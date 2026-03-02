package zybogpt

import spinal.core._
import spinal.core.sim._
import spinal.lib._

/** Multi-head attention module.
  *
  * Dataflow:
  * 1. QKV projections via TDot array (ternary weights), 2 batches each
  * 2. Reshape Q,K,V to (n_heads, head_dim)
  * 3. Store K,V in cache
  * 4. Attention scores: Q @ K^T via INT8 MAC array
  * 5. Scale by 1/sqrt(head_dim)
  * 6. Causal mask
  * 7. Softmax
  * 8. Weighted sum: attn @ V via INT8 MAC array
  * 9. Concat heads + O projection via TDot array (2 batches)
  *
  * scoreBuf uses Mem() (BRAM) instead of Vec(Reg()) to avoid 128-to-1 mux trees.
  * Softmax reads scores via address/data interface instead of Vec.
  */
case class Attention(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val io = new Bundle {
    val x = in Vec (SInt(8 bits), config.dModel)
    val position = in UInt (8 bits)
    val layerIdx = in UInt (log2Up(config.nLayers) bits)
    val start = in Bool ()

    val result = out Vec (SInt(8 bits), config.dModel)
    val done = out Bool ()

    // TDot array interface (shared resource)
    val tdotX = out Vec (SInt(8 bits), config.dModel)
    val tdotWeightAddr = out UInt (16 bits)
    val tdotStart = out Bool ()
    val tdotResult = in Vec (SInt(24 bits), config.numTDots)
    val tdotDone = in Bool ()

    // MAC array interface (shared resource)
    val macA = out Vec (SInt(8 bits), config.numDspMacs)
    val macB = out Vec (SInt(8 bits), config.numDspMacs)
    val macValid = out Bool ()
    val macClear = out Bool ()
    val macStart = out Bool ()
    val macResults = in Vec (SInt(24 bits), config.numDspMacs)
    val macDone = in Bool ()

    // KV cache interface
    val kvWriteK = out Vec (SInt(8 bits), config.headDim)
    val kvWriteV = out Vec (SInt(8 bits), config.headDim)
    val kvWriteHead = out UInt (log2Up(config.nHeads) bits)
    val kvWriteEn = out Bool ()
    val kvReadPos = out UInt (log2Up(config.ctxLen) bits)
    val kvReadHead = out UInt (log2Up(config.nHeads) bits)
    val kvReadEn = out Bool ()
    val kvReadK = in Vec (SInt(8 bits), config.headDim)
    val kvReadV = in Vec (SInt(8 bits), config.headDim)
    val kvReadValid = in Bool ()

    // Softmax interface: address/data for score reads (replaces Vec)
    val smScoreAddr = in UInt (7 bits)
    val smScoreData = out SInt (16 bits)
    val smLen = out UInt (8 bits)
    val smStart = out Bool ()
    // Prob read via address/data interface (replaces Vec to avoid mux trees)
    val smProbAddr = out UInt (7 bits)
    val smProbData = in UInt (8 bits)
    val smDone = in Bool ()
  }

  object State extends SpinalEnum {
    val IDLE, Q_PROJ, K_PROJ, V_PROJ, STORE_KV,
        ATTN_SCORE, ATTN_SCORE_MAC, ATTN_SCORE_SUM, ATTN_SCORE_WRITE,
        SCALE_MASK, SOFTMAX_WAIT,
        ATTN_VALUE, ATTN_VALUE_CALC, ATTN_VALUE_FINAL, CONCAT, O_PROJ, DONE = newElement()
  }

  val state = RegInit(State.IDLE); state.simPublic()
  val headIdx = Reg(UInt(log2Up(config.nHeads) bits)) init 0
  val posIdx = Reg(UInt(8 bits)) init 0
  val projBatch = Reg(UInt(1 bits)) init 0 // 0 or 1 for 2-batch TDot

  // Number of TDot batches per projection: dModel / numTDots
  val numProjBatches = config.dModel / config.numTDots // 64/32 = 2

  // Buffers for Q, K, V projections (full dModel width)
  val qBuf = Vec(Reg(SInt(8 bits)) init 0, config.dModel); qBuf.simPublic()
  val kBuf = Vec(Reg(SInt(8 bits)) init 0, config.dModel); kBuf.simPublic()
  val vBuf = Vec(Reg(SInt(8 bits)) init 0, config.dModel); vBuf.simPublic()

  // Score buffer uses Mem (BRAM) instead of Vec(Reg()) to avoid mux trees
  val scoreMem = Mem(SInt(16 bits), config.ctxLen)
  val attnOutBuf = Vec(Reg(SInt(8 bits)) init 0, config.dModel); attnOutBuf.simPublic()
  val oProjInputBuf = Vec(Reg(SInt(8 bits)) init 0, config.dModel) // Snapshot for O_PROJ input

  // STORE_KV write wait counter
  val kvWaitCounter = Reg(UInt(log2Up(config.headDim + 1) bits)) init 0

  // KV read pacing - prevents overlapping reads
  val readPending = RegInit(False)

  // Registered prob value: breaks BRAM→DSP timing path.
  val probReg = Reg(UInt(8 bits)) init 0

  // MAC array step counter for ATTN_SCORE_MAC (0=start, 1..4=data, 5+=wait done)
  val macStep = Reg(UInt(3 bits)) init 0
  // Dimension batch counter for ATTN_VALUE_CALC (0..4 for 8 dims per cycle + pipeline drain)
  val valueStep = Reg(UInt(3 bits)) init 0

  // Pipeline registers for ATTN_VALUE_CALC: 3-stage pipeline breaks DSP MACC feedback path.
  // Stage 1: vSlice mux → vSlicePipe (breaks readBufV → mux → DSP)
  // Stage 2: probSigned × vSlicePipe → productPipe (DSP multiply only, no accumulate)
  // Stage 3: attnAccum(d) += productPipe (carry chain only, no DSP in feedback loop)
  val vSlicePipe = Vec(Reg(SInt(8 bits)) init 0, config.numDspMacs)
  val vSlicePipeStep = Reg(UInt(3 bits)) init 0
  val vSlicePipeValid = RegInit(False)

  // Stage 2→3 product pipeline: prevents DSP48E1 MACC mode (which merges multiply+accumulate
  // into a feedback path).
  val productPipe = Vec(Reg(SInt(24 bits)) init 0, config.numDspMacs)
  val productPipeStep = Reg(UInt(3 bits)) init 0
  val productPipeValid = RegInit(False)

  // Accumulator for weighted value sum (ATTN_VALUE)
  val attnAccum = Vec(Reg(SInt(24 bits)) init 0, config.headDim)

  // Pipeline registers for ATTN_SCORE_MAC → scoreMem write.
  // Two-stage pipeline breaks 13-level path: DSP48E1 acc → adder tree → ×45 → scoreMem.
  // Stage 1 (ATTN_SCORE_SUM): macResultsReg → 8-element adder tree → macSumReg
  // Stage 2 (ATTN_SCORE_WRITE): macSumReg → ×45 scale → scoreMem write
  val macResultsReg = Vec(Reg(SInt(24 bits)) init 0, config.numDspMacs)
  val macSumReg = Reg(SInt(27 bits)) init 0

  // Softmax reads scores from scoreMem via address/data interface
  io.smScoreData := scoreMem.readSync(io.smScoreAddr)

  // Precompute SatInt8 for TDot results ONCE (shared across Q/K/V/O_PROJ states).
  // Without this, SpinalHDL generates separate SatInt8 per when-branch per call site
  // = 4 projections × 32 elements × 2 branches = 256 instances.
  // Precomputing reduces to 32 instances.
  val saturatedTdotResults = (0 until config.numTDots).map { i =>
    SatInt8(io.tdotResult(i) >> 4)
  }

  // Default outputs
  io.tdotX := io.x
  io.tdotWeightAddr := 0
  io.tdotStart := False
  io.macA.foreach(_ := 0)
  io.macB.foreach(_ := 0)
  io.macValid := False
  io.macClear := False
  io.macStart := False
  for (i <- 0 until config.headDim) {
    io.kvWriteK(i) := kBuf(i)
    io.kvWriteV(i) := vBuf(i)
  }
  io.kvWriteHead := headIdx
  io.kvWriteEn := False
  io.kvReadPos := posIdx.resize(log2Up(config.ctxLen) bits)
  io.kvReadHead := headIdx
  io.kvReadEn := False
  io.smLen := io.position + 1
  io.smStart := False
  io.smProbAddr := posIdx.resize(7 bits)
  for (i <- 0 until config.dModel) {
    io.result(i) := attnOutBuf(i)
  }
  io.done := False

  // Weight address offsets (per-block packed, word-aligned)
  val layerStride = config.layerPackedBytes
  val projStride = config.attnProjPackedBytes
  val batchWeightBytes = config.bytesPerTdotLoad

  // Registered weight address: eliminates combinational state → addr calc → BRAM path (7 logic levels).
  // Pre-computed at state transitions; incremented by batchWeightBytes within a projection,
  // by (projStride - batchWeightBytes) between projections.
  val projAddr = Reg(UInt(16 bits)) init 0

  /** Store TDot results into a buffer at the right batch offset.
    * Uses precomputed saturatedTdotResults to avoid duplicate SatInt8 hardware.
    */
  def storeTdotResults(buf: Vec[SInt]): Unit = {
    for (i <- 0 until config.numTDots) {
      when(projBatch === 0) {
        buf(i) := saturatedTdotResults(i)
      } otherwise {
        buf(config.numTDots + i) := saturatedTdotResults(i)
      }
    }
  }

  switch(state) {
    is(State.IDLE) {
      when(io.start) {
        state := State.Q_PROJ
        headIdx := 0
        projBatch := 0
        // Pre-compute initial weight address: projIdx=0, projBatch=0
        projAddr := ((io.layerIdx.resize(16 bits) * U(layerStride, 16 bits)).addAttribute("use_dsp", "no")).resize(16 bits)
      }
    }

    is(State.Q_PROJ) {
      io.tdotX := io.x
      io.tdotWeightAddr := projAddr
      io.tdotStart := True
      when(io.tdotDone) {
        storeTdotResults(qBuf)
        projAddr := projAddr + U(batchWeightBytes, 16 bits) // advance to next batch/proj
        when(projBatch === (numProjBatches - 1)) {
          projBatch := 0
          state := State.K_PROJ
        } otherwise {
          projBatch := projBatch + 1
        }
      }
    }

    is(State.K_PROJ) {
      io.tdotX := io.x
      io.tdotWeightAddr := projAddr
      io.tdotStart := True
      when(io.tdotDone) {
        storeTdotResults(kBuf)
        projAddr := projAddr + U(batchWeightBytes, 16 bits)
        when(projBatch === (numProjBatches - 1)) {
          projBatch := 0
          state := State.V_PROJ
        } otherwise {
          projBatch := projBatch + 1
        }
      }
    }

    is(State.V_PROJ) {
      io.tdotX := io.x
      io.tdotWeightAddr := projAddr
      io.tdotStart := True
      when(io.tdotDone) {
        storeTdotResults(vBuf)
        projAddr := projAddr + U(batchWeightBytes, 16 bits)
        when(projBatch === (numProjBatches - 1)) {
          projBatch := 0
          state := State.STORE_KV
          headIdx := 0
        } otherwise {
          projBatch := projBatch + 1
        }
      }
    }

    is(State.STORE_KV) {
      // Write K,V to cache for each head, waiting headDim cycles per write
      // Unroll headIdx to use compile-time constant indices
      for (i <- 0 until config.headDim) {
        io.kvWriteK(i) := kBuf(i) // default: head 0
        io.kvWriteV(i) := vBuf(i)
        for (h <- 1 until config.nHeads) {
          when(headIdx === h) {
            io.kvWriteK(i) := kBuf(h * config.headDim + i) // constant index
            io.kvWriteV(i) := vBuf(h * config.headDim + i) // constant index
          }
        }
      }
      io.kvWriteHead := headIdx

      when(kvWaitCounter === 0) {
        // Pulse writeEn for 1 cycle to start the write
        io.kvWriteEn := True
        kvWaitCounter := 1
      }.elsewhen(kvWaitCounter < config.headDim) {
        // Wait for write to complete (headDim cycles)
        kvWaitCounter := kvWaitCounter + 1
      }.otherwise {
        // Write done, advance to next head or next state
        kvWaitCounter := 0
        headIdx := headIdx + 1
        when(headIdx === (config.nHeads - 1)) {
          state := State.ATTN_SCORE
          headIdx := 0
          posIdx := 0
          readPending := False
        }
      }
    }

    is(State.ATTN_SCORE) {
      // Issue KV read to get K vector for dot product
      when(!readPending) {
        io.kvReadPos := posIdx.resize(log2Up(config.ctxLen) bits)
        io.kvReadEn := True
        readPending := True
      }
      when(io.kvReadValid) {
        readPending := False
        macStep := 0
        state := State.ATTN_SCORE_MAC
      }
    }

    is(State.ATTN_SCORE_MAC) {
      // Q[head] · K[pos] dot product via MAC array (8 MAC lanes × 4 steps).
      val numDotSteps = config.headDim / config.numDspMacs // 4

      // Feed Q and K data to MAC array (constant-indexed via head/step unrolling).
      // Range extended to numDotSteps+1 to drain AREG/BREG pipeline in Int8MacUnit.
      // Extra cycle (macStep=numDotSteps+1): macA/macB stay at 0 (default), but AREG
      // still holds last valid data, so the final product gets accumulated.
      when(macStep >= 1 && macStep <= numDotSteps + 1) {
        io.macValid := True
        when(macStep <= numDotSteps) {
          val dataStep = macStep - 1
          for (i <- 0 until config.numDspMacs) {
            for (s <- 0 until numDotSteps) {
              when(dataStep === s) {
                for (h <- 0 until config.nHeads) {
                  when(headIdx === h) {
                    io.macA(i) := qBuf(h * config.headDim + s * config.numDspMacs + i)
                  }
                }
                io.macB(i) := io.kvReadK(s * config.numDspMacs + i)
              }
            }
          }
        }
      }

      when(macStep === 0) {
        io.macStart := True
        macStep := 1
      }.elsewhen(macStep <= numDotSteps + 1) {
        macStep := macStep + 1
      }.otherwise {
        when(io.macDone) {
          // Register MAC results to break DSP→adder→scale→scoreMem path
          for (i <- 0 until config.numDspMacs) {
            macResultsReg(i) := io.macResults(i)
          }
          state := State.ATTN_SCORE_SUM
        }
      }
    }

    is(State.ATTN_SCORE_SUM) {
      // Pipeline stage 2: sum registered MAC results → register macSumReg
      macSumReg := macResultsReg.map(_.resize(27 bits)).reduce(_ + _)
      state := State.ATTN_SCORE_WRITE
    }

    is(State.ATTN_SCORE_WRITE) {
      // Pipeline stage 3: scale and write to scoreMem
      val scaledScore = (((macSumReg * 45).addAttribute("use_dsp", "no")) >> 8).resize(16 bits)
      scoreMem.write(posIdx.resize(log2Up(config.ctxLen) bits), scaledScore)

      when(posIdx === io.position) {
        state := State.SCALE_MASK
      }.otherwise {
        posIdx := posIdx + 1
        state := State.ATTN_SCORE
      }
    }

    is(State.SCALE_MASK) {
      io.smStart := True
      state := State.SOFTMAX_WAIT
    }

    is(State.SOFTMAX_WAIT) {
      when(io.smDone) {
        state := State.ATTN_VALUE
        posIdx := 0
      }
    }

    is(State.ATTN_VALUE) {
      // Weighted value sum: attn_weights @ V, serialized 8 dims per cycle.

      // Clear accumulators on entry (posIdx === 0 and not yet started reading)
      when(posIdx === 0 && !readPending) {
        for (d <- 0 until config.headDim) {
          attnAccum(d) := 0
        }
      }

      // Start a KV read only when previous read is done
      when(!readPending) {
        io.kvReadPos := posIdx.resize(log2Up(config.ctxLen) bits)
        io.kvReadEn := True
        readPending := True
      }

      when(io.kvReadValid) {
        readPending := False
        valueStep := 0
        probReg := io.smProbData // Capture prob before DSP multiply (breaks BRAM→DSP path)
        state := State.ATTN_VALUE_CALC
      }
    }

    is(State.ATTN_VALUE_CALC) {
      // Serialized prob × V accumulation: 8 dimensions per cycle, 4 cycles per position.
      // 3-stage pipeline prevents DSP48E1 MACC mode feedback path.
      // Stage 1: vSlice mux → vSlicePipe register (no DSP)
      // Stage 2: vSlicePipe × prob → productPipe register (DSP multiply only)
      // Stage 3: attnAccum(d) += productPipe (carry chain only, no DSP in loop)
      val numBatches = config.headDim / config.numDspMacs // 4

      // ---- Stage 1: Mux V inputs → pipeline register ----
      when(valueStep < numBatches) {
        for (i <- 0 until config.numDspMacs) {
          vSlicePipe(i) := io.kvReadV(i) // default: batch 0
          for (batch <- 1 until numBatches) {
            when(valueStep === batch) {
              vSlicePipe(i) := io.kvReadV(batch * config.numDspMacs + i)
            }
          }
        }
        vSlicePipeStep := valueStep
        vSlicePipeValid := True
        valueStep := valueStep + 1
      } otherwise {
        vSlicePipeValid := False
      }

      // ---- Stage 2: Multiply only → product pipeline register ----
      when(vSlicePipeValid) {
        val probSigned = probReg.resize(9 bits).asSInt
        for (i <- 0 until config.numDspMacs) {
          productPipe(i) := (probSigned * vSlicePipe(i)).addAttribute("use_dsp", "yes").resize(24 bits)
        }
        productPipeStep := vSlicePipeStep
        productPipeValid := True
      } otherwise {
        productPipeValid := False
      }

      // ---- Stage 3: Accumulate from product pipeline (no DSP in feedback loop) ----
      when(productPipeValid) {
        for (i <- 0 until config.numDspMacs) {
          for (batch <- 0 until numBatches) {
            when(productPipeStep === batch) {
              val d = batch * config.numDspMacs + i
              attnAccum(d) := attnAccum(d) + productPipe(i)
            }
          }
        }
      }

      // Transition when all 3 pipeline stages fully drained
      when(valueStep >= numBatches && !vSlicePipeValid && !productPipeValid) {
        when(posIdx === io.position) {
          state := State.ATTN_VALUE_FINAL
        }.otherwise {
          posIdx := posIdx + 1
          state := State.ATTN_VALUE
        }
      }
    }

    is(State.ATTN_VALUE_FINAL) {
      // Read settled attnAccum (last batch's accumulation registered last cycle),
      // apply SatInt8 and write to attnOutBuf. No DSP in critical path.
      for (d <- 0 until config.headDim) {
        val saturated = SatInt8(attnAccum(d) >> 8)
        for (h <- 0 until config.nHeads) {
          when(headIdx === h) {
            attnOutBuf(h * config.headDim + d) := saturated
          }
        }
      }

      headIdx := headIdx + 1
      when(headIdx === (config.nHeads - 1)) {
        state := State.CONCAT
      }.otherwise {
        posIdx := 0
        readPending := False
        state := State.ATTN_SCORE
      }
    }

    is(State.CONCAT) {
      // Snapshot attnOutBuf for O_PROJ input
      for (i <- 0 until config.dModel) {
        oProjInputBuf(i) := attnOutBuf(i)
      }
      projBatch := 0
      state := State.O_PROJ
    }

    is(State.O_PROJ) {
      // O projection via TDot (2 batches), reading from snapshot buffer
      for (i <- 0 until config.dModel) {
        io.tdotX(i) := oProjInputBuf(i)
      }
      io.tdotWeightAddr := projAddr
      io.tdotStart := True
      when(io.tdotDone) {
        storeTdotResults(attnOutBuf)
        projAddr := projAddr + U(batchWeightBytes, 16 bits)
        when(projBatch === (numProjBatches - 1)) {
          state := State.DONE
        } otherwise {
          projBatch := projBatch + 1
        }
      }
    }

    is(State.DONE) {
      io.done := True
      state := State.IDLE
    }
  }
}
