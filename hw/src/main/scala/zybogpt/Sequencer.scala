package zybogpt

import spinal.core._
import spinal.core.sim._
import spinal.lib._

/** Top-level sequencer FSM for single-token inference.
  *
  * States: IDLE -> EMBED -> EMB_READ -> LAYER_LOOP -> FINAL_NORM -> OUTPUT_LOGITS -> ARGMAX/SAMPLING -> DONE
  *
  * Each token inference:
  * 1. EMBED: Look up token + positional embedding
  * 2. LAYER_LOOP: Process through n_layers transformer layers
  * 3. FINAL_NORM: Apply final RMSNorm
  * 4. OUTPUT: Compute logits via tied embedding weights, find argmax
  * 5. DONE: Signal completion, output token available
  */
case class Sequencer(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val io = new Bundle {
    // Control from PS
    val tokenIn = in UInt (7 bits) // Input token ID
    val positionIn = in UInt (7 bits) // Current position
    val start = in Bool ()

    // Output
    val tokenOut = out UInt (7 bits) // Predicted next token
    val done = out Bool ()
    val busy = out Bool ()

    // Cycle counter for benchmarking
    val cycleCount = out UInt (32 bits)

    // Sampling configuration
    val invTemp = in UInt (16 bits)
    val seed = in UInt (32 bits)
    val seedWrite = in Bool ()  // Pulse to load seed into SamplingUnit LFSR

    // Embedding interface (addr/data for embedding readout)
    val embTokenId = out UInt (7 bits)
    val embPosition = out UInt (7 bits)
    val embStart = out Bool ()
    val embAddr = out UInt (log2Up(config.dModel) bits)
    val embData = in SInt (16 bits)
    val embDone = in Bool ()
    val embLogitMode = out Bool ()
    val embLogitVec = out Vec (SInt(8 bits), config.dModel)
    val embLogitTokenId = out UInt (7 bits)
    val embLogitResult = in SInt (24 bits)

    // Transformer layer interface
    val layerX = out Vec (SInt(8 bits), config.dModel)
    val layerPos = out UInt (8 bits)
    val layerIdx = out UInt (log2Up(config.nLayers) bits)
    val layerStart = out Bool ()
    val layerResult = in Vec (SInt(8 bits), config.dModel)
    val layerDone = in Bool ()

    // Final RMSNorm interface (gamma now via address/data in ZyboGPT.scala)
    val normX = out Vec (SInt(8 bits), config.dModel)
    val normStart = out Bool ()
    val normActive = out Bool () // True while final norm is running
    val normResult = in Vec (SInt(8 bits), config.dModel)
    val normDone = in Bool ()
  }

  object State extends SpinalEnum {
    val IDLE, EMBED, EMB_READ, LAYER_LOOP, FINAL_NORM, OUTPUT_LOGITS, ARGMAX, SAMPLING, DONE = newElement()
  }

  val state = RegInit(State.IDLE); state.simPublic()
  val layerCounter = Reg(UInt(log2Up(config.nLayers) bits)) init 0; layerCounter.simPublic()
  val cycleCounter = Reg(UInt(32 bits)) init 0

  // Activation buffer between layers
  val actBuf = Vec(Reg(SInt(8 bits)), config.dModel); actBuf.simPublic()

  // Embedding read counter (for sequential readout from Embedding Mem)
  val embReadIdx = Reg(UInt(log2Up(config.dModel) bits)) init 0

  // Pipeline registers for EMB_READ: 2-stage pipeline breaks BRAM → SatInt8 → actBuf critical path.
  // Stage 1: BRAM → SatInt8 → pipe reg. Stage 2: pipe reg → actBuf.
  val embPipeData = Reg(SInt(8 bits)) init 0
  val embPipeIdx = Reg(UInt(log2Up(config.dModel) bits)) init 0
  val embPipeValid = RegInit(False)
  val embReadDone = RegInit(False)

  // Logit computation
  val vocabIdx = Reg(UInt(7 bits)) init 0
  val maxLogit = Reg(SInt(24 bits))
  val maxToken = Reg(UInt(7 bits))
  val logitStarted = RegInit(False)

  // Logit buffer for temperature sampling (stores all 128 logits during OUTPUT_LOGITS)
  val logitBuf = Mem(SInt(24 bits), wordCount = config.vocabSize)

  // SamplingUnit for temperature sampling
  val sampler = SamplingUnit()
  sampler.io.logitData := logitBuf.readSync(sampler.io.logitAddr)
  sampler.io.maxLogit := maxLogit
  sampler.io.invTemp := io.invTemp
  sampler.io.seedLoad := io.seedWrite
  sampler.io.seedVal := io.seed
  sampler.io.start := False

  // Default outputs
  io.embTokenId := io.tokenIn
  io.embPosition := io.positionIn
  io.embStart := False
  io.embAddr := embReadIdx
  io.embLogitMode := False
  io.embLogitVec := actBuf
  io.embLogitTokenId := vocabIdx
  io.layerX := actBuf
  io.layerPos := io.positionIn.resize(8 bits)
  io.layerIdx := layerCounter
  io.layerStart := False
  io.normX := actBuf
  io.normStart := False
  io.normActive := (state === State.FINAL_NORM)
  io.tokenOut := maxToken
  io.done := False
  io.busy := state =/= State.IDLE
  io.cycleCount := cycleCounter

  // Count cycles while busy
  when(state =/= State.IDLE) {
    cycleCounter := cycleCounter + 1
  }

  switch(state) {
    is(State.IDLE) {
      when(io.start) {
        state := State.EMBED
        cycleCounter := 0
        layerCounter := 0
      }
    }

    is(State.EMBED) {
      io.embStart := True
      io.embLogitMode := False
      when(io.embDone) {
        // Transition to sequential embedding readout
        embReadIdx := 0
        embPipeValid := False
        embReadDone := False
        io.embAddr := U(0, log2Up(config.dModel) bits) // Prime read-ahead
        state := State.EMB_READ
      }
    }

    is(State.EMB_READ) {
      // 2-stage pipeline: BRAM → SatInt8 → pipeReg → actBuf write.
      // readSync latency: addr set at cycle N, data at cycle N+1.
      // EMBED→EMB_READ primes addr=0, so data[0] arrives at embReadIdx=0.

      // Stage 1: BRAM readSync data → SatInt8 → pipeline register
      when(!embReadDone) {
        io.embAddr := (embReadIdx + 1).resize(log2Up(config.dModel) bits) // read-ahead
        embPipeData := SatInt8(io.embData >> (config.fracBits - 7))
        embPipeIdx := embReadIdx
        embPipeValid := True
        when(embReadIdx === (config.dModel - 1)) {
          embReadDone := True
        } otherwise {
          embReadIdx := embReadIdx + 1
        }
      } otherwise {
        embPipeValid := False
      }

      // Stage 2: pipeline register → actBuf write (1 cycle behind)
      when(embPipeValid) {
        actBuf(embPipeIdx) := embPipeData
      }

      // Transition when pipeline fully drained
      when(embReadDone && !embPipeValid) {
        state := State.LAYER_LOOP
      }
    }

    is(State.LAYER_LOOP) {
      io.layerStart := True
      when(io.layerDone) {
        for (i <- 0 until config.dModel) {
          actBuf(i) := io.layerResult(i)
        }
        layerCounter := layerCounter + 1
        when(layerCounter === (config.nLayers - 1)) {
          state := State.FINAL_NORM
        }
      }
    }

    is(State.FINAL_NORM) {
      io.normStart := True
      when(io.normDone) {
        for (i <- 0 until config.dModel) {
          actBuf(i) := io.normResult(i)
        }
        state := State.OUTPUT_LOGITS
        vocabIdx := 0
        maxLogit := S(-8388607, 24 bits) // Near INT24_MIN
        maxToken := 0
        logitStarted := False
      }
    }

    is(State.OUTPUT_LOGITS) {
      // Compute logit for each vocab entry using tied embedding
      io.embLogitMode := True
      io.embLogitTokenId := vocabIdx

      // Pulse start once per vocab entry, wait for done
      when(!logitStarted) {
        io.embStart := True
        logitStarted := True
      }

      when(io.embDone) {
        logitStarted := False
        // Store logit in buffer for temperature sampling
        logitBuf.write(vocabIdx, io.embLogitResult)
        when(io.embLogitResult > maxLogit) {
          maxLogit := io.embLogitResult
          maxToken := vocabIdx
        }
        when(vocabIdx === (config.vocabSize - 1)) {
          // If temperature sampling enabled, go to SAMPLING; else greedy ARGMAX
          when(io.invTemp =/= 0) {
            state := State.SAMPLING
          } otherwise {
            state := State.ARGMAX
          }
        }.otherwise {
          vocabIdx := vocabIdx + 1
        }
      }
    }

    is(State.ARGMAX) {
      // Final argmax result is in maxToken (greedy mode)
      state := State.DONE
    }

    is(State.SAMPLING) {
      // Temperature sampling via SamplingUnit
      sampler.io.start := True
      when(sampler.io.done) {
        maxToken := sampler.io.selectedToken
        state := State.DONE
      }
    }

    is(State.DONE) {
      io.done := True
      // Return to IDLE on next cycle
      state := State.IDLE
    }
  }
}
