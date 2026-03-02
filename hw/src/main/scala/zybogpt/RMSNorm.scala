package zybogpt

import spinal.core._
import spinal.lib._

/** Integer RMSNorm implementation.
  *
  * Computes: y[i] = x[i] / rms(x) * gamma[i] * 64
  * where rms(x) = sqrt(mean(x^2))
  *
  * The 64x scale factor ensures the normalized output fills the INT8 range
  * [-128, 127]. Without it, RMSNorm normalizes to unit variance (~±2),
  * which is too small for downstream INT8 operations.
  *
  * Implementation uses a direct 256-entry inv_sqrt LUT indexed by meanSq >> 6,
  * covering the full INT8 range without LOD decomposition (avoids odd/even
  * exponent issues). For d_model=64 with INT8 inputs, meanSq max = 127^2 = 16129,
  * so meanSq >> 6 fits in 8 bits (max index = 251).
  *
  * LUT stores inv_sqrt values in Q2.14 format (14 fractional bits).
  * Final computation: y[i] = clamp((x[i] * LUT[idx] >> 8) * gamma[i] >> 10)
  * Effective shift = 18 instead of 24, providing the 64x scale (2^6).
  *
  * xBuf uses Mem() (BRAM) instead of Vec(Reg()) to avoid 64-to-1 mux trees.
  * Gamma is read via address/data interface from external normGammaMem.
  *
  * Uses Mem() for xMem and invSqrtLut (Vivado infers distributed RAM).
  */
case class RMSNorm(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val dim = config.dModel // 64
  val lutShift = 6 // meanSq >> 6 to get LUT index
  val lutFrac = 14 // Q2.14 format for LUT values

  val io = new Bundle {
    val x = in Vec (SInt(8 bits), dim)
    val gammaAddr = out UInt (log2Up(dim) bits)
    val gammaData = in SInt (16 bits)
    val start = in Bool ()
    val y = out Vec (SInt(8 bits), dim)
    val done = out Bool ()
  }

  // FSM
  object State extends SpinalEnum {
    val IDLE, LOAD_SUM_SQ, LOOKUP, SCALE, DONE = newElement()
  }
  val state = RegInit(State.IDLE)

  // BRAM for input buffer (replaces Vec(Reg()) to avoid mux trees)
  val xMem = Mem(SInt(8 bits), dim)
  val xAddr = UInt(log2Up(dim) bits)
  xAddr := 0 // default
  val xReadData = xMem.readSync(xAddr)

  val resultBuf = Vec(Reg(SInt(8 bits)) init 0, dim)

  val sumSq = Reg(UInt(32 bits)) init 0
  val step = Reg(UInt(7 bits)) init 0
  val invRmsVal = Reg(UInt(16 bits)) init 0

  // Pipeline registers for LOAD_SUM_SQ: breaks 12-level path from
  // 64-to-1 io.x mux (MUXF7/MUXF8) through DSP square + CARRY4 accumulate.
  // Stage 1: mux select → loadPipeReg. Stage 2: square + accumulate.
  val loadPipeReg = Reg(SInt(8 bits)) init 0
  val loadPipeIdx = Reg(UInt(log2Up(dim) bits)) init 0
  val loadPipeValid = RegInit(False)

  // Stage 3 pipeline: breaks loadPipeReg → square → sumSq accumulate path.
  // Square computation (LUT6 + CARRY4) is separated from accumulation (CARRY4 chain).
  val sqPipe = Reg(UInt(14 bits)) init 0 // max: 127² = 16129, fits 14 bits
  val sqPipeValid = RegInit(False)

  // Pipeline registers for SCALE state: 4-stage pipeline.
  // Stage 0: xMem readSync → xReadPipe register (breaks BRAM→DSP path)
  // Stage 1: xReadPipe * invRms (8×16, single DSP48E1) → scaleProd1Pipe
  // Stage 2: prod1 * gamma (16×16, single DSP48E1) → scaleProd2Shifted
  // Stage 3: clamp + resultBuf write
  // Using SInt(16) narrows multiply to fit DSP48E1 natively (no carry chain decomposition).
  val xReadPipe = Reg(SInt(8 bits)) init 0
  val gammaPipe0 = Reg(SInt(16 bits)) init 0
  val scaleIdx0 = Reg(UInt(log2Up(dim) bits)) init 0
  val scalePipe0Valid = RegInit(False)

  val scaleProd1Pipe = Reg(SInt(16 bits)) init 0
  val scaleGammaPipe = Reg(SInt(16 bits)) init 0
  val scaleWriteIdx = Reg(UInt(log2Up(dim) bits)) init 0
  val scalePipeValid = RegInit(False)

  // Stage 2→3 pipeline registers
  val scaleProd2Shifted = Reg(SInt(16 bits)) init 0
  val scaleProd2WriteIdx = Reg(UInt(log2Up(dim) bits)) init 0
  val scaleProd2Valid = RegInit(False)

  // Direct inv_sqrt LUT: 256 entries in Q2.14
  // Index i maps to meanSq ≈ i * 64 + 32 (midpoint of bin)
  // LUT[i] = round(16384 / sqrt(i * 64 + 32)) for i >= 1
  // LUT[0] = max value (protect against div-by-zero)
  val lutInitVals = (0 until 256).map { i =>
    if (i == 0) BigInt(16383) // Max representable, for near-zero input
    else {
      val meanSqApprox = i.toDouble * (1 << lutShift) + (1 << (lutShift - 1))
      val invSqrt = (1 << lutFrac).toDouble / Math.sqrt(meanSqApprox)
      BigInt(Math.round(invSqrt).max(0).min(16383))
    }
  }
  val invSqrtLut = Mem(UInt(16 bits), 256) initBigInt lutInitVals

  // LUT address from meanSq
  val meanSq = (sumSq >> log2Up(dim)).resize(16 bits)
  val lutIdx = (meanSq >> lutShift).resize(8 bits)

  // readSync with registered address
  val lutAddrReg = Reg(UInt(8 bits)) init 0
  val lutOut = invSqrtLut.readSync(lutAddrReg)

  io.done := False
  io.gammaAddr := 0

  switch(state) {
    is(State.IDLE) {
      when(io.start) {
        state := State.LOAD_SUM_SQ
        sumSq := 0
        step := 0
        loadPipeValid := False
      }
    }

    is(State.LOAD_SUM_SQ) {
      // Two-stage pipeline: breaks 12-level path (MUXF7/MUXF8 + DSP + CARRY4).
      // Stage 1: 64-to-1 mux select from io.x → pipeline register
      // Stage 2: square + accumulate + BRAM write from register (no mux in path)

      // Stage 1: mux and register
      when(step < dim) {
        loadPipeReg := io.x(step.resize(log2Up(dim) bits))
        loadPipeIdx := step.resize(log2Up(dim) bits)
        loadPipeValid := True
        step := step + 1
      } otherwise {
        loadPipeValid := False
      }

      // Stage 2: square only + BRAM write (no accumulation in this stage)
      when(loadPipeValid) {
        xMem.write(loadPipeIdx, loadPipeReg)
        val sq = (loadPipeReg * loadPipeReg).addAttribute("use_dsp", "no").asUInt.resize(14 bits)
        sqPipe := sq
        sqPipeValid := True
      } otherwise {
        sqPipeValid := False
      }

      // Stage 3: accumulate from sqPipe (CARRY4 chain only, no square in path)
      when(sqPipeValid) {
        sumSq := sumSq + sqPipe.resize(32 bits)
      }

      // Transition when all three pipeline stages fully drained
      when(step >= U(dim, 7 bits) && !loadPipeValid && !sqPipeValid) {
        state := State.LOOKUP
        step := 0
      }
    }

    is(State.LOOKUP) {
      // Step 0: sumSq register now has the full sum (all dim elements).
      //         Set LUT address from the correct meanSq.
      // Step 1: readSync latches lutAddrReg, BRAM read in progress.
      // Step 2: lutOut has valid data, capture invRmsVal.
      //         Also prime xAddr for SCALE read-ahead.
      when(step === 0) {
        lutAddrReg := lutIdx
        step := 1
      } elsewhen (step === 1) {
        step := 2
      } elsewhen (step === 2) {
        invRmsVal := lutOut
        step := 0
        state := State.SCALE
        // Prime read-ahead: xAddr=0 so xReadData has xMem[0] next cycle
        xAddr := U(0, log2Up(dim) bits)
        // Prime gamma read-ahead
        io.gammaAddr := U(0, log2Up(dim) bits)
      }
    }

    is(State.SCALE) {
      // Four-stage pipelined multiply to avoid BRAM→DSP and DSP timing violations.
      // Stage 0: xMem readSync → xReadPipe, gammaData → gammaPipe0 (breaks BRAM→DSP path)
      // Stage 1: xReadPipe * invRms (8×16, DSP48E1) → scaleProd1Pipe
      // Stage 2: prod1 * gamma (16×16, DSP48E1) → scaleProd2Shifted
      // Stage 3: clamp + resultBuf write (no DSP in path)
      val shift2 = 10

      // Stage 0: Register BRAM and gamma outputs (breaks BRAM CLK→DO → DSP path)
      when(step < dim) {
        xReadPipe := xReadData
        gammaPipe0 := io.gammaData
        scaleIdx0 := step.resize(log2Up(dim) bits)
        scalePipe0Valid := True
        step := step + 1
        xAddr := (step + 1).resize(log2Up(dim) bits)
        io.gammaAddr := (step + 1).resize(log2Up(dim) bits)
      } otherwise {
        scalePipe0Valid := False
      }

      // Stage 1: x * invRms from pipeline register (not directly from BRAM)
      val prod1 = xReadPipe * invRmsVal.asSInt // SInt(8) * SInt(16) = SInt(24)
      val prod1Trunc = (prod1 >> 8).resize(16 bits) // SInt(16), effective 14-bit range

      when(scalePipe0Valid) {
        scaleProd1Pipe := prod1Trunc
        scaleGammaPipe := gammaPipe0
        scaleWriteIdx := scaleIdx0
        scalePipeValid := True
      } otherwise {
        scalePipeValid := False
      }

      // Stage 2: prod1 * gamma (16×16, fits single DSP48E1) → shift → register
      val prod2 = scaleProd1Pipe * scaleGammaPipe // SInt(32)
      val shifted = (prod2 >> shift2).resize(16 bits) // SInt(16)
      scaleProd2Shifted := shifted
      scaleProd2WriteIdx := scaleWriteIdx
      scaleProd2Valid := scalePipeValid

      // Stage 3: clamp and write from pipeline (no DSP in path, just comparisons)
      when(scaleProd2Valid) {
        when(scaleProd2Shifted > 127) {
          resultBuf(scaleProd2WriteIdx) := S(127, 8 bits)
        } elsewhen (scaleProd2Shifted < -128) {
          resultBuf(scaleProd2WriteIdx) := S(-128, 8 bits)
        } otherwise {
          resultBuf(scaleProd2WriteIdx) := scaleProd2Shifted.resize(8 bits)
        }
      }

      // Done when all four pipeline stages drained
      when(step >= U(dim, 7 bits) && !scalePipe0Valid && !scalePipeValid && !scaleProd2Valid) {
        state := State.DONE
      }
    }

    is(State.DONE) {
      io.done := True
      state := State.IDLE
    }
  }

  for (i <- 0 until dim) {
    io.y(i) := resultBuf(i)
  }
}
