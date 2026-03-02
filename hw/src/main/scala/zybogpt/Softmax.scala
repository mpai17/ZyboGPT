package zybogpt

import spinal.core._
import spinal.lib._

/** Integer softmax implementation with piecewise-linear exp approximation.
  *
  * Steps:
  * 1. Load scores from external scoreMem via address/data interface + find max (sequential)
  * 2. Re-read scores + subtract max + piecewise-linear exp (sequential)
  * 3. Sum all exp values (sequential)
  * 4. Direct reciprocal LUT (no LOD, covers expSum range directly)
  * 5. Normalize: prob[i] = clamp(exp[i] * recipVal >> 8, 0, 255)
  *
  * Input: INT16 scores read via scoreAddr/scoreData interface (from Attention's scoreMem)
  * Output: UINT8 probabilities (0-255, sum ≈ 256)
  *
  * expBuf uses Mem() (infers BRAM) to avoid mux trees.
  * Scores are read directly from Attention's scoreMem (no internal copy needed
  * for SUBTRACT_EXP since we re-read via the address interface).
  *
  * Mem() instances: expMem, scoreMem, plExpLut, probMem, recipLut.
  */
case class Softmax(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val maxLen = config.ctxLen // 128

  val io = new Bundle {
    // Score read interface (reads from Attention's scoreMem)
    val scoreAddr = out UInt (7 bits)
    val scoreData = in SInt (16 bits)
    val len = in UInt (8 bits)
    val start = in Bool ()
    // Prob output via address/data interface (replaces Vec to avoid mux trees)
    val probAddr = in UInt (7 bits)
    val probData = out UInt (8 bits)
    val done = out Bool ()
  }

  object State extends SpinalEnum {
    val IDLE, LOAD_AND_FIND_MAX, SUBTRACT_EXP, SUM_EXP, RECIP, NORMALIZE, DONE = newElement()
  }
  val state = RegInit(State.IDLE)
  val step = Reg(UInt(8 bits)) init 0

  // BRAM-backed exp buffer
  val expMem = Mem(UInt(16 bits), maxLen)

  // Internal score buffer for SUBTRACT_EXP re-read
  // We need scores in two passes (LOAD_AND_FIND_MAX and SUBTRACT_EXP),
  // so we store them internally during the first pass.
  val scoreMem = Mem(SInt(16 bits), maxLen)

  // probMem replaces Vec(Reg()) resultBuf to avoid write mux tree
  val probMem = Mem(UInt(8 bits), maxLen)
  val maxVal = Reg(SInt(16 bits)) init 0
  val expSum = Reg(UInt(24 bits)) init 0
  val seqLen = Reg(UInt(8 bits)) init 0

  // Pipeline registers for SUBTRACT_EXP: 3-stage pipeline.
  // Stage 1: scoreMem read + subtract maxVal → subExpPipeShifted
  // Stage 2: clamp to [-128,0] + plExp LUT address → plExpLut readSync
  // Stage 3: plExpLut output → expMem write
  // Replaces 14-CARRY4-level combinational plExp with BRAM lookup.
  val subExpPipeShifted = Reg(SInt(16 bits)) init 0
  val subExpPipeValid = RegInit(False)
  val subExpPipeAddr = Reg(UInt(7 bits)) init 0
  val subExpPipeInRange = RegInit(False)

  // Stage 2→3 pipeline registers
  val plExpPipeAddr = Reg(UInt(7 bits)) init 0
  val plExpPipeValid = RegInit(False)
  val plExpPipeInRange = RegInit(False)

  // plExp LUT: replaces combinational piecewise-linear exp (14 carry chains)
  // with single-cycle BRAM readSync. Index = (xClamped + 128), range [0, 256].
  val plExpLutVals = (0 until 256).map { i =>
    val xClamped = i - 128
    val expVal = if (xClamped >= 0) 256
      else if (xClamped >= -3) 256 + xClamped * 64
      else if (xClamped >= -8) 64 + (xClamped + 3) * 11
      else if (xClamped >= -24) xClamped + 24
      else 0
    BigInt(scala.math.max(0, expVal))
  }
  val plExpLut = Mem(UInt(16 bits), 256) initBigInt plExpLutVals
  val plExpLutAddr = UInt(8 bits)
  plExpLutAddr := 0 // default
  val plExpLutData = plExpLut.readSync(plExpLutAddr)

  // Pipeline registers for NORMALIZE: 3-stage pipeline.
  // Stage 1: expMem readSync → normExpDataReg
  // Stage 2: normExpDataReg × invSumVal → normProductReg (PREG absorbs RegNext)
  // Stage 3: normProductReg >> 8 + clamp → probMem write
  val normExpDataReg = Reg(UInt(16 bits)) init 0
  val normPipeValid = RegInit(False)
  val normPipeAddr = Reg(UInt(7 bits)) init 0
  val normProductReg = Reg(UInt(32 bits)) init 0
  val normProduct2Valid = RegInit(False)
  val normProduct2Addr = Reg(UInt(7 bits)) init 0

  // Read address registers and readSync ports for internal BRAM
  val scoreLocalAddr = UInt(7 bits)
  scoreLocalAddr := step.resize(7 bits) // default
  val scoreLocalReadData = scoreMem.readSync(scoreLocalAddr)

  val expAddr = UInt(7 bits)
  expAddr := step.resize(7 bits) // default
  val expReadData = expMem.readSync(expAddr)

  // Direct reciprocal LUT: 256 entries
  val recipShift = 7
  val recipInitVals = (0 until 256).map { i =>
    if (i == 0) BigInt(65535)
    else {
      val expSumApprox = i.toDouble * (1 << recipShift) + (1 << (recipShift - 1))
      val recipVal = 65536.0 / expSumApprox
      BigInt(Math.round(recipVal).max(0).min(65535))
    }
  }
  val recipLut = Mem(UInt(16 bits), 256) initBigInt recipInitVals

  val recipAddr = Reg(UInt(8 bits)) init 0
  val recipOut = recipLut.readSync(recipAddr)
  val invSumVal = Reg(UInt(16 bits)) init 0

  // Prob readSync for address/data interface
  io.probData := probMem.readSync(io.probAddr)

  io.done := False
  io.scoreAddr := 0

  switch(state) {
    is(State.IDLE) {
      when(io.start) {
        state := State.LOAD_AND_FIND_MAX
        step := 0
        maxVal := S(-32767, 16 bits)
        expSum := 0
        seqLen := io.len
        // Prime read-ahead: request score at address 0
        io.scoreAddr := U(0, 7 bits)
      }
    }

    // LOAD_AND_FIND_MAX: read scores from external scoreMem via address/data,
    // store into internal scoreMem, and track max value.
    // readSync latency = 1 cycle: addr set at cycle N, data available at cycle N+1.
    // IDLE primes addr=0, so data[0] arrives at step=0.
    // step=0: data[0] ready from IDLE prime, request addr=1, process data[0]
    // step=1: data[1] ready, request addr=2, process data[1]
    // ...
    // step=127: data[127] ready, done, transition
    is(State.LOAD_AND_FIND_MAX) {
      // Read-ahead: request next score
      io.scoreAddr := (step + 1).resize(7 bits)

      // Process current score (data for addr=step arrived from previous cycle's request)
      scoreMem.write(step.resize(7 bits), io.scoreData)

      when(step < seqLen) {
        when(io.scoreData > maxVal) {
          maxVal := io.scoreData
        }
      }

      step := step + 1
      when(step === (maxLen - 1)) {
        state := State.SUBTRACT_EXP
        step := 0
        // Prime read-ahead for internal scoreMem
        scoreLocalAddr := U(0, 7 bits)
      }
    }

    // SUBTRACT_EXP: 3-stage pipeline using plExp BRAM LUT.
    // Replaces 14-CARRY4-level combinational plExp with single-cycle BRAM lookup.
    // Stage 1: scoreMem read → subtract maxVal → subExpPipeShifted
    // Stage 2: clamp [-128,0] → plExpLut address (readSync)
    // Stage 3: plExpLut data → expMem write
    is(State.SUBTRACT_EXP) {
      // Stage 1: read from scoreMem (primed last cycle), compute shifted, pipeline it
      when(step < U(maxLen, 8 bits)) {
        val shifted = (scoreLocalReadData - maxVal).resize(16 bits)
        subExpPipeShifted := shifted
        subExpPipeAddr := step.resize(7 bits)
        subExpPipeInRange := step < seqLen
        subExpPipeValid := True

        step := step + 1
        scoreLocalAddr := (step + 1).resize(7 bits)
      } otherwise {
        subExpPipeValid := False
      }

      // Stage 2: clamp and compute plExp LUT address (1 cycle behind stage 1)
      when(subExpPipeValid) {
        val x = subExpPipeShifted
        val clamped = SInt(8 bits)
        when(x >= S(0, 16 bits)) {
          clamped := S(0, 8 bits)
        } elsewhen (x < S(-128, 16 bits)) {
          clamped := S(-128, 8 bits)
        } otherwise {
          clamped := x.resize(8 bits)
        }
        plExpLutAddr := (clamped.asUInt + U(128, 8 bits)).resize(8 bits)
        plExpPipeAddr := subExpPipeAddr
        plExpPipeInRange := subExpPipeInRange
        plExpPipeValid := True
      } otherwise {
        plExpPipeValid := False
      }

      // Stage 3: write plExp LUT result to expMem (2 cycles behind stage 1)
      when(plExpPipeValid) {
        when(plExpPipeInRange) {
          expMem.write(plExpPipeAddr, plExpLutData)
        } otherwise {
          expMem.write(plExpPipeAddr, U(0, 16 bits))
        }
      }

      // Transition when all three stages drained
      when(step >= U(maxLen, 8 bits) && !subExpPipeValid && !plExpPipeValid) {
        state := State.SUM_EXP
        step := 0
        expAddr := U(0, 7 bits)
      }
    }

    // SUM_EXP: read from expMem via read-ahead, accumulate sum
    is(State.SUM_EXP) {
      expSum := expSum + expReadData.resize(24 bits)

      step := step + 1
      expAddr := (step + 1).resize(7 bits)

      when(step === (maxLen - 1)) {
        recipAddr := ((expSum + expReadData.resize(24 bits)) >> recipShift).resize(8 bits)
        state := State.RECIP
        step := 0
      }
    }

    is(State.RECIP) {
      expAddr := U(0, 7 bits)
      when(step === 0) {
        step := 1
      } elsewhen (step === 1) {
        invSumVal := recipOut
        step := 0
        state := State.NORMALIZE
      }
    }

    // NORMALIZE: 3-stage pipeline.
    // Stage 1: expMem readSync → normExpDataReg
    // Stage 2: normExpDataReg × invSumVal → normProductReg (Vivado absorbs into DSP48E1 PREG)
    // Stage 3: normProductReg >> 8 + clamp → probMem write
    is(State.NORMALIZE) {
      // Stage 1: capture BRAM read data into register
      when(step < U(maxLen, 8 bits)) {
        normExpDataReg := expReadData
        normPipeAddr := step.resize(7 bits)
        normPipeValid := True
        step := step + 1
        expAddr := (step + 1).resize(7 bits)
      } otherwise {
        normPipeValid := False
      }

      // Stage 2: multiply from registered value, register product (DSP48E1 PREG)
      when(normPipeValid) {
        normProductReg := normExpDataReg * invSumVal // UInt(16) × UInt(16) = UInt(32)
        normProduct2Addr := normPipeAddr
        normProduct2Valid := True
      } otherwise {
        normProduct2Valid := False
      }

      // Stage 3: shift + clamp + BRAM write (from registered DSP output)
      when(normProduct2Valid) {
        val prob = (normProductReg >> 8).resize(16 bits)
        val probClamped = Mux(prob > 255, U(255, 8 bits), prob.resize(8 bits))
        probMem.write(normProduct2Addr, probClamped)
      }

      // Done when all 3 pipeline stages drained
      when(step >= U(maxLen, 8 bits) && !normPipeValid && !normProduct2Valid) {
        state := State.DONE
      }
    }

    is(State.DONE) {
      io.done := True
      state := State.IDLE
    }
  }

}
