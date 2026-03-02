package zybogpt

import spinal.core._
import spinal.core.sim._

/** Temperature sampling unit with two-pass linear-exp approximation and Galois LFSR.
  *
  * Algorithm:
  *   Pass 1: For each vocab entry, compute (4-stage pipeline):
  *     Stage 1: diff = logitData - maxLogit, diffScaled = diff >> 12
  *     Stage 2a: shifted = (diffScaled * invTemp) >> 8 (DSP multiply, registered)
  *     Stage 2b: prob = max(0, min(256, 256 + shifted)) (bit manipulation, no DSP)
  *     Stage 3: totalProb += prob
  *
  *   Threshold (1 cycle): Advance LFSR, compute
  *     threshold = (lfsr[15:0] * totalProb) >> 16
  *
  *   Pass 2: Same pipeline, accumulate cumSum.
  *     When cumSum > threshold, selectedToken := current index, done.
  *
  * The 4-stage pipeline breaks the DSP → CARRY4 chain path.
  *
  * LFSR: Galois, polynomial 0xD000_0001 (matches Rust firmware sampling.rs)
  *
  * If invTemp == 0, caller should use greedy argmax instead (not handled here).
  */
case class SamplingUnit() extends Component {
  val io = new Bundle {
    // Logit memory read interface (Sequencer provides data via readSync)
    val logitAddr     = out UInt(7 bits)
    val logitData     = in  SInt(24 bits)
    val maxLogit      = in  SInt(24 bits)

    // Configuration
    val invTemp       = in  UInt(16 bits)
    val seedLoad      = in  Bool()
    val seedVal       = in  UInt(32 bits)

    // Handshake
    val start         = in  Bool()
    val done          = out Bool()

    // Result
    val selectedToken = out UInt(7 bits)
  }

  // Make signals visible in simulation
  io.logitAddr.simPublic()

  object State extends SpinalEnum {
    val IDLE, PASS1, THRESHOLD, PASS2, DONE = newElement()
  }

  val state = RegInit(State.IDLE); state.simPublic()

  // LFSR (persists across calls, loadable via seedLoad/seedVal)
  val lfsr = Reg(UInt(32 bits)) init 0xDEADBEEFL; lfsr.simPublic()

  // Step counter (0..130): step 0 is readSync fill, steps 1..128 feed stage 1,
  // steps 129..130 drain the pipeline
  val step          = Reg(UInt(9 bits)) init 0; step.simPublic()
  val totalProb     = Reg(UInt(24 bits)) init 0; totalProb.simPublic()
  val cumSum        = Reg(UInt(24 bits)) init 0; cumSum.simPublic()
  val threshold     = Reg(UInt(24 bits)) init 0; threshold.simPublic()
  val selectedToken = Reg(UInt(7 bits)) init 127; selectedToken.simPublic()
  val found         = RegInit(False)

  // Pipeline registers: break 27-level combinational path into 4 stages.
  // Stage 1 → pipe1: diff and scale
  val pipe1_diffScaled = Reg(SInt(32 bits)) init 0
  val pipe1_valid = RegInit(False)
  val pipe1_logitIdx = Reg(UInt(7 bits)) init 0
  // Stage 2a → pipe2a: DSP multiply result (breaks DSP → CARRY4 chain path)
  val pipe2a_shifted = Reg(SInt(32 bits)) init 0
  val pipe2a_valid = RegInit(False)
  val pipe2a_logitIdx = Reg(UInt(7 bits)) init 0
  // Stage 2b → pipe2: prob computation from registered DSP output
  val pipe2_prob = Reg(UInt(16 bits)) init 0
  val pipe2_valid = RegInit(False)
  val pipe2_logitIdx = Reg(UInt(7 bits)) init 0  // for PASS2 token selection

  // Drive logit address: during passes, addr = step when step < 128
  io.logitAddr := step.resize(7 bits) // wraps naturally for step >= 128
  io.done := False
  io.selectedToken := selectedToken

  // ---- Stage 1: diff computation (combinational from BRAM readSync) ----
  val diff = (io.logitData.resize(32 bits) - io.maxLogit.resize(32 bits))
  val LOGIT_SHIFT = 12
  val diffScaled = (diff >> LOGIT_SHIFT).resize(32 bits)

  // ---- Stage 2a: DSP multiply (from pipe1 registers) ----
  val invTempSigned = io.invTemp.resize(32 bits).asSInt
  val diffTimesInvTemp = pipe1_diffScaled * invTempSigned
  val shifted2a = (diffTimesInvTemp >> 8).resize(32 bits)

  // ---- Stage 2b: prob computation (from pipe2a registers, no DSP in path) ----
  // Optimized: bit manipulation replaces 32-bit add + compare (9 logic levels → 3).
  // prob = clamp(256 + shifted, 0, 256):
  //   shifted >= 0:     prob = 256
  //   shifted ∈ [-256, -1]: prob = shifted[7:0] (= 256+shifted, just wiring!)
  //   shifted < -256:   prob = 0
  val isNonNeg = !pipe2a_shifted.msb
  val inRange = pipe2a_shifted(30 downto 8).andR // True iff shifted ∈ [-256, -1] (given negative)
  val prob = UInt(16 bits)
  when(isNonNeg) {
    prob := U(256, 16 bits)
  } elsewhen(inRange) {
    prob := pipe2a_shifted(7 downto 0).asUInt.resize(16 bits)
  } otherwise {
    prob := U(0, 16 bits)
  }

  // ---- Pipeline register capture (always active, valid flags control usage) ----
  val inPass = (state === State.PASS1 || state === State.PASS2)
  // Stage 1 output is valid when readSync has delivered data (step >= 1, step <= 128)
  val stage1Valid = inPass && step >= 1 && step <= 128

  // Pipe1: capture diffScaled and valid flag
  pipe1_diffScaled := diffScaled
  pipe1_valid := stage1Valid
  pipe1_logitIdx := (step - 1).resize(7 bits) // logit index for this data

  // Pipe2a: capture DSP output (breaks DSP → CARRY4 chain timing path)
  pipe2a_shifted := shifted2a
  pipe2a_valid := pipe1_valid
  pipe2a_logitIdx := pipe1_logitIdx

  // Pipe2b: capture prob and valid flag (1 cycle behind pipe2a)
  pipe2_prob := prob
  pipe2_valid := pipe2a_valid
  pipe2_logitIdx := pipe2a_logitIdx

  switch(state) {
    is(State.IDLE) {
      when(io.start) {
        state := State.PASS1
        step := 0
        totalProb := 0
        found := False
        selectedToken := 127 // fallback
        pipe1_valid := False
        pipe2a_valid := False
        pipe2_valid := False
      }
    }

    is(State.PASS1) {
      // Stage 3: accumulate totalProb from pipeline output
      when(pipe2_valid) {
        totalProb := (totalProb + pipe2_prob.resize(24 bits)).resize(24 bits)
      }

      // Advance step (0..130)
      when(step <= 128) {
        step := step + 1
      }

      // Done when pipeline fully drained (step > 128, no more valid data in pipeline)
      when(step > 128 && !pipe1_valid && !pipe2a_valid && !pipe2_valid) {
        state := State.THRESHOLD
      }
    }

    is(State.THRESHOLD) {
      // Advance LFSR (Galois LFSR with polynomial 0xD000_0001)
      val bit = lfsr(0)
      val shiftedLfsr = lfsr |>> 1
      val newLfsr = UInt(32 bits)
      when(bit) {
        newLfsr := shiftedLfsr ^ U(0xD0000001L, 32 bits)
      } otherwise {
        newLfsr := shiftedLfsr
      }
      lfsr := newLfsr

      // threshold = (newLfsr[15:0] * totalProb) >> 16
      val lfsrLow = newLfsr(15 downto 0).resize(32 bits)
      val totalProbWide = totalProb.resize(32 bits)
      val product = lfsrLow * totalProbWide
      threshold := (product >> 16).resize(24 bits)

      state := State.PASS2
      step := 0
      cumSum := 0
      found := False
      pipe1_valid := False
      pipe2a_valid := False
      pipe2_valid := False
    }

    is(State.PASS2) {
      // Stage 3: accumulate cumSum and check threshold
      when(pipe2_valid && !found) {
        val newCumSum = (cumSum + pipe2_prob.resize(24 bits)).resize(24 bits)
        cumSum := newCumSum

        when(newCumSum > threshold) {
          selectedToken := pipe2_logitIdx
          found := True
        }
      }

      // Advance step (0..130)
      when(step <= 128) {
        step := step + 1
      }

      // Done when pipeline fully drained
      when(step > 128 && !pipe1_valid && !pipe2a_valid && !pipe2_valid) {
        state := State.DONE
      }
    }

    is(State.DONE) {
      io.done := True
      state := State.IDLE
    }
  }

  // Seed loading takes priority over FSM LFSR advance (last assignment wins)
  when(io.seedLoad) {
    lfsr := io.seedVal
  }
}
