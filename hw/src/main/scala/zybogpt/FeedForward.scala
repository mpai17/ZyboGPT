package zybogpt

import spinal.core._
import spinal.core.sim._
import spinal.lib._

/** Feed-forward network: down(relu(up(x))).
  *
  * Up projection: (d_model=64) -> (d_ff=256) via ternary TDot
  * ReLU: merged into UP_PROJ writes (max(0, result))
  * Down projection: (d_ff=256) -> (d_model=64) via ternary TDot
  *
  * upBuf uses Mem() (BRAM) instead of Vec(Reg()) to avoid 256-to-1 mux trees.
  * ReLU is applied inline during STORE_UP writes, eliminating the RELU state.
  * DOWN_PROJ pre-reads 64-element slices from upMem into a staging register.
  */
case class FeedForward(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val io = new Bundle {
    // Input
    val x = in Vec (SInt(8 bits), config.dModel)
    val layerIdx = in UInt (log2Up(config.nLayers) bits)
    val start = in Bool ()

    // Output
    val result = out Vec (SInt(8 bits), config.dModel)
    val done = out Bool ()

    // TDot array interface (shared)
    val tdotX = out Vec (SInt(8 bits), config.dModel)
    val tdotWeightAddr = out UInt (16 bits)
    val tdotStart = out Bool ()
    val tdotResult = in Vec (SInt(24 bits), config.numTDots)
    val tdotDone = in Bool ()
  }

  object State extends SpinalEnum {
    val IDLE, UP_PROJ, STORE_UP, LOAD_SLICE, DOWN_PROJ, DONE = newElement()
  }

  val state = RegInit(State.IDLE); state.simPublic()

  // BRAM for up-projection results (replaces Vec(Reg()) to avoid mux trees)
  val upMem = Mem(SInt(8 bits), config.dFf)

  // Capture register for TDot results (latched before serialized BRAM writes).
  // Stores SatInt8+ReLU result (8 bits) to avoid 32-to-1 mux → SatInt8 → BRAM critical path.
  val tdotCapture = Vec(Reg(SInt(8 bits)), config.numTDots)

  // Staging register for DOWN_PROJ input slices (pre-read from upMem)
  val tdotSliceBuf = Vec(Reg(SInt(8 bits)), config.dModel)

  val resultBuf = Vec(Reg(SInt(8 bits)), config.dModel); resultBuf.simPublic()
  val accumBuf = Vec(Reg(SInt(24 bits)) init 0, config.dModel) // For partial product accumulation
  val batchIdx = Reg(UInt(4 bits)) init 0
  val passIdx = Reg(UInt(3 bits)) init 0 // For 256/64=4 accumulation passes
  val writeIdx = Reg(UInt(log2Up(config.numTDots) bits)) init 0 // For serialized BRAM writes
  val readIdx = Reg(UInt(log2Up(config.dModel) bits)) init 0 // For pre-reading upMem slices

  // Read port for upMem (used in LOAD_SLICE)
  val upMemAddr = UInt(log2Up(config.dFf) bits)
  upMemAddr := 0 // default
  val upMemReadData = upMem.readSync(upMemAddr)

  // Weight address offsets for FFN (per-block packed, word-aligned)
  val attnWeightBytes = 4 * config.attnProjPackedBytes
  val upWeightBytes = config.ffUpPackedBytes
  val layerStride = config.layerPackedBytes

  // Registered weight addresses: eliminate combinational multiply chains
  // (batchIdx/passIdx → ×bytesPerTdotLoad → carry chain → BRAM, 8+ logic levels).
  // Initialized at state transitions, incremented by bytesPerTdotLoad each batch/pass.
  val upProjAddr = Reg(UInt(16 bits)) init 0
  val downProjAddr = Reg(UInt(16 bits)) init 0

  // Defaults
  io.tdotX := io.x
  io.tdotWeightAddr := 0
  io.tdotStart := False
  for (i <- 0 until config.dModel) {
    io.result(i) := resultBuf(i)
  }
  io.done := False

  switch(state) {
    is(State.IDLE) {
      when(io.start) {
        state := State.UP_PROJ
        batchIdx := 0
        // Pre-compute UP_PROJ base address (layerIdx multiply has full cycle to settle)
        upProjAddr := ((io.layerIdx.resize(16 bits) * U(layerStride, 16 bits)).addAttribute("use_dsp", "no")
          + U(attnWeightBytes, 16 bits)).resize(16 bits)
      }
    }

    is(State.UP_PROJ) {
      // Up projection: 64 -> 256, done in batches of 32
      io.tdotX := io.x
      // Use registered address (pre-computed, eliminates 8-level combinational multiply chain)
      io.tdotWeightAddr := upProjAddr
      io.tdotStart := True

      when(io.tdotDone) {
        // Advance address for next batch
        upProjAddr := upProjAddr + U(config.bytesPerTdotLoad, 16 bits)
        // Capture TDot results with inline SatInt8+ReLU (moved from STORE_UP to break
        // 32-to-1 mux → SatInt8 → ReLU → BRAM critical path; each capture has its own
        // dedicated result wire so no mux is needed here)
        for (i <- 0 until config.numTDots) {
          val shifted = SatInt8(io.tdotResult(i) >> 4)
          when(shifted < 0) {
            tdotCapture(i) := S(0, 8 bits)
          } otherwise {
            tdotCapture(i) := shifted
          }
        }
        writeIdx := 0
        state := State.STORE_UP
      }
    }

    is(State.STORE_UP) {
      // Serialize 32 writes to upMem (SatInt8+ReLU already applied during capture).
      val globalIdx = (batchIdx * U(config.numTDots) + writeIdx.resize(log2Up(config.dFf) bits))
        .resize(log2Up(config.dFf) bits)
      upMem.write(globalIdx, tdotCapture(writeIdx))

      writeIdx := writeIdx + 1
      when(writeIdx === (config.numTDots - 1)) {
        batchIdx := batchIdx + 1
        when(batchIdx === (config.dFf / config.numTDots - 1)) {
          // All up-projection done, pre-read first slice for DOWN_PROJ
          state := State.LOAD_SLICE
          batchIdx := 0
          passIdx := 0
          readIdx := 0
          // Prime read-ahead for upMem[0]
          upMemAddr := U(0, log2Up(config.dFf) bits)
          // Pre-compute initial DOWN_PROJ weight address (passIdx=0, batchIdx=0)
          downProjAddr := ((io.layerIdx.resize(16 bits) * U(layerStride, 16 bits)).addAttribute("use_dsp", "no")
            + U(attnWeightBytes + upWeightBytes, 16 bits)).resize(16 bits)
        } otherwise {
          // More UP_PROJ batches to process
          state := State.UP_PROJ
        }
      }
    }

    is(State.LOAD_SLICE) {
      // Pre-read 64 elements from upMem into tdotSliceBuf.
      // Transition already primed upMemAddr, so upMemReadData has data for readIdx=0.
      // Each cycle: store current data, request next address.
      val baseAddr = passIdx.resize(log2Up(config.dFf) bits) * U(config.dModel, log2Up(config.dFf) bits)
      tdotSliceBuf(readIdx.resize(log2Up(config.dModel) bits)) := upMemReadData

      // Read-ahead for next element
      upMemAddr := (baseAddr + readIdx.resize(log2Up(config.dFf) bits) + U(1, log2Up(config.dFf) bits)).resize(log2Up(config.dFf) bits)

      readIdx := readIdx + 1
      when(readIdx === (config.dModel - 1)) {
        state := State.DOWN_PROJ
      }
    }

    is(State.DOWN_PROJ) {
      // Down projection: 256 -> 64
      // Each output[i] = dot(upBuf[256], downWeight[i][256])
      // TDot is 64-wide, so each output needs 256/64=4 passes accumulated.
      // 32 TDots compute 32 outputs in parallel, so 64/32=2 batches.
      // Total: 4 passes * 2 batches = 8 TDot invocations.

      // Feed the pre-loaded 64-element slice
      for (i <- 0 until config.dModel) {
        io.tdotX(i) := tdotSliceBuf(i)
      }

      // Use registered address (pre-computed, eliminates 10+ level combinational multiply chain)
      io.tdotWeightAddr := downProjAddr
      io.tdotStart := True

      when(io.tdotDone) {
        // Advance address for next TDot block
        downProjAddr := downProjAddr + U(config.bytesPerTdotLoad, 16 bits)
        // Accumulate partial dot products using compile-time constant indexing.
        // batchIdx selects which half of accumBuf (0..31 or 32..63).
        // Unrolling over batch values avoids dynamic indexing mux trees.
        for (batch <- 0 until config.dModel / config.numTDots) {
          when(batchIdx === batch) {
            for (i <- 0 until config.numTDots) {
              val idx = batch * config.numTDots + i // compile-time constant
              when(passIdx === 0) {
                accumBuf(idx) := io.tdotResult(i)
              } otherwise {
                accumBuf(idx) := accumBuf(idx) + io.tdotResult(i)
              }
            }
          }
        }

        passIdx := passIdx + 1
        when(passIdx === (config.dFf / config.dModel - 1)) {
          passIdx := 0
          batchIdx := batchIdx + 1

          when(batchIdx === (config.dModel / config.numTDots - 1)) {
            state := State.DONE
          } otherwise {
            // Pre-load next slice (passIdx wraps to 0 for new batch, same slice needed)
            // Actually, passIdx=0 again means we need slice 0, but batchIdx changed.
            // The slice depends on passIdx, not batchIdx. passIdx resets to 0,
            // so we need slice 0 again. But wait - we already have it if we
            // just increment batchIdx and keep passIdx=0.
            // Actually no: the DOWN_PROJ loop over passIdx means we advance passIdx
            // for each TDot call, so next time passIdx=0 we DO need slice 0 again.
            // Since passIdx resets to 0, we need to re-load slice 0.
            readIdx := 0
            upMemAddr := U(0, log2Up(config.dFf) bits)
            state := State.LOAD_SLICE
          }
        } otherwise {
          // Next pass: need next 64-element slice
          val nextPassIdx = passIdx + 1
          readIdx := 0
          upMemAddr := (nextPassIdx.resize(log2Up(config.dFf) bits) * U(config.dModel, log2Up(config.dFf) bits)).resize(log2Up(config.dFf) bits)
          state := State.LOAD_SLICE
        }
      }
    }

    is(State.DONE) {
      // Quantize accumulated results to INT8 (accumBuf now has all passes settled)
      for (i <- 0 until config.dModel) {
        resultBuf(i) := SatInt8(accumBuf(i) >> 4)
      }
      io.done := True
      state := State.IDLE
    }
  }
}
