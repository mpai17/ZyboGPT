package zybogpt

import spinal.core._
import spinal.core.sim._
import spinal.lib._

/** Single transformer decoder layer: RMSNorm + Attention + Residual + RMSNorm + FFN + Residual.
  *
  * Orchestrates the attention and FFN sub-modules with shared compute resources.
  */
case class TransformerLayer(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val io = new Bundle {
    // Input/output activations
    val x = in Vec (SInt(8 bits), config.dModel)
    val position = in UInt (8 bits)
    val layerIdx = in UInt (log2Up(config.nLayers) bits)
    val start = in Bool ()
    val result = out Vec (SInt(8 bits), config.dModel)
    val done = out Bool ()

    // Shared TDot array interface
    val tdotX = out Vec (SInt(8 bits), config.dModel)
    val tdotWeightAddr = out UInt (16 bits)
    val tdotStart = out Bool ()
    val tdotResult = in Vec (SInt(24 bits), config.numTDots)
    val tdotDone = in Bool ()

    // Shared MAC array interface
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

    // RMSNorm gamma read interface (reads from external normGammaMem)
    // gammaAddr is the FULL address into normGammaMem (base + local offset)
    val gammaAddr = out UInt (log2Up((config.nLayers * 2 + 1) * config.dModel) bits)
    val gammaData = in SInt (16 bits)

    // Softmax score interface (Attention writes to scoreMem, Softmax reads via address)
    val smScoreAddr = in UInt (7 bits)
    val smScoreData = out SInt (16 bits)
  }

  object State extends SpinalEnum {
    val IDLE, ATTN_NORM, ATTENTION, ATTN_RESIDUAL,
        FF_NORM, FEEDFORWARD, FF_RESIDUAL, DONE = newElement()
  }

  val state = RegInit(State.IDLE); state.simPublic()

  // Instantiate sub-modules
  val rmsNorm = RMSNorm(config)
  val attention = Attention(config)
  val ffn = FeedForward(config)

  // Activation buffers
  val xBuf = Vec(Reg(SInt(8 bits)), config.dModel); xBuf.simPublic()
  val normedBuf = Vec(Reg(SInt(8 bits)), config.dModel); normedBuf.simPublic()
  val resultBuf = Vec(Reg(SInt(8 bits)), config.dModel); resultBuf.simPublic()

  val gammaAddrBits = log2Up((config.nLayers * 2 + 1) * config.dModel)

  // Default connections
  rmsNorm.io.x := xBuf
  rmsNorm.io.gammaData := io.gammaData
  rmsNorm.io.start := False

  // Gamma address routing: RMSNorm outputs a local offset (0..dim-1),
  // we add the appropriate base for attn norm vs ff norm
  val normGammaBase = Reg(UInt(gammaAddrBits bits)) init 0
  io.gammaAddr := (normGammaBase + rmsNorm.io.gammaAddr.resize(gammaAddrBits bits)).resize(gammaAddrBits bits)

  attention.io.x := normedBuf
  attention.io.position := io.position
  attention.io.layerIdx := io.layerIdx
  attention.io.start := False

  // Wire through shared resources from attention
  io.tdotX := attention.io.tdotX
  io.tdotWeightAddr := attention.io.tdotWeightAddr
  io.tdotStart := attention.io.tdotStart
  attention.io.tdotResult := io.tdotResult
  attention.io.tdotDone := io.tdotDone

  io.macA := attention.io.macA
  io.macB := attention.io.macB
  io.macValid := attention.io.macValid
  io.macClear := attention.io.macClear
  io.macStart := attention.io.macStart
  attention.io.macResults := io.macResults
  attention.io.macDone := io.macDone

  io.kvWriteK := attention.io.kvWriteK
  io.kvWriteV := attention.io.kvWriteV
  io.kvWriteHead := attention.io.kvWriteHead
  io.kvWriteEn := attention.io.kvWriteEn
  io.kvReadPos := attention.io.kvReadPos
  io.kvReadHead := attention.io.kvReadHead
  io.kvReadEn := attention.io.kvReadEn
  attention.io.kvReadK := io.kvReadK
  attention.io.kvReadV := io.kvReadV
  attention.io.kvReadValid := io.kvReadValid

  // Softmax instantiated here
  val softmax = Softmax(config)

  // Softmax reads scores from Attention's scoreMem via address/data interface
  attention.io.smScoreAddr := softmax.io.scoreAddr
  softmax.io.scoreData := attention.io.smScoreData
  softmax.io.len := attention.io.smLen
  softmax.io.start := attention.io.smStart
  // Prob read via address/data interface (replaces Vec)
  softmax.io.probAddr := attention.io.smProbAddr
  attention.io.smProbData := softmax.io.probData
  attention.io.smDone := softmax.io.done

  // Also expose score interface to top level (for potential external access)
  io.smScoreData := attention.io.smScoreData

  ffn.io.x := normedBuf
  ffn.io.layerIdx := io.layerIdx
  ffn.io.start := False
  ffn.io.tdotResult := io.tdotResult
  ffn.io.tdotDone := io.tdotDone
  // FFN TDot wiring will override attention's when active

  for (i <- 0 until config.dModel) {
    io.result(i) := resultBuf(i)
  }
  io.done := False

  switch(state) {
    is(State.IDLE) {
      when(io.start) {
        for (i <- 0 until config.dModel) {
          xBuf(i) := io.x(i)
        }
        // Set gamma base for attn norm: layerIdx * 2 * dModel
        normGammaBase := (io.layerIdx.resize(gammaAddrBits bits) *
          U(2 * config.dModel, gammaAddrBits bits))
          .resize(gammaAddrBits bits)
        state := State.ATTN_NORM
      }
    }

    is(State.ATTN_NORM) {
      rmsNorm.io.start := True
      when(rmsNorm.io.done) {
        for (i <- 0 until config.dModel) {
          normedBuf(i) := rmsNorm.io.y(i)
        }
        state := State.ATTENTION
      }
    }

    is(State.ATTENTION) {
      attention.io.start := True
      when(attention.io.done) {
        state := State.ATTN_RESIDUAL
      }
    }

    is(State.ATTN_RESIDUAL) {
      // x = x + attn_out
      for (i <- 0 until config.dModel) {
        val sum = xBuf(i).resize(16 bits) + attention.io.result(i).resize(16 bits)
        xBuf(i) := SatInt8(sum)
      }
      // Set gamma base for ff norm: layerIdx * 2 * dModel + dModel
      normGammaBase := (io.layerIdx.resize(gammaAddrBits bits) *
        U(2 * config.dModel, gammaAddrBits bits) +
        U(config.dModel, gammaAddrBits bits))
        .resize(gammaAddrBits bits)
      state := State.FF_NORM
    }

    is(State.FF_NORM) {
      rmsNorm.io.start := True
      when(rmsNorm.io.done) {
        for (i <- 0 until config.dModel) {
          normedBuf(i) := rmsNorm.io.y(i)
        }
        state := State.FEEDFORWARD
      }
    }

    is(State.FEEDFORWARD) {
      ffn.io.start := True
      // Override TDot wiring for FFN
      io.tdotX := ffn.io.tdotX
      io.tdotWeightAddr := ffn.io.tdotWeightAddr
      io.tdotStart := ffn.io.tdotStart
      ffn.io.tdotResult := io.tdotResult
      ffn.io.tdotDone := io.tdotDone

      when(ffn.io.done) {
        state := State.FF_RESIDUAL
      }
    }

    is(State.FF_RESIDUAL) {
      // x = x + ff_out
      for (i <- 0 until config.dModel) {
        val sum = xBuf(i).resize(16 bits) + ffn.io.result(i).resize(16 bits)
        resultBuf(i) := SatInt8(sum)
      }
      state := State.DONE
    }

    is(State.DONE) {
      io.done := True
      state := State.IDLE
    }
  }
}
