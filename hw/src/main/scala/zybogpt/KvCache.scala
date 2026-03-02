package zybogpt

import spinal.core._
import spinal.lib._

/** KV cache using dual-port BRAM circular buffer.
  *
  * Stores K and V vectors for each attention head in each layer.
  * Organized as circular buffer with write pointer advancing each token.
  *
  * Memory layout per head:
  *   K: ctx_len * head_dim INT8 values = 128 * 32 = 4096 bytes
  *   V: ctx_len * head_dim INT8 values = 128 * 32 = 4096 bytes
  *
  * Total: n_layers * n_heads * 2 * 4096 = 2 * 2 * 2 * 4096 = 32 KB (8 BRAM36)
  */
case class KvCache(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val io = new Bundle {
    // Write new K,V for current token
    val writeK = in Vec (SInt(8 bits), config.headDim)
    val writeV = in Vec (SInt(8 bits), config.headDim)
    val writeLayer = in UInt (log2Up(config.nLayers) bits)
    val writeHead = in UInt (log2Up(config.nHeads) bits)
    val writePos = in UInt (log2Up(config.ctxLen) bits) // Position in sequence
    val writeEn = in Bool ()

    // Read K,V for attention computation
    val readPos = in UInt (log2Up(config.ctxLen) bits)
    val readLayer = in UInt (log2Up(config.nLayers) bits)
    val readHead = in UInt (log2Up(config.nHeads) bits)
    val readEn = in Bool ()
    val readK = out Vec (SInt(8 bits), config.headDim)
    val readV = out Vec (SInt(8 bits), config.headDim)
    val readValid = out Bool ()

    // Current sequence length
    val seqLen = in UInt (8 bits)
  }

  // BRAM storage: total entries = nLayers * nHeads * ctxLen * headDim * 2 (K+V)
  // Organized as INT8 elements
  val totalEntries = config.nLayers * config.nHeads * config.ctxLen * config.headDim
  val kMem = Mem(SInt(8 bits), totalEntries)
  val vMem = Mem(SInt(8 bits), totalEntries)

  // Address computation: (layer * nHeads + head) * ctxLen * headDim + pos * headDim + dim
  val addrWidth = log2Up(totalEntries)

  // Use bit shifts instead of multiplications.
  // layer * (nHeads * ctxLen * headDim) = layer * 8192 = layer << 13
  // head * (ctxLen * headDim) = head * 4096 = head << 12
  // pos * headDim = pos * 32 = pos << 5
  def computeBaseAddr(layer: UInt, head: UInt, pos: UInt): UInt = {
    val layerOff = (layer.resize(addrWidth bits) << 13).resize(addrWidth bits)
    val headOff = (head.resize(addrWidth bits) << 12).resize(addrWidth bits)
    val posOff = (pos.resize(addrWidth bits) << 5).resize(addrWidth bits)
    (layerOff | headOff | posOff)
  }

  // Write: serialize headDim elements over multiple cycles.
  // Write directly from io.writeK/V (held stable by Attention during STORE_KV),
  // eliminating writeBufK/V registers and their parallel capture logic.
  val writeStep = Reg(UInt(log2Up(config.headDim) bits)) init 0
  val writing = RegInit(False)
  val writeBaseAddr = Reg(UInt(log2Up(totalEntries) bits))

  when(io.writeEn && !writing) {
    writing := True
    writeStep := 0
    writeBaseAddr := computeBaseAddr(io.writeLayer, io.writeHead, io.writePos)
  }

  when(writing) {
    kMem.write(writeBaseAddr + writeStep.resize(log2Up(totalEntries) bits), io.writeK(writeStep))
    vMem.write(writeBaseAddr + writeStep.resize(log2Up(totalEntries) bits), io.writeV(writeStep))
    writeStep := writeStep + 1
    when(writeStep === (config.headDim - 1)) {
      writing := False
    }
  }

  // Read: serialize headDim elements
  val readStep = Reg(UInt(log2Up(config.headDim) bits)) init 0
  val reading = RegInit(False)
  val readBufK = Vec(Reg(SInt(8 bits)), config.headDim)
  val readBufV = Vec(Reg(SInt(8 bits)), config.headDim)
  val readBaseAddr = Reg(UInt(log2Up(totalEntries) bits))
  val readDone = RegInit(False)

  when(io.readEn && !reading) {
    reading := True
    readStep := 0
    readDone := False
    readBaseAddr := computeBaseAddr(io.readLayer, io.readHead, io.readPos)
  }

  // Read data from BRAM (1-cycle latency for readSync)
  val kReadData = kMem.readSync(readBaseAddr + readStep.resize(addrWidth bits))
  val vReadData = vMem.readSync(readBaseAddr + readStep.resize(addrWidth bits))
  val readStepDelayed = RegNext(readStep)
  val readingDelayed = RegNext(reading, init = False)

  when(reading) {
    readStep := readStep + 1
    when(readStep === (config.headDim - 1)) {
      reading := False
    }
  }

  // Store read data 1 cycle after issuing read (accounts for readSync latency)
  when(readingDelayed) {
    readBufK(readStepDelayed) := kReadData
    readBufV(readStepDelayed) := vReadData
    when(!reading && readStepDelayed === (config.headDim - 1)) {
      readDone := True
    }
  }.otherwise {
    readDone := False
  }

  for (i <- 0 until config.headDim) {
    io.readK(i) := readBufK(i)
    io.readV(i) := readBufV(i)
  }
  io.readValid := readDone
}
