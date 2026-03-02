package zybogpt

import spinal.core._
import spinal.lib._

/** Hardware configuration for ZyboGPT accelerator.
  *
  * All model dimensions and compute array sizes are parameterized here.
  */
case class ZyboGPTHwConfig(
    // Model dimensions
    vocabSize: Int = 128,
    dModel: Int = 64,
    nHeads: Int = 2,
    nLayers: Int = 2,
    dFf: Int = 256,
    ctxLen: Int = 128,

    // Compute array
    numTDots: Int = 32, // Number of parallel ternary dot-product units
    numDspMacs: Int = 8, // Number of parallel INT8 MACs
    tdotWidth: Int = 64, // Dot product vector width (= dModel)

    // Clock
    clockMhz: Double = 150.0,

    // Quantization
    actBits: Int = 8, // INT8 activations
    accumBits: Int = 24, // Accumulator width
    fracBits: Int = 10, // Fixed-point fractional bits (Q5.10)
    normBits: Int = 16 // RMSNorm weight bits
) {
  val headDim: Int = dModel / nHeads

  // Ternary weight packing: 5 trits per byte
  val tritsPerByte: Int = 5

  // Weight memory sizes (in ternary parameters, for documentation)
  val ternaryParamsPerLayer: Int = {
    val attnParams = 4 * dModel * dModel // Q, K, V, O
    val ffnParams = dModel * dFf + dFf * dModel // up + down
    attnParams + ffnParams
  }
  val totalTernaryParams: Int = ternaryParamsPerLayer * nLayers

  // Per-TDot-load-block byte size (word-aligned for BRAM reads).
  // Each TDot load reads numTDots * tdotWidth = 2048 trits.
  // Packed: ceil(2048/5) = 410 bytes, word-aligned to 412 bytes.
  val bytesPerTdotLoad: Int = {
    val trits = numTDots * tdotWidth // 2048
    val dataBytes = (trits + tritsPerByte - 1) / tritsPerByte // 410
    ((dataBytes + 3) / 4) * 4 // 412 (word-aligned)
  }

  // Number of TDot load blocks per projection type
  val attnProjBlocks: Int = dModel / numTDots // 64/32 = 2
  val ffUpBlocks: Int = dFf / numTDots // 256/32 = 8
  val ffDownBlocks: Int = (dModel / numTDots) * (dFf / dModel) // 2*4 = 8

  // Packed byte sizes per projection type
  val attnProjPackedBytes: Int = attnProjBlocks * bytesPerTdotLoad // 824
  val ffUpPackedBytes: Int = ffUpBlocks * bytesPerTdotLoad // 3296
  val ffDownPackedBytes: Int = ffDownBlocks * bytesPerTdotLoad // 3296

  // Per-layer byte stride in BRAM
  val layerPackedBytes: Int = 4 * attnProjPackedBytes + ffUpPackedBytes + ffDownPackedBytes // 9888

  // Total packed weight size in BRAM
  val totalTernaryBytes: Int = layerPackedBytes * nLayers // 19776

  // BRAM usage (36Kb blocks)
  val weightBramBlocks: Int = (totalTernaryBytes + 4095) / 4096
  val kvCacheBramBlocks: Int = nLayers * nHeads * 2 // K+V per head per layer = 8
  val actBramBlocks: Int = 4 // Double-buffered activation storage

  // Cycles per operation
  val cyclesPerTernaryMatvec: Int = dModel / numTDots // 64/32 = 2 cycles for d_model width
  val cyclesPerDspMatvec: Int = ctxLen / numDspMacs // For attention score computation

  require(dModel % nHeads == 0, "dModel must be divisible by nHeads")
  require(dModel % numTDots == 0 || numTDots >= dModel, "numTDots must divide or exceed dModel")
}

/** Default configuration for Zybo Z7-10 (xc7z010). */
object ZyboGPTHwConfig {
  def default: ZyboGPTHwConfig = ZyboGPTHwConfig()
}

/** Saturating INT8 clamp helper for fixed-point datapath. */
object SatInt8 {
  /** Saturating clamp: clamp wider SInt to [-128, 127], return SInt(8 bits).
    *
    * Used instead of .resize(8 bits) at embedding, TDot, residual, and attention
    * outputs to match the training forward pass (which uses torch.clamp(-128,127)).
    * Prevents wrap-around artifacts that destroy learned representations.
    */
  def apply(value: SInt): SInt = {
    val result = SInt(8 bits)
    when(value > 127) {
      result := S(127, 8 bits)
    } elsewhen(value < -128) {
      result := S(-128, 8 bits)
    } otherwise {
      result := value.resize(8 bits)
    }
    result
  }
}
