package zybogpt

import spinal.core._
import spinal.lib._

/** Decodes 1.6-bit packed ternary weights to 2-bit per trit format.
  *
  * TerEffic packing scheme: 5 ternary trits per byte.
  * Encoding: trit values {-1,0,+1} mapped to {0,1,2}.
  * Packed as base-3: b0 + 3*b1 + 9*b2 + 27*b3 + 81*b4 (0-242 range).
  *
  * Output: 2-bit encoding per trit: 00=0, 01=+1, 11=-1.
  *
  * 5-stage pipeline: each divide-by-3 iteration is in its own pipeline stage
  * with registers between stages. This breaks the cascaded DSP multiply chain
  * (r*171 -> q -> r*171 -> ...) that caused timing violations when
  * Flow_AreaOptimized_high mapped all 5 multiplies to cascaded DSP48E1s.
  *
  * Throughput: 1 byte/cycle (after 5-cycle warmup). Latency: 6 cycles.
  * 5 r*171 multiply stages (Vivado may map to DSPs or LUTs depending on strategy).
  */
case class WeightDecoder() extends Component {
  val io = new Bundle {
    val packed = in UInt (8 bits) // Packed byte (5 trits)
    val valid_in = in Bool ()
    val trits = out Vec (Bits(2 bits), 5) // 5 decoded trits
    val valid_out = out Bool ()
  }

  // Helper: one divide-by-3 step. Combinational: q = (r * 171) >> 9, d = r - q*3
  def divmod3(r: UInt): (UInt, UInt) = {
    val q = ((r.resize(16 bits) * U(171, 8 bits)) >> 9).resize(8 bits)
    val d = (r - (q * U(3, 2 bits)).resize(8 bits)).resize(2 bits)
    (q, d)
  }

  // Map {0,1,2} -> {00=0, 01=+1, 11=-1}
  def mapTrit(digit: UInt): Bits = {
    digit.mux(
      U(0, 2 bits) -> B"11", // value -1 encoded as 0
      U(1, 2 bits) -> B"00", // value 0 encoded as 1
      U(2, 2 bits) -> B"01", // value +1 encoded as 2
      default -> B"00"
    )
  }

  // Stage 1: trit 0
  val (q0, d0) = divmod3(io.packed)
  val s1_rem = RegNext(q0)
  val s1_d0 = RegNext(d0)
  val s1_v = RegNext(io.valid_in, init = False)

  // Stage 2: trit 1
  val (q1, d1) = divmod3(s1_rem)
  val s2_rem = RegNext(q1)
  val s2_d0 = RegNext(s1_d0)
  val s2_d1 = RegNext(d1)
  val s2_v = RegNext(s1_v, init = False)

  // Stage 3: trit 2
  val (q2, d2) = divmod3(s2_rem)
  val s3_rem = RegNext(q2)
  val s3_d0 = RegNext(s2_d0)
  val s3_d1 = RegNext(s2_d1)
  val s3_d2 = RegNext(d2)
  val s3_v = RegNext(s2_v, init = False)

  // Stage 4: trit 3
  val (q3, d3) = divmod3(s3_rem)
  val s4_rem = RegNext(q3)
  val s4_d0 = RegNext(s3_d0)
  val s4_d1 = RegNext(s3_d1)
  val s4_d2 = RegNext(s3_d2)
  val s4_d3 = RegNext(d3)
  val s4_v = RegNext(s3_v, init = False)

  // Stage 5: trit 4
  val (_, d4) = divmod3(s4_rem)
  val s5_d0 = RegNext(s4_d0)
  val s5_d1 = RegNext(s4_d1)
  val s5_d2 = RegNext(s4_d2)
  val s5_d3 = RegNext(s4_d3)
  val s5_d4 = RegNext(d4)
  val s5_v = RegNext(s4_v, init = False)

  // Output: map digits to trit encoding
  io.trits(0) := mapTrit(s5_d0)
  io.trits(1) := mapTrit(s5_d1)
  io.trits(2) := mapTrit(s5_d2)
  io.trits(3) := mapTrit(s5_d3)
  io.trits(4) := mapTrit(s5_d4)
  io.valid_out := s5_v
}

/** Bank of weight decoders to feed the TDot array.
  *
  * Decodes enough packed bytes per cycle to fill all TDot units.
  * Each TDot needs 64 trits = 13 packed bytes (13*5=65 trits, 1 wasted).
  * 32 TDots need 32*13 = 416 bytes per cycle from BRAM.
  *
  * In practice, we time-multiplex: decode weights row-by-row.
  * One row = 64 trits = 13 bytes -> 3 cycles at 32-bit BRAM reads.
  */
case class WeightDecoderBank(
    numDecoders: Int = 16, // 16 decoders * 5 trits = 80 trits/cycle
    config: ZyboGPTHwConfig = ZyboGPTHwConfig.default
) extends Component {
  val io = new Bundle {
    // BRAM read data (32-bit words)
    val bramData = in Vec (Bits(32 bits), 4) // 4 BRAM ports * 32 bits = 16 bytes
    val bramValid = in Bool ()

    // Decoded trit output: up to 80 trits per cycle
    val trits = out Vec (Bits(2 bits), numDecoders * 5)
    val tritsValid = out Bool ()
  }

  val decoders = Array.fill(numDecoders)(WeightDecoder())

  // Distribute 16 bytes from 4x32-bit BRAM reads to 16 decoders
  for (i <- 0 until numDecoders) {
    val wordIdx = i / 4
    val byteIdx = i % 4
    decoders(i).io.packed := io.bramData(wordIdx)((byteIdx + 1) * 8 - 1 downto byteIdx * 8).asUInt
    decoders(i).io.valid_in := io.bramValid
  }

  // Flatten trit outputs
  for (i <- 0 until numDecoders) {
    for (j <- 0 until 5) {
      io.trits(i * 5 + j) := decoders(i).io.trits(j)
    }
  }

  io.tritsValid := decoders(0).io.valid_out
}
