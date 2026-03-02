package zybogpt

import spinal.core._
import spinal.lib._

/** Single ternary dot-product unit.
  *
  * Computes dot(x, w) where x is INT8 and w is ternary {-1, 0, +1}.
  * Uses mux-based selection (w=+1 -> +x, w=-1 -> -x, w=0 -> 0)
  * followed by a 3-stage pipelined binary adder tree.
  *
  * Pipeline stage 1: mux selection + first adder level (64 → 32 values)
  * Pipeline stage 2: next 2 adder levels (32 → 8 values)
  * Pipeline stage 3: last 3 adder levels (8 → 1 value)
  *
  * This breaks the 9-level combinational path (mux + 6-level adder tree)
  * into 3 pipeline stages.
  *
  * Input width is configurable (default 64 = dModel).
  * Latency: 3 cycles. Throughput unchanged.
  * Uses 0 DSPs (pure LUT adder tree).
  */
case class TDotUnit(width: Int = 64) extends Component {
  val io = new Bundle {
    val x = in Vec (SInt(8 bits), width) // INT8 activation vector
    val w = in Vec (Bits(2 bits), width) // Ternary weights: 00=0, 01=+1, 11=-1
    val valid_in = in Bool ()
    val result = out SInt (24 bits) // Accumulator output
    val valid_out = out Bool ()
  }

  // Stage 1: Mux selection based on ternary weight
  // Note: negate AFTER resize to 9 bits to handle x=-128 correctly
  // (-(-128) overflows 8-bit signed, but works in 9-bit)
  val muxed = Vec(SInt(9 bits), width)
  for (i <- 0 until width) {
    muxed(i) := io.w(i).mux(
      B"00" -> S(0, 9 bits), // w = 0
      B"01" -> io.x(i).resize(9 bits), // w = +1
      B"11" -> (-(io.x(i).resize(9 bits))), // w = -1
      B"10" -> S(0, 9 bits) // unused encoding, treat as 0
    )
  }

  // Adder tree helper: one level of pairwise addition
  def addPairs(values: Seq[SInt]): Seq[SInt] = {
    values.grouped(2).toSeq.map { pair =>
      if (pair.length == 2) {
        (pair(0).resize(pair(0).getWidth + 1 bits) +
          pair(1).resize(pair(1).getWidth + 1 bits))
      } else {
        pair(0).resize(pair(0).getWidth + 1 bits)
      }
    }
  }

  // Stage 1 (combinational): mux + first adder level: 64 → 32
  val level0 = addPairs(muxed.toSeq) // 64→32, 10 bits each

  // Pipeline register after stage 1
  val level0Reg = level0.map(v => RegNext(v))
  val valid1 = RegNext(io.valid_in, init = False)

  // Stage 2 (combinational): next 2 adder levels: 32 → 16 → 8
  val level1 = addPairs(level0Reg)    // 32→16, 11 bits each
  val level2 = addPairs(level1)       // 16→8, 12 bits each

  // Pipeline register after stage 2
  val level2Reg = level2.map(v => RegNext(v))
  val valid2 = RegNext(valid1, init = False)

  // Stage 3 (combinational): last 3 adder levels: 8 → 4 → 2 → 1
  val level3 = addPairs(level2Reg)    // 8→4, 13 bits each
  val level4 = addPairs(level3)       // 4→2, 14 bits each
  val level5 = addPairs(level4)       // 2→1, 15 bits

  // Output register
  val sum = RegNext(level5.head.resize(24 bits))
  io.result := sum
  io.valid_out := RegNext(valid2, init = False)
}
