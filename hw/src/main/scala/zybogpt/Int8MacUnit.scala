package zybogpt

import spinal.core._
import spinal.lib._

/** INT8 multiply-accumulate unit.
  *
  * Has use_dsp="yes" attribute, but Vivado may use LUTs for INT8×INT8.
  */
case class Int8MacUnit() extends Component {
  val io = new Bundle {
    val a = in SInt (8 bits)
    val b = in SInt (8 bits)
    val acc_in = in SInt (24 bits)
    val acc_en = in Bool ()
    val clear = in Bool ()
    val result = out SInt (24 bits)
  }

  // Input pipeline registers (absorbed into DSP48E1 AREG/BREG if DSP is inferred).
  // Zeroed on clear to prevent stale pipeline data from contaminating first accumulation.
  val aReg = Reg(SInt(8 bits)) init 0
  val bReg = Reg(SInt(8 bits)) init 0

  // Product: INT8 x INT8 -> INT16 (force DSP to save LUTs)
  val product = (aReg * bReg).addAttribute("use_dsp", "yes") // 16 bits

  // Accumulator
  val acc = Reg(SInt(24 bits)) init 0
  when(io.clear) {
    acc := 0
    aReg := 0
    bReg := 0
  }.elsewhen(io.acc_en) {
    aReg := io.a
    bReg := io.b
    acc := acc + product.resize(24 bits)
  } otherwise {
    aReg := io.a
    bReg := io.b
  }

  io.result := acc
}
