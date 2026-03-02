package zybogpt

import spinal.core._
import spinal.lib._

/** Array of 8 INT8 MACs for attention score computation.
  *
  * Used for:
  * 1. Q @ K^T: INT8 x INT8 -> INT24 (attention scores)
  * 2. attn_weights @ V: UINT8 x INT8 -> INT24 (weighted value sum)
  *
  * Operates in streaming mode: feeds vectors element-by-element,
  * accumulating products over the inner dimension.
  */
case class Int8MacArray(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val numMacs = config.numDspMacs

  val io = new Bundle {
    // Input vectors (streamed element by element)
    val a = in Vec (SInt(8 bits), numMacs) // 8 parallel lanes
    val b = in Vec (SInt(8 bits), numMacs) // 8 parallel lanes
    val valid = in Bool ()
    val clear = in Bool ()

    // Accumulated results
    val results = out Vec (SInt(24 bits), numMacs)
    val done = out Bool ()

    // Control
    val innerDim = in UInt (8 bits) // Length of dot product
    val start = in Bool ()
  }

  val macs = Array.fill(numMacs)(Int8MacUnit())
  val stepCounter = Reg(UInt(8 bits)) init 0
  val running = RegInit(False)

  for (i <- 0 until numMacs) {
    macs(i).io.a := io.a(i)
    macs(i).io.b := io.b(i)
    macs(i).io.acc_in := 0
    macs(i).io.acc_en := io.valid && running
    macs(i).io.clear := io.clear || io.start
    io.results(i) := macs(i).io.result
  }

  when(io.start) {
    running := True
    stepCounter := 0
  }

  when(running && io.valid) {
    stepCounter := stepCounter + 1
    when(stepCounter === io.innerDim - 1) {
      running := False
    }
  }

  io.done := RegNext(running && stepCounter === io.innerDim - 1, init = False)
}
