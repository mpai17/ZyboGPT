package zybogpt

import spinal.core._
import spinal.core.sim._
import scala.util.Random

/** Simulation testbench for WeightDecoder.
  *
  * Verifies that 1.6-bit packed ternary bytes are correctly
  * decoded back to 2-bit trit encoding.
  *
  * Packing: 5 trits per byte, base-3 encoding.
  * trit values {-1,0,+1} stored as {0,1,2}
  * packed = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
  *
  * Output encoding: 00=0, 01=+1, 11=-1
  */
object WeightDecoderSim extends App {
  val seed = 0xBEEF_0005L
  println(s"WeightDecoder seed: 0x${seed.toHexString}")
  val rng = new Random(seed)

  SimConfig.withWave.compile(WeightDecoder()).doSim { dut =>
    dut.clockDomain.forkStimulus(period = 10)

    def pack5Trits(trits: Seq[Int]): Int = {
      assert(trits.length == 5)
      val mapped = trits.map(_ + 1) // {-1,0,+1} -> {0,1,2}
      mapped(0) + 3 * mapped(1) + 9 * mapped(2) + 27 * mapped(3) + 81 * mapped(4)
    }

    def expectedEncoding(tritVal: Int): Int = tritVal match {
      case -1 => 3 // 11
      case 0  => 0 // 00
      case 1  => 1 // 01
      case _  => 0
    }

    // Deterministic test cases
    val testCases = Seq(
      ("all zeros", Seq(0, 0, 0, 0, 0)),
      ("all +1", Seq(1, 1, 1, 1, 1)),
      ("all -1", Seq(-1, -1, -1, -1, -1)),
      ("mixed 1", Seq(-1, 0, 1, 0, -1)),
      ("mixed 2", Seq(1, -1, 1, -1, 1)),
      ("single +1", Seq(1, 0, 0, 0, 0)),
      ("single -1 at end", Seq(0, 0, 0, 0, -1)),
    )

    for ((desc, trits) <- testCases) {
      val packed = pack5Trits(trits)
      println(s"Test: $desc -> packed=$packed (0x${packed.toHexString})")

      dut.io.packed #= packed
      dut.io.valid_in #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.valid_in #= false
      // Wait for 5-stage pipeline to produce valid output
      while (!dut.io.valid_out.toBoolean) {
        dut.clockDomain.waitRisingEdge()
      }

      for (i <- 0 until 5) {
        val expected = expectedEncoding(trits(i))
        val actual = dut.io.trits(i).toInt
        val pass = actual == expected
        println(s"  trit[$i]: expected=$expected (trit=${trits(i)}), got=$actual ${if (pass) "OK" else "FAIL"}")
        assert(pass, s"Trit $i mismatch for test '$desc'")
      }
    }

    // Randomized: exhaustive coverage of all 243 valid packed values
    println(s"\nRandomized exhaustive test (all 243 encodings, seed=0x${seed.toHexString}):")
    val indices = rng.shuffle((0 until 243).toList)
    var errors = 0
    for (packed <- indices) {
      // Decode expected trits from packed value
      var rem = packed
      val expectedTrits = new Array[Int](5)
      for (i <- 0 until 5) {
        val digit = rem % 3 // {0,1,2} -> {-1,0,+1}
        expectedTrits(i) = digit - 1
        rem = rem / 3
      }

      dut.io.packed #= packed
      dut.io.valid_in #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.valid_in #= false
      // Wait for 5-stage pipeline to produce valid output
      while (!dut.io.valid_out.toBoolean) {
        dut.clockDomain.waitRisingEdge()
      }

      for (i <- 0 until 5) {
        val expected = expectedEncoding(expectedTrits(i))
        val actual = dut.io.trits(i).toInt
        if (actual != expected) {
          println(s"  FAIL: packed=$packed, trit[$i]: expected=$expected (${expectedTrits(i)}), got=$actual")
          errors += 1
        }
      }
    }
    println(s"  Tested all 243 values, errors: $errors")
    assert(errors == 0, s"$errors decoding errors detected")

    println("\nAll WeightDecoder tests passed!")
  }
}
