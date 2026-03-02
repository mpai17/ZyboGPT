package zybogpt

import spinal.core._
import spinal.core.sim._
import scala.util.Random

/** Simulation testbench for TDotUnit.
  *
  * Verifies ternary dot product against known values and randomized vectors.
  */
object TDotUnitSim extends App {
  val seed = 0xBEEF_0001L
  println(s"TDotUnit seed: 0x${seed.toHexString}")
  val rng = new Random(seed)

  SimConfig.withWave.compile(TDotUnit(64)).doSim { dut =>
    dut.clockDomain.forkStimulus(period = 10)

    def computeTdot(x: Array[Int], w: Array[Int]): Int = {
      x.zip(w).map { case (xi, wi) =>
        val tritVal = wi match { case 0 => 0; case 1 => 1; case 3 => -1; case _ => 0 }
        xi * tritVal
      }.sum
    }

    def runTest(label: String, xVals: Array[Int], wEnc: Array[Int], expected: Int): Unit = {
      for (i <- 0 until 64) {
        dut.io.x(i) #= xVals(i)
        dut.io.w(i) #= wEnc(i)
      }
      dut.io.valid_in #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.valid_in #= false
      // Wait for 2-cycle pipeline (mux+half-tree → register → half-tree → register)
      while (!dut.io.valid_out.toBoolean) {
        dut.clockDomain.waitRisingEdge()
      }
      val actual = dut.io.result.toInt
      println(s"$label: result=$actual (expected $expected) ${if (actual == expected) "OK" else "FAIL"}")
      assert(actual == expected, s"$label: expected $expected, got $actual")
    }

    // Test 1: All zeros weight -> result should be 0
    println("Test 1: Zero weights")
    runTest("Test 1", Array.fill(64)(42), Array.fill(64)(0), 0)

    // Test 2: All +1 weights -> result should be sum of x
    println("Test 2: All +1 weights")
    runTest("Test 2", Array.fill(64)(1), Array.fill(64)(1), 64)

    // Test 3: All -1 weights -> result should be -sum(x)
    println("Test 3: All -1 weights")
    runTest("Test 3", Array.fill(64)(2), Array.fill(64)(3), -128)

    // Test 4: Mixed weights (half +1, half -1)
    println("Test 4: Mixed weights")
    val w4 = Array.tabulate(64)(i => if (i < 32) 1 else 3)
    runTest("Test 4", Array.fill(64)(10), w4, 0)

    // Test 5: Negative inputs with +1 weights
    println("Test 5: Negative x with +1 w")
    runTest("Test 5", Array.fill(64)(-3), Array.fill(64)(1), -192)

    // Test 6-15: Random vectors verified against software reference
    println("\nRandomized tests (10 iterations):")
    for (trial <- 0 until 10) {
      val x = Array.fill(64)(rng.nextInt(256) - 128)
      val wTrit = Array.fill(64)(rng.nextInt(3) - 1) // {-1, 0, +1}
      val wEnc = wTrit.map { case -1 => 3; case 0 => 0; case 1 => 1 }
      val expected = x.zip(wTrit).map { case (xi, wi) => xi * wi }.sum
      runTest(s"Random trial $trial", x, wEnc, expected)
    }

    println("\nAll TDotUnit tests passed!")
  }
}
