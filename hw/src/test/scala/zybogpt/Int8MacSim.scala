package zybogpt

import spinal.core._
import spinal.core.sim._
import scala.util.Random

/** Simulation testbench for Int8MacUnit and Int8MacArray.
  *
  * Verifies INT8 multiply-accumulate operations that will be mapped
  * to DSP48E1 slices for attention score computation.
  *
  * Int8MacUnit has AREG/BREG pipeline registers, adding 1 cycle latency.
  * Each test needs N+1 acc_en cycles for N data items (1 extra drain cycle).
  */
object Int8MacUnitSim extends App {
  val seed = 0xBEEF_0003L
  println(s"Int8MacUnit seed: 0x${seed.toHexString}")
  val rng = new Random(seed)

  SimConfig.withWave.compile(Int8MacUnit()).doSim { dut =>
    dut.clockDomain.forkStimulus(period = 10)

    // Test 1: Simple multiply (1 data + 1 drain = 2 acc_en cycles)
    println("Test 1: 7 * 3 = 21")
    dut.io.clear #= true
    dut.io.a #= 0
    dut.io.b #= 0
    dut.clockDomain.waitRisingEdge()
    dut.io.clear #= false
    dut.io.a #= 7
    dut.io.b #= 3
    dut.io.acc_en #= true
    dut.clockDomain.waitRisingEdge() // aReg=7, product=0*0=0, acc=0
    dut.io.a #= 0
    dut.io.b #= 0
    dut.clockDomain.waitRisingEdge() // aReg=0, product=7*3=21, acc=21
    dut.io.acc_en #= false
    dut.clockDomain.waitRisingEdge()
    println(s"  Result: ${dut.io.result.toInt} (expected 21)")
    assert(dut.io.result.toInt == 21)

    // Test 2: Accumulation (2 data + 1 drain = 3 acc_en cycles)
    println("Test 2: Accumulate 5*4 + 3*2 = 26")
    dut.io.clear #= true
    dut.io.a #= 0
    dut.io.b #= 0
    dut.clockDomain.waitRisingEdge()
    dut.io.clear #= false
    dut.io.a #= 5
    dut.io.b #= 4
    dut.io.acc_en #= true
    dut.clockDomain.waitRisingEdge() // aReg=5,4, product=0, acc=0
    dut.io.a #= 3
    dut.io.b #= 2
    dut.clockDomain.waitRisingEdge() // aReg=3,2, product=5*4=20, acc=20
    dut.io.a #= 0
    dut.io.b #= 0
    dut.clockDomain.waitRisingEdge() // aReg=0,0, product=3*2=6, acc=26
    dut.io.acc_en #= false
    dut.clockDomain.waitRisingEdge()
    println(s"  Result: ${dut.io.result.toInt} (expected 26)")
    assert(dut.io.result.toInt == 26)

    // Test 3: Negative numbers (1 data + 1 drain)
    println("Test 3: -5 * 10 = -50")
    dut.io.clear #= true
    dut.io.a #= 0
    dut.io.b #= 0
    dut.clockDomain.waitRisingEdge()
    dut.io.clear #= false
    dut.io.a #= -5
    dut.io.b #= 10
    dut.io.acc_en #= true
    dut.clockDomain.waitRisingEdge()
    dut.io.a #= 0
    dut.io.b #= 0
    dut.clockDomain.waitRisingEdge()
    dut.io.acc_en #= false
    dut.clockDomain.waitRisingEdge()
    println(s"  Result: ${dut.io.result.toInt} (expected -50)")
    assert(dut.io.result.toInt == -50)

    // Test 4: Full range INT8 (1 data + 1 drain)
    println("Test 4: 127 * 127 = 16129")
    dut.io.clear #= true
    dut.io.a #= 0
    dut.io.b #= 0
    dut.clockDomain.waitRisingEdge()
    dut.io.clear #= false
    dut.io.a #= 127
    dut.io.b #= 127
    dut.io.acc_en #= true
    dut.clockDomain.waitRisingEdge()
    dut.io.a #= 0
    dut.io.b #= 0
    dut.clockDomain.waitRisingEdge()
    dut.io.acc_en #= false
    dut.clockDomain.waitRisingEdge()
    println(s"  Result: ${dut.io.result.toInt} (expected 16129)")
    assert(dut.io.result.toInt == 16129)

    // Test 5: Worst-case accumulation (32 data + 1 drain = 33 acc_en cycles)
    println("Test 5: 32 accumulations of 127*-128")
    dut.io.clear #= true
    dut.io.a #= 0
    dut.io.b #= 0
    dut.clockDomain.waitRisingEdge()
    dut.io.clear #= false
    dut.io.a #= 127
    dut.io.b #= -128
    dut.io.acc_en #= true
    for (_ <- 0 until 32) dut.clockDomain.waitRisingEdge()
    // Drain cycle: pipeline still holds 127*-128
    dut.io.a #= 0
    dut.io.b #= 0
    dut.clockDomain.waitRisingEdge()
    dut.io.acc_en #= false
    dut.clockDomain.waitRisingEdge()
    val exp5 = 32 * 127 * (-128)
    println(s"  Result: ${dut.io.result.toInt} (expected $exp5)")
    assert(dut.io.result.toInt == exp5)

    // Test 6-15: Random dot products of various lengths
    println(s"\nRandomized dot product tests (10 iterations, seed=0x${seed.toHexString}):")
    for (trial <- 0 until 10) {
      val len = rng.nextInt(60) + 4  // 4 to 63
      val aVals = Array.fill(len)(rng.nextInt(256) - 128)
      val bVals = Array.fill(len)(rng.nextInt(256) - 128)
      val expected = aVals.zip(bVals).map { case (a, b) => a * b }.sum

      dut.io.clear #= true
      dut.io.a #= 0
      dut.io.b #= 0
      dut.clockDomain.waitRisingEdge()
      dut.io.clear #= false
      dut.io.acc_en #= true
      for (i <- 0 until len) {
        dut.io.a #= aVals(i)
        dut.io.b #= bVals(i)
        dut.clockDomain.waitRisingEdge()
      }
      // Drain cycle
      dut.io.a #= 0
      dut.io.b #= 0
      dut.clockDomain.waitRisingEdge()
      dut.io.acc_en #= false
      dut.clockDomain.waitRisingEdge()

      val actual = dut.io.result.toInt
      val pass = actual == expected
      println(s"  Trial $trial (len=$len): expected=$expected, got=$actual ${if (pass) "OK" else "FAIL"}")
      assert(pass, s"Trial $trial failed")
    }

    println("\nAll Int8MacUnit tests passed!")
  }
}

object Int8MacArraySim extends App {
  val seed = 0xBEEF_0004L
  println(s"Int8MacArray seed: 0x${seed.toHexString}")
  val rng = new Random(seed)

  SimConfig.withWave.compile(Int8MacArray()).doSim { dut =>
    dut.clockDomain.forkStimulus(period = 10)

    // Test 1: Deterministic 8-lane parallel dot product
    // With AREG/BREG pipeline, innerDim must be N+1 for N data items.
    println("Test 1: 8-lane parallel dot product, inner_dim=32")
    val innerDim = 32
    val aVecs = Array.fill(8)(Array.fill(innerDim)(rng.nextInt(20) - 10))
    val bVecs = Array.fill(8)(Array.fill(innerDim)(rng.nextInt(20) - 10))
    val expected = (0 until 8).map(i =>
      aVecs(i).zip(bVecs(i)).map { case (a, b) => a * b }.sum
    )

    dut.io.innerDim #= innerDim + 1 // +1 for pipeline drain
    dut.io.clear #= true
    dut.io.start #= true
    for (i <- 0 until 8) { dut.io.a(i) #= 0; dut.io.b(i) #= 0 }
    dut.clockDomain.waitRisingEdge()
    dut.io.clear #= false
    dut.io.start #= false
    dut.io.valid #= true

    for (step <- 0 until innerDim) {
      for (i <- 0 until 8) {
        dut.io.a(i) #= aVecs(i)(step)
        dut.io.b(i) #= bVecs(i)(step)
      }
      dut.clockDomain.waitRisingEdge()
    }
    // Drain cycle
    for (i <- 0 until 8) { dut.io.a(i) #= 0; dut.io.b(i) #= 0 }
    dut.clockDomain.waitRisingEdge()
    dut.io.valid #= false

    var timeout = 100
    while (!dut.io.done.toBoolean && timeout > 0) {
      dut.clockDomain.waitRisingEdge()
      timeout -= 1
    }
    assert(timeout > 0, "Timeout waiting for MAC array done")

    for (i <- 0 until 8) {
      val actual = dut.io.results(i).toInt
      val pass = actual == expected(i)
      println(s"  Lane $i: expected=${expected(i)}, got=$actual ${if (pass) "OK" else "FAIL"}")
      assert(pass, s"Lane $i mismatch")
    }

    // Test 2-6: Randomized multi-lane tests with varying inner dimensions
    println(s"\nRandomized tests (5 iterations, seed=0x${seed.toHexString}):")
    for (trial <- 0 until 5) {
      val dim = Seq(8, 16, 32, 48, 64)(trial % 5)
      val aV = Array.fill(8)(Array.fill(dim)(rng.nextInt(256) - 128))
      val bV = Array.fill(8)(Array.fill(dim)(rng.nextInt(256) - 128))
      val exp = (0 until 8).map(i => aV(i).zip(bV(i)).map { case (a, b) => a * b }.sum)

      dut.io.innerDim #= dim + 1 // +1 for pipeline drain
      dut.io.clear #= true
      dut.io.start #= true
      for (i <- 0 until 8) { dut.io.a(i) #= 0; dut.io.b(i) #= 0 }
      dut.clockDomain.waitRisingEdge()
      dut.io.clear #= false
      dut.io.start #= false
      dut.io.valid #= true

      for (step <- 0 until dim) {
        for (i <- 0 until 8) {
          dut.io.a(i) #= aV(i)(step)
          dut.io.b(i) #= bV(i)(step)
        }
        dut.clockDomain.waitRisingEdge()
      }
      // Drain cycle
      for (i <- 0 until 8) { dut.io.a(i) #= 0; dut.io.b(i) #= 0 }
      dut.clockDomain.waitRisingEdge()
      dut.io.valid #= false

      timeout = 100
      while (!dut.io.done.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"Timeout on trial $trial")

      var allOk = true
      for (i <- 0 until 8) {
        if (dut.io.results(i).toInt != exp(i)) allOk = false
      }
      println(s"  Trial $trial (dim=$dim): ${if (allOk) "OK" else "FAIL"}")
      assert(allOk, s"Trial $trial failed")
    }

    println("\nAll Int8MacArray tests passed!")
  }
}
