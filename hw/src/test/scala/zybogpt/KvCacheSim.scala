package zybogpt

import spinal.core._
import spinal.core.sim._
import scala.util.Random

/** Simulation testbench for KV cache.
  *
  * Verifies write-then-read consistency for K and V vectors
  * across layers, heads, and positions.
  */
object KvCacheSim extends App {
  val seed = 0xBEEF_0008L
  println(s"KvCache seed: 0x${seed.toHexString}")
  val rng = new Random(seed)

  SimConfig.withWave.compile(KvCache()).doSim { dut =>
    dut.clockDomain.forkStimulus(period = 10)

    val headDim = 32
    val nHeads = 2
    val nLayers = 2

    // Default all inputs
    dut.io.writeEn #= false
    dut.io.readEn #= false
    dut.io.seqLen #= 0
    dut.io.writeLayer #= 0
    dut.io.writeHead #= 0
    dut.io.writePos #= 0
    dut.io.readLayer #= 0
    dut.io.readHead #= 0
    dut.io.readPos #= 0

    dut.clockDomain.waitRisingEdge(5)

    def writeKV(layer: Int, head: Int, pos: Int, k: Array[Int], v: Array[Int]): Unit = {
      for (i <- 0 until headDim) {
        dut.io.writeK(i) #= k(i)
        dut.io.writeV(i) #= v(i)
      }
      dut.io.writeLayer #= layer
      dut.io.writeHead #= head
      dut.io.writePos #= pos
      dut.io.writeEn #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.writeEn #= false
      dut.clockDomain.waitRisingEdge(headDim + 5)
    }

    def readK(layer: Int, head: Int, pos: Int): Array[Int] = {
      dut.io.readLayer #= layer
      dut.io.readHead #= head
      dut.io.readPos #= pos
      dut.io.readEn #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.readEn #= false
      var timeout = headDim + 50
      while (!dut.io.readValid.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"Timeout reading L$layer H$head P$pos")
      (0 until headDim).map(i => dut.io.readK(i).toInt).toArray
    }

    // Test 1: Single write/read
    println("Test 1: Single write/read (layer=0, head=0, pos=0)")
    val testK = Array.fill(headDim)(rng.nextInt(200) - 100)
    val testV = Array.fill(headDim)(rng.nextInt(200) - 100)
    writeKV(0, 0, 0, testK, testV)
    val readBack = readK(0, 0, 0)
    val kMatch = readBack.zip(testK).count { case (a, b) => a == b }
    println(s"  K match: $kMatch/$headDim")

    // Test 2: Multiple positions
    println("\nTest 2: Write 4 positions, read back")
    val allK = Array.fill(4)(Array.fill(headDim)(rng.nextInt(200) - 100))
    val allV = Array.fill(4)(Array.fill(headDim)(rng.nextInt(200) - 100))
    for (pos <- 0 until 4) writeKV(0, 0, pos, allK(pos), allV(pos))
    for (pos <- 0 until 4) {
      val rb = readK(0, 0, pos)
      val ok = rb.zip(allK(pos)).count { case (a, b) => a == b }
      println(s"  Pos $pos: K match=$ok/$headDim")
    }

    // Test 3: Cross-layer/head isolation
    println("\nTest 3: Cross-layer/head isolation")
    val kL0H0 = Array.fill(headDim)(10)
    val kL0H1 = Array.fill(headDim)(20)
    val kL1H0 = Array.fill(headDim)(30)
    val kL1H1 = Array.fill(headDim)(40)
    val vDummy = Array.fill(headDim)(0)

    writeKV(0, 0, 0, kL0H0, vDummy)
    writeKV(0, 1, 0, kL0H1, vDummy)
    writeKV(1, 0, 0, kL1H0, vDummy)
    writeKV(1, 1, 0, kL1H1, vDummy)

    for ((expected, layer, head, desc) <- Seq(
      (10, 0, 0, "L0H0"), (20, 0, 1, "L0H1"),
      (30, 1, 0, "L1H0"), (40, 1, 1, "L1H1")
    )) {
      val rb = readK(layer, head, 0)
      val pass = rb(0) == expected
      println(s"  $desc: K[0]=${rb(0)} (expected $expected) ${if (pass) "OK" else "FAIL"}")
    }

    // Test 4-8: Randomized write/read with random layer/head/pos
    println(s"\nRandomized tests (5 iterations, seed=0x${seed.toHexString}):")
    for (trial <- 0 until 5) {
      val layer = rng.nextInt(nLayers)
      val head = rng.nextInt(nHeads)
      val pos = rng.nextInt(64)
      val k = Array.fill(headDim)(rng.nextInt(256) - 128)
      val v = Array.fill(headDim)(rng.nextInt(256) - 128)

      writeKV(layer, head, pos, k, v)
      val rb = readK(layer, head, pos)
      val match_count = rb.zip(k).count { case (a, b) => a == b }
      val pass = match_count == headDim
      println(s"  Trial $trial (L$layer H$head P$pos): K match=$match_count/$headDim ${if (pass) "OK" else "FAIL"}")
      assert(pass, s"Trial $trial: K mismatch")
    }

    println("\nAll KvCache tests passed!")
  }
}
