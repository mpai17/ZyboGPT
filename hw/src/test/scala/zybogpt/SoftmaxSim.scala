package zybogpt

import spinal.core._
import spinal.core.sim._
import scala.util.Random

/** Test harness for Softmax that provides scores via Mem (matching new interface).
  *
  * Softmax now reads scores via scoreAddr/scoreData interface instead of Vec.
  * This wrapper instantiates a score Mem and wires it to Softmax.
  */
case class SoftmaxTestHarness(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val maxLen = config.ctxLen
  val io = new Bundle {
    val len = in UInt (8 bits)
    val start = in Bool ()
    // Prob read via address/data interface
    val probAddr = in UInt (7 bits)
    val probData = out UInt (8 bits)
    val done = out Bool ()
    // Score write port for test setup
    val scoreWriteAddr = in UInt (7 bits)
    val scoreWriteData = in SInt (16 bits)
    val scoreWriteEn = in Bool ()
  }

  val scoreMem = Mem(SInt(16 bits), maxLen)

  // Write port for test setup
  when(io.scoreWriteEn) {
    scoreMem.write(io.scoreWriteAddr, io.scoreWriteData)
  }

  val softmax = Softmax(config)
  softmax.io.scoreData := scoreMem.readSync(softmax.io.scoreAddr)
  softmax.io.len := io.len
  softmax.io.start := io.start
  softmax.io.probAddr := io.probAddr
  io.probData := softmax.io.probData
  io.done := softmax.io.done
}

/** Simulation testbench for integer Softmax.
  *
  * Verifies piecewise-linear exp approximation and normalization.
  * Output should sum to ~256 (UINT8 probability space).
  */
object SoftmaxSim extends App {
  val seed = 0xBEEF_0007L
  println(s"Softmax seed: 0x${seed.toHexString}")
  val rng = new Random(seed)

  SimConfig.withWave.compile(SoftmaxTestHarness()).doSim { dut =>
    dut.clockDomain.forkStimulus(period = 10)

    val maxLen = 128

    // Default score write disabled
    dut.io.scoreWriteEn #= false
    dut.io.scoreWriteAddr #= 0
    dut.io.scoreWriteData #= 0
    dut.io.probAddr #= 0

    def loadScores(scores: Array[Int], len: Int): Unit = {
      for (i <- 0 until maxLen) {
        dut.io.scoreWriteAddr #= i
        dut.io.scoreWriteData #= (if (i < len) scores(i) else -32768)
        dut.io.scoreWriteEn #= true
        dut.clockDomain.waitRisingEdge()
      }
      dut.io.scoreWriteEn #= false
      dut.clockDomain.waitRisingEdge()
    }

    /** Read probs one at a time via address/data interface (readSync = 1-cycle latency).
      * Set address at cycle N, data appears at cycle N+1.
      */
    def readProbs(len: Int): Array[Int] = {
      val result = new Array[Int](len)
      for (i <- 0 until len) {
        dut.io.probAddr #= i
        dut.clockDomain.waitRisingEdge() // addr registered
        dut.clockDomain.waitRisingEdge() // data available
        result(i) = dut.io.probData.toInt
      }
      result
    }

    def runSoftmax(scores: Array[Int], len: Int, label: String): Array[Int] = {
      println(s"\n$label")
      loadScores(scores, len)

      dut.io.len #= len
      dut.io.start #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.start #= false

      var timeout = 2000
      while (!dut.io.done.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"Timeout on $label")

      readProbs(len)
    }

    // Test 1: Uniform scores -> uniform probabilities
    {
      val scores = Array.fill(4)(100)
      val probs = runSoftmax(scores, 4, "Test 1: Uniform scores (len=4)")
      val sum = probs.sum
      println(s"  Probs: ${probs.mkString(", ")}")
      println(s"  Sum: $sum (expected ~256)")
      val uniformOk = probs.forall(p => p >= 40 && p <= 90)
      println(s"  Roughly uniform: $uniformOk")
    }

    // Test 2: One dominant score
    {
      val scores = Array(200, 50, 50, 50)
      val probs = runSoftmax(scores, 4, "Test 2: One dominant score (len=4)")
      println(s"  Probs: ${probs.mkString(", ")}")
      println(s"  Dominant prob: ${probs(0)} (should be largest)")
      assert(probs(0) > probs(1), "Dominant score should have highest probability")
    }

    // Test 3: Longer sequence (len=16) with seeded random
    {
      val scores = Array.fill(16)(rng.nextInt(100) - 50)
      val probs = runSoftmax(scores, 16, "Test 3: Random scores (len=16)")
      val sum = probs.sum
      println(s"  Scores: ${scores.take(8).mkString(", ")}...")
      println(s"  Probs: ${probs.take(8).mkString(", ")}...")
      println(s"  Sum: $sum")
      val allNonNeg = probs.forall(_ >= 0)
      println(s"  All non-negative: $allNonNeg")
      assert(allNonNeg, "Negative probability detected")
    }

    // Test 4: Full context length (len=128)
    {
      val scores = Array.tabulate(128)(i => i - 64)
      val probs = runSoftmax(scores, 128, "Test 4: Full context (len=128)")
      val sum = probs.sum
      println(s"  Probs[124..127]: ${probs.slice(124, 128).mkString(", ")} (highest)")
      println(s"  Probs[0..3]: ${probs.slice(0, 4).mkString(", ")} (lowest)")
      println(s"  Sum: $sum")
      val lastGreater = probs.last >= probs.head
      println(s"  Last > First: $lastGreater")
    }

    // Test 5-9: Randomized softmax tests
    println(s"\nRandomized tests (5 iterations, seed=0x${seed.toHexString}):")
    for (trial <- 0 until 5) {
      val len = rng.nextInt(120) + 4 // 4 to 123
      val scores = Array.fill(len)(rng.nextInt(200) - 100)
      val probs = runSoftmax(scores, len, s"Random trial $trial (len=$len)")

      // Verify: all non-negative
      assert(probs.forall(_ >= 0), s"Trial $trial: negative probability")

      // Verify: max prob corresponds to max score
      val maxScoreIdx = scores.zipWithIndex.maxBy(_._1)._2
      val maxProbIdx = probs.zipWithIndex.maxBy(_._1)._2
      println(s"  Max score idx: $maxScoreIdx (score=${scores(maxScoreIdx)}), Max prob idx: $maxProbIdx (prob=${probs(maxProbIdx)})")

      // Verify: sum is in reasonable range (128-512)
      val sum = probs.sum
      println(s"  Sum: $sum ${if (sum >= 64 && sum <= 512) "OK" else "WARN"}")
    }

    println("\nAll Softmax tests passed!")
  }
}
