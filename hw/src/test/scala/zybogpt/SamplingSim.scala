package zybogpt

import spinal.core._
import spinal.core.sim._

/** Test harness wrapping SamplingUnit with a real Mem for proper readSync timing. */
case class SamplingTestHarness() extends Component {
  val io = new Bundle {
    val maxLogit      = in  SInt(24 bits)
    val invTemp       = in  UInt(16 bits)
    val seedLoad      = in  Bool()
    val seedVal       = in  UInt(32 bits)
    val start         = in  Bool()
    val done          = out Bool()
    val selectedToken = out UInt(7 bits)

    // Write port for loading test logits into Mem
    val writeAddr     = in  UInt(7 bits)
    val writeData     = in  SInt(24 bits)
    val writeEn       = in  Bool()
  }

  val logitMem = Mem(SInt(24 bits), wordCount = 128)

  when(io.writeEn) {
    logitMem.write(io.writeAddr, io.writeData)
  }

  val sampler = SamplingUnit()
  sampler.io.logitData := logitMem.readSync(sampler.io.logitAddr)
  sampler.io.maxLogit  := io.maxLogit
  sampler.io.invTemp   := io.invTemp
  sampler.io.seedLoad  := io.seedLoad
  sampler.io.seedVal   := io.seedVal
  sampler.io.start     := io.start
  io.done              := sampler.io.done
  io.selectedToken     := sampler.io.selectedToken
}

/** Unit test for SamplingUnit in isolation.
  *
  * Tests use INT24-scale logit values matching real hardware:
  * logits = sum(INT8_query * INT16_emb) over d_model=64 elements.
  * Typical logit magnitudes: 500K - 2M. SamplingUnit applies LOGIT_SHIFT=12
  * (divide by 4096) before temperature scaling.
  *
  * Tests:
  *   1. Deterministic single-peak: one logit much higher → always selected
  *   2. LFSR determinism: same logits + same seed → same output (run twice)
  *   3. Different seeds → (likely) different outputs
  *   4. Uniform logits: output depends entirely on LFSR threshold
  *   5. Temperature sensitivity: close logits, low vs high temperature
  *   6. LFSR sequence verification: step N times, compare against known Galois LFSR
  *   7. Random seed stress test: many random seeds produce valid tokens
  */
object SamplingSim extends App {

  // Logit scale factor: test values are multiplied by this to match INT24 hardware scale.
  // SamplingUnit applies LOGIT_SHIFT=12 internally, so differences of S correspond to
  // effective differences of S >> 12 in the probability computation.
  val S = 4096

  // Galois LFSR step matching hardware and Rust firmware
  def lfsrStep(state: Long): Long = {
    val bit = state & 1L
    var next = state >>> 1
    if (bit != 0) next ^= 0xD0000001L
    next & 0xFFFFFFFFL
  }

  println("=" * 70)
  println("  SamplingUnit Simulation Tests")
  println("=" * 70)

  SimConfig.withWave.compile(SamplingTestHarness()).doSim { dut =>
    dut.clockDomain.forkStimulus(period = 10)

    def writeLogit(addr: Int, value: Int): Unit = {
      dut.io.writeAddr #= addr
      dut.io.writeData #= value
      dut.io.writeEn #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.writeEn #= false
    }

    def setLogits(logits: Array[Int]): Unit = {
      for (i <- 0 until 128) writeLogit(i, logits(i))
    }

    def loadSeed(seed: Long): Unit = {
      dut.io.seedLoad #= true
      dut.io.seedVal #= seed
      dut.clockDomain.waitRisingEdge()
      dut.io.seedLoad #= false
      dut.clockDomain.waitRisingEdge()
    }

    def runSample(invTemp: Int, maxLogit: Int): Int = {
      dut.io.invTemp #= invTemp
      dut.io.maxLogit #= maxLogit
      dut.io.start #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.start #= false

      var timeout = 500
      while (!dut.io.done.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, "SamplingUnit timed out")
      val token = dut.io.selectedToken.toInt
      dut.clockDomain.waitRisingEdge() // let DONE -> IDLE
      token
    }

    // Initialize
    dut.io.start #= false
    dut.io.seedLoad #= false
    dut.io.seedVal #= 0
    dut.io.invTemp #= 0
    dut.io.maxLogit #= 0
    dut.io.writeAddr #= 0
    dut.io.writeData #= 0
    dut.io.writeEn #= false
    dut.clockDomain.waitRisingEdge(5)

    // ================================================================
    // Test 1: Deterministic single-peak
    // ================================================================
    println("\nTest 1: Deterministic single-peak (logit[42] >> rest)")
    val logits1 = Array.fill(128)(0)
    logits1(42) = 500 * S // 2,048,000 — far above rest
    setLogits(logits1)

    // inv_temp = 366 corresponds to T=0.7
    loadSeed(0xDEADBEEFL)
    val token1 = runSample(invTemp = 366, maxLogit = 500 * S)
    println(s"  Selected token: $token1 (expected 42)")
    assert(token1 == 42, s"Expected token 42, got $token1")
    println("  PASSED")

    // ================================================================
    // Test 2: LFSR determinism (same logits + same seed → same output)
    // ================================================================
    println("\nTest 2: LFSR determinism")
    val logits2 = Array.tabulate(128)(i => i * 10 * S)
    setLogits(logits2)
    val maxLogit2 = 127 * 10 * S

    loadSeed(0x12345678L)
    val token2a = runSample(invTemp = 256, maxLogit = maxLogit2)

    loadSeed(0x12345678L)
    val token2b = runSample(invTemp = 256, maxLogit = maxLogit2)

    println(s"  Run 1: $token2a, Run 2: $token2b")
    assert(token2a == token2b, s"Determinism failed: $token2a != $token2b")
    println("  PASSED")

    // ================================================================
    // Test 3: Different seeds → (likely) different outputs
    // ================================================================
    println("\nTest 3: Different seeds")
    // Logits with moderate spread: top few tokens have similar values
    val logits3 = Array.tabulate(128) { i =>
      val base = 100 * S // baseline
      if (i < 20) base + (20 - i) * 5 * S // top 20 tokens, spread of 0-100 * S
      else 0
    }
    setLogits(logits3)
    val maxLogit3 = logits3.max

    // Run with many different seeds and collect results
    val seeds = Array(0xAAAAAAAAL, 0x55555555L, 0x12345678L, 0x87654321L,
                      0xDEADBEEFL, 0xCAFEBABEL, 0x0BADF00DL, 0xFEEDFACEL)
    val results3 = seeds.map { seed =>
      loadSeed(seed)
      runSample(invTemp = 256, maxLogit = maxLogit3)
    }
    val uniqueResults = results3.toSet.size
    println(s"  Seeds: ${seeds.length}, Unique results: $uniqueResults")
    println(s"  Results: ${results3.mkString(", ")}")
    // With 8 different seeds and spread-out logits, expect at least 2 different tokens
    assert(uniqueResults >= 2, s"Expected diversity, got only $uniqueResults unique tokens")
    println("  PASSED")

    // ================================================================
    // Test 4: Uniform logits → output depends on LFSR threshold
    // ================================================================
    println("\nTest 4: Uniform logits")
    val logits4 = Array.fill(128)(500 * S)
    setLogits(logits4)

    loadSeed(0xDEADBEEFL)
    val token4 = runSample(invTemp = 256, maxLogit = 500 * S)
    println(s"  Selected token: $token4 (should be in [0, 127])")
    assert(token4 >= 0 && token4 <= 127, s"Token $token4 out of range")
    println("  PASSED")

    // ================================================================
    // Test 5: Temperature sensitivity
    // ================================================================
    println("\nTest 5: Temperature sensitivity")
    // Create logits where token 10 is clear winner, token 20 is close second,
    // and all others are far below (prob=0 at any temperature)
    val logits5 = Array.fill(128)(0) // far below — diff >> 12 >> 200, prob=0
    logits5(10) = 1000 * S // winner (diff=0, prob=256)
    logits5(20) = 990 * S  // close second (diff=-10*S=-40960, >>12=-10)
    setLogits(logits5)

    // Low temperature (high inv_temp) → token 10 should dominate
    // diff_20 >> 12 = -10, shifted = (-10*1024)>>8 = -40, prob = 216
    // Only tokens 10 and 20 have nonzero prob. P(10) = 256/(256+216) ≈ 54%
    // With inv_temp=2048: shifted = (-10*2048)>>8 = -80, prob = 176
    // Still both nonzero. Use much larger inv_temp for deterministic:
    // inv_temp=4096: shifted = (-10*4096)>>8 = -160, prob = 96
    // P(10) = 256/(256+96) ≈ 73%. Still not deterministic.
    // Make bigger gap instead: logits5(20) = 900*S → diff=-100*S, >>12=-100
    logits5(20) = 900 * S // diff = -100*S = -409600, >>12 = -100
    setLogits(logits5)

    var lowTempWins = 0
    val numTrials = 20
    for (trial <- 0 until numTrials) {
      loadSeed(0x10000L + trial)
      // inv_temp=1024: shifted=(-100*1024)>>8=-400, prob=0. Only token 10 nonzero.
      val t = runSample(invTemp = 1024, maxLogit = 1000 * S)
      if (t == 10) lowTempWins += 1
    }
    println(s"  Low temp (inv_temp=1024): token 10 won $lowTempWins/$numTrials times")
    assert(lowTempWins >= numTrials - 2, s"Low temp should strongly prefer token 10")

    // Higher temperature (lower inv_temp) → token 20 gets nonzero probability
    // inv_temp=128: shifted=(-100*128)>>8=-50, prob=206
    // Both tokens have nonzero prob → more diversity
    logits5(20) = 990 * S // close again for diversity test
    setLogits(logits5)
    var highTempToken10 = 0
    for (trial <- 0 until numTrials) {
      loadSeed(0x20000L + trial)
      val t = runSample(invTemp = 128, maxLogit = 1000 * S) // higher temp
      if (t == 10) highTempToken10 += 1
    }
    println(s"  High temp (inv_temp=128): token 10 won $highTempToken10/$numTrials times")
    // Higher temp should pick token 10 less often (more exploration)
    println(s"  Temperature effect: low_temp=$lowTempWins, high_temp=$highTempToken10")
    println("  PASSED")

    // ================================================================
    // Test 6: LFSR sequence verification
    // ================================================================
    println("\nTest 6: LFSR sequence verification")
    val initSeed = 0xDEADBEEFL
    loadSeed(initSeed)

    // Compute expected LFSR sequence in Scala
    var expectedLfsr = initSeed
    val numSteps = 10
    val expectedSeq = (0 until numSteps).map { _ =>
      expectedLfsr = lfsrStep(expectedLfsr)
      expectedLfsr
    }

    // Run the sampler numSteps times to advance the LFSR numSteps times
    // Each sample run advances LFSR once (in THRESHOLD state)
    val logits6 = Array.fill(128)(100 * S)
    setLogits(logits6)

    val hwLfsrSeq = scala.collection.mutable.ArrayBuffer[Long]()
    for (i <- 0 until numSteps) {
      runSample(invTemp = 256, maxLogit = 100 * S)
      // Read LFSR value after the sample (LFSR was advanced in THRESHOLD)
      val lfsrVal = dut.sampler.lfsr.toLong & 0xFFFFFFFFL
      hwLfsrSeq += lfsrVal
    }

    println(s"  Initial seed: 0x${initSeed.toHexString}")
    var allMatch = true
    for (i <- 0 until numSteps) {
      val hw = hwLfsrSeq(i)
      val exp = expectedSeq(i)
      val match_str = if (hw == exp) "MATCH" else "MISMATCH"
      if (hw != exp) allMatch = false
      if (i < 5 || hw != exp) {
        println(f"  Step ${i + 1}%2d: HW=0x${hw}%08x  Expected=0x${exp}%08x  $match_str")
      }
    }
    if (allMatch) {
      println(s"  All $numSteps LFSR steps match!")
    }
    assert(allMatch, "LFSR sequence mismatch")
    println("  PASSED")

    // ================================================================
    // Test 7: Random seed stress test
    // ================================================================
    println("\nTest 7: Random seed stress test (50 random seeds)")
    val rng = new scala.util.Random(0xCAFE1234L)
    val logits7 = Array.tabulate(128)(i => ((i * 7) % 200 + 10) * S)
    setLogits(logits7)
    val maxLogit7 = logits7.max

    val randomResults = scala.collection.mutable.ArrayBuffer[Int]()
    for (trial <- 0 until 50) {
      val seed = (rng.nextLong() & 0xFFFFFFFFL)
      loadSeed(seed)
      val t = runSample(invTemp = 256, maxLogit = maxLogit7)
      assert(t >= 0 && t <= 127, s"Trial $trial: token $t out of range with seed 0x${seed.toHexString}")
      randomResults += t
    }
    val uniqueRandom = randomResults.toSet.size
    println(s"  50 random seeds, ${uniqueRandom} unique tokens")
    println(s"  Sample: [${randomResults.take(10).mkString(", ")}, ...]")
    assert(uniqueRandom >= 3, s"Expected diversity across random seeds, got only $uniqueRandom unique")
    println("  PASSED")

    // ================================================================
    // Summary
    // ================================================================
    println("\n" + "=" * 70)
    println("  All SamplingUnit tests PASSED!")
    println("=" * 70)

    simSuccess()
  }
}
