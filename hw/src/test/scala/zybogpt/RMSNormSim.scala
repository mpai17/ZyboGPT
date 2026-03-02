package zybogpt

import spinal.core._
import spinal.core.sim._
import scala.util.Random

/** Test harness for RMSNorm that provides gamma via Mem (matching new interface).
  *
  * RMSNorm now reads gamma via gammaAddr/gammaData interface instead of Vec.
  * This wrapper instantiates a gamma Mem and wires it to RMSNorm.
  */
case class RMSNormTestHarness(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val dim = config.dModel
  val io = new Bundle {
    val x = in Vec (SInt(8 bits), dim)
    val start = in Bool ()
    val y = out Vec (SInt(8 bits), dim)
    val done = out Bool ()
    // Gamma write port for test setup
    val gammaWriteAddr = in UInt (log2Up(dim) bits)
    val gammaWriteData = in SInt (16 bits)
    val gammaWriteEn = in Bool ()
  }

  val gammaMem = Mem(SInt(16 bits), dim)
  gammaMem.init(Seq.fill(dim)(S(1024, 16 bits))) // default gamma=1.0

  // Write port for test setup
  when(io.gammaWriteEn) {
    gammaMem.write(io.gammaWriteAddr, io.gammaWriteData)
  }

  val norm = RMSNorm(config)
  norm.io.x := io.x
  norm.io.start := io.start
  io.y := norm.io.y
  io.done := norm.io.done

  // Wire gamma read port
  norm.io.gammaData := gammaMem.readSync(norm.io.gammaAddr)
}

/** Simulation testbench for integer RMSNorm.
  *
  * Computes expected output in software and compares against hardware,
  * with tolerance for fixed-point approximation errors.
  */
object RMSNormSim extends App {
  val seed = 0xBEEF_0006L
  println(s"RMSNorm seed: 0x${seed.toHexString}")
  val rng = new Random(seed)

  SimConfig.withWave.compile(RMSNormTestHarness()).doSim { dut =>
    dut.clockDomain.forkStimulus(period = 10)
    val dim = 64

    // Initialize ALL inputs to safe defaults before any clock edges
    dut.io.start #= false
    dut.io.gammaWriteEn #= false
    dut.io.gammaWriteAddr #= 0
    dut.io.gammaWriteData #= 0
    for (i <- 0 until dim) {
      dut.io.x(i) #= 0
    }

    def loadGammas(gammaQ510: Array[Int]): Unit = {
      for (i <- 0 until dim) {
        dut.io.gammaWriteAddr #= i
        dut.io.gammaWriteData #= gammaQ510(i)
        dut.io.gammaWriteEn #= true
        dut.clockDomain.waitRisingEdge()
      }
      dut.io.gammaWriteEn #= false
      dut.clockDomain.waitRisingEdge()
    }

    def waitDone(label: String): Unit = {
      var timeout = 200
      while (!dut.io.done.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"Timeout waiting for RMSNorm done in $label")
    }

    /** Software reference RMSNorm matching hardware integer arithmetic. */
    def refRmsNorm(x: Array[Int], gamma: Array[Double]): Array[Double] = {
      val meanSq = x.map(v => v.toDouble * v.toDouble).sum / dim
      val rms = Math.sqrt(meanSq)
      if (rms < 1e-6) return Array.fill(dim)(0.0)
      x.zip(gamma).map { case (xi, gi) => xi.toDouble / rms * gi * 64.0 }
    }

    def runTest(label: String, xVals: Array[Int], gammaQ510: Array[Int],
                gammaFloat: Array[Double], toleranceAbs: Int = 3): Unit = {
      println(s"\n$label")
      // Load gammas into BRAM
      loadGammas(gammaQ510)

      for (i <- 0 until dim) {
        dut.io.x(i) #= xVals(i)
      }
      dut.io.start #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.start #= false
      waitDone(label)

      val hwOut = (0 until dim).map(i => dut.io.y(i).toInt).toArray
      val expected = refRmsNorm(xVals, gammaFloat).map(v =>
        Math.max(-128, Math.min(127, Math.round(v).toInt))
      )

      println(s"  Input[0..5]:    ${xVals.take(6).mkString(", ")}")
      println(s"  Expected[0..5]: ${expected.take(6).mkString(", ")}")
      println(s"  Got[0..5]:      ${hwOut.take(6).mkString(", ")}")

      var signErrors = 0
      var absErrors = 0
      var maxErr = 0
      for (i <- 0 until dim) {
        val err = Math.abs(hwOut(i) - expected(i))
        maxErr = Math.max(maxErr, err)
        if (err > toleranceAbs) absErrors += 1
        if (Math.abs(expected(i)) > 2 && hwOut(i) != 0 && hwOut(i).signum != expected(i).signum) signErrors += 1
      }
      println(s"  Max error: $maxErr, elements exceeding tolerance($toleranceAbs): $absErrors, sign errors: $signErrors")

      assert(signErrors == 0, s"$label: $signErrors sign errors detected")
      assert(absErrors <= dim / 4, s"$label: too many elements ($absErrors/$dim) exceed tolerance")
    }

    // Test 1: Uniform input x=50, gamma=1.0 (64x scale: expect ~64)
    {
      val x = Array.fill(dim)(50)
      val gammaF = Array.fill(dim)(1.0)
      val gammaQ = gammaF.map(g => (g * 1024).toInt)
      runTest("Test 1: Uniform x=50, gamma=1.0 (expect ~64 with 64x scale)", x, gammaQ, gammaF, toleranceAbs = 8)
    }

    // Test 2: Random input, gamma=1.0
    {
      val x = Array.fill(dim)(rng.nextInt(200) - 100)
      val gammaF = Array.fill(dim)(1.0)
      val gammaQ = gammaF.map(g => (g * 1024).toInt)
      runTest("Test 2: Random input, gamma=1.0", x, gammaQ, gammaF, toleranceAbs = 12)

      val hwOut = (0 until dim).map(i => dut.io.y(i).toInt).toArray
      val uniqueVals = hwOut.toSet.size
      println(s"  Unique output values: $uniqueVals")
      assert(uniqueVals >= 3, s"Test 2: Expected diverse outputs, got only $uniqueVals unique values")
    }

    // Test 3: Negative inputs, gamma=0.5
    {
      val x = Array.fill(dim)(-80)
      val gammaF = Array.fill(dim)(0.5)
      val gammaQ = gammaF.map(g => (g * 1024).toInt)
      runTest("Test 3: All negative x=-80, gamma=0.5 (expect ~-32)", x, gammaQ, gammaF, toleranceAbs = 8)

      val hwOut = (0 until dim).map(i => dut.io.y(i).toInt).toArray
      assert(hwOut.forall(_ < 0), s"Test 3: Expected all negative outputs")
    }

    // Test 4: Mixed signs, gamma=1.0
    {
      val x = Array.tabulate(dim)(i => if (i < 32) 100 else -100)
      val gammaF = Array.fill(dim)(1.0)
      val gammaQ = gammaF.map(g => (g * 1024).toInt)
      runTest("Test 4: Mixed +/-100, gamma=1.0 (expect +-64)", x, gammaQ, gammaF, toleranceAbs = 8)
    }

    // Test 5: Sparse input, gamma=0.25
    {
      val x = Array.fill(dim)(0)
      x(0) = 127
      val gammaF = Array.fill(dim)(0.25)
      val gammaQ = gammaF.map(g => (g * 1024).toInt)
      runTest("Test 5: Sparse input (one non-zero), gamma=0.25", x, gammaQ, gammaF, toleranceAbs = 16)
    }

    // Test 6-10: Fully randomized (random x, random gamma in realistic range)
    println(s"\nRandomized tests (5 iterations, seed=0x${seed.toHexString}):")
    for (trial <- 0 until 5) {
      val x = Array.fill(dim)(rng.nextInt(200) - 100)
      val gammaScale = rng.nextDouble() * 1.5 + 0.5
      val gammaF = Array.fill(dim)(gammaScale + rng.nextGaussian() * 0.3)
        .map(g => Math.max(0.1, Math.min(4.0, g)))
      val gammaQ = gammaF.map(g => (g * 1024).toInt)
      runTest(s"Random trial $trial (gamma_scale=${gammaScale.formatted("%.1f")})", x, gammaQ, gammaF, toleranceAbs = 12)
    }

    println("\nAll RMSNorm tests passed!")
  }
}
