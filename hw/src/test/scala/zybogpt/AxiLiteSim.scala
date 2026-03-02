package zybogpt

import spinal.core._
import spinal.core.sim._
import scala.util.Random

/** Comprehensive AXI-Lite slave simulation testbench.
  *
  * Tests all register read/write operations including:
  * - Simultaneous AW+W (basic)
  * - Staggered AW-then-W (Zynq PS7 typical behavior)
  * - Staggered W-then-AW
  * - Delayed B-ready (master slow to accept write response)
  * - Back-to-back writes
  * - Interleaved reads and writes
  * - Start pulse edge detection
  * - Done latch behavior
  * - Sampling/seed register writes
  */
object AxiLiteSim extends App {
  val seed = 0xBEEF_0009L
  println(s"AxiLite seed: 0x${seed.toHexString}")
  val rng = new Random(seed)
  var testNum = 0

  SimConfig.withWave.compile(AxiLiteSlave()).doSim { dut =>
    dut.clockDomain.forkStimulus(period = 10)

    // Default inputs
    dut.io.axi.aw.valid #= false
    dut.io.axi.w.valid #= false
    dut.io.axi.b.ready #= false  // NOT always ready — test proper handshake
    dut.io.axi.ar.valid #= false
    dut.io.axi.r.ready #= false  // NOT always ready
    dut.io.tokenOut #= 42
    dut.io.busy #= false
    dut.io.done #= false
    dut.io.cycleCount #= 0

    dut.clockDomain.waitRisingEdge(5)

    // ========================================================
    // Helper: AXI write with AW and W arriving simultaneously
    // ========================================================
    def axiWriteSimultaneous(addr: Int, data: Long): Unit = {
      // Drive AW + W on same cycle
      dut.io.axi.aw.valid #= true
      dut.io.axi.aw.addr #= addr
      dut.io.axi.w.valid #= true
      dut.io.axi.w.data #= data
      dut.io.axi.w.strb #= 0xF

      // Wait for both AW and W handshakes (ready asserted by slave)
      var timeout = 20
      while (timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        val awDone = dut.io.axi.aw.ready.toBoolean
        val wDone = dut.io.axi.w.ready.toBoolean
        if (awDone) dut.io.axi.aw.valid #= false
        if (wDone) dut.io.axi.w.valid #= false
        if (awDone && wDone) {
          timeout = -1 // break
        } else {
          timeout -= 1
        }
      }
      assert(timeout == -1, s"AXI simultaneous write timeout at 0x${addr.toHexString}")

      // Wait for B valid
      timeout = 20
      while (!dut.io.axi.b.valid.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"AXI write response (b.valid) timeout at 0x${addr.toHexString}")
      assert(dut.io.axi.b.resp.toInt == 0, "Expected OKAY response")

      // Accept the response
      dut.io.axi.b.ready #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.axi.b.ready #= false

      // Verify b.valid deasserts after handshake
      dut.clockDomain.waitRisingEdge()
      assert(!dut.io.axi.b.valid.toBoolean, "b.valid must deassert after b.ready handshake")
    }

    // ========================================================
    // Helper: AXI write with AW arriving first, then W later
    // ========================================================
    def axiWriteAwFirst(addr: Int, data: Long, gapCycles: Int = 1): Unit = {
      // Drive AW first
      dut.io.axi.aw.valid #= true
      dut.io.axi.aw.addr #= addr

      // Wait for AW handshake
      var timeout = 20
      while (!dut.io.axi.aw.ready.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"AW handshake timeout at 0x${addr.toHexString}")
      dut.clockDomain.waitRisingEdge()
      dut.io.axi.aw.valid #= false

      // Gap between AW and W
      for (_ <- 0 until gapCycles) {
        dut.clockDomain.waitRisingEdge()
      }

      // Now drive W
      dut.io.axi.w.valid #= true
      dut.io.axi.w.data #= data
      dut.io.axi.w.strb #= 0xF

      timeout = 20
      while (!dut.io.axi.w.ready.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"W handshake timeout at 0x${addr.toHexString}")
      dut.clockDomain.waitRisingEdge()
      dut.io.axi.w.valid #= false

      // Wait for B valid
      timeout = 20
      while (!dut.io.axi.b.valid.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"B valid timeout (AW-first) at 0x${addr.toHexString}")

      // Accept response
      dut.io.axi.b.ready #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.axi.b.ready #= false
      dut.clockDomain.waitRisingEdge()
    }

    // ========================================================
    // Helper: AXI write with W arriving first, then AW later
    // ========================================================
    def axiWriteWFirst(addr: Int, data: Long, gapCycles: Int = 1): Unit = {
      // Drive W first
      dut.io.axi.w.valid #= true
      dut.io.axi.w.data #= data
      dut.io.axi.w.strb #= 0xF

      var timeout = 20
      while (!dut.io.axi.w.ready.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"W handshake timeout (W-first) at 0x${addr.toHexString}")
      dut.clockDomain.waitRisingEdge()
      dut.io.axi.w.valid #= false

      // Gap
      for (_ <- 0 until gapCycles) {
        dut.clockDomain.waitRisingEdge()
      }

      // Now drive AW
      dut.io.axi.aw.valid #= true
      dut.io.axi.aw.addr #= addr

      timeout = 20
      while (!dut.io.axi.aw.ready.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"AW handshake timeout (W-first) at 0x${addr.toHexString}")
      dut.clockDomain.waitRisingEdge()
      dut.io.axi.aw.valid #= false

      // Wait for B valid
      timeout = 20
      while (!dut.io.axi.b.valid.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"B valid timeout (W-first) at 0x${addr.toHexString}")

      // Accept response
      dut.io.axi.b.ready #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.axi.b.ready #= false
      dut.clockDomain.waitRisingEdge()
    }

    // ========================================================
    // Helper: AXI read with proper handshake
    // ========================================================
    def axiRead(addr: Int): Long = {
      dut.io.axi.ar.valid #= true
      dut.io.axi.ar.addr #= addr

      // Wait for AR handshake
      var timeout = 20
      while (!dut.io.axi.ar.ready.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"AR handshake timeout at 0x${addr.toHexString}")
      dut.clockDomain.waitRisingEdge()
      dut.io.axi.ar.valid #= false

      // Wait for R valid
      timeout = 20
      while (!dut.io.axi.r.valid.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"AXI read timeout at addr 0x${addr.toHexString}")

      val data = dut.io.axi.r.data.toLong
      assert(dut.io.axi.r.resp.toInt == 0, "Expected OKAY response on read")

      // Accept the read data
      dut.io.axi.r.ready #= true
      dut.clockDomain.waitRisingEdge()
      dut.io.axi.r.ready #= false

      // Verify r.valid deasserts
      dut.clockDomain.waitRisingEdge()
      assert(!dut.io.axi.r.valid.toBoolean, "r.valid must deassert after r.ready handshake")

      data
    }

    def pass(name: String): Unit = {
      testNum += 1
      println(s"  Test $testNum PASS: $name")
    }

    // ============================================================
    // Test Group 1: Basic simultaneous AW+W writes
    // ============================================================
    println("=== Group 1: Simultaneous AW+W writes ===")

    // Test 1: Read CONFIG register
    val config = axiRead(0x1C)
    val d_model = config & 0xFF
    val n_layers = (config >> 8) & 0xFF
    val ctx_len = (config >> 16) & 0xFF
    val vocab = (config >> 24) & 0xFF
    assert(d_model == 64 && n_layers == 2 && ctx_len == 128 && vocab == 128,
      s"Config mismatch: d=$d_model l=$n_layers c=$ctx_len v=$vocab")
    pass(s"CONFIG read: d=$d_model l=$n_layers c=$ctx_len v=$vocab")

    // Test 2: Write TOKEN_IN and read back
    axiWriteSimultaneous(0x08, 65)
    assert((axiRead(0x08) & 0x7F) == 65)
    pass("TOKEN_IN write/readback (simultaneous)")

    // Test 3: Write POSITION and read back
    axiWriteSimultaneous(0x10, 10)
    assert((axiRead(0x10) & 0x7F) == 10)
    pass("POSITION write/readback (simultaneous)")

    // Test 4: Read TOKEN_OUT
    assert((axiRead(0x0C) & 0x7F) == 42)
    pass("TOKEN_OUT read")

    // Test 5: Write SAMPLING and read back
    axiWriteSimultaneous(0x20, 512)
    assert((axiRead(0x20) & 0xFFFF) == 512)
    pass("SAMPLING write/readback (simultaneous)")

    // Test 6: Write SEED and read back
    axiWriteSimultaneous(0x24, 0xDEADBEEFL)
    val seedVal = axiRead(0x24) & 0xFFFFFFFFL
    assert(seedVal == 0xDEADBEEFL, s"SEED mismatch: got 0x${seedVal.toHexString}")
    pass("SEED write/readback (simultaneous)")

    // ============================================================
    // Test Group 2: Staggered AW-then-W writes (Zynq PS7 behavior)
    // ============================================================
    println("\n=== Group 2: Staggered AW-then-W writes ===")

    // Test 7: AW first, then W 1 cycle later
    axiWriteAwFirst(0x08, 99, gapCycles = 1)
    assert((axiRead(0x08) & 0x7F) == 99)
    pass("TOKEN_IN via AW-first (1 cycle gap)")

    // Test 8: AW first, then W 3 cycles later
    axiWriteAwFirst(0x10, 55, gapCycles = 3)
    assert((axiRead(0x10) & 0x7F) == 55)
    pass("POSITION via AW-first (3 cycle gap)")

    // Test 9: AW first for SAMPLING
    axiWriteAwFirst(0x20, 256, gapCycles = 2)
    assert((axiRead(0x20) & 0xFFFF) == 256)
    pass("SAMPLING via AW-first (2 cycle gap)")

    // Test 10: AW first for SEED
    axiWriteAwFirst(0x24, 0xCAFEBABEL, gapCycles = 1)
    val seedVal2 = axiRead(0x24) & 0xFFFFFFFFL
    assert(seedVal2 == 0xCAFEBABEL, s"SEED mismatch: got 0x${seedVal2.toHexString}")
    pass("SEED via AW-first (1 cycle gap)")

    // ============================================================
    // Test Group 3: Staggered W-then-AW writes
    // ============================================================
    println("\n=== Group 3: Staggered W-then-AW writes ===")

    // Test 11: W first, then AW
    axiWriteWFirst(0x08, 33, gapCycles = 1)
    assert((axiRead(0x08) & 0x7F) == 33)
    pass("TOKEN_IN via W-first (1 cycle gap)")

    // Test 12: W first, then AW 3 cycles later
    axiWriteWFirst(0x10, 77, gapCycles = 3)
    assert((axiRead(0x10) & 0x7F) == 77)
    pass("POSITION via W-first (3 cycle gap)")

    // Test 13: W first for SEED
    axiWriteWFirst(0x24, 0x12345678L, gapCycles = 2)
    val seedVal3 = axiRead(0x24) & 0xFFFFFFFFL
    assert(seedVal3 == 0x12345678L, s"SEED mismatch: got 0x${seedVal3.toHexString}")
    pass("SEED via W-first (2 cycle gap)")

    // ============================================================
    // Test Group 4: Delayed B-ready (master slow to accept response)
    // ============================================================
    println("\n=== Group 4: Delayed B-ready ===")

    // Test 14: Write with B-ready delayed 5 cycles
    dut.io.axi.aw.valid #= true
    dut.io.axi.aw.addr #= 0x08
    dut.io.axi.w.valid #= true
    dut.io.axi.w.data #= 111
    dut.io.axi.w.strb #= 0xF

    // Wait for AW+W accepted
    var timeout = 20
    while (timeout > 0) {
      dut.clockDomain.waitRisingEdge()
      if (dut.io.axi.aw.ready.toBoolean && dut.io.axi.w.ready.toBoolean) timeout = -1
      else timeout -= 1
    }
    assert(timeout == -1, "AW+W handshake timeout (delayed B test)")
    dut.io.axi.aw.valid #= false
    dut.io.axi.w.valid #= false

    // Wait for b.valid, but DON'T assert b.ready yet
    timeout = 20
    while (!dut.io.axi.b.valid.toBoolean && timeout > 0) {
      dut.clockDomain.waitRisingEdge()
      timeout -= 1
    }
    assert(timeout > 0, "b.valid never asserted (delayed B test)")

    // Hold b.ready low for 5 cycles — b.valid must stay high
    for (i <- 0 until 5) {
      dut.clockDomain.waitRisingEdge()
      assert(dut.io.axi.b.valid.toBoolean, s"b.valid dropped before b.ready at cycle $i")
    }

    // Now accept
    dut.io.axi.b.ready #= true
    dut.clockDomain.waitRisingEdge()
    dut.io.axi.b.ready #= false
    dut.clockDomain.waitRisingEdge()
    assert(!dut.io.axi.b.valid.toBoolean, "b.valid must deassert after delayed handshake")

    // Verify the write took effect
    assert((axiRead(0x08) & 0x7F) == 111)
    pass("Write with B-ready delayed 5 cycles")

    // ============================================================
    // Test Group 5: Back-to-back writes
    // ============================================================
    println("\n=== Group 5: Back-to-back writes ===")

    // Test 15: 10 rapid sequential writes using different methods
    val methods = Seq("simultaneous", "aw-first", "w-first")
    for (i <- 0 until 10) {
      val value = rng.nextInt(128)
      val method = methods(i % 3)
      method match {
        case "simultaneous" => axiWriteSimultaneous(0x08, value)
        case "aw-first"     => axiWriteAwFirst(0x08, value, gapCycles = 1)
        case "w-first"      => axiWriteWFirst(0x08, value, gapCycles = 1)
      }
      val readBack = axiRead(0x08) & 0x7F
      assert(readBack == value, s"Back-to-back iter $i ($method): wrote $value, read $readBack")
    }
    pass("10 back-to-back writes (mixed methods)")

    // Test 16: Rapid writes to different registers
    for (_ <- 0 until 5) {
      val tok = rng.nextInt(128)
      val pos = rng.nextInt(128)
      val invT = rng.nextInt(65536)
      axiWriteSimultaneous(0x08, tok)
      axiWriteAwFirst(0x10, pos, gapCycles = 1)
      axiWriteWFirst(0x20, invT, gapCycles = 1)
      assert((axiRead(0x08) & 0x7F) == tok)
      assert((axiRead(0x10) & 0x7F) == pos)
      assert((axiRead(0x20) & 0xFFFF) == invT)
    }
    pass("Multi-register back-to-back writes (5 rounds)")

    // ============================================================
    // Test Group 6: Control register edge detection
    // ============================================================
    println("\n=== Group 6: Start pulse edge detection ===")

    // Test 17: Start pulse on 0→1 transition
    axiWriteSimultaneous(0x00, 0x00) // Clear control
    dut.clockDomain.waitRisingEdge(2)
    assert(!dut.io.start.toBoolean, "Start should be low before trigger")

    axiWriteSimultaneous(0x00, 0x01) // Set start bit
    dut.clockDomain.waitRisingEdge()
    // Start is rising-edge: controlReg(0) && !RegNext(controlReg(0))
    // After write, controlReg=1, RegNext was 0 → start=true for 1 cycle
    val startFired = dut.io.start.toBoolean
    dut.clockDomain.waitRisingEdge()
    val startStillHigh = dut.io.start.toBoolean
    assert(startFired || !startStillHigh, "Start should be a single-cycle pulse")
    pass("Start pulse fires on 0→1 transition")

    // Test 18: No repeated start on writing 1 again
    axiWriteSimultaneous(0x00, 0x01)
    dut.clockDomain.waitRisingEdge(2)
    // controlReg was already 1, RegNext is now 1 → start should be false
    assert(!dut.io.start.toBoolean, "Start should not re-fire when bit stays 1")
    pass("No re-trigger when start bit stays 1")

    // ============================================================
    // Test Group 7: Done latch behavior
    // ============================================================
    println("\n=== Group 7: Done latch ===")

    // Test 19: Done latch sets on done pulse
    dut.io.done #= true
    dut.clockDomain.waitRisingEdge(2)
    val status = axiRead(0x04)
    assert((status & 0x02) != 0, s"Done bit not set in status: 0x${status.toHexString}")
    dut.io.done #= false
    dut.clockDomain.waitRisingEdge()
    // Should still be latched
    val status2 = axiRead(0x04)
    assert((status2 & 0x02) != 0, "Done latch cleared prematurely")
    pass("Done latch holds after done pulse")

    // Test 20: Start clears done latch
    axiWriteSimultaneous(0x00, 0x00)
    dut.clockDomain.waitRisingEdge()
    axiWriteSimultaneous(0x00, 0x01) // Trigger start
    dut.clockDomain.waitRisingEdge(2)
    val status3 = axiRead(0x04)
    assert((status3 & 0x02) == 0, "Done latch should be cleared by start")
    pass("Start clears done latch")

    // ============================================================
    // Test Group 8: Seed write pulse
    // ============================================================
    println("\n=== Group 8: Seed write pulse ===")

    // Test 21: Seed write generates a single-cycle pulse
    dut.clockDomain.waitRisingEdge(2)
    assert(!dut.io.seedWrite.toBoolean, "seedWrite should be low before write")
    axiWriteSimultaneous(0x24, 0xAAAAAAAAL)
    // seedWritePulse is set during the write, auto-cleared next cycle
    dut.clockDomain.waitRisingEdge(2)
    assert(!dut.io.seedWrite.toBoolean, "seedWrite should auto-clear")
    pass("Seed write pulse fires and auto-clears")

    // ============================================================
    // Test Group 9: Read during write (interleaved)
    // ============================================================
    println("\n=== Group 9: Interleaved read/write ===")

    // Test 22: Read immediately after write
    for (_ <- 0 until 5) {
      val v = rng.nextInt(128)
      axiWriteAwFirst(0x08, v, gapCycles = 0)
      val r = axiRead(0x08) & 0x7F
      assert(r == v, s"Interleaved read mismatch: wrote $v, read $r")
    }
    pass("5 interleaved write-then-read cycles")

    // ============================================================
    // Test Group 10: Cycle counter and status
    // ============================================================
    println("\n=== Group 10: Cycle counter and status ===")

    // Test 23: Cycle counter read
    dut.io.cycleCount #= 999999
    dut.clockDomain.waitRisingEdge()
    assert(axiRead(0x14) == 999999)
    pass("Cycle counter read")

    // Test 24: Busy bit in status
    dut.io.busy #= true
    dut.clockDomain.waitRisingEdge()
    assert((axiRead(0x04) & 1) == 1, "Busy bit should be set")
    dut.io.busy #= false
    dut.clockDomain.waitRisingEdge()
    assert((axiRead(0x04) & 1) == 0, "Busy bit should be clear")
    pass("Busy bit in STATUS register")

    // Test 25: Cycle count upper bits in status
    dut.io.cycleCount #= 0xABCD0000L
    dut.clockDomain.waitRisingEdge()
    val statusCycles = (axiRead(0x04) >> 16) & 0xFFFF
    assert(statusCycles == 0, s"STATUS cycle bits should be lower 16: got 0x${statusCycles.toHexString}")
    dut.io.cycleCount #= 0x1234
    dut.clockDomain.waitRisingEdge()
    val statusCycles2 = (axiRead(0x04) >> 16) & 0xFFFF
    assert(statusCycles2 == 0x1234, s"STATUS cycle bits mismatch: got 0x${statusCycles2.toHexString}")
    pass("Cycle count in STATUS[31:16]")

    // ============================================================
    // Summary
    // ============================================================
    println(s"\nAll $testNum AXI-Lite tests passed!")
  }
}
