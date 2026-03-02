package zybogpt

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axilite._

/** AXI4-Lite slave register interface for PS-PL communication.
  *
  * Register map (base: 0x43C0_0000):
  *   0x00 CONTROL   [W]  bit0: start, bit1: reset, bit2: mode (0=greedy, 1=sample)
  *   0x04 STATUS    [R]  bit0: busy, bit1: done, bit2: error, [31:16]: cycle_count[15:0]
  *   0x08 TOKEN_IN  [W]  [6:0]: input token ID
  *   0x0C TOKEN_OUT [R]  [6:0]: output token ID
  *   0x10 POSITION  [W]  [6:0]: current sequence position
  *   0x14 CYCLE_LO  [R]  cycle counter [31:0]
  *   0x18 CYCLE_HI  [R]  cycle counter [63:32] (reserved)
  *   0x1C CONFIG    [R]  [7:0]: d_model, [15:8]: n_layers, [23:16]: ctx_len, [31:24]: vocab_size
  *   0x20 SAMPLING  [W/R] [15:0]: inv_temp (0 = greedy argmax)
  *   0x24 SEED      [W/R] [31:0]: LFSR seed for temperature sampling
  */
case class AxiLiteSlave(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val io = new Bundle {
    val axi = slave(AxiLite4(AxiLite4Config(addressWidth = 8, dataWidth = 32)))

    // To accelerator
    val tokenIn = out UInt (7 bits)
    val position = out UInt (7 bits)
    val start = out Bool ()
    val reset_accel = out Bool ()
    val mode = out Bits (2 bits)

    // From accelerator
    val tokenOut = in UInt (7 bits)
    val busy = in Bool ()
    val done = in Bool ()
    val cycleCount = in UInt (32 bits)

    // Sampling configuration
    val invTemp = out UInt (16 bits)
    val seed = out UInt (32 bits)
    val seedWrite = out Bool ()  // Pulse on seed register write
  }

  // Registers
  val controlReg = Reg(Bits(32 bits)) init 0
  val tokenInReg = Reg(UInt(7 bits)) init 0
  val positionReg = Reg(UInt(7 bits)) init 0
  val doneLatched = RegInit(False)
  val invTempReg = Reg(UInt(16 bits)) init 0
  val seedReg = Reg(UInt(32 bits)) init 0xDEADBEEFL
  val seedWritePulse = RegInit(False)

  // Control outputs
  io.start := controlReg(0) && !RegNext(controlReg(0), init = False) // Rising edge
  io.reset_accel := controlReg(1)
  io.mode := controlReg(3 downto 2)
  io.tokenIn := tokenInReg
  io.position := positionReg
  io.invTemp := invTempReg
  io.seed := seedReg
  io.seedWrite := seedWritePulse

  // Latch done
  when(io.done) {
    doneLatched := True
  }
  when(io.start) {
    doneLatched := False
  }

  // Clear seed write pulse each cycle (single-cycle pulse)
  seedWritePulse := False

  // AXI4-Lite write logic
  // AW and W channels may arrive on different cycles. Latch each independently,
  // perform the register write when both have arrived, then send B response.
  val gotAw = RegInit(False)
  val gotW = RegInit(False)
  val writeAddr = Reg(Bits(8 bits))
  val writeData = Reg(Bits(32 bits))
  val bValid = RegInit(False)

  // Accept AW when we haven't already latched one (and no pending response)
  io.axi.aw.ready := !gotAw && !bValid
  when(io.axi.aw.valid && io.axi.aw.ready) {
    gotAw := True
    writeAddr := io.axi.aw.addr(7 downto 0).asBits
  }

  // Accept W when we haven't already latched one (and no pending response)
  io.axi.w.ready := !gotW && !bValid
  when(io.axi.w.valid && io.axi.w.ready) {
    gotW := True
    writeData := io.axi.w.data
  }

  // When both AW and W received, perform register write and assert B response
  when(gotAw && gotW) {
    gotAw := False
    gotW := False
    bValid := True

    switch(writeAddr) {
      is(0x00) { controlReg := writeData }
      is(0x08) { tokenInReg := writeData(6 downto 0).asUInt }
      is(0x10) { positionReg := writeData(6 downto 0).asUInt }
      is(0x20) { invTempReg := writeData(15 downto 0).asUInt }
      is(0x24) {
        seedReg := writeData.asUInt
        seedWritePulse := True
      }
    }
  }

  // Write response: clear after handshake
  io.axi.b.valid := bValid
  io.axi.b.resp := B"00" // OKAY
  when(io.axi.b.valid && io.axi.b.ready) {
    bValid := False
  }

  // AXI4-Lite read logic
  val readReady = RegInit(True).allowUnsetRegToAvoidLatch
  io.axi.ar.ready := readReady

  val readData = Reg(Bits(32 bits)) init 0
  val readValid = RegInit(False)

  when(io.axi.ar.valid && readReady) {
    val addr = io.axi.ar.addr(7 downto 0)
    readValid := True

    switch(addr) {
      is(0x00) { readData := controlReg }
      is(0x04) {
        // STATUS: [31:16]=cycleCount[15:0], [15:3]=0, bit2=0, bit1=done, bit0=busy
        readData := io.cycleCount(15 downto 0).asBits ##
          B(0, 13 bits) ## B"0" ## doneLatched.asBits ## io.busy.asBits
      }
      is(0x08) { readData := B(0, 25 bits) ## tokenInReg.asBits }
      is(0x0C) { readData := B(0, 25 bits) ## io.tokenOut.asBits }
      is(0x10) { readData := B(0, 25 bits) ## positionReg.asBits }
      is(0x14) { readData := io.cycleCount.asBits }
      is(0x18) { readData := B(0, 32 bits) }
      is(0x1C) {
        readData := B(config.vocabSize, 8 bits) ## B(config.ctxLen, 8 bits) ##
          B(config.nLayers, 8 bits) ## B(config.dModel, 8 bits)
      }
      is(0x20) { readData := B(0, 16 bits) ## invTempReg.asBits }
      is(0x24) { readData := seedReg.asBits }
      default { readData := B(0, 32 bits) }
    }
  }

  when(io.axi.r.ready && readValid) {
    readValid := False
  }

  io.axi.r.valid := readValid
  io.axi.r.data := readData
  io.axi.r.resp := B"00" // OKAY
}
