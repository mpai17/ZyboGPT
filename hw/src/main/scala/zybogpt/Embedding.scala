package zybogpt

import spinal.core._
import spinal.lib._

/** Token embedding + positional embedding lookup.
  *
  * Both embeddings stored in BRAM as INT16 (Q5.10 fixed-point).
  * Output is token_emb[token_id] + pos_emb[position] as INT16 (clamped to INT8 by Sequencer).
  *
  * Token embedding: 128 * 64 * 2 bytes = 16 KB (4 BRAM36 blocks)
  * Pos embedding:   128 * 64 * 2 bytes = 16 KB (4 BRAM36 blocks)
  * (Shared with output head via tied weights)
  *
  * Latency: 2 cycles (BRAM read + add).
  */
case class Embedding(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val io = new Bundle {
    val tokenId = in UInt (7 bits) // 0-127 ASCII
    val position = in UInt (7 bits) // 0-127 ctx_len
    val start = in Bool ()

    // Embedding output via address/data interface (replaces Vec to avoid mux trees)
    val embAddr = in UInt (log2Up(config.dModel) bits)
    val embData = out SInt (16 bits)
    val done = out Bool ()

    // For output head logit computation (reuse token embedding)
    val logitMode = in Bool ()
    val queryVec = in Vec (SInt(8 bits), config.dModel)
    val logitResult = out SInt (24 bits)
    val logitTokenId = in UInt (7 bits)
  }

  // Token embedding BRAM: 128 entries of d_model INT16 values
  // Organized as 128 * 64 = 8192 INT16 values
  // Using 32-bit wide BRAM: 128 * 32 = 4096 words (2 INT16 per word)
  val tokEmbMem = Mem(SInt(16 bits), config.vocabSize * config.dModel)
  val posEmbMem = Mem(SInt(16 bits), config.ctxLen * config.dModel)

  // Initialize from .mem files if available (loaded at elaboration time)
  {
    val tokFile = new java.io.File("../export/tok_emb_16b.mem")
    if (tokFile.exists()) {
      val lines = scala.io.Source.fromFile(tokFile).getLines()
        .map(_.trim).filter(_.nonEmpty).toSeq
      val initData = lines.map { hex =>
        val unsigned = Integer.parseInt(hex, 16)
        val signed = if (unsigned >= 0x8000) unsigned - 0x10000 else unsigned
        S(signed, 16 bits)
      }
      tokEmbMem.init(initData.padTo(config.vocabSize * config.dModel, S(0, 16 bits)))
    }
  }
  // Wide token embedding for parallel logit computation: 8 INT16 per 128-bit word
  // 1024 entries = 128 vocab × 8 groups (64 dims / 8 parallel)
  val tokEmbWide = Mem(Bits(128 bits), config.vocabSize * config.dModel / 8)
  locally {
    val tokFile = new java.io.File("../export/tok_emb_16b.mem")
    if (tokFile.exists()) {
      val lines = scala.io.Source.fromFile(tokFile).getLines()
        .map(_.trim).filter(_.nonEmpty).toSeq
      val values = lines.map(hex => Integer.parseInt(hex, 16))
      val initData = values.grouped(8).map { group =>
        val padded = group.padTo(8, 0)
        var bits = BigInt(0)
        for (i <- 0 until 8) {
          bits = bits | (BigInt(padded(i) & 0xFFFF) << (i * 16))
        }
        B(bits, 128 bits)
      }.toSeq
      tokEmbWide.init(initData.padTo(config.vocabSize * config.dModel / 8, B(0, 128 bits)))
    }
  }

  {
    val posFile = new java.io.File("../export/pos_emb_16b.mem")
    if (posFile.exists()) {
      val lines = scala.io.Source.fromFile(posFile).getLines()
        .map(_.trim).filter(_.nonEmpty).toSeq
      val initData = lines.map { hex =>
        val unsigned = Integer.parseInt(hex, 16)
        val signed = if (unsigned >= 0x8000) unsigned - 0x10000 else unsigned
        S(signed, 16 bits)
      }
      posEmbMem.init(initData.padTo(config.ctxLen * config.dModel, S(0, 16 bits)))
    }
  }

  // Element-by-element readout with counter
  val elemCounter = Reg(UInt(log2Up(config.dModel + 2) bits)) init 0
  val reading = RegInit(False)
  val outputMem = Mem(SInt(16 bits), config.dModel)

  // Addresses
  val tokBaseAddr = io.tokenId.resize(log2Up(config.vocabSize * config.dModel) bits) * config.dModel
  val posBaseAddr = io.position.resize(log2Up(config.ctxLen * config.dModel) bits) * config.dModel

  when(io.start && !io.logitMode) {
    reading := True
    elemCounter := 0
  }

  when(reading) {
    val tokAddr = tokBaseAddr + elemCounter.resize(tokBaseAddr.getWidth bits)
    val posAddr = posBaseAddr + elemCounter.resize(posBaseAddr.getWidth bits)

    val tokVal = tokEmbMem.readSync(tokAddr.resize(log2Up(config.vocabSize * config.dModel)))
    val posVal = posEmbMem.readSync(posAddr.resize(log2Up(config.ctxLen * config.dModel)))

    // Add with 1 cycle delay (sync read)
    val tokValD = RegNext(tokVal)
    val posValD = RegNext(posVal)
    val sum = tokValD + posValD

    when(elemCounter >= 2) { // Account for 2-cycle read latency
      outputMem.write((elemCounter - 2).resize(log2Up(config.dModel)), sum)
    }

    elemCounter := elemCounter + 1
    when(elemCounter === (config.dModel + 1)) {
      reading := False
    }
  }

  // Embedding output via address/data interface (readSync from BRAM)
  io.embData := outputMem.readSync(io.embAddr)
  val embDonePulse = RegNext(!reading && RegNext(reading, init = False), init = False)

  // --- Logit mode: 8× parallel dot product ---
  // Reads 8 INT16 embedding values per cycle from tokEmbWide (128-bit BRAM),
  // multiplies against 8 queryVec elements in parallel, adder tree → accumulate.
  // Processes 8 groups per logit via 4-stage pipeline.
  val numParallel = 8
  val numGroups = config.dModel / numParallel // 8

  val logitAcc = Reg(SInt(24 bits)) init 0
  val logitGroup = Reg(UInt(log2Up(numGroups + 2) bits)) init 0 // 0..numGroups+1
  val logitComputing = RegInit(False)
  val logitResultReg = Reg(SInt(24 bits)) init 0

  // 4-stage pipeline registers (breaks DSP cascade):
  // Stage 1: capture 8 queryVec + 8 embedding values from wide read
  val queryPipeRegs = Vec(Reg(SInt(8 bits)) init 0, numParallel)
  val embPipeRegs = Vec(Reg(SInt(16 bits)) init 0, numParallel)
  val logitPipeValid = RegInit(False)
  // Stage 2: 8 parallel multiplies → registered products (LUT-based, no DSPs)
  val productRegs = Vec(Reg(SInt(24 bits)) init 0, numParallel)
  for (i <- 0 until numParallel) {
    productRegs(i).addAttribute("USE_DSP", "no") // force LUT multiplies to prevent DSP cascade
  }
  val mulValid = RegInit(False)
  // Stage 3: balanced adder tree → registered partial sum (fabric CARRY4)
  val partialSumReg = Reg(SInt(28 bits)) init 0
  val addValid = RegInit(False)

  // Only start when not already computing (prevents restart race)
  when(io.start && io.logitMode && !logitComputing) {
    logitComputing := True
    logitGroup := 0
    logitAcc := 0
    logitPipeValid := False
    mulValid := False
    addValid := False
  }

  when(logitComputing) {
    // Wide BRAM read: addr = tokenId * numGroups + group (10 bits)
    val wideAddr = (io.logitTokenId.asBits ## logitGroup(log2Up(numGroups) - 1 downto 0)).asUInt
    val wideData = tokEmbWide.readSync(wideAddr)

    // Stage 1: Capture 8 queryVec + 8 embedding values from wide read
    // readSync latency: addr at group G → data at group G+1.
    when(logitGroup >= 1 && logitGroup <= numGroups) {
      // Extract 8 INT16 from 128-bit wide read
      for (i <- 0 until numParallel) {
        embPipeRegs(i) := wideData(i * 16 + 15 downto i * 16).asSInt
      }
      // Select 8 queryVec elements with compile-time constant indexing (avoids 64-to-1 mux)
      switch(logitGroup - 1) {
        for (g <- 0 until numGroups) {
          is(g) {
            for (i <- 0 until numParallel) {
              queryPipeRegs(i) := io.queryVec(g * numParallel + i)
            }
          }
        }
      }
      logitPipeValid := True
    } otherwise {
      logitPipeValid := False
    }

    // Stage 2: 8 parallel multiplies → product registers (LUT-based, no DSPs)
    when(logitPipeValid) {
      for (i <- 0 until numParallel) {
        productRegs(i) := queryPipeRegs(i) * embPipeRegs(i) // SInt(8) * SInt(16) = SInt(24)
      }
      mulValid := True
    } otherwise {
      mulValid := False
    }

    // Stage 3: balanced adder tree → registered partial sum (fabric CARRY4, no DSPs)
    when(mulValid) {
      val level1 = for (i <- 0 until 4) yield productRegs(2 * i) +^ productRegs(2 * i + 1)
      val level2 = for (i <- 0 until 2) yield level1(2 * i) +^ level1(2 * i + 1)
      partialSumReg := (level2(0) +^ level2(1)).resized
      addValid := True
    } otherwise {
      addValid := False
    }

    // Stage 4: accumulate registered partial sum into logitAcc
    when(addValid) {
      logitAcc := (logitAcc + partialSumReg).resized
    }

    // Advance group counter
    when(logitGroup <= numGroups) {
      logitGroup := logitGroup + 1
    }

    // Drain: all groups processed and all pipeline stages empty
    when(logitGroup > numGroups && !logitPipeValid && !mulValid && !addValid) {
      logitComputing := False
    }
  }

  // Latch result on falling edge of logitComputing (logitAcc has final value)
  val prevLogitComputing = RegNext(logitComputing, init = False)
  val logitJustDone = prevLogitComputing && !logitComputing
  when(logitJustDone) {
    logitResultReg := logitAcc
  }
  // Done pulse 1 cycle after latch (so logitResultReg has settled)
  val logitDonePulse = RegNext(logitJustDone, init = False)

  io.done := embDonePulse || logitDonePulse
  io.logitResult := logitResultReg
}
