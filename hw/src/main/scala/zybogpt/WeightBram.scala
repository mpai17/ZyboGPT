package zybogpt

import spinal.core._
import spinal.lib._

/** Weight BRAM storage with multi-cycle loading FSM + serial TDot compute.
  *
  * Stores all ternary weights in packed 1.6-bit format (32-bit wide BRAM).
  * On loadStart, reads packed bytes from BRAM, decodes via 4 WeightDecoder
  * instances (20 trits/cycle = 40 bits), and writes each batch to
  * decodedBankMem via a single write port (no register file = no mux tree).
  *
  * After loading, assembles 64-trit rows from decodedBankMem (4 reads per row)
  * and feeds them to a single TDotUnit serially.
  *
  * This eliminates the 103-bank register file and 31 TDotUnits.
  *
  * Total: 19,776 bytes ternary weights.
  */
case class WeightBram(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val totalTrits = config.numTDots * config.dModel // 32 * 64 = 2048
  val tritsPerRead = 20 // 4 bytes -> 4 decoders x 5 trits
  val readsNeeded = (totalTrits + tritsPerRead - 1) / tritsPerRead // ceil(2048/20) = 103
  val tritsPerRow = config.dModel // 64

  val io = new Bundle {
    // Weight loading interface
    val loadAddr = in UInt (16 bits)
    val loadStart = in Bool ()
    val loadDone = out Bool ()

    // TDot compute interface
    val tdotX = in Vec (SInt(8 bits), config.dModel)
    val computeStart = in Bool ()
    val computeDone = out Bool ()
    val results = out Vec (SInt(24 bits), config.numTDots)

    // Weight scale LUT
    val scaleAddr = in UInt (8 bits)
    val scaleData = out SInt (16 bits)
  }

  // Main weight BRAM (32-bit wide)
  val weightDepth = (config.totalTernaryBytes + 3) / 4
  val weightMem = Mem(Bits(32 bits), wordCount = weightDepth)

  {
    val memFile = new java.io.File("../export/weights_ternary_32b.mem")
    if (memFile.exists()) {
      val lines = scala.io.Source.fromFile(memFile).getLines()
        .map(_.trim).filter(_.nonEmpty).toSeq
      val initData = lines.map { hex =>
        B(BigInt(hex, 16), 32 bits)
      }
      weightMem.init(initData.padTo(weightDepth, B(0, 32 bits)))
    }
  }

  // ---- Decoded bank storage in BRAM ----
  // 103 banks of 40 bits (20 trits × 2 bits), replaces Vec(Reg()) register file.
  // Single write port eliminates 103-to-1 storeIdx decode tree.
  val decodedBankMem = Mem(Bits(tritsPerRead * 2 bits), readsNeeded) // 103 × 40 bits

  // 4 weight decoders
  val decoders = Array.fill(4)(WeightDecoder())

  // ---- Loading FSM ----
  val loading = RegInit(False)
  val readAddr = Reg(UInt(log2Up(weightDepth) bits)) init 0
  val readCounter = Reg(UInt(log2Up(readsNeeded + 1) bits)) init 0
  val loadCompleteReg = RegInit(False)

  val readData = weightMem.readSync(readAddr)
  val readValid = RegNext(loading, init = False)

  for (i <- 0 until 4) {
    decoders(i).io.packed := readData((i + 1) * 8 - 1 downto i * 8).asUInt
    decoders(i).io.valid_in := readValid
  }

  // Pack 20 decoded trits into 40-bit word
  val decodedBits = Bits(tritsPerRead * 2 bits)
  for (d <- 0 until 4) {
    for (t <- 0 until 5) {
      val idx = d * 5 + t
      decodedBits(idx * 2 + 1 downto idx * 2) := decoders(d).io.trits(t)
    }
  }

  val storeIdx = Reg(UInt(log2Up(readsNeeded + 1) bits)) init 0
  val decodersValid = decoders(0).io.valid_out

  when(io.loadStart && !loading) {
    loading := True
    readAddr := (io.loadAddr >> 2).resize(log2Up(weightDepth))
    readCounter := 0
    storeIdx := 0
    loadCompleteReg := False
  }

  when(loading) {
    readAddr := readAddr + 1
    readCounter := readCounter + 1
    when(readCounter === (readsNeeded - 1)) {
      loading := False
    }
  }

  // Write decoded trits to BRAM (single write port = no mux tree)
  when(decodersValid) {
    decodedBankMem.write(storeIdx.resize(log2Up(readsNeeded) bits), decodedBits)
    storeIdx := storeIdx + 1
    when(storeIdx === (readsNeeded - 1)) {
      loadCompleteReg := True
    }
  }

  when(io.loadStart) {
    loadCompleteReg := False
  }

  io.loadDone := loadCompleteReg

  // ---- Serial TDot Compute ----
  // For each of 32 TDot rows, assemble 64 trits from decodedBankMem banks.
  // Each row's trits are at flat indices [row*64 .. row*64+63].
  // Flat index / 20 = bank index, flat index % 20 = offset within bank.
  //
  // Row 0: banks 0,1,2,3 (trits 0-19, 20-39, 40-59, 60-63 from bank 3)
  // Row 1: banks 3,4,5,6 (trits 64-79 from bank 3, 80-99, 100-119, 120-127 from bank 6)
  // etc.
  //
  // Each row needs data from 4 banks (ceil(64/20) = 4, since 3*20=60 < 64).
  // Read 4 banks sequentially, assemble weight vector, then fire TDotUnit.
  //
  // Precompute at elaboration time which banks and offsets each row needs.

  val tdot = TDotUnit(config.dModel)

  object ComputeState extends SpinalEnum {
    val IDLE, ASSEMBLE, FIRE, CAPTURE = newElement()
  }
  val computeState = RegInit(ComputeState.IDLE)
  val computeRow = Reg(UInt(log2Up(config.numTDots) bits)) init 0
  val assembleStep = Reg(UInt(3 bits)) init 0 // 0..3 (4 bank reads per row)
  val computeDoneReg = RegInit(False)

  val weightReg = Vec(Reg(Bits(2 bits)) init B"00", config.dModel)
  val resultRegs = Vec(Reg(SInt(24 bits)) init 0, config.numTDots)

  // Bank read port
  val bankReadAddr = UInt(log2Up(readsNeeded) bits)
  bankReadAddr := 0
  val bankReadData = decodedBankMem.readSync(bankReadAddr)

  // Precompute bank mapping at elaboration time.
  // For TDot row r, the flat trit indices are [r*64 .. r*64+63].
  // We need to read from banks floor(r*64/20)..floor((r*64+63)/20) = up to 4 banks.
  // For each row, precompute (bankIdx, startOffset, numTrits) for each read step.
  case class BankSlice(bankIdx: Int, bankOffset: Int, rowOffset: Int, numTrits: Int)

  val rowSlices: Array[Array[BankSlice]] = Array.tabulate(config.numTDots) { row =>
    val startFlat = row * tritsPerRow // 0, 64, 128, ...
    val endFlat = startFlat + tritsPerRow - 1
    val startBank = startFlat / tritsPerRead
    val endBank = endFlat / tritsPerRead
    (startBank to endBank).zipWithIndex.map { case (bank, step) =>
      val bankStart = bank * tritsPerRead
      val sliceStart = Math.max(startFlat, bankStart) - bankStart // offset within bank
      val sliceEnd = Math.min(endFlat, bankStart + tritsPerRead - 1) - bankStart
      val rowOff = Math.max(startFlat, bankStart) - startFlat // offset within row
      BankSlice(bank, sliceStart, rowOff, sliceEnd - sliceStart + 1)
    }.toArray
  }

  // Maximum number of bank reads per row
  val maxReadsPerRow = rowSlices.map(_.length).max // should be 4

  // TDotUnit connections
  // Register tdotX to break combinational path from Attention FSM state
  // through the tdotX mux tree into TDot's adder tree.
  // tdotX is stable for many cycles before FIRE, so 1-cycle latency is transparent.
  val tdotXBuf = Vec(Reg(SInt(8 bits)) init 0, config.dModel)
  tdotXBuf := io.tdotX
  tdot.io.x := tdotXBuf
  tdot.io.w := weightReg
  tdot.io.valid_in := (computeState === ComputeState.FIRE)

  switch(computeState) {
    is(ComputeState.IDLE) {
      when(io.computeStart && loadCompleteReg) {
        computeState := ComputeState.ASSEMBLE
        computeRow := 0
        assembleStep := 0
        computeDoneReg := False
        // Prime read-ahead: bank for row 0, step 0
        bankReadAddr := U(rowSlices(0)(0).bankIdx, log2Up(readsNeeded) bits)
      }
    }

    is(ComputeState.ASSEMBLE) {
      // Read bank data and extract trits into weightReg.
      // readSync: addr primed previous cycle, data available now.
      // Each step processes one bank's contribution to the current row.

      // Unroll all possible (row, step) combinations at elaboration time
      // to determine bank addresses and trit mappings with constant indices.
      for (row <- 0 until config.numTDots) {
        when(computeRow === row) {
          val slices = rowSlices(row)
          for (step <- 0 until slices.length) {
            when(assembleStep === step) {
              // Extract trits from bankReadData at known offsets
              val slice = slices(step)
              for (t <- 0 until slice.numTrits) {
                val bankBitPos = (slice.bankOffset + t) * 2
                val rowPos = slice.rowOffset + t
                weightReg(rowPos) := bankReadData(bankBitPos + 1 downto bankBitPos)
              }

              // Prime next read
              val nextStep = step + 1
              if (nextStep < slices.length) {
                bankReadAddr := U(slices(nextStep).bankIdx, log2Up(readsNeeded) bits)
              }
            }
          }

          // Advance step
          assembleStep := assembleStep + 1
          when(assembleStep === (slices.length - 1)) {
            computeState := ComputeState.FIRE
            assembleStep := 0
          }
        }
      }
    }

    is(ComputeState.FIRE) {
      // TDotUnit valid_in driven combinationally above.
      // Result arrives 1 cycle later.
      computeState := ComputeState.CAPTURE
    }

    is(ComputeState.CAPTURE) {
      when(tdot.io.valid_out) {
        resultRegs(computeRow) := tdot.io.result
        computeRow := computeRow + 1
        when(computeRow === (config.numTDots - 1)) {
          computeDoneReg := True
          computeState := ComputeState.IDLE
        } otherwise {
          computeState := ComputeState.ASSEMBLE
          assembleStep := 0
          // Prime bank read for next row
          val nextRow = (computeRow + 1).resize(log2Up(config.numTDots) bits)
          // Use row 0's bank as default (will be overridden by unrolled logic)
          for (row <- 0 until config.numTDots) {
            when(nextRow === row) {
              bankReadAddr := U(rowSlices(row)(0).bankIdx, log2Up(readsNeeded) bits)
            }
          }
        }
      }
    }
  }

  when(io.loadStart) {
    computeDoneReg := False
  }

  io.computeDone := computeDoneReg
  io.results := resultRegs

  // ---- Scale LUT ----
  val numScales = config.nLayers * 6
  val scaleMem = Mem(SInt(16 bits), wordCount = numScales max 16)

  if (WeightInit.scaleValues.nonEmpty) {
    val scaleInit = WeightInit.scaleValues.map(v => S(v, 16 bits)).toSeq
    scaleMem.init(scaleInit.padTo(numScales max 16, S(0, 16 bits)))
  }

  io.scaleData := scaleMem.readSync(io.scaleAddr.resize(log2Up(numScales max 16)))
}
