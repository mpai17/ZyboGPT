package zybogpt

import spinal.core._
import spinal.core.sim._

/** Pipeline debug testbench: captures internal activations at every FSM stage
  * boundary during position 6 inference and compares against Python reference.
  *
  * Identifies the first pipeline stage where HW diverges from the bit-accurate
  * Python reference_inference.py, guiding root-cause analysis.
  */
object ZyboGPTPipelineDebugSim extends App {
  // ================================================================
  // Reference arrays from Python reference_inference.py --dump-step 6
  // (token=10, position=6, first autoregressive step)
  // Model: two-phase curriculum (float pretrain + hw-mode fine-tune)
  // ================================================================

  val refAfterEmbed = Array(
      -22, 38, 38, -23, -25, -8, -68, 86, 94, 24, 36, -125, 26, 5, 36, -120,
      -18, -86, -15, 37, 86, 46, 10, -90, 88, -20, -59, -21, 28, 23, 31, -34,
      65, 46, -44, -99, 9, 41, 56, -25, -86, -32, -24, 40, -23, -36, 22, 22,
      -38, -25, 35, -58, 14, 47, 110, -5, -91, 30, -29, 72, 8, 54, -13, 62)

  val refAfterLayer0AttnNorm = Array(
      -8, 17, 4, -7, -3, -3, -8, 14, 18, 15, 13, -15, 12, 1, 7, -16,
      -2, -9, -3, 5, 10, 5, 4, -12, 8, -9, -10, -6, 23, 3, 23, -11,
      8, 14, -7, -17, 1, 5, 9, -7, -13, -21, -8, 22, -4, -5, 3, 14,
      -28, -8, 25, -16, 9, 35, 13, -3, -12, 13, -24, 8, 2, 5, -2, 10)

  val refAfterLayer0QProj = Array(
      6, -7, 4, -1, -5, 5, 2, 6, 2, 1, -3, -3, -5, -9, 3, -2,
      0, -10, 4, -9, -7, 1, -4, -3, -3, -9, 1, -7, -3, 7, -1, -6,
      -5, -6, 3, 1, 4, -1, -4, -1, 4, -12, -4, -5, 2, 0, 8, 8,
      -2, 1, -8, -8, -3, 0, 3, -1, -3, 7, -10, 6, 3, 3, 5, 2)

  val refAfterLayer0KProj = Array(
      14, -5, 5, 20, 4, -25, -16, -23, -19, 16, 14, -15, -17, 28, -11, 14,
      -16, -18, -13, 23, 20, -12, -16, 17, 19, 26, 12, 20, 18, -25, 21, 19,
      23, 17, -20, -21, -21, 5, 17, -9, -18, 20, 23, -13, -18, -8, 10, -24,
      22, -23, 19, -14, -19, -19, -19, -17, 15, -23, -9, -21, -21, -16, -24, -18)

  val refAfterLayer0VProj = Array(
      1, -1, -11, -2, 4, -9, 8, -1, 7, 3, 5, 3, 2, -9, -2, 5,
      -1, 0, 0, -7, 1, 5, -3, -3, -1, 1, 1, 1, -2, 0, 1, 4,
      8, 0, -12, 19, 1, -11, 7, -10, 1, -8, -4, -20, 5, -13, 19, 2,
      1, 10, 2, -9, 8, 6, 11, -14, -11, -1, -7, 10, -4, -8, -5, -7)

  val refAfterLayer0AttnOut = Array(
      8, 7, 8, 7, -2, -7, -1, -4, -16, -5, -1, -8, -9, 0, 5, 1,
      -1, 8, 12, 5, -17, -1, -1, 3, -8, 10, 6, 22, -12, 5, 10, 3,
      5, 10, 14, -4, -5, 5, -11, -4, -10, 0, 0, 2, -9, -15, 4, -16,
      0, 4, -4, -3, 9, 2, -15, 10, 0, -2, 7, -4, 15, 7, 19, -1)

  val refAfterLayer0FfNorm = Array(
      -20, 46, 30, -14, -20, -17, -48, 72, 58, 26, 31, -92, 19, 6, 46, -80,
      -11, -42, -3, 38, 44, 28, 12, -58, 42, -14, -30, 0, 27, 21, 79, -25,
      39, 61, -25, -128, 2, 34, 39, -24, -87, -39, -24, 47, -21, -33, 15, 9,
      -60, -19, 54, -46, 36, 54, 61, 7, -47, 28, -43, 51, 36, 54, 5, 57)

  val refAfterLayer0FfDown = Array(
      13, 2, -43, -1, 28, -7, 51, -43, -49, -9, -11, 78, 3, 21, -5, 87,
      10, 50, -11, -16, -69, -40, 5, 51, -36, 18, 26, 15, 28, -11, -2, 3,
      -52, -12, 17, 77, -39, -10, -30, 23, 61, -10, 8, -1, 30, 44, -4, -6,
      -4, -8, 8, 50, 5, -16, -56, 0, 67, 6, -20, -69, 11, -45, -9, -47)

  val refAfterLayer1AttnNorm = Array(
      -2, 48, 2, -18, 0, -24, -23, 38, 31, 9, 27, -50, 18, 26, 33, -36,
      -8, -30, -17, 32, 0, 3, 17, -42, 41, 7, -21, 16, 46, 18, 42, -29,
      21, 49, -13, -32, -44, 46, 16, -8, -34, -46, -17, 45, -3, -10, 21, 0,
      -41, -32, 43, -14, 30, 30, 32, 5, -26, 37, -49, -2, 39, 15, -3, 16)

  val refAfterLayer1QProj = Array(
      -10, -8, 6, 3, -15, 11, -17, -9, 14, 5, 25, 5, -1, -27, -14, 4,
      -2, 4, -2, -1, 6, 16, 3, -2, 21, -4, 15, 13, -10, 8, 8, -13,
      -10, 19, 1, 2, -6, -9, -10, -1, -26, 3, 1, -6, -1, -1, 0, -5,
      -16, 5, 8, 3, -8, -11, -5, -15, 7, 10, -9, -3, -12, 9, 2, -16)

  val refAfterLayer1KProj = Array(
      -2, -2, 42, -32, 37, 24, 12, 28, -7, -9, -12, -59, -12, -43, 25, 39,
      -17, 6, 14, 3, -22, 7, -13, -11, 24, -40, 4, 15, -53, -55, -20, -20,
      -13, -3, 16, -42, -3, 12, -29, -31, 30, 25, 17, -6, -48, 24, -21, 2,
      -22, -5, 28, -31, 17, 22, -3, -28, 26, -1, -43, 4, 26, -10, 26, -8)

  val refAfterLayer1VProj = Array(
      -14, -7, 21, 19, -4, -14, 15, 6, 1, 14, -2, 15, -10, 12, 4, -32,
      2, -31, -3, -12, 15, 6, 6, 23, -5, -22, 33, -11, 18, 1, -21, -3,
      25, -38, 15, -17, -42, -8, -13, -16, -35, 36, -24, -8, -20, -17, 4, -16,
      7, -21, -29, -18, 30, -21, 2, -29, 15, 26, 6, -13, -30, -18, 16, -13)

  val refAfterLayer1AttnOut = Array(
      -17, -22, 16, 1, 0, 21, -7, -10, -14, -6, -3, 7, -20, -23, -19, 3,
      -2, 6, 14, 6, 11, -1, -19, 10, 0, -5, 11, -3, -23, -18, -18, 15,
      0, -17, 10, 5, 5, 5, 7, -10, 5, 16, 3, -25, 8, -4, -12, -1,
      19, 15, -16, 6, -22, -29, -8, -19, 5, -14, 23, 6, -18, 11, 14, -1)

  val refAfterLayer1FfNorm = Array(
      -114, 127, 118, -102, 6, -8, -128, 127, 94, 24, 127, -128, 0, 18, 106, -128,
      -70, -128, 0, 127, 68, 24, -32, -128, 127, 18, -102, 80, 127, -8, 127, -82,
      112, 127, -20, -128, -128, 127, 127, -102, -128, -128, -82, 100, 36, -70, 62, -8,
      -128, -88, 127, -32, 36, 24, 127, -88, -120, 124, -120, 30, 100, 127, 68, 80)

  val refAfterLayer1FfDown = Array(
      -8, -9, 51, 3, 42, 12, -38, 8, 38, 12, -34, -35, 23, 10, -17, 21,
      28, -21, 32, 20, 48, 27, -28, -27, 29, 5, 45, -38, -17, -62, -11, 1,
      28, 16, -50, 26, -21, -64, 16, 33, 2, 16, 38, -30, -11, 32, -30, -2,
      16, 21, 5, -31, -19, 12, -30, -4, 49, -4, 31, -11, -19, 9, -39, -3)

  val refAfterFinalNorm = Array(
      -104, 62, 127, -52, 127, 42, -128, 127, 127, 62, -52, -128, 90, 50, 0, -32,
      66, -128, 126, 127, 127, 122, -128, -128, 127, 30, 114, -100, 14, -128, 38, -48,
      127, 127, -128, 18, -128, -92, 127, 66, -112, -40, 98, -56, -20, 82, -80, -12,
      -28, 26, 110, -128, -52, 62, 2, -72, 118, 62, 46, -24, -12, 127, -112, 38)

  // Expected output token at position 6
  val expectedToken = 73

  // Ordered list of stages for comparison
  val stageOrder = Array(
    "after_embed",
    "after_layer0_attn_norm", "after_layer0_q_proj", "after_layer0_k_proj",
    "after_layer0_v_proj", "after_layer0_attn_out",
    "after_layer0_ff_norm", "after_layer0_ff_down",
    "after_layer1_attn_norm", "after_layer1_q_proj", "after_layer1_k_proj",
    "after_layer1_v_proj", "after_layer1_attn_out",
    "after_layer1_ff_norm", "after_layer1_ff_down",
    "after_final_norm"
  )

  val refMap: Map[String, Array[Int]] = Map(
    "after_embed" -> refAfterEmbed,
    "after_layer0_attn_norm" -> refAfterLayer0AttnNorm,
    "after_layer0_q_proj" -> refAfterLayer0QProj,
    "after_layer0_k_proj" -> refAfterLayer0KProj,
    "after_layer0_v_proj" -> refAfterLayer0VProj,
    "after_layer0_attn_out" -> refAfterLayer0AttnOut,
    "after_layer0_ff_norm" -> refAfterLayer0FfNorm,
    "after_layer0_ff_down" -> refAfterLayer0FfDown,
    "after_layer1_attn_norm" -> refAfterLayer1AttnNorm,
    "after_layer1_q_proj" -> refAfterLayer1QProj,
    "after_layer1_k_proj" -> refAfterLayer1KProj,
    "after_layer1_v_proj" -> refAfterLayer1VProj,
    "after_layer1_attn_out" -> refAfterLayer1AttnOut,
    "after_layer1_ff_norm" -> refAfterLayer1FfNorm,
    "after_layer1_ff_down" -> refAfterLayer1FfDown,
    "after_final_norm" -> refAfterFinalNorm
  )

  // FSM state ordinals (binarySequential encoding)
  object SeqSt  { val IDLE=0; val EMBED=1; val LAYER_LOOP=2; val FINAL_NORM=3; val OUTPUT_LOGITS=4; val ARGMAX=5; val SAMPLING=6; val DONE=7 }
  object LaySt  { val IDLE=0; val ATTN_NORM=1; val ATTENTION=2; val ATTN_RESIDUAL=3; val FF_NORM=4; val FEEDFORWARD=5; val FF_RESIDUAL=6; val DONE=7 }
  object AttnSt { val IDLE=0; val Q_PROJ=1; val K_PROJ=2; val V_PROJ=3; val STORE_KV=4; val ATTN_SCORE=5; val SCALE_MASK=6; val SOFTMAX_WAIT=7; val ATTN_VALUE=8; val CONCAT=9; val O_PROJ=10; val DONE=11 }
  object FfnSt  { val IDLE=0; val UP_PROJ=1; val STORE_UP=2; val LOAD_SLICE=3; val DOWN_PROJ=4; val DONE=5 }

  val prompt = Array(82, 79, 77, 69, 79, 58) // R, O, M, E, O, :

  println("=" * 70)
  println("  ZyboGPT Pipeline Debug: Position 6 Stage-by-Stage Comparison")
  println("=" * 70)

  SimConfig.withWave.compile(ZyboGPTTop()).doSim { dut =>
    dut.clockDomain.forkStimulus(period = 10)

    // Initialize AXI signals
    dut.io.axi.aw.valid #= false
    dut.io.axi.aw.addr #= 0
    dut.io.axi.w.valid #= false
    dut.io.axi.w.data #= 0
    dut.io.axi.w.strb #= 0xF
    dut.io.axi.b.ready #= true
    dut.io.axi.ar.valid #= false
    dut.io.axi.ar.addr #= 0
    dut.io.axi.r.ready #= true

    dut.clockDomain.waitRisingEdge(10)

    // Configure T=0.5 sampling (inv_temp=512) matching test vector generation
    def axiWriteInit(addr: Int, data: Long): Unit = {
      dut.io.axi.aw.valid #= true
      dut.io.axi.aw.addr #= addr
      dut.io.axi.w.valid #= true
      dut.io.axi.w.data #= data
      dut.io.axi.w.strb #= 0xF
      dut.clockDomain.waitRisingEdge()
      dut.io.axi.aw.valid #= false
      dut.io.axi.w.valid #= false
      dut.clockDomain.waitRisingEdge(2)
    }
    axiWriteInit(0x24, 0xDEADBEEFL) // SEED: must be set before inv_temp
    axiWriteInit(0x20, 0x0200) // SAMPLING: inv_temp=512 (T=0.5)

    // ---- AXI helpers (same as ZyboGPTRomeoSim) ----
    def axiWrite(addr: Int, data: Long): Unit = {
      dut.io.axi.aw.valid #= true
      dut.io.axi.aw.addr #= addr
      dut.io.axi.w.valid #= true
      dut.io.axi.w.data #= data
      dut.io.axi.w.strb #= 0xF
      dut.clockDomain.waitRisingEdge()
      dut.io.axi.aw.valid #= false
      dut.io.axi.w.valid #= false
      dut.clockDomain.waitRisingEdge(2)
    }

    def axiRead(addr: Int): Long = {
      dut.io.axi.ar.valid #= true
      dut.io.axi.ar.addr #= addr
      dut.clockDomain.waitRisingEdge()
      dut.io.axi.ar.valid #= false
      var timeout = 50
      while (!dut.io.axi.r.valid.toBoolean && timeout > 0) {
        dut.clockDomain.waitRisingEdge()
        timeout -= 1
      }
      assert(timeout > 0, s"AXI read timeout at addr 0x${addr.toHexString}")
      val data = dut.io.axi.r.data.toLong
      dut.clockDomain.waitRisingEdge()
      data
    }

    def runInference(token: Int, position: Int): (Int, Long) = {
      axiWrite(0x08, token.toLong)
      axiWrite(0x10, position.toLong)
      axiWrite(0x00, 0x01)
      dut.clockDomain.waitRisingEdge(3)
      axiWrite(0x00, 0x00)
      dut.clockDomain.waitRisingEdge(2)
      var timeout = 100000
      var done = false
      while (!done && timeout > 0) {
        dut.clockDomain.waitRisingEdge(50)
        timeout -= 50
        val status = axiRead(0x04)
        done = (status & 2) != 0
      }
      assert(done, s"Inference timed out at token=$token pos=$position")
      val tokenOut = (axiRead(0x0C) & 0x7F).toInt
      val cycles = axiRead(0x14)
      (tokenOut, cycles)
    }

    // ---- Signal read helpers ----
    // SInt(8 bits).toInt may return unsigned raw bits; ensure signed interpretation
    def s8(v: Int): Int = { val m = v & 0xFF; if (m >= 128) m - 256 else m }

    def readActBuf(): Array[Int] =
      (0 until 64).map(i => s8(dut.sequencer.actBuf(i).toInt)).toArray
    def readNormedBuf(): Array[Int] =
      (0 until 64).map(i => s8(dut.transformerLayer.normedBuf(i).toInt)).toArray
    def readQBuf(): Array[Int] =
      (0 until 64).map(i => s8(dut.transformerLayer.attention.qBuf(i).toInt)).toArray
    def readKBuf(): Array[Int] =
      (0 until 64).map(i => s8(dut.transformerLayer.attention.kBuf(i).toInt)).toArray
    def readVBuf(): Array[Int] =
      (0 until 64).map(i => s8(dut.transformerLayer.attention.vBuf(i).toInt)).toArray
    def readAttnOutBuf(): Array[Int] =
      (0 until 64).map(i => s8(dut.transformerLayer.attention.attnOutBuf(i).toInt)).toArray
    def readFfnResultBuf(): Array[Int] =
      (0 until 64).map(i => s8(dut.transformerLayer.ffn.resultBuf(i).toInt)).toArray

    // =========================================================
    // Phase 1: Process prompt (positions 0-5) — build KV cache
    // =========================================================
    println("\n--- Processing prompt ROMEO: (positions 0-5) ---")
    for (i <- prompt.indices) {
      val (tokenOut, cycles) = runInference(prompt(i), i)
      val ch = if (tokenOut >= 32 && tokenOut < 127) s"'${tokenOut.toChar}'" else f"0x$tokenOut%02x"
      println(f"  pos=$i: token=${prompt(i)}%3d -> $ch%6s ($cycles cycles)")
    }

    // =========================================================
    // Phase 2: Position 6 with pipeline monitoring
    // =========================================================
    println("\n--- Starting position 6 (token=10 '\n') with pipeline monitoring ---")

    // Start inference
    axiWrite(0x08, 10L)   // token = 10
    axiWrite(0x10, 6L)    // position = 6
    axiWrite(0x00, 0x01)  // trigger start
    dut.clockDomain.waitRisingEdge(3)
    axiWrite(0x00, 0x00)  // clear start
    dut.clockDomain.waitRisingEdge(2)

    // Fork monitoring thread to capture activations at FSM transitions
    val captures = scala.collection.mutable.LinkedHashMap[String, Array[Int]]()
    @volatile var monitoring = true

    fork {
      var prevSeq = -1
      var prevLay = -1
      var prevAttn = -1
      var prevFfn = -1
      var curLayer = 0

      while (monitoring) {
        dut.clockDomain.waitRisingEdge()

        val sq = dut.sequencer.state.toBigInt.toInt
        val ly = dut.transformerLayer.state.toBigInt.toInt

        // Track which layer we're in (capture on entry to ATTN_NORM)
        if (ly == LaySt.ATTN_NORM && prevLay != LaySt.ATTN_NORM) {
          curLayer = dut.sequencer.layerCounter.toInt
        }

        // === Sequencer transitions ===
        if (sq != prevSeq) {
          if (prevSeq == SeqSt.EMBED && sq == SeqSt.LAYER_LOOP) {
            dut.clockDomain.waitRisingEdge()
            captures("after_embed") = readActBuf()
          }
          if (prevSeq == SeqSt.FINAL_NORM && sq == SeqSt.OUTPUT_LOGITS) {
            dut.clockDomain.waitRisingEdge()
            captures("after_final_norm") = readActBuf()
          }
          if (sq == SeqSt.DONE) {
            monitoring = false
          }
        }

        // === TransformerLayer transitions (only during LAYER_LOOP) ===
        if (ly != prevLay && sq == SeqSt.LAYER_LOOP) {
          val L = curLayer

          if (prevLay == LaySt.ATTN_NORM && ly == LaySt.ATTENTION) {
            dut.clockDomain.waitRisingEdge()
            captures(s"after_layer${L}_attn_norm") = readNormedBuf()
          }
          if (prevLay == LaySt.FF_NORM && ly == LaySt.FEEDFORWARD) {
            dut.clockDomain.waitRisingEdge()
            captures(s"after_layer${L}_ff_norm") = readNormedBuf()
          }
        }

        // === Attention sub-state transitions ===
        if (ly == LaySt.ATTENTION && sq == SeqSt.LAYER_LOOP) {
          val at = dut.transformerLayer.attention.state.toBigInt.toInt
          val L = curLayer

          if (at != prevAttn) {
            if (prevAttn == AttnSt.Q_PROJ && at == AttnSt.K_PROJ) {
              dut.clockDomain.waitRisingEdge()
              captures(s"after_layer${L}_q_proj") = readQBuf()
            }
            if (prevAttn == AttnSt.K_PROJ && at == AttnSt.V_PROJ) {
              dut.clockDomain.waitRisingEdge()
              captures(s"after_layer${L}_k_proj") = readKBuf()
            }
            if (prevAttn == AttnSt.V_PROJ && at == AttnSt.STORE_KV) {
              dut.clockDomain.waitRisingEdge()
              captures(s"after_layer${L}_v_proj") = readVBuf()
            }
            if (prevAttn == AttnSt.O_PROJ && at == AttnSt.DONE) {
              dut.clockDomain.waitRisingEdge()
              captures(s"after_layer${L}_attn_out") = readAttnOutBuf()
            }
          }
          prevAttn = at
        } else {
          prevAttn = -1
        }

        // === FFN sub-state transitions ===
        if (ly == LaySt.FEEDFORWARD && sq == SeqSt.LAYER_LOOP) {
          val ff = dut.transformerLayer.ffn.state.toBigInt.toInt
          val L = curLayer

          if (ff != prevFfn) {
            if (prevFfn == FfnSt.DOWN_PROJ && ff == FfnSt.DONE) {
              // resultBuf := accumBuf >> 4 assigned in DONE, latched next edge
              dut.clockDomain.waitRisingEdge()
              dut.clockDomain.waitRisingEdge()
              captures(s"after_layer${L}_ff_down") = readFfnResultBuf()
            }
          }
          prevFfn = ff
        } else {
          prevFfn = -1
        }

        prevSeq = sq
        prevLay = ly
      }
    }

    // Main thread: poll STATUS for done
    var timeout = 100000
    var done = false
    while (!done && timeout > 0) {
      dut.clockDomain.waitRisingEdge(50)
      timeout -= 50
      val status = axiRead(0x04)
      done = (status & 2) != 0
    }
    assert(done, "Position 6 inference timed out after 100K cycles")

    val tokenOut = (axiRead(0x0C) & 0x7F).toInt
    monitoring = false
    dut.clockDomain.waitRisingEdge(10) // let monitor thread finish

    val ch = if (tokenOut >= 32 && tokenOut < 127) s"'${tokenOut.toChar}'" else f"0x$tokenOut%02x"
    println(s"\n  Position 6 output: $ch (token=$tokenOut)")
    println(s"  Expected:          token=$expectedToken")

    // =========================================================
    // Phase 3: Compare captures against Python reference
    // =========================================================
    println("\n" + "=" * 70)
    println("  STAGE-BY-STAGE COMPARISON (HW vs Python Reference)")
    println("=" * 70)
    println(f"  ${"Stage"}%-35s | ${"Len"}%3s | ${"MaxErr"}%6s | ${"#Mis"}%4s | ${"Status"}%-10s")
    println("  " + "-" * 68)

    var firstMismatch: Option[String] = None

    for (key <- stageOrder) {
      val hwOpt = captures.get(key)
      val refOpt = refMap.get(key)

      (hwOpt, refOpt) match {
        case (Some(hw), Some(ref)) =>
          val pairs = hw.zip(ref)
          val maxErr = pairs.map { case (h, r) => math.abs(h - r) }.max
          val misCount = pairs.count { case (h, r) => h != r }
          val status = if (misCount == 0) "MATCH" else "MISMATCH"
          println(f"  $key%-35s | ${hw.length}%3d | $maxErr%6d | $misCount%4d | $status%-10s")

          if (misCount > 0 && firstMismatch.isEmpty) {
            firstMismatch = Some(key)
            val diffs = pairs.zipWithIndex.filter { case ((h, r), _) => h != r }
            println(s"    First mismatches (up to 8):")
            for (((h, r), idx) <- diffs.take(8)) {
              println(f"      [$idx%3d] HW=$h%4d  Ref=$r%4d  diff=${h - r}%+d")
            }
          }

        case (None, Some(_)) =>
          println(f"  $key%-35s | ${refOpt.get.length}%3s |      - |    - | NOT_CAPTURED")

        case (Some(hw), None) =>
          println(f"  $key%-35s | ${hw.length}%3d |      - |    - | NO_REF")

        case _ =>
          println(f"  $key%-35s |   - |      - |    - | MISSING")
      }
    }

    // Check for unexpected captures
    val unexpected = captures.keys.toSet -- stageOrder.toSet
    for (key <- unexpected) {
      println(f"  $key%-35s | ${captures(key).length}%3d |      - |    - | UNEXPECTED")
    }

    // =========================================================
    // Summary & assertions
    // =========================================================
    println("\n" + "=" * 70)
    println("  SUMMARY")
    println("=" * 70)
    println(s"  Stages captured: ${captures.size} / ${stageOrder.length}")

    // Assert all expected stages were captured
    val notCaptured = stageOrder.filter(k => !captures.contains(k))
    if (notCaptured.nonEmpty) {
      println(s"  NOT CAPTURED: ${notCaptured.mkString(", ")}")
    }
    assert(notCaptured.isEmpty,
      s"Expected all ${stageOrder.length} stages to be captured, missing: ${notCaptured.mkString(", ")}")

    // Assert zero mismatches across all stages
    assert(firstMismatch.isEmpty,
      s"Pipeline mismatch at stage: ${firstMismatch.getOrElse("")}")
    println(s"  All ${captures.size} captured stages MATCH the Python reference.")

    // Assert output token matches
    assert(tokenOut == expectedToken,
      s"Output token mismatch: HW=$tokenOut expected=$expectedToken")
    println(s"  Output token: HW=$tokenOut  Expected=$expectedToken  MATCH")
    println("=" * 70)

    // =========================================================
    // Phase 4: Temperature sampling smoke test
    // =========================================================
    println("\n--- Temperature sampling smoke test ---")
    val rng = new scala.util.Random(42)
    axiWrite(0x20, 366) // inv_temp=366 (T~0.7)
    for (trial <- 0 until 5) {
      val seed = rng.nextLong() & 0xFFFFFFFFL
      axiWrite(0x24, seed)
      val token = 32 + (trial * 7) % 96 // varied ASCII tokens
      val (out, cycles) = runInference(token, 6 + trial)
      assert(out >= 0 && out <= 127, s"Token out of range: $out")
      val outCh = if (out >= 32 && out < 127) s"'${out.toChar}'" else f"0x$out%02x"
      println(f"  seed=0x${seed}%08x token=$token%3d -> $outCh%6s (token=$out%3d, $cycles cycles)")
    }
    println("  All 5 sampling trials completed successfully.")

    simSuccess()
  }
}
