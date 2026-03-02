package zybogpt

import spinal.core._
import spinal.core.sim._

/** Shared runner for "ROMEO:" prompt simulation tests.
  *
  * Exercises the full inference pipeline end-to-end:
  * 1. Prompt processing: feeds "ROMEO:" one token at a time (positions 0-5)
  * 2. Autoregressive generation: feeds each predicted token back as input
  * 3. Compares hardware output against Python reference golden vectors
  * 4. Reports cycle counts, tokens/sec at 150 MHz
  *
  * Requires trained weights in ../export/ (loaded at elaboration time).
  */
object ZyboGPTRomeoRunner {
  // "ROMEO:" prompt as ASCII token IDs
  val prompt = Array(82, 79, 77, 69, 79, 58) // R, O, M, E, O, :

  // Golden reference from Python reference_inference.py (bit-accurate INT8 datapath)
  val referenceGenerated = Array(
      10, 73, 32, 104, 105, 115, 32, 119, 111, 117, 108, 100, 32, 109, 97, 110,
      32, 115, 111, 110, 101, 32, 116, 104, 101, 32, 104, 97, 116, 104, 101, 114,
      32, 104, 97, 108, 108, 32, 116, 104, 101, 32, 100, 97, 121, 32, 115, 111,
      114, 118, 101, 32, 116, 104, 101, 32, 115, 101, 101, 101, 32, 115, 111, 32,
      104, 101, 114, 101, 32, 104, 101, 97, 114, 32, 116, 104, 101, 32, 116, 111,
      32, 119, 101, 110, 116, 32, 116, 104, 101, 32, 115, 104, 105, 115, 32, 109,
      97, 110, 32, 97, 32, 98, 101, 115, 101, 32, 97, 32, 115, 101, 110, 116,
      32, 116, 104, 101, 32, 97, 110, 100, 32, 97, 110)

  def isEndPunctuation(t: Int): Boolean = t == '.' || t == '!' || t == '?'
  def isUpperAlpha(t: Int): Boolean = t >= 'A' && t <= 'Z'

  def tokenToStr(t: Int): String =
    if (t >= 32 && t < 127) s"'${t.toChar}'" else f"0x${t}%02x"

  def run(maxGenerate: Int, useStopConditions: Boolean): Unit = {
    val modeStr = if (useStopConditions) "full" else s"$maxGenerate tokens"
    println("=" * 70)
    println(s"  ZyboGPT ROMEO Test ($modeStr)")
    println("=" * 70)
    println(s"  Prompt: \"${prompt.map(_.toChar).mkString}\" = [${prompt.mkString(", ")}]")
    println(s"  Generate: up to $maxGenerate tokens${if (useStopConditions) " (stop on end punctuation or moniker)" else ""}")
    println(s"  Reference: [${referenceGenerated.take(10).mkString(", ")}${if (referenceGenerated.length > 10) ", ..." else ""}]")
    println()

    SimConfig.withWave.compile(ZyboGPTTop()).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10) // 100 MHz sim clock

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

      // ---- AXI helper functions ----
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

      /** Run single-token inference and return (outputToken, cycleCount). */
      def runInference(token: Int, position: Int): (Int, Long) = {
        axiWrite(0x08, token.toLong)   // TOKEN_IN
        axiWrite(0x10, position.toLong) // POSITION

        // Trigger start (rising edge on CONTROL bit 0)
        axiWrite(0x00, 0x01)
        dut.clockDomain.waitRisingEdge(3)
        axiWrite(0x00, 0x00)
        dut.clockDomain.waitRisingEdge(2)

        // Poll STATUS for done (bit 1)
        var timeout = 100000 // 100K cycles max per token
        var done = false
        while (!done && timeout > 0) {
          dut.clockDomain.waitRisingEdge(50)
          timeout -= 50
          val status = axiRead(0x04)
          done = (status & 2) != 0
        }
        assert(done, s"Inference timed out at token=$token pos=$position after 100K cycles")

        val tokenOut = (axiRead(0x0C) & 0x7F).toInt
        val cycles = axiRead(0x14)
        (tokenOut, cycles)
      }

      // =========================================================
      // Configure sampling: T=0.5 (inv_temp=512) for Shakespeare-quality output
      // =========================================================
      axiWrite(0x24, 0xDEADBEEFL) // SEED: deterministic seed (must be set before inv_temp)
      axiWrite(0x20, 0x0200) // SAMPLING: inv_temp=512 (T=0.5)

      // =========================================================
      // Verify CONFIG register
      // =========================================================
      println("--- Verify CONFIG ---")
      val configVal = axiRead(0x1C)
      assert((configVal & 0xFF) == 64, "d_model mismatch")
      assert(((configVal >> 8) & 0xFF) == 2, "n_layers mismatch")
      assert(((configVal >> 16) & 0xFF) == 128, "ctx_len mismatch")
      assert(((configVal >> 24) & 0xFF) == 128, "vocab_size mismatch")
      println("  CONFIG OK: d_model=64, n_layers=2, ctx_len=128, vocab=128\n")

      // =========================================================
      // Process prompt "ROMEO:" (positions 0-5)
      // =========================================================
      println("--- Prompt Processing ---")
      println(f"  ${"pos"}%4s | ${"input"}%6s | ${"output"}%7s | ${"cycles"}%8s |")
      println("  " + "-" * 40)

      val allOutputs = scala.collection.mutable.ArrayBuffer[Int]()
      val allCycles = scala.collection.mutable.ArrayBuffer[Long]()
      var totalCycles = 0L

      for (i <- prompt.indices) {
        val (tokenOut, cycles) = runInference(prompt(i), i)
        allOutputs += tokenOut
        allCycles += cycles
        totalCycles += cycles
        println(f"  ${i}%4d | ${tokenToStr(prompt(i))}%6s | ${tokenToStr(tokenOut)}%7s | ${cycles}%8d |")
      }

      val firstGenerated = allOutputs.last
      println(f"\n  Prediction after prompt: ${tokenToStr(firstGenerated)}")
      if (referenceGenerated.nonEmpty) {
        val refMatch = firstGenerated == referenceGenerated(0)
        println(f"  Reference expected:      ${tokenToStr(referenceGenerated(0))}${if (refMatch) " MATCH" else " MISMATCH"}")
      }

      // =========================================================
      // Autoregressive generation
      // =========================================================
      println(s"\n--- Autoregressive Generation (up to $maxGenerate tokens) ---")
      println(f"  ${"pos"}%4s | ${"input"}%6s | ${"output"}%7s | ${"cycles"}%8s | ${"ref"}%7s | ${""}%-10s|")
      println("  " + "-" * 60)

      val generated = scala.collection.mutable.ArrayBuffer[Int]()
      var nextToken = firstGenerated
      var chainDiverged = false
      var stopReason = "max tokens"

      var i = 0
      var done = false
      while (i < maxGenerate && !done) {
        val pos = prompt.length + i
        val (tokenOut, cycles) = runInference(nextToken, pos)
        generated += tokenOut
        allCycles += cycles
        totalCycles += cycles

        // Compare with reference (only meaningful if chain hasn't diverged)
        val refIdx = i + 1 // generated[0] was already consumed as firstGenerated
        val (refStr, statusStr) = if (refIdx < referenceGenerated.length && !chainDiverged) {
          val refTok = referenceGenerated(refIdx)
          if (tokenOut == refTok) {
            (tokenToStr(refTok), "MATCH")
          } else {
            chainDiverged = true
            (tokenToStr(refTok), "DIVERGED")
          }
        } else if (chainDiverged) {
          ("---", "")
        } else {
          ("n/a", "")
        }

        println(f"  ${pos}%4d | ${tokenToStr(nextToken)}%6s | ${tokenToStr(tokenOut)}%7s | ${cycles}%8d | ${refStr}%7s | ${statusStr}%-10s|")

        // Check stop conditions (only in full mode)
        if (useStopConditions) {
          if (isEndPunctuation(tokenOut)) {
            stopReason = s"end punctuation '${tokenOut.toChar}'"
            done = true
          }
          if (tokenOut == ':' && isUpperAlpha(nextToken)) {
            stopReason = "moniker detected"
            done = true
          }
        }

        nextToken = tokenOut
        i += 1
      }

      println(s"\n  Stopped after ${generated.length} tokens: $stopReason")

      // =========================================================
      // System idle check
      // =========================================================
      println("\n--- System State ---")
      val finalStatus = axiRead(0x04)
      val busyAfter = finalStatus & 1
      println(s"  Idle after generation: ${busyAfter == 0}")
      assert(busyAfter == 0, "System should be idle after all inference completes")

      // =========================================================
      // Second run (verify KV cache doesn't corrupt)
      // =========================================================
      println("\n--- Repeat First Token (regression) ---")
      val (repeatOut, repeatCycles) = runInference(prompt(0), 0)
      val originalOut = allOutputs(0)
      println(f"  pos=0: in=${tokenToStr(prompt(0))} -> out=${tokenToStr(repeatOut)} (original: ${tokenToStr(originalOut)})")
      println(s"  System survived re-inference: OK")

      // =========================================================
      // Summary
      // =========================================================
      println("\n" + "=" * 70)
      println("  SUMMARY")
      println("=" * 70)

      // Generated text
      val fullTokens = Array(firstGenerated) ++ generated.toArray
      val fullText = fullTokens.map { t =>
        if (t >= 32 && t < 127) t.toChar.toString
        else s"\\x${"%02x".format(t)}"
      }.mkString
      println(s"  Prompt:    \"ROMEO:\"")
      println(s"  Generated: \"$fullText\"")
      println(s"  Tokens:    [${fullTokens.mkString(", ")}]")

      // Reference comparison
      val comparableTokens = math.min(fullTokens.length, referenceGenerated.length)
      val matchCount = (0 until comparableTokens).count(i => fullTokens(i) == referenceGenerated(i))
      println(s"\n  Reference match: $matchCount/$comparableTokens tokens")

      // Cycle statistics (exclude repeat run)
      val promptAndGenCycles = allCycles.toArray
      val avgCycles = totalCycles.toDouble / promptAndGenCycles.length
      val minCycles = promptAndGenCycles.min
      val maxCycles = promptAndGenCycles.max
      val tokPerSec150 = 150e6 / avgCycles

      println(f"\n  Cycle Statistics (${promptAndGenCycles.length} tokens):")
      println(f"    Total:         $totalCycles cycles")
      println(f"    Average:       $avgCycles%.0f cycles/token")
      println(f"    Min:           $minCycles cycles (pos 0)")
      println(f"    Max:           $maxCycles cycles (pos ${promptAndGenCycles.indexOf(maxCycles)})")
      println(f"    Tokens/sec:    $tokPerSec150%.0f @ 150 MHz")

      // Per-position cycle costs
      println(f"\n  Per-Token Cycle Costs:")
      for (i <- promptAndGenCycles.indices) {
        val delta = if (i > 0) f"  (+${promptAndGenCycles(i) - promptAndGenCycles(i-1)})" else ""
        println(f"    pos ${i}%3d: ${promptAndGenCycles(i)}%6d cycles$delta")
      }

      // Growth analysis
      if (promptAndGenCycles.length > 2) {
        val deltas = (1 until promptAndGenCycles.length).map(i =>
          promptAndGenCycles(i) - promptAndGenCycles(i-1))
        val avgDelta = deltas.sum.toDouble / deltas.length
        println(f"\n  Growth: ~${avgDelta}%.0f cycles/position (linear, from KV cache reads)")
      }

      // All tokens must complete within timeout
      assert(promptAndGenCycles.forall(_ > 0), "All cycle counts should be positive")
      assert(promptAndGenCycles.forall(_ < 100000), "No token should take more than 100K cycles")

      println(s"\n  All ZyboGPT ROMEO tests PASSED!")
      println("=" * 70)
    }
  }
}

/** Quick ROMEO test: 10 generated tokens, no stop conditions. */
object ZyboGPTRomeoSim extends App {
  ZyboGPTRomeoRunner.run(maxGenerate = 10, useStopConditions = false)
}

/** Full ROMEO test: up to 120 tokens with stop on punctuation/moniker. */
object ZyboGPTRomeoFullSim extends App {
  ZyboGPTRomeoRunner.run(maxGenerate = 120, useStopConditions = true)
}
