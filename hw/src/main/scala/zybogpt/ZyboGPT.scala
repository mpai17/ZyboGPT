package zybogpt

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axilite._

/** Top-level ZyboGPT accelerator module.
  *
  * Integrates:
  * - AXI-Lite slave for PS communication
  * - Sequencer FSM
  * - Embedding lookup
  * - Transformer layers (time-multiplexed)
  * - TDot array (shared ternary compute) with weight loading controller
  * - INT8 MAC array (shared attention compute)
  * - KV cache
  * - Weight BRAM with multi-cycle loading
  * - RMSNorm (final)
  * - Norm gamma weight storage (single BRAM for all gammas)
  */
case class ZyboGPTTop(config: ZyboGPTHwConfig = ZyboGPTHwConfig.default) extends Component {
  val io = new Bundle {
    val axi = slave(AxiLite4(AxiLite4Config(addressWidth = 8, dataWidth = 32)))
  }

  // Sub-modules (TDotArray removed — WeightBram now contains single TDotUnit)
  val axiSlave = AxiLiteSlave(config)
  val sequencer = Sequencer(config)
  val embedding = Embedding(config)
  val transformerLayer = TransformerLayer(config)
  val macArray = Int8MacArray(config)
  val kvCache = KvCache(config)
  val weightBram = WeightBram(config)
  val finalNorm = RMSNorm(config)

  // ================================================================
  // AXI connection
  // ================================================================
  axiSlave.io.axi <> io.axi

  // AXI slave <-> Sequencer
  sequencer.io.tokenIn := axiSlave.io.tokenIn
  sequencer.io.positionIn := axiSlave.io.position
  sequencer.io.start := axiSlave.io.start
  sequencer.io.invTemp := axiSlave.io.invTemp
  sequencer.io.seed := axiSlave.io.seed
  sequencer.io.seedWrite := axiSlave.io.seedWrite
  axiSlave.io.tokenOut := sequencer.io.tokenOut
  axiSlave.io.busy := sequencer.io.busy
  axiSlave.io.done := sequencer.io.done
  axiSlave.io.cycleCount := sequencer.io.cycleCount

  // ================================================================
  // Sequencer <-> Embedding
  // ================================================================
  embedding.io.tokenId := sequencer.io.embTokenId
  embedding.io.position := sequencer.io.embPosition
  embedding.io.start := sequencer.io.embStart
  embedding.io.embAddr := sequencer.io.embAddr
  sequencer.io.embData := embedding.io.embData
  sequencer.io.embDone := embedding.io.done
  embedding.io.logitMode := sequencer.io.embLogitMode
  embedding.io.queryVec := sequencer.io.embLogitVec
  embedding.io.logitTokenId := sequencer.io.embLogitTokenId
  sequencer.io.embLogitResult := embedding.io.logitResult

  // ================================================================
  // Sequencer <-> Transformer Layer
  // ================================================================
  transformerLayer.io.x := sequencer.io.layerX
  transformerLayer.io.position := sequencer.io.layerPos
  transformerLayer.io.layerIdx := sequencer.io.layerIdx
  transformerLayer.io.start := sequencer.io.layerStart
  sequencer.io.layerResult := transformerLayer.io.result
  sequencer.io.layerDone := transformerLayer.io.done

  // ================================================================
  // TDot Weight Loading + Serial Compute Controller
  // WeightBram now loads weights AND computes TDot results serially.
  // IDLE → LOADING (103 reads) → COMPUTE (32 rows serial) → DONE
  // ================================================================
  object TDotCtrl extends SpinalEnum {
    val IDLE, LOADING, COMPUTE, DONE = newElement()
  }
  val tdotCtrl = RegInit(TDotCtrl.IDLE)

  // Default connections
  weightBram.io.loadAddr := transformerLayer.io.tdotWeightAddr
  weightBram.io.loadStart := False
  weightBram.io.computeStart := False
  weightBram.io.tdotX := transformerLayer.io.tdotX
  weightBram.io.scaleAddr := 0

  transformerLayer.io.tdotResult := weightBram.io.results
  transformerLayer.io.tdotDone := False

  switch(tdotCtrl) {
    is(TDotCtrl.IDLE) {
      when(transformerLayer.io.tdotStart) {
        weightBram.io.loadStart := True
        tdotCtrl := TDotCtrl.LOADING
      }
    }
    is(TDotCtrl.LOADING) {
      when(weightBram.io.loadDone) {
        weightBram.io.computeStart := True
        tdotCtrl := TDotCtrl.COMPUTE
      }
    }
    is(TDotCtrl.COMPUTE) {
      when(weightBram.io.computeDone) {
        tdotCtrl := TDotCtrl.DONE
      }
    }
    is(TDotCtrl.DONE) {
      transformerLayer.io.tdotDone := True
      tdotCtrl := TDotCtrl.IDLE
    }
  }

  // ================================================================
  // Transformer Layer <-> MAC Array
  // ================================================================
  macArray.io.a := transformerLayer.io.macA
  macArray.io.b := transformerLayer.io.macB
  macArray.io.valid := transformerLayer.io.macValid
  macArray.io.clear := transformerLayer.io.macClear
  macArray.io.start := transformerLayer.io.macStart
  macArray.io.innerDim := config.headDim / config.numDspMacs + 1 // 32/8 + 1 = 5 steps (extra for AREG pipeline)
  transformerLayer.io.macResults := macArray.io.results
  transformerLayer.io.macDone := macArray.io.done

  // ================================================================
  // Transformer Layer <-> KV Cache
  // ================================================================
  kvCache.io.writeK := transformerLayer.io.kvWriteK
  kvCache.io.writeV := transformerLayer.io.kvWriteV
  kvCache.io.writeLayer := transformerLayer.io.layerIdx
  kvCache.io.writeHead := transformerLayer.io.kvWriteHead
  kvCache.io.writePos := sequencer.io.positionIn.resize(log2Up(config.ctxLen) bits)
  kvCache.io.writeEn := transformerLayer.io.kvWriteEn
  kvCache.io.readPos := transformerLayer.io.kvReadPos
  kvCache.io.readLayer := transformerLayer.io.layerIdx
  kvCache.io.readHead := transformerLayer.io.kvReadHead
  kvCache.io.readEn := transformerLayer.io.kvReadEn
  transformerLayer.io.kvReadK := kvCache.io.readK
  transformerLayer.io.kvReadV := kvCache.io.readV
  transformerLayer.io.kvReadValid := kvCache.io.readValid
  kvCache.io.seqLen := sequencer.io.positionIn.resize(8 bits) + 1

  // ================================================================
  // RMSNorm gamma weights - single BRAM for all gammas
  // ================================================================
  // Layout: (nLayers*2 + 1) * dModel entries
  // Indices 0..dModel-1: layer 0 attn norm
  // Indices dModel..2*dModel-1: layer 0 ff norm
  // Indices 2*dModel..3*dModel-1: layer 1 attn norm
  // Indices 3*dModel..4*dModel-1: layer 1 ff norm
  // Indices 4*dModel..5*dModel-1: final norm
  val numNormSets = config.nLayers * 2 + 1 // attn + ff per layer + final norm
  val totalGammaEntries = numNormSets * config.dModel
  val normGammaMem = Mem(SInt(16 bits), wordCount = totalGammaEntries)

  // Initialize norm gammas from WeightInit
  if (WeightInit.normGammas.nonEmpty) {
    val gammaInit = WeightInit.normGammas.take(totalGammaEntries)
      .map(v => S(v, 16 bits)).toSeq
    normGammaMem.init(gammaInit.padTo(totalGammaEntries, S(1024, 16 bits)))
  } else {
    normGammaMem.init(Seq.fill(totalGammaEntries)(S(1024, 16 bits)))
  }

  // Gamma read address computation:
  // TransformerLayer outputs full gammaAddr (base + local offset).
  // Final norm uses base = nLayers * 2 * dModel.
  val gammaAddrBits = log2Up(totalGammaEntries)

  // Mux between transformer layer gamma reads and final norm gamma reads
  val finalNormActive = sequencer.io.normStart || sequencer.io.normActive
  val gammaReadAddr = UInt(gammaAddrBits bits)

  when(finalNormActive) {
    // Final norm: base = nLayers * 2 * dModel
    gammaReadAddr := (U(config.nLayers * 2 * config.dModel, gammaAddrBits bits) +
      finalNorm.io.gammaAddr.resize(gammaAddrBits bits)).resize(gammaAddrBits bits)
  } otherwise {
    // Transformer layer norms (full address already computed by TransformerLayer)
    gammaReadAddr := transformerLayer.io.gammaAddr.resize(gammaAddrBits bits)
  }

  val gammaReadData = normGammaMem.readSync(gammaReadAddr)

  // Route gamma data to both consumers
  transformerLayer.io.gammaData := gammaReadData
  finalNorm.io.gammaData := gammaReadData

  // ================================================================
  // Sequencer <-> Final RMSNorm
  // ================================================================
  finalNorm.io.x := sequencer.io.normX
  finalNorm.io.start := sequencer.io.normStart
  sequencer.io.normResult := finalNorm.io.y
  sequencer.io.normDone := finalNorm.io.done
}

/** Verilog generation entry point. */
object ZyboGPTVerilog extends App {
  SpinalConfig(
    targetDirectory = "gen",
    defaultConfigForClockDomains = ClockDomainConfig(
      resetKind = SYNC,
      resetActiveLevel = LOW
    )
  ).generateVerilog(ZyboGPTTop())

  // Generate AXI-Lite wrapper with standard signal naming for Vivado block design
  val wrapper = """// Auto-generated AXI-Lite wrapper for Vivado block design integration.
// Translates standard AXI-Lite signal names to SpinalHDL io_axi_* naming
// so Vivado can auto-detect the AXI interface.
//
// Regenerated by: make spinal

module ZyboGPTWrapper (
    input  wire          s_axi_awvalid,
    output wire          s_axi_awready,
    input  wire [7:0]    s_axi_awaddr,
    input  wire [2:0]    s_axi_awprot,
    input  wire          s_axi_wvalid,
    output wire          s_axi_wready,
    input  wire [31:0]   s_axi_wdata,
    input  wire [3:0]    s_axi_wstrb,
    output wire          s_axi_bvalid,
    input  wire          s_axi_bready,
    output wire [1:0]    s_axi_bresp,
    input  wire          s_axi_arvalid,
    output wire          s_axi_arready,
    input  wire [7:0]    s_axi_araddr,
    input  wire [2:0]    s_axi_arprot,
    output wire          s_axi_rvalid,
    input  wire          s_axi_rready,
    output wire [31:0]   s_axi_rdata,
    output wire [1:0]    s_axi_rresp,
    input  wire          aclk,
    input  wire          aresetn
);

    ZyboGPTTop u_core (
        .clk                    (aclk),
        .resetn                 (aresetn),
        .io_axi_aw_valid        (s_axi_awvalid),
        .io_axi_aw_ready        (s_axi_awready),
        .io_axi_aw_payload_addr (s_axi_awaddr),
        .io_axi_aw_payload_prot (s_axi_awprot),
        .io_axi_w_valid         (s_axi_wvalid),
        .io_axi_w_ready         (s_axi_wready),
        .io_axi_w_payload_data  (s_axi_wdata),
        .io_axi_w_payload_strb  (s_axi_wstrb),
        .io_axi_b_valid         (s_axi_bvalid),
        .io_axi_b_ready         (s_axi_bready),
        .io_axi_b_payload_resp  (s_axi_bresp),
        .io_axi_ar_valid        (s_axi_arvalid),
        .io_axi_ar_ready        (s_axi_arready),
        .io_axi_ar_payload_addr (s_axi_araddr),
        .io_axi_ar_payload_prot (s_axi_arprot),
        .io_axi_r_valid         (s_axi_rvalid),
        .io_axi_r_ready         (s_axi_rready),
        .io_axi_r_payload_data  (s_axi_rdata),
        .io_axi_r_payload_resp  (s_axi_rresp)
    );

endmodule
"""
  val wrapperFile = new java.io.File("gen/ZyboGPTWrapper.v")
  val writer = new java.io.PrintWriter(wrapperFile)
  writer.write(wrapper)
  writer.close()
  println(s"[Done] Wrapper written to ${wrapperFile.getPath}")
}
