# ZyboGPT - Top-level Makefile
# Ternary LLM on Zybo Z7-10 FPGA
#
# Targets:
#   make train          - Train the model
#   make export         - Export weights to FPGA format
#   make validate       - Validate quantized model
#   make spinal         - Generate Verilog from SpinalHDL
#   make spinal-test    - Run SpinalHDL simulations
#   make vivado         - Create Vivado project
#   make vivado-synth   - Run synthesis
#   make vivado-impl    - Run implementation
#   make vivado-bit     - Generate bitstream
#   make rust           - Build Rust firmware
#   make flash          - Program Zybo board
#   make benchmark      - CPU/GPU/FPGA throughput comparison
#   make clean-train    - Clean training checkpoints, exports, data
#   make clean-hw       - Clean SpinalHDL generated files and sim artifacts
#   make clean-vivado   - Clean Vivado project, reports, and logs
#   make clean-rust     - Clean Rust build artifacts
#   make clean          - Clean everything
#   make all            - Full pipeline

.PHONY: all train train-phase1 train-phase2 export validate spinal spinal-test romeo-test romeo-test-full pipeline-debug vivado vivado-synth vivado-impl vivado-bit rust flash console benchmark clean clean-train clean-hw clean-vivado clean-rust

# Directories
PYTHON_DIR := python/train
HW_DIR := hw
RUST_DIR := rust
VIVADO_DIR := vivado
EXPORT_DIR := export
PHASE1_DIR ?= checkpoints/phase1
PHASE2_DIR ?= checkpoints/phase2
SCRIPTS_DIR := scripts

# Tools
PYTHON := venv/bin/python
SBT := cd $(HW_DIR) && sbt
VIVADO := $(HOME)/tools/Xilinx/Vivado/2024.1/bin/vivado
CARGO := cargo

DEVICE ?= auto

all: train export validate spinal vivado-bit rust

# ============================================================
# Phase 1: Training (two-phase curriculum)
# ============================================================

# Phase 1: Float pretrain (ternary weight STE but float activations/norms)
train-phase1: $(PHASE1_DIR)/best.pt

$(PHASE1_DIR)/best.pt:
	$(PYTHON) -m python.train.train --steps 50000 --lr 5e-4 --device $(DEVICE) --save-dir $(PHASE1_DIR)

# Phase 2: HW-mode fine-tune from Phase 1 checkpoint
train-phase2: $(PHASE2_DIR)/final.pt

$(PHASE2_DIR)/final.pt: $(PHASE1_DIR)/best.pt
	$(PYTHON) -m python.train.train --steps 50000 --lr 3e-4 --hw-mode \
		--device $(DEVICE) --save-dir $(PHASE2_DIR) --resume $(PHASE1_DIR)/best.pt

train: train-phase2

# ============================================================
# Phase 2: Export & Validation
# ============================================================

export: $(EXPORT_DIR)/weights_ternary.bin

$(EXPORT_DIR)/weights_ternary.bin: $(PHASE2_DIR)/best.pt
	$(PYTHON) $(SCRIPTS_DIR)/generate_weights.py $(PHASE2_DIR)/best.pt --output $(EXPORT_DIR)

validate: $(EXPORT_DIR)/weights_ternary.bin
	$(PYTHON) $(SCRIPTS_DIR)/validate_model.py $(PHASE2_DIR)/best.pt --export $(EXPORT_DIR)

# ============================================================
# Phase 3: SpinalHDL -> Verilog
# ============================================================

spinal: $(HW_DIR)/gen/ZyboGPTTop.v

$(HW_DIR)/gen/ZyboGPTTop.v: $(wildcard $(HW_DIR)/src/main/scala/zybogpt/*.scala)
	$(SBT) "runMain zybogpt.ZyboGPTVerilog"

spinal-test:
	rm -rf $(HW_DIR)/simWorkspace/ $(HW_DIR)/target/ $(HW_DIR)/project/target/
	$(SBT) clean \
		"Test/runMain zybogpt.RMSNormSim" \
		"Test/runMain zybogpt.SoftmaxSim" \
		"Test/runMain zybogpt.KvCacheSim" \
		"Test/runMain zybogpt.TDotUnitSim" \
		"Test/runMain zybogpt.Int8MacUnitSim" \
		"Test/runMain zybogpt.Int8MacArraySim" \
		"Test/runMain zybogpt.WeightDecoderSim" \
		"Test/runMain zybogpt.AxiLiteSim" \
		"Test/runMain zybogpt.ZyboGPTSim" \
		"Test/runMain zybogpt.SamplingSim" \
		"Test/runMain zybogpt.ZyboGPTRomeoSim"

romeo-test:
	$(SBT) "Test/runMain zybogpt.ZyboGPTRomeoSim"

romeo-test-full:
	$(SBT) "Test/runMain zybogpt.ZyboGPTRomeoFullSim"

pipeline-debug:
	$(SBT) "Test/runMain zybogpt.ZyboGPTPipelineDebugSim"

# ============================================================
# Phase 4: Vivado Synthesis & Implementation
# ============================================================

vivado: $(HW_DIR)/gen/ZyboGPTTop.v
	$(VIVADO) -mode batch -source $(VIVADO_DIR)/create_project.tcl -journal $(VIVADO_DIR)/create_project.jou -log $(VIVADO_DIR)/create_project.log

vivado-synth: vivado
	$(VIVADO) -mode batch -source $(VIVADO_DIR)/run_synth.tcl -journal $(VIVADO_DIR)/run_synth.jou -log $(VIVADO_DIR)/run_synth.log

vivado-impl: vivado-synth
	$(VIVADO) -mode batch -source $(VIVADO_DIR)/run_impl.tcl -journal $(VIVADO_DIR)/run_impl.jou -log $(VIVADO_DIR)/run_impl.log

vivado-bit: vivado-impl
	$(VIVADO) -mode batch -source $(VIVADO_DIR)/run_bitstream.tcl -journal $(VIVADO_DIR)/run_bitstream.jou -log $(VIVADO_DIR)/run_bitstream.log

# ============================================================
# Phase 5: Rust Firmware
# ============================================================

rust:
	cd $(RUST_DIR) && $(CARGO) build --release

# ============================================================
# Phase 6: Flash to Board
# ============================================================

BITSTREAM := $(VIVADO_DIR)/project/zybogpt.runs/impl_1/system_wrapper.bit
FIRMWARE := $(RUST_DIR)/target/armv7a-none-eabihf/release/zybogpt-fw
XSDB := $(HOME)/tools/Xilinx/Vivado/2024.1/bin/xsdb

PS7_INIT := $(VIVADO_DIR)/project/zybogpt.gen/sources_1/bd/system/ip/system_ps7_0/ps7_init.tcl

console:
	$(PYTHON) $(SCRIPTS_DIR)/cmd.py

flash: $(BITSTREAM) $(FIRMWARE)
	@echo "Programming Zybo Z7-10 via XSDB..."
	$(XSDB) $(SCRIPTS_DIR)/flash.tcl $(BITSTREAM) $(FIRMWARE) $(PS7_INIT)

benchmark:
	$(PYTHON) $(SCRIPTS_DIR)/benchmark.py

# ============================================================
# Clean
# ============================================================

clean-train:
	rm -rf checkpoints $(EXPORT_DIR) python/data test_vectors

clean-hw:
	rm -rf $(HW_DIR)/gen $(HW_DIR)/hw $(HW_DIR)/target $(HW_DIR)/project $(HW_DIR)/simWorkspace

clean-vivado:
	rm -rf $(VIVADO_DIR)/project $(VIVADO_DIR)/reports
	rm -f $(VIVADO_DIR)/*.jou $(VIVADO_DIR)/*.log $(VIVADO_DIR)/*.str

clean-rust:
	cd $(RUST_DIR) && $(CARGO) clean

clean: clean-train clean-hw clean-vivado clean-rust
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
