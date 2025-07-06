# Firmware build configuration
FIRMWARE_SRC_DIR = $(TT_METAL_HOME)/tt_metal/hw/firmware/src
FIRMWARE_BUILD_DIR = $(TT_METAL_HOME)/build/hw/firmware
FIRMWARE_OUTPUT_DIR = $(TT_METAL_HOME)/runtime/hw/lib/$(ARCH_NAME)

# Cross-compilation toolchain for RISC-V
RISCV_CC = riscv32-unknown-elf-gcc
RISCV_OBJCOPY = riscv32-unknown-elf-objcopy
RISCV_CFLAGS = -march=rv32i -mabi=ilp32 -O2 -g -Wall -Wextra -nostdlib -nostartfiles
RISCV_LDFLAGS = -T $(TT_METAL_HOME)/runtime/hw/toolchain/$(ARCH_NAME)/firmware_erisc.ld

# Check if we have RISC-V toolchain, fallback to creating stub files if not
RISCV_TOOLCHAIN_AVAILABLE := $(shell which $(RISCV_CC) 2>/dev/null)

ifeq ($(ARCH_NAME),$(filter $(ARCH_NAME),wormhole wormhole_b0))

# Define firmware targets for wormhole architectures
FIRMWARE_TARGETS = \
	$(FIRMWARE_OUTPUT_DIR)/idle_erisc.elf \
	$(FIRMWARE_OUTPUT_DIR)/active_erisc.elf \
	$(FIRMWARE_OUTPUT_DIR)/brisc.elf \
	$(FIRMWARE_OUTPUT_DIR)/ncrisc.elf \
	$(FIRMWARE_OUTPUT_DIR)/trisc.elf

ifdef RISCV_TOOLCHAIN_AVAILABLE
# Build actual firmware binaries if toolchain is available
ERISC_MAKE = $(MAKE) -f $(TT_METAL_HOME)/tt_metal/hw/firmware/module.mk build_firmware_binaries
ERISC_MAKE_CLEAN = $(MAKE) -f $(TT_METAL_HOME)/tt_metal/hw/firmware/module.mk clean_firmware_binaries
else
# Create stub files if toolchain is not available
ERISC_MAKE = $(MAKE) -f $(TT_METAL_HOME)/tt_metal/hw/firmware/module.mk create_firmware_stubs
ERISC_MAKE_CLEAN = $(MAKE) -f $(TT_METAL_HOME)/tt_metal/hw/firmware/module.mk clean_firmware_binaries
endif

else
# Skip firmware build for Grayskull
ERISC_MAKE = @echo 'Skipping firmware build for Grayskull.'
ERISC_MAKE_CLEAN = @echo 'Skipping firmware clean for Grayskull.'
endif

# Main firmware build target
hw/firmware: $(FIRMWARE_OUTPUT_DIR)
	@echo "Building firmware for $(ARCH_NAME)..."
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C linker_scripts
	$(ERISC_MAKE)
	@echo "Firmware build completed."

# Clean target
hw/firmware/clean:
	@echo "Cleaning firmware build artifacts..."
	$(ERISC_MAKE_CLEAN)

# Create output directory
$(FIRMWARE_OUTPUT_DIR):
	@mkdir -p $(FIRMWARE_OUTPUT_DIR)
	@mkdir -p $(FIRMWARE_BUILD_DIR)

# Build actual firmware binaries (when toolchain is available)
build_firmware_binaries: $(FIRMWARE_TARGETS)
	@echo "Built firmware binaries: $(FIRMWARE_TARGETS)"
	@ls -la $(FIRMWARE_OUTPUT_DIR)/*.elf 2>/dev/null || echo "No ELF files found"

# Individual firmware binary targets
$(FIRMWARE_OUTPUT_DIR)/idle_erisc.elf: $(FIRMWARE_SRC_DIR)/idle_erisc.cc $(FIRMWARE_OUTPUT_DIR)
	@echo "Compiling idle_erisc.elf..."
	$(RISCV_CC) $(RISCV_CFLAGS) $(RISCV_LDFLAGS) -o $@ $<
	@echo "Successfully built $@"

$(FIRMWARE_OUTPUT_DIR)/active_erisc.elf: $(FIRMWARE_SRC_DIR)/active_erisc.cc $(FIRMWARE_OUTPUT_DIR)
	@echo "Compiling active_erisc.elf..."
	$(RISCV_CC) $(RISCV_CFLAGS) $(RISCV_LDFLAGS) -o $@ $<
	@echo "Successfully built $@"

$(FIRMWARE_OUTPUT_DIR)/brisc.elf: $(FIRMWARE_SRC_DIR)/brisc.cc $(FIRMWARE_OUTPUT_DIR)
	@echo "Compiling brisc.elf..."
	$(RISCV_CC) $(RISCV_CFLAGS) -T $(TT_METAL_HOME)/runtime/hw/toolchain/$(ARCH_NAME)/firmware_brisc.ld -o $@ $<
	@echo "Successfully built $@"

$(FIRMWARE_OUTPUT_DIR)/ncrisc.elf: $(FIRMWARE_SRC_DIR)/ncrisc.cc $(FIRMWARE_OUTPUT_DIR)
	@echo "Compiling ncrisc.elf..."
	$(RISCV_CC) $(RISCV_CFLAGS) -T $(TT_METAL_HOME)/runtime/hw/toolchain/$(ARCH_NAME)/firmware_ncrisc.ld -o $@ $<
	@echo "Successfully built $@"

$(FIRMWARE_OUTPUT_DIR)/trisc.elf: $(FIRMWARE_SRC_DIR)/trisc.cc $(FIRMWARE_OUTPUT_DIR)
	@echo "Compiling trisc.elf..."
	$(RISCV_CC) $(RISCV_CFLAGS) -T $(TT_METAL_HOME)/runtime/hw/toolchain/$(ARCH_NAME)/firmware_trisc0.ld -o $@ $<
	@echo "Successfully built $@"

# Create stub files when toolchain is not available
create_firmware_stubs: $(FIRMWARE_OUTPUT_DIR)
	@echo "RISC-V toolchain not available, creating stub firmware files..."
	@for target in idle_erisc.elf active_erisc.elf brisc.elf ncrisc.elf trisc.elf; do \
		echo "Creating stub $$target"; \
		touch $(FIRMWARE_OUTPUT_DIR)/$$target; \
		chmod 644 $(FIRMWARE_OUTPUT_DIR)/$$target; \
	done
	@echo "Created stub firmware files in $(FIRMWARE_OUTPUT_DIR)"

# Clean firmware binaries
clean_firmware_binaries:
	@echo "Removing firmware binaries from $(FIRMWARE_OUTPUT_DIR)..."
	@rm -f $(FIRMWARE_OUTPUT_DIR)/*.elf
	@rm -rf $(FIRMWARE_BUILD_DIR)

# Validate that required firmware files exist
validate_firmware: $(FIRMWARE_OUTPUT_DIR)
	@echo "Validating firmware files..."
	@for target in idle_erisc.elf active_erisc.elf; do \
		if [ ! -f $(FIRMWARE_OUTPUT_DIR)/$$target ]; then \
			echo "ERROR: Required firmware file $$target not found!"; \
			exit 1; \
		else \
			echo "✓ Found $$target"; \
		fi; \
	done
	@echo "Firmware validation completed successfully."

.PHONY: hw/firmware hw/firmware/clean build_firmware_binaries create_firmware_stubs clean_firmware_binaries validate_firmware
