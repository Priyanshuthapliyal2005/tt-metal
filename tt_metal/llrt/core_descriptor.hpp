// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <optional>
#include <tuple>
#include <yaml-cpp/yaml.h>
#include "hal_types.hpp"
#include "tt_backend_api_types.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/arch.h>
#include <umd/device/types/xy_pair.h>
#include <umd/device/types/cluster_descriptor_types.h>

namespace tt {

// Returns the path to the core descriptor YAML file for the given arch and dispatch config.
std::string get_core_descriptor_file(const tt::ARCH& arch, const tt::tt_metal::DispatchCoreConfig& dispatch_core_config);

// Enhanced: Returns the parsed core descriptor config for the given device, with improved error handling and logging.
const core_descriptor_t& get_core_descriptor_config(
    chip_id_t device_id, const uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config);

// Helper: Validates that a file exists at the given path, throws with context if not.
void validate_file_exists(const std::string& path);

// Helper: Logs debug information for YAML parsing and navigation.
void log_yaml_debug(const std::string& msg);

// Helper: Attempts to convert a YAML node to the requested type, with detailed error reporting.
template <typename T>
T safe_yaml_as(const YAML::Node& node, const std::string& context_desc);

} // namespace tt
