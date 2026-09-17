// Raw-sample validation: required fields, finite numbers, ber > 0.
#ifndef OPTICS_VALIDATE_HPP
#define OPTICS_VALIDATE_HPP

#include <string>

#include <nlohmann/json.hpp>

#include "window.hpp"

namespace optics {

// Parse one JSONL line into a Sample.
// Returns false and sets `error` when the line must be dropped.
bool parse_sample(const std::string& line, Sample& out, std::string& error);

// Validate an already-parsed JSON object.
bool validate_sample(const nlohmann::json& doc, Sample& out, std::string& error);

}  // namespace optics

#endif  // OPTICS_VALIDATE_HPP
