#include "validate.hpp"

#include <cmath>

namespace optics {
namespace {

bool read_number(const nlohmann::json& doc, const char* key, double& out, std::string& error) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null()) {
    error = std::string("missing field: ") + key;
    return false;
  }
  if (!it->is_number()) {
    error = std::string("not a number: ") + key;
    return false;
  }
  const double value = it->get<double>();
  if (!std::isfinite(value)) {
    error = std::string("non-finite: ") + key;
    return false;
  }
  out = value;
  return true;
}

double read_optional_number(const nlohmann::json& doc, const char* key, double fallback) {
  const auto it = doc.find(key);
  if (it == doc.end() || it->is_null() || !it->is_number()) {
    return fallback;
  }
  const double value = it->get<double>();
  return std::isfinite(value) ? value : fallback;
}

bool read_string(const nlohmann::json& doc, const char* key, std::string& out, std::string& error) {
  const auto it = doc.find(key);
  if (it == doc.end() || !it->is_string()) {
    error = std::string("missing field: ") + key;
    return false;
  }
  out = it->get<std::string>();
  if (out.empty()) {
    error = std::string("empty field: ") + key;
    return false;
  }
  return true;
}

}  // namespace

bool validate_sample(const nlohmann::json& doc, Sample& out, std::string& error) {
  if (!doc.is_object()) {
    error = "not a JSON object";
    return false;
  }

  const auto ts_it = doc.find("ts_unix_ms");
  if (ts_it == doc.end() || !ts_it->is_number()) {
    error = "missing field: ts_unix_ms";
    return false;
  }
  const double ts_raw = ts_it->get<double>();
  if (!std::isfinite(ts_raw) || ts_raw < 0.0) {
    error = "invalid ts_unix_ms";
    return false;
  }
  out.ts_unix_ms = static_cast<std::int64_t>(ts_raw);

  if (!read_string(doc, "link_id", out.link_id, error)) {
    return false;
  }
  const auto ch_it = doc.find("channel_id");
  if (ch_it != doc.end() && ch_it->is_string() && !ch_it->get<std::string>().empty()) {
    out.channel_id = ch_it->get<std::string>();
  } else {
    out.channel_id = "C1";
  }

  if (!read_number(doc, "osnr_db", out.osnr_db, error)) {
    return false;
  }
  if (!read_number(doc, "ber", out.ber, error)) {
    return false;
  }
  if (out.ber <= 0.0) {
    error = "ber must be > 0";
    return false;
  }
  if (!read_number(doc, "laser_bias_ma", out.laser_bias_ma, error)) {
    return false;
  }

  out.edfa_pump_ma = read_optional_number(doc, "edfa_pump_ma", kEdfaPumpDefault);
  out.rx_power_dbm = read_optional_number(doc, "rx_power_dbm", kRxPowerDefault);
  error.clear();
  return true;
}

bool parse_sample(const std::string& line, Sample& out, std::string& error) {
  nlohmann::json doc = nlohmann::json::parse(line, nullptr, false);
  if (doc.is_discarded()) {
    error = "malformed JSON";
    return false;
  }
  return validate_sample(doc, out, error);
}

}  // namespace optics
