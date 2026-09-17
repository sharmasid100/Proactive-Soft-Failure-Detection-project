// CTest: rolling-window statistics on a synthetic ramp.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "window.hpp"

namespace {

int g_failures = 0;

void expect_near(const std::string& what, double got, double want, double tol = 1e-9) {
  if (std::fabs(got - want) > tol) {
    std::printf("FAIL %s: got %.12f want %.12f\n", what.c_str(), got, want);
    ++g_failures;
  } else {
    std::printf("ok   %s = %.6f\n", what.c_str(), got);
  }
}

void expect_true(const std::string& what, bool condition) {
  if (!condition) {
    std::printf("FAIL %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("ok   %s\n", what.c_str());
  }
}

std::vector<double> ramp(std::size_t n) {
  std::vector<double> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(static_cast<double>(i));
  }
  return out;
}

void test_free_functions() {
  const std::vector<double> values = ramp(optics::kWindowN);
  expect_near("mean(0..59)", optics::mean(values), 29.5);
  expect_near("slope(0..59)", optics::ols_slope(values), 1.0);
  // population std of 0..59 = sqrt((n^2 - 1) / 12)
  expect_near("std(0..59)", optics::stddev(values), std::sqrt((3600.0 - 1.0) / 12.0), 1e-9);

  const std::vector<double> constant(optics::kWindowN, 7.5);
  expect_near("slope(const)", optics::ols_slope(constant), 0.0);
  expect_near("std(const)", optics::stddev(constant), 0.0);

  expect_near("log10_ber(1e-6)", optics::log10_ber(1e-6), -6.0);
  expect_near("log10_ber(0) floored", optics::log10_ber(0.0), -15.0);
}

void test_window_stats() {
  optics::Window window(optics::kWindowN);
  expect_true("empty window not full", !window.full());

  for (std::size_t i = 0; i < optics::kWindowN; ++i) {
    optics::Sample sample;
    sample.ts_unix_ms = 1710000000000LL + static_cast<std::int64_t>(i) * 1000;
    sample.link_id = "L1";
    sample.channel_id = "C1";
    sample.osnr_db = static_cast<double>(i);          // ramp: slope 1, mean 29.5
    sample.ber = std::pow(10.0, -6.0);                // constant -> slope 0, mean -6
    sample.laser_bias_ma = 40.0 + 0.5 * static_cast<double>(i);  // slope 0.5
    sample.edfa_pump_ma = 180.0;
    sample.rx_power_dbm = -12.0;
    window.push(sample);
  }

  expect_true("window full at 60", window.full());
  expect_true("size == 60", window.size() == optics::kWindowN);

  const optics::WindowStats stats = window.stats();
  expect_near("osnr_mean", stats.osnr_mean, 29.5);
  expect_near("osnr_slope", stats.osnr_slope, 1.0);
  expect_near("osnr_std", stats.osnr_std, std::sqrt((3600.0 - 1.0) / 12.0), 1e-9);
  expect_near("log10_ber_mean", stats.log10_ber_mean, -6.0, 1e-9);
  expect_near("log10_ber_std", stats.log10_ber_std, 0.0, 1e-12);
  expect_near("log10_ber_slope", stats.log10_ber_slope, 0.0, 1e-12);
  expect_near("laser_bias_mean", stats.laser_bias_mean, 40.0 + 0.5 * 29.5);
  expect_near("laser_bias_slope", stats.laser_bias_slope, 0.5);

  const std::vector<double> osnr_seq = window.series(optics::Channel::kOsnrDb);
  expect_true("series length 60", osnr_seq.size() == optics::kWindowN);
  expect_near("series first", osnr_seq.front(), 0.0);
  expect_near("series last", osnr_seq.back(), 59.0);

  // 61st sample evicts the oldest; the window stays at 60 and shifts by one.
  optics::Sample extra;
  extra.ts_unix_ms = 1710000060000LL;
  extra.link_id = "L1";
  extra.channel_id = "C1";
  extra.osnr_db = 60.0;
  extra.ber = std::pow(10.0, -6.0);
  extra.laser_bias_ma = 40.0 + 0.5 * 60.0;
  window.push(extra);
  expect_true("size still 60 after eviction", window.size() == optics::kWindowN);
  expect_near("osnr_mean shifted", window.stats().osnr_mean, 30.5);
  expect_near("osnr_slope unchanged", window.stats().osnr_slope, 1.0);
}

void test_window_key_and_channels() {
  expect_true("window_key", optics::window_key("L1", "C1") == std::string("L1|C1"));
  expect_true("channel osnr", std::string(optics::channel_name(optics::Channel::kOsnrDb)) == "osnr_db");
  expect_true("channel log10_ber",
              std::string(optics::channel_name(optics::Channel::kLog10Ber)) == "log10_ber");
  expect_true("channel rx",
              std::string(optics::channel_name(optics::Channel::kRxPowerDbm)) == "rx_power_dbm");
}

}  // namespace

int main() {
  test_free_functions();
  test_window_stats();
  test_window_key_and_channels();
  if (g_failures > 0) {
    std::printf("%d assertion(s) failed\n", g_failures);
    return 1;
  }
  std::printf("all window tests passed\n");
  return 0;
}
