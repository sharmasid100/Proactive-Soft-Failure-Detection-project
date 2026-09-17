// Rolling telemetry window: 60 samples @ 1 Hz per (link_id, channel_id) key.
#ifndef OPTICS_WINDOW_HPP
#define OPTICS_WINDOW_HPP

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace optics {

constexpr std::size_t kWindowN = 60;
constexpr double kBerFloor = 1e-15;
constexpr double kEdfaPumpDefault = 0.0;
constexpr double kRxPowerDefault = -99.0;

// One validated raw telemetry sample.
struct Sample {
  std::int64_t ts_unix_ms = 0;
  std::string link_id;
  std::string channel_id;
  double osnr_db = 0.0;
  double ber = 0.0;
  double laser_bias_ma = 0.0;
  double edfa_pump_ma = kEdfaPumpDefault;
  double rx_power_dbm = kRxPowerDefault;
};

// The 8 Isolation Forest features, in the fixed contract order.
struct WindowStats {
  double osnr_mean = 0.0;
  double osnr_std = 0.0;
  double osnr_slope = 0.0;
  double log10_ber_mean = 0.0;
  double log10_ber_std = 0.0;
  double log10_ber_slope = 0.0;
  double laser_bias_mean = 0.0;
  double laser_bias_slope = 0.0;
};

// The 5 autoencoder channels, in the fixed contract order.
enum class Channel {
  kOsnrDb = 0,
  kLog10Ber = 1,
  kLaserBiasMa = 2,
  kEdfaPumpMa = 3,
  kRxPowerDbm = 4,
};

const char* channel_name(Channel channel);

// log10(max(ber, 1e-15)).
double log10_ber(double ber);

double mean(const std::vector<double>& values);
double stddev(const std::vector<double>& values);  // population std, matches numpy default
double ols_slope(const std::vector<double>& values);  // vs index 0..n-1

// Fixed-capacity rolling window; oldest sample is dropped once full.
class Window {
 public:
  explicit Window(std::size_t capacity = kWindowN);

  void push(const Sample& sample);
  bool full() const;
  std::size_t size() const;
  std::size_t capacity() const { return capacity_; }
  const Sample& back() const;

  std::vector<double> series(Channel channel) const;
  WindowStats stats() const;

 private:
  std::size_t capacity_;
  std::deque<Sample> samples_;
};

// key used to bucket samples: "<link_id>|<channel_id>".
std::string window_key(const std::string& link_id, const std::string& channel_id);

}  // namespace optics

#endif  // OPTICS_WINDOW_HPP
