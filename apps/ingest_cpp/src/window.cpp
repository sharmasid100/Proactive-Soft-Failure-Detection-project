#include "window.hpp"

#include <cmath>
#include <stdexcept>

namespace optics {

const char* channel_name(Channel channel) {
  switch (channel) {
    case Channel::kOsnrDb:
      return "osnr_db";
    case Channel::kLog10Ber:
      return "log10_ber";
    case Channel::kLaserBiasMa:
      return "laser_bias_ma";
    case Channel::kEdfaPumpMa:
      return "edfa_pump_ma";
    case Channel::kRxPowerDbm:
      return "rx_power_dbm";
  }
  return "unknown";
}

double log10_ber(double ber) {
  const double clamped = ber > kBerFloor ? ber : kBerFloor;
  return std::log10(clamped);
}

double mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (const double v : values) {
    sum += v;
  }
  return sum / static_cast<double>(values.size());
}

double stddev(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  const double mu = mean(values);
  double acc = 0.0;
  for (const double v : values) {
    const double d = v - mu;
    acc += d * d;
  }
  return std::sqrt(acc / static_cast<double>(values.size()));
}

double ols_slope(const std::vector<double>& values) {
  const std::size_t n = values.size();
  if (n < 2) {
    return 0.0;
  }
  const double x_mean = static_cast<double>(n - 1) / 2.0;
  const double y_mean = mean(values);
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = static_cast<double>(i) - x_mean;
    num += dx * (values[i] - y_mean);
    den += dx * dx;
  }
  if (den == 0.0) {
    return 0.0;
  }
  return num / den;
}

Window::Window(std::size_t capacity) : capacity_(capacity == 0 ? kWindowN : capacity) {}

void Window::push(const Sample& sample) {
  samples_.push_back(sample);
  while (samples_.size() > capacity_) {
    samples_.pop_front();
  }
}

bool Window::full() const { return samples_.size() == capacity_; }

std::size_t Window::size() const { return samples_.size(); }

const Sample& Window::back() const {
  if (samples_.empty()) {
    throw std::out_of_range("window is empty");
  }
  return samples_.back();
}

std::vector<double> Window::series(Channel channel) const {
  std::vector<double> out;
  out.reserve(samples_.size());
  for (const Sample& s : samples_) {
    switch (channel) {
      case Channel::kOsnrDb:
        out.push_back(s.osnr_db);
        break;
      case Channel::kLog10Ber:
        out.push_back(log10_ber(s.ber));
        break;
      case Channel::kLaserBiasMa:
        out.push_back(s.laser_bias_ma);
        break;
      case Channel::kEdfaPumpMa:
        out.push_back(s.edfa_pump_ma);
        break;
      case Channel::kRxPowerDbm:
        out.push_back(s.rx_power_dbm);
        break;
    }
  }
  return out;
}

WindowStats Window::stats() const {
  const std::vector<double> osnr = series(Channel::kOsnrDb);
  const std::vector<double> lber = series(Channel::kLog10Ber);
  const std::vector<double> bias = series(Channel::kLaserBiasMa);

  WindowStats stats;
  stats.osnr_mean = mean(osnr);
  stats.osnr_std = stddev(osnr);
  stats.osnr_slope = ols_slope(osnr);
  stats.log10_ber_mean = mean(lber);
  stats.log10_ber_std = stddev(lber);
  stats.log10_ber_slope = ols_slope(lber);
  stats.laser_bias_mean = mean(bias);
  stats.laser_bias_slope = ols_slope(bias);
  return stats;
}

std::string window_key(const std::string& link_id, const std::string& channel_id) {
  return link_id + "|" + channel_id;
}

}  // namespace optics
