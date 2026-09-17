// optics_ingest: JSONL TCP telemetry ingest, validation, rolling 60 s windows.
//
//   optics_ingest --listen 0.0.0.0:9000 --downstream 127.0.0.1:9001
//
// Reads raw telemetry samples (one JSON object per line) on --listen, validates
// them, keeps the last 60 samples per "<link_id>|<channel_id>" key, and emits a
// feature frame (JSONL) to --downstream on every accepted sample once the window
// is full. No ML happens here.

#include <csignal>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "jsonl_server.hpp"
#include "validate.hpp"
#include "window.hpp"

namespace {

constexpr double kInvalidLogIntervalS = 10.0;

optics::JsonlServer* g_server = nullptr;

void handle_signal(int /*signum*/) {
  if (g_server != nullptr) {
    g_server->stop();
  }
}

void log_json(const std::string& msg, const nlohmann::json& extra = nlohmann::json::object()) {
  nlohmann::json record;
  record["ts"] = static_cast<double>(std::time(nullptr));
  record["service"] = "optics_ingest";
  record["msg"] = msg;
  for (auto it = extra.begin(); it != extra.end(); ++it) {
    record[it.key()] = it.value();
  }
  std::cerr << record.dump() << std::endl;
}

nlohmann::json build_frame(const optics::Window& window) {
  const optics::Sample& last = window.back();
  const optics::WindowStats stats = window.stats();

  nlohmann::json seq = nlohmann::json::object();
  const optics::Channel channels[] = {
      optics::Channel::kOsnrDb,      optics::Channel::kLog10Ber,   optics::Channel::kLaserBiasMa,
      optics::Channel::kEdfaPumpMa,  optics::Channel::kRxPowerDbm,
  };
  for (const optics::Channel channel : channels) {
    seq[optics::channel_name(channel)] = window.series(channel);
  }

  nlohmann::json frame;
  frame["ts_unix_ms"] = last.ts_unix_ms;
  frame["link_id"] = last.link_id;
  frame["channel_id"] = last.channel_id;
  frame["n"] = static_cast<int>(window.size());
  frame["seq"] = seq;
  frame["stats"] = {
      {"osnr_mean", stats.osnr_mean},
      {"osnr_std", stats.osnr_std},
      {"osnr_slope", stats.osnr_slope},
      {"log10_ber_mean", stats.log10_ber_mean},
      {"log10_ber_std", stats.log10_ber_std},
      {"log10_ber_slope", stats.log10_ber_slope},
      {"laser_bias_mean", stats.laser_bias_mean},
      {"laser_bias_slope", stats.laser_bias_slope},
  };
  return frame;
}

std::string env_or(const char* key, const std::string& fallback) {
  const char* value = std::getenv(key);
  return (value == nullptr || *value == '\0') ? fallback : std::string(value);
}

struct Options {
  std::string listen = "0.0.0.0:9000";
  std::string downstream = "127.0.0.1:9001";
};

bool parse_args(int argc, char** argv, Options& options) {
  options.listen = env_or("INGEST_LISTEN", options.listen);
  options.downstream = env_or("INFER_HOST", options.downstream);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "--listen" || arg == "--downstream") && i + 1 < argc) {
      if (arg == "--listen") {
        options.listen = argv[++i];
      } else {
        options.downstream = argv[++i];
      }
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "usage: optics_ingest [--listen HOST:PORT] [--downstream HOST:PORT]\n";
      return false;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_args(argc, argv, options)) {
    return 2;
  }

  std::string error;
  optics::Endpoint listen_ep;
  optics::Endpoint downstream_ep;
  if (!optics::parse_endpoint(options.listen, listen_ep, error)) {
    log_json("bad_listen_endpoint", {{"error", error}});
    return 2;
  }
  if (!optics::parse_endpoint(options.downstream, downstream_ep, error)) {
    log_json("bad_downstream_endpoint", {{"error", error}});
    return 2;
  }

  optics::JsonlServer server(listen_ep.host, listen_ep.port);
  if (!server.start(error)) {
    log_json("listen_failed", {{"error", error}});
    return 1;
  }
  g_server = &server;
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);
  std::signal(SIGPIPE, SIG_IGN);

  optics::DownstreamClient downstream(downstream_ep.host, downstream_ep.port);
  std::unordered_map<std::string, optics::Window> windows;

  long long accepted = 0;
  long long invalid = 0;
  long long invalid_since_log = 0;
  long long frames = 0;
  double last_invalid_log_s = optics::now_seconds();

  log_json("listening", {{"listen", options.listen}, {"downstream", options.downstream}});

  server.serve_forever([&](const std::string& line) {
    optics::Sample sample;
    std::string parse_error;
    if (!optics::parse_sample(line, sample, parse_error)) {
      ++invalid;
      ++invalid_since_log;
      const double now = optics::now_seconds();
      if (now - last_invalid_log_s >= kInvalidLogIntervalS) {
        log_json("invalid_samples",
                 {{"dropped_since_last", invalid_since_log},
                  {"dropped_total", invalid},
                  {"last_error", parse_error}});
        last_invalid_log_s = now;
        invalid_since_log = 0;
      }
      return;
    }

    const std::string key = optics::window_key(sample.link_id, sample.channel_id);
    auto it = windows.find(key);
    if (it == windows.end()) {
      it = windows.emplace(key, optics::Window(optics::kWindowN)).first;
    }
    it->second.push(sample);
    ++accepted;

    if (!it->second.full()) {
      return;
    }
    const nlohmann::json frame = build_frame(it->second);
    if (downstream.send_line(frame.dump())) {
      ++frames;
      if (frames % 60 == 1) {
        log_json("frame_emitted",
                 {{"link_id", sample.link_id},
                  {"frames", frames},
                  {"accepted", accepted},
                  {"invalid", invalid}});
      }
    }
  });

  log_json("shutdown", {{"accepted", accepted}, {"invalid", invalid}, {"frames", frames}});
  return 0;
}
