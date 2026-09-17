// Minimal newline-delimited-JSON TCP server and outbound client (POSIX sockets).
#ifndef OPTICS_JSONL_SERVER_HPP
#define OPTICS_JSONL_SERVER_HPP

#include <functional>
#include <string>

namespace optics {

struct Endpoint {
  std::string host = "0.0.0.0";
  int port = 0;
};

// Parse "host:port" (host may be empty -> "0.0.0.0").
bool parse_endpoint(const std::string& text, Endpoint& out, std::string& error);

// Accepts one client at a time and hands every complete line to `handler`.
class JsonlServer {
 public:
  JsonlServer(std::string host, int port);
  ~JsonlServer();

  JsonlServer(const JsonlServer&) = delete;
  JsonlServer& operator=(const JsonlServer&) = delete;

  bool start(std::string& error);
  // Blocks: accept -> read lines -> on client EOF, accept again.
  void serve_forever(const std::function<void(const std::string&)>& handler);
  void stop();

  int port() const { return port_; }

 private:
  void handle_client(int client_fd, const std::function<void(const std::string&)>& handler);

  std::string host_;
  int port_;
  int listen_fd_ = -1;
  bool running_ = false;
};

// Outbound JSONL connection to the infer service; reconnects with 1 s backoff.
class DownstreamClient {
 public:
  DownstreamClient(std::string host, int port);
  ~DownstreamClient();

  DownstreamClient(const DownstreamClient&) = delete;
  DownstreamClient& operator=(const DownstreamClient&) = delete;

  // Appends '\n' and sends. Returns false when the peer is unreachable.
  bool send_line(const std::string& line);
  void close();
  bool connected() const { return fd_ >= 0; }

 private:
  bool ensure_connected();

  std::string host_;
  int port_;
  int fd_ = -1;
  double last_attempt_s_ = 0.0;
};

// Monotonic-ish wall clock seconds.
double now_seconds();

}  // namespace optics

#endif  // OPTICS_JSONL_SERVER_HPP
