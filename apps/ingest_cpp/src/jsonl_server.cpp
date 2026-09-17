#include "jsonl_server.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace optics {
namespace {

constexpr std::size_t kReadBufferSize = 8192;
constexpr double kReconnectBackoffS = 1.0;

void set_nodelay(int fd) {
  int flag = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
}

}  // namespace

double now_seconds() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

bool parse_endpoint(const std::string& text, Endpoint& out, std::string& error) {
  const std::size_t colon = text.rfind(':');
  if (colon == std::string::npos) {
    error = "expected host:port, got '" + text + "'";
    return false;
  }
  const std::string host = text.substr(0, colon);
  const std::string port_text = text.substr(colon + 1);
  if (port_text.empty()) {
    error = "missing port in '" + text + "'";
    return false;
  }
  char* end = nullptr;
  const long port = std::strtol(port_text.c_str(), &end, 10);
  if (end == nullptr || *end != '\0' || port <= 0 || port > 65535) {
    error = "invalid port in '" + text + "'";
    return false;
  }
  out.host = host.empty() ? "0.0.0.0" : host;
  out.port = static_cast<int>(port);
  error.clear();
  return true;
}

// --------------------------------------------------------------------------- //
// JsonlServer
// --------------------------------------------------------------------------- //

JsonlServer::JsonlServer(std::string host, int port) : host_(std::move(host)), port_(port) {}

JsonlServer::~JsonlServer() { stop(); }

bool JsonlServer::start(std::string& error) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    error = std::string("socket(): ") + std::strerror(errno);
    return false;
  }
  int reuse = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port_));
  if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
    error = "invalid bind address '" + host_ + "'";
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    error = std::string("bind(): ") + std::strerror(errno);
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  if (::listen(listen_fd_, 8) != 0) {
    error = std::string("listen(): ") + std::strerror(errno);
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  running_ = true;
  error.clear();
  return true;
}

void JsonlServer::serve_forever(const std::function<void(const std::string&)>& handler) {
  while (running_ && listen_fd_ >= 0) {
    sockaddr_in peer{};
    socklen_t peer_len = sizeof(peer);
    const int client_fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer), &peer_len);
    if (client_fd < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (!running_) {
        break;
      }
      std::cerr << R"({"service":"optics_ingest","msg":"accept_failed","error":")"
                << std::strerror(errno) << "\"}" << std::endl;
      continue;
    }
    set_nodelay(client_fd);
    std::cerr << R"({"service":"optics_ingest","msg":"client_connected"})" << std::endl;
    handle_client(client_fd, handler);
    ::close(client_fd);
    std::cerr << R"({"service":"optics_ingest","msg":"client_disconnected"})" << std::endl;
  }
}

void JsonlServer::handle_client(int client_fd,
                               const std::function<void(const std::string&)>& handler) {
  std::string pending;
  std::vector<char> buffer(kReadBufferSize);
  while (running_) {
    const ssize_t got = ::recv(client_fd, buffer.data(), buffer.size(), 0);
    if (got == 0) {
      break;
    }
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    pending.append(buffer.data(), static_cast<std::size_t>(got));
    std::size_t start = 0;
    while (true) {
      const std::size_t nl = pending.find('\n', start);
      if (nl == std::string::npos) {
        break;
      }
      std::string line = pending.substr(start, nl - start);
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (!line.empty()) {
        handler(line);
      }
      start = nl + 1;
    }
    pending.erase(0, start);
    if (pending.size() > 1024 * 1024) {  // runaway line without newline
      pending.clear();
    }
  }
  // Flush a trailing line without newline.
  if (!pending.empty()) {
    handler(pending);
  }
}

void JsonlServer::stop() {
  running_ = false;
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
}

// --------------------------------------------------------------------------- //
// DownstreamClient
// --------------------------------------------------------------------------- //

DownstreamClient::DownstreamClient(std::string host, int port)
    : host_(std::move(host)), port_(port) {}

DownstreamClient::~DownstreamClient() { close(); }

bool DownstreamClient::ensure_connected() {
  if (fd_ >= 0) {
    return true;
  }
  const double now = now_seconds();
  if (now - last_attempt_s_ < kReconnectBackoffS) {
    return false;
  }
  last_attempt_s_ = now;

  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* result = nullptr;
  const std::string port_text = std::to_string(port_);
  const int rc = ::getaddrinfo(host_.c_str(), port_text.c_str(), &hints, &result);
  if (rc != 0 || result == nullptr) {
    std::cerr << R"({"service":"optics_ingest","msg":"downstream_resolve_failed","host":")" << host_
              << R"(","error":")" << ::gai_strerror(rc) << "\"}" << std::endl;
    return false;
  }
  int fd = -1;
  for (addrinfo* it = result; it != nullptr; it = it->ai_next) {
    fd = ::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
    if (fd < 0) {
      continue;
    }
    if (::connect(fd, it->ai_addr, it->ai_addrlen) == 0) {
      break;
    }
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(result);
  if (fd < 0) {
    std::cerr << R"({"service":"optics_ingest","msg":"downstream_connect_failed","host":")" << host_
              << R"(","port":)" << port_ << "}" << std::endl;
    return false;
  }
  set_nodelay(fd);
  fd_ = fd;
  std::cerr << R"({"service":"optics_ingest","msg":"downstream_connected","host":")" << host_
            << R"(","port":)" << port_ << "}" << std::endl;
  return true;
}

bool DownstreamClient::send_line(const std::string& line) {
  if (!ensure_connected()) {
    return false;
  }
  std::string payload = line;
  payload.push_back('\n');
  std::size_t sent = 0;
  while (sent < payload.size()) {
#ifdef MSG_NOSIGNAL
    const int flags = MSG_NOSIGNAL;
#else
    const int flags = 0;
#endif
    const ssize_t wrote = ::send(fd_, payload.data() + sent, payload.size() - sent, flags);
    if (wrote <= 0) {
      if (errno == EINTR) {
        continue;
      }
      std::cerr << R"({"service":"optics_ingest","msg":"downstream_send_failed","error":")"
                << std::strerror(errno) << "\"}" << std::endl;
      close();
      return false;
    }
    sent += static_cast<std::size_t>(wrote);
  }
  return true;
}

void DownstreamClient::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

}  // namespace optics
