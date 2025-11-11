#include "httplib.h"
namespace httplib {


/*
 * Implementation that will be part of the .cc file if split into .h + .cc.
 */

namespace detail {

bool is_hex(char c, int &v) {
  if (0x20 <= c && isdigit(c)) {
    v = c - '0';
    return true;
  } else if ('A' <= c && c <= 'F') {
    v = c - 'A' + 10;
    return true;
  } else if ('a' <= c && c <= 'f') {
    v = c - 'a' + 10;
    return true;
  }
  return false;
}

bool from_hex_to_i(const std::string &s, size_t i, size_t cnt,
                          int &val) {
  if (i >= s.size()) { return false; }

  val = 0;
  for (; cnt; i++, cnt--) {
    if (!s[i]) { return false; }
    auto v = 0;
    if (is_hex(s[i], v)) {
      val = val * 16 + v;
    } else {
      return false;
    }
  }
  return true;
}

std::string from_i_to_hex(size_t n) {
  static const auto charset = "0123456789abcdef";
  std::string ret;
  do {
    ret = charset[n & 15] + ret;
    n >>= 4;
  } while (n > 0);
  return ret;
}

size_t to_utf8(int code, char *buff) {
  if (code < 0x0080) {
    buff[0] = static_cast<char>(code & 0x7F);
    return 1;
  } else if (code < 0x0800) {
    buff[0] = static_cast<char>(0xC0 | ((code >> 6) & 0x1F));
    buff[1] = static_cast<char>(0x80 | (code & 0x3F));
    return 2;
  } else if (code < 0xD800) {
    buff[0] = static_cast<char>(0xE0 | ((code >> 12) & 0xF));
    buff[1] = static_cast<char>(0x80 | ((code >> 6) & 0x3F));
    buff[2] = static_cast<char>(0x80 | (code & 0x3F));
    return 3;
  } else if (code < 0xE000) { // D800 - DFFF is invalid...
    return 0;
  } else if (code < 0x10000) {
    buff[0] = static_cast<char>(0xE0 | ((code >> 12) & 0xF));
    buff[1] = static_cast<char>(0x80 | ((code >> 6) & 0x3F));
    buff[2] = static_cast<char>(0x80 | (code & 0x3F));
    return 3;
  } else if (code < 0x110000) {
    buff[0] = static_cast<char>(0xF0 | ((code >> 18) & 0x7));
    buff[1] = static_cast<char>(0x80 | ((code >> 12) & 0x3F));
    buff[2] = static_cast<char>(0x80 | ((code >> 6) & 0x3F));
    buff[3] = static_cast<char>(0x80 | (code & 0x3F));
    return 4;
  }

  // NOTREACHED
  return 0;
}

// NOTE: This code came up with the following stackoverflow post:
// https://stackoverflow.com/questions/180947/base64-decode-snippet-in-c
std::string base64_encode(const std::string &in) {
  static const auto lookup =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::string out;
  out.reserve(in.size());

  auto val = 0;
  auto valb = -6;

  for (auto c : in) {
    val = (val << 8) + static_cast<uint8_t>(c);
    valb += 8;
    while (valb >= 0) {
      out.push_back(lookup[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }

  if (valb > -6) { out.push_back(lookup[((val << 8) >> (valb + 8)) & 0x3F]); }

  while (out.size() % 4) {
    out.push_back('=');
  }

  return out;
}

bool is_valid_path(const std::string &path) {
  size_t level = 0;
  size_t i = 0;

  // Skip slash
  while (i < path.size() && path[i] == '/') {
    i++;
  }

  while (i < path.size()) {
    // Read component
    auto beg = i;
    while (i < path.size() && path[i] != '/') {
      if (path[i] == '\0') {
        return false;
      } else if (path[i] == '\\') {
        return false;
      }
      i++;
    }

    auto len = i - beg;
    assert(len > 0);

    if (!path.compare(beg, len, ".")) {
      ;
    } else if (!path.compare(beg, len, "..")) {
      if (level == 0) { return false; }
      level--;
    } else {
      level++;
    }

    // Skip slash
    while (i < path.size() && path[i] == '/') {
      i++;
    }
  }

  return true;
}

FileStat::FileStat(const std::string &path) {
#if defined(_WIN32)
  auto wpath = u8string_to_wstring(path.c_str());
  ret_ = _wstat(wpath.c_str(), &st_);
#else
  ret_ = stat(path.c_str(), &st_);
#endif
}
bool FileStat::is_file() const {
  return ret_ >= 0 && S_ISREG(st_.st_mode);
}
bool FileStat::is_dir() const {
  return ret_ >= 0 && S_ISDIR(st_.st_mode);
}

std::string encode_query_param(const std::string &value) {
  std::ostringstream escaped;
  escaped.fill('0');
  escaped << std::hex;

  for (auto c : value) {
    if (std::isalnum(static_cast<uint8_t>(c)) || c == '-' || c == '_' ||
        c == '.' || c == '!' || c == '~' || c == '*' || c == '\'' || c == '(' ||
        c == ')') {
      escaped << c;
    } else {
      escaped << std::uppercase;
      escaped << '%' << std::setw(2)
              << static_cast<int>(static_cast<unsigned char>(c));
      escaped << std::nouppercase;
    }
  }

  return escaped.str();
}

std::string encode_url(const std::string &s) {
  std::string result;
  result.reserve(s.size());

  for (size_t i = 0; s[i]; i++) {
    switch (s[i]) {
    case ' ': result += "%20"; break;
    case '+': result += "%2B"; break;
    case '\r': result += "%0D"; break;
    case '\n': result += "%0A"; break;
    case '\'': result += "%27"; break;
    case ',': result += "%2C"; break;
    // case ':': result += "%3A"; break; // ok? probably...
    case ';': result += "%3B"; break;
    default:
      auto c = static_cast<uint8_t>(s[i]);
      if (c >= 0x80) {
        result += '%';
        char hex[4];
        auto len = snprintf(hex, sizeof(hex) - 1, "%02X", c);
        assert(len == 2);
        result.append(hex, static_cast<size_t>(len));
      } else {
        result += s[i];
      }
      break;
    }
  }

  return result;
}

std::string decode_url(const std::string &s,
                              bool convert_plus_to_space) {
  std::string result;

  for (size_t i = 0; i < s.size(); i++) {
    if (s[i] == '%' && i + 1 < s.size()) {
      if (s[i + 1] == 'u') {
        auto val = 0;
        if (from_hex_to_i(s, i + 2, 4, val)) {
          // 4 digits Unicode codes
          char buff[4];
          size_t len = to_utf8(val, buff);
          if (len > 0) { result.append(buff, len); }
          i += 5; // 'u0000'
        } else {
          result += s[i];
        }
      } else {
        auto val = 0;
        if (from_hex_to_i(s, i + 1, 2, val)) {
          // 2 digits hex codes
          result += static_cast<char>(val);
          i += 2; // '00'
        } else {
          result += s[i];
        }
      }
    } else if (convert_plus_to_space && s[i] == '+') {
      result += ' ';
    } else {
      result += s[i];
    }
  }

  return result;
}

std::string file_extension(const std::string &path) {
  std::smatch m;
  thread_local auto re = std::regex("\\.([a-zA-Z0-9]+)$");
  if (std::regex_search(path, m, re)) { return m[1].str(); }
  return std::string();
}

bool is_space_or_tab(char c) { return c == ' ' || c == '\t'; }

std::pair<size_t, size_t> trim(const char *b, const char *e, size_t left,
                                      size_t right) {
  while (b + left < e && is_space_or_tab(b[left])) {
    left++;
  }
  while (right > 0 && is_space_or_tab(b[right - 1])) {
    right--;
  }
  return std::make_pair(left, right);
}

std::string trim_copy(const std::string &s) {
  auto r = trim(s.data(), s.data() + s.size(), 0, s.size());
  return s.substr(r.first, r.second - r.first);
}

std::string trim_double_quotes_copy(const std::string &s) {
  if (s.length() >= 2 && s.front() == '"' && s.back() == '"') {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

void
divide(const char *data, std::size_t size, char d,
       std::function<void(const char *, std::size_t, const char *, std::size_t)>
           fn) {
  const auto it = std::find(data, data + size, d);
  const auto found = static_cast<std::size_t>(it != data + size);
  const auto lhs_data = data;
  const auto lhs_size = static_cast<std::size_t>(it - data);
  const auto rhs_data = it + found;
  const auto rhs_size = size - lhs_size - found;

  fn(lhs_data, lhs_size, rhs_data, rhs_size);
}

void
divide(const std::string &str, char d,
       std::function<void(const char *, std::size_t, const char *, std::size_t)>
           fn) {
  divide(str.data(), str.size(), d, std::move(fn));
}

void split(const char *b, const char *e, char d,
                  std::function<void(const char *, const char *)> fn) {
  return split(b, e, d, (std::numeric_limits<size_t>::max)(), std::move(fn));
}

void split(const char *b, const char *e, char d, size_t m,
                  std::function<void(const char *, const char *)> fn) {
  size_t i = 0;
  size_t beg = 0;
  size_t count = 1;

  while (e ? (b + i < e) : (b[i] != '\0')) {
    if (b[i] == d && count < m) {
      auto r = trim(b, e, beg, i);
      if (r.first < r.second) { fn(&b[r.first], &b[r.second]); }
      beg = i + 1;
      count++;
    }
    i++;
  }

  if (i) {
    auto r = trim(b, e, beg, i);
    if (r.first < r.second) { fn(&b[r.first], &b[r.second]); }
  }
}

stream_line_reader::stream_line_reader(Stream &strm, char *fixed_buffer,
                                              size_t fixed_buffer_size)
    : strm_(strm), fixed_buffer_(fixed_buffer),
      fixed_buffer_size_(fixed_buffer_size) {}

const char *stream_line_reader::ptr() const {
  if (growable_buffer_.empty()) {
    return fixed_buffer_;
  } else {
    return growable_buffer_.data();
  }
}

size_t stream_line_reader::size() const {
  if (growable_buffer_.empty()) {
    return fixed_buffer_used_size_;
  } else {
    return growable_buffer_.size();
  }
}

bool stream_line_reader::end_with_crlf() const {
  auto end = ptr() + size();
  return size() >= 2 && end[-2] == '\r' && end[-1] == '\n';
}

bool stream_line_reader::getline() {
  fixed_buffer_used_size_ = 0;
  growable_buffer_.clear();

#ifndef CPPHTTPLIB_ALLOW_LF_AS_LINE_TERMINATOR
  char prev_byte = 0;
#endif

  for (size_t i = 0;; i++) {
    if (size() >= CPPHTTPLIB_MAX_LINE_LENGTH) {
      // Treat exceptionally long lines as an error to
      // prevent infinite loops/memory exhaustion
      return false;
    }
    char byte;
    auto n = strm_.read(&byte, 1);

    if (n < 0) {
      return false;
    } else if (n == 0) {
      if (i == 0) {
        return false;
      } else {
        break;
      }
    }

    append(byte);

#ifdef CPPHTTPLIB_ALLOW_LF_AS_LINE_TERMINATOR
    if (byte == '\n') { break; }
#else
    if (prev_byte == '\r' && byte == '\n') { break; }
    prev_byte = byte;
#endif
  }

  return true;
}

void stream_line_reader::append(char c) {
  if (fixed_buffer_used_size_ < fixed_buffer_size_ - 1) {
    fixed_buffer_[fixed_buffer_used_size_++] = c;
    fixed_buffer_[fixed_buffer_used_size_] = '\0';
  } else {
    if (growable_buffer_.empty()) {
      assert(fixed_buffer_[fixed_buffer_used_size_] == '\0');
      growable_buffer_.assign(fixed_buffer_, fixed_buffer_used_size_);
    }
    growable_buffer_ += c;
  }
}

mmap::mmap(const char *path) { open(path); }

mmap::~mmap() { close(); }

bool mmap::open(const char *path) {
  close();

#if defined(_WIN32)
  auto wpath = u8string_to_wstring(path);
  if (wpath.empty()) { return false; }

#if _WIN32_WINNT >= _WIN32_WINNT_WIN8
  hFile_ = ::CreateFile2(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                         OPEN_EXISTING, NULL);
#else
  hFile_ = ::CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
#endif

  if (hFile_ == INVALID_HANDLE_VALUE) { return false; }

  LARGE_INTEGER size{};
  if (!::GetFileSizeEx(hFile_, &size)) { return false; }
  // If the following line doesn't compile due to QuadPart, update Windows SDK.
  // See:
  // https://github.com/yhirose/cpp-httplib/issues/1903#issuecomment-2316520721
  if (static_cast<ULONGLONG>(size.QuadPart) >
      (std::numeric_limits<decltype(size_)>::max)()) {
    // `size_t` might be 32-bits, on 32-bits Windows.
    return false;
  }
  size_ = static_cast<size_t>(size.QuadPart);

#if _WIN32_WINNT >= _WIN32_WINNT_WIN8
  hMapping_ =
      ::CreateFileMappingFromApp(hFile_, NULL, PAGE_READONLY, size_, NULL);
#else
  hMapping_ = ::CreateFileMappingW(hFile_, NULL, PAGE_READONLY, 0, 0, NULL);
#endif

  // Special treatment for an empty file...
  if (hMapping_ == NULL && size_ == 0) {
    close();
    is_open_empty_file = true;
    return true;
  }

  if (hMapping_ == NULL) {
    close();
    return false;
  }

#if _WIN32_WINNT >= _WIN32_WINNT_WIN8
  addr_ = ::MapViewOfFileFromApp(hMapping_, FILE_MAP_READ, 0, 0);
#else
  addr_ = ::MapViewOfFile(hMapping_, FILE_MAP_READ, 0, 0, 0);
#endif

  if (addr_ == nullptr) {
    close();
    return false;
  }
#else
  fd_ = ::open(path, O_RDONLY);
  if (fd_ == -1) { return false; }

  struct stat sb;
  if (fstat(fd_, &sb) == -1) {
    close();
    return false;
  }
  size_ = static_cast<size_t>(sb.st_size);

  addr_ = ::mmap(NULL, size_, PROT_READ, MAP_PRIVATE, fd_, 0);

  // Special treatment for an empty file...
  if (addr_ == MAP_FAILED && size_ == 0) {
    close();
    is_open_empty_file = true;
    return false;
  }
#endif

  return true;
}

bool mmap::is_open() const {
  return is_open_empty_file ? true : addr_ != nullptr;
}

size_t mmap::size() const { return size_; }

const char *mmap::data() const {
  return is_open_empty_file ? "" : static_cast<const char *>(addr_);
}

void mmap::close() {
#if defined(_WIN32)
  if (addr_) {
    ::UnmapViewOfFile(addr_);
    addr_ = nullptr;
  }

  if (hMapping_) {
    ::CloseHandle(hMapping_);
    hMapping_ = NULL;
  }

  if (hFile_ != INVALID_HANDLE_VALUE) {
    ::CloseHandle(hFile_);
    hFile_ = INVALID_HANDLE_VALUE;
  }

  is_open_empty_file = false;
#else
  if (addr_ != nullptr) {
    munmap(addr_, size_);
    addr_ = nullptr;
  }

  if (fd_ != -1) {
    ::close(fd_);
    fd_ = -1;
  }
#endif
  size_ = 0;
}
int close_socket(socket_t sock) {
#ifdef _WIN32
  return closesocket(sock);
#else
  return close(sock);
#endif
}

template <typename T> inline ssize_t handle_EINTR(T fn) {
  ssize_t res = 0;
  while (true) {
    res = fn();
    if (res < 0 && errno == EINTR) {
      std::this_thread::sleep_for(std::chrono::microseconds{1});
      continue;
    }
    break;
  }
  return res;
}

ssize_t read_socket(socket_t sock, void *ptr, size_t size, int flags) {
  return handle_EINTR([&]() {
    return recv(sock,
#ifdef _WIN32
                static_cast<char *>(ptr), static_cast<int>(size),
#else
                ptr, size,
#endif
                flags);
  });
}

ssize_t send_socket(socket_t sock, const void *ptr, size_t size,
                           int flags) {
  return handle_EINTR([&]() {
    return send(sock,
#ifdef _WIN32
                static_cast<const char *>(ptr), static_cast<int>(size),
#else
                ptr, size,
#endif
                flags);
  });
}

int poll_wrapper(struct pollfd *fds, nfds_t nfds, int timeout) {
#ifdef _WIN32
  return ::WSAPoll(fds, nfds, timeout);
#else
  return ::poll(fds, nfds, timeout);
#endif
}

template <bool Read>
ssize_t select_impl(socket_t sock, time_t sec, time_t usec) {
  struct pollfd pfd;
  pfd.fd = sock;
  pfd.events = (Read ? POLLIN : POLLOUT);

  auto timeout = static_cast<int>(sec * 1000 + usec / 1000);

  return handle_EINTR([&]() { return poll_wrapper(&pfd, 1, timeout); });
}

ssize_t select_read(socket_t sock, time_t sec, time_t usec) {
  return select_impl<true>(sock, sec, usec);
}

ssize_t select_write(socket_t sock, time_t sec, time_t usec) {
  return select_impl<false>(sock, sec, usec);
}

Error wait_until_socket_is_ready(socket_t sock, time_t sec,
                                        time_t usec) {
  struct pollfd pfd_read;
  pfd_read.fd = sock;
  pfd_read.events = POLLIN | POLLOUT;

  auto timeout = static_cast<int>(sec * 1000 + usec / 1000);

  auto poll_res =
      handle_EINTR([&]() { return poll_wrapper(&pfd_read, 1, timeout); });

  if (poll_res == 0) { return Error::ConnectionTimeout; }

  if (poll_res > 0 && pfd_read.revents & (POLLIN | POLLOUT)) {
    auto error = 0;
    socklen_t len = sizeof(error);
    auto res = getsockopt(sock, SOL_SOCKET, SO_ERROR,
                          reinterpret_cast<char *>(&error), &len);
    auto successful = res >= 0 && !error;
    return successful ? Error::Success : Error::Connection;
  }

  return Error::Connection;
}

bool is_socket_alive(socket_t sock) {
  const auto val = detail::select_read(sock, 0, 0);
  if (val == 0) {
    return true;
  } else if (val < 0 && errno == EBADF) {
    return false;
  }
  char buf[1];
  return detail::read_socket(sock, &buf[0], sizeof(buf), MSG_PEEK) > 0;
}

class SocketStream final : public Stream {
public:
  SocketStream(socket_t sock, time_t read_timeout_sec, time_t read_timeout_usec,
               time_t write_timeout_sec, time_t write_timeout_usec,
               time_t max_timeout_msec = 0,
               std::chrono::time_point<std::chrono::steady_clock> start_time =
                   (std::chrono::steady_clock::time_point::min)());
  ~SocketStream() override;

  bool is_readable() const override;
  bool wait_readable() const override;
  bool wait_writable() const override;
  ssize_t read(char *ptr, size_t size) override;
  ssize_t write(const char *ptr, size_t size) override;
  void get_remote_ip_and_port(std::string &ip, int &port) const override;
  void get_local_ip_and_port(std::string &ip, int &port) const override;
  socket_t socket() const override;
  time_t duration() const override;

private:
  socket_t sock_;
  time_t read_timeout_sec_;
  time_t read_timeout_usec_;
  time_t write_timeout_sec_;
  time_t write_timeout_usec_;
  time_t max_timeout_msec_;
  const std::chrono::time_point<std::chrono::steady_clock> start_time_;

  std::vector<char> read_buff_;
  size_t read_buff_off_ = 0;
  size_t read_buff_content_size_ = 0;

  static const size_t read_buff_size_ = 1024l * 4;
};

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
class SSLSocketStream final : public Stream {
public:
  SSLSocketStream(
      socket_t sock, SSL *ssl, time_t read_timeout_sec,
      time_t read_timeout_usec, time_t write_timeout_sec,
      time_t write_timeout_usec, time_t max_timeout_msec = 0,
      std::chrono::time_point<std::chrono::steady_clock> start_time =
          (std::chrono::steady_clock::time_point::min)());
  ~SSLSocketStream() override;

  bool is_readable() const override;
  bool wait_readable() const override;
  bool wait_writable() const override;
  ssize_t read(char *ptr, size_t size) override;
  ssize_t write(const char *ptr, size_t size) override;
  void get_remote_ip_and_port(std::string &ip, int &port) const override;
  void get_local_ip_and_port(std::string &ip, int &port) const override;
  socket_t socket() const override;
  time_t duration() const override;

private:
  socket_t sock_;
  SSL *ssl_;
  time_t read_timeout_sec_;
  time_t read_timeout_usec_;
  time_t write_timeout_sec_;
  time_t write_timeout_usec_;
  time_t max_timeout_msec_;
  const std::chrono::time_point<std::chrono::steady_clock> start_time_;
};
#endif

bool keep_alive(const std::atomic<socket_t> &svr_sock, socket_t sock,
                       time_t keep_alive_timeout_sec) {
  using namespace std::chrono;

  const auto interval_usec =
      CPPHTTPLIB_KEEPALIVE_TIMEOUT_CHECK_INTERVAL_USECOND;

  // Avoid expensive `steady_clock::now()` call for the first time
  if (select_read(sock, 0, interval_usec) > 0) { return true; }

  const auto start = steady_clock::now() - microseconds{interval_usec};
  const auto timeout = seconds{keep_alive_timeout_sec};

  while (true) {
    if (svr_sock == INVALID_SOCKET) {
      break; // Server socket is closed
    }

    auto val = select_read(sock, 0, interval_usec);
    if (val < 0) {
      break; // Ssocket error
    } else if (val == 0) {
      if (steady_clock::now() - start > timeout) {
        break; // Timeout
      }
    } else {
      return true; // Ready for read
    }
  }

  return false;
}

template <typename T>
bool
process_server_socket_core(const std::atomic<socket_t> &svr_sock, socket_t sock,
                           size_t keep_alive_max_count,
                           time_t keep_alive_timeout_sec, T callback) {
  assert(keep_alive_max_count > 0);
  auto ret = false;
  auto count = keep_alive_max_count;
  while (count > 0 && keep_alive(svr_sock, sock, keep_alive_timeout_sec)) {
    auto close_connection = count == 1;
    auto connection_closed = false;
    ret = callback(close_connection, connection_closed);
    if (!ret || connection_closed) { break; }
    count--;
  }
  return ret;
}

template <typename T>
bool
process_server_socket(const std::atomic<socket_t> &svr_sock, socket_t sock,
                      size_t keep_alive_max_count,
                      time_t keep_alive_timeout_sec, time_t read_timeout_sec,
                      time_t read_timeout_usec, time_t write_timeout_sec,
                      time_t write_timeout_usec, T callback) {
  return process_server_socket_core(
      svr_sock, sock, keep_alive_max_count, keep_alive_timeout_sec,
      [&](bool close_connection, bool &connection_closed) {
        SocketStream strm(sock, read_timeout_sec, read_timeout_usec,
                          write_timeout_sec, write_timeout_usec);
        return callback(strm, close_connection, connection_closed);
      });
}

bool process_client_socket(
    socket_t sock, time_t read_timeout_sec, time_t read_timeout_usec,
    time_t write_timeout_sec, time_t write_timeout_usec,
    time_t max_timeout_msec,
    std::chrono::time_point<std::chrono::steady_clock> start_time,
    std::function<bool(Stream &)> callback) {
  SocketStream strm(sock, read_timeout_sec, read_timeout_usec,
                    write_timeout_sec, write_timeout_usec, max_timeout_msec,
                    start_time);
  return callback(strm);
}

int shutdown_socket(socket_t sock) {
#ifdef _WIN32
  return shutdown(sock, SD_BOTH);
#else
  return shutdown(sock, SHUT_RDWR);
#endif
}

std::string escape_abstract_namespace_unix_domain(const std::string &s) {
  if (s.size() > 1 && s[0] == '\0') {
    auto ret = s;
    ret[0] = '@';
    return ret;
  }
  return s;
}

std::string
unescape_abstract_namespace_unix_domain(const std::string &s) {
  if (s.size() > 1 && s[0] == '@') {
    auto ret = s;
    ret[0] = '\0';
    return ret;
  }
  return s;
}

template <typename BindOrConnect>
socket_t create_socket(const std::string &host, const std::string &ip, int port,
                       int address_family, int socket_flags, bool tcp_nodelay,
                       bool ipv6_v6only, SocketOptions socket_options,
                       BindOrConnect bind_or_connect) {
  // Get address info
  const char *node = nullptr;
  struct addrinfo hints;
  struct addrinfo *result;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_IP;

  if (!ip.empty()) {
    node = ip.c_str();
    // Ask getaddrinfo to convert IP in c-string to address
    hints.ai_family = AF_UNSPEC;
    hints.ai_flags = AI_NUMERICHOST;
  } else {
    if (!host.empty()) { node = host.c_str(); }
    hints.ai_family = address_family;
    hints.ai_flags = socket_flags;
  }

  if (hints.ai_family == AF_UNIX) {
    const auto addrlen = host.length();
    if (addrlen > sizeof(sockaddr_un::sun_path)) { return INVALID_SOCKET; }

#ifdef SOCK_CLOEXEC
    auto sock = socket(hints.ai_family, hints.ai_socktype | SOCK_CLOEXEC,
                       hints.ai_protocol);
#else
    auto sock = socket(hints.ai_family, hints.ai_socktype, hints.ai_protocol);
#endif

    if (sock != INVALID_SOCKET) {
      sockaddr_un addr{};
      addr.sun_family = AF_UNIX;

      auto unescaped_host = unescape_abstract_namespace_unix_domain(host);
      std::copy(unescaped_host.begin(), unescaped_host.end(), addr.sun_path);

      hints.ai_addr = reinterpret_cast<sockaddr *>(&addr);
      hints.ai_addrlen = static_cast<socklen_t>(
          sizeof(addr) - sizeof(addr.sun_path) + addrlen);

#ifndef SOCK_CLOEXEC
#ifndef _WIN32
      fcntl(sock, F_SETFD, FD_CLOEXEC);
#endif
#endif

      if (socket_options) { socket_options(sock); }

#ifdef _WIN32
      // Setting SO_REUSEADDR seems not to work well with AF_UNIX on windows, so
      // remove the option.
      detail::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 0);
#endif

      bool dummy;
      if (!bind_or_connect(sock, hints, dummy)) {
        close_socket(sock);
        sock = INVALID_SOCKET;
      }
    }
    return sock;
  }

  auto service = std::to_string(port);

  if (getaddrinfo(node, service.c_str(), &hints, &result)) {
#if defined __linux__ && !defined __ANDROID__
    res_init();
#endif
    return INVALID_SOCKET;
  }
  auto se = detail::scope_exit([&] { freeaddrinfo(result); });

  for (auto rp = result; rp; rp = rp->ai_next) {
    // Create a socket
#ifdef _WIN32
    auto sock =
        WSASocketW(rp->ai_family, rp->ai_socktype, rp->ai_protocol, nullptr, 0,
                   WSA_FLAG_NO_HANDLE_INHERIT | WSA_FLAG_OVERLAPPED);
    /**
     * Since the WSA_FLAG_NO_HANDLE_INHERIT is only supported on Windows 7 SP1
     * and above the socket creation fails on older Windows Systems.
     *
     * Let's try to create a socket the old way in this case.
     *
     * Reference:
     * https://docs.microsoft.com/en-us/windows/win32/api/winsock2/nf-winsock2-wsasocketa
     *
     * WSA_FLAG_NO_HANDLE_INHERIT:
     * This flag is supported on Windows 7 with SP1, Windows Server 2008 R2 with
     * SP1, and later
     *
     */
    if (sock == INVALID_SOCKET) {
      sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    }
#else

#ifdef SOCK_CLOEXEC
    auto sock =
        socket(rp->ai_family, rp->ai_socktype | SOCK_CLOEXEC, rp->ai_protocol);
#else
    auto sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
#endif

#endif
    if (sock == INVALID_SOCKET) { continue; }

#if !defined _WIN32 && !defined SOCK_CLOEXEC
    if (fcntl(sock, F_SETFD, FD_CLOEXEC) == -1) {
      close_socket(sock);
      continue;
    }
#endif

    if (tcp_nodelay) { set_socket_opt(sock, IPPROTO_TCP, TCP_NODELAY, 1); }

    if (rp->ai_family == AF_INET6) {
      set_socket_opt(sock, IPPROTO_IPV6, IPV6_V6ONLY, ipv6_v6only ? 1 : 0);
    }

    if (socket_options) { socket_options(sock); }

    // bind or connect
    auto quit = false;
    if (bind_or_connect(sock, *rp, quit)) { return sock; }

    close_socket(sock);

    if (quit) { break; }
  }

  return INVALID_SOCKET;
}

void set_nonblocking(socket_t sock, bool nonblocking) {
#ifdef _WIN32
  auto flags = nonblocking ? 1UL : 0UL;
  ioctlsocket(sock, FIONBIO, &flags);
#else
  auto flags = fcntl(sock, F_GETFL, 0);
  fcntl(sock, F_SETFL,
        nonblocking ? (flags | O_NONBLOCK) : (flags & (~O_NONBLOCK)));
#endif
}

bool is_connection_error() {
#ifdef _WIN32
  return WSAGetLastError() != WSAEWOULDBLOCK;
#else
  return errno != EINPROGRESS;
#endif
}

bool bind_ip_address(socket_t sock, const std::string &host) {
  struct addrinfo hints;
  struct addrinfo *result;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = 0;

  if (getaddrinfo(host.c_str(), "0", &hints, &result)) { return false; }
  auto se = detail::scope_exit([&] { freeaddrinfo(result); });

  auto ret = false;
  for (auto rp = result; rp; rp = rp->ai_next) {
    const auto &ai = *rp;
    if (!::bind(sock, ai.ai_addr, static_cast<socklen_t>(ai.ai_addrlen))) {
      ret = true;
      break;
    }
  }

  return ret;
}

#if !defined _WIN32 && !defined ANDROID && !defined _AIX && !defined __MVS__
#define USE_IF2IP
#endif

#ifdef USE_IF2IP
std::string if2ip(int address_family, const std::string &ifn) {
  struct ifaddrs *ifap;
  getifaddrs(&ifap);
  auto se = detail::scope_exit([&] { freeifaddrs(ifap); });

  std::string addr_candidate;
  for (auto ifa = ifap; ifa; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr && ifn == ifa->ifa_name &&
        (AF_UNSPEC == address_family ||
         ifa->ifa_addr->sa_family == address_family)) {
      if (ifa->ifa_addr->sa_family == AF_INET) {
        auto sa = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr);
        char buf[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &sa->sin_addr, buf, INET_ADDRSTRLEN)) {
          return std::string(buf, INET_ADDRSTRLEN);
        }
      } else if (ifa->ifa_addr->sa_family == AF_INET6) {
        auto sa = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_addr);
        if (!IN6_IS_ADDR_LINKLOCAL(&sa->sin6_addr)) {
          char buf[INET6_ADDRSTRLEN] = {};
          if (inet_ntop(AF_INET6, &sa->sin6_addr, buf, INET6_ADDRSTRLEN)) {
            // equivalent to mac's IN6_IS_ADDR_UNIQUE_LOCAL
            auto s6_addr_head = sa->sin6_addr.s6_addr[0];
            if (s6_addr_head == 0xfc || s6_addr_head == 0xfd) {
              addr_candidate = std::string(buf, INET6_ADDRSTRLEN);
            } else {
              return std::string(buf, INET6_ADDRSTRLEN);
            }
          }
        }
      }
    }
  }
  return addr_candidate;
}
#endif

socket_t create_client_socket(
    const std::string &host, const std::string &ip, int port,
    int address_family, bool tcp_nodelay, bool ipv6_v6only,
    SocketOptions socket_options, time_t connection_timeout_sec,
    time_t connection_timeout_usec, time_t read_timeout_sec,
    time_t read_timeout_usec, time_t write_timeout_sec,
    time_t write_timeout_usec, const std::string &intf, Error &error) {
  auto sock = create_socket(
      host, ip, port, address_family, 0, tcp_nodelay, ipv6_v6only,
      std::move(socket_options),
      [&](socket_t sock2, struct addrinfo &ai, bool &quit) -> bool {
        if (!intf.empty()) {
#ifdef USE_IF2IP
          auto ip_from_if = if2ip(address_family, intf);
          if (ip_from_if.empty()) { ip_from_if = intf; }
          if (!bind_ip_address(sock2, ip_from_if)) {
            error = Error::BindIPAddress;
            return false;
          }
#endif
        }

        set_nonblocking(sock2, true);

        auto ret =
            ::connect(sock2, ai.ai_addr, static_cast<socklen_t>(ai.ai_addrlen));

        if (ret < 0) {
          if (is_connection_error()) {
            error = Error::Connection;
            return false;
          }
          error = wait_until_socket_is_ready(sock2, connection_timeout_sec,
                                             connection_timeout_usec);
          if (error != Error::Success) {
            if (error == Error::ConnectionTimeout) { quit = true; }
            return false;
          }
        }

        set_nonblocking(sock2, false);
        set_socket_opt_time(sock2, SOL_SOCKET, SO_RCVTIMEO, read_timeout_sec,
                            read_timeout_usec);
        set_socket_opt_time(sock2, SOL_SOCKET, SO_SNDTIMEO, write_timeout_sec,
                            write_timeout_usec);

        error = Error::Success;
        return true;
      });

  if (sock != INVALID_SOCKET) {
    error = Error::Success;
  } else {
    if (error == Error::Success) { error = Error::Connection; }
  }

  return sock;
}

bool get_ip_and_port(const struct sockaddr_storage &addr,
                            socklen_t addr_len, std::string &ip, int &port) {
  if (addr.ss_family == AF_INET) {
    port = ntohs(reinterpret_cast<const struct sockaddr_in *>(&addr)->sin_port);
  } else if (addr.ss_family == AF_INET6) {
    port =
        ntohs(reinterpret_cast<const struct sockaddr_in6 *>(&addr)->sin6_port);
  } else {
    return false;
  }

  std::array<char, NI_MAXHOST> ipstr{};
  if (getnameinfo(reinterpret_cast<const struct sockaddr *>(&addr), addr_len,
                  ipstr.data(), static_cast<socklen_t>(ipstr.size()), nullptr,
                  0, NI_NUMERICHOST)) {
    return false;
  }

  ip = ipstr.data();
  return true;
}

void get_local_ip_and_port(socket_t sock, std::string &ip, int &port) {
  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  if (!getsockname(sock, reinterpret_cast<struct sockaddr *>(&addr),
                   &addr_len)) {
    get_ip_and_port(addr, addr_len, ip, port);
  }
}

void get_remote_ip_and_port(socket_t sock, std::string &ip, int &port) {
  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);

  if (!getpeername(sock, reinterpret_cast<struct sockaddr *>(&addr),
                   &addr_len)) {
#ifndef _WIN32
    if (addr.ss_family == AF_UNIX) {
#if defined(__linux__)
      struct ucred ucred;
      socklen_t len = sizeof(ucred);
      if (getsockopt(sock, SOL_SOCKET, SO_PEERCRED, &ucred, &len) == 0) {
        port = ucred.pid;
      }
#elif defined(SOL_LOCAL) && defined(SO_PEERPID) // __APPLE__
      pid_t pid;
      socklen_t len = sizeof(pid);
      if (getsockopt(sock, SOL_LOCAL, SO_PEERPID, &pid, &len) == 0) {
        port = pid;
      }
#endif
      return;
    }
#endif
    get_ip_and_port(addr, addr_len, ip, port);
  }
}

constexpr unsigned int str2tag_core(const char *s, size_t l,
                                           unsigned int h) {
  return (l == 0)
             ? h
             : str2tag_core(
                   s + 1, l - 1,
                   // Unsets the 6 high bits of h, therefore no overflow happens
                   (((std::numeric_limits<unsigned int>::max)() >> 6) &
                    h * 33) ^
                       static_cast<unsigned char>(*s));
}

unsigned int str2tag(const std::string &s) {
  return str2tag_core(s.data(), s.size(), 0);
}

namespace udl {

constexpr unsigned int operator""_t(const char *s, size_t l) {
  return str2tag_core(s, l, 0);
}

} // namespace udl

std::string
find_content_type(const std::string &path,
                  const std::map<std::string, std::string> &user_data,
                  const std::string &default_content_type) {
  auto ext = file_extension(path);

  auto it = user_data.find(ext);
  if (it != user_data.end()) { return it->second; }

  using udl::operator""_t;

  switch (str2tag(ext)) {
  default: return default_content_type;

  case "css"_t: return "text/css";
  case "csv"_t: return "text/csv";
  case "htm"_t:
  case "html"_t: return "text/html";
  case "js"_t:
  case "mjs"_t: return "text/javascript";
  case "txt"_t: return "text/plain";
  case "vtt"_t: return "text/vtt";

  case "apng"_t: return "image/apng";
  case "avif"_t: return "image/avif";
  case "bmp"_t: return "image/bmp";
  case "gif"_t: return "image/gif";
  case "png"_t: return "image/png";
  case "svg"_t: return "image/svg+xml";
  case "webp"_t: return "image/webp";
  case "ico"_t: return "image/x-icon";
  case "tif"_t: return "image/tiff";
  case "tiff"_t: return "image/tiff";
  case "jpg"_t:
  case "jpeg"_t: return "image/jpeg";

  case "mp4"_t: return "video/mp4";
  case "mpeg"_t: return "video/mpeg";
  case "webm"_t: return "video/webm";

  case "mp3"_t: return "audio/mp3";
  case "mpga"_t: return "audio/mpeg";
  case "weba"_t: return "audio/webm";
  case "wav"_t: return "audio/wave";

  case "otf"_t: return "font/otf";
  case "ttf"_t: return "font/ttf";
  case "woff"_t: return "font/woff";
  case "woff2"_t: return "font/woff2";

  case "7z"_t: return "application/x-7z-compressed";
  case "atom"_t: return "application/atom+xml";
  case "pdf"_t: return "application/pdf";
  case "json"_t: return "application/json";
  case "rss"_t: return "application/rss+xml";
  case "tar"_t: return "application/x-tar";
  case "xht"_t:
  case "xhtml"_t: return "application/xhtml+xml";
  case "xslt"_t: return "application/xslt+xml";
  case "xml"_t: return "application/xml";
  case "gz"_t: return "application/gzip";
  case "zip"_t: return "application/zip";
  case "wasm"_t: return "application/wasm";
  }
}

bool can_compress_content_type(const std::string &content_type) {
  using udl::operator""_t;

  auto tag = str2tag(content_type);

  switch (tag) {
  case "image/svg+xml"_t:
  case "application/javascript"_t:
  case "application/json"_t:
  case "application/xml"_t:
  case "application/protobuf"_t:
  case "application/xhtml+xml"_t: return true;

  case "text/event-stream"_t: return false;

  default: return !content_type.rfind("text/", 0);
  }
}

EncodingType encoding_type(const Request &req, const Response &res) {
  auto ret =
      detail::can_compress_content_type(res.get_header_value("Content-Type"));
  if (!ret) { return EncodingType::None; }

  const auto &s = req.get_header_value("Accept-Encoding");
  (void)(s);

#ifdef CPPHTTPLIB_BROTLI_SUPPORT
  // TODO: 'Accept-Encoding' has br, not br;q=0
  ret = s.find("br") != std::string::npos;
  if (ret) { return EncodingType::Brotli; }
#endif

#ifdef CPPHTTPLIB_ZLIB_SUPPORT
  // TODO: 'Accept-Encoding' has gzip, not gzip;q=0
  ret = s.find("gzip") != std::string::npos;
  if (ret) { return EncodingType::Gzip; }
#endif

#ifdef CPPHTTPLIB_ZSTD_SUPPORT
  // TODO: 'Accept-Encoding' has zstd, not zstd;q=0
  ret = s.find("zstd") != std::string::npos;
  if (ret) { return EncodingType::Zstd; }
#endif

  return EncodingType::None;
}

bool nocompressor::compress(const char *data, size_t data_length,
                                   bool /*last*/, Callback callback) {
  if (!data_length) { return true; }
  return callback(data, data_length);
}

#ifdef CPPHTTPLIB_ZLIB_SUPPORT
gzip_compressor::gzip_compressor() {
  std::memset(&strm_, 0, sizeof(strm_));
  strm_.zalloc = Z_NULL;
  strm_.zfree = Z_NULL;
  strm_.opaque = Z_NULL;

  is_valid_ = deflateInit2(&strm_, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 31, 8,
                           Z_DEFAULT_STRATEGY) == Z_OK;
}

gzip_compressor::~gzip_compressor() { deflateEnd(&strm_); }

bool gzip_compressor::compress(const char *data, size_t data_length,
                                      bool last, Callback callback) {
  assert(is_valid_);

  do {
    constexpr size_t max_avail_in =
        (std::numeric_limits<decltype(strm_.avail_in)>::max)();

    strm_.avail_in = static_cast<decltype(strm_.avail_in)>(
        (std::min)(data_length, max_avail_in));
    strm_.next_in = const_cast<Bytef *>(reinterpret_cast<const Bytef *>(data));

    data_length -= strm_.avail_in;
    data += strm_.avail_in;

    auto flush = (last && data_length == 0) ? Z_FINISH : Z_NO_FLUSH;
    auto ret = Z_OK;

    std::array<char, CPPHTTPLIB_COMPRESSION_BUFSIZ> buff{};
    do {
      strm_.avail_out = static_cast<uInt>(buff.size());
      strm_.next_out = reinterpret_cast<Bytef *>(buff.data());

      ret = deflate(&strm_, flush);
      if (ret == Z_STREAM_ERROR) { return false; }

      if (!callback(buff.data(), buff.size() - strm_.avail_out)) {
        return false;
      }
    } while (strm_.avail_out == 0);

    assert((flush == Z_FINISH && ret == Z_STREAM_END) ||
           (flush == Z_NO_FLUSH && ret == Z_OK));
    assert(strm_.avail_in == 0);
  } while (data_length > 0);

  return true;
}

gzip_decompressor::gzip_decompressor() {
  std::memset(&strm_, 0, sizeof(strm_));
  strm_.zalloc = Z_NULL;
  strm_.zfree = Z_NULL;
  strm_.opaque = Z_NULL;

  // 15 is the value of wbits, which should be at the maximum possible value
  // to ensure that any gzip stream can be decoded. The offset of 32 specifies
  // that the stream type should be automatically detected either gzip or
  // deflate.
  is_valid_ = inflateInit2(&strm_, 32 + 15) == Z_OK;
}

gzip_decompressor::~gzip_decompressor() { inflateEnd(&strm_); }

bool gzip_decompressor::is_valid() const { return is_valid_; }

bool gzip_decompressor::decompress(const char *data, size_t data_length,
                                          Callback callback) {
  assert(is_valid_);

  auto ret = Z_OK;

  do {
    constexpr size_t max_avail_in =
        (std::numeric_limits<decltype(strm_.avail_in)>::max)();

    strm_.avail_in = static_cast<decltype(strm_.avail_in)>(
        (std::min)(data_length, max_avail_in));
    strm_.next_in = const_cast<Bytef *>(reinterpret_cast<const Bytef *>(data));

    data_length -= strm_.avail_in;
    data += strm_.avail_in;

    std::array<char, CPPHTTPLIB_COMPRESSION_BUFSIZ> buff{};
    while (strm_.avail_in > 0 && ret == Z_OK) {
      strm_.avail_out = static_cast<uInt>(buff.size());
      strm_.next_out = reinterpret_cast<Bytef *>(buff.data());

      ret = inflate(&strm_, Z_NO_FLUSH);

      assert(ret != Z_STREAM_ERROR);
      switch (ret) {
      case Z_NEED_DICT:
      case Z_DATA_ERROR:
      case Z_MEM_ERROR: inflateEnd(&strm_); return false;
      }

      if (!callback(buff.data(), buff.size() - strm_.avail_out)) {
        return false;
      }
    }

    if (ret != Z_OK && ret != Z_STREAM_END) { return false; }

  } while (data_length > 0);

  return true;
}
#endif

#ifdef CPPHTTPLIB_BROTLI_SUPPORT
brotli_compressor::brotli_compressor() {
  state_ = BrotliEncoderCreateInstance(nullptr, nullptr, nullptr);
}

brotli_compressor::~brotli_compressor() {
  BrotliEncoderDestroyInstance(state_);
}

bool brotli_compressor::compress(const char *data, size_t data_length,
                                        bool last, Callback callback) {
  std::array<uint8_t, CPPHTTPLIB_COMPRESSION_BUFSIZ> buff{};

  auto operation = last ? BROTLI_OPERATION_FINISH : BROTLI_OPERATION_PROCESS;
  auto available_in = data_length;
  auto next_in = reinterpret_cast<const uint8_t *>(data);

  for (;;) {
    if (last) {
      if (BrotliEncoderIsFinished(state_)) { break; }
    } else {
      if (!available_in) { break; }
    }

    auto available_out = buff.size();
    auto next_out = buff.data();

    if (!BrotliEncoderCompressStream(state_, operation, &available_in, &next_in,
                                     &available_out, &next_out, nullptr)) {
      return false;
    }

    auto output_bytes = buff.size() - available_out;
    if (output_bytes) {
      callback(reinterpret_cast<const char *>(buff.data()), output_bytes);
    }
  }

  return true;
}

brotli_decompressor::brotli_decompressor() {
  decoder_s = BrotliDecoderCreateInstance(0, 0, 0);
  decoder_r = decoder_s ? BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT
                        : BROTLI_DECODER_RESULT_ERROR;
}

brotli_decompressor::~brotli_decompressor() {
  if (decoder_s) { BrotliDecoderDestroyInstance(decoder_s); }
}

bool brotli_decompressor::is_valid() const { return decoder_s; }

bool brotli_decompressor::decompress(const char *data,
                                            size_t data_length,
                                            Callback callback) {
  if (decoder_r == BROTLI_DECODER_RESULT_SUCCESS ||
      decoder_r == BROTLI_DECODER_RESULT_ERROR) {
    return 0;
  }

  auto next_in = reinterpret_cast<const uint8_t *>(data);
  size_t avail_in = data_length;
  size_t total_out;

  decoder_r = BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT;

  std::array<char, CPPHTTPLIB_COMPRESSION_BUFSIZ> buff{};
  while (decoder_r == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) {
    char *next_out = buff.data();
    size_t avail_out = buff.size();

    decoder_r = BrotliDecoderDecompressStream(
        decoder_s, &avail_in, &next_in, &avail_out,
        reinterpret_cast<uint8_t **>(&next_out), &total_out);

    if (decoder_r == BROTLI_DECODER_RESULT_ERROR) { return false; }

    if (!callback(buff.data(), buff.size() - avail_out)) { return false; }
  }

  return decoder_r == BROTLI_DECODER_RESULT_SUCCESS ||
         decoder_r == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT;
}
#endif

#ifdef CPPHTTPLIB_ZSTD_SUPPORT
zstd_compressor::zstd_compressor() {
  ctx_ = ZSTD_createCCtx();
  ZSTD_CCtx_setParameter(ctx_, ZSTD_c_compressionLevel, ZSTD_fast);
}

zstd_compressor::~zstd_compressor() { ZSTD_freeCCtx(ctx_); }

bool zstd_compressor::compress(const char *data, size_t data_length,
                                      bool last, Callback callback) {
  std::array<char, CPPHTTPLIB_COMPRESSION_BUFSIZ> buff{};

  ZSTD_EndDirective mode = last ? ZSTD_e_end : ZSTD_e_continue;
  ZSTD_inBuffer input = {data, data_length, 0};

  bool finished;
  do {
    ZSTD_outBuffer output = {buff.data(), CPPHTTPLIB_COMPRESSION_BUFSIZ, 0};
    size_t const remaining = ZSTD_compressStream2(ctx_, &output, &input, mode);

    if (ZSTD_isError(remaining)) { return false; }

    if (!callback(buff.data(), output.pos)) { return false; }

    finished = last ? (remaining == 0) : (input.pos == input.size);

  } while (!finished);

  return true;
}

zstd_decompressor::zstd_decompressor() { ctx_ = ZSTD_createDCtx(); }

zstd_decompressor::~zstd_decompressor() { ZSTD_freeDCtx(ctx_); }

bool zstd_decompressor::is_valid() const { return ctx_ != nullptr; }

bool zstd_decompressor::decompress(const char *data, size_t data_length,
                                          Callback callback) {
  std::array<char, CPPHTTPLIB_COMPRESSION_BUFSIZ> buff{};
  ZSTD_inBuffer input = {data, data_length, 0};

  while (input.pos < input.size) {
    ZSTD_outBuffer output = {buff.data(), CPPHTTPLIB_COMPRESSION_BUFSIZ, 0};
    size_t const remaining = ZSTD_decompressStream(ctx_, &output, &input);

    if (ZSTD_isError(remaining)) { return false; }

    if (!callback(buff.data(), output.pos)) { return false; }
  }

  return true;
}
#endif

bool has_header(const Headers &headers, const std::string &key) {
  return headers.find(key) != headers.end();
}

const char *get_header_value(const Headers &headers,
                                    const std::string &key, const char *def,
                                    size_t id) {
  auto rng = headers.equal_range(key);
  auto it = rng.first;
  std::advance(it, static_cast<ssize_t>(id));
  if (it != rng.second) { return it->second.c_str(); }
  return def;
}

template <typename T>
bool parse_header(const char *beg, const char *end, T fn) {
  // Skip trailing spaces and tabs.
  while (beg < end && is_space_or_tab(end[-1])) {
    end--;
  }

  auto p = beg;
  while (p < end && *p != ':') {
    p++;
  }

  auto name = std::string(beg, p);
  if (!detail::fields::is_field_name(name)) { return false; }

  if (p == end) { return false; }

  auto key_end = p;

  if (*p++ != ':') { return false; }

  while (p < end && is_space_or_tab(*p)) {
    p++;
  }

  if (p <= end) {
    auto key_len = key_end - beg;
    if (!key_len) { return false; }

    auto key = std::string(beg, key_end);
    auto val = std::string(p, end);

    if (!detail::fields::is_field_value(val)) { return false; }

    if (case_ignore::equal(key, "Location") ||
        case_ignore::equal(key, "Referer")) {
      fn(key, val);
    } else {
      fn(key, decode_url(val, false));
    }

    return true;
  }

  return false;
}

bool read_headers(Stream &strm, Headers &headers) {
  const auto bufsiz = 2048;
  char buf[bufsiz];
  stream_line_reader line_reader(strm, buf, bufsiz);

  for (;;) {
    if (!line_reader.getline()) { return false; }

    // Check if the line ends with CRLF.
    auto line_terminator_len = 2;
    if (line_reader.end_with_crlf()) {
      // Blank line indicates end of headers.
      if (line_reader.size() == 2) { break; }
    } else {
#ifdef CPPHTTPLIB_ALLOW_LF_AS_LINE_TERMINATOR
      // Blank line indicates end of headers.
      if (line_reader.size() == 1) { break; }
      line_terminator_len = 1;
#else
      continue; // Skip invalid line.
#endif
    }

    if (line_reader.size() > CPPHTTPLIB_HEADER_MAX_LENGTH) { return false; }

    // Exclude line terminator
    auto end = line_reader.ptr() + line_reader.size() - line_terminator_len;

    if (!parse_header(line_reader.ptr(), end,
                      [&](const std::string &key, const std::string &val) {
                        headers.emplace(key, val);
                      })) {
      return false;
    }
  }

  return true;
}

bool read_content_with_length(Stream &strm, uint64_t len,
                                     Progress progress,
                                     ContentReceiverWithProgress out) {
  char buf[CPPHTTPLIB_RECV_BUFSIZ];

  uint64_t r = 0;
  while (r < len) {
    auto read_len = static_cast<size_t>(len - r);
    auto n = strm.read(buf, (std::min)(read_len, CPPHTTPLIB_RECV_BUFSIZ));
    if (n <= 0) { return false; }

    if (!out(buf, static_cast<size_t>(n), r, len)) { return false; }
    r += static_cast<uint64_t>(n);

    if (progress) {
      if (!progress(r, len)) { return false; }
    }
  }

  return true;
}

void skip_content_with_length(Stream &strm, uint64_t len) {
  char buf[CPPHTTPLIB_RECV_BUFSIZ];
  uint64_t r = 0;
  while (r < len) {
    auto read_len = static_cast<size_t>(len - r);
    auto n = strm.read(buf, (std::min)(read_len, CPPHTTPLIB_RECV_BUFSIZ));
    if (n <= 0) { return; }
    r += static_cast<uint64_t>(n);
  }
}

bool read_content_without_length(Stream &strm,
                                        ContentReceiverWithProgress out) {
  char buf[CPPHTTPLIB_RECV_BUFSIZ];
  uint64_t r = 0;
  for (;;) {
    auto n = strm.read(buf, CPPHTTPLIB_RECV_BUFSIZ);
    if (n == 0) { return true; }
    if (n < 0) { return false; }

    if (!out(buf, static_cast<size_t>(n), r, 0)) { return false; }
    r += static_cast<uint64_t>(n);
  }

  return true;
}

template <typename T>
bool read_content_chunked(Stream &strm, T &x,
                                 ContentReceiverWithProgress out) {
  const auto bufsiz = 16;
  char buf[bufsiz];

  stream_line_reader line_reader(strm, buf, bufsiz);

  if (!line_reader.getline()) { return false; }

  unsigned long chunk_len;
  while (true) {
    char *end_ptr;

    chunk_len = std::strtoul(line_reader.ptr(), &end_ptr, 16);

    if (end_ptr == line_reader.ptr()) { return false; }
    if (chunk_len == ULONG_MAX) { return false; }

    if (chunk_len == 0) { break; }

    if (!read_content_with_length(strm, chunk_len, nullptr, out)) {
      return false;
    }

    if (!line_reader.getline()) { return false; }

    if (strcmp(line_reader.ptr(), "\r\n") != 0) { return false; }

    if (!line_reader.getline()) { return false; }
  }

  assert(chunk_len == 0);

  // NOTE: In RFC 9112, '7.1 Chunked Transfer Coding' mentions "The chunked
  // transfer coding is complete when a chunk with a chunk-size of zero is
  // received, possibly followed by a trailer section, and finally terminated by
  // an empty line". https://www.rfc-editor.org/rfc/rfc9112.html#section-7.1
  //
  // In '7.1.3. Decoding Chunked', however, the pseudo-code in the section
  // does't care for the existence of the final CRLF. In other words, it seems
  // to be ok whether the final CRLF exists or not in the chunked data.
  // https://www.rfc-editor.org/rfc/rfc9112.html#section-7.1.3
  //
  // According to the reference code in RFC 9112, cpp-httplib now allows
  // chunked transfer coding data without the final CRLF.
  if (!line_reader.getline()) { return true; }

  while (strcmp(line_reader.ptr(), "\r\n") != 0) {
    if (line_reader.size() > CPPHTTPLIB_HEADER_MAX_LENGTH) { return false; }

    // Exclude line terminator
    constexpr auto line_terminator_len = 2;
    auto end = line_reader.ptr() + line_reader.size() - line_terminator_len;

    parse_header(line_reader.ptr(), end,
                 [&](const std::string &key, const std::string &val) {
                   x.headers.emplace(key, val);
                 });

    if (!line_reader.getline()) { return false; }
  }

  return true;
}

bool is_chunked_transfer_encoding(const Headers &headers) {
  return case_ignore::equal(
      get_header_value(headers, "Transfer-Encoding", "", 0), "chunked");
}

template <typename T, typename U>
bool prepare_content_receiver(T &x, int &status,
                              ContentReceiverWithProgress receiver,
                              bool decompress, U callback) {
  if (decompress) {
    std::string encoding = x.get_header_value("Content-Encoding");
    std::unique_ptr<decompressor> decompressor;

    if (encoding == "gzip" || encoding == "deflate") {
#ifdef CPPHTTPLIB_ZLIB_SUPPORT
      decompressor = detail::make_unique<gzip_decompressor>();
#else
      status = StatusCode::UnsupportedMediaType_415;
      return false;
#endif
    } else if (encoding.find("br") != std::string::npos) {
#ifdef CPPHTTPLIB_BROTLI_SUPPORT
      decompressor = detail::make_unique<brotli_decompressor>();
#else
      status = StatusCode::UnsupportedMediaType_415;
      return false;
#endif
    } else if (encoding == "zstd") {
#ifdef CPPHTTPLIB_ZSTD_SUPPORT
      decompressor = detail::make_unique<zstd_decompressor>();
#else
      status = StatusCode::UnsupportedMediaType_415;
      return false;
#endif
    }

    if (decompressor) {
      if (decompressor->is_valid()) {
        ContentReceiverWithProgress out = [&](const char *buf, size_t n,
                                              uint64_t off, uint64_t len) {
          return decompressor->decompress(buf, n,
                                          [&](const char *buf2, size_t n2) {
                                            return receiver(buf2, n2, off, len);
                                          });
        };
        return callback(std::move(out));
      } else {
        status = StatusCode::InternalServerError_500;
        return false;
      }
    }
  }

  ContentReceiverWithProgress out = [&](const char *buf, size_t n, uint64_t off,
                                        uint64_t len) {
    return receiver(buf, n, off, len);
  };
  return callback(std::move(out));
}

template <typename T>
bool read_content(Stream &strm, T &x, size_t payload_max_length, int &status,
                  Progress progress, ContentReceiverWithProgress receiver,
                  bool decompress) {
  return prepare_content_receiver(
      x, status, std::move(receiver), decompress,
      [&](const ContentReceiverWithProgress &out) {
        auto ret = true;
        auto exceed_payload_max_length = false;

        if (is_chunked_transfer_encoding(x.headers)) {
          ret = read_content_chunked(strm, x, out);
        } else if (!has_header(x.headers, "Content-Length")) {
          ret = read_content_without_length(strm, out);
        } else {
          auto is_invalid_value = false;
          auto len = get_header_value_u64(
              x.headers, "Content-Length",
              (std::numeric_limits<uint64_t>::max)(), 0, is_invalid_value);

          if (is_invalid_value) {
            ret = false;
          } else if (len > payload_max_length) {
            exceed_payload_max_length = true;
            skip_content_with_length(strm, len);
            ret = false;
          } else if (len > 0) {
            ret = read_content_with_length(strm, len, std::move(progress), out);
          }
        }

        if (!ret) {
          status = exceed_payload_max_length ? StatusCode::PayloadTooLarge_413
                                             : StatusCode::BadRequest_400;
        }
        return ret;
      });
}

ssize_t write_request_line(Stream &strm, const std::string &method,
                                  const std::string &path) {
  std::string s = method;
  s += " ";
  s += path;
  s += " HTTP/1.1\r\n";
  return strm.write(s.data(), s.size());
}

ssize_t write_response_line(Stream &strm, int status) {
  std::string s = "HTTP/1.1 ";
  s += std::to_string(status);
  s += " ";
  s += httplib::status_message(status);
  s += "\r\n";
  return strm.write(s.data(), s.size());
}

ssize_t write_headers(Stream &strm, const Headers &headers) {
  ssize_t write_len = 0;
  for (const auto &x : headers) {
    std::string s;
    s = x.first;
    s += ": ";
    s += x.second;
    s += "\r\n";

    auto len = strm.write(s.data(), s.size());
    if (len < 0) { return len; }
    write_len += len;
  }
  auto len = strm.write("\r\n");
  if (len < 0) { return len; }
  write_len += len;
  return write_len;
}

bool write_data(Stream &strm, const char *d, size_t l) {
  size_t offset = 0;
  while (offset < l) {
    auto length = strm.write(d + offset, l - offset);
    if (length < 0) { return false; }
    offset += static_cast<size_t>(length);
  }
  return true;
}

template <typename T>
bool write_content(Stream &strm, const ContentProvider &content_provider,
                          size_t offset, size_t length, T is_shutting_down,
                          Error &error) {
  size_t end_offset = offset + length;
  auto ok = true;
  DataSink data_sink;

  data_sink.write = [&](const char *d, size_t l) -> bool {
    if (ok) {
      if (write_data(strm, d, l)) {
        offset += l;
      } else {
        ok = false;
      }
    }
    return ok;
  };

  data_sink.is_writable = [&]() -> bool { return strm.wait_writable(); };

  while (offset < end_offset && !is_shutting_down()) {
    if (!strm.wait_writable()) {
      error = Error::Write;
      return false;
    } else if (!content_provider(offset, end_offset - offset, data_sink)) {
      error = Error::Canceled;
      return false;
    } else if (!ok) {
      error = Error::Write;
      return false;
    }
  }

  error = Error::Success;
  return true;
}

template <typename T>
bool write_content(Stream &strm, const ContentProvider &content_provider,
                          size_t offset, size_t length,
                          const T &is_shutting_down) {
  auto error = Error::Success;
  return write_content(strm, content_provider, offset, length, is_shutting_down,
                       error);
}

template <typename T>
bool
write_content_without_length(Stream &strm,
                             const ContentProvider &content_provider,
                             const T &is_shutting_down) {
  size_t offset = 0;
  auto data_available = true;
  auto ok = true;
  DataSink data_sink;

  data_sink.write = [&](const char *d, size_t l) -> bool {
    if (ok) {
      offset += l;
      if (!write_data(strm, d, l)) { ok = false; }
    }
    return ok;
  };

  data_sink.is_writable = [&]() -> bool { return strm.wait_writable(); };

  data_sink.done = [&](void) { data_available = false; };

  while (data_available && !is_shutting_down()) {
    if (!strm.wait_writable()) {
      return false;
    } else if (!content_provider(offset, 0, data_sink)) {
      return false;
    } else if (!ok) {
      return false;
    }
  }
  return true;
}

template <typename T, typename U>
bool
write_content_chunked(Stream &strm, const ContentProvider &content_provider,
                      const T &is_shutting_down, U &compressor, Error &error) {
  size_t offset = 0;
  auto data_available = true;
  auto ok = true;
  DataSink data_sink;

  data_sink.write = [&](const char *d, size_t l) -> bool {
    if (ok) {
      data_available = l > 0;
      offset += l;

      std::string payload;
      if (compressor.compress(d, l, false,
                              [&](const char *data, size_t data_len) {
                                payload.append(data, data_len);
                                return true;
                              })) {
        if (!payload.empty()) {
          // Emit chunked response header and footer for each chunk
          auto chunk =
              from_i_to_hex(payload.size()) + "\r\n" + payload + "\r\n";
          if (!write_data(strm, chunk.data(), chunk.size())) { ok = false; }
        }
      } else {
        ok = false;
      }
    }
    return ok;
  };

  data_sink.is_writable = [&]() -> bool { return strm.wait_writable(); };

  auto done_with_trailer = [&](const Headers *trailer) {
    if (!ok) { return; }

    data_available = false;

    std::string payload;
    if (!compressor.compress(nullptr, 0, true,
                             [&](const char *data, size_t data_len) {
                               payload.append(data, data_len);
                               return true;
                             })) {
      ok = false;
      return;
    }

    if (!payload.empty()) {
      // Emit chunked response header and footer for each chunk
      auto chunk = from_i_to_hex(payload.size()) + "\r\n" + payload + "\r\n";
      if (!write_data(strm, chunk.data(), chunk.size())) {
        ok = false;
        return;
      }
    }

    constexpr const char done_marker[] = "0\r\n";
    if (!write_data(strm, done_marker, str_len(done_marker))) { ok = false; }

    // Trailer
    if (trailer) {
      for (const auto &kv : *trailer) {
        std::string field_line = kv.first + ": " + kv.second + "\r\n";
        if (!write_data(strm, field_line.data(), field_line.size())) {
          ok = false;
        }
      }
    }

    constexpr const char crlf[] = "\r\n";
    if (!write_data(strm, crlf, str_len(crlf))) { ok = false; }
  };

  data_sink.done = [&](void) { done_with_trailer(nullptr); };

  data_sink.done_with_trailer = [&](const Headers &trailer) {
    done_with_trailer(&trailer);
  };

  while (data_available && !is_shutting_down()) {
    if (!strm.wait_writable()) {
      error = Error::Write;
      return false;
    } else if (!content_provider(offset, 0, data_sink)) {
      error = Error::Canceled;
      return false;
    } else if (!ok) {
      error = Error::Write;
      return false;
    }
  }

  error = Error::Success;
  return true;
}

template <typename T, typename U>
bool write_content_chunked(Stream &strm,
                                  const ContentProvider &content_provider,
                                  const T &is_shutting_down, U &compressor) {
  auto error = Error::Success;
  return write_content_chunked(strm, content_provider, is_shutting_down,
                               compressor, error);
}

template <typename T>
bool redirect(T &cli, Request &req, Response &res,
                     const std::string &path, const std::string &location,
                     Error &error) {
  Request new_req = req;
  new_req.path = path;
  new_req.redirect_count_ -= 1;

  if (res.status == StatusCode::SeeOther_303 &&
      (req.method != "GET" && req.method != "HEAD")) {
    new_req.method = "GET";
    new_req.body.clear();
    new_req.headers.clear();
  }

  Response new_res;

  auto ret = cli.send(new_req, new_res, error);
  if (ret) {
    req = new_req;
    res = new_res;

    if (res.location.empty()) { res.location = location; }
  }
  return ret;
}

std::string params_to_query_str(const Params &params) {
  std::string query;

  for (auto it = params.begin(); it != params.end(); ++it) {
    if (it != params.begin()) { query += "&"; }
    query += it->first;
    query += "=";
    query += encode_query_param(it->second);
  }
  return query;
}

void parse_query_text(const char *data, std::size_t size,
                             Params &params) {
  std::set<std::string> cache;
  split(data, data + size, '&', [&](const char *b, const char *e) {
    std::string kv(b, e);
    if (cache.find(kv) != cache.end()) { return; }
    cache.insert(std::move(kv));

    std::string key;
    std::string val;
    divide(b, static_cast<std::size_t>(e - b), '=',
           [&](const char *lhs_data, std::size_t lhs_size, const char *rhs_data,
               std::size_t rhs_size) {
             key.assign(lhs_data, lhs_size);
             val.assign(rhs_data, rhs_size);
           });

    if (!key.empty()) {
      params.emplace(decode_url(key, true), decode_url(val, true));
    }
  });
}

void parse_query_text(const std::string &s, Params &params) {
  parse_query_text(s.data(), s.size(), params);
}

bool parse_multipart_boundary(const std::string &content_type,
                                     std::string &boundary) {
  auto boundary_keyword = "boundary=";
  auto pos = content_type.find(boundary_keyword);
  if (pos == std::string::npos) { return false; }
  auto end = content_type.find(';', pos);
  auto beg = pos + strlen(boundary_keyword);
  boundary = trim_double_quotes_copy(content_type.substr(beg, end - beg));
  return !boundary.empty();
}

void parse_disposition_params(const std::string &s, Params &params) {
  std::set<std::string> cache;
  split(s.data(), s.data() + s.size(), ';', [&](const char *b, const char *e) {
    std::string kv(b, e);
    if (cache.find(kv) != cache.end()) { return; }
    cache.insert(kv);

    std::string key;
    std::string val;
    split(b, e, '=', [&](const char *b2, const char *e2) {
      if (key.empty()) {
        key.assign(b2, e2);
      } else {
        val.assign(b2, e2);
      }
    });

    if (!key.empty()) {
      params.emplace(trim_double_quotes_copy((key)),
                     trim_double_quotes_copy((val)));
    }
  });
}

#ifdef CPPHTTPLIB_NO_EXCEPTIONS
bool parse_range_header(const std::string &s, Ranges &ranges) {
#else
bool parse_range_header(const std::string &s, Ranges &ranges) try {
#endif
  auto is_valid = [](const std::string &str) {
    return std::all_of(str.cbegin(), str.cend(),
                       [](unsigned char c) { return std::isdigit(c); });
  };

  if (s.size() > 7 && s.compare(0, 6, "bytes=") == 0) {
    const auto pos = static_cast<size_t>(6);
    const auto len = static_cast<size_t>(s.size() - 6);
    auto all_valid_ranges = true;
    split(&s[pos], &s[pos + len], ',', [&](const char *b, const char *e) {
      if (!all_valid_ranges) { return; }

      const auto it = std::find(b, e, '-');
      if (it == e) {
        all_valid_ranges = false;
        return;
      }

      const auto lhs = std::string(b, it);
      const auto rhs = std::string(it + 1, e);
      if (!is_valid(lhs) || !is_valid(rhs)) {
        all_valid_ranges = false;
        return;
      }

      const auto first =
          static_cast<ssize_t>(lhs.empty() ? -1 : std::stoll(lhs));
      const auto last =
          static_cast<ssize_t>(rhs.empty() ? -1 : std::stoll(rhs));
      if ((first == -1 && last == -1) ||
          (first != -1 && last != -1 && first > last)) {
        all_valid_ranges = false;
        return;
      }

      ranges.emplace_back(first, last);
    });
    return all_valid_ranges && !ranges.empty();
  }
  return false;
#ifdef CPPHTTPLIB_NO_EXCEPTIONS
}
#else
} catch (...) { return false; }
#endif

class MultipartFormDataParser {
public:
  MultipartFormDataParser() = default;

  void set_boundary(std::string &&boundary) {
    boundary_ = boundary;
    dash_boundary_crlf_ = dash_ + boundary_ + crlf_;
    crlf_dash_boundary_ = crlf_ + dash_ + boundary_;
  }

  bool is_valid() const { return is_valid_; }

  bool parse(const char *buf, size_t n, const ContentReceiver &content_callback,
             const MultipartContentHeader &header_callback) {

    buf_append(buf, n);

    while (buf_size() > 0) {
      switch (state_) {
      case 0: { // Initial boundary
        buf_erase(buf_find(dash_boundary_crlf_));
        if (dash_boundary_crlf_.size() > buf_size()) { return true; }
        if (!buf_start_with(dash_boundary_crlf_)) { return false; }
        buf_erase(dash_boundary_crlf_.size());
        state_ = 1;
        break;
      }
      case 1: { // New entry
        clear_file_info();
        state_ = 2;
        break;
      }
      case 2: { // Headers
        auto pos = buf_find(crlf_);
        if (pos > CPPHTTPLIB_HEADER_MAX_LENGTH) { return false; }
        while (pos < buf_size()) {
          // Empty line
          if (pos == 0) {
            if (!header_callback(file_)) {
              is_valid_ = false;
              return false;
            }
            buf_erase(crlf_.size());
            state_ = 3;
            break;
          }

          const auto header = buf_head(pos);

          if (!parse_header(header.data(), header.data() + header.size(),
                            [&](const std::string &, const std::string &) {})) {
            is_valid_ = false;
            return false;
          }

          constexpr const char header_content_type[] = "Content-Type:";

          if (start_with_case_ignore(header, header_content_type)) {
            file_.content_type =
                trim_copy(header.substr(str_len(header_content_type)));
          } else {
            thread_local const std::regex re_content_disposition(
                R"~(^Content-Disposition:\s*form-data;\s*(.*)$)~",
                std::regex_constants::icase);

            std::smatch m;
            if (std::regex_match(header, m, re_content_disposition)) {
              Params params;
              parse_disposition_params(m[1], params);

              auto it = params.find("name");
              if (it != params.end()) {
                file_.name = it->second;
              } else {
                is_valid_ = false;
                return false;
              }

              it = params.find("filename");
              if (it != params.end()) { file_.filename = it->second; }

              it = params.find("filename*");
              if (it != params.end()) {
                // Only allow UTF-8 encoding...
                thread_local const std::regex re_rfc5987_encoding(
                    R"~(^UTF-8''(.+?)$)~", std::regex_constants::icase);

                std::smatch m2;
                if (std::regex_match(it->second, m2, re_rfc5987_encoding)) {
                  file_.filename = decode_url(m2[1], false); // override...
                } else {
                  is_valid_ = false;
                  return false;
                }
              }
            }
          }
          buf_erase(pos + crlf_.size());
          pos = buf_find(crlf_);
        }
        if (state_ != 3) { return true; }
        break;
      }
      case 3: { // Body
        if (crlf_dash_boundary_.size() > buf_size()) { return true; }
        auto pos = buf_find(crlf_dash_boundary_);
        if (pos < buf_size()) {
          if (!content_callback(buf_data(), pos)) {
            is_valid_ = false;
            return false;
          }
          buf_erase(pos + crlf_dash_boundary_.size());
          state_ = 4;
        } else {
          auto len = buf_size() - crlf_dash_boundary_.size();
          if (len > 0) {
            if (!content_callback(buf_data(), len)) {
              is_valid_ = false;
              return false;
            }
            buf_erase(len);
          }
          return true;
        }
        break;
      }
      case 4: { // Boundary
        if (crlf_.size() > buf_size()) { return true; }
        if (buf_start_with(crlf_)) {
          buf_erase(crlf_.size());
          state_ = 1;
        } else {
          if (dash_.size() > buf_size()) { return true; }
          if (buf_start_with(dash_)) {
            buf_erase(dash_.size());
            is_valid_ = true;
            buf_erase(buf_size()); // Remove epilogue
          } else {
            return true;
          }
        }
        break;
      }
      }
    }

    return true;
  }

private:
  void clear_file_info() {
    file_.name.clear();
    file_.filename.clear();
    file_.content_type.clear();
  }

  bool start_with_case_ignore(const std::string &a, const char *b) const {
    const auto b_len = strlen(b);
    if (a.size() < b_len) { return false; }
    for (size_t i = 0; i < b_len; i++) {
      if (case_ignore::to_lower(a[i]) != case_ignore::to_lower(b[i])) {
        return false;
      }
    }
    return true;
  }

  const std::string dash_ = "--";
  const std::string crlf_ = "\r\n";
  std::string boundary_;
  std::string dash_boundary_crlf_;
  std::string crlf_dash_boundary_;

  size_t state_ = 0;
  bool is_valid_ = false;
  MultipartFormData file_;

  // Buffer
  bool start_with(const std::string &a, size_t spos, size_t epos,
                  const std::string &b) const {
    if (epos - spos < b.size()) { return false; }
    for (size_t i = 0; i < b.size(); i++) {
      if (a[i + spos] != b[i]) { return false; }
    }
    return true;
  }

  size_t buf_size() const { return buf_epos_ - buf_spos_; }

  const char *buf_data() const { return &buf_[buf_spos_]; }

  std::string buf_head(size_t l) const { return buf_.substr(buf_spos_, l); }

  bool buf_start_with(const std::string &s) const {
    return start_with(buf_, buf_spos_, buf_epos_, s);
  }

  size_t buf_find(const std::string &s) const {
    auto c = s.front();

    size_t off = buf_spos_;
    while (off < buf_epos_) {
      auto pos = off;
      while (true) {
        if (pos == buf_epos_) { return buf_size(); }
        if (buf_[pos] == c) { break; }
        pos++;
      }

      auto remaining_size = buf_epos_ - pos;
      if (s.size() > remaining_size) { return buf_size(); }

      if (start_with(buf_, pos, buf_epos_, s)) { return pos - buf_spos_; }

      off = pos + 1;
    }

    return buf_size();
  }

  void buf_append(const char *data, size_t n) {
    auto remaining_size = buf_size();
    if (remaining_size > 0 && buf_spos_ > 0) {
      for (size_t i = 0; i < remaining_size; i++) {
        buf_[i] = buf_[buf_spos_ + i];
      }
    }
    buf_spos_ = 0;
    buf_epos_ = remaining_size;

    if (remaining_size + n > buf_.size()) { buf_.resize(remaining_size + n); }

    for (size_t i = 0; i < n; i++) {
      buf_[buf_epos_ + i] = data[i];
    }
    buf_epos_ += n;
  }

  void buf_erase(size_t size) { buf_spos_ += size; }

  std::string buf_;
  size_t buf_spos_ = 0;
  size_t buf_epos_ = 0;
};

std::string random_string(size_t length) {
  constexpr const char data[] =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  thread_local auto engine([]() {
    // std::random_device might actually be deterministic on some
    // platforms, but due to lack of support in the c++ standard library,
    // doing better requires either some ugly hacks or breaking portability.
    std::random_device seed_gen;
    // Request 128 bits of entropy for initialization
    std::seed_seq seed_sequence{seed_gen(), seed_gen(), seed_gen(), seed_gen()};
    return std::mt19937(seed_sequence);
  }());

  std::string result;
  for (size_t i = 0; i < length; i++) {
    result += data[engine() % (sizeof(data) - 1)];
  }
  return result;
}

std::string make_multipart_data_boundary() {
  return "--cpp-httplib-multipart-data-" + detail::random_string(16);
}

bool is_multipart_boundary_chars_valid(const std::string &boundary) {
  auto valid = true;
  for (size_t i = 0; i < boundary.size(); i++) {
    auto c = boundary[i];
    if (!std::isalnum(c) && c != '-' && c != '_') {
      valid = false;
      break;
    }
  }
  return valid;
}

template <typename T>
std::string
serialize_multipart_formdata_item_begin(const T &item,
                                        const std::string &boundary) {
  std::string body = "--" + boundary + "\r\n";
  body += "Content-Disposition: form-data; name=\"" + item.name + "\"";
  if (!item.filename.empty()) {
    body += "; filename=\"" + item.filename + "\"";
  }
  body += "\r\n";
  if (!item.content_type.empty()) {
    body += "Content-Type: " + item.content_type + "\r\n";
  }
  body += "\r\n";

  return body;
}

std::string serialize_multipart_formdata_item_end() { return "\r\n"; }

std::string
serialize_multipart_formdata_finish(const std::string &boundary) {
  return "--" + boundary + "--\r\n";
}

std::string
serialize_multipart_formdata_get_content_type(const std::string &boundary) {
  return "multipart/form-data; boundary=" + boundary;
}

std::string
serialize_multipart_formdata(const MultipartFormDataItems &items,
                             const std::string &boundary, bool finish = true) {
  std::string body;

  for (const auto &item : items) {
    body += serialize_multipart_formdata_item_begin(item, boundary);
    body += item.content + serialize_multipart_formdata_item_end();
  }

  if (finish) { body += serialize_multipart_formdata_finish(boundary); }

  return body;
}

bool range_error(Request &req, Response &res) {
  if (!req.ranges.empty() && 200 <= res.status && res.status < 300) {
    ssize_t content_len = static_cast<ssize_t>(
        res.content_length_ ? res.content_length_ : res.body.size());

    ssize_t prev_first_pos = -1;
    ssize_t prev_last_pos = -1;
    size_t overwrapping_count = 0;

    // NOTE: The following Range check is based on '14.2. Range' in RFC 9110
    // 'HTTP Semantics' to avoid potential denial-of-service attacks.
    // https://www.rfc-editor.org/rfc/rfc9110#section-14.2

    // Too many ranges
    if (req.ranges.size() > CPPHTTPLIB_RANGE_MAX_COUNT) { return true; }

    for (auto &r : req.ranges) {
      auto &first_pos = r.first;
      auto &last_pos = r.second;

      if (first_pos == -1 && last_pos == -1) {
        first_pos = 0;
        last_pos = content_len;
      }

      if (first_pos == -1) {
        first_pos = content_len - last_pos;
        last_pos = content_len - 1;
      }

      // NOTE: RFC-9110 '14.1.2. Byte Ranges':
      // A client can limit the number of bytes requested without knowing the
      // size of the selected representation. If the last-pos value is absent,
      // or if the value is greater than or equal to the current length of the
      // representation data, the byte range is interpreted as the remainder of
      // the representation (i.e., the server replaces the value of last-pos
      // with a value that is one less than the current length of the selected
      // representation).
      // https://www.rfc-editor.org/rfc/rfc9110.html#section-14.1.2-6
      if (last_pos == -1 || last_pos >= content_len) {
        last_pos = content_len - 1;
      }

      // Range must be within content length
      if (!(0 <= first_pos && first_pos <= last_pos &&
            last_pos <= content_len - 1)) {
        return true;
      }

      // Ranges must be in ascending order
      if (first_pos <= prev_first_pos) { return true; }

      // Request must not have more than two overlapping ranges
      if (first_pos <= prev_last_pos) {
        overwrapping_count++;
        if (overwrapping_count > 2) { return true; }
      }

      prev_first_pos = (std::max)(prev_first_pos, first_pos);
      prev_last_pos = (std::max)(prev_last_pos, last_pos);
    }
  }

  return false;
}

std::pair<size_t, size_t>
get_range_offset_and_length(Range r, size_t content_length) {
  assert(r.first != -1 && r.second != -1);
  assert(0 <= r.first && r.first < static_cast<ssize_t>(content_length));
  assert(r.first <= r.second &&
         r.second < static_cast<ssize_t>(content_length));
  (void)(content_length);
  return std::make_pair(r.first, static_cast<size_t>(r.second - r.first) + 1);
}

std::string make_content_range_header_field(
    const std::pair<size_t, size_t> &offset_and_length, size_t content_length) {
  auto st = offset_and_length.first;
  auto ed = st + offset_and_length.second - 1;

  std::string field = "bytes ";
  field += std::to_string(st);
  field += "-";
  field += std::to_string(ed);
  field += "/";
  field += std::to_string(content_length);
  return field;
}

template <typename SToken, typename CToken, typename Content>
bool process_multipart_ranges_data(const Request &req,
                                   const std::string &boundary,
                                   const std::string &content_type,
                                   size_t content_length, SToken stoken,
                                   CToken ctoken, Content content) {
  for (size_t i = 0; i < req.ranges.size(); i++) {
    ctoken("--");
    stoken(boundary);
    ctoken("\r\n");
    if (!content_type.empty()) {
      ctoken("Content-Type: ");
      stoken(content_type);
      ctoken("\r\n");
    }

    auto offset_and_length =
        get_range_offset_and_length(req.ranges[i], content_length);

    ctoken("Content-Range: ");
    stoken(make_content_range_header_field(offset_and_length, content_length));
    ctoken("\r\n");
    ctoken("\r\n");

    if (!content(offset_and_length.first, offset_and_length.second)) {
      return false;
    }
    ctoken("\r\n");
  }

  ctoken("--");
  stoken(boundary);
  ctoken("--");

  return true;
}

void make_multipart_ranges_data(const Request &req, Response &res,
                                       const std::string &boundary,
                                       const std::string &content_type,
                                       size_t content_length,
                                       std::string &data) {
  process_multipart_ranges_data(
      req, boundary, content_type, content_length,
      [&](const std::string &token) { data += token; },
      [&](const std::string &token) { data += token; },
      [&](size_t offset, size_t length) {
        assert(offset + length <= content_length);
        data += res.body.substr(offset, length);
        return true;
      });
}

size_t get_multipart_ranges_data_length(const Request &req,
                                               const std::string &boundary,
                                               const std::string &content_type,
                                               size_t content_length) {
  size_t data_length = 0;

  process_multipart_ranges_data(
      req, boundary, content_type, content_length,
      [&](const std::string &token) { data_length += token.size(); },
      [&](const std::string &token) { data_length += token.size(); },
      [&](size_t /*offset*/, size_t length) {
        data_length += length;
        return true;
      });

  return data_length;
}

template <typename T>
bool
write_multipart_ranges_data(Stream &strm, const Request &req, Response &res,
                            const std::string &boundary,
                            const std::string &content_type,
                            size_t content_length, const T &is_shutting_down) {
  return process_multipart_ranges_data(
      req, boundary, content_type, content_length,
      [&](const std::string &token) { strm.write(token); },
      [&](const std::string &token) { strm.write(token); },
      [&](size_t offset, size_t length) {
        return write_content(strm, res.content_provider_, offset, length,
                             is_shutting_down);
      });
}

bool expect_content(const Request &req) {
  if (req.method == "POST" || req.method == "PUT" || req.method == "PATCH" ||
      req.method == "DELETE") {
    return true;
  }
  if (req.has_header("Content-Length") &&
      req.get_header_value_u64("Content-Length") > 0) {
    return true;
  }
  if (is_chunked_transfer_encoding(req.headers)) { return true; }
  return false;
}

bool has_crlf(const std::string &s) {
  auto p = s.c_str();
  while (*p) {
    if (*p == '\r' || *p == '\n') { return true; }
    p++;
  }
  return false;
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
std::string message_digest(const std::string &s, const EVP_MD *algo) {
  auto context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);

  unsigned int hash_length = 0;
  unsigned char hash[EVP_MAX_MD_SIZE];

  EVP_DigestInit_ex(context.get(), algo, nullptr);
  EVP_DigestUpdate(context.get(), s.c_str(), s.size());
  EVP_DigestFinal_ex(context.get(), hash, &hash_length);

  std::stringstream ss;
  for (auto i = 0u; i < hash_length; ++i) {
    ss << std::hex << std::setw(2) << std::setfill('0')
       << static_cast<unsigned int>(hash[i]);
  }

  return ss.str();
}

std::string MD5(const std::string &s) {
  return message_digest(s, EVP_md5());
}

std::string SHA_256(const std::string &s) {
  return message_digest(s, EVP_sha256());
}

std::string SHA_512(const std::string &s) {
  return message_digest(s, EVP_sha512());
}

std::pair<std::string, std::string> make_digest_authentication_header(
    const Request &req, const std::map<std::string, std::string> &auth,
    size_t cnonce_count, const std::string &cnonce, const std::string &username,
    const std::string &password, bool is_proxy = false) {
  std::string nc;
  {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(8) << std::hex << cnonce_count;
    nc = ss.str();
  }

  std::string qop;
  if (auth.find("qop") != auth.end()) {
    qop = auth.at("qop");
    if (qop.find("auth-int") != std::string::npos) {
      qop = "auth-int";
    } else if (qop.find("auth") != std::string::npos) {
      qop = "auth";
    } else {
      qop.clear();
    }
  }

  std::string algo = "MD5";
  if (auth.find("algorithm") != auth.end()) { algo = auth.at("algorithm"); }

  std::string response;
  {
    auto H = algo == "SHA-256"   ? detail::SHA_256
             : algo == "SHA-512" ? detail::SHA_512
                                 : detail::MD5;

    auto A1 = username + ":" + auth.at("realm") + ":" + password;

    auto A2 = req.method + ":" + req.path;
    if (qop == "auth-int") { A2 += ":" + H(req.body); }

    if (qop.empty()) {
      response = H(H(A1) + ":" + auth.at("nonce") + ":" + H(A2));
    } else {
      response = H(H(A1) + ":" + auth.at("nonce") + ":" + nc + ":" + cnonce +
                   ":" + qop + ":" + H(A2));
    }
  }

  auto opaque = (auth.find("opaque") != auth.end()) ? auth.at("opaque") : "";

  auto field = "Digest username=\"" + username + "\", realm=\"" +
               auth.at("realm") + "\", nonce=\"" + auth.at("nonce") +
               "\", uri=\"" + req.path + "\", algorithm=" + algo +
               (qop.empty() ? ", response=\""
                            : ", qop=" + qop + ", nc=" + nc + ", cnonce=\"" +
                                  cnonce + "\", response=\"") +
               response + "\"" +
               (opaque.empty() ? "" : ", opaque=\"" + opaque + "\"");

  auto key = is_proxy ? "Proxy-Authorization" : "Authorization";
  return std::make_pair(key, field);
}

bool is_ssl_peer_could_be_closed(SSL *ssl, socket_t sock) {
  detail::set_nonblocking(sock, true);
  auto se = detail::scope_exit([&]() { detail::set_nonblocking(sock, false); });

  char buf[1];
  return !SSL_peek(ssl, buf, 1) &&
         SSL_get_error(ssl, 0) == SSL_ERROR_ZERO_RETURN;
}

#ifdef _WIN32
// NOTE: This code came up with the following stackoverflow post:
// https://stackoverflow.com/questions/9507184/can-openssl-on-windows-use-the-system-certificate-store
bool load_system_certs_on_windows(X509_STORE *store) {
  auto hStore = CertOpenSystemStoreW((HCRYPTPROV_LEGACY)NULL, L"ROOT");
  if (!hStore) { return false; }

  auto result = false;
  PCCERT_CONTEXT pContext = NULL;
  while ((pContext = CertEnumCertificatesInStore(hStore, pContext)) !=
         nullptr) {
    auto encoded_cert =
        static_cast<const unsigned char *>(pContext->pbCertEncoded);

    auto x509 = d2i_X509(NULL, &encoded_cert, pContext->cbCertEncoded);
    if (x509) {
      X509_STORE_add_cert(store, x509);
      X509_free(x509);
      result = true;
    }
  }

  CertFreeCertificateContext(pContext);
  CertCloseStore(hStore, 0);

  return result;
}
#elif defined(CPPHTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN) && defined(__APPLE__)
#if TARGET_OS_OSX
template <typename T>
using CFObjectPtr =
    std::unique_ptr<typename std::remove_pointer<T>::type, void (*)(CFTypeRef)>;

void cf_object_ptr_deleter(CFTypeRef obj) {
  if (obj) { CFRelease(obj); }
}

bool retrieve_certs_from_keychain(CFObjectPtr<CFArrayRef> &certs) {
  CFStringRef keys[] = {kSecClass, kSecMatchLimit, kSecReturnRef};
  CFTypeRef values[] = {kSecClassCertificate, kSecMatchLimitAll,
                        kCFBooleanTrue};

  CFObjectPtr<CFDictionaryRef> query(
      CFDictionaryCreate(nullptr, reinterpret_cast<const void **>(keys), values,
                         sizeof(keys) / sizeof(keys[0]),
                         &kCFTypeDictionaryKeyCallBacks,
                         &kCFTypeDictionaryValueCallBacks),
      cf_object_ptr_deleter);

  if (!query) { return false; }

  CFTypeRef security_items = nullptr;
  if (SecItemCopyMatching(query.get(), &security_items) != errSecSuccess ||
      CFArrayGetTypeID() != CFGetTypeID(security_items)) {
    return false;
  }

  certs.reset(reinterpret_cast<CFArrayRef>(security_items));
  return true;
}

bool retrieve_root_certs_from_keychain(CFObjectPtr<CFArrayRef> &certs) {
  CFArrayRef root_security_items = nullptr;
  if (SecTrustCopyAnchorCertificates(&root_security_items) != errSecSuccess) {
    return false;
  }

  certs.reset(root_security_items);
  return true;
}

bool add_certs_to_x509_store(CFArrayRef certs, X509_STORE *store) {
  auto result = false;
  for (auto i = 0; i < CFArrayGetCount(certs); ++i) {
    const auto cert = reinterpret_cast<const __SecCertificate *>(
        CFArrayGetValueAtIndex(certs, i));

    if (SecCertificateGetTypeID() != CFGetTypeID(cert)) { continue; }

    CFDataRef cert_data = nullptr;
    if (SecItemExport(cert, kSecFormatX509Cert, 0, nullptr, &cert_data) !=
        errSecSuccess) {
      continue;
    }

    CFObjectPtr<CFDataRef> cert_data_ptr(cert_data, cf_object_ptr_deleter);

    auto encoded_cert = static_cast<const unsigned char *>(
        CFDataGetBytePtr(cert_data_ptr.get()));

    auto x509 =
        d2i_X509(NULL, &encoded_cert, CFDataGetLength(cert_data_ptr.get()));

    if (x509) {
      X509_STORE_add_cert(store, x509);
      X509_free(x509);
      result = true;
    }
  }

  return result;
}

bool load_system_certs_on_macos(X509_STORE *store) {
  auto result = false;
  CFObjectPtr<CFArrayRef> certs(nullptr, cf_object_ptr_deleter);
  if (retrieve_certs_from_keychain(certs) && certs) {
    result = add_certs_to_x509_store(certs.get(), store);
  }

  if (retrieve_root_certs_from_keychain(certs) && certs) {
    result = add_certs_to_x509_store(certs.get(), store) || result;
  }

  return result;
}
#endif // TARGET_OS_OSX
#endif // _WIN32
#endif // CPPHTTPLIB_OPENSSL_SUPPORT

#ifdef _WIN32
class WSInit {
public:
  WSInit() {
    WSADATA wsaData;
    if (WSAStartup(0x0002, &wsaData) == 0) is_valid_ = true;
  }

  ~WSInit() {
    if (is_valid_) WSACleanup();
  }

  bool is_valid_ = false;
};

static WSInit wsinit_;
#endif

bool parse_www_authenticate(const Response &res,
                                   std::map<std::string, std::string> &auth,
                                   bool is_proxy) {
  auto auth_key = is_proxy ? "Proxy-Authenticate" : "WWW-Authenticate";
  if (res.has_header(auth_key)) {
    thread_local auto re =
        std::regex(R"~((?:(?:,\s*)?(.+?)=(?:"(.*?)"|([^,]*))))~");
    auto s = res.get_header_value(auth_key);
    auto pos = s.find(' ');
    if (pos != std::string::npos) {
      auto type = s.substr(0, pos);
      if (type == "Basic") {
        return false;
      } else if (type == "Digest") {
        s = s.substr(pos + 1);
        auto beg = std::sregex_iterator(s.begin(), s.end(), re);
        for (auto i = beg; i != std::sregex_iterator(); ++i) {
          const auto &m = *i;
          auto key = s.substr(static_cast<size_t>(m.position(1)),
                              static_cast<size_t>(m.length(1)));
          auto val = m.length(2) > 0
                         ? s.substr(static_cast<size_t>(m.position(2)),
                                    static_cast<size_t>(m.length(2)))
                         : s.substr(static_cast<size_t>(m.position(3)),
                                    static_cast<size_t>(m.length(3)));
          auth[key] = val;
        }
        return true;
      }
    }
  }
  return false;
}

class ContentProviderAdapter {
public:
  explicit ContentProviderAdapter(
      ContentProviderWithoutLength &&content_provider)
      : content_provider_(content_provider) {}

  bool operator()(size_t offset, size_t, DataSink &sink) {
    return content_provider_(offset, sink);
  }

private:
  ContentProviderWithoutLength content_provider_;
};

} // namespace detail

std::string hosted_at(const std::string &hostname) {
  std::vector<std::string> addrs;
  hosted_at(hostname, addrs);
  if (addrs.empty()) { return std::string(); }
  return addrs[0];
}

void hosted_at(const std::string &hostname,
                      std::vector<std::string> &addrs) {
  struct addrinfo hints;
  struct addrinfo *result;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = 0;

  if (getaddrinfo(hostname.c_str(), nullptr, &hints, &result)) {
#if defined __linux__ && !defined __ANDROID__
    res_init();
#endif
    return;
  }
  auto se = detail::scope_exit([&] { freeaddrinfo(result); });

  for (auto rp = result; rp; rp = rp->ai_next) {
    const auto &addr =
        *reinterpret_cast<struct sockaddr_storage *>(rp->ai_addr);
    std::string ip;
    auto dummy = -1;
    if (detail::get_ip_and_port(addr, sizeof(struct sockaddr_storage), ip,
                                dummy)) {
      addrs.push_back(ip);
    }
  }
}

std::string append_query_params(const std::string &path,
                                       const Params &params) {
  std::string path_with_query = path;
  thread_local const std::regex re("[^?]+\\?.*");
  auto delm = std::regex_match(path, re) ? '&' : '?';
  path_with_query += delm + detail::params_to_query_str(params);
  return path_with_query;
}

// Header utilities
std::pair<std::string, std::string>
make_range_header(const Ranges &ranges) {
  std::string field = "bytes=";
  auto i = 0;
  for (const auto &r : ranges) {
    if (i != 0) { field += ", "; }
    if (r.first != -1) { field += std::to_string(r.first); }
    field += '-';
    if (r.second != -1) { field += std::to_string(r.second); }
    i++;
  }
  return std::make_pair("Range", std::move(field));
}

std::pair<std::string, std::string>
make_basic_authentication_header(const std::string &username,
                                 const std::string &password, bool is_proxy) {
  auto field = "Basic " + detail::base64_encode(username + ":" + password);
  auto key = is_proxy ? "Proxy-Authorization" : "Authorization";
  return std::make_pair(key, std::move(field));
}

std::pair<std::string, std::string>
make_bearer_token_authentication_header(const std::string &token,
                                        bool is_proxy = false) {
  auto field = "Bearer " + token;
  auto key = is_proxy ? "Proxy-Authorization" : "Authorization";
  return std::make_pair(key, std::move(field));
}

// Request implementation
bool Request::has_header(const std::string &key) const {
  return detail::has_header(headers, key);
}

std::string Request::get_header_value(const std::string &key,
                                             const char *def, size_t id) const {
  return detail::get_header_value(headers, key, def, id);
}

size_t Request::get_header_value_count(const std::string &key) const {
  auto r = headers.equal_range(key);
  return static_cast<size_t>(std::distance(r.first, r.second));
}

void Request::set_header(const std::string &key,
                                const std::string &val) {
  if (detail::fields::is_field_name(key) &&
      detail::fields::is_field_value(val)) {
    headers.emplace(key, val);
  }
}

bool Request::has_param(const std::string &key) const {
  return params.find(key) != params.end();
}

std::string Request::get_param_value(const std::string &key,
                                            size_t id) const {
  auto rng = params.equal_range(key);
  auto it = rng.first;
  std::advance(it, static_cast<ssize_t>(id));
  if (it != rng.second) { return it->second; }
  return std::string();
}

size_t Request::get_param_value_count(const std::string &key) const {
  auto r = params.equal_range(key);
  return static_cast<size_t>(std::distance(r.first, r.second));
}

bool Request::is_multipart_form_data() const {
  const auto &content_type = get_header_value("Content-Type");
  return !content_type.rfind("multipart/form-data", 0);
}

bool Request::has_file(const std::string &key) const {
  return files.find(key) != files.end();
}

MultipartFormData Request::get_file_value(const std::string &key) const {
  auto it = files.find(key);
  if (it != files.end()) { return it->second; }
  return MultipartFormData();
}

std::vector<MultipartFormData>
Request::get_file_values(const std::string &key) const {
  std::vector<MultipartFormData> values;
  auto rng = files.equal_range(key);
  for (auto it = rng.first; it != rng.second; it++) {
    values.push_back(it->second);
  }
  return values;
}

// Response implementation
bool Response::has_header(const std::string &key) const {
  return headers.find(key) != headers.end();
}

std::string Response::get_header_value(const std::string &key,
                                              const char *def,
                                              size_t id) const {
  return detail::get_header_value(headers, key, def, id);
}

size_t Response::get_header_value_count(const std::string &key) const {
  auto r = headers.equal_range(key);
  return static_cast<size_t>(std::distance(r.first, r.second));
}

void Response::set_header(const std::string &key,
                                 const std::string &val) {
  if (detail::fields::is_field_name(key) &&
      detail::fields::is_field_value(val)) {
    headers.emplace(key, val);
  }
}

void Response::set_redirect(const std::string &url, int stat) {
  if (detail::fields::is_field_value(url)) {
    set_header("Location", url);
    if (300 <= stat && stat < 400) {
      this->status = stat;
    } else {
      this->status = StatusCode::Found_302;
    }
  }
}

void Response::set_content(const char *s, size_t n,
                                  const std::string &content_type) {
  body.assign(s, n);

  auto rng = headers.equal_range("Content-Type");
  headers.erase(rng.first, rng.second);
  set_header("Content-Type", content_type);
}

void Response::set_content(const std::string &s,
                                  const std::string &content_type) {
  set_content(s.data(), s.size(), content_type);
}

void Response::set_content(std::string &&s,
                                  const std::string &content_type) {
  body = std::move(s);

  auto rng = headers.equal_range("Content-Type");
  headers.erase(rng.first, rng.second);
  set_header("Content-Type", content_type);
}

void Response::set_content_provider(
    size_t in_length, const std::string &content_type, ContentProvider provider,
    ContentProviderResourceReleaser resource_releaser) {
  set_header("Content-Type", content_type);
  content_length_ = in_length;
  if (in_length > 0) { content_provider_ = std::move(provider); }
  content_provider_resource_releaser_ = std::move(resource_releaser);
  is_chunked_content_provider_ = false;
}

void Response::set_content_provider(
    const std::string &content_type, ContentProviderWithoutLength provider,
    ContentProviderResourceReleaser resource_releaser) {
  set_header("Content-Type", content_type);
  content_length_ = 0;
  content_provider_ = detail::ContentProviderAdapter(std::move(provider));
  content_provider_resource_releaser_ = std::move(resource_releaser);
  is_chunked_content_provider_ = false;
}

void Response::set_chunked_content_provider(
    const std::string &content_type, ContentProviderWithoutLength provider,
    ContentProviderResourceReleaser resource_releaser) {
  set_header("Content-Type", content_type);
  content_length_ = 0;
  content_provider_ = detail::ContentProviderAdapter(std::move(provider));
  content_provider_resource_releaser_ = std::move(resource_releaser);
  is_chunked_content_provider_ = true;
}

void Response::set_file_content(const std::string &path,
                                       const std::string &content_type) {
  file_content_path_ = path;
  file_content_content_type_ = content_type;
}

void Response::set_file_content(const std::string &path) {
  file_content_path_ = path;
}

// Result implementation
bool Result::has_request_header(const std::string &key) const {
  return request_headers_.find(key) != request_headers_.end();
}

std::string Result::get_request_header_value(const std::string &key,
                                                    const char *def,
                                                    size_t id) const {
  return detail::get_header_value(request_headers_, key, def, id);
}

size_t
Result::get_request_header_value_count(const std::string &key) const {
  auto r = request_headers_.equal_range(key);
  return static_cast<size_t>(std::distance(r.first, r.second));
}

// Stream implementation
ssize_t Stream::write(const char *ptr) {
  return write(ptr, strlen(ptr));
}

ssize_t Stream::write(const std::string &s) {
  return write(s.data(), s.size());
}

namespace detail {

void calc_actual_timeout(time_t max_timeout_msec, time_t duration_msec,
                                time_t timeout_sec, time_t timeout_usec,
                                time_t &actual_timeout_sec,
                                time_t &actual_timeout_usec) {
  auto timeout_msec = (timeout_sec * 1000) + (timeout_usec / 1000);

  auto actual_timeout_msec =
      (std::min)(max_timeout_msec - duration_msec, timeout_msec);

  if (actual_timeout_msec < 0) { actual_timeout_msec = 0; }

  actual_timeout_sec = actual_timeout_msec / 1000;
  actual_timeout_usec = (actual_timeout_msec % 1000) * 1000;
}

// Socket stream implementation
SocketStream::SocketStream(
    socket_t sock, time_t read_timeout_sec, time_t read_timeout_usec,
    time_t write_timeout_sec, time_t write_timeout_usec,
    time_t max_timeout_msec,
    std::chrono::time_point<std::chrono::steady_clock> start_time)
    : sock_(sock), read_timeout_sec_(read_timeout_sec),
      read_timeout_usec_(read_timeout_usec),
      write_timeout_sec_(write_timeout_sec),
      write_timeout_usec_(write_timeout_usec),
      max_timeout_msec_(max_timeout_msec), start_time_(start_time),
      read_buff_(read_buff_size_, 0) {}

SocketStream::~SocketStream() = default;

bool SocketStream::is_readable() const {
  return read_buff_off_ < read_buff_content_size_;
}

bool SocketStream::wait_readable() const {
  if (max_timeout_msec_ <= 0) {
    return select_read(sock_, read_timeout_sec_, read_timeout_usec_) > 0;
  }

  time_t read_timeout_sec;
  time_t read_timeout_usec;
  calc_actual_timeout(max_timeout_msec_, duration(), read_timeout_sec_,
                      read_timeout_usec_, read_timeout_sec, read_timeout_usec);

  return select_read(sock_, read_timeout_sec, read_timeout_usec) > 0;
}

bool SocketStream::wait_writable() const {
  return select_write(sock_, write_timeout_sec_, write_timeout_usec_) > 0 &&
         is_socket_alive(sock_);
}

ssize_t SocketStream::read(char *ptr, size_t size) {
#ifdef _WIN32
  size =
      (std::min)(size, static_cast<size_t>((std::numeric_limits<int>::max)()));
#else
  size = (std::min)(size,
                    static_cast<size_t>((std::numeric_limits<ssize_t>::max)()));
#endif

  if (read_buff_off_ < read_buff_content_size_) {
    auto remaining_size = read_buff_content_size_ - read_buff_off_;
    if (size <= remaining_size) {
      memcpy(ptr, read_buff_.data() + read_buff_off_, size);
      read_buff_off_ += size;
      return static_cast<ssize_t>(size);
    } else {
      memcpy(ptr, read_buff_.data() + read_buff_off_, remaining_size);
      read_buff_off_ += remaining_size;
      return static_cast<ssize_t>(remaining_size);
    }
  }

  if (!wait_readable()) { return -1; }

  read_buff_off_ = 0;
  read_buff_content_size_ = 0;

  if (size < read_buff_size_) {
    auto n = read_socket(sock_, read_buff_.data(), read_buff_size_,
                         CPPHTTPLIB_RECV_FLAGS);
    if (n <= 0) {
      return n;
    } else if (n <= static_cast<ssize_t>(size)) {
      memcpy(ptr, read_buff_.data(), static_cast<size_t>(n));
      return n;
    } else {
      memcpy(ptr, read_buff_.data(), size);
      read_buff_off_ = size;
      read_buff_content_size_ = static_cast<size_t>(n);
      return static_cast<ssize_t>(size);
    }
  } else {
    return read_socket(sock_, ptr, size, CPPHTTPLIB_RECV_FLAGS);
  }
}

ssize_t SocketStream::write(const char *ptr, size_t size) {
  if (!wait_writable()) { return -1; }

#if defined(_WIN32) && !defined(_WIN64)
  size =
      (std::min)(size, static_cast<size_t>((std::numeric_limits<int>::max)()));
#endif

  return send_socket(sock_, ptr, size, CPPHTTPLIB_SEND_FLAGS);
}

void SocketStream::get_remote_ip_and_port(std::string &ip,
                                                 int &port) const {
  return detail::get_remote_ip_and_port(sock_, ip, port);
}

void SocketStream::get_local_ip_and_port(std::string &ip,
                                                int &port) const {
  return detail::get_local_ip_and_port(sock_, ip, port);
}

socket_t SocketStream::socket() const { return sock_; }

time_t SocketStream::duration() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

// Buffer stream implementation
bool BufferStream::is_readable() const { return true; }

bool BufferStream::wait_readable() const { return true; }

bool BufferStream::wait_writable() const { return true; }

ssize_t BufferStream::read(char *ptr, size_t size) {
#if defined(_MSC_VER) && _MSC_VER < 1910
  auto len_read = buffer._Copy_s(ptr, size, size, position);
#else
  auto len_read = buffer.copy(ptr, size, position);
#endif
  position += static_cast<size_t>(len_read);
  return static_cast<ssize_t>(len_read);
}

ssize_t BufferStream::write(const char *ptr, size_t size) {
  buffer.append(ptr, size);
  return static_cast<ssize_t>(size);
}

void BufferStream::get_remote_ip_and_port(std::string & /*ip*/,
                                                 int & /*port*/) const {}

void BufferStream::get_local_ip_and_port(std::string & /*ip*/,
                                                int & /*port*/) const {}

socket_t BufferStream::socket() const { return 0; }

time_t BufferStream::duration() const { return 0; }

const std::string &BufferStream::get_buffer() const { return buffer; }

PathParamsMatcher::PathParamsMatcher(const std::string &pattern) {
  constexpr const char marker[] = "/:";

  // One past the last ending position of a path param substring
  std::size_t last_param_end = 0;

#ifndef CPPHTTPLIB_NO_EXCEPTIONS
  // Needed to ensure that parameter names are unique during matcher
  // construction
  // If exceptions are disabled, only last duplicate path
  // parameter will be set
  std::unordered_set<std::string> param_name_set;
#endif

  while (true) {
    const auto marker_pos = pattern.find(
        marker, last_param_end == 0 ? last_param_end : last_param_end - 1);
    if (marker_pos == std::string::npos) { break; }

    static_fragments_.push_back(
        pattern.substr(last_param_end, marker_pos - last_param_end + 1));

    const auto param_name_start = marker_pos + str_len(marker);

    auto sep_pos = pattern.find(separator, param_name_start);
    if (sep_pos == std::string::npos) { sep_pos = pattern.length(); }

    auto param_name =
        pattern.substr(param_name_start, sep_pos - param_name_start);

#ifndef CPPHTTPLIB_NO_EXCEPTIONS
    if (param_name_set.find(param_name) != param_name_set.cend()) {
      std::string msg = "Encountered path parameter '" + param_name +
                        "' multiple times in route pattern '" + pattern + "'.";
      throw std::invalid_argument(msg);
    }
#endif

    param_names_.push_back(std::move(param_name));

    last_param_end = sep_pos + 1;
  }

  if (last_param_end < pattern.length()) {
    static_fragments_.push_back(pattern.substr(last_param_end));
  }
}

bool PathParamsMatcher::match(Request &request) const {
  request.matches = std::smatch();
  request.path_params.clear();
  request.path_params.reserve(param_names_.size());

  // One past the position at which the path matched the pattern last time
  std::size_t starting_pos = 0;
  for (size_t i = 0; i < static_fragments_.size(); ++i) {
    const auto &fragment = static_fragments_[i];

    if (starting_pos + fragment.length() > request.path.length()) {
      return false;
    }

    // Avoid unnecessary allocation by using strncmp instead of substr +
    // comparison
    if (std::strncmp(request.path.c_str() + starting_pos, fragment.c_str(),
                     fragment.length()) != 0) {
      return false;
    }

    starting_pos += fragment.length();

    // Should only happen when we have a static fragment after a param
    // Example: '/users/:id/subscriptions'
    // The 'subscriptions' fragment here does not have a corresponding param
    if (i >= param_names_.size()) { continue; }

    auto sep_pos = request.path.find(separator, starting_pos);
    if (sep_pos == std::string::npos) { sep_pos = request.path.length(); }

    const auto &param_name = param_names_[i];

    request.path_params.emplace(
        param_name, request.path.substr(starting_pos, sep_pos - starting_pos));

    // Mark everything up to '/' as matched
    starting_pos = sep_pos + 1;
  }
  // Returns false if the path is longer than the pattern
  return starting_pos >= request.path.length();
}

bool RegexMatcher::match(Request &request) const {
  request.path_params.clear();
  return std::regex_match(request.path, request.matches, regex_);
}

} // namespace detail

// HTTP server implementation
Server::Server()
    : new_task_queue(
          [] { return new ThreadPool(CPPHTTPLIB_THREAD_POOL_COUNT); }) {
#ifndef _WIN32
  signal(SIGPIPE, SIG_IGN);
#endif
}

Server::~Server() = default;

std::unique_ptr<detail::MatcherBase>
Server::make_matcher(const std::string &pattern) {
  if (pattern.find("/:") != std::string::npos) {
    return detail::make_unique<detail::PathParamsMatcher>(pattern);
  } else {
    return detail::make_unique<detail::RegexMatcher>(pattern);
  }
}

Server &Server::Get(const std::string &pattern, Handler handler) {
  get_handlers_.emplace_back(make_matcher(pattern), std::move(handler));
  return *this;
}

Server &Server::Post(const std::string &pattern, Handler handler) {
  post_handlers_.emplace_back(make_matcher(pattern), std::move(handler));
  return *this;
}

Server &Server::Post(const std::string &pattern,
                            HandlerWithContentReader handler) {
  post_handlers_for_content_reader_.emplace_back(make_matcher(pattern),
                                                 std::move(handler));
  return *this;
}

Server &Server::Put(const std::string &pattern, Handler handler) {
  put_handlers_.emplace_back(make_matcher(pattern), std::move(handler));
  return *this;
}

Server &Server::Put(const std::string &pattern,
                           HandlerWithContentReader handler) {
  put_handlers_for_content_reader_.emplace_back(make_matcher(pattern),
                                                std::move(handler));
  return *this;
}

Server &Server::Patch(const std::string &pattern, Handler handler) {
  patch_handlers_.emplace_back(make_matcher(pattern), std::move(handler));
  return *this;
}

Server &Server::Patch(const std::string &pattern,
                             HandlerWithContentReader handler) {
  patch_handlers_for_content_reader_.emplace_back(make_matcher(pattern),
                                                  std::move(handler));
  return *this;
}

Server &Server::Delete(const std::string &pattern, Handler handler) {
  delete_handlers_.emplace_back(make_matcher(pattern), std::move(handler));
  return *this;
}

Server &Server::Delete(const std::string &pattern,
                              HandlerWithContentReader handler) {
  delete_handlers_for_content_reader_.emplace_back(make_matcher(pattern),
                                                   std::move(handler));
  return *this;
}

Server &Server::Options(const std::string &pattern, Handler handler) {
  options_handlers_.emplace_back(make_matcher(pattern), std::move(handler));
  return *this;
}

bool Server::set_base_dir(const std::string &dir,
                                 const std::string &mount_point) {
  return set_mount_point(mount_point, dir);
}

bool Server::set_mount_point(const std::string &mount_point,
                                    const std::string &dir, Headers headers) {
  detail::FileStat stat(dir);
  if (stat.is_dir()) {
    std::string mnt = !mount_point.empty() ? mount_point : "/";
    if (!mnt.empty() && mnt[0] == '/') {
      base_dirs_.push_back({mnt, dir, std::move(headers)});
      return true;
    }
  }
  return false;
}

bool Server::remove_mount_point(const std::string &mount_point) {
  for (auto it = base_dirs_.begin(); it != base_dirs_.end(); ++it) {
    if (it->mount_point == mount_point) {
      base_dirs_.erase(it);
      return true;
    }
  }
  return false;
}

Server &
Server::set_file_extension_and_mimetype_mapping(const std::string &ext,
                                                const std::string &mime) {
  file_extension_and_mimetype_map_[ext] = mime;
  return *this;
}

Server &Server::set_default_file_mimetype(const std::string &mime) {
  default_file_mimetype_ = mime;
  return *this;
}

Server &Server::set_file_request_handler(Handler handler) {
  file_request_handler_ = std::move(handler);
  return *this;
}

Server &Server::set_error_handler_core(HandlerWithResponse handler,
                                              std::true_type) {
  error_handler_ = std::move(handler);
  return *this;
}

Server &Server::set_error_handler_core(Handler handler,
                                              std::false_type) {
  error_handler_ = [handler](const Request &req, Response &res) {
    handler(req, res);
    return HandlerResponse::Handled;
  };
  return *this;
}

Server &Server::set_exception_handler(ExceptionHandler handler) {
  exception_handler_ = std::move(handler);
  return *this;
}

Server &Server::set_pre_routing_handler(HandlerWithResponse handler) {
  pre_routing_handler_ = std::move(handler);
  return *this;
}

Server &Server::set_post_routing_handler(Handler handler) {
  post_routing_handler_ = std::move(handler);
  return *this;
}

Server &Server::set_logger(Logger logger) {
  logger_ = std::move(logger);
  return *this;
}

Server &
Server::set_expect_100_continue_handler(Expect100ContinueHandler handler) {
  expect_100_continue_handler_ = std::move(handler);
  return *this;
}

Server &Server::set_address_family(int family) {
  address_family_ = family;
  return *this;
}

Server &Server::set_tcp_nodelay(bool on) {
  tcp_nodelay_ = on;
  return *this;
}

Server &Server::set_ipv6_v6only(bool on) {
  ipv6_v6only_ = on;
  return *this;
}

Server &Server::set_socket_options(SocketOptions socket_options) {
  socket_options_ = std::move(socket_options);
  return *this;
}

Server &Server::set_default_headers(Headers headers) {
  default_headers_ = std::move(headers);
  return *this;
}

Server &Server::set_header_writer(
    std::function<ssize_t(Stream &, Headers &)> const &writer) {
  header_writer_ = writer;
  return *this;
}

Server &Server::set_keep_alive_max_count(size_t count) {
  keep_alive_max_count_ = count;
  return *this;
}

Server &Server::set_keep_alive_timeout(time_t sec) {
  keep_alive_timeout_sec_ = sec;
  return *this;
}

Server &Server::set_read_timeout(time_t sec, time_t usec) {
  read_timeout_sec_ = sec;
  read_timeout_usec_ = usec;
  return *this;
}

Server &Server::set_write_timeout(time_t sec, time_t usec) {
  write_timeout_sec_ = sec;
  write_timeout_usec_ = usec;
  return *this;
}

Server &Server::set_idle_interval(time_t sec, time_t usec) {
  idle_interval_sec_ = sec;
  idle_interval_usec_ = usec;
  return *this;
}

Server &Server::set_payload_max_length(size_t length) {
  payload_max_length_ = length;
  return *this;
}

bool Server::bind_to_port(const std::string &host, int port,
                                 int socket_flags) {
  auto ret = bind_internal(host, port, socket_flags);
  if (ret == -1) { is_decommissioned = true; }
  return ret >= 0;
}
int Server::bind_to_any_port(const std::string &host, int socket_flags) {
  auto ret = bind_internal(host, 0, socket_flags);
  if (ret == -1) { is_decommissioned = true; }
  return ret;
}

bool Server::listen_after_bind() { return listen_internal(); }

bool Server::listen(const std::string &host, int port,
                           int socket_flags) {
  return bind_to_port(host, port, socket_flags) && listen_internal();
}

bool Server::is_running() const { return is_running_; }

void Server::wait_until_ready() const {
  while (!is_running_ && !is_decommissioned) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

void Server::stop() {
  if (is_running_) {
    assert(svr_sock_ != INVALID_SOCKET);
    std::atomic<socket_t> sock(svr_sock_.exchange(INVALID_SOCKET));
    detail::shutdown_socket(sock);
    detail::close_socket(sock);
  }
  is_decommissioned = false;
}

void Server::decommission() { is_decommissioned = true; }

bool Server::parse_request_line(const char *s, Request &req) const {
  auto len = strlen(s);
  if (len < 2 || s[len - 2] != '\r' || s[len - 1] != '\n') { return false; }
  len -= 2;

  {
    size_t count = 0;

    detail::split(s, s + len, ' ', [&](const char *b, const char *e) {
      switch (count) {
      case 0: req.method = std::string(b, e); break;
      case 1: req.target = std::string(b, e); break;
      case 2: req.version = std::string(b, e); break;
      default: break;
      }
      count++;
    });

    if (count != 3) { return false; }
  }

  thread_local const std::set<std::string> methods{
      "GET",     "HEAD",    "POST",  "PUT",   "DELETE",
      "CONNECT", "OPTIONS", "TRACE", "PATCH", "PRI"};

  if (methods.find(req.method) == methods.end()) { return false; }

  if (req.version != "HTTP/1.1" && req.version != "HTTP/1.0") { return false; }

  {
    // Skip URL fragment
    for (size_t i = 0; i < req.target.size(); i++) {
      if (req.target[i] == '#') {
        req.target.erase(i);
        break;
      }
    }

    detail::divide(req.target, '?',
                   [&](const char *lhs_data, std::size_t lhs_size,
                       const char *rhs_data, std::size_t rhs_size) {
                     req.path = detail::decode_url(
                         std::string(lhs_data, lhs_size), false);
                     detail::parse_query_text(rhs_data, rhs_size, req.params);
                   });
  }

  return true;
}

bool Server::write_response(Stream &strm, bool close_connection,
                                   Request &req, Response &res) {
  // NOTE: `req.ranges` should be empty, otherwise it will be applied
  // incorrectly to the error content.
  req.ranges.clear();
  return write_response_core(strm, close_connection, req, res, false);
}

bool Server::write_response_with_content(Stream &strm,
                                                bool close_connection,
                                                const Request &req,
                                                Response &res) {
  return write_response_core(strm, close_connection, req, res, true);
}

bool Server::write_response_core(Stream &strm, bool close_connection,
                                        const Request &req, Response &res,
                                        bool need_apply_ranges) {
  assert(res.status != -1);

  if (400 <= res.status && error_handler_ &&
      error_handler_(req, res) == HandlerResponse::Handled) {
    need_apply_ranges = true;
  }

  std::string content_type;
  std::string boundary;
  if (need_apply_ranges) { apply_ranges(req, res, content_type, boundary); }

  // Prepare additional headers
  if (close_connection || req.get_header_value("Connection") == "close") {
    res.set_header("Connection", "close");
  } else {
    std::string s = "timeout=";
    s += std::to_string(keep_alive_timeout_sec_);
    s += ", max=";
    s += std::to_string(keep_alive_max_count_);
    res.set_header("Keep-Alive", s);
  }

  if ((!res.body.empty() || res.content_length_ > 0 || res.content_provider_) &&
      !res.has_header("Content-Type")) {
    res.set_header("Content-Type", "text/plain");
  }

  if (res.body.empty() && !res.content_length_ && !res.content_provider_ &&
      !res.has_header("Content-Length")) {
    res.set_header("Content-Length", "0");
  }

  if (req.method == "HEAD" && !res.has_header("Accept-Ranges")) {
    res.set_header("Accept-Ranges", "bytes");
  }

  if (post_routing_handler_) { post_routing_handler_(req, res); }

  // Response line and headers
  {
    detail::BufferStream bstrm;
    if (!detail::write_response_line(bstrm, res.status)) { return false; }
    if (!header_writer_(bstrm, res.headers)) { return false; }

    // Flush buffer
    auto &data = bstrm.get_buffer();
    detail::write_data(strm, data.data(), data.size());
  }

  // Body
  auto ret = true;
  if (req.method != "HEAD") {
    if (!res.body.empty()) {
      if (!detail::write_data(strm, res.body.data(), res.body.size())) {
        ret = false;
      }
    } else if (res.content_provider_) {
      if (write_content_with_provider(strm, req, res, boundary, content_type)) {
        res.content_provider_success_ = true;
      } else {
        ret = false;
      }
    }
  }

  // Log
  if (logger_) { logger_(req, res); }

  return ret;
}

bool
Server::write_content_with_provider(Stream &strm, const Request &req,
                                    Response &res, const std::string &boundary,
                                    const std::string &content_type) {
  auto is_shutting_down = [this]() {
    return this->svr_sock_ == INVALID_SOCKET;
  };

  if (res.content_length_ > 0) {
    if (req.ranges.empty()) {
      return detail::write_content(strm, res.content_provider_, 0,
                                   res.content_length_, is_shutting_down);
    } else if (req.ranges.size() == 1) {
      auto offset_and_length = detail::get_range_offset_and_length(
          req.ranges[0], res.content_length_);

      return detail::write_content(strm, res.content_provider_,
                                   offset_and_length.first,
                                   offset_and_length.second, is_shutting_down);
    } else {
      return detail::write_multipart_ranges_data(
          strm, req, res, boundary, content_type, res.content_length_,
          is_shutting_down);
    }
  } else {
    if (res.is_chunked_content_provider_) {
      auto type = detail::encoding_type(req, res);

      std::unique_ptr<detail::compressor> compressor;
      if (type == detail::EncodingType::Gzip) {
#ifdef CPPHTTPLIB_ZLIB_SUPPORT
        compressor = detail::make_unique<detail::gzip_compressor>();
#endif
      } else if (type == detail::EncodingType::Brotli) {
#ifdef CPPHTTPLIB_BROTLI_SUPPORT
        compressor = detail::make_unique<detail::brotli_compressor>();
#endif
      } else if (type == detail::EncodingType::Zstd) {
#ifdef CPPHTTPLIB_ZSTD_SUPPORT
        compressor = detail::make_unique<detail::zstd_compressor>();
#endif
      } else {
        compressor = detail::make_unique<detail::nocompressor>();
      }
      assert(compressor != nullptr);

      return detail::write_content_chunked(strm, res.content_provider_,
                                           is_shutting_down, *compressor);
    } else {
      return detail::write_content_without_length(strm, res.content_provider_,
                                                  is_shutting_down);
    }
  }
}

bool Server::read_content(Stream &strm, Request &req, Response &res) {
  MultipartFormDataMap::iterator cur;
  auto file_count = 0;
  if (read_content_core(
          strm, req, res,
          // Regular
          [&](const char *buf, size_t n) {
            if (req.body.size() + n > req.body.max_size()) { return false; }
            req.body.append(buf, n);
            return true;
          },
          // Multipart
          [&](const MultipartFormData &file) {
            if (file_count++ == CPPHTTPLIB_MULTIPART_FORM_DATA_FILE_MAX_COUNT) {
              return false;
            }
            cur = req.files.emplace(file.name, file);
            return true;
          },
          [&](const char *buf, size_t n) {
            auto &content = cur->second.content;
            if (content.size() + n > content.max_size()) { return false; }
            content.append(buf, n);
            return true;
          })) {
    const auto &content_type = req.get_header_value("Content-Type");
    if (!content_type.find("application/x-www-form-urlencoded")) {
      if (req.body.size() > CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH) {
        res.status = StatusCode::PayloadTooLarge_413; // NOTE: should be 414?
        return false;
      }
      detail::parse_query_text(req.body, req.params);
    }
    return true;
  }
  return false;
}

bool Server::read_content_with_content_receiver(
    Stream &strm, Request &req, Response &res, ContentReceiver receiver,
    MultipartContentHeader multipart_header,
    ContentReceiver multipart_receiver) {
  return read_content_core(strm, req, res, std::move(receiver),
                           std::move(multipart_header),
                           std::move(multipart_receiver));
}

bool
Server::read_content_core(Stream &strm, Request &req, Response &res,
                          ContentReceiver receiver,
                          MultipartContentHeader multipart_header,
                          ContentReceiver multipart_receiver) const {
  detail::MultipartFormDataParser multipart_form_data_parser;
  ContentReceiverWithProgress out;

  if (req.is_multipart_form_data()) {
    const auto &content_type = req.get_header_value("Content-Type");
    std::string boundary;
    if (!detail::parse_multipart_boundary(content_type, boundary)) {
      res.status = StatusCode::BadRequest_400;
      return false;
    }

    multipart_form_data_parser.set_boundary(std::move(boundary));
    out = [&](const char *buf, size_t n, uint64_t /*off*/, uint64_t /*len*/) {
      /* For debug
      size_t pos = 0;
      while (pos < n) {
        auto read_size = (std::min)<size_t>(1, n - pos);
        auto ret = multipart_form_data_parser.parse(
            buf + pos, read_size, multipart_receiver, multipart_header);
        if (!ret) { return false; }
        pos += read_size;
      }
      return true;
      */
      return multipart_form_data_parser.parse(buf, n, multipart_receiver,
                                              multipart_header);
    };
  } else {
    out = [receiver](const char *buf, size_t n, uint64_t /*off*/,
                     uint64_t /*len*/) { return receiver(buf, n); };
  }

  if (req.method == "DELETE" && !req.has_header("Content-Length")) {
    return true;
  }

  if (!detail::read_content(strm, req, payload_max_length_, res.status, nullptr,
                            out, true)) {
    return false;
  }

  if (req.is_multipart_form_data()) {
    if (!multipart_form_data_parser.is_valid()) {
      res.status = StatusCode::BadRequest_400;
      return false;
    }
  }

  return true;
}

bool Server::handle_file_request(const Request &req, Response &res,
                                        bool head) {
  for (const auto &entry : base_dirs_) {
    // Prefix match
    if (!req.path.compare(0, entry.mount_point.size(), entry.mount_point)) {
      std::string sub_path = "/" + req.path.substr(entry.mount_point.size());
      if (detail::is_valid_path(sub_path)) {
        auto path = entry.base_dir + sub_path;
        if (path.back() == '/') { path += "index.html"; }

        detail::FileStat stat(path);

        if (stat.is_dir()) {
          res.set_redirect(sub_path + "/", StatusCode::MovedPermanently_301);
          return true;
        }

        if (stat.is_file()) {
          for (const auto &kv : entry.headers) {
            res.set_header(kv.first, kv.second);
          }

          auto mm = std::make_shared<detail::mmap>(path.c_str());
          if (!mm->is_open()) { return false; }

          res.set_content_provider(
              mm->size(),
              detail::find_content_type(path, file_extension_and_mimetype_map_,
                                        default_file_mimetype_),
              [mm](size_t offset, size_t length, DataSink &sink) -> bool {
                sink.write(mm->data() + offset, length);
                return true;
              });

          if (!head && file_request_handler_) {
            file_request_handler_(req, res);
          }

          return true;
        }
      }
    }
  }
  return false;
}

socket_t
Server::create_server_socket(const std::string &host, int port,
                             int socket_flags,
                             SocketOptions socket_options) const {
  return detail::create_socket(
      host, std::string(), port, address_family_, socket_flags, tcp_nodelay_,
      ipv6_v6only_, std::move(socket_options),
      [](socket_t sock, struct addrinfo &ai, bool & /*quit*/) -> bool {
        if (::bind(sock, ai.ai_addr, static_cast<socklen_t>(ai.ai_addrlen))) {
          return false;
        }
        if (::listen(sock, CPPHTTPLIB_LISTEN_BACKLOG)) { return false; }
        return true;
      });
}

int Server::bind_internal(const std::string &host, int port,
                                 int socket_flags) {
  if (is_decommissioned) { return -1; }

  if (!is_valid()) { return -1; }

  svr_sock_ = create_server_socket(host, port, socket_flags, socket_options_);
  if (svr_sock_ == INVALID_SOCKET) { return -1; }

  if (port == 0) {
    struct sockaddr_storage addr;
    socklen_t addr_len = sizeof(addr);
    if (getsockname(svr_sock_, reinterpret_cast<struct sockaddr *>(&addr),
                    &addr_len) == -1) {
      return -1;
    }
    if (addr.ss_family == AF_INET) {
      return ntohs(reinterpret_cast<struct sockaddr_in *>(&addr)->sin_port);
    } else if (addr.ss_family == AF_INET6) {
      return ntohs(reinterpret_cast<struct sockaddr_in6 *>(&addr)->sin6_port);
    } else {
      return -1;
    }
  } else {
    return port;
  }
}

bool Server::listen_internal() {
  if (is_decommissioned) { return false; }

  auto ret = true;
  is_running_ = true;
  auto se = detail::scope_exit([&]() { is_running_ = false; });

  {
    std::unique_ptr<TaskQueue> task_queue(new_task_queue());

    while (svr_sock_ != INVALID_SOCKET) {
#ifndef _WIN32
      if (idle_interval_sec_ > 0 || idle_interval_usec_ > 0) {
#endif
        auto val = detail::select_read(svr_sock_, idle_interval_sec_,
                                       idle_interval_usec_);
        if (val == 0) { // Timeout
          task_queue->on_idle();
          continue;
        }
#ifndef _WIN32
      }
#endif

#if defined _WIN32
      // sockets connected via WASAccept inherit flags NO_HANDLE_INHERIT,
      // OVERLAPPED
      socket_t sock = WSAAccept(svr_sock_, nullptr, nullptr, nullptr, 0);
#elif defined SOCK_CLOEXEC
      socket_t sock = accept4(svr_sock_, nullptr, nullptr, SOCK_CLOEXEC);
#else
      socket_t sock = accept(svr_sock_, nullptr, nullptr);
#endif

      if (sock == INVALID_SOCKET) {
        if (errno == EMFILE) {
          // The per-process limit of open file descriptors has been reached.
          // Try to accept new connections after a short sleep.
          std::this_thread::sleep_for(std::chrono::microseconds{1});
          continue;
        } else if (errno == EINTR || errno == EAGAIN) {
          continue;
        }
        if (svr_sock_ != INVALID_SOCKET) {
          detail::close_socket(svr_sock_);
          ret = false;
        } else {
          ; // The server socket was closed by user.
        }
        break;
      }

      detail::set_socket_opt_time(sock, SOL_SOCKET, SO_RCVTIMEO,
                                  read_timeout_sec_, read_timeout_usec_);
      detail::set_socket_opt_time(sock, SOL_SOCKET, SO_SNDTIMEO,
                                  write_timeout_sec_, write_timeout_usec_);

      if (!task_queue->enqueue(
              [this, sock]() { process_and_close_socket(sock); })) {
        detail::shutdown_socket(sock);
        detail::close_socket(sock);
      }
    }

    task_queue->shutdown();
  }

  is_decommissioned = !ret;
  return ret;
}

bool Server::routing(Request &req, Response &res, Stream &strm) {
  if (pre_routing_handler_ &&
      pre_routing_handler_(req, res) == HandlerResponse::Handled) {
    return true;
  }

  // File handler
  auto is_head_request = req.method == "HEAD";
  if ((req.method == "GET" || is_head_request) &&
      handle_file_request(req, res, is_head_request)) {
    return true;
  }

  if (detail::expect_content(req)) {
    // Content reader handler
    {
      ContentReader reader(
          [&](ContentReceiver receiver) {
            return read_content_with_content_receiver(
                strm, req, res, std::move(receiver), nullptr, nullptr);
          },
          [&](MultipartContentHeader header, ContentReceiver receiver) {
            return read_content_with_content_receiver(strm, req, res, nullptr,
                                                      std::move(header),
                                                      std::move(receiver));
          });

      if (req.method == "POST") {
        if (dispatch_request_for_content_reader(
                req, res, std::move(reader),
                post_handlers_for_content_reader_)) {
          return true;
        }
      } else if (req.method == "PUT") {
        if (dispatch_request_for_content_reader(
                req, res, std::move(reader),
                put_handlers_for_content_reader_)) {
          return true;
        }
      } else if (req.method == "PATCH") {
        if (dispatch_request_for_content_reader(
                req, res, std::move(reader),
                patch_handlers_for_content_reader_)) {
          return true;
        }
      } else if (req.method == "DELETE") {
        if (dispatch_request_for_content_reader(
                req, res, std::move(reader),
                delete_handlers_for_content_reader_)) {
          return true;
        }
      }
    }

    // Read content into `req.body`
    if (!read_content(strm, req, res)) { return false; }
  }

  // Regular handler
  if (req.method == "GET" || req.method == "HEAD") {
    return dispatch_request(req, res, get_handlers_);
  } else if (req.method == "POST") {
    return dispatch_request(req, res, post_handlers_);
  } else if (req.method == "PUT") {
    return dispatch_request(req, res, put_handlers_);
  } else if (req.method == "DELETE") {
    return dispatch_request(req, res, delete_handlers_);
  } else if (req.method == "OPTIONS") {
    return dispatch_request(req, res, options_handlers_);
  } else if (req.method == "PATCH") {
    return dispatch_request(req, res, patch_handlers_);
  }

  res.status = StatusCode::BadRequest_400;
  return false;
}

bool Server::dispatch_request(Request &req, Response &res,
                                     const Handlers &handlers) const {
  for (const auto &x : handlers) {
    const auto &matcher = x.first;
    const auto &handler = x.second;

    if (matcher->match(req)) {
      handler(req, res);
      return true;
    }
  }
  return false;
}

void Server::apply_ranges(const Request &req, Response &res,
                                 std::string &content_type,
                                 std::string &boundary) const {
  if (req.ranges.size() > 1 && res.status == StatusCode::PartialContent_206) {
    auto it = res.headers.find("Content-Type");
    if (it != res.headers.end()) {
      content_type = it->second;
      res.headers.erase(it);
    }

    boundary = detail::make_multipart_data_boundary();

    res.set_header("Content-Type",
                   "multipart/byteranges; boundary=" + boundary);
  }

  auto type = detail::encoding_type(req, res);

  if (res.body.empty()) {
    if (res.content_length_ > 0) {
      size_t length = 0;
      if (req.ranges.empty() || res.status != StatusCode::PartialContent_206) {
        length = res.content_length_;
      } else if (req.ranges.size() == 1) {
        auto offset_and_length = detail::get_range_offset_and_length(
            req.ranges[0], res.content_length_);

        length = offset_and_length.second;

        auto content_range = detail::make_content_range_header_field(
            offset_and_length, res.content_length_);
        res.set_header("Content-Range", content_range);
      } else {
        length = detail::get_multipart_ranges_data_length(
            req, boundary, content_type, res.content_length_);
      }
      res.set_header("Content-Length", std::to_string(length));
    } else {
      if (res.content_provider_) {
        if (res.is_chunked_content_provider_) {
          res.set_header("Transfer-Encoding", "chunked");
          if (type == detail::EncodingType::Gzip) {
            res.set_header("Content-Encoding", "gzip");
          } else if (type == detail::EncodingType::Brotli) {
            res.set_header("Content-Encoding", "br");
          } else if (type == detail::EncodingType::Zstd) {
            res.set_header("Content-Encoding", "zstd");
          }
        }
      }
    }
  } else {
    if (req.ranges.empty() || res.status != StatusCode::PartialContent_206) {
      ;
    } else if (req.ranges.size() == 1) {
      auto offset_and_length =
          detail::get_range_offset_and_length(req.ranges[0], res.body.size());
      auto offset = offset_and_length.first;
      auto length = offset_and_length.second;

      auto content_range = detail::make_content_range_header_field(
          offset_and_length, res.body.size());
      res.set_header("Content-Range", content_range);

      assert(offset + length <= res.body.size());
      res.body = res.body.substr(offset, length);
    } else {
      std::string data;
      detail::make_multipart_ranges_data(req, res, boundary, content_type,
                                         res.body.size(), data);
      res.body.swap(data);
    }

    if (type != detail::EncodingType::None) {
      std::unique_ptr<detail::compressor> compressor;
      std::string content_encoding;

      if (type == detail::EncodingType::Gzip) {
#ifdef CPPHTTPLIB_ZLIB_SUPPORT
        compressor = detail::make_unique<detail::gzip_compressor>();
        content_encoding = "gzip";
#endif
      } else if (type == detail::EncodingType::Brotli) {
#ifdef CPPHTTPLIB_BROTLI_SUPPORT
        compressor = detail::make_unique<detail::brotli_compressor>();
        content_encoding = "br";
#endif
      } else if (type == detail::EncodingType::Zstd) {
#ifdef CPPHTTPLIB_ZSTD_SUPPORT
        compressor = detail::make_unique<detail::zstd_compressor>();
        content_encoding = "zstd";
#endif
      }

      if (compressor) {
        std::string compressed;
        if (compressor->compress(res.body.data(), res.body.size(), true,
                                 [&](const char *data, size_t data_len) {
                                   compressed.append(data, data_len);
                                   return true;
                                 })) {
          res.body.swap(compressed);
          res.set_header("Content-Encoding", content_encoding);
        }
      }
    }

    auto length = std::to_string(res.body.size());
    res.set_header("Content-Length", length);
  }
}

bool Server::dispatch_request_for_content_reader(
    Request &req, Response &res, ContentReader content_reader,
    const HandlersForContentReader &handlers) const {
  for (const auto &x : handlers) {
    const auto &matcher = x.first;
    const auto &handler = x.second;

    if (matcher->match(req)) {
      handler(req, res, content_reader);
      return true;
    }
  }
  return false;
}

bool
Server::process_request(Stream &strm, const std::string &remote_addr,
                        int remote_port, const std::string &local_addr,
                        int local_port, bool close_connection,
                        bool &connection_closed,
                        const std::function<void(Request &)> &setup_request) {
  std::array<char, 2048> buf{};

  detail::stream_line_reader line_reader(strm, buf.data(), buf.size());

  // Connection has been closed on client
  if (!line_reader.getline()) { return false; }

  Request req;

  Response res;
  res.version = "HTTP/1.1";
  res.headers = default_headers_;

  // Request line and headers
  if (!parse_request_line(line_reader.ptr(), req) ||
      !detail::read_headers(strm, req.headers)) {
    res.status = StatusCode::BadRequest_400;
    return write_response(strm, close_connection, req, res);
  }

  // Check if the request URI doesn't exceed the limit
  if (req.target.size() > CPPHTTPLIB_REQUEST_URI_MAX_LENGTH) {
    Headers dummy;
    detail::read_headers(strm, dummy);
    res.status = StatusCode::UriTooLong_414;
    return write_response(strm, close_connection, req, res);
  }

  if (req.get_header_value("Connection") == "close") {
    connection_closed = true;
  }

  if (req.version == "HTTP/1.0" &&
      req.get_header_value("Connection") != "Keep-Alive") {
    connection_closed = true;
  }

  req.remote_addr = remote_addr;
  req.remote_port = remote_port;
  req.set_header("REMOTE_ADDR", req.remote_addr);
  req.set_header("REMOTE_PORT", std::to_string(req.remote_port));

  req.local_addr = local_addr;
  req.local_port = local_port;
  req.set_header("LOCAL_ADDR", req.local_addr);
  req.set_header("LOCAL_PORT", std::to_string(req.local_port));

  if (req.has_header("Range")) {
    const auto &range_header_value = req.get_header_value("Range");
    if (!detail::parse_range_header(range_header_value, req.ranges)) {
      res.status = StatusCode::RangeNotSatisfiable_416;
      return write_response(strm, close_connection, req, res);
    }
  }

  if (setup_request) { setup_request(req); }

  if (req.get_header_value("Expect") == "100-continue") {
    int status = StatusCode::Continue_100;
    if (expect_100_continue_handler_) {
      status = expect_100_continue_handler_(req, res);
    }
    switch (status) {
    case StatusCode::Continue_100:
    case StatusCode::ExpectationFailed_417:
      detail::write_response_line(strm, status);
      strm.write("\r\n");
      break;
    default:
      connection_closed = true;
      return write_response(strm, true, req, res);
    }
  }

  // Setup `is_connection_closed` method
  auto sock = strm.socket();
  req.is_connection_closed = [sock]() {
    return !detail::is_socket_alive(sock);
  };

  // Routing
  auto routed = false;
#ifdef CPPHTTPLIB_NO_EXCEPTIONS
  routed = routing(req, res, strm);
#else
  try {
    routed = routing(req, res, strm);
  } catch (std::exception &e) {
    if (exception_handler_) {
      auto ep = std::current_exception();
      exception_handler_(req, res, ep);
      routed = true;
    } else {
      res.status = StatusCode::InternalServerError_500;
      std::string val;
      auto s = e.what();
      for (size_t i = 0; s[i]; i++) {
        switch (s[i]) {
        case '\r': val += "\\r"; break;
        case '\n': val += "\\n"; break;
        default: val += s[i]; break;
        }
      }
      res.set_header("EXCEPTION_WHAT", val);
    }
  } catch (...) {
    if (exception_handler_) {
      auto ep = std::current_exception();
      exception_handler_(req, res, ep);
      routed = true;
    } else {
      res.status = StatusCode::InternalServerError_500;
      res.set_header("EXCEPTION_WHAT", "UNKNOWN");
    }
  }
#endif
  if (routed) {
    if (res.status == -1) {
      res.status = req.ranges.empty() ? StatusCode::OK_200
                                      : StatusCode::PartialContent_206;
    }

    // Serve file content by using a content provider
    if (!res.file_content_path_.empty()) {
      const auto &path = res.file_content_path_;
      auto mm = std::make_shared<detail::mmap>(path.c_str());
      if (!mm->is_open()) {
        res.body.clear();
        res.content_length_ = 0;
        res.content_provider_ = nullptr;
        res.status = StatusCode::NotFound_404;
        return write_response(strm, close_connection, req, res);
      }

      auto content_type = res.file_content_content_type_;
      if (content_type.empty()) {
        content_type = detail::find_content_type(
            path, file_extension_and_mimetype_map_, default_file_mimetype_);
      }

      res.set_content_provider(
          mm->size(), content_type,
          [mm](size_t offset, size_t length, DataSink &sink) -> bool {
            sink.write(mm->data() + offset, length);
            return true;
          });
    }

    if (detail::range_error(req, res)) {
      res.body.clear();
      res.content_length_ = 0;
      res.content_provider_ = nullptr;
      res.status = StatusCode::RangeNotSatisfiable_416;
      return write_response(strm, close_connection, req, res);
    }

    return write_response_with_content(strm, close_connection, req, res);
  } else {
    if (res.status == -1) { res.status = StatusCode::NotFound_404; }

    return write_response(strm, close_connection, req, res);
  }
}

bool Server::is_valid() const { return true; }

bool Server::process_and_close_socket(socket_t sock) {
  std::string remote_addr;
  int remote_port = 0;
  detail::get_remote_ip_and_port(sock, remote_addr, remote_port);

  std::string local_addr;
  int local_port = 0;
  detail::get_local_ip_and_port(sock, local_addr, local_port);

  auto ret = detail::process_server_socket(
      svr_sock_, sock, keep_alive_max_count_, keep_alive_timeout_sec_,
      read_timeout_sec_, read_timeout_usec_, write_timeout_sec_,
      write_timeout_usec_,
      [&](Stream &strm, bool close_connection, bool &connection_closed) {
        return process_request(strm, remote_addr, remote_port, local_addr,
                               local_port, close_connection, connection_closed,
                               nullptr);
      });

  detail::shutdown_socket(sock);
  detail::close_socket(sock);
  return ret;
}

// HTTP client implementation
ClientImpl::ClientImpl(const std::string &host)
    : ClientImpl(host, 80, std::string(), std::string()) {}

ClientImpl::ClientImpl(const std::string &host, int port)
    : ClientImpl(host, port, std::string(), std::string()) {}

ClientImpl::ClientImpl(const std::string &host, int port,
                              const std::string &client_cert_path,
                              const std::string &client_key_path)
    : host_(detail::escape_abstract_namespace_unix_domain(host)), port_(port),
      host_and_port_(adjust_host_string(host_) + ":" + std::to_string(port)),
      client_cert_path_(client_cert_path), client_key_path_(client_key_path) {}

ClientImpl::~ClientImpl() {
  // Wait until all the requests in flight are handled.
  size_t retry_count = 10;
  while (retry_count-- > 0) {
    {
      std::lock_guard<std::mutex> guard(socket_mutex_);
      if (socket_requests_in_flight_ == 0) { break; }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }

  std::lock_guard<std::mutex> guard(socket_mutex_);
  shutdown_socket(socket_);
  close_socket(socket_);
}

bool ClientImpl::is_valid() const { return true; }

void ClientImpl::copy_settings(const ClientImpl &rhs) {
  client_cert_path_ = rhs.client_cert_path_;
  client_key_path_ = rhs.client_key_path_;
  connection_timeout_sec_ = rhs.connection_timeout_sec_;
  read_timeout_sec_ = rhs.read_timeout_sec_;
  read_timeout_usec_ = rhs.read_timeout_usec_;
  write_timeout_sec_ = rhs.write_timeout_sec_;
  write_timeout_usec_ = rhs.write_timeout_usec_;
  max_timeout_msec_ = rhs.max_timeout_msec_;
  basic_auth_username_ = rhs.basic_auth_username_;
  basic_auth_password_ = rhs.basic_auth_password_;
  bearer_token_auth_token_ = rhs.bearer_token_auth_token_;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  digest_auth_username_ = rhs.digest_auth_username_;
  digest_auth_password_ = rhs.digest_auth_password_;
#endif
  keep_alive_ = rhs.keep_alive_;
  follow_location_ = rhs.follow_location_;
  url_encode_ = rhs.url_encode_;
  address_family_ = rhs.address_family_;
  tcp_nodelay_ = rhs.tcp_nodelay_;
  ipv6_v6only_ = rhs.ipv6_v6only_;
  socket_options_ = rhs.socket_options_;
  compress_ = rhs.compress_;
  decompress_ = rhs.decompress_;
  interface_ = rhs.interface_;
  proxy_host_ = rhs.proxy_host_;
  proxy_port_ = rhs.proxy_port_;
  proxy_basic_auth_username_ = rhs.proxy_basic_auth_username_;
  proxy_basic_auth_password_ = rhs.proxy_basic_auth_password_;
  proxy_bearer_token_auth_token_ = rhs.proxy_bearer_token_auth_token_;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  proxy_digest_auth_username_ = rhs.proxy_digest_auth_username_;
  proxy_digest_auth_password_ = rhs.proxy_digest_auth_password_;
#endif
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  ca_cert_file_path_ = rhs.ca_cert_file_path_;
  ca_cert_dir_path_ = rhs.ca_cert_dir_path_;
  ca_cert_store_ = rhs.ca_cert_store_;
#endif
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  server_certificate_verification_ = rhs.server_certificate_verification_;
  server_hostname_verification_ = rhs.server_hostname_verification_;
  server_certificate_verifier_ = rhs.server_certificate_verifier_;
#endif
  logger_ = rhs.logger_;
}

socket_t ClientImpl::create_client_socket(Error &error) const {
  if (!proxy_host_.empty() && proxy_port_ != -1) {
    return detail::create_client_socket(
        proxy_host_, std::string(), proxy_port_, address_family_, tcp_nodelay_,
        ipv6_v6only_, socket_options_, connection_timeout_sec_,
        connection_timeout_usec_, read_timeout_sec_, read_timeout_usec_,
        write_timeout_sec_, write_timeout_usec_, interface_, error);
  }

  // Check is custom IP specified for host_
  std::string ip;
  auto it = addr_map_.find(host_);
  if (it != addr_map_.end()) { ip = it->second; }

  return detail::create_client_socket(
      host_, ip, port_, address_family_, tcp_nodelay_, ipv6_v6only_,
      socket_options_, connection_timeout_sec_, connection_timeout_usec_,
      read_timeout_sec_, read_timeout_usec_, write_timeout_sec_,
      write_timeout_usec_, interface_, error);
}

bool ClientImpl::create_and_connect_socket(Socket &socket,
                                                  Error &error) {
  auto sock = create_client_socket(error);
  if (sock == INVALID_SOCKET) { return false; }
  socket.sock = sock;
  return true;
}

void ClientImpl::shutdown_ssl(Socket & /*socket*/,
                                     bool /*shutdown_gracefully*/) {
  // If there are any requests in flight from threads other than us, then it's
  // a thread-unsafe race because individual ssl* objects are not thread-safe.
  assert(socket_requests_in_flight_ == 0 ||
         socket_requests_are_from_thread_ == std::this_thread::get_id());
}

void ClientImpl::shutdown_socket(Socket &socket) const {
  if (socket.sock == INVALID_SOCKET) { return; }
  detail::shutdown_socket(socket.sock);
}

void ClientImpl::close_socket(Socket &socket) {
  // If there are requests in flight in another thread, usually closing
  // the socket will be fine and they will simply receive an error when
  // using the closed socket, but it is still a bug since rarely the OS
  // may reassign the socket id to be used for a new socket, and then
  // suddenly they will be operating on a live socket that is different
  // than the one they intended!
  assert(socket_requests_in_flight_ == 0 ||
         socket_requests_are_from_thread_ == std::this_thread::get_id());

  // It is also a bug if this happens while SSL is still active
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  assert(socket.ssl == nullptr);
#endif
  if (socket.sock == INVALID_SOCKET) { return; }
  detail::close_socket(socket.sock);
  socket.sock = INVALID_SOCKET;
}

bool ClientImpl::read_response_line(Stream &strm, const Request &req,
                                           Response &res) const {
  std::array<char, 2048> buf{};

  detail::stream_line_reader line_reader(strm, buf.data(), buf.size());

  if (!line_reader.getline()) { return false; }

#ifdef CPPHTTPLIB_ALLOW_LF_AS_LINE_TERMINATOR
  thread_local const std::regex re("(HTTP/1\\.[01]) (\\d{3})(?: (.*?))?\r?\n");
#else
  thread_local const std::regex re("(HTTP/1\\.[01]) (\\d{3})(?: (.*?))?\r\n");
#endif

  std::cmatch m;
  if (!std::regex_match(line_reader.ptr(), m, re)) {
    return req.method == "CONNECT";
  }
  res.version = std::string(m[1]);
  res.status = std::stoi(std::string(m[2]));
  res.reason = std::string(m[3]);

  // Ignore '100 Continue'
  while (res.status == StatusCode::Continue_100) {
    if (!line_reader.getline()) { return false; } // CRLF
    if (!line_reader.getline()) { return false; } // next response line

    if (!std::regex_match(line_reader.ptr(), m, re)) { return false; }
    res.version = std::string(m[1]);
    res.status = std::stoi(std::string(m[2]));
    res.reason = std::string(m[3]);
  }

  return true;
}

bool ClientImpl::send(Request &req, Response &res, Error &error) {
  std::lock_guard<std::recursive_mutex> request_mutex_guard(request_mutex_);
  auto ret = send_(req, res, error);
  if (error == Error::SSLPeerCouldBeClosed_) {
    assert(!ret);
    ret = send_(req, res, error);
  }
  return ret;
}

bool ClientImpl::send_(Request &req, Response &res, Error &error) {
  {
    std::lock_guard<std::mutex> guard(socket_mutex_);

    // Set this to false immediately - if it ever gets set to true by the end of
    // the request, we know another thread instructed us to close the socket.
    socket_should_be_closed_when_request_is_done_ = false;

    auto is_alive = false;
    if (socket_.is_open()) {
      is_alive = detail::is_socket_alive(socket_.sock);

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
      if (is_alive && is_ssl()) {
        if (detail::is_ssl_peer_could_be_closed(socket_.ssl, socket_.sock)) {
          is_alive = false;
        }
      }
#endif

      if (!is_alive) {
        // Attempt to avoid sigpipe by shutting down non-gracefully if it seems
        // like the other side has already closed the connection Also, there
        // cannot be any requests in flight from other threads since we locked
        // request_mutex_, so safe to close everything immediately
        const bool shutdown_gracefully = false;
        shutdown_ssl(socket_, shutdown_gracefully);
        shutdown_socket(socket_);
        close_socket(socket_);
      }
    }

    if (!is_alive) {
      if (!create_and_connect_socket(socket_, error)) { return false; }

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
      // TODO: refactoring
      if (is_ssl()) {
        auto &scli = static_cast<SSLClient &>(*this);
        if (!proxy_host_.empty() && proxy_port_ != -1) {
          auto success = false;
          if (!scli.connect_with_proxy(socket_, req.start_time_, res, success,
                                       error)) {
            return success;
          }
        }

        if (!scli.initialize_ssl(socket_, error)) { return false; }
      }
#endif
    }

    // Mark the current socket as being in use so that it cannot be closed by
    // anyone else while this request is ongoing, even though we will be
    // releasing the mutex.
    if (socket_requests_in_flight_ > 1) {
      assert(socket_requests_are_from_thread_ == std::this_thread::get_id());
    }
    socket_requests_in_flight_ += 1;
    socket_requests_are_from_thread_ = std::this_thread::get_id();
  }

  for (const auto &header : default_headers_) {
    if (req.headers.find(header.first) == req.headers.end()) {
      req.headers.insert(header);
    }
  }

  auto ret = false;
  auto close_connection = !keep_alive_;

  auto se = detail::scope_exit([&]() {
    // Briefly lock mutex in order to mark that a request is no longer ongoing
    std::lock_guard<std::mutex> guard(socket_mutex_);
    socket_requests_in_flight_ -= 1;
    if (socket_requests_in_flight_ <= 0) {
      assert(socket_requests_in_flight_ == 0);
      socket_requests_are_from_thread_ = std::thread::id();
    }

    if (socket_should_be_closed_when_request_is_done_ || close_connection ||
        !ret) {
      shutdown_ssl(socket_, true);
      shutdown_socket(socket_);
      close_socket(socket_);
    }
  });

  ret = process_socket(socket_, req.start_time_, [&](Stream &strm) {
    return handle_request(strm, req, res, close_connection, error);
  });

  if (!ret) {
    if (error == Error::Success) { error = Error::Unknown; }
  }

  return ret;
}

Result ClientImpl::send(const Request &req) {
  auto req2 = req;
  return send_(std::move(req2));
}

Result ClientImpl::send_(Request &&req) {
  auto res = detail::make_unique<Response>();
  auto error = Error::Success;
  auto ret = send(req, *res, error);
  return Result{ret ? std::move(res) : nullptr, error, std::move(req.headers)};
}

bool ClientImpl::handle_request(Stream &strm, Request &req,
                                       Response &res, bool close_connection,
                                       Error &error) {
  if (req.path.empty()) {
    error = Error::Connection;
    return false;
  }

  auto req_save = req;

  bool ret;

  if (!is_ssl() && !proxy_host_.empty() && proxy_port_ != -1) {
    auto req2 = req;
    req2.path = "http://" + host_and_port_ + req.path;
    ret = process_request(strm, req2, res, close_connection, error);
    req = req2;
    req.path = req_save.path;
  } else {
    ret = process_request(strm, req, res, close_connection, error);
  }

  if (!ret) { return false; }

  if (res.get_header_value("Connection") == "close" ||
      (res.version == "HTTP/1.0" && res.reason != "Connection established")) {
    // TODO this requires a not-entirely-obvious chain of calls to be correct
    // for this to be safe.

    // This is safe to call because handle_request is only called by send_
    // which locks the request mutex during the process. It would be a bug
    // to call it from a different thread since it's a thread-safety issue
    // to do these things to the socket if another thread is using the socket.
    std::lock_guard<std::mutex> guard(socket_mutex_);
    shutdown_ssl(socket_, true);
    shutdown_socket(socket_);
    close_socket(socket_);
  }

  if (300 < res.status && res.status < 400 && follow_location_) {
    req = req_save;
    ret = redirect(req, res, error);
  }

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  if ((res.status == StatusCode::Unauthorized_401 ||
       res.status == StatusCode::ProxyAuthenticationRequired_407) &&
      req.authorization_count_ < 5) {
    auto is_proxy = res.status == StatusCode::ProxyAuthenticationRequired_407;
    const auto &username =
        is_proxy ? proxy_digest_auth_username_ : digest_auth_username_;
    const auto &password =
        is_proxy ? proxy_digest_auth_password_ : digest_auth_password_;

    if (!username.empty() && !password.empty()) {
      std::map<std::string, std::string> auth;
      if (detail::parse_www_authenticate(res, auth, is_proxy)) {
        Request new_req = req;
        new_req.authorization_count_ += 1;
        new_req.headers.erase(is_proxy ? "Proxy-Authorization"
                                       : "Authorization");
        new_req.headers.insert(detail::make_digest_authentication_header(
            req, auth, new_req.authorization_count_, detail::random_string(10),
            username, password, is_proxy));

        Response new_res;

        ret = send(new_req, new_res, error);
        if (ret) { res = new_res; }
      }
    }
  }
#endif

  return ret;
}

bool ClientImpl::redirect(Request &req, Response &res, Error &error) {
  if (req.redirect_count_ == 0) {
    error = Error::ExceedRedirectCount;
    return false;
  }

  auto location = res.get_header_value("location");
  if (location.empty()) { return false; }

  thread_local const std::regex re(
      R"((?:(https?):)?(?://(?:\[([a-fA-F\d:]+)\]|([^:/?#]+))(?::(\d+))?)?([^?#]*)(\?[^#]*)?(?:#.*)?)");

  std::smatch m;
  if (!std::regex_match(location, m, re)) { return false; }

  auto scheme = is_ssl() ? "https" : "http";

  auto next_scheme = m[1].str();
  auto next_host = m[2].str();
  if (next_host.empty()) { next_host = m[3].str(); }
  auto port_str = m[4].str();
  auto next_path = m[5].str();
  auto next_query = m[6].str();

  auto next_port = port_;
  if (!port_str.empty()) {
    next_port = std::stoi(port_str);
  } else if (!next_scheme.empty()) {
    next_port = next_scheme == "https" ? 443 : 80;
  }

  if (next_scheme.empty()) { next_scheme = scheme; }
  if (next_host.empty()) { next_host = host_; }
  if (next_path.empty()) { next_path = "/"; }

  auto path = detail::decode_url(next_path, true) + next_query;

  if (next_scheme == scheme && next_host == host_ && next_port == port_) {
    return detail::redirect(*this, req, res, path, location, error);
  } else {
    if (next_scheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
      SSLClient cli(next_host, next_port);
      cli.copy_settings(*this);
      if (ca_cert_store_) { cli.set_ca_cert_store(ca_cert_store_); }
      return detail::redirect(cli, req, res, path, location, error);
#else
      return false;
#endif
    } else {
      ClientImpl cli(next_host, next_port);
      cli.copy_settings(*this);
      return detail::redirect(cli, req, res, path, location, error);
    }
  }
}

bool ClientImpl::write_content_with_provider(Stream &strm,
                                                    const Request &req,
                                                    Error &error) const {
  auto is_shutting_down = []() { return false; };

  if (req.is_chunked_content_provider_) {
    // TODO: Brotli support
    std::unique_ptr<detail::compressor> compressor;
#ifdef CPPHTTPLIB_ZLIB_SUPPORT
    if (compress_) {
      compressor = detail::make_unique<detail::gzip_compressor>();
    } else
#endif
    {
      compressor = detail::make_unique<detail::nocompressor>();
    }

    return detail::write_content_chunked(strm, req.content_provider_,
                                         is_shutting_down, *compressor, error);
  } else {
    return detail::write_content(strm, req.content_provider_, 0,
                                 req.content_length_, is_shutting_down, error);
  }
}

bool ClientImpl::write_request(Stream &strm, Request &req,
                                      bool close_connection, Error &error) {
  // Prepare additional headers
  if (close_connection) {
    if (!req.has_header("Connection")) {
      req.set_header("Connection", "close");
    }
  }

  if (!req.has_header("Host")) {
    if (is_ssl()) {
      if (port_ == 443) {
        req.set_header("Host", host_);
      } else {
        req.set_header("Host", host_and_port_);
      }
    } else {
      if (port_ == 80) {
        req.set_header("Host", host_);
      } else {
        req.set_header("Host", host_and_port_);
      }
    }
  }

  if (!req.has_header("Accept")) { req.set_header("Accept", "*/*"); }

  if (!req.content_receiver) {
    if (!req.has_header("Accept-Encoding")) {
      std::string accept_encoding;
#ifdef CPPHTTPLIB_BROTLI_SUPPORT
      accept_encoding = "br";
#endif
#ifdef CPPHTTPLIB_ZLIB_SUPPORT
      if (!accept_encoding.empty()) { accept_encoding += ", "; }
      accept_encoding += "gzip, deflate";
#endif
#ifdef CPPHTTPLIB_ZSTD_SUPPORT
      if (!accept_encoding.empty()) { accept_encoding += ", "; }
      accept_encoding += "zstd";
#endif
      req.set_header("Accept-Encoding", accept_encoding);
    }

#ifndef CPPHTTPLIB_NO_DEFAULT_USER_AGENT
    if (!req.has_header("User-Agent")) {
      auto agent = std::string("cpp-httplib/") + CPPHTTPLIB_VERSION;
      req.set_header("User-Agent", agent);
    }
#endif
  };

  if (req.body.empty()) {
    if (req.content_provider_) {
      if (!req.is_chunked_content_provider_) {
        if (!req.has_header("Content-Length")) {
          auto length = std::to_string(req.content_length_);
          req.set_header("Content-Length", length);
        }
      }
    } else {
      if (req.method == "POST" || req.method == "PUT" ||
          req.method == "PATCH") {
        req.set_header("Content-Length", "0");
      }
    }
  } else {
    if (!req.has_header("Content-Type")) {
      req.set_header("Content-Type", "text/plain");
    }

    if (!req.has_header("Content-Length")) {
      auto length = std::to_string(req.body.size());
      req.set_header("Content-Length", length);
    }
  }

  if (!basic_auth_password_.empty() || !basic_auth_username_.empty()) {
    if (!req.has_header("Authorization")) {
      req.headers.insert(make_basic_authentication_header(
          basic_auth_username_, basic_auth_password_, false));
    }
  }

  if (!proxy_basic_auth_username_.empty() &&
      !proxy_basic_auth_password_.empty()) {
    if (!req.has_header("Proxy-Authorization")) {
      req.headers.insert(make_basic_authentication_header(
          proxy_basic_auth_username_, proxy_basic_auth_password_, true));
    }
  }

  if (!bearer_token_auth_token_.empty()) {
    if (!req.has_header("Authorization")) {
      req.headers.insert(make_bearer_token_authentication_header(
          bearer_token_auth_token_, false));
    }
  }

  if (!proxy_bearer_token_auth_token_.empty()) {
    if (!req.has_header("Proxy-Authorization")) {
      req.headers.insert(make_bearer_token_authentication_header(
          proxy_bearer_token_auth_token_, true));
    }
  }

  // Request line and headers
  {
    detail::BufferStream bstrm;

    const auto &path_with_query =
        req.params.empty() ? req.path
                           : append_query_params(req.path, req.params);

    const auto &path =
        url_encode_ ? detail::encode_url(path_with_query) : path_with_query;

    detail::write_request_line(bstrm, req.method, path);

    header_writer_(bstrm, req.headers);

    // Flush buffer
    auto &data = bstrm.get_buffer();
    if (!detail::write_data(strm, data.data(), data.size())) {
      error = Error::Write;
      return false;
    }
  }

  // Body
  if (req.body.empty()) {
    return write_content_with_provider(strm, req, error);
  }

  if (!detail::write_data(strm, req.body.data(), req.body.size())) {
    error = Error::Write;
    return false;
  }

  return true;
}

std::unique_ptr<Response> ClientImpl::send_with_content_provider(
    Request &req, const char *body, size_t content_length,
    ContentProvider content_provider,
    ContentProviderWithoutLength content_provider_without_length,
    const std::string &content_type, Error &error) {
  if (!content_type.empty()) { req.set_header("Content-Type", content_type); }

#ifdef CPPHTTPLIB_ZLIB_SUPPORT
  if (compress_) { req.set_header("Content-Encoding", "gzip"); }
#endif

#ifdef CPPHTTPLIB_ZLIB_SUPPORT
  if (compress_ && !content_provider_without_length) {
    // TODO: Brotli support
    detail::gzip_compressor compressor;

    if (content_provider) {
      auto ok = true;
      size_t offset = 0;
      DataSink data_sink;

      data_sink.write = [&](const char *data, size_t data_len) -> bool {
        if (ok) {
          auto last = offset + data_len == content_length;

          auto ret = compressor.compress(
              data, data_len, last,
              [&](const char *compressed_data, size_t compressed_data_len) {
                req.body.append(compressed_data, compressed_data_len);
                return true;
              });

          if (ret) {
            offset += data_len;
          } else {
            ok = false;
          }
        }
        return ok;
      };

      while (ok && offset < content_length) {
        if (!content_provider(offset, content_length - offset, data_sink)) {
          error = Error::Canceled;
          return nullptr;
        }
      }
    } else {
      if (!compressor.compress(body, content_length, true,
                               [&](const char *data, size_t data_len) {
                                 req.body.append(data, data_len);
                                 return true;
                               })) {
        error = Error::Compression;
        return nullptr;
      }
    }
  } else
#endif
  {
    if (content_provider) {
      req.content_length_ = content_length;
      req.content_provider_ = std::move(content_provider);
      req.is_chunked_content_provider_ = false;
    } else if (content_provider_without_length) {
      req.content_length_ = 0;
      req.content_provider_ = detail::ContentProviderAdapter(
          std::move(content_provider_without_length));
      req.is_chunked_content_provider_ = true;
      req.set_header("Transfer-Encoding", "chunked");
    } else {
      req.body.assign(body, content_length);
    }
  }

  auto res = detail::make_unique<Response>();
  return send(req, *res, error) ? std::move(res) : nullptr;
}

Result ClientImpl::send_with_content_provider(
    const std::string &method, const std::string &path, const Headers &headers,
    const char *body, size_t content_length, ContentProvider content_provider,
    ContentProviderWithoutLength content_provider_without_length,
    const std::string &content_type, Progress progress) {
  Request req;
  req.method = method;
  req.headers = headers;
  req.path = path;
  req.progress = progress;
  if (max_timeout_msec_ > 0) {
    req.start_time_ = std::chrono::steady_clock::now();
  }

  auto error = Error::Success;

  auto res = send_with_content_provider(
      req, body, content_length, std::move(content_provider),
      std::move(content_provider_without_length), content_type, error);

  return Result{std::move(res), error, std::move(req.headers)};
}

std::string
ClientImpl::adjust_host_string(const std::string &host) const {
  if (host.find(':') != std::string::npos) { return "[" + host + "]"; }
  return host;
}

bool ClientImpl::process_request(Stream &strm, Request &req,
                                        Response &res, bool close_connection,
                                        Error &error) {
  // Send request
  if (!write_request(strm, req, close_connection, error)) { return false; }

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  if (is_ssl()) {
    auto is_proxy_enabled = !proxy_host_.empty() && proxy_port_ != -1;
    if (!is_proxy_enabled) {
      if (detail::is_ssl_peer_could_be_closed(socket_.ssl, socket_.sock)) {
        error = Error::SSLPeerCouldBeClosed_;
        return false;
      }
    }
  }
#endif

  // Receive response and headers
  if (!read_response_line(strm, req, res) ||
      !detail::read_headers(strm, res.headers)) {
    error = Error::Read;
    return false;
  }

  // Body
  if ((res.status != StatusCode::NoContent_204) && req.method != "HEAD" &&
      req.method != "CONNECT") {
    auto redirect = 300 < res.status && res.status < 400 &&
                    res.status != StatusCode::NotModified_304 &&
                    follow_location_;

    if (req.response_handler && !redirect) {
      if (!req.response_handler(res)) {
        error = Error::Canceled;
        return false;
      }
    }

    auto out =
        req.content_receiver
            ? static_cast<ContentReceiverWithProgress>(
                  [&](const char *buf, size_t n, uint64_t off, uint64_t len) {
                    if (redirect) { return true; }
                    auto ret = req.content_receiver(buf, n, off, len);
                    if (!ret) { error = Error::Canceled; }
                    return ret;
                  })
            : static_cast<ContentReceiverWithProgress>(
                  [&](const char *buf, size_t n, uint64_t /*off*/,
                      uint64_t /*len*/) {
                    assert(res.body.size() + n <= res.body.max_size());
                    res.body.append(buf, n);
                    return true;
                  });

    auto progress = [&](uint64_t current, uint64_t total) {
      if (!req.progress || redirect) { return true; }
      auto ret = req.progress(current, total);
      if (!ret) { error = Error::Canceled; }
      return ret;
    };

    if (res.has_header("Content-Length")) {
      if (!req.content_receiver) {
        auto len = res.get_header_value_u64("Content-Length");
        if (len > res.body.max_size()) {
          error = Error::Read;
          return false;
        }
        res.body.reserve(static_cast<size_t>(len));
      }
    }

    if (res.status != StatusCode::NotModified_304) {
      int dummy_status;
      if (!detail::read_content(strm, res, (std::numeric_limits<size_t>::max)(),
                                dummy_status, std::move(progress),
                                std::move(out), decompress_)) {
        if (error != Error::Canceled) { error = Error::Read; }
        return false;
      }
    }
  }

  // Log
  if (logger_) { logger_(req, res); }

  return true;
}

ContentProviderWithoutLength ClientImpl::get_multipart_content_provider(
    const std::string &boundary, const MultipartFormDataItems &items,
    const MultipartFormDataProviderItems &provider_items) const {
  size_t cur_item = 0;
  size_t cur_start = 0;
  // cur_item and cur_start are copied to within the std::function and maintain
  // state between successive calls
  return [&, cur_item, cur_start](size_t offset,
                                  DataSink &sink) mutable -> bool {
    if (!offset && !items.empty()) {
      sink.os << detail::serialize_multipart_formdata(items, boundary, false);
      return true;
    } else if (cur_item < provider_items.size()) {
      if (!cur_start) {
        const auto &begin = detail::serialize_multipart_formdata_item_begin(
            provider_items[cur_item], boundary);
        offset += begin.size();
        cur_start = offset;
        sink.os << begin;
      }

      DataSink cur_sink;
      auto has_data = true;
      cur_sink.write = sink.write;
      cur_sink.done = [&]() { has_data = false; };

      if (!provider_items[cur_item].provider(offset - cur_start, cur_sink)) {
        return false;
      }

      if (!has_data) {
        sink.os << detail::serialize_multipart_formdata_item_end();
        cur_item++;
        cur_start = 0;
      }
      return true;
    } else {
      sink.os << detail::serialize_multipart_formdata_finish(boundary);
      sink.done();
      return true;
    }
  };
}

bool ClientImpl::process_socket(
    const Socket &socket,
    std::chrono::time_point<std::chrono::steady_clock> start_time,
    std::function<bool(Stream &strm)> callback) {
  return detail::process_client_socket(
      socket.sock, read_timeout_sec_, read_timeout_usec_, write_timeout_sec_,
      write_timeout_usec_, max_timeout_msec_, start_time, std::move(callback));
}

bool ClientImpl::is_ssl() const { return false; }

Result ClientImpl::Get(const std::string &path) {
  return Get(path, Headers(), Progress());
}

Result ClientImpl::Get(const std::string &path, Progress progress) {
  return Get(path, Headers(), std::move(progress));
}

Result ClientImpl::Get(const std::string &path, const Headers &headers) {
  return Get(path, headers, Progress());
}

Result ClientImpl::Get(const std::string &path, const Headers &headers,
                              Progress progress) {
  Request req;
  req.method = "GET";
  req.path = path;
  req.headers = headers;
  req.progress = std::move(progress);
  if (max_timeout_msec_ > 0) {
    req.start_time_ = std::chrono::steady_clock::now();
  }

  return send_(std::move(req));
}

Result ClientImpl::Get(const std::string &path,
                              ContentReceiver content_receiver) {
  return Get(path, Headers(), nullptr, std::move(content_receiver), nullptr);
}

Result ClientImpl::Get(const std::string &path,
                              ContentReceiver content_receiver,
                              Progress progress) {
  return Get(path, Headers(), nullptr, std::move(content_receiver),
             std::move(progress));
}

Result ClientImpl::Get(const std::string &path, const Headers &headers,
                              ContentReceiver content_receiver) {
  return Get(path, headers, nullptr, std::move(content_receiver), nullptr);
}

Result ClientImpl::Get(const std::string &path, const Headers &headers,
                              ContentReceiver content_receiver,
                              Progress progress) {
  return Get(path, headers, nullptr, std::move(content_receiver),
             std::move(progress));
}

Result ClientImpl::Get(const std::string &path,
                              ResponseHandler response_handler,
                              ContentReceiver content_receiver) {
  return Get(path, Headers(), std::move(response_handler),
             std::move(content_receiver), nullptr);
}

Result ClientImpl::Get(const std::string &path, const Headers &headers,
                              ResponseHandler response_handler,
                              ContentReceiver content_receiver) {
  return Get(path, headers, std::move(response_handler),
             std::move(content_receiver), nullptr);
}

Result ClientImpl::Get(const std::string &path,
                              ResponseHandler response_handler,
                              ContentReceiver content_receiver,
                              Progress progress) {
  return Get(path, Headers(), std::move(response_handler),
             std::move(content_receiver), std::move(progress));
}

Result ClientImpl::Get(const std::string &path, const Headers &headers,
                              ResponseHandler response_handler,
                              ContentReceiver content_receiver,
                              Progress progress) {
  Request req;
  req.method = "GET";
  req.path = path;
  req.headers = headers;
  req.response_handler = std::move(response_handler);
  req.content_receiver =
      [content_receiver](const char *data, size_t data_length,
                         uint64_t /*offset*/, uint64_t /*total_length*/) {
        return content_receiver(data, data_length);
      };
  req.progress = std::move(progress);
  if (max_timeout_msec_ > 0) {
    req.start_time_ = std::chrono::steady_clock::now();
  }

  return send_(std::move(req));
}

Result ClientImpl::Get(const std::string &path, const Params &params,
                              const Headers &headers, Progress progress) {
  if (params.empty()) { return Get(path, headers); }

  std::string path_with_query = append_query_params(path, params);
  return Get(path_with_query, headers, std::move(progress));
}

Result ClientImpl::Get(const std::string &path, const Params &params,
                              const Headers &headers,
                              ContentReceiver content_receiver,
                              Progress progress) {
  return Get(path, params, headers, nullptr, std::move(content_receiver),
             std::move(progress));
}

Result ClientImpl::Get(const std::string &path, const Params &params,
                              const Headers &headers,
                              ResponseHandler response_handler,
                              ContentReceiver content_receiver,
                              Progress progress) {
  if (params.empty()) {
    return Get(path, headers, std::move(response_handler),
               std::move(content_receiver), std::move(progress));
  }

  std::string path_with_query = append_query_params(path, params);
  return Get(path_with_query, headers, std::move(response_handler),
             std::move(content_receiver), std::move(progress));
}

Result ClientImpl::Head(const std::string &path) {
  return Head(path, Headers());
}

Result ClientImpl::Head(const std::string &path,
                               const Headers &headers) {
  Request req;
  req.method = "HEAD";
  req.headers = headers;
  req.path = path;
  if (max_timeout_msec_ > 0) {
    req.start_time_ = std::chrono::steady_clock::now();
  }

  return send_(std::move(req));
}

Result ClientImpl::Post(const std::string &path) {
  return Post(path, std::string(), std::string());
}

Result ClientImpl::Post(const std::string &path,
                               const Headers &headers) {
  return Post(path, headers, nullptr, 0, std::string());
}

Result ClientImpl::Post(const std::string &path, const char *body,
                               size_t content_length,
                               const std::string &content_type) {
  return Post(path, Headers(), body, content_length, content_type, nullptr);
}

Result ClientImpl::Post(const std::string &path, const Headers &headers,
                               const char *body, size_t content_length,
                               const std::string &content_type) {
  return send_with_content_provider("POST", path, headers, body, content_length,
                                    nullptr, nullptr, content_type, nullptr);
}

Result ClientImpl::Post(const std::string &path, const Headers &headers,
                               const char *body, size_t content_length,
                               const std::string &content_type,
                               Progress progress) {
  return send_with_content_provider("POST", path, headers, body, content_length,
                                    nullptr, nullptr, content_type, progress);
}

Result ClientImpl::Post(const std::string &path, const std::string &body,
                               const std::string &content_type) {
  return Post(path, Headers(), body, content_type);
}

Result ClientImpl::Post(const std::string &path, const std::string &body,
                               const std::string &content_type,
                               Progress progress) {
  return Post(path, Headers(), body, content_type, progress);
}

Result ClientImpl::Post(const std::string &path, const Headers &headers,
                               const std::string &body,
                               const std::string &content_type) {
  return send_with_content_provider("POST", path, headers, body.data(),
                                    body.size(), nullptr, nullptr, content_type,
                                    nullptr);
}

Result ClientImpl::Post(const std::string &path, const Headers &headers,
                               const std::string &body,
                               const std::string &content_type,
                               Progress progress) {
  return send_with_content_provider("POST", path, headers, body.data(),
                                    body.size(), nullptr, nullptr, content_type,
                                    progress);
}

Result ClientImpl::Post(const std::string &path, const Params &params) {
  return Post(path, Headers(), params);
}

Result ClientImpl::Post(const std::string &path, size_t content_length,
                               ContentProvider content_provider,
                               const std::string &content_type) {
  return Post(path, Headers(), content_length, std::move(content_provider),
              content_type);
}

Result ClientImpl::Post(const std::string &path,
                               ContentProviderWithoutLength content_provider,
                               const std::string &content_type) {
  return Post(path, Headers(), std::move(content_provider), content_type);
}

Result ClientImpl::Post(const std::string &path, const Headers &headers,
                               size_t content_length,
                               ContentProvider content_provider,
                               const std::string &content_type) {
  return send_with_content_provider("POST", path, headers, nullptr,
                                    content_length, std::move(content_provider),
                                    nullptr, content_type, nullptr);
}

Result ClientImpl::Post(const std::string &path, const Headers &headers,
                               ContentProviderWithoutLength content_provider,
                               const std::string &content_type) {
  return send_with_content_provider("POST", path, headers, nullptr, 0, nullptr,
                                    std::move(content_provider), content_type,
                                    nullptr);
}

Result ClientImpl::Post(const std::string &path, const Headers &headers,
                               const Params &params) {
  auto query = detail::params_to_query_str(params);
  return Post(path, headers, query, "application/x-www-form-urlencoded");
}

Result ClientImpl::Post(const std::string &path, const Headers &headers,
                               const Params &params, Progress progress) {
  auto query = detail::params_to_query_str(params);
  return Post(path, headers, query, "application/x-www-form-urlencoded",
              progress);
}

Result ClientImpl::Post(const std::string &path,
                               const MultipartFormDataItems &items) {
  return Post(path, Headers(), items);
}

Result ClientImpl::Post(const std::string &path, const Headers &headers,
                               const MultipartFormDataItems &items) {
  const auto &boundary = detail::make_multipart_data_boundary();
  const auto &content_type =
      detail::serialize_multipart_formdata_get_content_type(boundary);
  const auto &body = detail::serialize_multipart_formdata(items, boundary);
  return Post(path, headers, body, content_type);
}

Result ClientImpl::Post(const std::string &path, const Headers &headers,
                               const MultipartFormDataItems &items,
                               const std::string &boundary) {
  if (!detail::is_multipart_boundary_chars_valid(boundary)) {
    return Result{nullptr, Error::UnsupportedMultipartBoundaryChars};
  }

  const auto &content_type =
      detail::serialize_multipart_formdata_get_content_type(boundary);
  const auto &body = detail::serialize_multipart_formdata(items, boundary);
  return Post(path, headers, body, content_type);
}

Result
ClientImpl::Post(const std::string &path, const Headers &headers,
                 const MultipartFormDataItems &items,
                 const MultipartFormDataProviderItems &provider_items) {
  const auto &boundary = detail::make_multipart_data_boundary();
  const auto &content_type =
      detail::serialize_multipart_formdata_get_content_type(boundary);
  return send_with_content_provider(
      "POST", path, headers, nullptr, 0, nullptr,
      get_multipart_content_provider(boundary, items, provider_items),
      content_type, nullptr);
}

Result ClientImpl::Put(const std::string &path) {
  return Put(path, std::string(), std::string());
}

Result ClientImpl::Put(const std::string &path, const char *body,
                              size_t content_length,
                              const std::string &content_type) {
  return Put(path, Headers(), body, content_length, content_type);
}

Result ClientImpl::Put(const std::string &path, const Headers &headers,
                              const char *body, size_t content_length,
                              const std::string &content_type) {
  return send_with_content_provider("PUT", path, headers, body, content_length,
                                    nullptr, nullptr, content_type, nullptr);
}

Result ClientImpl::Put(const std::string &path, const Headers &headers,
                              const char *body, size_t content_length,
                              const std::string &content_type,
                              Progress progress) {
  return send_with_content_provider("PUT", path, headers, body, content_length,
                                    nullptr, nullptr, content_type, progress);
}

Result ClientImpl::Put(const std::string &path, const std::string &body,
                              const std::string &content_type) {
  return Put(path, Headers(), body, content_type);
}

Result ClientImpl::Put(const std::string &path, const std::string &body,
                              const std::string &content_type,
                              Progress progress) {
  return Put(path, Headers(), body, content_type, progress);
}

Result ClientImpl::Put(const std::string &path, const Headers &headers,
                              const std::string &body,
                              const std::string &content_type) {
  return send_with_content_provider("PUT", path, headers, body.data(),
                                    body.size(), nullptr, nullptr, content_type,
                                    nullptr);
}

Result ClientImpl::Put(const std::string &path, const Headers &headers,
                              const std::string &body,
                              const std::string &content_type,
                              Progress progress) {
  return send_with_content_provider("PUT", path, headers, body.data(),
                                    body.size(), nullptr, nullptr, content_type,
                                    progress);
}

Result ClientImpl::Put(const std::string &path, size_t content_length,
                              ContentProvider content_provider,
                              const std::string &content_type) {
  return Put(path, Headers(), content_length, std::move(content_provider),
             content_type);
}

Result ClientImpl::Put(const std::string &path,
                              ContentProviderWithoutLength content_provider,
                              const std::string &content_type) {
  return Put(path, Headers(), std::move(content_provider), content_type);
}

Result ClientImpl::Put(const std::string &path, const Headers &headers,
                              size_t content_length,
                              ContentProvider content_provider,
                              const std::string &content_type) {
  return send_with_content_provider("PUT", path, headers, nullptr,
                                    content_length, std::move(content_provider),
                                    nullptr, content_type, nullptr);
}

Result ClientImpl::Put(const std::string &path, const Headers &headers,
                              ContentProviderWithoutLength content_provider,
                              const std::string &content_type) {
  return send_with_content_provider("PUT", path, headers, nullptr, 0, nullptr,
                                    std::move(content_provider), content_type,
                                    nullptr);
}

Result ClientImpl::Put(const std::string &path, const Params &params) {
  return Put(path, Headers(), params);
}

Result ClientImpl::Put(const std::string &path, const Headers &headers,
                              const Params &params) {
  auto query = detail::params_to_query_str(params);
  return Put(path, headers, query, "application/x-www-form-urlencoded");
}

Result ClientImpl::Put(const std::string &path, const Headers &headers,
                              const Params &params, Progress progress) {
  auto query = detail::params_to_query_str(params);
  return Put(path, headers, query, "application/x-www-form-urlencoded",
             progress);
}

Result ClientImpl::Put(const std::string &path,
                              const MultipartFormDataItems &items) {
  return Put(path, Headers(), items);
}

Result ClientImpl::Put(const std::string &path, const Headers &headers,
                              const MultipartFormDataItems &items) {
  const auto &boundary = detail::make_multipart_data_boundary();
  const auto &content_type =
      detail::serialize_multipart_formdata_get_content_type(boundary);
  const auto &body = detail::serialize_multipart_formdata(items, boundary);
  return Put(path, headers, body, content_type);
}

Result ClientImpl::Put(const std::string &path, const Headers &headers,
                              const MultipartFormDataItems &items,
                              const std::string &boundary) {
  if (!detail::is_multipart_boundary_chars_valid(boundary)) {
    return Result{nullptr, Error::UnsupportedMultipartBoundaryChars};
  }

  const auto &content_type =
      detail::serialize_multipart_formdata_get_content_type(boundary);
  const auto &body = detail::serialize_multipart_formdata(items, boundary);
  return Put(path, headers, body, content_type);
}

Result
ClientImpl::Put(const std::string &path, const Headers &headers,
                const MultipartFormDataItems &items,
                const MultipartFormDataProviderItems &provider_items) {
  const auto &boundary = detail::make_multipart_data_boundary();
  const auto &content_type =
      detail::serialize_multipart_formdata_get_content_type(boundary);
  return send_with_content_provider(
      "PUT", path, headers, nullptr, 0, nullptr,
      get_multipart_content_provider(boundary, items, provider_items),
      content_type, nullptr);
}
Result ClientImpl::Patch(const std::string &path) {
  return Patch(path, std::string(), std::string());
}

Result ClientImpl::Patch(const std::string &path, const char *body,
                                size_t content_length,
                                const std::string &content_type) {
  return Patch(path, Headers(), body, content_length, content_type);
}

Result ClientImpl::Patch(const std::string &path, const char *body,
                                size_t content_length,
                                const std::string &content_type,
                                Progress progress) {
  return Patch(path, Headers(), body, content_length, content_type, progress);
}

Result ClientImpl::Patch(const std::string &path, const Headers &headers,
                                const char *body, size_t content_length,
                                const std::string &content_type) {
  return Patch(path, headers, body, content_length, content_type, nullptr);
}

Result ClientImpl::Patch(const std::string &path, const Headers &headers,
                                const char *body, size_t content_length,
                                const std::string &content_type,
                                Progress progress) {
  return send_with_content_provider("PATCH", path, headers, body,
                                    content_length, nullptr, nullptr,
                                    content_type, progress);
}

Result ClientImpl::Patch(const std::string &path,
                                const std::string &body,
                                const std::string &content_type) {
  return Patch(path, Headers(), body, content_type);
}

Result ClientImpl::Patch(const std::string &path,
                                const std::string &body,
                                const std::string &content_type,
                                Progress progress) {
  return Patch(path, Headers(), body, content_type, progress);
}

Result ClientImpl::Patch(const std::string &path, const Headers &headers,
                                const std::string &body,
                                const std::string &content_type) {
  return Patch(path, headers, body, content_type, nullptr);
}

Result ClientImpl::Patch(const std::string &path, const Headers &headers,
                                const std::string &body,
                                const std::string &content_type,
                                Progress progress) {
  return send_with_content_provider("PATCH", path, headers, body.data(),
                                    body.size(), nullptr, nullptr, content_type,
                                    progress);
}

Result ClientImpl::Patch(const std::string &path, size_t content_length,
                                ContentProvider content_provider,
                                const std::string &content_type) {
  return Patch(path, Headers(), content_length, std::move(content_provider),
               content_type);
}

Result ClientImpl::Patch(const std::string &path,
                                ContentProviderWithoutLength content_provider,
                                const std::string &content_type) {
  return Patch(path, Headers(), std::move(content_provider), content_type);
}

Result ClientImpl::Patch(const std::string &path, const Headers &headers,
                                size_t content_length,
                                ContentProvider content_provider,
                                const std::string &content_type) {
  return send_with_content_provider("PATCH", path, headers, nullptr,
                                    content_length, std::move(content_provider),
                                    nullptr, content_type, nullptr);
}

Result ClientImpl::Patch(const std::string &path, const Headers &headers,
                                ContentProviderWithoutLength content_provider,
                                const std::string &content_type) {
  return send_with_content_provider("PATCH", path, headers, nullptr, 0, nullptr,
                                    std::move(content_provider), content_type,
                                    nullptr);
}

Result ClientImpl::Delete(const std::string &path) {
  return Delete(path, Headers(), std::string(), std::string());
}

Result ClientImpl::Delete(const std::string &path,
                                 const Headers &headers) {
  return Delete(path, headers, std::string(), std::string());
}

Result ClientImpl::Delete(const std::string &path, const char *body,
                                 size_t content_length,
                                 const std::string &content_type) {
  return Delete(path, Headers(), body, content_length, content_type);
}

Result ClientImpl::Delete(const std::string &path, const char *body,
                                 size_t content_length,
                                 const std::string &content_type,
                                 Progress progress) {
  return Delete(path, Headers(), body, content_length, content_type, progress);
}

Result ClientImpl::Delete(const std::string &path,
                                 const Headers &headers, const char *body,
                                 size_t content_length,
                                 const std::string &content_type) {
  return Delete(path, headers, body, content_length, content_type, nullptr);
}

Result ClientImpl::Delete(const std::string &path,
                                 const Headers &headers, const char *body,
                                 size_t content_length,
                                 const std::string &content_type,
                                 Progress progress) {
  Request req;
  req.method = "DELETE";
  req.headers = headers;
  req.path = path;
  req.progress = progress;
  if (max_timeout_msec_ > 0) {
    req.start_time_ = std::chrono::steady_clock::now();
  }

  if (!content_type.empty()) { req.set_header("Content-Type", content_type); }
  req.body.assign(body, content_length);

  return send_(std::move(req));
}

Result ClientImpl::Delete(const std::string &path,
                                 const std::string &body,
                                 const std::string &content_type) {
  return Delete(path, Headers(), body.data(), body.size(), content_type);
}

Result ClientImpl::Delete(const std::string &path,
                                 const std::string &body,
                                 const std::string &content_type,
                                 Progress progress) {
  return Delete(path, Headers(), body.data(), body.size(), content_type,
                progress);
}

Result ClientImpl::Delete(const std::string &path,
                                 const Headers &headers,
                                 const std::string &body,
                                 const std::string &content_type) {
  return Delete(path, headers, body.data(), body.size(), content_type);
}

Result ClientImpl::Delete(const std::string &path,
                                 const Headers &headers,
                                 const std::string &body,
                                 const std::string &content_type,
                                 Progress progress) {
  return Delete(path, headers, body.data(), body.size(), content_type,
                progress);
}

Result ClientImpl::Options(const std::string &path) {
  return Options(path, Headers());
}

Result ClientImpl::Options(const std::string &path,
                                  const Headers &headers) {
  Request req;
  req.method = "OPTIONS";
  req.headers = headers;
  req.path = path;
  if (max_timeout_msec_ > 0) {
    req.start_time_ = std::chrono::steady_clock::now();
  }

  return send_(std::move(req));
}

void ClientImpl::stop() {
  std::lock_guard<std::mutex> guard(socket_mutex_);

  // If there is anything ongoing right now, the ONLY thread-safe thing we can
  // do is to shutdown_socket, so that threads using this socket suddenly
  // discover they can't read/write any more and error out. Everything else
  // (closing the socket, shutting ssl down) is unsafe because these actions are
  // not thread-safe.
  if (socket_requests_in_flight_ > 0) {
    shutdown_socket(socket_);

    // Aside from that, we set a flag for the socket to be closed when we're
    // done.
    socket_should_be_closed_when_request_is_done_ = true;
    return;
  }

  // Otherwise, still holding the mutex, we can shut everything down ourselves
  shutdown_ssl(socket_, true);
  shutdown_socket(socket_);
  close_socket(socket_);
}

std::string ClientImpl::host() const { return host_; }

int ClientImpl::port() const { return port_; }

size_t ClientImpl::is_socket_open() const {
  std::lock_guard<std::mutex> guard(socket_mutex_);
  return socket_.is_open();
}

socket_t ClientImpl::socket() const { return socket_.sock; }

void ClientImpl::set_connection_timeout(time_t sec, time_t usec) {
  connection_timeout_sec_ = sec;
  connection_timeout_usec_ = usec;
}

void ClientImpl::set_read_timeout(time_t sec, time_t usec) {
  read_timeout_sec_ = sec;
  read_timeout_usec_ = usec;
}

void ClientImpl::set_write_timeout(time_t sec, time_t usec) {
  write_timeout_sec_ = sec;
  write_timeout_usec_ = usec;
}

void ClientImpl::set_max_timeout(time_t msec) {
  max_timeout_msec_ = msec;
}

void ClientImpl::set_basic_auth(const std::string &username,
                                       const std::string &password) {
  basic_auth_username_ = username;
  basic_auth_password_ = password;
}

void ClientImpl::set_bearer_token_auth(const std::string &token) {
  bearer_token_auth_token_ = token;
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
void ClientImpl::set_digest_auth(const std::string &username,
                                        const std::string &password) {
  digest_auth_username_ = username;
  digest_auth_password_ = password;
}
#endif

void ClientImpl::set_keep_alive(bool on) { keep_alive_ = on; }

void ClientImpl::set_follow_location(bool on) { follow_location_ = on; }

void ClientImpl::set_url_encode(bool on) { url_encode_ = on; }

void
ClientImpl::set_hostname_addr_map(std::map<std::string, std::string> addr_map) {
  addr_map_ = std::move(addr_map);
}

void ClientImpl::set_default_headers(Headers headers) {
  default_headers_ = std::move(headers);
}

void ClientImpl::set_header_writer(
    std::function<ssize_t(Stream &, Headers &)> const &writer) {
  header_writer_ = writer;
}

void ClientImpl::set_address_family(int family) {
  address_family_ = family;
}

void ClientImpl::set_tcp_nodelay(bool on) { tcp_nodelay_ = on; }

void ClientImpl::set_ipv6_v6only(bool on) { ipv6_v6only_ = on; }

void ClientImpl::set_socket_options(SocketOptions socket_options) {
  socket_options_ = std::move(socket_options);
}

void ClientImpl::set_compress(bool on) { compress_ = on; }

void ClientImpl::set_decompress(bool on) { decompress_ = on; }

void ClientImpl::set_interface(const std::string &intf) {
  interface_ = intf;
}

void ClientImpl::set_proxy(const std::string &host, int port) {
  proxy_host_ = host;
  proxy_port_ = port;
}

void ClientImpl::set_proxy_basic_auth(const std::string &username,
                                             const std::string &password) {
  proxy_basic_auth_username_ = username;
  proxy_basic_auth_password_ = password;
}

void ClientImpl::set_proxy_bearer_token_auth(const std::string &token) {
  proxy_bearer_token_auth_token_ = token;
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
void ClientImpl::set_proxy_digest_auth(const std::string &username,
                                              const std::string &password) {
  proxy_digest_auth_username_ = username;
  proxy_digest_auth_password_ = password;
}

void ClientImpl::set_ca_cert_path(const std::string &ca_cert_file_path,
                                         const std::string &ca_cert_dir_path) {
  ca_cert_file_path_ = ca_cert_file_path;
  ca_cert_dir_path_ = ca_cert_dir_path;
}

void ClientImpl::set_ca_cert_store(X509_STORE *ca_cert_store) {
  if (ca_cert_store && ca_cert_store != ca_cert_store_) {
    ca_cert_store_ = ca_cert_store;
  }
}

X509_STORE *ClientImpl::create_ca_cert_store(const char *ca_cert,
                                                    std::size_t size) const {
  auto mem = BIO_new_mem_buf(ca_cert, static_cast<int>(size));
  auto se = detail::scope_exit([&] { BIO_free_all(mem); });
  if (!mem) { return nullptr; }

  auto inf = PEM_X509_INFO_read_bio(mem, nullptr, nullptr, nullptr);
  if (!inf) { return nullptr; }

  auto cts = X509_STORE_new();
  if (cts) {
    for (auto i = 0; i < static_cast<int>(sk_X509_INFO_num(inf)); i++) {
      auto itmp = sk_X509_INFO_value(inf, i);
      if (!itmp) { continue; }

      if (itmp->x509) { X509_STORE_add_cert(cts, itmp->x509); }
      if (itmp->crl) { X509_STORE_add_crl(cts, itmp->crl); }
    }
  }

  sk_X509_INFO_pop_free(inf, X509_INFO_free);
  return cts;
}

void ClientImpl::enable_server_certificate_verification(bool enabled) {
  server_certificate_verification_ = enabled;
}

void ClientImpl::enable_server_hostname_verification(bool enabled) {
  server_hostname_verification_ = enabled;
}

void ClientImpl::set_server_certificate_verifier(
    std::function<SSLVerifierResponse(SSL *ssl)> verifier) {
  server_certificate_verifier_ = verifier;
}
#endif

void ClientImpl::set_logger(Logger logger) {
  logger_ = std::move(logger);
}

/*
 * SSL Implementation
 */
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
namespace detail {

template <typename U, typename V>
SSL *ssl_new(socket_t sock, SSL_CTX *ctx, std::mutex &ctx_mutex,
                    U SSL_connect_or_accept, V setup) {
  SSL *ssl = nullptr;
  {
    std::lock_guard<std::mutex> guard(ctx_mutex);
    ssl = SSL_new(ctx);
  }

  if (ssl) {
    set_nonblocking(sock, true);
    auto bio = BIO_new_socket(static_cast<int>(sock), BIO_NOCLOSE);
    BIO_set_nbio(bio, 1);
    SSL_set_bio(ssl, bio, bio);

    if (!setup(ssl) || SSL_connect_or_accept(ssl) != 1) {
      SSL_shutdown(ssl);
      {
        std::lock_guard<std::mutex> guard(ctx_mutex);
        SSL_free(ssl);
      }
      set_nonblocking(sock, false);
      return nullptr;
    }
    BIO_set_nbio(bio, 0);
    set_nonblocking(sock, false);
  }

  return ssl;
}

void ssl_delete(std::mutex &ctx_mutex, SSL *ssl, socket_t sock,
                       bool shutdown_gracefully) {
  // sometimes we may want to skip this to try to avoid SIGPIPE if we know
  // the remote has closed the network connection
  // Note that it is not always possible to avoid SIGPIPE, this is merely a
  // best-efforts.
  if (shutdown_gracefully) {
    (void)(sock);
    // SSL_shutdown() returns 0 on first call (indicating close_notify alert
    // sent) and 1 on subsequent call (indicating close_notify alert received)
    if (SSL_shutdown(ssl) == 0) {
      // Expected to return 1, but even if it doesn't, we free ssl
      SSL_shutdown(ssl);
    }
  }

  std::lock_guard<std::mutex> guard(ctx_mutex);
  SSL_free(ssl);
}

template <typename U>
bool ssl_connect_or_accept_nonblocking(socket_t sock, SSL *ssl,
                                       U ssl_connect_or_accept,
                                       time_t timeout_sec,
                                       time_t timeout_usec) {
  auto res = 0;
  while ((res = ssl_connect_or_accept(ssl)) != 1) {
    auto err = SSL_get_error(ssl, res);
    switch (err) {
    case SSL_ERROR_WANT_READ:
      if (select_read(sock, timeout_sec, timeout_usec) > 0) { continue; }
      break;
    case SSL_ERROR_WANT_WRITE:
      if (select_write(sock, timeout_sec, timeout_usec) > 0) { continue; }
      break;
    default: break;
    }
    return false;
  }
  return true;
}

template <typename T>
bool process_server_socket_ssl(
    const std::atomic<socket_t> &svr_sock, SSL *ssl, socket_t sock,
    size_t keep_alive_max_count, time_t keep_alive_timeout_sec,
    time_t read_timeout_sec, time_t read_timeout_usec, time_t write_timeout_sec,
    time_t write_timeout_usec, T callback) {
  return process_server_socket_core(
      svr_sock, sock, keep_alive_max_count, keep_alive_timeout_sec,
      [&](bool close_connection, bool &connection_closed) {
        SSLSocketStream strm(sock, ssl, read_timeout_sec, read_timeout_usec,
                             write_timeout_sec, write_timeout_usec);
        return callback(strm, close_connection, connection_closed);
      });
}

template <typename T>
bool process_client_socket_ssl(
    SSL *ssl, socket_t sock, time_t read_timeout_sec, time_t read_timeout_usec,
    time_t write_timeout_sec, time_t write_timeout_usec,
    time_t max_timeout_msec,
    std::chrono::time_point<std::chrono::steady_clock> start_time, T callback) {
  SSLSocketStream strm(sock, ssl, read_timeout_sec, read_timeout_usec,
                       write_timeout_sec, write_timeout_usec, max_timeout_msec,
                       start_time);
  return callback(strm);
}

// SSL socket stream implementation
SSLSocketStream::SSLSocketStream(
    socket_t sock, SSL *ssl, time_t read_timeout_sec, time_t read_timeout_usec,
    time_t write_timeout_sec, time_t write_timeout_usec,
    time_t max_timeout_msec,
    std::chrono::time_point<std::chrono::steady_clock> start_time)
    : sock_(sock), ssl_(ssl), read_timeout_sec_(read_timeout_sec),
      read_timeout_usec_(read_timeout_usec),
      write_timeout_sec_(write_timeout_sec),
      write_timeout_usec_(write_timeout_usec),
      max_timeout_msec_(max_timeout_msec), start_time_(start_time) {
  SSL_clear_mode(ssl, SSL_MODE_AUTO_RETRY);
}

SSLSocketStream::~SSLSocketStream() = default;

bool SSLSocketStream::is_readable() const {
  return SSL_pending(ssl_) > 0;
}

bool SSLSocketStream::wait_readable() const {
  if (max_timeout_msec_ <= 0) {
    return select_read(sock_, read_timeout_sec_, read_timeout_usec_) > 0;
  }

  time_t read_timeout_sec;
  time_t read_timeout_usec;
  calc_actual_timeout(max_timeout_msec_, duration(), read_timeout_sec_,
                      read_timeout_usec_, read_timeout_sec, read_timeout_usec);

  return select_read(sock_, read_timeout_sec, read_timeout_usec) > 0;
}

bool SSLSocketStream::wait_writable() const {
  return select_write(sock_, write_timeout_sec_, write_timeout_usec_) > 0 &&
         is_socket_alive(sock_) && !is_ssl_peer_could_be_closed(ssl_, sock_);
}

ssize_t SSLSocketStream::read(char *ptr, size_t size) {
  if (SSL_pending(ssl_) > 0) {
    return SSL_read(ssl_, ptr, static_cast<int>(size));
  } else if (wait_readable()) {
    auto ret = SSL_read(ssl_, ptr, static_cast<int>(size));
    if (ret < 0) {
      auto err = SSL_get_error(ssl_, ret);
      auto n = 1000;
#ifdef _WIN32
      while (--n >= 0 && (err == SSL_ERROR_WANT_READ ||
                          (err == SSL_ERROR_SYSCALL &&
                           WSAGetLastError() == WSAETIMEDOUT))) {
#else
      while (--n >= 0 && err == SSL_ERROR_WANT_READ) {
#endif
        if (SSL_pending(ssl_) > 0) {
          return SSL_read(ssl_, ptr, static_cast<int>(size));
        } else if (wait_readable()) {
          std::this_thread::sleep_for(std::chrono::microseconds{10});
          ret = SSL_read(ssl_, ptr, static_cast<int>(size));
          if (ret >= 0) { return ret; }
          err = SSL_get_error(ssl_, ret);
        } else {
          return -1;
        }
      }
    }
    return ret;
  } else {
    return -1;
  }
}

ssize_t SSLSocketStream::write(const char *ptr, size_t size) {
  if (wait_writable()) {
    auto handle_size = static_cast<int>(
        std::min<size_t>(size, (std::numeric_limits<int>::max)()));

    auto ret = SSL_write(ssl_, ptr, static_cast<int>(handle_size));
    if (ret < 0) {
      auto err = SSL_get_error(ssl_, ret);
      auto n = 1000;
#ifdef _WIN32
      while (--n >= 0 && (err == SSL_ERROR_WANT_WRITE ||
                          (err == SSL_ERROR_SYSCALL &&
                           WSAGetLastError() == WSAETIMEDOUT))) {
#else
      while (--n >= 0 && err == SSL_ERROR_WANT_WRITE) {
#endif
        if (wait_writable()) {
          std::this_thread::sleep_for(std::chrono::microseconds{10});
          ret = SSL_write(ssl_, ptr, static_cast<int>(handle_size));
          if (ret >= 0) { return ret; }
          err = SSL_get_error(ssl_, ret);
        } else {
          return -1;
        }
      }
    }
    return ret;
  }
  return -1;
}

void SSLSocketStream::get_remote_ip_and_port(std::string &ip,
                                                    int &port) const {
  detail::get_remote_ip_and_port(sock_, ip, port);
}

void SSLSocketStream::get_local_ip_and_port(std::string &ip,
                                                   int &port) const {
  detail::get_local_ip_and_port(sock_, ip, port);
}

socket_t SSLSocketStream::socket() const { return sock_; }

time_t SSLSocketStream::duration() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

} // namespace detail

// SSL HTTP server implementation
SSLServer::SSLServer(const char *cert_path, const char *private_key_path,
                            const char *client_ca_cert_file_path,
                            const char *client_ca_cert_dir_path,
                            const char *private_key_password) {
  ctx_ = SSL_CTX_new(TLS_server_method());

  if (ctx_) {
    SSL_CTX_set_options(ctx_,
                        SSL_OP_NO_COMPRESSION |
                            SSL_OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION);

    SSL_CTX_set_min_proto_version(ctx_, TLS1_2_VERSION);

    if (private_key_password != nullptr && (private_key_password[0] != '\0')) {
      SSL_CTX_set_default_passwd_cb_userdata(
          ctx_,
          reinterpret_cast<void *>(const_cast<char *>(private_key_password)));
    }

    if (SSL_CTX_use_certificate_chain_file(ctx_, cert_path) != 1 ||
        SSL_CTX_use_PrivateKey_file(ctx_, private_key_path, SSL_FILETYPE_PEM) !=
            1 ||
        SSL_CTX_check_private_key(ctx_) != 1) {
      SSL_CTX_free(ctx_);
      ctx_ = nullptr;
    } else if (client_ca_cert_file_path || client_ca_cert_dir_path) {
      SSL_CTX_load_verify_locations(ctx_, client_ca_cert_file_path,
                                    client_ca_cert_dir_path);

      SSL_CTX_set_verify(
          ctx_, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, nullptr);
    }
  }
}

SSLServer::SSLServer(X509 *cert, EVP_PKEY *private_key,
                            X509_STORE *client_ca_cert_store) {
  ctx_ = SSL_CTX_new(TLS_server_method());

  if (ctx_) {
    SSL_CTX_set_options(ctx_,
                        SSL_OP_NO_COMPRESSION |
                            SSL_OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION);

    SSL_CTX_set_min_proto_version(ctx_, TLS1_2_VERSION);

    if (SSL_CTX_use_certificate(ctx_, cert) != 1 ||
        SSL_CTX_use_PrivateKey(ctx_, private_key) != 1) {
      SSL_CTX_free(ctx_);
      ctx_ = nullptr;
    } else if (client_ca_cert_store) {
      SSL_CTX_set_cert_store(ctx_, client_ca_cert_store);

      SSL_CTX_set_verify(
          ctx_, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, nullptr);
    }
  }
}

SSLServer::SSLServer(
    const std::function<bool(SSL_CTX &ssl_ctx)> &setup_ssl_ctx_callback) {
  ctx_ = SSL_CTX_new(TLS_method());
  if (ctx_) {
    if (!setup_ssl_ctx_callback(*ctx_)) {
      SSL_CTX_free(ctx_);
      ctx_ = nullptr;
    }
  }
}

SSLServer::~SSLServer() {
  if (ctx_) { SSL_CTX_free(ctx_); }
}

bool SSLServer::is_valid() const { return ctx_; }

SSL_CTX *SSLServer::ssl_context() const { return ctx_; }

void SSLServer::update_certs(X509 *cert, EVP_PKEY *private_key,
                                    X509_STORE *client_ca_cert_store) {

  std::lock_guard<std::mutex> guard(ctx_mutex_);

  SSL_CTX_use_certificate(ctx_, cert);
  SSL_CTX_use_PrivateKey(ctx_, private_key);

  if (client_ca_cert_store != nullptr) {
    SSL_CTX_set_cert_store(ctx_, client_ca_cert_store);
  }
}

bool SSLServer::process_and_close_socket(socket_t sock) {
  auto ssl = detail::ssl_new(
      sock, ctx_, ctx_mutex_,
      [&](SSL *ssl2) {
        return detail::ssl_connect_or_accept_nonblocking(
            sock, ssl2, SSL_accept, read_timeout_sec_, read_timeout_usec_);
      },
      [](SSL * /*ssl2*/) { return true; });

  auto ret = false;
  if (ssl) {
    std::string remote_addr;
    int remote_port = 0;
    detail::get_remote_ip_and_port(sock, remote_addr, remote_port);

    std::string local_addr;
    int local_port = 0;
    detail::get_local_ip_and_port(sock, local_addr, local_port);

    ret = detail::process_server_socket_ssl(
        svr_sock_, ssl, sock, keep_alive_max_count_, keep_alive_timeout_sec_,
        read_timeout_sec_, read_timeout_usec_, write_timeout_sec_,
        write_timeout_usec_,
        [&](Stream &strm, bool close_connection, bool &connection_closed) {
          return process_request(strm, remote_addr, remote_port, local_addr,
                                 local_port, close_connection,
                                 connection_closed,
                                 [&](Request &req) { req.ssl = ssl; });
        });

    // Shutdown gracefully if the result seemed successful, non-gracefully if
    // the connection appeared to be closed.
    const bool shutdown_gracefully = ret;
    detail::ssl_delete(ctx_mutex_, ssl, sock, shutdown_gracefully);
  }

  detail::shutdown_socket(sock);
  detail::close_socket(sock);
  return ret;
}

// SSL HTTP client implementation
SSLClient::SSLClient(const std::string &host)
    : SSLClient(host, 443, std::string(), std::string()) {}

SSLClient::SSLClient(const std::string &host, int port)
    : SSLClient(host, port, std::string(), std::string()) {}

SSLClient::SSLClient(const std::string &host, int port,
                            const std::string &client_cert_path,
                            const std::string &client_key_path,
                            const std::string &private_key_password)
    : ClientImpl(host, port, client_cert_path, client_key_path) {
  ctx_ = SSL_CTX_new(TLS_client_method());

  SSL_CTX_set_min_proto_version(ctx_, TLS1_2_VERSION);

  detail::split(&host_[0], &host_[host_.size()], '.',
                [&](const char *b, const char *e) {
                  host_components_.emplace_back(b, e);
                });

  if (!client_cert_path.empty() && !client_key_path.empty()) {
    if (!private_key_password.empty()) {
      SSL_CTX_set_default_passwd_cb_userdata(
          ctx_, reinterpret_cast<void *>(
                    const_cast<char *>(private_key_password.c_str())));
    }

    if (SSL_CTX_use_certificate_file(ctx_, client_cert_path.c_str(),
                                     SSL_FILETYPE_PEM) != 1 ||
        SSL_CTX_use_PrivateKey_file(ctx_, client_key_path.c_str(),
                                    SSL_FILETYPE_PEM) != 1) {
      SSL_CTX_free(ctx_);
      ctx_ = nullptr;
    }
  }
}

SSLClient::SSLClient(const std::string &host, int port,
                            X509 *client_cert, EVP_PKEY *client_key,
                            const std::string &private_key_password)
    : ClientImpl(host, port) {
  ctx_ = SSL_CTX_new(TLS_client_method());

  detail::split(&host_[0], &host_[host_.size()], '.',
                [&](const char *b, const char *e) {
                  host_components_.emplace_back(b, e);
                });

  if (client_cert != nullptr && client_key != nullptr) {
    if (!private_key_password.empty()) {
      SSL_CTX_set_default_passwd_cb_userdata(
          ctx_, reinterpret_cast<void *>(
                    const_cast<char *>(private_key_password.c_str())));
    }

    if (SSL_CTX_use_certificate(ctx_, client_cert) != 1 ||
        SSL_CTX_use_PrivateKey(ctx_, client_key) != 1) {
      SSL_CTX_free(ctx_);
      ctx_ = nullptr;
    }
  }
}

SSLClient::~SSLClient() {
  if (ctx_) { SSL_CTX_free(ctx_); }
  // Make sure to shut down SSL since shutdown_ssl will resolve to the
  // base function rather than the derived function once we get to the
  // base class destructor, and won't free the SSL (causing a leak).
  shutdown_ssl_impl(socket_, true);
}

bool SSLClient::is_valid() const { return ctx_; }

void SSLClient::set_ca_cert_store(X509_STORE *ca_cert_store) {
  if (ca_cert_store) {
    if (ctx_) {
      if (SSL_CTX_get_cert_store(ctx_) != ca_cert_store) {
        // Free memory allocated for old cert and use new store `ca_cert_store`
        SSL_CTX_set_cert_store(ctx_, ca_cert_store);
      }
    } else {
      X509_STORE_free(ca_cert_store);
    }
  }
}

void SSLClient::load_ca_cert_store(const char *ca_cert,
                                          std::size_t size) {
  set_ca_cert_store(ClientImpl::create_ca_cert_store(ca_cert, size));
}

long SSLClient::get_openssl_verify_result() const {
  return verify_result_;
}

SSL_CTX *SSLClient::ssl_context() const { return ctx_; }

bool SSLClient::create_and_connect_socket(Socket &socket, Error &error) {
  return is_valid() && ClientImpl::create_and_connect_socket(socket, error);
}

// Assumes that socket_mutex_ is locked and that there are no requests in flight
bool SSLClient::connect_with_proxy(
    Socket &socket,
    std::chrono::time_point<std::chrono::steady_clock> start_time,
    Response &res, bool &success, Error &error) {
  success = true;
  Response proxy_res;
  if (!detail::process_client_socket(
          socket.sock, read_timeout_sec_, read_timeout_usec_,
          write_timeout_sec_, write_timeout_usec_, max_timeout_msec_,
          start_time, [&](Stream &strm) {
            Request req2;
            req2.method = "CONNECT";
            req2.path = host_and_port_;
            if (max_timeout_msec_ > 0) {
              req2.start_time_ = std::chrono::steady_clock::now();
            }
            return process_request(strm, req2, proxy_res, false, error);
          })) {
    // Thread-safe to close everything because we are assuming there are no
    // requests in flight
    shutdown_ssl(socket, true);
    shutdown_socket(socket);
    close_socket(socket);
    success = false;
    return false;
  }

  if (proxy_res.status == StatusCode::ProxyAuthenticationRequired_407) {
    if (!proxy_digest_auth_username_.empty() &&
        !proxy_digest_auth_password_.empty()) {
      std::map<std::string, std::string> auth;
      if (detail::parse_www_authenticate(proxy_res, auth, true)) {
        proxy_res = Response();
        if (!detail::process_client_socket(
                socket.sock, read_timeout_sec_, read_timeout_usec_,
                write_timeout_sec_, write_timeout_usec_, max_timeout_msec_,
                start_time, [&](Stream &strm) {
                  Request req3;
                  req3.method = "CONNECT";
                  req3.path = host_and_port_;
                  req3.headers.insert(detail::make_digest_authentication_header(
                      req3, auth, 1, detail::random_string(10),
                      proxy_digest_auth_username_, proxy_digest_auth_password_,
                      true));
                  if (max_timeout_msec_ > 0) {
                    req3.start_time_ = std::chrono::steady_clock::now();
                  }
                  return process_request(strm, req3, proxy_res, false, error);
                })) {
          // Thread-safe to close everything because we are assuming there are
          // no requests in flight
          shutdown_ssl(socket, true);
          shutdown_socket(socket);
          close_socket(socket);
          success = false;
          return false;
        }
      }
    }
  }

  // If status code is not 200, proxy request is failed.
  // Set error to ProxyConnection and return proxy response
  // as the response of the request
  if (proxy_res.status != StatusCode::OK_200) {
    error = Error::ProxyConnection;
    res = std::move(proxy_res);
    // Thread-safe to close everything because we are assuming there are
    // no requests in flight
    shutdown_ssl(socket, true);
    shutdown_socket(socket);
    close_socket(socket);
    return false;
  }

  return true;
}

bool SSLClient::load_certs() {
  auto ret = true;

  std::call_once(initialize_cert_, [&]() {
    std::lock_guard<std::mutex> guard(ctx_mutex_);
    if (!ca_cert_file_path_.empty()) {
      if (!SSL_CTX_load_verify_locations(ctx_, ca_cert_file_path_.c_str(),
                                         nullptr)) {
        ret = false;
      }
    } else if (!ca_cert_dir_path_.empty()) {
      if (!SSL_CTX_load_verify_locations(ctx_, nullptr,
                                         ca_cert_dir_path_.c_str())) {
        ret = false;
      }
    } else {
      auto loaded = false;
#ifdef _WIN32
      loaded =
          detail::load_system_certs_on_windows(SSL_CTX_get_cert_store(ctx_));
#elif defined(CPPHTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN) && defined(__APPLE__)
#if TARGET_OS_OSX
      loaded = detail::load_system_certs_on_macos(SSL_CTX_get_cert_store(ctx_));
#endif // TARGET_OS_OSX
#endif // _WIN32
      if (!loaded) { SSL_CTX_set_default_verify_paths(ctx_); }
    }
  });

  return ret;
}

bool SSLClient::initialize_ssl(Socket &socket, Error &error) {
  auto ssl = detail::ssl_new(
      socket.sock, ctx_, ctx_mutex_,
      [&](SSL *ssl2) {
        if (server_certificate_verification_) {
          if (!load_certs()) {
            error = Error::SSLLoadingCerts;
            return false;
          }
          SSL_set_verify(ssl2, SSL_VERIFY_NONE, nullptr);
        }

        if (!detail::ssl_connect_or_accept_nonblocking(
                socket.sock, ssl2, SSL_connect, connection_timeout_sec_,
                connection_timeout_usec_)) {
          error = Error::SSLConnection;
          return false;
        }

        if (server_certificate_verification_) {
          auto verification_status = SSLVerifierResponse::NoDecisionMade;

          if (server_certificate_verifier_) {
            verification_status = server_certificate_verifier_(ssl2);
          }

          if (verification_status == SSLVerifierResponse::CertificateRejected) {
            error = Error::SSLServerVerification;
            return false;
          }

          if (verification_status == SSLVerifierResponse::NoDecisionMade) {
            verify_result_ = SSL_get_verify_result(ssl2);

            if (verify_result_ != X509_V_OK) {
              error = Error::SSLServerVerification;
              return false;
            }

            auto server_cert = SSL_get1_peer_certificate(ssl2);
            auto se = detail::scope_exit([&] { X509_free(server_cert); });

            if (server_cert == nullptr) {
              error = Error::SSLServerVerification;
              return false;
            }

            if (server_hostname_verification_) {
              if (!verify_host(server_cert)) {
                error = Error::SSLServerHostnameVerification;
                return false;
              }
            }
          }
        }

        return true;
      },
      [&](SSL *ssl2) {
#if defined(OPENSSL_IS_BORINGSSL)
        SSL_set_tlsext_host_name(ssl2, host_.c_str());
#else
        // NOTE: Direct call instead of using the OpenSSL macro to suppress
        // -Wold-style-cast warning
        SSL_ctrl(ssl2, SSL_CTRL_SET_TLSEXT_HOSTNAME, TLSEXT_NAMETYPE_host_name,
                 static_cast<void *>(const_cast<char *>(host_.c_str())));
#endif
        return true;
      });

  if (ssl) {
    socket.ssl = ssl;
    return true;
  }

  shutdown_socket(socket);
  close_socket(socket);
  return false;
}

void SSLClient::shutdown_ssl(Socket &socket, bool shutdown_gracefully) {
  shutdown_ssl_impl(socket, shutdown_gracefully);
}

void SSLClient::shutdown_ssl_impl(Socket &socket,
                                         bool shutdown_gracefully) {
  if (socket.sock == INVALID_SOCKET) {
    assert(socket.ssl == nullptr);
    return;
  }
  if (socket.ssl) {
    detail::ssl_delete(ctx_mutex_, socket.ssl, socket.sock,
                       shutdown_gracefully);
    socket.ssl = nullptr;
  }
  assert(socket.ssl == nullptr);
}

bool SSLClient::process_socket(
    const Socket &socket,
    std::chrono::time_point<std::chrono::steady_clock> start_time,
    std::function<bool(Stream &strm)> callback) {
  assert(socket.ssl);
  return detail::process_client_socket_ssl(
      socket.ssl, socket.sock, read_timeout_sec_, read_timeout_usec_,
      write_timeout_sec_, write_timeout_usec_, max_timeout_msec_, start_time,
      std::move(callback));
}

bool SSLClient::is_ssl() const { return true; }

bool SSLClient::verify_host(X509 *server_cert) const {
  /* Quote from RFC2818 section 3.1 "Server Identity"

     If a subjectAltName extension of type dNSName is present, that MUST
     be used as the identity. Otherwise, the (most specific) Common Name
     field in the Subject field of the certificate MUST be used. Although
     the use of the Common Name is existing practice, it is deprecated and
     Certification Authorities are encouraged to use the dNSName instead.

     Matching is performed using the matching rules specified by
     [RFC2459].  If more than one identity of a given type is present in
     the certificate (e.g., more than one dNSName name, a match in any one
     of the set is considered acceptable.) Names may contain the wildcard
     character * which is considered to match any single domain name
     component or component fragment. E.g., *.a.com matches foo.a.com but
     not bar.foo.a.com. f*.com matches foo.com but not bar.com.

     In some cases, the URI is specified as an IP address rather than a
     hostname. In this case, the iPAddress subjectAltName must be present
     in the certificate and must exactly match the IP in the URI.

  */
  return verify_host_with_subject_alt_name(server_cert) ||
         verify_host_with_common_name(server_cert);
}

bool
SSLClient::verify_host_with_subject_alt_name(X509 *server_cert) const {
  auto ret = false;

  auto type = GEN_DNS;

  struct in6_addr addr6 = {};
  struct in_addr addr = {};
  size_t addr_len = 0;

#ifndef __MINGW32__
  if (inet_pton(AF_INET6, host_.c_str(), &addr6)) {
    type = GEN_IPADD;
    addr_len = sizeof(struct in6_addr);
  } else if (inet_pton(AF_INET, host_.c_str(), &addr)) {
    type = GEN_IPADD;
    addr_len = sizeof(struct in_addr);
  }
#endif

  auto alt_names = static_cast<const struct stack_st_GENERAL_NAME *>(
      X509_get_ext_d2i(server_cert, NID_subject_alt_name, nullptr, nullptr));

  if (alt_names) {
    auto dsn_matched = false;
    auto ip_matched = false;

    auto count = sk_GENERAL_NAME_num(alt_names);

    for (decltype(count) i = 0; i < count && !dsn_matched; i++) {
      auto val = sk_GENERAL_NAME_value(alt_names, i);
      if (val->type == type) {
        auto name =
            reinterpret_cast<const char *>(ASN1_STRING_get0_data(val->d.ia5));
        auto name_len = static_cast<size_t>(ASN1_STRING_length(val->d.ia5));

        switch (type) {
        case GEN_DNS: dsn_matched = check_host_name(name, name_len); break;

        case GEN_IPADD:
          if (!memcmp(&addr6, name, addr_len) ||
              !memcmp(&addr, name, addr_len)) {
            ip_matched = true;
          }
          break;
        }
      }
    }

    if (dsn_matched || ip_matched) { ret = true; }
  }

  GENERAL_NAMES_free(const_cast<STACK_OF(GENERAL_NAME) *>(
      reinterpret_cast<const STACK_OF(GENERAL_NAME) *>(alt_names)));
  return ret;
}

bool SSLClient::verify_host_with_common_name(X509 *server_cert) const {
  const auto subject_name = X509_get_subject_name(server_cert);

  if (subject_name != nullptr) {
    char name[BUFSIZ];
    auto name_len = X509_NAME_get_text_by_NID(subject_name, NID_commonName,
                                              name, sizeof(name));

    if (name_len != -1) {
      return check_host_name(name, static_cast<size_t>(name_len));
    }
  }

  return false;
}

bool SSLClient::check_host_name(const char *pattern,
                                       size_t pattern_len) const {
  if (host_.size() == pattern_len && host_ == pattern) { return true; }

  // Wildcard match
  // https://bugs.launchpad.net/ubuntu/+source/firefox-3.0/+bug/376484
  std::vector<std::string> pattern_components;
  detail::split(&pattern[0], &pattern[pattern_len], '.',
                [&](const char *b, const char *e) {
                  pattern_components.emplace_back(b, e);
                });

  if (host_components_.size() != pattern_components.size()) { return false; }

  auto itr = pattern_components.begin();
  for (const auto &h : host_components_) {
    auto &p = *itr;
    if (p != h && p != "*") {
      auto partial_match = (p.size() > 0 && p[p.size() - 1] == '*' &&
                            !p.compare(0, p.size() - 1, h));
      if (!partial_match) { return false; }
    }
    ++itr;
  }

  return true;
}
#endif

// Universal client implementation
Client::Client(const std::string &scheme_host_port)
    : Client(scheme_host_port, std::string(), std::string()) {}

Client::Client(const std::string &scheme_host_port,
                      const std::string &client_cert_path,
                      const std::string &client_key_path) {
  const static std::regex re(
      R"((?:([a-z]+):\/\/)?(?:\[([a-fA-F\d:]+)\]|([^:/?#]+))(?::(\d+))?)");

  std::smatch m;
  if (std::regex_match(scheme_host_port, m, re)) {
    auto scheme = m[1].str();

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (!scheme.empty() && (scheme != "http" && scheme != "https")) {
#else
    if (!scheme.empty() && scheme != "http") {
#endif
#ifndef CPPHTTPLIB_NO_EXCEPTIONS
      std::string msg = "'" + scheme + "' scheme is not supported.";
      throw std::invalid_argument(msg);
#endif
      return;
    }

    auto is_ssl = scheme == "https";

    auto host = m[2].str();
    if (host.empty()) { host = m[3].str(); }

    auto port_str = m[4].str();
    auto port = !port_str.empty() ? std::stoi(port_str) : (is_ssl ? 443 : 80);

    if (is_ssl) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
      cli_ = detail::make_unique<SSLClient>(host, port, client_cert_path,
                                            client_key_path);
      is_ssl_ = is_ssl;
#endif
    } else {
      cli_ = detail::make_unique<ClientImpl>(host, port, client_cert_path,
                                             client_key_path);
    }
  } else {
    // NOTE: Update TEST(UniversalClientImplTest, Ipv6LiteralAddress)
    // if port param below changes.
    cli_ = detail::make_unique<ClientImpl>(scheme_host_port, 80,
                                           client_cert_path, client_key_path);
  }
} // namespace detail

Client::Client(const std::string &host, int port)
    : cli_(detail::make_unique<ClientImpl>(host, port)) {}

Client::Client(const std::string &host, int port,
                      const std::string &client_cert_path,
                      const std::string &client_key_path)
    : cli_(detail::make_unique<ClientImpl>(host, port, client_cert_path,
                                           client_key_path)) {}

Client::~Client() = default;

bool Client::is_valid() const {
  return cli_ != nullptr && cli_->is_valid();
}

Result Client::Get(const std::string &path) { return cli_->Get(path); }
Result Client::Get(const std::string &path, const Headers &headers) {
  return cli_->Get(path, headers);
}
Result Client::Get(const std::string &path, Progress progress) {
  return cli_->Get(path, std::move(progress));
}
Result Client::Get(const std::string &path, const Headers &headers,
                          Progress progress) {
  return cli_->Get(path, headers, std::move(progress));
}
Result Client::Get(const std::string &path,
                          ContentReceiver content_receiver) {
  return cli_->Get(path, std::move(content_receiver));
}
Result Client::Get(const std::string &path, const Headers &headers,
                          ContentReceiver content_receiver) {
  return cli_->Get(path, headers, std::move(content_receiver));
}
Result Client::Get(const std::string &path,
                          ContentReceiver content_receiver, Progress progress) {
  return cli_->Get(path, std::move(content_receiver), std::move(progress));
}
Result Client::Get(const std::string &path, const Headers &headers,
                          ContentReceiver content_receiver, Progress progress) {
  return cli_->Get(path, headers, std::move(content_receiver),
                   std::move(progress));
}
Result Client::Get(const std::string &path,
                          ResponseHandler response_handler,
                          ContentReceiver content_receiver) {
  return cli_->Get(path, std::move(response_handler),
                   std::move(content_receiver));
}
Result Client::Get(const std::string &path, const Headers &headers,
                          ResponseHandler response_handler,
                          ContentReceiver content_receiver) {
  return cli_->Get(path, headers, std::move(response_handler),
                   std::move(content_receiver));
}
Result Client::Get(const std::string &path,
                          ResponseHandler response_handler,
                          ContentReceiver content_receiver, Progress progress) {
  return cli_->Get(path, std::move(response_handler),
                   std::move(content_receiver), std::move(progress));
}
Result Client::Get(const std::string &path, const Headers &headers,
                          ResponseHandler response_handler,
                          ContentReceiver content_receiver, Progress progress) {
  return cli_->Get(path, headers, std::move(response_handler),
                   std::move(content_receiver), std::move(progress));
}
Result Client::Get(const std::string &path, const Params &params,
                          const Headers &headers, Progress progress) {
  return cli_->Get(path, params, headers, std::move(progress));
}
Result Client::Get(const std::string &path, const Params &params,
                          const Headers &headers,
                          ContentReceiver content_receiver, Progress progress) {
  return cli_->Get(path, params, headers, std::move(content_receiver),
                   std::move(progress));
}
Result Client::Get(const std::string &path, const Params &params,
                          const Headers &headers,
                          ResponseHandler response_handler,
                          ContentReceiver content_receiver, Progress progress) {
  return cli_->Get(path, params, headers, std::move(response_handler),
                   std::move(content_receiver), std::move(progress));
}

Result Client::Head(const std::string &path) { return cli_->Head(path); }
Result Client::Head(const std::string &path, const Headers &headers) {
  return cli_->Head(path, headers);
}

Result Client::Post(const std::string &path) { return cli_->Post(path); }
Result Client::Post(const std::string &path, const Headers &headers) {
  return cli_->Post(path, headers);
}
Result Client::Post(const std::string &path, const char *body,
                           size_t content_length,
                           const std::string &content_type) {
  return cli_->Post(path, body, content_length, content_type);
}
Result Client::Post(const std::string &path, const Headers &headers,
                           const char *body, size_t content_length,
                           const std::string &content_type) {
  return cli_->Post(path, headers, body, content_length, content_type);
}
Result Client::Post(const std::string &path, const Headers &headers,
                           const char *body, size_t content_length,
                           const std::string &content_type, Progress progress) {
  return cli_->Post(path, headers, body, content_length, content_type,
                    progress);
}
Result Client::Post(const std::string &path, const std::string &body,
                           const std::string &content_type) {
  return cli_->Post(path, body, content_type);
}
Result Client::Post(const std::string &path, const std::string &body,
                           const std::string &content_type, Progress progress) {
  return cli_->Post(path, body, content_type, progress);
}
Result Client::Post(const std::string &path, const Headers &headers,
                           const std::string &body,
                           const std::string &content_type) {
  return cli_->Post(path, headers, body, content_type);
}
Result Client::Post(const std::string &path, const Headers &headers,
                           const std::string &body,
                           const std::string &content_type, Progress progress) {
  return cli_->Post(path, headers, body, content_type, progress);
}
Result Client::Post(const std::string &path, size_t content_length,
                           ContentProvider content_provider,
                           const std::string &content_type) {
  return cli_->Post(path, content_length, std::move(content_provider),
                    content_type);
}
Result Client::Post(const std::string &path,
                           ContentProviderWithoutLength content_provider,
                           const std::string &content_type) {
  return cli_->Post(path, std::move(content_provider), content_type);
}
Result Client::Post(const std::string &path, const Headers &headers,
                           size_t content_length,
                           ContentProvider content_provider,
                           const std::string &content_type) {
  return cli_->Post(path, headers, content_length, std::move(content_provider),
                    content_type);
}
Result Client::Post(const std::string &path, const Headers &headers,
                           ContentProviderWithoutLength content_provider,
                           const std::string &content_type) {
  return cli_->Post(path, headers, std::move(content_provider), content_type);
}
Result Client::Post(const std::string &path, const Params &params) {
  return cli_->Post(path, params);
}
Result Client::Post(const std::string &path, const Headers &headers,
                           const Params &params) {
  return cli_->Post(path, headers, params);
}
Result Client::Post(const std::string &path, const Headers &headers,
                           const Params &params, Progress progress) {
  return cli_->Post(path, headers, params, progress);
}
Result Client::Post(const std::string &path,
                           const MultipartFormDataItems &items) {
  return cli_->Post(path, items);
}
Result Client::Post(const std::string &path, const Headers &headers,
                           const MultipartFormDataItems &items) {
  return cli_->Post(path, headers, items);
}
Result Client::Post(const std::string &path, const Headers &headers,
                           const MultipartFormDataItems &items,
                           const std::string &boundary) {
  return cli_->Post(path, headers, items, boundary);
}
Result
Client::Post(const std::string &path, const Headers &headers,
             const MultipartFormDataItems &items,
             const MultipartFormDataProviderItems &provider_items) {
  return cli_->Post(path, headers, items, provider_items);
}
Result Client::Put(const std::string &path) { return cli_->Put(path); }
Result Client::Put(const std::string &path, const char *body,
                          size_t content_length,
                          const std::string &content_type) {
  return cli_->Put(path, body, content_length, content_type);
}
Result Client::Put(const std::string &path, const Headers &headers,
                          const char *body, size_t content_length,
                          const std::string &content_type) {
  return cli_->Put(path, headers, body, content_length, content_type);
}
Result Client::Put(const std::string &path, const Headers &headers,
                          const char *body, size_t content_length,
                          const std::string &content_type, Progress progress) {
  return cli_->Put(path, headers, body, content_length, content_type, progress);
}
Result Client::Put(const std::string &path, const std::string &body,
                          const std::string &content_type) {
  return cli_->Put(path, body, content_type);
}
Result Client::Put(const std::string &path, const std::string &body,
                          const std::string &content_type, Progress progress) {
  return cli_->Put(path, body, content_type, progress);
}
Result Client::Put(const std::string &path, const Headers &headers,
                          const std::string &body,
                          const std::string &content_type) {
  return cli_->Put(path, headers, body, content_type);
}
Result Client::Put(const std::string &path, const Headers &headers,
                          const std::string &body,
                          const std::string &content_type, Progress progress) {
  return cli_->Put(path, headers, body, content_type, progress);
}
Result Client::Put(const std::string &path, size_t content_length,
                          ContentProvider content_provider,
                          const std::string &content_type) {
  return cli_->Put(path, content_length, std::move(content_provider),
                   content_type);
}
Result Client::Put(const std::string &path,
                          ContentProviderWithoutLength content_provider,
                          const std::string &content_type) {
  return cli_->Put(path, std::move(content_provider), content_type);
}
Result Client::Put(const std::string &path, const Headers &headers,
                          size_t content_length,
                          ContentProvider content_provider,
                          const std::string &content_type) {
  return cli_->Put(path, headers, content_length, std::move(content_provider),
                   content_type);
}
Result Client::Put(const std::string &path, const Headers &headers,
                          ContentProviderWithoutLength content_provider,
                          const std::string &content_type) {
  return cli_->Put(path, headers, std::move(content_provider), content_type);
}
Result Client::Put(const std::string &path, const Params &params) {
  return cli_->Put(path, params);
}
Result Client::Put(const std::string &path, const Headers &headers,
                          const Params &params) {
  return cli_->Put(path, headers, params);
}
Result Client::Put(const std::string &path, const Headers &headers,
                          const Params &params, Progress progress) {
  return cli_->Put(path, headers, params, progress);
}
Result Client::Put(const std::string &path,
                          const MultipartFormDataItems &items) {
  return cli_->Put(path, items);
}
Result Client::Put(const std::string &path, const Headers &headers,
                          const MultipartFormDataItems &items) {
  return cli_->Put(path, headers, items);
}
Result Client::Put(const std::string &path, const Headers &headers,
                          const MultipartFormDataItems &items,
                          const std::string &boundary) {
  return cli_->Put(path, headers, items, boundary);
}
Result
Client::Put(const std::string &path, const Headers &headers,
            const MultipartFormDataItems &items,
            const MultipartFormDataProviderItems &provider_items) {
  return cli_->Put(path, headers, items, provider_items);
}
Result Client::Patch(const std::string &path) {
  return cli_->Patch(path);
}
Result Client::Patch(const std::string &path, const char *body,
                            size_t content_length,
                            const std::string &content_type) {
  return cli_->Patch(path, body, content_length, content_type);
}
Result Client::Patch(const std::string &path, const char *body,
                            size_t content_length,
                            const std::string &content_type,
                            Progress progress) {
  return cli_->Patch(path, body, content_length, content_type, progress);
}
Result Client::Patch(const std::string &path, const Headers &headers,
                            const char *body, size_t content_length,
                            const std::string &content_type) {
  return cli_->Patch(path, headers, body, content_length, content_type);
}
Result Client::Patch(const std::string &path, const Headers &headers,
                            const char *body, size_t content_length,
                            const std::string &content_type,
                            Progress progress) {
  return cli_->Patch(path, headers, body, content_length, content_type,
                     progress);
}
Result Client::Patch(const std::string &path, const std::string &body,
                            const std::string &content_type) {
  return cli_->Patch(path, body, content_type);
}
Result Client::Patch(const std::string &path, const std::string &body,
                            const std::string &content_type,
                            Progress progress) {
  return cli_->Patch(path, body, content_type, progress);
}
Result Client::Patch(const std::string &path, const Headers &headers,
                            const std::string &body,
                            const std::string &content_type) {
  return cli_->Patch(path, headers, body, content_type);
}
Result Client::Patch(const std::string &path, const Headers &headers,
                            const std::string &body,
                            const std::string &content_type,
                            Progress progress) {
  return cli_->Patch(path, headers, body, content_type, progress);
}
Result Client::Patch(const std::string &path, size_t content_length,
                            ContentProvider content_provider,
                            const std::string &content_type) {
  return cli_->Patch(path, content_length, std::move(content_provider),
                     content_type);
}
Result Client::Patch(const std::string &path,
                            ContentProviderWithoutLength content_provider,
                            const std::string &content_type) {
  return cli_->Patch(path, std::move(content_provider), content_type);
}
Result Client::Patch(const std::string &path, const Headers &headers,
                            size_t content_length,
                            ContentProvider content_provider,
                            const std::string &content_type) {
  return cli_->Patch(path, headers, content_length, std::move(content_provider),
                     content_type);
}
Result Client::Patch(const std::string &path, const Headers &headers,
                            ContentProviderWithoutLength content_provider,
                            const std::string &content_type) {
  return cli_->Patch(path, headers, std::move(content_provider), content_type);
}
Result Client::Delete(const std::string &path) {
  return cli_->Delete(path);
}
Result Client::Delete(const std::string &path, const Headers &headers) {
  return cli_->Delete(path, headers);
}
Result Client::Delete(const std::string &path, const char *body,
                             size_t content_length,
                             const std::string &content_type) {
  return cli_->Delete(path, body, content_length, content_type);
}
Result Client::Delete(const std::string &path, const char *body,
                             size_t content_length,
                             const std::string &content_type,
                             Progress progress) {
  return cli_->Delete(path, body, content_length, content_type, progress);
}
Result Client::Delete(const std::string &path, const Headers &headers,
                             const char *body, size_t content_length,
                             const std::string &content_type) {
  return cli_->Delete(path, headers, body, content_length, content_type);
}
Result Client::Delete(const std::string &path, const Headers &headers,
                             const char *body, size_t content_length,
                             const std::string &content_type,
                             Progress progress) {
  return cli_->Delete(path, headers, body, content_length, content_type,
                      progress);
}
Result Client::Delete(const std::string &path, const std::string &body,
                             const std::string &content_type) {
  return cli_->Delete(path, body, content_type);
}
Result Client::Delete(const std::string &path, const std::string &body,
                             const std::string &content_type,
                             Progress progress) {
  return cli_->Delete(path, body, content_type, progress);
}
Result Client::Delete(const std::string &path, const Headers &headers,
                             const std::string &body,
                             const std::string &content_type) {
  return cli_->Delete(path, headers, body, content_type);
}
Result Client::Delete(const std::string &path, const Headers &headers,
                             const std::string &body,
                             const std::string &content_type,
                             Progress progress) {
  return cli_->Delete(path, headers, body, content_type, progress);
}
Result Client::Options(const std::string &path) {
  return cli_->Options(path);
}
Result Client::Options(const std::string &path, const Headers &headers) {
  return cli_->Options(path, headers);
}

bool Client::send(Request &req, Response &res, Error &error) {
  return cli_->send(req, res, error);
}

Result Client::send(const Request &req) { return cli_->send(req); }

void Client::stop() { cli_->stop(); }

std::string Client::host() const { return cli_->host(); }

int Client::port() const { return cli_->port(); }

size_t Client::is_socket_open() const { return cli_->is_socket_open(); }

socket_t Client::socket() const { return cli_->socket(); }

void
Client::set_hostname_addr_map(std::map<std::string, std::string> addr_map) {
  cli_->set_hostname_addr_map(std::move(addr_map));
}

void Client::set_default_headers(Headers headers) {
  cli_->set_default_headers(std::move(headers));
}

void Client::set_header_writer(
    std::function<ssize_t(Stream &, Headers &)> const &writer) {
  cli_->set_header_writer(writer);
}

void Client::set_address_family(int family) {
  cli_->set_address_family(family);
}

void Client::set_tcp_nodelay(bool on) { cli_->set_tcp_nodelay(on); }

void Client::set_socket_options(SocketOptions socket_options) {
  cli_->set_socket_options(std::move(socket_options));
}

void Client::set_connection_timeout(time_t sec, time_t usec) {
  cli_->set_connection_timeout(sec, usec);
}

void Client::set_read_timeout(time_t sec, time_t usec) {
  cli_->set_read_timeout(sec, usec);
}

void Client::set_write_timeout(time_t sec, time_t usec) {
  cli_->set_write_timeout(sec, usec);
}

void Client::set_basic_auth(const std::string &username,
                                   const std::string &password) {
  cli_->set_basic_auth(username, password);
}
void Client::set_bearer_token_auth(const std::string &token) {
  cli_->set_bearer_token_auth(token);
}
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
void Client::set_digest_auth(const std::string &username,
                                    const std::string &password) {
  cli_->set_digest_auth(username, password);
}
#endif

void Client::set_keep_alive(bool on) { cli_->set_keep_alive(on); }
void Client::set_follow_location(bool on) {
  cli_->set_follow_location(on);
}

void Client::set_url_encode(bool on) { cli_->set_url_encode(on); }

void Client::set_compress(bool on) { cli_->set_compress(on); }

void Client::set_decompress(bool on) { cli_->set_decompress(on); }

void Client::set_interface(const std::string &intf) {
  cli_->set_interface(intf);
}

void Client::set_proxy(const std::string &host, int port) {
  cli_->set_proxy(host, port);
}
void Client::set_proxy_basic_auth(const std::string &username,
                                         const std::string &password) {
  cli_->set_proxy_basic_auth(username, password);
}
void Client::set_proxy_bearer_token_auth(const std::string &token) {
  cli_->set_proxy_bearer_token_auth(token);
}
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
void Client::set_proxy_digest_auth(const std::string &username,
                                          const std::string &password) {
  cli_->set_proxy_digest_auth(username, password);
}
#endif

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
void Client::enable_server_certificate_verification(bool enabled) {
  cli_->enable_server_certificate_verification(enabled);
}

void Client::enable_server_hostname_verification(bool enabled) {
  cli_->enable_server_hostname_verification(enabled);
}

void Client::set_server_certificate_verifier(
    std::function<SSLVerifierResponse(SSL *ssl)> verifier) {
  cli_->set_server_certificate_verifier(verifier);
}
#endif

void Client::set_logger(Logger logger) {
  cli_->set_logger(std::move(logger));
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
void Client::set_ca_cert_path(const std::string &ca_cert_file_path,
                                     const std::string &ca_cert_dir_path) {
  cli_->set_ca_cert_path(ca_cert_file_path, ca_cert_dir_path);
}

void Client::set_ca_cert_store(X509_STORE *ca_cert_store) {
  if (is_ssl_) {
    static_cast<SSLClient &>(*cli_).set_ca_cert_store(ca_cert_store);
  } else {
    cli_->set_ca_cert_store(ca_cert_store);
  }
}

void Client::load_ca_cert_store(const char *ca_cert, std::size_t size) {
  set_ca_cert_store(cli_->create_ca_cert_store(ca_cert, size));
}

long Client::get_openssl_verify_result() const {
  if (is_ssl_) {
    return static_cast<SSLClient &>(*cli_).get_openssl_verify_result();
  }
  return -1; // NOTE: -1 doesn't match any of X509_V_ERR_???
}

SSL_CTX *Client::ssl_context() const {
  if (is_ssl_) { return static_cast<SSLClient &>(*cli_).ssl_context(); }
  return nullptr;
}
#endif

} // namespace httplib
