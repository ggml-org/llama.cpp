#include "frameforge-ipc.h"

#include <cstdint>
#include <cstring>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#endif

namespace frameforge {

// IPC constants
constexpr size_t MAX_MESSAGE_SIZE   = 1024 * 1024;  // 1MB
constexpr int    MAX_PIPE_INSTANCES = 1;
constexpr int    PIPE_BUFFER_SIZE   = 4096;

// IPCServer implementation

IPCServer::IPCServer(const std::string & pipe_name)
    : pipe_name_(pipe_name)
    , running_(false)
    , message_callback_(nullptr)
#ifdef _WIN32
    , pipe_handle_(INVALID_HANDLE_VALUE)
#else
    , pipe_fd_(-1)
#endif
{
}

IPCServer::~IPCServer() {
    stop();
}

bool IPCServer::start() {
    if (running_) {
        return false;
    }
    
    if (!create_pipe()) {
        return false;
    }
    
    running_ = true;
    return true;
}

void IPCServer::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    close_pipe();
}

bool IPCServer::send_message(const std::string & message) {
    if (!running_) {
        return false;
    }
    
#ifdef _WIN32
    if (pipe_handle_ == INVALID_HANDLE_VALUE) {
        return false;
    }
    
    DWORD bytes_written;
    uint32_t msg_size = static_cast<uint32_t>(message.size());
    
    // Write message size first
    if (!WriteFile(pipe_handle_, &msg_size, sizeof(msg_size), &bytes_written, NULL)) {
        return false;
    }
    
    // Write message data
    if (!WriteFile(pipe_handle_, message.c_str(), msg_size, &bytes_written, NULL)) {
        return false;
    }
    
    FlushFileBuffers(pipe_handle_);
    return true;
#else
    if (pipe_fd_ < 0) {
        return false;
    }
    
    uint32_t msg_size = static_cast<uint32_t>(message.size());
    
    // Write message size first
    if (write(pipe_fd_, &msg_size, sizeof(msg_size)) != sizeof(msg_size)) {
        return false;
    }
    
    // Write message data
    if (write(pipe_fd_, message.c_str(), msg_size) != static_cast<ssize_t>(msg_size)) {
        return false;
    }
    
    return true;
#endif
}

void IPCServer::set_message_callback(std::function<void(const std::string &)> callback) {
    message_callback_ = callback;
}

#ifdef _WIN32

bool IPCServer::create_pipe() {
    std::string pipe_path = "\\\\.\\pipe\\" + pipe_name_;

    pipe_handle_ = CreateNamedPipeA(pipe_path.c_str(), PIPE_ACCESS_DUPLEX,
                                    PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT, MAX_PIPE_INSTANCES,
                                    PIPE_BUFFER_SIZE,  // out buffer size
                                    PIPE_BUFFER_SIZE,  // in buffer size
                                    0,                 // default timeout
                                    NULL);

    if (pipe_handle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to create named pipe: " << GetLastError() << std::endl;
        return false;
    }
    
    return true;
}

void IPCServer::close_pipe() {
    if (pipe_handle_ != INVALID_HANDLE_VALUE) {
        DisconnectNamedPipe(pipe_handle_);
        CloseHandle(pipe_handle_);
        pipe_handle_ = INVALID_HANDLE_VALUE;
    }
}

#else

bool IPCServer::create_pipe() {
    std::string pipe_path = "/tmp/" + pipe_name_;
    
    // Remove existing pipe if it exists
    unlink(pipe_path.c_str());
    
    // Create FIFO (named pipe)
    if (mkfifo(pipe_path.c_str(), 0666) != 0) {
        std::cerr << "Failed to create named pipe: " << strerror(errno) << std::endl;
        return false;
    }
    
    // Open pipe for reading and writing (non-blocking initially)
    pipe_fd_ = open(pipe_path.c_str(), O_RDWR | O_NONBLOCK);
    if (pipe_fd_ < 0) {
        std::cerr << "Failed to open named pipe: " << strerror(errno) << std::endl;
        unlink(pipe_path.c_str());
        return false;
    }
    
    return true;
}

void IPCServer::close_pipe() {
    if (pipe_fd_ >= 0) {
        close(pipe_fd_);
        pipe_fd_ = -1;
        
        std::string pipe_path = "/tmp/" + pipe_name_;
        unlink(pipe_path.c_str());
    }
}

#endif

// IPCClient implementation

IPCClient::IPCClient(const std::string & pipe_name)
    : pipe_name_(pipe_name)
    , connected_(false)
#ifdef _WIN32
    , pipe_handle_(INVALID_HANDLE_VALUE)
#else
    , pipe_fd_(-1)
#endif
{
}

IPCClient::~IPCClient() {
    disconnect();
}

bool IPCClient::connect() {
    if (connected_) {
        return false;
    }
    
#ifdef _WIN32
    std::string pipe_path = "\\\\.\\pipe\\" + pipe_name_;
    
    // Try to connect to the pipe
    pipe_handle_ = CreateFileA(
        pipe_path.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        0,
        NULL
    );
    
    if (pipe_handle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to connect to pipe: " << GetLastError() << std::endl;
        return false;
    }
    
    // Set pipe to message-read mode
    DWORD mode = PIPE_READMODE_MESSAGE;
    if (!SetNamedPipeHandleState(pipe_handle_, &mode, NULL, NULL)) {
        std::cerr << "Failed to set pipe mode: " << GetLastError() << std::endl;
        CloseHandle(pipe_handle_);
        pipe_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }
#else
    std::string pipe_path = "/tmp/" + pipe_name_;
    
    pipe_fd_ = open(pipe_path.c_str(), O_RDWR);
    if (pipe_fd_ < 0) {
        std::cerr << "Failed to connect to pipe: " << strerror(errno) << std::endl;
        return false;
    }
#endif
    
    connected_ = true;
    return true;
}

void IPCClient::disconnect() {
    if (!connected_) {
        return;
    }
    
#ifdef _WIN32
    if (pipe_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(pipe_handle_);
        pipe_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (pipe_fd_ >= 0) {
        close(pipe_fd_);
        pipe_fd_ = -1;
    }
#endif
    
    connected_ = false;
}

bool IPCClient::send_message(const std::string & message) {
    if (!connected_) {
        return false;
    }
    
#ifdef _WIN32
    if (pipe_handle_ == INVALID_HANDLE_VALUE) {
        return false;
    }
    
    DWORD bytes_written;
    uint32_t msg_size = static_cast<uint32_t>(message.size());
    
    // Write message size first
    if (!WriteFile(pipe_handle_, &msg_size, sizeof(msg_size), &bytes_written, NULL)) {
        return false;
    }
    
    // Write message data
    if (!WriteFile(pipe_handle_, message.c_str(), msg_size, &bytes_written, NULL)) {
        return false;
    }
    
    return true;
#else
    if (pipe_fd_ < 0) {
        return false;
    }
    
    uint32_t msg_size = static_cast<uint32_t>(message.size());
    
    // Write message size first
    if (write(pipe_fd_, &msg_size, sizeof(msg_size)) != sizeof(msg_size)) {
        return false;
    }
    
    // Write message data
    if (write(pipe_fd_, message.c_str(), msg_size) != static_cast<ssize_t>(msg_size)) {
        return false;
    }
    
    return true;
#endif
}

std::string IPCClient::receive_message() {
    if (!connected_) {
        return "";
    }
    
#ifdef _WIN32
    if (pipe_handle_ == INVALID_HANDLE_VALUE) {
        return "";
    }
    
    DWORD bytes_read;
    uint32_t msg_size = 0;
    
    // Read message size first
    if (!ReadFile(pipe_handle_, &msg_size, sizeof(msg_size), &bytes_read, NULL)) {
        return "";
    }

    if (msg_size == 0 || msg_size > MAX_MESSAGE_SIZE) {
        return "";
    }

    // Read message data
    std::string message(msg_size, '\0');
    if (!ReadFile(pipe_handle_, &message[0], msg_size, &bytes_read, NULL)) {
        return "";
    }
    
    return message;
#else
    if (pipe_fd_ < 0) {
        return "";
    }
    
    uint32_t msg_size = 0;
    
    // Read message size first
    if (read(pipe_fd_, &msg_size, sizeof(msg_size)) != sizeof(msg_size)) {
        return "";
    }

    if (msg_size == 0 || msg_size > MAX_MESSAGE_SIZE) {
        return "";
    }

    // Read message data
    std::string message(msg_size, '\0');
    if (read(pipe_fd_, &message[0], msg_size) != static_cast<ssize_t>(msg_size)) {
        return "";
    }
    
    return message;
#endif
}

} // namespace frameforge
