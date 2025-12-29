#pragma once

#include "log.h"

#include <string>
#include <functional>
#include <thread>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#endif

namespace semantic_server {

// Callback for when a message is received via IPC
using ipc_message_callback = std::function<void(const std::string &)>;

class IPCHandler {
public:
    IPCHandler(const std::string & pipe_name)
        : pipe_name(pipe_name), running(false) {
    }
    
    ~IPCHandler() {
        stop();
    }
    
    // Start the IPC server (listens for incoming messages)
    bool start(ipc_message_callback callback) {
        if (running.exchange(true)) {
            return false; // Already running
        }
        
        message_callback = callback;
        
#ifdef _WIN32
        return start_windows();
#else
        return start_unix();
#endif
    }
    
    // Stop the IPC server
    void stop() {
        if (!running.exchange(false)) {
            return; // Not running
        }
        
        // Signal the thread to stop
        if (listener_thread.joinable()) {
            listener_thread.join();
        }
        
#ifdef _WIN32
        if (pipe_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(pipe_handle);
            pipe_handle = INVALID_HANDLE_VALUE;
        }
#else
        if (pipe_fd >= 0) {
            close(pipe_fd);
            pipe_fd = -1;
        }
        // Clean up the named pipe file
        unlink(pipe_name.c_str());
#endif
    }
    
    // Send a message through the pipe (for client mode or bidirectional communication)
    bool send_message(const std::string & message) {
#ifdef _WIN32
        return send_message_windows(message);
#else
        return send_message_unix(message);
#endif
    }

private:
    std::string pipe_name;
    std::atomic<bool> running;
    ipc_message_callback message_callback;
    std::thread listener_thread;
    
#ifdef _WIN32
    HANDLE pipe_handle = INVALID_HANDLE_VALUE;
    
    bool start_windows() {
        listener_thread = std::thread([this]() {
            std::string full_pipe_name = "\\\\.\\pipe\\" + pipe_name;
            
            while (running) {
                // Create named pipe
                pipe_handle = CreateNamedPipeA(
                    full_pipe_name.c_str(),
                    PIPE_ACCESS_DUPLEX,
                    PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
                    1,                  // max instances
                    4096,               // output buffer size
                    4096,               // input buffer size
                    0,                  // default timeout
                    NULL
                );
                
                if (pipe_handle == INVALID_HANDLE_VALUE) {
                    LOG_ERR("Failed to create named pipe: %d\n", GetLastError());
                    return;
                }
                
                LOG_INF("Waiting for client connection on pipe: %s\n", full_pipe_name.c_str());
                
                // Wait for client to connect
                BOOL connected = ConnectNamedPipe(pipe_handle, NULL) ?
                    TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
                
                if (!connected) {
                    CloseHandle(pipe_handle);
                    continue;
                }
                
                LOG_INF("Client connected to pipe\n");
                
                // Read messages
                while (running) {
                    char buffer[4096];
                    DWORD bytes_read;
                    
                    BOOL success = ReadFile(
                        pipe_handle,
                        buffer,
                        sizeof(buffer) - 1,
                        &bytes_read,
                        NULL
                    );
                    
                    if (!success || bytes_read == 0) {
                        break;
                    }
                    
                    buffer[bytes_read] = '\0';
                    std::string message(buffer, bytes_read);
                    
                    if (message_callback) {
                        message_callback(message);
                    }
                }
                
                DisconnectNamedPipe(pipe_handle);
                CloseHandle(pipe_handle);
            }
        });
        
        return true;
    }
    
    bool send_message_windows(const std::string & message) {
        if (pipe_handle == INVALID_HANDLE_VALUE) {
            return false;
        }
        
        DWORD bytes_written;
        BOOL success = WriteFile(
            pipe_handle,
            message.c_str(),
            message.size(),
            &bytes_written,
            NULL
        );
        
        return success && bytes_written == message.size();
    }
#else
    int pipe_fd = -1;
    
    bool start_unix() {
        // Create named pipe (FIFO)
        unlink(pipe_name.c_str()); // Remove if exists
        
        if (mkfifo(pipe_name.c_str(), 0666) == -1) {
            LOG_ERR("Failed to create named pipe: %s\n", strerror(errno));
            return false;
        }
        
        LOG_INF("Created named pipe: %s\n", pipe_name.c_str());
        
        listener_thread = std::thread([this]() {
            while (running) {
                // Open pipe for reading (blocking until a writer connects)
                pipe_fd = open(pipe_name.c_str(), O_RDONLY);
                if (pipe_fd < 0) {
                    if (running) {
                        LOG_ERR("Failed to open pipe for reading: %s\n", strerror(errno));
                    }
                    break;
                }
                
                LOG_INF("Client connected to pipe\n");
                
                // Read messages
                while (running) {
                    char buffer[4096];
                    ssize_t bytes_read = read(pipe_fd, buffer, sizeof(buffer) - 1);
                    
                    if (bytes_read <= 0) {
                        break;
                    }
                    
                    buffer[bytes_read] = '\0';
                    std::string message(buffer, bytes_read);
                    
                    if (message_callback) {
                        message_callback(message);
                    }
                }
                
                close(pipe_fd);
                pipe_fd = -1;
            }
        });
        
        return true;
    }
    
    bool send_message_unix(const std::string & message) {
        // Open pipe for writing
        int fd = open(pipe_name.c_str(), O_WRONLY | O_NONBLOCK);
        if (fd < 0) {
            LOG_ERR("Failed to open pipe for writing: %s\n", strerror(errno));
            return false;
        }
        
        ssize_t bytes_written = write(fd, message.c_str(), message.size());
        close(fd);
        
        return bytes_written == (ssize_t)message.size();
    }
#endif
};

} // namespace semantic_server
