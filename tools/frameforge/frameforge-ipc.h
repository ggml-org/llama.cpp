#ifndef FRAMEFORGE_IPC_H
#define FRAMEFORGE_IPC_H

#include <string>
#include <functional>

namespace frameforge {

// IPC Server for Named Pipes
class IPCServer {
public:
    IPCServer(const std::string & pipe_name);
    ~IPCServer();
    
    // Start the IPC server
    bool start();
    
    // Stop the IPC server
    void stop();
    
    // Check if server is running
    bool is_running() const { return running_; }
    
    // Send a message through the pipe
    bool send_message(const std::string & message);
    
    // Set callback for received messages
    void set_message_callback(std::function<void(const std::string &)> callback);
    
private:
    std::string pipe_name_;
    bool running_;
    std::function<void(const std::string &)> message_callback_;
    
#ifdef _WIN32
    void * pipe_handle_;  // HANDLE on Windows
    void server_loop_windows();
#else
    int pipe_fd_;
    void server_loop_unix();
#endif
    
    // Platform-specific implementations
    bool create_pipe();
    void close_pipe();
};

// IPC Client for Named Pipes
class IPCClient {
public:
    IPCClient(const std::string & pipe_name);
    ~IPCClient();
    
    // Connect to the IPC server
    bool connect();
    
    // Disconnect from the server
    void disconnect();
    
    // Check if connected
    bool is_connected() const { return connected_; }
    
    // Send a message through the pipe
    bool send_message(const std::string & message);
    
    // Receive a message from the pipe (blocking)
    std::string receive_message();
    
private:
    std::string pipe_name_;
    bool connected_;
    
#ifdef _WIN32
    void * pipe_handle_;  // HANDLE on Windows
#else
    int pipe_fd_;
#endif
};

} // namespace frameforge

#endif // FRAMEFORGE_IPC_H
