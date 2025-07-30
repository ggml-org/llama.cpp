#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <fstream>
#include <sstream>
#include <functional>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#endif

class ProgressFeatureTest {
private:
    std::string server_url;
    int server_port;
    
    std::string create_long_prompt() {
        return R"(Please provide a comprehensive analysis of artificial intelligence and machine learning, including but not limited to:

1. Historical Development: Trace the evolution of AI from its early beginnings in the 1950s through the various AI winters and recent breakthroughs. Discuss key milestones such as the Dartmouth Conference, expert systems, neural networks, and deep learning.

2. Machine Learning Fundamentals: Explain the core concepts of supervised learning, unsupervised learning, and reinforcement learning. Describe different types of algorithms including decision trees, support vector machines, neural networks, and ensemble methods.

3. Deep Learning Revolution: Detail the resurgence of neural networks through deep learning, including convolutional neural networks (CNNs) for computer vision, recurrent neural networks (RNNs) and transformers for natural language processing, and generative adversarial networks (GANs).

4. Natural Language Processing: Discuss the evolution from rule-based systems to statistical methods to neural approaches. Cover topics like word embeddings, sequence-to-sequence models, attention mechanisms, and large language models like GPT, BERT, and their successors.

5. Computer Vision: Explore the development of computer vision from traditional image processing to deep learning approaches. Discuss object detection, image segmentation, face recognition, and recent advances in vision transformers.

6. Applications and Impact: Analyze how AI is transforming various industries including healthcare, finance, transportation, education, and entertainment. Discuss both the benefits and potential risks of AI deployment.

7. Ethical Considerations: Address important ethical issues such as bias in AI systems, privacy concerns, job displacement, and the need for responsible AI development and deployment.

8. Future Directions: Speculate on emerging trends in AI research, including multimodal AI, few-shot learning, explainable AI, and the pursuit of artificial general intelligence (AGI).

Please provide detailed explanations with specific examples and technical details where appropriate. This should be a thorough, academic-level analysis suitable for someone with a background in computer science or related fields.)";
    }
    
    std::string create_completion_request(const std::string& prompt, bool return_progress = true) {
        std::ostringstream oss;
        oss << "POST /completion HTTP/1.1\r\n";
        oss << "Host: localhost:" << server_port << "\r\n";
        oss << "Content-Type: application/json\r\n";
        oss << "Connection: close\r\n";
        
        std::string json_body = "{"
            "\"prompt\": \"" + escape_json_string(prompt) + "\","
            "\"stream\": true,"
            "\"return_progress\": " + (return_progress ? "true" : "false") + ","
            "\"max_tokens\": 20,"
            "\"temperature\": 0.7"
            "}";
        
        oss << "Content-Length: " << json_body.length() << "\r\n";
        oss << "\r\n";
        oss << json_body;
        
        return oss.str();
    }
    
    std::string create_chat_completion_request(const std::string& prompt, bool return_progress = true) {
        std::ostringstream oss;
        oss << "POST /v1/chat/completions HTTP/1.1\r\n";
        oss << "Host: localhost:" << server_port << "\r\n";
        oss << "Content-Type: application/json\r\n";
        oss << "Connection: close\r\n";
        
        std::string json_body = "{"
            "\"model\": \"test\","
            "\"messages\": [{\"role\": \"user\", \"content\": \"" + escape_json_string(prompt) + "\"}],"
            "\"stream\": true,"
            "\"return_progress\": " + (return_progress ? "true" : "false") + ","
            "\"max_tokens\": 20,"
            "\"temperature\": 0.7"
            "}";
        
        oss << "Content-Length: " << json_body.length() << "\r\n";
        oss << "\r\n";
        oss << json_body;
        
        return oss.str();
    }
    
    std::string escape_json_string(const std::string& str) {
        std::string result;
        for (char c : str) {
            if (c == '"' || c == '\\' || c == '\n' || c == '\r' || c == '\t') {
                result += '\\';
                switch (c) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '\n': result += 'n'; break;
                    case '\r': result += 'r'; break;
                    case '\t': result += 't'; break;
                }
            } else {
                result += c;
            }
        }
        return result;
    }
    
    bool send_http_request(const std::string& request, std::string& response) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            std::cerr << "Failed to create socket" << std::endl;
            return false;
        }
        
        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(server_port);
        server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        
        if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Failed to connect to server" << std::endl;
            close(sock);
            return false;
        }
        
        if (send(sock, request.c_str(), request.length(), 0) < 0) {
            std::cerr << "Failed to send request" << std::endl;
            close(sock);
            return false;
        }
        
        char buffer[4096];
        response.clear();
        
        while (true) {
            int bytes_received = recv(sock, buffer, sizeof(buffer) - 1, 0);
            if (bytes_received <= 0) break;
            
            buffer[bytes_received] = '\0';
            response += buffer;
        }
        
        close(sock);
        return true;
    }
    
    bool parse_progress_responses(const std::string& response, std::vector<std::string>& progress_responses, std::vector<std::string>& content_responses) {
        std::istringstream iss(response);
        std::string line;
        
        while (std::getline(iss, line)) {
            if (line.substr(0, 6) == "data: ") {
                std::string data = line.substr(6);
                if (data.find("\"n_prompt_tokens_processed\"") != std::string::npos || data.find("\"progress\"") != std::string::npos) {
                    progress_responses.push_back(data);
                } else if (data.find("\"content\"") != std::string::npos || data.find("\"choices\"") != std::string::npos) {
                    content_responses.push_back(data);
                }
            }
        }
        
        return true;
    }
    
    bool check_progress_completion(const std::vector<std::string>& progress_responses) {
        if (progress_responses.empty()) {
            return false;
        }
        
        // Check if the last progress response shows 100% completion
        std::string last_response = progress_responses.back();
        return last_response.find("\"progress\":1.0") != std::string::npos;
    }

public:
    ProgressFeatureTest(int port = 8081) : server_port(port) {}
    
    bool test_completion_endpoint_progress() {
        std::cout << "\n=== Testing /completion endpoint progress ===" << std::endl;
        
        std::string prompt = create_long_prompt();
        std::string request = create_completion_request(prompt, true);
        std::string response;
        
        if (!send_http_request(request, response)) {
            std::cout << "Failed to send request" << std::endl;
            return false;
        }
        
        std::vector<std::string> progress_responses, content_responses;
        parse_progress_responses(response, progress_responses, content_responses);
        
        std::cout << "Received " << progress_responses.size() << " progress responses" << std::endl;
        std::cout << "Received " << content_responses.size() << " content responses" << std::endl;
        
        if (check_progress_completion(progress_responses)) {
            std::cout << "Progress reached 100% as expected" << std::endl;
            return true;
        } else {
            std::cout << "Progress did not reach 100%" << std::endl;
            return false;
        }
    }
    
    bool test_chat_completion_endpoint_progress() {
        std::cout << "\n=== Testing /v1/chat/completions endpoint progress ===" << std::endl;
        
        std::string prompt = create_long_prompt();
        std::string request = create_chat_completion_request(prompt, true);
        std::string response;
        
        if (!send_http_request(request, response)) {
            std::cout << "Failed to send request" << std::endl;
            return false;
        }
        
        std::vector<std::string> progress_responses, content_responses;
        parse_progress_responses(response, progress_responses, content_responses);
        
        std::cout << "Received " << progress_responses.size() << " progress responses" << std::endl;
        std::cout << "Received " << content_responses.size() << " content responses" << std::endl;
        
        if (check_progress_completion(progress_responses)) {
            std::cout << "Progress reached 100% as expected" << std::endl;
            return true;
        } else {
            std::cout << "Progress did not reach 100%" << std::endl;
            return false;
        }
    }
    
    bool test_progress_disabled() {
        std::cout << "\n=== Testing progress disabled ===" << std::endl;
        
        std::string prompt = create_long_prompt();
        std::string request = create_completion_request(prompt, false);
        std::string response;
        
        if (!send_http_request(request, response)) {
            std::cout << "Failed to send request" << std::endl;
            return false;
        }
        
        std::vector<std::string> progress_responses, content_responses;
        parse_progress_responses(response, progress_responses, content_responses);
        
        std::cout << "Received " << progress_responses.size() << " progress responses (should be 0)" << std::endl;
        std::cout << "Received " << content_responses.size() << " content responses" << std::endl;
        
        if (progress_responses.empty()) {
            std::cout << "No progress responses when disabled, as expected" << std::endl;
            return true;
        } else {
            std::cout << "Progress responses received when disabled" << std::endl;
            return false;
        }
    }
    
    bool run_all_tests() {
        std::cout << "Starting Progress Feature Tests" << std::endl;
        std::cout << "==================================================" << std::endl;
        
        std::vector<std::pair<std::string, std::function<bool()>>> tests = {
            {"Completion endpoint progress", [this]() { return test_completion_endpoint_progress(); }},
            {"Chat completion endpoint progress", [this]() { return test_chat_completion_endpoint_progress(); }},
            {"Progress disabled", [this]() { return test_progress_disabled(); }},
        };
        
        int passed = 0;
        int total = tests.size();
        
        for (const auto& test : tests) {
            std::cout << "\n==================== " << test.first << " ====================" << std::endl;
            if (test.second()) {
                std::cout << "PASSED" << std::endl;
                passed++;
            } else {
                std::cout << "FAILED" << std::endl;
            }
        }
        
        std::cout << "\n==================================================" << std::endl;
        std::cout << "Test Results: " << passed << "/" << total << " tests passed" << std::endl;
        
        return passed == total;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 1) {
        std::cout << "Usage: " << argv[0] << std::endl;
        std::cout << "Make sure the server is running on localhost:8081" << std::endl;
        return 1;
    }
    
    ProgressFeatureTest tester;
    bool success = tester.run_all_tests();
    
    if (success) {
        std::cout << "\nAll tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome tests failed!" << std::endl;
        return 1;
    }
} 