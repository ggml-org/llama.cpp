// thermal_control_pid.h
#ifndef THERMAL_CONTROL_PID_H
#define THERMAL_CONTROL_PID_H

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <fstream>       
#include <chrono>
#include <errno.h>
#include <cmath>
#include <algorithm>

#define GPU_TEMP_PATH "/sys/class/kgsl/kgsl-3d0/temp"
#define GPU_MIN_FREQ_PATH "/sys/class/kgsl/kgsl-3d0/devfreq/min_freq"
#define GPU_MAX_FREQ_PATH "/sys/class/kgsl/kgsl-3d0/devfreq/max_freq"

// PID 설정
#define TARGET_TEMPERATURE 60.0  // 목표 온도 (°C)
#define CHECK_INTERVAL_MS 500    // 체크 간격 (ms) - 더 자주 체크할수록 더 정확함

// PID 게인 (튜닝 필요)
#define KP 0.8   // Proportional gain
#define KI 0.1   // Integral gain  
#define KD 0.5   // Derivative gain

// CSV 로깅용 파일 스트림
static std::ofstream g_temp_csv;  // pid_temp.csv
static std::ofstream g_freq_csv;  // pid_freq.csv

// Snapdragon Elite GPU 주파수 레벨 (Hz)
static const int GPU_FREQ_LEVELS[] = {
    900000000,  // 0
    832000000,  // 1
    734000000,  // 2
    660000000,  // 3
    607000000,  // 4
    525000000,  // 5
    443000000,  // 6
    389000000,  // 7
    342000000,  // 8
    222000000,  // 9
    160000000   // 10
};
static const int NUM_FREQ_LEVELS = sizeof(GPU_FREQ_LEVELS) / sizeof(GPU_FREQ_LEVELS[0]);

// PID 상태 구조체
struct PIDState {
    double integral;           // 오차 적분값
    double prev_error;         // 이전 오차
    int current_freq_index;    // 현재 주파수 인덱스
    bool active;               // PID 활성화 여부
    std::chrono::steady_clock::time_point last_check;
};

static PIDState g_pid_state = {0.0, 0.0, 0, false};

// Read GPU Temperature
static inline double read_gpu_temp() {
    const std::string path = GPU_TEMP_PATH;
    std::ifstream file(path);

    if (!file.is_open()) {
        fprintf(stderr, "Thermal: Cannot open %s: %s\n", path.c_str(), strerror(errno));
        return -1.0;
    }

    int temp_milli = 0;
    file >> temp_milli;
    file.close();

    // 밀리도 → 섭씨
    return temp_milli / 1000.0;
}

// GPU frequency 설정
static inline bool set_gpu_freq(int freq_hz) {
    char freq_str[32];
    snprintf(freq_str, sizeof(freq_str), "%d\n", freq_hz);
    
    bool success = false;
    
    // min_freq 설정
    int fd_min = open(GPU_MIN_FREQ_PATH, O_WRONLY | O_TRUNC);
    if (fd_min >= 0) {
        write(fd_min, freq_str, strlen(freq_str));
        close(fd_min);
        success = true;
    } else {
        fprintf(stderr, "Thermal: Cannot open %s: %s\n", GPU_MIN_FREQ_PATH, strerror(errno));
    }
    
    // max_freq 설정
    int fd_max = open(GPU_MAX_FREQ_PATH, O_WRONLY | O_TRUNC);
    if (fd_max >= 0) {
        write(fd_max, freq_str, strlen(freq_str));
        close(fd_max);
    } else {
        fprintf(stderr, "Thermal: Cannot open %s: %s\n", GPU_MAX_FREQ_PATH, strerror(errno));
    }
    
    return success;
}

// CSV 파일 초기화
static inline void init_thermal_csv() {
    g_temp_csv.open("pid_temp.csv", std::ios::out | std::ios::trunc);
    if (g_temp_csv.is_open()) {
        g_temp_csv << "timestamp_sec,temperature_celsius\n";
        g_temp_csv.flush();
        fprintf(stderr, "Thermal: Logging temperature to pid_temp.csv\n");
    }
    
    g_freq_csv.open("pid_freq.csv", std::ios::out | std::ios::trunc);
    if (g_freq_csv.is_open()) {
        g_freq_csv << "timestamp_sec,temperature_celsius,freq_index_from,freq_index_to,frequency_hz,frequency_mhz,error\n";
        g_freq_csv.flush();
        fprintf(stderr, "Thermal: Logging frequency changes to pid_freq.csv\n");
    }
}

// CSV 파일 닫기
static inline void close_thermal_csv() {
    if (g_temp_csv.is_open()) {
        g_temp_csv.close();
        fprintf(stderr, "Thermal: Closed pid_temp.csv\n");
    }
    if (g_freq_csv.is_open()) {
        g_freq_csv.close();
        fprintf(stderr, "Thermal: Closed pid_freq.csv\n");
    }
}


// PID 계산 함수
static inline int compute_pid(double current_temp, double dt) {
    // 오차 계산 (목표 - 현재)
    // 양수면 온도가 낮음 → 주파수 올려야 함
    // 음수면 온도가 높음 → 주파수 낮춰야 함
    double error = TARGET_TEMPERATURE - current_temp;
    
    // P: 비례 항 (현재 오차)
    double p_term = KP * error;
    
    // I: 적분 항 (누적 오차)
    g_pid_state.integral += error * dt;
    
    // Anti-windup: integral 값 제한 (너무 커지는 것 방지)
    const double INTEGRAL_MAX = 50.0;
    g_pid_state.integral = std::max(-INTEGRAL_MAX, std::min(INTEGRAL_MAX, g_pid_state.integral));
    double i_term = KI * g_pid_state.integral;
    
    // D: 미분 항 (오차 변화율)
    double derivative = (error - g_pid_state.prev_error) / dt;
    double d_term = KD * derivative;
    
    // PID 출력 계산
    double pid_output = p_term + i_term + d_term;
    
    // 이전 오차 저장
    g_pid_state.prev_error = error;
    
    // PID 출력을 주파수 인덱스 변화량으로 변환
    // 양수 → 주파수 올림 (인덱스 감소)
    // 음수 → 주파수 낮춤 (인덱스 증가)
    int index_delta = 0;
    
    if (pid_output > 2.0) {
        index_delta = -2;  // 빠르게 주파수 올림
    } else if (pid_output > 0.5) {
        index_delta = -1;  // 주파수 올림
    } else if (pid_output < -2.0) {
        index_delta = 2;   // 빠르게 주파수 낮춤
    } else if (pid_output < -0.5) {
        index_delta = 1;   // 주파수 낮춤
    }
    // -0.5 ~ 0.5 사이면 유지 (dead zone)
    
    return index_delta;
}

// PID 기반 thermal control
static inline void thermal_control_check() {
    static bool initialized = false;
    static auto start_time = std::chrono::steady_clock::now();
    static auto last_temp_log = std::chrono::steady_clock::now();
    
    // 초기화
    if (!initialized) {
        initialized = true;
        g_pid_state.current_freq_index = 0;  // 최대 주파수에서 시작
        g_pid_state.last_check = std::chrono::steady_clock::now();
        start_time = std::chrono::steady_clock::now();
        last_temp_log = std::chrono::steady_clock::now();
        
        // CSV 로깅 초기화
        init_thermal_csv();
        
        fprintf(stderr, "Thermal: PID Controller initialized\n");
        fprintf(stderr, "Thermal: Target temperature: %.1f°C\n", TARGET_TEMPERATURE);
        fprintf(stderr, "Thermal: PID gains - Kp:%.2f Ki:%.2f Kd:%.2f\n", KP, KI, KD);
    }
    
    // 현재 시간
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_pid_state.last_check);
    
    // 온도 읽기
    double temp = read_gpu_temp();
    if (temp <= 0) return;
    
    // 🔥 매초마다 온도를 pid_temp.csv에 기록
    auto time_since_last_temp_log = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_temp_log);
    if (time_since_last_temp_log.count() >= 1000) {  // 1초마다
        if (g_temp_csv.is_open()) {
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            g_temp_csv << elapsed_sec << "," << temp << "\n";
            g_temp_csv.flush();
        }
        last_temp_log = now;
    }
    
    // 체크 간격이 안 됐으면 리턴
    if (elapsed.count() < CHECK_INTERVAL_MS) {
        return;
    }
    
    // PID 활성화 조건: 목표 온도 근처 또는 이미 활성화됨
    if (!g_pid_state.active && temp >= TARGET_TEMPERATURE - 5.0) {
        g_pid_state.active = true;
        fprintf(stderr, "Thermal: PID activated at %.1f°C\n", temp);
    }
    
    if (!g_pid_state.active) {
        g_pid_state.last_check = now;
        return;  // 아직 PID 작동 안 함
    }
    
    // dt 계산 (초 단위)
    double dt = elapsed.count() / 1000.0;
    
    // PID 계산
    int index_delta = compute_pid(temp, dt);
    
    // 주파수 인덱스 업데이트
    int new_index = g_pid_state.current_freq_index + index_delta;
    new_index = std::max(0, std::min(NUM_FREQ_LEVELS - 1, new_index));
    
    // 주파수가 바뀌어야 하면 설정
    if (new_index != g_pid_state.current_freq_index) {
        int new_freq = GPU_FREQ_LEVELS[new_index];
        
        if (set_gpu_freq(new_freq)) {
            // 디버그 출력
            double error = TARGET_TEMPERATURE - temp;
            fprintf(stderr, "Thermal: T=%.1f°C (error=%.1f) → Freq[%d->%d] = %d Hz (%.0f MHz)\n",
                    temp, error, 
                    g_pid_state.current_freq_index, new_index,
                    new_freq, new_freq / 1e6);
            
            // 🔥 주파수 변경을 pid_freq.csv에 기록
            if (g_freq_csv.is_open()) {
                auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                g_freq_csv << elapsed_sec << ","
                          << temp << ","
                          << g_pid_state.current_freq_index << ","
                          << new_index << ","
                          << new_freq << ","
                          << (new_freq / 1000000) << ","
                          << error << "\n";
                g_freq_csv.flush();
            }
            
            g_pid_state.current_freq_index = new_index;
        } else {
            fprintf(stderr, "Thermal: Failed to set GPU frequency (try sudo)\n");
        }
    } else {
        // 주파수 안 바뀜 (안정 상태)
        static int stable_count = 0;
        stable_count++;
        
        if (stable_count % 20 == 0) {  // 20회마다 한 번 출력
            fprintf(stderr, "Thermal: Stable at T=%.1f°C, Freq=%d MHz\n",
                    temp, GPU_FREQ_LEVELS[g_pid_state.current_freq_index] / 1000000);
        }
    }
    
    g_pid_state.last_check = now;
}

// PID 리셋 (필요시)
static inline void reset_pid() {
    g_pid_state.integral = 0.0;
    g_pid_state.prev_error = 0.0;
    g_pid_state.current_freq_index = 0;
    g_pid_state.active = false;
    fprintf(stderr, "Thermal: PID reset\n");
}

// 현재 PID 상태 출력 (디버깅용)
static inline void print_pid_status() {
    double temp = read_gpu_temp();
    int freq = GPU_FREQ_LEVELS[g_pid_state.current_freq_index];
    
    fprintf(stderr, "\n=== PID Status ===\n");
    fprintf(stderr, "Current Temp: %.1f°C\n", temp);
    fprintf(stderr, "Target Temp:  %.1f°C\n", TARGET_TEMPERATURE);
    fprintf(stderr, "Error:        %.1f°C\n", TARGET_TEMPERATURE - temp);
    fprintf(stderr, "Integral:     %.2f\n", g_pid_state.integral);
    fprintf(stderr, "Current Freq: %d MHz (Level %d/%d)\n", 
            freq / 1000000, g_pid_state.current_freq_index, NUM_FREQ_LEVELS - 1);
    fprintf(stderr, "PID Active:   %s\n", g_pid_state.active ? "Yes" : "No");
    fprintf(stderr, "==================\n\n");
}

#endif // THERMAL_CONTROL_PID_H