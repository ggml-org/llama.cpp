// thermal_control.h
#ifndef THERMAL_CONTROL_H
#define THERMAL_CONTROL_H

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
#include <map>

#define GPU_TEMP_PATH "/sys/class/kgsl/kgsl-3d0/temp"
#define GPU_MIN_FREQ_PATH "/sys/class/kgsl/kgsl-3d0/devfreq/min_freq"
#define GPU_MAX_FREQ_PATH "/sys/class/kgsl/kgsl-3d0/devfreq/max_freq"
// #define CHECK_INTERVAL 5  // 10 토큰마다 한 번 체크

#define TARGET_TEMPERATURE 60

// 온도-주파수 매핑 (온도(°C) -> GPU frequency(Hz))
static std::map<int, int> temp_to_freq = {
    {60, 443000000},
    {65, 660000000}
};

// 🔥 Throughput monitoring CSV (llama.cpp에서 정의됨)
extern std::ofstream g_csv;


// Read GPU Temperature

double read_gpu_temp() {
    const std::string path = "/sys/class/kgsl/kgsl-3d0/temp";
    std::ifstream file(path);

    if (!file.is_open()) {
        fprintf(stderr, "Thermal: Cannot open %s: %s\n", GPU_MIN_FREQ_PATH, strerror(errno));
        return -1.0; // 오류 시 음수 반환
    }

    int temp_milli = 0;
    file >> temp_milli;
    file.close();

    // 밀리도 단위 → 섭씨
    return temp_milli;
}

// GPU frequency 설정 - echo처럼
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

// 🔥 CSV에 thermal 이벤트 기록
static inline void log_thermal_event(const char* event, double temp_celsius, int freq_hz = 0) {
    if (!g_csv.is_open()) return;
    
    auto ts = std::chrono::system_clock::now().time_since_epoch();
    auto ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ts).count();
    
    // CSV 형식: timestamp,-1,event_type,temp,freq
    g_csv << ts_ms << ",-1," << event << "," << temp_celsius << "," << freq_hz << "\n";
    g_csv.flush();
}

// 온도에 맞는 GPU frequency 찾기
static inline int get_freq_for_temp(int temp_celsius) {
    // 온도가 높을수록 낮은 주파수 사용
    // 매핑된 온도 중 현재 온도 이상인 가장 낮은 온도 찾기
    int target_freq = 0;
    for (auto& pair : temp_to_freq) {
        if (temp_celsius >= pair.first) {
            target_freq = pair.second;
        }
    }
    return target_freq;
}

// 온도 기반 thermal control
static inline void thermal_control_check() {
    static bool initialized = false;
    static bool finished = false;
    static int call_count = 0;
    static int current_freq = 0;
    
    // 초기화
    if (!initialized) {
        initialized = true;
        fprintf(stderr, "Thermal: Control initialized\n");
    }
    
    int temp_mc = read_gpu_temp();
    if (temp_mc <= 0) return;
    
    int temp_celsius = temp_mc / 1000;

    if (temp_celsius >= TARGET_TEMPERATURE && !finished){
        finished = true;
    
        // 온도에 맞는 주파수 찾기
        int target_freq = get_freq_for_temp(TARGET_TEMPERATURE);
        // int target_freq = TARGET_TEMPERATURE;
        
        // 주파수가 바뀌어야 할 때만 설정
        if (set_gpu_freq(target_freq)) {
            current_freq = target_freq;
            
            // 콘솔 출력
            fprintf(stderr, "Thermal: wants to set GPU temp to be %d°C, setting GPU freq to %d Hz\n", 
                    temp_celsius, target_freq);
            
            // CSV에 기록
            // log_thermal_event("FREQ_CHANGE", (double)temp_celsius, target_freq);
        } else {
            fprintf(stderr, "Thermal: Failed to set GPU frequency (try sudo)\n");
        }
        
    }
}

#endif // THERMAL_CONTROL_H