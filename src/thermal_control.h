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
    {70, 660000000}
};

// 🔥 Throughput monitoring CSV (llama.cpp에서 정의됨)
extern std::ofstream g_csv;

// GPU 온도 읽기 (millidegree C)
static inline int read_gpu_temp() {
    int fd = open(GPU_TEMP_PATH, O_RDONLY);
    if (fd < 0) return -1;
    
    char buf[32];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    
    if (n <= 0) return -1;
    buf[n] = '\0';
    
    // Fast parse
    int temp = 0;
    for (int i = 0; i < n && buf[i] >= '0' && buf[i] <= '9'; i++) {
        temp = temp * 10 + (buf[i] - '0');
    }
    return temp;
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

// CPU frequency 설정 - echo처럼 (주석 처리)
/*
static inline bool set_cpu_freq(int freq_khz) {
    DIR *dir = opendir("/sys/devices/system/cpu");
    if (!dir) {
        fprintf(stderr, "Thermal: Cannot open /sys/devices/system/cpu: %s\n", strerror(errno));
        return false;
    }
    
    char freq_str[32];
    snprintf(freq_str, sizeof(freq_str), "%d\n", freq_khz);
    
    bool success = false;
    struct dirent *entry;
    
    while ((entry = readdir(dir)) != NULL) {
        // cpu0, cpu1, ... 찾기
        if (strncmp(entry->d_name, "cpu", 3) != 0) continue;
        char c = entry->d_name[3];
        if (c < '0' || c > '9') continue;
        
        char path[256];
        
        // scaling_max_freq 먼저 (min보다 크거나 같아야 함)
        snprintf(path, sizeof(path), 
                 "/sys/devices/system/cpu/%s/cpufreq/scaling_max_freq", 
                 entry->d_name);
        int fd = open(path, O_WRONLY | O_TRUNC);
        if (fd >= 0) {
            write(fd, freq_str, strlen(freq_str));
            close(fd);
            success = true;
        }
        
        // scaling_min_freq
        snprintf(path, sizeof(path), 
                 "/sys/devices/system/cpu/%s/cpufreq/scaling_min_freq", 
                 entry->d_name);
        fd = open(path, O_WRONLY | O_TRUNC);
        if (fd >= 0) {
            write(fd, freq_str, strlen(freq_str));
            close(fd);
        }
    }
    
    closedir(dir);
    return success;
}
*/

// GPU 온도 읽기 (millidegree C) - FD 재사용
static inline int read_gpu_temp_fast(int fd) {
    if (fd < 0) return -1;
    
    if (lseek(fd, 0, SEEK_SET) < 0) return -1;
    
    char buf[32];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    if (n <= 0) return -1;
    buf[n] = '\0';
    
    // Fast parse
    int temp = 0;
    for (int i = 0; i < n && buf[i] >= '0' && buf[i] <= '9'; i++) {
        temp = temp * 10 + (buf[i] - '0');
    }
    return temp;
}

// CPU frequency 설정 - 캐시된 FD 사용
static inline bool set_cpu_freq_cached(int *fd_cache, int num_fds, int freq_khz) {
    if (num_fds == 0) return false;
    
    char freq_str[16];
    int len = snprintf(freq_str, sizeof(freq_str), "%d", freq_khz);
    
    bool success = true;
    for (int i = 0; i < num_fds; i += 2) {
        // max 먼저
        if (fd_cache[i + 1] >= 0) {
            if (lseek(fd_cache[i + 1], 0, SEEK_SET) >= 0) {
                write(fd_cache[i + 1], freq_str, len);
            } else {
                success = false;
            }
        }
        // min 나중에
        if (fd_cache[i] >= 0) {
            if (lseek(fd_cache[i], 0, SEEK_SET) >= 0) {
                write(fd_cache[i], freq_str, len);
            } else {
                success = false;
            }
        }
    }
    
    return success;
}

// 초기화: CPU FD들을 미리 열어두기
static inline int init_cpu_freq_fds(int *fd_cache, int max_fds) {
    DIR *dir = opendir("/sys/devices/system/cpu");
    if (!dir) return 0;
    
    int count = 0;
    struct dirent *entry;
    
    while ((entry = readdir(dir)) != NULL && count < max_fds - 1) {
        if (strncmp(entry->d_name, "cpu", 3) != 0) continue;
        if (entry->d_name[3] < '0' || entry->d_name[3] > '9') continue;
        
        char min_path[256], max_path[256];
        snprintf(min_path, sizeof(min_path), 
                 "/sys/devices/system/cpu/%s/cpufreq/scaling_min_freq", 
                 entry->d_name);
        snprintf(max_path, sizeof(max_path), 
                 "/sys/devices/system/cpu/%s/cpufreq/scaling_max_freq", 
                 entry->d_name);
        
        fd_cache[count++] = open(min_path, O_WRONLY);
        fd_cache[count++] = open(max_path, O_WRONLY);
    }
    
    closedir(dir);
    return count;
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
    if (temp_mc < 0) return;
    
    int temp_celsius = temp_mc / 1000;

    if (temp_celsius >= temp_celsius && !finished){
        finished = true;
    
        // 온도에 맞는 주파수 찾기
        int target_freq = get_freq_for_temp(temp_celsius);
        
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