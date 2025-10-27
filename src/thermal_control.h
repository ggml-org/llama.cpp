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
#include <fstream>       // 🔥 추가
#include <chrono>        // 🔥 추가

#define GPU_TEMP_PATH "/sys/class/kgsl/kgsl-3d0/temp"
#define TEMP_THRESHOLD_MC 60000  // 60도 = 60000 millidegree C
#define TARGET_CPU_FREQ 960000
#define CHECK_INTERVAL 10  // 10 토큰마다 한 번 체크

// 🔥 Throughput monitoring CSV (llama.cpp에서 정의됨)
extern std::ofstream g_csv;

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
static inline void log_thermal_event(const char* event, double temp_celsius, int freq_khz = 0) {
    if (!g_csv.is_open()) return;
    
    auto ts = std::chrono::system_clock::now().time_since_epoch();
    auto ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ts).count();
    
    // CSV 형식: timestamp,-1,event_type,temp,freq
    g_csv << ts_ms << ",-1," << event << "," << temp_celsius << "," << freq_khz << "\n";
    g_csv.flush();
}

// 온도 기반 thermal control - 최적화 버전
static inline void thermal_control_check() {
    static int gpu_temp_fd = -1;
    static int cpu_freq_fds[32];  // max 16 CPUs = 32 FDs (min/max pairs)
    static int num_cpu_fds = 0;
    static bool initialized = false;
    static int call_count = 0;
    static bool throttled = false;
    
    // 초기화
    if (!initialized) {
        gpu_temp_fd = open(GPU_TEMP_PATH, O_RDONLY);
        num_cpu_fds = init_cpu_freq_fds(cpu_freq_fds, 32);
        initialized = true;
        if (gpu_temp_fd < 0 || num_cpu_fds == 0) {
            fprintf(stderr, "Thermal: Failed to initialize (fd=%d, num_fds=%d)\n", 
                    gpu_temp_fd, num_cpu_fds);
        }
    }
    
    // CHECK_INTERVAL 토큰마다만 체크
    call_count++;
    if (call_count % CHECK_INTERVAL != 0) {
        return;
    }
    
    int temp_mc = read_gpu_temp_fast(gpu_temp_fd);
    if (temp_mc < 0) return;
    
    double temp_celsius = temp_mc / 1000.0;  // 🔥 추가
    
    // 60도 이상이면 throttle
    if (temp_mc >= TEMP_THRESHOLD_MC && !throttled) {
        set_cpu_freq_cached(cpu_freq_fds, num_cpu_fds, TARGET_CPU_FREQ);
        throttled = true;
        
        // 🔥 콘솔 출력
        fprintf(stderr, "Thermal: GPU temp %.1f°C >= 60°C, throttling CPU to %d KHz\n", 
                temp_celsius, TARGET_CPU_FREQ);
        
        // 🔥 CSV에 기록
        log_thermal_event("THROTTLE", temp_celsius, TARGET_CPU_FREQ);
    }
    // 55도 이하로 내려가면 throttle 해제
    else if (temp_mc < (TEMP_THRESHOLD_MC - 5000) && throttled) {
        throttled = false;
        
        // 🔥 콘솔 출력
        fprintf(stderr, "Thermal: GPU temp %.1f°C < 55°C, releasing throttle\n", 
                temp_celsius);
        
        // 🔥 CSV에 기록
        log_thermal_event("RELEASE", temp_celsius, 0);
        
        // 필요시 원래 frequency로 복구
        // set_cpu_freq_cached(cpu_freq_fds, num_cpu_fds, ORIGINAL_FREQ);
    }
}

#endif // THERMAL_CONTROL_H