// thermal_control.h
#ifndef THERMAL_CONTROL_H
#define THERMAL_CONTROL_H

#include <stdio.h>       // fprintf, stderr, snprintf 추가
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <stdbool.h>     // bool 타입 추가

#define GPU_TEMP_PATH "/sys/class/kgsl/kgsl-3d0/temp"
#define TEMP_THRESHOLD_MC 70000  // 70도 = 70000 millidegree C
#define TARGET_CPU_FREQ 1785600

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

// CPU frequency 설정
static inline bool set_cpu_freq(int freq_khz) {
    DIR *dir = opendir("/sys/devices/system/cpu");
    if (!dir) return false;
    
    bool success = true;
    struct dirent *entry;
    
    while ((entry = readdir(dir)) != NULL) {
        // cpu0, cpu1, ... 찾기
        if (strncmp(entry->d_name, "cpu", 3) != 0) continue;
        if (entry->d_name[3] < '0' || entry->d_name[3] > '9') continue;
        
        char min_path[256], max_path[256];
        snprintf(min_path, sizeof(min_path), 
                 "/sys/devices/system/cpu/%s/cpufreq/scaling_min_freq", 
                 entry->d_name);
        snprintf(max_path, sizeof(max_path), 
                 "/sys/devices/system/cpu/%s/cpufreq/scaling_max_freq", 
                 entry->d_name);
        
        // scaling_max_freq 먼저 설정 (min보다 높아야 함)
        int fd_max = open(max_path, O_WRONLY);
        if (fd_max >= 0) {
            char freq_str[16];
            int len = snprintf(freq_str, sizeof(freq_str), "%d", freq_khz);
            write(fd_max, freq_str, len);
            close(fd_max);
        } else {
            success = false;
        }
        
        // scaling_min_freq 설정
        int fd_min = open(min_path, O_WRONLY);
        if (fd_min >= 0) {
            char freq_str[16];
            int len = snprintf(freq_str, sizeof(freq_str), "%d", freq_khz);
            write(fd_min, freq_str, len);
            close(fd_min);
        } else {
            success = false;
        }
    }
    
    closedir(dir);
    return success;
}

// 온도 기반 thermal control
static inline void thermal_control_check() {
    static int last_temp = -1;
    static bool throttled = false;
    
    int temp_mc = read_gpu_temp();
    if (temp_mc < 0) return;  // 읽기 실패시 무시
    
    // 70도 이상이면 throttle
    if (temp_mc >= TEMP_THRESHOLD_MC && !throttled) {
        set_cpu_freq(TARGET_CPU_FREQ);
        throttled = true;
        fprintf(stderr, "Thermal: GPU temp %.1f°C >= 70°C, throttling CPU to %d KHz\n", 
                temp_mc / 1000.0, TARGET_CPU_FREQ);
    }
    // 65도 이하로 내려가면 throttle 해제 (hysteresis)
    else if (temp_mc < (TEMP_THRESHOLD_MC - 5000) && throttled) {
        // 원하는 경우 여기서 원래 frequency로 복구 가능
        throttled = false;
        fprintf(stderr, "Thermal: GPU temp %.1f°C < 65°C, releasing throttle\n", 
                temp_mc / 1000.0);
    }
    
    last_temp = temp_mc;
}

#endif // THERMAL_CONTROL_H