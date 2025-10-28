// thermal_control.h
#ifndef THERMAL_CONTROL_H
#define THERMAL_CONTROL_H

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <stdbool.h>
#include <errno.h>

#define GPU_TEMP_PATH "/sys/class/kgsl/kgsl-3d0/temp"
#define TEMP_THRESHOLD_MC 60000  // 60도 = 60000 millidegree C
#define TARGET_CPU_FREQ 960000
#define CHECK_INTERVAL 10  // 10 토큰마다 한 번 체크

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

// CPU frequency 설정 - echo처럼
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

// 온도 기반 thermal control
static inline void thermal_control_check() {
    static bool initialized = false;
    static int call_count = 0;
    static bool throttled = false;
    
    if (!initialized) {
        initialized = true;
        fprintf(stderr, "Thermal: Control initialized\n");
    }
    
    call_count++;
    if (call_count % CHECK_INTERVAL != 0) {
        return;
    }
    
    int temp_mc = read_gpu_temp();
    if (temp_mc < 0) return;
    
    // 60도 이상이면 throttle
    if (temp_mc >= TEMP_THRESHOLD_MC && !throttled) {
        fprintf(stderr, "Thermal: GPU %.1f°C >= 60°C, throttling CPU to %d KHz\n", 
                temp_mc / 1000.0, TARGET_CPU_FREQ);
        
        if (set_cpu_freq(TARGET_CPU_FREQ)) {
            throttled = true;
            fprintf(stderr, "Thermal: Throttling activated\n");
        } else {
            fprintf(stderr, "Thermal: Failed (try sudo)\n");
        }
    }
    // 55도 이하면 해제
    else if (temp_mc < (TEMP_THRESHOLD_MC - 5000) && throttled) {
        throttled = false;
        fprintf(stderr, "Thermal: GPU %.1f°C < 55°C, releasing throttle\n", 
                temp_mc / 1000.0);
    }
}

#endif // THERMAL_CONTROL_H