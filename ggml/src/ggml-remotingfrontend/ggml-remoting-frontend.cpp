#include "ggml-remoting-frontend.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "remoting.h"

#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>
#include <thread>

int ggml_backend_remoting_get_device_count();

struct remoting_device_struct {
    std::mutex mutex;
};
