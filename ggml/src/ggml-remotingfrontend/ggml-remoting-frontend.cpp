#include <ostream>
#include <iostream>
#include <mutex>
#include <memory>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <sys/sysmacros.h>
#include <sys/stat.h>

#include "ggml-remoting-frontend.h"
#include "remoting.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"



int ggml_backend_remoting_get_device_count();




struct remoting_device_struct {
    std::mutex mutex;
};
