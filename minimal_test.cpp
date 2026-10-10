#include <sycl/sycl.hpp>
int main() {
    sycl::queue q;
    sycl::malloc_device(1024, q);
    return 0;
}
