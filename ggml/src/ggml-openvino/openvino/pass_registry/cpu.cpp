#include "pipeline.h"

namespace ov {
namespace frontend {
namespace ggml {
namespace pipeline {

// The CPU plugin runs the device-neutral model as is.
void register_cpu_passes(ov::pass::Manager & manager, const GraphTraits & traits) {
    (void) manager;
    (void) traits;
}

}  // namespace pipeline
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
