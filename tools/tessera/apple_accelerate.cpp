#include <Accelerate/Accelerate.h>

#include <cstddef>
#include <cstdint>

extern "C" int tessera_accelerate_weighted_square_error(
        const float * weights,
        const float * reconstructed,
        const float * importance_squared,
        float * error,
        std::size_t rows,
        std::size_t columns) {
    if (!weights || !reconstructed || !error || rows == 0 || columns == 0) {
        return -1;
    }

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t offset = row * columns;
        vDSP_vsub(
            reconstructed + offset, 1,
            weights + offset, 1,
            error + offset, 1,
            static_cast<vDSP_Length>(columns));
        vDSP_vsq(
            error + offset, 1,
            error + offset, 1,
            static_cast<vDSP_Length>(columns));
        if (importance_squared) {
            vDSP_vmul(
                error + offset, 1,
                importance_squared, 1,
                error + offset, 1,
                static_cast<vDSP_Length>(columns));
        }
    }
    return 0;
}

extern "C" int tessera_accelerate_lane_targets(
        const float * weights,
        const int8_t * ternary,
        float * targets,
        std::size_t rows,
        std::size_t columns,
        std::size_t lane_width) {
    if (!weights || !ternary || !targets || rows == 0 || columns == 0 ||
            lane_width == 0 || columns % lane_width != 0) {
        return -1;
    }

    const std::size_t lanes = columns / lane_width;
    float magnitudes[64];
    float retained[64];
    if (lane_width > 64) {
        return -2;
    }

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            const std::size_t offset = row * columns + lane * lane_width;
            vDSP_vabs(
                weights + offset, 1,
                magnitudes, 1,
                static_cast<vDSP_Length>(lane_width));
            float count = 0.0f;
            for (std::size_t i = 0; i < lane_width; ++i) {
                retained[i] = ternary[offset + i] == 0 ? 0.0f : 1.0f;
                count += retained[i];
            }
            float sum = 0.0f;
            vDSP_dotpr(
                magnitudes, 1,
                retained, 1,
                &sum,
                static_cast<vDSP_Length>(lane_width));
            targets[row * lanes + lane] = count > 0.0f ? sum / count : 0.0f;
        }
    }
    return 0;
}
