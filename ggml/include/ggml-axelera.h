/*
 * Copyright (c) 2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum number of Axelera devices supported.
 */
#define GGML_AXELERA_MAX_DEVICES 16

/**
 * @brief Retrieves the Axelera backend registration structure.
 *
 * This function returns the backend registration structure for the Axelera backend,
 * which is used by GGML to register and manage the backend.
 *
 * @return A pointer to the Axelera backend registration structure.
 */
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_axelera_reg(void);

/**
 * @brief Initializes the Axelera backend for a specified device.
 *
 * This function initializes the Axelera backend for the given device.
 * It verifies the device index, allocates a context, and creates a backend
 * instance.
 *
 * @param device The index of the device to initialize (0-based).
 * @return A pointer to the initialized backend instance, or nullptr on failure.
 */
GGML_BACKEND_API ggml_backend_t ggml_backend_axelera_init(int32_t device);

/**
 * @brief Checks if a given backend is an Axelera backend.
 *
 * This function verifies if the provided backend is an Axelera backend by comparing
 * its GUID with the Axelera backend's GUID.
 *
 * @param backend The backend instance to check.
 * @return True if the backend is an Axelera backend, false otherwise.
 */
GGML_BACKEND_API bool ggml_backend_is_axelera(ggml_backend_t backend);

/**
 * @brief Retrieves the Axelera buffer type for a specified device.
 *
 * This function initializes and returns the buffer type interface associated
 * with the given device. It ensures thread-safe access using a mutex.
 *
 * @param device The device index for which to retrieve the buffer type.
 * @return A pointer to the buffer type interface for the specified device, or
 * nullptr if the device index is out of range.
 */
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_axelera_buffer_type(int32_t device);

/**
 * @brief Retrieves the number of Axelera devices available.
 *
 * This function returns the number of Axelera devices available in the system.
 *
 * @return The number of Axelera devices available.
 */
GGML_BACKEND_API int32_t ggml_backend_axelera_get_device_count(void);

/**
 * @brief Pinned host buffer for use with the CPU backend for faster copies between CPU and Axelera.
 *
 * This function returns a buffer type for pinned host memory that can be used
 * for efficient data transfers between CPU and Axelera devices.
 *
 * @return A pointer to the host buffer type interface.
 */
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_axelera_host_buffer_type(void);

/**
 * @brief Retrieves the description of a specific Axelera device.
 *
 * This function retrieves the device name and capabilities information
 * and writes it into the provided description buffer.
 *
 * @param device The device index to retrieve the description for.
 * @param description Pointer to a buffer where the description will be written.
 * @param description_size Size of the description buffer.
 */
GGML_BACKEND_API void ggml_backend_axelera_get_device_description(
    int32_t device, char* description, size_t description_size);

/**
 * @brief Retrieves the memory information of a specific Axelera device.
 *
 * This function retrieves the free and total memory information
 * for the specified device and stores them in the provided pointers.
 *
 * @param device The device index to retrieve memory information for.
 * @param free Pointer to a variable where the free memory size will be stored.
 * @param total Pointer to a variable where the total memory size will be stored.
 */
GGML_BACKEND_API void ggml_backend_axelera_get_device_memory(
    int32_t device, size_t* free, size_t* total);

#ifdef __cplusplus
}
#endif
