

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <array>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <future>
#include <queue>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
    #include <windows.h>
    #include <direct.h> // For _mkdir on Windows
    #include <algorithm> // For std::replace on w64devkit
#else
    #include <unistd.h>
    #include <sys/wait.h>
    #include <fcntl.h>
#endif

#define ASYNCIO_CONCURRENCY 64

std::mutex lock;
std::vector<std::pair<std::string, std::string>> shader_fnames;

std::string GLSLC = "glslc";
std::string input_dir = "vulkan-shaders";
std::string output_dir = "/tmp";
std::string target_hpp = "ggml-vulkan-shaders.hpp";
std::string target_cpp = "ggml-vulkan-shaders.cpp";
bool no_clean = false;

const std::vector<std::string> type_names = {
    "f32",
    "f16",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq4_nl"
};

void execute_command(const std::string& command, std::string& stdout_str, std::string& stderr_str) {
#ifdef _WIN32
    HANDLE stdout_read, stdout_write;
    HANDLE stderr_read, stderr_write;
    SECURITY_ATTRIBUTES sa = { sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };

    if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) ||
        !SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0)) {
        throw std::runtime_error("Failed to create stdout pipe");
    }

    if (!CreatePipe(&stderr_read, &stderr_write, &sa, 0) ||
        !SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
        throw std::runtime_error("Failed to create stderr pipe");
    }

    PROCESS_INFORMATION pi;
    STARTUPINFOA si = { sizeof(STARTUPINFOA) };
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = stdout_write;
    si.hStdError = stderr_write;

    std::vector<char> cmd(command.begin(), command.end());
    cmd.push_back('\0');

    if (!CreateProcessA(NULL, cmd.data(), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        throw std::runtime_error("Failed to create process");
    }

    CloseHandle(stdout_write);
    CloseHandle(stderr_write);

    std::array<char, 128> buffer;
    DWORD bytes_read;

    while (ReadFile(stdout_read, buffer.data(), buffer.size(), &bytes_read, NULL) && bytes_read > 0) {
        stdout_str.append(buffer.data(), bytes_read);
    }

    while (ReadFile(stderr_read, buffer.data(), buffer.size(), &bytes_read, NULL) && bytes_read > 0) {
        stderr_str.append(buffer.data(), bytes_read);
    }

    CloseHandle(stdout_read);
    CloseHandle(stderr_read);
    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
#else
int stdout_pipe[2];
    int stderr_pipe[2];

    if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        throw std::runtime_error("Failed to create pipes");
    }

    pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("Failed to fork process");
    }

    if (pid == 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        execl("/bin/sh", "sh", "-c", command.c_str(), (char*) nullptr);
        _exit(EXIT_FAILURE);
    } else {
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        std::array<char, 128> buffer;
        ssize_t bytes_read;

        while ((bytes_read = read(stdout_pipe[0], buffer.data(), buffer.size())) > 0) {
            stdout_str.append(buffer.data(), bytes_read);
        }

        while ((bytes_read = read(stderr_pipe[0], buffer.data(), buffer.size())) > 0) {
            stderr_str.append(buffer.data(), bytes_read);
        }

        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        waitpid(pid, nullptr, 0);
    }
#endif
}

bool directory_exists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false; // Path doesn't exist or can't be accessed
    }
    return (info.st_mode & S_IFDIR) != 0; // Check if it is a directory
}

bool create_directory(const std::string& path) {
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST; // EEXIST means the directory already exists
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST; // 0755 is the directory permissions
#endif
}

std::string to_uppercase(const std::string& input) {
    std::string result = input;
    for (char& c : result) {
        c = std::toupper(c);
    }
    return result;
}

bool string_ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

static const char path_separator = '/';

std::string join_paths(const std::string& path1, const std::string& path2) {
    return path1 + path_separator + path2;
}

std::string basename(const std::string &path) {
    return path.substr(path.find_last_of("/\\") + 1);
}

void string_to_spv(const std::string& _name, const std::string& in_fname, const std::map<std::string, std::string>& defines, bool fp16 = true) {
    std::string name = _name + (fp16 ? "" : "_fp32");
    std::string out_fname = join_paths(output_dir, name + ".spv");
    std::string in_path = join_paths(input_dir, in_fname);

    #ifdef _WIN32
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", "--target-env=vulkan1.2", "-O", "\"" + in_path + "\"", "-o", "\"" + out_fname + "\""};
    #else
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", "--target-env=vulkan1.2", "-O", in_path, "-o",  out_fname};
    #endif
    for (const auto& define : defines) {
        cmd.push_back("-D" + define.first + "=" + define.second);
    }

    std::string command;
    for (const auto& part : cmd) {
        command += part + " ";
    }

    std::string stdout_str, stderr_str;
    try {
        // std::cout << "Executing command: ";
        // for (const auto& part : cmd) {
        //     std::cout << part << " ";
        // }
        // std::cout << std::endl;

        execute_command(command, stdout_str, stderr_str);
        if (!stderr_str.empty()) {
            std::cerr << "cannot compile " << name << "\n\n" << command << "\n\n" << stderr_str << std::endl;
            return;
        }

        std::lock_guard<std::mutex> guard(lock);
        shader_fnames.push_back(std::make_pair(name, out_fname));
    } catch (const std::exception& e) {
        std::cerr << "Error executing command for " << name << ": " << e.what() << std::endl;
    }
}

std::map<std::string, std::string> merge_maps(const std::map<std::string, std::string>& a, const std::map<std::string, std::string>& b) {
    std::map<std::string, std::string> result = a;
    result.insert(b.begin(), b.end());
    return result;
}

void matmul_shaders(std::vector<std::future<void>>& tasks, bool fp16, bool matmul_id) {
    std::string load_vec = fp16 ? "8" : "4";
    std::string aligned_b_type_f32 = fp16 ? "mat2x4" : "vec4";
    std::string aligned_b_type_f16 = fp16 ? "f16mat2x4" : "f16vec4";

    std::map<std::string, std::string> base_dict = {{"FLOAT_TYPE", fp16 ? "float16_t" : "float"}};
    std::string shader_name = "matmul";

    if (matmul_id) {
        base_dict["MUL_MAT_ID"] = "1";
        shader_name = "matmul_id";
    }

    if (fp16) {
        base_dict["FLOAT16"] = "1";
    }

    // Shaders with f16 B_TYPE
    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv(shader_name + "_f32_f16", "mul_mm.comp", merge_maps(base_dict, {{"DATA_A_F32", "1"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}), fp16);
    }));
    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv(shader_name + "_f32_f16_aligned", "mul_mm.comp", merge_maps(base_dict, {{"DATA_A_F32", "1"}, {"LOAD_VEC_A", load_vec}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"D_TYPE", "float"}}), fp16);
    }));

    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv(shader_name + "_f16", "mul_mm.comp", merge_maps(base_dict, {{"DATA_A_F16", "1"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}), fp16);
    }));
    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv(shader_name + "_f16_aligned", "mul_mm.comp", merge_maps(base_dict, {{"DATA_A_F16", "1"}, {"LOAD_VEC_A", load_vec}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"D_TYPE", "float"}}), fp16);
    }));

    for (const auto& tname : type_names) {
        std::string data_a_key = "DATA_A_" + to_uppercase(tname);
        // For unaligned, load one at a time for f32/f16, or two at a time for quants
        std::string load_vec_a_unaligned = (tname == "f32" || tname == "f16") ? "1" : "2";
        // For aligned matmul loads
        std::string load_vec_a = (tname == "f32" || tname == "f16") ? load_vec : "2";
        tasks.push_back(std::async(std::launch::async, [=] {
            string_to_spv(shader_name + "_" + tname + "_f32", "mul_mm.comp", merge_maps(base_dict, {{data_a_key, "1"}, {"LOAD_VEC_A", load_vec_a_unaligned}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}), fp16);
        }));
        tasks.push_back(std::async(std::launch::async, [=] {
            string_to_spv(shader_name + "_" + tname + "_f32_aligned", "mul_mm.comp", merge_maps(base_dict, {{data_a_key, "1"}, {"LOAD_VEC_A", load_vec_a}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32}, {"D_TYPE", "float"}}), fp16);
        }));
    }
}

void process_shaders(std::vector<std::future<void>>& tasks) {
    std::cout << "ggml_vulkan: Generating and compiling shaders to SPIR-V" << std::endl;
    std::map<std::string, std::string> base_dict = {{"FLOAT_TYPE", "float"}};

    for (const auto& fp16 : {false, true}) {
        matmul_shaders(tasks, fp16, false);
        matmul_shaders(tasks, fp16, true);
    }

    for (const auto& tname : type_names) {
        // mul mat vec
        std::string data_a_key = "DATA_A_" + to_uppercase(tname);
        std::string shader = (string_ends_with(tname, "_k")) ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";

        tasks.push_back(std::async(std::launch::async, [=] {
            string_to_spv("mul_mat_vec_" + tname + "_f32_f32", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
        }));
        tasks.push_back(std::async(std::launch::async, [=] {
            string_to_spv("mul_mat_vec_" + tname + "_f16_f32", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
        }));

        tasks.push_back(std::async(std::launch::async, [=] {
            string_to_spv("mul_mat_vec_id_" + tname + "_f32", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_key, "1"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
        }));

        // Dequant shaders
        if (tname != "f16") {
            tasks.push_back(std::async(std::launch::async, [=] {
                string_to_spv("dequant_" + tname, "dequant_" + tname + ".comp", merge_maps(base_dict, {{data_a_key, "1"}, {"D_TYPE", "float16_t"}}));
            }));
        }

        if (!string_ends_with(tname, "_k")) {
            shader = (tname == "f32" || tname == "f16") ? "get_rows.comp" : "get_rows_quant.comp";

            if (tname == "f16") {
                tasks.push_back(std::async(std::launch::async, [=] {
                    string_to_spv("get_rows_" + tname, shader, {{data_a_key, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
                }));
            } else {
                tasks.push_back(std::async(std::launch::async, [=] {
                    string_to_spv("get_rows_" + tname, shader, {{data_a_key, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}});
                }));
            }
            tasks.push_back(std::async(std::launch::async, [=] {
                string_to_spv("get_rows_" + tname + "_f32", shader, {{data_a_key, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}});
            }));
        }
    }

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("mul_mat_vec_p021_f16_f32", "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("mul_mat_vec_nc_f16_f32", "mul_mat_vec_nc.comp", {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});
    }));

    // Norms
    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv("norm_f32", "norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    }));
    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv("group_norm_f32", "group_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    }));
    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv("rms_norm_f32", "rms_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("cpy_f32_f32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("cpy_f32_f16", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("cpy_f16_f16", "copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("add_f32", "add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("add_f16_f32_f16", "add.comp", {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("acc_f32", "acc.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("split_k_reduce", "mul_mat_split_k_reduce.comp", {});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("mul_f32", "mul.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("div_f32", "div.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("repeat_f32", "repeat.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("scale_f32", "scale.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("sqr_f32", "square.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("clamp_f32", "clamp.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("pad_f32", "pad.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("concat_f32", "concat.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("concat_f16", "concat.comp", {{"A_TYPE", "float16_t"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("concat_i32", "concat.comp", {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("upscale_f32", "upscale.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("gelu_f32", "gelu.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("gelu_quick_f32", "gelu_quick.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("silu_f32", "silu.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("relu_f32", "relu.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("leaky_relu_f32", "leaky_relu.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("tanh_f32", "tanh.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("diag_mask_inf_f32", "diag_mask_inf.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv("soft_max_f32", "soft_max.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    }));
    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv("soft_max_f32_f16", "soft_max.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("rope_norm_f32", "rope_norm.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("rope_norm_f16", "rope_norm.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("rope_neox_f32", "rope_neox.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    }));
    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("rope_neox_f16", "rope_neox.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    }));

    tasks.push_back(std::async(std::launch::async, [] {
        string_to_spv("argsort_f32", "argsort.comp", {{"A_TYPE", "float"}});
    }));

    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv("sum_rows_f32", "sum_rows.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    }));

    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv("im2col_f32", "im2col.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    }));
    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv("im2col_f32_f16", "im2col.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}}));
    }));

    tasks.push_back(std::async(std::launch::async, [=] {
        string_to_spv("timestep_embedding_f32", "timestep_embedding.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    }));
}

void write_output_files() {
    FILE* hdr = fopen(target_hpp.c_str(), "w");
    FILE* src = fopen(target_cpp.c_str(), "w");

    fprintf(hdr, "#include <cstdint>\n\n");
    fprintf(src, "#include \"%s\"\n\n", basename(target_hpp).c_str());

    for (const auto& pair : shader_fnames) {
        const std::string& name = pair.first;
        #ifdef _WIN32
            std::string path = pair.second;
            std::replace(path.begin(), path.end(), '/', '\\' );
        #else
            const std::string& path = pair.second;
        #endif

        FILE* spv = fopen(path.c_str(), "rb");
        if (!spv) {
            std::cerr << "Error opening SPIR-V file: " << path << " (" << strerror(errno) << ")\n";
            continue;
        }

        fseek(spv, 0, SEEK_END);
        size_t size = ftell(spv);
        fseek(spv, 0, SEEK_SET);

        std::vector<unsigned char> data(size);
        size_t read_size = fread(data.data(), 1, size, spv);
        fclose(spv);
        if (read_size != size) {
            std::cerr << "Error reading SPIR-V file: " << path << " (" << strerror(errno) << ")\n";
            continue;
        }

        fprintf(hdr, "extern unsigned char %s_data[%zu];\n", name.c_str(), size);
        fprintf(hdr, "const uint64_t %s_len = %zu;\n\n", name.c_str(), size);

        fprintf(src, "unsigned char %s_data[%zu] = {\n", name.c_str(), size);
        for (size_t i = 0; i < size; ++i) {
            fprintf(src, "0x%02x,", data[i]);
            if ((i + 1) % 12 == 0) fprintf(src, "\n");
        }
        fprintf(src, "\n};\n\n");

        if (!no_clean) {
            std::remove(path.c_str());
        }
    }

    fclose(hdr);
    fclose(src);
}

int main(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) {
            args[argv[i]] = argv[i + 1];
        }
    }

    if (args.find("--glslc") != args.end()) {
        GLSLC = args["--glslc"]; // Path to glslc
    }
    if (args.find("--input-dir") != args.end()) {
        input_dir = args["--input-dir"]; // Directory containing shader sources
    }
    if (args.find("--output-dir") != args.end()) {
        output_dir = args["--output-dir"]; // Directory for containing SPIR-V output
    }
    if (args.find("--target-hpp") != args.end()) {
        target_hpp = args["--target-hpp"]; // Path to generated header file
    }
    if (args.find("--target-cpp") != args.end()) {
        target_cpp = args["--target-cpp"]; // Path to generated cpp file
    }
    if (args.find("--no-clean") != args.end()) {
        no_clean = true; // Keep temporary SPIR-V files in output-dir after build
    }

    if (!directory_exists(input_dir)) {
        std::cerr << "\"" << input_dir << "\" must be a valid directory containing shader sources" << std::endl;
        return EXIT_FAILURE;
    }

    if (!directory_exists(output_dir)) {
        if (!create_directory(output_dir)) {
            std::cerr << "Error creating output directory: " << output_dir << "\n";
            return EXIT_FAILURE;
        }
    }

    std::vector<std::future<void>> tasks;
    process_shaders(tasks);

    for (auto& task : tasks) {
        task.get();
    }

    write_output_files();

    return EXIT_SUCCESS;
}
