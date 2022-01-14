#include <string>
#include <istream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <complex>

#include <CL/opencl.h>

#include "const.h"

#ifndef CHECK_ERR
#define CHECK_ERR(intro, result, exit_label)        \
do {                                                \
    if ((result) != 0)                              \
    {                                               \
        fprintf(stderr, "%s: %d\n", intro, result); \
        goto exit_label;                            \
    }                                               \
} while (false)
#endif

#ifndef CHECK_AND_RET_ERR
#define CHECK_AND_RET_ERR(intro, result)            \
do {                                                \
    if ((result) != 0)                              \
    {                                               \
        fprintf(stderr, "%s: %d\n", intro, result); \
        return result;                              \
    }                                               \
} while (false)
#endif

static inline
std::string load_source_file(char const* file_name)
{
    std::ifstream input(file_name);
    std::stringstream ss;
    ss << input.rdbuf();
    return ss.str();
}

struct gpu_context
{
    size_t frame_size_x{0};
    size_t frame_size_y{0};

    cl_program program = nullptr;
    cl_mem result_array_buf = nullptr;
    std::vector<cl_kernel> kernels;
    cl_device_id selected_device = nullptr;
    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;

    ~gpu_context() noexcept
    {
        if (program)
            clReleaseProgram(program);
        if (result_array_buf)
            clReleaseMemObject(result_array_buf);

        for (auto kernel : kernels)
            if (kernel)
                clReleaseKernel(kernel);

        if (context)
            clReleaseContext(context);
        if (command_queue)
            clReleaseCommandQueue(command_queue);
        if (selected_device)
            clReleaseDevice(selected_device);
    }
};

static inline
char const* get_mem_type_str(cl_device_local_mem_type t)
{
    switch (t)
    {
    case CL_LOCAL:
        return "local";
    case CL_GLOBAL:
        return "global";
    default:
        return "other";
    }
}

/**
 * Setup device for the specified \ref gpu_context. Doesn't change over kernel.
 * \param context Context to be initialized
 * \return error code or zero on success
 */
cl_int select_device(gpu_context& context)
{
    cl_int error_code = 0;
    cl_uint num_platforms = 0;

    error_code = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (error_code)
    {
        fprintf(stderr, "Error getting platforms list!\n");
        return error_code;
    }

    std::vector<cl_platform_id> platforms(num_platforms);

    error_code = clGetPlatformIDs(num_platforms, platforms.data(), &num_platforms);
    if (error_code)
    {
        fprintf(stderr, "Error getting platforms list!\n");
        return error_code;
    }

    cl_uint num_devices = 0;
    size_t max_devices = 42;
    cl_device_id device_list[max_devices];
    char device_name[64];

    cl_device_local_mem_type mem_type = CL_NONE;
    size_t max_work_group_size = 0;

    for (size_t i = 0; i < num_platforms; ++i)
    {
        error_code = clGetDeviceIDs(
            platforms[i], CL_DEVICE_TYPE_ALL, max_devices, device_list,
            &num_devices
        );

        if (error_code) continue;

        for (size_t j = 0; j < num_devices; ++j)
        {
            if (!context.selected_device)
                context.selected_device = device_list[j];

            size_t ret_sz = 0;
            cl_device_local_mem_type cur_mem_type = 0;
            size_t work_group_size = 0;

            error_code = clGetDeviceInfo(
                device_list[j], CL_DEVICE_LOCAL_MEM_TYPE,
                sizeof(cl_device_local_mem_type), &cur_mem_type, &ret_sz
            );

            if (error_code)
            {
                if (context.selected_device != device_list[j])
                    clReleaseDevice(device_list[j]);
                continue;
            }

            error_code = clGetDeviceInfo(
                device_list[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                sizeof(size_t), &work_group_size, &ret_sz
            );

            if (error_code)
            {
                if (context.selected_device != device_list[j])
                    clReleaseDevice(device_list[j]);
                continue;
            }

            error_code = clGetDeviceInfo(
                device_list[j], CL_DEVICE_NAME, 63, device_name, &ret_sz
            );

            device_name[ret_sz] = '\0';
            fprintf(
                stderr,
                "Found device \"%s\": mem type %s, max workgroup size %zu\n",
                device_name, get_mem_type_str(cur_mem_type), work_group_size
            );

            if ((mem_type != CL_LOCAL && cur_mem_type == CL_LOCAL)
                || (cur_mem_type == mem_type
                    && max_work_group_size < work_group_size))
            {
                clReleaseDevice(context.selected_device);
                context.selected_device = device_list[j];
                max_work_group_size = work_group_size;
                mem_type = cur_mem_type;
            }
            else if (context.selected_device != device_list[j])
                clReleaseDevice(device_list[j]);
        }
    }

    if (!context.selected_device)
        return error_code;
    else
    {
        size_t ret_sz;
        clGetDeviceInfo(
            context.selected_device, CL_DEVICE_NAME, 63, device_name, &ret_sz
        );
        device_name[ret_sz] = '\0';

        fprintf(stderr, "Selected device: %s\n", device_name);
    }

    context.context = clCreateContext(
        nullptr, 1, &context.selected_device, nullptr, nullptr, &error_code
    );

    if (error_code) return error_code;

    context.command_queue = clCreateCommandQueue(
        context.context, context.selected_device, CL_QUEUE_PROFILING_ENABLE,
        &error_code
    );

    return error_code;
}

/// Loads and compile the kernel for the specified \ref gpu_context
cl_int load_program(gpu_context& context,
                    std::vector<char const*> const& sources_paths)
{
    assert(context.selected_device);
    assert(context.context);

    std::vector<std::string> sources;
    std::vector<char const*> sources_c_strs;
    std::vector<size_t> sources_lens;
    sources.reserve(sources_paths.size());
    sources_c_strs.reserve(sources_paths.size());
    sources_lens.reserve(sources_paths.size());
    cl_int result = 0;

    for (auto&& source_path : sources_paths)
    {
        sources.emplace_back(load_source_file(source_path));
        if (sources.back().empty())
        {
            fprintf(stderr, "Error: empty source file %s\n", source_path);
            return -1;
        }

        sources_c_strs.emplace_back(sources.back().c_str());
        sources_lens.emplace_back(sources.back().size());
    }

    context.program = clCreateProgramWithSource(
        context.context, sources.size(), sources_c_strs.data(),
        sources_lens.data(), &result
    );
    CHECK_ERR("Failed to create clProgram", result, return_error);

    result = clBuildProgram(
        context.program, 1, &context.selected_device, "", nullptr, nullptr
    );

    if (result)
    {
        fprintf(stderr, "kernel compilation failed\n");
        size_t log_len = 0;
        cl_int saved_error_code = result;
        char* build_log;

        result = clGetProgramBuildInfo(
            context.program, context.selected_device,
            CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len
        );
        CHECK_ERR("Failed to retrieve build's log", result, return_error);

        build_log = static_cast<char*>(malloc(log_len));
        result = clGetProgramBuildInfo(
            context.program, context.selected_device,
            CL_PROGRAM_BUILD_LOG, log_len, build_log, &log_len
        );

        if (result)
        {
            fprintf(stderr, "Failed to retrieve build's log: %d", result);
            free(build_log);
            goto return_error;
        }

        fprintf(stderr, "Kernel compilation log:\n%s\n", build_log);
        result = saved_error_code;
        goto return_error;
    }

return_error:
    return result;
}

/// Setups kernel & kernel structs like mem buffers for the \ref gpu_context
cl_int setup_kernels(gpu_context& context,
                     std::vector<char const*> const& kernel_names)
{
    cl_int result = 0;

    assert(context.selected_device);
    assert(context.context);
    assert(context.command_queue);

    for (auto&& kernel_name : kernel_names)
    {
        context.kernels.emplace_back(clCreateKernel(
            context.program, kernel_name, &result
        ));

        if (result)
            return result;
    }

    size_t frame_size = context.frame_size_x * context.frame_size_y;
    context.result_array_buf = clCreateBuffer(
            context.context, CL_MEM_READ_WRITE, frame_size * sizeof(precise_t),
            nullptr, &result
    );
    CHECK_AND_RET_ERR("Error creating buffer", result);
    return 0;
}

gpu_context setup_gpu_context(std::vector<char const*> const& sources_list,
                              std::vector<char const*> const& kernel_names,
                              cl_int* error, size_t N, size_t M)
{
    assert(error != nullptr);

    *error = 0;

    gpu_context context;

    context.frame_size_x = N;
    context.frame_size_y = M;

    *error = select_device(context);
    if (*error) return context;

    *error = load_program(context, sources_list);
    if (*error) return context;

    *error = setup_kernels(context, kernel_names);
    if (*error) return context;

    return context;
}

int main()
{
    // N = frame X size
    // M = frame Y size
//    constexpr cl_uint M = 3840;
//    constexpr cl_uint N = 2160;
//    constexpr precise_t scale = 0.0005;

    constexpr cl_uint const M = 640;
    constexpr cl_uint const N = 480;
    constexpr precise_t scale = 0.0025;

    constexpr precise_t focus_x = 0;
    constexpr precise_t focus_y = 0;


    std::vector<char const*> kernel_names = {
        "mandelbrot"
    };

    std::vector<char const*> sources_list {
        "const.h",
        "kernel.cl"
    };

    int exit_code = 0;
    cl_int error_code = 0;

    gpu_context context = setup_gpu_context(
            sources_list, kernel_names, &error_code, N, M
    );
    CHECK_AND_RET_ERR("startup failed", error_code);

    // Setting up "mandelbrot" kernel args
    clSetKernelArg( // focus_x
        context.kernels[0], 0, sizeof(precise_t), &focus_x
    );
    clSetKernelArg( // focus_x
        context.kernels[0], 1, sizeof(precise_t), &focus_y
    );
    clSetKernelArg( // scale
        context.kernels[0], 2, sizeof(precise_t), &scale
    );
    clSetKernelArg(
        context.kernels[0], 3, sizeof(cl_uint), &N
    );
    clSetKernelArg(
        context.kernels[0], 4, sizeof(cl_uint), &M
    );
    clSetKernelArg(
        context.kernels[0], 5, sizeof(cl_mem), &context.result_array_buf
    );

    size_t work_size[] = {M, N};
    size_t work_offset[] = {0, 0};
    std::vector<cl_event> run_events(kernel_names.size());

    error_code = clEnqueueNDRangeKernel(
        context.command_queue, context.kernels[0], 2, work_offset, work_size,
        nullptr, 0, nullptr, &run_events[0]
    );
    CHECK_AND_RET_ERR("Error enqueuing kernel", error_code);

    std::vector<precise_t> result(N * M, 0);

    clEnqueueReadBuffer(
        context.command_queue, context.result_array_buf, true, 0,
        N * M * sizeof(precise_t), result.data(), 0, nullptr, nullptr
    );

    long double elapsed_time = 0;

    for (auto event : run_events)
    {
        cl_ulong t_start = 0, t_end = 0;
        clGetEventProfilingInfo(
            event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, nullptr
        );
        clGetEventProfilingInfo(
            event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, nullptr
        );

        elapsed_time += t_end - t_start;
    }

    printf("%.4Lf ms elapsed max=%f min=%f", elapsed_time / 1e6,
           *std::max_element(result.begin(), result.end()),
           *std::min_element(result.begin(), result.end()));

    std::ofstream ppm("result.ppm");
    ppm << "P3\n"
        << M << " " << N << "\n"
        << 255 << "\n";
    for (auto x : result)
        ppm << 0 << " " << (int) (x * 255) << " 0\n";

    return exit_code;
}
