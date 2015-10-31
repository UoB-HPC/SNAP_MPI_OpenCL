
#include "profiler.h"
#include "ocl_global.h"
#include "ocl_kernels.h"

#define MAX_DEVICES 8

void check_ocl(const cl_int err, const char *msg)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "OpenCL Error: %d. %s\n", err, msg);
        exit(err);
    }
}

void init_ocl(struct context * context)
{
    cl_int err;

    // Get list of platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    check_ocl(err, "Getting number of platforms");
    cl_platform_id *platforms = malloc(num_platforms*sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, &num_platforms);
    check_ocl(err, "Getting platforms");

    // Get a GPU device
    cl_device_type type = CL_DEVICE_TYPE_GPU;
    cl_uint num_devices = 0;
    cl_device_id devices[MAX_DEVICES];
    for (unsigned int i = 0; i < num_platforms; i++)
    {
        clGetDeviceIDs(platforms[i], type, MAX_DEVICES, devices, &num_devices);
        if (num_devices == 1)
            break;
    }
    free(platforms);
    if (num_devices == 0)
        check_ocl(CL_DEVICE_NOT_FOUND, "Cannot find a GPU device");

    // Just pick the first GPU device
    context->device = devices[0];
#ifdef __APPLE__
    // If we on my MacBook we need the second GPU (the discrete one)
    context->device = devices[1];
#endif

    // Create a context and command queue for the device
    context->context = clCreateContext(0, 1, &context->device, NULL, NULL, &err);
    check_ocl(err, "Creating context");

    if (profiling)
        context->queue = clCreateCommandQueue(context->context, context->device, CL_QUEUE_PROFILING_ENABLE, &err);
    else
        context->queue = clCreateCommandQueue(context->context, context->device, 0, &err);
    check_ocl(err, "Creating command queue");

    // Create program
    context->program = clCreateProgramWithSource(context->context, sizeof(ocl_kernels)/sizeof(char*), ocl_kernels, NULL, &err);
    check_ocl(err, "Creating program");

    // Build program
    char *options = "-cl-mad-enable -cl-fast-relaxed-math";
    cl_int build_err = clBuildProgram(context->program, 1, &context->device, options, NULL, NULL);
    if (build_err == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t log_size;
        err = clGetProgramBuildInfo(context->program, context->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        check_ocl(err, "Getting build log size");
        char *build_log = malloc(log_size);
        err = clGetProgramBuildInfo(context->program, context->device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        check_ocl(err, "Getting build log");
        fprintf(stderr, "OpenCL Build log: %s\n", build_log);
        free(build_log);
    }
    check_ocl(build_err, "Building program");

    // Create the kernels
    context->kernels.calc_velocity_delta = clCreateKernel(context->program, "calc_velocity_delta", &err);
    check_ocl(err, "Creating velocity delta kernel");
    context->kernels.calc_dd_coeff = clCreateKernel(context->program, "calc_dd_coeff", &err);
    check_ocl(err, "Creating diamond difference constants kernel");
    context->kernels.calc_denominator = clCreateKernel(context->program, "calc_denominator", &err);
    check_ocl(err, "Creating denominator kernel");
    context->kernels.zero_buffer = clCreateKernel(context->program, "zero_buffer", &err);
    check_ocl(err, "Creating buffer zeroing kernel");
    context->kernels.outer_source = clCreateKernel(context->program, "calc_outer_source", &err);
    check_ocl(err, "Creating outer source kernel");
    context->kernels.inner_source = clCreateKernel(context->program, "calc_inner_source", &err);
    check_ocl(err, "Creating inner source kernel");
    context->kernels.sweep_plane = clCreateKernel(context->program, "sweep_plane", &err);
    check_ocl(err, "Creating sweep plane kernel");
    context->kernels.reduce_flux = clCreateKernel(context->program, "reduce_flux", &err);
    check_ocl(err, "Creating scalar flux reduction kernel");
    context->kernels.reduce_flux_moments = clCreateKernel(context->program, "reduce_flux_moments", &err);
    check_ocl(err, "Creating scalar flux moments reduction kernel");

}

void release_context(struct context * context)
{
    cl_int err;
    err = clReleaseProgram(context->program);
    check_ocl(err, "Releasing program");

#ifdef CL_VERSION_1_2
    err = clReleaseDevice(context->device);
    check_ocl(err, "Releasing device");
#endif

    err = clReleaseCommandQueue(context->queue);
    check_ocl(err, "Releasing command queue");

    err = clReleaseContext(context->context);
    check_ocl(err, "Releasing context");

}
