
#include "ocl_global.h"
#include "ocl_kernels.h"

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
    cl_int num_devices = 0;
    for (unsigned int i = 0; i < num_platforms; i++)
    {
        clGetDeviceIDs(platforms[i], type, 1, &context->device, &num_devices);
        if (num_devices == 1)
            break;
    }
    free(platforms);
    if (num_devices == 0)
        check_ocl(CL_DEVICE_NOT_FOUND, "Cannot find a GPU device");

    // Create a context and command queue for the device
    context->context = clCreateContext(0, 1, &context->device, NULL, NULL, &err);
    check_ocl(err, "Creating context");

    context->queue = clCreateCommandQueue(context->context, context->device, 0, &err);
    check_ocl(err, "Creating command queue");

    // Create program
    context->program = clCreateProgramWithSource(context->context, 1, &ocl_kernels_ocl, NULL, &err);
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
