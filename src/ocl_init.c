
#include "ocl_global.h"

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

}
