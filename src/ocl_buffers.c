
#include "ocl_buffers.h"


void check_device_memory_requirements(
    struct problem * problem, struct rankinfo * rankinfo,
    struct context * context)
{
    cl_int err;
    cl_ulong global;
    err = clGetDeviceInfo(context->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global, NULL);
    cl_ulong total = 0;
    // TODO - add up the memory requirements, in bytes.

    if (global < total)
    {
        fprintf(stderr,"Error: Device does not have enough global memory.\n");
        fprintf(stderr, "Required: %.1f GB\n", (double)total/(1024.0*1024.0*1024.0));
        fprintf(stderr, "Available: %.1f GB\n", (double)global/(1024.0*1024.0*1024.0));
        exit(EXIT_FAILURE);
    }
}

void allocate_buffers(
    struct problem * problem, struct rankinfo * rankinfo,
    struct context * context, struct buffers * buffers)
{
    cl_int err;

    // Angular flux
    for (int i = 0; i < 8; i++)
    {
        buffers->angular_flux_in[i] = clCreateBuffer(
            context->context,
            CL_MEM_READ_WRITE,
            sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz,
            NULL, &err);
        check_ocl(err, "Creating an angular flux in buffer");

        buffers->angular_flux_out[i] = clCreateBuffer(
            context->context,
            CL_MEM_READ_WRITE,
            sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz,
            NULL, &err);
        check_ocl(err, "Creating an angular flux out buffer");
    }
}
