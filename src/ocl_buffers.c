
#include "ocl_buffers.h"

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
