
#include "ocl_sweep.h"

void sweep(
    const int octant,
    const unsigned int num_planes,
    const struct plane * planes,
    struct problem * problem,
    struct context * context,
    struct buffers * buffers
    )
{
    cl_int err;

    for (unsigned int p = 0; p < num_planes; p++)
    {
        size_t global[] = {problem->nang*problem->ng, planes[p].num_cells};
        err = clEnqueueNDRangeKernel(context->queue,
            context->kernels.sweep_plane,
            2, 0, global, NULL,
            0, NULL, NULL);
        check_ocl(err, "Enqueue plane sweep kernel");
    }
}
