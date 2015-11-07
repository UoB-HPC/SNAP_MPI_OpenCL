
#pragma once

#include <stdlib.h>

#include "global.h"
#include "ocl_global.h"
#include "ocl_buffers.h"

struct cell_id
{
    unsigned int i, j, k;
};

struct plane
{
    unsigned int num_cells;
    struct cell_id * cell_ids;
};

/** \brief Create a list of cell indexes in the planes in the XY plane determined by chunk */
void init_planes(struct plane** planes, unsigned int *num_planes, struct problem * problem, struct rankinfo * rankinfo);

void copy_planes(const struct plane * planes, const unsigned int num_planes, struct context * context, struct buffers * buffers);

/** \brief Enqueue the kernels to sweep a plane */
void sweep_plane(const unsigned int z_pos, const int octant, const int istep, const int jstep, const int kstep, const unsigned int plane_num, const struct plane * planes, struct problem * problem, struct rankinfo * rankinfo, struct context * context, struct buffers * buffers);

