
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


void init_planes(struct plane** planes, unsigned int *num_planes, struct rankinfo * rankinfo);

void copy_planes(const struct plane * planes, const unsigned int num_planes, struct context * context, struct buffers * buffers);

