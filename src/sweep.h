
#pragma once

#include <stdlib.h>

#include "global.h"

struct cell_id
{
    unsigned int i, j, k;
};

struct plane
{
    unsigned int num_cells;
    struct cell_id * cell_ids;
};


void init_planes(struct plane* planes, unsigned int *num_planes, struct rankinfo * rankinfo);
