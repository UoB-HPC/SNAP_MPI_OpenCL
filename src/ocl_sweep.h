
#pragma once

#include "global.h"
#include "sweep.h"

#include "ocl_global.h"
#include "ocl_buffers.h"

void sweep(const int octant, const unsigned int num_planes, const struct plane * planes, struct problem * problem, struct context * context, struct buffers * buffers);
