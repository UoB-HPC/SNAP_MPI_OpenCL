
#pragma once

#include "global.h"
#include "sweep.h"

#include "ocl_global.h"
#include "ocl_buffers.h"

/** \brief Enqueue the kernels to sweep a plane */
void sweep_plane(const int octant, const unsigned int plane_num, const struct plane * planes, struct problem * problem, struct rankinfo * rankinfo, struct context * context, struct buffers * buffers);
