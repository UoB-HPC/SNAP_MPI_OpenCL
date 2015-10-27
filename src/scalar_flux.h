
#pragma once

#include <math.h>

#include "global.h"

#include "ocl_global.h"
#include "ocl_buffers.h"


/** \brief Enqueue kernel to compute scalar flux (non-blocking) */
void compute_scalar_flux(struct problem * problem, struct rankinfo * rankinfo, struct context * context, struct buffers * buffers);
