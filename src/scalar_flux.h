
#pragma once

#include <math.h>

#include "global.h"

#include "ocl_global.h"
#include "ocl_buffers.h"

#include "profiler.h"

/** \brief Enqueue kernel to compute scalar flux (non-blocking) */
void compute_scalar_flux(struct problem * problem, struct rankinfo * rankinfo, struct context * context, struct buffers * buffers);

/** \brief Enqueue kernel to compute scalar flux moments (non-blocking) */
void compute_scalar_flux_moments(struct problem * problem, struct rankinfo * rankinfo, struct context * context, struct buffers * buffers);


/** \brief Copy the scalar flux back to the host (choose blocking) */
void copy_back_scalar_flux(struct problem *problem, struct rankinfo * rankinfo, struct context * context, struct buffers * buffers, double * scalar_flux, cl_bool blocking);
