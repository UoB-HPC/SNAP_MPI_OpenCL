
#pragma once

#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include "global.h"
#include "ocl_global.h"

static const bool profiling = true;

// Timers
struct timers
{
    double setup_time;
    double outer_source_time;
    double inner_source_time;
    double sweep_time;
    double reduction_time;
    double simulation_time;
    double convergence_time;
    double outer_params_time;
};

cl_event outer_source_event;
cl_event inner_source_event;

cl_event scalar_flux_event;
cl_event scalar_flux_moments_event;

cl_event velocity_delta_event;
cl_event denominator_event;

double wtime(void);

void outer_profiler(struct timers * timers);

void inner_profiler(struct timers * timers, struct problem * problem);

