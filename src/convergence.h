
#pragma once

#include <stdbool.h>
#include <math.h>
#include <float.h>

#include "global.h"
#include "comms.h"
#include "problem.h"
#include "allocate.h"

/** \brief Check inner convergence - requires MPI_AllReduce*/
bool inner_convergence(const struct problem * problem, const struct rankinfo * rankinfo, const struct memory * memory);
