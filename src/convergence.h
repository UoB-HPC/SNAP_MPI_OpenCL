
#pragma once

#include <stdbool.h>
#include <math.h>
#include <float.h>

#include "global.h"
#include "comms.h"
#include "problem.h"
#include "allocate.h"

/** \brief Check inner convergence - requires MPI_AllReduce*/
bool inner_convergence(struct problem * problem, struct rankinfo * rankinfo, struct memory * memory);
