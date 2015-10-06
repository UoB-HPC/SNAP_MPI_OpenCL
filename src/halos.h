
#pragma once

#include "problem.h"
#include "comms.h"
#include <stdlib.h>


// Halo data arrays
struct dir
{
	double * in;
	double * out;
};
struct halo
{
	struct dir x, y, z;
};

void allocate_halos(struct problem * problem, struct rankinfo * locals, struct halo * halos);
void free_halos(struct problem * problem, struct halo * halos);