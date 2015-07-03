
#pragma once

#include "problem.h"
#include <stdlib.h>


// Halo data arrays
struct dir
{
	double * in;
	double * out;
};
struct halo
{
	struct dir x, y;
};

void allocate_halos(struct problem * globals, struct rankinfo * locals, struct halo * halos);
void free_halos(struct halo * halos);
