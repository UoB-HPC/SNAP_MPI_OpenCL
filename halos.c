
#include "halos.h"

void allocate_halos(struct problem * globals, struct rankinfo * locals, struct halo * halos)
{
	// Allocate halos
    halos->x.in = malloc(sizeof(double)*globals->nang*globals->ng*locals->ny*locals->nz*8);
    halos->x.out = malloc(sizeof(double)*globals->nang*globals->ng*locals->ny*locals->nz*8);
    halos->y.in = malloc(sizeof(double)*globals->nang*globals->ng*locals->nx*locals->nz*8);
    halos->y.out = malloc(sizeof(double)*globals->nang*globals->ng*locals->nx*locals->nz*8);

}

void free_halos(struct halo * halos)
{
    free(halos->x.in);
    free(halos->x.out);
    free(halos->y.in);
    free(halos->y.out);
}
