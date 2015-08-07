
#include "halos.h"

void allocate_halos(struct problem * globals, struct rankinfo * locals, struct halo * halos)
{
    // Allocate halos
    if (globals->npex > 1)
    {
        halos->x.in = malloc(sizeof(double)*globals->nang*globals->ng*locals->ny*locals->nz*8);
        halos->x.out = malloc(sizeof(double)*globals->nang*globals->ng*locals->ny*locals->nz*8);
    }
    if (globals->npey > 1)
    {
        halos->y.in = malloc(sizeof(double)*globals->nang*globals->ng*locals->nx*locals->nz*8);
        halos->y.out = malloc(sizeof(double)*globals->nang*globals->ng*locals->nx*locals->nz*8);
    }
    if (globals->npez > 1)
    {
        halos->z.in = malloc(sizeof(double)*globals->nang*globals->ng*locals->nx*locals->ny*8);
        halos->z.out = malloc(sizeof(double)*globals->nang*globals->ng*locals->nx*locals->ny*8);
    }
}

void free_halos(struct problem * globals, struct halo * halos)
{
    if (globals->npex > 1)
    {
        free(halos->x.in);
        free(halos->x.out);
    }
    if (globals->npey > 1)
    {
        free(halos->y.in);
        free(halos->y.out);
    }
    if (globals->npez > 1)
    {
        free(halos->z.in);
        free(halos->z.out);
    }
}

void send_halo(int rank_to, struct rankinfo * local)
{
    // Don't need to do an MPI communication here
    if (rank_to == local->rank)
        return;
    
}
