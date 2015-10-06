
#include "halos.h"

void allocate_halos(struct problem * problem, struct rankinfo * rankinfos, struct halo * halos)
{
    // Allocate halos
    if (problem->npex > 1)
    {
        halos->x.in = malloc(sizeof(double)*problem->nang*problem->ng*rankinfos->ny*rankinfos->nz*8);
        halos->x.out = malloc(sizeof(double)*problem->nang*problem->ng*rankinfos->ny*rankinfos->nz*8);
    }
    if (problem->npey > 1)
    {
        halos->y.in = malloc(sizeof(double)*problem->nang*problem->ng*rankinfos->nx*rankinfos->nz*8);
        halos->y.out = malloc(sizeof(double)*problem->nang*problem->ng*rankinfos->nx*rankinfos->nz*8);
    }
    if (problem->npez > 1)
    {
        halos->z.in = malloc(sizeof(double)*problem->nang*problem->ng*rankinfos->nx*rankinfos->ny*8);
        halos->z.out = malloc(sizeof(double)*problem->nang*problem->ng*rankinfos->nx*rankinfos->ny*8);
    }
}

void free_halos(struct problem * problem, struct halo * halos)
{
    if (problem->npex > 1)
    {
        free(halos->x.in);
        free(halos->x.out);
    }
    if (problem->npey > 1)
    {
        free(halos->y.in);
        free(halos->y.out);
    }
    if (problem->npez > 1)
    {
        free(halos->z.in);
        free(halos->z.out);
    }
}

void send_halo(int rank_to, struct rankinfo * rankinfo)
{
    // Don't need to do an MPI communication here
    if (rank_to == rankinfo->rank)
        return;
    
}
