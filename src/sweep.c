
#include "sweep.h"

void init_planes(struct plane* planes, unsigned int *num_planes, struct rankinfo * rankinfo)
{
    *num_planes = rankinfo->nx + rankinfo->ny + rankinfo->nz - 2;
    planes = malloc(sizeof(struct plane) * *num_planes);

    for (unsigned int p = 0; p < *num_planes; p++)
        planes[p].num_cells = 0;

    for (unsigned int k = 0; k < rankinfo->nz; k++)
        for (unsigned int j = 0; j < rankinfo->ny; j++)
            for (unsigned int i = 0; i < rankinfo->nx; i++)
            {
                unsigned int p = i + j + k;
                planes[p].num_cells += 1;
            }

    for (unsigned int p = 0; p < *num_planes; p++)
        planes[p].cell_ids = malloc(sizeof(struct cell_id) * planes[p].num_cells);

    unsigned int index[*num_planes];
    for (unsigned int p = 0; p < *num_planes; p++)
        index[p] = 0;

    for (unsigned int k = 0; k < rankinfo->nz; k++)
        for (unsigned int j = 0; j < rankinfo->ny; j++)
            for (unsigned int i = 0; i < rankinfo->nx; i++)
            {
                unsigned int p = i + j + k;
                planes[p].cell_ids[index[p]].i = i;
                planes[p].cell_ids[index[p]].j = j;
                planes[p].cell_ids[index[p]].k = k;
                index[p] += 1;
            }
}
