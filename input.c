
#include <stdio.h>
#include <mpi.h>

#include "problem.h"

void read_input(char *file, struct problem *globals)
{
    FILE *fp;
    fp = fopen(file, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: Could not open file %s\n", file);
        exit(-1);
    }
    char *line = NULL;
    ssize_t read;
    size_t len = 0;

    // Read the lines in the file
    while ((read = getline(&line, &len, fp)) != -1)
    {
        // Cycle over whitespace
        int i = 0;
        while (isspace(line[i]))
            i++;

        if (strncmp(line+i, "nx", strlen("nx")) == 0)
        {
            i += strlen("nx");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->nx = atoi(line+i);
        }
        else if (strncmp(line+i, "ny", strlen("ny")) == 0)
        {
            i += strlen("ny");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->ny = atoi(line+i);
        }
        else if (strncmp(line+i, "nz", strlen("nz")) == 0)
        {
            i += strlen("nz");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->nz = atoi(line+i);
        }
        else if (strncmp(line+i, "lx", strlen("lx")) == 0)
        {
            i += strlen("lx");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->lx = atoi(line+i);
        }
        else if (strncmp(line+i, "ly", strlen("ly")) == 0)
        {
            i += strlen("ly");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->ly = atoi(line+i);
        }
        else if (strncmp(line+i, "lz", strlen("lz")) == 0)
        {
            i += strlen("lz");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->lz = atoi(line+i);
        }
        else if (strncmp(line+i, "ng", strlen("ng")) == 0)
        {
            i += strlen("ng");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->ng = atoi(line+i);
        }
        else if (strncmp(line+i, "nang", strlen("nang")) == 0)
        {
            i += strlen("nang");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->nang = atoi(line+i);
        }
        else if (strncmp(line+i, "nmom", strlen("nmom")) == 0)
        {
            i += strlen("nmom");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->nmom = atoi(line+i);
        }
        else if (strncmp(line+i, "iitm", strlen("iitm")) == 0)
        {
            i += strlen("iitm");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->iitm = atoi(line+i);
        }
        else if (strncmp(line+i, "oitm", strlen("oitm")) == 0)
        {
            i += strlen("oitm");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->oitm = atoi(line+i);
        }
        else if (strncmp(line+i, "nsteps", strlen("nsteps")) == 0)
        {
            i += strlen("nsteps");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->nsteps = atoi(line+i);
        }
        else if (strncmp(line+i, "tf", strlen("tf")) == 0)
        {
            i += strlen("tf");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->tf = atof(line+i);
        }
        else if (strncmp(line+i, "epsi", strlen("epsi")) == 0)
        {
            i += strlen("epsi");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->epsi = atof(line+i);
        }
        else if (strncmp(line+i, "npex", strlen("npex")) == 0)
        {
            i += strlen("npex");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->npex = atoi(line+i);
        }
            else if (strncmp(line+i, "npey", strlen("npey")) == 0)
        {
            i += strlen("npey");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->npey = atoi(line+i);
        }
            else if (strncmp(line+i, "npez", strlen("npez")) == 0)
        {
            i += strlen("npez");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->npez = atoi(line+i);
        }
    }
    free(line);
}

void broadcast_problem(struct problem *globals, int rank)
{
    unsigned int ints[] = {
        globals->nx,
        globals->ny,
        globals->nz,
        globals->ng,
        globals->nang,
        globals->nmom,
        globals->iitm,
        globals->oitm,
        globals->nsteps,
        globals->npex,
        globals->npey,
        globals->npez
    };
    double doubles[] = {
        globals->lx,
        globals->ly,
        globals->lz,
        globals->tf,
        globals->epsi
    };
    MPI_Bcast(ints, 12, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(doubles, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        globals->nx = ints[0];
        globals->ny = ints[1];
        globals->nz = ints[2];
        globals->ng = ints[3];
        globals->nang = ints[4];
        globals->nmom = ints[5];
        globals->iitm = ints[6];
        globals->oitm = ints[7];
        globals->nsteps = ints[8];
        globals->npex = ints[9];
        globals->npey = ints[10];
        globals->npez = ints[11];

        globals->lx = doubles[0];
        globals->ly = doubles[1];
        globals->lz = doubles[2];
        globals->tf = doubles[3];
        globals->epsi = doubles[4];
    }
}
