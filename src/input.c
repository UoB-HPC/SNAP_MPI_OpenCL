
#include "input.h"

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
            globals->lx = atof(line+i);
        }
        else if (strncmp(line+i, "ly", strlen("ly")) == 0)
        {
            i += strlen("ly");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->ly = atof(line+i);
        }
        else if (strncmp(line+i, "lz", strlen("lz")) == 0)
        {
            i += strlen("lz");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->lz = atof(line+i);
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
        else if (strncmp(line+i, "chunk", strlen("chunk")) == 0)
        {
            i += strlen("chunk");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals->chunk = atoi(line+i);
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
        globals->dx,
        globals->dy,
        globals->dz,
        globals->dt,
        globals->tf,
        globals->epsi
    };
    MPI_Bcast(ints, 12, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(doubles, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
        globals->dx = doubles[3];
        globals->dy = doubles[4];
        globals->dz = doubles[5];
        globals->dt = doubles[6];
        globals->tf = doubles[7];
        globals->epsi = doubles[8];
    }
    globals->cmom = globals->nmom * globals->nmom;
}

void check_decomposition(struct problem * input)
{
    bool err = false;

    // Check we have at least a 1x1x1 processor array
    if (input->npex < 1)
    {
        fprintf(stderr, "Input error: npex must be >= 1\n");
        err = true;
    }
    if (input->npey < 1)
    {
        fprintf(stderr, "Input error: npey must be >= 1\n");
        err = true;
    }

    // Check npez = 1 (regular KBA for now)
    if (input->npez != 1)
    {
        fprintf(stderr, "Input error: npez must equal 1 (for KBA)\n");
        err = true;
    }

    // Check grid divides across processor array
    if (input->nx % input->npex != 0)
    {
        fprintf(stderr, "Input error: npex should divide nx\n");
        err = true;
    }
    if (input->ny % input->npey != 0)
    {
        fprintf(stderr, "Input error: npey should divide ny\n");
        err = true;
    }
    if (input->nz % input->npez != 0)
    {
        fprintf(stderr, "Input error: npez should divide nz\n");
        err = true;
    }

    // Check chunk size divides nz
    if (input->nz % input->chunk != 0)
    {
        fprintf(stderr, "Input error: chunk should divide nz\n");
        err = true;
    }

    if (err)
        exit(-1);
}

