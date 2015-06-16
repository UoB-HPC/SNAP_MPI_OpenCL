
#include <stdio.h>

#include "problem.h"

void read_input(char *file)
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

    struct problem globals;

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
            globals.nx = atoi(line+i);
        }
        else if (strncmp(line+i, "ny", strlen("ny")) == 0)
        {
            i += strlen("ny");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals.ny = atoi(line+i);
        }
        else if (strncmp(line+i, "nz", strlen("nz")) == 0)
        {
            i += strlen("nz");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals.nz = atoi(line+i);
        }
        else if (strncmp(line+i, "lx", strlen("lx")) == 0)
        {
            i += strlen("lx");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals.lx = atoi(line+i);
        }
        else if (strncmp(line+i, "ly", strlen("ly")) == 0)
        {
            i += strlen("ly");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals.ly = atoi(line+i);
        }
        else if (strncmp(line+i, "lz", strlen("lz")) == 0)
        {
            i += strlen("lz");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals.lz = atoi(line+i);
        }
        else if (strncmp(line+i, "ng", strlen("ng")) == 0)
        {
            i += strlen("ng");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals.ng = atoi(line+i);
        }
        else if (strncmp(line+i, "nang", strlen("nang")) == 0)
        {
            i += strlen("nang");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals.nang = atoi(line+i);
        }
        else if (strncmp(line+i, "nmom", strlen("nmom")) == 0)
        {
            i += strlen("nmom");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals.nmom = atoi(line+i);
        }
        else if (strncmp(line+i, "iitm", strlen("iitm")) == 0)
        {
            i += strlen("iitm");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals.iitm = atoi(line+i);
        }
        else if (strncmp(line+i, "oitm", strlen("oitm")) == 0)
        {
            i += strlen("oitm");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals.oitm = atoi(line+i);
        }
        else if (strncmp(line+i, "nsteps", strlen("nsteps")) == 0)
        {
            i += strlen("nsteps");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals.nsteps = atoi(line+i);
        }
        else if (strncmp(line+i, "epsi", strlen("epsi")) == 0)
        {
            i += strlen("epsi");
            // Cycle to after the equals
            while (isspace(line[i]) || line[i] == '=')
                i++;
            globals.epsi = atof(line+i);
        }
    }
    free(line);
}
