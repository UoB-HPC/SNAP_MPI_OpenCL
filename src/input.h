
#pragma once

/** \file
* \brief Handles reading in problem data from file
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <mpi.h>

#include "problem.h"

/** \brief Read the input data from a file and populare the problem structure
*
* Note: should only be called by master rank.
* \param nx Number of cells in x direction
* \param ny Number of cells in y direction
* \param nz Number of cells in z direction
* \param lx Physical size in x direction
* \param ly Physical size in y direction
* \param lz Physical size in z direction
* \param ng Number of energy groups
* \param nang Number of angles per octant
* \param nmom Number of moments
* \param iitm Maximum number of inner iterations per outer
* \param oitm Maximum number of outer iterations per timestep
* \param nsteps Number of timesteps
* \param tf Physical time to simulate
* \param epsi Convergence criteria
* \param npex MPI decomposition: number of processors in x direction
* \param npey MPI decomposition: number of processors in y direction
* \param npez MPI decomposition: number of processors in z direction
* \param chunk Number of x-y planes to calculate before communication
*/
void read_input(char *file, struct problem *globals);

/** \brief Send problem data from master to all MPI ranks */
void broadcast_problem(struct problem *globals, int rank);

/** \brief Check MPI decomposition is valid */
void check_decomposition(struct problem * input);
