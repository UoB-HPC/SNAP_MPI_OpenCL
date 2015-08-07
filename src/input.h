
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <mpi.h>

#include "problem.h"

void read_input(char *file, struct problem *globals);
void broadcast_problem(struct problem *globals, int rank);
void check_decomposition(struct problem * input);
