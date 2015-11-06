
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Just zeros the buffer!
kernel void zero_buffer(global double *buffer)
{
    size_t id = get_global_id(0);
    buffer[id] = 0.0;
}

// Zeros a 2D buffer
kernel void zero_buffer_2D(global double *buffer)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	buffer[j * get_global_size(0) + i] = 0.0;
}
