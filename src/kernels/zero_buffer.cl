
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Just zeros the buffer!
kernel void zero_buffer(global double *buffer)
{
    size_t id = get_global_id(0);
    buffer[id] = 0.0;
}

// Zeros a 3D buffer
kernel void zero_buffer_3D(global double *buffer)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t k = get_global_id(2);
	size_t index =
		(k * get_global_size(1) * get_global_size(0)) +
		(j * get_global_size(0)) + i;
	buffer[index] = 0.0;
}
