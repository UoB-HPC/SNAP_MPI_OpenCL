
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Just zeros the buffer!
kernel void zero_buffer(global double *buffer)
{
    size_t id = get_global_id(0);
    buffer[id] = 0.0;
}
