__kernel void multiply (__global float* A, __global float* B, __global float* C)
{
	uint local_id = get_local_id(0);
	C[local_id] = A[local_id] * B[local_id];
}