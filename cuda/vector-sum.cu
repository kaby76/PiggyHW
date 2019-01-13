/**
 * Vector sum: C = A + B.
 */

#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "float.h"
#include <builtin_types.h>

// System includes
#include <stdio.h>
#include <assert.h>


extern "C" {

    /**
     * Vector sum on the GPU: C = A + B
     */
    __global__ void VectorSumParallel(int *A, int *B, int *C, int n)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n)
        {
            C[i] = A[i] + B[i];
        }
    }
}
