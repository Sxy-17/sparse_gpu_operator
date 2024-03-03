#include <cuda.h>
// #include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>

#define ROW_OPT 4096
#define COL_OPT 11008
#define MAX_BATCH_SIZE_DIV_32 8

// 当前代码必须提前知道__shared__ float4 mat_up_shared[ROW_OPT / 8];的大小

// Col major
__global__ void ffn_fuse_23(nv_bfloat16 *vec_sparse, nv_bfloat16 *vec_input,
                            nv_bfloat16 *mat_up, nv_bfloat16 *res, unsigned int batch_size,
                            unsigned int mat_row, unsigned int mat_col, float threshold)
{
#ifdef USE_CONSTANT
    mat_row = ROW_OPT;
    mat_col = COL_OPT;
#endif

    int col_id = blockIdx.y * 32 + threadIdx.y;
    int num_per_threadx = mat_row / 32;
    int row_chunk_id = threadIdx.x;
    int row_id = row_chunk_id * num_per_threadx;

    nv_bfloat16 *vec_sparse_p = &vec_sparse[col_id];            // per thread
    nv_bfloat16 *vec_input_p = &vec_input[row_id];              // per thread
    nv_bfloat16 *mat_up_p = &mat_up[col_id * mat_row + row_id]; // per thread, col-major
    nv_bfloat16 *res_p = &res[col_id];                          // per thread

    float4 *vec_input_f4 = reinterpret_cast<float4 *>(vec_input_p);
    float4 vec_input_f_val;
    float4 *mat_up_f4 = reinterpret_cast<float4 *>(mat_up_p);
    float4 mat_up_f_val;

    nv_bfloat16 vec_sparse_val;

    // bit-wise index
    bool is_activated = false;
    unsigned int batch_idx_arr[MAX_BATCH_SIZE_DIV_32] = {0}; // max supported batch size is 256
    for (int batch_iter = 0; batch_iter < batch_size; batch_iter++)
    {
        vec_sparse_val = *(vec_sparse_p + batch_iter * mat_col);
        // if(threadIdx.x == 0 && threadIdx.y == 0 && batch_iter == 0) printf("col = %d \tvec_sparse_val = %f \n", col_id, __bfloat162float(vec_sparse_val));
        if (__bfloat162float(vec_sparse_val) > threshold)
        {
            is_activated = true;
            // Set the corresponding bit in batch_idx_arr to 1
            batch_idx_arr[batch_iter / 32] |= (1u << (batch_iter % 32));
        }
    }
    // if(threadIdx.x == 0 && col_id == 11000) {
    // printf("col = %d \n", col_id);
    // printf("batch_idx_arr[0] = %u \tbatch_idx_arr[7] = %u\n", batch_idx_arr[0], batch_idx_arr[7]);}
    // printf("batch_idx_arr[0] = %x \tbatch_idx_arr[1] = %x\n", batch_idx_arr[0], batch_idx_arr[1]);
    // printf("batch_idx_arr[0]");

    // 改成if (is_activated) {xxx}
    if (!is_activated)
        ; // TODO 没有被active的batch也应该要写入0
    else
    {
    // if(threadIdx.x == 0 && col_id == 0)  printf("HI2, X = %d; Y = %d\n", blockIdx.x, blockIdx.y);
    if(threadIdx.x == 0 && col_id == 0)  printf("HI, Y = %d\n", threadIdx.y);
        // __shared__ float4 mat_up_shared[ROW_OPT / 8];
        __shared__ float4 mat_up_shared[16]; // num_per_threadx /8 
        bool use_shared_memory = false;

        // Loop through batch_idx_arr and get decimal value of batch_activated
        for (int batch_chunk_iter = 0; batch_chunk_iter < MAX_BATCH_SIZE_DIV_32; batch_chunk_iter++)
        {
            unsigned int chunk = batch_idx_arr[batch_chunk_iter];
            int bit_position = 0;
            while (chunk > 0)
            {
                if (chunk & 1) // found the activated batch
                {
                    unsigned int batch_activated = batch_chunk_iter * 32 + bit_position;
                    // Reset sum for each iteration
                    float sum = 0.0f;

#pragma unroll
                    for (int i = 0; i < (num_per_threadx / 8); i++) // read eight 16-bit elements in each loop
                    {
                        vec_input_f_val = vec_input_f4[(mat_row / 8) * batch_activated + i];
                        const nv_bfloat162 *vec_input_h1 = (nv_bfloat162 *)&vec_input_f_val.x;
                        const nv_bfloat162 *vec_input_h2 = (nv_bfloat162 *)&vec_input_f_val.y;
                        const nv_bfloat162 *vec_input_h3 = (nv_bfloat162 *)&vec_input_f_val.z;
                        const nv_bfloat162 *vec_input_h4 = (nv_bfloat162 *)&vec_input_f_val.w;

                        if (!use_shared_memory)
                        {
                            mat_up_f_val = mat_up_f4[i];
                        }
                        else
                        {
                            // Read mat_up_f_val from shared memory
                            // mat_up_f_val = mat_up_f4[i];
                            mat_up_f_val = mat_up_shared[i];
                        }

                        const nv_bfloat162 *mat_up_h1 = (nv_bfloat162 *)&mat_up_f_val.x;
                        const nv_bfloat162 *mat_up_h2 = (nv_bfloat162 *)&mat_up_f_val.y;
                        const nv_bfloat162 *mat_up_h3 = (nv_bfloat162 *)&mat_up_f_val.z;
                        const nv_bfloat162 *mat_up_h4 = (nv_bfloat162 *)&mat_up_f_val.w;

                        sum += __bfloat162float(vec_input_h1->x) * __bfloat162float(mat_up_h1->x);
                        sum += __bfloat162float(vec_input_h1->y) * __bfloat162float(mat_up_h1->y);
                        sum += __bfloat162float(vec_input_h2->x) * __bfloat162float(mat_up_h2->x);
                        sum += __bfloat162float(vec_input_h2->y) * __bfloat162float(mat_up_h2->y);
                        sum += __bfloat162float(vec_input_h3->x) * __bfloat162float(mat_up_h3->x);
                        sum += __bfloat162float(vec_input_h3->y) * __bfloat162float(mat_up_h3->y);
                        sum += __bfloat162float(vec_input_h4->x) * __bfloat162float(mat_up_h4->x);
                        sum += __bfloat162float(vec_input_h4->y) * __bfloat162float(mat_up_h4->y);

                        // Update use_shared_memory for the next iteration
                        if (!use_shared_memory)
                        {
                            // Write mat_up_f_val to shared memory for the next iteration
                            // mat_up_shared[row_chunk_id * (num_per_threadx / 8) + i] = mat_up_f_val;
                            mat_up_shared[i] = mat_up_f_val;
                        }
                    __syncthreads();
                    __syncwarp();
    // if(threadIdx.x == 0 && col_id == 0)  printf("i = %d \t shared index = %d\n", i, (row_chunk_id * (num_per_threadx / 8) + i));
                    }
                    // __syncwarp();
                    // __syncthreads();

                    __shared__ float warp_sum[32];
                    warp_sum[threadIdx.y] = 0.0f;
                    atomicAdd(&warp_sum[threadIdx.y], sum);
                    __syncthreads();

                    if (threadIdx.x == 0)
                    {
                        float sum = warp_sum[threadIdx.y];
                        // vec_sparse_val = *(vec_sparse_p + batch_activated * mat_col);
                        // sum = sum * __bfloat162float(vec_sparse_val);
                        *(res_p + batch_activated * mat_col) = __float2bfloat16(sum);
                    }

                    use_shared_memory = true;
                } // end of one batch
                chunk >>= 1;
                bit_position++;
            } // end of one index chunk
        }     // end of all index chunks
        __syncwarp();

    }
}

void launch_ffn_fuse_23(nv_bfloat16 *vec_sparse, nv_bfloat16 *vec_input,
                        nv_bfloat16 *mat_up, nv_bfloat16 *res, unsigned int batch_size,
                        unsigned int mat_row, unsigned int mat_col, float threshold)
{
#ifdef USE_CONSTANT
    mat_row = ROW_OPT;
    mat_col = COL_OPT;
#endif
    dim3 grid_dim(1, mat_col / 32);
    dim3 block_dim(32, 32, 1);

    ffn_fuse_23<<<grid_dim, block_dim>>>(vec_sparse, vec_input, mat_up, res, batch_size,
                                         mat_row, mat_col, threshold);
}