import time
import numpy as np
import torch
import ffn_23

# nwarmup = 10
# ntest = 100

nwarmup = 0
ntest = 1

def show_time(func):
    times = list()
    res = None
    # GPU warm up
    for _ in range(nwarmup):
        res = func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        res = func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time-start_time) * 1e6)
    return times, res

# def compare_tensors(res_cuda, res_torch, tolerance):
#     if res_cuda.shape != res_torch.shape:
#         print("Tensor shapes are different.")
#         return False
#     res_cuda_list = res_cuda.view(-1).tolist()
#     res_torch_list = res_torch.view(-1).tolist()

#     for index, (a, b) in enumerate(zip(res_cuda_list, res_torch_list)):
#         if (abs(b) == 0 and abs(a) > tolerance or abs(b) > 0 and abs(a - b) / abs(b) > tolerance):
#             print(f"Index {index}: diff = {a-b}")
#             return False
#     return True

def compare_tensors(res_cuda, res_torch, tolerance):
    if res_cuda.shape != res_torch.shape:
        print("Tensor shapes are different.")
    res_cuda_list = res_cuda.view(-1).tolist()
    res_torch_list = res_torch.view(-1).tolist()

    for index, (a, b) in enumerate(zip(res_cuda_list, res_torch_list)):
        if (abs(b) == 0 and abs(a) > tolerance or abs(b) > 0 and abs(a - b) / abs(b) > tolerance):
        # if (abs(a - b) > tolerance):
            print(f"Index {index}: diff = {a-b}; res_cuda_list: {res_cuda_list[index]}; res_torch_list: {res_torch_list[index]}")
        # print(f"Index {index}: diff = {a-b}; res_cuda_list: {res_cuda_list[index]}; res_torch_list: {res_torch_list[index]}")



mat_row = 4096
mat_col = 11008
fatrelu_threshold = 0.

# file_path = 'pytorch/sparse_vec.npy'
# data = np.load(file_path)
# first_row = data[9, :]
# vec_sparse = torch.tensor(first_row, device="cuda:0", dtype=torch.bfloat16)

for idx in range(1):

    batch_size = 2
    vec_sparse = torch.rand(mat_col, device="cuda:0", dtype=torch.bfloat16)
    vec_sparse = torch.relu(vec_sparse - 1 / 10)
    # vec_sparse = torch.relu(vec_sparse - 9 / 10)
    # vec_sparse = torch.arange(-10000, -10000 + 11008, device="cuda:0", dtype=torch.bfloat16)
    # vec_sparse = torch.relu(vec_sparse)


    print(">>> act_rate:", round(torch.sum(vec_sparse > 0).item() * 100 / vec_sparse.numel(), 2))
    vec_sparse = vec_sparse.unsqueeze(0).expand(batch_size, -1).contiguous()
    # print(f"vec_sparse.shape: {vec_sparse.shape}")
    
    vec = torch.rand(mat_row, device="cuda:0", dtype=torch.bfloat16)
    vec = vec.unsqueeze(0).expand(batch_size, -1).contiguous()
    # print(f"vec.shape: {vec.shape}")
    # vec[:2] = 1
    # vec[0, :2] = 1
    # vec[1, :2] = 1
    mat = torch.rand(mat_row, mat_col, device="cuda:0", dtype=torch.bfloat16)

    
    cuda_res = torch.zeros(mat_col, device="cuda:0", dtype=torch.bfloat16)
    cuda_res = cuda_res.unsqueeze(0).expand(batch_size, -1).contiguous()
    # print(f"cuda_res.shape: {vec.shape}")
    # exit()


    def run_torch():
        res = torch.matmul(vec, mat)
        res = res * vec_sparse
        # res[torch.eq(vec_sparse, 0)] = 0
        res = res.contiguous()
        return res

    def run_cuda():
        ffn_23.torch_launch_ffn_fuse_23(vec_sparse, vec, mat, cuda_res, batch_size, mat_row, mat_col, fatrelu_threshold)
        return cuda_res

    # 使用大量计算清空 GPU 缓存
    for _ in range(100):
        x = torch.rand(1000, 2000, device="cuda:0", dtype=torch.bfloat16)
        y = torch.rand(2000, 1000, device="cuda:0", dtype=torch.bfloat16)
        x = x ** 2
        y = y ** 0.5
        z = x @ y
    print("Running torch...")
    torch_time, torch_res = show_time(run_torch)
    print("Torch time:  {:.4f} us".format(np.mean(torch_time)))

    mat = mat.t().contiguous() # mat转变为列主序存储后再传给kernel
    # 使用大量计算清空 GPU 缓存
    for _ in range(100):
        x = torch.rand(1000, 2000, device="cuda:0", dtype=torch.bfloat16)
        y = torch.rand(2000, 1000, device="cuda:0", dtype=torch.bfloat16)
        x = x ** 2
        y = y ** 0.5
        z = x @ y
    print("Running cuda...")
    cuda_time, cuda_res = show_time(run_cuda)
    print("Cuda time:  {:.4f} us".format(np.mean(cuda_time)))

    tolerance = 0.01
    compare_tensors(cuda_res, torch_res, tolerance)
    # if not compare_tensors(cuda_res, torch_res, tolerance):
    #     from IPython import embed
    #     embed()
    #     exit()
