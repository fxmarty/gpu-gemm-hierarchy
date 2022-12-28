# A description of a simple GEMM hierarchy on Nvidia GPUs

This doc is heavily based on the [references](#references), and describes how a GEMM *C = A * B* is abstraced into smaller pieces, all down to the GPU thread level. Dummy Python code is used for illustration.

## Tiling

### Thread block tile

For each block, a tile (Mtile, Ktile) of A and (Ktile, Ntile) of B are loaded into the shared memory, accessible by all warps.

```python
import torch

M, N, K = 48, 128, 12

A = torch.rand((M, K))
B = torch.rand((K, N))

# for simplification tile shapes are all multiple of matrix shapes
# otherwise we would need to check matrix bounds and mask out of bounds values by 0s in tiles
Mtile = M // 6
Ntile = N // 4
Ktile = K // 2

print("Mtile:", Mtile)
print("Ntile:", Ntile)
print("Ktile:", Ktile)

# This version does the tiling on the A and B matrices as well,
# allowing to load only a submatrix (Mtile, Ktile) of A, submatrix (Ktile, Ntile) of B
# into the inner loop `for kb in range(0, K, Ktile)`.
# Therefore, the three innermost loops are dispatched to thread blocks for each
# (mb, nb, kb).
# Meaning that given tiles (Mtile, Ktile) of A and (Ktile, Ntile) of B are loaded
# into shared memory for a thread block.

output = torch.zeros((M, N))
for mb in range(0, M, Mtile):  # iterate over M dimension
    for nb in range(0, N, Ntile):  # iterate over N dimension
        for kb in range(0, K, Ktile):

            # classic GEMM
            for k in range(Ktile):
                for i in range(Mtile):  # compute one tile
                    for j in range(Ntile):
                        row = mb + i
                        col = nb + j
                        output[row][col] += A[row][kb + k] * B[kb + k][col]

assert torch.allclose(A @ B, output)
```


### Warp tile

The tile output of C of size (Mtile, Ntile) is partitioned between the warps. There is still some data reload from shared memory (for example a submatrice of A is reloaded 4 times from Warps 0, 2, 4, 6).

```python
import torch

M, N, K = 48, 128, 12

A = torch.rand((M, K))
B = torch.rand((K, N))

# for simplification tile shapes are all multiple of matrix shapes
# otherwise we would need to check matrix bounds and mask out of bounds values by 0s in tiles
Mtile = M // 6
Ntile = N // 4
Ktile = K // 2

print("Mtile:", Mtile)
print("Ntile:", Ntile)
print("Ktile:", Ktile)

warp_per_col = 2
warp_per_row = 4
warp_per_inner = 2

warp_m = Mtile // warp_per_col
warp_n = Ntile // warp_per_row
warp_k = Ktile // warp_per_inner

print("output col items per warp:", warp_m)
print("output row items per warp:", warp_n)
print("input inner items per warp:", warp_k)

output = torch.zeros((M, N))
for mb in range(0, M, Mtile):  # iterate over M dimension
    for nb in range(0, N, Ntile):  # iterate over N dimension
        for kb in range(0, K, Ktile):

            # a thread block handle it from here
            # load A and B tiles in shared memory here and
            # compute GEMM over (warpm_m * warp_k) and (warp_k * warp_n)
            for kw in range(0, Ktile, warp_k):
                for iw in range(0, Mtile, warp_m):
                    for jw in range(0, Ntile, warp_n):

                        # classic GEMM (handled by threads)
                        for k in range(warp_k):
                            for i in range(warp_m):
                                for j in range(warp_n):
                                    row = mb + iw + i
                                    col = nb + jw + j

                                    output[row][col] += A[row][kb + kw + k] * B[kb + kw + k][col]

assert torch.allclose(A @ B, output)
```

### Thread tile

Each thread is responsible for processing a certain number of elements. Note that **threads cannot access each other’s registers**., so the idea is to choose an organization that enables reuse of values held in registers for multiple math instructions.

```python
import torch

M, N, K = 48, 128, 12

A = torch.rand((M, K))
B = torch.rand((K, N))

# for simplification tile shapes are all multiple of matrix shapes
# otherwise we would need to check matrix bounds and mask out of bounds values by 0s in tiles
Mtile = M // 6
Ntile = N // 4
Ktile = K // 2

print("Mtile:", Mtile)
print("Ntile:", Ntile)
print("Ktile:", Ktile)

warp_per_col = 2
warp_per_row = 4
warp_per_inner = 2

warp_m = Mtile // warp_per_col
warp_n = Ntile // warp_per_row
warp_k = Ktile // warp_per_inner

print("output col items per warp:", warp_m)
print("output row items per warp:", warp_n)
print("input inner items per warp:", warp_k)

thread_m = 4
tread_n = 4
thread_k = 1

output = torch.zeros((M, N))

for mb in range(0, M, Mtile):  # iterate over M dimension
    for nb in range(0, N, Ntile):  # iterate over N dimension
        for kb in range(0, K, Ktile):
            # a block handle the GEMM of (Mtile, Ktile) of A and (Ktile, Ntile) of B
            # to compute a tile (Mtile, Ntile) of the output
            # load A and B tiles in shared memory here

            for iw in range(0, Mtile, warp_m):
                for jw in range(0, Ntile, warp_n):
                    for kw in range(0, Ktile, warp_k):
                        # split the output tile (Mtile, Ntile) into smaller ones (warpm_m, warp_n)
                        # each dispatched on a warp
                        # a warp compute the GEMM over (warpm_m * warp_k) and (warp_k * warp_n)

                        for kt in range(0, warp_k, thread_k):
                            for it in range(0, warp_m, thread_m):
                                for jt in range(0, warp_n, tread_n):
                                    # a thread handle it from here, does a classic GEMM
                                    # we tile at the thread level because registers are not shared

                                    for i in range(thread_m):
                                        for j in range(tread_n):
                                            for k in range(thread_k):
                                                row = mb + iw + it + i
                                                col = nb + jw + + jt + j

                                                output[row][col] += A[row][kb + kw + kt + k] * B[kb + kw + kt + k][col]

assert torch.allclose(A @ B, output)
```

## References

* Why the loop order matters: https://stackoverflow.com/questions/7395556/why-does-the-order-of-loops-in-a-matrix-multiply-algorithm-affect-performance
* Loop order GIFs: https://www.adityaagrawal.net/blog/architecture/matrix_multiplication
* Tiling GIFs: https://penny-xu.github.io/blog/tiled-matrix-multiplication
* Python code: https://github.com/ELS-RD/kernl/blob/main/tutorial/1%20-%20tiled%20matmul.ipynb
* CUTLASS: Fast Linear Algebra in CUDA C++: https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
* CUTLASS: Software Primitives for Dense Linear Algebra at All Levels and Scales within CUDA: https://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf
* Efficient GEMM in CUDA: https://github.com/NVIDIA/cutlass/blob/master/media/docs/efficient_gemm.md
