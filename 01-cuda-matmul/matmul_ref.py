"""Pure-Python / NumPy reference for the CUDA matmul kernels.

`matmul_naive` mirrors the naive kernel (one dot product per output element).
`matmul_tiled` mirrors the tiled kernel: same tile loop, same zero padding.
All matrices are row-major float32.
"""
import numpy as np

TILE = 16


def matmul_naive(A, B):
    """Triple loop. Slow. Use only for small sizes."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            acc = np.float32(0)
            for k in range(K):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc
    return C


def matmul_tiled(A, B, tile=TILE):
    """Emulates one thread block per output tile. Edge tiles are zero padded."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.float32)
    for bi in range(0, M, tile):
        for bj in range(0, N, tile):
            acc = np.zeros((tile, tile), dtype=np.float32)
            for t in range(0, K, tile):
                As = np.zeros((tile, tile), dtype=np.float32)  # "shared memory"
                Bs = np.zeros((tile, tile), dtype=np.float32)
                a = A[bi:bi + tile, t:t + tile]
                b = B[t:t + tile, bj:bj + tile]
                As[:a.shape[0], :a.shape[1]] = a
                Bs[:b.shape[0], :b.shape[1]] = b
                acc += As @ Bs
            h = min(tile, M - bi)
            w_ = min(tile, N - bj)
            C[bi:bi + h, bj:bj + w_] = acc[:h, :w_]
    return C


def gflops(M, N, K, ms):
    return 2.0 * M * N * K / (ms * 1e6)
