import numpy as np
import pytest
from matmul_ref import matmul_naive, matmul_tiled, TILE

SHAPES = [(1, 1, 1), (16, 16, 16), (17, 19, 23), (33, 5, 40), (48, 64, 32)]


def rand(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_tiled_matches_numpy(M, N, K):
    A, B = rand((M, K), 0), rand((K, N), 1)
    np.testing.assert_allclose(matmul_tiled(A, B), A @ B, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("M,N,K", [(1, 1, 1), (5, 7, 9), (17, 3, 20)])
def test_naive_matches_numpy(M, N, K):
    A, B = rand((M, K), 2), rand((K, N), 3)
    np.testing.assert_allclose(matmul_naive(A, B), A @ B, rtol=1e-4, atol=1e-4)


def test_identity():
    A = rand((20, 20), 4)
    np.testing.assert_allclose(matmul_tiled(A, np.eye(20, dtype=np.float32)), A, atol=1e-6)


def test_cuda_source_has_tiling():
    # Guard: the .cu file must keep shared-memory tiling and a TILE macro.
    src = open("src/kernels.cuh").read()
    assert "__shared__" in src and "__syncthreads" in src and "#define TILE" in src
