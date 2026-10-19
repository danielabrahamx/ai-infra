import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import pytest, torch
import kernels

cuda_triton = pytest.mark.skipif(not (kernels.HAS_TRITON and torch.cuda.is_available()),
                                 reason="needs Triton + CUDA")


# ---- CPU: reference implementations vs independent torch ops (always run)
def test_rmsnorm_ref_matches_torch_rmsnorm():
    x, w = torch.randn(8, 257), torch.randn(257)
    m = torch.nn.RMSNorm(257, eps=1e-6)
    with torch.no_grad():
        m.weight.copy_(w)
    torch.testing.assert_close(kernels.rmsnorm_ref(x, w), m(x), rtol=1e-5, atol=1e-5)


def test_rmsnorm_ref_unit_rms():
    y = kernels.rmsnorm_ref(torch.randn(4, 64) * 7, torch.ones(64))
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(4), rtol=1e-3, atol=1e-3)


def test_softmax_ref_matches_torch():
    x = torch.randn(8, 1000) * 5
    torch.testing.assert_close(kernels.softmax_ref(x), torch.softmax(x, -1), rtol=1e-5, atol=1e-6)


def test_softmax_ref_stable_for_large_values():
    y = kernels.softmax_ref(torch.tensor([[1e4, 1e4 + 1.0, -1e4]]))
    assert torch.isfinite(y).all() and abs(y.sum().item() - 1) < 1e-6


def test_wrappers_raise_clearly_without_triton():
    if kernels.HAS_TRITON:
        pytest.skip("triton present")
    with pytest.raises(RuntimeError, match="triton-windows"):
        kernels.rmsnorm(torch.randn(2, 8), torch.randn(8))


# ---- GPU: Triton vs reference (skipped without CUDA)
@cuda_triton
@pytest.mark.parametrize("n", [63, 128, 1000, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_triton_rmsnorm(n, dtype):
    x = torch.randn(33, n, device="cuda", dtype=dtype); w = torch.randn(n, device="cuda", dtype=dtype)
    tol = 1e-5 if dtype == torch.float32 else 2e-3
    torch.testing.assert_close(kernels.rmsnorm(x, w), kernels.rmsnorm_ref(x, w), rtol=tol, atol=tol)


@cuda_triton
@pytest.mark.parametrize("n", [63, 128, 1000, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_triton_softmax(n, dtype):
    x = torch.randn(33, n, device="cuda", dtype=dtype) * 5
    tol = 1e-5 if dtype == torch.float32 else 2e-3
    torch.testing.assert_close(kernels.softmax(x), kernels.softmax_ref(x), rtol=tol, atol=tol)
