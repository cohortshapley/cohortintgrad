import pytest
import torch
import math
import cohortintgrad as csig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@pytest.fixture()
def ref():
    x = torch.Tensor([[0,0], [0,1],[1,0], [1,2]])
    return x

@pytest.fixture()
def ref_t():
    y = torch.Tensor([0,1,1,2])
    return y

def test_igcs_single(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    assert math.isclose(-0.5, IG.igcs_single(0)[0], rel_tol=0.01)
