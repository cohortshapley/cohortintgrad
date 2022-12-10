import pytest
import torch
import math
import cohortintgrad as csig
import numpy as np
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
    igcs_single0 = IG.igcs_single(0) #(-0.5, -0.5) in analytical result but contains numerical error in practice
    assert math.isclose(-0.5, igcs_single0[0], rel_tol=0.01)

def test_igcs_single_2(ref, ref_t): #the case where IGCS and CS dont match
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    igcs_single3 = IG.igcs_single(3)
    analytic = 3/7 + 2/(7 * np.sqrt(7)) * (np.arctan(1/np.sqrt(7)) - np.arctan(5/np.sqrt(7))) # analytic result
    assert math.isclose(analytic, igcs_single3[0], rel_tol=0.01)