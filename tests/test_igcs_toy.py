import math

import numpy as np
import pytest
import torch

import cohortintgrad as csig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def ref():
    x = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 2]])
    return x


@pytest.fixture()
def ref_t():
    y = torch.Tensor([0, 1, 1, 2])
    return y


@pytest.fixture()
def diagonal():
    n_step = 5
    z = torch.vstack(
        [
            torch.arange(0, 1 + 1.0 / n_step, 1.0 / n_step)
            for _ in range(2)  #: ref.shape[1]
        ]
    ).T.to(device)
    return z


def test_igcs_single(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    igcs_single0 = IG.igcs_single(
        0
    )  # (-0.5, -0.5) in analytical result but contains numerical error in practice
    np.testing.assert_allclose(
        [-0.5, -0.5], igcs_single0.to("cpu").detach().numpy(), rtol=0.01
    )


def test_igcs_single_2(ref, ref_t):  # the case where IGCS and CS dont match
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    igcs_single3 = IG.igcs_single(3)
    analytic_0 = 3 / 7 + 2 / (7 * np.sqrt(7)) * (
        np.arctan(1 / np.sqrt(7)) - np.arctan(5 / np.sqrt(7))
    )  # analytic result
    analytic_1 = 4 / 7 - 2 / (7 * np.sqrt(7)) * (
        np.arctan(1 / np.sqrt(7)) - np.arctan(5 / np.sqrt(7))
    )  # analytic result
    np.testing.assert_allclose(
        [analytic_0, analytic_1], igcs_single3.to("cpu").detach().numpy(), rtol=0.01
    )


def test_summation_rule(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    igcs_single3 = IG.igcs_single(3)
    summation = torch.sum(igcs_single3).item()
    y_diff = (ref_t[3] - torch.mean(ref_t)).item()
    assert math.isclose(y_diff, summation, rel_tol=0.01)


def test_dissimilarity(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    dissimilar = (
        IG._similarity(ref[:, 1], ref[3, 1]).to("cpu").detach().numpy()
    )  # "2" in ref is not similar to other 2nd element of ref but always pick itself
    np.testing.assert_array_equal(dissimilar, np.array([0, 0, 0, 1]).astype(np.int32))


def test_similar_ratio(ref, ref_t):
    IG = csig.CohortIntGrad(
        ref.to(device), ref_t.to(device), ratio=0.5, n_step=500
    )  # similar ratio is so large that similar to "1" in 2nd data
    dissimilar = IG._similarity(ref[:, 1], ref[3, 1]).to("cpu").detach().numpy()
    np.testing.assert_array_equal(dissimilar, np.array([0, 1, 0, 1]).astype(np.int32))


def test_sjw_origin(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    sjw_origin = (
        IG._sjw(torch.Tensor([0.0]).to(device), ref[:, 1], ref[3, 1])
        .to("cpu")
        .detach()
        .numpy()
    )  # at the origin, 1 + z_j (S_j(x_i)-1) = 1 regardless of similarity
    np.testing.assert_allclose(np.ones((1, ref.shape[0])), sjw_origin, rtol=0.01)


def test_sjw_target(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    sjw_target = (
        IG._sjw(torch.Tensor([1.0]).to(device), ref[:, 0], ref[3, 0])
        .to("cpu")
        .detach()
        .numpy()
    )  # at (1,1) in z, 1 + z_j (S_j(x_i)-1) = 0 in dissimilar element and =1 in similar
    np.testing.assert_allclose([[0, 0, 1, 1]], sjw_target, rtol=0.01)


def test_sjw_mlt_zj_once(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    sjw_target = (
        IG._sjw(torch.Tensor([0.0, 1.0]).to(device), ref[:, 0], ref[3, 0])
        .to("cpu")
        .detach()
        .numpy()
    )  # 1 + z_j (S_j(x_i)-1) =1  on the edge near to origin
    np.testing.assert_allclose([[1, 1, 1, 1], [0, 0, 1, 1]], sjw_target, rtol=0.01)


def test_sjw_midpoint(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    sjw_midpoint = (
        IG._sjw(torch.Tensor([0.5]).to(device), ref[:, 0], ref[3, 0])
        .to("cpu")
        .detach()
        .numpy()
    )  # 1 + z_j (S_j(x_i)-1) depends on the coordinate z in dissimilar data
    np.testing.assert_allclose([[0.5, 0.5, 1, 1]], sjw_midpoint, rtol=0.01)


def test_sz_origin(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    z = torch.Tensor([[0, 0]]).to(device)
    sz_origin = (
        IG._sz(t_id=3, z=z).to("cpu").detach().numpy()
    )  # soft similarity function sz is a product over features about _sjw, which is always 1 at the origin
    np.testing.assert_allclose([[1, 1, 1, 1]], sz_origin, rtol=0.01)


def test_sz_target(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    z = torch.Tensor([[1, 1]]).to(device)
    sz_target = (
        IG._sz(t_id=3, z=z).to("cpu").detach().numpy()
    )  # soft similarity function sz is a product over features about _sjw -> Hadamard prod of [0, 0, 1, 1] and [0, 0, 0, 1]
    np.testing.assert_allclose([[0, 0, 0, 1]], sz_target, rtol=0.01)


def test_sz_diagonal(ref, ref_t, diagonal):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    sz_diagonal = (
        IG._sz(t_id=3, z=diagonal).to("cpu").detach().numpy()
    )  # soft similarity function sz is a product over features about _sjw
    # -> vstack of (Hadamard prod of [1-z, 1-z, 1, 1] (test_sjw_midpoint above) and [1-z, 1-z, 1-z, 1]) on diagonal line z1
    step = diagonal.shape[0] - 1
    sz_true = np.vstack(
        [
            np.array([[(1 - i / step) ** 2, (1 - i / step) ** 2, 1 - i / step, 1]])
            for i in range(step + 1)
        ]
    )
    np.testing.assert_allclose(sz_true, sz_diagonal, rtol=0.01)


def test_sz_nondiagonal(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    nondiag_z = torch.Tensor([[0, 1], [1.0, 0.0]]).to(device)
    sz_nondiag = (
        IG._sz(t_id=3, z=nondiag_z).to("cpu").detach().numpy()
    )  # the direction whose coordinate is zero contributes as multiplication of one
    # then it is samne as sjw(zj=1, ref[:, j], ref[3, j])
    np.testing.assert_allclose([[0, 0, 0, 1], [0, 0, 1, 1]], sz_nondiag, rtol=0.01)
