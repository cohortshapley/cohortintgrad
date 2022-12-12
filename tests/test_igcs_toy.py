import math

import numpy as np
import pytest
import torch

import cohortintgrad as csig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def ref():
    x = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 2]])
    return x


@pytest.fixture(scope="module")
def ref_t():
    y = torch.Tensor([0, 1, 1, 2])
    return y


@pytest.fixture(scope="module")
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


def test_np_shape_cast(ref, ref_t):
    IG = csig.CohortIntGrad(np.array(ref), np.array(ref_t), ratio=0.1, n_step=500)
    igcs_single0 = IG.igcs_single(0)
    np.testing.assert_allclose(
        [-0.5, -0.5], igcs_single0.to("cpu").detach().numpy(), rtol=0.01
    )
    # works for numpy inputs but returns torch.Tensor


def test_np_partially_shape_cast(ref, ref_t):
    IG = csig.CohortIntGrad(ref, np.array(ref_t), ratio=0.1, n_step=500)
    igcs_single0 = IG.igcs_single(0)
    np.testing.assert_allclose(
        [-0.5, -0.5], igcs_single0.to("cpu").detach().numpy(), rtol=0.01
    )


def test_y_shape_cast(ref):
    ref_t = torch.Tensor([[0], [1], [1], [2]])
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    igcs_single0 = IG.igcs_single(0)
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


def test_sz_typecast(ref, ref_t):
    IG = csig.CohortIntGrad(
        ref.to(device), ref_t.half().to(device), ratio=0.1, n_step=500
    )
    z = torch.Tensor([[0, 0]]).to(device)
    sz_origin = IG._sz(t_id=3, z=z)
    assert sz_origin.dtype == torch.float16


def test_nu_typecast(ref, ref_t):
    IG = csig.CohortIntGrad(
        ref.to(device), ref_t.half().to(device), ratio=0.1, n_step=500
    )
    z = torch.Tensor([[0, 0]]).to(device)
    nu_origin = IG._nu(t_id=3, z=z)
    assert nu_origin.dtype == torch.float16


def test_nu_diagonal(ref, ref_t, diagonal):  # arbitrary z
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    nu_diagonal = (
        IG._nu(t_id=3, z=diagonal).to("cpu").detach().numpy()
    )  # weighted average of ref_t :
    weighted_average = (
        (
            torch.sum(IG._sz(t_id=3, z=diagonal) * ref_t.to("cuda"), axis=1)
            / torch.sum(IG._sz(t_id=3, z=diagonal), axis=1)
        )
        .to("cpu")
        .detach()
        .numpy()
    )
    np.testing.assert_allclose(weighted_average, nu_diagonal, rtol=0.01)


def test_nu_nondiagonal(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    nondiag_z = torch.Tensor([[0, 1], [1.0, 0.0]]).to(device)
    nu_nondiag = IG._nu(t_id=3, z=nondiag_z).to("cpu").detach().numpy()
    np.testing.assert_allclose([1 * 2 / 1, (1 + 2) / (1 + 1)], nu_nondiag, rtol=0.01)


def test_summand_diagonal(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    summand_diagonal = IG._summand(t_id=3).to("cpu").detach().numpy()

    def partial1_nu(z):  # analytic results
        return (z**2 - 5 * z + 4) / ((2 * z**2 - 5 * z + 4) ** 2)
        # denominator: ((1-z)^2 + (1-z)^2 + (1-z) + 1)^2
        # numerator: (2z^2-5z+4)(z-1)-2(z-1)((1-z)^2 + (1-z) + 2)

    def partial2_nu(z):
        return (4 - 3 * z) / ((2 * z**2 - 5 * z + 4) ** 2)

    analytic_result = np.vstack(
        [np.array([partial1_nu(i / 500), partial2_nu(i / 500)]) for i in range(500 + 1)]
    )
    np.testing.assert_allclose(analytic_result, summand_diagonal, rtol=0.01)


def test_remaining_delta(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    rd = IG._remaining_delta(t_id=3, ig=IG.igcs_single(t_id=3)).item()
    content = (
        torch.sum(IG.igcs_single(t_id=3)).item() - (ref_t[3] - torch.mean(ref_t)).item()
    )
    math.isclose(rd, content, rel_tol=0.01)


def test_remaining_delta_degenerated():
    ref_d = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 2], [1, 2]])
    ref_td = torch.Tensor([0, 1, 1, 2, 100])
    IG = csig.CohortIntGrad(ref_d.to(device), ref_td.to(device), ratio=0.1, n_step=500)
    rd = IG._remaining_delta(t_id=3, ig=IG.igcs_single(t_id=3)).item()
    content = (
        torch.sum(IG.igcs_single(t_id=3)).item()
        - ((ref_td[3] + ref_td[4]) / 2 - torch.mean(ref_td)).item()
        # when there are completely similar data to the target, the target value in the summation rule is modified to their average
    )
    np.testing.assert_allclose(rd, content, rtol=0.01)


def test_igcs_stack(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    ig, rd = IG.igcs_stack(list(range(4)))
    analytic_30 = 3 / 7 + 2 / (7 * np.sqrt(7)) * (
        np.arctan(1 / np.sqrt(7)) - np.arctan(5 / np.sqrt(7))
    )  # analytic result
    analytic_31 = 4 / 7 - 2 / (7 * np.sqrt(7)) * (
        np.arctan(1 / np.sqrt(7)) - np.arctan(5 / np.sqrt(7))
    )  # analytic result
    analytic_10 = 1 / 7 - 10 / (7 * np.sqrt(7)) * (
        np.pi / 2 - np.arctan(3 / np.sqrt(7))
    )
    np.testing.assert_allclose(
        [
            [-1 / 2, -1 / 2],
            [analytic_10, -analytic_10],
            [1 / 2, -1 / 2],
            [analytic_30, analytic_31],
        ],
        ig.to("cpu").detach().numpy(),
        rtol=0.01,
    )


def test_igcs_stack_rd(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    ig, rd = IG.igcs_stack(list(range(4)))
    np.testing.assert_allclose(
        np.zeros((1, 4)), rd.to("cpu").detach().numpy(), atol=0.01
    )


def test_predict_ks(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    predict_ks = IG._predict_ks(binary_vectors=[[0, 0], [0, 1], [1, 0], [1, 1]], t_id=3)
    average_00 = np.average([ref_t[0], ref_t[1], ref_t[2], ref_t[3]])
    average_10 = np.average([ref_t[2], ref_t[3]])
    average_01 = np.average([ref_t[3]])
    average_11 = np.average([ref_t[3]])
    np.testing.assert_allclose(
        [average_00, average_01, average_10, average_11], predict_ks, rtol=0.01
    )


def test_cs(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    cs = np.vstack([IG.cohort_kernel_shap(t_id=i) for i in range(4)])
    average_00 = np.average([ref_t[0], ref_t[1], ref_t[2], ref_t[3]])
    average10 = list()
    average01 = list()
    average11 = list()

    average10.append(np.average([ref_t[0], ref_t[1]]))
    average01.append(np.average([ref_t[0], ref_t[2]]))
    average11.append(np.average([ref_t[0]]))
    average10.append(np.average([ref_t[0], ref_t[1]]))
    average01.append(np.average([ref_t[1]]))
    average11.append(np.average([ref_t[1]]))
    average10.append(np.average([ref_t[2], ref_t[3]]))
    average01.append(np.average([ref_t[0], ref_t[2]]))
    average11.append(np.average([ref_t[2]]))
    average10.append(np.average([ref_t[2], ref_t[3]]))
    average01.append(np.average([ref_t[3]]))
    average11.append(np.average([ref_t[3]]))

    average10 = np.hstack(average10)  # array of average at z=(1,0)
    average01 = np.hstack(average01)  # array of average at z=(0,1)
    average11 = np.hstack(average11)  # array of average at z=(1,1)

    cs_explicit = np.vstack(
        [
            np.average(
                np.vstack([average10 - average_00, average11 - average01]), axis=0
            ),
            np.average(
                np.vstack([average01 - average_00, average11 - average10]), axis=0
            ),
        ]
    ).T  # explicit averaging

    np.testing.assert_allclose(cs_explicit, cs, rtol=0.01)


def test_perm_cs(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    cs = np.vstack(
        [IG.cohort_permutation_shap(t_id=i, max_evals=5) for i in range(4)]
    )  # least 2 * num_features + 1 = 5 > 2!
    average_00 = np.average([ref_t[0], ref_t[1], ref_t[2], ref_t[3]])
    average10 = list()
    average01 = list()
    average11 = list()

    average10.append(np.average([ref_t[0], ref_t[1]]))
    average01.append(np.average([ref_t[0], ref_t[2]]))
    average11.append(np.average([ref_t[0]]))
    average10.append(np.average([ref_t[0], ref_t[1]]))
    average01.append(np.average([ref_t[1]]))
    average11.append(np.average([ref_t[1]]))
    average10.append(np.average([ref_t[2], ref_t[3]]))
    average01.append(np.average([ref_t[0], ref_t[2]]))
    average11.append(np.average([ref_t[2]]))
    average10.append(np.average([ref_t[2], ref_t[3]]))
    average01.append(np.average([ref_t[3]]))
    average11.append(np.average([ref_t[3]]))

    average10 = np.hstack(average10)  # array of average at z=(1,0)
    average01 = np.hstack(average01)  # array of average at z=(0,1)
    average11 = np.hstack(average11)  # array of average at z=(1,1)

    cs_explicit = np.vstack(
        [
            np.average(
                np.vstack([average10 - average_00, average11 - average01]), axis=0
            ),
            np.average(
                np.vstack([average01 - average_00, average11 - average10]), axis=0
            ),
        ]
    ).T  # explicit averaging

    np.testing.assert_allclose(cs_explicit, cs, rtol=0.01)


def test_numsim(ref, ref_t):
    IG = csig.CohortIntGrad(ref.to(device), ref_t.to(device), ratio=0.1, n_step=500)
    numsim = np.vstack(
        [IG.num_sim(t_id=i).to("cpu").detach().numpy() for i in range(4)]
    )

    numsim10 = list()
    numsim01 = list()
    numsim10.append(len([ref_t[0], ref_t[1]]))
    numsim01.append(len([ref_t[0], ref_t[2]]))
    numsim10.append(len([ref_t[0], ref_t[1]]))
    numsim01.append(len([ref_t[1]]))
    numsim10.append(len([ref_t[2], ref_t[3]]))
    numsim01.append(len([ref_t[0], ref_t[2]]))
    numsim10.append(len([ref_t[2], ref_t[3]]))
    numsim01.append(len([ref_t[3]]))
    numsim10 = np.hstack(numsim10)  # array of number of similar data in first direction
    numsim01 = np.hstack(numsim01)

    numsim_explicit = np.vstack([numsim10, numsim01]).T
    np.testing.assert_array_equal(numsim_explicit, numsim)
