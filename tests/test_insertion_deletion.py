import math

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import cohortintgrad as csig

device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model_data_setup():
    data = fetch_california_housing()
    x = data["data"][:1000]
    y = data["target"][:1000]

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=1018
    )

    lr = LinearRegression()
    lr.fit(train_x, train_y)
    return (
        test_x,
        test_y,
        lr,
        lr.coef_,
    )  # arbitrary array (#elem=#feat) can be used as feat importance


@pytest.fixture(scope="module")
def target_data_id():
    return 0


class simple_model(torch.nn.Module):
    def __init__(self, input_dim: int = 3):
        super(simple_model, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, 1))
        self.layers = torch.nn.Sequential(*layers)

    def predict(self, embedded):
        out = self.layers(embedded)
        return out

    def forward(self, x):
        out = self.layers(x)
        return out


def test_wrap_torch_model(model_data_setup):
    model = simple_model(input_dim=8)
    wf = csig.insertion_deletion.wrap_torch_model(model_data_setup[0], model.to(device))
    assert type(wf) == np.ndarray


def test_mode_assertion(model_data_setup, target_data_id):
    x, y, model, coeff = model_data_setup
    with pytest.raises(AssertionError):
        _ = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
            target=x[target_data_id],
            reference=np.zeros(x.shape[1:]),
            feat_attr=coeff,
            pred_function=model.predict,
            mode="hoge",
            torch_cast=False,
        )


def test_syn_data_insertion(model_data_setup, target_data_id):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[target_data_id],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="insertion",
        torch_cast=False,
    )
    syn_data = np.zeros((x.shape[1] + 1, x.shape[1]))
    for i, j in enumerate(
        np.argsort(-coeff)
    ):  # in insertion, synthesized data are input to reference in ordered by feat attr (from large positive to large negative)
        syn_data[i + 1 :, j] += x[target_data_id, j]
    np.testing.assert_allclose(syn_data, id_calc.synthetic_data_generator(), rtol=0.01)


def test_syn_data_deletion(model_data_setup, target_data_id):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[target_data_id],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="deletion",
        torch_cast=False,
    )
    syn_data = np.repeat(
        x[target_data_id].reshape(1, x.shape[1]), x.shape[1] + 1, axis=0
    )
    for i, j in enumerate(
        np.argsort(-coeff)
    ):  # in deletion, synthesized data are deleted to input in ordered by feat attr (from large positive to large negative)
        syn_data[i + 1 :, j] -= x[target_data_id, j]
    np.testing.assert_allclose(syn_data, id_calc.synthetic_data_generator(), rtol=0.01)


def test_typecast_syn_data(model_data_setup, target_data_id):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[target_data_id],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="insertion",
        torch_cast=True,
    )
    assert type(id_calc.synthetic_data_generator()) == torch.Tensor


def test_straight_insertion(model_data_setup, target_data_id):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[target_data_id],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="insertion",
        torch_cast=False,
    )
    stc = (
        id_calc.straight_points()
    )  # y coordinate of the straight line from ref to target
    differentiate = [
        stc[i + 1] - stc[i] for i in range(x.shape[1])
    ]  # then the differences between neighbor are constant
    ones = (
        np.ones(x.shape[1])
        * (
            model.predict(x)[target_data_id]
            - model.predict(np.zeros(x.shape[1:]).reshape(1, -1))
        )
        / x.shape[1]
    )
    np.testing.assert_allclose(ones, differentiate, rtol=0.01)


def test_straight_insertion_ep(model_data_setup, target_data_id):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[target_data_id],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="insertion",
        torch_cast=False,
    )
    stc = (
        id_calc.straight_points()
    )  # y coordinate of the straight line from ref to target
    ep = [stc[0], stc[-1]]  # end points are y value of ref and target
    np.testing.assert_allclose(
        [
            model.predict(np.zeros(x.shape[1:]).reshape(1, -1)).item(),
            model.predict(x)[target_data_id],
        ],
        ep,
        rtol=0.01,
    )


def test_straight_deletion(model_data_setup, target_data_id):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[target_data_id],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="deletion",
        torch_cast=False,
    )
    stc = (
        id_calc.straight_points()
    )  # y coordinate of the straight line from ref to target
    differentiate = [
        stc[i + 1] - stc[i] for i in range(x.shape[1])
    ]  # then the differences between neighbor are constant
    ones = (
        np.ones(x.shape[1])
        * (
            model.predict(np.zeros(x.shape[1:]).reshape(1, -1))
            - model.predict(x)[target_data_id]
        )
        / x.shape[1]
    )
    np.testing.assert_allclose(ones, differentiate, rtol=0.01)


def test_straight_deletion_ep(model_data_setup, target_data_id):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[target_data_id],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="deletion",
        torch_cast=False,
    )
    stc = (
        id_calc.straight_points()
    )  # y coordinate of the straight line from target to ref
    ep = [stc[0], stc[-1]]  # end points are y value of target and ref
    np.testing.assert_allclose(
        [
            model.predict(x)[target_data_id],
            model.predict(np.zeros(x.shape[1:]).reshape(1, -1)).item(),
        ],
        ep,
        rtol=0.01,
    )


def test_insertion_pt_linear_regression(model_data_setup, target_data_id):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[target_data_id],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="insertion",
        torch_cast=False,
    )
    pred_syn_data = list()
    tmp_pred = model.predict(np.zeros(x.shape[1:]).reshape(1, -1)).item()
    pred_syn_data.append(tmp_pred)
    for j in np.argsort(-coeff):
        tmp_pred += coeff[j] * x[target_data_id, j]  # explicitly linear regression
        pred_syn_data.append(tmp_pred)
    pt, abc = id_calc.calc_abc()
    np.testing.assert_allclose(pred_syn_data, pt)


def test_deletion_pt_linear_regression(model_data_setup, target_data_id):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[target_data_id],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="deletion",
        torch_cast=False,
    )
    pred_syn_data = list()
    tmp_pred = model.predict(x)[target_data_id]
    pred_syn_data.append(tmp_pred)
    for j in np.argsort(-coeff):
        tmp_pred -= coeff[j] * x[target_data_id, j]  # explicitly linear regression
        pred_syn_data.append(tmp_pred)
    pt, abc = id_calc.calc_abc()
    np.testing.assert_allclose(pred_syn_data, pt)


def test_insertion_abc(model_data_setup, target_data_id):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[target_data_id],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="insertion",
        torch_cast=False,
    )
    stc = id_calc.straight_points()
    pt, abc = id_calc.calc_abc()
    trepoid_cum = (
        np.sum([((pt - stc)[i] + (pt - stc)[i + 1]) / 2 for i in range(x.shape[1])])
        / x.shape[1]
    )  # another derivation of ABC
    np.testing.assert_allclose(trepoid_cum, abc)


def test_deletion_abc(model_data_setup, target_data_id):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[target_data_id],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="deletion",
        torch_cast=False,
    )
    stc = id_calc.straight_points()
    pt, abc = id_calc.calc_abc()
    trepoid_cum = (
        np.sum([((stc - pt)[i] + (stc - pt)[i + 1]) / 2 for i in range(x.shape[1])])
        / x.shape[1]
    )  # deletion counts the area below the straight line in our convention
    np.testing.assert_allclose(trepoid_cum, abc)
