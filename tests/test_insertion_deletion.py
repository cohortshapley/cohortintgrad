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
    return test_x, test_y, lr, lr.coef_


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


def test_mode_assertion(model_data_setup):
    x, y, model, coeff = model_data_setup
    with pytest.raises(AssertionError):
        _ = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
            target=x[0],
            reference=np.zeros(x.shape[1:]),
            feat_attr=coeff,
            pred_function=model.predict,
            mode="hoge",
            torch_cast=False,
        )


def test_syn_data_insertion(model_data_setup):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[0],
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
        syn_data[i + 1 :, j] += x[0, j]
    np.testing.assert_allclose(syn_data, id_calc.synthetic_data_generator(), rtol=0.01)


def test_syn_data_deletion(model_data_setup):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[0],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="deletion",
        torch_cast=False,
    )
    syn_data = np.repeat(x[0].reshape(1, x.shape[1]), x.shape[1] + 1, axis=0)
    for i, j in enumerate(
        np.argsort(-coeff)
    ):  # in deletion, synthesized data are deleted to input in ordered by feat attr (from large positive to large negative)
        syn_data[i + 1 :, j] -= x[0, j]
    np.testing.assert_allclose(syn_data, id_calc.synthetic_data_generator(), rtol=0.01)


def test_typecast_syn_data(model_data_setup):
    x, y, model, coeff = model_data_setup
    id_calc = csig.insertion_deletion.Insertion_Deletion_ABC_calc(
        target=x[0],
        reference=np.zeros(x.shape[1:]),
        feat_attr=coeff,
        pred_function=model.predict,
        mode="insertion",
        torch_cast=True,
    )
    assert type(id_calc.synthetic_data_generator()) == torch.Tensor
