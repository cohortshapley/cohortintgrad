from functools import partial
from typing import Literal

import numpy as np
import torch

from ..igcs import CohortIntGrad
from .insertion_deletion import Insertion_Deletion_ABC_calc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Loaded_Feat_Attr(CohortIntGrad):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        feat_attr: np.ndarray,
        ratio: float = 0.1,
        # n_step: int = 500,
    ):
        """The wrapper class to evaluate XAI methods in observational way
        See in Section 5.1.2 of our paper (arXiv:2211.08414 [cs.LG]) in detail.

        Args:
            x (torch.Tensor): input data, 1st axis is data in cohort
            y (torch.Tensor): outcome of data
            ratio (float, optional): threshold of similarity. Defaults to 0.1.
            feat_attr (np.ndarray, optional): the result of feature attribution to be evaluated. same shape as x
        """
        assert (
            x.shape == feat_attr.shape
        ), f"invalid data shape: input x shape {x.shape} != feat attr shape {feat_attr.shape}"
        super().__init__(x=x, y=y, ratio=ratio, n_step=1)
        self.feat_attr = feat_attr

    def insertion_deletion_test(
        self,
        t_id: int,
        mode: Literal["insertion", "deletion"] = "insertion",
        # torch_cast: bool = False,
    ):
        """evaluation in observational ABC

        Args:
            t_id (int): target data ID
            mode (Literal[&quot;insertion&quot;, &quot;deletion&quot;], optional): test mode. Defaults to "insertion".
            torch_cast (bool, optional): whether cast the synthetic data to torch.Tensor to be evaluated in pred_function. Defaults to False.

        Returns:
           Tuple[np.ndarray, float]: plots of outcome, ABC
        """
        target = np.ones(self.x.shape[1])
        baseline = np.zeros(self.x.shape[1])
        pred = partial(self._predict_ks, t_id=t_id)
        self.id_test = Insertion_Deletion_ABC_calc(
            target=target,
            reference=baseline,
            feat_attr=self.feat_attr[t_id],
            pred_function=pred,
            mode=mode,
            torch_cast=False,
        )
        return self.id_test.calc_abc()
