from functools import partial
from typing import Union

import numpy as np
import shap
import torch
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


class CohortIntGrad:
    def __init__(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],  # TODO: dataloader as input
        ratio: float = 0.1,
        n_step: int = 500,
    ):
        """the instance of calculator of Cohort Shapley Integrated Gradients

        Args:
            x (torch.Tensor): input data, 1st axis is data in cohort
            y (torch.Tensor): outcome of data
            ratio (float, optional): threshold of similarity. Defaults to 0.1.
            n_step (int, optional): number of steps of path integral. Defaults to 500.
        """
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        self.x_shape = x.shape[1:]
        self.x = x.reshape((x.shape[0], -1))
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.y = y
        self.ratio = ratio
        self.n_step = n_step

    def _sjw(
        self, zj: torch.Tensor, xij: torch.Tensor, xtj: torch.Tensor
    ) -> torch.Tensor:
        sim = self._similarity(xij, xtj).to(device)
        y = torch.mul(zj.reshape(-1, 1), (sim - 1).reshape(1, -1)) + 1
        return y

    # TODO: other similarity measure in CS
    def _similarity(self, xij: torch.Tensor, xtj: torch.Tensor) -> torch.Tensor:
        ranges = xij.max() - xij.min()
        xtjs = xtj.repeat(xij.shape)
        return (abs(xtjs - xij) <= ranges * self.ratio).int()

    def _sz(self, t_id: int, z: torch.Tensor) -> torch.Tensor:
        m = [
            self._sjw(z[:, i], self.x[:, i], self.x[t_id, i])
            .unsqueeze(0)
            .to(self.y.dtype)  #
            for i in range(self.x.shape[1])
        ]
        return torch.cumprod(torch.vstack(m), dim=0)[
            -1
        ]  # TODO: "cumprod_out_cpu" not implemented for 'Half'

    def _nu(self, t_id: int, z: torch.Tensor) -> torch.Tensor:
        sz_ = self._sz(t_id, z).to(self.y.dtype)
        denom = torch.sum(sz_, dim=1).to(self.y.dtype)
        num_elem = torch.matmul(sz_, self.y)[:, 0]  # TODO: torch.autocast?
        return num_elem / denom

    def igcs_single(self, t_id: int) -> torch.Tensor:
        """CSIG for single data

        Args:
            t_id (int): data id of target data

        Returns:
            torch.Tensor: CSIG value of target data
        """
        summand = self._summand(t_id)
        ig = torch.mean(summand, dim=0)
        ig = ig.reshape(self.x_shape)
        return ig

    # TODO: scaler.scale(summand).backward()?
    def _summand(self, t_id: int):
        z = (
            torch.vstack(
                [
                    torch.arange(0, 1 + 1.0 / self.n_step, 1.0 / self.n_step)
                    for _ in range(self.x.shape[1])
                ]
            )
            .T.to(device)
            .requires_grad_()
        )
        summand = self._nu(t_id, z)
        summand.backward(
            gradient=torch.ones(self.n_step + 1).to(device),
            retain_graph=False,
            inputs=z,
        )
        return z.grad

    def _remaining_delta(self, t_id: int, ig: torch.Tensor) -> torch.Tensor:
        cache = self._sz(t_id=t_id, z=torch.ones(1, self.x.shape[1]).to(device))
        residual = torch.matmul(cache, self.y) / torch.sum(cache)
        rd = torch.sum(ig) - (residual - torch.mean(self.y))

        return rd

    def igcs_stack(self, stack_target: list) -> tuple[torch.Tensor, torch.Tensor]:
        """CSIG for multiple data

        Args:
            stack_target (list): list of data in for target data

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: CSIG for target data, residue of Riemann sum
        """
        stack = [
            self.igcs_single(t_id) for t_id in tqdm.tqdm(stack_target)
        ]  # TODO: avoid tqdm
        ig_all = torch.vstack(
            [ig.unsqueeze(0) for ig in stack]
        )  # TODO: check difference of api in CPU/cuda
        rd_all = torch.hstack(
            [self._remaining_delta(t_id, ig) for (t_id, ig) in zip(stack_target, stack)]
        )
        return ig_all, rd_all

    def _predict_ks(
        self,
        binary_vectors: np.ndarray,
        t_id: int,
    ) -> np.ndarray:
        z = torch.Tensor(binary_vectors).to(device)  # .int()?
        nu = self._nu(t_id, z)
        return nu.to("cpu").detach().numpy()

    def cohort_kernel_shap(self, t_id: int):
        pred = partial(self._predict_ks, t_id=t_id)
        explainer = shap.KernelExplainer(pred, data=np.zeros((1, self.x.shape[1])))
        sv = explainer.shap_values(
            np.ones((1, self.x.shape[1])),
            nsamples=2 ** self.x.shape[1],
            l1_reg=f"num_features({self.x.shape[1]})",
            silent=True,
        )
        return sv[0]

    def cohort_permutation_shap(self, t_id: int, max_evals: int = 10000):
        pred = partial(self._predict_ks, t_id=t_id)
        explainer = shap.explainers.Permutation(pred, np.zeros((1, self.x.shape[1])))
        sv = explainer(
            np.ones((1, self.x.shape[1])),
            max_evals=max_evals,
            # batch_size=bs,
            silent=True,
        )
        return sv.values[0]

    def num_sim(self, t_id: int) -> torch.Tensor:
        """Rarity for each feature of the target data

        Args:
            t_id (int): target data id

        Returns:
            torch.Tensor: number of similar data in cohort for each feature
        """
        z = torch.eye(self.x.shape[1]).to(device)
        sz = self._sz(t_id, z)
        return torch.sum(sz, axis=1).int().reshape(self.x_shape)
