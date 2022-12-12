from typing import Any, Literal, Tuple, Union

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class Insertion_Deletion_ABC_calc:
    def __init__(
        self,
        target: np.ndarray,  # TODO: input and cast torch.Tensor
        reference: np.ndarray,
        feat_attr: np.ndarray,  # TODO: reshape(1,-1) and return with sort‘
        pred_function: Any,
        mode: Literal["insertion", "deletion"] = "insertion",
        torch_cast: bool = False,
    ):
        """Calculator of ABC for insertion/deletion game
        The behavior is discussed in arXiv:2205.12423 [cs.LG] in detail

        Args:
            target (np.ndarray): target data to be deleted from
            reference (np.ndarray): reference data to be inserted to
            feat_attr (np.ndarray): feature attribution that sorts features
            pred_function (Any): model prediction function that cast synthesized np.ndarray to outcome
            mode (Literal[&quot;insertion&quot;, &quot;deletion&quot;], optional): test mode. Defaults to "insertion".
            torch_cast (bool): whether cast the synthetic data to torch.Tensor. Defaults to False.
        """
        self.feat_attr = feat_attr
        self.pred_method = pred_function
        assert mode in [
            "insertion",
            "deletion",
        ], f'mode must be "insertion" or "deletion", but input {mode}'
        self.mode = mode
        self.cast = torch_cast
        if self.mode == "deletion":
            self.start = target
            self.goal = reference
        elif self.mode == "insertion":
            self.start = reference
            self.goal = target

    def synthetic_data_generator(self) -> Union[np.ndarray, torch.Tensor]:  # len=d+1
        """Generated the datapoints evaluated in the insertion/deletion process

        Returns:
            Union[np.ndarray, torch.Tensor]: data points that appear in insertion/deletion
        """
        synthetics = list()
        sort = np.argsort(
            -self.feat_attr
        )  # sorted feat id from positive large to negative large

        synthetics.append(self.start)
        start = self.start.copy()
        goal = self.goal.copy()
        for feat in range(self.feat_attr.shape[0]):
            where = sort[feat]
            start[where] = goal[where]
            synthetics.append(start.copy())

        if self.cast:
            return torch.Tensor(np.array(synthetics))
        return np.array(synthetics)  # TODO: vstack?

    def calc_abc(self) -> Tuple[np.ndarray, float, float]:
        """execution of calculation

        Returns:
            Tuple[np.ndarray, float, float]: plots of outcome, AUC, ABC
        """
        points = self.pred_method(self.synthetic_data_generator())
        auc = (np.sum(points) - (points[0] + points[-1]) / 2) / (
            self.feat_attr.shape[0]
        )
        abc = auc - (points[-1] + points[0]) / 2
        if self.mode == "deletion":
            abc = -abc
        return points, abc

    def straight_points(self) -> np.ndarray:  # len=d+1
        """give the y-coordinates of the straight line that connects the start point and the end point

        Returns:
            np.ndarray: the d+1 y-coordinates of the straight line including two end points
        """
        end_points = self.pred_method(np.array([self.start, self.goal]))
        line_val = np.array(
            [
                end_points[0]
                + (end_points[1] - end_points[0]) * rank / self.feat_attr.shape[0]
                for rank in range(self.feat_attr.shape[0] + 1)
            ]
        )
        return line_val


def wrap_torch_model(x: np.ndarray, model: torch.nn.Module):
    """wrapper function of a model in pytorch to use in insertion/deletion

    Args:
        x (np.ndarray): placeholder input in insertion/deletion
        model (torch.nn.Module): pytorch model tested by insertin/deletion

    Returns:
        map from synthesized data to model output
    """
    return (
        model.predict(torch.Tensor(x).to(device)).to("cpu").detach().numpy()
    )  # TODO: forward or __call__
