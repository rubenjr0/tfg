import depth_pro
import numpy as np
import torch
from scipy import ndimage

from .sensor import Sensor


class Pipeline:
    def __init__(self, sensor: Sensor):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = torch.float16 if torch.cuda.is_available() else torch.float32
        self.depth_pro, _ = depth_pro.create_model_and_transforms(
            device=self.device, precision=self.precision
        )
        self.depth_pro.eval()
        self.depth_pro.compile()
        self.sensor = sensor

    def process_depth(
        depth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process depth image to create a mask, edges, and laplacian.
        Args:
            depth (np.ndarray): Depth image, shape (H, W, 1).
        Returns:
            depth (np.ndarray): Processed depth image.
            depth_edges (np.ndarray): Edges of the depth image.
            depth_laplacian (np.ndarray): Laplacian of the depth image.
        """

        mask = depth >= np.percentile(depth, 99.9)
        mask = 1.0 - ndimage.binary_dilation(mask).astype(np.float32)
        depth_edges = ndimage.sobel(depth, axis=0) + ndimage.sobel(depth, axis=1)
        depth_edges = np.linalg.norm(depth_edges, axis=-1, keepdims=True)
        depth_laplacian = ndimage.laplace(depth)
        depth_laplacian = np.linalg.norm(depth_laplacian, axis=-1, keepdims=True)
        return depth, depth_edges, depth_laplacian, mask

    def refine(
        mu_1: torch.Tensor, var_1: torch.Tensor, mu_2: torch.Tensor, var_2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Refine the depth image using the uncertainty estimates.
        Args:
            mu_1 ( torch.Tensor): Mean of the first depth image.
            var_1 ( torch.Tensor): Variance of the first depth image.
            mu_2 ( torch.Tensor): Mean of the second depth image.
            var_2 ( torch.Tensor): Variance of the second depth image.
        Returns:
            mu_refined ( torch.Tensor): Refined mean of the depth image.
            var_refined ( torch.Tensor): Refined variance of the depth image.
        """

        mu_refined = (mu_1 * var_2 + mu_2 * var_1) / (var_1 + var_2)
        var_refined = 1.0 / (1.0 / var_1 + 1.0 / var_2)
        mu_diff = torch.abs(mu_1 - mu_2)
        conssitency = torch.exp(-(mu_diff**2) / (2 * (var_1 + var_2)))
        adjusted = conssitency * mu_refined + (1 - conssitency) * torch.minimum(
            mu_1, mu_2
        )
        return adjusted, var_refined

    def __call__(self, rgbd: np.ndarray):
        _image = rgbd[:, :, :3]
        depth = rgbd[:, :, 3]
        depth, depth_edges, depth_laplacian = Pipeline.process_depth(depth)


if __name__ == "__main__":
    import rerun as rr

    rr.init("depth-pipeline", spawn=True)

    depth = np.load(
        "data/val/outdoor/scene_00022/scan_00194/00022_00194_outdoor_000_000_depth.npy"
    )
    depth, depth_edges, depth_laplacian, mask = Pipeline.process_depth(depth)

    rr.log("depth", rr.DepthImage(depth))
    rr.log("depth_edges", rr.Tensor(depth_edges))
    rr.log("depth_laplacian", rr.Tensor(depth_laplacian))
    rr.log("mask", rr.Tensor(mask))
