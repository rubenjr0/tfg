import depth_pro
import numpy as np
import torch
from scipy import ndimage

from uncertainty_estimation.utils import Sensor


class Pipeline:
    def __init__(self, sensor: Sensor):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth_pro, _ = depth_pro.create_model_and_transforms(device=self.device)
        self.depth_pro.eval()
        self.depth_pro.compile()
        self.sensor = sensor

    def estimate_depth(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.depth_pro.infer(x)["depth"]

    def process_depth(
        depth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process depth image to create a mask, edges, and laplacian.
        Args:
            depth (np.ndarray): Depth image, shape (H, W, 1).
        Returns:
            depth_edges (np.ndarray): Edges of the depth image.
            depth_laplacian (np.ndarray): Laplacian of the depth image.
            mask (np.ndarray): Mask of the depth image.
        """

        mask = depth >= np.percentile(depth, 99.9)
        mask = 1.0 - ndimage.binary_dilation(mask).astype(np.float32)
        depth_edges = ndimage.sobel(depth, axis=0) + ndimage.sobel(depth, axis=1)
        depth_edges = np.linalg.norm(depth_edges, axis=-1, keepdims=True)
        depth_laplacian = ndimage.laplace(depth)
        depth_laplacian = np.linalg.norm(depth_laplacian, axis=-1, keepdims=True)
        return depth_edges, depth_laplacian, mask

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
    from uncertainty_estimation.data import ImageDepthDataset
    from pathlib import Path
    from tqdm import tqdm
    from torchvision.transforms import v2 as T

    t = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float, scale=True),
        ]
    )
    pipeline = Pipeline(None)
    dataset = ImageDepthDataset(root="data/geosynth")
    for entry in tqdm(dataset.data):
        image = entry.rgb.read()
        image = t(image)
        estimated = pipeline.estimate_depth(image)
        estimated = estimated.cpu().numpy()
        path = Path(entry.path)
        np.save(path / "est.npy", estimated)
