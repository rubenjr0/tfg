import depth_pro
import numpy as np
import torch
from scipy import ndimage

from uncertainty_estimation.model import UncertaintyEstimator
from uncertainty_estimation.utils import Sensor


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
    consistency = torch.exp(-mu_diff.pow(2) / (2 * (var_1 + var_2)))
    adjusted = consistency * mu_refined + (1 - consistency) * torch.minimum(mu_1, mu_2)
    return adjusted, var_refined


class Pipeline:
    def __init__(self, sensor: Sensor | None, checkpoint_path: str | None):
        self.sensor = sensor
        print("Creating pipeline...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print("Loading DepthPro...")
        self.depth_pro, _ = depth_pro.create_model_and_transforms(device=self.device)
        self.depth_pro.eval()
        self.depth_pro.compile()

        if checkpoint_path is not None:
            print("Loading Uncertainty model...")
            self.unc_model = UncertaintyEstimator.load_from_checkpoint(
                checkpoint_path=checkpoint_path
            )
            self.unc_model.to(self.device)
            self.unc_model.eval()
            # self.unc_model.compile()

    def estimate_depth(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.depth_pro.infer(x)["depth"]

    def __call__(self, rgbd: np.ndarray):
        assert self.sensor is not None
        image = rgbd[:, :, :3]
        depth = rgbd[:, :, 3][:, :, np.newaxis]

        print("Processing depth image...")
        depth_edges, depth_laplacian, _ = process_depth(depth)
        depth_unc = self.sensor.compute_uncertainty(depth, depth_edges)
        depth = torch.from_numpy(depth).permute(2, 0, 1).to(self.device)
        depth_edges = torch.from_numpy(depth_edges).permute(2, 0, 1).to(self.device)
        depth_laplacian = (
            torch.from_numpy(depth_laplacian).permute(2, 0, 1).to(self.device)
        )
        depth_unc = torch.from_numpy(depth_unc).permute(2, 0, 1).to(self.device)

        print("Estimating depth...")
        image = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        est = self.estimate_depth(image)
        est = est.unsqueeze(0)
        est_np = est.cpu().numpy().transpose(1, 2, 0)
        est_edges, est_laplacian, _ = process_depth(est_np)
        est_edges = torch.from_numpy(est_edges).permute(2, 0, 1).to(self.device)
        est_laplacian = torch.from_numpy(est_laplacian).permute(2, 0, 1).to(self.device)

        print("Estimating uncertainty...")
        x = torch.cat([est, est_edges, est_laplacian], dim=0).unsqueeze(0)
        est_log_var = self.unc_model(image, x)
        est_var = torch.clamp(est_log_var, -6, 6).exp()

        depth = depth.squeeze()
        depth_unc = depth_unc.squeeze()
        est = est.squeeze()
        est_var = est_var.squeeze()
        mu, var = refine(depth, depth_unc, est, est_var)
        print(f"mu: {mu.shape}, var: {var.shape}")
        return mu, var


if __name__ == "__main__":
    import rerun as rr

    from uncertainty_estimation.data import ImageDepthDataset

    sensor = Sensor(3, 35, (0.0008, 0.0016, 0.0018))
    pipeline = Pipeline(sensor, "checkpoints/epoch=749-step=3000-v1.ckpt")
    dataset = ImageDepthDataset(root="data/ai_001_001/images")
    sample = dataset[0]
    rgb = sample["image"]
    depth = sample["depth"]
    rgbd = torch.cat([rgb, depth], dim=0).permute(1, 2, 0).numpy()
    mu, var = pipeline(rgbd)
    mu = mu.detach().cpu().numpy()
    var = var.detach().cpu().numpy()

    rr.init("uncertainty-predictor", spawn=True)
    rr.log("rgbd/rgb", rr.Image(rgb.numpy().transpose(1, 2, 0)))
    rr.log("rgbd/depth", rr.DepthImage(depth))
    rr.log("refined/mu", rr.DepthImage(mu))
    rr.log("refined/var", rr.Tensor(var))
