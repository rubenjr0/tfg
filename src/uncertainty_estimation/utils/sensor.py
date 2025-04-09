import numpy as np
from scipy import ndimage


class Sensor:
    """
    Good defaults: Sensor(3, 35, (0.0008, 0.0016, 0.0018))
    """

    def __init__(
        self, min_range: float, max_range: float, coefs: tuple[float, float, float]
    ):
        self.min_range = min_range
        self.max_range = max_range
        self.base_uncertainty = coefs[0]
        self.linear_coeff = coefs[1]
        self.quadratic_coeff = coefs[2]

    def compute_uncertainty_batch(self, depth: np.ndarray) -> np.ndarray:
        """
        Compute the uncertainty of a batch of depth images.
        Args:
            depth (np.ndarray): The depth images to compute uncertainty for. (N, H, W, 1)
        """
        uncs = [self.compute_uncertainty(d) for d in depth]
        return np.stack(uncs, axis=0)

    def compute_uncertainty(self, depth: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Compute the uncertainty of a depth image based on some sensor parameters.

        Args:
            depth (np.ndarray): The depth image to compute uncertainty for. (H, W, 1)
            MAX_VAR (float): The maximum uncertainty value.
        """
        unc = (
            self.base_uncertainty
            + self.linear_coeff * depth
            + self.quadratic_coeff * depth**2
        )
        invalid_mask = (depth < self.min_range) | (depth > self.max_range)
        invalid_mask = ndimage.binary_dilation(invalid_mask)
        unc += edges * 0.05
        unc[invalid_mask] = depth.max() - depth.min()
        unc = ndimage.gaussian_filter(unc, sigma=2)
        return unc


if __name__ == "__main__":
    sensor = Sensor(3, 35, (0.0008, 0.0016, 0.0018))
    depth = np.random.rand(1, 256, 256).astype(np.float32)
    uncertainty = sensor.compute_uncertainty(depth)
    print(uncertainty.shape)
