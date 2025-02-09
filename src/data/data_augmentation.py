import numpy as np
import torch
import random
from scipy.ndimage import gaussian_filter1d

class DataAugmentation:
    """Apply augmentation techniques to handwriting stroke sequences."""

    def __init__(self, config):
        self.enable_augmentation = config.enable_augmentation

    def apply(self, data):
        """Applies random augmentation techniques."""
        if not self.enable_augmentation:
            return data

        # Randomly select augmentation methods
        if random.random() < 0.3:
            data = self.add_noise(data)
        if random.random() < 0.3:
            data = self.time_warp(data)
        if random.random() < 0.3:
            data = self.smoothing(data)

        return data

    def add_noise(self, data, noise_level=0.05):
        """Adds Gaussian noise to the sequence."""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    def time_warp(self, data, alpha=2):
        """Performs random time warping by stretching or compressing the sequence."""
        num_points = len(data)
        idx = np.arange(num_points)

        # Generate a smooth warping function
        warping = np.interp(
            idx, np.linspace(0, num_points - 1, num_points // alpha), idx[::alpha]
        )
        return np.interp(idx, warping, data)

    def smoothing(self, data, sigma=1):
        """Applies Gaussian smoothing."""
        return gaussian_filter1d(data, sigma=sigma, axis=0)
