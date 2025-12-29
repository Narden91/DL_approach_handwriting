from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict
import torch


class BalancedBatchSampler(Sampler):
    """Sampler that ensures each batch contains both positive and negative samples."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)

        # Get indices for each class (ultra-fast with new dataset structure)
        self.label_to_indices = defaultdict(list)

        # Optimized: Use pre-computed arrays from FastHandwritingDataset
        if hasattr(dataset, 'labels') and hasattr(dataset, 'windows'):
            # Ultra-fast path: directly access labels array and window start indices
            for idx in range(len(dataset.windows)):
                start_idx = dataset.windows[idx, 3]  # start_index is 4th column
                label = int(dataset.labels[start_idx])
                self.label_to_indices[label].append(idx)
        elif hasattr(dataset, 'labels_array') and hasattr(dataset, 'windows'):
            # Fast path: use numpy array and window indices (legacy compatibility)
            for idx, (_, _, _, window_indices) in enumerate(dataset.windows):
                if isinstance(window_indices, np.ndarray):
                    label = int(dataset.labels_array[window_indices[0]])
                else:
                    label = int(dataset.labels_array[window_indices])
                self.label_to_indices[label].append(idx)
        else:
            # Fallback: iterate through dataset (slow but compatible)
            for idx in range(len(dataset)):
                _, label, _, _ = dataset[idx]
                self.label_to_indices[label.item()].append(idx)

        # Store number of samples per class
        self.count_per_class = {
            label: len(indices)
            for label, indices in self.label_to_indices.items()
        }

        # Calculate number of batches
        self.num_batches = len(self.dataset) // self.batch_size

    def __iter__(self):
        temperature = 0.5
        probs = np.array([len(indices) for indices in self.label_to_indices.values()])
        probs = probs ** temperature
        probs = probs / probs.sum()

        # Create a copy of indices for each class
        indices_by_class = {
            label: indices.copy()
            for label, indices in self.label_to_indices.items()
        }

        # Calculate samples per class per batch to maintain distribution
        total_samples = sum(self.count_per_class.values())
        samples_per_class = {
            label: max(1, int(self.batch_size * prob))
            for label, prob in zip(self.label_to_indices.keys(), probs)
        }

        # Generate batches
        batches = []
        for _ in range(self.num_batches):
            batch = []

            # Add minimum samples from each class
            for label, n_samples in samples_per_class.items():
                if len(indices_by_class[label]) < n_samples:
                    # If we run out of samples, reshuffle the indices
                    indices_by_class[label] = self.label_to_indices[label].copy()
                    np.random.shuffle(indices_by_class[label])

                # Take required samples
                batch.extend(indices_by_class[label][:n_samples])
                indices_by_class[label] = indices_by_class[label][n_samples:]

            # Fill remaining slots randomly
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                # Pool remaining indices from all classes
                all_remaining = []
                for indices in indices_by_class.values():
                    all_remaining.extend(indices)

                if len(all_remaining) < remaining:
                    # If we don't have enough remaining samples, reshuffle all indices
                    all_indices = []
                    for label_indices in self.label_to_indices.values():
                        all_indices.extend(label_indices)
                    np.random.shuffle(all_indices)
                    all_remaining = all_indices

                # Add remaining samples to batch
                np.random.shuffle(all_remaining)
                batch.extend(all_remaining[:remaining])

            # Shuffle the batch
            np.random.shuffle(batch)
            batches.append(batch)

        # Flatten and return all batches
        all_indices = []
        for batch in batches:
            all_indices.append(torch.tensor(batch))

        return iter(all_indices)

    def __len__(self):
        return self.num_batches
