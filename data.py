import numpy as np
import torch


class Exchangeable2DGaussian(torch.utils.data.Dataset):
    def __init__(self, dim, size, modes):
        super(Exchangeable2DGaussian, self).__init__()
        noise = np.random.normal(size=(size, dim, 2))
        self.data = modes + noise
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class TwoDPoints(torch.utils.data.Dataset):
    def __init__(self, name, size):
        super(TwoDPoints, self).__init__()

        self.dataset_name = name
        self.size = size

        rng = np.random.RandomState()
        if self.dataset_name == "8gaussians":
            scale = 4.
            centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                       (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                             1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
            centers = [(scale * x, scale * y) for x, y in centers]

            dataset = []
            for i in range(size):
                point = rng.randn(2) * 0.5
                idx = rng.randint(8)
                center = centers[idx]
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            dataset /= 1.414
            self.data = dataset

        elif self.dataset_name == "pinwheel":
            radial_std = 0.3
            tangential_std = 0.1
            num_classes = 5
            num_per_class = size // 5
            rate = 0.25
            rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

            features = rng.randn(num_classes * num_per_class, 2) \
                       * np.array([radial_std, tangential_std])
            features[:, 0] += 1.
            labels = np.repeat(np.arange(num_classes), num_per_class)

            angles = rads[labels] + rate * np.exp(features[:, 0])
            rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
            rotations = np.reshape(rotations.T, (-1, 2, 2))

            self.data = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

        elif self.dataset_name == "2spirals":
            n = np.sqrt(np.random.rand(size, 1)) * 540 * (2 * np.pi) / 360
            d1x = -np.cos(n) * n + np.random.rand(size, 1) * 0.5
            d1y = np.sin(n) * n + np.random.rand(size, 1) * 0.5
            sign = np.random.choice([-1.0,1.0], size = size)
            x = sign[:,None]*np.hstack((d1x, d1y)) / 3
            x += np.random.randn(*x.shape) * 0.1
            self.data = x

        elif self.dataset_name == "checkerboard":
            x1 = np.random.rand(size) * 4 - 2
            x2_ = np.random.rand(size) - np.random.randint(0, 2, size) * 2
            x2 = x2_ + (np.floor(x1) % 2)
            self.data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2

        elif self.dataset_name == "line":
            x = rng.rand(size) * 5 - 2.5
            y = x
            self.data = np.stack((x, y), 1)
        elif self.dataset_name == "cos":
            x = rng.rand(size) * 5 - 2.5
            y = np.sin(x) * 2.5
            self.data = np.stack((x, y), 1)
        else:
            raise NotImplementedError

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def get_loader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True
    )
