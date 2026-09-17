import numpy as np


class MacenkoNormalizer:
    def __init__(self, Io: float = 240.0, alpha: float = 1.0, beta: float = 0.15):
        self.Io = Io
        self.alpha = alpha
        self.beta = beta

        self.HERef = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def fit(self, target_img_rgb: np.ndarray):
        HE, maxC = self._get_stain_matrix_and_max_c(target_img_rgb)
        if HE is not None:
            self.HERef = HE
            self.maxCRef = maxC

    def _get_stain_matrix_and_max_c(self, img_rgb: np.ndarray):
        img_rgb = img_rgb.astype(np.float64)
        OD = -np.log((img_rgb + 1.0) / self.Io)
        ODhat = OD.reshape(-1, 3)

        mask = (ODhat > self.beta).any(axis=1)
        ODhat = ODhat[mask]
        if ODhat.shape[0] < 100:
            return None, None

        _, V = np.linalg.eigh(np.cov(ODhat, rowvar=False))
        V = V[:, [2, 1]]

        That = np.dot(ODhat, V)
        phi = np.arctan2(That[:, 1], That[:, 0])

        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)

        vMin = np.dot(V, np.array([np.cos(min_phi), np.sin(min_phi)]))
        vMax = np.dot(V, np.array([np.cos(max_phi), np.sin(max_phi)]))

        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax]).T
        else:
            HE = np.array([vMax, vMin]).T

        Y = np.reshape(OD, (-1, 3)).T
        C, _, _, _ = np.linalg.lstsq(HE, Y, rcond=None)
        maxC = np.percentile(C, 99, axis=1)

        return HE, maxC

    def transform(self, img_rgb: np.ndarray) -> np.ndarray:
        orig_shape = img_rgb.shape
        img_rgb = img_rgb.astype(np.float64)
        OD = -np.log((img_rgb + 1.0) / self.Io)

        HE, maxC = self._get_stain_matrix_and_max_c(img_rgb)
        if HE is None or maxC is None:
            return img_rgb.astype(np.uint8)

        Y = np.reshape(OD, (-1, 3)).T
        C, _, _, _ = np.linalg.lstsq(HE, Y, rcond=None)

        maxC = np.maximum(maxC, 1e-6)
        C = C * (self.maxCRef[:, np.newaxis] / maxC[:, np.newaxis])

        normalized_OD = np.dot(self.HERef, C)
        normalized_img = self.Io * np.exp(-normalized_OD)
        normalized_img = normalized_img.T.reshape(orig_shape)
        normalized_img = np.clip(normalized_img, 0, 255).astype(np.uint8)

        return normalized_img
