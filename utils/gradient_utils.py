import numpy as np
from sklearn.preprocessing import MinMaxScaler
class GradientUtils:
    """
    Base class for computing gradient-based tiling statistics, histograms,
    and peakiness scores.
    """

    def __init__(self, imgs: np.ndarray, tile_size, border_size=None, bin_edges=None, channel=None):
        self.imgs = imgs
        self.tile_size = np.array(tile_size) if isinstance(tile_size, (tuple, list, np.ndarray)) else tile_size
        self.border_size = border_size
        self._bin_edges = bin_edges

        # Remove borders
        self.imgs_wo_borders = self.border_free(self.imgs, self.border_size)

        # Compute gradients along each axis
        self.gradients = self.compute_gradients(self.imgs_wo_borders, self.border_size)
        
        
        # Gradients along tile grid
        self.grad_edges = self.get_gradients_at(position="edge", channels=channel)
        self.grad_middle = self.get_gradients_at(position="middle", channels=channel)

        self.mean = self.grad_middle.mean()
        self.std = self.grad_middle.std()
        # print(f"Mean: {self.mean}, STD : {self.std}")
        # Bin edges
        if self._bin_edges is None:
            self._bin_edges = self.get_bin_edges(list(self.gradients) + [self.grad_edges, self.grad_middle])


    # ---------------- STATIC METHODS ----------------
    @staticmethod
    def get_bin_edges(gradients: list, num_bins=200):
        flattened = np.concatenate([g.flatten() for g in gradients])
        _, bin_edges = np.histogram(flattened, bins=num_bins)
        return bin_edges

    @staticmethod
    def compute_histograms(gradients: np.ndarray, bin_edges: np.ndarray):
        return np.histogram(gradients, bins=bin_edges)[0]

    @staticmethod
    def wiener_entropy(hist: np.ndarray, eps=1e-12):
        w = np.hanning(len(hist))
        X = np.fft.rfft(hist * w)
        P = np.abs(X) ** 2 + eps
        geom_mean = np.exp(np.mean(np.log(P)))
        arith_mean = np.mean(P)
        return 1.0 - float(geom_mean / (arith_mean + eps))

    # ----------------- PUBLIC METHODS -----------------
    def make_bin_edges(self, n_bins=2000):
        return self.get_bin_edges(list(self.gradients) + [self.grad_edges, self.grad_middle], num_bins=n_bins)

    def get_peakiness_scores(histogram_edges, histogram_middle, eps=1e-12):
        scores = []
        for x in [histogram_edges, histogram_middle, histogram_middle - histogram_edges]:
            scores.append(GradientUtils.wiener_entropy(x, eps=eps))
        return scores

    def _normalize_gradients(self, gradients, mu=None, sigma=None):
        """
        Normalize gradients using z-score normalization.
        
        Args:
            gradients: Array to normalize
            mu: Mean to use for normalization. If None, uses self.mean
            sigma: Std to use for normalization. If None, uses self.std
        
        Returns:
            Normalized array
        """
        if mu is None:
            mu = self.mean
        if sigma is None:
            sigma = self.std
        
        return (gradients - mu) / (sigma + 1e-8)




# ----------------- 2D IMPLEMENTATION -----------------
class GradientUtils2D(GradientUtils):

    @staticmethod
    def border_free(imgs, border_size):
        if border_size == 0:
            return imgs
        else:
            return imgs[:, border_size:-border_size, border_size:-border_size, :]
    
    @staticmethod
    def compute_gradients(imgs, border_size=0):
        grad_x = imgs[:, :, 1:, :] - imgs[:, :, :-1, :]
        grad_y = imgs[:, 1:, :, :] - imgs[:, :-1, :, :]
        return grad_x, grad_y

    def _gradients_along_tile_grid(self, offset, channels=None):
        oy , ox = offset if isinstance(offset, (tuple, list, np.ndarray)) else (offset, offset)
        tile_sz_y, tile_sz_x = self.tile_size if isinstance(self.tile_size, (tuple, list, np.ndarray)) else (self.tile_size, self.tile_size)
        grad_x, grad_y = self.gradients
        if channels is None:
            grad_x_slice = grad_x[:, :, ox::tile_sz_x,:]
            grad_y_slice = grad_y[:, oy::tile_sz_y, :,:]
        elif isinstance(channels, int):
            grad_x_slice =  grad_x[:, :, ox::tile_sz_x, channels]
            grad_y_slice = grad_y[:, oy::tile_sz_y, :, channels]
        else:
            raise ValueError("channels must be None or int")
        return np.concatenate([grad_x_slice.flatten(), grad_y_slice.flatten()])

    def get_gradients_at(self, position="edge", channels=None):
        if isinstance(position, str):
            position = position.lower()
            if position == "edge":
                offset = self.tile_size - 1 
            elif position == "middle":
                offset = self.tile_size // 2 - 1
            else:
                raise ValueError("position must be 'edge' or 'middle'")
        elif isinstance(position, int):
            offset = position
        else:
            raise TypeError("position must be string or int")
        return self._gradients_along_tile_grid(offset, channels)


# ----------------- 3D IMPLEMENTATION -----------------
class GradientUtils3D(GradientUtils):

    @staticmethod
    def border_free(imgs, border_size):
        bz, by, bx = border_size
        z_slice = slice(bz, -bz if bz != 0 else None)
        y_slice = slice(by, -by if by != 0 else None)
        x_slice = slice(bx, -bx if bx != 0 else None)
        return imgs[:, z_slice, y_slice, x_slice, :]

    @staticmethod
    def compute_gradients(imgs, border_size):
        wb = GradientUtils3D.border_free(imgs, border_size)
        grad_z = wb[:, 1:, :, :, :] - wb[:, :-1, :, :, :]
        grad_y = wb[:, :, 1:, :, :] - wb[:, :, :-1, :, :]
        grad_x = wb[:, :, :, 1:, :] - wb[:, :, :, :-1, :]
        return grad_z, grad_y, grad_x

    def _gradients_along_tile_grid(self, offset, channels=None):
        oz, oy, ox = offset
        oz = 8 #!AMAN Hardcoded
        if channels is None:
            channels = list(range(self.gradients[2].shape[-1]))
        elif isinstance(channels, int):
            channels = [channels]

        grad_z, grad_y, grad_x = self.gradients
        grad_x_slice = grad_x[:, :, :, ox::self.tile_size[2], channels]
        grad_y_slice = grad_y[:, :, oy::self.tile_size[1], :, channels]
        grad_z_slice = grad_z[:, oz::self.tile_size[0], :, :, channels]
        return np.concatenate([grad_x_slice.flatten(), grad_y_slice.flatten(), grad_z_slice.flatten()])

    def get_gradients_at(self, position="edge", channels=None):
        if isinstance(position, str):
            position = position.lower()
            if position == "edge":
                offset = self.tile_size - 1 
            elif position == "middle":
                offset = self.tile_size // 2 - 1
            else:
                raise ValueError("position must be 'edge' or 'middle'")
        elif isinstance(position, int):
            offset = np.array([position] * 3)
        else:
            offset = np.array(position)
        return self._gradients_along_tile_grid(offset, channels)
