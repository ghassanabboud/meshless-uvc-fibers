import numpy as np
import SimpleITK as sitk
import logging
import open3d as o3d
import torch
import pickle as pkl
import torch
import torch.utils.dlpack
from .SparseVolume import SparseVolume
from typing_extensions import override

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
    
class SparseVectors(SparseVolume):

    def __init__(self, points, vectors, shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), device="cpu"):
        """
        Initialize the SparseVectors from a sparse 3D numpy array.

        Parameters
        ----------
        points: np.ndarray or torch.Tensor
            2D numpy array / torch Tensor with shape (N, 3) containing the coordinates of nonzero voxels. coordinates must be in (z, y, x) order.
        vectors: np.ndarray or torch.Tensor
            2D numpy array / torch Tensor with shape (N, 3) containing the vector components at the sparse coordinates. vector components must be in (z, y, x) order.
        shape: torch.Size 
            Shape of the full 3D volume. (z, y, x).
        spacing: tuple, optional
            Spacing between voxels in each dimension. (x, y, z).
        origin: tuple
            Coordinates of the origin. (x, y, z).
        device: str
            Device string for torch, e.g., "cuda:#" or "cpu". Default is "cpu".
        """

        super().__init__(points, shape, spacing=spacing, origin=origin, device=device)

        if isinstance(vectors, torch.Tensor):
            if vectors.ndim != 2 or vectors.shape[0] != len(points) :
                raise ValueError("Input torch tensor must have 2 dimensions (N,3) and have the same length as points.")
            vectors = vectors.to(dtype=torch.float32, device=self.device)
        elif isinstance(vectors, np.ndarray):
            if vectors.ndim != 2 or vectors.shape[0] != len(points):
                raise ValueError("Input numpy array must have 2 dimensions (N,3) and have the same length as points.")
            vectors = torch.tensor(vectors, dtype=torch.float32, device=self.device)
        
        self.vectors = vectors

        #if isinstance(arr, torch.Tensor):
        #    if arr.shape[-1] != 3 or arr.ndim != 4:
        #        raise ValueError("Input torch tensor must have shape (z, y, x, 3).")
        #    arr = arr.to(dtype=torch.float32, device=device)
        #elif isinstance(arr, np.ndarray):
        #    if arr.shape[-1] != 3 or arr.ndim != 4:
        #        raise ValueError("Input numpy array must have shape (z, y, x, 3).")
        #    arr = torch.tensor(arr, dtype=torch.float32, device=device)
        
        #nonzero_vector_mask = torch.any(arr != 0, dim=3)
        #super().__init__(nonzero_vector_mask, spacing=spacing, origin=origin, device=device)
        #self.vectors = arr[tuple(self.points.T)]
    
    @classmethod
    @override
    def from_array(cls, arr, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), device="cpu"):
        """
        Initialize a SparseField from a sparse 3D numpy array.

        Parameters
        ----------
        arr: np.ndarray or torch.Tensor
            4D numpy array / torch Tensor with shape (z,y,x, 4) containing vectors at each point.
        spacing: tuple, optional
            Spacing between voxels in each dimension. (x, y, z). Default is (1.0, 1.0, 1.0).
        origin: tuple, optional
            Coordinates of the origin. (x, y, z). Default is (0.0, 0.0, 0.0).
        device: str, optional
            Device string for torch, e.g., "cuda:#" or "cpu". Default is "cpu".

        Returns
        -------
        SparseVolume
            An instance of SparseVolume initialized from the input array.
        """

        device = cls._resolve_device(device)
        if isinstance(arr, torch.Tensor):
            if arr.ndim != 4 or arr.shape[-1] != 3:
                raise ValueError("Input torch tensor must have 4 dimensions (z,y,x,3).")
            arr = arr.to(dtype=torch.float32, device=device)
        elif isinstance(arr, np.ndarray):
            if arr.ndim != 4 or arr.shape[-1] != 3:
                raise ValueError("Input numpy array must have 4 dimensions (z,y,x,3).")
            arr = torch.tensor(arr, dtype=torch.float32, device=device)
        
        points = torch.argwhere(torch.any(arr != 0, dim=-1))
        vectors = arr[tuple(points.T)]
        return cls(points, vectors, shape=arr.shape, spacing=spacing, origin=origin, device=device)

    @classmethod
    def from_Sparse_with_vectors(cls, other, vectors):
        """
        Create a new SparseVectors by copying all attributes from another SparseVolume,
        except for the vectors, which are provided as an argument.

        Parameters
        ----------
        other: SparseVolume
            The source SparseVolume to copy attributes and points from.
        vectors: np.ndarray
            3D numpy array with shape (N, 3) containing vector components at sparse coordinates. vector components must be in (z, y, x) order.


        Returns
        -------
        SparseVectors
            A new SparseVectors instance with the specified vectors.
        """
        return cls(points=other.points, vectors=vectors, shape=other.shape, spacing=other.spacing, origin=other.origin, device=other.device)
    
    @classmethod
    @override
    def combine(cls, sparse_vectors):
        """
        Combine multiple SparseVectors instances into a single SparseVectors instance.

        Parameters
        ----------
        sparse_vectors: list of SparseVectors
            List of SparseVectors instances to combine.
    
        Returns
        -------
        SparseVectors
            A new SparseVectors instance containing the combined points and vectors from all input instances.
        """
        if not sparse_vectors:
            raise ValueError("Input list of SparseVolume is empty.")
        origin = sparse_vectors[0].origin
        spacing = sparse_vectors[0].spacing
        shape = sparse_vectors[0].shape
        device = sparse_vectors[0].device

        for v in sparse_vectors:
            if v.origin != origin:
                raise ValueError("All SparseVolumes must have the same origin.")
            if v.spacing != spacing:
                raise ValueError("All SparseVolumes must have the same spacing.")
            if v.shape != shape:
                raise ValueError("All SparseVolumes must have the same shape.")
            if v.device != device:
                raise ValueError("All SparseVolumes must be on the same device.")
            
        all_points = torch.cat([v.points for v in sparse_vectors], dim=0)
        all_vectors = torch.cat([v.vectors for v in sparse_vectors], dim=0)
        return cls(all_points, all_vectors, shape=shape, spacing=spacing, origin=origin, device=device)

    @override
    def reconstruct(self, numpy=True):
        """
        Reconstruct the full 3D numpy array from the sparse representation, including vector components.

        Parameters
        ----------
        numpy: bool
            If True, returns a numpy array. If False, returns a torch tensor.

        Returns
        -------
        np.ndarray
            3D numpy array with nonzero elements at stored coordinates, filled with vector components. Matrix shape is (z, y, x, 3).
        """
        arr = torch.zeros(self.shape, dtype=self.vectors.dtype)
        if len(self.points) > 0:
            arr[tuple(self.points.T)] = self.vectors
        if numpy:
            arr = arr.cpu().numpy()
        return arr
    
    @override
    def extract_points_based_on_mask(self, mask):
        """
        Create a new SparseVectors consisting only of points that match the given boolean mask.

        Parameters
        ----------
        mask: torch.Tensor
            A boolean tensor of shape (N,) indicating which points to keep. Must be the same length as self.points.

        Returns
        -------
        SparseVectors
            New SparseVectors containing only the points that match the mask.
        """
        if len(mask) != len(self.points):
            raise ValueError("Mask must have the same length as SparseVectors's points.")
        
        points = self.points[mask]
        vectors = self.vectors[mask]

        new = SparseVectors(points, vectors, self.shape, self.spacing, self.origin, device=self.device)
        return new
    
    def rotate(self, axis, angle):
        """
        Rotate the vectors around a specified axis by a given angle in the direction of the LEFT hand rule.

        Parameters
        ----------
        axis: SparseVectors
            Axis of rotation as a SparseVectors instance. MUST BE UNIT VECTORS. Must have the same number of points as self and be on the same device. Must be in (z, y, x) order.
        angle: SparseField
            Angles of rotation in degrees as a SparseField instance. must be on the same device.

        Returns
        -------
        SparseVectors
            A new SparseVectors instance with the rotated vectors.
        """

        if len(axis.points) != len(self.points) or len(angle.values) != len(self.points):
            raise ValueError("Axis, angle, and self must have the same number of points for rotation.")
        if axis.device != self.device or angle.device != self.device:
            raise ValueError("Axis and angle must be on the same device as SparseVectors.")
        
        # WE TAKE THE NEGATIVE ANGLE BECAUSE THE RODRIGUES FORMULA IS FOR RIGHT-HAND RULE, WE WANT TO ROTATE IN THE OPPOSITE DIRECTION
        angle_values = -angle.values

        cos_angle = torch.cos(torch.deg2rad(angle_values))
        sin_angle = torch.sin(torch.deg2rad(angle_values))

        # I am saving my vectors as (z, y, x) , so if i want to use Rodrigues formula, I bring them back to (x, y, z) order to make sure they respect the right-hand rule
        vector_values= self.vectors.flip(1)
        axis_values = axis.vectors.flip(1)

        dot_product = torch.sum(vector_values * axis_values, dim=1)
        cross_product = torch.cross(axis_values, vector_values, dim=1)
        rotated = vector_values * cos_angle[:, None] + \
                  cross_product * sin_angle[:, None] + \
                  axis_values * dot_product[:, None] * (1 - cos_angle[:, None])

        # Normalize the rotated vectors
        norms = torch.linalg.norm(rotated, axis=-1)
        norms[norms == 0] = 1.0
        rotated /= norms[:, None]
        # after i'm done, I flip the results back to (z, y, x) order to store.
        rotated = rotated.flip(1)

        return SparseVectors.from_Sparse_with_vectors(self, rotated)
    
    def interpolate_missing_vectors(self):
        """
        Find fibers with norms 0 and replace them with the average of their 6 closest neighbors.

        Returns
        -------
        SparseVectors
            A new SparseVectors instance with missing vectors interpolated.
        """

        valid_indices = torch.where(torch.linalg.norm(self.vectors, dim=1) > 1e-3)[0]
        missing_indices = torch.where(torch.linalg.norm(self.vectors, dim=1) <= 1e-3)[0]

        if len(missing_indices) == 0:
            logger.debug("No vectors with norm 0 found. No interpolation needed. Returning original SparseVectors.")
            return self
        
        logger.debug(f"number of vectors with norm close to 0: {len(missing_indices)}. Percentage: {len(missing_indices) / len(self.vectors) * 100:.2f}%")
        
        valid_vectors = self.vectors[valid_indices]
        valid_points = self.points[valid_indices]
        missing_points = self.points[missing_indices]

        valid_sparse = SparseVectors(valid_points, valid_vectors, shape=self.shape, spacing=self.spacing, origin=self.origin, device=self.device)
        valid_sparse._init_nns()

        target_tensor = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(missing_points.to(torch.float32).contiguous()))
        indices, _ = valid_sparse.nns.knn_search(target_tensor, knn=6)

        indices = torch.utils.dlpack.from_dlpack(indices.to_dlpack())
        neighbouring_vectors = valid_vectors[indices]
        average_vectors = torch.mean(neighbouring_vectors, dim=1)
        average_vectors = average_vectors / torch.linalg.norm(average_vectors, dim=1, keepdim=True)
        self.vectors[missing_indices] = average_vectors

        logger.debug(f"Missing vectors were replaced with the average of their 6 closest points.")

    def reorder(self, sparse_template):
        """
        Reorder the points and values of this SparseField to match the order of points in another SparseVolume.

        Parameters
        ----------
        sparse_template: SparseVolume
            The SparseVolume whose points will be used as the new order.

        Returns
        -------
        SparseField
            A new SparseField with points and values reordered to match sparse_template.
        """
        if sparse_template.device != self.device:
            raise ValueError("sparse_template must be on the same device as self.")
        if sparse_template.n_points != self.n_points:
            raise ValueError("sparse_template must have the same number of points as self.")
        
        members, indices = self.check_point_membership(sparse_template.points, return_indices=True)
        if not torch.all(members):
            raise ValueError("Not all points in self are present in sparse_template.")
        reordered_vectors = self.vectors[indices]
        return SparseVectors.from_Sparse_with_vectors(sparse_template, reordered_vectors)

    @override
    def __str__(self):
        return (f"SparseVectors(spacing={self.spacing}, origin={self.origin}, "
                f"num_points={len(self.points)}, device={self.device}, "
                f"shape={self.shape}, "
                f"norms_mean={torch.mean(torch.linalg.norm(self.vectors, dim=-1)):.2f})")

    @override
    def to_pickle(self, filename):
        """
        Save the SparseVectors to a pickle file as a dictionary. Can then be loaded with `SparseVectorsd.from_pickle(filename)`.

        Parameters
        ----------
        filename: str
            Path to the output pickle file.
        """
        dict = {"origin": self.origin,
                "spacing": self.spacing,
                "shape": self.shape,
                "points": self.points.cpu().numpy(),
                "vectors": self.vectors.cpu().numpy()}
        with open(filename, 'wb') as f:
            pkl.dump(dict, f)
        logger.info(f"SparseVectors saved to {filename}")

    @classmethod
    @override
    def from_pickle(cls, filename, device="cpu"):
        """
        Load a SparseVectors from a pickle file, pickle file must have been created with `SparseVectors.to_pickle(filename)`
        and is saved as a dictionary with keys: "origin", "spacing", "shape", "points", "vectors".

        Parameters
        ----------
        filename: str
            Path to the input pickle file.
        device: str
            Device string for torch, e.g., "cuda:#" or "cpu". Default is "cpu".

        Returns
        -------
        SparseVectors
            An instance of SparseVectors loaded from the pickle file.
        """
        with open(filename, 'rb') as f:
            class_dict = pkl.load(f)

        if not all(key in class_dict for key in ["origin", "spacing", "shape", "points", "vectors"]):
            raise ValueError("Pickle file must contain 'origin', 'spacing', 'shape', 'points', and 'vectors' keys.")
        
        return cls(class_dict["points"], class_dict["vectors"], class_dict["shape"], class_dict["spacing"], class_dict["origin"], device=device)

    def to_mha(self, filename):
        raise NotImplementedError