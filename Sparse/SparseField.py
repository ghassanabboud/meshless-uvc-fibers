import numpy as np
import SimpleITK as sitk
import logging
import open3d as o3d
import scipy.sparse.linalg
import torch
import pickle as pkl
import torch
import torch.utils.dlpack
from .SparseVolume import SparseVolume
from .SparseVectors import SparseVectors
from typing_extensions import override
import scipy
import scipy.sparse
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

  
class SparseField(SparseVolume):
    def __init__(self, points, values, shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), device="cpu"):
        """
        Initialize the SparseField from a sparse 3D numpy array.

        Parameters
        ----------
        points: np.ndarray or torch.Tensor
            2D numpy array / torch Tensor with shape (N, 3) containing the coordinates of the sparse points.
        values: np.ndarray or torch.Tensor
            1D numpy array / torch Tensor with shape (N,) containing scalar values at the sparse coordinates.
        shape: tuple
            Shape of the full 3D volume. (z, y, x).
        spacing: tuple, optional
            Spacing between voxels in each dimension. (x, y, z). Default is (1.0, 1.0, 1.0).
        origin: tuple, optional
            Coordinates of the origin. (x, y, z). Default is (0.0, 0.0, 0.0).
        device: str
            Device string for torch, e.g., "cuda:#" or "cpu". Default is "cpu".
        """


        super().__init__(points, shape, spacing=spacing, origin=origin, device=device)

        if isinstance(values, torch.Tensor):
            if values.ndim != 1 or values.shape[0] != len(self.points):
                raise ValueError("Input torch tensor must be 1D and the same length as points.")
            values = values.to(dtype=torch.float32, device=self.device)
        elif isinstance(values, np.ndarray):
            if values.ndim != 1 or values.shape[0] != len(self.points):
                raise ValueError("Input numpy array must be 1D and the same length as points.")
            values = torch.tensor(values, dtype=torch.float32, device=self.device)

        self.values = values

    @classmethod
    @override
    def from_array(cls, arr, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), device="cpu"):
        """
        Initialize a SparseField from a sparse 3D numpy array.

        Parameters
        ----------
        arr: np.ndarray or torch.Tensor
            3D numpy array / torch Tensor with shape (z,y,x) containing nonzero voxels.
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
            if arr.ndim != 3:
                raise ValueError("Input torch tensor must have 3 dimensions (z,y,x).")
            arr = arr.to(dtype=torch.float32, device=device)
        elif isinstance(arr, np.ndarray):
            if arr.ndim != 3:
                raise ValueError("Input numpy array must have 3 dimensions (z,y,x).")
            arr = torch.tensor(arr, dtype=torch.float32, device=device)
        
        points = torch.argwhere(arr > 0)
        values = arr[tuple(points.T)]
        return cls(points, values, shape=arr.shape, spacing=spacing, origin=origin, device=device)

    @classmethod
    def from_Sparse_with_values(cls, other, values):
        """
        Create a new SparseField by copying all attributes from another SparseVolume or SparseField,
        except for the values, which are provided as an argument.
        
        Parameters
        ----------
        other: SparseVolume or SparseField
            The source SparseVolume or SparseField to copy attributes and points from
        values: np.ndarray / torch.Tensor
            1D numpy array of scalar values corresponding to the points in the new SparseField.

        Returns
        -------
        SparseField
            A new SparseField instance with the specified values.
        """
        return cls(points=other.points, values=values, shape=other.shape, spacing=other.spacing, origin=other.origin, device=other.device)

    @classmethod
    @override
    def combine(cls, sparse_fields):
        """
        Combine multiple SparseFields into a single SparseField by concatenating their points and values.

        Parameters
        ----------
        sparse_fields: list of SparseField
            List of SparseField instances to combine.

        Returns
        -------
        SparseField
            A new SparseField instance containing the combined points and values.
        """
        if not sparse_fields:
            raise ValueError("Input list of SparseVolume is empty.")
        origin = sparse_fields[0].origin
        spacing = sparse_fields[0].spacing
        shape = sparse_fields[0].shape
        device = sparse_fields[0].device

        for v in sparse_fields:
            if v.origin != origin:
                raise ValueError("All SparseVolumes must have the same origin.")
            if v.spacing != spacing:
                raise ValueError("All SparseVolumes must have the same spacing.")
            if v.shape != shape:
                raise ValueError("All SparseVolumes must have the same shape.")
            if v.device != device:
                raise ValueError("All SparseVolumes must be on the same device.")
            
        all_points = torch.cat([v.points for v in sparse_fields], dim=0)
        all_values = torch.cat([v.values for v in sparse_fields], dim=0)
        return cls(all_points, all_values, shape=shape, spacing=spacing, origin=origin, device=device)

    @override
    def reconstruct(self, numpy=True):
        """
        Reconstruct the full 3D numpy array from the sparse representation, including scalar values.

        Parameters
        ----------
        numpy: bool
            If True, returns a numpy array. If False, returns a torch tensor.

        Returns
        -------
        np.ndarray
            3D numpy array with nonzero elements at stored coordinates, filled with scalar values. dimensions are (z, y, x).
        """
        arr = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        arr[tuple(self.points.T)] = self.values
        if numpy:
            arr = arr.cpu().numpy()
        return arr
    
    @override
    def extract_points_based_on_mask(self, mask):
        """
        Create a new SparseField consisting only of points that match the given boolean mask.

        Parameters
        ----------
        mask: torch.Tensor
            A boolean tensor of shape (N,) indicating which points to keep. Must be the same length as self.points.

        Returns
        -------
        SparseField
            New SparseField containing only the points that match the mask.
        """
        if len(mask) != len(self.points):
            raise ValueError("Mask must have the same length as SparseField's points.")
        
        points = self.points[mask]
        values = self.values[mask]

        new = SparseField(points, values, self.shape, self.spacing, self.origin, device=self.device)
        return new

    
    def values_for_closest_points(self, sparse_volume):
        """
        For each point in the input SparseVolume, find the closest point in self
        and assign its scalar value. Returns a new SparseField with the same points
        as sparse_volume and values from the closest neighbors in self.

        Parameters
        ----------
        sparse_volume: SparseVolume
            The input SparseVolume containing points for which to find closest values. must be on the same device as self.

        Returns
        -------
        SparseField
            A new SparseField with points from sparse_volume and values from closest points in self.
        """
        if sparse_volume.device != self.device:
            raise ValueError("sparse_volume must be on the same device as SparseField.")

        target_tensor = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(sparse_volume.points.to(torch.float32).contiguous()))

        # Build or reuse NearestNeighborSearch on self.points
        self._init_nns()

        indices, _ = self.nns.knn_search(target_tensor, knn=1)
        indices = torch.utils.dlpack.from_dlpack(indices.to_dlpack()).flatten()
        values = self.values[indices]

        return SparseField.from_Sparse_with_values(sparse_volume, values)
    
    def extract_isosurface(self, value, tol = None):
        """
        Create a new SparseVolume containing only the points whose value is equal to the specified value.

        Parameters
        ----------
        value: float
            The value to filter points by.
        
        Returns
        -------
        SparseVolume
            A new SparseVolume containing only the points where self.values == value.
        """
        if tol is None:
            mask = self.values == value
        else:
            mask = torch.abs(self.values - value) < tol
        selected_points = self.points[mask]
        return SparseVolume.from_Sparse_with_points(self, selected_points)
    
    def gradient(self):
        """
        Calculate the gradient of the scalar field represented by this SparseField.

        Returns
        -------
        SparseVectors
            A SparseVectors object containing the gradient vectors at each point. Components of the vectors are in the order (z, y, x).
        """
        self._init_nns()
        
        gradient_x = torch.zeros_like(self.values, dtype=torch.float32)
        edge_x_plus, neighbours_x_plus = self._find_edge_specific_offset(torch.tensor([0, 0, 1], device=self.device))
        edge_x_minus, neighbours_x_minus = self._find_edge_specific_offset(torch.tensor([0, 0, -1], device=self.device))
        gradient_x[edge_x_plus] = self.values[edge_x_plus] - self.values[neighbours_x_minus[edge_x_plus]]
        gradient_x[edge_x_minus] = self.values[neighbours_x_plus[edge_x_minus]] - self.values[edge_x_minus]
        not_edge_x = torch.logical_not(torch.logical_or(edge_x_plus, edge_x_minus))
        gradient_x[not_edge_x] = (self.values[neighbours_x_plus[not_edge_x]] - self.values[neighbours_x_minus[not_edge_x]])/2

        gradient_y = torch.zeros_like(self.values, dtype=torch.float32)
        edge_y_plus, neighbours_y_plus = self._find_edge_specific_offset(torch.tensor([0, 1, 0], device=self.device))
        edge_y_minus, neighbours_y_minus = self._find_edge_specific_offset(torch.tensor([0, -1, 0], device=self.device))
        gradient_y[edge_y_plus] = self.values[edge_y_plus] - self.values[neighbours_y_minus[edge_y_plus]]
        gradient_y[edge_y_minus] = self.values[neighbours_y_plus[edge_y_minus]] - self.values[edge_y_minus]
        not_edge_y = torch.logical_not(torch.logical_or(edge_y_plus, edge_y_minus))
        gradient_y[not_edge_y] = (self.values[neighbours_y_plus[not_edge_y]] - self.values[neighbours_y_minus[not_edge_y]])/2

        gradient_z = torch.zeros_like(self.values, dtype=torch.float32)
        edge_z_plus, neighbours_z_plus = self._find_edge_specific_offset(torch.tensor([1, 0, 0], device=self.device))
        edge_z_minus, neighbours_z_minus = self._find_edge_specific_offset(torch.tensor([-1, 0, 0], device=self.device))
        gradient_z[edge_z_plus] = self.values[edge_z_plus] - self.values[neighbours_z_minus[edge_z_plus]]
        gradient_z[edge_z_minus] = self.values[neighbours_z_plus[edge_z_minus]] - self.values[edge_z_minus]
        not_edge_z = torch.logical_not(torch.logical_or(edge_z_plus, edge_z_minus))
        gradient_z[not_edge_z] = (self.values[neighbours_z_plus[not_edge_z]] - self.values[neighbours_z_minus[not_edge_z]])/2

        gradients = torch.stack((gradient_z, gradient_y, gradient_x), axis=-1)
        norms = torch.linalg.norm(gradients, axis=-1, keepdims=True)
        nb_null = torch.sum(norms == 0)
        if nb_null > 0:
            logger.warning(f"Gradient calculation resulted in {nb_null} null vectors")
        # Avoid division by zero
        norms[norms == 0] = 1.0
        gradients /= norms

        # Create SparseVectors object with the calculated gradients
        sparse_vectors = SparseVectors.from_Sparse_with_vectors(self, gradients)
        return sparse_vectors
    
    
    def backward_smooth(self, lambda_param=1.0, tol =1e-5, maxiter=20, values_to_keep= [0,1]):
        """
        Smoothen the scalar field using a Laplacian smoothing approach.

        Parameters
        ----------
        lambda_param: float
            The smoothing parameter, controls the strength of the smoothing.
        iterations: int
            Number of iterations to perform for smoothing.
        tol: float
            Tolerance for convergence of the iterative linear solver.
        max_iter: int
            Maximum number of iterations for the iterative linear solve.
        values_to_keep: list of float
            List of scalar values to keep unchanged during smoothing. 

        Returns
        -------
        SparseField
            A new SparseField with the smoothed values.
        exit_code: int
            Exit code from the GMRES solver, 0 indicates successful exit because of convergence. if not, equal to the number of iterations performed.
        """

        A,b = self.setup_A_system(values_to_keep=values_to_keep, lambda_param=lambda_param)
        print("start resolution")
        precond = scipy.sparse.diags(1/A.diagonal(), format='csr')
        result, info = scipy.sparse.linalg.gmres(A,b, x0=b, maxiter=maxiter, M=precond, tol=tol, callback_type="legacy", callback=lambda pr_norm: print("Residual norm:", pr_norm))
        return SparseField.from_Sparse_with_values(self, result), info
    
    def setup_laplacian_system(self, values_to_keep=[0,1]):
        """
        Get the Laplacian matrix of the SparseField.

        Returns
        -------
        scipy.sparse.csr_matrix
            The laplacian matrix in CSR format.
        np.ndarray
            The RHS vector for the linear system, containing the values of the SparseField.
        """
        rows = []
        cols = []
        data = []
        
        all_neighbours = -1 * torch.ones(self.n_points,6, dtype=torch.int64) # initialize with -1 to indicate no neighbour
        b = self.values.clone().detach()

        edge_x_plus, neighbours_x_plus = self._find_edge_specific_offset(torch.tensor([0, 0, 1], device=self.device))
        edge_x_minus, neighbours_x_minus = self._find_edge_specific_offset(torch.tensor([0, 0, -1], device=self.device))
        all_neighbours[~edge_x_plus, 0] = neighbours_x_plus[~edge_x_plus]
        all_neighbours[~edge_x_minus, 1] = neighbours_x_minus[~edge_x_minus]

        edge_y_plus, neighbours_y_plus = self._find_edge_specific_offset(torch.tensor([0, 1, 0], device=self.device))
        edge_y_minus, neighbours_y_minus = self._find_edge_specific_offset(torch.tensor([0, -1, 0], device=self.device))
        all_neighbours[~edge_y_plus, 2] = neighbours_y_plus[~edge_y_plus]
        all_neighbours[~edge_y_minus, 3] = neighbours_y_minus[~edge_y_minus]

        edge_z_plus, neighbours_z_plus = self._find_edge_specific_offset(torch.tensor([1, 0, 0], device=self.device))
        edge_z_minus, neighbours_z_minus = self._find_edge_specific_offset(torch.tensor([-1, 0, 0], device=self.device))
        all_neighbours[~edge_z_plus, 4] = neighbours_z_plus[~edge_z_plus]
        all_neighbours[~edge_z_minus, 5] = neighbours_z_minus[~edge_z_minus]

        for value in values_to_keep:
            print(f"Keeping value {value} unchanged during Laplacian calculation.")
            mask = self.values == value # (N,)
            bc_indices = torch.where(mask)[0] #(k,) with k the number of BC points

            # step 1: set that the bc have no neighbours for when constructing their rows in laplacian matrix
            all_neighbours[mask] = -1  # Set neighbours to -1 for points with the value to keep, as if they have no neighbours
            b[mask] = value  # Set the corresponding b values to the value to keep

            #step 2: remove bc points that ended up as neighbours of others and pass their values to the b vector
            mask_for_bc_indices_to_remove = torch.isin(all_neighbours, bc_indices) #(N,6)
            number_of_bc_neighbours_per_row = torch.sum(mask_for_bc_indices_to_remove, dim=1) #(N,)
            all_neighbours[mask_for_bc_indices_to_remove] = -1  # Remove bc points from neighbours
            b -= value * number_of_bc_neighbours_per_row  # Subtract the value from
        
        #return all_neighbours, b
        diagonal_weights = torch.sum(all_neighbours != -1, dim=1, dtype=torch.float32)  # Count valid neighbours for each point

        # This transform the laplacian matrix to iden - lambda_param * laplacian, this operation only changes the diagonal weights and also sets the weights of the bc points to 1
        #diagonal_weights = 1 - lambda_param * diagonal_weights  # Calculate diagonal weights
        
        diagonal_weights[diagonal_weights == 0] = 1.0  # for the bc points we fixed at having no neighbours
        
        diagonal_rows = torch.arange(self.n_points, device=self.device)
        diagonal_cols = diagonal_rows.clone()


        rows_off = torch.arange(self.n_points, device=self.device).repeat_interleave(6)
        cols_off = all_neighbours.flatten()
        weights_off = -1 * torch.ones_like(cols_off, dtype=torch.float32, device=self.device) 

        mask = cols_off != -1  # Filter out -1 neighbours
        rows_off = rows_off[mask]
        cols_off = cols_off[mask]
        weights_off = weights_off[mask]


        rows = torch.concatenate([diagonal_rows, rows_off]).cpu().numpy()
        cols = torch.concatenate([diagonal_cols, cols_off]).cpu().numpy()
        data = torch.concatenate([diagonal_weights, weights_off]).cpu().numpy()


        laplacian = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(self.n_points, self.n_points)).tocsr()
        b = b.cpu().numpy()


        return laplacian, b


    def setup_A_system(self, values_to_keep=[0,1], lambda_param=1.0):
        """
        Set up the system for backward laplacian smoothing where the matrix A is defined as A = I - lambda_param * L, where L is the laplacian matrix.

        Parameters
        ----------
        values_to_keep: list of float
            List of scalar values of the field that should be fixed as boundary conditions during the smoothing process.
        lambda_param: float
            The smoothing parameter, controls the strength of the smoothing.

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix A in CSR format.
        np.ndarray
            The RHS vector for the linear system, containing the values of the SparseField and incorporating the boundary conditions.
        """
        rows = []
        cols = []
        data = []
        
        all_neighbours = self._find_all_neighbours()
        b = self.values.clone().detach()

        #return all_neighbours, b
        diagonal_weights = torch.sum(all_neighbours != -1, dim=1, dtype=torch.float32)  # Count valid neighbours for each point

        # This transform the laplacian matrix to iden - lambda_param * laplacian, this operation only changes the diagonal weights and also sets the weights of the bc points to 1
        diagonal_weights = 1 - lambda_param * diagonal_weights  # Calculate diagonal weights
        diagonal_rows = torch.arange(self.n_points, device=self.device)
        diagonal_cols = diagonal_rows.clone()

        for value in values_to_keep:
            print(f"Keeping value {value} unchanged during Laplacian calculation.")
            mask = self.values == value # (N,)
            bc_indices = torch.where(mask)[0] #(k,) with k the number of BC points

            # step 1: set that the bc have no neighbours for when constructing their rows in laplacian matrix
            all_neighbours[mask] = -1 
            b[mask] = value  
            diagonal_weights[mask] = 1.0  # Set the diagonal weights to 1 for the bc points

            #step 2: remove bc points that ended up as neighbours of others and pass their values to the b vector
            mask_for_bc_indices_to_remove = torch.isin(all_neighbours, bc_indices) #(N,6)
            number_of_bc_neighbours_per_row = torch.sum(mask_for_bc_indices_to_remove, dim=1) #(N,)
            all_neighbours[mask_for_bc_indices_to_remove] = -1  # Remove bc points from neighbours
            b -= value *lambda_param * number_of_bc_neighbours_per_row 


        rows_off = torch.arange(self.n_points, device=self.device).repeat_interleave(6)
        cols_off = all_neighbours.flatten()
        weights_off = lambda_param * torch.ones_like(cols_off, dtype=torch.float64, device=self.device) # -1 for the laplacian becomes +lambda_param in A = iden - lambda_param * laplacian
    
        mask = cols_off != -1  # Filter out -1 neighbours
        rows_off = rows_off[mask]
        cols_off = cols_off[mask]
        weights_off = weights_off[mask]

        rows = torch.concatenate([diagonal_rows, rows_off]).cpu().numpy()
        cols = torch.concatenate([diagonal_cols, cols_off]).cpu().numpy()
        data = torch.concatenate([diagonal_weights, weights_off]).cpu().numpy()
        laplacian = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(self.n_points, self.n_points)).tocsr()
        b = b.cpu().numpy()

        return laplacian, b
    
    def forward_smooth(self, lambda_param=0.01, iter=20, values_to_keep=[0,1], norm_list=None):
        """
        Smoothen the scalar field using a forward Laplacian smoothing approach. it is recommended to use a small lambda_param (e.g., 0.01) and to instead increase the number of iterations for better results.

        Parameters
        ----------
        lambda_param: float
            The smoothing parameter, controls the strength of the smoothing.
        iter: int
            Number of iterations to perform for smoothing.
        values_to_keep: list of float
            List of scalar values of the field that should be fixed as boundary conditions during the smoothing process.
        norm_list: list of float, optional
            If provided, will store the norm of the difference between the new and old values at each iteration for analysis.

        Returns
        -------
        SparseField
            A new SparseField with the smoothed values.
        """
        
        neighbours = self._find_all_neighbours()
        mask = neighbours != -1
        nb_neighbours_per_row = torch.sum(mask, axis=1)

        # find bc indices
        bc_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        for value in values_to_keep:
            bc_indices = torch.cat((bc_indices, torch.where(self.values == value)[0]), dim=0)

        old_values = self.values
        new_values = self.values

        for i in tqdm(range(iter)):
            temp_values = new_values.clone().detach()
            add_values = neighbours.clone().detach().to(dtype=torch.float32) # create a copy of neighbours to add values
            add_values[mask] = new_values[neighbours[mask]]
            add_values[~mask] = 0

            add_values_sum = torch.sum(add_values, axis=1) # sum of all neighbours
            laplacian = (- nb_neighbours_per_row * new_values + add_values_sum) / nb_neighbours_per_row # this is the laplacian:  m * value - sum of m neighbours 

            laplacian[bc_indices] = 0 # so the points at bc never change
            new_values = new_values + lambda_param * laplacian
            old_values = temp_values
            if norm_list is not None:
                norm_list.append(torch.norm(new_values - old_values).item())
                print(f"Iteration {i+1}/{iter}, Norm of change: {norm_list[-1]}")
                print(f"norm of laplacian: {torch.norm(laplacian).item()}")
        
        return SparseField.from_Sparse_with_values(self, new_values)
 
    def replace_nan(self):
        """
        Replace NaN values in the SparseField with mean of closest non-NaN values.

        Returns
        -------
        SparseField
            A new SparseField with NaN values replaced by zeros.
        """
        mask = torch.isnan(self.values)
        nb_nans = torch.sum(mask)
        if nb_nans == 0:
            return self
        
        logger.warning(f"{nb_nans} points have NaN values found in SparseField. Replacing with closest values.")
        valid_field  = self.extract_points_based_on_mask(~mask)
        missing_points = self.extract_points_based_on_mask(mask)
        new_values = valid_field.values_for_closest_points(missing_points)
        new_values = new_values.values
        self.values[mask] = new_values

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
            raise ValueError("sparse_template must be on the same device as SparseField.")
        if sparse_template.n_points != self.n_points:
            raise ValueError("sparse_template must have the same number of points as SparseField.")
        
        members, indices = self.check_point_membership(sparse_template.points, return_indices=True)
        if not torch.all(members):
            raise ValueError("Not all points in self are present in sparse_template.")
        reordered_values = self.values[indices]
        return SparseField.from_Sparse_with_values(sparse_template, reordered_values)
    
            
    #def setup_heat_solve(self, lambda_param=1.0):
    #    """
    #    Get the Laplacian matrix of the SparseField.
#
    #    Returns
    #    -------
    #    scipy.sparse.csr_matrix
    #        The laplacian matrix in CSR format.
    #    np.ndarray
    #        The RHS vector for the linear system, containing the values of the SparseField.
    #    """
    #    rows = []
    #    cols = []
    #    data = []
    #    
    #    all_neighbours = -1 * torch.ones(self.n_points,6, dtype=torch.int64) # initialize with -1 to indicate no neighbour
    #    b = torch.zeros(self.n_points, dtype=torch.float32, device=self.device)  # Initialize b with zeros
#
    #    edge_x_plus, neighbours_x_plus = self._find_edge_specific_offset(torch.tensor([0, 0, 1], device=self.device))
    #    edge_x_minus, neighbours_x_minus = self._find_edge_specific_offset(torch.tensor([0, 0, -1], device=self.device))
    #    all_neighbours[~edge_x_plus, 0] = neighbours_x_plus[~edge_x_plus]
    #    all_neighbours[~edge_x_minus, 1] = neighbours_x_minus[~edge_x_minus]
#
    #    edge_y_plus, neighbours_y_plus = self._find_edge_specific_offset(torch.tensor([0, 1, 0], device=self.device))
    #    edge_y_minus, neighbours_y_minus = self._find_edge_specific_offset(torch.tensor([0, -1, 0], device=self.device))
    #    all_neighbours[~edge_y_plus, 2] = neighbours_y_plus[~edge_y_plus]
    #    all_neighbours[~edge_y_minus, 3] = neighbours_y_minus[~edge_y_minus]
#
    #    edge_z_plus, neighbours_z_plus = self._find_edge_specific_offset(torch.tensor([1, 0, 0], device=self.device))
    #    edge_z_minus, neighbours_z_minus = self._find_edge_specific_offset(torch.tensor([-1, 0, 0], device=self.device))
    #    all_neighbours[~edge_z_plus, 4] = neighbours_z_plus[~edge_z_plus]
    #    all_neighbours[~edge_z_minus, 5] = neighbours_z_minus[~edge_z_minus]
    #    #return all_neighbours, b
    #    diagonal_weights = torch.sum(all_neighbours != -1, dim=1, dtype=torch.float32)  # Count valid neighbours for each point
#
    #    # This transform the laplacian matrix to iden - lambda_param * laplacian, this operation only changes the diagonal weights and also sets the weights of the bc points to 1
    #    diagonal_weights = 1 - lambda_param * diagonal_weights  # Calculate diagonal weights
    #    diagonal_rows = torch.arange(self.n_points, device=self.device)
    #    diagonal_cols = diagonal_rows.clone()
#
    #    heat_injection = 1
    #    mask = self.values == 0 # (N,)
    #    bc_indices = torch.where(mask)[0] #(k,) with k the number of BC points
#
    #    # step 1: set that the bc have no neighbours for when constructing their rows in laplacian matrix
    #    all_neighbours[mask] = -1  # Set neighbours to -1 for points with the value to keep, as if they have no neighbours
    #    b[mask] = heat_injection  # Set the corresponding b values to the value to keep
    #    diagonal_weights[mask] = 1.0  # Set the diagonal weights to 1 for the bc points
#
    #    #step 2: remove bc points that ended up as neighbours of others and pass their values to the b vector
    #    mask_for_bc_indices_to_remove = torch.isin(all_neighbours, bc_indices) #(N,6)
    #    number_of_bc_neighbours_per_row = torch.sum(mask_for_bc_indices_to_remove, dim=1) #(N,)
    #    all_neighbours[mask_for_bc_indices_to_remove] = -1  # Remove bc points from neighbours
    #    #b -= value * lambda_param * number_of_bc_neighbours_per_row  # Subtract the value from RHS
    #    b -= heat_injection *lambda_param * number_of_bc_neighbours_per_row 
#
    #    rows_off = torch.arange(self.n_points, device=self.device).repeat_interleave(6)
    #    cols_off = all_neighbours.flatten()
    #    weights_off = lambda_param * torch.ones_like(cols_off, dtype=torch.float32, device=self.device) # -1 for the laplacian becomes +lambda_param in A = iden - lambda_param * laplacian
    #    mask = cols_off != -1  # Filter out -1 neighbours
    #    rows_off = rows_off[mask]
    #    cols_off = cols_off[mask]
    #    weights_off = weights_off[mask]
#
    #    rows = torch.concatenate([diagonal_rows, rows_off]).cpu().numpy()
    #    cols = torch.concatenate([diagonal_cols, cols_off]).cpu().numpy()
    #    data = torch.concatenate([diagonal_weights, weights_off]).cpu().numpy()
#
    #    laplacian = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(self.n_points, self.n_points)).tocsr()
    #    b = b.cpu().numpy()
#
#
    #    return laplacian, b
    #
    #def solve_heat(self, lambda_param=1, maxiter=50, tol=1e-5):
#
    #    A, b = self.setup_heat_solve(lambda_param=lambda_param)
#
    #    precond = scipy.sparse.diags(1/A.diagonal(), format='csr')
    #    result, info = scipy.sparse.linalg.gmres(A,b, x0=b, maxiter=maxiter, tol=tol, M=precond, callback_type="pr_norm", callback=lambda pr_norm: print("Residual norm:", pr_norm))
    #    return SparseField.from_Sparse_with_values(self, result), info

    @override
    def __str__(self):
        return (f"SparseField(spacing={self.spacing}, origin={self.origin}, "
                f"shape={self.shape}, "
                f"num_points={len(self.points)}, device={self.device}, "
                f"values_mean={torch.mean(self.values):.2f})")

    @override
    def to_pickle(self, filename):
        """
        Save the SparseField to a pickle file as a dictionary. Can then be loaded with `SparseField.from_pickle(filename)`.

        Parameters
        ----------
        filename: str
            Path to the output pickle file.
        """
        dict = {"origin": self.origin,
                "spacing": self.spacing,
                "shape": self.shape,
                "points": self.points.cpu().numpy(),
                "values": self.values.cpu().numpy()}
        with open(filename, 'wb') as f:
            pkl.dump(dict, f)
        logger.info(f"SparseField saved to {filename}")

    @classmethod
    @override
    def from_pickle(cls, filename, device="cpu"):
        """
        Load a SparseField from a pickle file, pickle file must have been created with `SparseField.to_pickle(filename)`
        and is saved as a dictionary with keys: "origin", "spacing", "shape", "points", "values".

        Parameters
        ----------
        filename: str
            Path to the input pickle file.
        device: str
            Device string for torch, e.g., "cuda:#" or "cpu". Default is "cpu".

        Returns
        -------
        SparseField
            An instance of SparseField loaded from the pickle file.
        """
        with open(filename, 'rb') as f:
            class_dict = pkl.load(f)

        if not all(key in class_dict for key in ["origin", "spacing", "shape", "points", "values"]):
            raise ValueError("Pickle file must contain 'origin', 'spacing', 'shape', 'points', and 'values' keys.")
        
        return cls(class_dict["points"], class_dict["values"], class_dict["shape"], class_dict["spacing"], class_dict["origin"], device=device)

