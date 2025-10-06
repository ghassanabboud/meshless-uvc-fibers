import numpy as np
import SimpleITK as sitk
import logging
import open3d as o3d
import torch
import pickle as pkl
import torch.utils.dlpack
import cc3d
import skfmm
import numpy.ma as ma
import networkx as nx


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SparseVolume:
    """
    A class representing a sparse 3D matrix, storing only the coordinates of nonzero voxels. While the nonzero voxel coordinates are integers values, representing slice indices in a 3D matrix, 
    attributes "spacing" and "origin" represent continuous physical coordinates and are allow to map the 3D matrix to a continuous-space volume. This choice allows to interface with 
    libraries like SimpleITK that process images as 3D numpy arrays with associated spacing and origin. This class is best used when reading 3D data from files like .mha.

    "spacing" and "origin" have coordinates in order (x, y, z) while the nonzero voxel indices are stored in order (z, y, x). This choice is consistent with SimpleITK.
    """
    def __init__(self, points, shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), device="cpu"):
        """
        Initialize the SparseVolume from a list of 3D points.

        Parameters
        ----------
        points: np.ndarray or torch.Tensor
            2D numpy array / torch Tensor with shape (N, 3) containing the coordinates of nonzero voxels. order is (z, y, x).
        shape: torch.Size
            Shape of the 3D volume (z, y, x).
        spacing: tuple, optional
            Spacing between voxels in each dimension. (x, y, z). Default is (1.0, 1.0, 1.0).
        origin: tuple, optional
            Coordinates of the origin. (x, y, z). Default is (0.0, 0.0, 0.0).
        device: str
            Device string for torch, e.g., "cuda:0" or "cpu". Default is "cpu".

        """
        self.device = self._resolve_device(device)
        
        if isinstance(points, torch.Tensor):
            if points.ndim != 2:
                raise ValueError("Input torch tensor must have 2 dimensions (N,3).")
            points = points.to(dtype=torch.int64, device=device)
        elif isinstance(points, np.ndarray):
            if points.ndim != 2:
                raise ValueError("Input numpy array must have 2 dimensions (N,3).")
            points = torch.tensor(points, dtype=torch.int64, device=device)

        if len(points) == 0:
            raise ValueError("Cannot create SparseVolume with no points.")
        
        self.points = points
        self.shape = shape
        self.spacing = spacing
        self.origin = origin
        self.n_points = len(self.points)
        self.nns = None  # NearestNeighborSearch object for Open3D
        self.graph = None
        self.laplacian = None  

    @classmethod
    def from_array(cls, arr, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), device="cpu"):
        """
        Initialize a SparseVolume from a sparse 3D numpy array.

        Parameters
        ----------
        arr: np.ndarray or torch.Tensor
            3D numpy array / torch Tensor with shape (z,y,x) containing nonzero voxels.
        spacing: tuple, optional
            Spacing between voxels in each dimension. (x, y, z). Default is (1.0, 1.0, 1.0).
        origin: tuple, optional
            Coordinates of the origin. (x, y, z). Default is (0.0, 0.0, 0.0).
        device: str
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
            arr = arr.to(dtype=torch.int64, device=device)
        elif isinstance(arr, np.ndarray):
            if arr.ndim != 3:
                raise ValueError("Input numpy array must have 3 dimensions (z,y,x).")
            arr = torch.tensor(arr, dtype=torch.int64, device=device)

        points = torch.argwhere(arr > 0)
        obj = cls(points, shape=arr.shape, spacing=spacing, origin=origin, device=device)

        return obj


    @classmethod
    def from_sitk_image(cls, sitk_image, device="cpu"):
        """
        Initialize a SparseVolume from a SimpleITK image.

        Parameters
        ----------
        sitk_image: SimpleITK.Image
            The input SimpleITK image.
        device: str
            Device string for torch, e.g, "cuda:#" or "cpu". Default is "cpu".

        Returns
        -------
        SparseVolume
            An instance of SparseVolume initialized from the SimpleITK image.
        """
        arr = sitk.GetArrayFromImage(sitk_image)
        spacing = tuple(sitk_image.GetSpacing())
        origin = tuple(sitk_image.GetOrigin())
        obj = cls.from_array(arr, spacing=spacing, origin=origin, device=device)
        return obj
    
    @classmethod
    def from_mha_file(cls, filename, device="cpu"):
        """
        Initialize a SparseVolume directly from an .mha file.

        Parameters
        ----------
        filename: str
            Path to the .mha file.
        device: str
            Device string for torch, e.g., "cuda:#" or "cpu". Default is "cpu".

        Returns
        -------
        SparseVolume
            An instance of SparseVolume initialized from the .mha file.
        """
        sitk_image = sitk.ReadImage(filename)
        obj = cls.from_sitk_image(sitk_image, device=device)
        return obj
    
    @classmethod
    def from_Sparse_with_points(cls, other, points):
        """
        Create a new SparseVolume by copying all attributes from another SparseVolume,
        except for the points, which are provided as an argument. Tensor is created on the same device as other.

        Parameters
        ----------
        other: SparseVolume
            The source SparseVolume to copy attributes from.
        points: np.ndarray / torch.Tensor
            New points array (N, 3) to use in the new SparseVolume.

        Returns
        -------
        SparseVolume
            A new SparseVolume instance with the specified points.
        """

        return cls(points, shape=other.shape, spacing=other.spacing, origin=other.origin, device=other.device)

    
    @classmethod
    def combine(cls, volumes):
        """
        Combine a list of SparseVolume objects into a single SparseVolume.
        Checks that all have the same origin, spacing, shape, and device.

        Parameters
        ----------
        volumes: list of SparseVolume
            The SparseVolumes to combine.

        Returns
        -------
        SparseVolume
            A new SparseVolume with points from all input volumes stacked together.
        """
        if not volumes:
            raise ValueError("Input list of SparseVolume is empty.")
        origin = volumes[0].origin
        spacing = volumes[0].spacing
        shape = volumes[0].shape
        device = volumes[0].device

        for v in volumes:
            if v.origin != origin:
                raise ValueError("All SparseVolumes must have the same origin.")
            if v.spacing != spacing:
                raise ValueError("All SparseVolumes must have the same spacing.")
            if v.shape != shape:
                raise ValueError("All SparseVolumes must have the same shape.")
            if v.device != device:
                raise ValueError("All SparseVolumes must be on the same device.")

        all_points = torch.vstack([v.points for v in volumes if len(v.points) > 0])
        return cls(all_points, shape=shape, spacing=spacing, origin=origin, device=device)
    
    @staticmethod
    def _resolve_device(device):
        """
        Handle device selection for pytorch 
        """
        if isinstance(device, str) and device.startswith("cuda"):
            # Check if CUDA is available in Open3D
            if torch.cuda.is_available():
                return device
            else:
                logger.warning("CUDA not available. Falling back to CPU.")
                return "cpu"
        elif device == "cpu":
            return device
        else:
            raise ValueError(f"Unsupported device: {device}. Use 'cuda:#' or 'cpu'.")
        
    def _init_nns(self):
        """
        Initialize the NearestNeighborSearch object for Open3D if not already done.

        Returns
        -------
        bool
            True if the NearestNeighborSearch was initialized, False if it was already initialized.
        """
        if self.nns is None:
            points_tensor = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(self.points.to(torch.float32).contiguous()))
            self.nns = o3d.core.nns.NearestNeighborSearch(points_tensor)
            self.nns.knn_index()
            return True
        else:
            return False
        
        
    def _init_graph(self):
        """
        Initialize the graph representation of the SparseVolume if not already done.

        Returns
        -------
        bool
            True if the graph was initialized, False if it was already initialized.
        """
        if self.graph is None:

            indices = torch.arange(self.n_points, device=self.device)
            G = nx.Graph()
            G.add_nodes_from(range(self.n_points))

            offsets = torch.tensor([
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1]
            ], device=self.device)

            self._init_nns()

            for offset in offsets:
                neighbours = self.points + offset
                neighbour_query = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(neighbours.to(torch.float32).contiguous()))
                neighbours, sq_dist = self.nns.knn_search(neighbour_query, knn=1)

                neighbours = torch.utils.dlpack.from_dlpack(neighbours.to_dlpack()).flatten()
                sq_dist = torch.utils.dlpack.from_dlpack(sq_dist.to_dlpack()).flatten()
                on_edge = sq_dist > 0.25 # if sq_dist > 0.25, then distance > 0.5, so point is on edge because we could not exactly find a neighbour
                
                edges_to_add = zip(indices[~on_edge].cpu().numpy(), neighbours[~on_edge].cpu().numpy())
                G.add_edges_from(edges_to_add)
            
            self.graph = G
            return True
        else:
            return False
        
    def _init_laplacian(self):
        """
        Initialize the graph Laplacian of the SparseVolume if not already done.

        Returns
        -------
        bool
            True if the Laplacian was initialized, False if it was already initialized.
        """
        if self.laplacian is None:
            self._init_graph()
            self.laplacian = nx.laplacian_matrix(self.graph)
            with open("/home/jtso3/ghassan/fibers_mha_LDRBM/graph_laplacian.pkl ", "wb") as f:
                pkl.dump(self.laplacian, f)
            return True
        else:
            return False
        
    def _get_o3d_device(self):
        """
        Get the Open3D device corresponding to the SparseVolume's device.

        Returns
        -------
        o3d.core.Device
            The Open3D device.
        """
        if self.device.startswith("cuda"):
            _, device_id = self.device.split(":")
            return o3d.core.Device("CUDA"+f":{device_id}")
        elif self.device == "cpu":
            return o3d.core.Device("CPU:0")
        else:
            raise ValueError(f"Unsupported device: {self.device}. Use 'cuda:#' or 'cpu'.")

    
    def reconstruct(self, numpy=True):
        """
        Reconstruct the full 3D array from the sparse representation.

        Parameters
        ----------
        numpy: bool
            If True, returns a numpy array. If False, returns a torch tensor.

        Returns
        -------
        np.ndarray / torch.Tensor (dtype=torch.int64)
            3D numpy array / torch tensor with nonzero elements at stored coordinates. dimensions are (z, y, x).
        """
        arr = torch.zeros(self.shape, dtype=torch.int64, device=self.device)
        if len(self.points) > 0:
            arr[self.points[:, 0], self.points[:, 1], self.points[:, 2]] = 1
        if numpy:
            arr = arr.cpu().numpy()
        return arr

    def physical_point_to_index(self, point):
        """
        Convert a physical (continuous) point to a discrete array index.

        Parameters
        ----------
        point: tuple
            Physical coordinates (x, y, z).

        Returns
        -------
        torch.Tensor
            Corresponding discrete array index (z, y, x). shape (3,).
        """
        if len(point) != 3:
            raise ValueError("Point must be a tuple of 3 elements (x, y, z).")
        idx = []
        for i in range(3):
            index = int(np.floor((point[i] - self.origin[i]) / self.spacing[i] + 0.5))
            if index < 0 or index >= self.shape[i]:
                raise IndexError(f"Physical point {point} is out of bounds in dimension {i}.")
            idx.append(index)
        idx = idx[::-1] # Return as (k, j, i) i.e (z, y, x) to match torch indexing
        idx = torch.tensor(idx, dtype=torch.int32, device=self.device)
        return idx
    
    def physical_point_to_closest_index(self, point):
        """
        Transform a physical point to the closest discrete array index corresponding to a nonzero voxel.

        Parameters
        ----------
        point: tuple
            Physical coordinates (x, y, z).

        Returns
        -------
        torch.Tensor
            Corresponding discrete array index (z, y, x) of the point in the volume closest to input. shape (3,)
        """
        idx = self.physical_point_to_index(point)


        # Check if idx is one of the nonzero points
        if torch.any(torch.all(self.points == idx, dim=1)):
            return idx

        # Prepare Open3D tensor for the query point
        query_tensor = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(idx.to(torch.float32).unsqueeze(0).contiguous()))

        self._init_nns()

        indices, squared_distances = self.nns.knn_search(query_tensor, knn=1)
        indices = torch.utils.dlpack.from_dlpack(indices.to_dlpack())
        squared_distances = torch.utils.dlpack.from_dlpack(squared_distances.to_dlpack())
        closest_idx = self.points[indices][0][0]
        distance = torch.sqrt(squared_distances)[0][0]

        logger.warning(
            f"physical_point_to_closest_index: Point {idx} is not a nonzero voxel. "
            f"Closest nonzero voxel is {closest_idx} (distance {distance:.2f})."
        )
        return closest_idx
    
    @staticmethod
    def read_fcsv_coordinates(filename):
        """
        Read a .fcsv file and return the coordinates of the points in the file.

        Parameters
        ----------
        filename: str
            Path to the .fcsv file. Coordinate are in order (x, y, z).

        Returns
        -------
        list of list
            A list of coordinates, where each coordinate is a list of [x, y, z] values.
            If the file is empty or contains no valid coordinates, returns an empty list.
        """
        coordinates = []
        with open(filename, 'r') as file:
            for line in file:
                if not line.startswith('#') and line.strip():
                    parts = line.split(',')
                    if len(parts) >= 4:
                        x, y, z = map(float, parts[1:4])
                        coordinates.append([x, y, z])

        return coordinates
    
    def extract_shortest_path(self, source, target):
        """
        Extract the shortest path between two points in the SparseVolume using Dijkstra's algorithm.

        Parameters
        ----------
        source: SparseVolume
            The source SparseVolume from which to start the path. All source points must be in self.points.
        target: SparseVolume
            The target SparseVolume to which to find the path. Must have only one point in self.points.
        """
        self._init_graph()
        
        mask = source.check_point_membership(self.points)
        source_indices = torch.where(mask)[0]
        mask = target.check_point_membership(self.points)
        target_index = torch.where(mask)[0]

        distance, path_indices = nx.multi_source_dijkstra(self.graph, sources=set(source_indices.cpu().numpy()), target = target_index.item(), weight='weight')
        mask = torch.zeros(self.n_points, dtype=torch.bool, device=self.device)
        mask[path_indices] = True
        path = self.extract_points_based_on_mask(mask)

        return path
    

    def indices_from_fcsv(self, filename):
        """
        Read a .fcsv file and return the  indices in the sparse volume for each physical point.

        Parameters
        ----------
        filename: str
            Path to the .fcsv file. Coordinate are in order (x, y, z).

        Returns
        -------
        torch.Tensor
            A Tensor of discrete array indices  (z, y, x) corresponding to the closest nonzero voxels
            for each point in the .fcsv file. shape (N,3). device is the same as self.device.
        """
        coordinates = self.read_fcsv_coordinates(filename)
        if not coordinates:
            return torch.empty((0, 3), dtype=torch.int32, device=self.device)

        indices = []
        for pt in coordinates:
            idx = self.physical_point_to_index(tuple(pt))
            indices.append(idx)
        return torch.stack(indices, dim=0)
      
    def closest_indices_from_fcsv(self, filename):
        """
        Read a .fcsv file and return the closest indices in the sparse volume for each physical point.

        Parameters
        ----------
        filename: str
            Path to the .fcsv file. Coordinate are in order (x, y, z).

        Returns
        -------
        torch.Tensor
            A Tensor of discrete array indices  (z, y, x) corresponding to the closest nonzero voxels
            for each point in the .fcsv file. shape (N,3). device is the same as self.device.
        """
        coordinates = self.read_fcsv_coordinates(filename)
        if not coordinates:
            return torch.empty((0, 3), dtype=torch.int32, device=self.device)

        indices = []
        for pt in coordinates:
            idx = self.physical_point_to_closest_index(tuple(pt))
            indices.append(idx)
        return torch.stack(indices, dim=0)
    
    def extract_closest_connected_surfaces(self, filenames):
        """
        Partition self into surfaces of connected points and return new SparseVolumes including only the connected surfaces
        closest to the points in the respective .fcsv files.

        Parameters
        ----------
        filenames: list(str)
            list of paths to .fcsv files. Coordinate are in order (x, y, z).

        Returns
        -------
        list(SparseVolume)
            list of SparseVolumes containing the closest connected point sets to each landmark file.
        """
        full_matrix = self.reconstruct(numpy=True)
        connected_surfaces = cc3d.connected_components(full_matrix, connectivity=26)

        surfaces = []
        for filename in filenames:
            index= self.closest_indices_from_fcsv(filename)[0]
            index = index.cpu().numpy()
            surface_points = np.argwhere(connected_surfaces == connected_surfaces[tuple(index)])
            surface_points = torch.tensor(surface_points, dtype=torch.int32, device=self.device)
            surface = SparseVolume.from_Sparse_with_points(self, surface_points)
            surfaces.append(surface)
        
        return surfaces
    
    def extract_connected_surfaces(self):
        """
        Partition self into surfaces of connected points and return a list of SparseVolumes, each containing one connected surface.

        Returns
        -------
        list(SparseVolume)
            List of SparseVolumes, each containing one connected surface.
        """
        full_matrix = self.reconstruct(numpy=True)
        connected_surfaces = cc3d.connected_components(full_matrix, connectivity=26)

        surfaces = []
        for i in range(1, connected_surfaces.max() + 1):
            surface_points = np.argwhere(connected_surfaces == i)
            surface = SparseVolume.from_Sparse_with_points(self, surface_points)
            surfaces.append(surface)

        return surfaces
    
    def index_to_physical_point(self, index):
        """
        Convert a discrete array index to a physical (continuous) point.

        Parameters
        ----------
        index: tuple
            Discrete array index (z, y, x).

        Returns
        -------
        tuple
            Corresponding physical coordinates (x, y, z).
        """
        #TODO: reimplement without tuples?
        if len(index) != 3:
            raise ValueError("Index must be a tuple of 3 elements (i, j, k).")
        point = tuple(self.origin[i] + index[i] * self.spacing[i] for i in range(3))
        return point[::-1]  # Return as (x, y, z) to match physical coordinates order
    

    def distances_to_points(self, points_target, fmm= True):
        """
        Compute the distances from a set of 3D points to the nearest nonzero voxel in self using Open3D's core NearestNeighborSearch.

        Parameters
        ----------
        points_target: torch.Tensor
            2D torch Tensor of shape (N, 3) containing the target points in index coordinates (z,y,x). same device as self.points.
        fim: bool, optional
            If True, uses the fast iterative method to get the distance of the shortest path through the volume. If False, uses Euclidean distance

        Returns
        -------
        torch.Tensor
            1D tensor of distances from each point in points_target to the nearest nonzero voxel in self. same device as self.points.
        """

        if fmm:
            return self._fmm_distances_to_points(points_target)
        else:
            return self._euclidian_distances_to_points(points_target)
    
    def _euclidian_distances_to_points(self, points_target):
        """
        Compute the Euclidian distances from a set of 3D points to the nearest nonzero voxel in self using Open3D's core NearestNeighborSearch.

        Parameters
        ----------
        points_target: torch.Tensor
            2D torch Tensor of shape (N, 3) containing the target points in index coordinates (z,y,x). same device as self.points.

        Returns
        -------
        torch.Tensor
            1D tensor of distances from each point in points_target to the nearest nonzero voxel in self. same device as self.points.
        """
        if points_target.device != self.points.device:
            raise ValueError("points_target must be on the same device as SparseVolume's points.")

        target_tensor = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(points_target.to(torch.float32).contiguous()))
        self._init_nns()

        _, squared_distances = self.nns.knn_search(target_tensor, knn=1)

        # Convert to torch tensor and get distances
        distances_torch = torch.utils.dlpack.from_dlpack(squared_distances.to_dlpack())
        distances = torch.sqrt(distances_torch)
        return distances.reshape(-1)
    
    def _fmm_distances_to_points(self, points_target):
        """
        Compute the distances of the shortest paths from a set of 3D points to the nearest nonzero voxel in self using the numpy fast marching method.

        Parameters
        ----------
        points_target: torch.Tensor
            2D torch Tensor of shape (N, 3) containing the target points in index coordinates (z,y,x). same device as self.points.

        Returns
        -------
        torch.Tensor
            1D tensor of distances from each point in points_target to the nearest nonzero voxel in self. same device as self.points.
        """
        if points_target.device != self.points.device:
            raise ValueError("points_target must be on the same device as SparseVolume's points.")
        
        volume = self.reconstruct(numpy=True)
        phi = np.ones_like(volume)
        phi[volume>0]= 0
        mask = np.ones_like(volume, dtype=bool)
        mask[points_target[:, 0], points_target[:, 1], points_target[:, 2]] = False
        mask = np.logical_and(mask, volume == 0)  # mask only the points that are not in the volume
        phi = ma.masked_array(phi, mask=mask)

        dist = skfmm.distance(phi)

        if isinstance(dist, ma.MaskedArray):
            sum = np.sum(dist.mask[points_target[:, 0], points_target[:, 1], points_target[:, 2]])
            if sum>0:
                logger.warning(f"{sum} target points were not connected to the volume through a path, their distance values will be NaN.")
                logger.warning("Please make sure your input volume is fully connected to the target points.")
            
        distances = dist[points_target[:, 0], points_target[:, 1], points_target[:, 2]]
        distances = torch.tensor(distances, dtype=torch.float32, device=self.device)
        return distances
    
    def get_centroid(self):
        """
        Compute the centroid of the SparseVolume.

        Returns
        -------
        torch.Tensor
            A tensor of shape (3,) representing the centroid coordinates (z,y,x).
        """

        centroid = torch.mean(self.points.to(torch.float32), dim=0)
        return centroid

        
    def extract_edge_points(self):
        """
        Create a new SparseVolume consisting only of edge points.
        A point is considered at the edge if at least one of its 6-connected neighbors
        is not present in the volume (distance > 0.5).

        Returns
        -------
        SparseVolume
            New SparseVolume containing only the edge points.
        """

        offsets = torch.tensor([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], device=self.device)

        self._init_nns()

        on_edge = torch.tensor([False] * len(self.points), device=self.device)
        for offset in offsets:
            on_edge_based_on_this_loop, _ = self._find_edge_specific_offset(offset)
            on_edge = torch.logical_or(on_edge, on_edge_based_on_this_loop)

        edge_points = self.points[on_edge]
        return SparseVolume.from_Sparse_with_points(self, edge_points)
    
    def check_point_membership(self, query_points, return_indices=False):
        """
        Check if the given points are part of the SparseVolume.

        Parameters
        ----------
        query_points: torch.Tensor
            2D tensor of shape (N, 3) containing the points to check in index coordinates (z,y,x).

        Returns
        -------
        torch.Tensor
            A boolean tensor of shape (N,) indicating whether each point is part of the SparseVolume.
        """
        if query_points.device != self.points.device:
            raise ValueError("Input points must be on the same device as SparseVolume's points.")
        
        self._init_nns()

        query_points = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(query_points.to(torch.float32).contiguous()))
        neighbours, sq_dist = self.nns.knn_search(query_points, knn=1)

        sq_dist = torch.utils.dlpack.from_dlpack(sq_dist.to_dlpack()).flatten()
        members = sq_dist < 0.25 # if sq_dist < 0.25, then distance < 0.5, so points are present
        if return_indices:
            indices = torch.utils.dlpack.from_dlpack(neighbours.to_dlpack()).flatten()
            return members, indices
        return members
    
    def eliminate_non_connected_points(self):
        """
        Eliminate points that are not connected to the main component of the SparseVolume.
        This is done by defining 26-connected regions using cc3d and keeping only the largest connected component.

        Returns
        -------
        SparseVolume
            A new SparseVolume containing only the points that are connected to the main component.
        """
        full_matrix = self.reconstruct(numpy=True)
        connected_surfaces = cc3d.connected_components(full_matrix, connectivity=6)
        unique_components, counts = np.unique(connected_surfaces, return_counts=True)

        # Remove component 0 (background) and its count
        mask = unique_components != 0
        unique_components = unique_components[mask]
        counts = counts[mask]

        if len(unique_components) == 1:
            logger.info("All points are connected, returning original SparseVolume.")
            return self
        
        # CC3D does not guarantee that the components are labeled from bigger to smaller so we need to check which label has the highest count
        logger.info(f"Found {len(unique_components)} connected components with counts: {counts}")
        logger.info("Eliminating non-connected points, keeping only the largest component.")
        largest_component = unique_components[np.argmax(counts)]
        points = np.argwhere(connected_surfaces == largest_component)
        new_volume = SparseVolume.from_Sparse_with_points(self, points)
        return new_volume
    

    def extract_points_based_on_mask(self, mask):
        """
        Create a new SparseVolume consisting only of points that match the given boolean mask.

        Parameters
        ----------
        mask: torch.Tensor
            A boolean tensor of shape (N,) indicating which points to keep. Must be the same length as self.points.

        Returns
        -------
        SparseVolume
            New SparseVolume containing only the points that match the mask.
        """
        if len(mask) != len(self.points):
            raise ValueError("Mask must have the same length as SparseVolume's points.")
        
        points = self.points[mask]

        new = SparseVolume.from_Sparse_with_points(self, points)
        return new
    
    
    def _find_edge_specific_offset(self, offset):
        """
        Create a mask for points on the edge based on a specific offset, eg. points who do not have top neighbour based on offset [1,0,0].

        Parameters
        ----------
        offset: torch.Tensor
            Offset vector to check for edge points. (3,)

        Returns
        -------
        torch.Tensor
            Boolean mask indicating which points are on the edge based on the offset. (N,)
        """
        self._init_nns()

        neighbours = self.points + offset
        neighbour_query = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(neighbours.to(torch.float32).contiguous()))
        neighbours, sq_dist = self.nns.knn_search(neighbour_query, knn=1)

        neighbours = torch.utils.dlpack.from_dlpack(neighbours.to_dlpack()).flatten()
        sq_dist = torch.utils.dlpack.from_dlpack(sq_dist.to_dlpack()).flatten()
        on_edge = sq_dist > 0.25 # if sq_dist > 0.25, then distance > 0.5, so point is on edge because we could not exactly find a neighbour
        return on_edge, neighbours
    
    def _find_all_neighbours(self):
        """
        Find the indices of the 6 neighbours for each point in the SparseVolume.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, 6) where N is the number of points in the SparseVolume.
            Each row contains the indices of the 6 neighbours in self.points for the corresponding point.
            Values of -1 indicate that the neighbour does not exist (i.e., the point is on the edge of the volume).
        """

        all_neighbours = -1 * torch.ones(self.n_points,6, dtype=torch.int64) # initialize with -1 to indicate no neighbou

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

        return all_neighbours
    
    def __str__(self):
        return (f"SparseVolume(spacing={self.spacing}, origin={self.origin}, "
                f"shape={self.shape}, "
                f"num_nonzero_points={len(self.points)}, device={self.device})")

    def __repr__(self):
        return self.__str__()
    
    def to_pickle(self, filename):
        """
        Save the SparseVolume to a pickle file as a dictionary. Can then be loaded with `SparseVolume.from_pickle(filename)`.

        Parameters
        ----------
        filename: str
            Path to the output pickle file.
        """
        dict = {"origin": self.origin,
                "spacing": self.spacing,
                "shape": self.shape,
                "points": self.points.cpu().numpy()}
        with open(filename, 'wb') as f:
            pkl.dump(dict, f)
        logger.info(f"SparseVolume saved to {filename}")

    @classmethod
    def from_pickle(cls, filename, device="cpu"):
        """
        Load a SparseVolume from a pickle file, pickle file must have been created with `SparseVolume.to_pickle(filename)`
        and is saved as a dictionary with keys: "origin", "spacing", "shape", "points".

        Parameters
        ----------
        filename: str
            Path to the input pickle file.
        device: str
            Device string for torch, e.g., "cuda:#" or "cpu". Default is "cpu".

        Returns
        -------
        SparseVolume
            An instance of SparseVolume loaded from the pickle file.
        """
        with open(filename, 'rb') as f:
            class_dict = pkl.load(f)

        if not all(key in class_dict for key in ["origin", "spacing", "shape", "points"]):
            raise ValueError("Pickle file must contain 'origin', 'spacing', 'shape', and 'points' keys.")
        
        return cls(class_dict["points"], class_dict["shape"], class_dict["spacing"], class_dict["origin"], device=device)
    
    def to_mha(self, filename):
        """
        Save the SparseVolume to an .mha file.
        
        Parameters
        ----------
        filename: str
            Path to the output .mha file.

        Returns
        -------
        None
        
        """
        arr = self.reconstruct(numpy=True)
        sitk_image = sitk.GetImageFromArray(arr)
        sitk_image.SetSpacing(self.spacing)
        sitk_image.SetOrigin(self.origin)
        sitk.WriteImage(sitk_image, filename)



  