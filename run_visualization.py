import pyvista as pv
import numpy as np
import pickle as pkl
from cemetery.utils import get_image_array
import logging
from omegaconf import DictConfig
import hydra
from Sparse import SparseVectors, SparseField, SparseVolume
import torch

@hydra.main(version_base=None, config_path= "configs", config_name="visualization_config")
def run_visualization(cfg: DictConfig) -> None:

    if cfg.files.coords_path is not None:
        data = SparseField.from_pickle(cfg.files.coords_path, device="cpu")

    if cfg.files.fibers_path is not None:
        data = SparseVectors.from_pickle(cfg.files.fibers_path, device="cpu")


    if cfg.segment != "full":
        if cfg.files.border_path is None:
            raise ValueError("Border path must be provided if segment is not 'full'.")
        
        border = SparseField.from_pickle(cfg.files.border_path, device="cpu")

        if cfg.segment == "RV_endo":
            surface = border.extract_isosurface(cfg.labels.RV_endo_label)
        elif cfg.segment == "LV_endo":
            surface = border.extract_isosurface(cfg.labels.LV_endo_label)
        elif cfg.segment == "endo":
            RV_endo = border.extract_isosurface(cfg.labels.RV_endo_label)
            LV_endo = border.extract_isosurface(cfg.labels.LV_endo_label)
            surface = SparseVolume.combine([RV_endo, LV_endo])
        elif cfg.segment == "epi":
            surface = border.extract_isosurface(cfg.labels.epi_label)

        elif cfg.segment == "base":
            surface = border.extract_isosurface(cfg.labels.base_label)
        else:
            raise ValueError(f"Unknown segment: {cfg.segment}. Choose from 'RV_endo', 'LV_endo', 'endo', 'epi', or 'base'.")
        
        mask = surface.check_point_membership(data.points)
        data = data.extract_points_based_on_mask(mask)
    
    if cfg.extract.use:
        if cfg.extract.extractor_path is None:
            raise ValueError("Selector path must be provided for extraction.")
        
        selector = SparseField.from_pickle(cfg.extract.extractor_path, device="cpu")
        points_of_interest = selector.extract_isosurface(cfg.extract.value, tol=cfg.extract.tolerance)
        mask = points_of_interest.check_point_membership(data.points)
        data = data.extract_points_based_on_mask(mask)

    
    if cfg.cut.use:
        points = data.points.numpy()
        if cfg.cut.along not in ["x", "y", "z"]:
            raise ValueError("Cut direction must be 'x', 'y', or 'z'.")
        if cfg.cut.along == "z":
            cut_threshold = cfg.cut.position if cfg.cut.position is not None else np.mean(points, axis=0)[0]
            mask = points[:, 0] < cut_threshold
        elif cfg.cut.along == "y":
            cut_threshold = cfg.cut.position if cfg.cut.position is not None else np.mean(points, axis=0)[1]
            mask = points[:, 1] < cut_threshold
        elif cfg.cut.along == "x":
            cut_threshold = cfg.cut.position if cfg.cut.position is not None else np.mean(points, axis=0)[2]
            mask = points[:, 2] < cut_threshold
        
        if cfg.cut.reverse:
            mask = ~mask
        data = data.extract_points_based_on_mask(mask)

    points = data.points.numpy()
    points = points[:, ::-1]
    cloud = pv.PolyData(points)
    
    plotter = pv.Plotter()

    #with open("/home/jtso3/shape_files/p03/sampling_01_points.pkl", "rb") as f:
    #    test_points = pkl.load(f)
    #
    #test_points = test_points.numpy()[:, ::-1]
#
    #test_cloud = pv.PolyData(test_points)
    #plotter.add_mesh(test_cloud, color="green", point_size=5, render_points_as_spheres=True)
#
    if cfg.files.fibers_path is not None:
        vectors = data.vectors.numpy()
        vectors = vectors[:,::-1]
        cloud.point_data["vectors"]= vectors
        plotter.add_mesh(cloud, opacity=0.2)

        if cfg.fibers.nb_points is not None:
            nb_points = min(cfg.fibers.nb_points, len(points))
            print(f"Showing {nb_points} points with associated fibers.")
            random_indices = np.random.choice(len(points), size=nb_points, replace=False)
            cloud = cloud.extract_points(random_indices, adjacent_cells=False)

        plotter.add_mesh(cloud.glyph(orient="vectors", scale="vectors", factor =cfg.fibers.fiber_scale), color= cfg.fibers.fiber_color)

    if cfg.files.coords_path is not None:
        values = data.values.numpy()
        cloud.point_data["coords"]= values

        if cfg.coords.aha:
            annotations = {i: f"{i}" for i in range(1,18)}
            plotter.add_mesh(cloud, scalars="coords", show_edges=False, opacity=cfg.coords.opacity, cmap="rainbow", annotations=annotations, scalar_bar_args={"title": "AHA segment", "n_labels": 0})
            for i in range(1, 18):
                segment = data.extract_isosurface(i, tol=0.05)
                center = segment.get_centroid()
                center = tuple(center.numpy()[::-1])
                text_mesh = pv.Text3D(f"{i}", center=center, height=10)
                plotter.add_mesh(text_mesh)
        else:
            if cfg.coords.nb_points is not None:
                nb_points = min(cfg.coords.nb_points, len(points))
                print(f"Showing {nb_points} points with associated coords.")
                random_indices = np.random.choice(len(points), size=nb_points, replace=False)
                cloud = cloud.extract_points(random_indices, adjacent_cells=False)

            plotter.add_mesh(cloud, scalars="coords", show_edges=False, opacity=cfg.coords.opacity, cmap=cfg.coords.cmap)
            

        if cfg.isolines.use:
            isoline_values = np.arange(cfg.isolines.start, cfg.isolines.end, cfg.isolines.step)
            for value in isoline_values:
                isoline = data.extract_isosurface(value, tol=cfg.isolines.tolerance)
                isoline_points = isoline.points.numpy()
                isoline_points = isoline_points[:, ::-1]
                isoline_cloud = pv.PolyData(isoline_points)
                plotter.add_mesh(isoline_cloud, color=cfg.isolines.color, line_width=cfg.isolines.line_width)


    plotter.show()
    

if __name__ == "__main__":
    run_visualization()