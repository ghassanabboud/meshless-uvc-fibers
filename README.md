# Universal Ventricular Coordinates and Myocardial Fiber Orientation on Meshless Grids Using Rule-Based Methods

<a href="https://Hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

<p align="center">
    <img src="imgs/lv_apico_fibers.gif" width="90%" style="display:inline-block; vertical-align:top;">
</p>

## Goal 

This project sought to develop an open-source Python implementation of Universal Ventricular Coordinates (UVCs) and myocardial fiber assignment on ventricular grids using Rule-Based Methods (RBMs). Such fibers are essential for simulating cardiac electrophysiology, as they define the anisotropic conduction properties of the heart tissue. UVCs are also used to position landmarks across different hearts and study shape variations across cardiac cohorts. Conventional methods create tetrahedral meshes from segmented cardiac images and then use specialized Laplace solvers to generate UVCs and fibers. Generating such meshes can be time-consuming and distort the original segmented heart's shape. However, such methods are convenient for those using Finite-Element Methods (FEM) to simulate cardiac electrophysiology, as FEM requires such a mesh.

With the introduction of Lattice-Boltzmann methods (LBM) for electrophysiological simulations, there is no more need for meshes: all calculations can be performed on the uniform grid of points generated through segmentation. Thus, there is a need for RBMs that calculate fiber orientation directly on grids without generating meshes.

## Past Works

[Bayer et al.](https://pubmed.ncbi.nlm.nih.gov/22648575/) introduced in 2012 the first rule-based method using solutions to the Laplace equation with different boundary conditions. This Laplace-Dirichlet Rule-Based Method (LDRBM) set the foundations of all subsequent algorithms, notably through the definition of the apicobasal and the transmural directions. It guaranteed a smooth transition in fiber orientation between the ventricles using bi-directional spherical linear interpolation (bislerp). In a later work, [Bayer et al. (2018)](https://pubmed.ncbi.nlm.nih.gov/29414438/) formalized the definition of universal ventricular coordinates (UVCs), both for transferring data between different geometries and for generating fiber orientations.

 [Doste et al. (2019)](http://pubmed.ncbi.nlm.nih.gov/30721579/) treated the left and right ventricles (LV, RV) separately. This new method allowed both specialized angles for the different ventricles and more accurate fiber orientation around the right ventricular outflow tract. [Piersanti et al. (2020)](https://www.sciencedirect.com/science/article/pii/S0045782520306538) compared both methods and showed that different alpha angles between the LV and RV are essential for accurate fiber assignment. [Piersanti et al. (2020)](https://www.sciencedirect.com/science/article/pii/S0045782520306538) also introduced a new RBM for the generation of fibers in the atria. 
 
 [Gillette et al. (2020)](https://www.sciencedirect.com/science/article/pii/S1361841521001262?via%3Dihub#sec0038) introduced the use of Bi-eikonal Normalization (BN) fields to replace the Laplace fields, underlining that those are more geometrically robust and are a better measure of normalized distance. [Biasi et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0010482524016147?via%3Dihub) applied the RBM on a meshless voxel-based grid using BN fields, pairing their RBM with a meshless electrophysiology solver. Biasi ensured a smooth transition across the septum not by using bislerp but by introducing a transventricular coordinate that controls the change of the alpha angle across the septum. Finally, [Schuler et al. (2021)](https://www.sciencedirect.com/science/article/pii/S1361841521002929) improved upon the definition of UVCs with better consistency and lower transfer errors through their open-source tool [CobivecoX](https://github.com/KIT-IBT/Cobiveco).


## Current implementation


The current implementation has a use case similar to that of Biasi et al.: it operates on meshless grids and defines coordinates as BN fields. However, BN fields are by definition discontinuous at Voronoi boundaries, which leads to the formation of cusps in the isolines of the coordinates as well as discontinuities in the fiber orientation. This pipeline thus includes a post-processing step that smoothes the bi-eikonal fields by solving a diffusion PDE on the grid. Apart from the use of meshless BN fields, this implementation sticks more closely to the RBM of Doste et al., separating the ventricles and ensuring accurate fibers around the right ventricular outflow tract (RVOT) and interventricular septum. This implementation also draws inspiration from [CobivecoX](https://github.com/KIT-IBT/Cobiveco) for the definition of the transventricular and rotational coordinates.

The current implementation offers 5 scripts for UVCs and fiber generation on 5 different geometries:
- `run_LV.py`: for the left ventricle (LV) with one valve rim joining the mitral and aortic valves (or a flat base)
- `run_RV_no_OT.py`: for the right ventricle (RV) with one valve rim joining the tricuspid and pulmonary valves (or a flat base)
- `run_RVOT.py`: for the right ventricle (RV) with two valve rims, tricuspid and pulmonary.
- `run_biventricular_no_RVOT.py`: for a biventricular grid with a flat base.
- `run_biventricular_RVOT.py`: for a biventricular grid with one valve rim on the left ventricle and two valve rims on the right ventricle.

We do not provide scripts for left ventricular grids with two valve rims as [Piersanti et al. (2020)](https://www.sciencedirect.com/science/article/pii/S0045782520306538) showed that applying the Doste method on the left ventricle leads to minimal differences in fiber orientation because of the proximity of the mitral and aortic valves.

To optimize memory usage, this implementation relies on the representation of sparse 3D arrays using custom classes that store only non-zero values and their coordinates. Kindly refer to the documentation of the [`SparseVolume` class](Sparse/SparseVolume.py) for more implementation details. Note that some parts of our method rely on the chirality of the biventricular grid. This means it will yield unpredictable results on grids where the left and right sides are flipped (dextrocardia).


## Installation

This repo was developed on WSL2 (Windows Subsystem for Linux) using an Ubuntu distribution. It has not been tested on macOS or Windows but should work on pure Linux systems.

### Downloading 3DSlicer

All scripts support optional manual selection of the apex. These markings can be generated as .fcsv files using 3DSlicer. Here are installation steps for those unfamiliar with the program.

Installing requirements:
```bash
sudo apt-get install libglu1-mesa libpulse-mainloop-glib0 libnss3 libasound2 qt5dxcb-plugin
```

Download the latest version of 3DSlicer from the [official website](https://download.slicer.org/). The version used in this project is 5.8.0. Download manually or using the following command:
```bash
wget https://download.slicer.org/bitstream/679325961357655fd585ffb5
```

Extract the downloaded file:
```bash
tar -xvzf Slicer-4.11.20210226-linux-amd64.tar.gz
```

Add the Slicer executable to the PATH variable to be able to run it anywhere from the terminal:
```bash
cd Slicer-[version]-linux-amd64
pwd
```
Copy the output of the `pwd` command and add the following line to the `.bashrc` file:
```bash
export PATH=$PATH:[output of pwd command]
```

### Installing Python Dependencies

Specific libraries are required to run this pipeline. To create a virtual environment and install the required packages, run these instructions in the repo's root directory using Anaconda/Miniconda as package manager:

```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate fibers_mha
```


## Usage

This repository provides 7 main scripts: 5 generate UVCs and fibers as mentioned above, `define_AHA_segments.py` provides an additional functionality to divide the LV into a 17-segment model standardized by the [American Heart Association](https://www.ahajournals.org/doi/pdf/10.1161/hc0402.102975) and widely used in cardiac research, and  `run_visualization.py` helps visualize both vector (fiber, sheet direction, etc.) and scalar (apicobasal coordinates, transmural coordinates, AHA segments, etc.) fields using [PyVista](https://docs.pyvista.org/).

Configuration for all scripts uses Hydra. Configuration files can be overridden using the command line or by editing the config files in the `configs` folder. Certain configs are organized in groups (angles, labels). One can thus easily try running the pipeline with different angles while keeping the rest of the configuration fixed. One can also change the output directory from the default using `hydra.run.dir`. For more information on how to use Hydra, refer to the [documentation](https://hydra.cc/docs/intro/).

For a detailed description of the parameters, use the command line help option:
```bash
python run_LV.py --help
```

Here is an example run of the pipeline specifying an output directory, saving all coordinates and saving a .h5 file to use in a subsequent LBM pipeline:
```bash
python run_LV.py files.volume_path=/path/to/volume_file.mha files.border_path=/path/to/border_file.mha Hydra.run.dir=./output_directory/patient01 prepare_h5=true
```


## Limitations


- [Schuler et al. (2021)](https://www.sciencedirect.com/science/article/pii/S1361841521002929) criticized the use of bi-eikonal fields as universal coordinates because of their lack of bijectivity and the presence of cusps in the associated contour lines. They also highlighted the limitations of Laplace Dirichlet fields, whose formulation creates distortions for geometries with uneven wall thickness. 
- While the configuration supports a `device` parameter, only CPU computation is supported for the time being because of the lack of a GPU tool for bi-eikonal calculations. This might change in the future.



