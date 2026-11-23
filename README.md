# mesh_to_stl

Extract a 3D isosurface from an MCNP meshtal file and export it as a binary
STL, using the `neutron_tools` meshtal reader and VTK.

## Installation

With Python 3.10 or later, install the dependencies in a virtual environment:

```sh
python -m pip install -r requirements.txt
```

The requirements install `neutron_tools` from a pinned GitHub source archive.

## Usage

```sh
python mesh_to_stl.py meshtal contour.stl 1e-5 --scale 2.0 --tally 14
```

The positional value is the contour level **after** multiplying tally values
by `--scale` (default 1). Omit `--tally` if the file contains only one mesh.
For a tally with multiple energy or time bins, select one with `--energy` or
`--time`, using the numeric bin value printed in the file (for example
`--energy 1e36` for a total energy bin). Run `python mesh_to_stl.py --help`
for all options.

Only complete Cartesian meshes with at least two voxels along each axis are
supported; nonuniform spacing is supported. Values are interpolated between
voxel centers, so the contour is limited to the volume between the outermost
centers. A contour intersecting that volume's boundary is open: the STL is
not automatically capped or guaranteed watertight. An empty contour is an error.
Coordinate units are those of the input mesh (typically cm); STL has no units.
The reader's own supported meshtal formats and bin handling apply.
