"""Extract an isosurface from a Cartesian MCNP mesh tally."""

import argparse
from pathlib import Path
import tempfile

import numpy as np
from neutron_tools.mcnp.meshtal_analysis import read_meshtally_file
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid
from vtkmodules.vtkFiltersCore import vtkContourFilter
from vtkmodules.vtkIOGeometry import vtkSTLWriter


def contour_mesh(mesh, level, scale=1.0, energy=None, time=None):
    """Return a VTK isosurface using one value per Cartesian voxel center."""
    if not np.isfinite(level) or not np.isfinite(scale):
        raise ValueError("contour level and scale must be finite")
    if energy is not None and time is not None:
        raise ValueError("select either an energy or a time bin, not both")

    data = mesh.data.dropna(how="all")
    for column, selected in (("Energy", energy), ("Time", time)):
        if selected is not None and column not in data:
            raise ValueError(f"this tally has no {column.lower()} bins")
        if column in data:
            bins = np.unique(data[column].to_numpy(dtype=float))
            if not np.isfinite(bins).all():
                raise ValueError(f"{column.lower()} bins must be finite")
            if selected is None:
                if len(bins) != 1:
                    raise ValueError(f"select a {column.lower()} bin")
                selected = bins[0]
            if not np.isfinite(selected):
                raise ValueError(f"{column.lower()} bin must be finite")
            matches = bins[np.isclose(bins, selected, rtol=1e-6, atol=0)]
            if len(matches) != 1:
                raise ValueError(f"{column.lower()} bin {selected} not found or ambiguous")
            data = data[data[column] == matches[0]]

    axes = []
    for name in ("x", "y", "z"):
        bounds = np.asarray(getattr(mesh, f"{name}_bounds"), dtype=float)
        coordinates = data[name].to_numpy(dtype=float)
        axis = np.unique(coordinates)
        if (len(bounds) < 3 or len(axis) != len(bounds) - 1
                or not np.isfinite(bounds).all() or not np.isfinite(axis).all()
                or np.any(np.diff(bounds) <= 0)
                or np.any(axis <= bounds[:-1]) or np.any(axis >= bounds[1:])):
            raise ValueError("a complete 3D Cartesian mesh with at least two bins per axis is required")
        axes.append(axis)

    values = data["value"].to_numpy(dtype=float) * scale
    if not np.isfinite(values).all():
        raise ValueError("mesh values after scaling must be finite")
    indices = [np.searchsorted(axis, data[name].to_numpy(dtype=float))
               for name, axis in zip(("x", "y", "z"), axes)]
    flat = np.ravel_multi_index(indices, tuple(map(len, axes)), order="F")
    if len(flat) != np.prod([len(axis) for axis in axes]) or len(np.unique(flat)) != len(flat):
        raise ValueError("mesh has missing or duplicate voxel values")
    ordered = np.empty(len(flat), dtype=float)
    ordered[flat] = values

    grid = vtkRectilinearGrid()
    grid.SetDimensions(*(len(axis) for axis in axes))
    for setter, axis in zip((grid.SetXCoordinates, grid.SetYCoordinates,
                             grid.SetZCoordinates), axes):
        setter(numpy_to_vtk(axis, deep=True))
    scalars = numpy_to_vtk(ordered, deep=True)
    scalars.SetName("value")
    grid.GetPointData().SetScalars(scalars)

    contour = vtkContourFilter()
    contour.SetInputData(grid)
    contour.SetValue(0, level)
    contour.Update()
    surface = contour.GetOutput()
    if surface.GetNumberOfPolys() == 0:
        raise ValueError("contour produced no triangles; check the level and mesh values")
    return surface


def read_mesh(input_path, tally=None):
    """Read one tally, isolating it because the upstream reader resets its type."""
    with open(input_path) as source:
        sections = []
        for line in source:
            if " Mesh Tally Number" in line:
                sections.append(int(line.split()[-1]))
        if tally is None:
            if len(sections) != 1:
                raise ValueError("file must contain one tally or specify --tally")
            tally = sections[0]
        if sections.count(tally) != 1:
            raise ValueError(f"tally {tally} not found or not unique")

        source.seek(0)
        with tempfile.TemporaryDirectory() as directory:
            selected_path = Path(directory) / "meshtal"
            selected = False
            with selected_path.open("w") as target:
                for line in source:
                    if " Mesh Tally Number" in line:
                        if selected:
                            break
                        selected = int(line.split()[-1]) == tally
                    if selected and line.strip():
                        target.write(line)
            return read_meshtally_file(str(selected_path))[0]


def convert(input_path, output_path, level, scale=1.0, tally=None, energy=None, time=None):
    """Read a meshtal file and write the requested isosurface to STL."""
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("input and output paths must be different")
    mesh = read_mesh(input_path, tally)
    surface = contour_mesh(mesh, level, scale, energy, time)
    writer = vtkSTLWriter()
    writer.SetFileName(str(output_path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(surface)
    if not writer.Write():
        raise OSError(f"could not write STL file: {output_path}")
    return surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="MCNP meshtal file")
    parser.add_argument("output", help="output STL file")
    parser.add_argument("level", type=float, help="contour value after scaling")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply tally values (default: 1)")
    parser.add_argument("--tally", type=int, help="mesh tally number")
    bins = parser.add_mutually_exclusive_group()
    bins.add_argument("--energy", type=float, help="energy bin value as printed in the meshtal")
    bins.add_argument("--time", type=float, help="time bin value as printed in the meshtal")
    args = parser.parse_args()
    try:
        convert(args.input, args.output, args.level, args.scale, args.tally,
                args.energy, args.time)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
