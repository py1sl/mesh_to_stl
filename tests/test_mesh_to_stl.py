import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from neutron_tools.mcnp.meshtal_analysis import read_meshtally_file
from vtkmodules.vtkIOGeometry import vtkSTLReader

from mesh_to_stl import contour_mesh, convert


def tally(number, energy=False):
    header = (
        f" Mesh Tally Number {number}\n"
        " X direction: 0 2 6\n"
        " Y direction: 0 4 10\n"
        " Z direction: 0 6 14\n"
    )
    if energy:
        header += " Energy bin boundaries: 0 1 2\n"
        header += " Energy         X         Y         Z     Result     Rel Error\n"
    else:
        header += " X         Y         Z     Result     Rel Error\n"
    rows = []
    for bin_value in ([1, 2] if energy else [None]):
        for x in (1, 4):
            for y in (2, 7):
                for z in (3, 10):
                    prefix = f"{bin_value} " if energy else ""
                    rows.append(f"{prefix}{x} {y} {z} {x} 0.1\n")
    return header + "".join(rows)


class MeshToStlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.input = Path(self.tmp.name) / "meshtal"
        self.output = Path(self.tmp.name) / "contour.stl"
        self.input.write_text(tally(14))

    def test_scaled_contour_and_stl_round_trip(self):
        surface = convert(self.input, self.output, 5, scale=2)
        self.assertGreater(surface.GetNumberOfPolys(), 0)
        np.testing.assert_allclose(surface.GetBounds(), (2.5, 2.5, 2, 7, 3, 10))
        reader = vtkSTLReader()
        reader.SetFileName(str(self.output))
        reader.Update()
        self.assertGreater(reader.GetOutput().GetNumberOfPolys(), 0)
        np.testing.assert_allclose(reader.GetOutput().GetBounds(), surface.GetBounds())

    def test_bins_and_tally_selection(self):
        self.input.write_text(tally(14) + "\n\n" + tally(24, energy=True) + "\n")
        with self.assertRaisesRegex(ValueError, "specify --tally"):
            convert(self.input, self.output, 2)
        with self.assertRaisesRegex(ValueError, "not found"):
            convert(self.input, self.output, 2, tally=99)
        with self.assertRaisesRegex(ValueError, "select a energy bin"):
            convert(self.input, self.output, 2, tally=24)
        surface = convert(self.input, self.output, 2, tally=24, energy=2)
        np.testing.assert_allclose(surface.GetBounds(), (2, 2, 2, 7, 3, 10))
        with self.assertRaisesRegex(ValueError, "not found"):
            convert(self.input, self.output, 2, tally=24, energy=3)

    def test_invalid_mesh_and_empty_contour(self):
        mesh = read_meshtally_file(str(self.input))[0]
        with self.assertRaisesRegex(ValueError, "no triangles"):
            contour_mesh(mesh, 100)
        with self.assertRaisesRegex(ValueError, "finite"):
            contour_mesh(mesh, 2, scale=float("nan"))
        with self.assertRaisesRegex(ValueError, "no time bins"):
            contour_mesh(mesh, 2, time=1)
        mesh.data = mesh.data.sample(frac=1, random_state=1)
        np.testing.assert_allclose(contour_mesh(mesh, 2).GetBounds(), (2, 2, 2, 7, 3, 10))
        mesh.data.iloc[0] = mesh.data.iloc[1]
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            contour_mesh(mesh, 2)
        mesh = read_meshtally_file(str(self.input))[0]
        mesh.data = mesh.data.iloc[:-1]
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            contour_mesh(mesh, 2)
        with self.assertRaisesRegex(ValueError, "different"):
            convert(self.input, self.input, 2)

    def test_time_bins(self):
        contents = tally(14, energy=True).replace("Energy", "Time")
        self.input.write_text(contents)
        with self.assertRaisesRegex(ValueError, "select a time bin"):
            convert(self.input, self.output, 2)
        surface = convert(self.input, self.output, 2, time=1)
        np.testing.assert_allclose(surface.GetBounds(), (2, 2, 2, 7, 3, 10))

    def test_cli(self):
        script = Path(__file__).resolve().parents[1] / "mesh_to_stl.py"
        result = subprocess.run(
            [sys.executable, str(script), str(self.input), str(self.output), "5",
             "--scale", "2"], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.output.is_file())
        result = subprocess.run(
            [sys.executable, str(script), str(self.input), str(self.output), "100"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no triangles", result.stderr)


if __name__ == "__main__":
    unittest.main()
