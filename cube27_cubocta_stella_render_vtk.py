#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cube27_cubocta_stella_render_vtk.py

Version GPU / VTK du renderer cuboctaedre <-> cube ternaire 27 <->
stella octangula resolue.

Objectif principal
------------------
Rendre la manipulation manuelle beaucoup plus fluide que le backend
Matplotlib initial. La geometrie des spheres n'est jamais reconstruite
pendant le deplacement du parametre: deux ensembles de glyphes GPU sont
crees une seule fois (E13 et O14), puis seules leurs echelles sont mises
a jour.

Parametrisation exacte conservee
--------------------------------
lambda = -1 : E13 = L0 U L2, 13 spheres, rayon 1
lambda =  0 : G = sqrt(2) * {-1,0,1}^3, 27 spheres, rayon 1/sqrt(2)
lambda = +1 : O14 = L1 U L3, 14 spheres, rayon 1

Le document de reference utilise a=1/sqrt(2) comme pas des centres et
un diametre initial a. Ici toutes les longueurs sont homothetiees x2,
comme dans la premiere version, afin que le rayon du cube 27 a lambda=0
soit exactement 1/sqrt(2).

Le chemin de rayons est la version x2 du domaine exact de non-recouvrement:
    R_E <= 1, R_O <= 1, R_E + R_O <= sqrt(2).

Moteur
------
- VTK / OpenGL
- vtkGlyph3DMapper pour instancier les 13 + 14 spheres sur GPU
- aucune recreation de mesh lors du drag du slider
- acteurs de lignes preconstruits, seule l'opacite varie
- interaction TrackballCamera native VTK

Commandes
---------
Souris gauche       rotation libre de la camera
Molette             zoom
Souris milieu       pan
Slider              reglage continu de lambda
Fleche gauche/droite pas manuel
Home                lambda=-1, cuboctaedre E13
0                   lambda=0, cube 27
End                 lambda=+1, stella O14
Espace               lecture/pause de l'oscillation
R                   reinitialise la camera
E                   active/desactive les graphes d'aretes
C                   active/desactive le cube de reference
S                   capture PNG instantanee
Q / Escape           quitter

Dependances
-----------
    pip install numpy vtk
Optionnel pour export anime:
    pip install imageio imageio-ffmpeg pillow
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import vtk
except ImportError as exc:  # pragma: no cover - message utilisateur
    raise SystemExit(
        "VTK n'est pas installe. Installez-le avec: pip install vtk\n"
        "Le renderer Matplotlib historique reste utilisable separement."
    ) from exc


SQRT2 = math.sqrt(2.0)
R_MID = 1.0 / SQRT2
R_END = 1.0
STEP = SQRT2
DEFAULT_LIMIT = 3.05


@dataclass(frozen=True)
class Node:
    ternary: tuple[int, int, int]
    xyz: np.ndarray
    support: int
    parity: int  # 0 -> E13, 1 -> O14


def build_nodes() -> list[Node]:
    out: list[Node] = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                support = int(i != 0) + int(j != 0) + int(k != 0)
                out.append(
                    Node(
                        ternary=(i, j, k),
                        xyz=STEP * np.array((i, j, k), dtype=float),
                        support=support,
                        parity=support & 1,
                    )
                )
    return out


NODES = build_nodes()
E_NODES = [n for n in NODES if n.parity == 0]
O_NODES = [n for n in NODES if n.parity == 1]


def radii_from_lambda(lam: float) -> tuple[float, float]:
    """Chemin piecewise-lineaire exact dans le domaine de non-recouvrement."""
    lam = float(np.clip(lam, -1.0, 1.0))
    if lam <= 0.0:
        s = lam + 1.0
        r_e = (1.0 - s) * R_END + s * R_MID
        r_o = s * R_MID
    else:
        s = lam
        r_e = (1.0 - s) * R_MID
        r_o = (1.0 - s) * R_MID + s * R_END
    return r_e, r_o


def state_name(lam: float) -> str:
    if abs(lam + 1.0) < 1e-8:
        return "E13 — centre + cuboctaedre (13 boules)"
    if abs(lam) < 1e-8:
        return "G — cube ternaire 27"
    if abs(lam - 1.0) < 1e-8:
        return "O14 — stella octangula resolue (14 boules)"
    if lam < 0.0:
        return "transition E13 -> cube 27"
    return "transition cube 27 -> O14"


def ternary_hasse_edges() -> list[tuple[np.ndarray, np.ndarray]]:
    by_u = {n.ternary: n for n in NODES}
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for n in NODES:
        u = list(n.ternary)
        for axis in range(3):
            if u[axis] < 1:
                v = list(u)
                v[axis] += 1
                edges.append((n.xyz, by_u[tuple(v)].xyz))
    return edges


def internal_contact_edges(nodes: list[Node], target_distance: float = 2.0):
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if abs(np.linalg.norm(nodes[i].xyz - nodes[j].xyz) - target_distance) < 1e-10:
                edges.append((nodes[i].xyz, nodes[j].xyz))
    return edges


CUBE_EDGES = ternary_hasse_edges()
CUBO_EDGES = internal_contact_edges(E_NODES)
STELLA_EDGES = internal_contact_edges(O_NODES)

assert len(NODES) == 27
assert len(E_NODES) == 13
assert len(O_NODES) == 14
assert len(CUBE_EDGES) == 54
assert len(CUBO_EDGES) == 36
assert len(STELLA_EDGES) == 36
assert np.allclose(radii_from_lambda(-1.0), (1.0, 0.0))
assert np.allclose(radii_from_lambda(0.0), (R_MID, R_MID))
assert np.allclose(radii_from_lambda(+1.0), (0.0, 1.0))


def polydata_for_nodes(nodes: list[Node], scalar_name: str = "radius"):
    points = vtk.vtkPoints()
    scales = vtk.vtkFloatArray()
    scales.SetName(scalar_name)
    scales.SetNumberOfComponents(1)
    scales.SetNumberOfTuples(len(nodes))

    for idx, n in enumerate(nodes):
        points.InsertNextPoint(float(n.xyz[0]), float(n.xyz[1]), float(n.xyz[2]))
        scales.SetValue(idx, 1.0)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.GetPointData().AddArray(scales)
    poly.GetPointData().SetActiveScalars(scalar_name)
    return poly, scales


def make_glyph_actor(
    nodes: list[Node],
    sphere_source: "vtk.vtkSphereSource",
    color: tuple[float, float, float],
):
    poly, scales = polydata_for_nodes(nodes)

    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(poly)
    mapper.SetSourceConnection(sphere_source.GetOutputPort())
    mapper.ScalingOn()
    mapper.SetScaleArray("radius")
    mapper.SetScaleModeToScaleByMagnitude()
    mapper.SetScaleFactor(1.0)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetInterpolationToPhong()
    prop.SetAmbient(0.12)
    prop.SetDiffuse(0.76)
    prop.SetSpecular(0.34)
    prop.SetSpecularPower(36.0)
    return actor, poly, scales, mapper


def make_line_actor(
    edges: Iterable[tuple[np.ndarray, np.ndarray]],
    color=(0.16, 0.16, 0.18),
    width=1.25,
):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for p, q in edges:
        i0 = points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        i1 = points.InsertNextPoint(float(q[0]), float(q[1]), float(q[2]))
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i0)
        line.GetPointIds().SetId(1, i1)
        lines.InsertNextCell(line)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(width)
    actor.GetProperty().SetOpacity(0.0)
    return actor


def make_reference_cube_actor():
    cube = vtk.vtkCubeSource()
    cube.SetCenter(0.0, 0.0, 0.0)
    cube.SetXLength(2.0 * STEP)
    cube.SetYLength(2.0 * STEP)
    cube.SetZLength(2.0 * STEP)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cube.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetRepresentationToWireframe()
    prop.SetColor(0.24, 0.24, 0.28)
    prop.SetLineWidth(1.0)
    prop.SetOpacity(0.22)
    return actor


def make_axes_actor():
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(1.0, 1.0, 1.0)
    axes.SetShaftTypeToCylinder()
    axes.SetCylinderRadius(0.015)
    axes.AxisLabelsOff()
    return axes


def smoothstep_scalar(x: float) -> float:
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def cycle_lambda(phase: float) -> float:
    """phase dans [0,4): -1 -> 0 -> +1 -> 0 -> -1."""
    t = phase % 4.0
    seg = int(math.floor(t))
    x = smoothstep_scalar(t - seg)
    if seg == 0:
        return -1.0 + x
    if seg == 1:
        return x
    if seg == 2:
        return 1.0 - x
    return -x


QUALITY = {
    "low": (14, 20),
    "medium": (22, 32),
    "high": (32, 48),
    "ultra": (48, 72),
}


class Cube27VTKApp:
    def __init__(self, args, offscreen: bool = False):
        self.args = args
        self.offscreen = offscreen
        self.lam = float(np.clip(args.lambda_value, -1.0, 1.0))
        self.playing = False
        self.play_direction = 1.0
        self.edges_enabled = not args.no_edges
        self.cube_enabled = not args.no_cube
        self.last_tick = time.perf_counter()
        self.snapshot_counter = 0

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.965, 0.965, 0.975)
        self.renderer.SetBackground2(0.80, 0.84, 0.90)
        self.renderer.GradientBackgroundOn()

        self.window = vtk.vtkRenderWindow()
        self.window.SetWindowName("Cube 27 — Cuboctaedre <-> Stella Octangula — VTK/OpenGL")
        self.window.AddRenderer(self.renderer)
        self.window.SetSize(args.width, args.height)
        self.window.SetMultiSamples(args.msaa)
        if offscreen:
            self.window.SetOffScreenRendering(1)

        self.interactor = None
        self.slider_widget = None
        if not offscreen:
            self.interactor = vtk.vtkRenderWindowInteractor()
            self.interactor.SetRenderWindow(self.window)
            self.interactor.SetDesiredUpdateRate(float(args.target_fps))
            style = vtk.vtkInteractorStyleTrackballCamera()
            self.interactor.SetInteractorStyle(style)

        theta_res, phi_res = QUALITY[args.quality]
        self.sphere_source = vtk.vtkSphereSource()
        self.sphere_source.SetRadius(1.0)
        self.sphere_source.SetThetaResolution(theta_res)
        self.sphere_source.SetPhiResolution(phi_res)
        self.sphere_source.LatLongTessellationOff()

        self.e_actor, self.e_poly, self.e_scales, self.e_mapper = make_glyph_actor(
            E_NODES, self.sphere_source, (0.16, 0.53, 0.88)
        )
        self.o_actor, self.o_poly, self.o_scales, self.o_mapper = make_glyph_actor(
            O_NODES, self.sphere_source, (0.95, 0.47, 0.13)
        )
        self.renderer.AddActor(self.e_actor)
        self.renderer.AddActor(self.o_actor)

        self.cube_edges_actor = make_line_actor(CUBE_EDGES, width=1.0)
        self.cubo_edges_actor = make_line_actor(CUBO_EDGES, width=1.55)
        self.stella_edges_actor = make_line_actor(STELLA_EDGES, width=1.55)
        self.renderer.AddActor(self.cube_edges_actor)
        self.renderer.AddActor(self.cubo_edges_actor)
        self.renderer.AddActor(self.stella_edges_actor)

        self.reference_cube = make_reference_cube_actor()
        self.renderer.AddActor(self.reference_cube)

        if args.axes:
            self.axes_actor = make_axes_actor()
            self.renderer.AddActor(self.axes_actor)
        else:
            self.axes_actor = None

        self.text = vtk.vtkTextActor()
        self.text.SetPosition(18, args.height - 78)
        text_prop = self.text.GetTextProperty()
        text_prop.SetFontFamilyToArial()
        text_prop.SetFontSize(20)
        text_prop.SetColor(0.08, 0.08, 0.10)
        text_prop.SetBold(1)
        self.renderer.AddViewProp(self.text)

        self.help_text = vtk.vtkTextActor()
        self.help_text.SetInput(
            "drag slider | souris: rotation/zoom/pan | <- -> | Home / 0 / End | "
            "Space: auto | E: edges | C: cube | S: PNG | R: camera"
        )
        self.help_text.SetPosition(18, 18)
        hp = self.help_text.GetTextProperty()
        hp.SetFontFamilyToArial()
        hp.SetFontSize(13)
        hp.SetColor(0.18, 0.18, 0.20)
        self.renderer.AddViewProp(self.help_text)

        lightkit = vtk.vtkLightKit()
        lightkit.SetKeyLightIntensity(0.78)
        lightkit.SetKeyToFillRatio(3.0)
        lightkit.SetKeyToHeadRatio(2.2)
        lightkit.SetKeyLightWarmth(0.55)
        lightkit.SetFillLightWarmth(0.45)
        lightkit.AddLightsToRenderer(self.renderer)

        self.set_camera(args.elev, args.azim, args.distance)
        self.update_state(self.lam, render=False)

        if not offscreen:
            self._build_slider()
            self.interactor.AddObserver("KeyPressEvent", self.on_key)
            self.interactor.AddObserver("TimerEvent", self.on_timer)

    def set_camera(self, elev: float, azim: float, distance: float):
        er = math.radians(elev)
        ar = math.radians(azim)
        x = distance * math.cos(er) * math.cos(ar)
        y = distance * math.cos(er) * math.sin(ar)
        z = distance * math.sin(er)
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(x, y, z)
        cam.SetFocalPoint(0.0, 0.0, 0.0)
        cam.SetViewUp(0.0, 0.0, 1.0)
        cam.SetClippingRange(0.1, 100.0)

    def _build_slider(self):
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(-1.0)
        rep.SetMaximumValue(1.0)
        rep.SetValue(self.lam)
        rep.SetTitleText("lambda")
        rep.SetLabelFormat("%+.3f")
        rep.SetSliderLength(0.025)
        rep.SetSliderWidth(0.030)
        rep.SetTubeWidth(0.006)
        rep.SetEndCapLength(0.012)
        rep.SetEndCapWidth(0.020)
        rep.SetTitleHeight(0.020)
        rep.SetLabelHeight(0.018)

        rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint1Coordinate().SetValue(0.16, 0.095)
        rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint2Coordinate().SetValue(0.84, 0.095)

        widget = vtk.vtkSliderWidget()
        widget.SetInteractor(self.interactor)
        widget.SetRepresentation(rep)
        widget.SetAnimationModeToAnimate()
        widget.EnabledOn()
        widget.AddObserver("InteractionEvent", self.on_slider)
        widget.AddObserver("EndInteractionEvent", self.on_slider)
        self.slider_widget = widget

    def set_scale_array(self, array: "vtk.vtkFloatArray", value: float):
        for i in range(array.GetNumberOfTuples()):
            array.SetValue(i, float(value))
        array.Modified()

    def update_state(self, lam: float, render: bool = True):
        self.lam = float(np.clip(lam, -1.0, 1.0))
        r_e, r_o = radii_from_lambda(self.lam)

        # Mise a jour O(27) de scalaires uniquement; le maillage sphere reste resident.
        self.set_scale_array(self.e_scales, max(r_e, 1e-6))
        self.set_scale_array(self.o_scales, max(r_o, 1e-6))
        self.e_poly.Modified()
        self.o_poly.Modified()
        self.e_actor.SetVisibility(r_e > 2e-5)
        self.o_actor.SetVisibility(r_o > 2e-5)

        if self.edges_enabled:
            a_mid = max(0.0, 1.0 - abs(self.lam))
            a_cubo = max(0.0, -self.lam)
            a_stella = max(0.0, self.lam)
            self.cube_edges_actor.GetProperty().SetOpacity(0.26 * a_mid)
            self.cubo_edges_actor.GetProperty().SetOpacity(0.42 * a_cubo)
            self.stella_edges_actor.GetProperty().SetOpacity(0.42 * a_stella)
        else:
            self.cube_edges_actor.GetProperty().SetOpacity(0.0)
            self.cubo_edges_actor.GetProperty().SetOpacity(0.0)
            self.stella_edges_actor.GetProperty().SetOpacity(0.0)

        self.reference_cube.SetVisibility(self.cube_enabled)
        self.text.SetInput(
            f"{state_name(self.lam)}\n"
            f"lambda = {self.lam:+.4f}    R_E = {r_e:.6f}    R_O = {r_o:.6f}"
        )

        if self.slider_widget is not None:
            self.slider_widget.GetRepresentation().SetValue(self.lam)

        if render:
            self.window.Render()

    def on_slider(self, obj, event):
        rep = obj.GetRepresentation()
        self.update_state(rep.GetValue())

    def on_key(self, obj, event):
        key = self.interactor.GetKeySym()
        key_low = key.lower() if key else ""

        if key in ("Left", "KP_Left"):
            self.update_state(self.lam - self.args.step)
        elif key in ("Right", "KP_Right"):
            self.update_state(self.lam + self.args.step)
        elif key == "Home":
            self.update_state(-1.0)
        elif key == "End":
            self.update_state(1.0)
        elif key in ("0", "KP_0", "Insert"):
            self.update_state(0.0)
        elif key_low == "space":
            self.playing = not self.playing
            self.last_tick = time.perf_counter()
        elif key_low == "e":
            self.edges_enabled = not self.edges_enabled
            self.update_state(self.lam)
        elif key_low == "c":
            self.cube_enabled = not self.cube_enabled
            self.update_state(self.lam)
        elif key_low == "r":
            self.set_camera(self.args.elev, self.args.azim, self.args.distance)
            self.window.Render()
        elif key_low == "s":
            self.snapshot_counter += 1
            path = Path(f"cube27_snapshot_{self.snapshot_counter:03d}.png")
            self.write_png(path)
            print(f"Capture: {path.resolve()}")
        elif key in ("Escape",) or key_low == "q":
            self.interactor.TerminateApp()

    def on_timer(self, obj, event):
        if not self.playing:
            return
        now = time.perf_counter()
        dt = min(now - self.last_tick, 0.1)
        self.last_tick = now

        x = self.lam + self.play_direction * self.args.auto_speed * dt
        if x >= 1.0:
            x = 1.0
            self.play_direction = -1.0
        elif x <= -1.0:
            x = -1.0
            self.play_direction = 1.0
        self.update_state(x)

    def write_png(self, path: Path):
        self.window.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.window)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()

    def start(self):
        if self.offscreen:
            return
        self.window.Render()
        self.interactor.Initialize()
        interval_ms = max(1, int(round(1000.0 / self.args.target_fps)))
        self.interactor.CreateRepeatingTimer(interval_ms)
        self.interactor.Start()


def export_animation(args, path: Path):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit("Pour --save: pip install imageio imageio-ffmpeg pillow") from exc

    app = Cube27VTKApp(args, offscreen=True)
    total = max(8, int(args.frames))
    frames = []

    # WindowToImageFilter reemploye sur toute la sequence.
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(app.window)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()

    from vtk.util.numpy_support import vtk_to_numpy

    for idx in range(total):
        phase = 4.0 * idx / total
        lam = cycle_lambda(phase)
        app.update_state(lam, render=False)

        if args.rotate_camera:
            app.set_camera(args.elev, args.azim + 360.0 * idx / total, args.distance)

        app.window.Render()
        w2i.Modified()
        w2i.Update()
        vtk_img = w2i.GetOutput()
        width, height, _ = vtk_img.GetDimensions()
        arr = vtk_to_numpy(vtk_img.GetPointData().GetScalars())
        arr = arr.reshape(height, width, -1)
        arr = np.flipud(arr)
        frames.append(arr.copy())

        if idx % max(1, total // 10) == 0:
            print(f"render {idx + 1}/{total}")

    suffix = path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(path, frames, fps=args.fps, loop=0)
    elif suffix == ".mp4":
        imageio.mimsave(
            path,
            frames,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
    else:
        raise SystemExit("--save attend un .gif ou .mp4")
    print(f"Animation ecrite: {path}")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Renderer VTK/OpenGL fluide pour E13 cuboctaedrique <-> cube 27 <-> "
            "O14 stella octangula, rayon median 1/sqrt(2)."
        )
    )
    p.add_argument("--lambda-value", type=float, default=-1.0)
    p.add_argument("--quality", choices=tuple(QUALITY), default="medium")
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--msaa", type=int, default=4, choices=(0, 2, 4, 8))
    p.add_argument("--target-fps", type=int, default=60)
    p.add_argument("--step", type=float, default=0.025)
    p.add_argument("--auto-speed", type=float, default=0.42, help="unites lambda/seconde")
    p.add_argument("--elev", type=float, default=23.0)
    p.add_argument("--azim", type=float, default=34.0)
    p.add_argument("--distance", type=float, default=10.5)
    p.add_argument("--axes", action="store_true")
    p.add_argument("--no-edges", action="store_true")
    p.add_argument("--no-cube", action="store_true")
    p.add_argument("--still", type=Path, help="capture PNG via rendu VTK offscreen")
    p.add_argument("--save", type=Path, help="export GIF ou MP4 via rendu VTK offscreen")
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--rotate-camera", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.still:
        if args.still.suffix.lower() != ".png":
            raise SystemExit("--still utilise actuellement le format PNG.")
        app = Cube27VTKApp(args, offscreen=True)
        app.update_state(args.lambda_value, render=False)
        app.write_png(args.still)
        print(f"Image ecrite: {args.still}")
        return

    if args.save:
        export_animation(args, args.save)
        return

    app = Cube27VTKApp(args, offscreen=False)
    print("Renderer VTK/OpenGL initialise.")
    print("  Slider ou <- -> : lambda")
    print("  Home / 0 / End : E13 / cube27 / O14")
    print("  Space          : oscillation auto")
    print("  Souris         : camera native VTK")
    print("  E / C          : aretes / cube de reference")
    print("  S              : capture PNG")
    print("  Q / Esc        : quitter")
    app.start()


if __name__ == "__main__":
    main()
