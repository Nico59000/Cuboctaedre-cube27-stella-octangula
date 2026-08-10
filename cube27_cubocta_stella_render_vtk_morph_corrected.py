#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cube27_cubocta_stella_render_vtk_morph.py

Renderer VTK/OpenGL fluide pour l'alternance :
    cuboctaedre de boules (E13)
    <-> cube ternaire 27
    <-> stella octangula resolue (O14)

NOUVEAUTE
---------
Cette version CORRIGE le morphing geometrique des centres L2 sur la branche
vers la stella. Le mode par defaut FIXED supprime la contraction vers le
centre. Les modes SAFE et RAW conservent l'homothetie comme objet d'etude.

Convention geometrique
----------------------
On reprend la meme homothetie x2 que dans les versions precedentes :
- au milieu (lambda = 0), les 27 centres sont ceux du cube ternaire
  G = sqrt(2) * {-1,0,1}^3
- le rayon des boules au milieu est exactement 1/sqrt(2)

Les rayons restent :
    lambda = -1 : R_E = 1,         R_O = 0
    lambda =  0 : R_E = R_O = 1/sqrt(2)
    lambda = +1 : R_E = 0,         R_O = 1

Morphing des centres (choix de rendu)
-------------------------------------
Le document fixe les ensembles exacts E13 et O14, mais pas une interpolation
cinematique entre eux. Le present script ajoute donc un moteur d'interpolation
geometrique cohérent avec ces structures :

Branche lambda < 0  (vers le regime cuboctaedrique)
    - Les 13 centres de E13 restent a leurs positions exactes.
    - Les 6 centres de faces L1 de O14 restent sur les axes ; ils coïncident
      deja avec les centres des 6 faces carrees du cuboctaedre.
    - Les 8 sommets du cube L3 glissent le long des diagonales du cube vers
      les centres des 8 faces triangulaires du cuboctaedre, c.-a-d. vers
          (2/3) * sommet

Branche lambda > 0  (vers le regime stella octangula)
    - Les 14 centres de O14 restent a leurs positions exactes.
    - Le centre de E13 reste en 0.
    - Les 12 centres de type L2 glissent vers les milieux des 12 aretes du
      l'octaedre central O0, c.-a-d. vers
          (1/2) * centre_L2

Ce morphing laisse lambda = 0 EXACT, tout en fournissant une deformation
continue, symetrique et interactive des positions.

Commandes
---------
Souris gauche        rotation libre
Molette              zoom
Souris milieu        pan
Slider               reglage continu de lambda
Fleche gauche/droite pas manuel
Home                 lambda = -1
0                    lambda = 0
End                  lambda = +1
Espace               lecture/pause
M                    active/desactive le morphing des centres
E                    active/desactive les graphes d'aretes
C                    active/desactive le cube de reference
S                    capture PNG
R                    reinitialise la camera
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
except ImportError as exc:
    raise SystemExit(
        "VTK n'est pas installe. Installez-le avec: pip install vtk"
    ) from exc


SQRT2 = math.sqrt(2.0)
PHI = (1.0 + math.sqrt(5.0)) / 2.0
ALPHA_PHI = 1.0 / PHI          # phi^-1
BETA_PHI = 1.0 / (PHI * PHI)   # phi^-2 = ALPHA_PHI^2 = 1 - ALPHA_PHI

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
NODE_BY_KEY = {n.ternary: n for n in NODES}
E_NODES = [n for n in NODES if n.parity == 0]
O_NODES = [n for n in NODES if n.parity == 1]
E_INDEX = {n.ternary: i for i, n in enumerate(E_NODES)}
O_INDEX = {n.ternary: i for i, n in enumerate(O_NODES)}


def radii_from_lambda(lam: float) -> tuple[float, float]:
    """Chemin exact des rayons dans le domaine de non-recouvrement homothetie x2."""
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


def smoothstep_scalar(x: float) -> float:
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


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


def left_target_position(node: Node) -> np.ndarray:
    """
    Cible de la branche cuboctaedrique (lambda -> -1).
    - E13 fixe
    - L1 fixe (centres des faces carrees du cuboctaedre)
    - L3 -> centres des faces triangulaires du cuboctaedre = (2/3) * sommet
    """
    if node.parity == 0:
        return node.xyz.copy()
    if node.support == 1:
        return node.xyz.copy()
    if node.support == 3:
        return (2.0 / 3.0) * node.xyz
    return node.xyz.copy()


def raw_right_homothety_scale(lam: float) -> float:
    """
    Facteur de l'ancienne homothetie centrale explicite sur L2.

        q_raw(lambda) = 1 - smoothstep(lambda)/2

    Il descend de 1 a 1/2. Ce chemin est conserve uniquement comme
    diagnostic; il devient non admissible au regard du non-recouvrement
    sur une partie de la branche droite.
    """
    lam = float(np.clip(lam, 0.0, 1.0))
    return 1.0 - 0.5 * smoothstep_scalar(lam)


def right_radius_sum(lam: float) -> float:
    """R_E + R_O sur la branche lambda >= 0."""
    lam = float(np.clip(lam, 0.0, 1.0))
    r_e, r_o = radii_from_lambda(lam)
    return r_e + r_o


def safe_right_homothety_floor(lam: float) -> float:
    """
    Borne exacte de non-recouvrement pour un noeud L2 contracte radialement
    p -> q p face au noeud L1 le plus proche, avec q >= 1/2.

    Pour p = sqrt(2)(1,1,0), a = sqrt(2)(1,0,0),

        d(q)^2 = 1 + (2q - 1)^2.

    La condition d(q) >= R_E + R_O donne

        q >= (1 + sqrt((R_E+R_O)^2 - 1))/2.

    Sur lambda in [0,1], R_E+R_O in [1,sqrt(2)].
    """
    s = right_radius_sum(lam)
    return 0.5 * (1.0 + math.sqrt(max(0.0, s * s - 1.0)))


def safe_right_homothety_scale(lam: float) -> float:
    """
    Version corrigee de l'homothetie : on ne laisse jamais q descendre
    sous la borne exacte de non-recouvrement.
    """
    return max(raw_right_homothety_scale(lam), safe_right_homothety_floor(lam))


def right_critical_clearance(lam: float, q: float) -> float:
    """
    Jeu exact du couple critique L2--L1 :
        clearance = distance_centres - (R_E + R_O).
    Une valeur negative signifie recouvrement.
    """
    lam = float(np.clip(lam, 0.0, 1.0))
    d = math.sqrt(1.0 + (2.0 * q - 1.0) ** 2)
    return d - right_radius_sum(lam)


def raw_golden_lambda() -> float:
    """
    Unique lambda dans (0,1) tel que q_raw(lambda)=phi^-1.
    Equivalent a smoothstep(lambda)=2 phi^-2.
    """
    target = 2.0 * BETA_PHI
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if smoothstep_scalar(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def safe_golden_lambda() -> float:
    """
    Parametre exact auquel la borne de non-recouvrement autorise pour la
    premiere fois q = phi^-1 sur la branche droite.

    A q=alpha=phi^-1:
        2 alpha - 1 = alpha^3,
        d_phi = sqrt(1 + alpha^6).

    Comme R_E+R_O = sqrt(2) - (sqrt(2)-1) lambda,
        lambda_g = (sqrt(2)-d_phi)/(sqrt(2)-1).
    """
    alpha = ALPHA_PHI
    d_phi = math.sqrt(1.0 + alpha ** 6)
    return (SQRT2 - d_phi) / (SQRT2 - 1.0)


RAW_GOLDEN_LAMBDA = raw_golden_lambda()
SAFE_GOLDEN_LAMBDA = safe_golden_lambda()


def right_l2_scale(lam: float, mode: str, morph_strength: float = 1.0) -> float:
    """
    Facteur radial applique aux douze noeuds L2 pour lambda > 0.

    fixed:
        q=1. Correction par defaut: aucun deplacement vers le centre.

    safe:
        homothetie experimentale retardee/projetee dans le domaine exact
        de non-recouvrement.

    raw:
        ancienne homothetie 1 -> 1/2, conservee pour diagnostic.
    """
    lam = float(np.clip(lam, 0.0, 1.0))
    strength = float(np.clip(morph_strength, 0.0, 1.0))

    if mode == "fixed":
        q_target = 1.0
    elif mode == "safe":
        q_target = safe_right_homothety_scale(lam)
    elif mode == "raw":
        q_target = raw_right_homothety_scale(lam)
    else:
        raise ValueError(f"mode de morphing inconnu: {mode}")

    # morph_strength interpole entre centres fixes et le mode demande.
    return 1.0 - strength * (1.0 - q_target)


def positions_from_lambda(
    lam: float,
    center_morph_mode: str = "fixed",
    morph_strength: float = 1.0,
) -> dict[tuple[int, int, int], np.ndarray]:
    """
    Positions courantes de tous les centres.

    lambda = 0:
        cube ternaire 27 exact.

    lambda < 0:
        le morphing historique de la classe O14 vers les centres de faces
        du cuboctaedre est conserve; il ne contracte pas les noeuds L2.

    lambda > 0:
        les douze noeuds L2 utilisent l'un des trois modes:
        fixed (defaut rigoureux), safe (homothetie admissible),
        raw (ancienne homothetie diagnostique).
    """
    lam = float(np.clip(lam, -1.0, 1.0))
    morph_strength = float(np.clip(morph_strength, 0.0, 1.0))
    out: dict[tuple[int, int, int], np.ndarray] = {}

    if lam < 0.0:
        t = morph_strength * smoothstep_scalar(-lam)
        for n in NODES:
            target = left_target_position(n)
            out[n.ternary] = (1.0 - t) * n.xyz + t * target
        return out

    if lam > 0.0:
        q = right_l2_scale(lam, center_morph_mode, morph_strength)
        for n in NODES:
            if n.parity == 0 and n.support == 2:
                out[n.ternary] = q * n.xyz
            else:
                out[n.ternary] = n.xyz.copy()
        return out

    for n in NODES:
        out[n.ternary] = n.xyz.copy()
    return out

    if lam < 0.0:
        t = morph_strength * smoothstep_scalar(-lam)
        for n in NODES:
            target = left_target_position(n)
            out[n.ternary] = (1.0 - t) * n.xyz + t * target
    elif lam > 0.0:
        t = morph_strength * smoothstep_scalar(lam)
        for n in NODES:
            target = right_target_position(n)
            out[n.ternary] = (1.0 - t) * n.xyz + t * target
    else:
        for n in NODES:
            out[n.ternary] = n.xyz.copy()

    return out


def ternary_hasse_pairs() -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    pairs: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for n in NODES:
        u = list(n.ternary)
        for axis in range(3):
            if u[axis] < 1:
                v = list(u)
                v[axis] += 1
                pairs.append((tuple(u), tuple(v)))
    return pairs


def internal_contact_pairs(nodes: list[Node], target_distance: float = 2.0):
    pairs: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if abs(np.linalg.norm(nodes[i].xyz - nodes[j].xyz) - target_distance) < 1e-10:
                pairs.append((nodes[i].ternary, nodes[j].ternary))
    return pairs


CUBE_PAIRS = ternary_hasse_pairs()
CUBO_PAIRS = internal_contact_pairs(E_NODES)
STELLA_PAIRS = internal_contact_pairs(O_NODES)

assert len(NODES) == 27
assert len(E_NODES) == 13
assert len(O_NODES) == 14
assert len(CUBE_PAIRS) == 54
assert len(CUBO_PAIRS) == 36
assert len(STELLA_PAIRS) == 36
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
    return poly, points, scales


def make_glyph_actor(
    nodes: list[Node],
    sphere_source: "vtk.vtkSphereSource",
    color: tuple[float, float, float],
):
    poly, points, scales = polydata_for_nodes(nodes)

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
    return actor, poly, points, scales


class DynamicEdgeActor:
    def __init__(
        self,
        pairs: Iterable[tuple[tuple[int, int, int], tuple[int, int, int]]],
        color=(0.16, 0.16, 0.18),
        width=1.25,
    ):
        self.pairs = list(pairs)
        self.points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()

        for _ in self.pairs:
            i0 = self.points.InsertNextPoint(0.0, 0.0, 0.0)
            i1 = self.points.InsertNextPoint(0.0, 0.0, 0.0)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i0)
            line.GetPointIds().SetId(1, i1)
            self.lines.InsertNextCell(line)

        self.poly = vtk.vtkPolyData()
        self.poly.SetPoints(self.points)
        self.poly.SetLines(self.lines)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.poly)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(*color)
        self.actor.GetProperty().SetLineWidth(width)
        self.actor.GetProperty().SetOpacity(0.0)

    def update_positions(self, current_positions: dict[tuple[int, int, int], np.ndarray]):
        point_id = 0
        for u, v in self.pairs:
            p = current_positions[u]
            q = current_positions[v]
            self.points.SetPoint(point_id, float(p[0]), float(p[1]), float(p[2]))
            self.points.SetPoint(point_id + 1, float(q[0]), float(q[1]), float(q[2]))
            point_id += 2
        self.points.Modified()
        self.poly.Modified()


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
        self.center_morph_mode = args.center_morph_mode
        self.morph_strength = float(np.clip(args.morph_strength, 0.0, 1.0))
        self.edges_enabled = not args.no_edges
        self.cube_enabled = not args.no_cube
        self.last_tick = time.perf_counter()
        self.snapshot_counter = 0
        self.current_positions = {n.ternary: n.xyz.copy() for n in NODES}

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.965, 0.965, 0.975)
        self.renderer.SetBackground2(0.80, 0.84, 0.90)
        self.renderer.GradientBackgroundOn()

        self.window = vtk.vtkRenderWindow()
        self.window.SetWindowName(
            "Cube 27 — Cuboctaedre <-> Stella Octangula — VTK/OpenGL + Morph centres"
        )
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

        self.e_actor, self.e_poly, self.e_points, self.e_scales = make_glyph_actor(
            E_NODES, self.sphere_source, (0.16, 0.53, 0.88)
        )
        self.o_actor, self.o_poly, self.o_points, self.o_scales = make_glyph_actor(
            O_NODES, self.sphere_source, (0.95, 0.47, 0.13)
        )
        self.renderer.AddActor(self.e_actor)
        self.renderer.AddActor(self.o_actor)

        self.cube_edges = DynamicEdgeActor(CUBE_PAIRS, width=1.0)
        self.cubo_edges = DynamicEdgeActor(CUBO_PAIRS, width=1.55)
        self.stella_edges = DynamicEdgeActor(STELLA_PAIRS, width=1.55)
        self.renderer.AddActor(self.cube_edges.actor)
        self.renderer.AddActor(self.cubo_edges.actor)
        self.renderer.AddActor(self.stella_edges.actor)

        self.reference_cube = make_reference_cube_actor()
        self.renderer.AddActor(self.reference_cube)

        if args.axes:
            self.axes_actor = make_axes_actor()
            self.renderer.AddActor(self.axes_actor)
        else:
            self.axes_actor = None

        self.text = vtk.vtkTextActor()
        self.text.SetPosition(18, args.height - 92)
        text_prop = self.text.GetTextProperty()
        text_prop.SetFontFamilyToArial()
        text_prop.SetFontSize(19)
        text_prop.SetColor(0.08, 0.08, 0.10)
        text_prop.SetBold(1)
        self.renderer.AddViewProp(self.text)

        self.help_text = vtk.vtkTextActor()
        self.help_text.SetInput(
            "slider | souris | <- -> | Home / 0 / End | Space: auto | "
            "M: fixed/safe/raw | G: seuil dore safe | E: edges | C: cube | S: PNG | R: camera"
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

    def update_node_points(self):
        for i, n in enumerate(E_NODES):
            p = self.current_positions[n.ternary]
            self.e_points.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))
        self.e_points.Modified()
        self.e_poly.Modified()

        for i, n in enumerate(O_NODES):
            p = self.current_positions[n.ternary]
            self.o_points.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))
        self.o_points.Modified()
        self.o_poly.Modified()

    def update_edges(self):
        self.cube_edges.update_positions(self.current_positions)
        self.cubo_edges.update_positions(self.current_positions)
        self.stella_edges.update_positions(self.current_positions)

    def update_state(self, lam: float, render: bool = True):
        self.lam = float(np.clip(lam, -1.0, 1.0))
        r_e, r_o = radii_from_lambda(self.lam)

        self.current_positions = positions_from_lambda(
            self.lam,
            center_morph_mode=self.center_morph_mode,
            morph_strength=self.morph_strength,
        )
        self.update_node_points()
        self.update_edges()

        self.set_scale_array(self.e_scales, max(r_e, 1e-6))
        self.set_scale_array(self.o_scales, max(r_o, 1e-6))
        self.e_actor.SetVisibility(r_e > 2e-5)
        self.o_actor.SetVisibility(r_o > 2e-5)

        if self.edges_enabled:
            a_mid = max(0.0, 1.0 - abs(self.lam))
            a_cubo = max(0.0, -self.lam)
            a_stella = max(0.0, self.lam)
            self.cube_edges.actor.GetProperty().SetOpacity(0.26 * a_mid)
            self.cubo_edges.actor.GetProperty().SetOpacity(0.42 * a_cubo)
            self.stella_edges.actor.GetProperty().SetOpacity(0.42 * a_stella)
        else:
            self.cube_edges.actor.GetProperty().SetOpacity(0.0)
            self.cubo_edges.actor.GetProperty().SetOpacity(0.0)
            self.stella_edges.actor.GetProperty().SetOpacity(0.0)

        self.reference_cube.SetVisibility(self.cube_enabled)

        if self.lam > 0.0:
            q_l2 = right_l2_scale(
                self.lam, self.center_morph_mode, self.morph_strength
            )
            gap = right_critical_clearance(self.lam, q_l2)
            delta = 1.0 - q_l2
            golden_dist = abs(q_l2 - ALPHA_PHI)
            diag_line = (
                f"q_L2 = {q_l2:.6f}    delta = {delta:.6f}    "
                f"gap_crit = {gap:+.6e}    |q-phi^-1| = {golden_dist:.3e}"
            )
        else:
            diag_line = "q_L2 = 1.000000    branche droite inactive"

        self.text.SetInput(
            f"{state_name(self.lam)}\n"
            f"lambda = {self.lam:+.4f}    R_E = {r_e:.6f}    R_O = {r_o:.6f}\n"
            f"morph L2 = {self.center_morph_mode.upper()}    force = {self.morph_strength:.2f}\n"
            f"{diag_line}"
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
        elif key_low == "m":
            modes = ("fixed", "safe", "raw")
            i = modes.index(self.center_morph_mode)
            self.center_morph_mode = modes[(i + 1) % len(modes)]
            self.update_state(self.lam)
        elif key_low == "g":
            self.center_morph_mode = "safe"
            self.update_state(SAFE_GOLDEN_LAMBDA)
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
            path = Path(f"cube27_morph_snapshot_{self.snapshot_counter:03d}.png")
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
            "O14 stella octangula, avec morphing geometrique continu des centres."
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
    p.add_argument(
        "--center-morph-mode",
        choices=("fixed", "safe", "raw"),
        default="fixed",
        help=(
            "fixed: correction rigoureuse sans contraction L2; "
            "safe: homothetie projetee dans le domaine de non-recouvrement; "
            "raw: ancienne homothetie diagnostique."
        ),
    )
    p.add_argument("--morph-strength", type=float, default=1.0)
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
    print("  M              : cycle FIXED / SAFE / RAW pour les noeuds L2")
    print(f"  G              : seuil dore SAFE lambda={SAFE_GOLDEN_LAMBDA:.9f}")
    print("  Souris         : camera native VTK")
    print("  E / C          : aretes / cube de reference")
    print("  S              : capture PNG")
    print("  Q / Esc        : quitter")
    app.start()


if __name__ == "__main__":
    main()
