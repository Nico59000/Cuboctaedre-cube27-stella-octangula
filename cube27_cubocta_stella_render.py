#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cube27_cubocta_stella_render.py

Animation parametrique exacte inspiree de:
"Cube ternaire de 27 boules, decomposition cuboctaedrique 13/14
et complexe resolu de la stella octangula" (R1).

Etats clefs
-----------
lambda = -1 :
    E13 = L0 U L2
    13 boules : centre + 12 sommets du cuboctaedre
    rayon = 1

lambda = 0 :
    G = {-sqrt(2), 0, +sqrt(2)}^3
    27 boules du cube ternaire
    rayon = 1/sqrt(2)

lambda = +1 :
    O14 = L1 U L3
    14 boules : 6 sommets de l'octaedre central + 8 sommets externes
    du complexe resolu de la stella octangula
    rayon = 1

Pourquoi ce changement d'echelle ?
----------------------------------
Le document travaille avec a = 1/sqrt(2), et le bloc initial de 27 boules
a un *diametre* a, donc un rayon 1/(2 sqrt(2)).
On multiplie ici toutes les longueurs du document par 2 afin que le rayon
du cube 27 au milieu de la parametrisation soit exactement 1/sqrt(2),
comme demande.

La trajectoire des rayons est la mise a l'echelle x2 du domaine exact
de non-recouvrement de la proposition 6.3:
    R_E <= 1
    R_O <= 1
    R_E + R_O <= sqrt(2)

Le chemin piecewise-lineaire utilise ici relie:
    (R_E,R_O) = (1,0)
                 -> (1/sqrt(2), 1/sqrt(2))
                 -> (0,1)

Le script offre:
- un mode interactif avec slider;
- touches gauche/droite pour regler lambda;
- Home / End / 0 pour aller aux trois etats clefs;
- espace pour lancer/arreter une oscillation automatique;
- sauvegarde GIF ou MP4 si demandee.

Dependances:
    numpy
    matplotlib
    pillow  (seulement pour GIF)
    ffmpeg  (seulement pour MP4)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.widgets import Slider


SQRT2 = math.sqrt(2.0)
R_MID = 1.0 / SQRT2
R_END = 1.0

# Le document place G = a{-1,0,1}^3 avec a=1/sqrt(2).
# Mise a l'echelle x2 => pas du cube ternaire = sqrt(2).
STEP = SQRT2

# Reglages de rendu.
DEFAULT_SPHERE_U = 18
DEFAULT_SPHERE_V = 32
DEFAULT_LIMIT = 2.95


@dataclass(frozen=True)
class Node:
    ternary: tuple[int, int, int]
    xyz: np.ndarray
    support: int
    parity: int  # 0 = E13, 1 = O14


def build_nodes() -> list[Node]:
    nodes: list[Node] = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                u = (i, j, k)
                support = int(i != 0) + int(j != 0) + int(k != 0)
                nodes.append(
                    Node(
                        ternary=u,
                        xyz=STEP * np.array(u, dtype=float),
                        support=support,
                        parity=support % 2,
                    )
                )
    return nodes


NODES = build_nodes()
E_NODES = [n for n in NODES if n.parity == 0]  # 1 + 12 = 13
O_NODES = [n for n in NODES if n.parity == 1]  # 6 + 8 = 14


def radii_from_lambda(lam: float) -> tuple[float, float]:
    """
    Retourne (R_E, R_O), avec lambda dans [-1,1].

    Le chemin est exactement dans le domaine de non-recouvrement mis a
    l'echelle x2:
        2 R_E <= 2
        2 R_O <= 2
        R_E + R_O <= sqrt(2).
    """
    lam = float(np.clip(lam, -1.0, 1.0))

    if lam <= 0.0:
        s = lam + 1.0  # -1 -> 0 ; 0 -> 1
        r_e = (1.0 - s) * R_END + s * R_MID
        r_o = s * R_MID
    else:
        s = lam  # 0 -> 0 ; 1 -> 1
        r_e = (1.0 - s) * R_MID
        r_o = (1.0 - s) * R_MID + s * R_END

    return r_e, r_o


def state_name(lam: float) -> str:
    if abs(lam + 1.0) < 1e-9:
        return "E13 : cuboctaedre de boules (13)"
    if abs(lam) < 1e-9:
        return "G : cube ternaire 27"
    if abs(lam - 1.0) < 1e-9:
        return "O14 : stella octangula resolue (14)"
    if lam < 0:
        return "transition cuboctaedre -> cube 27"
    return "transition cube 27 -> stella octangula"


def sphere_mesh(center: np.ndarray, radius: float, nu: int, nv: int):
    u = np.linspace(0.0, 2.0 * np.pi, nv)
    v = np.linspace(0.0, np.pi, nu)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def ternary_hasse_edges() -> list[tuple[np.ndarray, np.ndarray]]:
    """Les 54 contacts du cube 3x3x3 : +/-1 dans une seule coordonnee."""
    by_u = {n.ternary: n for n in NODES}
    edges = []
    for n in NODES:
        u = list(n.ternary)
        for axis in range(3):
            if u[axis] < 1:
                v = list(u)
                v[axis] += 1
                vn = by_u[tuple(v)]
                edges.append((n.xyz, vn.xyz))
    assert len(edges) == 54
    return edges


def internal_contact_edges(nodes: list[Node], distance: float = 2.0):
    """Contacts internes dans E13 ou O14 aux etats unitaires (echelle x2)."""
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            d = np.linalg.norm(nodes[i].xyz - nodes[j].xyz)
            if abs(d - distance) < 1e-10:
                edges.append((nodes[i].xyz, nodes[j].xyz))
    return edges


CUBE_EDGES = ternary_hasse_edges()     # 54
CUBO_EDGES = internal_contact_edges(E_NODES, 2.0)  # 36
STELLA_EDGES = internal_contact_edges(O_NODES, 2.0)  # 36

assert len(E_NODES) == 13
assert len(O_NODES) == 14
assert len(CUBO_EDGES) == 36
assert len(STELLA_EDGES) == 36


def draw_segment_set(ax, edges, alpha, linewidth=0.8):
    if alpha <= 1e-4:
        return
    for p, q in edges:
        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            [p[2], q[2]],
            linewidth=linewidth,
            alpha=float(alpha),
            zorder=1,
        )


def draw_reference_cube(ax, alpha=0.16):
    """Cadre de conv(O14), soit le cube aux sommets (+/-sqrt(2))^3."""
    s = STEP
    verts = [
        np.array((x, y, z), float)
        for x in (-s, s)
        for y in (-s, s)
        for z in (-s, s)
    ]
    for i, p in enumerate(verts):
        for q in verts[i + 1:]:
            diff = np.abs(p - q)
            if np.count_nonzero(diff > 1e-12) == 1:
                ax.plot(
                    [p[0], q[0]],
                    [p[1], q[1]],
                    [p[2], q[2]],
                    linewidth=0.7,
                    alpha=alpha,
                    zorder=0,
                )


def configure_axes(ax, elev=22.0, azim=34.0):
    ax.set_xlim(-DEFAULT_LIMIT, DEFAULT_LIMIT)
    ax.set_ylim(-DEFAULT_LIMIT, DEFAULT_LIMIT)
    ax.set_zlim(-DEFAULT_LIMIT, DEFAULT_LIMIT)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(False)


def draw_state(
    ax,
    lam: float,
    sphere_u: int,
    sphere_v: int,
    elev: float,
    azim: float,
    show_edges: bool = True,
):
    ax.cla()
    configure_axes(ax, elev=elev, azim=azim)
    draw_reference_cube(ax)

    r_e, r_o = radii_from_lambda(lam)

    # Topologie de contact interpolee:
    # - cuboctaedre dominant a gauche;
    # - cube de Hasse dominant au milieu;
    # - stella dominante a droite.
    if show_edges:
        a_mid = max(0.0, 1.0 - abs(lam))
        a_cubo = max(0.0, -lam)
        a_stella = max(0.0, lam)
        draw_segment_set(ax, CUBE_EDGES, alpha=0.28 * a_mid, linewidth=0.7)
        draw_segment_set(ax, CUBO_EDGES, alpha=0.38 * a_cubo, linewidth=0.9)
        draw_segment_set(ax, STELLA_EDGES, alpha=0.38 * a_stella, linewidth=0.9)

    # E13 : centre + milieux d'aretes du cube = classe de parite paire.
    if r_e > 1e-4:
        for n in E_NODES:
            x, y, z = sphere_mesh(n.xyz, r_e, sphere_u, sphere_v)
            ax.plot_surface(
                x, y, z,
                linewidth=0.0,
                antialiased=True,
                alpha=0.72,
                shade=True,
                zorder=3,
            )

    # O14 : centres de faces + sommets du cube = classe de parite impaire.
    if r_o > 1e-4:
        for n in O_NODES:
            x, y, z = sphere_mesh(n.xyz, r_o, sphere_u, sphere_v)
            ax.plot_surface(
                x, y, z,
                linewidth=0.0,
                antialiased=True,
                alpha=0.72,
                shade=True,
                zorder=2,
            )

    ax.set_title(
        f"{state_name(lam)}\n"
        f"$\\lambda={lam:+.3f}$   "
        f"$R_E={r_e:.6f}$   $R_O={r_o:.6f}$"
    )


def smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


def cycle_parameter(frames: int) -> np.ndarray:
    """
    Cycle ferme:
        -1 -> 0 -> +1 -> 0 -> -1
    avec interpolation smoothstep sur chaque quart.
    """
    if frames < 8:
        frames = 8
    u = np.linspace(0.0, 4.0, frames, endpoint=False)
    out = np.empty_like(u)

    for idx, t in enumerate(u):
        seg = int(math.floor(t))
        x = smoothstep(np.array([t - seg]))[0]
        if seg == 0:
            out[idx] = -1.0 + x
        elif seg == 1:
            out[idx] = 0.0 + x
        elif seg == 2:
            out[idx] = 1.0 - x
        else:
            out[idx] = 0.0 - x
    return out


def make_figure(figsize=(9.0, 8.0)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def interactive(args):
    fig, ax = make_figure()
    plt.subplots_adjust(bottom=0.16)

    slider_ax = fig.add_axes([0.16, 0.055, 0.68, 0.035])
    slider = Slider(
        slider_ax,
        "lambda",
        -1.0,
        1.0,
        valinit=-1.0,
        valstep=0.01,
    )

    playing = {"on": False, "direction": 1.0}

    def redraw(_=None):
        draw_state(
            ax,
            slider.val,
            args.sphere_u,
            args.sphere_v,
            args.elev,
            args.azim,
            not args.no_edges,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "left":
            slider.set_val(max(-1.0, slider.val - 0.04))
        elif event.key == "right":
            slider.set_val(min(1.0, slider.val + 0.04))
        elif event.key == "home":
            slider.set_val(-1.0)
        elif event.key in ("0", "insert"):
            slider.set_val(0.0)
        elif event.key == "end":
            slider.set_val(1.0)
        elif event.key == " ":
            playing["on"] = not playing["on"]

    def timer_step():
        if not playing["on"]:
            return
        x = slider.val + playing["direction"] * 0.025
        if x >= 1.0:
            x = 1.0
            playing["direction"] = -1.0
        elif x <= -1.0:
            x = -1.0
            playing["direction"] = 1.0
        slider.set_val(x)

    slider.on_changed(redraw)
    fig.canvas.mpl_connect("key_press_event", on_key)

    timer = fig.canvas.new_timer(interval=35)
    timer.add_callback(timer_step)
    timer.start()

    redraw()
    print("Commandes interactives:")
    print("  slider        : deplacer lambda")
    print("  gauche/droite : pas de 0.04")
    print("  Home          : cuboctaedre E13")
    print("  0             : cube 27, rayon 1/sqrt(2)")
    print("  End           : stella O14")
    print("  espace        : lecture/pause")
    plt.show()


def save_animation(args, out_path: Path):
    fig, ax = make_figure()
    values = cycle_parameter(args.frames)

    def update(frame_index):
        lam = float(values[frame_index])
        # Rotation camera legere optionnelle.
        azim = args.azim
        if args.rotate_camera:
            azim = args.azim + 360.0 * frame_index / len(values)

        draw_state(
            ax,
            lam,
            args.sphere_u,
            args.sphere_v,
            args.elev,
            azim,
            not args.no_edges,
        )
        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(values),
        interval=1000.0 / args.fps,
        blit=False,
    )

    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        writer = PillowWriter(fps=args.fps)
    elif suffix == ".mp4":
        writer = FFMpegWriter(fps=args.fps, bitrate=args.bitrate)
    else:
        raise SystemExit("Extension de sortie attendue: .gif ou .mp4")

    print(f"Rendu de {len(values)} images -> {out_path}")
    anim.save(out_path, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print("Termine.")


def save_still(args, out_path: Path, lam: float):
    fig, ax = make_figure()
    draw_state(
        ax,
        lam,
        args.sphere_u,
        args.sphere_v,
        args.elev,
        args.azim,
        not args.no_edges,
    )
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Image enregistree -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Animation E13 cuboctaedrique <-> cube 27 <-> O14 stella octangula, "
            "avec rayon 1/sqrt(2) au milieu."
        )
    )
    p.add_argument(
        "--save",
        type=Path,
        help="Sauvegarde une animation .gif ou .mp4 au lieu du mode interactif.",
    )
    p.add_argument(
        "--still",
        type=Path,
        help="Sauvegarde une image fixe PNG/JPG.",
    )
    p.add_argument(
        "--lambda-value",
        type=float,
        default=0.0,
        help="Valeur de lambda pour --still (defaut: 0).",
    )
    p.add_argument("--frames", type=int, default=160)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--bitrate", type=int, default=6000)
    p.add_argument("--sphere-u", type=int, default=DEFAULT_SPHERE_U)
    p.add_argument("--sphere-v", type=int, default=DEFAULT_SPHERE_V)
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=34.0)
    p.add_argument("--rotate-camera", action="store_true")
    p.add_argument("--no-edges", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Sanity checks exacts des trois etats clefs.
    assert np.allclose(radii_from_lambda(-1.0), (1.0, 0.0))
    assert np.allclose(radii_from_lambda(0.0), (R_MID, R_MID))
    assert np.allclose(radii_from_lambda(+1.0), (0.0, 1.0))

    if args.still:
        save_still(args, args.still, float(np.clip(args.lambda_value, -1.0, 1.0)))
    elif args.save:
        save_animation(args, args.save)
    else:
        interactive(args)


if __name__ == "__main__":
    main()
