"""
retina_animation.py — Manim-Animation: Retina Schicht für Schicht aufbauen.

Verwendet die EXAKT gleichen Positions-Generatoren wie retina_main.py
(uniform_cap_points, uniform_sphere_band, radial_project — über
Retina._generate_positions). Keine Doppel-Implementierung der Geometrie.

USAGE
-----
Zwingende Files im selben Ordner (oder via PYTHONPATH erreichbar):
  retina_main.py
  retina_scales.py

Render-Befehle:

  # Schnelle Vorschau (480p, ~10-20 Min mit Cairo bei scale=2)
  manim -pql retina_animation.py RetinaBuildup

  # Hochauflösend (1080p, lange Renderzeit)
  manim -pqh retina_animation.py RetinaBuildup

  # OpenGL-Renderer ist mit ~4500 Dots DEUTLICH schneller — empfohlen:
  manim --renderer=opengl -pql retina_animation.py RetinaBuildup

NEST muss NICHT installiert sein, um die Animation zu rendern: das Skript
stub'd `nest` falls nicht vorhanden, weil compute_counts() und
_generate_positions() reines Python sind und `nest.Create` nicht aufrufen.
"""

# ---------------------------------------------------------------------------
# Optional NEST-Stub
# ---------------------------------------------------------------------------
# retina_main.py macht `import nest` ganz oben. Für Positions-Berechnung
# brauchen wir aber kein NEST. Daher: wenn nest nicht installiert ist, ein
# Dummy-Modul anlegen, BEVOR retina_main importiert wird.
import sys
import types

try:
    import nest  # noqa: F401
except ImportError:
    _stub = types.ModuleType("nest")
    _stub.spatial = types.SimpleNamespace(free=lambda *a, **k: None)
    _stub.Create = lambda *a, **k: None
    _stub.Connect = lambda *a, **k: None
    _stub.NodeCollection = object
    _stub.set = lambda *a, **k: None
    sys.modules["nest"] = _stub

import numpy as np
from manim import (
    ThreeDScene, VGroup, Dot, Dot3D, Text, FadeIn,
    DEGREES, WHITE, UL, RIGHT, DOWN, LEFT,
)

from retina_main import (
    Retina,
    POP_L_FOVEAL, POP_M_FOVEAL,
    POP_L_PERIPHERAL, POP_M_PERIPHERAL, POP_S_PERIPHERAL,
    POP_RODS,
    POP_HORIZONTAL_FOVEAL, POP_HORIZONTAL_PERIPHERAL,
    POP_MIDGET_ON_BIP_FOVEAL, POP_MIDGET_OFF_BIP_FOVEAL,
    POP_MIDGET_ON_BIP_PERIPHERAL, POP_MIDGET_OFF_BIP_PERIPHERAL,
    POP_PARASOL_ON_BIP, POP_PARASOL_OFF_BIP,
    POP_KONIO_S_BIP,
    POP_AMACRINE_FOVEAL, POP_AMACRINE_PERIPHERAL,
    POP_MIDGET_ON_GANG_FOVEAL, POP_MIDGET_OFF_GANG_FOVEAL,
    POP_MIDGET_ON_GANG_PERIPHERAL, POP_MIDGET_OFF_GANG_PERIPHERAL,
    POP_PARASOL_ON_GANG, POP_PARASOL_OFF_GANG,
    POP_KONIO_GANG,
)
from retina_scales import get_config


# ===========================================================================
# CONFIG
# ===========================================================================

# '1' = ~2.5k Punkte (Default, schneller Render).
# '2' = ~4.4k (dichter, schöner; ~2x Renderzeit).
# '3'+ wird ab ~30k Dots zäh — nur mit OpenGL-Renderer empfohlen.
SCALE = "1"
VARIANT = "default"


# ===========================================================================
# KOORDINATEN-REMAP
# ===========================================================================
# In retina_main.py ist +y die foveale Blickrichtung (siehe uniform_cap_points
# Kommentar: "Y ist die nach-vorne-Achse: Fovea schaut in +y-Richtung").
# Manim's ThreeDScene rendert mit +z als "oben" beim Default-Kamerawinkel.
# Wir tauschen y und z, damit die Foveal-Kappe "nach oben" zeigt und die
# Default-Kameraneigung (phi=70°) sie aus einem 3/4-Winkel zeigt.
def to_manim(p: np.ndarray) -> np.ndarray:
    """(x_r, y_r, z_r) -> (x_m, z_r, y_r)  — Foveal-Achse y wird zu manim-z."""
    return np.column_stack([p[:, 0], p[:, 2], p[:, 1]])


# ===========================================================================
# FARBEN — biologische Rollen, NICHT NEST-Typen
# ===========================================================================

CLR_L = "#E63946"        # L-Zapfen — rot-empfindlich
CLR_M = "#52B788"        # M-Zapfen — grün-empfindlich
CLR_S = "#4895EF"        # S-Zapfen — blau-empfindlich
CLR_RODS = "#A0A0A0"     # Stäbchen — Schwachlicht
CLR_HORIZ = "#F77F00"    # Horizontalzellen — laterale Hemmung
CLR_AMAC = "#FF6FA3"     # Amakrinzellen — zeitliche Verarbeitung
CLR_MIDGET = "#C77DFF"   # Midget-Pfad — Detail/Farbe (Parvo)
CLR_PARASOL = "#4CC9F0"  # Parasol-Pfad — Bewegung/Kontrast (Magno)
CLR_KONIO = "#FFD60A"    # Konio-Pfad — Blau-Gelb-Opponenz

# Population -> Farbe. Bipolare und Ganglien des gleichen Pfads teilen sich
# die Farbe (die Schichtung unterscheidet sie räumlich genug).
POP_COLORS = {
    POP_L_FOVEAL: CLR_L,
    POP_L_PERIPHERAL: CLR_L,
    POP_M_FOVEAL: CLR_M,
    POP_M_PERIPHERAL: CLR_M,
    POP_S_PERIPHERAL: CLR_S,
    POP_RODS: CLR_RODS,

    POP_HORIZONTAL_FOVEAL: CLR_HORIZ,
    POP_HORIZONTAL_PERIPHERAL: CLR_HORIZ,
    POP_AMACRINE_FOVEAL: CLR_AMAC,
    POP_AMACRINE_PERIPHERAL: CLR_AMAC,

    POP_MIDGET_ON_BIP_FOVEAL: CLR_MIDGET,
    POP_MIDGET_OFF_BIP_FOVEAL: CLR_MIDGET,
    POP_MIDGET_ON_BIP_PERIPHERAL: CLR_MIDGET,
    POP_MIDGET_OFF_BIP_PERIPHERAL: CLR_MIDGET,
    POP_MIDGET_ON_GANG_FOVEAL: CLR_MIDGET,
    POP_MIDGET_OFF_GANG_FOVEAL: CLR_MIDGET,
    POP_MIDGET_ON_GANG_PERIPHERAL: CLR_MIDGET,
    POP_MIDGET_OFF_GANG_PERIPHERAL: CLR_MIDGET,

    POP_PARASOL_ON_BIP: CLR_PARASOL,
    POP_PARASOL_OFF_BIP: CLR_PARASOL,
    POP_PARASOL_ON_GANG: CLR_PARASOL,
    POP_PARASOL_OFF_GANG: CLR_PARASOL,

    POP_KONIO_S_BIP: CLR_KONIO,
    POP_KONIO_GANG: CLR_KONIO,
}

# Legende — links oben, fix im Bild
LEGEND = [
    (CLR_L,       "L-Zapfen (rot-empfindlich)"),
    (CLR_M,       "M-Zapfen (grün-empfindlich)"),
    (CLR_S,       "S-Zapfen (blau-empfindlich)"),
    (CLR_RODS,    "Stäbchen (Schwachlicht)"),
    (CLR_HORIZ,   "Horizontalzellen (laterale Hemmung)"),
    (CLR_MIDGET,  "Midget-Pfad (Detail / Farbe)"),
    (CLR_PARASOL, "Parasol-Pfad (Bewegung / Kontrast)"),
    (CLR_KONIO,   "Koniozelluläre Bahn (Blau-Gelb)"),
    (CLR_AMAC,    "Amakrinzellen (zeitliche Verarbeitung)"),
]


# ===========================================================================
# BUILD-STAGES
# ===========================================================================
# Reihenfolge entspricht _generate_positions in retina_main.py:
#   Layer 0: Photorezeptoren (außen)
#   Layer 1: Horizontalzellen
#   Layer 2: Midget-Bipolare
#   Layer 3: Parasol- + Konio-Bipolare
#   Layer 4: Amakrinzellen
#   Layer 5: Ganglienzellen (innen — Sehnerv-Output)
#
# Pro Stage werden ALLE genannten Pops gleichzeitig gefadet — z.B. ON+OFF
# oder foveal+peripheral zusammen.

STAGES = [
    # Layer 0 — Photorezeptoren, ein Typ nach dem anderen
    ("L-Zapfen",    [POP_L_FOVEAL, POP_L_PERIPHERAL]),
    ("M-Zapfen",    [POP_M_FOVEAL, POP_M_PERIPHERAL]),
    ("S-Zapfen",    [POP_S_PERIPHERAL]),
    ("Stäbchen",    [POP_RODS]),

    # Layer 1
    ("Horizontalzellen",
        [POP_HORIZONTAL_FOVEAL, POP_HORIZONTAL_PERIPHERAL]),

    # Layer 2 — Midget Bipolare (ON+OFF, foveal+peripher in einem Beat;
    # im Code 1:1:1-aligned mit den fovealen Zapfen via radial_project)
    ("Midget-Bipolare",
        [POP_MIDGET_ON_BIP_FOVEAL, POP_MIDGET_OFF_BIP_FOVEAL,
         POP_MIDGET_ON_BIP_PERIPHERAL, POP_MIDGET_OFF_BIP_PERIPHERAL]),

    # Layer 3 — Parasol + Konio (parallele periphere Pfade)
    ("Parasol-Bipolare",
        [POP_PARASOL_ON_BIP, POP_PARASOL_OFF_BIP]),
    ("Konio-Bipolare",
        [POP_KONIO_S_BIP]),

    # Layer 4
    ("Amakrinzellen",
        [POP_AMACRINE_FOVEAL, POP_AMACRINE_PERIPHERAL]),

    # Layer 5 — Output
    ("Midget-Ganglien (Output)",
        [POP_MIDGET_ON_GANG_FOVEAL, POP_MIDGET_OFF_GANG_FOVEAL,
         POP_MIDGET_ON_GANG_PERIPHERAL, POP_MIDGET_OFF_GANG_PERIPHERAL]),
    ("Parasol-Ganglien (Output)",
        [POP_PARASOL_ON_GANG, POP_PARASOL_OFF_GANG]),
    ("Konio-Ganglien (Output)",
        [POP_KONIO_GANG]),
]


# ===========================================================================
# HELPER
# ===========================================================================

def make_dot_group(positions: np.ndarray, color: str,
                    radius: float = 0.025) -> VGroup:
    """VGroup aus Dot3D-Punkten an den (manim-remappten) Positionen."""
    if len(positions) == 0:
        return VGroup()
    pts_m = to_manim(positions)
    dots = VGroup(*[
        # resolution=(8,8) -> 64 Polygone pro Dot. Niedrig genug für
        # mehrere tausend Dots bei akzeptabler Renderzeit.
        Dot3D(point=p, radius=radius, color=color, resolution=(8, 8))
        for p in pts_m
    ])
    return dots


def adaptive_radius(n: int) -> float:
    """Kleinere Pops bekommen größere Dots, damit nichts verschwindet."""
    if n < 30:
        return 0.045
    if n < 100:
        return 0.035
    if n < 400:
        return 0.028
    return 0.022


def build_legend() -> VGroup:
    """Legende oben links, fix im Frame."""
    rows = VGroup()
    for color, label in LEGEND:
        dot = Dot(radius=0.10, color=color)
        text = Text(label, font_size=20, color=WHITE).next_to(
            dot, RIGHT, buff=0.20)
        rows.add(VGroup(dot, text))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.20)
    rows.to_corner(UL, buff=0.45)
    return rows


# ===========================================================================
# SCENE
# ===========================================================================

class RetinaBuildup(ThreeDScene):
    """Schicht-für-Schicht-Aufbau einer biologisch plausiblen Retina.

    Positionen werden über Retina._generate_positions() berechnet — die
    EXAKT gleiche Funktion, die im NEST-Build genutzt wird. Keine Mock-
    Geometrie, kein Resampling.
    """

    def construct(self):
        # --- Positionen aus dem echten Retina-Code holen ---
        params, neuron_params, _ = get_config(SCALE, VARIANT)
        retina = Retina(params=params, neuron_params=neuron_params,
                        verbose=False)
        retina.compute_counts()
        positions = retina._generate_positions()

        # --- Legende (fix im Frame) ---
        legend = build_legend()
        self.add_fixed_in_frame_mobjects(legend)

        # --- 3D-Kamera ---
        # phi=70° kippt vom Default-"von oben" Richtung Horizont,
        # theta dreht um die vertikale Achse. Foveal-Kappe wurde via
        # to_manim() auf manim-+z gemappt -> sie zeigt "nach oben".
        self.set_camera_orientation(
            phi=70 * DEGREES,
            theta=-50 * DEGREES,
        )
        # Sanfte Rotation, damit der 3D-Aufbau klar wird.
        self.begin_ambient_camera_rotation(rate=0.08, about="theta")

        # --- Stages der Reihe nach einblenden ---
        for stage_name, pop_names in STAGES:
            anims = []
            for pop_name in pop_names:
                pos = positions.get(pop_name)
                if pos is None or len(pos) == 0:
                    continue
                color = POP_COLORS[pop_name]
                radius = adaptive_radius(len(pos))
                group = make_dot_group(pos, color, radius=radius)
                # lag_ratio<<1 -> stark gestaffelter Fade ("Wachsen" der Schicht).
                anims.append(
                    FadeIn(group, lag_ratio=0.004, run_time=1.4)
                )
            if not anims:
                continue
            self.play(*anims)
            self.wait(0.35)

        # --- Letzter Halt + langsames Auslaufen der Rotation ---
        self.wait(2.0)
        self.stop_ambient_camera_rotation()
        self.wait(1.5)
