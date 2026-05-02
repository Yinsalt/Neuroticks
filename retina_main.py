"""
===============================================================================
RETINA v2 — Biologisch plausible Retina mit räumlicher Konnektivität
===============================================================================

Eine von Grund auf überarbeitete Retina-Simulation für NEST, mit Fokus auf:

  1. BIOLOGISCH KORREKTE KONVERGENZ via nearest-neighbor
     Jede Zelle wird explizit mit ihren räumlichen Nachbarn verbunden.
     Keine pairwise_bernoulli Zufallsverbindungen mehr.

  2. MIDGET vs PARASOL STRIKT GETRENNT
     Zwei parallele Pfade vom Zapfen bis zur Ganglienzelle. Midget =
     hohe Auflösung/Farbe/Parvo-LGN. Parasol = Bewegung/Kontrast/Magno-LGN.

  3. PHOTOREZEPTOREN ALS STROMQUELLEN, NICHT ALS SPIKES
     Echte Photorezeptoren spiken nicht. Wir modellieren sie als
     parrot_neurons (nur für Positionsreferenz) und injizieren den
     visuellen Input als Strom DIREKT in die Bipolars.

  4. HORIZONTAL + AMACRINE für laterale Verarbeitung

  5. MEHRERE AGENTEN via frei wählbarem origin-Offset

  6. SEPARATER INPUT-FEEDER als eigenständiges Objekt

===============================================================================
PUBLIC API
===============================================================================

    from retina import Retina

    retina = Retina(params={'n_cones_foveal': 300, 'origin': (10, 0, 0)})
    retina.build()
    retina.connect()

    feeder_config = {
        'generator_type': 'step_current',
        'input_resolution': (64, 64),
        'max_current_pa': 400.0,
    }
    feeder = retina.create_input_feeder(feeder_config)

    pops = retina.get_populations()
    nest.Connect(pops['midget_ON_ganglion_foveal'], my_lgn_target,
                 'one_to_one', {'weight': 2.0, 'delay': 1.0})

    feeder.feed(lms_frame, intensity_frame)
    nest.Simulate(50.0)

===============================================================================
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import nest


# ============================================================================
#  SECTION 1: BIOLOGIE-REFERENZ (als Code-Kommentar)
# ============================================================================
#
# Die Retina hat ZWEI parallele Pfade vom Photorezeptor bis zum Sehnerv.
# Sie sind anatomisch UND funktional getrennt, landen in verschiedenen
# LGN-Schichten und codieren unterschiedliche Bildaspekte.
#
# MIDGET-PFAD (hohe Auflösung, Farbe, Parvo-LGN)
#   Fovea:      1 Zapfen -> 1 Midget-Bipolar -> 1 Midget-Ganglion  (1:1:1)
#   Peripheral: 2-3 Zapfen pro Midget-Bipolar, 3 Bipolars pro Ganglion
#   --> Kleine rezeptive Felder, langsame Integration, rot/grün-Gegenfarben.
#       Deswegen liest du mit der Fovea und siehst Farben im Zentrum.
#
# PARASOL-PFAD (Bewegung, Kontrast, Magno-LGN)
#   Diffuse Bipolar integriert ~8 Zapfen
#   Parasol-Ganglion integriert ~6 Bipolars
#   Parasol existiert nur peripheral (keine foveale Parasol-Population!)
#   --> Große rezeptive Felder, schnelle Antwort, achromatisch.
#       Deswegen siehst du Bewegung auch am Rand des Blickfelds.
#
# ON vs OFF: jeder Pfad kommt doppelt vor — ON-Zellen feuern bei lokalem
# Helligkeitsanstieg, OFF-Zellen bei Abfall. Zusammen codieren sie sowohl
# positive als auch negative Kontraste (und vermeiden negative Feuerraten).
#
# HORIZONTAL-ZELLEN: laterale Inhibition auf Bipolar-Ebene.
#   Bilden das "Surround" im klassischen Center-Surround-RF. Ohne sie würde
#   die Retina nur absolute Helligkeit codieren statt lokalen Kontrast.
#
# AMACRINE: laterale Inhibition auf Ganglien-Ebene + temporale Verarbeitung.
#   Zuständig u.a. für transiente Antworten und Bewegungssensitivität.
#   Hier als AdEx (aeif_cond_exp) modelliert wegen Adaptations-Dynamik.
#
# SCHICHTUNG (von außen nach innen, Richtung Licht->Sehnerv):
#   Photorezeptoren -> Horizontal -> Bipolar -> Amacrine -> Ganglion
#   Im Code wird das durch abnehmende Kugelradien nachgebaut (radius_step).


# ============================================================================
#  SECTION 2: POPULATIONS-NAMEN ALS KONSTANTEN
# ============================================================================
#
# Alle Populationen werden über String-Keys angesprochen (self.populations,
# self.positions, self.counts, mapping etc.). Die Namen hier festzupinnen
# verhindert Tippfehler-Bugs, die sonst erst zur Laufzeit auffallen würden,
# und ermöglicht IDE-Autovervollständigung.

# --- PHOTOREZEPTOREN ---
# Drei Zapfentypen (L/M/S) + Stäbchen. L und M existieren foveal UND peripher,
# S nur peripher (die Fovea hat biologisch fast keine S-Zapfen, deswegen
# siehst du Blau mittig schlechter als Rot/Grün). Rods nur peripher.

POP_L_FOVEAL = 'L_foveal'
POP_M_FOVEAL = 'M_foveal'
POP_L_PERIPHERAL = 'L_peripheral'
POP_M_PERIPHERAL = 'M_peripheral'
POP_S_PERIPHERAL = 'S_peripheral'
POP_RODS = 'rods'

PHOTORECEPTOR_POPS = [
    POP_L_FOVEAL, POP_M_FOVEAL,
    POP_L_PERIPHERAL, POP_M_PERIPHERAL, POP_S_PERIPHERAL,
    POP_RODS,
]

# --- HORIZONTAL ---
# Foveal und peripher getrennt, weil Dichte/Reichweite stark unterschiedlich.
POP_HORIZONTAL_FOVEAL = 'horizontal_foveal'
POP_HORIZONTAL_PERIPHERAL = 'horizontal_peripheral'

# --- MIDGET-BIPOLARS ---
# Getrennt ON/OFF und getrennt foveal/peripher -> 4 Populationen.
# Foveal ist Midget die DOMINANTE Bipolar-Klasse (hohe Auflösung, 1:1).
POP_MIDGET_ON_BIP_FOVEAL = 'midget_ON_bipolar_foveal'
POP_MIDGET_OFF_BIP_FOVEAL = 'midget_OFF_bipolar_foveal'
POP_MIDGET_ON_BIP_PERIPHERAL = 'midget_ON_bipolar_peripheral'
POP_MIDGET_OFF_BIP_PERIPHERAL = 'midget_OFF_bipolar_peripheral'

# --- PARASOL-BIPOLARS ---
# Nur peripher! Der Parasol-Pfad existiert in der zentralen Fovea nicht.
# Biologisch: die Foveola wird vollständig vom Midget-System bedient,
# Parasol-Dendriten sind zu groß um dort Platz zu finden.
POP_PARASOL_ON_BIP = 'parasol_ON_bipolar'
POP_PARASOL_OFF_BIP = 'parasol_OFF_bipolar'

# --- KONIOZELLULÄRE BAHN (S-Cone-Bipolar + small-bistratified Ganglion) ---
# Eigene parallele Bahn für Blau-Information, neben Midget und Parasol.
# Pfad: S-Zapfen -> BB-Bipolar (Blue-cone-bistratified) -> small-bistratified
# Ganglion -> LGN K-Schichten. Nur peripher, weil die Foveola praktisch
# keine S-Zapfen hat (gleicher Grund wieso wir Blau zentral schlechter sehen).
# RF-Bauplan nach Dacey 1996: Center = +S, Surround = -(L+M) -> Blue-Yellow-
# Opponenz. Genau das machen wir hier durch zwei Eingangsgewichte: positiv
# vom S-Zapfen, negativ von benachbarten L+M-Zapfen.
POP_KONIO_S_BIP = 'konio_S_bipolar_peripheral'
POP_KONIO_GANG = 'konio_ganglion_peripheral'

# --- AMACRINE ---
POP_AMACRINE_FOVEAL = 'amacrine_foveal'
POP_AMACRINE_PERIPHERAL = 'amacrine_peripheral'

# --- GANGLIEN (Output der Retina -> Sehnerv -> LGN) ---
# Das sind die einzigen Populationen, die ein Downstream-Target
# (z.B. LGN) direkt abgreifen sollte. Siehe get_output_populations().
POP_MIDGET_ON_GANG_FOVEAL = 'midget_ON_ganglion_foveal'
POP_MIDGET_OFF_GANG_FOVEAL = 'midget_OFF_ganglion_foveal'
POP_MIDGET_ON_GANG_PERIPHERAL = 'midget_ON_ganglion_peripheral'
POP_MIDGET_OFF_GANG_PERIPHERAL = 'midget_OFF_ganglion_peripheral'

POP_PARASOL_ON_GANG = 'parasol_ON_ganglion'
POP_PARASOL_OFF_GANG = 'parasol_OFF_ganglion'

BIPOLAR_POPS = [
    POP_MIDGET_ON_BIP_FOVEAL, POP_MIDGET_OFF_BIP_FOVEAL,
    POP_MIDGET_ON_BIP_PERIPHERAL, POP_MIDGET_OFF_BIP_PERIPHERAL,
    POP_PARASOL_ON_BIP, POP_PARASOL_OFF_BIP,
    POP_KONIO_S_BIP,
]

GANGLION_POPS = [
    POP_MIDGET_ON_GANG_FOVEAL, POP_MIDGET_OFF_GANG_FOVEAL,
    POP_MIDGET_ON_GANG_PERIPHERAL, POP_MIDGET_OFF_GANG_PERIPHERAL,
    POP_PARASOL_ON_GANG, POP_PARASOL_OFF_GANG,
    POP_KONIO_GANG,
]

# Erlaubte Input-Auflösungen: Zweierpotenzen von 32 bis 16384.
# Zweierpotenz-Beschränkung ist kein biologisches Muss, sondern eine
# Safeguard gegen versehentliche krumme Werte aus Godot. Bei Bedarf lockern.
VALID_INPUT_RESOLUTIONS = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}


# ============================================================================
#  SECTION 3: GEOMETRIE-HELPER
# ============================================================================
#
# Positionsgeneratoren arbeiten in lokalen Koordinaten um (0,0,0).
# Der origin-Offset wird erst nach der Generierung in der Retina-Klasse
# aufaddiert — so bleiben Nachbarschaftsberechnungen konsistent, weil
# kNN translations-invariant ist.


def uniform_cap_points(n: int, r: float, angle_deg: float,
                       noise: float = 0.0) -> np.ndarray:
    """Gleichmäßig verteilte Punkte auf einer Kugelkappe (Foveola).

    Fibonacci-Spirale auf der Kugel: bringt n Punkte weitgehend
    gleichverteilt auf die Kappe zwischen y=cos(angle) und y=1.
    Alternative wäre zufälliges Sampling, aber Fibonacci hat deutlich
    geringere Klumpen-Varianz bei kleinen n — wichtig bei ~150 Zapfen
    in der Foveola, wo Zufall sichtbare Löcher hinterlassen würde.
    """
    if n <= 0:
        return np.zeros((0, 3))

    angle_rad = np.radians(angle_deg)
    # Die Kappe reicht in y-Richtung von cos(angle) bis 1 (Nordpol).
    # Y ist die "nach vorne"-Achse: Fovea schaut in +y-Richtung.
    y_min = np.cos(angle_rad)
    y_max = 1.0
    # Goldener Winkel ~137.5° — Fibonacci-Spiralkonstante für gleichmäßige
    # Verteilung ohne offensichtliche Streifen/Ringe.
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    pts = np.zeros((n, 3))
    for i in range(n):
        # t läuft von 0 bis 1 -> y läuft von y_max (Pol) bis y_min (Rand).
        t = i / (n - 1) if n > 1 else 0.5
        y = y_max - t * (y_max - y_min)
        theta = golden_angle * i
        # Noise simuliert biologische Unregelmäßigkeit. Foveale Zapfen sind
        # sehr regelmäßig gepackt, deswegen wird noise hier meist gering
        # gewählt (noise * 0.3 im Aufrufer).
        if noise > 0:
            y = np.clip(y + np.random.uniform(-1, 1) * noise, y_min, y_max)
            theta += np.random.uniform(-1, 1) * noise * np.pi
        # rxz = Radius in der xz-Ebene bei gegebenem y (Pythagoras auf
        # Einheitskugel), dann mit r skaliert.
        rxz = np.sqrt(max(0, 1.0 - y ** 2))
        pts[i] = [np.cos(theta) * rxz * r, y * r, np.sin(theta) * rxz * r]

    return pts


def uniform_sphere_band(n: int, r: float, a0_deg: float, a1_deg: float,
                         noise: float = 0.0) -> np.ndarray:
    """Gleichmäßig verteilte Punkte auf einem Kugelband (Peripherie).

    Gleiche Fibonacci-Logik wie uniform_cap_points, aber mit UNTERER und
    OBERER Winkelgrenze statt bis zum Pol. Das Band entspricht dem Bereich
    zwischen Foveola-Rand (a0 ~5°) und Iris-Rand (a1 ~120°).

    Warum nicht bis 180°? Biologisch endet die Retina an der ora serrata
    (~110°); dahinter sitzt die Iris.
    """
    if n <= 0:
        return np.zeros((0, 3))

    # cos() ist monoton fallend auf [0, pi], also y_max bei kleinerem Winkel.
    y_max = np.cos(np.radians(a0_deg))
    y_min = np.cos(np.radians(a1_deg))
    if y_min >= y_max:
        raise ValueError(f"Ungültige Winkel: {a0_deg}° / {a1_deg}°")

    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    pts = np.zeros((n, 3))
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.5
        y = y_max - t * (y_max - y_min)
        theta = golden_angle * i
        if noise > 0:
            y = np.clip(y + np.random.uniform(-1, 1) * noise, y_min, y_max)
            theta += np.random.uniform(-1, 1) * noise * np.pi
        rxz = np.sqrt(max(0, 1.0 - y ** 2))
        pts[i] = [np.cos(theta) * rxz * r, y * r, np.sin(theta) * rxz * r]

    return pts


def radial_project(points: np.ndarray, new_radius: float,
                    center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Projiziert Punkte auf eine Kugel mit neuem Radius.
    Wenn center gegeben: Projektion um dieses Zentrum, sonst um (0,0,0).

    Genutzt um Bipolars/Ganglien GENAU UNTER den fovealen Zapfen zu
    platzieren (1:1:1-Alignment). Jeder foveale Zapfen wirft sozusagen
    einen Schatten nach innen durch die retinalen Schichten.
    """
    if len(points) == 0:
        return points
    if center is None:
        center = np.zeros(3)
    relative = points - center
    norms = np.linalg.norm(relative, axis=1, keepdims=True)
    # Division-by-zero Schutz: Punkt im Zentrum -> bleibt im Zentrum.
    norms[norms == 0] = 1.0
    return center + relative * (new_radius / norms)


def nearest_k_indices(sources: np.ndarray, targets: np.ndarray,
                       k: int) -> np.ndarray:
    """
    Für jede Target-Position: Indizes der k nächsten Sources.

    Herzstück der räumlichen Konnektivität. KD-Tree statt naiver
    Distanzmatrix, weil wir schnell in die 10k+ Knoten-Region kommen.

    WICHTIG: kNN ist translations-invariant. Wenn sources und targets
    beide um denselben Offset verschoben sind, bleiben die Nachbarschaften
    identisch. Das macht origin-Offset unproblematisch — wir können
    Positionen vor ODER nach dem Offset-Addieren vergleichen.
    """
    if len(sources) == 0 or len(targets) == 0:
        return np.zeros((0, k), dtype=int)

    from scipy.spatial import cKDTree
    # k kann nicht größer als die Anzahl Sources sein (kDTree würde sonst
    # NaN/-1 zurückgeben, je nach scipy-Version).
    k = min(k, len(sources))
    tree = cKDTree(sources)
    _, idx = tree.query(targets, k=k)
    # cKDTree gibt bei k=1 ein 1D-Array zurück — wir wollen konsistent 2D.
    if k == 1:
        idx = idx.reshape(-1, 1)
    return idx


# ============================================================================
#  SECTION 4: DEFAULT-PARAMETER
# ============================================================================

DEFAULT_PARAMS: Dict = {

    # --- GEOMETRIE ---
    # Abstrakte Weltkoordinaten (keine realen mm/µm). radius=2.0 ist eine
    # bequeme Einheit für KD-Tree-Distanzen und die 3D-Visualisierung.
    'radius': 2.0,
    # Foveola = Zentrum der Fovea, höchste Zapfendichte. 5° ist der
    # biologische Wert für den reinen Zapfen-Bereich ohne Stäbchen.
    'foveola_angle_deg': 5.0,
    # Iris-Rand bei ~120°: bis dahin reicht die funktionale Retina.
    'iris_angle_deg': 120.0,
    # Positions-Jitter. 0.02 = 2% Rauschen -> sichtbar aber nicht chaotisch.
    'noise': 0.02,

    # Origin-Offset für mehrere Agenten
    # Default (0,0,0). Für mehrere Retinas: Abstände >= 3 * radius empfohlen,
    # damit sich die Kugeln nicht überlappen.
    'origin': (0.0, 0.0, 0.0),

    # Layer-Stacking: jede innere Schicht bekommt etwas kleineren Radius.
    # radius_step=0.02 -> Layer i hat Radius r*(1 - 0.02*i), aber gekappt
    # bei max_radius_delta damit die Ganglien nicht im Kugelzentrum landen.
    'radius_step': 0.02,
    'max_radius_delta': 0.25,

    # --- POPULATIONSGRÖSSEN (primäre Steuerung) ---
    # Skalierbar — für Debugging klein lassen, für finale Runs hochdrehen.
    # Biologisch hat ein menschliches Auge ~6 Mio Zapfen + 120 Mio Stäbchen;
    # diese Werte sind stark runterskaliert aber biologisch plausibel im
    # Verhältnis (Rods ca. 10-20x Zapfenzahl).
    'n_cones_foveal': 150,
    'n_cones_peripheral': 600,
    'n_S_cones': 60,           # Nur peripher. S-Anteil bei ~5-10% der Zapfen.
    'n_rods': 6000,

    # L-Zapfen-Anteil (Rot-empfindlich) unter L+M. Biologisch ~60-70%,
    # daher 0.667. Der Rest wird M.
    'L_fraction': 0.667,

    # --- KONVERGENZ ---
    # Fovea: 1:1:1 — das ist DER strukturelle Grund für hohe Sehschärfe.
    'k_cones_per_midget_bipolar_foveal': 1,
    'k_bipolars_per_midget_ganglion_foveal': 1,
    # Peripher: mehrere Zapfen konvergieren. Auflösung runter, SNR rauf.
    'k_cones_per_midget_bipolar_peripheral': 3,
    'k_bipolars_per_midget_ganglion_peripheral': 3,
    # Parasol: starke Konvergenz -> große rezeptive Felder, schnell/grob.
    'k_cones_per_parasol_bipolar': 8,
    'k_bipolars_per_parasol_ganglion': 6,

    # Konio: 1 S-Zapfen direkt als Center (via Feeder-Strom analog zu Midget),
    # Yellow-Surround kommt NICHT direkt von L+M-Zapfen — die sind parrot_neurons
    # und spiken nicht — sondern von Midget-ON-Bipolaren peripheral, die L+M
    # bereits encoden und spiken. k_midget_surround = wie viele Midget-Bipolare
    # auf einen Konio-Bipolar inhibitorisch wirken. 2 Bipolars konvergieren
    # auf 1 Konio-Ganglion.
    'k_S_cones_per_konio_bipolar': 1,
    'k_midget_surround_per_konio_bipolar': 2,
    'k_konio_bipolars_per_konio_ganglion': 2,

    # Horizontal-RF-Größen. Peripher größer, weil Horizontals dort weite
    # laterale Reichweite haben (Gap Junctions in realer Retina).
    'k_cones_per_horizontal_foveal': 5,
    'k_cones_per_horizontal_peripheral': 12,
    # Wie viele Horizontals inhibieren EIN Bipolar (Surround-Stärke).
    'k_horizontals_surround_per_cone': 3,

    # Amacrine-Konnektivität: mehrere Bipolars -> Amacrine,
    # und eine Amacrine inhibiert mehrere Ganglien + Feedback auf Bipolars.
    'k_bipolars_per_amacrine': 8,
    'k_amacrines_per_ganglion': 5,
    'k_amacrines_per_bipolar_feedback': 3,

    # --- POPULATIONS-RATIOS ---
    # Parasol macht nur ~10% der Ganglien aus (biologisch realistisch).
    'parasol_to_midget_ratio_peripheral': 0.10,
    # Konio macht ca. 8-10% der Ganglien aus, je nach Spezies. Eigentlich
    # 1:1 mit S-Zapfen pro Bipolar, dann ~2:1 zu Ganglion -> Anzahl folgt
    # direkt aus n_S_cones.
    'konio_ganglion_to_bipolar_ratio': 0.5,
    # Horizontal-Dichte: 1 Horizontal pro ~5-10 Zapfen.
    'horizontal_to_cone_ratio_foveal': 0.2,
    'horizontal_to_cone_ratio_peripheral': 0.1,
    # Amacrine: ~1/3 der Bipolar-Anzahl.
    'amacrine_to_bipolar_ratio': 0.3,

    # --- RIBBON-SYNAPSEN ---
    # Echte Bipolars haben Ribbon-Synapsen mit 10-100 Vesikelfreisetzungen
    # pro AP. Wir emulieren das durch mehrfache Connect-Aufrufe
    # (n_ribbon parallele Synapsen mit vollem Gewicht).
    # Validiert bei 20 -> gesunde Midget/Parasol-Feuerraten (10-50 Hz).
    'ribbon_bipolar_to_ganglion': 20,
    'ribbon_horizontal': 3,
    'ribbon_amacrine': 3,
    # Konio-Bahn ist anatomisch dünner als Midget — entsprechend kleinerer
    # ribbon-Faktor. 10 ist ein konservativer Mittelwert.
    'ribbon_konio': 10,

    # --- SYNAPTISCHE GEWICHTE (nS) ---
    # Positive = exzitatorisch, negative = inhibitorisch. In NEST werden
    # inhibitorische Synapsen über negative Gewichte bei iaf_cond_exp
    # realisiert (Umleitung auf tau_syn_in).
    'w_horizontal_to_cone': -0.5,        # Surround-Inhibition
    'w_bipolar_to_ganglion_midget': 3.0, # Midget: starker einzelner Input
    'w_bipolar_to_ganglion_parasol': 2.5,# Parasol: viele Inputs, etwas schwächer
    'w_bipolar_to_amacrine': 1.5,
    'w_amacrine_to_ganglion': -1.0,
    'w_amacrine_to_bipolar': -0.8,
    # Konio: Center-Strom kommt via Feeder (S-Cone-Helligkeit -> step_current),
    # genau wie bei Midget. Der Yellow-Surround entsteht durch INHIBITORISCHE
    # Synapsen von Midget-ON-Bipolaren peripheral (die L+M-Helligkeit codieren
    # und spiken). w_midget_to_konio_bipolar negativ -> wenn L+M-Region hell,
    # wird Konio-Bipolar gehemmt. Das erzeugt direkt am Bipolar Blue-Yellow-
    # Opponenz (Dacey 1996). w_konio_bip_to_ganglion entspricht Midget-Stärke;
    # die ribbon-Multiplicity ist hier kleiner.
    'w_midget_to_konio_bipolar': -1.5,
    'w_konio_bipolar_to_ganglion': 3.0,

    # --- SIMULATION ---
    # NEST min_delay: Untergrenze für alle Synapsen-Delays.
    # 1.0 ms ist konservativ und erlaubt grobes Parallelisieren.
    'min_delay': 1.0,

    # --- GRID-BUCKET-MAPPING ---
    # Auflösung der internen Pixel-Bucket-Maps. Default 32x32 matcht das
    # cortikale Adapter-Grid in V1-V6. Über get_grid_maps(resolution=(H,W))
    # können on-the-fly auch andere Auflösungen berechnet werden.
    'bucket_resolution': (32, 32),
}


DEFAULT_NEURON_PARAMS: Dict = {

    # Photorezeptoren: parrot_neurons in NEST.
    # parrot_neuron leitet eingehende Spikes 1:1 weiter, hat selbst keine
    # Dynamik. Wir brauchen sie nur als Positions-Anker — echte Photo-
    # rezeptoren spiken ohnehin nicht (graded potentials), der visuelle
    # Input geht über step_current_generator DIREKT in die Bipolars.

    # Horizontal: Standard iaf_cond_exp (LIF mit konduktanz-basierten Syn.)
    # Kein Adaptationsterm nötig — Horizontals feuern eher kontinuierlich
    # und modulieren lateral.
    'horizontal': {
        "C_m": 250.0, "g_L": 8.33,      # tau_m = C_m/g_L ~ 30 ms (langsam)
        "E_L": -70.0, "E_ex": 0.0, "E_in": -85.0,
        "V_th": -57.0, "V_reset": -70.0, "t_ref": 2.0,
        "tau_syn_ex": 3.0, "tau_syn_in": 8.0,
        "I_e": 0.0,
    },

    # Bipolar ON: Standard LIF.
    # Photorezeptor-Input kommt als Strom (step_current_generator),
    # nicht als synaptische Events.
    'bipolar_on': {
        # Kein Dark-Current — Strom kommt via step_current_generator
        # V_th auf -58 mV gesenkt -> Rheobase = g_L * (V_th - E_L) = 20 * 12 = 240 pA
        # Zusammen mit max_current_pa=800 im Feeder ergibt das einen gesunden
        # Arbeitsbereich: typische Bipolars feuern bei 20-70 Hz je nach Input.
        "C_m": 200.0, "g_L": 20.0,      # tau_m = 10 ms (schnell genug für Frames)
        "E_L": -70.0, "E_ex": 0.0, "E_in": -85.0,
        "V_th": -58.0, "V_reset": -70.0, "t_ref": 2.0,
        "tau_syn_ex": 1.5, "tau_syn_in": 4.0,
        "I_e": 0.0,
    },

    # Bipolar OFF: IDENTISCHE Neuron-Parameter wie ON.
    # Der Unterschied liegt NICHT im Neuronenmodell sondern im Feeder:
    # _compute_bipolar_amplitudes() invertiert das Vorzeichen für OFF-Zellen.
    # Dadurch bleibt ON/OFF-Symmetrie exakt erhalten.
    'bipolar_off': {
        # Invertierung passiert im Feeder, nicht hier.
        # V_th auf -58 mV gesenkt -> Rheobase 240 pA (symmetrisch zu ON)
        "C_m": 200.0, "g_L": 20.0,
        "E_L": -70.0, "E_ex": 0.0, "E_in": -85.0,
        "V_th": -58.0, "V_reset": -70.0, "t_ref": 2.0,
        "tau_syn_ex": 1.5, "tau_syn_in": 4.0,
        "I_e": 0.0,
    },

    # Amacrine: Adaptive Exponential IaF (aeif_cond_exp).
    # Der Adaptations-Term w (über a, b, tau_w) ist hier WICHTIG — echte
    # Amacrines zeigen starke Spike-Frequency-Adaptation, das erzeugt
    # transiente Antworten auf Bewegung/Kontraständerung.
    'amacrine': {
        "C_m": 200.0, "g_L": 10.0,
        "E_L": -70.0, "V_th": -55.0, "V_reset": -60.0, "t_ref": 2.0,
        # a = sub-threshold adaptation, b = spike-triggered adaptation jump,
        # tau_w = wie schnell w zurückkehrt. Delta_T = exponential slope.
        "a": 2.0, "b": 40.0, "tau_w": 100.0, "Delta_T": 2.0, "V_peak": 0.0,
        "E_ex": 0.0, "E_in": -85.0,
        "tau_syn_ex": 2.0, "tau_syn_in": 5.0,
    },

    # Midget-Ganglien: AdEx. Stärkere Adaptation als Amacrine,
    # liefert saubere sustained/transient-Antworten.
    'midget_ganglion': {
        "C_m": 250.0, "g_L": 12.0,
        "E_L": -70.0, "V_th": -52.0, "V_reset": -60.0, "t_ref": 2.0,
        "a": 4.0, "b": 80.0, "tau_w": 150.0, "Delta_T": 2.0, "V_peak": 0.0,
        "E_ex": 0.0, "E_in": -85.0,
        "tau_syn_ex": 2.0, "tau_syn_in": 5.0,
    },

    # Parasol-Ganglien: AdEx, aber SCHNELLER (kleineres tau_w, kürzere
    # t_ref, kleineres tau_syn). Das matcht die biologische Rolle
    # "schnelle Bewegungsdetektion" -> hohe Feuerraten bei Flickern.
    'parasol_ganglion': {
        "C_m": 400.0, "g_L": 20.0,
        "E_L": -70.0, "V_th": -50.0, "V_reset": -60.0, "t_ref": 1.0,
        "a": 2.0, "b": 60.0, "tau_w": 80.0, "Delta_T": 2.0, "V_peak": 0.0,
        "E_ex": 0.0, "E_in": -85.0,
        "tau_syn_ex": 1.0, "tau_syn_in": 3.0,
    },
}


DEFAULT_FEEDER_CONFIG: Dict = {
    'generator_type': 'step_current',
    'input_resolution': (64, 64),
    'max_current_pa': 800.0,     # Doppelt so hoch wie vorher:
                                  # typischer Pixel (0.4) -> 320 pA (80 pA über Rheobase)
                                  # heller Pixel (0.8) -> 640 pA (400 pA über Rheobase)
    'max_rate_hz': 4000.0,
    'poisson_weight_nS': 2.0,
    'contrast_gain': 2.0,         # Verstärkung der Mittel-adaptierten Kontraste.
                                  # 1.0 = nur Mittel abziehen, 2.0 = doppelt verstärkt,
                                  # höher = empfindlicher für kleine Kontraste.
}


def _validate_resolution(resolution: Tuple[int, int]) -> None:
    """Prüft dass resolution aus Zweierpotenzen zwischen 32 und 16384 besteht.

    Warnt ab 2048x2048 -> ~4 Mio Pixel pro Frame, spürbare Kosten im
    feed()-Sampling. Kein hartes Limit, nur ein Hinweis an den User.
    """
    if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
        raise ValueError(f"resolution muss (H, W) Tupel sein, bekam {resolution}")
    h, w = resolution
    for val, name in [(h, 'H'), (w, 'W')]:
        if val not in VALID_INPUT_RESOLUTIONS:
            raise ValueError(
                f"Ungültige {name}={val}. Erlaubt sind Zweierpotenzen "
                f"von 32 bis 16384: {sorted(VALID_INPUT_RESOLUTIONS)}"
            )
    if h * w >= 2048 * 2048:
        print(f"WARNING: input_resolution {resolution} ist sehr groß "
              f"({h*w:,} Pixel). Erwarte hohen Speicherverbrauch pro Frame.")


# ============================================================================
#  SECTION 5: INPUT-FEEDER
# ============================================================================
#
# Der RetinaInputFeeder ist die externe Schnittstelle für visuellen Input.
# Er existiert als eigenständiges Objekt, das von Retina.create_input_feeder()
# erzeugt wird. Einmal erzeugt kann er außerhalb der Retina-Klasse benutzt
# werden — z.B. in einer Godot-Loop, die einzelne Frames reinreicht.
#
# Der Feeder kapselt:
#   - NEST-Generator-NodeCollections (step_current oder poisson)
#   - Verbindungen Generator -> Bipolar
#   - Mapping Bipolar -> Photorezeptor-Indizes
#   - Sampling-Logik (3D-Position -> Pixel im Eingangsbild)
#   - Polaritäts-Logik (ON sieht Helligkeit, OFF sieht Dunkelheit)


class RetinaInputFeeder:
    """
    Externe Input-Schnittstelle für eine Retina.

    Wird von Retina.create_input_feeder(config) erzeugt. Danach eigenständig
    nutzbar — hält keine Referenz auf das Retina-Objekt.

    Thread-safety: nein.

    Datenfluss pro feed():
      Frame (H,W,3) + Intensity (H,W)
        -> ggf. resizen auf config['input_resolution']
        -> pro Photorezeptor den Pixel-Wert an seiner 3D-Position samplen
        -> pro Bipolar den Mittelwert seiner zugeordneten Photorezeptoren
        -> Weber-Fechner-Adaptation: Abweichung vom globalen Bild-Mittel
        -> ON vs OFF: Vorzeichen umdrehen für OFF
        -> Amplitude [0,1] auf Strom (pA) oder Rate (Hz) skalieren
        -> in die NEST-Generatoren schreiben
    """

    def __init__(self,
                 config: Dict,
                 bipolar_populations: Dict[str, 'nest.NodeCollection'],
                 bipolar_to_photo_mapping: Dict[str, List],
                 origin: np.ndarray,
                 world_radius: float,
                 min_delay: float):
        # User-Config über die Defaults drüberlegen.
        self.config = {**DEFAULT_FEEDER_CONFIG, **config}
        _validate_resolution(self.config['input_resolution'])

        # Alles was wir von der Retina brauchen wird HIER reingereicht —
        # danach ist der Feeder unabhängig.
        self._bipolar_pops = bipolar_populations
        self._mapping = bipolar_to_photo_mapping
        self._origin = np.asarray(origin, dtype=float)
        self._world_radius = world_radius
        self._min_delay = min_delay

        self.generator_type = self.config['generator_type']
        if self.generator_type not in ('step_current', 'poisson'):
            raise ValueError(
                f"generator_type muss 'step_current' oder 'poisson' sein, "
                f"bekam '{self.generator_type}'"
            )

        # generators: pro Bipolar-Population eine NodeCollection mit
        # EINEM Generator pro Bipolar-Neuron (1:1 Verbindung).
        self.generators: Dict[str, 'nest.NodeCollection'] = {}
        # Photo-Positionen werden per set_photo_positions() nachgereicht,
        # weil sie ans feed()-Zeitpunkt-Sampling gebunden sind (nicht ans Setup).
        self._photo_positions_cache: Optional[Dict[str, np.ndarray]] = None

        self._build_generators()

    def _build_generators(self):
        """Erzeugt Generator-Knoten und verbindet sie mit den Bipolars.

        Zwei Modi:
          step_current: direkte Stromeinspeisung in pA. Deterministisch,
                        perfekt für Kontrolle der Feuerraten.
          poisson: stochastische Spike-Events mit variabler Rate.
                   Biologischer, aber mehr Varianz pro Trial.
        """
        gen_model = ('step_current_generator'
                     if self.generator_type == 'step_current'
                     else 'poisson_generator')

        for bip_name, bip_pop in self._bipolar_pops.items():
            n = len(bip_pop)
            if n == 0:
                continue

            # EIN Generator pro Bipolar-Neuron — so kann jeder Bipolar
            # individuell je nach Pixel-Input angesteuert werden.
            gens = nest.Create(gen_model, n)
            self.generators[bip_name] = gens

            if self.generator_type == 'step_current':
                # step_current_generator injiziert einen Strom, kein Spike.
                # weight=1.0 lässt den gesetzten Ampere-Wert unverändert
                # durchlaufen (Skalierung passiert über amplitude_values).
                nest.Connect(
                    gens, bip_pop, 'one_to_one',
                    syn_spec={'weight': 1.0, 'delay': self._min_delay},
                )
            else:
                # poisson_generator: Spikes mit Poisson-verteiltem Inter-
                # Spike-Intervall. Hier skaliert das Gewicht die synaptische
                # Stärke pro Event.
                weight = self.config['poisson_weight_nS']
                nest.Connect(
                    gens, bip_pop, 'one_to_one',
                    syn_spec={'weight': weight, 'delay': self._min_delay},
                )

    def set_photo_positions(self, photo_positions: Dict[str, np.ndarray]):
        """
        Wird von Retina.create_input_feeder() aufgerufen um die Positionen
        der Photorezeptoren zu übergeben (für Frame-Sampling).
        """
        self._photo_positions_cache = photo_positions

    def feed(self, lms_frame: np.ndarray, intensity_frame: np.ndarray) -> None:
        """
        Überträgt EIN Frame auf die Retina.

        Args:
            lms_frame: (H, W, 3) Array mit L/M/S-Kanälen, Werte in [0, 1]
            intensity_frame: (H, W) Array, Werte in [0, 1]

        Auflösung des Frames darf von der Feeder-Config abweichen — wird
        dann intern via scipy.ndimage.zoom angepasst. Gleiche Auflösung ist
        effizienter.

        Call-Pattern im closed-loop:
          for frame in godot_frames:
              feeder.feed(lms, intensity)
              nest.Simulate(sim_step_ms)   # typisch 20-50 ms
              spikes = recorder.get(...)
        """
        if self._photo_positions_cache is None:
            raise RuntimeError(
                "Photo-Positionen nicht gesetzt. Feeder muss über "
                "Retina.create_input_feeder() erzeugt werden."
            )

        # Auflösungs-Mismatch? Dann resizen. Kostet extra Rechenzeit —
        # deswegen am besten Godot-Seite direkt in der Ziel-Auflösung rendern.
        target_res = self.config['input_resolution']
        if lms_frame.shape[:2] != target_res:
            lms_frame, intensity_frame = self._resize_frame(
                lms_frame, intensity_frame, target_res
            )

        # Zwei-Phasen-Pipeline:
        # 1. Sampling: welcher Photorezeptor sieht welchen Pixel?
        # 2. Update: pro Bipolar Mittel bilden, ON/OFF flippen, in NEST pushen.
        photo_values = self._sample_frame(lms_frame, intensity_frame)

        if self.generator_type == 'step_current':
            self._update_step_current(photo_values)
        else:
            self._update_poisson(photo_values)

    def _resize_frame(self, lms_frame, intensity_frame, target_res):
        """Skaliert Frames auf die Ziel-Auflösung mit bilinearer Interpolation.

        order=1 = bilinear. Bikubisch (order=3) wäre schöner, aber hier
        geht's um den Input für die Retina — bilinear reicht dicke.
        """
        from scipy.ndimage import zoom
        src_h, src_w = lms_frame.shape[:2]
        tgt_h, tgt_w = target_res
        sh = tgt_h / src_h
        sw = tgt_w / src_w
        # (sh, sw, 1) für lms: 1 = keine Skalierung auf der Channel-Achse.
        lms_resized = zoom(lms_frame, (sh, sw, 1), order=1)
        intensity_resized = zoom(intensity_frame, (sh, sw), order=1)
        return lms_resized, intensity_resized

    def _sample_frame(self, lms_frame, intensity_frame) -> Dict[str, np.ndarray]:
        """
        Sampelt den Frame an jeder Photorezeptor-Position.

        3D -> 2D Projektion: X,Z auf Bildebene, origin abgezogen damit
        das Sampling relativ zur Retina funktioniert.

        VEREINFACHUNG: Wir projizieren direkt orthogonal, statt eine
        echte Linse + Strahlengang zu simulieren. Das heißt: kein
        Bildumkehrung wie in einer realen Optik, kein Sehfeld-Winkel,
        kein Blendenmodell. Der Input-Frame wird einfach in uv-Koordinaten
        auf die xz-Ebene gemappt. Für closed-loop RL reicht das —
        die Agenten lernen ja die Pixel-zu-Retina-Beziehung implizit.

        Nur X und Z werden genutzt; Y (Tiefe) wird ignoriert. Die Kappe
        ist eine Halbkugel in +y, also haben alle Photorezeptoren
        ähnliche y-Werte und unterschiedliche x/z — das mappt sauber
        auf ein 2D-Bild.
        """
        H, W = lms_frame.shape[:2]
        world_r = self._world_radius
        origin = self._origin

        def sample(positions: np.ndarray, channel_idx: Optional[int]) -> np.ndarray:
            if len(positions) == 0:
                return np.zeros(0)
            # Relativ zum Retina-Zentrum (wichtig für origin != 0).
            rel = positions - origin
            x = rel[:, 0]
            z = rel[:, 2]
            # Mapping [-world_r, +world_r] -> [0, 1]
            u = (x / world_r + 1.0) / 2.0
            v = (z / world_r + 1.0) / 2.0
            # Diskrete Pixel-Koordinaten, geclippt damit Randpunkte nicht out-of-bounds gehen.
            col = np.clip((u * W).astype(int), 0, W - 1)
            row = np.clip((v * H).astype(int), 0, H - 1)
            # channel_idx None -> Intensitätsbild (für Rods, die achromatisch sind)
            if channel_idx is None:
                return intensity_frame[row, col]
            # Sonst: L=0 (rot-Kanal), M=1 (grün-Kanal), S=2 (blau-Kanal)
            return lms_frame[row, col, channel_idx]

        pp = self._photo_positions_cache
        return {
            POP_L_FOVEAL: sample(pp[POP_L_FOVEAL], 0),
            POP_M_FOVEAL: sample(pp[POP_M_FOVEAL], 1),
            POP_L_PERIPHERAL: sample(pp[POP_L_PERIPHERAL], 0),
            POP_M_PERIPHERAL: sample(pp[POP_M_PERIPHERAL], 1),
            POP_S_PERIPHERAL: sample(pp[POP_S_PERIPHERAL], 2),
            POP_RODS: sample(pp[POP_RODS], None),
        }

    def _compute_bipolar_amplitudes(self,
                                      photo_values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Pro Bipolar-Population: Array [0,1] von Aktivierungen.

        ADAPTATION AUF GLOBALEN BILDMITTELWERT (Weber-Fechner-artig):
        Statt absolute Helligkeit zu codieren, codieren beide Pfade die
        ABWEICHUNG vom Bildmittelwert. Das macht die Retina kontrast-
        sensitiv (statt helligkeitssensitiv) und symmetriert ON/OFF auch
        bei stark schiefen Helligkeitsverteilungen (z.B. dunkle Videos).

        ON-Bipolar:  amp = (intensity - mean) * gain + 0.5  → feuert wenn lokal heller
        OFF-Bipolar: amp = (mean - intensity) * gain + 0.5  → feuert wenn lokal dunkler

        Bei einem komplett uniformen Bild liegen beide Pfade bei amp=0.5
        (Baseline). Bei Kontrast-Patches reagieren ON und OFF symmetrisch.

        Der gain-Faktor bestimmt wie stark Kontraste verstärkt werden.
        gain=2.0 bedeutet: ein Pixel der 25% über dem Mittel liegt sättigt
        das ON-Signal voll aus.
        """
        # --- Schritt 1: Globaler Bildmittelwert ---
        # Über ALLE Photorezeptor-Werte gemittelt (inkl. Rods), damit L/M/S-
        # Zapfen und Stäbchen dieselbe Baseline sehen. Das ist eine grobe
        # Vereinfachung der realen Weber-Adaptation (die im Retina-
        # biologisch eher auf Zell-Ebene stattfindet), funktioniert aber
        # robust gegen Helligkeitssprünge in Game-Frames.
        all_values = np.concatenate([v for v in photo_values.values() if len(v) > 0])
        global_mean = float(all_values.mean()) if len(all_values) > 0 else 0.5

        # --- Schritt 2: Kontrast-Gain aus Config ---
        # Verstärkungsfaktor: 2.0 ist ein vernünftiger Startpunkt,
        # höher = empfindlicher, niedriger = robuster gegen Rauschen.
        gain = self.config.get('contrast_gain', 2.0)

        result = {}
        # --- Schritt 3: Pro Bipolar-Population ---
        for bip_name, mapping in self._mapping.items():
            # Skip wenn kein Generator für diese Population existiert
            # (z.B. wenn Foveal-Parasol deaktiviert wäre — aktuell nie der Fall).
            if bip_name not in self.generators:
                continue

            # Lokaler Mittelwert: für jeden Bipolar den Durchschnitt seiner
            # zugeordneten Photorezeptoren. mapping[i] = Liste von
            # (pop_name, photo_idx)-Tupeln.
            n_bip = len(mapping)
            local_means = np.zeros(n_bip)
            for bip_idx, photo_list in enumerate(mapping):
                if not photo_list:
                    continue
                vals = [photo_values[pop_name][idx] for pop_name, idx in photo_list]
                local_means[bip_idx] = np.mean(vals)

            # --- Schritt 4: ON/OFF-Inversion ---
            # Einzige Stelle im Code, wo ON vs OFF unterschieden wird!
            # Alles andere (Neuronenparameter, Konnektivität) ist symmetrisch.
            if 'OFF' in bip_name:
                amplitudes = (global_mean - local_means) * gain + 0.5
            else:
                amplitudes = (local_means - global_mean) * gain + 0.5

            # Auf [0, 1] clippen damit der Strom nicht negativ wird
            # (negativer Strom = hyperpolarisierend, würde Bipolar stummschalten).
            amplitudes = np.clip(amplitudes, 0.0, 1.0)
            result[bip_name] = amplitudes

        return result

    def _update_step_current(self, photo_values: Dict[str, np.ndarray]):
        """Schreibt neue Strom-Amplituden in die step_current_generators."""
        amplitudes_all = self._compute_bipolar_amplitudes(photo_values)

        max_current = self.config['max_current_pa']
        # WICHTIG: amplitude_times muss in der Zukunft liegen, sonst
        # verwirft NEST den Eintrag. Deswegen +min_delay auf biological_time.
        current_time = nest.biological_time
        next_time = current_time + self._min_delay

        for bip_name, amps in amplitudes_all.items():
            gens = self.generators[bip_name]
            # Skalierung: [0,1] -> [0, max_current_pa] in pA.
            currents = (amps * max_current).astype(float)
            # Jeder Generator bekommt seinen individuellen Wert.
            # TODO: das ist eine Python-Schleife über ~n Generatoren —
            # könnte für große Populationen gebündelt werden, aber
            # NEST's set() hat keine einfache vektorisierte API für
            # step_current_generator-Listen.
            for i, gen in enumerate(gens):
                gen.set(
                    amplitude_times=[next_time],
                    amplitude_values=[float(currents[i])],
                )

    def _update_poisson(self, photo_values: Dict[str, np.ndarray]):
        """Schreibt neue Raten in die poisson_generators."""
        amplitudes_all = self._compute_bipolar_amplitudes(photo_values)
        max_rate = self.config['max_rate_hz']

        for bip_name, amps in amplitudes_all.items():
            gens = self.generators[bip_name]
            # Skalierung: [0,1] -> [0, max_rate_hz] in Hz.
            # Hier geht die vektorisierte Version — poisson_generator
            # nimmt eine Liste für 'rate' direkt an.
            rates = (amps * max_rate).astype(float)
            gens.set(rate=rates.tolist())


# ============================================================================
#  SECTION 6: DIE RETINA-KLASSE
# ============================================================================


class Retina:
    """
    Biologisch strukturierte Retina mit räumlicher Konnektivität.

    Nach build() + connect():
        populations     Dict[str, NodeCollection]  — alle NEST-Populationen
        positions       Dict[str, np.ndarray]      — 3D-Positionen inkl origin
        counts          Dict[str, int]             — Populationsgrößen
    """

    # ------------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------------

    def __init__(self,
                 params: Optional[Dict] = None,
                 neuron_params: Optional[Dict] = None,
                 verbose: bool = True):
        self.verbose = verbose

        # Merge-Strategie: User-Params überschreiben Defaults, fehlende Keys
        # kommen aus DEFAULT_PARAMS. So kann der User punktuell einzelne
        # Werte override-n ohne das ganze Dict ausfüllen zu müssen.
        self.params = {**DEFAULT_PARAMS, **(params or {})}

        # Origin validieren — muss (x,y,z) sein, sonst bricht das Multi-Agent-
        # Setup später subtil.
        origin = self.params['origin']
        self._origin = np.asarray(origin, dtype=float)
        if self._origin.shape != (3,):
            raise ValueError(f"origin muss (x,y,z) sein, bekam {origin}")

        # Neuron-Params: pro Zelltyp Deep-Merge. Der User kann z.B. nur
        # {'bipolar_on': {'V_th': -60.0}} reinreichen und bekommt den Rest
        # aus DEFAULT_NEURON_PARAMS['bipolar_on'].
        user_neurons = neuron_params or {}
        self.neuron_params = {}
        for key, defaults in DEFAULT_NEURON_PARAMS.items():
            self.neuron_params[key] = {**defaults, **user_neurons.get(key, {})}

        # Werden in build() befüllt:
        self.populations: Dict[str, 'nest.NodeCollection'] = {}
        self.positions: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, int] = {}

        # Wird in build() via _compute_bipolar_to_photo_mapping gefüllt,
        # und an den Feeder weitergereicht.
        self._bipolar_to_photo_mapping: Dict[str, List] = {}

        # Grid-Bucket-Maps (sparse): pop_name -> dict[(row, col) -> [nest_id]]
        # Wird in build() via _compute_grid_maps() gefüllt.
        self._grid_maps: Dict[str, Dict[Tuple[int, int], List[int]]] = {}
        self._id_to_bucket: Dict[str, Dict[int, Tuple[int, int]]] = {}

        # State-Flags verhindern doppeltes build()/connect() — sonst würden
        # doppelte Neuronen und Synapsen in NEST entstehen.
        self._built = False
        self._connected = False

    # ------------------------------------------------------------------------
    # Populations-Größen aus biologischen Ratios
    # ------------------------------------------------------------------------

    def compute_counts(self) -> Dict[str, int]:
        """Leitet alle Populationsgrößen aus den biologischen Ratios ab.

        Die "primären" Größen sind n_cones_foveal, n_cones_peripheral,
        n_S_cones, n_rods. Alles andere wird als feste Ratio davon berechnet.
        Dadurch reicht im Config das Skalieren von z.B. n_cones_peripheral
        um die ganze Retina zu vergrößern, ohne einzelne Populationen
        aus dem biologischen Gleichgewicht zu bringen.
        """
        p = self.params

        n_fov = p['n_cones_foveal']
        n_periph = p['n_cones_peripheral']
        n_S = p['n_S_cones']
        n_rods = p['n_rods']

        # --- Zapfen-Aufteilung L/M ---
        # L_fraction=0.667 -> 2:1 Verhältnis L:M (biologisch typisch).
        # S wird separat gezählt, ist KEIN Anteil von n_periph.
        L_frac = p['L_fraction']
        n_L_fov = int(round(n_fov * L_frac))
        n_M_fov = n_fov - n_L_fov
        n_L_periph = int(round(n_periph * L_frac))
        n_M_periph = n_periph - n_L_periph

        # --- Midget foveal: 1:1:1 ---
        # Für jeden fovealen Zapfen ein Bipolar und ein Ganglion.
        # Das ist KEINE Design-Entscheidung sondern biologische Realität.
        n_midget_bip_fov = n_fov
        n_midget_gang_fov = n_fov

        # --- Midget peripheral: Konvergenz ---
        # Bipolars konvergieren k_c2b Zapfen, Ganglien konvergieren k_b2g Bipolars.
        # max(1, ...) verhindert Zero-Division bei sehr kleinen Populationen.
        k_c2b = p['k_cones_per_midget_bipolar_peripheral']
        k_b2g = p['k_bipolars_per_midget_ganglion_peripheral']
        n_midget_bip_periph = max(1, n_periph // k_c2b)
        n_midget_gang_periph = max(1, n_midget_bip_periph // k_b2g)

        # --- Parasol ---
        # Nur ~10% der Midget-Ganglien (parasol_to_midget_ratio_peripheral).
        # Bipolar-Anzahl ergibt sich rückwärts aus dem Konvergenzfaktor:
        # wenn 6 Bipolars pro Ganglion nötig sind, brauchen wir 6x so viele Bipolars.
        n_parasol_gang = max(1, int(n_midget_gang_periph *
                                     p['parasol_to_midget_ratio_peripheral']))
        n_parasol_bip = max(1, n_parasol_gang *
                             p['k_bipolars_per_parasol_ganglion'])

        # --- Konio ---
        # 1 Bipolar pro S-Zapfen (Center via 1:1), Konvergenz zu Ganglion mit
        # 'konio_ganglion_to_bipolar_ratio'. Bei n_S=60 ergibt das 60 Bipolare
        # und 30 Ganglien — eine kleine, aber nicht null Population.
        n_konio_bip = max(5, n_S)
        n_konio_gang = max(5, int(round(n_konio_bip *
                                         p['konio_ganglion_to_bipolar_ratio'])))

        # --- Horizontal ---
        # Als Anteil der Zapfenanzahl. Periph schließt S-Zapfen mit ein,
        # weil Horizontals auch S-Input verarbeiten (für S-Off-Pfad).
        n_horiz_fov = max(5, int(n_fov * p['horizontal_to_cone_ratio_foveal']))
        n_horiz_periph = max(10, int((n_periph + n_S) *
                                      p['horizontal_to_cone_ratio_peripheral']))

        # --- Amacrine ---
        # ~30% der Gesamt-Bipolar-Anzahl (ON+OFF zusammen, daher *2).
        # Foveal: nur Midget-Bipolars (Parasol+Konio existieren foveal nicht).
        # Peripheral: Midget + Parasol + Konio. Konio einfach (kein ON/OFF-Split).
        n_amac_fov = max(5, int((n_midget_bip_fov * 2) *
                                 p['amacrine_to_bipolar_ratio']))
        n_amac_periph = max(10, int(
            (n_midget_bip_periph * 2 + n_parasol_bip * 2 + n_konio_bip) *
            p['amacrine_to_bipolar_ratio']
        ))

        self.counts = {
            POP_L_FOVEAL: n_L_fov,
            POP_M_FOVEAL: n_M_fov,
            POP_L_PERIPHERAL: n_L_periph,
            POP_M_PERIPHERAL: n_M_periph,
            POP_S_PERIPHERAL: n_S,
            POP_RODS: n_rods,

            POP_HORIZONTAL_FOVEAL: n_horiz_fov,
            POP_HORIZONTAL_PERIPHERAL: n_horiz_periph,

            # ON und OFF GLEICHE Anzahl — exakte Symmetrie pro Pfad.
            POP_MIDGET_ON_BIP_FOVEAL: n_midget_bip_fov,
            POP_MIDGET_OFF_BIP_FOVEAL: n_midget_bip_fov,
            POP_MIDGET_ON_BIP_PERIPHERAL: n_midget_bip_periph,
            POP_MIDGET_OFF_BIP_PERIPHERAL: n_midget_bip_periph,

            POP_PARASOL_ON_BIP: n_parasol_bip,
            POP_PARASOL_OFF_BIP: n_parasol_bip,

            # Konio: keine ON/OFF-Trennung — der bistratified Bipolar feuert
            # auf Blue-Yellow-Differenz, nicht auf separate Polaritäten.
            POP_KONIO_S_BIP: n_konio_bip,

            POP_AMACRINE_FOVEAL: n_amac_fov,
            POP_AMACRINE_PERIPHERAL: n_amac_periph,

            POP_MIDGET_ON_GANG_FOVEAL: n_midget_gang_fov,
            POP_MIDGET_OFF_GANG_FOVEAL: n_midget_gang_fov,
            POP_MIDGET_ON_GANG_PERIPHERAL: n_midget_gang_periph,
            POP_MIDGET_OFF_GANG_PERIPHERAL: n_midget_gang_periph,

            POP_PARASOL_ON_GANG: n_parasol_gang,
            POP_PARASOL_OFF_GANG: n_parasol_gang,

            POP_KONIO_GANG: n_konio_gang,
        }

        return self.counts

    # ------------------------------------------------------------------------
    # Positions-Generierung mit Origin-Offset
    # ------------------------------------------------------------------------

    def _generate_positions(self) -> Dict[str, np.ndarray]:
        """
        Generiert alle 3D-Positionen lokal um (0,0,0), addiert am Ende origin.
        Layer-Stacking: mehrere Schichten mit abnehmendem Radius.

        LAYER-REIHENFOLGE (von außen nach innen, in Lichtrichtung):
          Layer 0: Photorezeptoren (am weitesten außen — Licht kommt von innen!)
          Layer 1: Horizontal
          Layer 2: Midget-Bipolars
          Layer 3: Parasol-Bipolars
          Layer 4: Amacrine
          Layer 5: Ganglien (innen — Axone bilden den Sehnerv)

        Biologisch sind die Layers natürlich invertiert (Licht muss erst
        durch alle inneren Schichten, bevor es die Photorezeptoren trifft,
        außer in der Fovea wo die Schichten zur Seite geschoben sind).
        Wir machen das hier umgekehrt, weil es für 3D-Visualisierung
        intuitiver ist (Photorezeptoren "schauen" nach außen).
        """
        p = self.params
        c = self.counts
        r = p['radius']
        noise = p['noise']
        fov_a = p['foveola_angle_deg']
        iris_a = p['iris_angle_deg']
        step = p['radius_step']
        max_delta = p['max_radius_delta']

        def layer_radius(layer_idx: int) -> float:
            # Jeder Layer liegt step*r weiter innen als der vorige,
            # aber insgesamt nicht mehr als max_delta vom Außenradius.
            # Das verhindert, dass die Ganglien-Schicht im Zentrum kollabiert.
            delta = min(layer_idx * step * r, max_delta)
            return r - delta

        pos = {}

        # --- Layer 0: Photorezeptoren ---
        # Fovea = reine Zapfen (Stäbchen peripher). S-Zapfen nur peripher.
        # noise * 0.3 foveal: dichte regelmäßige Packung, kaum Jitter.
        r_photo = layer_radius(0)
        pos[POP_L_FOVEAL] = uniform_cap_points(
            c[POP_L_FOVEAL], r_photo, fov_a, noise * 0.3)
        pos[POP_M_FOVEAL] = uniform_cap_points(
            c[POP_M_FOVEAL], r_photo, fov_a, noise * 0.3)
        pos[POP_L_PERIPHERAL] = uniform_sphere_band(
            c[POP_L_PERIPHERAL], r_photo, fov_a, iris_a, noise)
        pos[POP_M_PERIPHERAL] = uniform_sphere_band(
            c[POP_M_PERIPHERAL], r_photo, fov_a, iris_a, noise)
        pos[POP_S_PERIPHERAL] = uniform_sphere_band(
            c[POP_S_PERIPHERAL], r_photo, fov_a, iris_a, noise)
        pos[POP_RODS] = uniform_sphere_band(
            c[POP_RODS], r_photo, fov_a, iris_a, noise)

        # --- Layer 1: Horizontal ---
        r_horiz = layer_radius(1)
        pos[POP_HORIZONTAL_FOVEAL] = uniform_cap_points(
            c[POP_HORIZONTAL_FOVEAL], r_horiz, fov_a, noise * 0.5)
        pos[POP_HORIZONTAL_PERIPHERAL] = uniform_sphere_band(
            c[POP_HORIZONTAL_PERIPHERAL], r_horiz, fov_a, iris_a, noise)

        # --- Layer 2: Midget-Bipolars ---
        # Foveal: ALIGNMENT mit Zapfen! Kombinierte L+M-Positionen
        # werden radial nach innen projiziert. So liegt jeder foveale
        # Midget-Bipolar EXAKT unter seinem zugehörigen Zapfen
        # — das ist Voraussetzung für die 1:1:1-Foveal-Architektur.
        foveal_cone_pos_local = np.vstack([pos[POP_L_FOVEAL], pos[POP_M_FOVEAL]])
        r_bip = layer_radius(2)
        pos[POP_MIDGET_ON_BIP_FOVEAL] = radial_project(foveal_cone_pos_local, r_bip)
        pos[POP_MIDGET_OFF_BIP_FOVEAL] = radial_project(foveal_cone_pos_local, r_bip)

        # Peripheral: freie Verteilung, wird über kNN mit Zapfen verbunden.
        pos[POP_MIDGET_ON_BIP_PERIPHERAL] = uniform_sphere_band(
            c[POP_MIDGET_ON_BIP_PERIPHERAL], r_bip, fov_a, iris_a, noise)
        pos[POP_MIDGET_OFF_BIP_PERIPHERAL] = uniform_sphere_band(
            c[POP_MIDGET_OFF_BIP_PERIPHERAL], r_bip, fov_a, iris_a, noise)

        # --- Layer 3: Parasol-Bipolars ---
        # NUR peripheral — keine Foveal-Version (biologisch kein Parasol-Pfad
        # in der Foveola). Eigener Layer-Radius damit sie sich nicht mit
        # Midget-Bipolars überlagern bei KD-Tree-Queries.
        r_parasol_bip = layer_radius(3)
        pos[POP_PARASOL_ON_BIP] = uniform_sphere_band(
            c[POP_PARASOL_ON_BIP], r_parasol_bip, fov_a, iris_a, noise)
        pos[POP_PARASOL_OFF_BIP] = uniform_sphere_band(
            c[POP_PARASOL_OFF_BIP], r_parasol_bip, fov_a, iris_a, noise)

        # --- Layer 3 (Konio-Bipolar): teilt sich den Radius mit Parasol ---
        # Beide sind periphere "parallele" Pfade neben Midget. Es gibt keinen
        # Konio-Konflikt mit Parasol bei kNN, weil sie nie als gemeinsame
        # Source/Target gequeryt werden. uniform_sphere_band statt 1:1-Align
        # mit S-Zapfen, weil n_konio_bip auf max(5, n_S) gefloort ist und so
        # bei sehr kleinen S-Pops nicht in eine Index-Mismatch läuft. Das
        # 1:1-Mapping zum Center-S-Zapfen erledigt ohnehin _compute_bipolar_
        # to_photo_mapping per kNN.
        pos[POP_KONIO_S_BIP] = uniform_sphere_band(
            c[POP_KONIO_S_BIP], r_parasol_bip, fov_a, iris_a, noise)

        # --- Layer 4: Amacrine ---
        r_amac = layer_radius(4)
        pos[POP_AMACRINE_FOVEAL] = uniform_cap_points(
            c[POP_AMACRINE_FOVEAL], r_amac, fov_a, noise)
        pos[POP_AMACRINE_PERIPHERAL] = uniform_sphere_band(
            c[POP_AMACRINE_PERIPHERAL], r_amac, fov_a, iris_a, noise)

        # --- Layer 5: Ganglien ---
        # Foveal wieder mit radial_project() aus Zapfen-Positionen für
        # 1:1:1-Alignment. Peripheral frei verteilt.
        r_gang = layer_radius(5)
        pos[POP_MIDGET_ON_GANG_FOVEAL] = radial_project(foveal_cone_pos_local, r_gang)
        pos[POP_MIDGET_OFF_GANG_FOVEAL] = radial_project(foveal_cone_pos_local, r_gang)
        pos[POP_MIDGET_ON_GANG_PERIPHERAL] = uniform_sphere_band(
            c[POP_MIDGET_ON_GANG_PERIPHERAL], r_gang, fov_a, iris_a, noise)
        pos[POP_MIDGET_OFF_GANG_PERIPHERAL] = uniform_sphere_band(
            c[POP_MIDGET_OFF_GANG_PERIPHERAL], r_gang, fov_a, iris_a, noise)
        pos[POP_PARASOL_ON_GANG] = uniform_sphere_band(
            c[POP_PARASOL_ON_GANG], r_gang, fov_a, iris_a, noise)
        pos[POP_PARASOL_OFF_GANG] = uniform_sphere_band(
            c[POP_PARASOL_OFF_GANG], r_gang, fov_a, iris_a, noise)
        pos[POP_KONIO_GANG] = uniform_sphere_band(
            c[POP_KONIO_GANG], r_gang, fov_a, iris_a, noise)

        # --- ORIGIN-OFFSET ---
        # Erst JETZT addieren, am Ende. Alle kNN-Berechnungen oben (v.a.
        # radial_project) laufen in lokalen Koordinaten — das vermeidet
        # numerische Fehler bei großen Origin-Werten und stellt sicher,
        # dass Multi-Agent-Retinas exakt gleich strukturiert sind.
        if not np.allclose(self._origin, 0):
            for name in pos:
                if len(pos[name]) > 0:
                    pos[name] = pos[name] + self._origin

        return pos

    # ------------------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------------------

    def build(self) -> 'Retina':
        """Erzeugt alle NEST-Populationen mit Positionen und Neuronenmodellen.

        Reihenfolge wichtig: zuerst counts+positions (reine Python-Logik),
        dann nest.Create() für jede Population. Positionen werden über
        nest.spatial.free() angeheftet — das erlaubt spätere räumliche
        Queries in NEST selbst (nutzen wir aktuell nicht, könnte aber für
        Visualisierung oder NEST-interne spatial connect_layers nützlich
        werden).
        """
        if self._built:
            if self.verbose:
                print("WARNING: Retina bereits gebaut")
            return self

        # min_delay muss VOR Connect gesetzt werden, damit NEST die
        # Synapsen-Queues richtig dimensioniert. try/except weil in
        # manchen NEST-Versionen set() fehlschlägt wenn bereits Knoten
        # existieren — in dem Fall vertrauen wir auf den bestehenden Wert.
        try:
            nest.set(min_delay=self.params['min_delay'])
        except Exception:
            pass

        self.compute_counts()
        self.positions = self._generate_positions()

        if self.verbose:
            total = sum(self.counts.values())
            print(f"\n=== BUILD: {total:,} Zielneuronen (origin={tuple(self._origin)}) ===")

        np_p = self.neuron_params

        # --- Photorezeptoren als parrot_neurons ---
        # parrot_neuron ist ein spike-transparentes Neuron: es hat kein
        # eigenes Membranpotenzial, sondern leitet eingehende Spikes einfach
        # weiter. Wir brauchen sie hier NUR für die Position — der visuelle
        # Input geht via step_current_generator direkt in die Bipolars.
        # edge_wrap=False: keine periodischen Randbedingungen (keine Kugel-
        # Toroid-Magie).
        for name in PHOTORECEPTOR_POPS:
            positions = self.positions[name]
            if len(positions) == 0:
                continue
            self.populations[name] = nest.Create(
                'parrot_neuron',
                positions=nest.spatial.free(positions.tolist(), edge_wrap=False),
            )

        # --- Horizontal: iaf_cond_exp ---
        # Leaky Integrate-and-Fire mit konduktanz-basierten Synapsen.
        # Keine Adaptation — Horizontals sollen möglichst linear integrieren.
        for name in [POP_HORIZONTAL_FOVEAL, POP_HORIZONTAL_PERIPHERAL]:
            positions = self.positions[name]
            if len(positions) == 0:
                continue
            self.populations[name] = nest.Create(
                'iaf_cond_exp',
                positions=nest.spatial.free(positions.tolist(), edge_wrap=False),
                params=np_p['horizontal'],
            )

        # --- Midget-Bipolars ON ---
        # Foveal + peripheral mit demselben Neuronenmodell (bipolar_on).
        # Unterschied foveal vs peripheral liegt nur in der Anzahl und in
        # der Konnektivität, nicht im Einzel-Neuron.
        for name in [POP_MIDGET_ON_BIP_FOVEAL, POP_MIDGET_ON_BIP_PERIPHERAL]:
            positions = self.positions[name]
            if len(positions) == 0:
                continue
            self.populations[name] = nest.Create(
                'iaf_cond_exp',
                positions=nest.spatial.free(positions.tolist(), edge_wrap=False),
                params=np_p['bipolar_on'],
            )

        # --- Midget-Bipolars OFF ---
        # Gleiches Modell (bipolar_off hat identische Parameter wie bipolar_on).
        # Trennung existiert nur, damit ein User die Parameter bei Bedarf
        # separat überschreiben könnte.
        for name in [POP_MIDGET_OFF_BIP_FOVEAL, POP_MIDGET_OFF_BIP_PERIPHERAL]:
            positions = self.positions[name]
            if len(positions) == 0:
                continue
            self.populations[name] = nest.Create(
                'iaf_cond_exp',
                positions=nest.spatial.free(positions.tolist(), edge_wrap=False),
                params=np_p['bipolar_off'],
            )

        # --- Parasol-Bipolars ---
        # Keine eigenen Parasol-Neuronenparameter — nutzen bipolar_on/off.
        # Unterschied zu Midget: nur die Konnektivität (mehr Zapfen pro Bipolar,
        # größere rezeptive Felder).
        for name, model_key in [(POP_PARASOL_ON_BIP, 'bipolar_on'),
                                 (POP_PARASOL_OFF_BIP, 'bipolar_off')]:
            positions = self.positions[name]
            if len(positions) == 0:
                continue
            self.populations[name] = nest.Create(
                'iaf_cond_exp',
                positions=nest.spatial.free(positions.tolist(), edge_wrap=False),
                params=np_p[model_key],
            )

        # --- Konio-Bipolar (BB-bistratified) ---
        # Auch iaf_cond_exp mit bipolar_on-Params. Die Blue-Yellow-Opponenz
        # entsteht ausschließlich durch die Eingangsgewichte (positiv von S,
        # negativ von L+M) — das Neuron selbst ist nichts Besonderes.
        if POP_KONIO_S_BIP in self.positions:
            positions = self.positions[POP_KONIO_S_BIP]
            if len(positions) > 0:
                self.populations[POP_KONIO_S_BIP] = nest.Create(
                    'iaf_cond_exp',
                    positions=nest.spatial.free(positions.tolist(), edge_wrap=False),
                    params=np_p['bipolar_on'],
                )

        # --- Amacrine: aeif_cond_exp ---
        # Adaptive Exponential IaF. Brauchen wir wegen der Adaptation —
        # Amacrines zeigen starke transiente Antworten.
        for name in [POP_AMACRINE_FOVEAL, POP_AMACRINE_PERIPHERAL]:
            positions = self.positions[name]
            if len(positions) == 0:
                continue
            self.populations[name] = nest.Create(
                'aeif_cond_exp',
                positions=nest.spatial.free(positions.tolist(), edge_wrap=False),
                params=np_p['amacrine'],
            )

        # --- Midget-Ganglien: aeif_cond_exp ---
        # Ganglien adaptieren biologisch ebenfalls — sustained vs transient
        # Antworten sind teilweise durch Spike-Frequenz-Adaptation geprägt.
        for name in [POP_MIDGET_ON_GANG_FOVEAL, POP_MIDGET_OFF_GANG_FOVEAL,
                     POP_MIDGET_ON_GANG_PERIPHERAL, POP_MIDGET_OFF_GANG_PERIPHERAL]:
            positions = self.positions[name]
            if len(positions) == 0:
                continue
            self.populations[name] = nest.Create(
                'aeif_cond_exp',
                positions=nest.spatial.free(positions.tolist(), edge_wrap=False),
                params=np_p['midget_ganglion'],
            )

        # --- Parasol-Ganglien: aeif_cond_exp, schneller parametrisiert ---
        for name in [POP_PARASOL_ON_GANG, POP_PARASOL_OFF_GANG]:
            positions = self.positions[name]
            if len(positions) == 0:
                continue
            self.populations[name] = nest.Create(
                'aeif_cond_exp',
                positions=nest.spatial.free(positions.tolist(), edge_wrap=False),
                params=np_p['parasol_ganglion'],
            )

        # --- Konio-Ganglion (small-bistratified): aeif_cond_exp ---
        # Wir nutzen midget_ganglion-Parameter (langsame Antwort, hohe
        # Adaptation), weil small-bistratified-Zellen biologisch eher
        # sustained antworten — passt zum Midget-Profil. Wer eine schnellere
        # Konio-Antwort will, kann auf parasol_ganglion umschalten.
        if POP_KONIO_GANG in self.positions:
            positions = self.positions[POP_KONIO_GANG]
            if len(positions) > 0:
                self.populations[POP_KONIO_GANG] = nest.Create(
                    'aeif_cond_exp',
                    positions=nest.spatial.free(positions.tolist(), edge_wrap=False),
                    params=np_p['midget_ganglion'],
                )

        # --- Mapping für den Feeder ---
        # MUSS nach _generate_positions() passieren, aber KANN vor connect()
        # stehen (das Mapping betrifft nur die Feeder-Sampling-Logik,
        # nicht die intra-retinalen Verbindungen).
        self._compute_bipolar_to_photo_mapping()

        # --- Grid-Bucket-Mapping ---
        # Pro Pop: dict[(row, col) -> [nest_id]] in der Default-Resolution.
        # Wird via get_grid_maps() für externe Bucket-Connect-Logik genutzt.
        self._compute_grid_maps()

        self._built = True

        if self.verbose:
            actual = sum(len(p) for p in self.populations.values())
            print(f"Gebaut: {actual:,} Neuronen über {len(self.populations)} Populationen")

        return self

    # ------------------------------------------------------------------------
    # Bipolar -> Photorezeptor Mapping
    # ------------------------------------------------------------------------

    def _compute_bipolar_to_photo_mapping(self):
        """
        Für jeden Bipolar: welche Photorezeptor-Indizes "beobachtet" er?
        Wird vom Feeder beim feed() benutzt.

        Struktur:
          self._bipolar_to_photo_mapping[bip_name] = [
              [(photo_pop, photo_idx), ...],   # Bipolar 0
              [(photo_pop, photo_idx), ...],   # Bipolar 1
              ...
          ]

        Jeder Eintrag ist eine Liste von (pop_name, idx)-Tupeln, damit ein
        Bipolar z.B. sowohl L- als auch M-Zapfen mischen kann (Parasol-Pfad).

        Biologisch: das entspricht den Dendriten-Baum eines Bipolars.
        Foveal sehr schmal (1 Zapfen), peripheral breiter (3-8 Zapfen).
        """
        p = self.params
        self._bipolar_to_photo_mapping = {}

        # --- Midget foveal: strikt 1:1 ---
        # Alignment existiert schon über radial_project() in _generate_positions.
        # Hier nur Index-Zuordnung: Bipolar i gehört zu Zapfen i (erst alle L,
        # dann alle M — so wurde auch foveal_cone_pos_local gebaut).
        n_L_fov = self.counts[POP_L_FOVEAL]
        n_M_fov = self.counts[POP_M_FOVEAL]
        fov_mapping = []
        for i in range(n_L_fov):
            fov_mapping.append([(POP_L_FOVEAL, i)])
        for i in range(n_M_fov):
            fov_mapping.append([(POP_M_FOVEAL, i)])
        # Dasselbe Mapping für ON und OFF — beide "schauen" denselben Zapfen.
        # Der Unterschied kommt erst im Feeder (Vorzeichen-Flip).
        self._bipolar_to_photo_mapping[POP_MIDGET_ON_BIP_FOVEAL] = fov_mapping
        self._bipolar_to_photo_mapping[POP_MIDGET_OFF_BIP_FOVEAL] = fov_mapping

        # --- Midget peripheral: k nächste L/M-Zapfen ---
        # S wird NICHT mitgenommen — Midget-Pfad ist biologisch auf L/M
        # beschränkt (S hat einen eigenen spezialisierten Bipolar-Typ).
        L_periph_pos = self.positions[POP_L_PERIPHERAL]
        M_periph_pos = self.positions[POP_M_PERIPHERAL]
        combined_periph = np.vstack([L_periph_pos, M_periph_pos])
        n_L = len(L_periph_pos)

        for bip_name in [POP_MIDGET_ON_BIP_PERIPHERAL, POP_MIDGET_OFF_BIP_PERIPHERAL]:
            if bip_name not in self.positions:
                continue
            bip_pos = self.positions[bip_name]
            k = p['k_cones_per_midget_bipolar_peripheral']
            # kNN im kombinierten L+M-Array.
            neighbor_idx = nearest_k_indices(combined_periph, bip_pos, k)

            # Indizes zurück in (pop_name, local_idx) auflösen.
            # Trick: wenn idx < n_L, ist es ein L-Zapfen mit Index idx,
            # sonst ein M-Zapfen mit Index idx - n_L.
            mapping = []
            for indices in neighbor_idx:
                targets = []
                for idx in indices:
                    idx = int(idx)
                    if idx < n_L:
                        targets.append((POP_L_PERIPHERAL, idx))
                    else:
                        targets.append((POP_M_PERIPHERAL, idx - n_L))
                mapping.append(targets)
            self._bipolar_to_photo_mapping[bip_name] = mapping

        # --- Parasol: k nächste L/M/S-Zapfen ---
        # Im Gegensatz zu Midget: S MIT dabei! Parasol ist achromatisch,
        # summiert alle Zapfentypen und verliert damit Farbinformation
        # zugunsten hoher Empfindlichkeit.
        S_periph_pos = self.positions[POP_S_PERIPHERAL]
        combined_parasol = np.vstack([L_periph_pos, M_periph_pos, S_periph_pos])
        n_L_p = len(L_periph_pos)
        n_M_p = len(M_periph_pos)

        for bip_name in [POP_PARASOL_ON_BIP, POP_PARASOL_OFF_BIP]:
            if bip_name not in self.positions:
                continue
            bip_pos = self.positions[bip_name]
            k = p['k_cones_per_parasol_bipolar']
            neighbor_idx = nearest_k_indices(combined_parasol, bip_pos, k)

            # Drei-Stufen-Auflösung: L / M / S je nach Index-Bereich.
            mapping = []
            for indices in neighbor_idx:
                targets = []
                for idx in indices:
                    idx = int(idx)
                    if idx < n_L_p:
                        targets.append((POP_L_PERIPHERAL, idx))
                    elif idx < n_L_p + n_M_p:
                        targets.append((POP_M_PERIPHERAL, idx - n_L_p))
                    else:
                        targets.append((POP_S_PERIPHERAL, idx - n_L_p - n_M_p))
                mapping.append(targets)
            self._bipolar_to_photo_mapping[bip_name] = mapping

        # --- Konio-Bipolar: nur S-Zapfen als Center ---
        # Hier mappen wir AUSSCHLIESSLICH den nächsten S-Zapfen pro Konio-
        # Bipolar. Den Yellow-Surround (L+M) erzeugen wir NICHT über das
        # Feeder-Mapping, sondern in _connect_konio_pathway via inhibitorische
        # Synapsen von Midget-ON-Bipolaren — analog zur Horizontal-Surround-
        # Strategie, weil Photorezeptoren als parrot_neurons nicht spiken
        # können und als Synapsen-Quelle für Inhibition unbrauchbar sind.
        if POP_KONIO_S_BIP in self.positions and len(S_periph_pos) > 0:
            konio_pos = self.positions[POP_KONIO_S_BIP]
            k_S = p['k_S_cones_per_konio_bipolar']
            # Safeguard: bei sehr kleinen S-Pops nie mehr fragen als vorhanden.
            k_S = min(k_S, len(S_periph_pos))
            neighbor_idx = nearest_k_indices(S_periph_pos, konio_pos, k_S)
            mapping = []
            for indices in neighbor_idx:
                targets = [(POP_S_PERIPHERAL, int(idx)) for idx in indices]
                mapping.append(targets)
            self._bipolar_to_photo_mapping[POP_KONIO_S_BIP] = mapping

    # ========================================================================
    # CONNECT — nachbarschafts-basierte Verdrahtung
    # ========================================================================

    def connect(self) -> 'Retina':
        """Baut alle intra-retinalen Verbindungen.

        Reihenfolge:
          1. Horizontal laterale Inhibition (Surround-Bildung)
          2. Bipolar -> Midget-Ganglion (Center-Signal, foveal + peripheral)
          3. Bipolar -> Parasol-Ganglion (nur peripheral)
          4. Amacrine-Zirkus (Input von Bipolars, Feedback auf Bipolars,
             Inhibition auf Ganglien)

        Die Reihenfolge ist unkritisch, alles sind nur Connect-Aufrufe —
        aber so bleibt die Logik nah am biologischen Pfad.
        """
        if not self._built:
            raise RuntimeError("build() muss vor connect() laufen")
        if self._connected:
            if self.verbose:
                print("WARNING: bereits verbunden")
            return self

        if self.verbose:
            print("\n=== CONNECT ===")

        self._connect_horizontal_lateral()
        self._connect_bipolar_to_midget_ganglion()
        self._connect_bipolar_to_parasol_ganglion()
        self._connect_konio_pathway()
        self._connect_amacrine_circuit()

        self._connected = True

        if self.verbose:
            total = len(nest.GetConnections())
            print(f"Intra-retinale Verbindungen: {total:,}")

        return self

    # ------------------------------------------------------------------------
    # Connection-Primitive
    # ------------------------------------------------------------------------

    def _connect_nearest(self, src_pop, src_positions, tgt_pop, tgt_positions,
                          k: int, weight: float, delay: float = 1.0,
                          n_ribbon: int = 1):
        """
        Jedes Target bekommt seine k räumlich nächsten Sources als Inputs.
        Nutzt Ribbon-Multiplicity für starke Verbindungen.

        Ribbon-Synapse: biologisch setzen echte Bipolars pro AP mehrere
        Vesikel-Gruppen frei (glutamaterge Ribbon-Synapsen). Das simulieren
        wir durch n_ribbon parallele Connect-Calls — NEST sieht dann mehrere
        Synapsen zwischen denselben Knoten, was effektiv die synaptische
        Stärke multipliziert ohne das Membran-Zeitverhalten zu verändern.
        """
        if len(src_pop) == 0 or len(tgt_pop) == 0 or k == 0:
            return

        k = min(k, len(src_pop))
        # kNN liefert für jedes Target die Indizes seiner k nächsten Sources.
        neighbor_idx = nearest_k_indices(src_positions, tgt_positions, k)

        src_list = src_pop.tolist()
        tgt_list = tgt_pop.tolist()

        # Pro Target: NodeCollection aus den ausgewählten Sources bauen.
        # sorted(set(...)) dedupliziert — theoretisch könnte kNN denselben
        # Source zweimal zurückgeben wenn src_positions Duplikate hat (sollte
        # nicht vorkommen, aber Safeguard).
        for tgt_idx, src_indices in enumerate(neighbor_idx):
            tgt_node = nest.NodeCollection([tgt_list[tgt_idx]])
            src_nodes = nest.NodeCollection(sorted(set(
                src_list[si] for si in src_indices
            )))
            # Ribbon-Multiplicity: n_ribbon parallele Synapsen-Bündel.
            # Jedes Bündel verbindet alle gewählten Sources mit dem Target.
            for _ in range(n_ribbon):
                nest.Connect(
                    src_nodes, tgt_node,
                    conn_spec='all_to_all',
                    syn_spec={'weight': weight, 'delay': delay},
                )

    def _connect_one_to_one_ribbon(self, src_pop, tgt_pop, weight: float,
                                     delay: float = 1.0, n_ribbon: int = 1):
        """1:1 mit Ribbon-Multiplicity.

        Nur für die foveale Midget-Schicht genutzt, wo Index i im Bipolar
        genau dem Index i im Ganglion entspricht (durch das radial_project-
        Alignment). Schnellerer Weg als _connect_nearest mit k=1.
        """
        if len(src_pop) == 0 or len(tgt_pop) == 0:
            return
        if len(src_pop) != len(tgt_pop):
            raise ValueError(
                f"1:1 erfordert gleiche Größe: {len(src_pop)} vs {len(tgt_pop)}"
            )
        # n_ribbon parallele 1:1-Verbindungen — keine explizite Deduplikation
        # nötig weil one_to_one sauber pairt.
        for _ in range(n_ribbon):
            nest.Connect(src_pop, tgt_pop, 'one_to_one',
                         syn_spec={'weight': weight, 'delay': delay})

    # ------------------------------------------------------------------------
    # Spezifische Connection-Routinen
    # ------------------------------------------------------------------------

    def _connect_horizontal_lateral(self):
        """Horizontal: Input von Bipolars, inhib Feedback auf Bipolars.

        WICHTIGER HINWEIS ZUR VEREINFACHUNG:
        Biologisch bekommen Horizontal-Zellen ihren Input direkt von den
        Photorezeptoren (Zapfen). Wir können das aber nicht so modellieren,
        weil unsere Photorezeptoren parrot_neurons sind und keine Spikes
        auslösen — der visuelle Input geht via step_current_generator an
        die Bipolars. Deswegen greifen wir HIER den OFF-Bipolar als Proxy
        für das Photorezeptor-Signal ab (OFF-Bipolar feuert invers zur
        lokalen Helligkeit, ist aber der schnellste Signalweg nach dem
        Input).

        Die resultierende Inhibition auf die Bipolars bildet dann das
        klassische Center-Surround-RF: ein Bipolar ist aktiv wenn sein
        Center hell ist UND der Surround dunkel — genau die Voraussetzung
        für Kanten- und Kontrast-Detektion.
        """
        p = self.params

        # --- Foveal-Zirkus ---
        hf = self.populations.get(POP_HORIZONTAL_FOVEAL)
        if hf is not None and POP_MIDGET_OFF_BIP_FOVEAL in self.populations:
            # Input: OFF-Bipolars foveal -> Horizontal (exzitatorisch, w=1.0).
            # k_cones_per_horizontal_foveal=5 -> jede Horizontal sammelt
            # über ~5 Bipolars -> räumliche Integration.
            self._connect_nearest(
                src_pop=self.populations[POP_MIDGET_OFF_BIP_FOVEAL],
                src_positions=self.positions[POP_MIDGET_OFF_BIP_FOVEAL],
                tgt_pop=hf,
                tgt_positions=self.positions[POP_HORIZONTAL_FOVEAL],
                k=p['k_cones_per_horizontal_foveal'],
                weight=1.0, delay=1.0,
                n_ribbon=p['ribbon_horizontal'],
            )
            # Feedback: Horizontal -> OFF-Bipolars (inhibitorisch, w=-0.5).
            # Nur OFF foveal — die ON-Pfad-Foveal-Inhibition kommt nicht
            # zustande, weil foveal keine ON-Bipolars via Horizontal moduliert
            # werden (Vereinfachung; biologisch inhibieren Horizontals beide
            # über H1/H2-Subtypen).
            self._connect_nearest(
                src_pop=hf,
                src_positions=self.positions[POP_HORIZONTAL_FOVEAL],
                tgt_pop=self.populations[POP_MIDGET_OFF_BIP_FOVEAL],
                tgt_positions=self.positions[POP_MIDGET_OFF_BIP_FOVEAL],
                k=p['k_horizontals_surround_per_cone'],
                weight=p['w_horizontal_to_cone'],
                delay=1.5,   # leicht verzögert -> Surround kommt NACH Center
                n_ribbon=p['ribbon_horizontal'],
            )

        # --- Peripheral-Zirkus ---
        # Hier vollständiger: Input kommt von Midget-OFF UND Parasol-OFF,
        # Feedback geht auf ALLE vier Bipolar-Typen (ON+OFF × Midget+Parasol).
        hp = self.populations.get(POP_HORIZONTAL_PERIPHERAL)
        if hp is not None:
            # Exzitatorischer Input von OFF-Bipolars (beide Pfade).
            for bip_name in [POP_MIDGET_OFF_BIP_PERIPHERAL, POP_PARASOL_OFF_BIP]:
                if bip_name in self.populations:
                    self._connect_nearest(
                        src_pop=self.populations[bip_name],
                        src_positions=self.positions[bip_name],
                        tgt_pop=hp,
                        tgt_positions=self.positions[POP_HORIZONTAL_PERIPHERAL],
                        k=p['k_cones_per_horizontal_peripheral'],
                        weight=1.0, delay=1.0,
                        n_ribbon=p['ribbon_horizontal'],
                    )
            # Inhibitorischer Feedback auf ALLE Bipolar-Typen peripheral.
            # Dadurch entsteht das Surround für alle vier Pfade.
            for bip_name in [POP_MIDGET_OFF_BIP_PERIPHERAL, POP_PARASOL_OFF_BIP,
                             POP_MIDGET_ON_BIP_PERIPHERAL, POP_PARASOL_ON_BIP]:
                if bip_name in self.populations:
                    self._connect_nearest(
                        src_pop=hp,
                        src_positions=self.positions[POP_HORIZONTAL_PERIPHERAL],
                        tgt_pop=self.populations[bip_name],
                        tgt_positions=self.positions[bip_name],
                        k=p['k_horizontals_surround_per_cone'],
                        weight=p['w_horizontal_to_cone'],
                        delay=1.5,
                        n_ribbon=p['ribbon_horizontal'],
                    )

    def _connect_bipolar_to_midget_ganglion(self):
        """Midget-Bipolar -> Midget-Ganglion. Foveal 1:1, peripheral k nearest.

        HIER wird der Midget-Pfad geschlossen. Foveal ist das die kritische
        1:1:1-Verbindung (Zapfen -> Bipolar -> Ganglion), die im menschlichen
        Auge die höchste Sehschärfe erzeugt.
        """
        p = self.params
        # abs() weil w_bipolar_to_ganglion_midget als positive (exzitatorische)
        # Verbindung gedacht ist. Falls im Config versehentlich negativ.
        w = abs(p['w_bipolar_to_ganglion_midget'])
        n_ribbon = p['ribbon_bipolar_to_ganglion']  # = 20 (validiert)

        # --- Foveal: strikt 1:1 ---
        # Kein kNN nötig, Index-Alignment existiert dank radial_project.
        self._connect_one_to_one_ribbon(
            self.populations[POP_MIDGET_ON_BIP_FOVEAL],
            self.populations[POP_MIDGET_ON_GANG_FOVEAL],
            weight=w, delay=1.0, n_ribbon=n_ribbon,
        )
        self._connect_one_to_one_ribbon(
            self.populations[POP_MIDGET_OFF_BIP_FOVEAL],
            self.populations[POP_MIDGET_OFF_GANG_FOVEAL],
            weight=w, delay=1.0, n_ribbon=n_ribbon,
        )

        # --- Peripheral: k nächste Bipolars -> Ganglion ---
        # k=3 typisch -> leichte Konvergenz, Auflösung nimmt nach peripher ab.
        k = p['k_bipolars_per_midget_ganglion_peripheral']
        self._connect_nearest(
            src_pop=self.populations[POP_MIDGET_ON_BIP_PERIPHERAL],
            src_positions=self.positions[POP_MIDGET_ON_BIP_PERIPHERAL],
            tgt_pop=self.populations[POP_MIDGET_ON_GANG_PERIPHERAL],
            tgt_positions=self.positions[POP_MIDGET_ON_GANG_PERIPHERAL],
            k=k, weight=w, delay=1.0, n_ribbon=n_ribbon,
        )
        self._connect_nearest(
            src_pop=self.populations[POP_MIDGET_OFF_BIP_PERIPHERAL],
            src_positions=self.positions[POP_MIDGET_OFF_BIP_PERIPHERAL],
            tgt_pop=self.populations[POP_MIDGET_OFF_GANG_PERIPHERAL],
            tgt_positions=self.positions[POP_MIDGET_OFF_GANG_PERIPHERAL],
            k=k, weight=w, delay=1.0, n_ribbon=n_ribbon,
        )

    def _connect_bipolar_to_parasol_ganglion(self):
        """Parasol-Bipolar -> Parasol-Ganglion (k nearest).

        Nur peripheral. k=6 -> Parasol-Ganglion integriert über 6 Bipolars,
        das ergibt zusammen mit 8 Zapfen pro Bipolar ein RF aus ~48 Zapfen
        (mit Overlap durchaus ~30-40 effektive Zapfen pro Parasol-RF —
        passt zur biologisch dokumentierten Größe).
        """
        p = self.params
        w = abs(p['w_bipolar_to_ganglion_parasol'])
        n_ribbon = p['ribbon_bipolar_to_ganglion']
        k = p['k_bipolars_per_parasol_ganglion']

        self._connect_nearest(
            src_pop=self.populations[POP_PARASOL_ON_BIP],
            src_positions=self.positions[POP_PARASOL_ON_BIP],
            tgt_pop=self.populations[POP_PARASOL_ON_GANG],
            tgt_positions=self.positions[POP_PARASOL_ON_GANG],
            k=k, weight=w, delay=0.8, n_ribbon=n_ribbon,   # delay=0.8 schneller als Midget
        )
        self._connect_nearest(
            src_pop=self.populations[POP_PARASOL_OFF_BIP],
            src_positions=self.positions[POP_PARASOL_OFF_BIP],
            tgt_pop=self.populations[POP_PARASOL_OFF_GANG],
            tgt_positions=self.positions[POP_PARASOL_OFF_GANG],
            k=k, weight=w, delay=0.8, n_ribbon=n_ribbon,
        )

    def _connect_konio_pathway(self):
        """Konio-Bahn: Yellow-Surround-Inhibition + Bipolar -> Ganglion.

        Drei-Schritt-Aufbau:
          1. Midget-ON-Bipolar peripheral -> Konio-Bipolar (INHIBITORISCH).
             Yellow-Surround. Midget-ON-peripheral ist unser Proxy für L+M-
             Helligkeit (genau wie Horizontal-Surround Midget-OFF als L+M-
             Proxy nutzt — selbe Strategie, andere Vorzeichen-Konvention).
          2. Konio-Bipolar -> Konio-Ganglion (EXZITATORISCH).
             k=2 nächste Bipolare pro Ganglion.

        Center-Strom (S-Cone -> Konio-Bipolar) kommt NICHT hier, sondern via
        Feeder als step_current — analog Midget-Bipolaren. Die Mapping-Logik
        in _compute_bipolar_to_photo_mapping liefert die richtigen S-Indizes
        pro Konio-Bipolar.

        Wichtig: dieser Pfad existiert nur peripheral, weil POP_KONIO_S_BIP
        und POP_KONIO_GANG nur peripheral angelegt werden (S-Zapfen fehlen
        in der Foveola).
        """
        if POP_KONIO_S_BIP not in self.populations:
            return
        if POP_KONIO_GANG not in self.populations:
            return

        p = self.params

        # --- Step 1: Midget-ON-Bipolar peripheral -> Konio-Bipolar (Yellow-Surround) ---
        # Negatives Gewicht erzeugt die Inhibition. Wir nehmen Midget-ON statt
        # Midget-OFF, weil Midget-ON ON-Antwort auf L+M-Helligkeit liefert
        # (entspricht "Gelb hell"). Bei Midget-OFF wäre das Vorzeichen invers.
        if POP_MIDGET_ON_BIP_PERIPHERAL in self.populations:
            w_surround = -abs(p['w_midget_to_konio_bipolar'])
            k_surround = p['k_midget_surround_per_konio_bipolar']
            n_ribbon_surround = p['ribbon_konio']
            self._connect_nearest(
                src_pop=self.populations[POP_MIDGET_ON_BIP_PERIPHERAL],
                src_positions=self.positions[POP_MIDGET_ON_BIP_PERIPHERAL],
                tgt_pop=self.populations[POP_KONIO_S_BIP],
                tgt_positions=self.positions[POP_KONIO_S_BIP],
                k=k_surround,
                weight=w_surround,
                delay=1.5,        # leicht verzögert wie Horizontal-Surround
                n_ribbon=n_ribbon_surround,
            )

        # --- Step 2: Konio-Bipolar -> Konio-Ganglion ---
        # Exzitatorisch, ribbon=10 (kleiner als Midget=20 weil dünnere Bahn).
        w_kg = abs(p['w_konio_bipolar_to_ganglion'])
        k_kg = p['k_konio_bipolars_per_konio_ganglion']
        n_ribbon_kg = p['ribbon_konio']
        self._connect_nearest(
            src_pop=self.populations[POP_KONIO_S_BIP],
            src_positions=self.positions[POP_KONIO_S_BIP],
            tgt_pop=self.populations[POP_KONIO_GANG],
            tgt_positions=self.positions[POP_KONIO_GANG],
            k=k_kg,
            weight=w_kg,
            delay=1.0,
            n_ribbon=n_ribbon_kg,
        )

    def _connect_amacrine_circuit(self):
        """Amacrine: Input von Bipolars, inhib Output auf Bipolars + Ganglien.

        Dreischritt:
          1. Bipolar -> Amacrine (exzitatorisch)
          2. Amacrine -> Ganglion (inhibitorisch) — moduliert Output
          3. Amacrine -> Bipolar (inhibitorisch, Feedback) — reziprok

        Die Delays stufen sich: Input (1.0ms) -> Inhibition-auf-Ganglion (1.5ms)
        -> Feedback-auf-Bipolar (2.0ms). Dadurch folgt die Inhibition zeitlich
        der Aktivierung — das erzeugt transiente Antworten (Bipolar-Burst
        wird schnell durch Amacrine-Feedback gedämpft).

        Peripheral: ALLE vier Bipolar-Typen beteiligt.
        Foveal: NUR Midget-ON/OFF (kein Parasol-Pfad foveal).
        """
        p = self.params
        # Gewichte: abs(...) stellt sicher dass Input positiv, Inhibition negativ.
        w_bip_in = abs(p['w_bipolar_to_amacrine'])
        w_amac_gang = -abs(p['w_amacrine_to_ganglion'])
        w_amac_bip = -abs(p['w_amacrine_to_bipolar'])
        k_bip2amac = p['k_bipolars_per_amacrine']
        k_amac2gang = p['k_amacrines_per_ganglion']
        k_amac2bip = p['k_amacrines_per_bipolar_feedback']
        n_ribbon = p['ribbon_amacrine']

        # --- Peripheral-Zirkus ---
        ap = self.populations.get(POP_AMACRINE_PERIPHERAL)
        if ap is not None:
            ap_pos = self.positions[POP_AMACRINE_PERIPHERAL]

            # Step 1: Bipolar -> Amacrine (exzitatorisch).
            # Alle vier peripheral-Bipolar-Typen speisen gemeinsam die
            # peripheral-Amacrines — biologisch integrieren Amacrines
            # auch Pfad-übergreifend.
            for bip_name in [POP_MIDGET_ON_BIP_PERIPHERAL, POP_MIDGET_OFF_BIP_PERIPHERAL,
                             POP_PARASOL_ON_BIP, POP_PARASOL_OFF_BIP]:
                if bip_name not in self.populations:
                    continue
                self._connect_nearest(
                    src_pop=self.populations[bip_name],
                    src_positions=self.positions[bip_name],
                    tgt_pop=ap, tgt_positions=ap_pos,
                    k=k_bip2amac, weight=w_bip_in, delay=1.0,
                    n_ribbon=n_ribbon,
                )

            # Step 2: Amacrine -> Ganglion (inhibitorisch).
            # Dämpft die Ganglion-Aktivität direkt — Quelle der transienten
            # Antworten auf Bewegung/Flicker.
            for gang_name in [POP_MIDGET_ON_GANG_PERIPHERAL, POP_MIDGET_OFF_GANG_PERIPHERAL,
                              POP_PARASOL_ON_GANG, POP_PARASOL_OFF_GANG]:
                if gang_name not in self.populations:
                    continue
                self._connect_nearest(
                    src_pop=ap, src_positions=ap_pos,
                    tgt_pop=self.populations[gang_name],
                    tgt_positions=self.positions[gang_name],
                    k=k_amac2gang, weight=w_amac_gang, delay=1.5,
                    n_ribbon=n_ribbon,
                )

            # Step 3: Amacrine -> Bipolar (inhibitorisches Feedback).
            # Reziproke Amacrine-Synapse. Biologisch GABAerg, dämpft
            # anhaltende Bipolar-Aktivität -> High-Pass-ähnliche Filterung.
            for bip_name in [POP_MIDGET_ON_BIP_PERIPHERAL, POP_MIDGET_OFF_BIP_PERIPHERAL,
                             POP_PARASOL_ON_BIP, POP_PARASOL_OFF_BIP]:
                if bip_name not in self.populations:
                    continue
                self._connect_nearest(
                    src_pop=ap, src_positions=ap_pos,
                    tgt_pop=self.populations[bip_name],
                    tgt_positions=self.positions[bip_name],
                    k=k_amac2bip, weight=w_amac_bip, delay=2.0,
                    n_ribbon=n_ribbon,
                )

        # --- Foveal-Zirkus ---
        # Nur Midget-Pfad. min(k, len(...)) als Safeguard weil die foveale
        # Amacrine-Population klein sein kann (bei wenigen Zapfen).
        af = self.populations.get(POP_AMACRINE_FOVEAL)
        if af is not None:
            af_pos = self.positions[POP_AMACRINE_FOVEAL]

            # Input von Midget-Bipolars (ON+OFF) foveal.
            for bip_name in [POP_MIDGET_ON_BIP_FOVEAL, POP_MIDGET_OFF_BIP_FOVEAL]:
                if bip_name in self.populations:
                    self._connect_nearest(
                        src_pop=self.populations[bip_name],
                        src_positions=self.positions[bip_name],
                        tgt_pop=af, tgt_positions=af_pos,
                        k=min(k_bip2amac, len(self.populations[bip_name])),
                        weight=w_bip_in, delay=1.0, n_ribbon=n_ribbon,
                    )

            # Output: Inhibition auf Midget-Ganglien foveal.
            # Kein reziprokes Feedback foveal — bewusste Vereinfachung
            # (im echten Foveal-Midget-Pfad ist Amacrine-Beteiligung
            # minimal, sustain-Antworten dominieren).
            for gang_name in [POP_MIDGET_ON_GANG_FOVEAL, POP_MIDGET_OFF_GANG_FOVEAL]:
                if gang_name in self.populations:
                    self._connect_nearest(
                        src_pop=af, src_positions=af_pos,
                        tgt_pop=self.populations[gang_name],
                        tgt_positions=self.positions[gang_name],
                        k=min(k_amac2gang, len(af)),
                        weight=w_amac_gang, delay=1.5, n_ribbon=n_ribbon,
                    )

    # ========================================================================
    # FEEDER ERZEUGEN
    # ========================================================================

    def create_input_feeder(self, config: Optional[Dict] = None) -> RetinaInputFeeder:
        """
        Erzeugt einen RetinaInputFeeder als externe Input-Schnittstelle.

        Args:
            config: Dict, mergt mit DEFAULT_FEEDER_CONFIG.
                Keys: 'generator_type' ('step_current' oder 'poisson'),
                      'input_resolution' (Zweierpotenz-Tuple),
                      'max_current_pa', 'max_rate_hz', 'poisson_weight_nS'

        Returns:
            Eigenständiger Feeder für externe Nutzung.

        Der Feeder wird NICHT in self gespeichert — der Caller ist dafür
        verantwortlich, ihn zu halten. Damit kann theoretisch auch mehrere
        Feeder an eine Retina gehängt werden (z.B. für A/B-Testing von
        Input-Modi), aber das ist eher ein edge-case.
        """
        if not self._built:
            raise RuntimeError("build() muss vor create_input_feeder() laufen")

        config = config or {}

        # Nur Bipolar-Populationen an den Feeder geben — Photorezeptoren
        # sind parrot_neurons und bekommen keinen Input (siehe Section 1).
        bipolar_pops = {
            name: self.populations[name]
            for name in BIPOLAR_POPS
            if name in self.populations
        }

        feeder = RetinaInputFeeder(
            config=config,
            bipolar_populations=bipolar_pops,
            bipolar_to_photo_mapping=self._bipolar_to_photo_mapping,
            origin=self._origin,
            world_radius=self.params['radius'],
            min_delay=self.params['min_delay'],
        )

        # Photo-Positionen nachreichen — die braucht der Feeder für das
        # 3D->2D-Sampling in _sample_frame.
        photo_positions = {
            name: self.positions[name]
            for name in PHOTORECEPTOR_POPS
            if name in self.positions
        }
        feeder.set_photo_positions(photo_positions)

        if self.verbose:
            total_gens = sum(len(g) for g in feeder.generators.values())
            print(f"\n=== FEEDER: {feeder.generator_type}, "
                  f"{total_gens} Generatoren, "
                  f"resolution={feeder.config['input_resolution']} ===")

        return feeder

    # ========================================================================
    # PUBLIC API
    # ========================================================================
    # Alle Getter geben Kopien/flache Kopien zurück, damit der Caller
    # nicht versehentlich den Retina-State mutieren kann.

    def get_populations(self) -> Dict[str, 'nest.NodeCollection']:
        """Alle NEST-Populationen als Dict. Für eigene Connect-Aufrufe."""
        return dict(self.populations)

    def get_output_populations(self) -> Dict[str, 'nest.NodeCollection']:
        """Nur die Ganglien — das sind die "Sehnerv-Fasern".

        Hier an den LGN dran-connecten. Alle anderen Populationen sind
        intra-retinale Stufen und sollten kein Downstream-Target haben.
        """
        return {k: v for k, v in self.populations.items() if k in GANGLION_POPS}

    def get_photoreceptor_populations(self) -> Dict[str, 'nest.NodeCollection']:
        """Nur die Photorezeptoren (parrot_neurons). Primär für Debugging/Viz."""
        return {k: v for k, v in self.populations.items() if k in PHOTORECEPTOR_POPS}

    def get_positions(self) -> Dict[str, np.ndarray]:
        """3D-Positionen aller Populationen (inkl. Origin-Offset)."""
        return dict(self.positions)

    def get_counts(self) -> Dict[str, int]:
        """Populationsgrößen."""
        return dict(self.counts)

    def get_origin(self) -> np.ndarray:
        """Origin-Offset (für Multi-Agent-Setups nützlich). copy() weil np-Array."""
        return self._origin.copy()

    # ------------------------------------------------------------------------
    # Grid-Bucket-Mapping (Position → Pixel-Bucket → NEST-IDs)
    # ------------------------------------------------------------------------

    def _compute_grid_maps(self):
        """Build-time: cache Default-Grid-Maps für alle Pops.

        Resultat in self._grid_maps und self._id_to_bucket.
        """
        H, W = self.params['bucket_resolution']
        if self.verbose:
            print(f"\n=== GRID-MAPS (default cache): bucket_resolution=({H}, {W}) ===")

        grid_maps, id_to_bucket = self._compute_grid_for_resolution(
            H, W, verbose=self.verbose)
        self._grid_maps = grid_maps
        self._id_to_bucket = id_to_bucket

    def _compute_grid_for_resolution(self, H: int, W: int,
                                      verbose: bool = False
                                     ) -> Tuple[
                                         Dict[str, Dict[Tuple[int, int], List[int]]],
                                         Dict[str, Dict[int, Tuple[int, int]]]
                                     ]:
        """Berechne (forward, reverse) Grid-Maps für beliebige (H, W)-Auflösung.

        Logik identisch zu Feeder._sample_frame: x->col, z->row, y ignoriert.
        """
        world_r = self.params['radius']
        origin = self._origin

        grid_maps: Dict[str, Dict[Tuple[int, int], List[int]]] = {}
        id_to_bucket: Dict[str, Dict[int, Tuple[int, int]]] = {}

        for pop_name, pop in self.populations.items():
            positions = self.positions.get(pop_name)
            if positions is None or len(positions) == 0:
                continue

            rel = positions - origin
            x = rel[:, 0]
            z = rel[:, 2]
            u = (x / world_r + 1.0) / 2.0
            v = (z / world_r + 1.0) / 2.0
            col = np.clip((u * W).astype(int), 0, W - 1)
            row = np.clip((v * H).astype(int), 0, H - 1)

            nest_ids = pop.tolist()
            grid: Dict[Tuple[int, int], List[int]] = {}
            id_map: Dict[int, Tuple[int, int]] = {}
            for idx, nest_id in enumerate(nest_ids):
                key = (int(row[idx]), int(col[idx]))
                if key not in grid:
                    grid[key] = []
                grid[key].append(nest_id)
                id_map[nest_id] = key

            grid_maps[pop_name] = grid
            id_to_bucket[pop_name] = id_map

            if verbose:
                max_occ = max((len(v) for v in grid.values()), default=0)
                print(f"  {pop_name:<35}  n={len(nest_ids):>6}  "
                      f"occupied_buckets={len(grid):>6}  max_per_bucket={max_occ}")

        return grid_maps, id_to_bucket

    def get_grid_maps(self, resolution: Optional[Tuple[int, int]] = None
                      ) -> Dict[str, Dict[Tuple[int, int], List[int]]]:
        """Pixel-Bucket-Mapping pro Population (sparse).

        Args:
            resolution: optional (H, W). None = cached default.
                        Sonst on-the-fly für andere Auflösung.

        Returns:
            Dict[pop_name -> Dict[(row, col) -> list[nest_id]]].
        """
        if resolution is None:
            return dict(self._grid_maps)
        H, W = resolution
        if not isinstance(H, int) or not isinstance(W, int):
            raise ValueError(f"resolution muss (int, int) sein, bekam {resolution}")
        if H < 2 or W < 2:
            raise ValueError(f"resolution zu klein: {resolution}, mindestens (2, 2)")
        grid_maps, _ = self._compute_grid_for_resolution(H, W, verbose=False)
        return grid_maps

    def get_id_to_bucket(self, resolution: Optional[Tuple[int, int]] = None
                         ) -> Dict[str, Dict[int, Tuple[int, int]]]:
        """Reverse-Map: nest_id -> (row, col) pro Population."""
        if resolution is None:
            return dict(self._id_to_bucket)
        H, W = resolution
        _, id_to_bucket = self._compute_grid_for_resolution(H, W, verbose=False)
        return id_to_bucket

    def get_bucket_resolution(self) -> Tuple[int, int]:
        """Aktive Default-Bucket-Resolution (H, W)."""
        return tuple(self.params['bucket_resolution'])

    # ========================================================================
    # DIAGNOSTIK
    # ========================================================================

    def print_counts(self):
        """Sauber formatierter Größenreport, gruppiert nach Zelltyp."""
        if not self.counts:
            self.compute_counts()

        print("\n=== POPULATIONS-GRÖSSEN ===")
        # Gruppierung spiegelt die Pathway-Struktur wider: Photorezeptoren
        # oben, dann Horizontal, dann Bipolar (Midget+Parasol), dann Amacrine,
        # dann Ganglien unten (output layer).
        groups = [
            ("Photoreceptors", PHOTORECEPTOR_POPS),
            ("Horizontal", [POP_HORIZONTAL_FOVEAL, POP_HORIZONTAL_PERIPHERAL]),
            ("Midget Bipolars", [POP_MIDGET_ON_BIP_FOVEAL, POP_MIDGET_OFF_BIP_FOVEAL,
                                   POP_MIDGET_ON_BIP_PERIPHERAL, POP_MIDGET_OFF_BIP_PERIPHERAL]),
            ("Parasol Bipolars", [POP_PARASOL_ON_BIP, POP_PARASOL_OFF_BIP]),
            ("Amacrine", [POP_AMACRINE_FOVEAL, POP_AMACRINE_PERIPHERAL]),
            ("Midget Ganglions", [POP_MIDGET_ON_GANG_FOVEAL, POP_MIDGET_OFF_GANG_FOVEAL,
                                    POP_MIDGET_ON_GANG_PERIPHERAL, POP_MIDGET_OFF_GANG_PERIPHERAL]),
            ("Parasol Ganglions", [POP_PARASOL_ON_GANG, POP_PARASOL_OFF_GANG]),
        ]
        for group_name, pops in groups:
            print(f"\n  [{group_name}]")
            for pop in pops:
                if pop in self.counts:
                    print(f"    {pop:<40} {self.counts[pop]:>6}")

        print(f"\n  TOTAL: {sum(self.counts.values()):,}")

    def create_spike_recorders(self) -> Dict[str, 'nest.NodeCollection']:
        """Erzeugt einen spike_recorder pro Population.

        Convenience-Methode fürs Debugging — verbindet ALLE Populationen
        inklusive Photorezeptoren (die spiken zwar nicht selbst, aber
        nested-Connects könnten indirekt Events erzeugen, v.a. wenn jemand
        die parrot_neurons später ersetzt).

        Für Production lieber selektiv aufzeichnen (nur Ganglien ->
        spart massiv RAM bei langen Simulationen).
        """
        recorders = {}
        for name, pop in self.populations.items():
            sr = nest.Create('spike_recorder')
            nest.Connect(pop, sr)
            recorders[name] = sr
        return recorders
