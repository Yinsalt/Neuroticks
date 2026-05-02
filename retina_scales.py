"""
retina_scales.py — Skalen- und Varianten-Profile für die Retina

Liefert fertige `params`-Dicts, die direkt an `Retina(params=...)` gehen:

    from retina_scales import get_config
    params, neuron_params, feeder_config = get_config(scale=3, variant='default')
    retina = Retina(params=params, neuron_params=neuron_params, verbose=True)
    retina.build()
    retina.connect()
    feeder = retina.create_input_feeder(feeder_config)

Die 5 Skalen (1..5) sind kalibriert für unterschiedliche Input-Auflösungen:

    Skala  Input     Pop      Use-Case
    1      32x32     ~2.5k    Smoke-Test, sub-second-Build
    2      64x64     ~4.4k    Schneller Loop, Connectivity-Tests
    3      128x128   ~29k     Iterative Entwicklung, V1-V6 mit ~500 PYR
    4      256x256   ~144k    Realistische Dynamik, ~30 GB RAM
    5      512x512   ~310k    Voll-Detail, entspricht alter 'medium'

NAMENSKONVENTION:
  SCALES   -> kontrollieren absolute Populationsgrößen
              (n_cones_foveal, n_cones_peripheral, n_S_cones, n_rods)
              + dazu passende input_resolution
  VARIANTS -> kontrollieren Tuning (ribbons, weights, dark currents)
  NEURONS  -> Override-Block für DEFAULT_NEURON_PARAMS

Jede (scale × variant)-Kombination ist gültig und gibt einen runnbaren Config.

BACKWARDS-COMPAT: Die alten String-Namen 'tiny'/'small'/'medium'/'large' werden
auf 1/2/3/5 gemappt damit alter Code weiterläuft. 'large' wird nicht mehr
unterstützt im neuen Schema (Skala 5 ist äquivalent zur alten 'medium').
"""

from copy import deepcopy
from typing import Dict, Tuple, Union


# ============================================================================
# SCALES — absolute Populationsgrößen, indiziert von 1 bis 5
# ============================================================================
#
# Pixel/Cone-Coverage progressiv: kleine Skalen knapp (Konvergenz fängt es ab),
# große Skalen komfortabel.
#
# Periphere Verhältnisse skalieren konservativ mit Input-Größe:
#   Skalen 1-2: peri:fov = 10:1, rods:fov = 50:1   (knapp peripheral)
#   Skala 3:    peri:fov = 14:1, rods:fov = 100:1  (Mittelfeld)
#   Skalen 4-5: peri:fov = 20:1, rods:fov = 200:1  (volle Bio-Verhältnisse)
# ============================================================================

SCALES: Dict[int, Dict] = {
    1: {
        'description': 'Sub-second-Build — ~2.5k Retina-Neurone',
        'estimated_total': 2_500,
        'estimated_build_seconds': 1,
        'estimated_ram_mb': 30,
        'use_case': ('Smoke-Tests, schneller Connectivity-Walk-through. '
                     'Pop-Groessen so klein dass Build praktisch kostenfrei ist.'),
        'n_cones_foveal':     30,
        'n_cones_peripheral': 270,
        'n_S_cones':          60,
        'n_rods':             1_500,
        # Ganglion-Targets bei diesen Werten:
        # midget_fov=30, midget_peri=30, parasol=3, konio=30
        'param_overrides': {
            # Default parasol_to_midget=0.10 → 30 * 0.10 = 3 (passt)
        },
    },
    2: {
        'description': 'Schneller Iterations-Loop — ~5k Retina-Neurone',
        'estimated_total': 5_000,
        'estimated_build_seconds': 3,
        'estimated_ram_mb': 60,
        'use_case': ('Connectivity-Patterns ausprobieren, mit V1-V6 ~100 PYR pro '
                     'Areal sinnvoll kombinierbar.'),
        'n_cones_foveal':     100,
        'n_cones_peripheral': 450,
        'n_S_cones':          100,
        'n_rods':             2_500,
        # midget_fov=100, midget_peri=50, parasol=5, konio=50
        'param_overrides': {},
    },
    3: {
        'description': 'Iterative Entwicklung — ~33k Retina-Neurone',
        'estimated_total': 33_000,
        'estimated_build_seconds': 10,
        'estimated_ram_mb': 280,
        'use_case': ('Erste echte Closed-Loop-Tests. V1-V6 mit ~300-500 PYR pro '
                     'Areal, Pulvinar reduziert. Sweet-Spot fuer Architektur-Iteration.'),
        'n_cones_foveal':     500,
        'n_cones_peripheral': 2_304,
        'n_S_cones':          200,
        'n_rods':             17_600,
        # midget_fov=500, midget_peri=256 (16x16), parasol=25, konio=100
        # Default ratio 0.10 ergibt 256*0.10 = 25.6 -> 25, passt
        'param_overrides': {},
    },
    4: {
        'description': 'Realistische Dynamik — ~155k Retina-Neurone (1:1 GRID-MATCH)',
        'estimated_total': 155_000,
        'estimated_build_seconds': 30,
        'estimated_ram_mb': 950,
        'use_case': ('Hauptarbeitsskala fuer 30-40 GB RAM-Workflow. V1-V6 mit '
                     '~1000 PYR pro Areal. EXAKTER 1:1-Match zu Cortex-Grids: '
                     'midget_fov=1800 (45x40), midget_peri=1024 (32x32), '
                     'parasol=100 (10x10), konio=480 (24x20).'),
        'n_cones_foveal':     1_800,
        'n_cones_peripheral': 9_216,
        'n_S_cones':          960,
        'n_rods':             92_400,
        # midget_fov=1800 (1:1), midget_peri=9216//9=1024
        # parasol_target=100, midget_peri=1024 → ratio = 100/1024 = 0.09765625
        # konio_target=480, n_S=960 → ratio = 480/960 = 0.5 (passt zu default)
        'param_overrides': {
            'parasol_to_midget_ratio_peripheral': 0.09765625,  # = 100/1024
        },
    },
    5: {
        'description': 'Voll-Detail — ~310k Retina-Neurone',
        'estimated_total': 310_000,
        'estimated_build_seconds': 60,
        'estimated_ram_mb': 1_500,
        'use_case': ('Validierungs-Runs mit voller Foveal-Aufloesung. V1-V6 mit '
                     '~1500-2000 PYR pro Areal. Build dauert ~1 min.'),
        'n_cones_foveal':     2_400,
        'n_cones_peripheral': 18_000,
        'n_S_cones':          1_600,
        'n_rods':             199_000,
        # midget_fov=2400, midget_peri=2000, parasol=200 (default ratio passt)
        # konio=800
        'param_overrides': {},
    },
}


# ----------------------------------------------------------------------------
# input_resolution pro Skala
# ----------------------------------------------------------------------------
#
# Pixel-pro-Foveal-Cone für jede Skala (Foveola-Patch ~17% Bildkante):
#
#   Skala 1: 32x32  ->  5x5 = 25 px /  30 cones = 0.83 px/cone (knapp, Bipolar-Konvergenz fängt ab)
#   Skala 2: 64x64  -> 10x10 = 100 px /  50 cones = 2.0 px/cone
#   Skala 3: 128x128 -> 21x21 = 441 px / 176 cones = 2.5 px/cone
#   Skala 4: 256x256 -> 43x43 = 1849 px / 462 cones = 4.0 px/cone
#   Skala 5: 512x512 -> 87x87 = 7569 px / 995 cones = 7.6 px/cone
#
# Memory pro Frame (RGBA float32):
#   32x32   =   1k px  =  16 KB / Frame
#   64x64   =   4k px  =  64 KB / Frame
#   128x128 =  16k px  = 256 KB / Frame
#   256x256 =  65k px  =   1 MB / Frame
#   512x512 = 262k px  =   4 MB / Frame
# ----------------------------------------------------------------------------

_RESOLUTION_FOR_SCALE: Dict[int, Tuple[int, int]] = {
    1: (32, 32),
    2: (64, 64),
    3: (128, 128),
    4: (256, 256),
    5: (512, 512),
}


# ============================================================================
# Backwards-Compat: alte String-Namen auf neue Zahlen mappen
# ============================================================================

_LEGACY_NAME_MAP: Dict[str, int] = {
    'tiny':   1,
    'small':  2,
    'medium': 4,   # alte 'medium' war ~260k, neue Skala 4 ist ~144k.
                   # Wir wählen 4 weil das das praktische Workflow-Mittelfeld ist.
                   # Wer wirklich die alte 'medium'-Pop-Größe braucht, nutze Skala 5.
    'large':  5,   # alte 'large' war ~770k — die neue Skala-Familie geht nicht
                   # mehr so hoch. 5 ist das Maximum.
}


def _resolve_scale(scale: Union[int, str]) -> int:
    """Übersetzt String oder Int in eine numerische Skala 1..5."""
    if isinstance(scale, int):
        if scale not in SCALES:
            raise ValueError(
                f"Unbekannte scale={scale}. Verfuegbar: {sorted(SCALES.keys())}"
            )
        return scale
    if isinstance(scale, str):
        # Versuche direkt als Zahl
        try:
            n = int(scale)
            return _resolve_scale(n)
        except ValueError:
            pass
        # Legacy-Namen
        if scale in _LEGACY_NAME_MAP:
            return _LEGACY_NAME_MAP[scale]
        raise ValueError(
            f"Unbekannte scale '{scale}'. "
            f"Verfuegbar: {sorted(SCALES.keys())} oder Legacy-Namen "
            f"{sorted(_LEGACY_NAME_MAP.keys())}"
        )
    raise TypeError(f"scale muss int oder str sein, bekam {type(scale).__name__}")


# ============================================================================
# VARIANTS — Tuning-Layer ueber den DEFAULT_PARAMS
# ============================================================================

VARIANTS: Dict[str, Dict] = {
    'default': {
        'description': 'Validierter Baseline aus den Diagnose-Tests',
    },

    'strong_ribbon': {
        'description': 'Doppelte Ribbon-Multiplicity — staerkeres Signal',
        'ribbon_bipolar_to_ganglion': 40,
        'ribbon_horizontal': 6,
        'ribbon_amacrine': 6,
        'ribbon_konio': 20,
    },

    'weak_amacrine': {
        'description': 'Reduzierte Amacrine-Inhibition — weniger Transient',
        'w_amacrine_to_ganglion': -0.4,
        'w_amacrine_to_bipolar': -0.3,
    },

    'strong_konio': {
        'description': 'Staerkere Blue-Yellow-Opponenz',
        'ribbon_konio': 20,
        'w_midget_to_konio_bipolar': -2.5,
        'w_konio_bipolar_to_ganglion': 4.0,
    },

    'fast_response': {
        'description': 'Hoehere direkte Gewichte, weniger Ribbons — transienter',
        'w_bipolar_to_ganglion_midget': 5.0,
        'w_bipolar_to_ganglion_parasol': 4.0,
        'ribbon_bipolar_to_ganglion': 15,
    },

    'conservative': {
        'description': 'Niedrigerer Drive ueberall — weniger Saettigung',
        'w_bipolar_to_ganglion_midget': 2.0,
        'w_bipolar_to_ganglion_parasol': 1.5,
        'ribbon_bipolar_to_ganglion': 15,
    },
}


# ============================================================================
# NEURON-PARAMS — Overrides fuer DEFAULT_NEURON_PARAMS
# ============================================================================

NEURON_OVERRIDES: Dict[str, Dict] = {
    # leer — die Defaults sind validiert.
}


# ============================================================================
# FEEDER-CONFIG-Defaults pro Skala
# ============================================================================

def _feeder_config_for(scale: int) -> Dict:
    return {
        'generator_type': 'step_current',
        'input_resolution': _RESOLUTION_FOR_SCALE[scale],
        'max_current_pa': 800.0,
        'contrast_gain': 2.0,
    }


# ============================================================================
# Cortex-Skala -> Retina-Skala-Empfehlung
# ============================================================================

_GRAPH_TO_RETINA_RECOMMENDATION: Dict[str, int] = {
    'current_v32': 1,    # ~100 PYR pro Areal -> Skala 1
    'small':       2,    # ~300 PYR -> Skala 2-3
    'medium':      3,    # ~500-1000 PYR -> Skala 3
    'workflow':    4,    # ~1000 PYR + 30 GB Ziel -> Skala 4
    'large':       5,    # ~2000+ PYR -> Skala 5
}


# ============================================================================
# API
# ============================================================================

def get_config(scale: Union[int, str] = 3,
               variant: str = 'default') -> Tuple[Dict, Dict, Dict]:
    """Liefert (params, neuron_params, feeder_config) fuer die Retina.

    Args:
        scale: 1..5 (oder Legacy-String 'tiny'/'small'/'medium'/'large').
               Default ist 3 (iterative Entwicklung).
        variant: einer von VARIANTS.keys() — kontrolliert Tuning.

    Returns:
        Tuple aus drei Dicts:
          params         -> direkt an Retina(params=...)
          neuron_params  -> direkt an Retina(neuron_params=...) (kann {} sein)
          feeder_config  -> direkt an retina.create_input_feeder(...)

    Raises:
        ValueError: wenn scale oder variant nicht existiert.
    """
    scale_id = _resolve_scale(scale)

    if variant not in VARIANTS:
        raise ValueError(
            f"Unbekannte variant '{variant}'. Verfuegbar: {sorted(VARIANTS.keys())}"
        )

    # 1. Skalen-Werte als Basis
    params: Dict = {}
    scale_dict = SCALES[scale_id]
    for k in ('n_cones_foveal', 'n_cones_peripheral', 'n_S_cones', 'n_rods'):
        params[k] = scale_dict[k]

    # 1b. Per-Skala param_overrides (z.B. parasol_to_midget_ratio_peripheral
    #     für 1:1-Grid-Match in Skala 4)
    scale_overrides = scale_dict.get('param_overrides', {})
    params.update(scale_overrides)

    # 2. Variant-Overlay
    variant_dict = deepcopy(VARIANTS[variant])
    variant_dict.pop('description', None)
    params.update(variant_dict)

    # 3. Neuron-Overrides (deep merge)
    neuron_params: Dict = deepcopy(NEURON_OVERRIDES.get(variant, {}))

    # 4. Feeder-Config
    feeder_config = _feeder_config_for(scale_id)

    return params, neuron_params, feeder_config


def list_scales():
    """Listet alle verfuegbaren Skalen mit Beschreibung."""
    return [(k, v['description']) for k, v in sorted(SCALES.items())]


def list_variants():
    """Listet alle verfuegbaren Varianten mit Beschreibung."""
    return [(k, v['description']) for k, v in VARIANTS.items()]


def recommend_for_graph(graph_scale: str) -> int:
    """Empfiehlt eine Retina-Skala (1..5) passend zur Cortex-Pop-Größe.

    Args:
        graph_scale: 'current_v32', 'small', 'medium', 'workflow', 'large'

    Returns:
        Skalen-Nummer 1..5.
    """
    if graph_scale not in _GRAPH_TO_RETINA_RECOMMENDATION:
        valid = list(_GRAPH_TO_RETINA_RECOMMENDATION.keys())
        raise ValueError(
            f"Unbekannte graph_scale '{graph_scale}'. Verfuegbar: {valid}"
        )
    return _GRAPH_TO_RETINA_RECOMMENDATION[graph_scale]


def describe(scale: Union[int, str], variant: str = 'default') -> str:
    """Menschenlesbare Zusammenfassung der gewaehlten Konfiguration."""
    try:
        scale_id = _resolve_scale(scale)
    except (ValueError, TypeError) as e:
        return f"Unbekannt: scale={scale} ({e})"

    if variant not in VARIANTS:
        return f"Unbekannt: variant={variant}"

    s = SCALES[scale_id]
    total_photo = (s['n_cones_foveal'] + s['n_cones_peripheral']
                    + s['n_S_cones'] + s['n_rods'])
    res = _RESOLUTION_FOR_SCALE[scale_id]

    foveola_pixel_side = int(res[0] * 0.17)
    foveola_pixels = foveola_pixel_side * foveola_pixel_side
    pixels_per_foveal_cone = foveola_pixels / max(1, s['n_cones_foveal'])

    if pixels_per_foveal_cone >= 1.0:
        coverage_status = 'OK'
    elif pixels_per_foveal_cone >= 0.3:
        coverage_status = 'knapp (mehrere Cones pro Pixel — Konvergenz fangt es ab)'
    else:
        coverage_status = 'SUBNYQUIST! Aufloesung erhoehen'

    return (
        f"Retina-Konfiguration:\n"
        f"  Skala         : {scale_id}\n"
        f"                  {s['description']}\n"
        f"                  {s['use_case']}\n"
        f"  Variant       : {variant}\n"
        f"                  {VARIANTS[variant]['description']}\n"
        f"  ----- Photorezeptoren -----\n"
        f"  Foveal cones  : {s['n_cones_foveal']:>10,}\n"
        f"  Peri cones    : {s['n_cones_peripheral']:>10,}\n"
        f"  S cones       : {s['n_S_cones']:>10,}\n"
        f"  Rods          : {s['n_rods']:>10,}\n"
        f"  Photo gesamt  : {total_photo:>10,}\n"
        f"  ----- Estimates -----\n"
        f"  Total Neurone : {s['estimated_total']:>10,}\n"
        f"  Build-Zeit    : {s['estimated_build_seconds']:>4} s (geschaetzt)\n"
        f"  RAM-Bedarf    : {s['estimated_ram_mb']:>4} MB (geschaetzt, NEST nodes+conns)\n"
        f"  ----- Feeder -----\n"
        f"  Input-Reso    : {res[0]}x{res[1]} = {res[0]*res[1]:,} px/Frame\n"
        f"  Foveola-Patch : {foveola_pixel_side}x{foveola_pixel_side} = {foveola_pixels:,} px\n"
        f"  Pixel/Cone    : {pixels_per_foveal_cone:.1f}  ({coverage_status})\n"
    )


if __name__ == '__main__':
    print("Verfuegbare Skalen:")
    for k, d in list_scales():
        print(f"  {k}  {d}")
    print()
    print("Verfuegbare Varianten:")
    for k, d in list_variants():
        print(f"  {k:14s}  {d}")
    print()
    for s in [1, 2, 3, 4, 5]:
        print(describe(s, 'default'))
        print()
    print("Empfehlungen Cortex-Skala -> Retina-Skala:")
    for cs in ['current_v32', 'small', 'medium', 'workflow', 'large']:
        print(f"  {cs:14s} -> Skala {recommend_for_graph(cs)}")
    print()
    print("Backwards-Compat (Legacy-Namen):")
    for legacy in ['tiny', 'small', 'medium', 'large']:
        print(f"  '{legacy}' -> Skala {_LEGACY_NAME_MAP[legacy]}")
