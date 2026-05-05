"""
RetinaFactory — nimmt ein Positions-Array entgegen.

Du übergibst eine Liste von Positionen [(x,y,z), ...] und bekommst eine
Retina pro Position. Außerdem ist die Retina->LGN-Verkabelungsfunktion
hier drin, weil sie zum Biom-Kern gehört.

USAGE:
    import nest
    nest.ResetKernel()
    nest.resolution = 0.1

    from retina_factory import build_retina_array, connect_retina_to_lgn

    positions = [(0,0,0), (10,0,0), (20,0,0)]

    retinas = build_retina_array(
        positions=positions,
        scale='3',
        variant='default',
    )

    # Pro Agent das Brain-Set bauen (graph_factory.build_graph_array)
    # und dann die Retina ans LGN verkabeln:
    connect_retina_to_lgn(retinas[0]['retina'], agent_graphs_by_name)

    retinas[0]['feeder'].feed(lms, intensity)
    nest.Simulate(50.0)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Sequence
from copy import deepcopy

from retina_main import Retina
from retina_scales import get_config


# ═══════════════════════════════════════════════════════════════════
# RETINA-ARRAY BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_retina_array(
    positions: Sequence[Sequence[float]],
    scale: str = '3',
    variant: str = 'default',
    feeder_config: Optional[Dict] = None,
    foveola_angle_deg: Optional[float] = 20.0,
    foveal_cone_multiplier: Optional[float] = 4.0,
    verbose: bool = True,
    min_spacing_check: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Erstellt eine Retina pro Position.

    Args:
        positions:     Liste von (x, y, z) Tupeln — eine Position pro Retina.
        scale:         '1'..'5' (oder Legacy 'tiny'/'small'/'medium'/'large').
        variant:       'default', 'strong_ribbon', etc.
        feeder_config: Überschreibt Feeder-Defaults für alle Retinas.
        foveola_angle_deg:        wenn nicht None: param-override (Default 20).
        foveal_cone_multiplier:   wenn nicht None und nicht 1.0: skaliert
                                  n_cones_foveal (Default 4.0).
        verbose:       Debug-Output
        min_spacing_check: Warnt wenn Positionen zu dicht beieinander sind.

    Returns:
        Dict mit Integer-Keys (0, 1, 2, ...) und pro Retina:
            'retina':      Retina-Objekt
            'feeder':      RetinaInputFeeder
            'recorders':   Dict[str, nest.NodeCollection]
            'position':    np.ndarray [x, y, z]
            'output_pops': Dict[str, nest.NodeCollection]
            'feeder_cfg':  Dict mit der Feeder-Konfiguration (input_resolution!)
    """
    import nest

    positions = [np.array(p, dtype=float) for p in positions]
    n = len(positions)

    if n == 0:
        raise ValueError("positions-Liste ist leer")

    # FIX: get_config liefert 3 Werte, nicht 2.
    base_params, base_neuron_params, base_feeder_cfg = get_config(scale, variant)

    # Foveola-Anpassungen wie im health-check (besser für Mini-Auge)
    if foveola_angle_deg is not None:
        base_params['foveola_angle_deg'] = float(foveola_angle_deg)
    if (foveal_cone_multiplier is not None and
            foveal_cone_multiplier != 1.0 and
            'n_cones_foveal' in base_params):
        base_params['n_cones_foveal'] = int(
            base_params['n_cones_foveal'] * float(foveal_cone_multiplier)
        )

    radius = base_params.get('radius', 2.0)
    if min_spacing_check:
        _warn_if_too_close(positions, min_dist=3.0 * radius)

    # User-feeder_config Overlay über das aus get_config
    effective_feeder_cfg = deepcopy(base_feeder_cfg)
    if feeder_config:
        effective_feeder_cfg.update(feeder_config)

    if verbose:
        print(f"\n{'='*60}")
        print(f"build_retina_array: {n} Retinas, "
              f"scale={scale}, variant={variant}")
        print(f"{'='*60}")

    retinas = {}

    for i, pos in enumerate(positions):
        if verbose:
            print(f"\n--- Retina {i} @ {pos} ---")

        params = deepcopy(base_params)
        neuron_params = deepcopy(base_neuron_params)
        params['origin'] = tuple(pos)

        retina = Retina(params=params, neuron_params=neuron_params,
                        verbose=verbose)
        retina.build()
        retina.connect()

        feeder = retina.create_input_feeder(deepcopy(effective_feeder_cfg))

        recorders = {}
        output_pops = retina.get_output_populations()
        for name, pop in output_pops.items():
            sr = nest.Create('spike_recorder')
            nest.Connect(pop, sr)
            recorders[name] = sr

        retinas[i] = {
            'retina': retina,
            'feeder': feeder,
            'recorders': recorders,
            'position': pos,
            'output_pops': output_pops,
            'feeder_cfg': deepcopy(effective_feeder_cfg),
        }

    if verbose:
        total = sum(sum(r['retina'].get_counts().values())
                    for r in retinas.values())
        print(f"\n{'='*60}")
        print(f"FERTIG: {n} Retinas, {total:,} Neuronen total")
        print(f"{'='*60}")

    return retinas


def _warn_if_too_close(positions: List[np.ndarray], min_dist: float) -> None:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                print(f"WARNING: Retina {i} und {j} nur {d:.2f} auseinander "
                      f"(empfohlen: >= {min_dist:.1f})")


# ═══════════════════════════════════════════════════════════════════
# RETINA -> LGN VERKABELUNG
# ═══════════════════════════════════════════════════════════════════
#
# Übernommen aus graph_health_tracked.connect_retina_to_lgn — gehört
# konzeptionell ins Biom-Kernsystem, nicht ins Diagnose-Skript.
# ═══════════════════════════════════════════════════════════════════

def connect_retina_to_lgn(retina, graphs_by_name, verbose=True):
    """Verkabelt Retina-Ganglien -> LGN-Relays mit fixed_indegree.

    fixed_indegree bleibt skalen-stabil pro LGN-Neuron. Werte gewählt für
    LGN-Output ~15-30 Hz bei medium-Retina mit ~60 Hz Ganglion-Drive.
    Bei kleineren Skalen kappt fixed_indegree automatisch auf den
    verfügbaren Pool.

    Args:
        retina: Retina-Instanz mit get_output_populations()
        graphs_by_name: dict {graph_name: Graph}, muss 'LGN_RELAY_POPS' enthalten
        verbose: Debug-Output

    Returns:
        Anzahl total angelegter Synapsen.
    """
    import nest

    retina_outputs = retina.get_output_populations()

    lgn_relay_graph = graphs_by_name.get('LGN_RELAY_POPS')
    if lgn_relay_graph is None:
        raise RuntimeError("Graph 'LGN_RELAY_POPS' nicht gefunden")

    lgn_nodes_by_name = {node.name: node for node in lgn_relay_graph.node_list}

    # (retina_pop_name, lgn_node_name, indegree_per_lgn_neuron, weight, delay)
    mapping = [
        # Foveal Parvo: 1:1-Fokus, ~5-30 Cones pro Midget-Relay
        ('midget_ON_ganglion_foveal',      'LGN_PARVO_ON_FOVEAL',      30, 1.2, 1.5),
        ('midget_OFF_ganglion_foveal',     'LGN_PARVO_OFF_FOVEAL',     30, 1.2, 1.5),
        # Peripheral Parvo: mehr Konvergenz, größere RFs
        ('midget_ON_ganglion_peripheral',  'LGN_PARVO_ON_PERIPHERAL',  60, 0.8, 1.5),
        ('midget_OFF_ganglion_peripheral', 'LGN_PARVO_OFF_PERIPHERAL', 60, 0.8, 1.5),
        # Magno: Parasol konvergiert ~30-80 pro Magno-Relay; bei kleinen
        # Skalen (n_parasol=25) cappt indegree=50 auf 24 -> 96% connectivity
        # ohne multapses. NEST's FixedInDegreeBuilder geht dabei in einen
        # Rejection-Sampler-Pfad und braucht Minuten pro Verkabelung.
        # 15 Eingänge pro Magno-Relay sind biologisch noch breit (60% bei
        # 25 Sources, skaliert sauber nach oben bei größeren Skalen).
        ('parasol_ON_ganglion',  'LGN_MAGNO_ON',  15, 1.5, 1.5),
        ('parasol_OFF_ganglion', 'LGN_MAGNO_OFF', 15, 1.5, 1.5),
        # Konio: kleinere Konvergenz weil S-Cones rar
        ('konio_ganglion_peripheral', 'LGN_KONIOCELLULAR', 40, 1.0, 1.5),
    ]

    if verbose:
        print('\nVerkabele Retina -> LGN (fixed_indegree):')

    total = 0
    for retina_pop_name, lgn_node_name, indegree, weight, delay in mapping:
        src = retina_outputs.get(retina_pop_name)
        tgt_node = lgn_nodes_by_name.get(lgn_node_name)

        if src is None:
            if verbose:
                print(f"  WARN: Retina-Pop '{retina_pop_name}' nicht gefunden")
            continue
        if tgt_node is None:
            if verbose:
                print(f"  WARN: LGN-Node '{lgn_node_name}' nicht gefunden")
            continue
        if not tgt_node.population:
            if verbose:
                print(f"  WARN: '{lgn_node_name}' hat keine Population")
            continue

        tgt = tgt_node.population[0]

        # fixed_indegree kann nicht mehr verlangen als die Source-Pop hergibt;
        # cappen mit kleinem Sicherheitsabstand (allow_multapses=False)
        n_src = len(src)
        eff_indegree = min(indegree, max(1, n_src - 1))
        capped = (eff_indegree < indegree)

        try:
            nest.Connect(
                src, tgt,
                conn_spec={
                    'rule': 'fixed_indegree',
                    'indegree': eff_indegree,
                    'allow_autapses': False,
                    'allow_multapses': False,
                },
                syn_spec={'weight': weight, 'delay': delay,
                          'synapse_model': 'static_synapse'},
            )
            n_new = len(tgt) * eff_indegree
            total += n_new
            if verbose:
                cap_note = f' (capped from {indegree})' if capped else ''
                print(f"  {retina_pop_name:<32} ({n_src:>4}) -> "
                      f"{lgn_node_name:<26} ({len(tgt):>4}):  {n_new:>5} "
                      f"conns [indeg={eff_indegree}{cap_note}, w={weight}]")
        except Exception as e:
            print(f"  FEHLER {retina_pop_name} -> {lgn_node_name}: {e}")

    if verbose:
        print(f'Gesamt Retina->LGN: {total} connections')
    return total
