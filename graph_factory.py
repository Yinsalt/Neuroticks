"""
GraphFactory — baut N "Brain-Sets" aus einer graphs.json (oder v*.json) Blaupause.

Pro Retina-Position werden ALLE Graphen aus der JSON gebaut und senkrecht
zur Retina-Achse versetzt. Nutzt Graph.load_all_from_json() mit id_offset,
um Cross-Graph-Connections innerhalb des Files automatisch korrekt zu remappen.

USAGE:
    positions = [(0,0,0), (10,0,0), (20,0,0)]

    agents = build_graph_array(
        retina_positions=positions,
        perpendicular_distance=15.0,
        perpendicular_axis='y',
        json_path='v204.json',   # oder 'graphs.json'
    )

    # agents[i] = {
    #     'graphs': {graph_name: Graph, ...},
    #     'position': np.ndarray [x, y, z],
    #     'retina_position': np.ndarray,
    #     'recorders': [...],
    # }

    lgn = agents[0]['graphs']['LGN_RELAY_POPS']
    controls = agents[0]['graphs']['Controls']
"""

import json
import os
import re
import numpy as np
from copy import deepcopy
from typing import Dict, List, Optional, Any, Sequence

from neuron_toolbox import Graph


# ID-Offset pro Agent, damit Cross-Graph-Connections im JSON korrekt remappt
# werden. Agent 0 bekommt 1000+, Agent 1 bekommt 2000+, usw.
# In load_all_from_json wird id_offset auf jede graph_id im Block addiert,
# UND auf die graph_id-Targets aller connections.
ID_OFFSET_PER_AGENT = 1000


def _shift_positions_in_blueprint(blueprint_data: Dict,
                                   origin_offset: np.ndarray) -> Dict:
    """
    Tiefe Kopie des JSON-Daten-Dicts; verschiebt alle Positionen
    (init_position, parameters.m, top-level center_of_mass, parameters.center_of_mass,
    parameters.old_center_of_mass) um origin_offset.

    positions-Listen (per-neuron Koordinaten) werden auch verschoben falls
    nicht-leer.

    Original-Daten werden nicht modifiziert.
    """
    data = deepcopy(blueprint_data)

    for g in data.get('graphs', []):
        if 'init_position' in g:
            ip = np.array(g['init_position'], dtype=float) + origin_offset
            g['init_position'] = ip.tolist()

        for n in g.get('nodes', []):
            params = n.setdefault('parameters', {})

            for key in ('m', 'center_of_mass', 'old_center_of_mass'):
                if key in params and params[key] is not None:
                    arr = np.array(params[key], dtype=float)
                    if arr.shape == (3,):
                        params[key] = (arr + origin_offset).tolist()

            if 'center_of_mass' in n and n['center_of_mass'] is not None:
                arr = np.array(n['center_of_mass'], dtype=float)
                if arr.shape == (3,):
                    n['center_of_mass'] = (arr + origin_offset).tolist()

            # positions: list of arrays (per-neuron), kann leer sein
            if 'positions' in n and n['positions']:
                shifted_positions = []
                for cluster in n['positions']:
                    if cluster is None:
                        shifted_positions.append(None)
                        continue
                    arr = np.asarray(cluster, dtype=float)
                    if arr.size == 0:
                        shifted_positions.append(arr.tolist() if hasattr(arr, 'tolist') else cluster)
                        continue
                    shifted_positions.append((arr + origin_offset).tolist())
                n['positions'] = shifted_positions

    return data


def build_graph_array(
    retina_positions: Sequence[Sequence[float]],
    perpendicular_distance: float,
    perpendicular_axis: str = 'y',
    json_path: str = 'graphs.json',
    verbose: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Baut N Brain-Sets. Pro Agent werden ALLE Graphen aus der JSON gebaut.

    Returns:
        Dict[agent_idx, {
            'graphs': Dict[graph_name -> Graph],
            'position': np.ndarray (Brain-Origin),
            'retina_position': np.ndarray,
            'recorders': List[Dict],
        }]
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON nicht gefunden: {json_path}")

    with open(json_path, 'r') as f:
        blueprint_data = json.load(f)

    graph_blocks = blueprint_data.get('graphs', [])
    if not graph_blocks:
        raise ValueError(f"{json_path} enthält keine 'graphs'")

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if perpendicular_axis not in axis_map:
        raise ValueError("perpendicular_axis muss 'x', 'y' oder 'z' sein")
    axis_idx = axis_map[perpendicular_axis]

    perp_offset = np.zeros(3)
    perp_offset[axis_idx] = perpendicular_distance

    retina_positions = [np.array(p, dtype=float) for p in retina_positions]
    n_agents = len(retina_positions)
    n_graphs_per_agent = len(graph_blocks)

    if verbose:
        print(f"\n{'='*60}")
        print(f"build_graph_array: {n_agents} Agents x "
              f"{n_graphs_per_agent} Graphen aus '{json_path}'")
        for bp in graph_blocks:
            print(f"  - '{bp.get('graph_name', 'unnamed')}' "
                  f"({len(bp.get('nodes', []))} Nodes)")
        print(f"Perp offset: {perpendicular_distance} auf "
              f"{perpendicular_axis}-Achse")
        print(f"{'='*60}")

    agents_out = {}

    for agent_idx, retina_pos in enumerate(retina_positions):
        graph_origin = retina_pos + perp_offset

        if verbose:
            print(f"\n--- Agent {agent_idx} @ Brain-Origin {graph_origin} ---")

        # Origin-Shift des Blueprints
        shifted_data = _shift_positions_in_blueprint(blueprint_data,
                                                      graph_origin)

        # ID-Offset: Agent 0 -> 1000, Agent 1 -> 2000, etc.
        # Cross-Graph-Connections werden automatisch von load_all_from_json
        # remappt (innerhalb des Files; Aggregator->Liquid bleibt korrekt).
        id_off = (agent_idx + 1) * ID_OFFSET_PER_AGENT

        graphs_loaded = Graph.load_all_from_json(
            shifted_data,
            populate=True,
            build_connections=True,
            id_offset=id_off,
            verbose=verbose,
        )

        agent_graphs = {g.graph_name: g for g in graphs_loaded}

        agents_out[agent_idx] = {
            'graphs': agent_graphs,
            'position': graph_origin,
            'retina_position': retina_pos,
            'recorders': [],
        }

        # ─── DIAGNOSE: connection-graph_ids in node.connections ───
        # Nach dem Build prüfen: zeigen die Node-Connections auf existierende
        # Graphen (mit dem gleichen id_offset) oder auf die alten unremappten?
        if verbose:
            existing_gids = {g.graph_id for g in graphs_loaded}
            target_counts = {}  # tgt_gid -> count
            for g in graphs_loaded:
                for n in g.node_list:
                    for c in (n.connections or []):
                        tgt_gid = c.get('target', {}).get('graph_id')
                        if tgt_gid is None: continue
                        target_counts[tgt_gid] = target_counts.get(tgt_gid, 0) + 1
            unreachable = {k: v for k, v in target_counts.items()
                           if k not in existing_gids}
            print(f"  [diag] existing graph_ids: "
                  f"{sorted(existing_gids)}")
            print(f"  [diag] connection target counts: "
                  f"{dict(sorted(target_counts.items()))}")
            if unreachable:
                print(f"  [diag] ⚠ unreachable connection targets: "
                      f"{unreachable}")

    # Recorder pro Agent sammeln (falls Devices in JSON definiert)
    for agent_data in agents_out.values():
        recs = []
        for graph in agent_data['graphs'].values():
            recs.extend(_collect_recorders_for_graph(graph))
        agent_data['recorders'] = recs

    if verbose:
        total_graphs = sum(len(a['graphs']) for a in agents_out.values())
        total_nodes = sum(len(g.node_list)
                          for a in agents_out.values()
                          for g in a['graphs'].values())
        total_recs = sum(len(a['recorders']) for a in agents_out.values())
        print(f"\n{'='*60}")
        print(f"FERTIG: {n_agents} Agents, {total_graphs} Graphen, "
              f"{total_nodes} Nodes, {total_recs} Recorder")
        print(f"{'='*60}")

    return agents_out


def _collect_recorders_for_graph(graph: Graph) -> List[Dict]:
    """Sammelt alle aktiven spike_recorder/multimeter eines Graphen
    (falls Devices in JSON definiert wurden)."""
    import nest

    active = []
    for node in graph.node_list:
        if not hasattr(node, 'devices'):
            continue
        for dev in node.devices:
            gid_col = dev.get('runtime_gid')
            if gid_col is None:
                continue
            try:
                if hasattr(gid_col, 'tolist'):
                    gid_int = int(gid_col.tolist()[0])
                elif isinstance(gid_col, list):
                    gid_int = int(gid_col[0])
                else:
                    gid_int = int(gid_col)
            except (ValueError, TypeError, IndexError):
                continue
            model = dev.get('model', '')
            if 'recorder' in model or 'meter' in model:
                try:
                    handle = nest.NodeCollection([gid_int])
                    active.append({
                        'handle': handle,
                        'graph_id': graph.graph_id,
                        'graph_name': graph.graph_name,
                        'node_id': node.id,
                        'device_id': str(dev.get('id', '?')),
                        'model': model,
                    })
                except Exception as e:
                    print(f"  ⚠ device {gid_int} attach failed: {e}")
    return active


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def find_latest_graph_json(search_dir: str) -> Optional[str]:
    """Findet höchste v*.json oder V*.json im Verzeichnis. Returns Pfad oder None."""
    pattern = re.compile(r'^[Vv](\d+)\.json$')
    candidates = []
    try:
        for fname in os.listdir(search_dir):
            m = pattern.match(fname)
            if m:
                candidates.append((int(m.group(1)), fname))
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return os.path.join(search_dir, candidates[0][1])


def get_pop_for_graph_node(graph: Graph, node_name: str):
    """Gibt die NEST-NodeCollection einer Pop in einem Graph zurück.
    None wenn nicht gefunden oder leer."""
    for node in graph.node_list:
        if node.name != node_name:
            continue
        if not node.population or node.population[0] is None:
            return None
        pop = node.population[0]
        if len(pop) == 0:
            return None
        return pop
    return None


# ═══════════════════════════════════════════════════════════════════
#  VERIFICATION
# ═══════════════════════════════════════════════════════════════════

def verify_graph_positions(
    agents: Dict[int, Dict[str, Any]],
    verbose: bool = True,
) -> bool:
    """Prüft dass alle Graphen aller Agents räumlich getrennt sind."""
    if verbose:
        print(f"\n{'='*60}")
        print("VERIFY: Graph-Positionen")
        print(f"{'='*60}")

    n = len(agents)
    if n < 2:
        if verbose:
            print("Nur ein Agent — kein Cross-Vergleich nötig.")
        return True

    all_good = True

    if verbose:
        print(f"\n[1] Brain-Origins:")
    origins = {}
    for i, ad in agents.items():
        origins[i] = ad['position']
        names = list(ad['graphs'].keys())
        if verbose:
            print(f"  Agent {i}: {origins[i]}  ({len(names)} graphs)")

    for i in range(n):
        for j in range(i + 1, n):
            if np.allclose(origins[i], origins[j]):
                print(f"  FEHLER: Agent {i} und {j} haben identischen Origin")
                all_good = False

    if verbose:
        print(f"\n[2] Cross-Agent Node-Position Check:")

    common_graphs = set(agents[0]['graphs'].keys())
    for i in range(1, n):
        common_graphs &= set(agents[i]['graphs'].keys())

    matches = 0
    for graph_name in common_graphs:
        for node_id in {nd.id for nd in agents[0]['graphs'][graph_name].node_list}:
            coms = []
            for i in range(n):
                graph = agents[i]['graphs'].get(graph_name)
                if graph is None:
                    continue
                node = next((nd for nd in graph.node_list if nd.id == node_id), None)
                if node is not None and hasattr(node, 'center_of_mass'):
                    coms.append((i, np.asarray(node.center_of_mass)))
            for ii in range(len(coms)):
                for jj in range(ii + 1, len(coms)):
                    if np.allclose(coms[ii][1], coms[jj][1]):
                        print(f"  FEHLER: Agent {coms[ii][0]} und {coms[jj][0]} "
                              f"haben in '{graph_name}' Node {node_id} "
                              f"auf identischer Position {coms[ii][1]}")
                        matches += 1
                        all_good = False

    if verbose and matches == 0:
        print(f"  OK: alle Cross-Agent Positionen unterschiedlich "
              f"({len(common_graphs)} gemeinsame Graphen)")

    if verbose:
        print(f"\n{'='*60}")
        print("VERIFY: " + ("✓ sauber" if all_good else "✗ Konflikte"))
        print(f"{'='*60}\n")

    return all_good
