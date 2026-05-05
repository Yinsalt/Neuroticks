"""
Headless run script for a Neuroticks project export.

Layout:
  ./graphs.json            project data (nodes, connections, devices)
  ./neuron_toolbox.py      core builders (Graph, Node, populate_node, VT cache)
  ./functional_models.json model presets (optional, for graph creator UI)
  ./run.py                 this file
  ./Simulation_History/    output JSON files (created on first run)

What this script does:
  1. Resets NEST and clears the dopamine-VT cache
  2. Loads graphs.json via Graph.load_all_from_json — the same canonical
     loader used by the GUI and server.py. This handles thin/fat formats
     transparently, rebuilds cross-graph topology backrefs, sets up
     volume_transmitters for stdp_dopamine_synapse connections, and wires
     all connections in one sweep.
  3. Caches every spike_recorder / multimeter device into a flat list
  4. Runs nest.Simulate() in DUMP_INTERVAL chunks across one or more
     RUNS, flushing recorders after each chunk
  5. Writes a history JSON loadable in the Neuroticks History browser

Edit the `=== CONFIGURATION ===` block at the top to change runtime
behavior — no code changes below should be necessary for typical use.
"""
import nest
import numpy as np
import json
import os
import sys
import copy
from pathlib import Path
from datetime import datetime

import neuron_toolbox
from neuron_toolbox import Graph, Node, clear_dopa_vt_cache


# ════════════════════════════════════════════════════════════════════
# === CONFIGURATION ==================================================
# ════════════════════════════════════════════════════════════════════

# --- Files ---
JSON_FILENAME    = "__JSON_FILENAME__"
OUTPUT_DIR       = Path("Simulation_History")
OUTPUT_PREFIX    = "history_headless"   # → history_headless_<runIdx>_<timestamp>.json

# --- NEST kernel ---
NEST_RESOLUTION  = 0.1                  # ms
NUM_THREADS      = __NUM_THREADS__      # local_num_threads
RNG_SEED         = None                 # int or None for NEST default
PRINT_TIME       = False                # NEST's own progress prints
ENABLE_STRUCTURAL_PLASTICITY = True     # tolerated as no-op on builds without it

# --- Simulation timing ---
SIM_DURATION     = __SIM_DURATION__     # ms per run
DUMP_INTERVAL    = 2000.0               # ms — flush recorders every N ms
NUM_RUNS         = 1                    # repeat the whole simulation N times
                                        # each run starts from a fresh kernel
                                        # → independent statistics

# --- Build-time verbosity ---
VERBOSE_BUILD    = False                # True = neuron_toolbox prints per-node info

# --- Output ---
HISTORY_INDENT   = None                 # None = compact, 2 = pretty

# ════════════════════════════════════════════════════════════════════
# === END CONFIG — code below should not need editing ================
# ════════════════════════════════════════════════════════════════════


class NumpyNestEncoder(json.JSONEncoder):
    """Numpy-aware JSON encoder. NEST commonly returns np scalars."""
    def default(self, obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _get_nest_time():
    """NEST 3.x exposes simulated time as biological_time. The legacy
    'time' key returns 0.0 in 3.8 — same fix as the GUI."""
    try:
        st = nest.GetKernelStatus()
        return float(st.get("biological_time", st.get("time", 0.0)) or 0.0)
    except Exception:
        return 0.0


def init_kernel():
    """Reset and configure the NEST kernel.  Always clears the dopa-VT
    cache too, otherwise stdp_dopamine_synapse setups crash on the
    second run with `UnknownNode in CopyModel_l_l_D`."""
    nest.ResetKernel()
    clear_dopa_vt_cache()
    
    kernel = {
        "resolution": NEST_RESOLUTION,
        "print_time": PRINT_TIME,
        "local_num_threads": int(NUM_THREADS),
    }
    if RNG_SEED is not None:
        kernel["rng_seed"] = int(RNG_SEED)
    nest.SetKernelStatus(kernel)
    
    if ENABLE_STRUCTURAL_PLASTICITY:
        try:
            nest.EnableStructuralPlasticity()
        except Exception:
            # Not all NEST builds have it — non-fatal.
            pass
    
    if hasattr(neuron_toolbox, "VERBOSE_BUILD"):
        neuron_toolbox.VERBOSE_BUILD = bool(VERBOSE_BUILD)


def load_and_build():
    """Load graphs.json and build the entire network in NEST.
    
    Uses Graph.load_all_from_json — the canonical loader. This single
    call handles:
      - thin / fat format (positions optional)
      - cross-graph connections (source / target / da_source remap)
      - volume_transmitter setup for stdp_dopamine_synapse
      - next_ids / prev_ids reconstruction
      - shape_params_per_pop, population_nest_params, devices
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, JSON_FILENAME)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    print(f"Loading {JSON_FILENAME}...")
    graphs = Graph.load_all_from_json(
        json_path,
        populate=True,
        build_connections=True,
        verbose=VERBOSE_BUILD,
    )
    
    # Index by graph_id for downstream code (cache_active_recorders etc.)
    graphs_by_id = {g.graph_id: g for g in graphs}
    
    n_nodes = sum(len(g.node_list) for g in graphs)
    n_conns = sum(
        len(getattr(n, "connections", []) or [])
        for g in graphs for n in g.node_list
    )
    print(f"  {len(graphs)} graphs, {n_nodes} nodes, {n_conns} connections")
    return graphs_by_id


def cache_active_recorders(graphs):
    """Flatten every recording device across all graphs into one list,
    so the per-chunk flush doesn't have to retraverse the graph tree."""
    active = []
    for graph in graphs.values():
        for node in graph.node_list:
            for dev in (getattr(node, "devices", []) or []):
                gid_col = dev.get("runtime_gid")
                if gid_col is None:
                    continue
                # runtime_gid can be a NodeCollection, list, or int —
                # NEST's API returns NodeCollection on Create() but the
                # GUI sometimes serializes it as int. Normalize.
                try:
                    if hasattr(gid_col, "tolist"):
                        gid_int = int(gid_col.tolist()[0])
                    elif isinstance(gid_col, list):
                        gid_int = int(gid_col[0])
                    else:
                        gid_int = int(gid_col)
                except (ValueError, TypeError, IndexError):
                    continue
                
                model = dev.get("model", "")
                if "recorder" not in model and "meter" not in model:
                    continue
                try:
                    handle = nest.NodeCollection([gid_int])
                except Exception as e:
                    print(f"  Warning: couldn't get NodeCollection for gid {gid_int}: {e}")
                    continue
                active.append({
                    "handle": handle,
                    "graph_id": graph.graph_id,
                    "node_id": node.id,
                    "device_id": str(dev.get("id", gid_int)),
                    "model": model,
                })
    
    print(f"  {len(active)} active recording devices")
    if not active:
        print("  WARNING: no spike_recorders or multimeters — simulation will produce no data.")
    return active


def collect_and_flush(active_recorders):
    """Pull buffered events out of every recorder and reset its counter.
    Without the reset, NEST's internal buffer grows unbounded over long
    runs; with the reset, we get clean per-chunk segments."""
    collected = {}
    total_events = 0
    
    for rec in active_recorders:
        try:
            status = nest.GetStatus(rec["handle"])[0]
            n_events = status.get("n_events", 0)
            if n_events <= 0:
                continue
            events = status.get("events", {})
            clean_data = {
                k: v.tolist() if isinstance(v, np.ndarray) else list(v)
                for k, v in events.items()
            }
            key = (rec["graph_id"], rec["node_id"], rec["device_id"])
            collected[key] = {"model": rec["model"], "data": clean_data}
            total_events += n_events
            nest.SetStatus(rec["handle"], {"n_events": 0})
        except Exception as e:
            print(f"  Read error on device {rec['device_id']}: {e}")
    
    return collected, total_events


def simulate_one_run(graphs, recorders):
    """Run nest.Simulate() in DUMP_INTERVAL chunks until SIM_DURATION
    is reached. Returns the list of per-chunk recorder dumps."""
    accumulated = []
    current_time = 0.0
    step_count = 0
    
    while current_time < SIM_DURATION:
        step = min(DUMP_INTERVAL, SIM_DURATION - current_time)
        nest.Simulate(step)
        
        # NEST's biological_time is authoritative; the local accumulator
        # is a fallback in case the kernel state is somehow inaccessible.
        nest_t = _get_nest_time()
        current_time = max(current_time + step, nest_t)
        step_count += 1
        
        run_data, n_events = collect_and_flush(recorders)
        accumulated.append(run_data)
        
        pct = current_time / SIM_DURATION * 100
        print(f"    [{pct:5.1f}%] {current_time:.0f}/{SIM_DURATION:.0f} ms"
              f"  |  {n_events} events")
    
    return accumulated, step_count


def build_history_file(graphs, accumulated_runs, run_index, total_runs):
    """Build a v3.0 history file that's loadable by the Neuroticks
    History Browser. accumulated_runs is a list-of-list-of-dict:
      [run0_chunks, run1_chunks, ...]
    where each chunk is {(gid, nid, did): {model, data}}.
    """
    history = {
        "meta": {
            "version": "3.0",
            "type": "neuroticks_history",
            "timestamp": str(datetime.now()),
            "source": "headless_export",
            "run_index": run_index,
            "total_runs": total_runs,
            "sim_duration_ms": SIM_DURATION,
            "dump_interval_ms": DUMP_INTERVAL,
        },
        "graphs": [],
    }
    
    for graph in graphs.values():
        g_entry = {
            "graph_id": graph.graph_id,
            "graph_name": getattr(graph, "graph_name", f"Graph_{graph.graph_id}"),
            "nodes": [],
        }
        
        for node in graph.node_list:
            clean_devices = []
            for dev in (getattr(node, "devices", []) or []):
                d_copy = dev.copy()
                d_copy.pop("runtime_gid", None)
                clean_devices.append(d_copy)
            
            n_entry = {
                "id": node.id,
                "name": getattr(node, "name", f"Node_{node.id}"),
                "neuron_models": getattr(node, "neuron_models", []) or [],
                "types": list(getattr(node, "types", []) or []),
                "devices": clean_devices,
                "results": {"history": []},
            }
            
            for chunk in accumulated_runs:
                run_entry = {"devices": {}}
                for (gid, nid, did), dev_result in chunk.items():
                    if gid == graph.graph_id and nid == node.id:
                        run_entry["devices"][did] = dev_result
                if run_entry["devices"]:
                    n_entry["results"]["history"].append(run_entry)
            
            g_entry["nodes"].append(n_entry)
        
        history["graphs"].append(g_entry)
    
    return history


def write_history(history, output_file):
    with open(output_file, "w") as f:
        json.dump(history, f, cls=NumpyNestEncoder, indent=HISTORY_INDENT)
    
    total_recordings = sum(
        len(run.get("devices", {}))
        for g in history["graphs"]
        for n in g["nodes"]
        for run in n["results"]["history"]
    )
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  → {output_file}")
    print(f"    {size_mb:.1f} MB, {total_recordings} device entries")


def run_one(run_index):
    """Initialize, build, simulate, write — one full run."""
    print(f"\n{'='*64}")
    print(f"Run {run_index + 1} / {NUM_RUNS}")
    print(f"{'='*64}")
    
    init_kernel()
    graphs = load_and_build()
    recorders = cache_active_recorders(graphs)
    
    print(f"\n  Simulating {SIM_DURATION} ms (dump every {DUMP_INTERVAL} ms)...")
    accumulated, step_count = simulate_one_run(graphs, recorders)
    print(f"  {step_count} chunks collected")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if NUM_RUNS > 1:
        fname = f"{OUTPUT_PREFIX}_run{run_index + 1:03d}_{timestamp}.json"
    else:
        fname = f"{OUTPUT_PREFIX}_{timestamp}.json"
    output_file = OUTPUT_DIR / fname
    
    history = build_history_file(graphs, accumulated, run_index, NUM_RUNS)
    write_history(history, output_file)
    return output_file


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Neuroticks headless export")
    print(f"  JSON:        {JSON_FILENAME}")
    print(f"  Sim duration: {SIM_DURATION} ms × {NUM_RUNS} run(s)")
    print(f"  Threads:     {NUM_THREADS}")
    print(f"  Output dir:  {OUTPUT_DIR.resolve()}")
    
    outputs = []
    for i in range(int(NUM_RUNS)):
        try:
            outputs.append(run_one(i))
        except Exception as e:
            print(f"\n  Run {i + 1} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*64}")
    print(f"DONE — {len(outputs)} / {NUM_RUNS} runs successful")
    print(f"{'='*64}")
    for p in outputs:
        print(f"  {p}")
    print("\nLoadable in the Neuroticks History Browser.")


if __name__ == "__main__":
    main()
