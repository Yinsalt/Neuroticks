# NeuroTicks

**A Visual CAD Tool for Spiking Neural Networks**

[![Demo Video](https://img.shields.io/badge/Demo-YouTube-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=cYMDaO1GlLc)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg?style=for-the-badge&logo=python)](https://python.org)
[![NEST 3.8](https://img.shields.io/badge/NEST-3.8-orange.svg?style=for-the-badge)](https://nest-simulator.org)

---

## Demo

[![NeuroTicks Demo](https://img.youtube.com/vi/cYMDaO1GlLc/maxresdefault.jpg)](https://www.youtube.com/watch?v=cYMDaO1GlLc)

*Click to watch the demo on YouTube*

---

## What is NeuroTicks?

NeuroTicks is a graphical construction kit for the **NEST Simulator**. It bridges the gap between abstract network definitions and spatially explicit, biological architectures.

Think of it as **CAD software for the brain** — visually design, manipulate, and wire neural populations in 3D space before running simulations.

If you've ever scripted distance-dependent connectivity, procedural population layouts, or cross-area projections in pure Python, you know it gets messy fast. NeuroTicks is built for **connectomic hypothesis testing**:

- *How does the physical arrangement of populations affect synchronization?*
- *What changes when I tilt this cortical sheet, or relocate this attractor?*
- *Does the network still learn if I switch the synapse model on this projection?*

Solo-developed, single-author. Functional first, polished later.

---

## Features

### Sculpt Neural Tissue

Populations are placed in 3D space using configurable shape primitives — **Grid**, **Blob**, **Ring (CCW)**, **Cone/Column** — or procedural generation via **Wave Function Collapse** and **polynomial flow fields** that warp spatial distributions into biologically plausible morphologies.

Each population in a tool node carries its **own** geometry parameters, neuron model, and NEST parameters. A single CCW node can host one ring of `iaf_psc_alpha` plus another ring of `aeif_cond_alpha` with different radii — they coexist as parallel populations under the same node.

### Connection Rules and Plasticity

Per-projection connection rules: `similarity` (the original CCW/Cone angular-similarity weighting), `one_to_one`, `all_to_all`, `pairwise_bernoulli`, `fixed_indegree`, `fixed_outdegree`. Plus full per-synapse-type configuration:

- Pick `static_synapse`, `stdp_synapse`, `stdp_synapse_hom`, `tsodyks_synapse`, etc.
- Model-specific parameters appear as collapsible UI fields (tau_plus, lambda, alpha, Wmax for STDP; U, tau_rec, tau_fac for Tsodyks; …)
- Plasticity safety net: STDP can't accept negative weights, so inhibitory connections in a CCW/Cone fall back to `static_synapse` automatically. `Wmax` gets auto-adjusted upward when it's smaller than the largest weight in the projection.

### Sew Networks Together

Build functional modules in isolation, then combine them. Import existing graph definitions and use the **Fusion Tool** to merge independent networks into a single simulation. Cross-graph connections are first-class — projections from `LGN` to `V1` to `V2` work identically whether the graphs were built together or fused later.

### Live Interaction

Watch the network spike in real-time:

- **Simulate Lesions** — set connection weights to zero during runtime
- **Tweak Dynamics** — adjust synaptic delays and weights on the fly
- **Inject & Record** — click neurons in 3D, attach voltmeters, inject Poisson currents

### 3D Visualization

Built on **PyVista**. Three modes:

- **Neurons View** — point-cloud of every spiking neuron, colored by model
- **Graph View** — abstracted skeleton with one sphere per node and arrows per connection. Click a node or arrow for full inspection details. Multiple parallel projections between the same pair fan out as separate visible arrows.
- **Firing Patterns View** — animated edges that flash on activity, with adjustable base opacity

### Multi-Agent / Brain-Set Builder

For evolutionary or comparative experiments, scripts in `build_system.py` + `retina_factory.py` + `graph_factory.py` build **N parallel agents** out of a single `graphs.json` blueprint. Each agent gets its own retina and full brain stack, spatially separated in NEST so plasticity stays local. Connections between agents are explicit — fitness comparisons, tournament selection, recombination all happen in user code.

### Headless Export

Design in the GUI, export to Python + JSON. The `run.py` and `build_system.py` scripts run on clusters without graphics, with a `VERBOSE_BUILD` flag in `neuron_toolbox` that switches per-pop build logs on/off depending on whether you want diagnostics or a quiet run.

### Simulation History

Automatic snapshots of every simulation run. Reload past states and compare dynamics through the analysis dashboard — spike rasters, firing-rate histograms, membrane-potential traces.

---

## Architecture

NeuroTicks is organized around two core data structures — `Graph` and `Node` — that form a hierarchical, tree-like representation of neural populations. Graphs own and manage a collection of Nodes. Each Node encapsulates one or more spatial populations, their NEST-level instantiation, attached recording/stimulation devices, and synaptic projections to other Nodes (possibly across Graphs).

The spatial layout of each Node is governed by a `PolynomGenerator` which defines polynomial vector fields (flow fields) used to deform grid-based neuron placements into non-trivial 3D morphologies.

```
┌─────────────────────────────────────────────────────────────────────┐
│                             Graph                                   │
│                                                                     │
│  graph_id, graph_name                                               │
│  node_list: List[Node]          ← ordered list of all nodes        │
│  node_dict: Dict[id|name, Node] ← fast lookup by id or name        │
│  root: Node                     ← first node, tree root            │
│  polynom_generator: PolynomGenerator                                │
│                                                                     │
│  create_node(params, parent, ...) → Node                            │
│  populate(n_iterations, ...)      → builds tree of nodes            │
│  build_all()                      → calls node.build() for all      │
│  build_connections(ext_graphs)    → wires NEST synapses             │
│  remove_node(node) / dispose()                                      │
│  get_node(id) → Node                                                │
├─────────────────────────────────────────────────────────────────────┤
│  1                                                                  │
│  │ owns *                                                           │
│  ▼                                                                  │
│                             Node                                    │
│                                                                     │
│  id, name, graph_id                                                 │
│  parameters: dict               ← full parameter snapshot          │
│  center_of_mass: np.array(3)    ← spatial origin in 3D             │
│                                                                     │
│  ── Per-population data (parallel lists, one entry per pop) ──     │
│  positions: List[np.array(N,3)] ← per-pop neuron coordinates       │
│  population: List[NodeCollection]← NEST neuron handles             │
│  neuron_models: List[str]       ← NEST model per pop               │
│  population_nest_params: List[dict] ← per-pop neuron params       │
│  shape_params_per_pop: List[dict]   ← per-pop tool geometry/syn   │
│  types: List[int]               ← sub-population type indices      │
│                                                                     │
│  devices: List[dict]            ← spike recorders, multimeters, …  │
│  connections: List[dict]        ← outgoing projection descriptors  │
│  results: dict                  ← recorded data (history entries)  │
│                                                                     │
│  ── Tree links ──                                                   │
│  parent: Node | None                                                │
│  prev: List[Node]               ← predecessors (incoming edges)    │
│  next: List[Node]               ← successors  (outgoing edges)     │
│                                                                     │
│  build()            → generates positions via WFC / shapes + flow  │
│  populate_node()    → creates NEST neurons at positions            │
│  connect(registry)  → instantiates NEST synapses from connections  │
│  add_neighbor(other) / remove()                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Connection Model

Connections are stored as declarative dicts on each Node:

```json
{
  "id": 0,
  "source": { "graph_id": 0, "node_id": 0, "pop_id": 0 },
  "target": { "graph_id": 0, "node_id": 1, "pop_id": 2 },
  "params": {
    "rule": "pairwise_bernoulli",
    "p": 0.1,
    "synapse_model": "stdp_synapse",
    "weight": 1.5,
    "delay": 1.0,
    "tau_plus": 20.0,
    "lambda": 0.01,
    "Wmax": 5.0,
    "use_spatial": false
  }
}
```

At simulation time, `Node.connect()` resolves source/target references through a graph registry (cross-graph connections are supported) and delegates to `nest.Connect()` with the appropriate `conn_spec` and `syn_spec`. Spatial connectivity (distance-dependent masks) is supported via NEST's `CreateMask` API when `use_spatial: true`.

Tool nodes (CCW, Cone, Blob) generate their internal projections from a per-pop `shape_params` dict that holds the geometry, weight, delay, connection rule, and synapse model + extras. The tool builder reads this dict instead of node-global values, so multiple populations under the same node can have entirely different connectivity logic.

### Device Model

Devices (spike recorders, multimeters, generators) are attached per-population:

```json
{
  "id": 0,
  "model": "spike_recorder",
  "target_pop_id": 0,
  "params": { "label": "spikes_pop0" },
  "conn_params": { "weight": 1.0, "delay": 1.0 },
  "runtime_gid": null
}
```

Multi-compartment models (`iaf_cond_alpha_mc`, `cm_default`, etc.) are detected automatically — recordables and receptor types are adjusted so that devices connect correctly without manual configuration.

### Connection Lifecycle Safety

When a Node is deleted, its outgoing connections are cleared from its own `connections` list, and incoming connections are removed from every predecessor in `node.prev`. Cross-graph predecessors are tracked through `add_neighbor` calls during connection creation and project loading. As a safety net, every Reset and project Load also runs `prune_dangling_connections()` which sweeps every node's connection list and drops entries whose source or target node no longer exists. This prevents stale references from causing `InvalidNodeCollection` errors when NEST is reset.

---

## Project Structure

```
NeuroTicks/
├── Main.py                  # Entry point, main window, 3D viewport, sim loop
├── neuron_toolbox.py        # Graph, Node, PolynomGenerator, NEST builders
│                            #   create_CCW, connect_cone, create_blob_population
├── WidgetLib.py             # UI components, parameter editors, analysis dashboard
│                            #   InlineSynapseConfig, ConnectionTool,
│                            #   prune_dangling_connections, ...
├── CustomExtension.py       # Simulation history browser
├── ExtraTab.py              # Fusion tool — sewing graphs together
├── OtherTab.py              # User-editable sandbox for ad-hoc widgets
├── functional_models.json   # Preset network templates
│
├── build_system.py          # Multi-agent builder — N retinas + N brain sets
├── retina_factory.py        # Retina array constructor (configurable scales)
├── graph_factory.py         # Graph array constructor from graphs.json
└── visualize.py             # Standalone 3D visualizer for multi-agent setups
```

---

## Installation

### Quick Install (Ubuntu 22.04 / 24.04)

```bash
wget https://raw.githubusercontent.com/Yinsalt/NeuroTicks/main/install.sh
chmod +x install.sh
./install.sh
```

This installs all system dependencies, creates a Python virtual environment, compiles NEST 3.8 from source, and clones NeuroTicks.

After installation:

```bash
source ~/neuroticks/activate.sh
python Main.py
```

### Docker

```bash
docker build -t neuroticks:latest .
./run_neuroticks_docker.sh
```

### Manual Installation

| Package | Version |
|---------|---------|
| Python | 3.10+ |
| NEST Simulator | 3.8 |
| PyQt6 | 6.x |
| PyVista | 0.46+ |
| PyQtGraph | 0.13+ |
| PyOpenGL | 3.1.10 |
| NumPy | 2.x |
| SciPy | 1.x |
| Cython | < 3.0 (for NEST build) |

---

## Usage

### Single Network (GUI)

1. **Create Graph** — new graph structure via menu or button
2. **Add Nodes** — define populations. Pick a tool type (CCW, Cone, Blob, Grid, or Custom WFC) per node; configure per-population geometry, neuron model, NEST parameters, and synapse config separately
3. **Connect** — open the Connection Manager. Pick source/target population, choose connection rule and synapse model. Model-specific parameters (tau_plus, Wmax, ...) appear inline
4. **Attach Devices** — spike recorders, multimeters, current generators
5. **Simulate** — click the global play button. The timer counts in NEST biological time; the 3D view animates spikes live
6. **Analyze** — review past runs in the History tab with rasters, firing-rate histograms, and voltage traces
7. **Save / Load** — projects serialize to JSON, including all per-pop params, devices, and connections

### Multi-Agent (Headless Script)

```python
from build_system import build_system

retinas, agents = build_system(
    num_agents=4,
    retina_spacing=10.0,
    perpendicular_distance=15.0,
    retina_scale='tiny',
    graphs_json='graphs.json',
)

# Each agent is independent — feed it visual input, simulate, read out spikes
for i in range(len(retinas)):
    lms = get_video_frame_for_agent(i)
    retinas[i]['feeder'].feed(lms, intensity=lms.mean(axis=2))

nest.Simulate(50.0)

for i in range(len(retinas)):
    spike_counts = {
        name: nest.GetStatus(rec, 'n_events')[0]
        for name, rec in retinas[i]['recorders'].items()
    }
```

`graph_factory.build_graph_array()` builds N copies of a brain blueprint with disjoint NEST `graph_id`s, so plasticity in one agent doesn't bleed into another.

---

## Retina Pipeline

Anatomically motivated retina model with:

- 6 bipolar populations (Midget + Parasol, ON + OFF, foveal + peripheral)
- 6 ganglion populations matched 1:1 to bipolars
- Mean-adapted contrast gain (Weber-Fechner scaling)
- 1:1 foveal mapping, k-NN peripheral pooling
- Configurable scales (`tiny`, `small`, `medium`, `large`) and tuning variants
- LMS color input via `RetinaInputFeeder`

Output rates match published targets: Midget 10–50 Hz, Parasol 25–40 Hz, with symmetric ON/OFF responses. See `retina_factory.py` for usage.

---

## Known Limitations

- Large networks (10k+ neurons) slow down the 3D visualization
- Tested primarily on Linux (Ubuntu 22.04 / 24.04)
- Some UI elements are rough around the edges — solo-developer project, function-before-elegance
- Refactor pending: the codebase grew organically through several feature waves; cleanup is on the roadmap once the neuroscience side stabilizes

---

## License

GPL-3.0
