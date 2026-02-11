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

Think of it as **"CAD software for the brain"**: visually design, manipulate, and wire neural populations in 3D space before running simulations.

If you've tried scripting complex 3D topologies, distance-dependent connectivity rules, or procedural population distributions in pure Python — you know it gets messy fast.

NeuroTicks is built for **connectomic hypothesis testing**:

- *How does the physical arrangement of populations affect synchronization?*
- *What happens to signal propagation if I twist this cortical sheet?*

---

## Features

### Sculpt Neural Tissue

Define neuron **geometry**, not just numbers. Populations are placed in 3D space using configurable shape primitives (Grids, Blobs, Rings, Cones) or procedural generation via **Wave Function Collapse** and **polynomial flow fields** that warp spatial distributions into biologically plausible structures.

### Sew Networks Together

Build functional modules in isolation, then combine them. Import existing graph definitions and use the **Fusion Tool** to merge independent networks into a single simulation.

### Live Interaction

Watch the network spike in real-time:

- **Simulate Lesions** — Set connection weights to zero during runtime
- **Tweak Dynamics** — Adjust synaptic delays and weights on the fly
- **Inject & Record** — Click neurons in 3D, attach voltmeters, inject currents

### 3D Visualization

Built on **PyVista** for high-performance rendering of neuron positions and connectivity.

### Headless Export

Design in the GUI, export to Python + JSON, run on clusters without graphics.

### Simulation History

Automatic snapshots of every simulation run. Reload past states and compare dynamics through the built-in analysis dashboard with spike raster plots, firing rate histograms, and membrane potential traces.

---

## Architecture

NeuroTicks is organized around two core data structures — `Graph` and `Node` — that form a hierarchical, tree-like representation of neural populations. Graphs own and manage a collection of Nodes. Each Node encapsulates a spatial population of neurons, its NEST-level instantiation, attached recording/stimulation devices, and its synaptic connections to other Nodes (possibly across Graphs).

The spatial layout of each Node is governed by a `PolynomGenerator` which defines polynomial vector fields (flow fields) used to deform grid-based neuron placements into non-trivial 3D morphologies.

```
┌─────────────────────────────────────────────────────────────────────┐
│                             Graph                                   │
│                                                                     │
│  graph_id, graph_name                                               │
│  node_list: List[Node]          ← ordered list of all nodes        │
│  node_dict: Dict[id|name, Node] ← fast lookup by id or name       │
│  root: Node                     ← first node, tree root            │
│  polynom_generator: PolynomGenerator                                │
│                                                                     │
│  create_node(params, parent, ...) → Node                           │
│  populate(n_iterations, ...)      → builds tree of nodes           │
│  build_all()                      → calls node.build() for all     │
│  build_connections(ext_graphs)    → wires NEST synapses            │
│  remove_node(node)                                                  │
│  get_node(id) → Node                                               │
├─────────────────────────────────────────────────────────────────────┤
│  1                                                                  │
│  │                                                                  │
│  │ owns *                                                           │
│  ▼                                                                  │
│                             Node                                    │
│                                                                     │
│  id, name, graph_id                                                 │
│  parameters: dict               ← full parameter snapshot          │
│  center_of_mass: np.array(3)    ← spatial origin in 3D             │
│  positions: List[np.array(N,3)] ← per-type neuron coordinates     │
│  population: List[NodeCollection]← NEST neuron handles             │
│  types: List[int]               ← sub-population type indices      │
│  neuron_models: List[str]       ← NEST model per type              │
│  devices: List[dict]            ← spike recorders, multimeters, …  │
│  connections: List[dict]        ← synaptic projection descriptors  │
│  results: dict                  ← recorded data (history entries)  │
│  function_generator: PolynomGenerator                               │
│                                                                     │
│  ── Tree links ──                                                   │
│  parent: Node | None                                                │
│  prev: List[Node]               ← predecessors (incoming edges)    │
│  next: List[Node]               ← successors  (outgoing edges)     │
│                                                                     │
│  build()            → generates positions via WFC / shapes + flow  │
│  populate_node()    → creates NEST neurons at positions            │
│  connect(registry)  → instantiates NEST synapses from connections  │
│  instantiate_devices()                                              │
│  add_neighbor(other) / remove()                                     │
├─────────────────────────────────────────────────────────────────────┤
│  used by Node                                                       │
│  ▼                                                                  │
│                        PolynomGenerator                             │
│                                                                     │
│  n: int                         ← max polynomial power             │
│  coefficients: np.array(n+1, 3) ← coefficient matrix              │
│  terms: np.array(n+1, 3)        ← monomial basis functions        │
│  decay: float                   ← exponential power prior          │
│                                                                     │
│  generate(num_terms, ...) → callable  (x,y,z) → scalar            │
│  encode(polynom_func) → dict          serializable representation  │
│  decode(encoded) → callable           reconstruct from dict        │
│  decode_multiple(list) → [fx, fy, fz] full 3D flow field          │
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
    "use_spatial": false
  }
}
```

At simulation time, `Node.connect()` resolves source/target references through a graph registry (supporting cross-graph connections) and delegates to `nest.Connect()` with the appropriate `conn_spec` and `syn_spec`. Spatial connectivity (distance-dependent masks) is supported via NEST's `CreateMask` API when `use_spatial: true`.

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

---

## Project Structure

```
NeuroTicks/
├── Main.py                  # Entry point, main window, 3D viewport
├── neuron_toolbox.py        # Graph, Node, PolynomGenerator, NEST wrappers
├── WidgetLib.py             # UI components, parameter editors, analysis dashboard
├── CustomExtension.py       # Simulation history browser
├── ExtraTab.py              # Additional tools (fusion, import/export)
└── functional_models.json   # Preset network templates
```

---

## Installation

### Quick Install (Ubuntu 22.04 / 24.04)

```bash
wget https://raw.githubusercontent.com/Yinsalt/NeuroTicks/main/install.sh
chmod +x install.sh
./install.sh
```

This will install all system dependencies, create a Python virtual environment, compile NEST 3.8 from source, and clone NeuroTicks.

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

1. **Create Graph** — New graph structure via menu
2. **Add Nodes** — Define populations with spatial parameters (shape type, grid size, flow fields)
3. **Connect** — Use the connection tool to define synaptic projections between populations
4. **Attach Devices** — Spike recorders, multimeters, current generators
5. **Simulate** — Run and observe dynamics in real-time with live 3D visualization
6. **Analyze** — Review past runs in the history tab with raster plots, rate histograms, and voltage traces

---

## Work in Progress

### Retina Module

An anatomically correct retina module is under development:

- Polar projection of video input
- Spherical photoreceptor arrangement
- Correct foveal density distribution
- Direct video-to-spike conversion

---

## Known Limitations

- Large networks (10k+ neurons) will slow down the 3D visualization
- Tested primarily on Linux (Ubuntu 22.04 / 24.04)
- Some UI elements are rough around the edges

---

## License

GPL-3.0
