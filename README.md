# NeuroTicks

**A visual CAD tool for spiking neural networks**

[![Demo Video](https://img.shields.io/badge/Demo-YouTube-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=cYMDaO1GlLc)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg?style=for-the-badge&logo=python)](https://python.org)
[![NEST 3.8](https://img.shields.io/badge/NEST-3.8-orange.svg?style=for-the-badge)](https://nest-simulator.org)

[![NeuroTicks Demo](https://img.youtube.com/vi/cYMDaO1GlLc/maxresdefault.jpg)](https://www.youtube.com/watch?v=cYMDaO1GlLc)

NeuroTicks is a graphical construction kit for the [NEST Simulator](https://nest-simulator.org). You place neural populations in 3D space, wire them, simulate, and watch the result live. It is built for questions where geometry matters: how the arrangement of populations affects synchronization, or what changes when a projection gets a different synapse model.

Solo-developed. Functional first, polished later.

## Features

- **Spatial populations.** Shape primitives (Grid, Blob, Ring, Cone/Column) plus procedural placement with Wave Function Collapse and polynomial flow fields. Every population carries its own geometry, neuron model and NEST parameters, and several populations can share one node.
- **Connections and plasticity.** Per-projection rules (`all_to_all`, `one_to_one`, `pairwise_bernoulli`, `fixed_indegree`, `fixed_outdegree`, angular `similarity`), optional distance-dependent masks, and any NEST synapse model (`static`, `stdp`, `tsodyks`, dopamine-modulated STDP) with its parameters as UI fields.
- **Fusion tool.** Build modules separately, then merge them into one project. Connections across graphs work the same as within one graph.
- **Live interaction.** Click neurons in 3D to inject Poisson input or attach recorders, and change weights and delays while the simulation runs.
- **3D views.** Neuron point cloud, graph skeleton with clickable nodes and connections, and a firing-pattern view. Built on PyVista.
- **Retina.** A retina model (cones, horizontal, bipolar, amacrine and ganglion cells, seven ganglion populations) that can be driven by video and wired into an LGN graph. Five size scales and several tuning variants, including biophysical cone phototransduction and short-term plasticity.
- **History.** Every run can be saved and reopened in the analysis dashboard: rasters, rate histograms, voltage traces.
- **Headless export.** Export a project as `run.py` + `graphs.json` and run it on a machine without graphics.

## Installation

Ubuntu 22.04 / 24.04.

1. Download `neuroticks.zip` from this repository and unpack it into a folder of your choice.
2. Run the installer in that folder. It installs the system packages (asks for sudo), creates the virtual environment `venv/` in the same folder and builds NEST 3.8 into it. Building NEST takes a while.
3. Activate the environment and start the program.

```bash
unzip neuroticks.zip -d ~/NeuroTicks
cd ~/NeuroTicks
bash install.sh
source activate.sh
python3 Main.py
```

For every later start: `source ~/NeuroTicks/activate.sh` (it also changes into the folder), then `python3 Main.py`.

The example projects are in `Examples/` in this repository. Copy that folder next to `Main.py` to see them under "Population Examples".

Manual setup needs Python 3.10+, NEST 3.8, PyQt6, PyQtGraph 0.13+, PyVista 0.46+ with pyvistaqt, PyOpenGL 3.1.10, NumPy, SciPy, pandas, networkx, matplotlib and IPython. See `requirements.txt`. `opencv-python-headless` is only needed for video input in the retina tab.

## Usage

1. **Editor.** Create a graph, add nodes, pick a tool type and neuron model per population.
2. **Connection Creator.** Choose source and target population, a rule and a synapse model, then create the connection in NEST.
3. **Instrumentation.** Attach spike recorders, multimeters or generators.
4. **Play.** The global play button runs the simulation. The Interactive and Live Plots tabs show activity while it runs.
5. **History.** Save a run and inspect it later.
6. **Save / Load.** Projects are JSON files. `Examples/` contains ready-made ones, reachable through "Population Examples".

### Headless

The export writes a folder with `run.py`, `graphs.json`, `neuron_toolbox.py` and `functional_models.json`. It needs NEST and the Python packages above, but no Qt and nothing else from this repository:

```bash
python run.py
```

Results land in `Simulation_History/` and can be opened in the History tab.

### Several agents from one blueprint

```python
import nest
from neuroticks.retina.factory import build_retina_array, connect_retina_to_lgn
from neuroticks.core.graph_factory import build_graph_array

positions = [(0, 0, 0), (10, 0, 0), (20, 0, 0)]
retinas = build_retina_array(positions, scale=1, connect_recorders=False)
agents = build_graph_array(positions, perpendicular_distance=15.0, json_path='graphs.json')

connect_retina_to_lgn(retinas[0]['retina'], agents[0]['graphs'])   # needs a graph named LGN_RELAY_POPS
retinas[0]['feeder'].feed(lms, intensity)                           # lms: HxWx3 frame, intensity: HxW
nest.Simulate(50.0)
```

Each agent gets its own copy of every graph with separate IDs, so plasticity in one agent does not affect another.

## Project layout

```
Main.py                  entry point
neuron_toolbox.py        Graph, Node, population builders (also shipped with every headless export)
functional_models.json   neuron model parameters
install.sh               installer, creates venv/ and activate.sh in this folder
neuroticks/
    constants/           node tools, synapse models, colors
    utils/               NEST helpers, serialization, spatial connection parameters
    core/                connection execution, graph building, script export
    retina/              retina model, feeder, scales, cone cascade
    gui/                 widgets, 3D views, dashboards, tabs, main window
    state.py             shared runtime state
docs/                    refactoring report and function inventory
Examples/                example projects
OtherTab.py              sandbox for your own widgets, loaded as the "Other" view
```

`WidgetLib.py`, `ExtraTab.py`, `retina_factory.py` and the other old module names in the root only forward to the package, so older scripts keep working. `old/` and `old_version/` hold the previous single-file version and are not used.

## Known limitations

- Large networks (10k+ neurons) slow down the 3D views.
- Tested primarily on Linux (Ubuntu 22.04 / 24.04).
- No automated tests yet.
- Some UI parts are rough.

## License

GPL-3.0
