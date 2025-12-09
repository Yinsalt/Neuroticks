# Neuroticks

A GUI for NEST-based neural network simulations. Built with PyQt6 and PyVista.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NEST](https://img.shields.io/badge/NEST-3.x-green)
![License](https://img.shields.io/badge/License-GPL--3.0-orange)

## What is this?

Neuroticks is a research tool for building and simulating spiking neural networks with explicit spatial structure. It's designed for connectomic hypothesis testing — when you want to explore how network topology and spatial arrangement affect dynamics.

The tool is suited for constructing larger, position-based neural architectures where biological plausibility matters: neurons have 3D coordinates, connections can depend on distance, and populations are organized in configurable spatial clusters.

Not a replacement for scripting complex experiments. More useful when you need to quickly prototype network structures, test connectivity hypotheses, or visualize what's actually happening in your simulation.

## Features

- **3D visualization** of network structure via PyVista
- **Graph-based editor** for neuron populations with spatial parameters
- **Wave Function Collapse** algorithm for procedural neuron clustering
- **Polynomial vector fields** for spatial organization of populations
- **Live simulation** with real-time spike visualization
- **Recording devices** (spike recorder, voltmeter, etc.) attachable via menu
- **Simulation history** — save, reload, and analyze past runs

## Installation

```bash
pip install pyqt6 pyvista pyvistaqt nest-simulator numpy scipy pyqtgraph matplotlib
```

Then:
```bash
python Main.py
```

## Dependencies

| Category | Package | Version |
|----------|---------|---------|
| Core | Python | 3.10+ |
| Simulation | NEST Simulator | 3.x |
| GUI | PyQt6 | 6.x |
| 3D Viz | PyVista | 0.46+ |
| Plotting | PyQtGraph | 0.13+ |
| Plotting | Matplotlib | 3.x |
| Numerics | NumPy | 2.x |
| Numerics | SciPy | 1.x |

## Structure

```
├── Main.py              # Entry point, main window
├── neuron_toolbox.py    # Graph/Node classes, NEST wrappers
├── WidgetLib.py         # UI components, editors, analysis tools
├── CustomExtension.py   # History browser tab
├── ExtraTab.py          # Additional tools
└── functional_models.json
```

## Usage

1. **Create graph** — Use menu to create a new graph structure
2. **Add nodes** — Define neuron populations with spatial and model parameters
3. **Connect** — Use connection tool to define synaptic projections
4. **Attach devices** — Add spike recorders, voltmeters via menu
5. **Simulate** — Run simulation, observe dynamics in real-time
6. **Analyze** — Use history tab to review and compare past runs

## Limitations

- Documentation is minimal
- Error handling is inconsistent
- Large networks (10k+ neurons) will slow down visualization
- Some UI elements are rough around the edges
- Tested primarily on Linux

## Status

Work in progress. Functional for research use, but expect bugs.

## License

GPL-3.0
