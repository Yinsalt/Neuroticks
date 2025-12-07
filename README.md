# Neuroticks

A GUI for NEST-based neural network simulations. Built with PyQt6 and PyVista.

## What is this?

Neuroticks lets you visually assemble, simulate, and analyze spiking neural networks — without writing code for every little thing. Click together populations, connect them, hit play, watch spikes fly.

Built as a research tool for exploring network dynamics and testing hypotheses quickly.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NEST](https://img.shields.io/badge/NEST-3.x-green)
![License](https://img.shields.io/badge/License-GPL--3.0-orange)

## Features

- **3D visualization** of network structure via PyVista
- **Graph editor** for neuron populations with drag & drop
- **Live simulation** with real-time spike visualization
- **Recording devices** (spike recorder, voltmeter, etc.) attachable directly in the GUI
- **Simulation history** — reload and analyze past runs
- **Wave Function Collapse** for procedural neuron clustering

## Installation

```bash
pip install pyqt6 pyvista pyvistaqt nest-simulator numpy pyqtgraph
```

Then:

```bash
python CleanAlpha.py
```

## Dependencies

- Python ≥ 3.10
- NEST Simulator 3.x
- PyQt6
- PyVista + pyvistaqt
- NumPy, Matplotlib, pyqtgraph

## Structure

```
├── CleanAlpha.py        # Main window & entry point
├── neuron_toolbox.py    # Graph/Node classes, NEST wrappers
├── WidgetLib.py         # UI components, analysis dashboard
├── CustomExtension.py   # History browser tab
├── ExtraTab.py          # Additional tools
└── functional_models.json
```

## Usage

1. **Create graph** → Right-click in editor
2. **Add nodes** → Neuron populations with parameters
3. **Connect** → Use connection tool for synapses
4. **Attach devices** → Spike recorder, voltmeter
5. **Simulate** → Hit play, collect data
6. **Analyze** → History tab for past runs

## Status

Work in progress. Runs, but expect rough edges.

## License

GPL-3.0
