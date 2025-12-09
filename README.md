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
python Main.py
```

## Dependencies

  NEST Simulator       3.9.0-post0.dev0
  NumPy                2.2.6
  SciPy                1.16.3

GUI DEPENDENCIES
----------------------------------------
  PyQt6                installed (version unknown)
  PyQtGraph            0.13.7
  PyVista              0.46.4
  PyVistaQt            0.11.3
  VTK                  9.5.2
    └─ Qt Version:     6.10.0
    └─ PyQt Version:   6.10.0

VISUALIZATION
----------------------------------------
  Matplotlib           3.10.7
  PyOpenGL             3.1.10

OPTIONAL
----------------------------------------
  Pillow               12.0.0
  Pandas               2.3.3
  NetworkX             3.5

============================================================

PIP FREEZE (relevante Pakete):
----------------------------------------
  matplotlib==3.10.7
  matplotlib-inline==0.2.1
  nest-asyncio==1.6.0
  nest-desktop==4.2.0
  nest-neat==1.0
  numpy==2.2.6
  PyOpenGL==3.1.10
  PyQt6==6.10.0
  PyQt6-Qt6==6.10.0
  PyQt6_sip==13.10.2
  pyqtgraph==0.13.7
  pyvista==0.46.4
  pyvistaqt==0.11.3
  scipy==1.16.3
  trame-vtk==2.10.0
  vtk==9.5.2

## Structure

```
├── Main.py        # Main window & entry point
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
