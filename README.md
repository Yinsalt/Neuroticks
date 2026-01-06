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

### Why?

If you've tried scripting complex 3D topologies, distance-dependent connectivity rules, or procedural population distributions in pure Python — you know it gets messy fast.

NeuroTicks is built for **connectomic hypothesis testing**:

- *How does the physical arrangement of populations affect synchronization?*
- *What happens to signal propagation if I twist this cortical sheet?*

> **Note:** This is a research prototype, not a replacement for PyNN. It's a prototyping assistant for spatial SNNs that lets you see what you're doing.

---

## Features

### Sculpt Neural Tissue
Define neuron **geometry**, not just numbers:
- Grids, Layers, Spheres
- **Flow Fields** (polynomial vector fields) to warp structures
- **Wave Function Collapse** for procedural, non-uniform distributions

### Sew Networks Together
Build functional modules in isolation, then combine them:
- Import existing graphs
- Use the **Fusion Tool** to merge networks

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
Automatic snapshots of simulation runs. Reload and compare past dynamics.

---

## Installation

### Quick Install (Ubuntu 22.04/24.04)

```bash
wget https://raw.githubusercontent.com/Yinsalt/NeuroTicks/main/install.sh
chmod +x install.sh
./install.sh
```

This will:
- Install all system dependencies
- Create a Python virtual environment
- Compile NEST 3.8 from source
- Clone NeuroTicks

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

If you prefer manual setup, ensure you have:

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
| Cython | <3.0 (for NEST build) |

---

## Usage

1. **Create Graph** — New graph structure via menu
2. **Add Nodes** — Define populations with spatial parameters
3. **Connect** — Use connection tool for synaptic projections
4. **Attach Devices** — Spike recorders, voltmeters
5. **Simulate** — Run and observe dynamics in real-time
6. **Analyze** — Review past runs in history tab

---

## Project Structure

```
NeuroTicks/
├── Main.py              # Entry point
├── neuron_toolbox.py    # Graph/Node classes, NEST wrappers
├── WidgetLib.py         # UI components, editors, analysis
├── CustomExtension.py   # History browser
├── ExtraTab.py          # Additional tools
└── functional_models.json
```

---

## Work in Progress

### Retina Module
An anatomically correct retina module is under development:
- Polar projection of video input
- Spherical photoreceptor arrangement
- Correct foveal density distribution
- Direct video-to-spike conversion

---

## Limitations

- Large networks (10k+ neurons) will slow down visualization
- Tested primarily on Linux
- Documentation is minimal
- Some UI elements are rough

---

## License

GPL-3.0
