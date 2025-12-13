# Neuroticks

**A Visual CAD Tool for Spiking Neural Networks**

- **Language:** Python  
- **Backend:** NEST Simulator  
- **License:** —  
- **Status:** Research Prototype / Work in Progress  

---

## What is this?

**Neuroticks** is a graphical interface and construction kit for the **NEST Simulator**.  
It bridges the gap between abstract network definitions and spatially explicit, biological architectures.

If you have ever tried to script complex 3D topologies, distance-dependent connectivity rules, or procedural population distributions purely in Python code, you know it can get messy.  
Neuroticks acts as **“CAD software for the brain”**: it allows you to visually design, manipulate, and wire neural populations in 3D space *before* you ever hit the run button.

It is designed for **connectomic hypothesis testing** and helps answer questions like:

- *How does the physical arrangement of these two populations affect their synchronization?*
- *What happens to signal propagation if I physically twist this cortical sheet?*

> **Honesty Disclaimer**  
> This is a research tool created for specific experiments. It is **not** a replacement for PyNN or a full-blown IDE.  
> It is a powerful **prototyping assistant for spatial SNNs** that lets you see what you are doing.

---

## What can you actually do with it?

Neuroticks is not just for looking at pretty neurons — it is for **building and testing functional circuits**.

### Sculpt Neural Tissue
Instead of just defining *“1000 neurons”*, you define their **geometry**:
- Grids  
- Layers  
- Spheres  

You can:
- Use **Flow Fields** (polynomial vector fields) to warp structures
- Use **Wave Function Collapse (WFC)** to generate procedural, non-uniform distributions

### “Sew” Networks
Build functional modules (e.g. a visual cortex patch and a thalamic relay) in isolation and then:
- Import them
- “Sew” them together into a larger system using the **Fusion Tool**

### Interact Live
Watch the network spike **in real time** and interact with it:

- **Simulate Lesions**  
  Select a connection bundle and lower its weight to `0.0` *during runtime*.  
  This simulates severing a pathway and lets you immediately observe how the network adapts (or fails).

- **Tweak Dynamics**  
  Adjust synaptic delays and weights on the fly without restarting the simulation.

### Inject & Record
- Click any neuron in 3D space
- Attach voltmeters
- Inject current spikes manually

---

## Features

- **3D Visualization**  
  Built on **PyVista** for high-performance rendering of neuron positions and connectivity skeletons.

- **Procedural Generation**  
  Create complex spatial distributions using **Wave Function Collapse** and **vector fields**.

- **Topological Wiring**  
  Define connections based on spatial rules:
  - Connect to nearest *k* neighbors
  - Probability decays with distance
  - Spherical masking

- **Headless Export**  
  Design in the GUI, export to a clean Python setup (`run.py` + JSON), and run heavy simulations on a cluster or server without graphical overhead.

- **Simulation History**  
  Automatically snapshots simulation runs.  
  Reload and replay past simulations to compare dynamics.

---

## Work in Progress: The Retina Module

An anatomically correct **Retina Module** is currently under development.

### Concept
- Polar projection of video footage
- Spherical arrangement of photoreceptors
- Correct foveal density and peripheral vision

### Video Input
- Pipe real-world visual data directly into the graph structure via spike generators

### Interactive
- Select a video file directly via the GUI for quick testing

### Automation
- Drop video files into the project folder
- Or pass them as parameters to the headless script

This allows automated loops over different visual stimuli without manual intervention.

**Status:**  
Mathematical foundation is solid; integration into the main graph is currently being implemented.

---

## Installation

Neuroticks relies on the **NEST Simulator**.

- Ensure you have a working NEST installation  
- Alternatively, use a Conda environment



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
