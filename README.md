# Neuroticks

Neural network simulation GUI built on [NEST Simulator](https://www.nest-simulator.org/).

![Python](https://img.shields.io/badge/Python-3.12-blue)
![NEST](https://img.shields.io/badge/NEST-3.9-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- Visual neural network editor with 3D spatial positioning
- Real-time simulation with live spike visualization
- Multiple neuron models (LIF, AdEx, Hodgkin-Huxley, ...)
- Configurable synaptic connections and stimulation
- History recording and analysis dashboard

## Quick Start (Docker)

```bash
# Download and run
curl -O https://raw.githubusercontent.com/YOURUSER/neuroticks/main/run_neuroticks.sh
chmod +x run_neuroticks.sh
./run_neuroticks.sh
```

First run pulls the image (~2-3 GB). Data is saved to `~/neuroticks_data/`.

## Manual Installation

Requires: Python 3.12, NEST Simulator 3.9+

```bash
# Install dependencies
pip install PyQt6 pyqtgraph pyvista pyvistaqt numpy scipy matplotlib

# Run
python CleanAlpha.py
```

## Data Storage

```
~/neuroticks_data/
├── Simulation_History/   # Recorded simulation data
├── projects/             # Saved network graphs
└── exports/              # Screenshots, exports
```

## License

MIT
