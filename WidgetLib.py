import sys
from PyQt6.QtWidgets import QApplication,QDialog,QAbstractSpinBox,QDialogButtonBox,QListWidget,QSlider, QMainWindow,QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtWidgets import QSizePolicy,QListWidgetItem, QFrame,QPushButton,QLabel,QGroupBox, QStackedWidget, QToolBar, QMenu, QGridLayout, QStackedLayout
from PyQt6.QtGui import QColor, QPalette, QAction,QIcon,QBrush
from PyQt6.QtCore import QSize, Qt, pyqtSignal,QTimer
import code
import time
import pyvista as pv
import scipy.ndimage as ndimage
from io import StringIO
from PyQt6 import QtGui  # WICHTIG für QtGui.QTextCursor
from PyQt6.QtCore import QEvent # WICHTIG für QEvent.Type.KeyPress
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QVector3D
from pyvistaqt import QtInteractor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pyqtgraph.dockarea as dock
import numpy as np
from pathlib import Path
from datetime import datetime
import pyqtgraph.opengl as gl
import time
import vtk
import pyqtgraph.opengl as gl
from typing import Dict, Any, Tuple, Optional, List
import matplotlib.colors as mcolors
from datetime import datetime
from pathlib import Path
from neuron_toolbox import *
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSlider, QCheckBox, QFrame)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QVector3D
from neuron_toolbox import successful_neuron_models
from PyQt6.QtWidgets import (
    QLineEdit,      
    QTextEdit,      
    QSpinBox,       
    QDoubleSpinBox, 
    QComboBox,      
    QCheckBox,      
    QRadioButton,   
    QSlider,       
    QDial,         
    QPushButton,   
    QFileDialog,   
    QColorDialog,   
    QTreeWidget,    
    QTreeWidgetItem 
)
from PyQt6.QtWidgets import QScrollArea, QInputDialog
current_graph_metadata = []
graph_parameters = {}     #graph_id 
next_graph_id = 0
POLYNOM_NOISE_LEVEL = 0.05
pg.setConfigOption('background', '#1e1e1e')
pg.setConfigOption('foreground', 'd')
pg.setConfigOption('antialias', True)

_nest_simulation_has_run = False


NODE_TOOLS = {
    "custom": {
        "label": "WFC & Flow Field (Custom)",
        "params": ["grid_size", "sparsity_factor", "sparse_holes", "num_steps", "dt"]
    },
    "CCW": {
        "label": "CCW Ring (Attractor)",
        "params": ["n_neurons", "radius", "k", "bidirectional"] 
    },
    "Blob": {
        "label": "Random Blob",
        "params": ["n_neurons", "radius"]
    },
    "Cone": {
        "label": "Cone / Column",
        "params": ["n_neurons", "radius_bottom", "radius_top", "height"]
    },
    "Grid": {
        "label": "2D Grid",
        "params": ["grid_side_length"]
    }
}

__all__ = [
    'SYNAPSE_MODELS',
    "DeviceConfigPage",
    "AnalysisDashboard",
    "LiveDataDashboard",
    'create_nest_mask',
    "SimulationControlWidget",
    "SimulationDashboardWidget",
    'create_distance_dependent_weight',
    'SynapseParamWidget',
    "SimulationViewWidget",
    'SynapseCreationWidget',
    'ConnectionExecutor',
    'ConnectionQueueWidget',
    'BlinkingNetworkWidget', 
    'FlowFieldWidget',       
    'GraphCreatorWidget',
    "StructuresWidget",
    'EditGraphWidget',
    'GraphOverviewWidget',
    'ConnectionTool',
    'ToolsWidget',
    'SimulationControlWidget',
    'graph_parameters',
    'next_graph_id',
    '_clean_params',
    '_serialize_connections',
    'NumpyEncoder',
    'create_nest_connections_from_stored',
    'safe_nest_reset_and_repopulate',
    'repopulate_all_graphs',
    'get_all_connections_summary',
    'export_connections_to_dict',
    'import_connections_from_dict',
    '_nest_simulation_has_run'
]

SYNAPSE_MODELS = {
    "static_synapse": {
        "category": "basic",
        "description": "Standard-Synapse mit konstantem Gewicht",
        "params": {}
    },
    
    "bernoulli_synapse": {
        "category": "basic",
        "description": "Stochastische Synapse mit Transmissionswahrscheinlichkeit",
        "params": {
            "p_transmit": {
                "default": 1.0, "type": "float", "min": 0.0, "max": 1.0,
                "unit": "", "description": "Transmission probability"
            }
        }
    },

    "cont_delay_synapse": {
        "category": "basic",
        "description": "Synapse mit kontinuierlichem Delay",
        "params": {}
    },

    "gap_junction": {
        "category": "electrical",
        "description": "Elektrische Synapse (Kein Delay!)",
        "no_delay": True,
        "params": {}
    },

    "stdp_synapse": {
        "category": "plasticity",
        "description": "Standard STDP nach Song et al.",
        "params": {
            "tau_plus": {"default": 20.0, "type": "float", "min": 0.1, "max": 1000.0, "unit": "ms",
                         "description": "Time constant of potentiation window"},
            "lambda": {"default": 0.01, "type": "float", "min": 0.0, "max": 1.0, "unit": "",
                       "description": "Learning rate"},
            "alpha": {"default": 1.0, "type": "float", "min": 0.0, "max": 10.0, "unit": "",
                      "description": "Asymmetry parameter (depression scale)"},
            "mu_plus": {"default": 1.0, "type": "float", "min": 0.0, "max": 10.0, "unit": "",
                        "description": "Weight dependence exponent (potentiation)"},
            "mu_minus": {"default": 1.0, "type": "float", "min": 0.0, "max": 10.0, "unit": "",
                         "description": "Weight dependence exponent (depression)"},
            "Wmax": {"default": 100.0, "type": "float", "min": 0.0, "max": 10000.0, "unit": "",
                     "description": "Maximum weight"}
        }
    },

    "stdp_synapse_hom": {
        "category": "plasticity",
        "description": "Homeostatic STDP",
        "params": {
            "tau_plus": {"default": 20.0, "type": "float", "min": 0.1, "max": 1000.0, "unit": "ms"},
            "lambda": {"default": 0.01, "type": "float", "min": 0.0, "max": 1.0, "unit": ""},
            "alpha": {"default": 1.0, "type": "float", "min": 0.0, "max": 10.0, "unit": ""},
            "Wmax": {"default": 100.0, "type": "float", "min": 0.0, "max": 10000.0, "unit": ""}
        }
    },

    "stdp_triplet_synapse": {
        "category": "plasticity",
        "description": "Triplet STDP (Pfister & Gerstner 2006)",
        "params": {
            "tau_plus": {"default": 16.8, "type": "float", "min": 0.1, "unit": "ms",
                         "description": "Time constant of 1st potentiation window"},
            "tau_minus": {"default": 33.7, "type": "float", "min": 0.1, "unit": "ms",
                          "description": "Time constant of 1st depression window"},
            "tau_x": {"default": 101.0, "type": "float", "min": 0.1, "unit": "ms",
                      "description": "Time constant of presynaptic trace"},
            "tau_y": {"default": 125.0, "type": "float", "min": 0.1, "unit": "ms",
                      "description": "Time constant of postsynaptic trace"},
            "Wmax": {"default": 100.0, "type": "float", "min": 0.0, "unit": ""}
        }
    },

    "stdp_dopamine_synapse": {
        "category": "plasticity",
        "description": "Neuromoduliertes STDP (Reward-Learning)",
        "params": {
            "b": {"default": 0.0, "type": "float", "min": -10.0, "max": 10.0, "unit": "",
                  "description": "Dopamine baseline concentration"},
            "tau_plus": {"default": 20.0, "type": "float", "min": 0.1, "unit": "ms"},
            "tau_n": {"default": 200.0, "type": "float", "min": 0.1, "unit": "ms",
                      "description": "Time constant of eligibility trace"},
            "Wmin": {"default": 0.0, "type": "float", "min": -10000.0, "unit": ""},
            "Wmax": {"default": 100.0, "type": "float", "min": 0.0, "unit": ""}
        }
    },

    "vogels_sprekeler_synapse": {
        "category": "plasticity",
        "description": "Inhibitory STDP (Vogels et al. 2011)",
        "params": {
            "tau": {"default": 20.0, "type": "float", "min": 0.1, "unit": "ms",
                    "description": "STDP time constant"},
            "eta": {"default": 0.001, "type": "float", "min": 0.0, "unit": "",
                    "description": "Learning rate"},
            "alpha": {"default": 0.12, "type": "float", "min": 0.0, "unit": "",
                      "description": "Target firing rate parameter"},
            "Wmax": {"default": 100.0, "type": "float", "min": 0.0, "unit": ""}
        }
    },

    "clopath_synapse": {
        "category": "plasticity",
        "description": "Voltage-based STDP (Clopath et al. 2010)",
        "params": {
            "tau_x": {"default": 10.0, "type": "float", "min": 0.1, "unit": "ms"},
            "A_LTP": {"default": 0.00014, "type": "float", "min": 0.0, "unit": "",
                      "description": "LTP amplitude"},
            "A_LTD": {"default": 0.00008, "type": "float", "min": 0.0, "unit": "",
                      "description": "LTD amplitude"},
            "Wmax": {"default": 100.0, "type": "float", "min": 0.0, "unit": ""}
        }
    },

    "urbanczik_synapse": {
        "category": "plasticity",
        "description": "Gradient-based learning (Dendritic)",
        "params": {
            "tau_Delta": {"default": 100.0, "type": "float", "min": 0.1, "unit": "ms",
                          "description": "Time constant of eligibility trace"},
            "eta": {"default": 0.01, "type": "float", "min": 0.0, "unit": "",
                    "description": "Learning rate"},
            "Wmax": {"default": 100.0, "type": "float", "min": 0.0, "unit": ""}
        }
    },

    "tsodyks_synapse": {
        "category": "stp",
        "description": "Short-Term Plasticity (Tsodyks & Markram)",
        "params": {
            "U": {"default": 0.5, "type": "float", "min": 0.0, "max": 1.0, "unit": "",
                  "description": "Utilization of synaptic efficacy"},
            "tau_rec": {"default": 800.0, "type": "float", "min": 0.1, "unit": "ms",
                        "description": "Recovery time constant"},
            "tau_fac": {"default": 0.0, "type": "float", "min": 0.0, "unit": "ms",
                        "description": "Facilitation time constant"},
            "x": {"default": 1.0, "type": "float", "min": 0.0, "max": 1.0, "unit": "",
                  "description": "Initial fraction of resources"}
        }
    },

    "tsodyks2_synapse": {
        "category": "stp",
        "description": "Short-Term Plasticity (Alternative)",
        "params": {
            "U": {"default": 0.5, "type": "float", "min": 0.0, "max": 1.0, "unit": ""},
            "tau_rec": {"default": 800.0, "type": "float", "min": 0.1, "unit": "ms"},
            "tau_fac": {"default": 0.0, "type": "float", "min": 0.0, "unit": "ms"}
        }
    },

    "quantal_stp_synapse": {
        "category": "stp",
        "description": "Quantal STP mit Release Sites",
        "params": {
            "U": {"default": 0.5, "type": "float", "min": 0.0, "max": 1.0, "unit": ""},
            "tau_rec": {"default": 800.0, "type": "float", "min": 0.1, "unit": "ms"},
            "tau_fac": {"default": 0.0, "type": "float", "min": 0.0, "unit": "ms"},
            "n": {"default": 1, "type": "int", "min": 1, "max": 100, "unit": "",
                  "description": "Number of release sites"},
            "a": {"default": 1, "type": "int", "min": 0, "max": 100, "unit": "",
                  "description": "Initial available release sites"}
        }
    },

    "ht_synapse": {
        "category": "special",
        "description": "Hill-Tononi Synapse",
        "params": {}
    },

    "diffusion_connection": {
        "category": "special",
        "description": "Diffusion-Verbindung (rate neurons)",
        "params": {}
    },

    "rate_connection_instantaneous": {
        "category": "special",
        "description": "Instantane Rate-Verbindung",
        "no_delay": True,
        "params": {}
    },

    "rate_connection_delayed": {
        "category": "special",
        "description": "Verzögerte Rate-Verbindung",
        "params": {}
    }
}
"""
# Synaptic Elements 
nest.SetDefaults('iaf_psc_alpha', {
    'synaptic_elements': {
        'Den_ex': {'growth_curve': 'gaussian', 'z': 1.0, ...},
        'Axon_ex': {'growth_curve': 'gaussian', 'z': 1.0, ...}
    }
})

# Structural Plasticity 
nest.structural_plasticity_synapses = {
    'synapse_ex': {
        'synapse_model': 'static_synapse',
        'post_synaptic_element': 'Den_ex',
        'pre_synaptic_element': 'Axon_ex'
    }
}
nest.structural_plasticity_update_interval = 100.0  # ms
"""




def generate_biased_polynomial(axis_idx, max_degree=2, num_noise_terms=2):

    global POLYNOM_NOISE_LEVEL
    noise = POLYNOM_NOISE_LEVEL
    
    main_coeff = 1.0 + random.uniform(-noise, noise)
    
    indices = [[axis_idx, 1]]
    coefficients = [main_coeff]
    
    for _ in range(num_noise_terms):
        v_idx = random.randint(0, 2) 
        power = random.randint(0, max_degree)
        
        if v_idx == axis_idx and power == 1:
            power = 0 
            
        coeff = random.uniform(-1.0, 1.0) * noise
        
        indices.append([v_idx, power])
        coefficients.append(coeff)
        
    return {
        'indices': indices,
        'coefficients': coefficients,
        'n': max_degree,
        'decay': 0.5
    }






class PositionDialog(QDialog):
    def __init__(self, current_pos=[0.0, 0.0, 0.0], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Target Position for Twin")
        self.selected_pos = list(current_pos)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        info = QLabel("Set the Center of Mass (m) for the Twin Node.\n"
                      "The structure will be regenerated locally at this new position.")
        layout.addWidget(info)

        form = QFormLayout()
        self.spins = []
        labels = ['X:', 'Y:', 'Z:']
        
        for i, lbl in enumerate(labels):
            spin = QDoubleSpinBox()
            spin.setRange(-10000, 10000)
            spin.setDecimals(3)
            spin.setValue(self.selected_pos[i])
            form.addRow(lbl, spin)
            self.spins.append(spin)
            
        layout.addLayout(form)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_position(self):
        return [s.value() for s in self.spins]
    

def create_nest_mask(mask_type: str, params: Dict[str, Any]) -> Optional[Any]:

    try:
        if mask_type == 'spherical' or mask_type == 'sphere':
            radius = float(params.get('radius', params.get('r', 1.0)))
            return nest.CreateMask('spherical', {'radius': radius}) 
            
        elif mask_type == 'rectangular' or mask_type == 'box':
            size = float(params.get('size', params.get('r', 1.0)))
            return nest.CreateMask('rectangular', {
                'lower_left': [-size, -size, -size],
                'upper_right': [size, size, size]
            })
            
        elif mask_type == 'elliptical':
            major = float(params.get('major_axis', 1.0))
            minor = float(params.get('minor_axis', 0.5))
            return nest.CreateMask('elliptical', {
                'major_axis': major,
                'minor_axis': minor
            })
            
        elif mask_type == 'doughnut':
            inner = float(params.get('inner_radius', params.get('inner', 0.2)))
            outer = float(params.get('outer_radius', params.get('outer', 1.0)))
            return nest.CreateMask('doughnut', {
                'inner_radius': inner,
                'outer_radius': outer
            })
            
        else:
            print(f"⚠ Unknown mask type: {mask_type}")
            return None
            
    except Exception as e:
        print(f"⚠ Mask creation failed: {e}")
        return None


def create_distance_dependent_weight(base_weight: float, 
                                     factor: float = 1.0, 
                                     offset: float = 0.0,
                                     mode: str = 'linear') -> Any:
    
    try:
        dist = nest.spatial.distance
        
        if mode == 'linear':
            return base_weight + dist * factor + offset
            
        elif mode == 'exponential':
            import nest.math as nm
            return base_weight * nm.exp(-dist * factor)
            
        elif mode == 'gaussian':
            import nest.math as nm
            sigma = factor if factor > 0 else 1.0
            return base_weight * nm.exp(-(dist / sigma) ** 2)
            
        else:
            return base_weight
            
    except Exception as e:
        print(f"Distance weight creation failed: {e}, using base weight")
        return base_weight
    


class SynapseParamWidget(QWidget):
    
    valueChanged = pyqtSignal()
    
    def __init__(self, name: str, info: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.name = name
        self.info = info
        self._init_ui()
        
    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        
        label = QLabel(f"{self.name}:")
        label.setMinimumWidth(100)
        desc = self.info.get('description', self.name)
        unit = self.info.get('unit', '')
        if unit:
            label.setToolTip(f"{desc} [{unit}]")
        else:
            label.setToolTip(desc)
        layout.addWidget(label)
        
        ptype = self.info.get('type', 'float')
        default = self.info.get('default', 0.0)
        
        if ptype == 'float':
            self.input = QDoubleSpinBox()
            self.input.setDecimals(6)
            self.input.setRange(
                self.info.get('min', -1e10),
                self.info.get('max', 1e10)
            )
            self.input.setValue(float(default))
            self.input.valueChanged.connect(self.valueChanged.emit)
            
        elif ptype == 'int' or ptype == 'integer':
            self.input = QSpinBox()
            self.input.setRange(
                int(self.info.get('min', 0)),
                int(self.info.get('max', 10000))
            )
            self.input.setValue(int(default))
            self.input.valueChanged.connect(self.valueChanged.emit)
            
        elif ptype == 'bool':
            self.input = QCheckBox()
            self.input.setChecked(bool(default))
            self.input.stateChanged.connect(self.valueChanged.emit)
            
        else:
            self.input = QLineEdit(str(default))
            self.input.textChanged.connect(self.valueChanged.emit)
        
        layout.addWidget(self.input, 1)
        
        if unit:
            unit_label = QLabel(unit)
            unit_label.setStyleSheet("color: #888; font-size: 10px;")
            layout.addWidget(unit_label)
    
    def get_value(self):
        if isinstance(self.input, (QDoubleSpinBox, QSpinBox)):
            return self.input.value()
        elif isinstance(self.input, QCheckBox):
            return self.input.isChecked()
        else:
            return self.input.text()
    
    def set_value(self, value):
        if isinstance(self.input, (QDoubleSpinBox, QSpinBox)):
            self.input.setValue(value)
        elif isinstance(self.input, QCheckBox):
            self.input.setChecked(bool(value))
        else:
            self.input.setText(str(value))

class SynapseCreationWidget(QWidget):

    
    synapseCreated = pyqtSignal(str, dict) 
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.param_widgets: Dict[str, SynapseParamWidget] = {}
        self.custom_synapses: Dict[str, Dict] = {}  
        self._init_ui()
        
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        header = QLabel("Synapse Configuration")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3;")
        main_layout.addWidget(header)
        
        sel_group = QGroupBox("Model Selection")
        sel_layout = QFormLayout(sel_group)
        
        self.category_combo = QComboBox()
        self.category_combo.addItems(["All", "basic", "plasticity", "stp", "electrical", "special"])
        self.category_combo.currentTextChanged.connect(self._filter_models)
        sel_layout.addRow("Category:", self.category_combo)
        
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        sel_layout.addRow("Synapse Model:", self.model_combo)
        
        self.desc_label = QLabel("")
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: #888; font-style: italic; padding: 5px;")
        sel_layout.addRow(self.desc_label)
        
        main_layout.addWidget(sel_group)
        
        base_group = QGroupBox("Base Parameters")
        base_layout = QFormLayout(base_group)
        
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(-100000, 100000)
        self.weight_spin.setValue(1.0)
        self.weight_spin.setDecimals(4)
        base_layout.addRow("Weight:", self.weight_spin)
        
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 10000)
        self.delay_spin.setValue(1.0)
        self.delay_spin.setDecimals(2)
        self.delay_spin.setSuffix(" ms")
        base_layout.addRow("Delay:", self.delay_spin)
        
        self.delay_warning = QLabel("")
        self.delay_warning.setStyleSheet("color: #FF9800; font-size: 10px;")
        base_layout.addRow(self.delay_warning)
        
        main_layout.addWidget(base_group)
        
        params_group = QGroupBox("Model-Specific Parameters")
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.params_scroll.setMaximumHeight(200)
        
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(5, 5, 5, 5)
        self.params_scroll.setWidget(self.params_container)
        
        params_group_layout = QVBoxLayout(params_group)
        params_group_layout.addWidget(self.params_scroll)
        main_layout.addWidget(params_group)
        
        custom_group = QGroupBox("Custom Model (Optional)")
        custom_layout = QFormLayout(custom_group)
        
        self.custom_name_input = QLineEdit()
        self.custom_name_input.setPlaceholderText("Leave empty to use base model")
        custom_layout.addRow("Custom Name:", self.custom_name_input)
        
        save_btn = QPushButton("Save as Preset")
        save_btn.clicked.connect(self._save_preset)
        custom_layout.addRow(save_btn)
        
        main_layout.addWidget(custom_group)
        
        action_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply Synapse")
        self.apply_btn.setStyleSheet("background-color: #4CAF50; font-weight: bold;")
        self.apply_btn.clicked.connect(self._emit_synapse)
        action_layout.addWidget(self.apply_btn)
        
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_to_defaults)
        action_layout.addWidget(reset_btn)
        
        main_layout.addLayout(action_layout)
        main_layout.addStretch()
        
        self._filter_models("All")
        



    def _filter_models(self, category: str):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        
        for name, info in SYNAPSE_MODELS.items():
            if category == "All" or info.get('category', 'basic') == category:
                self.model_combo.addItem(name)
        
        self.model_combo.blockSignals(False)
        if self.model_combo.count() > 0:
            self._on_model_changed(self.model_combo.currentText())
    



    def _on_model_changed(self, model_name: str):
        if not model_name or model_name not in SYNAPSE_MODELS:
            return
            
        info = SYNAPSE_MODELS[model_name]
        
        self.desc_label.setText(info.get('description', ''))
        
        if info.get('no_delay', False):
            self.delay_spin.setEnabled(False)
            self.delay_warning.setText("This synapse type does not use delay!")
        else:
            self.delay_spin.setEnabled(True)
            self.delay_warning.setText("")
        
        for widget in self.param_widgets.values():
            widget.deleteLater()
        self.param_widgets.clear()
        
        params = info.get('params', {})
        for param_name, param_info in params.items():
            widget = SynapseParamWidget(param_name, param_info)
            self.params_layout.addWidget(widget)
            self.param_widgets[param_name] = widget
        
        self.params_layout.addStretch()
    
    def _reset_to_defaults(self):
        model = self.model_combo.currentText()
        if model in SYNAPSE_MODELS:
            info = SYNAPSE_MODELS[model]
            for name, widget in self.param_widgets.items():
                if name in info.get('params', {}):
                    default = info['params'][name].get('default', 0)
                    widget.set_value(default)
        
        self.weight_spin.setValue(1.0)
        self.delay_spin.setValue(1.0)
        self.custom_name_input.clear()
    
    def _save_preset(self):
        name = self.custom_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Name Required", 
                              "Please enter a custom name to save as preset.")
            return
        
        self.custom_synapses[name] = self.get_synapse_config()
        QMessageBox.information(self, "Saved", 
                               f"Synapse preset '{name}' saved!")
    
    def get_synapse_config(self) -> Dict[str, Any]:
        model = self.model_combo.currentText()
        info = SYNAPSE_MODELS.get(model, {})
        
        config = {
            'base_model': model,
            'weight': self.weight_spin.value(),
            'delay': self.delay_spin.value() if not info.get('no_delay') else None,
            'custom_name': self.custom_name_input.text().strip() or None,
            'no_delay': info.get('no_delay', False),
            'params': {}
        }
        
        for name, widget in self.param_widgets.items():
            config['params'][name] = widget.get_value()
        
        return config
    
    def set_synapse_config(self, config: Dict[str, Any]):
        model = config.get('base_model', 'static_synapse')
        
        if model in SYNAPSE_MODELS:
            cat = SYNAPSE_MODELS[model].get('category', 'All')
            self.category_combo.setCurrentText(cat)
        
        idx = self.model_combo.findText(model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        
        self.weight_spin.setValue(config.get('weight', 1.0))
        if config.get('delay') is not None:
            self.delay_spin.setValue(config['delay'])
        
        self.custom_name_input.setText(config.get('custom_name', '') or '')
        
        for name, value in config.get('params', {}).items():
            if name in self.param_widgets:
                self.param_widgets[name].set_value(value)
    
    def _emit_synapse(self):
        config = self.get_synapse_config()
        name = config['custom_name'] or config['base_model']
        self.synapseCreated.emit(name, config)




def create_nest_mask_safe(mask_type, params):
    try:
        if mask_type in ('sphere', 'spherical'):
            return nest.CreateMask('spherical', {'r': params.get('radius', 1.0)})
        elif mask_type in ('box', 'rectangular'):
            s = params.get('size', 1.0)
            return nest.CreateMask('rectangular', {
                'lower_left': [-s, -s, -s], 'upper_right': [s, s, s]
            })
        elif mask_type == 'doughnut':
            return nest.CreateMask('doughnut', {
                'inner_radius': params.get('inner', 0.2),
                'outer_radius': params.get('outer', 1.0)
            })
    except Exception as e:
        print(f"⚠ Mask: {e}")
    return None


class DoubleInputField(QWidget):
    def __init__(self, param_name, default_value=0.0, min_val=0.0, max_val=100.0, decimals=2):
        super().__init__()
        self.param_name = param_name
        layout = QHBoxLayout()
        self.label = QLabel(f"{param_name}:")
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setDecimals(decimals)
        self.spinbox.setValue(default_value)
        layout.addWidget(self.label)
        layout.addWidget(self.spinbox)
        self.setLayout(layout)
    
    def get_value(self):
        return self.spinbox.value()

class IntegerInputField(QWidget):
    def __init__(self, param_name, default_value=0, min_val=0, max_val=100):
        super().__init__()
        self.param_name = param_name
        layout = QHBoxLayout()
        self.label = QLabel(f"{param_name}:")
        self.spinbox = QSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(default_value)
        layout.addWidget(self.label)
        layout.addWidget(self.spinbox)
        self.setLayout(layout)
    
    def get_value(self):
        return self.spinbox.value()

class DropdownField(QWidget):
    def __init__(self, param_name, options, default_index=0):
        super().__init__()
        self.param_name = param_name
        layout = QHBoxLayout()
        self.label = QLabel(f"{param_name}:")
        self.combobox = QComboBox()
        self.combobox.addItems(options)
        self.combobox.setCurrentIndex(default_index)
        layout.addWidget(self.label)
        layout.addWidget(self.combobox)
        self.setLayout(layout)
    
    def get_value(self):
        return self.combobox.currentText()

class CheckboxField(QWidget):
    def __init__(self, param_name, default_checked=False):
        super().__init__()
        self.param_name = param_name
        layout = QHBoxLayout()
        self.checkbox = QCheckBox(param_name)
        self.checkbox.setChecked(default_checked)
        layout.addWidget(self.checkbox)
        self.setLayout(layout)
    
    def get_value(self):
        return self.checkbox.isChecked()



class NeuronParametersWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.all_models = {}
        self.current_model = None
        self.parameter_widgets = {}
        self.load_json()
        self.setup_ui()

    def load_json(self):
        try:
            with open("functional_models.json") as f:
                self.all_models = json.load(f)
            
            global successful_neuron_models
            
            for model in self.all_models:
                if model not in neuron_colors:
                    neuron_colors[model] = "#FFFFFF"
            
            raw = list(self.all_models.keys())
            successful_neuron_models = sorted(raw, key=lambda x: x.lower())
            
            priority = ['iaf_psc_alpha']
            for p in reversed(priority):
                if p in successful_neuron_models:
                    successful_neuron_models.remove(p)
                    successful_neuron_models.insert(0, p)
            # -----------------------------------

        except FileNotFoundError:
            print("Warning: functional_models.json not found")
            self.all_models = {}

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Model Dropdown
        model_layout = QHBoxLayout()
        model_label = QLabel("Neuron Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.all_models.keys()))
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # Scrollable Parameters
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_container.setLayout(self.params_layout)
        self.params_scroll.setWidget(self.params_container)
        layout.addWidget(self.params_scroll)

        # Save Button
        self.save_button = QPushButton("Save Parameters")
        self.save_button.clicked.connect(self.save_parameters)
        layout.addWidget(self.save_button)

        # Initial load
        if self.model_combo.count() > 0:
            self.on_model_changed(self.model_combo.currentText())

    def on_model_changed(self, model_name):
        self.current_model = model_name
        self.clear_params()
        if model_name in self.all_models:
            params = self.all_models[model_name]
            for param_name, param_info in params.items():
                widget = self.create_param_widget(param_name, param_info)
                if widget:
                    self.params_layout.addWidget(widget)
                    self.parameter_widgets[param_name] = widget

    def clear_params(self):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.parameter_widgets.clear()

    def create_param_widget(self, param_name, param_info):
        param_type = param_info.get('type', 'float')
        default = param_info.get('default', 0.0)
        min_val = param_info.get('min')
        min_val = -1e9 if min_val is None else min_val
        max_val = param_info.get('max')
        max_val = 1e9 if max_val is None else max_val
        
        widget = None
        layout = QHBoxLayout()
        label = QLabel(f"{param_name}:")
        layout.addWidget(label)

        if param_type == 'float':
            spinbox = QDoubleSpinBox()
            spinbox.setRange(min_val, max_val)
            
            if abs(default) < 0.001 and default != 0.0:
                spinbox.setDecimals(10)  
                spinbox.setSingleStep(default / 10)
            elif abs(default) < 1.0:
                spinbox.setDecimals(6)
            else:
                spinbox.setDecimals(2)
            
            spinbox.setValue(default)
            layout.addWidget(spinbox)
            widget = QWidget()
            widget.setLayout(layout)
            widget.get_value = spinbox.value
            
        elif param_type == 'integer':
            spinbox = QSpinBox()
            spinbox.setRange(int(min_val), int(max_val))
            spinbox.setValue(int(default))
            layout.addWidget(spinbox)
            widget = QWidget()
            widget.setLayout(layout)
            widget.get_value = spinbox.value
            
        elif param_type == 'boolean':
            checkbox = QCheckBox()
            checkbox.setChecked(default)
            layout.addWidget(checkbox)
            widget = QWidget()
            widget.setLayout(layout)
            widget.get_value = checkbox.isChecked
            
        return widget

    def save_parameters(self):
        if not self.current_model:
            print("No model selected")
            return
        params = {}
        for param_name, widget in self.parameter_widgets.items():
            params[param_name] = widget.get_value()
        print({"model": self.current_model, "params": params})


#################################################################################


import sys
import ast
import json
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, 
    QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox, 
    QLineEdit, QTextEdit, QFormLayout, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt
import json
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, 
    QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit
)
from PyQt6.QtCore import pyqtSignal
class NodeParametersWidget(QWidget):

    paramsChanged = pyqtSignal(dict)

    def __init__(self, node_data=None):
        super().__init__()
        self.node_data = node_data if node_data else {}
        self.widgets = {}         
        self.tool_panels = {}    
        self.auto_save = True
        self.num_populations = 0
        self.init_ui()
        if node_data:
            self.load_data(node_data)

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        self.add_section("🔧 Node Type")
        self.tool_combo = QComboBox()
        for key, info in NODE_TOOLS.items():
            self.tool_combo.addItem(info["label"], key)
        self.tool_combo.currentIndexChanged.connect(self.on_tool_changed)
        self.content_layout.addWidget(self.tool_combo)
        self.widgets['tool_type'] = {'type': 'combo', 'widget': self.tool_combo}
        

        self.add_section("Basic Info")
        self.add_text_field("name", "Name")
        self.add_int_field("id", "ID", min_val=0)
        

        self.add_section("Position & Transform")
        self.add_vector3_field("center_of_mass", "Center (X,Y,Z)")
        self.add_vector3_field("displacement", "Displacement")
        self.add_float_field("displacement_factor", "Disp. Factor", 0.0, 100.0)
        self.add_float_field("rot_theta", "Rotation θ (X-Axis)", -360, 360)
        self.add_float_field("rot_phi", "Rotation φ (Y-Axis)", -360, 360)
        self.add_float_field("stretch_x", "Scale X", 0.1, 100.0)
        self.add_float_field("stretch_y", "Scale Y", 0.1, 100.0)
        self.add_float_field("stretch_z", "Scale Z", 0.1, 100.0)
        

        self.add_section("Tool Parameters")
        self.create_tool_stack()
        

        self.add_section("Probability Vector (Population Weights)")
        self.probability_container = QWidget()
        self.probability_layout = QVBoxLayout(self.probability_container)
        self.content_layout.addWidget(self.probability_container)
        

        self.on_tool_changed() 
        self.content_layout.addStretch()


    def _add_model_selector(self, layout, key="tool_neuron_model", label="Neuron Model"):
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        
        combo = QComboBox()
        

        combo.addItems(successful_neuron_models)
        
        combo.currentTextChanged.connect(self.on_change)
        
        row.addWidget(lbl)
        row.addWidget(combo)
        layout.addLayout(row)
        
        self.widgets[key] = {'type': 'combo_text', 'widget': combo, 'row_layout': row}

    def create_tool_stack(self):

        self.tool_stack = QStackedWidget()
        self.content_layout.addWidget(self.tool_stack)

        panel_custom = self._create_panel_custom()
        self.tool_stack.addWidget(panel_custom)
        self.tool_panels['custom'] = 0
        

        panel_ccw = self._create_panel_ccw()
        self.tool_stack.addWidget(panel_ccw)
        self.tool_panels['CCW'] = 1
        

        panel_blob = self._create_panel_blob()
        self.tool_stack.addWidget(panel_blob)
        self.tool_panels['Blob'] = 2
        

        panel_cone = self._create_panel_cone()
        self.tool_stack.addWidget(panel_cone)
        self.tool_panels['Cone'] = 3
        

        panel_grid = self._create_panel_grid()
        self.tool_stack.addWidget(panel_grid)
        self.tool_panels['Grid'] = 4



    def _create_panel_custom(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Header
        header = QLabel("Wave Function Collapse + Flow Field")
        header.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 12px;")
        layout.addWidget(header)
        
        desc = QLabel("Generates Cluster via WFC, moves them through their vectorfields.")
        desc.setStyleSheet("color: #888; font-size: 10px; margin-bottom: 10px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        self._add_field_to_layout(layout, "old", "Use WFC (Multitype)", field_type="bool", default=False)

        self._add_field_to_layout(layout, "grid_size", "Grid Size (X,Y,Z)", field_type="vector3_int")
        self._add_field_to_layout(layout, "sparsity_factor", "Sparsity Factor", field_type="float", 
                                  min_val=0.0, max_val=1.0, default=0.9)
        self._add_field_to_layout(layout, "sparse_holes", "Sparse Holes", field_type="int", 
                                  min_val=0, max_val=100, default=0)
        self._add_field_to_layout(layout, "num_steps", "Flow Steps", field_type="int", 
                                  min_val=1, max_val=100, default=8)
        self._add_field_to_layout(layout, "dt", "Time Step (dt)", field_type="float", 
                                  min_val=0.001, max_val=1.0, default=0.01)
        
        layout.addStretch()
        return panel

    def _create_panel_ccw(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        header = QLabel("CCW Ring Attractor")
        header.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 12px;")
        layout.addWidget(header)
        
        self._add_model_selector(layout, "tool_neuron_model", "Neuron Model")
        self._add_neuron_edit_button(layout)
        
        self._add_field_to_layout(layout, "n_neurons", "Number of Neurons", field_type="int", min_val=3, max_val=100000, default=100)
        self._add_field_to_layout(layout, "radius", "Ring Radius", field_type="float", min_val=0.1, max_val=1000.0, default=5.0)
        self._add_field_to_layout(layout, "bidirectional", "Bidirectional", field_type="bool", default=False)
        
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); line.setStyleSheet("color: #555;")
        layout.addWidget(line)
        layout.addWidget(QLabel("Connection Settings (Overrides):", styleSheet="font-weight:bold; color:#FF9800"))
        
        row_syn = QHBoxLayout()
        row_syn.addWidget(QLabel("Synapse Model:"))
        syn_combo = QComboBox()
        syn_combo.addItems(["static_synapse", "stdp_synapse", "tsodyks_synapse"]) 
        row_syn.addWidget(syn_combo)
        layout.addLayout(row_syn)
        self.widgets["ccw_syn_model"] = {'type': 'combo_text', 'widget': syn_combo}
        
        self._add_field_to_layout(layout, "ccw_weight_ex", "Exc Weight", field_type="float", default=30.0)
        self._add_field_to_layout(layout, "ccw_delay_ex", "Delay (ms)", field_type="float", default=1.0)
        self._add_field_to_layout(layout, "k", "Inh Factor (k)", field_type="float", default=10.0)
        
        layout.addStretch()
        return panel

    def _create_panel_blob(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        header = QLabel("Random Blob (Sphere)")
        header.setStyleSheet("color: #9C27B0; font-weight: bold; font-size: 12px;")
        layout.addWidget(header)
        
        self._add_model_selector(layout, "tool_neuron_model", "Neuron Model") 
        self._add_neuron_edit_button(layout) 
        
        self._add_field_to_layout(layout, "n_neurons", "Number of Neurons", field_type="int", min_val=1, max_val=100000, default=100)
        self._add_field_to_layout(layout, "radius", "Sphere Radius", field_type="float", min_val=0.1, max_val=1000.0, default=5.0)
        layout.addStretch()
        return panel

    def _create_panel_cone(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        header = QLabel("Cone / Cortical Column")
        header.setStyleSheet("color: #E91E63; font-weight: bold; font-size: 12px;")
        layout.addWidget(header)
        
        self._add_model_selector(layout, "tool_neuron_model", "Neuron Model")
        self._add_neuron_edit_button(layout) 
        
        self._add_field_to_layout(layout, "n_neurons", "Number of Neurons", field_type="int", min_val=1, max_val=100000, default=500)
        self._add_field_to_layout(layout, "radius_bottom", "Bottom Radius", field_type="float", min_val=0.1, max_val=1000.0, default=5.0)
        self._add_field_to_layout(layout, "radius_top", "Top Radius", field_type="float", min_val=0.0, max_val=1000.0, default=1.0)
        self._add_field_to_layout(layout, "height", "Height", field_type="float", min_val=0.1, max_val=1000.0, default=10.0)
        layout.addStretch()
        return panel

    def _create_panel_grid(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        header = QLabel("▦ 2D Grid Layer")
        header.setStyleSheet("color: #00BCD4; font-weight: bold; font-size: 12px;")
        layout.addWidget(header)
        
        self._add_model_selector(layout, "tool_neuron_model", "Neuron Model")
        
        self._add_field_to_layout(layout, "grid_side_length", "Grid Side Length", field_type="int", min_val=1, max_val=1000, default=10)
        layout.addStretch()
        return panel


    def _add_field_to_layout(self, layout, key, label, field_type="float", 
                             min_val=None, max_val=None, default=None):
 
        row = QHBoxLayout()
        
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        row.addWidget(lbl)
        
        if field_type == "int":
            widget = QSpinBox()
            widget.setRange(min_val or 0, max_val or 100000)
            widget.setValue(default or 0)
            widget.valueChanged.connect(self.on_change)
            row.addWidget(widget)
            self.widgets[key] = {'type': 'int', 'widget': widget, 'row_layout': row}
            
        elif field_type == "float":
            widget = QDoubleSpinBox()
            widget.setRange(min_val or -1000.0, max_val or 1000.0)
            widget.setDecimals(4)
            widget.setValue(default or 0.0)
            widget.valueChanged.connect(self.on_change)
            row.addWidget(widget)
            self.widgets[key] = {'type': 'float', 'widget': widget, 'row_layout': row}
            
        elif field_type == "bool":
            widget = QCheckBox()
            widget.setChecked(default or False)
            widget.stateChanged.connect(self.on_change)
            row.addWidget(widget)
            row.addStretch()
            self.widgets[key] = {'type': 'bool', 'widget': widget, 'row_layout': row}
            
        elif field_type == "vector3_int":
            widgets = []
            for prefix in ["X:", "Y:", "Z:"]:
                spin = QSpinBox()
                spin.setRange(1, 1000)
                spin.setPrefix(prefix + " ")
                spin.setValue(default or 10)
                spin.valueChanged.connect(self.on_change)
                widgets.append(spin)
                row.addWidget(spin)
            self.widgets[key] = {'type': 'vector3_int', 'widgets': widgets, 'row_layout': row}
        
        layout.addLayout(row)

    def on_tool_changed(self):
        tool_id = self.tool_combo.currentData()
        
        panel_index = self.tool_panels.get(tool_id, 0)
        
        self.tool_stack.setCurrentIndex(panel_index)
        
        self.on_change()

    def add_section(self, title):
        label = QLabel(title)
        label.setStyleSheet("font-weight: bold; font-size: 13px; color: #2196F3; margin-top: 10px; border-bottom: 2px solid #2196F3; padding-bottom: 5px;")
        self.content_layout.addWidget(label)

    def add_text_field(self, key, label, parent=None):
        target_layout = parent if parent else self.content_layout
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        edit = QLineEdit()
        edit.textChanged.connect(self.on_change)
        row.addWidget(lbl)
        row.addWidget(edit)
        target_layout.addLayout(row)
        self.widgets[key] = {'type': 'text', 'widget': edit, 'row_layout': row}

    def add_int_field(self, key, label, min_val=0, max_val=10000, parent=None):
        target_layout = parent if parent else self.content_layout
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.valueChanged.connect(self.on_change)
        row.addWidget(lbl)
        row.addWidget(spin)
        target_layout.addLayout(row)
        self.widgets[key] = {'type': 'int', 'widget': spin, 'row_layout': row}

    def add_float_field(self, key, label, min_val=-1000.0, max_val=1000.0, parent=None):
        target_layout = parent if parent else self.content_layout
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(4)
        spin.valueChanged.connect(self.on_change)
        row.addWidget(lbl)
        row.addWidget(spin)
        target_layout.addLayout(row)
        self.widgets[key] = {'type': 'float', 'widget': spin, 'row_layout': row}

    def add_bool_field(self, key, label, parent=None):
        target_layout = parent if parent else self.content_layout
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        check = QCheckBox()
        check.stateChanged.connect(self.on_change)
        row.addWidget(lbl)
        row.addWidget(check)
        row.addStretch()
        target_layout.addLayout(row)
        self.widgets[key] = {'type': 'bool', 'widget': check, 'row_layout': row}

    def add_vector3_field(self, key, label, parent=None):
        target_layout = parent if parent else self.content_layout
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        
        widgets = []
        for prefix in ["X: ", "Y: ", "Z: "]:
            spin = QDoubleSpinBox()
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(3)
            spin.setPrefix(prefix)
            spin.valueChanged.connect(self.on_change)
            widgets.append(spin)
        
        row.addWidget(lbl)
        for w in widgets:
            row.addWidget(w)
        
        target_layout.addLayout(row)
        self.widgets[key] = {'type': 'vector3', 'widgets': widgets, 'row_layout': row}

    def add_vector3_int_field(self, key, label, parent=None):
        target_layout = parent if parent else self.content_layout
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        
        widgets = []
        for prefix in ["X: ", "Y: ", "Z: "]:
            spin = QSpinBox()
            spin.setRange(1, 1000)
            spin.setPrefix(prefix)
            spin.valueChanged.connect(self.on_change)
            widgets.append(spin)
        
        row.addWidget(lbl)
        for w in widgets:
            row.addWidget(w)
            
        target_layout.addLayout(row)
        self.widgets[key] = {'type': 'vector3_int', 'widgets': widgets, 'row_layout': row}

    
    def on_change(self):
        if self.auto_save:
            params = self.get_current_params()
            self.paramsChanged.emit(params)
    

    
    def set_population_count(self, count):
        self.num_populations = count
        
        while self.probability_layout.count():
            item = self.probability_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if 'probability_vector' in self.widgets:
            del self.widgets['probability_vector']
        
        if count == 0:
            label = QLabel("No populations defined yet")
            label.setStyleSheet("color: #999; font-style: italic;")
            self.probability_layout.addWidget(label)
        else:
            prob_widgets = []
            for i in range(count):
                row = QHBoxLayout()
                label = QLabel(f"Population {i+1}:")
                label.setMinimumWidth(120)
                
                spin = QDoubleSpinBox()
                spin.setRange(0.0, 1.0)
                spin.setDecimals(3)
                spin.setValue(1.0 / count)
                spin.valueChanged.connect(self.on_change)
                
                row.addWidget(label)
                row.addWidget(spin)
                self.probability_layout.addLayout(row)
                prob_widgets.append(spin)
            
            self.widgets['probability_vector'] = {'type': 'prob_list', 'widgets': prob_widgets}

        if 'old' in self.widgets:
            wfc_widget = self.widgets['old']['widget']
            
            wfc_widget.blockSignals(True)
            
            if count > 1:
                wfc_widget.setChecked(True)
            else:
                wfc_widget.setChecked(False)
                
            wfc_widget.blockSignals(False)
    
    def get_current_params(self):
        result = {}
        for key, info in self.widgets.items():
            wtype = info['type']
            
            if wtype == 'text':
                result[key] = info['widget'].text()
            elif wtype == 'combo_text': 
                result[key] = info['widget'].currentText()
            
            elif wtype == 'int':
                result[key] = info['widget'].value()
            elif wtype == 'float':
                result[key] = info['widget'].value()
            elif wtype == 'bool':
                result[key] = info['widget'].isChecked()
            elif wtype == 'combo':
                result[key] = info['widget'].currentData()
            elif wtype == 'vector3':
                result[key] = [s.value() for s in info['widgets']]
            elif wtype == 'vector3_int':
                result[key] = [s.value() for s in info['widgets']]
            elif wtype == 'prob_list':
                result[key] = [s.value() for s in info['widgets']]
        
        if 'center_of_mass' in result:
            result['m'] = result['center_of_mass'].copy()
        
        sx = result.get('stretch_x', 1.0)
        sy = result.get('stretch_y', 1.0)
        sz = result.get('stretch_z', 1.0)
        
        result['transform_matrix'] = [
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, sz]
        ]
        return result

    def load_data(self, data):
        self.auto_save = False
        self.node_data = data
        
        for key, info in self.widgets.items():
            if key not in data:
                continue
            value = data[key]
            wtype = info['type']
            
            
            if wtype == 'combo_text': 
                idx = info['widget'].findText(str(value))
                if idx >= 0: info['widget'].setCurrentIndex(idx)
            elif wtype == 'text':
                info['widget'].setText(str(value))
            elif wtype == 'int':
                info['widget'].setValue(int(value))
            elif wtype == 'float':
                info['widget'].setValue(float(value))
            elif wtype == 'bool':
                info['widget'].setChecked(bool(value))
            elif wtype == 'combo':
                index = info['widget'].findData(value)
                if index >= 0:
                    info['widget'].setCurrentIndex(index)
            elif wtype in ['vector3', 'vector3_int']:
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
                    for i, spin in enumerate(info['widgets']):
                        spin.blockSignals(True)
                        spin.setValue(float(value[i]) if wtype == 'vector3' else int(value[i]))
                        spin.blockSignals(False)
            elif wtype == 'prob_list':
                if isinstance(value, (list, tuple)):
                    for i, v in enumerate(value):
                        if i < len(info['widgets']):
                            info['widgets'][i].blockSignals(True)
                            info['widgets'][i].setValue(float(v))
                            info['widgets'][i].blockSignals(False)
        
        if 'tool_type' in data:
            tool_id = data['tool_type']
            panel_idx = self.tool_panels.get(tool_id, 0)
            self.tool_stack.setCurrentIndex(panel_idx)
        
        self.auto_save = True
    def _add_neuron_edit_button(self, layout):
        """Fügt einen Button hinzu, der zur Population-Edit-Ansicht springt."""
        btn = QPushButton("⚙️ Edit Neuron Parameters (Pop 0)")
        btn.setStyleSheet("background-color: #37474F; color: #EEE; border: 1px solid #555; padding: 5px;")
        btn.clicked.connect(self._jump_to_neuron_editor)
        layout.addWidget(btn)

    def _jump_to_neuron_editor(self):

        self.parent().parent().setCurrentIndex(2)


def _set_visible(layout, visible):
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget():
            item.widget().setVisible(visible)
        if item.layout():
            _set_visible(item.layout(), visible)

class PolynomialTrioWidget(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 20px;
                font-weight: bold;
                color: #EEE;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #00E5FF; /* Cyan Titel */
            }
        """)
        
        layout = QVBoxLayout(self)
        
        self.builder_x = PolynomialBuilderWidget("dx/dt")
        self.builder_y = PolynomialBuilderWidget("dy/dt")
        self.builder_z = PolynomialBuilderWidget("dz/dt")
        
        layout.addWidget(self.builder_x)
        line1 = QFrame(); line1.setFrameShape(QFrame.Shape.HLine); line1.setStyleSheet("background: #333;")
        layout.addWidget(line1)
        layout.addWidget(self.builder_y)
        line2 = QFrame(); line2.setFrameShape(QFrame.Shape.HLine); line2.setStyleSheet("background: #333;")
        layout.addWidget(line2)
        layout.addWidget(self.builder_z)

    def _convert_to_ui_terms(self, data):
        """Converts JSON encoded format {'indices':..., 'coefficients':...} to UI list."""
        if isinstance(data, list):
            return data
        
        if isinstance(data, dict) and 'indices' in data and 'coefficients' in data:
            terms = []
            indices = data['indices']
            coeffs = data['coefficients']
            var_map = {0: 'x', 1: 'y', 2: 'z'}
            
            for (var_idx, power), coeff in zip(indices, coeffs):
                term = {
                    'coeff': float(coeff),
                    'var': var_map.get(var_idx, 'x'),
                    'power': int(power)
                }
                terms.append(term)
            return terms
        
        return []

    def _convert_from_ui_terms(self, ui_terms):
        indices = []
        coefficients = []
        var_map = {'x': 0, 'y': 1, 'z': 2}
        
        for t in ui_terms:
            c = t['coeff']
            v = t['var']
            p = t['power']
            
            if v == '1':
                v_idx = 0
                p = 0   
            else:
                v_idx = var_map.get(v, 0)
            
            indices.append([v_idx, p])
            coefficients.append(c)
        
        return {
            'indices': indices,
            'coefficients': coefficients,
            'n': 5,      
            'decay': 0.5 
        }

    def load_polynomials(self, poly_dict):
        if not poly_dict: return
        
        x_terms = self._convert_to_ui_terms(poly_dict.get('x', []))
        y_terms = self._convert_to_ui_terms(poly_dict.get('y', []))
        z_terms = self._convert_to_ui_terms(poly_dict.get('z', []))
        
        for term in x_terms: self.builder_x.add_term(term)
        for term in y_terms: self.builder_y.add_term(term)
        for term in z_terms: self.builder_z.add_term(term)

    def get_polynomials(self):
        return {
            'x': self._convert_from_ui_terms(self.builder_x.get_terms()),
            'y': self._convert_from_ui_terms(self.builder_y.get_terms()),
            'z': self._convert_from_ui_terms(self.builder_z.get_terms())
        }



class PolynomialTermRow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(5)

        self.spin_coeff = QDoubleSpinBox()
        self.spin_coeff.setRange(-100.0, 100.0)
        self.spin_coeff.setSingleStep(0.1)
        self.spin_coeff.setValue(1.0)
        self.spin_coeff.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin_coeff.setFixedWidth(60)
        self.spin_coeff.setStyleSheet("""
            background-color: #2b2b2b; color: #00E5FF; font-weight: bold;
            border: 1px solid #444; border-radius: 3px;
        """)
        self.spin_coeff.setToolTip("Coefficient (Multiplikator)")
        layout.addWidget(self.spin_coeff)

        lbl_mult = QLabel("×")
        lbl_mult.setStyleSheet("color: #777; font-weight: bold;")
        layout.addWidget(lbl_mult)

        self.combo_var = QComboBox()
        self.combo_var.addItems(["x", "y", "z", "1"])
        self.combo_var.setFixedWidth(50)
        self.combo_var.setStyleSheet("""
            background-color: #2b2b2b; color: #FFEB3B;
            border: 1px solid #444; border-radius: 3px;
        """)
        layout.addWidget(self.combo_var)

        lbl_pow = QLabel("^")
        lbl_pow.setStyleSheet("color: #777; font-weight: bold;")
        layout.addWidget(lbl_pow)

        self.spin_pow = QSpinBox()
        self.spin_pow.setRange(0, 10)
        self.spin_pow.setFixedWidth(40)
        self.spin_pow.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin_pow.setStyleSheet("""
            background-color: #2b2b2b; color: #FF9800;
            border: 1px solid #444; border-radius: 3px;
        """)
        layout.addWidget(self.spin_pow)

        self.btn_del = QPushButton("×")
        self.btn_del.setFixedSize(20, 20)
        self.btn_del.setStyleSheet("""
            QPushButton { background: transparent; color: #F44336; border: none; font-weight: bold; }
            QPushButton:hover { background: #330000; border-radius: 10px; }
        """)
        layout.addWidget(self.btn_del)

    def get_data(self):
        return {
            "coeff": self.spin_coeff.value(),
            "var": self.combo_var.currentText(),
            "power": self.spin_pow.value()
        }

    def set_data(self, data):
        self.spin_coeff.setValue(data.get("coeff", 1.0))
        idx = self.combo_var.findText(data.get("var", "x"))
        if idx >= 0: self.combo_var.setCurrentIndex(idx)
        self.spin_pow.setValue(data.get("power", 1))


class PolynomialManagerWidget(QWidget):
    polynomialsChanged = pyqtSignal(int, list)
    
    def __init__(self):
        super().__init__()
        self.population_polynomials = {} 
        self.current_node_idx = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        header_bg = QWidget()
        header_bg.setStyleSheet("background-color: #222; border-bottom: 1px solid #444;")
        hb_layout = QVBoxLayout(header_bg)
        
        header = QLabel("FLOW FIELD DEFINITION")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #FF9800; letter-spacing: 1px;")
        hb_layout.addWidget(header)
        layout.addWidget(header_bg)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #1e1e1e; }")
        
        self.content = QWidget()
        self.content.setStyleSheet("background-color: #1e1e1e;")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setSpacing(15)
        
        scroll.setWidget(self.content)
        layout.addWidget(scroll)
        
        self.apply_btn = QPushButton("✓ APPLY FLOW FIELDS")
        self.apply_btn.setMinimumHeight(45)
        self.apply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #D84315; 
                color: white; 
                font-weight: bold; 
                font-size: 13px;
                border-top: 2px solid #BF360C;
            }
            QPushButton:hover { background-color: #FF5722; }
            QPushButton:pressed { background-color: #BF360C; }
        """)
        self.apply_btn.clicked.connect(self.apply_changes)
        layout.addWidget(self.apply_btn)
    
    def set_populations(self, population_list, node_idx=None):
        self.current_node_idx = node_idx
        
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.population_polynomials.clear()
        
        if not population_list:
            placeholder = QLabel("No populations found for this graph node.")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #666; font-style: italic; margin-top: 50px;")
            self.content_layout.addWidget(placeholder)
            return
        
        for i, pop in enumerate(population_list):
            pop_name = f"POPULATION {i} ({pop.get('model', 'unknown')})"
            
            trio = PolynomialTrioWidget(pop_name)
            
            if 'polynomials' in pop and pop['polynomials']:
                trio.load_polynomials(pop['polynomials'])
            
            self.content_layout.addWidget(trio)
            self.population_polynomials[i] = trio
            
        self.content_layout.addStretch()
    
    def apply_changes(self):
        if self.current_node_idx is None:
            return
        all_polynomials = self.get_all_polynomials()
        self.polynomialsChanged.emit(self.current_node_idx, all_polynomials)
        self.apply_btn.setText("✓ SAVED!")
        QTimer.singleShot(1000, lambda: self.apply_btn.setText("✓ APPLY FLOW FIELDS"))

    def get_all_polynomials(self):
        return [trio.get_polynomials() for trio in self.population_polynomials.values()]


class GraphNameColorWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Graph Name:"))
        self.name_input = QLineEdit()
        name_row.addWidget(self.name_input)
        layout.addLayout(name_row)
        
        save_btn = QPushButton("Create Graph")
        save_btn.clicked.connect(self.create_graph)
        layout.addWidget(save_btn)
    
    def generate_random_color(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return f"0x{r:02x}{g:02x}{b:02x}"
    
    def create_graph(self):
        data = {
            'graph_name': self.name_input.text(),
            'color': self.generate_random_color()
        }
        print(data)
        return data


class GraphOverviewWidget(QWidget):

    node_selected = pyqtSignal(int, int)
    population_selected = pyqtSignal(int, int, int)
    connection_selected = pyqtSignal(dict)
    requestConnectionCreation = pyqtSignal(int, int, int) 
    requestConnectionDeletion = pyqtSignal(dict)  
    device_selected = pyqtSignal(dict)

    COLOR_GRAPH_BG = "#000000"      
    COLOR_GRAPH_FG = "#87CEEB"      
    COLOR_NODE_BG = "#8B0000"       
    COLOR_NODE_FG = "#FFFF00"      
    COLOR_POP_BG = "#424242"        
    COLOR_POP_FG = "#FFFFFF"        
    COLOR_CONN_BG = "#aaaa00"       
    COLOR_CONN_FG = "#841414" 
    COLOR_DEVICE_BG = "#4A148C"     
    COLOR_DEVICE_FG = "#E0E0E0"     
    
    def __init__(self, parent=None, graph_list=None):
        super().__init__(parent)
        self.main_window = parent
        self.graph_list = graph_list if graph_list is not None else []
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        header = QHBoxLayout()
        title = QLabel("Graph Overview")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")
        header.addWidget(title)
        header.addStretch()
        
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setFixedSize(28, 28)
        self.refresh_btn.setToolTip("Refresh List")
        self.refresh_btn.clicked.connect(self.update_tree)
        header.addWidget(self.refresh_btn)
        
        self.expand_btn = QPushButton("⊕")
        self.expand_btn.setFixedSize(28, 28)
        self.expand_btn.setToolTip("Expand All (Alles aufklappen)")
        self.expand_btn.clicked.connect(self._expand_all)
        header.addWidget(self.expand_btn)
        
        self.collapse_btn = QPushButton("⊖")
        self.collapse_btn.setFixedSize(28, 28)
        self.collapse_btn.setToolTip("Collapse All")
        self.collapse_btn.clicked.connect(self._collapse_all)
        header.addWidget(self.collapse_btn)
        
        layout.addLayout(header)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Element", "Details"])
        self.tree.setColumnWidth(0, 240)
        self.tree.setColumnWidth(1, 200)
        self.tree.setAlternatingRowColors(False)
        self.tree.setAnimated(True)
        self.tree.setIndentation(20)
        
        self.tree.setItemsExpandable(True) 
        
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #444444; 
                border-radius: 4px;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 12px;
                outline: 0; 
            }
            
            QTreeWidget::item {
                padding: 6px 4px;
                border-bottom: 1px solid #2a2a2a;
                margin: 0px;
            }
            
            QTreeWidget::item:selected {
                background-color: #264f78;
                color: #ffffff;
                border: none;
            }
            
            QTreeWidget::item:hover {
                background-color: #2d2d2d;
            }

            
            QHeaderView {
                background-color: #333333;
                border: none;
                margin: 0px;
                padding: 0px;
            }
            
            QHeaderView::section {
                background-color: #333333;
                color: #ffffff;
                padding: 6px;
                border: none;
                border-bottom: 2px solid #555555;
                border-right: 1px solid #444444; 
                font-weight: bold;
                margin: 0px;
            }
            
            QTableCornerButton::section {
                background-color: #333333;
                border: none;
            }
            
            QScrollBar:vertical {
                border: none;
                background: #2b2b2b;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        self.tree.setFrameShape(QFrame.Shape.NoFrame)  
        self.tree.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False) 
        
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        
        layout.addWidget(self.tree)
        
        self.status_label = QLabel("No graphs")
        self.status_label.setStyleSheet("color: #aaaaaa; font-size: 11px; padding: 3px;") 
        layout.addWidget(self.status_label)

    def _show_context_menu(self, position):
        item = self.tree.itemAt(position)
        if not item: return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data: return

        item_type = data.get('type')
        menu = QMenu(self)

        if item_type in ['node', 'population']:
            pass

        elif item_type == 'connection':
            conn_data = data.get('connection')
            conn_name = conn_data.get('name', 'Connection')
            
            action_del = QAction(f"Delete '{conn_name}'", self)
            action_del.triggered.connect(lambda: self.requestConnectionDeletion.emit(conn_data))
            menu.addAction(action_del)
        elif item_type == 'device':
            dev_data = data.get('device')
            model = dev_data.get('model', 'Device')
            action_info = QAction(f"ℹDevice: {model}", self)
            action_info.setEnabled(False)
            menu.addAction(action_info)

        menu.exec(self.tree.viewport().mapToGlobal(position))


    def update_tree(self):
        self.tree.clear()
        
        if not self.graph_list:
            self.status_label.setText("No graphs loaded")
            return
        
        total_nodes = 0
        total_pops = 0
        total_conns = 0
        total_devs = 0 
        
        for graph in self.graph_list:
            graph_item = QTreeWidgetItem(self.tree)
            graph_name = getattr(graph, 'graph_name', f'Graph_{graph.graph_id}')
            graph_item.setText(0, f"{graph_name}")
            graph_item.setText(1, f"ID: {graph.graph_id} | {len(graph.node_list)} nodes")
            
            self._style_item(graph_item, self.COLOR_GRAPH_BG, self.COLOR_GRAPH_FG, bold=True)
            graph_item.setData(0, Qt.ItemDataRole.UserRole, {'type': 'graph', 'graph_id': graph.graph_id})
            graph_item.setExpanded(True)
            
            for node in graph.node_list:
                total_nodes += 1
                node_item = self._create_node_item(graph, node, graph_item)
                
                populations = self._get_populations(node)
                for pop_idx, pop_info in enumerate(populations):
                    total_pops += 1
                    pop_item = self._create_population_item(graph, node, pop_idx, pop_info, node_item)
                    
                    connections = self._get_connections_for_pop(node, pop_idx)
                    for conn in connections:
                        total_conns += 1
                        self._create_connection_item(conn, pop_item)
                    
                    devices = self._get_devices_for_pop(node, pop_idx)
                    for dev in devices:
                        total_devs += 1
                        # HIER IST DIE ÄNDERUNG: Wir übergeben IDs an _create_device_item
                        self._create_device_item(dev, pop_item, graph.graph_id, node.id, pop_idx)
        
        self.status_label.setText(
            f"{len(self.graph_list)} graphs | 🟡 {total_nodes} nodes | 🟠 {total_pops} pops | → {total_conns} conns |  {total_devs} devs"
        )
    
    def _get_devices_for_pop(self, node, pop_idx):

        
        dev_list = []
        
        if hasattr(node, 'parameters') and 'devices' in node.parameters:
            for dev in node.parameters['devices']:
                if dev.get('target_pop_id') == pop_idx:
                    dev_list.append(dev)
        elif hasattr(node, 'devices'):
            for dev in node.devices:
                if dev.get('target_pop_id') == pop_idx:
                    dev_list.append(dev)
                    
        return dev_list

    def _create_device_item(self, dev, parent_item, graph_id, node_id, pop_idx):
        item = QTreeWidgetItem(parent_item)
        
        model = dev.get('model', 'unknown_device')
        dev_id = dev.get('id', '?')
        
        params = dev.get('params', {})
        param_str = ""
        if 'rate' in params: param_str += f"Rate={params['rate']}Hz "
        if 'amplitude' in params: param_str += f"Amp={params['amplitude']}pA "
        if 'mean' in params: param_str += f"Mean={params['mean']} "
        if 'label' in params: param_str += f"File='{params['label']}' "
        if not param_str: param_str = "params..."
        
        item.setText(0, f"       {model}")
        item.setText(1, f"ID: {dev_id} | {param_str}")
        
        self._style_item(item, self.COLOR_DEVICE_BG, self.COLOR_DEVICE_FG, bold=False)
        
        # WICHTIG: Target-Informationen in das Datenpaket injizieren!
        # Damit weiß der Editor später, welches Device er löschen muss.
        device_data_with_context = dev.copy()
        device_data_with_context['target'] = {
            'graph_id': graph_id,
            'node_id': node_id,
            'pop_id': pop_idx
        }
        
        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'device',
            'device': device_data_with_context 
        })
        
        return item


    def _style_item(self, item, bg_color, fg_color, bold=False):
        bg_brush = QBrush(QColor(bg_color))
        fg_brush = QBrush(QColor(fg_color))
        
        for col in range(2):
            item.setBackground(col, bg_brush)
            item.setForeground(col, fg_brush)
        
        if bold:
            font = item.font(0)
            font.setBold(True)
            font.setPointSize(font.pointSize() + 1)
            item.setFont(0, font)
            item.setFont(1, font)
    
    def _create_node_item(self, graph, node, parent_item):
        item = QTreeWidgetItem(parent_item)
        
        is_root = not hasattr(node, 'parent') or node.parent is None
        icon = "🔵" if is_root else "🟡"
        
        node_name = getattr(node, 'name', f'Node_{node.id}')
        item.setText(0, f"{icon} Node {node.id}: {node_name}")
        
        n_pops = len(node.types) if hasattr(node, 'types') and node.types else 0
        n_conns = len(node.connections) if hasattr(node, 'connections') and node.connections else 0
        item.setText(1, f"{n_pops} pops | {n_conns} connections")
        
        self._style_item(item, self.COLOR_NODE_BG, self.COLOR_NODE_FG, bold=False)
        
        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'node',
            'graph_id': graph.graph_id,
            'node_id': node.id
        })
        
        item.setExpanded(True)
        return item
    
    def _create_population_item(self, graph, node, pop_idx, pop_info, parent_item):
        item = QTreeWidgetItem(parent_item)
        
        model = pop_info.get('model', 'unknown')
        n_neurons = pop_info.get('n_neurons', '?')
        
        item.setText(0, f"    🟠 Pop {pop_idx}: {model}")
        item.setText(1, f"[{n_neurons} neurons]")
        
        self._style_item(item, self.COLOR_POP_BG, self.COLOR_POP_FG, bold=False)
        
        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'population',
            'graph_id': graph.graph_id,
            'node_id': node.id,
            'pop_id': pop_idx,
            'info': pop_info
        })
        
        item.setExpanded(True)
        
        return item
    
    def _create_connection_item(self, conn, parent_item):
        item = QTreeWidgetItem(parent_item)
        
        source = conn.get('source', {})
        target = conn.get('target', {})
        params = conn.get('params', {})
        
        is_self = (
            source.get('graph_id') == target.get('graph_id') and
            source.get('node_id') == target.get('node_id') and
            source.get('pop_id') == target.get('pop_id')
        )
        
        if is_self:
            icon = "↻"
            target_str = "Self"
        else:
            icon = "→"
            if source.get('graph_id') != target.get('graph_id'):
                target_str = f"G{target.get('graph_id', '?')}N{target.get('node_id', '?')}P{target.get('pop_id', '?')}"
            else:
                target_str = f"N{target.get('node_id', '?')}P{target.get('pop_id', '?')}"
        
        synapse = params.get('synapse_model', 'static')
        if len(synapse) > 15:
            synapse = synapse[:13] + ".."
        
        weight = params.get('weight', '?')
        if isinstance(weight, (int, float)):
            weight_str = f"{weight:.2f}"
        else:
            weight_str = str(weight)
        
        rule = params.get('rule', 'all_to_all')
        if len(rule) > 12:
            rule = rule[:10] + ".."
        
        item.setText(0, f"        {icon} {target_str}")
        item.setText(1, f"{synapse} | w={weight_str} | {rule}")
        
        self._style_item(item, self.COLOR_CONN_BG, self.COLOR_CONN_FG, bold=True)
        
        font = item.font(0)
        font.setPointSize(max(10, font.pointSize()))
        font.setBold(True)
        item.setFont(0, font)
        item.setFont(1, font)
        
        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'connection',
            'connection': conn
        })
        
        return item
    
    def _get_populations(self, node):
        populations = []
        types = node.types if hasattr(node, 'types') and node.types else []
        models = node.neuron_models if hasattr(node, 'neuron_models') and node.neuron_models else []
        
        for i, t in enumerate(types):
            n_neurons = 0
            if hasattr(node, 'positions') and node.positions and i < len(node.positions):
                pos = node.positions[i]
                if pos is not None and hasattr(pos, '__len__'):
                    n_neurons = len(pos)
            
            if n_neurons == 0 and hasattr(node, 'population') and node.population:
                if i < len(node.population) and node.population[i] is not None:
                    n_neurons = len(node.population[i])
            
            pop_info = {
                'type': t,
                'model': models[i] if i < len(models) else 'iaf_psc_alpha',
                'n_neurons': n_neurons
            }
            populations.append(pop_info)
        return populations
    
    def _get_connections_for_pop(self, node, pop_idx):
        if not hasattr(node, 'connections') or not node.connections:
            return []
        
        result = []
        for conn in node.connections:
            source = conn.get('source', {})
            if source.get('node_id') == node.id and source.get('pop_id') == pop_idx:
                result.append(conn)
        return result
    
    def _expand_all(self):
        self.tree.expandAll()
    
    def _collapse_all(self):
        self.tree.collapseAll()
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setExpanded(True)
    
    def _on_item_clicked(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data: return
        
        item_type = data.get('type')
        
        if item_type == 'node':
            self.node_selected.emit(data['graph_id'], data['node_id'])
        elif item_type == 'population':
            self.population_selected.emit(data['graph_id'], data['node_id'], data['pop_id'])
        elif item_type == 'connection':
            self.connection_selected.emit(data['connection'])
        elif item_type == 'device':
            self.device_selected.emit(data['device'])
    
    def _on_item_double_clicked(self, item, column):
        pass
            
################################################################################
node_parameters1 = {
    "grid_size": [8, 8, 8], 
    "m": [0.0, 0.0, 0.0],
    "rot_theta": 0.0,
    "rot_phi": 0.0,
    "stretch_x": 1.0, 
    "stretch_y": 1.0, 
    "stretch_z": 1.0,
    "transform_matrix": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
    "dt": 0.001,
    "old": True,
    "num_steps": 5,
    "sparse_holes": 0,
    "sparsity_factor": 0.9,
    "probability_vector": [0.3, 0.2, 0.4],
    "name": "Node",
    "id": 0,
    "polynom_max_power": 2,
    "center_of_mass": [0.0, 0.0, 0.0],
    "displacement": [0.0, 0.0, 0.0],
    "displacement_factor": 1.0, 
    "devices": [],
    "connections": []
}


################################################################################
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, 
    QSpinBox, QDoubleSpinBox, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt

class PolynomialBuilderWidget(QWidget):
    def __init__(self, axis_name="f(x)"):
        super().__init__()
        self.axis_name = axis_name
        self.rows = []
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(2)
        
        header_layout = QHBoxLayout()
        self.lbl_title = QLabel(f"{axis_name} =")
        self.lbl_title.setStyleSheet("color: #AAA; font-weight: bold; font-family: Consolas;")
        header_layout.addWidget(self.lbl_title)
        
        btn_add = QPushButton("+ Term")
        btn_add.setFixedSize(60, 20)
        btn_add.setStyleSheet("background: #333; color: #8BC34A; border: 1px solid #444; border-radius: 3px; font-size: 10px;")
        btn_add.clicked.connect(self.add_term)
        header_layout.addWidget(btn_add)
        header_layout.addStretch()
        
        self.layout.addLayout(header_layout)
        
        self.rows_layout = QVBoxLayout()
        self.layout.addLayout(self.rows_layout)
    
    def add_term(self, data=None):
        row = PolynomialTermRow()
        if data:
            row.set_data(data)
        
        row.btn_del.clicked.connect(lambda: self.remove_term(row))
        
        self.rows_layout.addWidget(row)
        self.rows.append(row)

    def remove_term(self, row):
        self.rows_layout.removeWidget(row)
        row.deleteLater()
        if row in self.rows:
            self.rows.remove(row)

    def get_terms(self):
        return [r.get_data() for r in self.rows]


def generate_random_polynomial(max_degree=2, num_terms=3):
    indices = []
    coefficients = []
        
    for _ in range(num_terms):
        var_idx = random.randint(0, 2)
            
        power = random.randint(0, max_degree)
            
        coeff = random.uniform(-1.0, 1.0)
            
        indices.append([var_idx, power])
        coefficients.append(coeff)
        
    return {
            'indices': indices,
            'coefficients': coefficients,
            'n': max_degree,
            'decay': 0.5
    }


import random
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QScrollArea, QStackedLayout
)
from PyQt6.QtCore import pyqtSignal



class GraphCreatorWidget(QWidget):
    graphCreated = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        self.node_list = []
        self.current_node_idx = None
        self.current_pop_idx = None
        self._last_polynomial_node_idx = None 
        self.init_ui()

        self.polynom_manager.polynomialsChanged.connect(self.on_polynomials_changed)




    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Graph Name:"))
        self.graph_name_input = QLineEdit()
        
        global next_graph_id 
        self.graph_name_input.setText(f"Graph_{next_graph_id}")
        self.graph_name_input.setPlaceholderText("e.g., Visual Cortex")
        
        header_layout.addWidget(self.graph_name_input)
        main_layout.addLayout(header_layout)

        noise_layout = QHBoxLayout()
        noise_layout.setContentsMargins(0, 5, 0, 10)
        
        lbl_noise = QLabel("Polynomial Noise:")
        lbl_noise.setToolTip("Determines how much new nodes deviate from Identity (f(x)=x).")
        
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 200) # 0.0 bis 2.0
        self.noise_slider.setValue(int(POLYNOM_NOISE_LEVEL * 100))
        self.noise_slider.valueChanged.connect(self._update_noise_level)
        
        self.noise_value_lbl = QLabel(f"{POLYNOM_NOISE_LEVEL:.2f}")
        self.noise_value_lbl.setFixedWidth(40)
        self.noise_value_lbl.setStyleSheet("color: #00E5FF; font-weight: bold;")
        
        noise_layout.addWidget(lbl_noise)
        noise_layout.addWidget(self.noise_slider)
        noise_layout.addWidget(self.noise_value_lbl)
        
        main_layout.addLayout(noise_layout)
        
        content_layout = QHBoxLayout()
        
        node_col = QVBoxLayout()
        node_col.addWidget(QLabel("NODES", alignment=Qt.AlignmentFlag.AlignCenter))
        
        node_scroll = QScrollArea()
        node_scroll.setWidgetResizable(True)
        self.node_list_widget = QWidget()
        self.node_list_layout = QVBoxLayout(self.node_list_widget)
        node_scroll.setWidget(self.node_list_widget)
        node_col.addWidget(node_scroll)
        
        btns_layout = QHBoxLayout()
        
        self.btn_add_node = QPushButton("+ Add Node")
        self.btn_add_node.clicked.connect(self.add_node)
        self.btn_add_node.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        
        self.btn_add_node.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.btn_add_node.customContextMenuRequested.connect(self.show_add_node_menu)
        self.btn_add_node.setToolTip("Left-Click: Add Empty Node\nRight-Click: Add Cortical Structure Preset")
        
        btns_layout.addWidget(self.btn_add_node)
        
        twin_btn = QPushButton("👥 Twin")
        twin_btn.setToolTip("Clone selected node with same parameters at new position")
        twin_btn.clicked.connect(self.create_twin_node)
        twin_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        btns_layout.addWidget(twin_btn)

        self.remove_node_btn = QPushButton("🗑️")
        self.remove_node_btn.setFixedWidth(40)
        self.remove_node_btn.setToolTip("Remove selected node")
        self.remove_node_btn.clicked.connect(self.remove_node)
        self.remove_node_btn.setEnabled(False)
        self.remove_node_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        btns_layout.addWidget(self.remove_node_btn)
        
        node_col.addLayout(btns_layout)
        content_layout.addLayout(node_col, 2)
        
        pop_col = QVBoxLayout()
        pop_col.addWidget(QLabel("POPULATIONS", alignment=Qt.AlignmentFlag.AlignCenter))
        
        pop_scroll = QScrollArea()
        pop_scroll.setWidgetResizable(True)
        self.pop_list_widget = QWidget()
        self.pop_list_layout = QVBoxLayout(self.pop_list_widget)
        pop_scroll.setWidget(self.pop_list_widget)
        pop_col.addWidget(pop_scroll)
        
        pop_btns_layout = QHBoxLayout()
        
        self.add_pop_btn = QPushButton("+ Add Pop")
        self.add_pop_btn.clicked.connect(self.add_population)
        self.add_pop_btn.setEnabled(False)
        self.add_pop_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        pop_btns_layout.addWidget(self.add_pop_btn)
        
        self.remove_pop_btn = QPushButton("🗑️")
        self.remove_pop_btn.setFixedWidth(40)
        self.remove_pop_btn.setToolTip("Remove selected population")
        self.remove_pop_btn.clicked.connect(self.remove_population)
        self.remove_pop_btn.setEnabled(False)
        self.remove_pop_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        pop_btns_layout.addWidget(self.remove_pop_btn)
        
        pop_col.addLayout(pop_btns_layout)
        
        content_layout.addLayout(pop_col, 2)
        
        editor_col = QVBoxLayout()
        self.editor_stack = QStackedWidget()
        placeholder = QLabel("← Select a Node or Population", alignment=Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("font-size: 14px; color: #999;")
        self.editor_stack.addWidget(placeholder)
        self.node_param_widget = NodeParametersWidget(node_parameters1.copy())
        self.node_param_widget.paramsChanged.connect(self.save_node_params)
        self.editor_stack.addWidget(self.node_param_widget)
        self.pop_param_widget = NeuronParametersWidget()
        self.editor_stack.addWidget(self.pop_param_widget)
        self.polynom_manager = PolynomialManagerWidget()
        self.editor_stack.addWidget(self.polynom_manager)
        editor_col.addWidget(self.editor_stack)
        content_layout.addLayout(editor_col, 6)
        
        main_layout.addLayout(content_layout)
        
        bottom_layout = QHBoxLayout()
        polynom_btn = QPushButton("Edit Polynomial Flow Field")
        polynom_btn.clicked.connect(self.open_polynomial_editor)
        polynom_btn.setMinimumHeight(50)
        polynom_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        bottom_layout.addWidget(polynom_btn)
        
        create_btn = QPushButton("CREATE GRAPH")
        create_btn.setMinimumHeight(60)
        create_btn.clicked.connect(self.create_graph)
        create_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 16px;")
        bottom_layout.addWidget(create_btn)
        
        main_layout.addLayout(bottom_layout)


    def show_add_node_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background-color: #2b2b2b; color: white; border: 1px solid #555; }")
        
        header = QAction("Add Structure Patch:", self)
        header.setEnabled(False)
        menu.addAction(header)
        menu.addSeparator()

        region_data = {r_key: {} for r_key in region_names.values()}
        for model, region_map in distributions.items():
            for r_key, prob in region_map.items():
                if prob > 0: region_data[r_key][model] = prob

        for display_name, r_key in region_names.items():
            if r_key not in region_data or not region_data[r_key]: continue
            
            models_probs = region_data[r_key]
            model_list = list(models_probs.keys())
            prob_list = list(models_probs.values())
            
            total = sum(prob_list)
            if total > 0: prob_list = [p/total for p in prob_list]

            action = QAction(display_name, self)
            action.triggered.connect(lambda checked, n=display_name, m=model_list, p=prob_list: 
                                   self.add_structure_node(n, m, p))
            menu.addAction(action)

        menu.exec(self.btn_add_node.mapToGlobal(pos))

    def add_structure_node(self, name, models, probs):
        
        self.add_node()
        
        node_idx = len(self.node_list) - 1
        node_data = self.node_list[node_idx]
        
        safe_name = name.replace(" ", "_").replace("/", "-")
        node_data['params']['name'] = f"{safe_name}_{node_idx}"
        node_data['params']['grid_size'] = [10, 10, 10]
        node_data['params']['probability_vector'] = probs
        node_data['params']['sparsity_factor'] = 0.85
        
        node_data['button'].setText(f"Node {node_idx + 1}: {node_data['params']['name']}")
        
        populations = []
        for i, model in enumerate(models):
            default_polynomials = {
                'x': generate_biased_polynomial(axis_idx=0, max_degree=2),
                'y': generate_biased_polynomial(axis_idx=1, max_degree=2),
                'z': generate_biased_polynomial(axis_idx=2, max_degree=2)
            }
            
            populations.append({
                'model': model,
                'params': {}, 
                'polynomials': default_polynomials
            })
            
        node_data['populations'] = populations
        
        self.select_node(node_idx)
        print(f"Added Structure Node: {name}")

    def _update_noise_level(self, value):
        global POLYNOM_NOISE_LEVEL
        POLYNOM_NOISE_LEVEL = value / 100.0
        self.noise_value_lbl.setText(f"{POLYNOM_NOISE_LEVEL:.2f}")



    def create_twin_node(self):
        if self.current_node_idx is None:
            print("Please select a node to clone first.")
            return
            
        # 1. Aktuellen Zustand sichern
        self.save_node_params(self.node_param_widget.get_current_params())
        self.save_current_population_params()
        
        # 2. Quelldaten holen
        source_node = self.node_list[self.current_node_idx]
        
        # --- FIX: Populationen bereinigen (UI-Buttons entfernen) ---
        clean_populations = []
        if 'populations' in source_node:
            for pop in source_node['populations']:
                # Wir kopieren nur die Daten, nicht den 'button' Key
                clean_pop = {
                    'model': pop.get('model'),
                    'params': pop.get('params', {}),
                    'polynomials': pop.get('polynomials', {})
                }
                clean_populations.append(clean_pop)
        # -----------------------------------------------------------

        data_to_copy = {
            'params': source_node['params'],
            'populations': clean_populations # Verwende bereinigte Liste
        }
        
        # Jetzt ist deepcopy sicher, da keine QPushButtons mehr enthalten sind
        new_node_data = copy.deepcopy(data_to_copy)
        
        current_pos = source_node['params'].get('center_of_mass', [0,0,0])
        old_id = source_node['params'].get('id')
        
        # 3. Dialog für neue Position
        dlg = PositionDialog(current_pos, self)
        if dlg.exec():
            new_pos = dlg.get_position()
            
            import copy
            import time
            
            # 4. Deepcopy der Datenstruktur (bereits oben passiert, hier nur Zuweisung)
            # new_node_data ist bereits sauber
            
            # WICHTIG: Referenzen bereinigen!
            new_node_data['original_node'] = None 
            
            # 5. Parameter anpassen
            new_idx = len(self.node_list)
            new_node_data['params']['id'] = new_idx
            new_node_data['params']['name'] = f"{source_node['params'].get('name', 'Node')}_Twin"
            
            # Position setzen
            new_node_data['params']['m'] = list(new_pos)
            new_node_data['params']['center_of_mass'] = list(new_pos)
            new_node_data['params']['old_center_of_mass'] = list(new_pos)
            
            # 6. Interne Verbindungen klonen (Self-Connections)
            source_conns = source_node['params'].get('connections', [])
            twin_conns = []
            
            for i, conn in enumerate(source_conns):
                src_id = conn['source'].get('node_id')
                tgt_id = conn['target'].get('node_id')
                
                # Nur Verbindungen kopieren, die INNERHALB des Nodes bleiben (Self-Loops)
                if src_id == old_id and tgt_id == old_id:
                    # Hier ist deepcopy sicher, da connections reine dicts sind
                    new_conn = copy.deepcopy(conn)
                    
                    # IDs auf den neuen Node mappen
                    new_conn['source']['node_id'] = new_idx
                    new_conn['target']['node_id'] = new_idx
                    
                    # Neue eindeutige ID generieren
                    new_conn['id'] = int(time.time() * 10000) + i
                    new_conn['name'] = f"{conn.get('name', 'conn')}_Twin"
                    
                    twin_conns.append(new_conn)
            
            new_node_data['params']['connections'] = twin_conns
            
            # 7. UI Button erstellen
            node_btn = QPushButton(f"Node {new_idx + 1}: {new_node_data['params']['name']}")
            node_btn.setMinimumHeight(50)
            node_btn.clicked.connect(lambda checked=False, idx=new_idx: self.select_node(idx))
            self.node_list_layout.addWidget(node_btn)
            
            # Button wieder in das Dictionary einfügen (für die Laufzeit)
            new_node_data['button'] = node_btn
            
            # 8. Hinzufügen und Auswählen
            self.node_list.append(new_node_data)
            self.select_node(new_idx)
            
            print(f"Twin created at {new_pos} with {len(twin_conns)} internal connections.")


    def add_population(self):
        if self.current_node_idx is None:
            return
        
        self.save_current_population_params()
        
        node = self.node_list[self.current_node_idx]
        pop_idx = len(node['populations'])
        
        default_polynomials = {
            'x': generate_biased_polynomial(axis_idx=0, max_degree=2),
            'y': generate_biased_polynomial(axis_idx=1, max_degree=2),
            'z': generate_biased_polynomial(axis_idx=2, max_degree=2)
        }
        
        node['populations'].append({
            'model': 'iaf_psc_alpha',
            'params': {},
            'polynomials': default_polynomials  
        })
        
        num_pops = len(node['populations'])
        self.node_param_widget.set_population_count(num_pops)
        node['params']['probability_vector'] = [1.0/num_pops] * num_pops
        self.node_param_widget.load_data(node['params'])
        
        self.update_population_list()
        self.select_population(pop_idx)
    
    def open_polynomial_editor(self):
        if self.current_node_idx is None:
            print("Please select a node first!")
            return
        
        node = self.node_list[self.current_node_idx]
        if not node['populations']:
            print("Please add populations first!")
            return
        
        self.save_current_population_params()
        
        if hasattr(self, '_last_polynomial_node_idx') and self._last_polynomial_node_idx is not None:
            last_node = self.node_list[self._last_polynomial_node_idx]
            all_polynomials = self.polynom_manager.get_all_polynomials()
            for i, poly_dict in enumerate(all_polynomials):
                if i < len(last_node['populations']):
                    last_node['populations'][i]['polynomials'] = poly_dict
        
        self._last_polynomial_node_idx = self.current_node_idx
        
        self.polynom_manager.set_populations(node['populations'], self.current_node_idx)
        self.editor_stack.setCurrentIndex(3)
    
    def on_polynomials_changed(self, node_idx, polynomials):
        if node_idx >= len(self.node_list):
            return
        
        node = self.node_list[node_idx]
        
        for i, poly_dict in enumerate(polynomials):
            if i < len(node['populations']):
                node['populations'][i]['polynomials'] = poly_dict
        
        print(f"Updated polynomials for Node {node_idx}")
    
    def create_graph(self):
        global next_graph_id, graph_parameters
        
        if not self.graph_name_input.text():
            print("ERROR: Graph name required!")
            return
        
        if not self.node_list:
            print("ERROR: Add at least one node!")
            return
        
        for i, node in enumerate(self.node_list):
            tool_type = node['params'].get('tool_type', 'custom')
            
            if tool_type == 'custom' and not node['populations']:
                print(f"ERROR: Node {i+1} (Custom) has no populations! Please add populations.")
                return
            
            if node['populations']:
                prob_vec = node['params'].get('probability_vector', [])
                total_prob = sum(prob_vec)
                if abs(total_prob - 1.0) > 0.01:
                    print(f"ERROR: Node {i+1} probability vector sums to {total_prob:.2f}, must be 1.0!")
                    return
        
        if self.current_node_idx is not None:
            current_params = self.node_param_widget.get_current_params()
            if 'center_of_mass' in current_params:
                current_params['m'] = current_params['center_of_mass'].copy()
            self.node_list[self.current_node_idx]['params'] = current_params
        
        self.save_current_population_params()
        
        if hasattr(self, '_last_polynomial_node_idx') and self._last_polynomial_node_idx is not None:
            last_node = self.node_list[self._last_polynomial_node_idx]
            all_polynomials = self.polynom_manager.get_all_polynomials()
            for i, poly_dict in enumerate(all_polynomials):
                if i < len(last_node['populations']):
                    last_node['populations'][i]['polynomials'] = poly_dict
        
        graph_id = next_graph_id
        next_graph_id += 1
        
        converted_nodes = []
        
        for node_idx, node in enumerate(self.node_list):
            tool_type = node['params'].get('tool_type', 'custom')
            populations = node['populations']
            
            if tool_type != 'custom' and not populations:
                print(f"Auto-configuring {tool_type} node structure...")
                
                selected_model = node['params'].get('tool_neuron_model', 'iaf_psc_alpha')
                neuron_models = [selected_model] 
                
                types = [0]
                encoded_polynoms_per_type = [[]] 
                prob_vec = [1.0]
                pop_nest_params = [{}]
            else:
                neuron_models = [pop['model'] for pop in populations]
                types = list(range(len(populations)))
                
                encoded_polynoms_per_type = []
                for pop in populations:
                    poly_dict = pop.get('polynomials', None)
                    if poly_dict and all(k in poly_dict for k in ['x', 'y', 'z']):
                        encoded_polynoms_per_type.append([poly_dict['x'], poly_dict['y'], poly_dict['z']])
                    else:
                        encoded_polynoms_per_type.append([])
                
                num_pops = len(populations)
                prob_vec = node['params'].get('probability_vector', [])
                if not prob_vec or len(prob_vec) != num_pops:
                    prob_vec = [1.0 / num_pops] * num_pops if num_pops > 0 else []
                
                if num_pops > 0 and abs(sum(prob_vec) - 1.0) > 0.01:
                    s = sum(prob_vec)
                    prob_vec = [p/s for p in prob_vec] if s > 0 else [1.0/num_pops]*num_pops
                
                pop_nest_params = [pop.get('params', {}) for pop in populations]

            node_params = {
                'grid_size': node['params'].get('grid_size', [10, 10, 10]),
                'm': node['params'].get('center_of_mass', [0.0, 0.0, 0.0]),
                'center_of_mass': node['params'].get('center_of_mass', [0.0, 0.0, 0.0]),
                'displacement': node['params'].get('displacement', [0.0, 0.0, 0.0]),
                'displacement_factor': node['params'].get('displacement_factor', 1.0),
                'rot_theta': node['params'].get('rot_theta', 0.0),
                'rot_phi': node['params'].get('rot_phi', 0.0),
                
                'tool_type': tool_type,
                'n_neurons': node['params'].get('n_neurons', 100),
                'radius': node['params'].get('radius', 5.0),
                'radius_top': node['params'].get('radius_top', 1.0),
                'radius_bottom': node['params'].get('radius_bottom', 5.0),
                'height': node['params'].get('height', 10.0),
                'grid_side_length': node['params'].get('grid_side_length', 10),
                
                'k': node['params'].get('k', 10.0),
                'bidirectional': node['params'].get('bidirectional', False),
                
                'stretch_x': node['params'].get('stretch_x', 1.0),
                'stretch_y': node['params'].get('stretch_y', 1.0),
                'stretch_z': node['params'].get('stretch_z', 1.0),
                'transform_matrix': node['params'].get('transform_matrix', [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                
                'dt': node['params'].get('dt', 0.01),
                'old': node['params'].get('old', True),
                'num_steps': node['params'].get('num_steps', 8),
                'sparse_holes': node['params'].get('sparse_holes', 0),
                'sparsity_factor': node['params'].get('sparsity_factor', 0.9),
                
                'name': node['params'].get('name', f'Node_{node_idx}'),
                'id': node_idx,
                'graph_id': graph_id,
                'neuron_models': neuron_models,
                'types': types,
                'probability_vector': prob_vec,    
                'distribution': prob_vec,          
                'encoded_polynoms_per_type': encoded_polynoms_per_type,
                'population_nest_params': pop_nest_params,
                'polynom_max_power': node['params'].get('polynom_max_power', 5),
                'conn_prob': [],
                'field': None,
                'coefficients': None,
                'connections': []
            }
            
            node['populations'] = populations.copy()
            converted_nodes.append(node_params)
        
        graph_parameters[graph_id] = {
            'graph_name': self.graph_name_input.text(),
            'graph_id': graph_id,
            'color': self.generate_random_color(),
            'parameter_list': converted_nodes,
            'max_nodes': len(converted_nodes)
        }
        
        self.graphCreated.emit(graph_id)
        self.reset()
        
    def add_node(self):
        node_idx = len(self.node_list)
        node_btn = QPushButton(f"Node {node_idx + 1}")
        node_btn.setMinimumHeight(50)
        node_btn.clicked.connect(lambda checked=False, idx=node_idx: self.select_node(idx))
        self.node_list_layout.addWidget(node_btn)
        
        node_params = copy.deepcopy(node_parameters1)
        
        node_params['id'] = node_idx
        node_params['name'] = f"Node_{node_idx}"
        node_params['probability_vector'] = []
        
        node_params['m'] = [0.0, 0.0, 0.0]
        node_params['center_of_mass'] = [0.0, 0.0, 0.0]
        node_params['displacement'] = [0.0, 0.0, 0.0]
        
        self.node_list.append({
            'params': node_params,
            'populations': [],
            'button': node_btn
        })
        
        self.select_node(node_idx)

    def remove_population(self):
        if self.current_node_idx is None or self.current_pop_idx is None:
            return
            
        node = self.node_list[self.current_node_idx]
        
        node['populations'].pop(self.current_pop_idx)
        
        prob_vec = node['params'].get('probability_vector', [])
        if len(prob_vec) > self.current_pop_idx:
            prob_vec.pop(self.current_pop_idx)
            total = sum(prob_vec)
            if total > 0:
                node['params']['probability_vector'] = [p/total for p in prob_vec]
            else:
                if prob_vec:
                    node['params']['probability_vector'] = [1.0/len(prob_vec)] * len(prob_vec)
        
        self.update_population_list()
        
        num_pops = len(node['populations'])
        
        self.node_param_widget.set_population_count(num_pops)
        self.node_param_widget.load_data(node['params'])
        
        if num_pops > 0:
            new_idx = max(0, self.current_pop_idx - 1)
            self.select_population(new_idx)
        else:
            self.current_pop_idx = None
            self.remove_pop_btn.setEnabled(False)
            self.editor_stack.setCurrentIndex(1)
    def remove_node(self):
        if self.current_node_idx is None:
            return

        node_data = self.node_list.pop(self.current_node_idx)
        
        if 'button' in node_data:
            node_data['button'].deleteLater()

        for i, node in enumerate(self.node_list):
            name = node['params'].get('name', 'Node')
            node['button'].setText(f"Node {i + 1}: {name}")
            
            node['params']['id'] = i
            
            try: node['button'].clicked.disconnect()
            except: pass
            node['button'].clicked.connect(lambda checked=False, idx=i: self.select_node(idx))

        if self.node_list:
            new_idx = max(0, self.current_node_idx - 1)
            self.select_node(new_idx)
        else:
            self.current_node_idx = None
            self.editor_stack.setCurrentIndex(0)
            self.remove_node_btn.setEnabled(False)
    def select_node(self, node_idx):
        if node_idx < 0 or node_idx >= len(self.node_list): return
        
        self.save_current_population_params()
        self.current_node_idx = node_idx
        self.current_pop_idx = None
        
        # Buttons aktivieren
        self.remove_node_btn.setEnabled(True)
        self.add_pop_btn.setEnabled(True)
        self.remove_pop_btn.setEnabled(False)
        
        for i, node in enumerate(self.node_list):
            node['button'].setStyleSheet("background-color: #2196F3; color: white;" if i == node_idx else "")
        self.remove_node_btn.setEnabled(len(self.node_list) > 1)

        node_data = self.node_list[node_idx]
        tool_type = node_data['params'].get('tool_type', 'custom')

        if tool_type != 'custom' and not node_data['populations']:
            model = node_data['params'].get('tool_neuron_model', 'iaf_psc_alpha')
            node_data['populations'].append({
                'model': model,
                'params': {}, 
                'polynomials': {'x': [], 'y': [], 'z': []} 
            })

        num_pops = len(node_data['populations'])
        self.node_param_widget.set_population_count(num_pops)
        self.node_param_widget.load_data(node_data['params'])
        self.editor_stack.setCurrentIndex(1)
        
        self.update_population_list()
        
        self.add_pop_btn.setEnabled(tool_type == 'custom')

    def load_structure_preset(self, name, models, probs, grid_size=[10,10,10]):
        """Lädt ein Struktur-Preset und bereitet den Node vor."""
        print(f"Loading Structure Preset: {name}")
        
        self.reset()
        
        safe_name = name.replace(" ", "_").replace("/", "-")
        self.graph_name_input.setText(f"Graph_{safe_name}")
        
        self.add_node()
        
        node = self.node_list[0]
        
        node['params']['name'] = f"{safe_name}_Node"
        node['params']['grid_size'] = grid_size
        node['params']['probability_vector'] = probs
        
        node['params']['sparsity_factor'] = 0.85 
        node['params']['dt'] = 0.01
        
        node['populations'] = []
        
        for i, model in enumerate(models):
            default_polynomials = {
                'x': generate_biased_polynomial(axis_idx=0, max_degree=2),
                'y': generate_biased_polynomial(axis_idx=1, max_degree=2),
                'z': generate_biased_polynomial(axis_idx=2, max_degree=2)
            }
            
            node['populations'].append({
                'model': model,
                'params': {}, 
                'polynomials': default_polynomials
            })
            
        self.select_node(0)
    
    
    def save_node_params(self, params):
        if self.current_node_idx is not None:
            if 'center_of_mass' in params:
                params['m'] = params['center_of_mass'].copy()
            
            num_pops = len(self.node_list[self.current_node_idx]['populations'])
            if num_pops > 0:
                current_probs = params.get('probability_vector', [])
                while len(current_probs) < num_pops:
                    current_probs.append(0.0)
                if len(current_probs) > num_pops:
                    current_probs = current_probs[:num_pops]
                params['probability_vector'] = current_probs
                
                self.node_param_widget.auto_save = False
                self.node_param_widget.load_data(params)
                self.node_param_widget.auto_save = True
            
            self.node_list[self.current_node_idx]['params'] = params
    
    
    def update_population_list(self):
        while self.pop_list_layout.count():
            item = self.pop_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if self.current_node_idx is None:
            return
        
        node = self.node_list[self.current_node_idx]
        for i, pop in enumerate(node['populations']):
            pop_btn = QPushButton(f"Pop {i+1}: {pop['model']}")
            pop_btn.setMinimumHeight(40)
            pop_btn.clicked.connect(lambda checked, idx=i: self.select_population(idx))
            self.pop_list_layout.addWidget(pop_btn)
            pop['button'] = pop_btn
    
    
    def select_population(self, pop_idx):
        self.save_current_population_params()
        
        self.current_pop_idx = pop_idx
        self.remove_pop_btn.setEnabled(True) 
        node = self.node_list[self.current_node_idx]
        for i, pop in enumerate(node['populations']):
            if i == pop_idx:
                pop['button'].setStyleSheet("background-color: #2196F3; color: white;")
            else:
                pop['button'].setStyleSheet("")
        
        pop = node['populations'][pop_idx]
        if pop['params']:
            self.pop_param_widget.model_combo.setCurrentText(pop['model'])
        
        self.editor_stack.setCurrentIndex(2)
    
    def save_current_population_params(self):
        if self.current_node_idx is not None and self.current_pop_idx is not None:
            if self.current_node_idx >= len(self.node_list):
                print(f"[DEBUG] Skipping save_current_population_params - invalid index {self.current_node_idx}")
                return
            
            node = self.node_list[self.current_node_idx]
            
            if self.current_pop_idx >= len(node['populations']):
                print(f"[DEBUG] Skipping save_current_population_params - invalid pop index {self.current_pop_idx}")
                return
            
            pop = node['populations'][self.current_pop_idx]
            
            if self.pop_param_widget.current_model:
                pop['model'] = self.pop_param_widget.current_model
                pop['params'] = {k: w.get_value() for k, w in self.pop_param_widget.parameter_widgets.items()}
                
                if 'button' in pop:
                    pop['button'].setText(f"Pop {self.current_pop_idx+1}: {pop['model']}")

    
    def generate_random_color(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return f"0x{r:02x}{g:02x}{b:02x}"
    
    def reset(self):
        self.current_node_idx = None
        self.current_pop_idx = None
        
        self.node_list.clear()
        
        while self.node_list_layout.count():
            item = self.node_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        while self.pop_list_layout.count():
            item = self.pop_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.editor_stack.setCurrentIndex(0)
        self.add_pop_btn.setEnabled(False)
        
        custom_idx = self.node_param_widget.tool_combo.findData("custom")
        if custom_idx >= 0:
            self.node_param_widget.tool_combo.setCurrentIndex(custom_idx)
        
        global next_graph_id
        next_name = f"Graph_{next_graph_id}"
        
        self.graph_name_input.setText(next_name)
        self.graph_name_input.setCursorPosition(len(next_name))




class EditGraphWidget(QWidget):    
    graphUpdated = pyqtSignal(int)
    
    def __init__(self, graph_list=None):
        super().__init__()
        self.graph_list = graph_list if graph_list is not None else []
        self.current_graph = None
        self.current_graph_id = None
        self.node_list = []
        self.current_node_idx = None
        self.current_pop_idx = None
        self._last_polynomial_node_idx = None
        self.init_ui()
    
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select Graph to Edit:"))
        
        self.graph_selector = QComboBox()
        self.graph_selector.currentIndexChanged.connect(self.on_graph_selected)
        selector_layout.addWidget(self.graph_selector)
        
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.refresh_graph_list)
        selector_layout.addWidget(refresh_btn)
        
        main_layout.addLayout(selector_layout)
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Graph Name:"))
        self.graph_name_input = QLineEdit()
        self.graph_name_input.setPlaceholderText("Edit graph name...")
        name_layout.addWidget(self.graph_name_input)
        main_layout.addLayout(name_layout)
        
        content_layout = QHBoxLayout()
        
        node_col = QVBoxLayout()
        node_col.addWidget(QLabel("NODES", alignment=Qt.AlignmentFlag.AlignCenter))
        
        node_scroll = QScrollArea()
        node_scroll.setWidgetResizable(True)
        self.node_list_widget = QWidget()
        self.node_list_layout = QVBoxLayout(self.node_list_widget)
        node_scroll.setWidget(self.node_list_widget)
        node_col.addWidget(node_scroll)
        
        node_buttons_layout = QHBoxLayout()
        
        self.btn_add_node = QPushButton("+ Add")
        self.btn_add_node.clicked.connect(self.add_node)
        self.btn_add_node.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        
        self.btn_add_node.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.btn_add_node.customContextMenuRequested.connect(self.show_add_node_menu)
        self.btn_add_node.setToolTip("Left-Click: Empty Node\nRight-Click: Structure Patch")

        node_buttons_layout.addWidget(self.btn_add_node)
        
        twin_btn = QPushButton("👥 Twin")
        twin_btn.setToolTip("Clone selected node as a NEW node at new position")
        twin_btn.clicked.connect(self.create_twin_node)
        twin_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        node_buttons_layout.addWidget(twin_btn)

        self.remove_node_btn = QPushButton("🗑️ Remove")
        self.remove_node_btn.clicked.connect(self.remove_node)
        self.remove_node_btn.setEnabled(False)
        self.remove_node_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        node_buttons_layout.addWidget(self.remove_node_btn)
        
        node_col.addLayout(node_buttons_layout)
        
        content_layout.addLayout(node_col, 2)
        
        pop_col = QVBoxLayout()
        pop_scroll = QScrollArea()
        pop_scroll.setWidgetResizable(True)
        self.pop_list_widget = QWidget()
        self.pop_list_layout = QVBoxLayout(self.pop_list_widget)
        pop_scroll.setWidget(self.pop_list_widget)
        pop_col.addWidget(pop_scroll)
        
        pop_btns_layout = QHBoxLayout()
        
        self.add_pop_btn = QPushButton("+ Add Pop")
        self.add_pop_btn.clicked.connect(self.add_population)
        self.add_pop_btn.setEnabled(False)
        self.add_pop_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        pop_btns_layout.addWidget(self.add_pop_btn)
        
        self.remove_pop_btn = QPushButton("🗑️")
        self.remove_pop_btn.setFixedWidth(40)
        self.remove_pop_btn.setToolTip("Remove selected population")
        self.remove_pop_btn.clicked.connect(self.remove_population)
        self.remove_pop_btn.setEnabled(False)
        self.remove_pop_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        pop_btns_layout.addWidget(self.remove_pop_btn)
        
        pop_col.addLayout(pop_btns_layout)
        
        content_layout.addLayout(pop_col, 2)
        
        #  EDITOR COLUMN 
        editor_col = QVBoxLayout()
        
        self.editor_stack = QStackedWidget()
        
        placeholder = QLabel("← Select a Graph to Edit", alignment=Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("font-size: 14px; color: #999;")
        self.editor_stack.addWidget(placeholder)
        
        self.node_param_widget = NodeParametersWidget(node_parameters1.copy())
        self.node_param_widget.paramsChanged.connect(self.save_node_params)
        self.editor_stack.addWidget(self.node_param_widget)
        
        self.pop_param_widget = NeuronParametersWidget()
        self.editor_stack.addWidget(self.pop_param_widget)
        
        self.polynom_manager = PolynomialManagerWidget()
        self.editor_stack.addWidget(self.polynom_manager)
        self.conn_editor = ConnectionParamWidget()
        self.editor_stack.addWidget(self.conn_editor)
        editor_col.addWidget(self.editor_stack)
        
        content_layout.addLayout(editor_col, 6)
        
        main_layout.addLayout(content_layout)
        
        bottom_layout = QHBoxLayout()
        
        polynom_btn = QPushButton("Edit Polynomial Flow Field")
        polynom_btn.clicked.connect(self.open_polynomial_editor)
        polynom_btn.setMinimumHeight(50)
        polynom_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        bottom_layout.addWidget(polynom_btn)
        
        delete_btn = QPushButton("🗑️ Delete Graph")
        delete_btn.setMinimumHeight(50)
        delete_btn.clicked.connect(self.delete_graph)
        delete_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        bottom_layout.addWidget(delete_btn)
        
        save_btn = QPushButton("SAVE CHANGES")
        save_btn.setMinimumHeight(60)
        save_btn.clicked.connect(self.save_changes)
        save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 16px;")
        bottom_layout.addWidget(save_btn)
        
        main_layout.addLayout(bottom_layout)
        
        self.refresh_graph_list()

    def select_node_by_id(self, graph_id, node_id):
        index = self.graph_selector.findData(graph_id)
        if index >= 0:
            if self.graph_selector.currentIndex() != index:
                self.graph_selector.setCurrentIndex(index)
        
        target_idx = None
        for i, node_data in enumerate(self.node_list):
            if node_data['params'].get('id') == node_id:
                target_idx = i
                break
        
        if target_idx is not None:
            self.select_node(target_idx)
        else:
            print(f"⚠ Node ID {node_id} not found in Editor list.")

    def select_population_by_ids(self, graph_id, node_id, pop_id):
        self.select_node_by_id(graph_id, node_id)
        
        if self.current_node_idx is not None:
            node_data = self.node_list[self.current_node_idx]
            if pop_id < len(node_data['populations']):
                self.select_population(pop_id)
    def delete_connection_by_data(self, conn_data):
        
        src_gid = conn_data['source']['graph_id']
        src_nid = conn_data['source']['node_id']
        conn_id = conn_data.get('id')

        if self.current_graph_id != src_gid:
            idx = self.graph_selector.findData(src_gid)
            if idx >= 0:
                self.graph_selector.setCurrentIndex(idx)
            else:
                print(f"Error: Graph {src_gid} not found for deletion.")
                return

        target_node_idx = None
        for i, node in enumerate(self.node_list):
            if node['params']['id'] == src_nid:
                target_node_idx = i
                break
        
        if target_node_idx is None:
            print(f"Error: Source Node {src_nid} not found in current editor list.")
            return

        # 1. Remove from parameters dictionary (GUI representation)
        node_data_wrapper = self.node_list[target_node_idx]
        connections = node_data_wrapper['params'].get('connections', [])
        conn_id = conn_data.get('id')
        new_conns = [c for c in connections if c.get('id') != conn_id]
        node_data_wrapper['params']['connections'] = new_conns
        
        # 2. Remove from actual Node Object (Backend) & Update Topology
        if node_data_wrapper.get('original_node'):
            src_node_obj = node_data_wrapper['original_node']
            
            # Aus der Liste im Objekt entfernen
            if hasattr(src_node_obj, 'connections'):
                conn_id = conn_data.get('id')
                src_node_obj.connections = [c for c in src_node_obj.connections if c.get('id') != conn_id]
            
            # Ziel-Node finden
            tgt_gid = conn_data['target']['graph_id']
            tgt_nid = conn_data['target']['node_id']
            
            # Suche Ziel-Graph und Ziel-Node
            tgt_node_obj = None
            target_graph = None
            
            if self.current_graph and self.current_graph.graph_id == tgt_gid:
                target_graph = self.current_graph
            else:
                for g in self.graph_list:
                    if g.graph_id == tgt_gid:
                        target_graph = g
                        break
            
            if target_graph:
                tgt_node_obj = target_graph.get_node(tgt_nid)
            
            # --- NEU: Prüfen ob Topologie getrennt werden muss ---
            if src_node_obj and tgt_node_obj:
                src_node_obj.remove_neighbor_if_isolated(tgt_node_obj)

        self.save_changes()

    def show_add_node_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background-color: #2b2b2b; color: white; border: 1px solid #555; }")
        
        header = QAction("Add Structure Patch:", self)
        header.setEnabled(False)
        menu.addAction(header)
        menu.addSeparator()

        region_data = {r_key: {} for r_key in region_names.values()}
        for model, region_map in distributions.items():
            for r_key, prob in region_map.items():
                if prob > 0: region_data[r_key][model] = prob

        for display_name, r_key in region_names.items():
            if r_key not in region_data or not region_data[r_key]: continue
            
            models_probs = region_data[r_key]
            model_list = list(models_probs.keys())
            prob_list = list(models_probs.values())
            total = sum(prob_list)
            if total > 0: prob_list = [p/total for p in prob_list]

            action = QAction(display_name, self)
            action.triggered.connect(lambda checked, n=display_name, m=model_list, p=prob_list: 
                                   self.add_structure_node(n, m, p))
            menu.addAction(action)

        menu.exec(self.btn_add_node.mapToGlobal(pos))

    def add_structure_node(self, name, models, probs):
        """Fügt Struktur im Editor hinzu."""
        if self.current_graph is None: return

        self.add_node()
        
        node_idx = len(self.node_list) - 1
        node_data = self.node_list[node_idx]
        
        safe_name = name.replace(" ", "_").replace("/", "-")
        node_data['params']['name'] = f"{safe_name}_{node_idx}"
        node_data['params']['grid_size'] = [10, 10, 10]
        node_data['params']['probability_vector'] = probs
        node_data['params']['sparsity_factor'] = 0.85
        
        is_new = node_data.get('original_node') is None
        suffix = " (NEW)" if is_new else ""
        node_data['button'].setText(f"Node {node_idx + 1}{suffix}: {node_data['params']['name']}")
        
        populations = []
        for i, model in enumerate(models):
            default_polynomials = {
                'x': generate_biased_polynomial(axis_idx=0, max_degree=2),
                'y': generate_biased_polynomial(axis_idx=1, max_degree=2),
                'z': generate_biased_polynomial(axis_idx=2, max_degree=2)
            }
            populations.append({
                'model': model,
                'params': {}, 
                'polynomials': default_polynomials
            })
            
        node_data['populations'] = populations
        
        self.select_node(node_idx)
        print(f"Added Structure Node to Editor: {name}")

    def load_connection_editor(self, connection_data):
        src_gid = connection_data['source']['graph_id']
        index = self.graph_selector.findData(src_gid)
        if index >= 0 and self.graph_selector.currentIndex() != index:
            self.graph_selector.setCurrentIndex(index)
            
        
        src_nid = connection_data['source']['node_id']
        
        target_node_data = None
        for nd in self.node_list:
            if nd['params'].get('id') == src_nid:
                target_node_data = nd
                break
        
        if target_node_data:

            
            found_conn = None

            original_node = target_node_data.get('original_node')
            
            if original_node and hasattr(original_node, 'connections'):
                for conn in original_node.connections:
                    if conn.get('id') == connection_data.get('id'):
                        found_conn = conn
                        break
            
            if not found_conn:
                 conns = target_node_data['params'].get('connections', [])
                 for conn in conns:
                     if conn.get('id') == connection_data.get('id'):
                         found_conn = conn
                         break
            
            if found_conn:
                self.conn_editor.load_data(found_conn)
                self.editor_stack.setCurrentIndex(4)
                print(f"Editing Connection ID {found_conn.get('id')}")
            else:
                print("Connection reference not found in Editor list (maybe unsaved?). Editing copy.")
                self.conn_editor.load_data(connection_data)
                self.editor_stack.setCurrentIndex(4)
        else:
            print("Source Node for connection not found in Editor.")



    def remove_node(self):
        if self.current_node_idx is None: return
            
        node_data = self.node_list.pop(self.current_node_idx)
        
        # GUI Cleanup
        if 'button' in node_data:
            self.node_list_layout.removeWidget(node_data['button'])
            node_data['button'].deleteLater()
            
        # --- NEU: Backend Cleanup (Topologie) ---
        if self.current_graph and node_data.get('original_node'):
            original_node = node_data['original_node']
            # Graph.remove_node ruft jetzt Node.remove() auf
            self.current_graph.remove_node(original_node)
            
        for i, node in enumerate(self.node_list):
            node['params']['id'] = i
            
            old_name = node['params'].get('name', '')
            if old_name.startswith("Node_"):
                node['params']['name'] = f"Node_{i}"
            
            node['button'].setText(f"Node {i + 1}: {node['params']['name']}")
            
            try: node['button'].clicked.disconnect()
            except: pass
            node['button'].clicked.connect(lambda checked=False, idx=i: self.select_node(idx))

        if self.node_list:
            new_idx = max(0, self.current_node_idx - 1)
            self.select_node(new_idx)
        else:
            self.current_node_idx = None
            self.current_pop_idx = None
            self.editor_stack.setCurrentIndex(0) 
            self.remove_node_btn.setEnabled(False)
            self.add_pop_btn.setEnabled(False)
            self.remove_pop_btn.setEnabled(False)
            
            while self.pop_list_layout.count():
                item = self.pop_list_layout.takeAt(0)
                if item.widget(): item.widget().deleteLater()
    def remove_population(self):
        if self.current_node_idx is None or self.current_pop_idx is None:
            return
            
        node_wrapper = self.node_list[self.current_node_idx]
        
        node_wrapper['populations'].pop(self.current_pop_idx)
        

        if node_wrapper.get('original_node'):
            node_obj = node_wrapper['original_node']
            
            new_conns = []
            targets_to_check = set()
            
            # Alle Verbindungen filtern
            if hasattr(node_obj, 'connections'):
                for conn in node_obj.connections:
                    # Wenn Quelle die gelöschte Pop ID ist -> Wegwerfen & Target merken
                    if conn['source']['pop_id'] == self.current_pop_idx:
                        tgt_gid = conn['target']['graph_id']
                        tgt_nid = conn['target']['node_id']
                        targets_to_check.add((tgt_gid, tgt_nid))
                        continue 
                    
                    # Wenn Quelle > gelöschte ID -> ID dekrementieren
                    if conn['source']['pop_id'] > self.current_pop_idx:
                        conn['source']['pop_id'] -= 1
                    
                    # Target IDs müssen ebenfalls angepasst werden, wenn es ein Self-Loop ist
                    if (conn['target']['graph_id'] == node_obj.graph_id and 
                        conn['target']['node_id'] == node_obj.id and 
                        conn['target']['pop_id'] > self.current_pop_idx):
                        conn['target']['pop_id'] -= 1
                        
                    new_conns.append(conn)
                
                node_obj.connections = new_conns
                
                # --- NEU: Topologie für betroffene Ziele prüfen ---
                for (gid, nid) in targets_to_check:
                    # Graph/Node finden (vereinfacht)
                    tgt_graph = next((g for g in self.graph_list if g.graph_id == gid), None)
                    if tgt_graph:
                        tgt_node = tgt_graph.get_node(nid)
                        if tgt_node:
                            node_obj.remove_neighbor_if_isolated(tgt_node)



    def refresh_graph_list(self):
        
        current_graph_id = self.graph_selector.currentData()
        
        self.graph_selector.blockSignals(True)
        self.graph_selector.clear()
        
        if not self.graph_list: 
            self.graph_selector.addItem("No graphs available")
            self.graph_selector.setEnabled(False)
        else:
            for graph in self.graph_list: 
                name = getattr(graph, 'graph_name', f'Graph {graph.graph_id}')
                self.graph_selector.addItem(f"{name} (ID: {graph.graph_id})", graph.graph_id)
            self.graph_selector.setEnabled(True)
        
        target_index = 0
        if current_graph_id is not None:
            index = self.graph_selector.findData(current_graph_id)
            if index >= 0:
                target_index = index
        
        self.graph_selector.setCurrentIndex(target_index)
        self.graph_selector.blockSignals(False)
        
        if self.graph_selector.count() > 0 and self.graph_list:
            self.on_graph_selected(target_index)
    
    def on_graph_selected(self, index):
        
        if not self.graph_list or index < 0:
            return
        
        graph_id = self.graph_selector.currentData()
        if graph_id is None:
            return
        
        self.current_graph = None
        for graph in self.graph_list:
            if graph.graph_id == graph_id:
                self.current_graph = graph
                break
        
        if not self.current_graph:
            print(f"ERROR: Graph {graph_id} not found!")
            return
        
        self.current_graph_id = graph_id
        
        self.load_graph_data()
    
    def load_graph_data(self):
        if not self.current_graph:
            return
        
        print(f"\nLoading Graph {self.current_graph_id}")
        
        self.current_node_idx = None
        self.current_pop_idx = None
        
        self.node_list.clear()
        while self.node_list_layout.count():
            item = self.node_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        while self.pop_list_layout.count():
            item = self.pop_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        graph_name = getattr(self.current_graph, 'graph_name', f'Graph_{self.current_graph_id}')
        self.graph_name_input.setText(graph_name)
        
        for node in self.current_graph.node_list:
            self.load_node_from_graph(node)
        
        print(f"Loaded {len(self.node_list)} nodes")
        
        if self.node_list:
            self.select_node(0)
    
    def load_node_from_graph(self, node):
        node_idx = len(self.node_list)
        
        node_btn = QPushButton(f"Node {node_idx + 1}: {node.name}")
        node_btn.setMinimumHeight(50)
        node_btn.clicked.connect(lambda checked=False, idx=node_idx: self.select_node(idx))
        self.node_list_layout.addWidget(node_btn)
        
        populations = []
        if hasattr(node, 'population') and node.population:
            for pop_idx, nest_pop in enumerate(node.population):
                if nest_pop is None or len(nest_pop) == 0:
                    continue
                
                model = nest.GetStatus(nest_pop, 'model')[0]
                
                params = {}
                try:
                    status = nest.GetStatus(nest_pop)
                    if status:
                        first_neuron = status[0]
                        param_keys = ['V_m', 'E_L', 'tau_m', 'C_m', 'V_th', 
                                     't_ref', 'V_reset', 'I_e', 'tau_syn_ex', 'tau_syn_in']
                        for key in param_keys:
                            if key in first_neuron:
                                params[key] = first_neuron[key]
                except Exception as e:
                    print(f"Could not extract params from pop {pop_idx}: {e}")
                
                polynomials = {}
                if hasattr(node, 'parameters'):
                    encoded = node.parameters.get('encoded_polynoms_per_type', [])
                    if pop_idx < len(encoded) and encoded[pop_idx]:
                        poly_list = encoded[pop_idx]
                        if len(poly_list) >= 3:
                            polynomials = {
                                'x': poly_list[0],
                                'y': poly_list[1],
                                'z': poly_list[2]
                            }
                
                populations.append({
                    'model': model,
                    'params': params,
                    'polynomials': polynomials
                })
        
        node_data = {
            'params': node.parameters.copy() if hasattr(node, 'parameters') else {},
            'populations': populations,
            'button': node_btn,
            'original_node': node 
        }
        


        node_data['params'].update({
            'id': node.id,
            'name': node.name,
            'center_of_mass': node.center_of_mass.tolist(),
            'm': node.center_of_mass.tolist(),
            'old_center_of_mass': getattr(node, 'old_center_of_mass', node.center_of_mass).tolist() if isinstance(getattr(node, 'old_center_of_mass', None), np.ndarray) else node.parameters.get('old_center_of_mass'),
            'probability_vector': node.parameters.get('probability_vector', [])
        })
        
        self.node_list.append(node_data)
        
        print(f"  Loaded Node {node.id}: {node.name} ({len(populations)} populations)")
    
    def add_node(self):
        import copy
        node_idx = len(self.node_list)
        node_btn = QPushButton(f"Node {node_idx + 1} (NEW)")
        node_btn.setMinimumHeight(50)
        node_btn.clicked.connect(lambda checked=False, idx=node_idx: self.select_node(idx))
        self.node_list_layout.addWidget(node_btn)
        
        node_params = copy.deepcopy(node_parameters1)
        
        node_params['id'] = node_idx
        node_params['name'] = f"Node_{node_idx}"
        node_params['probability_vector'] = []
        
        node_params['m'] = [0.0, 0.0, 0.0]
        node_params['center_of_mass'] = [0.0, 0.0, 0.0]
        node_params['displacement'] = [0.0, 0.0, 0.0]
        
        self.node_list.append({
            'params': node_params,
            'populations': [],
            'button': node_btn,
            'original_node': None  
        })
        
        self.select_node(node_idx)
    
    def select_node(self, node_idx):
        if node_idx < 0 or node_idx >= len(self.node_list):
            print(f"Invalid node index {node_idx} (max: {len(self.node_list)-1})")
            return
        
        self.save_current_population_params()
        
        self.current_node_idx = node_idx
        self.current_pop_idx = None
        
        for i, node in enumerate(self.node_list):
            if i == node_idx:
                node['button'].setStyleSheet("background-color: #2196F3; color: white;")
            else:
                node['button'].setStyleSheet("")
        self.remove_node_btn.setEnabled(len(self.node_list) > 1)

        
        num_pops = len(self.node_list[node_idx]['populations'])
        self.node_param_widget.set_population_count(num_pops)
        self.node_param_widget.load_data(self.node_list[node_idx]['params'])
        self.editor_stack.setCurrentIndex(1)
        
        self.update_population_list()
        self.add_pop_btn.setEnabled(True)
    
    def save_node_params(self, params):
        if self.current_node_idx is not None:
            params['m'] = params['center_of_mass'].copy()
            
            num_pops = len(self.node_list[self.current_node_idx]['populations'])
            if num_pops > 0:
                current_probs = params.get('probability_vector', [])
                while len(current_probs) < num_pops:
                    current_probs.append(0.0)
                if len(current_probs) > num_pops:
                    current_probs = current_probs[:num_pops]
                params['probability_vector'] = current_probs
                
                self.node_param_widget.auto_save = False
                self.node_param_widget.load_data(params)
                self.node_param_widget.auto_save = True
            
            self.node_list[self.current_node_idx]['params'] = params
    
    def add_population(self):
        if self.current_node_idx is None:
            return
        
        self.save_current_population_params()
        
        node = self.node_list[self.current_node_idx]
        pop_idx = len(node['populations'])
        
        default_polynomials = {
            'x': generate_biased_polynomial(axis_idx=0, max_degree=2),
            'y': generate_biased_polynomial(axis_idx=1, max_degree=2),
            'z': generate_biased_polynomial(axis_idx=2, max_degree=2)
        }
        # ---------------------------------------------------
        
        node['populations'].append({
            'model': 'iaf_psc_alpha',
            'params': {},
            'polynomials': default_polynomials
        })
        
        num_pops = len(node['populations'])
        self.node_param_widget.set_population_count(num_pops)
        node['params']['probability_vector'] = [1.0/num_pops] * num_pops
        self.node_param_widget.load_data(node['params'])
        
        self.update_population_list()
        self.select_population(pop_idx)
    
    def update_population_list(self):
        while self.pop_list_layout.count():
            item = self.pop_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if self.current_node_idx is None:
            return
        
        node = self.node_list[self.current_node_idx]
        for i, pop in enumerate(node['populations']):
            pop_btn = QPushButton(f"Pop {i+1}: {pop['model']}")
            pop_btn.setMinimumHeight(40)
            pop_btn.clicked.connect(lambda checked, idx=i: self.select_population(idx))
            self.pop_list_layout.addWidget(pop_btn)
            pop['button'] = pop_btn
    
    def select_population(self, pop_idx):
        self.save_current_population_params()
        
        self.current_pop_idx = pop_idx
        
        node = self.node_list[self.current_node_idx]
        for i, pop in enumerate(node['populations']):
            if i == pop_idx:
                pop['button'].setStyleSheet("background-color: #2196F3; color: white;")
            else:
                pop['button'].setStyleSheet("")
        
        pop = node['populations'][pop_idx]
        if pop['params']:
            self.pop_param_widget.model_combo.setCurrentText(pop['model'])
        
        self.editor_stack.setCurrentIndex(2)

    def create_twin_node(self):
        # IMPORTS MÜSSEN GANZ OBEN STEHEN
        import copy
        import time
        from PyQt6.QtWidgets import QMessageBox

        if self.current_node_idx is None:
            return
            
        # 1. Sichern
        self.save_node_params(self.node_param_widget.get_current_params())
        self.save_current_population_params()
        
        # 2. Quelle
        source_node = self.node_list[self.current_node_idx]
        
        # --- BEREINIGUNG VOR DEM KOPIEREN (Pickle Fix) ---
        # Wir müssen eine saubere Version der Populations-Liste erstellen, 
        # die keine GUI-Elemente (QPushButton) enthält.
        clean_populations = []
        if 'populations' in source_node:
            for pop in source_node['populations']:
                clean_pop = {
                    'model': pop.get('model'),
                    'params': pop.get('params', {}),
                    'polynomials': pop.get('polynomials', {})
                }
                clean_populations.append(clean_pop)
        
        data_to_copy = {
            'params': source_node['params'],
            'populations': clean_populations
        }
        # -------------------------------------------------

        # Jetzt ist deepcopy sicher
        new_node_data = copy.deepcopy(data_to_copy)
        
        current_pos = source_node['params'].get('center_of_mass', [0,0,0])
        old_id = source_node['params'].get('id')
        
        # 3. Position Dialog
        dlg = PositionDialog(current_pos, self)
        if dlg.exec():
            new_pos = dlg.get_position()
            
            # WICHTIG: Referenzen löschen!
            if 'button' in new_node_data:
                del new_node_data['button']
            
            # Damit save_changes() weiß, dass dies ein NEUER Node ist
            new_node_data['original_node'] = None 
            
            # 5. Parameter anpassen
            new_idx = len(self.node_list)
            
            # Name generieren
            old_name = source_node['params'].get('name', 'Node')
            new_name = f"{old_name}_Twin"
            
            new_node_data['params']['id'] = new_idx
            new_node_data['params']['name'] = new_name
            
            new_node_data['params']['m'] = list(new_pos)
            new_node_data['params']['center_of_mass'] = list(new_pos)
            new_node_data['params']['old_center_of_mass'] = list(new_pos) 
            
            # 6. Verbindungen klonen (nur interne Self-Loops)
            source_conns = source_node['params'].get('connections', [])
            twin_conns = []
            
            # Höchste existierende Connection ID finden
            max_conn_id = 0
            for node in self.node_list:
                for c in node['params'].get('connections', []):
                    if isinstance(c.get('id'), int):
                        max_conn_id = max(max_conn_id, c.get('id'))
            
            for i, conn in enumerate(source_conns):
                src_id = conn['source'].get('node_id')
                tgt_id = conn['target'].get('node_id')
                
                # Nur interne Verbindungen kopieren
                if src_id == old_id and tgt_id == old_id:
                    new_conn = copy.deepcopy(conn)
                    
                    new_conn['source']['node_id'] = new_idx
                    new_conn['target']['node_id'] = new_idx
                    
                    max_conn_id += 1
                    new_conn['id'] = max_conn_id
                    new_conn['name'] = f"{conn.get('name', 'conn')}_Twin"
                    
                    if 'error' in new_conn:
                        del new_conn['error']
                        
                    twin_conns.append(new_conn)
            
            new_node_data['params']['connections'] = twin_conns
            
            # 7. Button erstellen
            node_btn = QPushButton(f"Node {new_idx + 1} (NEW TWIN)")
            node_btn.setMinimumHeight(50)
            # Wichtig: Lambda muss idx binden
            node_btn.clicked.connect(lambda checked=False, idx=new_idx: self.select_node(idx))
            
            self.node_list_layout.addWidget(node_btn)
            new_node_data['button'] = node_btn
            
            # 8. Liste
            self.node_list.append(new_node_data)
            self.select_node(new_idx)
            
            # Info
            print(f"Twin created in Editor with {len(twin_conns)} internal connections.")
            QMessageBox.information(self, "Twin Created", 
                                  f"Twin '{new_name}' added at {new_pos}.\n"
                                  f"Inherited {len(twin_conns)} internal connections.\n"
                                  "Click 'SAVE CHANGES' to generate the neurons in NEST.")
            
    def save_current_population_params(self):
        if self.current_node_idx is not None and self.current_pop_idx is not None:
            if self.current_node_idx < 0 or self.current_node_idx >= len(self.node_list):
                print(f"[DEBUG] Skipping save (invalid node index {self.current_node_idx}, list size {len(self.node_list)})")
                return
            
            node = self.node_list[self.current_node_idx]
            
            if self.current_pop_idx < 0 or self.current_pop_idx >= len(node['populations']):
                print(f"[DEBUG] Skipping save (invalid pop index {self.current_pop_idx})")
                return
            
            pop = node['populations'][self.current_pop_idx]
            
            if self.pop_param_widget.current_model:
                pop['model'] = self.pop_param_widget.current_model
                pop['params'] = {k: w.get_value() for k, w in self.pop_param_widget.parameter_widgets.items()}
                
                if 'button' in pop:
                    pop['button'].setText(f"Pop {self.current_pop_idx+1}: {pop['model']}")
    
    def open_polynomial_editor(self):
        if self.current_node_idx is None:
            print("Please select a node first!")
            return
        
        node = self.node_list[self.current_node_idx]
        if not node['populations']:
            print("Please add populations first!")
            return
        
        self.save_current_population_params()
        
        if hasattr(self, '_last_polynomial_node_idx') and self._last_polynomial_node_idx is not None:
            last_node = self.node_list[self._last_polynomial_node_idx]
            all_polynomials = self.polynom_manager.get_all_polynomials()
            for i, poly_dict in enumerate(all_polynomials):
                if i < len(last_node['populations']):
                    last_node['populations'][i]['polynomials'] = poly_dict
        
        self._last_polynomial_node_idx = self.current_node_idx
        
        self.polynom_manager.set_populations(node['populations'])
        self.editor_stack.setCurrentIndex(3)
    def save_changes(self):
        if not self.current_graph:
            print("ERROR: No graph selected!")
            return
        
        if not self.graph_name_input.text():
            print("ERROR: Graph name required!")
            return
        

        for i, node in enumerate(self.node_list):
            tool_type = node['params'].get('tool_type', 'custom')
            if tool_type == 'custom' and not node['populations']:
                print(f"ERROR: Node {i+1} (Custom) has no populations! Please add populations.")
                return

        new_name = self.graph_name_input.text()
        self.current_graph.graph_name = new_name
        self.save_current_population_params()
        
        if hasattr(self, '_last_polynomial_node_idx') and self._last_polynomial_node_idx is not None:
            last_node = self.node_list[self._last_polynomial_node_idx]
            all_polynomials = self.polynom_manager.get_all_polynomials()
            for i, poly_dict in enumerate(all_polynomials):
                if i < len(last_node['populations']):
                    last_node['populations'][i]['polynomials'] = poly_dict
        
        print(f"\n=== Saving changes to Graph {self.current_graph_id} ===")
        
        self.current_graph.graph_name = self.graph_name_input.text()
        
        edited_analysis = [self._node_was_edited(n) for n in self.node_list]
        nodes_with_structural_changes = [
            (n, i) for i, (n, (_, _, structural)) in enumerate(zip(self.node_list, edited_analysis)) 
            if structural
        ]
        
        structural_change_indices = {idx for _, idx in nodes_with_structural_changes}

        global _nest_simulation_has_run
        nest.ResetKernel()
        _nest_simulation_has_run = False  
        
        self.current_graph.node_list.clear()
        self.current_graph.nodes = 0
        self.current_graph._next_id = 0
        
        for node_idx, node_data in enumerate(self.node_list):
            node_params = self._build_node_params(node_idx, node_data)
            original_node = node_data.get('original_node')
            
            if node_idx == 0:
                new_node = self.current_graph.create_node(
                    parameters=node_params,
                    is_root=True,
                    auto_build=False
                )
            else:
                user_pos = node_params.get('center_of_mass', [0.0, 0.0, 0.0])
                has_explicit_position = not np.allclose(user_pos, [0.0, 0.0, 0.0])
                
                if has_explicit_position:
                    new_node = self.current_graph.create_node(
                        parameters=node_params,
                        auto_build=False
                    )
                else:
                    new_node = self.current_graph.create_node(
                        parameters=node_params,
                        other=self.current_graph.node_list[node_idx - 1],
                        auto_build=False
                    )
            
            has_structural_change = node_idx in structural_change_indices
            
            original_has_points = False
            if original_node and hasattr(original_node, 'positions') and original_node.positions:
                original_has_points = any(len(c) > 0 for c in original_node.positions if c is not None)

            if has_structural_change or not original_has_points:
                reason = "Structural change" if has_structural_change else "Original positions empty/invalid"
                print(f"  Node {node_idx}: REBUILD ({reason})")
                try:
                    new_node.build()
                except Exception as e:
                    print(f"Build failed for Node {node_idx}: {e}")
            
            elif original_node:
                old_com = np.array(original_node.center_of_mass)
                new_com = np.array(node_params['center_of_mass'])
                delta = new_com - old_com
                
                if np.any(np.abs(delta) > 1e-6):
                    print(f"  Node {node_idx}: Translation by {delta}")
                    new_node.positions = []
                    for pos_cluster in original_node.positions:
                        if pos_cluster is not None and len(pos_cluster) > 0:
                            translated = pos_cluster + delta
                            new_node.positions.append(translated)
                        else:
                            new_node.positions.append(np.array([]))
                else:
                    print(f"  Node {node_idx}: Reusing positions (copy)")
                    new_node.positions = [cluster.copy() for cluster in original_node.positions]
                
                new_node.center_of_mass = new_com
            
            new_node.populate_node()
        
        for graph in self.graph_list:
            if graph.graph_id == self.current_graph_id:
                continue
            for node in graph.node_list:
                node.populate_node()
        
        print(f"Graph '{self.current_graph.graph_name}' updated!")
        
        self.graphUpdated.emit(-1)
        idx = self.graph_selector.currentIndex()
        if idx >= 0:
            new_label = f"{new_name} (ID: {self.current_graph_id})"
            self.graph_selector.setItemText(idx, new_label)

    def _node_was_edited(self, node_data):

        original = node_data.get('original_node')
        if not original:
            return False, False, False 
        
        any_change = False
        polynomials_changed = False
        structural_changed = False
        
        if not hasattr(original, 'parameters'):
            return False, False, False
        
        old_params = original.parameters
        new_params = node_data['params']
        

        structural_keys = [
            'grid_size', 'num_steps', 'sparsity_factor', 'sparse_holes',
            'dt', 'displacement', 'displacement_factor', 
            'rot_theta', 'rot_phi', 'polynom_max_power',
            'stretch_x', 'stretch_y', 'stretch_z' 
        ]
        for key in structural_keys:
            old_val = old_params.get(key)
            new_val = new_params.get(key)
            
            if old_val != new_val:
                print(f"    Structural: {key} changed: {old_val} → {new_val}")
                any_change = True
                structural_changed = True
        

        old_com = np.array(original.center_of_mass)
        new_com = np.array(new_params['center_of_mass'])
        if not np.allclose(old_com, new_com, atol=1e-6):
            print(f"    COM changed: {old_com} → {new_com}")
            any_change = True
        

        old_pop_count = len(original.population) if hasattr(original, 'population') and original.population else 0
        new_pop_count = len(node_data['populations'])
        if old_pop_count != new_pop_count:
            print(f"    Population count changed: {old_pop_count} → {new_pop_count}")
            any_change = True
            structural_changed = True 
        

        old_models = original.neuron_models if hasattr(original, 'neuron_models') else []
        new_models = [pop['model'] for pop in node_data['populations']]
        if old_models != new_models:
            print(f"    Neuron models changed")
            any_change = True
            structural_changed = True
        

        if 'population_nest_params' in old_params:
            old_nest_params = old_params['population_nest_params']
            new_nest_params = [pop.get('params', {}) for pop in node_data['populations']]
            for i, (old_p, new_p) in enumerate(zip(old_nest_params, new_nest_params)):
                if old_p != new_p:
                    print(f"    Population {i} NEST parameters changed")
                    any_change = True

        old_prob = old_params.get('probability_vector', [])
        new_prob = new_params.get('probability_vector', [])
        if old_prob != new_prob:
            print(f"    Probability vector changed")
            any_change = True
            structural_changed = True 


        if 'encoded_polynoms_per_type' in old_params:
            old_polys = old_params['encoded_polynoms_per_type']
            new_polys = []
            for pop in node_data['populations']:
                poly_dict = pop.get('polynomials', {})
                if poly_dict and all(k in poly_dict for k in ['x', 'y', 'z']):
                    new_polys.append([poly_dict['x'], poly_dict['y'], poly_dict['z']])
                else:
                    new_polys.append([])
            
            if len(old_polys) != len(new_polys):
                print(f"    Polynomial count changed")
                any_change = True
                polynomials_changed = True
                structural_changed = True
            else:
                for i, (old_poly, new_poly) in enumerate(zip(old_polys, new_polys)):
                    if old_poly != new_poly:
                        print(f"    Pop {i} polynomials changed")
                        any_change = True
                        polynomials_changed = True
                        structural_changed = True
                        break
        
        return any_change, polynomials_changed, structural_changed


    def _build_node_params(self, node_idx, node_data):
        populations = node_data['populations']
        tool_type = node_data['params'].get('tool_type', 'custom')


        if tool_type != 'custom' and not populations:
            selected_model = node_data['params'].get('tool_neuron_model', 'iaf_psc_alpha')
            neuron_models = [selected_model]
            types = [0]
            encoded_polynoms_per_type = [[]]
            prob_vec = [1.0]
            pop_nest_params = [{}]
        else:
            neuron_models = [pop['model'] for pop in populations]
            types = list(range(len(populations)))
            
            encoded_polynoms_per_type = []
            for pop in populations:
                poly_dict = pop.get('polynomials', None)
                if poly_dict and all(k in poly_dict for k in ['x', 'y', 'z']):
                    encoded_polynoms_per_type.append([poly_dict['x'], poly_dict['y'], poly_dict['z']])
                else:
                    encoded_polynoms_per_type.append([])
            
            prob_vec = node_data['params'].get('probability_vector', [])
            if not prob_vec and len(populations) > 0:
                prob_vec = [1.0/len(populations)] * len(populations)
            
            pop_nest_params = [pop.get('params', {}) for pop in populations]

        old_com = node_data['params'].get('old_center_of_mass', None)
        if old_com is None and node_data.get('original_node'):
            orig = node_data['original_node']
            if hasattr(orig, 'old_center_of_mass'):
                old_com = orig.old_center_of_mass
        
        if old_com is None:
            old_com = node_data['params'].get('center_of_mass', [0.0, 0.0, 0.0])

        sx = node_data['params'].get('stretch_x', 1.0)
        sy = node_data['params'].get('stretch_y', 1.0)
        sz = node_data['params'].get('stretch_z', 1.0)
        
        transform_matrix = [
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, sz]
        ]
        raw_params = node_data['params']
        tool_type = raw_params.get('tool_type', 'custom')
        return {
            'name': raw_params.get('name', f'Node_{node_idx}'),
            'id': node_idx,
            'graph_id': self.current_graph_id,
            'tool_type': tool_type, 
            'neuron_models': neuron_models,
            'types': types,
            'distribution': prob_vec,
            'probability_vector': prob_vec,
            'encoded_polynoms_per_type': encoded_polynoms_per_type,
            'population_nest_params': pop_nest_params,
            'ccw_syn_model': raw_params.get('ccw_syn_model', 'static_synapse'),
            'ccw_weight_ex': float(raw_params.get('ccw_weight_ex', 30.0)),
            'ccw_delay_ex': float(raw_params.get('ccw_delay_ex', 1.0)),
            'k': float(raw_params.get('k', 10.0)), # inhibition factor
            'bidirectional': bool(raw_params.get('bidirectional', False)),
            'm': raw_params.get('center_of_mass', [0.0, 0.0, 0.0]),
            'center_of_mass': raw_params.get('center_of_mass', [0.0, 0.0, 0.0]),
            'displacement': raw_params.get('displacement', [0.0, 0.0, 0.0]),
            'displacement_factor': raw_params.get('displacement_factor', 1.0),
            'rot_theta': raw_params.get('rot_theta', 0.0),
            'rot_phi': raw_params.get('rot_phi', 0.0),
            'transform_matrix': transform_matrix,
            'stretch_x': sx, 'stretch_y': sy, 'stretch_z': sz,
            'old_center_of_mass': old_com,
            'n_neurons': int(raw_params.get('n_neurons', 100)),
            'radius': float(raw_params.get('radius', 5.0)),
            'radius_top': float(raw_params.get('radius_top', 1.0)),
            'radius_bottom': float(raw_params.get('radius_bottom', 5.0)),
            'height': float(raw_params.get('height', 10.0)),
            'grid_side_length': int(raw_params.get('grid_side_length', 10)),
            'k': float(raw_params.get('k', 10.0)),
            'bidirectional': bool(raw_params.get('bidirectional', False)),
            'grid_size': raw_params.get('grid_size', [10, 10, 10]),
            'dt': raw_params.get('dt', 0.01),
            'old': raw_params.get('old', True),
            'num_steps': raw_params.get('num_steps', 8),
            'sparse_holes': raw_params.get('sparse_holes', 0),
            'sparsity_factor': raw_params.get('sparsity_factor', 0.9),
            'polynom_max_power': raw_params.get('polynom_max_power', 5),
            'conn_prob': [],
            'field': None,
            'coefficients': None,
            'connections': [] 
        }
    
    def delete_graph(self):
        """Löscht aktuellen Graph"""
        if not self.current_graph:
            print("No graph selected!")
            return
        
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            'Delete Graph',
            f"Really delete '{self.current_graph.graph_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.graph_list.remove(self.current_graph)
            print(f"Graph {self.current_graph_id} deleted")
            
            global _nest_simulation_has_run
            import nest
            nest.ResetKernel()
            _nest_simulation_has_run = False 
            print("NEST Kernel reset")
            
            for graph in self.graph_list:
                print(f"  Repopulating Graph {graph.graph_id}: {graph.graph_name}")
                for node in graph.node_list:
                    node.populate_node()
            
            self.current_graph = None
            self.current_graph_id = None
            self.refresh_graph_list()
            self.graphUpdated.emit(-1)


class ConnectionTargetRow(QWidget):

    removeClicked = pyqtSignal(QWidget)
    def __init__(self, graph_list, index, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.index = index
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        frame = QFrame()
        frame.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444; border-radius: 4px;")
        frame_layout = QGridLayout(frame)
        frame_layout.setContentsMargins(5, 5, 5, 5)
        
        lbl_title = QLabel(f"Target #{self.index + 1}")
        lbl_title.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 10px;")
        
        self.btn_del = QPushButton("×")
        self.btn_del.setFixedSize(20, 20)
        self.btn_del.setStyleSheet("""
            QPushButton { background-color: #D32F2F; color: white; border: none; border-radius: 2px; font-weight: bold; }
            QPushButton:hover { background-color: #F44336; }
        """)
        self.btn_del.clicked.connect(lambda: self.removeClicked.emit(self))
        
        if self.index == 0:
            self.btn_del.setVisible(False)

        # Combos
        self.combo_graph = QComboBox()
        self.combo_node = QComboBox()
        self.combo_pop = QComboBox()
        
        for c in [self.combo_graph, self.combo_node, self.combo_pop]:
            c.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555; padding: 2px;")
            c.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Logik
        self.combo_graph.currentIndexChanged.connect(self.on_graph_changed)
        self.combo_node.currentIndexChanged.connect(self.on_node_changed)

        # Layout
        frame_layout.addWidget(lbl_title, 0, 0)
        frame_layout.addWidget(self.btn_del, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)
        
        frame_layout.addWidget(QLabel("G:"), 1, 0)
        frame_layout.addWidget(self.combo_graph, 1, 1)
        
        frame_layout.addWidget(QLabel("N:"), 2, 0)
        frame_layout.addWidget(self.combo_node, 2, 1)
        
        frame_layout.addWidget(QLabel("P:"), 3, 0)
        frame_layout.addWidget(self.combo_pop, 3, 1)

        layout.addWidget(frame)
        
        self.refresh_data()

    def refresh_data(self, new_graph_list=None):
        if new_graph_list is not None:
            self.graph_list = new_graph_list
            
        current_g = self.combo_graph.currentData()
        
        self.combo_graph.blockSignals(True)
        self.combo_graph.clear()
        for graph in self.graph_list:
            name = getattr(graph, 'graph_name', f'Graph {graph.graph_id}')
            self.combo_graph.addItem(f"{name} (ID: {graph.graph_id})", graph.graph_id)
        
        if current_g is not None:
            idx = self.combo_graph.findData(current_g)
            if idx >= 0: self.combo_graph.setCurrentIndex(idx)
            
        self.combo_graph.blockSignals(False)
        
        if self.combo_graph.count() > 0:
            self.on_graph_changed()

    def on_graph_changed(self):
        self.combo_node.blockSignals(True)
        self.combo_node.clear()
        
        graph_id = self.combo_graph.currentData()
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        
        if graph:
            for node in graph.node_list:
                self.combo_node.addItem(f"{node.name} (ID: {node.id})", node.id)
        
        self.combo_node.blockSignals(False)
        self.on_node_changed()

    def on_node_changed(self):
        self.combo_pop.clear()
        graph_id = self.combo_graph.currentData()
        node_id = self.combo_node.currentData()
        
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if graph:
            node = next((n for n in graph.node_list if n.id == node_id), None)
            if node and hasattr(node, 'population'):
                for i, pop in enumerate(node.population):
                    model = nest.GetStatus(pop, 'model')[0] if len(pop) > 0 else "empty"
                    self.combo_pop.addItem(f"Pop {i}: {model}", i)

    def get_selection(self):
        g = self.combo_graph.currentData()
        n = self.combo_node.currentData()
        p = self.combo_pop.currentData()
        return g, n, p




class ConnectionTool(QWidget):
    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list
        self.connections = []
        self.next_conn_id = 0
        self.current_conn_idx = None
        self.syn_param_widgets = {} 
        
        self.target_rows = []
        
        self.init_ui()

    def refresh(self):
        self.refresh_graph_list()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        left_col = QVBoxLayout()
        left_col.setContentsMargins(5, 5, 5, 5)
        
        src_group = QGroupBox("Source Population (1)")
        src_group.setStyleSheet("QGroupBox { border: 1px solid #4CAF50; font-weight: bold; margin-top: 10px; } QGroupBox::title { color: #4CAF50; }")
        src_layout = QFormLayout(src_group)
        
        self.source_graph_combo = QComboBox()
        self.source_graph_combo.currentIndexChanged.connect(self.on_source_graph_changed)
        src_layout.addRow("Graph:", self.source_graph_combo)
        
        self.source_node_combo = QComboBox()
        self.source_node_combo.currentIndexChanged.connect(self.on_source_node_changed)
        src_layout.addRow("Node:", self.source_node_combo)
        
        self.source_pop_combo = QComboBox()
        src_layout.addRow("Pop:", self.source_pop_combo)
        
        left_col.addWidget(src_group)
        
        lbl_arrow = QLabel("⬇ connects to ⬇")
        lbl_arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_arrow.setStyleSheet("font-weight: bold; color: #888; margin: 5px;")
        left_col.addWidget(lbl_arrow)
        
        tgt_group = QGroupBox("Target Populations (N)")
        tgt_group.setStyleSheet("QGroupBox { border: 1px solid #FF9800; font-weight: bold; margin-top: 10px; } QGroupBox::title { color: #FF9800; }")
        tgt_outer_layout = QVBoxLayout(tgt_group)
        tgt_outer_layout.setContentsMargins(2, 10, 2, 2)
        
        self.scroll_targets = QScrollArea()
        self.scroll_targets.setWidgetResizable(True)
        self.scroll_targets.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_targets.setStyleSheet("background: transparent;")
        
        self.targets_container = QWidget()
        self.targets_layout = QVBoxLayout(self.targets_container)
        self.targets_layout.setContentsMargins(0, 0, 0, 0)
        self.targets_layout.setSpacing(5)
        self.targets_layout.addStretch() # Push items up
        
        self.scroll_targets.setWidget(self.targets_container)
        tgt_outer_layout.addWidget(self.scroll_targets)
        
        self.btn_add_target = QPushButton("+ Add Target")
        self.btn_add_target.setStyleSheet("background-color: #333; color: #FF9800; border: 1px solid #FF9800; border-radius: 4px; padding: 4px;")
        self.btn_add_target.clicked.connect(self.add_target_row)
        tgt_outer_layout.addWidget(self.btn_add_target)
        
        left_col.addWidget(tgt_group, 2) 
        
        main_layout.addLayout(left_col, 2)
        

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        middle_container = QWidget()
        middle_col = QVBoxLayout(middle_container)
        middle_col.setContentsMargins(10, 0, 10, 0)
        
        name_layout = QHBoxLayout()
        self.conn_name_input = QLineEdit()
        self.conn_name_input.setPlaceholderText("Connection Name (Base)")
        name_layout.addWidget(QLabel("Name:"))
        name_layout.addWidget(self.conn_name_input)
        middle_col.addLayout(name_layout)

        self.tabs = QTabWidget()
        self.tab_spatial = QWidget()
        self._init_spatial_tab()
        self.tabs.addTab(self.tab_spatial, "🌍 Spatial")
        
        self.tab_topo = QWidget()
        self._init_topological_tab()
        self.tabs.addTab(self.tab_topo, "🕸️ Topological")
        
        middle_col.addWidget(self.tabs)
        
        syn_group = QGroupBox("Synapse Properties")
        syn_layout = QFormLayout(syn_group)
        
        self.syn_model_combo = QComboBox()
        self.syn_model_combo.addItems(sorted(SYNAPSE_MODELS.keys()))
        self.syn_model_combo.currentTextChanged.connect(self.on_synapse_model_changed)
        syn_layout.addRow("Model:", self.syn_model_combo)
        
        wd_layout = QHBoxLayout()
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(-10000, 10000)
        self.weight_spin.setValue(1.0)
        self.weight_spin.setDecimals(3)
        self.weight_spin.setPrefix("W: ")
        
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 1000)
        self.delay_spin.setValue(1.0)
        self.delay_spin.setDecimals(2)
        self.delay_spin.setPrefix("D: ")
        self.delay_spin.setSuffix(" ms")
        
        wd_layout.addWidget(self.weight_spin)
        wd_layout.addWidget(self.delay_spin)
        syn_layout.addRow("Base Params:", wd_layout)
        
        self.dynamic_syn_params_container = QWidget()
        self.dynamic_syn_params_layout = QVBoxLayout(self.dynamic_syn_params_container)
        self.dynamic_syn_params_layout.setContentsMargins(0,0,0,0)
        syn_layout.addRow(self.dynamic_syn_params_container)
        
        opts_layout = QHBoxLayout()
        self.allow_autapses_check = QCheckBox("Autapses")
        self.allow_autapses_check.setChecked(True)
        self.allow_multapses_check = QCheckBox("Multapses")
        self.allow_multapses_check.setChecked(True)
        self.receptor_spin = QSpinBox()
        self.receptor_spin.setRange(0, 255)
        self.receptor_spin.setPrefix("Receptor: ")
        
        opts_layout.addWidget(self.allow_autapses_check)
        opts_layout.addWidget(self.allow_multapses_check)
        opts_layout.addWidget(self.receptor_spin)
        syn_layout.addRow(opts_layout)
        
        middle_col.addWidget(syn_group)
        middle_col.addStretch()
        
        scroll_area.setWidget(middle_container)
        main_layout.addWidget(scroll_area, 3)
        

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Connection Queue", alignment=Qt.AlignmentFlag.AlignCenter))
        
        self.conn_list_widget = QWidget()
        self.conn_list_layout = QVBoxLayout(self.conn_list_widget)
        conn_scroll = QScrollArea()
        conn_scroll.setWidgetResizable(True)
        conn_scroll.setWidget(self.conn_list_widget)
        right_col.addWidget(conn_scroll)
        
        self.add_conn_btn = QPushButton("⬇ Add All to Queue")
        self.add_conn_btn.setStyleSheet("background-color: #2196F3; font-weight: bold; padding: 8px;")
        self.add_conn_btn.clicked.connect(self.add_connection)
        right_col.addWidget(self.add_conn_btn)
        
        self.delete_conn_btn = QPushButton("Remove Selected")
        self.delete_conn_btn.clicked.connect(self.delete_connection)
        self.delete_conn_btn.setEnabled(False)
        right_col.addWidget(self.delete_conn_btn)
        
        self.create_all_btn = QPushButton("🚀 CREATE IN NEST")
        self.create_all_btn.setMinimumHeight(50)
        self.create_all_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")
        self.create_all_btn.clicked.connect(self.create_all_connections)
        right_col.addWidget(self.create_all_btn)
        
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #bbb; font-size: 11px; font-weight: bold;")
        self.status_label.setWordWrap(True)
        right_col.addWidget(self.status_label)
        
        main_layout.addLayout(right_col, 2)
        
        self.refresh_graph_list()
        self.add_target_row() 
        if self.syn_model_combo.count() > 0:
            self.on_synapse_model_changed(self.syn_model_combo.currentText())

    def add_target_row(self):
        idx = len(self.target_rows)
        row = ConnectionTargetRow(self.graph_list, idx, self)
        
        row.removeClicked.connect(self.remove_target_row)
        
        self.target_rows.append(row)
        self.targets_layout.insertWidget(idx, row)
        
        QTimer.singleShot(100, lambda: self.scroll_targets.verticalScrollBar().setValue(
            self.scroll_targets.verticalScrollBar().maximum()
        ))

    def remove_target_row(self, row_widget):
        if row_widget in self.target_rows:
            self.target_rows.remove(row_widget)
            self.targets_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            
            for i, r in enumerate(self.target_rows):
                r.index = i

    def refresh_graph_list(self):
        self.source_graph_combo.clear()
        for graph in self.graph_list:
            name = getattr(graph, 'graph_name', f'Graph {graph.graph_id}')
            self.source_graph_combo.addItem(f"{name} (ID: {graph.graph_id})", graph.graph_id)
        
        if len(self.graph_list) > 0:
            self.on_source_graph_changed(0)
            
        for row in self.target_rows:
            row.refresh_data(self.graph_list)


    def on_source_graph_changed(self, index):
        self.source_node_combo.clear()
        self.source_pop_combo.clear()
        if index < 0: return
        graph_id = self.source_graph_combo.currentData()
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph: return
        for node in graph.node_list:
            self.source_node_combo.addItem(f"{node.name} (ID: {node.id})", node.id)
        if len(graph.node_list) > 0: self.on_source_node_changed(0)

    def on_source_node_changed(self, index):
        self.source_pop_combo.clear()
        if index < 0: return
        graph_id = self.source_graph_combo.currentData()
        node_id = self.source_node_combo.currentData()
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph: return
        node = next((n for n in graph.node_list if n.id == node_id), None)
        if not node: return
        if hasattr(node, 'population') and node.population:
            for i, pop in enumerate(node.population):
                model = nest.GetStatus(pop, 'model')[0] if len(pop) > 0 else "empty"
                self.source_pop_combo.addItem(f"Pop {i}: {model}", i)

    def add_connection(self):
        s_gid = self.source_graph_combo.currentData()
        s_nid = self.source_node_combo.currentData()
        s_pid = self.source_pop_combo.currentData()
        
        if None in [s_gid, s_nid, s_pid]:
            print("Source not fully selected!")
            return

        base_params = self._get_current_params()
        base_name = self.conn_name_input.text().strip() or "Conn"
        
        count_added = 0
        
        for i, row in enumerate(self.target_rows):
            t_gid, t_nid, t_pid = row.get_selection()
            
            if None in [t_gid, t_nid, t_pid]:
                print(f"Skipping Target Row {i+1}: Incomplete selection")
                continue
            
            current_name = base_name
            if len(self.target_rows) > 1:
                current_name = f"{base_name}_{i+1}"
                
            params_copy = copy.deepcopy(base_params)
            
            conn_dict = {
                'id': self.next_conn_id,
                'name': current_name,
                'source': {
                    'graph_id': s_gid,
                    'node_id': s_nid,
                    'pop_id': s_pid
                },
                'target': {
                    'graph_id': t_gid,
                    'node_id': t_nid,
                    'pop_id': t_pid
                },
                'params': params_copy
            }
            
            self.connections.append(conn_dict)
            self.next_conn_id += 1
            count_added += 1
            
        if count_added > 0:
            self.update_connection_list()
            self.status_label.setText(f"Added {count_added} connections to queue.")
        else:
            self.status_label.setText("No valid targets found.")

    
    def update_connection_list(self):
        while self.conn_list_layout.count():
            item = self.conn_list_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for i, conn in enumerate(self.connections):
            mode_icon = "🌍" if conn['params'].get('use_spatial', False) else "🕸️"
            btn = QPushButton(f"{mode_icon} {conn['name']}")
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda checked, idx=i: self.select_connection(idx))
            if 'error' in conn:
                btn.setStyleSheet("background-color: #4a1818; color: #ff9999; border: 1px solid #ff5555;")
                btn.setToolTip(f"Failed: {conn['error']}")
            if i == self.current_conn_idx:
                btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
            self.conn_list_layout.addWidget(btn)
        self.conn_list_layout.addStretch()

    def select_connection(self, idx):
        self.current_conn_idx = idx
        self.delete_conn_btn.setEnabled(True)
        self.update_connection_list()

    def delete_connection(self):
        if self.current_conn_idx is not None:
            self.connections.pop(self.current_conn_idx)
            self.current_conn_idx = None
            self.delete_conn_btn.setEnabled(False)
            self.update_connection_list()

    def create_all_connections(self):
        if not self.connections: return
        graphs_dict = {g.graph_id: g for g in self.graph_list}
        self.connection_executor = ConnectionExecutor(graphs_dict)
        success, fail, failed_items = self.connection_executor.execute_pending_connections(self.connections)
        self.connections = failed_items
        self.update_connection_list()
        if fail == 0: self.status_label.setText(f"All {success} created.")
        else: self.status_label.setText(f"{success} created, {fail} failed.")

    def set_source(self, graph_id, node_id, pop_id):
        idx_graph = self.source_graph_combo.findData(graph_id)
        if idx_graph >= 0:
            self.source_graph_combo.setCurrentIndex(idx_graph)
            idx_node = self.source_node_combo.findData(node_id)
            if idx_node >= 0:
                self.source_node_combo.setCurrentIndex(idx_node)
                idx_pop = self.source_pop_combo.findData(pop_id)
                if idx_pop >= 0: self.source_pop_combo.setCurrentIndex(idx_pop)

    def _init_spatial_tab(self):
        layout = QVBoxLayout(self.tab_spatial)
        layout.addWidget(QLabel("Connects neurons based on spatial positions.", styleSheet="color: #AAA; font-style: italic;"))
        form = QFormLayout()
        self.mask_type_combo = QComboBox()
        self.mask_type_combo.addItems(["sphere", "box", "doughnut"])
        form.addRow("Mask Shape:", self.mask_type_combo)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.01, 1000.0); self.radius_spin.setValue(0.5)
        form.addRow("Outer Radius/Size:", self.radius_spin)
        self.inner_radius_spin = QDoubleSpinBox()
        self.inner_radius_spin.setRange(0.0, 1000.0); self.inner_radius_spin.setValue(0.0)
        form.addRow("Inner Radius:", self.inner_radius_spin)
        layout.addLayout(form)
        dist_layout = QHBoxLayout()
        self.dist_dep_check = QCheckBox("Scale Weight by Distance")
        self.dist_dep_check.toggled.connect(lambda c: [self.dist_factor_spin.setEnabled(c), self.dist_offset_spin.setEnabled(c)])
        self.dist_factor_spin = QDoubleSpinBox(); self.dist_factor_spin.setEnabled(False); self.dist_factor_spin.setValue(1.0)
        self.dist_offset_spin = QDoubleSpinBox(); self.dist_offset_spin.setEnabled(False); self.dist_offset_spin.setValue(0.0)
        dist_layout.addWidget(self.dist_dep_check); dist_layout.addWidget(self.dist_factor_spin); dist_layout.addWidget(self.dist_offset_spin)
        layout.addLayout(dist_layout)
        prob_layout = QHBoxLayout()
        self.spatial_prob_spin = QDoubleSpinBox(); self.spatial_prob_spin.setRange(0.0, 1.0); self.spatial_prob_spin.setValue(1.0)
        prob_layout.addWidget(QLabel("Probability (p):")); prob_layout.addWidget(self.spatial_prob_spin)
        layout.addLayout(prob_layout); layout.addStretch()

    def _init_topological_tab(self):
        layout = QVBoxLayout(self.tab_topo)
        self.rule_combo = QComboBox()
        self.rule_combo.addItems(["all_to_all", "fixed_indegree", "fixed_outdegree", "fixed_total_number", "pairwise_bernoulli", "one_to_one"])
        self.rule_combo.currentTextChanged.connect(self.on_rule_changed)
        layout.addWidget(QLabel("Connection Rule:"))
        layout.addWidget(self.rule_combo)
        self.topo_params_widget = QWidget(); self.topo_params_layout = QFormLayout(self.topo_params_widget)
        layout.addWidget(self.topo_params_widget)
        self.indegree_spin = QSpinBox(); self.indegree_spin.setRange(1, 100000); self.indegree_spin.setValue(10)
        self.outdegree_spin = QSpinBox(); self.outdegree_spin.setRange(1, 100000); self.outdegree_spin.setValue(10)
        self.total_num_spin = QSpinBox(); self.total_num_spin.setRange(1, 1000000); self.total_num_spin.setValue(100)
        self.topo_prob_spin = QDoubleSpinBox(); self.topo_prob_spin.setRange(0, 1); self.topo_prob_spin.setValue(0.1)
        self.on_rule_changed("all_to_all"); layout.addStretch()

    def on_rule_changed(self, rule):
        while self.topo_params_layout.count():
            item = self.topo_params_layout.takeAt(0)
            if item.widget(): item.widget().setParent(None)
        if rule == "fixed_indegree": self.topo_params_layout.addRow("Indegree:", self.indegree_spin)
        elif rule == "fixed_outdegree": self.topo_params_layout.addRow("Outdegree:", self.outdegree_spin)
        elif rule == "fixed_total_number": self.topo_params_layout.addRow("Total Connections:", self.total_num_spin)
        elif "bernoulli" in rule: self.topo_params_layout.addRow("Probability:", self.topo_prob_spin)

    def on_synapse_model_changed(self, model_name):
        while self.dynamic_syn_params_layout.count():
            item = self.dynamic_syn_params_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.syn_param_widgets.clear()
        if model_name not in SYNAPSE_MODELS: return
        params = SYNAPSE_MODELS[model_name]
        for param_name, info in params.items():
            p_type = info.get('type', 'float'); p_default = info.get('default', 0.0)
            widget = None
            if p_type == 'float': widget = DoubleInputField(param_name, default_value=float(p_default))
            elif p_type == 'integer': widget = IntegerInputField(param_name, default_value=int(p_default))
            if widget:
                self.dynamic_syn_params_layout.addWidget(widget)
                self.syn_param_widgets[param_name] = widget

    def _get_current_params(self):
        is_spatial = (self.tabs.currentIndex() == 0)
        params = {
            'synapse_model': self.syn_model_combo.currentText(),
            'weight': self.weight_spin.value(),
            'delay': self.delay_spin.value(),
            'allow_autapses': self.allow_autapses_check.isChecked(),
            'allow_multapses': self.allow_multapses_check.isChecked(),
            'receptor_type': self.receptor_spin.value(),
            'use_spatial': is_spatial
        }
        if is_spatial:
            params.update({'rule': 'pairwise_bernoulli', 'p': self.spatial_prob_spin.value(), 'mask_type': self.mask_type_combo.currentText(), 'mask_radius': self.radius_spin.value(), 'mask_inner_radius': self.inner_radius_spin.value(), 'distance_dependent_weight': self.dist_dep_check.isChecked(), 'dist_factor': self.dist_factor_spin.value(), 'dist_offset': self.dist_offset_spin.value()})
        else:
            rule = self.rule_combo.currentText(); params['rule'] = rule
            if rule == 'fixed_indegree': params['indegree'] = self.indegree_spin.value()
            elif rule == 'fixed_outdegree': params['outdegree'] = self.outdegree_spin.value()
            elif rule == 'fixed_total_number': params['N'] = self.total_num_spin.value()
            elif 'bernoulli' in rule: params['p'] = self.topo_prob_spin.value()
        for name, widget in self.syn_param_widgets.items(): params[name] = widget.get_value()
        return params



"""
for graph in self.graph_list:
    for node in graph.node_list:
        if hasattr(node, 'connections'):
            for conn in node.connections:
                #  NEST Connections
                pass
            

"""


class BlinkingNetworkWidget(QWidget):
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # PyVista Setup
        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        self.plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
        self.plotter.enable_anti_aliasing()
        
        self.layout.addWidget(self.plotter)
        
        # State Arrays
        self.neuron_mesh = None
        self.edge_mesh = None
        
        self.activity = None        
        self.edge_intensities = None 
        self.base_colors = None
        
        self.connected_mask = None 
        self.base_opacity = 0.3 
        
        self.connectivity_map = [] 
        self.population_ranges = {} 
        
        self.simulation_running = False
        self.is_active = True
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        
        self.build_scene()
        self.timer.start(16)

    def set_base_opacity(self, value):
        self.base_opacity = value
        if self.edge_intensities is not None and self.edge_mesh is not None:
            mask = self.edge_intensities < 0.9
            self.edge_intensities[mask] = self.base_opacity
            self.edge_mesh.cell_data["glow"] = self.edge_intensities
            self.plotter.update()

    def build_scene(self):
        self.plotter.clear()
        self.connectivity_map = []
        self.population_ranges = {}
        
        if not self.graph_list: return
        
        all_points = []
        all_colors = []
        current_idx = 0
        
        for graph in self.graph_list:
            for node in graph.node_list:
                if not hasattr(node, 'positions') or not node.positions: continue
                for pop_idx, cluster in enumerate(node.positions):
                    if cluster is None or len(cluster) == 0: continue
                    
                    start = current_idx
                    end = current_idx + len(cluster)
                    self.population_ranges[(graph.graph_id, node.id, pop_idx)] = (start, end)
                    current_idx = end
                    
                    model = node.neuron_models[pop_idx] if pop_idx < len(node.neuron_models) else "unknown"
                    hex_c = neuron_colors.get(model, "#ffffff")
                    rgb = mcolors.to_rgb(hex_c)
                    
                    all_points.append(cluster)
                    all_colors.extend([rgb] * len(cluster))
        
        if not all_points: return

        points = np.vstack(all_points)
        n_points = len(points)
        self.base_colors = np.array(all_colors)
        self.activity = np.zeros(n_points)
        self.connected_mask = np.zeros(n_points, dtype=bool)
        
        self.neuron_mesh = pv.PolyData(points)
        self.neuron_mesh.point_data["display_color"] = self.base_colors * 0.3
        
        self.plotter.add_mesh(
            self.neuron_mesh, scalars="display_color", rgb=True,
            point_size=6, render_points_as_spheres=True, ambient=0.4
        )
        
        lines_data = []
        current_edge_idx = 0
        VISUAL_EDGE_LIMIT = 80
        
        for graph in self.graph_list:
            for node in graph.node_list:
                if not hasattr(node, 'connections'): continue
                for conn in node.connections:
                    src = conn.get('source', {})
                    tgt = conn.get('target', {})
                    params = conn.get('params', {})
                    
                    s_key = (src.get('graph_id'), src.get('node_id'), src.get('pop_id'))
                    t_key = (tgt.get('graph_id'), tgt.get('node_id'), tgt.get('pop_id'))
                    
                    if s_key in self.population_ranges and t_key in self.population_ranges:
                        s_start, s_end = self.population_ranges[s_key]
                        t_start, t_end = self.population_ranges[t_key]
                        
                        self.connected_mask[s_start:s_end] = True
                        self.connected_mask[t_start:t_end] = True
                        
                        n_possible = (s_end-s_start) * (t_end-t_start)
                        n_show = min(VISUAL_EDGE_LIMIT, int(np.sqrt(n_possible)))
                        n_show = max(1, n_show)
                        
                        edge_slice = slice(current_edge_idx, current_edge_idx + n_show)
                        current_edge_idx += n_show
                        
                        self.connectivity_map.append({
                            'src_slice': slice(s_start, s_end),
                            'tgt_indices': np.arange(t_start, t_end),
                            'weight': params.get('weight', 1.0),
                            'edge_slice': edge_slice
                        })
                        
                        p1s = np.random.randint(s_start, s_end, n_show)
                        p2s = np.random.randint(t_start, t_end, n_show)
                        for p1, p2 in zip(p1s, p2s):
                            lines_data.extend([2, p1, p2])

        if lines_data:
            self.edge_mesh = pv.PolyData(points, lines=np.array(lines_data))
            
            self.edge_intensities = np.full(self.edge_mesh.n_cells, self.base_opacity)
            self.edge_mesh.cell_data["glow"] = self.edge_intensities
            
            colors = ["#0a0500", "#8B5A2B", "#FFD700", "#FFFFFF"]
            gold_cmap = mcolors.LinearSegmentedColormap.from_list("gold", colors, N=256)
            
            self.plotter.add_mesh(
                self.edge_mesh, 
                scalars="glow", 
                cmap=gold_cmap,
                clim=[0.0, 1.0], 
                opacity="linear", 
                
                render_lines_as_tubes=True, 
                line_width=1.5,
                
                use_transparency=True, 
                show_scalar_bar=False
            )
            
        self.plotter.reset_camera()

    def start_simulation(self):
        self.simulation_running = True
        if self.activity is not None: self.activity[:] = 0
        if self.edge_intensities is not None: self.edge_intensities[:] = self.base_opacity

    def stop_simulation(self):
        self.simulation_running = False

    def stop_animation(self):
        self.stop_simulation()

    def animate(self):
        if (not self.isVisible() or 
            not self.is_active or 
            self.neuron_mesh is None or 
            not hasattr(self, 'plotter') or 
            self.plotter.render_window is None):
            return
        
        try:
            if self.simulation_running:
                self.activity *= 0.90
                
                if self.edge_intensities is not None:
                    diff = self.edge_intensities - self.base_opacity
                    self.edge_intensities = self.base_opacity + (diff * 0.80)
                    self.edge_intensities = np.maximum(self.edge_intensities, self.base_opacity)

                connected_indices = np.where(self.connected_mask)[0]
                if len(connected_indices) > 0:
                    n_fire = max(1, int(len(connected_indices) * 0.03))
                    noise_idx = np.random.choice(connected_indices, n_fire)
                    self.activity[noise_idx] += 0.6

                for conn in self.connectivity_map:
                    src_act = self.activity[conn['src_slice']]
                    mean_src = np.mean(src_act)
                    
                    if mean_src > 0.2:
                        if self.edge_intensities is not None:
                            flash_val = np.clip(mean_src * 2.5, 0.8, 1.0)
                            current = self.edge_intensities[conn['edge_slice']]
                            self.edge_intensities[conn['edge_slice']] = np.maximum(current, flash_val)
                        
                        strength = mean_src * abs(conn['weight']) * 0.5
                        targets = conn['tgt_indices']
                        n_hits = max(1, int(len(targets) * 0.3))
                        hit_idx = targets[np.random.choice(len(targets), size=n_hits)]
                        self.activity[hit_idx] += strength

                self.activity = np.clip(self.activity, 0, 1.5)
                
                display_act = np.clip(self.activity, 0, 1)
                colors = self.base_colors * 0.3 + (np.ones_like(self.base_colors) * display_act[:, None] * 0.9)
                mask = display_act > 0.8
                colors[mask] = [1.0, 1.0, 1.0] 
                self.neuron_mesh.point_data["display_color"] = colors
                
                if self.edge_mesh:
                    self.edge_mesh.cell_data["glow"] = self.edge_intensities

            else:
                # IDLE MODE
                import time
                t = time.time()
                pulse = (np.sin(t * 2) + 1) / 2 * 0.2 + 0.2
                if self.neuron_mesh and "display_color" in self.neuron_mesh.point_data:
                    self.neuron_mesh.point_data["display_color"] = self.base_colors * pulse
                
                if self.edge_intensities is not None and self.edge_mesh:
                    noise = np.random.uniform(-0.01, 0.01, len(self.edge_intensities))
                    self.edge_intensities[:] = np.clip(self.base_opacity + noise, 0, 1)
                    self.edge_mesh.cell_data["glow"] = self.edge_intensities

            if self.plotter.render_window:
                self.plotter.update()
            
        except Exception:
            pass

    def closeEvent(self, event):
        self.is_active = False
        self.simulation_running = False
        
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        if hasattr(self, 'plotter'):
            self.plotter.close()
            
        super().closeEvent(event)










    def create_nest_connections(self,graph_list):

        
        total_connections_created = 0
        
        print("CREATING NEST CONNECTIONS")
        
        for graph in self.graph_list:
            graph_id = graph.graph_id
            print(f"\nProcessing Graph {graph_id}...")
            
            for node in graph.node_list:
                node_id = node.id
                
                if not hasattr(node, 'connections') or not node.connections:
                    continue
                
                print(f"Node {node_id} has {len(node.connections)} connection(s)")
                
                for conn in node.connections:
                    try:
                        source_graph_id = conn['source']['graph_id']
                        source_node_id = conn['source']['node_id']
                        source_pop_id = conn['source']['pop_id']
                        
                        source_graph = next((g for g in graph_list if g.graph_id == source_graph_id), None)
                        if not source_graph:
                            print(f"Source Graph {source_graph_id} not found!")
                            continue
                        
                        source_node = next((n for n in source_graph.node_list if n.id == source_node_id), None)
                        if not source_node:
                            print(f"Source Node {source_node_id} not found!")
                            continue
                        
                        if source_pop_id >= len(source_node.population):
                            print(f"    ⚠ Source Population {source_pop_id} out of range!")
                            continue
                        source_population = source_node.population[source_pop_id]
                        
                        target_graph_id = conn['target']['graph_id']
                        target_node_id = conn['target']['node_id']
                        target_pop_id = conn['target']['pop_id']
                        
                        target_graph = next((g for g in graph_list if g.graph_id == target_graph_id), None)
                        if not target_graph:
                            print(f"    ⚠ Target Graph {target_graph_id} not found!")
                            continue
                        
                        target_node = next((n for n in target_graph.node_list if n.id == target_node_id), None)
                        if not target_node:
                            print(f"    ⚠ Target Node {target_node_id} not found!")
                            continue
                        
                        if target_pop_id >= len(target_node.population):
                            print(f"    ⚠ Target Population {target_pop_id} out of range!")
                            continue
                        target_population = target_node.population[target_pop_id]
                        
                        params = conn['params']
                        
                        conn_spec = {'rule': params['rule']}
                        
                        if 'indegree' in params:
                            conn_spec['indegree'] = params['indegree']
                        if 'outdegree' in params:
                            conn_spec['outdegree'] = params['outdegree']
                        if 'N' in params:
                            conn_spec['N'] = params['N']
                        if 'p' in params:
                            conn_spec['p'] = params['p']
                        
                        conn_spec['allow_autapses'] = params.get('allow_autapses', True)
                        conn_spec['allow_multapses'] = params.get('allow_multapses', True)
                        
                        syn_spec = {
                            'synapse_model': params.get('synapse_model', 'static_synapse'),
                            'weight': params.get('weight', 1.0),
                            'delay': params.get('delay', 1.0)
                        }
                        
                        nest.Connect(
                            source_population,
                            target_population,
                            conn_spec=conn_spec,
                            syn_spec=syn_spec
                        )
                        
                        total_connections_created += 1
                        
                        print(f"{conn['name']}")
                        print(f"{len(source_population)} → {len(target_population)} neurons")
                        print(f"Rule: {params['rule']}, Weight: {params['weight']}, Delay: {params['delay']}ms")
                    
                    except Exception as e:
                        print(f"Error creating connection {conn.get('name', 'unknown')}: {e}")
                        import traceback
                        traceback.print_exc()
        
        print(f"Total NEST connections created: {total_connections_created}")
        
        return total_connections_created







from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass



SYNAPSE_MODELS = {
    'static_synapse': {
        'weight': {'type': 'float', 'default': 1.0, 'unit': 'pA or nS'},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1, 'unit': 'ms'},
    },
    'stdp_synapse': {
        'weight': {'type': 'float', 'default': 1.0},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1},
        'tau_plus': {'type': 'float', 'default': 20.0, 'unit': 'ms'},
        'lambda': {'type': 'float', 'default': 0.01},
        'alpha': {'type': 'float', 'default': 1.0},
        'mu_plus': {'type': 'float', 'default': 1.0},
        'mu_minus': {'type': 'float', 'default': 1.0},
        'Wmax': {'type': 'float', 'default': 100.0},
    },
    'stdp_synapse_hom': {
        'weight': {'type': 'float', 'default': 1.0},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1},
        'tau_plus': {'type': 'float', 'default': 20.0},
        'lambda': {'type': 'float', 'default': 0.01},
        'alpha': {'type': 'float', 'default': 1.0},
        'mu_plus': {'type': 'float', 'default': 1.0},
        'mu_minus': {'type': 'float', 'default': 1.0},
        'Wmax': {'type': 'float', 'default': 100.0},
    },
    'tsodyks_synapse': {
        'weight': {'type': 'float', 'default': 1.0},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1},
        'U': {'type': 'float', 'default': 0.5, 'min': 0.0, 'max': 1.0},
        'tau_rec': {'type': 'float', 'default': 800.0, 'unit': 'ms'},
        'tau_fac': {'type': 'float', 'default': 0.0, 'unit': 'ms'},
    },
    'tsodyks2_synapse': {
        'weight': {'type': 'float', 'default': 1.0},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1},
        'U': {'type': 'float', 'default': 0.5},
        'u': {'type': 'float', 'default': 0.5},
        'x': {'type': 'float', 'default': 1.0},
        'tau_rec': {'type': 'float', 'default': 800.0},
        'tau_fac': {'type': 'float', 'default': 0.0},
    },
    'quantal_stp_synapse': {
        'weight': {'type': 'float', 'default': 1.0},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1},
        'U': {'type': 'float', 'default': 0.5},
        'tau_rec': {'type': 'float', 'default': 800.0},
        'tau_fac': {'type': 'float', 'default': 0.0},
        'n': {'type': 'int', 'default': 1},
    },
    'stdp_dopamine_synapse': {
        'weight': {'type': 'float', 'default': 1.0},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1},
        'tau_plus': {'type': 'float', 'default': 20.0},
        'tau_c': {'type': 'float', 'default': 1000.0},
        'tau_n': {'type': 'float', 'default': 200.0},
        'b': {'type': 'float', 'default': 0.0},
        'Wmin': {'type': 'float', 'default': 0.0},
        'Wmax': {'type': 'float', 'default': 200.0},
    },
    'ht_synapse': {
        'weight': {'type': 'float', 'default': 1.0},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1},
    },
    'bernoulli_synapse': {
        'weight': {'type': 'float', 'default': 1.0},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1},
        'p_transmit': {'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 1.0},
    },
    'cont_delay_synapse': {
        'weight': {'type': 'float', 'default': 1.0},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1},
    },
    'clopath_synapse': {
        'weight': {'type': 'float', 'default': 0.5},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1},
        'Wmax': {'type': 'float', 'default': 1.0},
        'Wmin': {'type': 'float', 'default': 0.0},
    },
    'vogels_sprekeler_synapse': {
        'weight': {'type': 'float', 'default': 1.0},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1},
        'tau': {'type': 'float', 'default': 20.0},
        'eta': {'type': 'float', 'default': 0.001},
        'alpha': {'type': 'float', 'default': 0.12},
        'Wmax': {'type': 'float', 'default': -1.0},
    },
    'gap_junction': {
        'weight': {'type': 'float', 'default': 1.0},
    },
    'rate_connection_instantaneous': {
        'weight': {'type': 'float', 'default': 1.0},
    },
    'rate_connection_delayed': {
        'weight': {'type': 'float', 'default': 1.0},
        'delay': {'type': 'float', 'default': 1.0, 'min': 0.1},
    },
    'diffusion_connection': {
        'weight': {'type': 'float', 'default': 1.0},
    },
}

CONNECTION_RULES = {
    'all_to_all': {
        'params': ['allow_autapses', 'allow_multapses'],
        'description': 'Connect all sources to all targets'
    },
    'one_to_one': {
        'params': [],
        'description': 'Connect source i to target i (requires equal sizes)'
    },
    'fixed_indegree': {
        'params': ['indegree', 'allow_autapses', 'allow_multapses'],
        'description': 'Each target receives fixed number of connections'
    },
    'fixed_outdegree': {
        'params': ['outdegree', 'allow_autapses', 'allow_multapses'],
        'description': 'Each source sends fixed number of connections'
    },
    'fixed_total_number': {
        'params': ['N', 'allow_autapses', 'allow_multapses'],
        'description': 'Create exactly N connections total'
    },
    'pairwise_bernoulli': {
        'params': ['p', 'allow_autapses', 'allow_multapses'],
        'description': 'Connect each pair with probability p'
    },
    'symmetric_pairwise_bernoulli': {
        'params': ['p', 'allow_autapses', 'allow_multapses'],
        'description': 'Like pairwise_bernoulli but symmetric'
    },
}



def create_connection_dict(
    connection_id: int,
    name: str,
    source_graph_id: int,
    source_node_id: int,
    source_pop_id: int,
    target_graph_id: int,
    target_node_id: int,
    target_pop_id: int,
    rule: str = 'all_to_all',
    synapse_model: str = 'static_synapse',
    weight: float = 1.0,
    delay: float = 1.0,
    allow_autapses: bool = False,
    allow_multapses: bool = True,
    **extra_params
) -> Dict[str, Any]:
    return {
        'id': connection_id,
        'name': name,
        'source': {
            'graph_id': source_graph_id,
            'node_id': source_node_id,
            'pop_id': source_pop_id
        },
        'target': {
            'graph_id': target_graph_id,
            'node_id': target_node_id,
            'pop_id': target_pop_id
        },
        'params': {
            'rule': rule,
            'synapse_model': synapse_model,
            'weight': weight,
            'delay': delay,
            'allow_autapses': allow_autapses,
            'allow_multapses': allow_multapses,
            **extra_params
        }
    }


def validate_delay(delay: float, resolution: float = None) -> float:
    if resolution is None:
        try:
            resolution = nest.resolution
        except:
            resolution = 0.1  
    
    if delay < resolution:
        print(f"⚠ Delay {delay}ms < resolution {resolution}ms, adjusting to {resolution}ms")
        return resolution
    return delay


def validate_connection_params(params: Dict[str, Any]) -> Tuple[Dict, Dict, List[str]]:
    warnings = []
    

    gui_keys = {
        'use_spatial', 
        'mask_type', 'mask_radius', 'mask_inner_radius', 'mask_size', 
        'distance_dependent_weight', 'dist_factor', 'dist_offset',
        'custom_name', 'base_model', 'topology_type' 
    }

    rule = params.get('rule', 'all_to_all')
    synapse_model = params.get('synapse_model', 'static_synapse')
    
    conn_spec = {'rule': rule}
    
    if rule in CONNECTION_RULES:
        rule_params = CONNECTION_RULES[rule]['params']
        for p in rule_params:
            if p in params:
                conn_spec[p] = params[p]
    
    if rule == 'fixed_indegree' and 'indegree' not in conn_spec: conn_spec['indegree'] = 1
    if rule == 'fixed_outdegree' and 'outdegree' not in conn_spec: conn_spec['outdegree'] = 1
    if rule == 'fixed_total_number' and 'N' not in conn_spec: conn_spec['N'] = 10
    if rule == 'pairwise_bernoulli' and 'p' not in conn_spec: conn_spec['p'] = 0.1
    
    conn_spec['allow_autapses'] = params.get('allow_autapses', True)
    conn_spec['allow_multapses'] = params.get('allow_multapses', True)
    
    syn_spec = {'synapse_model': synapse_model}
    syn_spec['weight'] = float(params.get('weight', 1.0))
    syn_spec['delay'] = validate_delay(float(params.get('delay', 1.0)))
    
    if 'receptor_type' in params and params['receptor_type'] != 0:
        syn_spec['receptor_type'] = int(params['receptor_type'])

    known_syn_params = set()
    if synapse_model in SYNAPSE_MODELS:
        model_params_def = SYNAPSE_MODELS[synapse_model].get('params', {})
        for param_name in model_params_def:
            known_syn_params.add(param_name)
            if param_name in params:
                syn_spec[param_name] = params[param_name]
    
    
    
    handled_keys = {'rule', 'synapse_model', 'weight', 'delay', 
                    'allow_autapses', 'allow_multapses', 'indegree', 
                    'outdegree', 'N', 'p', 'receptor_type'}
    
    for key, value in params.items():
        if key in handled_keys: continue
        if key in gui_keys: continue 
        if key in known_syn_params: continue 
        
        warnings.append(f"Ignored unknown parameter '{key}' (not in known synapse params)")
    
    return conn_spec, syn_spec, warnings


class ConnectionExecutor:

    
    def __init__(self, graphs: Dict[int, Any]):

        self.graphs = graphs
        self._connection_counter = 0
        self._created_models: List[str] = [] 
    
    def _get_next_connection_id(self) -> int:
        self._connection_counter += 1
        return self._connection_counter
    
    def _get_population(self, graph_id: int, node_id: int, pop_id: int):
        if graph_id not in self.graphs:
            return None, f"Graph {graph_id} not found"
        
        graph = self.graphs[graph_id]
        node = graph.get_node(node_id)
        
        if node is None:
            return None, f"Node {node_id} not in Graph {graph_id}"
        
        if not node.population:
            return None, f"Node {node_id} has no populations"
        
        if pop_id >= len(node.population):
            return None, f"Pop {pop_id} not in Node {node_id}"
        
        pop = node.population[pop_id]
        if pop is None or len(pop) == 0:
            return None, f"Pop {pop_id} is empty"
        
        return pop, None
    




    def execute_connection(self, connection: Dict[str, Any]) -> Tuple[bool, str]:
        # --- FIX: Import muss am Anfang stehen, um UnboundLocalError zu vermeiden ---
        import copy 
        import time
        # ----------------------------------------------------------------------------
        
        try:
            source_info = connection['source']
            target_info = connection['target']
            params = connection.get('params', {})
            conn_name = connection.get('name', f"Conn_{connection.get('id', '?')}")
            
            src_pop, err = self._get_population(
                source_info['graph_id'],
                source_info['node_id'],
                source_info['pop_id']
            )
            if err:
                return False, f"Source: {err}"
            
            tgt_pop, err = self._get_population(
                target_info['graph_id'],
                target_info['node_id'],
                target_info['pop_id']
            )
            if err:
                return False, f"Target: {err}"
            
            topo_type = params.get('topology_type')
            
            # --- SPECIAL TOPOLOGY: CCW RING ---
            if topo_type == 'ring_ccw':
                src_ids = nest.GetStatus(src_pop, 'global_id')
                tgt_ids = nest.GetStatus(tgt_pop, 'global_id')
                
                n = len(src_ids)
                if len(tgt_ids) != n:
                    return False, f"CCW Ring requires equal size (Src: {n}, Tgt: {len(tgt_ids)})"
                
                w = float(params.get('weight', 10.0))
                d = float(params.get('delay', 1.0))
                
                pre_neurons = []
                post_neurons = []
                
                for i in range(n):
                    pre = src_ids[i]
                    post = tgt_ids[(i + 1) % n] 
                    pre_neurons.append(pre)
                    post_neurons.append(post)
                
                nest.Connect(pre_neurons, post_neurons, 
                             {'rule': 'one_to_one'}, 
                             {'weight': w, 'delay': d, 'synapse_model': 'static_synapse'})

                try:
                    src_graph = self.graphs.get(source_info['graph_id'])
                    if src_graph:
                        src_node_obj = src_graph.get_node(source_info['node_id'])
                        if src_node_obj:
                            if not hasattr(src_node_obj, 'connections'):
                                src_node_obj.connections = []
                            
                            exists = False
                            for existing in src_node_obj.connections:
                                if existing.get('id') == connection.get('id'):
                                    exists = True; break
                            
                            if not exists:
                                src_node_obj.connections.append(copy.deepcopy(connection))
                except Exception as e:
                    print(f"Warning: Could not save ring connection to model: {e}")
                
                return True, f"✓ CCW Ring created ({n} connections)"

            # --- STANDARD VERBINDUNGEN ---
            
            # HT Neuron Receptor Mapping Fix
            try:
                target_model = nest.GetStatus(tgt_pop, 'model')[0]
            except:
                target_model = 'unknown'
            
            if 'ht_neuron' in str(target_model):
                rec_map = {1: 'AMPA', 2: 'NMDA', 3: 'GABA_A', 4: 'GABA_B'}
                current_rec = params.get('receptor_type', 0)
                if isinstance(current_rec, int) and current_rec in rec_map:
                    params['receptor_type'] = rec_map[current_rec]
                elif current_rec == 0:
                    params.pop('receptor_type', None)
            
            # Regel Parameter
            rule = params.get('rule', 'all_to_all')
            conn_spec = {'rule': rule}
            
            if rule == 'fixed_indegree':
                conn_spec['indegree'] = int(params.get('indegree', 1))
            elif rule == 'fixed_outdegree':
                conn_spec['outdegree'] = int(params.get('outdegree', 1))
            elif rule == 'fixed_total_number':
                conn_spec['N'] = int(params.get('N', 1))
            elif 'bernoulli' in rule:
                conn_spec['p'] = float(params.get('p', 0.1))
            
            conn_spec['allow_autapses'] = params.get('allow_autapses', True)
            conn_spec['allow_multapses'] = params.get('allow_multapses', True)
            
            # Spatial Mask
            if params.get('use_spatial', False):
                mask_type = params.get('mask_type', 'spherical')
                mask_params = {
                    'radius': params.get('mask_radius', 1.0),
                    'r': params.get('mask_radius', 1.0),
                    'inner_radius': params.get('mask_inner_radius', 0.0),
                    'outer_radius': params.get('mask_radius', 1.0),
                    'size': params.get('mask_radius', 1.0)
                }
                
                mask_type_map = {
                    'sphere': 'spherical', 'spherical': 'spherical',
                    'box': 'rectangular', 'rectangular': 'rectangular',
                    'doughnut': 'doughnut'
                }
                actual_type = mask_type_map.get(mask_type, 'spherical')
                mask = create_nest_mask(actual_type, mask_params)
                if mask is not None:
                    conn_spec['mask'] = mask
            
            # Synapse Specification
            base_model = params.get('synapse_model', 'static_synapse')
            no_delay = base_model in ['gap_junction', 'rate_connection_instantaneous']
            
            control_keys = {
                'rule', 'indegree', 'outdegree', 'N', 'p',
                'synapse_model', 'weight', 'delay',
                'allow_autapses', 'allow_multapses', 'receptor_type',
                'use_spatial', 'mask_type', 'mask_radius', 'mask_inner_radius',
                'distance_dependent_weight', 'dist_factor', 'dist_offset',
                'custom_name', 'no_delay', 'base_model'
            }
            
            model_params = {
                k: v for k, v in params.items() 
                if k not in control_keys and v is not None
            }
            
            final_model = base_model
            if model_params:
                ts = int(time.time() * 1e6)
                custom_name = params.get('custom_name') or f"{base_model}_{self._connection_counter}_{ts}"
                try:
                    nest.CopyModel(base_model, custom_name, model_params)
                    final_model = custom_name
                    self._created_models.append(custom_name)
                except Exception as e:
                    print(f"CopyModel failed: {e}, using {base_model}")
            
            syn_spec = {'synapse_model': final_model}
            
            weight = float(params.get('weight', 1.0))
            if params.get('use_spatial') and params.get('distance_dependent_weight'):
                factor = float(params.get('dist_factor', 1.0))
                offset = float(params.get('dist_offset', 0.0))
                syn_spec['weight'] = create_distance_dependent_weight(weight, factor, offset)
            else:
                syn_spec['weight'] = weight
            
            if not no_delay:
                delay = float(params.get('delay', 1.0))
                syn_spec['delay'] = max(delay, nest.resolution)
            
            receptor = params.get('receptor_type')
            if receptor is not None and receptor != 0:
                syn_spec['receptor_type'] = receptor
            
            if rule == 'one_to_one' and len(src_pop) != len(tgt_pop):
                return False, f"one_to_one size mismatch: {len(src_pop)} vs {len(tgt_pop)}"
            
            # --- EXECUTE NEST CONNECT ---
            nest.Connect(src_pop, tgt_pop, conn_spec, syn_spec)
            
            # --- SAVE TO INTERNAL MODEL ---
            try:
                src_graph = self.graphs.get(source_info['graph_id'])
                tgt_graph = self.graphs.get(target_info['graph_id']) 

                if src_graph and tgt_graph:
                    src_node_obj = src_graph.get_node(source_info['node_id'])
                    tgt_node_obj = tgt_graph.get_node(target_info['node_id']) 

                    if src_node_obj and tgt_node_obj:
                        
                        # 1. Save connection dict
                        if not hasattr(src_node_obj, 'connections'):
                            src_node_obj.connections = []
                        
                        exists = False
                        for existing in src_node_obj.connections:
                            if existing.get('id') == connection.get('id'):
                                exists = True
                                break
                        
                        if not exists:
                            # HIER WAR DER FEHLER: copy war lokal gebunden aber unassigned
                            src_node_obj.connections.append(copy.deepcopy(connection))
                            
                        # 2. Update Topology
                        src_node_obj.add_neighbor(tgt_node_obj)
                    
                return True, f"✓ {conn_name} created"

            except Exception as save_err:
                # Fallback, falls speichern fehlschlägt, aber NEST erfolgreich war
                print(f"Warning: Connection created in NEST but failed to save to Model: {save_err}")
                return True, f"✓ {conn_name} (Model save failed)"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)



    def execute_pending_connections(self, connections: List[Dict[str, Any]]) -> Tuple[int, int, List[Dict[str, Any]]]:

        success_count = 0
        fail_count = 0
        failed_items = []

        print(f"Processing {len(connections)} pending connections...")

        for conn in connections:
            success, msg = self.execute_connection(conn)
            
            if success:
                success_count += 1
                print(f"  {msg}")
            else:
                fail_count += 1
                conn['error'] = msg
                failed_items.append(conn)
                print(f"  ✗ {conn.get('name', '?')}: {msg}")

        return success_count, fail_count, failed_items
    
    def execute_all(self, connections: List[Dict[str, Any]], 
                   progress_callback=None) -> Tuple[int, int, List[str]]:
        
        created = 0
        failed = 0
        errors = []
        total = len(connections)
        
        print(f"\nCreating {total} connections...")
        
        for i, conn in enumerate(connections):
            success, msg = self.execute_connection(conn)
            
            if success:
                created += 1
                print(f"  {msg}")
            else:
                failed += 1
                err_msg = f"✗ {conn.get('name', '?')}: {msg}"
                errors.append(err_msg)
                print(f"  {err_msg}")
            
            if progress_callback:
                progress_callback(i + 1, total, msg)
        
        print(f"\nConnections: {created} created, {failed} failed")
        
        return created, failed, errors
    
    def cleanup_custom_models(self):

        self._created_models.clear()


class ConnectionQueueWidget(QWidget):
    
    connectionSelected = pyqtSignal(int) 
    connectionRemoved = pyqtSignal(int)  
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.connections: List[Dict] = []
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("Connection Queue")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)
        
        # List
        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.remove_btn = QPushButton("🗑 Remove")
        self.remove_btn.clicked.connect(self._remove_selected)
        self.remove_btn.setEnabled(False)
        btn_layout.addWidget(self.remove_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all)
        btn_layout.addWidget(clear_btn)
        
        layout.addLayout(btn_layout)
        
        self.status_label = QLabel("0 connections")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)
    
    def add_connection(self, conn: Dict):
        """Fügt eine Verbindung hinzu."""
        self.connections.append(conn)
        
        name = conn.get('name', f"Connection {len(self.connections)}")
        src = conn.get('source', {})
        tgt = conn.get('target', {})
        
        label = f"{name}: G{src.get('graph_id',0)}N{src.get('node_id',0)}P{src.get('pop_id',0)} → " \
                f"G{tgt.get('graph_id',0)}N{tgt.get('node_id',0)}P{tgt.get('pop_id',0)}"
        
        item = QListWidgetItem(label)
        
        syn = conn.get('params', {}).get('synapse_model', 'static')
        if 'stdp' in syn:
            item.setBackground(QColor(100, 50, 150, 50))
        elif 'gap' in syn:
            item.setBackground(QColor(50, 150, 100, 50))
        elif 'tsodyks' in syn or 'stp' in syn:
            item.setBackground(QColor(150, 100, 50, 50))
        
        self.list_widget.addItem(item)
        self._update_status()
    
    def _on_item_clicked(self, item):
        idx = self.list_widget.row(item)
        self.remove_btn.setEnabled(True)
        self.connectionSelected.emit(idx)
    
    def _remove_selected(self):
        idx = self.list_widget.currentRow()
        if idx >= 0:
            self.list_widget.takeItem(idx)
            del self.connections[idx]
            self.connectionRemoved.emit(idx)
            self._update_status()
        self.remove_btn.setEnabled(False)
    
    def clear_all(self):
        self.list_widget.clear()
        self.connections.clear()
        self._update_status()
    
    def get_all_connections(self) -> List[Dict]:
        return self.connections.copy()
    
    def _update_status(self):
        n = len(self.connections)
        self.status_label.setText(f"{n} connection{'s' if n != 1 else ''}")









def create_nest_connections_from_stored(
    graphs: Dict[int, Any],
    verbose: bool = True
) -> Tuple[int, int, List[Dict]]:
    
    total_created = 0
    total_failed = 0
    failed_connections = []
    
    if verbose:
        print("\n" + "="*60)
        print("RECREATING ALL STORED CONNECTIONS")
        print("="*60)
    
    all_connections = []
    
    for graph_id, graph in graphs.items():
        if verbose:
            print(f"\nGraph {graph_id} ({graph.graph_name}):")
        
        for node in graph.node_list:
            if node.connections:
                if verbose:
                    print(f"Node {node.id} ({node.name}): {len(node.connections)} connections")
                
                for conn in node.connections:
                    all_connections.append(conn)
    
    if not all_connections:
        if verbose:
            print("\nNo stored connections found")
        return 0, 0, []
    
    if verbose:
        print(f"\nTotal connections to recreate: {len(all_connections)}")
    
    for conn in all_connections:
        conn_name = conn.get('name', f"Connection_{conn['id']}")
        source = conn['source']
        target = conn['target']
        params = conn['params']
        
        try:
            source_graph = graphs.get(source['graph_id'])
            target_graph = graphs.get(target['graph_id'])
            
            if source_graph is None:
                raise ValueError(f"Source graph {source['graph_id']} not found")
            if target_graph is None:
                raise ValueError(f"Target graph {target['graph_id']} not found")
            
            source_node = source_graph.get_node(source['node_id'])
            target_node = target_graph.get_node(target['node_id'])
            
            if source_node is None:
                raise ValueError(f"Source node {source['node_id']} not found")
            if target_node is None:
                raise ValueError(f"Target node {target['node_id']} not found")
            
            if not source_node.population or source['pop_id'] >= len(source_node.population):
                raise ValueError(f"Source population {source['pop_id']} not available")
            if not target_node.population or target['pop_id'] >= len(target_node.population):
                raise ValueError(f"Target population {target['pop_id']} not available")
            
            source_pop = source_node.population[source['pop_id']]
            target_pop = target_node.population[target['pop_id']]
            
            if source_pop is None or len(source_pop) == 0:
                raise ValueError("Source population is empty")
            if target_pop is None or len(target_pop) == 0:
                raise ValueError("Target population is empty")
            
            conn_spec, syn_spec, warnings = validate_connection_params(params)
            
            if conn_spec['rule'] == 'one_to_one' and len(source_pop) != len(target_pop):
                raise ValueError(f"one_to_one: size mismatch ({len(source_pop)} vs {len(target_pop)})")
            
            nest.Connect(source_pop, target_pop, conn_spec, syn_spec)
            
            total_created += 1
            if verbose:
                print(f"  ✓ {conn_name}: G{source['graph_id']}N{source['node_id']}P{source['pop_id']} → "
                      f"G{target['graph_id']}N{target['node_id']}P{target['pop_id']}")
        
        except Exception as e:
            total_failed += 1
            error_msg = str(e)
            failed_connections.append({**conn, 'error': error_msg})
            if verbose:
                print(f"  ✗ {conn_name}: {error_msg}")
    
    if verbose:
        print(f"✓ Created: {total_created} | ✗ Failed: {total_failed}")
    
    return total_created, total_failed, failed_connections


def repopulate_all_graphs(graphs: Dict[int, Any], verbose: bool = False) -> int:

    total_pops = 0
    
    if verbose:
        print("REPOPULATING ALL GRAPHS")
    
    for graph_id, graph in graphs.items():
        if verbose:
            print(f"\nGraph {graph_id} ({graph.graph_name}):")
        
        for node in graph.node_list:
            try:
                if not node.positions or all(len(c) == 0 for c in node.positions):
                    if verbose:
                        print(f"Building Node {node.id}...")
                    node.build()
                
                if verbose:
                    print(f"Populating Node {node.id}...")
                node.populate_node()
                
                n_pops = len(node.population) if node.population else 0
                total_pops += n_pops
                
                if verbose:
                    print(f"    ✓ {n_pops} populations created")
            
            except Exception as e:
                if verbose:
                    print(f"    ✗ Error: {e}")
    
    if verbose:
        print(f"\n✓ Total populations created: {total_pops}")
    
    return total_pops


def safe_nest_reset_and_repopulate(
    graphs: Dict[int, Any],
    enable_structural_plasticity: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    
    if verbose:
        print("# SAFE NEST RESET AND REPOPULATE")
    
    global _nest_simulation_has_run
    if verbose:
        print("\nResetting NEST kernel...")
    nest.ResetKernel()
    _nest_simulation_has_run = False 
    
    if enable_structural_plasticity:
        try:
            nest.EnableStructuralPlasticity()
            if verbose:
                print("✓ Structural plasticity enabled")
        except Exception as e:
            if verbose:
                print(f"Could not enable structural plasticity: {e}")
            enable_structural_plasticity = False
    
    total_pops = repopulate_all_graphs(graphs, verbose=verbose)
    
    created, failed, failed_list = create_nest_connections_from_stored(graphs, verbose=verbose)
    
    result = {
        'populations_created': total_pops,
        'connections_created': created,
        'connections_failed': failed,
        'failed_connections': failed_list,
        'structural_plasticity': enable_structural_plasticity
    }
    
    if verbose:
        print("# RESET COMPLETE")
        print(f"# Populations: {total_pops}")
        print(f"# Connections: {created} created, {failed} failed")
    
    return result


def get_all_connections_summary(graphs: Dict[int, Any]) -> Dict[str, Any]:

    summary = {
        'total_connections': 0,
        'by_graph': {},
        'by_synapse_model': {},
        'by_rule': {}
    }
    
    for graph_id, graph in graphs.items():
        graph_conns = 0
        
        for node in graph.node_list:
            n_conns = len(node.connections) if node.connections else 0
            graph_conns += n_conns
            
            for conn in (node.connections or []):
                params = conn.get('params', {})
                
                model = params.get('synapse_model', 'static_synapse')
                summary['by_synapse_model'][model] = summary['by_synapse_model'].get(model, 0) + 1
                
                rule = params.get('rule', 'all_to_all')
                summary['by_rule'][rule] = summary['by_rule'].get(rule, 0) + 1
        
        summary['by_graph'][graph_id] = graph_conns
        summary['total_connections'] += graph_conns
    
    return summary


def clear_all_connections(graphs: Dict[int, Any], verbose: bool = True) -> int:
    
    total_cleared = 0
    
    for graph_id, graph in graphs.items():
        for node in graph.node_list:
            if node.connections:
                n = len(node.connections)
                node.connections = []
                total_cleared += n
    
    if verbose:
        print(f"✓ Cleared {total_cleared} connections from all graphs")
    
    return total_cleared


def export_connections_to_dict(graphs: Dict[int, Any]) -> List[Dict]:
    
    all_connections = []
    
    for graph_id, graph in graphs.items():
        for node in graph.node_list:
            for conn in (node.connections or []):
                all_connections.append(copy.deepcopy(conn))
    
    return all_connections


def import_connections_from_dict(
    graphs: Dict[int, Any],
    connections: List[Dict],
    clear_existing: bool = True
) -> int:
    
    if clear_existing:
        clear_all_connections(graphs, verbose=False)
    
    imported = 0
    
    for conn in connections:
        source = conn.get('source', {})
        graph_id = source.get('graph_id')
        node_id = source.get('node_id')
        
        if graph_id in graphs:
            node = graphs[graph_id].get_node(node_id)
            if node is not None:
                node.connections.append(copy.deepcopy(conn))
                imported += 1
    
    return imported


def create_connection_from_gui_params(
    source_graph_id: int,
    source_node_id: int,
    source_pop_id: int,
    target_graph_id: int,
    target_node_id: int,
    target_pop_id: int,
    rule: str,
    synapse_model: str,
    weight: float,
    delay: float,
    allow_autapses: bool = False,
    allow_multapses: bool = True,
    connection_name: str = None,
    indegree: int = None,
    outdegree: int = None,
    N: int = None,
    p: float = None,
    **synapse_params
) -> Dict[str, Any]:
    
    params = {
        'rule': rule,
        'synapse_model': synapse_model,
        'weight': weight,
        'delay': delay,
        'allow_autapses': allow_autapses,
        'allow_multapses': allow_multapses,
    }
    
    if indegree is not None:
        params['indegree'] = indegree
    if outdegree is not None:
        params['outdegree'] = outdegree
    if N is not None:
        params['N'] = N
    if p is not None:
        params['p'] = p
    
    params.update(synapse_params)
    
    return {
        'source_graph_id': source_graph_id,
        'source_node_id': source_node_id,
        'source_pop_id': source_pop_id,
        'target_graph_id': target_graph_id,
        'target_node_id': target_node_id,
        'target_pop_id': target_pop_id,
        'params': params,
        'name': connection_name
    }








def validate_SYNAPSE_MODELS():
    
    nest_synapses = set(nest.SYNAPSE_MODELS)
    
    
    invalid = []
    valid = []
    
    for model in SYNAPSE_MODELS.keys():
        if model in nest_synapses:
            valid.append(model)
            print(f"{model}")
        else:
            invalid.append(model)
            print(f"{model} - NOT A VALID SYNAPSE MODEL!")
    
    print(f"\nValid: {len(valid)} | Invalid: {len(invalid)}")
    
    if invalid:
        print("\nREMOVE THESE FROM SYNAPSE_MODELS:")
        for m in invalid:
            print(f'    "{m}": {{}},')
    
    return invalid











def generate_node_parameters_list(n_nodes=5, 
                                   n_types=5, 
                                   vary_polynoms=True,
                                   vary_types_per_node=True,
                                   safe_mode=True,      
                                   max_power=2,
                                   max_coeff=0.8,
                                   graph_id=0,
                                   add_self_connections=False, 
                                   self_conn_probability=0.3): 
   
    params_list = []
    
    for i in range(n_nodes):
        if vary_types_per_node:
            node_n_types = np.random.randint(1, n_types + 1)
        else:
            node_n_types = n_types
        
        types = list(range(node_n_types))
        
        available_models = successful_neuron_models
        neuron_models = [available_models[t % len(available_models)] 
                        for t in range(node_n_types)]
        
        probability_vector = list(np.random.dirichlet([1] * node_n_types))
        
        params = {
            "types": types,
            "neuron_models": neuron_models,
            "grid_size": [
                np.random.randint(8, 15),
                np.random.randint(8, 15),
                np.random.randint(8, 15)
            ],
            "m": [
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.0, 1.0)
            ],
            "rot_theta": np.random.uniform(-np.pi, np.pi),
            "rot_phi": np.random.uniform(-np.pi, np.pi),
            "transform_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "dt": np.random.uniform(0.005, 0.02),
            "old": True,
            "num_steps": np.random.randint(5, 12),
            "plot_clusters": False,
            "title": f"Node_{i}",
            "sparse_holes": np.random.randint(0, 3),
            "sparsity_factor": np.random.uniform(0.85, 0.95),
            "probability_vector": probability_vector,
            "name": f"Node_{i}",
            "id": i,
            "graph_id": graph_id,
            "distribution": [],
            "conn_prob": [],
            "polynom_max_power": 5,
            "coefficients": None,
            "center_of_mass": np.array([0.0, 0.0, 0.0]),
            "displacement": np.array([0.0, 0.0, 0.0]),
            "displacement_factor": 1.0,
            "field": None,
            "population_nest_params": [],
            "connections": [], 
        }
        
        if add_self_connections and np.random.random() < self_conn_probability:
            for pop_id in range(node_n_types):
                self_conn = {
                    'id': len(params['connections']) + 1,
                    'name': f'SelfConn_N{i}_P{pop_id}',
                    'source': {
                        'graph_id': graph_id,
                        'node_id': i,
                        'pop_id': pop_id
                    },
                    'target': {
                        'graph_id': graph_id,
                        'node_id': i, 
                        'pop_id': pop_id
                    },
                    'params': {
                        'rule': 'fixed_indegree',
                        'indegree': np.random.randint(1, 5),
                        'synapse_model': 'static_synapse',
                        'weight': np.random.uniform(0.5, 2.0),
                        'delay': 1.0,
                        'allow_autapses': False,  
                        'allow_multapses': True
                    }
                }
                params['connections'].append(self_conn)
        
        if vary_polynoms:
            encoded_polynoms = []
            
            for type_idx in range(node_n_types): 
                type_polynoms = []
                
                for coord in range(3):
                    num_terms = np.random.randint(2, 5)
                    
                    indices = []
                    coeffs = []
                    
                    for _ in range(num_terms):
                        idx = np.random.choice([0, 1, 2])
                        
                        if safe_mode:
                            power = np.random.choice(range(min(max_power + 1, 4)))
                            coeff = np.random.uniform(-max_coeff, max_coeff)
                        else:
                            power = np.random.choice([0, 1, 2, 3])
                            coeff = np.random.randn() * 0.5
                        
                        indices.append([idx, power])
                        coeffs.append(float(coeff))
                    
                    poly_encoded = {
                        'indices': indices,
                        'coefficients': coeffs,
                        'n': 5,
                        'decay': 0.5
                    }
                    type_polynoms.append(poly_encoded)
                
                encoded_polynoms.append(type_polynoms)
            
            params["encoded_polynoms_per_type"] = encoded_polynoms
        else:
            identity_polynoms = []
            for type_idx in range(node_n_types):
                type_polynoms = [
                    {'indices': [[0, 1]], 'coefficients': [1.0], 'n': 5, 'decay': 0.5},
                    {'indices': [[1, 1]], 'coefficients': [1.0], 'n': 5, 'decay': 0.5},
                    {'indices': [[2, 1]], 'coefficients': [1.0], 'n': 5, 'decay': 0.5}
                ]
                identity_polynoms.append(type_polynoms)
            
            params["encoded_polynoms_per_type"] = identity_polynoms
        
        params_list.append(params)
    
    return params_list

class SaveLoadWidget(QWidget):
    
    def __init__(self, main_window, Graph_class=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.Graph_class = Graph_class
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Save Button
        self.save_btn = QPushButton("Save Graph")
        self.save_btn.setMinimumWidth(100)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
        """)
        self.save_btn.clicked.connect(self.save_graph_dialog)
        
        self.load_btn = QPushButton("Load Graph")
        self.load_btn.setMinimumWidth(100)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #1565C0; }
        """)
        self.load_btn.clicked.connect(self.load_graph_dialog)
        
        layout.addWidget(self.save_btn)
        layout.addWidget(self.load_btn)
        layout.addStretch()

    def save_graph_dialog(self):
        if not hasattr(self.main_window, 'graph_list') or not self.main_window.graph_list:
            QMessageBox.warning(self, "No Graph", "no graph available")
            return
        
        current_idx = getattr(self.main_window, 'current_graph_idx', -1)
        if current_idx < 0 or current_idx >= len(self.main_window.graph_list):
            current_idx = len(self.main_window.graph_list) - 1
        
        graph = self.main_window.graph_list[current_idx]
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Graph speichern",
            f"{getattr(graph, 'graph_name', 'graph')}.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filepath:
            if not filepath.endswith('.json'):
                filepath += '.json'
            
            success = save_graph(graph, filepath)
            
            if success:
                QMessageBox.information(self, "Gespeichert", 
                    f"Graph '{graph.graph_name}' erfolgreich gespeichert!")
            else:
                QMessageBox.critical(self, "Fehler", 
                    "Fehler beim Speichern des Graphs!")

    def load_graph_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Graph", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not filepath:
            return
        
        GraphClass = self.Graph_class
        if GraphClass is None and hasattr(self.main_window, 'Graph'):
            GraphClass = self.main_window.Graph
        
        if GraphClass is None:
            GraphClass = Graph 
        
        if GraphClass is None:
            QMessageBox.critical(self, "Error", "Graph type not available")
            return
        
        graph = load_graph(filepath, GraphClass)
        
        if graph:
            if hasattr(self.main_window, 'graph_list'):
                self.main_window.graph_list.append(graph)
            
            if hasattr(self.main_window, 'update_visualizations'):
                self.main_window.update_visualizations()
            if hasattr(self.main_window, 'graph_overview'):
                self.main_window.graph_overview.update_tree()
            
            QMessageBox.information(self, "Loaded", 
                f"Graph '{graph.graph_name}' loaded.\nNodes: {len(graph.node_list)}")
        else:
            QMessageBox.critical(self, "Error", "Loading Error!")

def add_save_load_buttons(main_window, Graph_class=None, target_layout=None):
    
    widget = SaveLoadWidget(main_window, Graph_class)
    
    if target_layout:
        target_layout.addWidget(widget)
    elif hasattr(main_window, 'statusBar'):
        main_window.statusBar().addPermanentWidget(widget)
    else:
        widget.setParent(main_window)
        widget.move(main_window.width() - 260, main_window.height() - 60)
        widget.show()
    
    main_window.save_load_widget = widget
    print("Save/Load Buttons hinzugefügt")
    return widget

class SimulationControlWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)  
        
        # Styles
        btn_style = """
            QPushButton {
                border: none;
                border-radius: 0px; /* Eckig, um Lücken zu füllen */
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { border: 2px solid rgba(255,255,255,0.3); }
        """
        
        start_style = btn_style + "QPushButton { background-color: #4CAF50; color: white; }"
        stop_style = btn_style + "QPushButton { background-color: #F44336; color: white; }"
        io_style = btn_style + "QPushButton { background-color: #2196F3; color: white; }"
        
        self.btn_start = QPushButton("Simulate")
        self.btn_start.setStyleSheet(start_style)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet(stop_style)
        
        self.btn_save = QPushButton("Save Graphs")
        self.btn_save.setStyleSheet(io_style)
    
        self.btn_load = QPushButton("Load Graphs")
        self.btn_load.setStyleSheet(io_style)
        
        for btn in [self.btn_start, self.btn_stop, self.btn_save, self.btn_load]:
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_load)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return super().default(obj)

def _clean_params(params: dict) -> dict:
    clean = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (np.integer, np.int64)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            clean[k] = float(v)
        elif isinstance(v, dict):
            clean[k] = _clean_params(v)
        elif isinstance(v, list):
            clean_list = []
            for x in v:
                if isinstance(x, np.ndarray):
                    clean_list.append(x.tolist())
                elif isinstance(x, (np.floating, np.float64)):
                    clean_list.append(float(x))
                elif isinstance(x, (np.integer, np.int64)):
                    clean_list.append(int(x))
                elif isinstance(x, dict):
                    clean_list.append(_clean_params(x))
                else:
                    clean_list.append(x)
            clean[k] = clean_list
        else:
            clean[k] = v
    return clean

def _serialize_connections(connections: list) -> list:
    if not connections:
        return []
    result = []
    for conn in connections:
        conn_data = {
            'id': conn.get('id'),
            'name': conn.get('name'),
            'source': conn.get('source'),
            'target': conn.get('target'),
            'params': _clean_params(conn.get('params', {}))
        }
        result.append(conn_data)
    return result

def save_graph(graph, filepath: str) -> bool:
    try:
        data = {
            'meta': {
                'version': '1.0',
                'saved_at': datetime.now().isoformat(),
            },
            'graph': {
                'graph_id': graph.graph_id,
                'graph_name': graph.graph_name,
                'max_nodes': graph.max_nodes,
                'init_position': list(graph.init_position) if hasattr(graph, 'init_position') else [0,0,0],
                'polynom_max_power': graph.polynom_max_power if hasattr(graph, 'polynom_max_power') else 5,
                'polynom_decay': graph.polynom_decay if hasattr(graph, 'polynom_decay') else 0.8,
            },
            'nodes': []
        }
        
        for node in graph.node_list:
            cleaned_devices = []
            
            source_devices = getattr(node, 'devices', [])
            if not source_devices and hasattr(node, 'parameters') and 'devices' in node.parameters:
                source_devices = node.parameters['devices']

            for dev in source_devices:
                dev_copy = dev.copy()
                if 'runtime_gid' in dev_copy:
                    del dev_copy['runtime_gid']
                if 'params' in dev_copy:
                    dev_copy['params'] = _clean_params(dev_copy['params'])
                cleaned_devices.append(dev_copy)
            
            safe_params = _clean_params(node.parameters) if hasattr(node, 'parameters') else {}
            
            if 'devices' in safe_params:
                del safe_params['devices']

            node_data = {
                'id': node.id,
                'name': node.name,
                'graph_id': node.graph_id if hasattr(node, 'graph_id') else 0,
                'parameters': safe_params,
                'center_of_mass': list(node.center_of_mass) if hasattr(node, 'center_of_mass') else [0,0,0],
                'types': node.types if hasattr(node, 'types') else [],
                'devices': cleaned_devices, 
                'connections': _serialize_connections(node.connections) if hasattr(node, 'connections') else [],
                'neuron_models': node.neuron_models if hasattr(node, 'neuron_models') else [],
                'distribution': list(node.distribution) if hasattr(node, 'distribution') and node.distribution else [],
                'parent_id': node.parent.id if hasattr(node, 'parent') and node.parent else None,
                'next_ids': [n.id for n in node.next] if hasattr(node, 'next') else [],
                'prev_ids': [n.id for n in node.prev] if hasattr(node, 'prev') else [],
                'positions': [
                    pos.tolist() if isinstance(pos, np.ndarray) else list(pos)
                    for pos in node.positions
                ] if hasattr(node, 'positions') and node.positions else []
            }
            data['nodes'].append(node_data)
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2)
        
        print(f"Saved: {graph.graph_name} ({len(graph.node_list)} nodes) → {filepath}")
        return True
        
    except Exception as e:
        print(f"Save failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_graph(filepath: str, Graph_class):
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        g_data = data['graph']
        nodes_data = sorted(data['nodes'], key=lambda x: x['id'])
        
        print(f"Loading: {g_data['graph_name']} ({len(nodes_data)} nodes)...")
        
        graph = Graph_class(
            graph_name=g_data['graph_name'],
            graph_id=g_data['graph_id'],
            parameter_list=[],
            polynom_max_power=g_data.get('polynom_max_power', 5),
            polynom_decay=g_data.get('polynom_decay', 0.8),
            position=g_data.get('init_position', [0, 0, 0]),
            max_nodes=g_data.get('max_nodes', len(nodes_data))
        )
        
        for nd in nodes_data:
            params = nd.get('parameters', {}).copy()
            params['id'] = nd['id']
            params['name'] = nd['name']
            params['graph_id'] = nd.get('graph_id', 0)
            params['connections'] = nd.get('connections', [])
            
            if 'devices' in nd:
                params['devices'] = nd['devices']

            if 'center_of_mass' in nd:
                params['center_of_mass'] = np.array(nd['center_of_mass'])
            
            new_node = graph.create_node(
                parameters=params,
                is_root=(nd['id']==0), 
                auto_build=False
            )
            
            if nd.get('positions'):
                new_node.positions = [np.array(pos) for pos in nd['positions']]
                new_node.center_of_mass = np.array(nd['center_of_mass'])
            
            new_node.populate_node()

        for nd in nodes_data:
            node = graph.get_node(nd['id'])
            if not node:
                continue
            
            for next_id in nd.get('next_ids', []):
                next_node = graph.get_node(next_id)
                if next_node and next_node not in node.next:
                    node.next.append(next_node)
            
            for prev_id in nd.get('prev_ids', []):
                prev_node = graph.get_node(prev_id)
                if prev_node and prev_node not in node.prev:
                    node.prev.append(prev_node)
        
        print(f"Loaded: {graph.graph_name} ({len(graph.node_list)} nodes)")
        return graph
        
    except Exception as e:
        print(f"Load failed: {e}")
        import traceback
        traceback.print_exc()
        return None

class ConnectionParamWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.current_conn_data = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("EDIT CONNECTION")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #673AB7; padding: 5px; border-bottom: 2px solid #673AB7;")
        layout.addWidget(header)
        
        form_layout = QFormLayout()
        
        # Name
        self.name_input = QLineEdit()
        form_layout.addRow("Name:", self.name_input)
        
        # Weight
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(-10000.0, 10000.0)
        self.weight_spin.setDecimals(3)
        form_layout.addRow("Weight:", self.weight_spin)
        
        # Delay
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 1000.0)
        self.delay_spin.setDecimals(2)
        self.delay_spin.setSuffix(" ms")
        form_layout.addRow("Delay:", self.delay_spin)
        
        # Rule
        self.rule_lbl = QLabel("-")
        form_layout.addRow("Rule:", self.rule_lbl)
        
        # Dynamic Params (p, N, indegree)
        self.dyn_label = QLabel("Parameter:")
        self.dyn_spin = QDoubleSpinBox() 
        self.dyn_spin.setRange(0, 1000000)
        self.dyn_spin.setDecimals(4)
        self.dyn_spin.setVisible(False)
        form_layout.addRow(self.dyn_label, self.dyn_spin)
        
        # Synapse Model
        self.syn_lbl = QLabel("-")
        form_layout.addRow("Model:", self.syn_lbl)
        
        layout.addLayout(form_layout)
        
        # Apply Button
        self.btn_apply = QPushButton("Update Connection Params")
        self.btn_apply.setStyleSheet("background-color: #673AB7; color: white; font-weight: bold; padding: 10px;")
        self.btn_apply.clicked.connect(self.apply_changes)
        layout.addWidget(self.btn_apply)
        
        layout.addStretch()
        
    def load_data(self, conn_data):
        self.current_conn_data = conn_data
        params = conn_data.get('params', {})
        
        self.name_input.setText(conn_data.get('name', ''))
        self.weight_spin.setValue(params.get('weight', 1.0))
        self.delay_spin.setValue(params.get('delay', 1.0))
        
        rule = params.get('rule', 'all_to_all')
        self.rule_lbl.setText(rule)
        self.syn_lbl.setText(params.get('synapse_model', 'static_synapse'))
        
        self.dyn_spin.setVisible(False)
        self.dyn_label.setText("")
        
        if rule == 'fixed_indegree' and 'indegree' in params:
            self.dyn_label.setText("Indegree:")
            self.dyn_spin.setDecimals(0)
            self.dyn_spin.setValue(params['indegree'])
            self.dyn_spin.setVisible(True)
        elif rule == 'fixed_outdegree' and 'outdegree' in params:
            self.dyn_label.setText("Outdegree:")
            self.dyn_spin.setDecimals(0)
            self.dyn_spin.setValue(params['outdegree'])
            self.dyn_spin.setVisible(True)
        elif rule == 'fixed_total_number' and 'N' in params:
            self.dyn_label.setText("Total N:")
            self.dyn_spin.setDecimals(0)
            self.dyn_spin.setValue(params['N'])
            self.dyn_spin.setVisible(True)
        elif 'p' in params: 
            self.dyn_label.setText("Probability (p):")
            self.dyn_spin.setDecimals(4)
            self.dyn_spin.setValue(params['p'])
            self.dyn_spin.setVisible(True)

    def apply_changes(self):
        if self.current_conn_data is None: return
        
        self.current_conn_data['name'] = self.name_input.text()
        params = self.current_conn_data['params']
        
        params['weight'] = self.weight_spin.value()
        params['delay'] = self.delay_spin.value()
        
        rule = params.get('rule', '')
        if self.dyn_spin.isVisible():
            val = self.dyn_spin.value()
            if rule == 'fixed_indegree': params['indegree'] = int(val)
            elif rule == 'fixed_outdegree': params['outdegree'] = int(val)
            elif rule == 'fixed_total_number': params['N'] = int(val)
            elif 'p' in params: params['p'] = val
            
        print(f"Connection '{self.current_conn_data['name']}' updated in local memory.")

class DeviceTargetSelector(QGroupBox):

    def __init__(self, graph_list, parent=None):
        super().__init__("Target Selection", parent)
        self.graph_list = graph_list
        self.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; color: #ddd; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)
        
        self.combo_graph = QComboBox()
        self.combo_node = QComboBox()
        self.combo_pop = QComboBox()
        
        layout.addRow("Graph:", self.combo_graph)
        layout.addRow("Node:", self.combo_node)
        layout.addRow("Population:", self.combo_pop)
        
        self.combo_graph.currentIndexChanged.connect(self.on_graph_changed)
        self.combo_node.currentIndexChanged.connect(self.on_node_changed)
        
        self.refresh()

    def refresh(self):
        self.combo_graph.blockSignals(True)
        self.combo_graph.clear()
        for graph in self.graph_list:
            name = getattr(graph, 'graph_name', f'Graph {graph.graph_id}')
            self.combo_graph.addItem(f"{name} (ID: {graph.graph_id})", graph.graph_id)
        self.combo_graph.blockSignals(False)
        
        if self.graph_list:
            self.on_graph_changed()

    def on_graph_changed(self):
        self.combo_node.blockSignals(True)
        self.combo_node.clear()
        
        graph_id = self.combo_graph.currentData()
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        
        if graph:
            for node in graph.node_list:
                self.combo_node.addItem(f"{node.name} (ID: {node.id})", node.id)
        
        self.combo_node.blockSignals(False)
        self.on_node_changed()

    def on_node_changed(self):
        self.combo_pop.clear()
        graph_id = self.combo_graph.currentData()
        node_id = self.combo_node.currentData()
        
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if graph:
            node = next((n for n in graph.node_list if n.id == node_id), None)
            if node and hasattr(node, 'population'):
                for i, pop in enumerate(node.population):
                    model = nest.GetStatus(pop, 'model')[0] if len(pop) > 0 else "empty"
                    self.combo_pop.addItem(f"Pop {i}: {model}", i)

    def get_selection(self):
        return {
            'graph_id': self.combo_graph.currentData(),
            'node_id': self.combo_node.currentData(),
            'pop_id': self.combo_pop.currentData()
        }



class DeviceConfigPage(QWidget):
    deviceCreated = pyqtSignal(dict)
    deviceUpdated = pyqtSignal(dict, dict) # (old_data, new_data)

    def __init__(self, device_label, device_type, graph_list, parent=None):
        super().__init__(parent)
        self.device_type = device_type
        self.device_label = device_label
        self.graph_list = graph_list
        self.param_widgets = {}
        self.current_edit_device = None 
        
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Linke Seite: Ziel-Auswahl
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0,0,0,0)
        
        self.target_selector = DeviceTargetSelector(self.graph_list)
        left_layout.addWidget(self.target_selector)
        left_layout.addStretch()
        
        # Rechte Seite: Parameter
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        
        self.header_label = QLabel(f"Configure {self.device_label}")
        self.header_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {self._get_color()}; margin-bottom: 5px;")
        right_layout.addWidget(self.header_label)
        
        form_group = QGroupBox("Device Parameters")
        self.form_layout = QFormLayout(form_group)
        
        self._build_parameters()
        
        right_layout.addWidget(form_group)
        right_layout.addStretch()
        
        # --- BUTTON STYLES (Push Effect) ---
        base_btn_style = """
            QPushButton {
                color: white; 
                font-weight: bold; 
                font-size: 14px;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:pressed {
                margin-top: 2px;
                border-bottom: 0px solid;
            }
        """
        
        # Farben definieren
        c_main = self._get_color()
        c_dark = QColor(c_main).darker(150).name()
        
        style_create = base_btn_style + f"""
            QPushButton {{
                background-color: {c_main};
                border-bottom: 3px solid {c_dark};
            }}
            QPushButton:hover {{ background-color: {QColor(c_main).lighter(110).name()}; }}
        """
        
        style_save = base_btn_style + """
            QPushButton {
                background-color: #4CAF50;
                border-bottom: 3px solid #2E7D32;
            }
            QPushButton:hover { background-color: #66BB6A; }
        """
        
        style_cancel = base_btn_style + """
            QPushButton {
                background-color: #607D8B;
                border-bottom: 3px solid #455A64;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #78909C; }
        """

        # 1. CREATE Button
        self.btn_create = QPushButton(f"✚ Create {self.device_label}")
        self.btn_create.setMinimumHeight(45)
        self.btn_create.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_create.setStyleSheet(style_create)
        self.btn_create.clicked.connect(self.create_device)
        right_layout.addWidget(self.btn_create)

        # 2. SAVE Button
        self.btn_save = QPushButton(f"💾 Save Changes & Reset")
        self.btn_save.setMinimumHeight(45)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setStyleSheet(style_save)
        self.btn_save.clicked.connect(self.update_device)
        self.btn_save.setVisible(False)
        right_layout.addWidget(self.btn_save)
        
        # 3. CANCEL Button
        self.btn_cancel = QPushButton("Cancel Edit")
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.setStyleSheet(style_cancel)
        self.btn_cancel.clicked.connect(self.reset_to_create_mode)
        self.btn_cancel.setVisible(False)
        right_layout.addWidget(self.btn_cancel)

        main_layout.addWidget(left_container, 1)
        main_layout.addWidget(right_container, 2)

    def load_device_data(self, device_data):
        """Lädt Daten in die Maske und schaltet in den Edit-Modus."""
        print(f"Loading data into editor for: {device_data.get('model')}")
        self.current_edit_device = device_data
        
        # UI umschalten
        self.header_label.setText(f"EDIT MODE: {self.device_label} (ID: {device_data.get('id')})")
        self.header_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: #4CAF50; border-bottom: 2px solid #4CAF50;")
        
        self.btn_create.setVisible(False)
        self.btn_save.setVisible(True)
        self.btn_cancel.setVisible(True)
        
        # Target Selector befüllen
        target = device_data.get('target', {})
        gid = target.get('graph_id')
        idx_g = self.target_selector.combo_graph.findData(gid)
        if idx_g >= 0:
            self.target_selector.combo_graph.setCurrentIndex(idx_g)
            self.target_selector.on_graph_changed()
        
        nid = target.get('node_id')
        idx_n = self.target_selector.combo_node.findData(nid)
        if idx_n >= 0:
            self.target_selector.combo_node.setCurrentIndex(idx_n)
            self.target_selector.on_node_changed()
        
        pid = target.get('pop_id')
        idx_p = self.target_selector.combo_pop.findData(pid)
        if idx_p >= 0:
            self.target_selector.combo_pop.setCurrentIndex(idx_p)
        
        # Parameter Felder befüllen
        params = device_data.get('params', {})
        conn_params = device_data.get('conn_params', {})
        all_data = {**params, **conn_params}
        
        if 'weight' in conn_params: all_data['conn_weight'] = conn_params['weight']
        if 'delay' in conn_params: all_data['conn_delay'] = conn_params['delay']

        for key, widget in self.param_widgets.items():
            if key in all_data:
                val = all_data[key]
                if isinstance(widget, QDoubleSpinBox):
                    try: widget.setValue(float(val))
                    except: pass
                elif isinstance(widget, QLineEdit):
                    if isinstance(val, list):
                        widget.setText(", ".join(map(str, val)))
                    else:
                        widget.setText(str(val))

    def reset_to_create_mode(self):
        """Setzt die Maske sauber zurück auf Erstellen."""
        self.current_edit_device = None
        
        self.header_label.setText(f"Configure {self.device_label}")
        self.header_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {self._get_color()};")
        
        self.btn_create.setVisible(True)
        self.btn_save.setVisible(False)
        self.btn_cancel.setVisible(False)
        
        # HINWEIS: Wir behalten die Werte in den Feldern absichtlich (Comfort Feature),
        # damit man schnell ähnliche Devices erstellen kann. 
        # Wenn Sie leere Felder wollen, können Sie hier eine clear() Schleife einbauen.

    def _gather_data(self):
        target = self.target_selector.get_selection()
        if target['graph_id'] is None: 
            print("No target selected!")
            return None

        device_params = {}
        conn_params = {}
        
        for key, widget in self.param_widgets.items():
            val = None
            if isinstance(widget, QDoubleSpinBox):
                val = widget.value()
            elif isinstance(widget, QLineEdit):
                text_val = widget.text()
                if key in ["amplitude_times", "amplitude_values", "spike_times"]:
                    val = self._parse_list_input(text_val, dtype=float)
                elif key == "spike_multiplicities":
                    val = self._parse_list_input(text_val, dtype=int)
                elif key == "record_from":
                    cleaned = text_val.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
                    parts = [s.strip() for s in cleaned.split(',') if s.strip()]
                    val = parts if parts else ["V_m"]
                else:
                    val = text_val
            
            if key.startswith("conn_"):
                clean_key = key.replace("conn_", "")
                conn_params[clean_key] = val
            else:
                if val is not None:
                    device_params[key] = val
                    
        return {
            "model": self.device_type,
            "target": target,
            "params": device_params,
            "conn_params": conn_params
        }

    def create_device(self):
        data = self._gather_data()
        if data:
            self.deviceCreated.emit(data)

    def update_device(self):
        if self.current_edit_device is None: return
        
        new_data = self._gather_data()
        if new_data:
            new_data['id'] = self.current_edit_device.get('id')
            self.deviceUpdated.emit(self.current_edit_device, new_data)
            self.reset_to_create_mode()

    # --- HELPER ---
    def _get_color(self):
        if "generator" in self.device_type: return "#FF9800"
        if "recorder" in self.device_type or "meter" in self.device_type: return "#E91E63"
        return "#999"

    def _build_parameters(self):
        self.form_layout.addRow(QLabel("--- Device Settings ---"))
        self._add_param("start", 0.0, "Start Time (ms)")
        self._add_param("stop", 10000.0, "Stop Time (ms)")
        
        if "poisson_generator" in self.device_type:
            self._add_param("rate", 1000.0, "Rate (Hz)")
        elif "noise_generator" in self.device_type:
            self._add_param("mean", 0.0, "Mean (pA)")
            self._add_param("std", 200.0, "Std Dev (pA)")
            self._add_param("dt", 1.0, "Time Step (ms)")
        elif "dc_generator" in self.device_type:
            self._add_param("amplitude", 100.0, "Amplitude (pA)")
        elif "ac_generator" in self.device_type:
            self._add_param("amplitude", 50.0, "Amplitude (pA)")
            self._add_param("frequency", 10.0, "Frequency (Hz)")
            self._add_param("phase", 0.0, "Phase (deg)")
            self._add_param("offset", 0.0, "Offset (pA)")
        elif "step_current_generator" in self.device_type:
            self._add_text_param("amplitude_times", "100.0, 300.0, 500.0", "Times (ms)")
            self._add_text_param("amplitude_values", "100.0, 0.0, -50.0", "Currents (pA)")
        elif "spike_generator" in self.device_type:
             self._add_text_param("spike_times", "10.0, 20.5, 50.0", "Spike Times (ms)")
             self._add_text_param("spike_multiplicities", "", "Multiplicities")
        elif "recorder" in self.device_type:
            self._add_text_param("label", "record", "File Label")
        elif "multimeter" in self.device_type:
            self._add_param("interval", 1.0, "Recording Interval (ms)")
            self._add_text_param("record_from", "V_m", "Record (e.g. V_m)")
        elif "voltmeter" in self.device_type:
            self._add_param("interval", 1.0, "Recording Interval (ms)")

        self.form_layout.addRow(QLabel("--- Connection Settings ---"))
        default_weight = 10.0 if "generator" in self.device_type else 1.0
        self._add_param("conn_weight", default_weight, "Synapse Weight (pA)")
        self._add_param("conn_delay", 1.0, "Synapse Delay (ms)")

    def _add_param(self, key, default, label):
        spin = QDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(2)
        spin.setValue(default)
        self.form_layout.addRow(label, spin)
        self.param_widgets[key] = spin

    def _add_text_param(self, key, default, label):
        le = QLineEdit(str(default))
        self.form_layout.addRow(label, le)
        self.param_widgets[key] = le

    def _parse_list_input(self, text, dtype=float):
        if not text.strip(): return []
        try:
            cleaned = text.replace('[', '').replace(']', '')
            parts = cleaned.split(',')
            return [dtype(p.strip()) for p in parts if p.strip()]
        except: return []




class ToolsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.graph_list = [] 
        self.button_map = {}
        self.init_ui()
        
    def update_graphs(self, graph_list):
        self.graph_list = graph_list
        current_widget = self.config_stack.currentWidget()
        if isinstance(current_widget, DeviceConfigPage):
            current_widget.target_selector.graph_list = graph_list
            current_widget.target_selector.refresh()
            
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        menu_container = QWidget()
        menu_container.setFixedWidth(160)
        menu_container.setStyleSheet("background-color: #232323; border-right: 1px solid #444;")
        menu_layout = QVBoxLayout(menu_container)
        
        lbl_menu = QLabel("TOOLBOX")
        lbl_menu.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_menu.setStyleSheet("color: #888; font-weight: bold; font-size: 10px; letter-spacing: 1px;")
        menu_layout.addWidget(lbl_menu)
        
        self.devices = [
            ("Spike Recorder", "spike_recorder", "#E91E63"),
            ("Voltmeter", "voltmeter", "#E91E63"),
            ("Multimeter", "multimeter", "#E91E63"),
            None,
            ("Poisson Gen", "poisson_generator", "#FF9800"),
            ("Noise Gen", "noise_generator", "#FF9800"),
            ("DC Gen", "dc_generator", "#FF9800"),
            ("AC Gen", "ac_generator", "#FF9800"),
            ("Step Current", "step_current_generator", "#FF9800"),
            ("Spike Gen", "spike_generator", "#FF9800"),
        ]
        
        self.config_stack = QStackedWidget()
        
        # --- INDEX 0: HELP PAGE (NEU) ---
        self.help_page = DeviceHelpWidget()
        self.config_stack.addWidget(self.help_page)
        
        # Button Gruppe für Exklusivität
        self.btn_group = []

        for item in self.devices:
            if item is None:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet("color: #444;")
                menu_layout.addWidget(line)
                continue
                
            label, model, color = item
            
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet(f"""
                QPushButton {{
                    text-align: left; padding: 8px; border: none; background: transparent; color: #bbb; border-left: 3px solid transparent;
                }}
                QPushButton:checked {{ background: #333; color: white; border-left: 3px solid {color}; }}
                QPushButton:hover {{ background: #2c2c2c; }}
            """)
            
            page = DeviceConfigPage(label, model, self.graph_list)
            page.deviceCreated.connect(self.on_device_created)
            
            idx = self.config_stack.addWidget(page)
            btn.clicked.connect(lambda checked, i=idx, b=btn: self.switch_tool(i, b))
            
            self.button_map[btn] = idx
            self.btn_group.append(btn)
            menu_layout.addWidget(btn)
            
        menu_layout.addStretch()
        
        main_layout.addWidget(menu_container)
        main_layout.addWidget(self.config_stack)
        
    def switch_tool(self, index, clicked_btn):
        # 1. Altes Widget resetten
        current = self.config_stack.currentWidget()
        if isinstance(current, DeviceConfigPage):
            current.reset_to_create_mode()

        # 2. Buttons umschalten
        for btn in self.button_map:
            btn.setChecked(btn == clicked_btn)
            
        self.config_stack.setCurrentIndex(index)
        
        # 3. Neues Widget updaten
        widget = self.config_stack.widget(index)
        if isinstance(widget, DeviceConfigPage):
            widget.target_selector.graph_list = self.graph_list
            widget.target_selector.refresh()
            widget.reset_to_create_mode()

    def reset_view(self):
        """Springt zurück auf die Hilfeseite und deselektiert alle Buttons."""
        self.config_stack.setCurrentIndex(0) # Index 0 ist jetzt DeviceHelpWidget
        for btn in self.button_map:
            btn.setChecked(False)

    def open_device_editor(self, device_data):
        model = device_data.get('model')
        target_index = -1
        target_widget = None
        
        for i in range(self.config_stack.count()):
            widget = self.config_stack.widget(i)
            if isinstance(widget, DeviceConfigPage):
                if widget.device_type == model:
                    target_index = i
                    target_widget = widget
                    break
        
        if target_widget:
            for btn, idx in self.button_map.items():
                btn.setChecked(idx == target_index)
            
            self.config_stack.setCurrentIndex(target_index)
            
            target_widget.target_selector.graph_list = self.graph_list
            target_widget.target_selector.refresh()
            target_widget.load_device_data(device_data)
        else:
            print(f"No editor for '{model}' found.")

    def on_device_created(self, data):
        global _nest_simulation_has_run
        
        model_name = data['model']
        print(f"\nCreating Device: {model_name}")
        
        if _nest_simulation_has_run:
            print(" NEST Reset (Simulation has run)...")
            try:
                nest.ResetKernel()
                _nest_simulation_has_run = False
                for graph in self.graph_list:
                    for node in graph.node_list:
                        if not hasattr(node, 'positions') or not node.positions: node.build()
                        elif all(len(p) == 0 for p in node.positions if p is not None): node.build()
                        node.populate_node()
                print("  ✓ Reset Done")
            except Exception as e:
                print(f"  Reset failed: {e}")
                return
        
        target = data['target']
        graph_id = target['graph_id']
        node_id = target['node_id']
        pop_id = target['pop_id']
        
        target_graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not target_graph: 
            print(f"Error: Graph {graph_id} not found.")
            return
            
        target_node = next((n for n in target_graph.node_list if n.id == node_id), None)
        if not target_node: 
            print(f"Error: Node {node_id} not found.")
            return
        
        conn_params = data.get('conn_params', {})
        
        if not hasattr(target_node, 'devices'): target_node.devices = []
        if 'devices' not in target_node.parameters: target_node.parameters['devices'] = []
            
        device_record = {
            "id": len(target_node.parameters['devices']),
            "model": model_name,
            "target_pop_id": pop_id,
            "params": data['params'],     
            "conn_params": conn_params,    
            "runtime_gid": None           
        }
            
        target_node.parameters['devices'].append(device_record)
        target_node.devices.append(device_record)
        print(f"  ✓ Saved to Node Parameters (Total: {len(target_node.parameters['devices'])})")

        try:
            nest_device = nest.Create(model_name, params=data['params'])
            device_record['runtime_gid'] = nest_device
            print(f"  ✓ NEST ID: {nest_device}")
            
            if pop_id >= len(target_node.population):
                raise ValueError(f"Population index {pop_id} out of range.")

            pop_nest = target_node.population[pop_id]
            weight = float(conn_params.get('weight', 1.0))
            delay = max(float(conn_params.get('delay', 1.0)), 0.1)
            
            syn_spec = {'synapse_model': 'static_synapse', 'weight': weight, 'delay': delay}
            
            is_source = False
            if "generator" in model_name or "stimulator" in model_name: is_source = True
            elif "meter" in model_name: is_source = True
            
            if is_source:
                nest.Connect(nest_device, pop_nest, syn_spec=syn_spec)
                direction_str = "-> Population (Source)"
            else:
                nest.Connect(pop_nest, nest_device, syn_spec=syn_spec)
                direction_str = "<- Population (Target)"
                
            print(f"  ✓ Connected: {model_name} {direction_str} (W={weight}, D={delay})")
            
            # --- RESET TO BLANK PAGE AFTER CREATION ---
            self.reset_view()  # <--- HIER
            
        except Exception as e:
            print(f" Error creating device in NEST: {e}")
            import traceback; traceback.print_exc()
            if device_record in target_node.parameters['devices']: target_node.parameters['devices'].remove(device_record)
            if device_record in target_node.devices: target_node.devices.remove(device_record)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "NEST Error", f"Could not create device:\n{str(e)}")




class DeviceHelpWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Icon / Header
        lbl_icon = QLabel("🛠️")
        lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_icon.setStyleSheet("font-size: 48px; margin-bottom: 10px;")
        layout.addWidget(lbl_icon)
        
        lbl_title = QLabel("Device Toolbox Guide")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #FF9800; font-size: 18px; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(lbl_title)
        
        # Help Text
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                color: #ccc;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        
        html_content = """
        <div style='font-family: sans-serif;'>
            <p style='text-align: center; color: #888;'>Select a tool from the left menu to configure it.</p>
            
            <h3 style='color: #E91E63;'>📡 Recorders & Meters (Output)</h3>
            <ul>
                <li><b>Spike Recorder:</b> Records firing times (spikes) from neurons. Essential for Raster Plots.</li>
                <li><b>Voltmeter / Multimeter:</b> Records continuous membrane potentials (V_m). connect to neurons to see analog traces.</li>
            </ul>

            <h3 style='color: #FF9800;'>⚡ Generators (Input)</h3>
            <ul>
                <li><b>Poisson Generator:</b> Injects random spikes with a specific frequency (Hz). Simulates background activity.</li>
                <li><b>Noise Generator:</b> Injects continuous current noise (Gaussian white noise).</li>
                <li><b>DC Generator:</b> Injects a constant current (pA). Useful for driving neurons to threshold.</li>
                <li><b>Step Current:</b> Injects current pulses at specific times.</li>
                <li><b>Spike Generator:</b> Injects exact spikes at pre-defined timestamps.</li>
            </ul>
            
            <hr style='border-color: #444;'>
            <p style='color: #aaa; font-style: italic; font-size: 11px;'>
                <b>Note:</b> Generators are connected <i>TO</i> the population (Source). 
                Recorders are connected <i>FROM</i> the population (Target).
            </p>
        </div>
        """
        help_text.setHtml(html_content)
        layout.addWidget(help_text)


class SafeGLViewWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rendering_enabled = True

    def paintGL(self, *args, **kwargs):

        if not self.rendering_enabled or not self.isVisible() or not self.isValid():
            return
            
        try:
            super().paintGL(*args, **kwargs)
        except Exception:
            pass

    def stop_rendering(self):
        self.rendering_enabled = False
        
    def start_rendering(self):
        self.rendering_enabled = True

class StructuresWidget(QWidget):

    structureSelected = pyqtSignal(str, list, list)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_lbl = QLabel("CORTICAL STRUCTURES")
        header_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #E91E63; border-bottom: 2px solid #E91E63; padding-bottom: 5px;")
        layout.addWidget(header_lbl)
        
        info = QLabel("Select a region to auto-generate a node patch (10x10x10)\nwith biological neuron distributions.")
        info.setStyleSheet("color: #888; font-style: italic; margin-bottom: 10px;")
        layout.addWidget(info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        
        container = QWidget()
        self.grid_layout = QGridLayout(container)
        self.grid_layout.setSpacing(10)

        region_data = {r_key: {} for r_key in region_names.values()}
        
        for model, region_map in distributions.items():
            for r_key, prob in region_map.items():
                if prob > 0:
                    region_data[r_key][model] = prob

        row, col = 0, 0
        for display_name, r_key in region_names.items():
            if r_key not in region_data or not region_data[r_key]:
                continue
            
            models_probs = region_data[r_key]
            model_list = list(models_probs.keys())
            prob_list = list(models_probs.values())
            
            total = sum(prob_list)
            if total > 0:
                prob_list = [p/total for p in prob_list]
            
            # Button erstellen
            btn = QPushButton(display_name)
            btn.setMinimumHeight(60)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #37474F;
                    color: white;
                    font-weight: bold;
                    border: 1px solid #455A64;
                    border-radius: 5px;
                    font-size: 12px;
                    text-align: left;
                    padding-left: 10px;
                }
                QPushButton:hover {
                    background-color: #E91E63;
                    border: 1px solid #F48FB1;
                }
            """)
            
            tooltip_txt = f"<b>{display_name}</b><br>Contains:<br>"
            for m, p in zip(model_list, prob_list):
                tooltip_txt += f"• {m}: {p*100:.1f}%<br>"
            btn.setToolTip(tooltip_txt)
            
            btn.clicked.connect(lambda checked, n=display_name, m=model_list, p=prob_list: 
                              self.structureSelected.emit(n, m, p))
            
            self.grid_layout.addWidget(btn, row, col)
            
            col += 1
            if col > 1: 
                row += 1
        
        scroll.setWidget(container)
        layout.addWidget(scroll)







class FlowFieldWidget(QWidget):
    """
    Visualisiert die Vektorfelder (Flow Fields) für den aktuell ausgewählten Node.
    Erlaubt interaktives 'Stepping' durch das Feld mit Step-Counter und +/- 10 Schritten.
    """
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.target_graph_id = None
        self.target_node_id = None
        
        # Speicher für Simulation
        self.dt = 0.05
        self.current_step = 0  # <--- Step Counter
        self.active_populations_data = [] 
        
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Toolbar für Steuerung ---
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(5, 5, 5, 5)
        toolbar.setSpacing(5)
        
        # Reset
        btn_reset = QPushButton("↺ Reset")
        btn_reset.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        btn_reset.clicked.connect(self.reset_positions)
        toolbar.addWidget(btn_reset)
        
        # Prev 10
        btn_prev10 = QPushButton("⏪ -10")
        btn_prev10.setStyleSheet("background-color: #5D4037; color: white; font-weight: bold; padding: 5px;")
        btn_prev10.clicked.connect(lambda: self.perform_step(-10))
        toolbar.addWidget(btn_prev10)

        # Prev 1
        btn_prev = QPushButton("◀ -1")
        btn_prev.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 5px;")
        btn_prev.clicked.connect(lambda: self.perform_step(-1))
        toolbar.addWidget(btn_prev)
        
        # Step Label
        self.lbl_step = QLabel("Step: 0")
        self.lbl_step.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_step.setFixedWidth(80)
        self.lbl_step.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 12px; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        toolbar.addWidget(self.lbl_step)
        
        # Next 1
        btn_next = QPushButton("+1 ▶")
        btn_next.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        btn_next.clicked.connect(lambda: self.perform_step(1))
        toolbar.addWidget(btn_next)

        # Next 10
        btn_next10 = QPushButton("⏩ +10")
        btn_next10.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 5px;")
        btn_next10.clicked.connect(lambda: self.perform_step(10))
        toolbar.addWidget(btn_next10)
        
        # DT Spinbox
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 1.0)
        self.dt_spin.setSingleStep(0.01)
        self.dt_spin.setValue(0.05)
        self.dt_spin.setPrefix("dt: ")
        self.dt_spin.setDecimals(3)
        self.dt_spin.valueChanged.connect(self.update_dt)
        toolbar.addWidget(self.dt_spin)
        
        self.layout.addLayout(toolbar)
        
        # --- PyVista Plotter ---
        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        self.plotter.add_axes() 
        
        self.layout.addWidget(self.plotter)
        
        # Info Text Actor
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.GetTextProperty().SetFontSize(14)
        self.text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        self.text_actor.SetPosition(10, 10)
        self.plotter.renderer.AddViewProp(self.text_actor)
        
        self.build_scene()

    def update_dt(self, val):
        self.dt = val

    def set_target_node(self, graph_id, node_id):
        if self.target_graph_id == graph_id and self.target_node_id == node_id:
            return 
        self.target_graph_id = graph_id
        self.target_node_id = node_id
        if self.isVisible():
            self.build_scene()

    def build_scene(self):
        self.plotter.clear()
        self.plotter.renderer.AddViewProp(self.text_actor) 
        self.active_populations_data = [] 
        self.current_step = 0
        self.lbl_step.setText("Step: 0")
        
        if not self.graph_list:
            self.text_actor.SetInput("No graphs available.")
            self.plotter.render()
            return

        if self.target_graph_id is None or self.target_node_id is None:
            self.text_actor.SetInput("Select a node to inspect Flow Field.")
            self.plotter.render()
            return

        target_graph = next((g for g in self.graph_list if g.graph_id == self.target_graph_id), None)
        if not target_graph: return
        target_node = next((n for n in target_graph.node_list if n.id == self.target_node_id), None)
        if not target_node: return

        self.text_actor.SetInput(f"Flow Field: {target_node.name} (Graph {target_graph.graph_id})")

        if not hasattr(target_node, 'positions') or not target_node.positions:
            return

        params = target_node.parameters
        encoded_per_type = params.get("encoded_polynoms_per_type", [])
        poly_gen = PolynomGenerator(n=params.get('polynom_max_power', 5))

        show_legend = True 

        for pop_idx, positions in enumerate(target_node.positions):
            if positions is None or len(positions) == 0:
                continue
            
            funcs = (None, None, None)
            if pop_idx < len(encoded_per_type):
                try:
                    funcs = poly_gen.decode_multiple(encoded_per_type[pop_idx])
                except Exception as e:
                    print(f"Error decoding polys: {e}")
            
            f1, f2, f3 = funcs
            if f1 is None: continue 

            current_pos = positions.copy()
            
            points_mesh = pv.PolyData(current_pos)
            self.plotter.add_mesh(
                points_mesh, 
                color="#aaaaaa", 
                point_size=5, 
                render_points_as_spheres=True
            )
            
            vectors_mesh = pv.PolyData(current_pos)
            self._update_vectors(vectors_mesh, current_pos, f1, f2, f3)
            
            glyph = vectors_mesh.glyph(orient='vectors', scale='scale', factor=0.8)
            
            actor_arrows = self.plotter.add_mesh(
                glyph, 
                scalars='mag', 
                cmap='jet', 
                opacity=0.8,
                show_scalar_bar=show_legend,
                scalar_bar_args={'title': 'Flow Magnitude', 'color': 'white'}
            )
            show_legend = False
            
            self.active_populations_data.append({
                'points_mesh': points_mesh,
                'vectors_mesh': vectors_mesh,
                'arrow_actor': actor_arrows,
                'original_pos': positions.copy(),
                'current_pos': current_pos,
                'funcs': (f1, f2, f3)
            })

        self.plotter.reset_camera()
        self.plotter.update()

    def _update_vectors(self, mesh, positions, f1, f2, f3):
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        
        u = f1(x, y, z)
        v = f2(x, y, z)
        w = f3(x, y, z)
        
        vectors = np.column_stack((u, v, w))
        magnitudes = np.linalg.norm(vectors, axis=1)
        safe_mags = np.where(magnitudes == 0, 1, magnitudes)
        
        mesh.points = positions
        mesh['vectors'] = vectors / safe_mags[:, None]
        mesh['mag'] = magnitudes
        mesh['scale'] = np.clip(magnitudes, 0, 2.0)

    def perform_step(self, n_steps):
        """
        Führt n_steps Iterationen durch (positiv = vorwärts, negativ = rückwärts).
        """
        if not self.active_populations_data:
            return

        direction = 1 if n_steps > 0 else -1
        iterations = abs(n_steps)

        # Wir iterieren durch die Anzahl der gewünschten Schritte
        # Wichtig: Wir müssen für jeden kleinen Schritt die neuen Vektoren berechnen,
        # da sich das Vektorfeld räumlich ändert.
        
        for data in self.active_populations_data:
            pos = data['current_pos']
            f1, f2, f3 = data['funcs']
            
            for _ in range(iterations):
                x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
                u = f1(x, y, z)
                v = f2(x, y, z)
                w = f3(x, y, z)
                
                velocity = np.column_stack((u, v, w))
                
                # Euler Integration Step
                pos = pos + (direction * velocity * self.dt)
            
            # Daten aktualisieren nach allen Iterationen
            data['current_pos'] = pos
            
            # Visualisierung updaten
            data['points_mesh'].points = pos
            
            # Pfeile an neuer Position berechnen
            self._update_vectors(data['vectors_mesh'], pos, f1, f2, f3)
            
            self.plotter.remove_actor(data['arrow_actor'])
            glyph = data['vectors_mesh'].glyph(orient='vectors', scale='scale', factor=0.8)
            new_actor = self.plotter.add_mesh(
                glyph, 
                scalars='mag', 
                cmap='jet', 
                opacity=0.8,
                show_scalar_bar=False 
            )
            data['arrow_actor'] = new_actor

        self.current_step += n_steps
        self.lbl_step.setText(f"Step: {self.current_step}")
        self.plotter.update()

    def reset_positions(self):
        if not self.active_populations_data: return
        
        self.current_step = 0
        self.lbl_step.setText("Step: 0")
        
        for data in self.active_populations_data:
            data['current_pos'] = data['original_pos'].copy()
            
            f1, f2, f3 = data['funcs']
            data['points_mesh'].points = data['current_pos']
            
            self._update_vectors(data['vectors_mesh'], data['current_pos'], f1, f2, f3)
            
            self.plotter.remove_actor(data['arrow_actor'])
            glyph = data['vectors_mesh'].glyph(orient='vectors', scale='scale', factor=0.8)
            data['arrow_actor'] = self.plotter.add_mesh(
                glyph, scalars='mag', cmap='jet', opacity=0.8, show_scalar_bar=False
            )
            
        self.plotter.update()



class SimulationViewWidget(QWidget):
    # --- SIGNALS ---
    sigPauseSimulation = pyqtSignal()
    sigStopSimulation = pyqtSignal()
    sigStartContinuous = pyqtSignal(float, float) 
    sigStepSimulation = pyqtSignal(float)
    sigResetSimulation = pyqtSignal() 
    
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.scene_loaded = False
        self.spike_tool_active = False
        self.is_paused = True 
        self.all_points = None      
        self.base_colors = None     
        self.current_colors = None  
        self.global_ids = None      
        self.gid_to_idx = {}        
        self.anim_rg = None         
        self.anim_b = None          
        self.heat = None            
        self.dynamic_generators = [] 
        self.generator_btns = []     
        self.stim_params = {'radius': 2.0, 'weight_ex': 10.0, 'weight_in': -10.0, 'delay': 1.0, 'model': 'spike_generator', 'rate': 500.0, 'multiplicity': 1.0}
        self.point_size = 7.0       
        self.decay_flash = 0.60     
        self.decay_tail = 0.90      
        self.decay_heat = 0.80      
        self.view = None
        self.scatter_item = None
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.update_animation)
        self.init_ui()
        self.update_button_styles() 
        self._original_mouse_press = self.view.mousePressEvent
        self.view.mousePressEvent = self._wrapped_mouse_press

    # --- FIX: Methode für sicheren Reset ---
    def stop_rendering_safe(self):
        self.is_paused = True
        if hasattr(self, 'anim_timer'):
            self.anim_timer.stop()
        if self.scatter_item:
            try: self.scatter_item.setVisible(False)
            except: pass

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        sidebar = QWidget(); sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("background-color: #232323; border-right: 1px solid #444;")
        sb_layout = QVBoxLayout(sidebar); sb_layout.setContentsMargins(8,8,8,8); sb_layout.setSpacing(10)
        sb_layout.addWidget(QLabel("SIMULATION CONTROL", styleSheet="color:#FF9800; font-weight:bold; font-size:10px;"))
        
        row_load = QHBoxLayout()
        self.btn_load = QPushButton("⚡ LOAD SCENE"); self.btn_load.clicked.connect(self.load_scene)
        self.btn_load.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold;")
        self.btn_clear_scene = QPushButton("✕"); self.btn_clear_scene.setFixedWidth(30)
        self.btn_clear_scene.clicked.connect(self.unload_scene)
        self.btn_clear_scene.setStyleSheet("background-color: #444; color: white;")
        row_load.addWidget(self.btn_load); row_load.addWidget(self.btn_clear_scene)
        sb_layout.addLayout(row_load)
        
        sim_grid = QGridLayout(); sim_grid.setSpacing(5)
        self.btn_start = QPushButton("▶ RUN"); self.btn_start.setMinimumHeight(50); self.btn_start.clicked.connect(self.action_start)
        self.btn_step = QPushButton("⏯ STEP"); self.btn_step.setMinimumHeight(50); self.btn_step.clicked.connect(self.action_step)
        self.btn_pause = QPushButton("⏸ PAUSE"); self.btn_pause.clicked.connect(self.action_pause); self.btn_pause.setStyleSheet("background-color: #424242; color: #ccc;")
        self.btn_reset = QPushButton("↺ RESET"); self.btn_reset.clicked.connect(self.action_reset); self.btn_reset.setStyleSheet("background-color: #BF360C; color: white;")
        sim_grid.addWidget(self.btn_start, 0, 0); sim_grid.addWidget(self.btn_step, 0, 1)
        sim_grid.addWidget(self.btn_pause, 1, 0); sim_grid.addWidget(self.btn_reset, 1, 1)
        sb_layout.addLayout(sim_grid)
        
        tg = QGroupBox("Time Settings"); tl = QFormLayout(tg)
        self.spin_step_size = QDoubleSpinBox(); self.spin_step_size.setRange(0.1, 1000); self.spin_step_size.setValue(25.0); self.spin_step_size.setSuffix(" ms")
        self.spin_duration = QDoubleSpinBox(); self.spin_duration.setRange(0, 1e6); self.spin_duration.setValue(1000); self.spin_duration.setSuffix(" ms")
        tl.addRow("Step:", self.spin_step_size); tl.addRow("Max:", self.spin_duration)
        sb_layout.addWidget(tg)
        
        self.btn_spike_tool = QPushButton("Inject Tool"); self.btn_spike_tool.setCheckable(True)
        self.btn_spike_tool.clicked.connect(self.toggle_spike_tool)
        self.btn_spike_tool.setStyleSheet("background-color: #333; color: #aaa; padding: 6px;")
        sb_layout.addWidget(self.btn_spike_tool)
        
        self.tool_settings = QWidget(); self.tool_settings.setVisible(False)
        ts_layout = QVBoxLayout(self.tool_settings); ts_layout.setContentsMargins(0,5,0,5)
        self.combo_model = QComboBox(); self.combo_model.addItems(["spike_generator", "poisson_generator"])
        self.combo_model.currentTextChanged.connect(lambda t: self.stim_params.update({'model': t}))
        ts_layout.addWidget(self.combo_model)
        
        gp = QGridLayout()
        self.spin_w_ex = QDoubleSpinBox(); self.spin_w_ex.setRange(0, 10000); self.spin_w_ex.setValue(10.0); self.spin_w_ex.valueChanged.connect(lambda v: self.stim_params.update({'weight_ex': v}))
        self.spin_w_in = QDoubleSpinBox(); self.spin_w_in.setRange(-10000, 0); self.spin_w_in.setValue(-10.0); self.spin_w_in.valueChanged.connect(lambda v: self.stim_params.update({'weight_in': v}))
        gp.addWidget(QLabel("Ex:"),0,0); gp.addWidget(self.spin_w_ex,0,1); gp.addWidget(QLabel("In:"),1,0); gp.addWidget(self.spin_w_in,1,1)
        self.lbl_radius = QLabel(f"Radius: {self.stim_params['radius']:.1f}")
        self.slider_radius = QSlider(Qt.Orientation.Horizontal); self.slider_radius.setRange(1, 100); self.slider_radius.setValue(20)
        self.slider_radius.valueChanged.connect(self._update_radius)
        ts_layout.addLayout(gp); ts_layout.addWidget(self.lbl_radius); ts_layout.addWidget(self.slider_radius)
        sb_layout.addWidget(self.tool_settings)
        
        sb_layout.addWidget(QLabel("INJECTORS", styleSheet="color:#ccc; font-weight:bold; margin-top:10px;"))
        self.scroll_area = QScrollArea(); self.scroll_area.setWidgetResizable(True); self.scroll_area.setStyleSheet("border:1px solid #333; background:#1e1e1e;")
        self.grid_container = QWidget(); self.grid_layout = QGridLayout(self.grid_container); self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.grid_container)
        sb_layout.addWidget(self.scroll_area)
        
        self.btn_clear_gens = QPushButton("🗑 Clear All"); self.btn_clear_gens.clicked.connect(self.clear_generators)
        self.btn_clear_gens.setStyleSheet("background-color: #5d4037; color: white;")
        sb_layout.addWidget(self.btn_clear_gens)
        
        layout.addWidget(sidebar)
        self.view = gl.GLViewWidget(); self.view.opts['distance'] = 40; self.view.setBackgroundColor('#050505')
        layout.addWidget(self.view)

    def update_button_styles(self):
        style_run = "QPushButton { background-color: #2E7D32; color: #ccc; font-weight: bold; } QPushButton:hover { background-color: #388E3C; }"
        style_active = "QPushButton { background-color: #00E676; color: black; border: 2px solid white; font-weight: bold; }"
        if not self.is_paused:
            self.btn_start.setStyleSheet(style_active); self.btn_start.setText("RUNNING")
            self.btn_step.setEnabled(False); self.btn_pause.setEnabled(True)
        else:
            self.btn_start.setStyleSheet(style_run); self.btn_start.setText("▶ RUN")
            self.btn_step.setEnabled(True); self.btn_pause.setEnabled(False)

    def action_start(self):
        if not self.scene_loaded: return
        self.is_paused = False; self.update_button_styles()
        self.sigStartContinuous.emit(self.spin_step_size.value(), self.spin_duration.value())
    
    def action_step(self):
        if not self.scene_loaded: return
        self.is_paused = True; self.update_button_styles()
        self.sigStepSimulation.emit(self.spin_step_size.value())
        
    def action_pause(self):
        self.is_paused = True; self.update_button_styles(); self.sigPauseSimulation.emit()
        
    def action_reset(self):
        self.is_paused = True; self.update_button_styles()
        if self.anim_rg is not None: self.anim_rg[:] = 0; self.anim_b[:] = 0; self.heat[:] = 0
        self.update_animation()
        self.sigResetSimulation.emit()

    # ... (Helper methods same as before: toggle_spike_tool, _update_radius, add_gen_button) ...
    def toggle_spike_tool(self, checked): self.tool_settings.setVisible(checked)
    def _update_radius(self, val): 
        self.stim_params['radius'] = val/10.0; self.lbl_radius.setText(f"Radius: {self.stim_params['radius']:.1f}")

    def add_gen_button(self, data):
        idx = len(self.generator_btns); btn = QPushButton(); btn.setFixedSize(75, 45)
        m = "E" if data['type']=="excitatory" else "I"; btn.setText(f"{m} #{idx+1}\n({len(data['targets'])})")
        c = "#00E5FF" if data['type']=="excitatory" else "#FF5252"
        btn.setStyleSheet(f"QPushButton {{ background-color: {c}; color: black; font-weight: 900; border: 2px solid #222; border-radius: 8px; }}")
        btn.clicked.connect(lambda: self.inject_stimulus(data))
        self.grid_layout.addWidget(btn, idx//3, idx%3)
        self.generator_btns.append(btn)

    def load_scene(self):
        if self.scene_loaded: self.unload_scene()
        print("Loading Scene...")
        points = []; colors = []; gids_list = []
        for graph in self.graph_list:
            for node in graph.node_list:
                if hasattr(node, 'positions') and node.positions:
                    for pop_idx, cluster in enumerate(node.positions):
                        if cluster is None or len(cluster)==0: continue
                        model = node.neuron_models[pop_idx] if hasattr(node, 'neuron_models') and pop_idx < len(node.neuron_models) else "unknown"
                        hex_c = neuron_colors.get(model, "#ffffff"); rgb = mcolors.to_rgba(hex_c, alpha=0.6)
                        count = len(cluster); points.append(cluster); colors.append(np.tile(rgb, (count, 1)))
                        if hasattr(node, 'population') and len(node.population)>pop_idx:
                            pop_ids = nest.GetStatus(node.population[pop_idx], 'global_id')
                            gids_list.append(np.array(pop_ids) if len(pop_ids)==count else np.zeros(count, dtype=int))
                        else: gids_list.append(np.zeros(count, dtype=int))
        if not points: return
        self.all_points = np.vstack(points); self.base_colors = np.vstack(colors)
        self.global_ids = np.concatenate(gids_list)
        self.gid_to_idx = {gid: i for i, gid in enumerate(self.global_ids) if gid > 0}
        self.current_colors = self.base_colors.copy()
        N = len(self.all_points)
        self.anim_rg = np.zeros(N); self.anim_b = np.zeros(N); self.heat = np.zeros(N)
        self.scatter_item = gl.GLScatterPlotItem(pos=self.all_points, color=self.current_colors, size=self.point_size, pxMode=True)
        self.scatter_item.setGLOptions('translucent')
        self.view.addItem(self.scatter_item)
        self.scene_loaded = True; self.anim_timer.start(30); self.update_button_styles()

    def unload_scene(self):
        self.anim_timer.stop()
        if self.scatter_item and self.view:
            try: self.view.removeItem(self.scatter_item)
            except: pass
        self.scatter_item = None; self.all_points = None; self.gid_to_idx = {}; self.scene_loaded = False

    def feed_spikes(self, spike_gids):
        if not self.scene_loaded: return
        indices = [self.gid_to_idx[gid] for gid in spike_gids if gid in self.gid_to_idx]
        if indices:
            self.anim_rg[indices] = 1.0; self.anim_b[indices] = 1.0; self.heat[indices] += 1.0
            if self.view: self.view.update()

    def update_animation(self):
        if not self.scene_loaded or self.scatter_item is None: return
        self.anim_rg *= self.decay_tail; self.anim_b *= self.decay_flash; self.heat *= self.decay_heat
        if np.max(self.anim_rg) < 0.01: return
        new_c = self.base_colors.copy()
        mask = self.anim_rg > 0.05
        if np.any(mask):
            val = self.anim_rg[mask, np.newaxis]
            new_c[mask, 0:3] += val # Brighten
            new_c[mask, 3] = np.minimum(1.0, 0.3 + val.flatten()) # Alpha
        mask_burst = self.heat > 8.0
        if np.any(mask_burst): new_c[mask_burst] = [0.0, 0.8, 1.0, 1.0]
        self.scatter_item.setData(color=new_c)

    # ... (Injection helpers same as before: _project_points, _get_nearest, _wrapped_mouse_press, _find_targets) ...
    def _project_points(self):
        if self.all_points is None: return None, None
        v = self.view.viewMatrix(); p = self.view.projectionMatrix(); mvp = p * v
        mat = np.array(mvp.data()).reshape(4,4)
        pts = np.hstack([self.all_points, np.ones((len(self.all_points),1))])
        clip = pts @ mat
        return clip, clip[:,3]>0.001

    def _get_nearest_neuron_idx(self, pos):
        clip, valid = self._project_points()
        if clip is None: return None, None
        idx = np.where(valid)[0]
        if len(idx)==0: return None, None
        vc = clip[idx]; ndc = vc[:,:3]/vc[:,3][:,np.newaxis]
        w, h = self.view.width(), self.view.height()
        sx = (ndc[:,0]+1)*w/2; sy = (1-ndc[:,1])*h/2
        dist = (sx-pos.x())**2 + (sy-pos.y())**2
        mini = np.argmin(dist)
        if dist[mini] < 900: return idx[mini], self.all_points[idx[mini]]
        return None, None

    def _wrapped_mouse_press(self, event):
        self._original_mouse_press(event)
        if not self.scene_loaded or not self.spike_tool_active: return
        idx, center = self._get_nearest_neuron_idx(event.pos())
        if center is not None:
            mode = "excitatory" if event.button() == Qt.MouseButton.LeftButton else "inhibitory"
            self.create_injector(center, mode)

    def _find_targets(self, pos, r):
        if self.all_points is None: return []
        hits = np.where(np.sum((self.all_points-pos)**2, axis=1) <= r**2)[0]
        ts = self.global_ids[hits].tolist()
        return [g for g in ts if g > 0]

    def create_injector(self, pos, mode):
        targets = self._find_targets(pos, self.stim_params['radius'])
        if not targets: return
        was_running = not self.is_paused
        if was_running: self.sigPauseSimulation.emit(); QApplication.processEvents()
        
        model = self.stim_params['model']
        try: status = nest.GetKernelStatus(); t = status.get('time', 0.0)
        except: t = 0.0
        
        params = {}
        if model == 'spike_generator':
            params = {'spike_times': [t+1.0], 'spike_weights': [float(self.stim_params['multiplicity'])]}
        elif model == 'poisson_generator':
            params = {'rate': self.stim_params['rate'], 'start': t}
        
        try:
            gid = nest.Create(model, params=params)
            w = self.stim_params['weight_ex'] if mode=="excitatory" else self.stim_params['weight_in']
            nest.Connect(gid, targets, syn_spec={'weight': w, 'delay': 1.0})
            
            data = {'gid': gid, 'model': model, 'targets': targets, 'pos': pos, 'radius': self.stim_params['radius'], 'weight': w, 'delay': 1.0, 'type': mode, 'params': params}
            self.dynamic_generators.append(data)
            self.add_gen_button(data)
            self.feed_spikes(targets)
        except Exception as e: print(f"Error: {e}")
        if was_running: self.action_start()

    def inject_stimulus(self, data):
        try: 
            status = nest.GetKernelStatus(); t = status.get('time', 0.0)
            if data['model'] == 'spike_generator':
                nest.SetStatus(data['gid'], {'spike_times': [t+1.0], 'spike_weights': [1.0]})
            self.feed_spikes(data['targets'])
        except: pass

    def clear_generators(self):
        self.dynamic_generators.clear()
        for b in self.generator_btns: self.grid_layout.removeWidget(b); b.deleteLater()
        self.generator_btns.clear()

    def restore_injectors(self):
        if not self.dynamic_generators: return
        old = list(self.dynamic_generators); self.clear_generators()
        for o in old:
            try:
                ts = self._find_targets(o['pos'], o['radius'])
                if not ts: continue
                gid = nest.Create(o['model'], params=o['params'])
                nest.Connect(gid, ts, syn_spec={'weight': o['weight'], 'delay': o['delay']})
                o['gid'] = gid; o['targets'] = ts
                self.dynamic_generators.append(o)
                self.add_gen_button(o)
            except: pass






class AnalysisDashboard(QWidget):
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # === LEFT MENU (SIDEBAR) ===
        menu_frame = QFrame()
        menu_frame.setFixedWidth(200)
        menu_frame.setStyleSheet("background-color: #232323; border-right: 1px solid #444;")
        menu_layout = QVBoxLayout(menu_frame)
        menu_layout.setContentsMargins(10, 20, 10, 10)
        menu_layout.setSpacing(10)
        
        lbl_title = QLabel("DATA ANALYSIS")
        lbl_title.setStyleSheet("color: #bbb; font-weight: bold; font-size: 12px; letter-spacing: 1px; border-bottom: 1px solid #444; padding-bottom: 5px;")
        menu_layout.addWidget(lbl_title)
        
        # Buttons für die Tabs
        self.btn_graph_info = QPushButton("Graph Info")
        self.btn_time_series = QPushButton("Time Series")
        self.btn_tool_inspector = QPushButton("Tool Inspector")
        self.btn_terminal = QPushButton("Terminal")  # <--- NEU: 4. Button
        
        # Liste der Buttons
        self.buttons = [
            self.btn_graph_info, 
            self.btn_time_series, 
            self.btn_tool_inspector,
            self.btn_terminal
        ]
        
        for i, btn in enumerate(self.buttons):
            btn.setCheckable(True)
            btn.setStyleSheet(self._get_btn_style())
            # Verbindung mit Index
            btn.clicked.connect(lambda checked, idx=i: self.switch_view(idx))
            menu_layout.addWidget(btn)
        
        menu_layout.addStretch()
        
        # === RIGHT CONTENT (STACK) ===
        self.content_stack = QStackedWidget()
        
        # Tab 0: Graph Info
        self.graph_info = GraphInfoWidget(self.graph_list)
        self.content_stack.addWidget(self.graph_info)
        
        # Tab 1: Time Series Plots
        self.time_series = TimeSeriesPlotWidget(self.graph_list)
        self.content_stack.addWidget(self.time_series)
        
        # Tab 2: Tool Inspector
        self.tool_inspector = ToolInspectorWidget(self.graph_list)
        self.content_stack.addWidget(self.tool_inspector)

        # Tab 3: Terminal (NEU)
        # Wir übergeben den Kontext (nest, graph_list, etc.)
        context = {
            'graph_list': self.graph_list,
            'nest': nest,
            'np': np,
            'self': self
        }
        self.terminal_widget = PythonConsoleWidget(context_vars=context)
        self.content_stack.addWidget(self.terminal_widget)
        
        layout.addWidget(menu_frame)
        layout.addWidget(self.content_stack)
        
        # Start auf Tab 0
        self.switch_view(0)

    def switch_view(self, index):
        self.content_stack.setCurrentIndex(index)
        
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)
            
        # Refresh Data Logic beim Umschalten
        if index == 0:
            self.graph_info.refresh_combo()
        elif index == 1:
            self.time_series.refresh_graphs()
        # Index 2 (Tools) & 3 (Terminal) brauchen keinen expliziten Refresh

    def _get_btn_style(self):
        return """
            QPushButton {
                text-align: left; padding: 10px; border: none;
                background-color: transparent; color: #aaa; font-weight: bold;
            }
            QPushButton:hover { background-color: #2c2c2c; color: white; }
            QPushButton:checked { background-color: #1976D2; color: white; border-left: 4px solid #64B5F6; }
        """









class SimulationViewWidget(QWidget):
    # --- SIGNALS ---
    sigPauseSimulation = pyqtSignal()
    sigStopSimulation = pyqtSignal()
    sigStartContinuous = pyqtSignal(float, float) # step_size, max_duration
    sigStepSimulation = pyqtSignal(float)
    sigResetSimulation = pyqtSignal() 
    
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        
        self.scene_loaded = False
        self.spike_tool_active = False
        self.is_paused = True 
        
        # Data Containers
        self.all_points = None      
        self.base_colors = None     
        self.current_colors = None  
        self.global_ids = None      
        self.gid_to_idx = {}        
        
        # Activity Arrays
        self.anim_rg = None         
        self.anim_b = None          
        self.heat = None            
        
        self.dynamic_generators = [] 
        self.generator_btns = []     
        
        self.stim_params = {
            'radius': 2.0,
            'weight_ex': 10.0,
            'weight_in': -10.0,
            'delay': 1.0,
            'model': 'spike_generator', 
            'rate': 500.0,
            'multiplicity': 1.0
        }
        
        self.point_size = 7.0       
        self.decay_flash = 0.60     
        self.decay_tail = 0.90      
        self.decay_heat = 0.80      
        
        self.view = None
        self.scatter_item = None
        
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.update_animation)
        
        self.init_ui()
        self.update_button_styles() 
        
        self._original_mouse_press = self.view.mousePressEvent
        self.view.mousePressEvent = self._wrapped_mouse_press

    # --- FIX: Fehlende Methode für sicheren Reset ---
    def stop_rendering_safe(self):
        """Stoppt Rendering und Timer sicher, bevor NEST resetet wird."""
        self.is_paused = True
        if hasattr(self, 'anim_timer'):
            self.anim_timer.stop()
        
        if self.scatter_item:
            try:
                self.scatter_item.setVisible(False)
            except: pass
    # ------------------------------------------------

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # === SIDEBAR ===
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("background-color: #232323; border-right: 1px solid #444;")
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(10)
        
        lbl_ctrl = QLabel("SIMULATION CONTROL")
        lbl_ctrl.setStyleSheet("color: #FF9800; font-weight: bold; font-size: 10px; letter-spacing: 1px;")
        sidebar_layout.addWidget(lbl_ctrl)

        # 1. Load / Clear
        row_load = QHBoxLayout()
        self.btn_load = QPushButton("⚡ LOAD SCENE")
        self.btn_load.clicked.connect(self.load_scene)
        self.btn_load.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold; padding: 6px;")
        
        self.btn_clear_scene = QPushButton("✕")
        self.btn_clear_scene.setFixedWidth(30)
        self.btn_clear_scene.setStyleSheet("background-color: #424242; color: white;")
        self.btn_clear_scene.clicked.connect(self.unload_scene)
        
        row_load.addWidget(self.btn_load)
        row_load.addWidget(self.btn_clear_scene)
        sidebar_layout.addLayout(row_load)
        
        # 2. Controls Grid
        sim_grid = QGridLayout()
        sim_grid.setSpacing(5)
        
        self.btn_start = QPushButton("▶ RUN")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self.action_start)
        
        self.btn_step = QPushButton("⏯ STEP")
        self.btn_step.setMinimumHeight(50)
        self.btn_step.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_step.clicked.connect(self.action_step)
        
        self.btn_pause = QPushButton("⏸ PAUSE")
        self.btn_pause.clicked.connect(self.action_pause)
        self.btn_pause.setStyleSheet("background-color: #424242; color: #ccc; font-weight: bold;")
        
        self.btn_reset = QPushButton("↺ RESET")
        self.btn_reset.clicked.connect(self.action_reset)
        self.btn_reset.setStyleSheet("background-color: #BF360C; color: white; font-weight: bold;")
        
        sim_grid.addWidget(self.btn_start, 0, 0)
        sim_grid.addWidget(self.btn_step, 0, 1)
        sim_grid.addWidget(self.btn_pause, 1, 0)
        sim_grid.addWidget(self.btn_reset, 1, 1)
        
        sidebar_layout.addLayout(sim_grid)
        
        # 3. Time Settings
        time_group = QGroupBox("Time Settings")
        time_layout = QFormLayout(time_group)
        
        self.spin_step_size = QDoubleSpinBox()
        self.spin_step_size.setRange(0.1, 1000.0); self.spin_step_size.setValue(25.0); self.spin_step_size.setSuffix(" ms")
        
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.0, 1000000.0); self.spin_duration.setValue(1000.0); self.spin_duration.setSuffix(" ms")
        
        time_layout.addRow("Step Size:", self.spin_step_size)
        time_layout.addRow("Max Time:", self.spin_duration)
        sidebar_layout.addWidget(time_group)
        
        # 4. Tool Settings
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); line.setStyleSheet("color: #444;")
        sidebar_layout.addWidget(line)
        
        self.btn_spike_tool = QPushButton("Inject Tool")
        self.btn_spike_tool.setCheckable(True)
        self.btn_spike_tool.setStyleSheet("""
            QPushButton { background-color: #333; color: #aaa; border: 1px solid #555; padding: 6px; font-weight: bold;}
            QPushButton:checked { background-color: #00E5FF; color: black; border: 1px solid #00E5FF; }
        """)
        self.btn_spike_tool.clicked.connect(self.toggle_spike_tool)
        sidebar_layout.addWidget(self.btn_spike_tool)
        
        self.tool_settings = QWidget()
        self.tool_settings.setVisible(False)
        tool_layout = QVBoxLayout(self.tool_settings)
        tool_layout.setContentsMargins(0,5,0,5)
        
        self.combo_model = QComboBox()
        self.combo_model.addItems(["spike_generator", "poisson_generator"])
        self.combo_model.currentTextChanged.connect(lambda t: self.stim_params.update({'model': t}))
        tool_layout.addWidget(self.combo_model)
        
        grid_params = QGridLayout()
        self.spin_w_ex = QDoubleSpinBox(); self.spin_w_ex.setRange(0, 10000); self.spin_w_ex.setValue(50.0)
        self.spin_w_ex.valueChanged.connect(lambda v: self.stim_params.update({'weight_ex': v}))
        
        self.spin_w_in = QDoubleSpinBox(); self.spin_w_in.setRange(-10000, 0); self.spin_w_in.setValue(-50.0)
        self.spin_w_in.valueChanged.connect(lambda v: self.stim_params.update({'weight_in': v}))
        
        grid_params.addWidget(QLabel("Ex:"), 0, 0); grid_params.addWidget(self.spin_w_ex, 0, 1)
        grid_params.addWidget(QLabel("In:"), 1, 0); grid_params.addWidget(self.spin_w_in, 1, 1)
        
        self.lbl_radius = QLabel(f"Radius: {self.stim_params['radius']:.1f}")
        self.slider_radius = QSlider(Qt.Orientation.Horizontal)
        self.slider_radius.setRange(1, 100); self.slider_radius.setValue(20)
        self.slider_radius.valueChanged.connect(self._update_radius)
        
        tool_layout.addLayout(grid_params)
        tool_layout.addWidget(self.lbl_radius)
        tool_layout.addWidget(self.slider_radius)
        sidebar_layout.addWidget(self.tool_settings)
        
        # 5. Generator List
        sidebar_layout.addWidget(QLabel("ACTIVE INJECTORS", styleSheet="color: #ccc; font-size:10px; font-weight:bold; margin-top:10px;"))
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: 1px solid #333; background-color: #1e1e1e;")
        
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(5)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        self.scroll_area.setWidget(self.grid_container)
        sidebar_layout.addWidget(self.scroll_area)
        
        self.btn_clear_gens = QPushButton("🗑 Clear All Injectors")
        self.btn_clear_gens.clicked.connect(self.clear_generators)
        self.btn_clear_gens.setStyleSheet("background-color: #5d4037; color: white;")
        sidebar_layout.addWidget(self.btn_clear_gens)
        
        # === VIEWPORT ===
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 40
        self.view.setBackgroundColor('#050505')
        
        layout.addWidget(sidebar)
        layout.addWidget(self.view)

    def update_button_styles(self):
        style_run_idle = "QPushButton { background-color: #2E7D32; color: #ccc; border: 1px solid #1B5E20; border-radius: 4px; font-weight: bold; font-size: 14px; } QPushButton:hover { background-color: #388E3C; color: white; }"
        style_run_active = "QPushButton { background-color: #00E676; color: black; border: 2px solid white; border-radius: 4px; font-weight: bold; font-size: 14px; box-shadow: 0px 0px 10px #00E676; }"
        style_step_idle = "QPushButton { background-color: #F57F17; color: #ccc; border: 1px solid #E65100; border-radius: 4px; font-weight: bold; font-size: 14px; } QPushButton:hover { background-color: #F9A825; color: white; }"
        style_step_ready = "QPushButton { background-color: #FFEA00; color: black; border: 2px solid white; border-radius: 4px; font-weight: bold; font-size: 14px; }"
        
        if not self.is_paused:
            self.btn_start.setStyleSheet(style_run_active); self.btn_start.setText("RUNNING")
            self.btn_step.setStyleSheet(style_step_idle); self.btn_step.setEnabled(False)
            self.btn_pause.setEnabled(True)
        else:
            self.btn_start.setStyleSheet(style_run_idle); self.btn_start.setText("▶ RUN")
            self.btn_step.setStyleSheet(style_step_ready); self.btn_step.setEnabled(True)
            self.btn_pause.setEnabled(False)

    def action_start(self):
        if not self.scene_loaded: return
        self.is_paused = False
        self.update_button_styles()
        self.sigStartContinuous.emit(self.spin_step_size.value(), self.spin_duration.value())

    def action_step(self):
        if not self.scene_loaded: return
        self.is_paused = True
        self.update_button_styles()
        self.sigStepSimulation.emit(self.spin_step_size.value())

    def action_pause(self):
        self.is_paused = True
        self.update_button_styles()
        self.sigPauseSimulation.emit()

    def action_reset(self):
        self.is_paused = True
        if self.anim_rg is not None: self.anim_rg[:] = 0; self.anim_b[:] = 0; self.heat[:] = 0
        self.update_animation()
        self.update_button_styles()
        self.sigResetSimulation.emit()

    def toggle_spike_tool(self, checked):
        self.tool_settings.setVisible(checked)

    def _update_radius(self, val):
        self.stim_params['radius'] = val / 10.0
        self.lbl_radius.setText(f"Radius: {self.stim_params['radius']:.1f}")

    def add_gen_button(self, data):
        idx = len(self.generator_btns)
        btn = QPushButton()
        btn.setFixedSize(75, 45)
        mode_char = "E" if data['type'] == "excitatory" else "I"
        count = len(data['targets'])
        btn.setText(f"{mode_char} #{idx+1}\n({count})")
        
        c = "#00E5FF" if data['type'] == "excitatory" else "#FF5252"
        btn.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {c}; color: black; font-weight: 900;
                border: 2px solid #222; border-radius: 8px; 
                font-size: 10px;
            }}
            QPushButton:hover {{ border: 2px solid white; }}
            QPushButton:pressed {{ background-color: white; }}
        """)
        btn.setToolTip(f"{data['model']} (ID: {data['gid']})\nTargets: {count}\nPos: {data['pos']}")
        btn.clicked.connect(lambda: self.inject_stimulus(data))
        
        row = idx // 3
        col = idx % 3
        self.grid_layout.addWidget(btn, row, col)
        self.generator_btns.append(btn)

    def load_scene(self):
        if self.scene_loaded: self.unload_scene()
        print("Loading Scene...")
        points = []; colors = []; gids_list = []
        for graph in self.graph_list:
            for node in graph.node_list:
                if hasattr(node, 'positions') and node.positions:
                    for pop_idx, cluster in enumerate(node.positions):
                        if cluster is None or len(cluster) == 0: continue
                        model = node.neuron_models[pop_idx] if hasattr(node, 'neuron_models') and pop_idx < len(node.neuron_models) else "unknown"
                        hex_c = neuron_colors.get(model, "#ffffff")
                        rgb = mcolors.to_rgba(hex_c, alpha=0.6)
                        count = len(cluster)
                        points.append(cluster)
                        colors.append(np.tile(rgb, (count, 1)))
                        if hasattr(node, 'population') and len(node.population) > pop_idx:
                            pop_ids = nest.GetStatus(node.population[pop_idx], 'global_id')
                            if len(pop_ids) == count: gids_list.append(np.array(pop_ids))
                            else: gids_list.append(np.zeros(count, dtype=int))
                        else: gids_list.append(np.zeros(count, dtype=int))
        if not points: return
        self.all_points = np.vstack(points)
        self.base_colors = np.vstack(colors)
        self.global_ids = np.concatenate(gids_list)
        self.gid_to_idx = {gid: i for i, gid in enumerate(self.global_ids) if gid > 0}
        self.current_colors = self.base_colors.copy()
        N = len(self.all_points)
        self.anim_rg = np.zeros(N); self.anim_b = np.zeros(N); self.heat = np.zeros(N)
        self.scatter_item = gl.GLScatterPlotItem(pos=self.all_points, color=self.current_colors, size=self.point_size, pxMode=True)
        self.scatter_item.setGLOptions('translucent')
        self.view.addItem(self.scatter_item)
        self.scene_loaded = True
        self.anim_timer.start(30)
        self.update_button_styles()

    def unload_scene(self):
        self.anim_timer.stop()
        if self.scatter_item and self.view:
            try: self.view.removeItem(self.scatter_item)
            except: pass
        self.scatter_item = None
        self.all_points = None
        self.gid_to_idx = {}
        self.scene_loaded = False

    def feed_spikes(self, spike_gids):
        if not self.scene_loaded: return
        indices = [self.gid_to_idx[gid] for gid in spike_gids if gid in self.gid_to_idx]
        if not indices: return
        self.anim_rg[indices] = 1.0; self.anim_b[indices] = 1.0; self.heat[indices] += 1.0

    def update_animation(self):
        if not self.scene_loaded or self.scatter_item is None: return
        self.anim_rg *= self.decay_tail; self.anim_b *= self.decay_flash; self.heat *= self.decay_heat
        if np.max(self.anim_rg) < 0.01: return
        new_c = self.base_colors.copy()
        mask_active = self.anim_rg > 0.05
        if np.any(mask_active):
            val_rg = self.anim_rg[mask_active, np.newaxis]
            new_c[mask_active, 0] = np.minimum(1.0, new_c[mask_active, 0] + val_rg.flatten())
            new_c[mask_active, 1] = np.minimum(1.0, new_c[mask_active, 1] + val_rg.flatten())
            val_b = self.anim_b[mask_active]
            new_c[mask_active, 2] = np.minimum(1.0, new_c[mask_active, 2] + val_b)
            new_c[mask_active, 3] = np.minimum(1.0, 0.4 + val_rg.flatten() * 0.6)
        mask_burst = self.heat > 8.0 
        if np.any(mask_burst): new_c[mask_burst] = [0.0, 0.8, 1.0, 1.0]
        self.scatter_item.setData(color=new_c)

    def _project_points(self):
        if self.all_points is None: return None, None
        view_m = self.view.viewMatrix(); proj_m = self.view.projectionMatrix()
        mvp = proj_m * view_m
        mat_data = np.array(mvp.data()).reshape(4, 4)
        N = len(self.all_points)
        points_4d = np.hstack([self.all_points, np.ones((N, 1))])
        clip = points_4d @ mat_data
        return clip, clip[:, 3] > 0.001

    def _get_nearest_neuron_idx(self, mouse_pos):
        clip, valid = self._project_points()
        if clip is None: return None, None
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0: return None, None
        v_clip = clip[valid_idx]
        ndc = v_clip[:, :3] / v_clip[:, 3][:, np.newaxis]
        w, h = self.view.width(), self.view.height()
        sx = (ndc[:, 0] + 1) * w / 2; sy = (1 - ndc[:, 1]) * h / 2
        dist_sq = (sx - mouse_pos.x())**2 + (sy - mouse_pos.y())**2
        min_loc = np.argmin(dist_sq)
        if dist_sq[min_loc] < 900: return valid_idx[min_loc], self.all_points[valid_idx[min_loc]]
        return None, None

    def _wrapped_mouse_press(self, event):
        self._original_mouse_press(event)
        if not self.scene_loaded or not self.spike_tool_active: return
        idx, center = self._get_nearest_neuron_idx(event.pos())
        if center is not None:
            mode = "excitatory" if event.button() == Qt.MouseButton.LeftButton else "inhibitory"
            self.create_injector(center, mode)

    def _find_targets(self, pos, radius):
        if self.all_points is None: return []
        diff = self.all_points - pos
        dist_sq = np.sum(diff**2, axis=1)
        hits = np.where(dist_sq <= radius**2)[0]
        if len(hits) == 0: return []
        targets = self.global_ids[hits].tolist()
        return [g for g in targets if g > 0]

    def create_injector(self, center_pos, mode):
        r = self.stim_params['radius']
        targets = self._find_targets(center_pos, r)
        if not targets: return
        
        was_running = not self.is_paused
        if was_running:
            self.sigPauseSimulation.emit()
            QApplication.processEvents()
        
        model = self.stim_params['model']
        try: status = nest.GetKernelStatus(); current_time = status.get('time', 0.0)
        except: current_time = 0.0
        
        params = {}
        if model == 'spike_generator':
            spike_time = current_time + self.stim_params['delay'] + 1.0
            params = {'spike_times': [spike_time], 'spike_weights': [float(self.stim_params['multiplicity'])]}
        elif model == 'poisson_generator':
            params = {'rate': self.stim_params['rate'], 'start': current_time}
        
        try:
            gen_id = nest.Create(model, params=params)
            w = self.stim_params['weight_ex'] if mode == "excitatory" else self.stim_params['weight_in']
            d = max(self.stim_params['delay'], 0.1)
            nest.Connect(gen_id, targets, syn_spec={'weight': w, 'delay': d})
            
            gen_data = {
                'gid': gen_id, 
                'model': model, 'targets': targets, 'pos': center_pos, 
                'radius': r, 'weight': w, 'delay': d, 'type': mode, 'params': params
            }
            self.dynamic_generators.append(gen_data)
            self.add_gen_button(gen_data)
            self.feed_spikes(targets)
            print(f"Injector created: {model} -> {len(targets)} targets")
        except Exception as e: print(f"Injector Error: {e}")
        
        if was_running: self.action_start()

    def inject_stimulus(self, data):
        gid = data['gid']
        model = data['model']
        try: status = nest.GetKernelStatus(); current_time = status.get('time', 0.0)
        except: current_time = 0.0
        spike_time = current_time + data['delay'] + 1.0 
        
        try:
            if model == 'spike_generator':
                # Handle NodeCollection properly
                if isinstance(gid, int):
                     nest.SetStatus([gid], {'spike_times': [spike_time], 'spike_weights': [1.0]})
                else:
                     nest.SetStatus(gid, {'spike_times': [spike_time], 'spike_weights': [1.0]})

            self.feed_spikes(data['targets'])
            print(f"Manual Injection at {spike_time}ms")
        except Exception as e: print(f"Injection Failed: {e}")

    def clear_generators(self):
        for gen in self.dynamic_generators:
            try:
                gid = gen['gid']
                if gen['model'] == 'spike_generator': nest.SetStatus(gid, {'spike_times': []})
                elif gen['model'] == 'poisson_generator': nest.SetStatus(gid, {'rate': 0.0})
            except: pass
        self.dynamic_generators.clear()
        for btn in self.generator_btns: self.grid_layout.removeWidget(btn); btn.deleteLater()
        self.generator_btns.clear()

    def restore_injectors(self):
        if not self.dynamic_generators: return
        print("Restoring injectors spatially...")
        old_list = list(self.dynamic_generators) 
        self.dynamic_generators = []
        for btn in self.generator_btns: 
            self.grid_layout.removeWidget(btn)
            btn.deleteLater()
        self.generator_btns.clear()
        
        for old in old_list:
            try:
                new_targets = self._find_targets(old['pos'], old['radius'])
                if not new_targets: continue
                
                new_id = nest.Create(old['model'], params=old['params'])
                nest.Connect(new_id, new_targets, syn_spec={'weight': old['weight'], 'delay': old['delay']})
                
                new_data = old.copy()
                new_data['gid'] = new_id
                new_data['targets'] = new_targets
                
                self.dynamic_generators.append(new_data)
                self.add_gen_button(new_data)
            except Exception as e:
                print(f"Failed to restore injector: {e}")














class SimulationDashboardWidget(QWidget):
    """
    Zentrale Steuereinheit für die Headless-Simulation.
    """
    requestStartSimulation = pyqtSignal(float) 
    requestStopSimulation = pyqtSignal()
    requestResetKernel = pyqtSignal()

    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("HEADLESS SIMULATION DASHBOARD")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF9800; letter-spacing: 2px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        content_layout = QHBoxLayout()
        
        # --- LEFT COLUMN: CONFIG ---
        left_col = QVBoxLayout()
        
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout(config_group)
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 10000000.0) # Großzügiges Limit
        self.duration_spin.setValue(1000.0)
        self.duration_spin.setSuffix(" ms")
        self.duration_spin.setStyleSheet("font-size: 14px; padding: 5px; color: #00E5FF; font-weight: bold;")
        
        self.input_placeholder = QLineEdit("No external input file selected")
        self.input_placeholder.setReadOnly(True)
        self.input_placeholder.setStyleSheet("color: #888; font-style: italic;")
        
        config_layout.addRow("Duration:", self.duration_spin)
        config_layout.addRow("Input:", self.input_placeholder)
        
        left_col.addWidget(config_group)
        
        # Stats Group
        stats_group = QGroupBox("Network Status")
        self.stats_layout = QFormLayout(stats_group)
        self.lbl_nodes = QLabel("0"); self.lbl_nodes.setStyleSheet("color: #00E5FF; font-weight:bold;")
        self.lbl_neurons = QLabel("0"); self.lbl_neurons.setStyleSheet("color: #00E5FF; font-weight:bold;")
        self.lbl_conns = QLabel("0"); self.lbl_conns.setStyleSheet("color: #00E5FF; font-weight:bold;")
        
        self.stats_layout.addRow("Nodes:", self.lbl_nodes)
        self.stats_layout.addRow("Total Neurons:", self.lbl_neurons)
        self.stats_layout.addRow("Connections:", self.lbl_conns)
        
        left_col.addWidget(stats_group)
        left_col.addStretch()
        content_layout.addLayout(left_col, 1)

        # --- RIGHT COLUMN: CHART ---
        right_col = QVBoxLayout()
        chart_group = QGroupBox("Model Distribution")
        chart_layout = QVBoxLayout(chart_group)
        
        self.figure = Figure(figsize=(4, 4), facecolor='#2b2b2b')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #2b2b2b;")
        chart_layout.addWidget(self.canvas)
        
        right_col.addWidget(chart_group)
        content_layout.addLayout(right_col, 1)

        main_layout.addLayout(content_layout)

        # --- ACTION BUTTONS ---
        action_container = QWidget()
        action_layout = QHBoxLayout(action_container) # Horizontal für Start/Stop/Reset
        action_layout.setSpacing(15)
        
        # Start Button
        self.btn_start = QPushButton("▶ RUN HEADLESS")
        self.btn_start.setMinimumHeight(60)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32; color: white; font-weight: bold; font-size: 16px; 
                border-radius: 8px; border-bottom: 4px solid #1B5E20;
            }
            QPushButton:hover { background-color: #388E3C; margin-top: 2px; border-bottom: 2px solid #1B5E20; }
            QPushButton:pressed { background-color: #1B5E20; margin-top: 4px; border-bottom: none; }
            QPushButton:disabled { background-color: #444; border: 1px solid #555; color: #888; }
        """)
        self.btn_start.clicked.connect(self._on_start_clicked)
        
        # Stop Button
        self.btn_stop = QPushButton("⏹ STOP")
        self.btn_stop.setMinimumHeight(60)
        self.btn_stop.setEnabled(False) # Initial aus
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #C62828; color: white; font-weight: bold; font-size: 16px; 
                border-radius: 8px; border-bottom: 4px solid #B71C1C;
            }
            QPushButton:hover { background-color: #D32F2F; margin-top: 2px; border-bottom: 2px solid #B71C1C; }
            QPushButton:pressed { background-color: #B71C1C; margin-top: 4px; border-bottom: none; }
            QPushButton:disabled { background-color: #444; border: 1px solid #555; color: #888; }
        """)
        self.btn_stop.clicked.connect(self.requestStopSimulation.emit)
        
        # Reset Button
        self.btn_reset = QPushButton("↺ RESET KERNEL")
        self.btn_reset.setMinimumHeight(60)
        self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #E65100; color: white; font-weight: bold; font-size: 16px; 
                border-radius: 8px; border-bottom: 4px solid #BF360C;
            }
            QPushButton:hover { background-color: #EF6C00; margin-top: 2px; border-bottom: 2px solid #BF360C; }
            QPushButton:pressed { background-color: #BF360C; margin-top: 4px; border-bottom: none; }
            QPushButton:disabled { background-color: #444; border: 1px solid #555; color: #888; }
        """)
        self.btn_reset.clicked.connect(self.requestResetKernel.emit)

        action_layout.addWidget(self.btn_start, 2)
        action_layout.addWidget(self.btn_stop, 1)
        action_layout.addWidget(self.btn_reset, 1)
        
        main_layout.addWidget(action_container)

    def _on_start_clicked(self):
        duration = self.duration_spin.value()
        self.requestStartSimulation.emit(duration)

    def set_ui_locked(self, locked):
        """Sperrt/Entsperrt UI-Elemente während der Simulation."""
        self.btn_start.setEnabled(not locked)
        self.btn_reset.setEnabled(not locked)
        self.duration_spin.setEnabled(not locked)
        
        self.btn_stop.setEnabled(locked) # Stop nur aktiv wenn läuft
        
        if locked:
            self.btn_start.setText("Running...")
            self.btn_start.setStyleSheet("background-color: #444; color: #888; border: 1px solid #555; border-radius: 8px;")
        else:
            self.btn_start.setText("▶ RUN HEADLESS")
            self.btn_start.setStyleSheet("""
                QPushButton {
                    background-color: #2E7D32; color: white; font-weight: bold; font-size: 16px; 
                    border-radius: 8px; border-bottom: 4px solid #1B5E20;
                }
                QPushButton:hover { background-color: #388E3C; margin-top: 2px; border-bottom: 2px solid #1B5E20; }
                QPushButton:pressed { background-color: #1B5E20; margin-top: 4px; border-bottom: none; }
            """)

    def refresh_data(self):
        # ... (Identisch zum vorherigen Code für PieChart und Labels) ...
        # Nur Labels updaten
        total_nodes = 0
        total_neurons = 0
        total_conns = 0
        model_counts = {}

        for graph in self.graph_list:
            total_nodes += len(graph.node_list)
            for node in graph.node_list:
                if hasattr(node, 'connections'): total_conns += len(node.connections)
                if hasattr(node, 'population') and node.population:
                    for i, nest_pop in enumerate(node.population):
                        if nest_pop:
                            count = len(nest_pop)
                            total_neurons += count
                            model_name = node.neuron_models[i] if hasattr(node, 'neuron_models') and i < len(node.neuron_models) else "unknown"
                            model_counts[model_name] = model_counts.get(model_name, 0) + count

        self.lbl_nodes.setText(str(total_nodes))
        self.lbl_neurons.setText(str(total_neurons))
        self.lbl_conns.setText(str(total_conns))
        self._plot_pie_chart(model_counts)

    def _plot_pie_chart(self, counts):
        self.figure.clear()
        if not counts:
            ax = self.figure.add_subplot(111); ax.text(0.5, 0.5, "No Data", ha='center', color='white'); ax.axis('off')
            self.canvas.draw(); return

        labels = list(counts.keys()); sizes = list(counts.values())
        colors = [neuron_colors.get(l, "#888") for l in labels]
        ax = self.figure.add_subplot(111)
        wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, colors=colors, textprops=dict(color="white"), pctdistance=0.85)
        for t in autotexts: t.set_color("black"); t.set_weight("bold")
        ax.legend(wedges, labels, title="Models", loc="center left", bbox_to_anchor=(0.9, 0, 0.5, 1), frameon=False, labelcolor='#ccc')
        self.figure.patch.set_facecolor('#2b2b2b'); self.figure.subplots_adjust(left=0, bottom=0, right=0.75, top=1)
        self.canvas.draw()





def build_gid_to_meta_map(graph_list):
    gid_map = {}
    
    for graph in graph_list:
        for node in graph.node_list:
            if hasattr(node, 'population') and node.population:
                for pop_idx, pop in enumerate(node.population):
                    if pop:
                        for local_idx, gid in enumerate(pop.tolist()):
                            gid_map[gid] = {
                                'graph_id': graph.graph_id,
                                'node_id': node.id,
                                'node_name': getattr(node, 'name', 'unnamed'),
                                'pop_idx': pop_idx,
                                'local_idx': local_idx
                            }
    return gid_map

class GraphInfoWidget(QWidget):
    """
    Zeigt statische Graphen-Statistiken.
    Layout: Dark Mode.
    Plots: Light Mode (Scientific Style).
    """
    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Selector
        sel_layout = QHBoxLayout()
        lbl = QLabel("Select Graph:")
        lbl.setStyleSheet("color: white;") # Label weiß für Darkmode GUI
        sel_layout.addWidget(lbl)
        
        self.combo = QComboBox()
        self.combo.setStyleSheet("background-color: #333; color: white; border: 1px solid #555;")
        self.combo.currentIndexChanged.connect(self.generate_report)
        sel_layout.addWidget(self.combo)
        layout.addLayout(sel_layout)
        
        # Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("background-color: #1e1e1e; border: none;") # Hintergrund dunkel
        
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background-color: #1e1e1e;") 
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(20)
        
        self.scroll.setWidget(self.content_widget)
        layout.addWidget(self.scroll)
        
        self.refresh_combo()

    def refresh_combo(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        for g in self.graph_list:
            name = getattr(g, 'graph_name', f"Graph {g.graph_id}")
            self.combo.addItem(f"{name} (ID: {g.graph_id})", g)
        self.combo.blockSignals(False)
        if self.combo.count() > 0:
            self.generate_report()

    def generate_report(self):
        # Cleanup
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        
        graph = self.combo.currentData()
        if not graph: return

        # --- HEADER INFO (Dark Mode Style) ---
        total_nodes = len(graph.node_list)
        total_conns = sum(len(n.connections) for n in graph.node_list if hasattr(n, 'connections'))
        
        info_group = QFrame()
        info_group.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444; border-radius: 5px; padding: 15px;")
        ig_layout = QVBoxLayout(info_group)
        
        lbl_name = QLabel(getattr(graph, 'graph_name', 'Unknown Graph'))
        lbl_name.setStyleSheet("font-size: 20px; font-weight: bold; color: #00E5FF;")
        lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_stats = QLabel(f"Nodes: {total_nodes}  |  Connections: {total_conns}")
        lbl_stats.setStyleSheet("font-size: 14px; color: #ccc; margin-top: 5px;")
        lbl_stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        ig_layout.addWidget(lbl_name)
        ig_layout.addWidget(lbl_stats)
        self.content_layout.addWidget(info_group)

        # --- GRAPH SKELETON (Light Plot) ---
        skel_container = QFrame()
        skel_container.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444; border-radius: 5px;")
        skel_layout = QVBoxLayout(skel_container)
        skel_layout.addWidget(QLabel("Graph Topology", styleSheet="color: #FF9800; font-weight: bold; font-size: 14px; padding: 5px;"))
        
        # White Figure -> "Paper Style"
        fig_skel = Figure(figsize=(5, 5), facecolor='white')
        canvas_skel = FigureCanvas(fig_skel)
        ax_skel = fig_skel.add_subplot(111, projection='3d')
        
        # Hintergrund und Achsen Standard (Weiß/Grau/Schwarz)
        # Wir setzen NICHTS auf schwarz oder dunkelgrau
        
        centers = np.array([n.center_of_mass for n in graph.node_list])
        if len(centers) > 0:
            # Nodes: Dunkelrot für guten Kontrast auf Weiß
            ax_skel.scatter(centers[:,0], centers[:,1], centers[:,2], c='darkred', s=50, edgecolors='black', alpha=0.8)
            
            # Edges: Dunkelgrau
            for node in graph.node_list:
                start = node.center_of_mass
                if hasattr(node, 'next'):
                    for neighbor in node.next:
                        end = neighbor.center_of_mass
                        ax_skel.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c='#333333', alpha=0.6, linewidth=1.5)
        
        # Standard Labels (Schwarz auf Weiß)
        ax_skel.set_xlabel('X')
        ax_skel.set_ylabel('Y')
        ax_skel.set_zlabel('Z')
        
        skel_layout.addWidget(canvas_skel)
        self.content_layout.addWidget(skel_container)

        # --- SUBPOPULATIONS GRID (Light Plot) ---
        if total_nodes > 0:
            grid_container = QFrame()
            grid_container.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444; border-radius: 5px;")
            gc_layout = QVBoxLayout(grid_container)
            gc_layout.addWidget(QLabel("Node Populations", styleSheet="color: #E91E63; font-weight: bold; font-size: 14px; padding: 5px;"))
            
            cols = 4
            rows = int(np.ceil(total_nodes / cols))
            fig_height = max(4, rows * 3) 
            
            # Große weiße Figur
            fig_grid = Figure(figsize=(10, fig_height), facecolor='white')
            canvas_grid = FigureCanvas(fig_grid)
            
            for i, node in enumerate(graph.node_list):
                ax = fig_grid.add_subplot(rows, cols, i+1, projection='3d')
                # Titel Schwarz
                ax.set_title(f"Node {node.id}: {node.name}", color='black', fontsize=9, fontweight='bold')
                
                if hasattr(node, 'positions') and node.positions:
                    for j, cluster in enumerate(node.positions):
                        if cluster is not None and len(cluster) > 0:
                            cmap = plt.get_cmap("tab10")
                            color = cmap(j % 10)
                            # Punkte
                            ax.scatter(cluster[:,0], cluster[:,1], cluster[:,2], color=color, s=2, alpha=0.6)
                
                # Achsen ausblenden für cleanen Look, aber Box behalten
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

            fig_grid.tight_layout()
            gc_layout.addWidget(canvas_grid)
            self.content_layout.addWidget(grid_container)







class TimeSeriesPlotWidget(QWidget):

    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Controls (Dark Mode Style) ---
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444; border-radius: 4px;")
        ctrl_layout = QHBoxLayout(ctrl_frame)
        
        lbl_style = "color: #ddd; font-weight: bold;"
        combo_style = "background-color: #1e1e1e; color: white; border: 1px solid #555; padding: 2px;"
        
        self.combo_graph = QComboBox()
        self.combo_graph.setStyleSheet(combo_style)
        self.combo_node = QComboBox()
        self.combo_node.setStyleSheet(combo_style)
        self.combo_device = QComboBox()
        self.combo_device.setStyleSheet(combo_style)
        
        self.btn_plot = QPushButton("Plot Data")
        self.btn_plot.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; border-radius: 3px; padding: 5px 15px;")
        
        ctrl_layout.addWidget(QLabel("Graph:", styleSheet=lbl_style))
        ctrl_layout.addWidget(self.combo_graph)
        ctrl_layout.addWidget(QLabel("Node:", styleSheet=lbl_style))
        ctrl_layout.addWidget(self.combo_node)
        ctrl_layout.addWidget(QLabel("Device:", styleSheet=lbl_style))
        ctrl_layout.addWidget(self.combo_device)
        ctrl_layout.addWidget(self.btn_plot)
        ctrl_layout.addStretch()
        
        layout.addWidget(ctrl_frame)
        
        # --- Plot Area (Light Mode) ---
        # Figure facecolor='white' macht den Hintergrund weiß
        self.figure = Figure(figsize=(8, 6), facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        
        # Canvas expandieren lassen
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setSizePolicy(sizePolicy)
        
        layout.addWidget(self.canvas)
        
        # Events
        self.combo_graph.currentIndexChanged.connect(self.on_graph_changed)
        self.combo_node.currentIndexChanged.connect(self.on_node_changed)
        self.btn_plot.clicked.connect(self.plot_data)
        
        self.refresh_graphs()

    def refresh_graphs(self):
        self.combo_graph.blockSignals(True)
        self.combo_graph.clear()
        for g in self.graph_list:
            name = getattr(g, 'graph_name', f"Graph {g.graph_id}")
            self.combo_graph.addItem(f"{name} (ID: {g.graph_id})", g.graph_id)
        self.combo_graph.blockSignals(False)
        self.on_graph_changed()

    def on_graph_changed(self):
        self.combo_node.blockSignals(True)
        self.combo_node.clear()
        gid = self.combo_graph.currentData()
        
        graph = next((g for g in self.graph_list if g.graph_id == gid), None)
        if graph:
            for node in graph.node_list:
                self.combo_node.addItem(f"{node.name} (ID: {node.id})", node)
        
        self.combo_node.blockSignals(False)
        self.on_node_changed()

    def on_node_changed(self):
        self.combo_device.clear()
        node = self.combo_node.currentData()
        if not node: return
        
        known_devices = set()
        
        if hasattr(node, 'devices'):
            for d in node.devices:
                dev_id = d.get('id')
                model = d.get('model', 'unknown')
                known_devices.add((dev_id, model))
        
        if hasattr(node, 'results') and 'history' in node.results:
            for entry in node.results['history']:
                if 'devices' in entry:
                    for dev_id_str, info in entry['devices'].items():
                        try:
                            did = int(dev_id_str)
                            model = info.get('model', 'unknown')
                            known_devices.add((did, model))
                        except: pass
        
        for did, model in sorted(known_devices):
            self.combo_device.addItem(f"Dev {did}: {model}", did)

    def plot_data(self):
        self.figure.clear()
        # Weißer Subplot
        ax = self.figure.add_subplot(111)
        
        node = self.combo_node.currentData()
        dev_id = self.combo_device.currentData()
        
        if not node or dev_id is None:
            ax.text(0.5, 0.5, "No device selected", ha='center', color='black')
            self.canvas.draw()
            return
            
        if not hasattr(node, 'results') or 'history' not in node.results or not node.results['history']:
            ax.text(0.5, 0.5, "No history data available", ha='center', color='black')
            self.canvas.draw()
            return

        all_times = []
        all_senders = [] 
        all_values = {}  
        
        found_any = False
        device_type = "unknown"
        
        for run in node.results['history']:
            dev_data = run['devices'].get(dev_id) or run['devices'].get(str(dev_id))
            
            if not dev_data or 'data' not in dev_data: continue
            
            found_any = True
            raw = dev_data['data']
            
            if 'times' in raw: all_times.extend(raw['times'])
            if 'senders' in raw:
                all_senders.extend(raw['senders'])
                device_type = "spike_recorder"
            
            for k, v in raw.items():
                if k not in ['times', 'senders']:
                    if k not in all_values: all_values[k] = []
                    all_values[k].extend(v)
                    device_type = "meter"

        if not found_any or not all_times:
            ax.text(0.5, 0.5, "No data in history", ha='center', color='black')
            self.canvas.draw()
            return

        times = np.array(all_times)
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        
        if device_type == "spike_recorder":
            senders = np.array(all_senders)[sort_idx]
            # Schwarze Punkte für Spikes
            ax.scatter(times, senders, s=5, c='black', marker='|')
            ax.set_ylabel("Neuron ID", color='black')
            ax.set_title(f"Raster Plot (Dev {dev_id})", color='black')
            
        elif device_type == "meter":
            for key, vals in all_values.items():
                vals = np.array(vals)[sort_idx]
                ax.plot(times, vals, label=key, linewidth=1.5)
            
            ax.set_ylabel("Value", color='black')
            # Legende mit weißem Hintergrund und schwarzem Rahmen
            ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
            ax.set_title(f"Analog Signals (Dev {dev_id})", color='black')
            ax.grid(True, linestyle='--', alpha=0.5, color='gray')
            
        ax.set_xlabel("Time (ms)", color='black')
        
        # Achsenfarben auf Schwarz setzen
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        
        self.figure.tight_layout()
        self.canvas.draw()




class PythonConsoleWidget(QWidget):
    def __init__(self, context_vars=None, parent=None):
        super().__init__(parent)
        # Kontext für den Interpreter (Zugriff auf graph_list, nest, etc.)
        self.local_vars = context_vars if context_vars is not None else {}
        self.history = []
        self.history_idx = 0
        
        self.init_ui()
        
        # Initialisiert den Python-Interpreter im Kontext
        self.interpreter = code.InteractiveConsole(self.local_vars)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Output Area
        self.output_view = QTextEdit()
        self.output_view.setReadOnly(True)
        self.output_view.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00FF00;
                font-family: Consolas, Monospace;
                font-size: 12px;
                border: 1px solid #444;
            }
        """)
        
        # Input Line
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText(">>> Enter Python command...")
        self.input_line.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: Consolas, Monospace;
                font-size: 12px;
                border: 1px solid #555;
                padding: 4px;
            }
        """)
        self.input_line.returnPressed.connect(self.execute_command)
        self.input_line.installEventFilter(self) # Für History (Pfeil hoch/runter)

        layout.addWidget(self.output_view)
        layout.addWidget(self.input_line)
        
        self.write_output("Python Console initialized.\nContext available: 'nest', 'graph_list', 'np'\n")

    def write_output(self, text):
        # Hier lag der NameError: QtGui muss importiert sein
        self.output_view.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.output_view.insertPlainText(text)
        self.output_view.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def execute_command(self):
        command = self.input_line.text()
        self.input_line.clear()
        
        if not command.strip():
            return

        self.write_output(f">>> {command}\n")
        self.history.append(command)
        self.history_idx = len(self.history)

        # Stdout Capture: Print-Ausgaben abfangen und ins GUI umleiten
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = mystdout = StringIO()
        sys.stderr = mystderr = StringIO()

        try:
            # Versuche erst eval() für Expressions (damit man Resultate sieht ohne print)
            try:
                result = eval(command, self.local_vars)
                if result is not None:
                    print(repr(result))
            except SyntaxError:
                # Bei Statements (z.B. a = 1) exec via runsource nutzen
                self.interpreter.runsource(command)
            except Exception:
                # Runtime Errors im eval auch an runsource übergeben (für Tracebacks)
                self.interpreter.runsource(command)
                
        except Exception:
            self.interpreter.runsource(command)
        
        finally:
            # Stdout wiederherstellen
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        output = mystdout.getvalue()
        error = mystderr.getvalue()

        if output: self.write_output(output)
        if error: self.write_output(error)

    def eventFilter(self, obj, event):
        # Hier lag der AttributeError: In PyQt6 ist es QEvent.Type.KeyPress
        if obj == self.input_line and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Up:
                if self.history_idx > 0:
                    self.history_idx -= 1
                    self.input_line.setText(self.history[self.history_idx])
                return True
            elif key == Qt.Key.Key_Down:
                if self.history_idx < len(self.history) - 1:
                    self.history_idx += 1
                    self.input_line.setText(self.history[self.history_idx])
                else:
                    self.history_idx = len(self.history)
                    self.input_line.clear()
                return True
        return super().eventFilter(obj, event)





































class ToolInspectorWidget(QWidget):
    """
    Der Rohling für die Analyse von Tools (Devices).
    Iteriert durch alle Graphen -> Nodes -> Devices.
    """
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Header ---
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444; border-radius: 4px;")
        header_layout = QHBoxLayout(header_frame)
        
        lbl_title = QLabel("TOOL INSPECTOR (BATCH PROCESSING)")
        lbl_title.setStyleSheet("color: #FFC107; font-weight: bold; font-size: 14px;")
        
        btn_run = QPushButton("▶ Run Processing")
        btn_run.setStyleSheet("background-color: #FFC107; color: black; font-weight: bold; padding: 5px 15px;")
        btn_run.clicked.connect(self.run_inspector)
        
        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(btn_run)
        
        layout.addWidget(header_frame)
        
        # --- Output Log ---
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Output log will appear here...")
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e; 
                color: #00E5FF; 
                font-family: Consolas, Monospace; 
                font-size: 12px;
                border: 1px solid #444;
            }
        """)
        layout.addWidget(self.log_output)

    def log(self, message):
        self.log_output.append(message)
        # Scroll to bottom
        sb = self.log_output.verticalScrollBar()
        sb.setValue(sb.maximum())


    def run_inspector(self):
        """
        Iteriert durch die Struktur und prüft Tools/Devices.
        """
        self.log_output.clear()
        self.log(">>> Starting Inspector...")
        
        tools_found = 0
        
        for graph in self.graph_list:
            # self.log(f"Graph {graph.graph_id}...")
            
            for node in graph.node_list:
                
                # Devices können in node.devices (Runtime) oder node.parameters (Config) sein
                # Wir nehmen node.devices bevorzugt, wenn vorhanden
                devices = getattr(node, 'devices', [])
                if not devices and hasattr(node, 'parameters') and 'devices' in node.parameters:
                    devices = node.parameters['devices']
                
                for tool in devices:
                    # 'tool' ist ein Dictionary, z.B. {'model': 'spike_recorder', 'params': ...}
                    
                    # ----------------------------------------------------
                    # HIER IST DER PLATZHALTER FÜR DEINE LOGIK
                    # ----------------------------------------------------
                    
                    pass
                    
                    # ----------------------------------------------------
                    
                    # Optional: Loggen, dass wir hier waren (kannst du auskommentieren)
                    model = tool.get('model', 'unknown')
                    node_name = getattr(node, 'name', f"Node_{node.id}")
                    # self.log(f"  [Pass] Processed {model} on {node_name}")
                    
                    tools_found += 1

        self.log(f"\n>>> Done. Iterated over {tools_found} tools/devices.")














class LivePlotWidget(QWidget):
    """
    Einzelner Plot für ein Device (im Detail-Bereich).
    Puffert Daten lokal, um das NEST-Queue-Limit zu umgehen.
    """
    def __init__(self, device_id, device_type, params, node_info, max_points=100000, parent=None):
        super().__init__(parent)
        self.device_id = device_id
        self.device_type = device_type
        self.params = params
        self.node_info = node_info 
        self.nest_gid = None 
        
        # Puffer-Einstellungen
        self.max_points = max_points
        self.global_time_offset = 0.0 
        self.last_recorded_time = 0.0 # FIX: Initialisierung
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Header Zeile (Mini)
        dev_name = params.get('label', device_type)
        if "recorder" in device_type: dev_name = "Spikes"
        elif "meter" in device_type: dev_name = "Analog"
        
        # Farbe aus Node-Info
        c_hex = self.node_info.get('color', '#FFFFFF')
        
        lbl = QLabel(f"<span style='color:{c_hex}'><b>{dev_name}</b></span> <span style='color:#666'>(#{device_id})</span>")
        lbl.setStyleSheet("font-size: 9pt; margin-left: 4px;")
        self.layout.addWidget(lbl)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setBackground(None) # Transparent
        self.plot_widget.setMinimumHeight(120)
        
        # Interaktion: Pan/Zoom erlauben, aber Standard ist Auto-Scroll
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.layout.addWidget(self.plot_widget)
        
        # Datenspeicher (Listen sind performant genug für append)
        self.times = []
        self.senders = []      # Nur für Spike Recorder
        self.values = {}       # Nur für Multimeter {var_name: []}
        
        self.scatter_item = None
        self.curves = {}
        
        self._init_plot_type()

    def _init_plot_type(self):
        c = QColor(self.node_info.get('color', '#FFFFFF'))
        
        if "spike_recorder" in self.device_type:
            self.plot_widget.setLabel('left', 'Neuron ID')
            # pxMode=True verhindert "Morphen" beim Zoomen (Punkte bleiben gleich groß)
            self.scatter_item = pg.ScatterPlotItem(
                size=3, pen=None, brush=pg.mkBrush(c), 
                symbol='s', pxMode=True
            )
            self.plot_widget.addItem(self.scatter_item)
            
        elif "meter" in self.device_type:
            rec_vars = self.params.get('record_from', ['V_m'])
            if isinstance(rec_vars, str): rec_vars = [rec_vars]
            
            for i, var in enumerate(rec_vars):
                # Variation der Farbe für verschiedene Variablen
                pen_c = c
                if i == 1: pen_c = c.lighter(150)
                
                self.curves[var] = self.plot_widget.plot(name=var, pen=pg.mkPen(pen_c, width=1.5))
                self.values[var] = []

    def set_time_offset(self, offset):
        """Verschiebt die Zeitachse für neue Daten (bei Reset mit Keep Data)."""
        self.global_time_offset = offset

    def update_data(self, events):
        """Nimmt Daten aus NEST entgegen und puffert sie."""
        raw_times = events.get('times', [])
        if len(raw_times) == 0: return
        
        # Zeit korrigieren (Offset addieren)
        adj_times = np.array(raw_times) + self.global_time_offset
        self.last_recorded_time = adj_times[-1]
        
        # --- Spikes ---
        if self.scatter_item:
            new_senders = events.get('senders', [])
            self.times.extend(adj_times)
            self.senders.extend(new_senders)
            
            # Limitierung (FIFO Queue)
            if len(self.times) > self.max_points:
                cut = len(self.times) - self.max_points
                self.times = self.times[cut:]
                self.senders = self.senders[cut:]
            
            # Plot Update
            # Performance-Trick: downsampling für Anzeige bei extrem vielen Punkten?
            # Hier zeigen wir alles im Puffer an.
            self.scatter_item.setData(self.times, self.senders)

        # --- Analog ---
        elif self.curves:
            self.times.extend(adj_times)
            
            # Limitierung Times
            if len(self.times) > self.max_points:
                cut = len(self.times) - self.max_points
                self.times = self.times[cut:]
            
            # Werte updaten
            for k in self.curves:
                if k in events:
                    self.values[k].extend(events[k])
                    # Limitierung Values
                    if len(self.values[k]) > self.max_points:
                        v_cut = len(self.values[k]) - self.max_points
                        self.values[k] = self.values[k][v_cut:]
            
            # Kurven zeichnen
            for k, curve in self.curves.items():
                if k in self.values:
                    # Länge angleichen (Sicherheit)
                    min_len = min(len(self.times), len(self.values[k]))
                    curve.setData(self.times[:min_len], self.values[k][:min_len])
        
        # Auto-Scroll (Window: Letzte 2000ms)
        if len(self.times) > 0:
            last = self.times[-1]
            # update range, disable auto-range to prevent flickering
            self.plot_widget.setXRange(max(0, last - 2000), last + 10, padding=0)

    def clear_data(self):
        self.times = []
        self.senders = []
        for k in self.values: self.values[k] = []
        self.global_time_offset = 0.0
        self.last_recorded_time = 0.0
        
        if self.scatter_item: self.scatter_item.clear()
        for c in self.curves.values(): c.clear()
    
    def get_full_history(self):
        """Exportiert alle Daten im aktuellen Puffer."""
        d = {
            "times": list(self.times), 
            "type": self.device_type, 
            "id": self.device_id
        }
        if self.scatter_item:
            d["senders"] = list(self.senders)
        elif self.curves:
            d["values"] = {k: list(v) for k, v in self.values.items()}
        return d


class NodeLiveGroup(QGroupBox):
    """Gruppiert Plots eines Nodes."""
    def __init__(self, graph_id, node_id, node_name, model_name, color_hex, parent=None):
        super().__init__(parent)
        
        self.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid #333;
                border-radius: 4px;
                margin-top: 1.2em;
                background-color: #181818;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {color_hex};
                font-weight: bold;
            }}
        """)
        self.setTitle(f"{node_name} (ID:{node_id}) - {model_name}")
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 15, 4, 4)
        self.layout.setSpacing(2)
        
        # --- Activity Sum Plot (Oben) ---
        self.rate_plot = pg.PlotWidget()
        self.rate_plot.setBackground(None)
        self.rate_plot.setMaximumHeight(50)
        self.rate_plot.hideAxis('left')
        self.rate_plot.hideAxis('bottom')
        
        # Einfache Kurve für Aktivität
        c = QColor(color_hex)
        self.rate_curve = self.rate_plot.plot(pen=pg.mkPen(c, width=2), fillLevel=0, brush=pg.mkBrush(c.red(), c.green(), c.blue(), 40))
        self.layout.addWidget(self.rate_plot)
        
        # --- Toggle Button ---
        self.btn_toggle = QPushButton("▼ Details")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setStyleSheet("text-align: left; color: #666; background: transparent; border: none; font-size: 10px;")
        self.btn_toggle.toggled.connect(self._toggle)
        self.layout.addWidget(self.btn_toggle)
        
        # --- Details Area ---
        self.details = QWidget()
        self.details.setVisible(False)
        self.details_layout = QVBoxLayout(self.details)
        self.details_layout.setContentsMargins(10,0,0,0)
        self.layout.addWidget(self.details)
        
        self.plots = []

    def add_plot(self, widget):
        self.details_layout.addWidget(widget)
        self.plots.append(widget)
        self.btn_toggle.setText(f"▶ Details ({len(self.plots)} Devices)")

    def _toggle(self, checked):
        self.details.setVisible(checked)
        arrow = "▼" if checked else "▶"
        self.btn_toggle.setText(f"{arrow} Details ({len(self.plots)} Devices)")
        
    def update_summary(self):
        # Dummy für Summen-Aktivität (kann später erweitert werden)
        pass


class LiveDataDashboard(QWidget):
    # Signale an MainWindow
    sigStartLive = pyqtSignal(float, float) 
    sigPauseLive = pyqtSignal()
    sigResetLive = pyqtSignal(bool) 
    
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.node_groups = {}
        self.plot_widgets_map = {} 
        self.current_max_time = 0.0
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Sidebar ---
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("background-color: #232323; border-right: 1px solid #444;")
        sb = QVBoxLayout(sidebar)
        
        sb.addWidget(QLabel("LIVE DATA CONTROL", styleSheet="color:#00E5FF; font-weight:bold; margin-bottom:10px;"))
        
        self.btn_start = QPushButton("▶ RUN")
        self.btn_start.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 8px;")
        self.btn_start.clicked.connect(self.on_start)
        sb.addWidget(self.btn_start)
        
        self.btn_pause = QPushButton("⏸ PAUSE")
        self.btn_pause.setStyleSheet("background-color: #FBC02D; color: black; font-weight: bold; padding: 8px;")
        self.btn_pause.clicked.connect(self.sigPauseLive.emit)
        sb.addWidget(self.btn_pause)
        
        self.btn_reset = QPushButton("💾 RESET & SAVE")
        self.btn_reset.setToolTip("Saves history to JSON and resets simulation.")
        self.btn_reset.setStyleSheet("background-color: #BF360C; color: white; font-weight: bold; padding: 8px;")
        self.btn_reset.clicked.connect(self.on_reset)
        sb.addWidget(self.btn_reset)
        
        self.chk_keep = QCheckBox("Keep Data on Reset")
        self.chk_keep.setChecked(True)
        self.chk_keep.setStyleSheet("color: #ddd; margin-top: 5px;")
        sb.addWidget(self.chk_keep)
        
        self.btn_clear = QPushButton("🗑 Clear Plots")
        self.btn_clear.clicked.connect(self.clear_all_plots)
        self.btn_clear.setStyleSheet("background-color: #444; color: #fff;")
        sb.addWidget(self.btn_clear)
        
        sb.addSpacing(20)
        
        grp = QGroupBox("Config")
        f = QFormLayout(grp)
        self.spin_step = QDoubleSpinBox(); self.spin_step.setRange(1, 1000); self.spin_step.setValue(25.0); self.spin_step.setSuffix(" ms")
        self.spin_hist = QSpinBox(); self.spin_hist.setRange(1000, 1000000); self.spin_hist.setValue(50000); self.spin_hist.setSuffix(" pts")
        self.spin_hist.setToolTip("Max points per plot (FIFO Buffer)")
        self.spin_hist.valueChanged.connect(self.update_history_len)
        
        f.addRow("Step:", self.spin_step)
        f.addRow("Buffer:", self.spin_hist)
        sb.addWidget(grp)
        
        btn_scan = QPushButton("↻ Rescan Devices")
        btn_scan.clicked.connect(self.scan_for_devices)
        sb.addWidget(btn_scan)
        
        sb.addStretch()
        layout.addWidget(sidebar)
        
        # --- Content Area ---
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background-color: #121212;")
        
        self.content = QWidget()
        self.content.setStyleSheet("background-color: #121212;")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setSpacing(5)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll)
        
        self.rescan_timer = QTimer()
        self.rescan_timer.setSingleShot(True)
        self.rescan_timer.timeout.connect(self.scan_for_devices)

    def on_start(self):
        self.sigStartLive.emit(self.spin_step.value(), 0.0) # 0.0 Duration = Infinite

    def on_reset(self):
        keep = self.chk_keep.isChecked()
        if keep:
            max_t = 0
            for w in self.plot_widgets_map.values():
                if w.last_recorded_time > max_t: max_t = w.last_recorded_time
            self.current_max_time = max_t
        else:
            self.current_max_time = 0
            self.clear_all_plots()
            
        self.sigResetLive.emit(keep)
        self.rescan_timer.start(500)

    def clear_all_plots(self):
        for w in self.plot_widgets_map.values(): w.clear_data()
        self.current_max_time = 0

    def update_history_len(self, val):
        for w in self.plot_widgets_map.values():
            w.max_points = val

    def scan_for_devices(self):
        print("Scanning for live devices...")
        
        # Bestehende Devices markieren
        found_dev_keys = set()
        
        for graph in self.graph_list:
            for node in graph.node_list:
                # 1. Gruppe (Node) finden oder erstellen
                g_key = (graph.graph_id, node.id)
                if g_key not in self.node_groups:
                    model = node.neuron_models[0] if node.neuron_models else "unknown"
                    c = neuron_colors.get(model, "#ffffff")
                    group = NodeLiveGroup(graph.graph_id, node.id, node.name, model, c)
                    self.content_layout.addWidget(group)
                    self.node_groups[g_key] = group
                
                group = self.node_groups[g_key]
                
                # 2. Devices finden
                devices = getattr(node, 'devices', [])
                if not devices and 'devices' in node.parameters: devices = node.parameters['devices']
                
                relevant = [d for d in devices if "recorder" in d.get('model','') or "meter" in d.get('model','')]
                
                for dev in relevant:
                    dtype = dev.get('model')
                    did = dev.get('id')
                    d_key = (graph.graph_id, node.id, did)
                    found_dev_keys.add(d_key)
                    
                    nest_gid = dev.get('runtime_gid')
                    
                    if d_key in self.plot_widgets_map:
                        # Update
                        w = self.plot_widgets_map[d_key]
                        w.nest_gid = nest_gid
                        w.set_time_offset(self.current_max_time)
                    else:
                        # Neu erstellen
                        model = node.neuron_models[0] if node.neuron_models else "unknown"
                        c = neuron_colors.get(model, "#fff")
                        info = {'name': node.name, 'color': c}
                        
                        w = LivePlotWidget(did, dtype, dev.get('params',{}), info, max_points=self.spin_hist.value())
                        w.nest_gid = nest_gid
                        w.set_time_offset(self.current_max_time)
                        
                        self.plot_widgets_map[d_key] = w
                        group.add_plot(w)
        
        # Cleanup Orphans (Entfernte Devices)
        for k in list(self.plot_widgets_map.keys()):
            if k not in found_dev_keys:
                w = self.plot_widgets_map.pop(k)
                w.deleteLater()
                
        # Cleanup Empty Groups
        for k in list(self.node_groups.keys()):
            # Prüfen ob die Gruppe noch Plots hat (einfacher Hack: Wir löschen sie hier nicht, 
            # sondern lassen sie leer stehen, um Flackern zu vermeiden. 
            # Ein Full Refresh Button könnte aufräumen.)
            pass

    def update_plots(self):
        if not self.isVisible(): return
        for w in self.plot_widgets_map.values():
            gid = w.nest_gid
            if gid is None: continue
            try:
                status = nest.GetStatus(gid)[0]
                if status.get('n_events', 0) > 0:
                    w.update_data(status.get('events', {}))
                    nest.SetStatus(gid, {'n_events': 0})
            except: pass

    def get_all_data(self):
        """Exportiert History."""
        out = {}
        for k, w in self.plot_widgets_map.items():
            name = f"Graph{k[0]}_Node{k[1]}_Dev{k[2]}"
            out[name] = w.get_full_history()
        return out








class NodeLiveGroup(QGroupBox):
    """
    Ein Container für einen Node, der eine Summen-Aktivität (oben)
    und einklappbare Detail-Plots (unten) enthält.
    """
    def __init__(self, graph_id, node_id, node_name, model_name, color, parent=None):
        super().__init__(parent)
        self.graph_id = graph_id
        self.node_id = node_id
        self.node_name = node_name
        
        # Styling
        self.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 24px;
                background-color: #1a1a1a;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: {color};
                font-weight: bold;
                font-size: 13px;
            }}
        """)
        self.setTitle(f"Graph {graph_id} | {node_name} ({model_name})")
        
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(2)
        self.layout.setContentsMargins(5, 15, 5, 5)
        
        # --- 1. Summen-Plot (Activity Histogram) ---
        self.sum_plot = pg.PlotWidget()
        self.sum_plot.setBackground('#121212')
        self.sum_plot.setMaximumHeight(80) # Schlank halten
        self.sum_plot.hideAxis('left')
        self.sum_plot.setLabel('bottom', 'Spike Rate (Population)', units='')
        self.sum_plot.getAxis('bottom').setHeight(15)
        
        # Wir nutzen eine Kurve für die Feuerrate
        self.rate_curve = self.sum_plot.plot(pen=pg.mkPen(color, width=2, fillLevel=0, brush=(50,50,50,100)))
        self.layout.addWidget(self.sum_plot)
        
        # --- 2. Collapse Button ---
        self.btn_toggle = QPushButton("▼ Show 0 Devices")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(True) # Initial offen
        self.btn_toggle.setStyleSheet("""
            QPushButton {
                text-align: left; background: #222; color: #888; border: none; padding: 4px;
            }
            QPushButton:hover { background: #333; color: white; }
            QPushButton:checked { color: #00E5FF; }
        """)
        self.btn_toggle.toggled.connect(self.toggle_details)
        self.layout.addWidget(self.btn_toggle)
        
        # --- 3. Detail Container ---
        self.details_container = QWidget()
        self.details_layout = QVBoxLayout(self.details_container)
        self.details_layout.setContentsMargins(0,0,0,0)
        self.details_layout.setSpacing(5)
        self.layout.addWidget(self.details_container)
        
        # Daten für Rate
        self.spike_times_buffer = [] # Alle Spikes aller Recorder dieses Nodes
        self.plot_widgets = [] # Liste der Detail-Widgets

    def add_device_widget(self, widget):
        self.details_layout.addWidget(widget)
        self.plot_widgets.append(widget)
        self.btn_toggle.setText(f"▼ Show {len(self.plot_widgets)} Devices")

    def toggle_details(self, checked):
        self.details_container.setVisible(checked)
        arrow = "▼" if checked else "▶"
        self.btn_toggle.setText(f"{arrow} Show {len(self.plot_widgets)} Devices")

    def update_sum_plot(self, time_window=100.0):
        """Berechnet einfache Feuerrate aus allen Spikes dieses Nodes."""
        # Sammle neue Spikes von allen Recordern
        new_spikes = []
        for w in self.plot_widgets:
            if "spike_recorder" in w.device_type:
                # Wir nehmen die letzten hinzugefügten Zeiten aus dem Widget
                # Das ist ein Hack, effizienter wäre direkte Datenübergabe.
                # Da update_data schon lief, nehmen wir die letzten N
                if w.times:
                    # Letzte 50ms
                    last_t = w.times[-1]
                    cutoff = last_t - time_window
                    
                    # Finde Index (numpy searchsorted wäre schneller, aber hier Python list)
                    # Wir nehmen einfach an, dass die neuen Daten am Ende sind
                    # Wir nutzen einfach die Daten im Widget Puffer für die Visualisierung
                    
                    # Besser: Wir machen ein Histogramm über die GESAMTEN times im Buffer des Widgets
                    # und plotten das als Linie.
                    
                    y, x = np.histogram(w.times, bins=50) # 50 Bins über die gesamte Zeit
                    # Das wird langsam bei vielen Daten.
                    # Vereinfachung: Wir plotten nur ein "Live Meter" (Balken) oder lassen es weg für Performance.
                    pass
        
        # Für den Moment lassen wir den Sum-Plot leer oder zeigen stattdessen nur Text an,
        # um Performance zu sparen, bis wir einen echten Rate-Calculator haben.
        self.sum_plot.setTitle("Aggregate Activity (ToDo)", size='8pt')


class LiveDataDashboard(QWidget):
    sigStartLive = pyqtSignal(float, float) 
    sigPauseLive = pyqtSignal()
    sigResetLive = pyqtSignal(bool) 
    
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.node_groups = {} # Map: (graph_id, node_id) -> NodeLiveGroup
        self.plot_widgets_map = {} # Map: key -> widget
        self.current_max_time = 0.0
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Sidebar ---
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("background-color: #232323; border-right: 1px solid #444;")
        sb_layout = QVBoxLayout(sidebar)
        
        sb_layout.addWidget(QLabel("LIVE MONITOR", styleSheet="color:#00E5FF; font-weight:bold;"))
        
        self.btn_start = QPushButton("▶ RUN"); self.btn_start.clicked.connect(self.on_start)
        self.btn_start.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
        sb_layout.addWidget(self.btn_start)
        
        self.btn_pause = QPushButton("⏸ PAUSE"); self.btn_pause.clicked.connect(self.sigPauseLive.emit)
        self.btn_pause.setStyleSheet("background-color: #FBC02D; color: black; font-weight: bold;")
        sb_layout.addWidget(self.btn_pause)
        
        self.btn_reset = QPushButton("💾 RESET"); self.btn_reset.clicked.connect(self.on_reset)
        self.btn_reset.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold;")
        sb_layout.addWidget(self.btn_reset)
        
        self.chk_keep = QCheckBox("Keep Data"); self.chk_keep.setChecked(True)
        sb_layout.addWidget(self.chk_keep)
        
        btn_scan = QPushButton("↻ Rescan"); btn_scan.clicked.connect(self.scan_for_devices)
        sb_layout.addWidget(btn_scan)
        
        sb_layout.addStretch()
        layout.addWidget(sidebar)
        
        # --- Content (Scroll Area) ---
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background-color: #121212;")
        
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setSpacing(10)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll)
        
        self.rescan_timer = QTimer(); self.rescan_timer.setSingleShot(True); self.rescan_timer.timeout.connect(self.scan_for_devices)

    def on_start(self):
        self.sigStartLive.emit(25.0, 5000.0) # Defaults

    def on_reset(self):
        keep = self.chk_keep.isChecked()
        if keep:
            max_t = 0
            for w in self.plot_widgets_map.values():
                if w.last_recorded_time > max_t: max_t = w.last_recorded_time
            self.current_max_time = max_t
        else:
            self.current_max_time = 0
            self.clear_all_plots()
            
        self.sigResetLive.emit(keep)
        self.rescan_timer.start(200)

    def clear_all_plots(self):
        for w in self.plot_widgets_map.values(): w.clear_data()
        self.current_max_time = 0

    def scan_for_devices(self):
        # 1. Bestehende Struktur prüfen
        found_dev_keys = set()
        
        for graph in self.graph_list:
            for node in graph.node_list:
                # Group Key
                g_key = (graph.graph_id, node.id)
                
                # Gruppe erstellen falls nicht existent
                if g_key not in self.node_groups:
                    model = node.neuron_models[0] if node.neuron_models else "unknown"
                    color = neuron_colors.get(model, "#ffffff")
                    
                    group = NodeLiveGroup(graph.graph_id, node.id, node.name, model, color)
                    self.content_layout.addWidget(group)
                    self.node_groups[g_key] = group
                
                group_widget = self.node_groups[g_key]
                
                # Devices scannen
                devices = getattr(node, 'devices', [])
                if not devices and 'devices' in node.parameters: devices = node.parameters['devices']
                
                for dev in devices:
                    dtype = dev.get('model', '')
                    if "recorder" in dtype or "meter" in dtype:
                        dev_id = dev.get('id')
                        d_key = (graph.graph_id, node.id, dev_id)
                        found_dev_keys.add(d_key)
                        nest_gid = dev.get('runtime_gid')
                        
                        if d_key in self.plot_widgets_map:
                            # Update
                            self.plot_widgets_map[d_key].nest_gid = nest_gid
                            self.plot_widgets_map[d_key].set_time_offset(self.current_max_time)
                        else:
                            # Create
                            model = node.neuron_models[0] if node.neuron_models else "unknown"
                            c = neuron_colors.get(model, "#fff")
                            info = {'name': node.name, 'color': c}
                            
                            w = LivePlotWidget(dev_id, dtype, dev.get('params',{}), info)
                            w.nest_gid = nest_gid
                            w.set_time_offset(self.current_max_time)
                            
                            self.plot_widgets_map[d_key] = w
                            group_widget.add_device_widget(w)
        
        # Cleanup orphans (Devices)
        to_remove_devs = []
        for k, w in self.plot_widgets_map.items():
            if k not in found_dev_keys:
                w.deleteLater()
                to_remove_devs.append(k)
        for k in to_remove_devs: del self.plot_widgets_map[k]
        
        # Cleanup empty groups
        # (Optional: Gruppen löschen die keine Devices mehr haben)

    def update_plots(self):
        if not self.isVisible(): return
        for w in self.plot_widgets_map.values():
            gid = w.nest_gid
            if gid is None: continue
            try:
                status = nest.GetStatus(gid)[0]
                if status.get('n_events', 0) > 0:
                    w.update_data(status.get('events', {}))
                    nest.SetStatus(gid, {'n_events': 0})
            except: pass
        
        # Optional: Sum plots updaten
        # for g in self.node_groups.values(): g.update_sum_plot()

    def get_all_data(self):
        data = {}
        for k, w in self.plot_widgets_map.items():
            s = f"G{k[0]}_N{k[1]}_D{k[2]}"
            data[s] = w.get_full_history()
        return data


#

















