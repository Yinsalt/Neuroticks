import sys
from PyQt6.QtWidgets import QApplication,QDialog,QAbstractSpinBox,QDialogButtonBox,QListWidget,QSlider, QMainWindow,QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtWidgets import QSizePolicy,QListWidgetItem, QFrame,QPushButton,QLabel,QGroupBox, QStackedWidget, QToolBar, QMenu, QGridLayout, QStackedLayout
from PyQt6.QtGui import QColor, QPalette, QAction,QIcon,QBrush
from PyQt6.QtCore import QSize, Qt, pyqtSignal,QTimer
import pyvista as pv
from pyvistaqt import QtInteractor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import time
import vtk
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from typing import Dict, Any, Tuple, Optional, List
import matplotlib.colors as mcolors
from datetime import datetime
from pathlib import Path
from neuron_toolbox import *
from WidgetLib import *
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
    'create_nest_mask',
    "LiveSpectatorWindow",
    "SimulationDashboardWidget",
    'create_distance_dependent_weight',
    'SynapseParamWidget',
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
    'import_connections_from_dict'
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




def generate_biased_polynomial(axis_idx, max_degree=2, num_noise_terms=2):
    """
    Generiert ein Polynom, das nahe an der Identität (f(x)=x) liegt,
    aber durch POLYNOM_NOISE_LEVEL gestört wird.
    
    axis_idx: 0=x, 1=y, 2=z (Welche Achse ist die Identität?)
    """
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
        print(f"⚠ Distance weight creation failed: {e}, using base weight")
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
        
        header = QLabel("⚡ Synapse Configuration")
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
        
        save_btn = QPushButton("💾 Save as Preset")
        save_btn.clicked.connect(self._save_preset)
        custom_layout.addRow(save_btn)
        
        main_layout.addWidget(custom_group)
        
        action_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("✓ Apply Synapse")
        self.apply_btn.setStyleSheet("background-color: #4CAF50; font-weight: bold;")
        self.apply_btn.clicked.connect(self._emit_synapse)
        action_layout.addWidget(self.apply_btn)
        
        reset_btn = QPushButton("↺ Reset")
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
            self.delay_warning.setText("⚠ This synapse type does not use delay!")
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
        priority = ['iaf_psc_alpha', 'iaf_cond_exp', 'aeif_cond_alpha', 'hh_psc_alpha']
        sorted_models = sorted([m for m in successful_neuron_models if m not in priority])
        combo.addItems(priority + sorted_models)
        
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
        
        desc = QLabel("Generates Cluster via WFC,moves them through their vectorfields.")
        desc.setStyleSheet("color: #888; font-size: 10px; margin-bottom: 10px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
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
        
        self._add_field_to_layout(layout, "n_neurons", "Number of Neurons", field_type="int", min_val=3, max_val=100000, default=100)
        self._add_field_to_layout(layout, "radius", "Ring Radius", field_type="float", min_val=0.1, max_val=1000.0, default=5.0)
        self._add_field_to_layout(layout, "k", "Inhibition Strength (k)", field_type="float", min_val=0.0, max_val=1000.0, default=10.0)
        self._add_field_to_layout(layout, "bidirectional", "Bidirectional Connections", field_type="bool", default=False)
        layout.addStretch()
        return panel

    def _create_panel_blob(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        header = QLabel(" Random Blob (Sphere)")
        header.setStyleSheet("color: #9C27B0; font-weight: bold; font-size: 12px;")
        layout.addWidget(header)
        
        self._add_model_selector(layout, "tool_neuron_model", "Neuron Model") 
        
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
        
        self._add_field_to_layout(layout, "n_neurons", "Number of Neurons", field_type="int", min_val=1, max_val=100000, default=500)
        self._add_field_to_layout(layout, "radius_bottom", "Bottom Radius (Base)", field_type="float", min_val=0.1, max_val=1000.0, default=5.0)
        self._add_field_to_layout(layout, "radius_top", "Top Radius (Apex)", field_type="float", min_val=0.0, max_val=1000.0, default=1.0)
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
        self.on_change()  # Parameter-Update triggern

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
            return
        
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
                            
        if 'transform_matrix' in data:
             pass
             
        self.auto_save = True




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
        self.spin_coeff.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons) # Cleaner look
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


################################################################################
class GraphOverviewWidget(QWidget):

    node_selected = pyqtSignal(int, int)
    population_selected = pyqtSignal(int, int, int)
    connection_selected = pyqtSignal(dict)
    requestConnectionCreation = pyqtSignal(int, int, int) # graph_id, node_id, pop_id    # Farben (High Contrast)
    requestConnectionDeletion = pyqtSignal(dict)  

    COLOR_GRAPH_BG = "#000000"      
    COLOR_GRAPH_FG = "#87CEEB"      
    COLOR_NODE_BG = "#8B0000"       
    COLOR_NODE_FG = "#FFFF00"      
    COLOR_POP_BG = "#424242"        
    COLOR_POP_FG = "#FFFFFF"        
    COLOR_CONN_BG = "#aaaa00"       
    COLOR_CONN_FG = "#841414"       
    
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
        title = QLabel("📊 Graph Overview")
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
        self.collapse_btn.setToolTip("Collapse All (Alles einklappen)")
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
            
            action_del = QAction(f"🗑️ Delete '{conn_name}'", self)
            action_del.triggered.connect(lambda: self.requestConnectionDeletion.emit(conn_data))
            menu.addAction(action_del)

        menu.exec(self.tree.viewport().mapToGlobal(position))


    def update_tree(self):
        self.tree.clear()
        
        if not self.graph_list:
            self.status_label.setText("No graphs loaded")
            return
        
        total_nodes = 0
        total_pops = 0
        total_conns = 0
        
        for graph in self.graph_list:
            graph_item = QTreeWidgetItem(self.tree)
            graph_name = getattr(graph, 'graph_name', f'Graph_{graph.graph_id}')
            graph_item.setText(0, f"📊 {graph_name}")
            graph_item.setText(1, f"ID: {graph.graph_id} | {len(graph.node_list)} nodes")
            
            self._style_item(graph_item, self.COLOR_GRAPH_BG, self.COLOR_GRAPH_FG, bold=True)
            
            graph_item.setData(0, Qt.ItemDataRole.UserRole, {
                'type': 'graph', 
                'graph_id': graph.graph_id
            })
            
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
        
        self.status_label.setText(
            f"📊 {len(self.graph_list)} graphs | 🟡 {total_nodes} nodes | 🟠 {total_pops} pops | → {total_conns} conns"
        )
    
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
    
    def _on_item_double_clicked(self, item, column):
        pass
            
################################################################################

node_parameters1 = {
    "grid_size": [5, 5, 5],
    "m": [0.0, 0.0, 0.0],
    "rot_theta": 0.0,
    "rot_phi": 0.0,
    "transform_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "dt": 0.01,
    "old": True,
    "num_steps": 2,
    "sparse_holes": 0,
    "sparsity_factor": 0.9,
    "probability_vector": [0.3, 0.2, 0.4],
    "name": "Node",
    "id": 0,
    "polynom_max_power": 2,
    "center_of_mass": [0.0, 0.0, 0.0],
    "displacement": [0.0, 0.0, 0.0],
    "displacement_factor": 1.0,
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
        # ------------------------------------
        
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
        
        self.add_pop_btn = QPushButton("+ Add Population")
        self.add_pop_btn.clicked.connect(self.add_population)
        self.add_pop_btn.setEnabled(False)
        self.add_pop_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        pop_col.addWidget(self.add_pop_btn)
        
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
        """Zeigt das Menü mit Cortex-Strukturen beim Rechtsklick an."""
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
        node_data['params']['grid_size'] = [10, 10, 10] # Kleiner Patch wie gewünscht
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
            
        self.save_node_params(self.node_param_widget.get_current_params())
        self.save_current_population_params()
        
        source_node = self.node_list[self.current_node_idx]
        current_pos = source_node['params'].get('center_of_mass', [0,0,0])
        old_id = source_node['params'].get('id')
        
        dlg = PositionDialog(current_pos, self)
        if dlg.exec():
            new_pos = dlg.get_position()
            
            import copy
            import time
            new_node_data = copy.deepcopy(source_node)
            
            new_idx = len(self.node_list)
            new_node_data['params']['id'] = new_idx
            new_node_data['params']['name'] = f"{source_node['params']['name']}_Twin"
            
            new_node_data['params']['m'] = list(new_pos)
            new_node_data['params']['center_of_mass'] = list(new_pos)
            new_node_data['params']['old_center_of_mass'] = list(new_pos)
            
            source_conns = source_node['params'].get('connections', [])
            twin_conns = []
            
            for conn in source_conns:
                src_id = conn['source'].get('node_id')
                tgt_id = conn['target'].get('node_id')
                
                if src_id == old_id and tgt_id == old_id:
                    new_conn = copy.deepcopy(conn)
                    
                    new_conn['source']['node_id'] = new_idx
                    new_conn['target']['node_id'] = new_idx
                    
                    new_conn['id'] = int(time.time() * 1000) + len(twin_conns)
                    new_conn['name'] = f"{conn.get('name', 'conn')}_Twin"
                    
                    twin_conns.append(new_conn)
            
            new_node_data['params']['connections'] = twin_conns
            print(f"  Twin inherited {len(twin_conns)} internal connections.")
            
            node_btn = QPushButton(f"Node {new_idx + 1}: {new_node_data['params']['name']}")
            node_btn.setMinimumHeight(50)
            node_btn.clicked.connect(lambda checked=False, idx=new_idx: self.select_node(idx))
            self.node_list_layout.addWidget(node_btn)
            new_node_data['button'] = node_btn
            
            self.node_list.append(new_node_data)
            self.select_node(new_idx)
            print(f"Twin created at {new_pos}")

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
        # ----------------------------------------
        
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
            
            # Nur für Custom Nodes (WFC) sind manuelle Populationen zwingend
            if tool_type == 'custom' and not node['populations']:
                print(f"ERROR: Node {i+1} (Custom) has no populations! Please add populations.")
                return
            
            # Probability Check nur wenn manuelle Populationen da sind
            if node['populations']:
                prob_vec = node['params'].get('probability_vector', [])
                total_prob = sum(prob_vec)
                if abs(total_prob - 1.0) > 0.01:
                    print(f"ERROR: Node {i+1} probability vector sums to {total_prob:.2f}, must be 1.0!")
                    return
        
        # Aktuelle GUI-Werte des selektierten Nodes speichern
        if self.current_node_idx is not None:
            current_params = self.node_param_widget.get_current_params()
            if 'center_of_mass' in current_params:
                current_params['m'] = current_params['center_of_mass'].copy()
            self.node_list[self.current_node_idx]['params'] = current_params
        
        self.save_current_population_params()
        
        # Polynomials speichern (falls vorhanden)
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
            
            # === LOGIK-WEICHE: TOOL VS CUSTOM ===
            if tool_type != 'custom' and not populations:
                # AUTOMATISCHE KONFIGURATION FÜR TOOLS (z.B. CCW, Blob)
                print(f"Auto-configuring {tool_type} node structure...")
                
                # --- ÄNDERUNG START ---
                # Hole das Modell aus den Parametern, Fallback auf iaf_psc_alpha
                selected_model = node['params'].get('tool_neuron_model', 'iaf_psc_alpha')
                neuron_models = [selected_model] 
                # --- ÄNDERUNG ENDE ---
                
                types = [0]
                encoded_polynoms_per_type = [[]] 
                prob_vec = [1.0]
                pop_nest_params = [{}]
            else:
                # STANDARD LOGIK (WFC / Manuelle Pops)
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
                
                # Normalisieren
                if num_pops > 0 and abs(sum(prob_vec) - 1.0) > 0.01:
                    s = sum(prob_vec)
                    prob_vec = [p/s for p in prob_vec] if s > 0 else [1.0/num_pops]*num_pops
                
                pop_nest_params = [pop.get('params', {}) for pop in populations]

            # Parameter zusammenbauen
            node_params = {
                # Geometrie & Position
                'grid_size': node['params'].get('grid_size', [10, 10, 10]),
                'm': node['params'].get('center_of_mass', [0.0, 0.0, 0.0]),
                'center_of_mass': node['params'].get('center_of_mass', [0.0, 0.0, 0.0]),
                'displacement': node['params'].get('displacement', [0.0, 0.0, 0.0]),
                'displacement_factor': node['params'].get('displacement_factor', 1.0),
                'rot_theta': node['params'].get('rot_theta', 0.0),
                'rot_phi': node['params'].get('rot_phi', 0.0),
                
                # Tool Settings
                'tool_type': tool_type,
                'n_neurons': node['params'].get('n_neurons', 100),
                'radius': node['params'].get('radius', 5.0),
                'radius_top': node['params'].get('radius_top', 1.0),
                'radius_bottom': node['params'].get('radius_bottom', 5.0),
                'height': node['params'].get('height', 10.0),
                'grid_side_length': node['params'].get('grid_side_length', 10),
                
                # Circuit Specifics (CCW)
                'k': node['params'].get('k', 10.0),
                'bidirectional': node['params'].get('bidirectional', False),
                
                # Transformations Matrix
                'stretch_x': node['params'].get('stretch_x', 1.0),
                'stretch_y': node['params'].get('stretch_y', 1.0),
                'stretch_z': node['params'].get('stretch_z', 1.0),
                'transform_matrix': node['params'].get('transform_matrix', [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                
                # WFC Stuff
                'dt': node['params'].get('dt', 0.01),
                'old': node['params'].get('old', True),
                'num_steps': node['params'].get('num_steps', 8),
                'sparse_holes': node['params'].get('sparse_holes', 0),
                'sparsity_factor': node['params'].get('sparsity_factor', 0.9),
                
                # NEST / Structure
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
            
            # Originale Populations-Liste für GUI-Status mitschleifen
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
        
        node_params = node_parameters1.copy()
        node_params['id'] = node_idx
        node_params['name'] = f"Node_{node_idx}"
        node_params['probability_vector'] = []
        
        self.node_list.append({
            'params': node_params,
            'populations': [],
            'button': node_btn
        })
        
        self.select_node(node_idx)
    
    def select_node(self, node_idx):
        self.save_current_population_params()
        
        self.current_node_idx = node_idx
        self.current_pop_idx = None
        
        for i, node in enumerate(self.node_list):
            if i == node_idx:
                node['button'].setStyleSheet("background-color: #2196F3; color: white;")
            else:
                node['button'].setStyleSheet("")
        
        
        num_pops = len(self.node_list[node_idx]['populations'])
        self.node_param_widget.set_population_count(num_pops)
        self.node_param_widget.load_data(self.node_list[node_idx]['params'])
        self.editor_stack.setCurrentIndex(1)
        
        self.update_population_list()
        self.add_pop_btn.setEnabled(True)

    def load_structure_preset(self, name, models, probs, grid_size=[10,10,10]):
        """Lädt ein Struktur-Preset und bereitet den Node vor."""
        print(f"Loading Structure Preset: {name}")
        
        # 1. Resetten
        self.reset()
        
        # 2. Namen setzen
        safe_name = name.replace(" ", "_").replace("/", "-")
        self.graph_name_input.setText(f"Graph_{safe_name}")
        
        # 3. Node hinzufügen (Node 0)
        self.add_node()
        
        # 4. Zugriff auf den neu erstellten Node
        node = self.node_list[0]
        
        # 5. Parameter setzen
        node['params']['name'] = f"{safe_name}_Node"
        node['params']['grid_size'] = grid_size
        node['params']['probability_vector'] = probs
        
        # WFC Parameter leicht anpassen für Strukturen
        node['params']['sparsity_factor'] = 0.85 # Etwas dichter
        node['params']['dt'] = 0.01
        
        # 6. Populationen erstellen
        node['populations'] = []
        
        for i, model in enumerate(models):
            # --- UPDATE ---
            # Auch hier: Sanfte Abweichung statt totalem Chaos
            default_polynomials = {
                'x': generate_biased_polynomial(axis_idx=0, max_degree=2),
                'y': generate_biased_polynomial(axis_idx=1, max_degree=2),
                'z': generate_biased_polynomial(axis_idx=2, max_degree=2)
            }
            # --------------
            
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
        # 1. Interne Auswahl zurücksetzen
        self.current_node_idx = None
        self.current_pop_idx = None
        
        # 2. Listen leeren
        self.node_list.clear()
        
        # 3. UI-Layouts leeren (Buttons entfernen)
        while self.node_list_layout.count():
            item = self.node_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        while self.pop_list_layout.count():
            item = self.pop_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 4. Editor-Ansicht zurücksetzen
        self.editor_stack.setCurrentIndex(0)
        self.add_pop_btn.setEnabled(False)
        
        # --- NEU: Tool-Auswahl auf Standard ("custom") zurücksetzen ---
        # Sucht den Index für "custom" und setzt ihn aktiv
        custom_idx = self.node_param_widget.tool_combo.findData("custom")
        if custom_idx >= 0:
            self.node_param_widget.tool_combo.setCurrentIndex(custom_idx)
        
        # 5. NAME AKTUALISIEREN
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
        
        #  GRAPH SELECTOR 
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select Graph to Edit:"))
        
        self.graph_selector = QComboBox()
        self.graph_selector.currentIndexChanged.connect(self.on_graph_selected)
        selector_layout.addWidget(self.graph_selector)
        
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.refresh_graph_list)
        selector_layout.addWidget(refresh_btn)
        
        main_layout.addLayout(selector_layout)
        
        #  GRAPH NAME 
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Graph Name:"))
        self.graph_name_input = QLineEdit()
        self.graph_name_input.setPlaceholderText("Edit graph name...")
        name_layout.addWidget(self.graph_name_input)
        main_layout.addLayout(name_layout)
        
        # MAIN CONTENT 
        content_layout = QHBoxLayout()
        
        #  NODE COLUMN 
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
        
        # Rechtsklick Menü
        self.btn_add_node.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.btn_add_node.customContextMenuRequested.connect(self.show_add_node_menu)
        self.btn_add_node.setToolTip("Left-Click: Empty Node\nRight-Click: Structure Patch")

        node_buttons_layout.addWidget(self.btn_add_node)
        
        # --- NEUER BUTTON ---
        twin_btn = QPushButton("👥 Twin")
        twin_btn.setToolTip("Clone selected node as a NEW node at new position")
        twin_btn.clicked.connect(self.create_twin_node)
        twin_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        node_buttons_layout.addWidget(twin_btn)
        # --------------------

        self.remove_node_btn = QPushButton("🗑️ Remove")
        self.remove_node_btn.clicked.connect(self.remove_node)
        self.remove_node_btn.setEnabled(False)
        self.remove_node_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        node_buttons_layout.addWidget(self.remove_node_btn)
        
        node_col.addLayout(node_buttons_layout)
        
        content_layout.addLayout(node_col, 2)
        
        #  POPULATION COLUMN 
        pop_col = QVBoxLayout()
        pop_col.addWidget(QLabel("POPULATIONS", alignment=Qt.AlignmentFlag.AlignCenter))
        
        pop_scroll = QScrollArea()
        pop_scroll.setWidgetResizable(True)
        self.pop_list_widget = QWidget()
        self.pop_list_layout = QVBoxLayout(self.pop_list_widget)
        pop_scroll.setWidget(self.pop_list_widget)
        pop_col.addWidget(pop_scroll)
        
        self.add_pop_btn = QPushButton("+ Add Population")
        self.add_pop_btn.clicked.connect(self.add_population)
        self.add_pop_btn.setEnabled(False)
        self.add_pop_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        pop_col.addWidget(self.add_pop_btn)
        
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
        
        #  BOTTOM BUTTONS 
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
        
        # Initial refresh
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
        """Löscht eine Verbindung basierend auf dem Dictionary und speichert."""
        
        src_gid = conn_data['source']['graph_id']
        src_nid = conn_data['source']['node_id']
        conn_id = conn_data.get('id')

        # 1. Prüfen, ob wir im richtigen Graphen sind
        if self.current_graph_id != src_gid:
            # Wir laden den Graphen, falls er nicht aktiv ist
            idx = self.graph_selector.findData(src_gid)
            if idx >= 0:
                self.graph_selector.setCurrentIndex(idx)
            else:
                print(f"Error: Graph {src_gid} not found for deletion.")
                return

        # 2. Node in der lokalen node_list finden
        target_node_idx = None
        for i, node in enumerate(self.node_list):
            if node['params']['id'] == src_nid:
                target_node_idx = i
                break
        
        if target_node_idx is None:
            print(f"Error: Source Node {src_nid} not found in current editor list.")
            return

        # 3. Verbindung aus der Parameter-Liste entfernen
        node = self.node_list[target_node_idx]
        connections = node['params'].get('connections', [])
        
        # Filtern: Behalte alle, die NICHT die ID haben
        new_conns = [c for c in connections if c.get('id') != conn_id]
        
        if len(new_conns) == len(connections):
            print("Warning: Connection ID not found in node parameters. Already deleted?")
            return

        # Update List
        node['params']['connections'] = new_conns
        
        # 4. Auch aus 'original_node' entfernen (falls vorhanden), 
        # damit die Änderung als "Structural Change" erkannt oder direkt übernommen wird
        if node.get('original_node'):
            orig = node['original_node']
            if hasattr(orig, 'connections'):
                # Filtern im Objekt
                orig.connections = [c for c in orig.connections if c.get('id') != conn_id]

        print(f"🗑️ Connection {conn_id} removed from Node {src_nid}. Saving...")
        
        # 5. Sofort speichern und neu laden (Triggered NEST Reset wenn nötig)
        self.save_changes()

    def show_add_node_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background-color: #2b2b2b; color: white; border: 1px solid #555; }")
        
        header = QAction("🧠 Add Structure Patch:", self)
        header.setEnabled(False)
        menu.addAction(header)
        menu.addSeparator()

        # Daten aufbereiten (Identisch zu Creator)
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

        # 1. Basis-Node
        self.add_node()
        
        # 2. Daten holen
        node_idx = len(self.node_list) - 1
        node_data = self.node_list[node_idx]
        
        # 3. Params
        safe_name = name.replace(" ", "_").replace("/", "-")
        node_data['params']['name'] = f"{safe_name}_{node_idx}"
        node_data['params']['grid_size'] = [10, 10, 10]
        node_data['params']['probability_vector'] = probs
        node_data['params']['sparsity_factor'] = 0.85
        
        # UI Update
        is_new = node_data.get('original_node') is None
        suffix = " (NEW)" if is_new else ""
        node_data['button'].setText(f"Node {node_idx + 1}{suffix}: {node_data['params']['name']}")
        
        # 4. Pops
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
        
        # 5. Refresh Params im UI Widget (da add_node evtl schon leere Daten geladen hat)
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
                print(f"🔌 Editing Connection ID {found_conn.get('id')}")
            else:
                print("Connection reference not found in Editor list (maybe unsaved?). Editing copy.")
                self.conn_editor.load_data(connection_data)
                self.editor_stack.setCurrentIndex(4)
        else:
            print("Source Node for connection not found in Editor.")



    def remove_node(self):
        if self.current_node_idx is None:
            return

        if len(self.node_list) <= 1:
            QMessageBox.warning(self, "Cannot Remove", "A graph must have at least one node.")
            return

        node_name = self.node_list[self.current_node_idx]['params'].get('name', 'Node')

        reply = QMessageBox.question(
            self, 'Remove Node',
            f"Remove '{node_name}'?\n\nThis will trigger NEST reset on save.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            removed_node = self.node_list.pop(self.current_node_idx)
            if 'button' in removed_node:
                removed_node['button'].deleteLater()

            for i, node in enumerate(self.node_list):
                if 'button' in node:
                    try:
                        node['button'].clicked.disconnect()
                    except:
                        pass
                    
                    # Reconnect with correct index using default argument trick
                    node['button'].clicked.connect(lambda checked=False, idx=i: self.select_node(idx))
                    
                    name = node['params'].get('name', 'Node')
                    is_new = node.get('original_node') is None
                    suffix = " (NEW)" if is_new else ""
                    node['button'].setText(f"Node {i+1}{suffix}: {name}")

            self.current_node_idx = None
            self.current_pop_idx = None
            self.editor_stack.setCurrentIndex(0)
            self.remove_node_btn.setEnabled(len(self.node_list) > 1)
            self.add_pop_btn.setEnabled(False)

            while self.pop_list_layout.count():
                item = self.pop_list_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            print(f"Node '{node_name}' removed from UI (will be deleted on save)")




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
        
        # Clear existing UI
        self.node_list.clear()
        while self.node_list_layout.count():
            item = self.node_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        while self.pop_list_layout.count():
            item = self.pop_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Load graph name
        graph_name = getattr(self.current_graph, 'graph_name', f'Graph_{self.current_graph_id}')
        self.graph_name_input.setText(graph_name)
        
        # Load nodes
        for node in self.current_graph.node_list:
            self.load_node_from_graph(node)
        
        print(f"Loaded {len(self.node_list)} nodes")
        
        # Select first node
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
        node_idx = len(self.node_list)
        node_btn = QPushButton(f"Node {node_idx + 1} (NEW)")
        node_btn.setMinimumHeight(50)
        node_btn.clicked.connect(lambda checked=False, idx=node_idx: self.select_node(idx))
        self.node_list_layout.addWidget(node_btn)
        
        node_params = node_parameters1.copy()
        node_params['id'] = node_idx
        node_params['name'] = f"Node_{node_idx}"
        node_params['probability_vector'] = []
        
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
        
        # --- HIER: Benutze ebenfalls den neuen Generator ---
        # Greift auf die gleiche globale Variable zu, die im Creator-Slider gesetzt wird
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
        if self.current_node_idx is None:
            return
            
        # 1. Save current state
        self.save_node_params(self.node_param_widget.get_current_params())
        self.save_current_population_params()
        
        # 2. Source Data
        source_node = self.node_list[self.current_node_idx]
        current_pos = source_node['params'].get('center_of_mass', [0,0,0])
        old_id = source_node['params'].get('id')
        
        # 3. Dialog
        dlg = PositionDialog(current_pos, self)
        if dlg.exec():
            new_pos = dlg.get_position()
            
            # 4. Copy
            import copy
            import time
            new_node_data = copy.deepcopy(source_node)
            
            # 5. Adjustments
            new_idx = len(self.node_list)
            new_name = f"{source_node['params'].get('name', 'Node')}_Twin"
            
            new_node_data['params']['id'] = new_idx
            new_node_data['params']['name'] = new_name
            
            # Position Update
            new_node_data['params']['m'] = list(new_pos)
            new_node_data['params']['center_of_mass'] = list(new_pos)
            new_node_data['params']['old_center_of_mass'] = list(new_pos) 
            
            # WICHTIG: Als neuen Node markieren
            new_node_data['original_node'] = None 
            
            # 6. Connections (Nur interne kopieren)
            source_conns = source_node['params'].get('connections', [])
            twin_conns = []
            
            # Wir suchen die höchste existierende Connection ID im ganzen Graphen, um Konflikte zu vermeiden
            max_conn_id = 0
            for node in self.node_list:
                for c in node['params'].get('connections', []):
                    if isinstance(c.get('id'), int):
                        max_conn_id = max(max_conn_id, c.get('id'))
            
            for conn in source_conns:
                src_id = conn['source'].get('node_id')
                tgt_id = conn['target'].get('node_id')
                
                # CHECK: Interne Verbindung?
                if src_id == old_id and tgt_id == old_id:
                    new_conn = copy.deepcopy(conn)
                    
                    # IDs auf Twin biegen
                    new_conn['source']['node_id'] = new_idx
                    new_conn['target']['node_id'] = new_idx
                    
                    # Neue ID vergeben
                    max_conn_id += 1
                    new_conn['id'] = max_conn_id
                    new_conn['name'] = f"{conn.get('name', 'conn')}_Twin"
                    
                    # Fehlerstatus zurücksetzen (falls das Original fehlerhaft war)
                    if 'error' in new_conn:
                        del new_conn['error']
                        
                    twin_conns.append(new_conn)
            
            new_node_data['params']['connections'] = twin_conns
            
            # 7. UI Button
            node_btn = QPushButton(f"Node {new_idx + 1} (NEW TWIN)")
            node_btn.setMinimumHeight(50)
            node_btn.clicked.connect(lambda checked=False, idx=new_idx: self.select_node(idx))
            self.node_list_layout.addWidget(node_btn)
            new_node_data['button'] = node_btn
            
            self.node_list.append(new_node_data)
            self.select_node(new_idx)
            
            print(f"Twin created in Editor with {len(twin_conns)} internal connections.")
            QMessageBox.information(self, "Twin Created", 
                                  f"Twin '{new_name}' added at {new_pos}.\n"
                                  f"Inherited {len(twin_conns)} internal connections.\n"
                                  "Don't forget to click 'SAVE CHANGES' to generate the neurons.")
            
    def save_current_population_params(self):
        if self.current_node_idx is not None and self.current_pop_idx is not None:
            # Prüfe ob Index noch gültig ist
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
        
        # --- VALIDATION UPDATE START ---
        # Nur Custom-Nodes (WFC) benötigen zwingend manuelle Populationen.
        # Tool-Nodes (Cone, CCW etc.) können automatisch generiert werden.
        for i, node in enumerate(self.node_list):
            tool_type = node['params'].get('tool_type', 'custom')
            if tool_type == 'custom' and not node['populations']:
                print(f"ERROR: Node {i+1} (Custom) has no populations! Please add populations.")
                return
        # --- VALIDATION UPDATE END ---

        # Update graph name im Objekt
        new_name = self.graph_name_input.text()
        self.current_graph.graph_name = new_name
        self.save_current_population_params()
        
        # Save polynomials
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

        nest.ResetKernel()
        
        self.current_graph.node_list.clear()
        self.current_graph.nodes = 0
        self.current_graph._next_id = 0
        
        for node_idx, node_data in enumerate(self.node_list):
            # Hier wird jetzt die robustere _build_node_params aufgerufen
            node_params = self._build_node_params(node_idx, node_data)
            original_node = node_data.get('original_node')
            
            if node_idx == 0:
                new_node = self.current_graph.create_node(
                    parameters=node_params,
                    is_root=True,
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

            # Logik für Rebuild vs. Copy
            if has_structural_change or not original_has_points:
                reason = "Structural change" if has_structural_change else "Original positions empty/invalid"
                print(f"  Node {node_idx}: REBUILD ({reason})")
                try:
                    new_node.build()
                except Exception as e:
                    print(f"  ❌ Build failed for Node {node_idx}: {e}")
            
            elif original_node:
                # Versuche Positionen zu übernehmen, wenn keine strukturelle Änderung vorliegt
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
        
        # Andere Graphen repopulaten (da NEST Reset)
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

        # --- AUTOMATIC CONFIGURATION LOGIC START ---
        # Wenn es ein Tool ist und keine Populationen manuell angelegt wurden, 
        # erstellen wir die Struktur automatisch (wie im Creator).
        if tool_type != 'custom' and not populations:
            selected_model = node_data['params'].get('tool_neuron_model', 'iaf_psc_alpha')
            neuron_models = [selected_model]
            types = [0]
            encoded_polynoms_per_type = [[]]
            prob_vec = [1.0]
            pop_nest_params = [{}]
        else:
            # Standard Logik für Custom/Existierende Populationen
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
        # --- AUTOMATIC CONFIGURATION LOGIC END ---

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

        return {
            # Core Params
            'name': node_data['params'].get('name', f'Node_{node_idx}'),
            'id': node_idx,
            'graph_id': self.current_graph_id,
            'tool_type': tool_type,
            
            # Structure (Autogenerated or Manual)
            'neuron_models': neuron_models,
            'types': types,
            'distribution': prob_vec,
            'probability_vector': prob_vec,
            'encoded_polynoms_per_type': encoded_polynoms_per_type,
            'population_nest_params': pop_nest_params,

            # Geometry & Physics
            'grid_size': node_data['params'].get('grid_size', [10, 10, 10]),
            'm': node_data['params'].get('center_of_mass', [0.0, 0.0, 0.0]),
            'center_of_mass': node_data['params'].get('center_of_mass', [0.0, 0.0, 0.0]),
            'displacement': node_data['params'].get('displacement', [0.0, 0.0, 0.0]),
            'displacement_factor': node_data['params'].get('displacement_factor', 1.0),
            'rot_theta': node_data['params'].get('rot_theta', 0.0),
            'rot_phi': node_data['params'].get('rot_phi', 0.0),
            'transform_matrix': transform_matrix,
            'stretch_x': sx, 'stretch_y': sy, 'stretch_z': sz,
            'old_center_of_mass': old_com,
            
            # Tool Specifics
            'n_neurons': node_data['params'].get('n_neurons', 100),
            'radius': node_data['params'].get('radius', 5.0),
            'radius_top': node_data['params'].get('radius_top', 1.0),
            'radius_bottom': node_data['params'].get('radius_bottom', 5.0),
            'height': node_data['params'].get('height', 10.0),
            'grid_side_length': node_data['params'].get('grid_side_length', 10),
            'k': node_data['params'].get('k', 10.0),
            'bidirectional': node_data['params'].get('bidirectional', False),

            # WFC
            'dt': node_data['params'].get('dt', 0.01),
            'old': node_data['params'].get('old', True),
            'num_steps': node_data['params'].get('num_steps', 8),
            'sparse_holes': node_data['params'].get('sparse_holes', 0),
            'sparsity_factor': node_data['params'].get('sparsity_factor', 0.9),
            'polynom_max_power': node_data['params'].get('polynom_max_power', 5),
            
            # Empty fields
            'conn_prob': [],
            'field': None,
            'coefficients': None,
            'connections': [] # Connections werden in der Loop außen behandelt/bewahrt
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
            
            import nest
            nest.ResetKernel()
            print("NEST Kernel reset")
            
            for graph in self.graph_list:
                print(f"  Repopulating Graph {graph.graph_id}: {graph.graph_name}")
                for node in graph.node_list:
                    node.populate_node()
            
            self.current_graph = None
            self.current_graph_id = None
            self.refresh_graph_list()
            self.graphUpdated.emit(-1)


# In WidgetLib.py, ersetze die komplette class ConnectionTool


class ConnectionTool(QWidget):
    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list
        self.connections = []
        self.next_conn_id = 0
        self.current_conn_idx = None
        self.syn_param_widgets = {} 
        self.init_ui()
    def refresh(self):
        """Alias für refresh_graph_list, wird von CleanAlpha aufgerufen."""
        self.refresh_graph_list()
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # ==========================================
        # SPALTE 1: SOURCE & TARGET (Links)
        # ==========================================
        left_col = QVBoxLayout()
        
        # --- Source Group ---
        src_group = QGroupBox("Source Population")
        src_group.setStyleSheet("QGroupBox { border: 1px solid #4CAF50; margin-top: 10px; } QGroupBox::title { color: #4CAF50; }")
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
        lbl_arrow.setStyleSheet("font-weight: bold; color: #777; margin: 5px;")
        left_col.addWidget(lbl_arrow)
        
        # --- Target Group ---
        tgt_group = QGroupBox("Target Population")
        tgt_group.setStyleSheet("QGroupBox { border: 1px solid #FF9800; margin-top: 10px; } QGroupBox::title { color: #FF9800; }")
        tgt_layout = QFormLayout(tgt_group)
        
        self.target_graph_combo = QComboBox()
        self.target_graph_combo.currentIndexChanged.connect(self.on_target_graph_changed)
        tgt_layout.addRow("Graph:", self.target_graph_combo)
        
        self.target_node_combo = QComboBox()
        self.target_node_combo.currentIndexChanged.connect(self.on_target_node_changed)
        tgt_layout.addRow("Node:", self.target_node_combo)
        
        self.target_pop_combo = QComboBox()
        tgt_layout.addRow("Pop:", self.target_pop_combo)
        
        left_col.addWidget(tgt_group)
        left_col.addStretch()
        
        main_layout.addLayout(left_col, 2)
        
        # ==========================================
        # SPALTE 2: PARAMETER (Mitte)
        # ==========================================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        middle_container = QWidget()
        middle_col = QVBoxLayout(middle_container)
        middle_col.setContentsMargins(10, 0, 10, 0)
        
        # Name
        name_layout = QHBoxLayout()
        self.conn_name_input = QLineEdit()
        self.conn_name_input.setPlaceholderText("Connection Name (Optional)")
        name_layout.addWidget(QLabel("Name:"))
        name_layout.addWidget(self.conn_name_input)
        middle_col.addLayout(name_layout)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #AAA; padding: 8px 20px; }
            QTabBar::tab:selected { background: #2196F3; color: white; font-weight: bold; }
        """)
        
        self.tab_spatial = QWidget()
        self._init_spatial_tab()
        self.tabs.addTab(self.tab_spatial, "🌍 Spatial (Geometric)")
        
        self.tab_topo = QWidget()
        self._init_topological_tab()
        self.tabs.addTab(self.tab_topo, "🕸️ Topological (Graph)")
        
        middle_col.addWidget(self.tabs)
        
        # Synapse Settings
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
        
        # ==========================================
        # SPALTE 3: LISTE & ACTIONS (Rechts)
        # ==========================================
        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Connection Queue", alignment=Qt.AlignmentFlag.AlignCenter))
        
        self.conn_list_widget = QWidget()
        self.conn_list_layout = QVBoxLayout(self.conn_list_widget)
        conn_scroll = QScrollArea()
        conn_scroll.setWidgetResizable(True)
        conn_scroll.setWidget(self.conn_list_widget)
        right_col.addWidget(conn_scroll)
        
        # Buttons
        self.add_conn_btn = QPushButton("⬇ Add to Queue")
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
        
        # ✅ HIER FEHLTE DAS STATUS LABEL
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #bbb; font-size: 11px; font-weight: bold;")
        self.status_label.setWordWrap(True)
        right_col.addWidget(self.status_label)
        # ==========================================
        
        main_layout.addLayout(right_col, 2)
        
        # Initial Refresh
        self.refresh_graph_list()
        if self.syn_model_combo.count() > 0:
            self.on_synapse_model_changed(self.syn_model_combo.currentText())

    def _init_spatial_tab(self):
        layout = QVBoxLayout(self.tab_spatial)
        info = QLabel("Connects neurons based on spatial positions.")
        info.setStyleSheet("color: #AAA; font-style: italic;")
        layout.addWidget(info)
        form = QFormLayout()
        self.mask_type_combo = QComboBox()
        self.mask_type_combo.addItems(["sphere", "box", "doughnut"])
        form.addRow("Mask Shape:", self.mask_type_combo)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.01, 1000.0)
        self.radius_spin.setValue(0.5)
        self.radius_spin.setSuffix(" mm")
        form.addRow("Outer Radius/Size:", self.radius_spin)
        self.inner_radius_spin = QDoubleSpinBox()
        self.inner_radius_spin.setRange(0.0, 1000.0)
        self.inner_radius_spin.setValue(0.0)
        self.inner_radius_spin.setSuffix(" mm")
        form.addRow("Inner Radius:", self.inner_radius_spin)
        layout.addLayout(form)


        dist_layout = QHBoxLayout()
        
        self.dist_dep_check = QCheckBox("Scale Weight by Distance")
        self.dist_dep_check.setToolTip("Multiply base weight by (distance * factor) + offset")
        self.dist_dep_check.toggled.connect(self._toggle_dist_inputs)
        
        self.dist_factor_spin = QDoubleSpinBox()
        self.dist_factor_spin.setRange(-100.0, 100.0)
        self.dist_factor_spin.setValue(1.0)
        self.dist_factor_spin.setPrefix("Factor: ")
        self.dist_factor_spin.setEnabled(False)
        
        self.dist_offset_spin = QDoubleSpinBox()
        self.dist_offset_spin.setRange(-100.0, 100.0)
        self.dist_offset_spin.setValue(0.0)
        self.dist_offset_spin.setPrefix("Offset: ")
        self.dist_offset_spin.setEnabled(False)

        dist_layout.addWidget(self.dist_dep_check)
        dist_layout.addWidget(self.dist_factor_spin)
        dist_layout.addWidget(self.dist_offset_spin)
        
        layout.addLayout(dist_layout)
        layout.addSpacing(10)
        layout.addWidget(QLabel("Connectivity:"))
        prob_layout = QHBoxLayout()
        prob_layout.addWidget(QLabel("Probability (p):"))
        self.spatial_prob_spin = QDoubleSpinBox()
        self.spatial_prob_spin.setRange(0.0, 1.0)
        self.spatial_prob_spin.setValue(1.0)
        self.spatial_prob_spin.setSingleStep(0.1)
        prob_layout.addWidget(self.spatial_prob_spin)
        layout.addLayout(prob_layout)
        layout.addStretch()
    def reset_interface(self):
        """Setzt die Eingabefelder auf Standardwerte zurück."""
        self.conn_name_input.clear()
        
        # Spatial Tab Defaults
        self.mask_type_combo.setCurrentIndex(0) # sphere
        self.radius_spin.setValue(0.5)
        self.inner_radius_spin.setValue(0.0)
        self.dist_dep_check.setChecked(False)
        self.spatial_prob_spin.setValue(1.0)
        
        # Dist inputs (die werden durch toggle automatisch disabled)
        self.dist_factor_spin.setValue(1.0)
        self.dist_offset_spin.setValue(0.0)
        
        # Topological Tab Defaults
        self.rule_combo.setCurrentIndex(0) # all_to_all
        self.indegree_spin.setValue(10)
        self.outdegree_spin.setValue(10)
        self.total_num_spin.setValue(100)
        self.topo_prob_spin.setValue(0.1)
        
        # Synapse Defaults
        # Tipp: Wir lassen das Synapsen-Modell absichtlich stehen, 
        # damit man schneller mehrere ähnliche Verbindungen erstellen kann.
        self.weight_spin.setValue(1.0)
        self.delay_spin.setValue(1.0)
        self.allow_autapses_check.setChecked(True)
        self.allow_multapses_check.setChecked(True)
        self.receptor_spin.setValue(0)
        
        # Source/Target Combos lassen wir so, wie sie sind, 
        # damit der Workflow nicht unterbrochen wird.
    def _toggle_dist_inputs(self, checked):
        self.dist_factor_spin.setEnabled(checked)
        self.dist_offset_spin.setEnabled(checked)
        
    def _init_topological_tab(self):
        layout = QVBoxLayout(self.tab_topo)
        self.rule_combo = QComboBox()
        self.rule_combo.addItems(["all_to_all", "fixed_indegree", "fixed_outdegree", "fixed_total_number", "pairwise_bernoulli", "one_to_one"])
        self.rule_combo.currentTextChanged.connect(self.on_rule_changed)
        layout.addWidget(QLabel("Connection Rule:"))
        layout.addWidget(self.rule_combo)
        self.topo_params_widget = QWidget()
        self.topo_params_layout = QFormLayout(self.topo_params_widget)
        layout.addWidget(self.topo_params_widget)
        self.indegree_spin = QSpinBox(); self.indegree_spin.setRange(1, 100000); self.indegree_spin.setValue(10)
        self.outdegree_spin = QSpinBox(); self.outdegree_spin.setRange(1, 100000); self.outdegree_spin.setValue(10)
        self.total_num_spin = QSpinBox(); self.total_num_spin.setRange(1, 1000000); self.total_num_spin.setValue(100)
        self.topo_prob_spin = QDoubleSpinBox(); self.topo_prob_spin.setRange(0, 1); self.topo_prob_spin.setValue(0.1)
        self.on_rule_changed("all_to_all")
        layout.addStretch()

    def on_rule_changed(self, rule):
        while self.topo_params_layout.count():
            item = self.topo_params_layout.takeAt(0)
            if item.widget(): item.widget().setParent(None)
        if rule == "fixed_indegree": self.topo_params_layout.addRow("Indegree:", self.indegree_spin)
        elif rule == "fixed_outdegree": self.topo_params_layout.addRow("Outdegree:", self.outdegree_spin)
        elif rule == "fixed_total_number": self.topo_params_layout.addRow("Total Connections:", self.total_num_spin)
        elif "bernoulli" in rule: self.topo_params_layout.addRow("Probability:", self.topo_prob_spin)

    def _get_current_params(self):
        """Sammelt Parameter basierend auf dem aktiven Tab."""
        is_spatial = (self.tabs.currentIndex() == 0)
        
        params = {
            'synapse_model': self.syn_model_combo.currentText(), # ✅ FIX: self.syn_model_combo
            'weight': self.weight_spin.value(),
            'delay': self.delay_spin.value(),
            'allow_autapses': self.allow_autapses_check.isChecked(),
            'allow_multapses': self.allow_multapses_check.isChecked(),
            'receptor_type': self.receptor_spin.value(),
            'use_spatial': is_spatial
        }
        
        if is_spatial:
            params['rule'] = 'pairwise_bernoulli'
            params['p'] = self.spatial_prob_spin.value()
            params['mask_type'] = self.mask_type_combo.currentText()
            params['mask_radius'] = self.radius_spin.value()
            params['mask_inner_radius'] = self.inner_radius_spin.value()
            params['distance_dependent_weight'] = self.dist_dep_check.isChecked()
            params['dist_factor'] = self.dist_factor_spin.value()
            params['dist_offset'] = self.dist_offset_spin.value()
        else:
            rule = self.rule_combo.currentText()
            params['rule'] = rule
            if rule == 'fixed_indegree': params['indegree'] = self.indegree_spin.value()
            elif rule == 'fixed_outdegree': params['outdegree'] = self.outdegree_spin.value()
            elif rule == 'fixed_total_number': params['N'] = self.total_num_spin.value()
            elif 'bernoulli' in rule: params['p'] = self.topo_prob_spin.value()
                
        for name, widget in self.syn_param_widgets.items():
             params[name] = widget.get_value()
        return params

    def on_synapse_model_changed(self, model_name):
        while self.dynamic_syn_params_layout.count():
            item = self.dynamic_syn_params_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.syn_param_widgets.clear()
        if model_name not in SYNAPSE_MODELS: return
        params = SYNAPSE_MODELS[model_name]
        for param_name, info in params.items():
            p_type = info.get('type', 'float')
            p_default = info.get('default', 0.0)
            widget = None
            if p_type == 'float': widget = DoubleInputField(param_name, default_value=float(p_default))
            elif p_type == 'integer': widget = IntegerInputField(param_name, default_value=int(p_default))
            if widget:
                self.dynamic_syn_params_layout.addWidget(widget)
                self.syn_param_widgets[param_name] = widget

    def refresh_graph_list(self):
        self.source_graph_combo.clear()
        self.target_graph_combo.clear()
        for graph in self.graph_list:
            name = getattr(graph, 'graph_name', f'Graph {graph.graph_id}')
            self.source_graph_combo.addItem(f"{name} (ID: {graph.graph_id})", graph.graph_id)
            self.target_graph_combo.addItem(f"{name} (ID: {graph.graph_id})", graph.graph_id)
        if len(self.graph_list) > 0:
            self.on_source_graph_changed(0)
            self.on_target_graph_changed(0)

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
        for i, pop in enumerate(node.population):
            self.source_pop_combo.addItem(f"Pop {i}", i)

    def on_target_graph_changed(self, index):
        self.target_node_combo.clear()
        self.target_pop_combo.clear()
        if index < 0: return
        graph_id = self.target_graph_combo.currentData()
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph: return
        for node in graph.node_list:
            self.target_node_combo.addItem(f"{node.name} (ID: {node.id})", node.id)
        if len(graph.node_list) > 0: self.on_target_node_changed(0)

    def on_target_node_changed(self, index):
        self.target_pop_combo.clear()
        if index < 0: return
        graph_id = self.target_graph_combo.currentData()
        node_id = self.target_node_combo.currentData()
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph: return
        node = next((n for n in graph.node_list if n.id == node_id), None)
        if not node: return
        for i, pop in enumerate(node.population):
            self.target_pop_combo.addItem(f"Pop {i}", i)

    def add_connection(self):
        s_gid = self.source_graph_combo.currentData()
        s_nid = self.source_node_combo.currentData()
        s_pid = self.source_pop_combo.currentData()
        t_gid = self.target_graph_combo.currentData()
        t_nid = self.target_node_combo.currentData()
        t_pid = self.target_pop_combo.currentData()
        
        if None in [s_gid, s_nid, s_pid, t_gid, t_nid, t_pid]:
            print("Select Source and Target!")
            return

        params = self._get_current_params()
        name = self.conn_name_input.text() or f"Conn_{self.next_conn_id}"
        
        # --- ÄNDERUNG START ---
        # Wir erstellen jetzt eine VERSCHACHTELTE Struktur, die der Executor erwartet
        conn_dict = {
            'id': self.next_conn_id,
            'name': name,
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
            'params': params
        }
        
        self.connections.append(conn_dict)
        self.next_conn_id += 1
        self.update_connection_list()
        print(f"Added to Queue: {name}")

    def update_connection_list(self):
        while self.conn_list_layout.count():
            item = self.conn_list_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        for i, conn in enumerate(self.connections):
            mode = "🌍" if conn['params'].get('use_spatial', False) else "🕸️"
            btn = QPushButton(f"{mode} {conn['name']}")
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda checked, idx=i: self.select_connection(idx))
            self.conn_list_layout.addWidget(btn)
        self.conn_list_layout.addStretch()

    def select_connection(self, idx):
        self.current_conn_idx = idx
        self.delete_conn_btn.setEnabled(True)
        print(f"Selected connection index {idx}")

    def delete_connection(self):
        if self.current_conn_idx is not None:
            self.connections.pop(self.current_conn_idx)
            self.current_conn_idx = None
            self.delete_conn_btn.setEnabled(False)
            self.update_connection_list()

    def create_all_connections(self):
        if not self.connections:
            print("No connections to create!")
            return
        
        print("Creating connections...")
        
        # ✅ FIX: Executor JEDES MAL neu erstellen mit aktuellem Graph-Dictionary
        # Das ist wichtig, da sich die Graph-Objekte nach einem Rebuild ändern können!
        graphs_dict = {g.graph_id: g for g in self.graph_list}
        self.connection_executor = ConnectionExecutor(graphs_dict)

        # Ausführen
        success_count, fail_count, failed_items = self.connection_executor.execute_pending_connections(self.connections)
        
        print(f"Created: {success_count}, Failed: {fail_count}")
        
        # Update List Logic
        self.connections = failed_items
        self.update_connection_list()
        
        if fail_count == 0:
            self.status_label.setText(f"✅ All {success_count} connections created successfully.")
            self.reset_interface()
        else:
            self.status_label.setText(f"⚠️ {success_count} created, {fail_count} failed. Check red items.")

    def update_connection_list(self):
        """Aktualisiert die Liste. Färbt fehlerhafte Einträge rot."""
        
        # Alte Widgets entfernen
        while self.conn_list_layout.count():
            item = self.conn_list_layout.takeAt(0)
            if item.widget(): 
                item.widget().deleteLater()
            
        for i, conn in enumerate(self.connections):
            # Standard-Text
            mode_icon = "🌍" if conn['params'].get('use_spatial', False) else "🕸️"
            btn_text = f"{mode_icon} {conn['name']}"
            
            btn = QPushButton(btn_text)
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda checked, idx=i: self.select_connection(idx))
            
            # === FEHLER BEHANDLUNG ===
            if 'error' in conn:
                # Fehlerhafte Connection: Rötlicher Style
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4a1818; /* Dunkelrot */
                        color: #ff9999;            /* Helles Rot für Text */
                        border: 1px solid #ff5555; /* Roter Rand */
                        font-weight: bold;
                        text-align: left;
                        padding-left: 10px;
                    }
                    QPushButton:hover { background-color: #662222; }
                    QPushButton:pressed { background-color: #331111; }
                """)
                
                # Tooltip mit Fehlergrund (Rein informierend)
                error_msg = conn['error']
                # Bereinige evtl. vorhandene "NEST error:" Präfixe für saubere Anzeige
                clean_msg = error_msg.replace("NEST error:", "").strip()
                btn.setToolTip(f"Failed: {clean_msg}")
                
            else:
                # Normale (wartende) Connection
                btn.setStyleSheet("") # Standard Style
                btn.setToolTip("Pending creation")

            # Selektions-Highlight (überschreibt Style temporär, wenn ausgewählt)
            if i == self.current_conn_idx:
                btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
            
            self.conn_list_layout.addWidget(btn)
            
        self.conn_list_layout.addStretch()
    def set_source(self, graph_id, node_id, pop_id):
        """
        Setzt die Source-Comboboxen programmatisch auf die angegebenen IDs.
        Wird vom Kontextmenü im GraphOverview aufgerufen.
        """
        # 1. Graph auswählen
        idx_graph = self.source_graph_combo.findData(graph_id)
        if idx_graph >= 0:
            self.source_graph_combo.setCurrentIndex(idx_graph)
            # Durch das Signal 'currentIndexChanged' werden die Nodes neu geladen.
            # Wir erzwingen hier aber ein sofortiges Update der Node-Box, falls nötig, 
            # aber meistens reicht das Signal. Da Qt Signale synchron sein können, 
            # probieren wir den direkten Zugriff danach.
            
            # 2. Node auswählen
            idx_node = self.source_node_combo.findData(node_id)
            if idx_node >= 0:
                self.source_node_combo.setCurrentIndex(idx_node)
                
                # 3. Population auswählen
                idx_pop = self.source_pop_combo.findData(pop_id)
                if idx_pop >= 0:
                    self.source_pop_combo.setCurrentIndex(idx_pop)
                else:
                    print(f"⚠ Pop ID {pop_id} not found in source combo")
            else:
                print(f"⚠ Node ID {node_id} not found in source combo")
        else:
            print(f"⚠ Graph ID {graph_id} not found in source combo")

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
                        # === SOURCE ===
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
    
    rule = params.get('rule', 'all_to_all')
    synapse_model = params.get('synapse_model', 'static_synapse')
    
    conn_spec = {'rule': rule}
    
    if rule in CONNECTION_RULES:
        rule_params = CONNECTION_RULES[rule]['params']
        for p in rule_params:
            if p in params:
                conn_spec[p] = params[p]
    
    if rule == 'fixed_indegree' and 'indegree' not in conn_spec:
        conn_spec['indegree'] = 1
        warnings.append("fixed_indegree requires 'indegree', defaulting to 1")
    
    if rule == 'fixed_outdegree' and 'outdegree' not in conn_spec:
        conn_spec['outdegree'] = 1
        warnings.append("fixed_outdegree requires 'outdegree', defaulting to 1")
    
    if rule == 'fixed_total_number' and 'N' not in conn_spec:
        conn_spec['N'] = 10
        warnings.append("fixed_total_number requires 'N', defaulting to 10")
    
    if rule == 'pairwise_bernoulli' and 'p' not in conn_spec:
        conn_spec['p'] = 0.1
        warnings.append("pairwise_bernoulli requires 'p', defaulting to 0.1")
    
    syn_spec = {'synapse_model': synapse_model}
    
    syn_spec['weight'] = params.get('weight', 1.0)
    syn_spec['delay'] = validate_delay(params.get('delay', 1.0))
    if 'receptor_type' in params:
        syn_spec['receptor_type'] = int(params['receptor_type'])
    if synapse_model in SYNAPSE_MODELS:
        model_params = SYNAPSE_MODELS[synapse_model]
        for param_name, param_info in model_params.items():
            if param_name in params and param_name not in ['weight', 'delay']:
                syn_spec[param_name] = params[param_name]
    
    known_params = {'rule', 'synapse_model', 'weight', 'delay', 
                    'allow_autapses', 'allow_multapses', 'indegree', 
                    'outdegree', 'N', 'p', 'receptor_type'}
    if synapse_model in SYNAPSE_MODELS:
        known_params.update(SYNAPSE_MODELS[synapse_model].keys())
    
    for key in params:
        if key not in known_params:
            syn_spec[key] = params[key]
            warnings.append(f"Unknown parameter '{key}' passed to syn_spec")
    
    return conn_spec, syn_spec, warnings


class ConnectionExecutor:
    """
    Führt NEST-Verbindungen aus.
    
    Korrigierte Version mit NEST 3.x Mask-Support.
    """
    
    def __init__(self, graphs: Dict[int, Any]):
        """
        Args:
            graphs: Dictionary {graph_id: Graph object}
        """
        self.graphs = graphs
        self._connection_counter = 0
        self._created_models: List[str] = []  # Track für Cleanup
    
    def _get_next_connection_id(self) -> int:
        self._connection_counter += 1
        return self._connection_counter
    
    def _get_population(self, graph_id: int, node_id: int, pop_id: int):
        """Holt eine Population aus Graph/Node/Pop-IDs."""
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
        """
        Führt eine einzelne Verbindung aus.
        
        Args:
            connection: Dict mit 'source', 'target', 'params', 'name', 'id'
            
        Returns:
            (success: bool, message: str)
        """
        try:
            source_info = connection['source']
            target_info = connection['target']
            params = connection.get('params', {})
            conn_name = connection.get('name', f"Conn_{connection.get('id', '?')}")
            
            # === 1. Get Populations ===
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
            # === SPECIAL TOPOLOGY CHECK (CCW Ring, etc.) ===
            topo_type = params.get('topology_type')
            
            if topo_type == 'ring_ccw':
                # IDs holen
                src_ids = nest.GetStatus(src_pop, 'global_id')
                tgt_ids = nest.GetStatus(tgt_pop, 'global_id')
                
                n = len(src_ids)
                if len(tgt_ids) != n:
                    return False, f"CCW Ring requires equal size (Src: {n}, Tgt: {len(tgt_ids)})"
                
                w = float(params.get('weight', 10.0))
                d = float(params.get('delay', 1.0))
                
                # Listen für one_to_one mapping erstellen: i -> (i+1)%n
                pre_neurons = []
                post_neurons = []
                
                for i in range(n):
                    pre = src_ids[i]
                    post = tgt_ids[(i + 1) % n] # Wrap around
                    pre_neurons.append(pre)
                    post_neurons.append(post)
                
                # Verbindung erstellen
                nest.Connect(pre_neurons, post_neurons, 
                             {'rule': 'one_to_one'}, 
                             {'weight': w, 'delay': d, 'synapse_model': 'static_synapse'})

                # --- WICHTIG: AUCH HIER SPEICHERN DAMIT DIE GUI ES SIEHT ---
                try:
                    src_graph = self.graphs.get(source_info['graph_id'])
                    if src_graph:
                        src_node_obj = src_graph.get_node(source_info['node_id'])
                        if src_node_obj:
                            if not hasattr(src_node_obj, 'connections'):
                                src_node_obj.connections = []
                            
                            # Check auf Duplikate
                            exists = False
                            for existing in src_node_obj.connections:
                                if existing.get('id') == connection.get('id'):
                                    exists = True; break
                            
                            if not exists:
                                import copy
                                src_node_obj.connections.append(copy.deepcopy(connection))
                except Exception as e:
                    print(f"Warning: Could not save ring connection to model: {e}")
                # -----------------------------------------------------------
                
                return True, f"✓ CCW Ring created ({n} connections)"
            # === 2. Target Model Check (für HT-Neuron Fix) ===
            try:
                target_model = nest.GetStatus(tgt_pop, 'model')[0]
            except:
                target_model = 'unknown'
            
            # HT-Neuron Receptor Fix
            if 'ht_neuron' in str(target_model):
                rec_map = {1: 'AMPA', 2: 'NMDA', 3: 'GABA_A', 4: 'GABA_B'}
                current_rec = params.get('receptor_type', 0)
                if isinstance(current_rec, int) and current_rec in rec_map:
                    params['receptor_type'] = rec_map[current_rec]
                elif current_rec == 0:
                    params.pop('receptor_type', None)
            
            # === 3. Build conn_spec ===
            rule = params.get('rule', 'all_to_all')
            conn_spec = {'rule': rule}
            
            # Rule-specific params
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
            
            # === 4. Spatial Mask (NEST 3.x kompatibel!) ===
            if params.get('use_spatial', False):
                mask_type = params.get('mask_type', 'spherical')
                mask_params = {
                    'radius': params.get('mask_radius', 1.0),
                    'r': params.get('mask_radius', 1.0),
                    'inner_radius': params.get('mask_inner_radius', 0.0),
                    'outer_radius': params.get('mask_radius', 1.0),
                    'size': params.get('mask_radius', 1.0)
                }
                
                # Korrekter Mask-Typ Mapping
                mask_type_map = {
                    'sphere': 'spherical',
                    'spherical': 'spherical',
                    'box': 'rectangular',
                    'rectangular': 'rectangular',
                    'doughnut': 'doughnut'
                }
                actual_type = mask_type_map.get(mask_type, 'spherical')
                
                mask = create_nest_mask(actual_type, mask_params)
                if mask is not None:
                    conn_spec['mask'] = mask
            
            # === 5. Build Synapse Model ===
            base_model = params.get('synapse_model', 'static_synapse')
            
            # Check if delay should be excluded
            no_delay = base_model in ['gap_junction', 'rate_connection_instantaneous']
            
            # Keys that shouldn't go into CopyModel
            control_keys = {
                'rule', 'indegree', 'outdegree', 'N', 'p',
                'synapse_model', 'weight', 'delay',
                'allow_autapses', 'allow_multapses', 'receptor_type',
                'use_spatial', 'mask_type', 'mask_radius', 'mask_inner_radius',
                'distance_dependent_weight', 'dist_factor', 'dist_offset',
                'custom_name', 'no_delay', 'base_model'
            }
            
            # Extract model-specific params
            model_params = {
                k: v for k, v in params.items() 
                if k not in control_keys and v is not None
            }
            
            # Create custom model if needed
            final_model = base_model
            if model_params:
                ts = int(time.time() * 1e6)
                custom_name = params.get('custom_name') or f"{base_model}_{self._connection_counter}_{ts}"
                try:
                    nest.CopyModel(base_model, custom_name, model_params)
                    final_model = custom_name
                    self._created_models.append(custom_name)
                except Exception as e:
                    print(f"  ⚠ CopyModel failed: {e}, using {base_model}")
            
            # === 6. Build syn_spec ===
            syn_spec = {'synapse_model': final_model}
            
            # Weight (possibly distance-dependent)
            weight = float(params.get('weight', 1.0))
            if params.get('use_spatial') and params.get('distance_dependent_weight'):
                factor = float(params.get('dist_factor', 1.0))
                offset = float(params.get('dist_offset', 0.0))
                syn_spec['weight'] = create_distance_dependent_weight(
                    weight, factor, offset
                )
            else:
                syn_spec['weight'] = weight
            
            # Delay (nur wenn erlaubt)
            if not no_delay:
                delay = float(params.get('delay', 1.0))
                syn_spec['delay'] = max(delay, nest.resolution)
            
            # Receptor Type
            receptor = params.get('receptor_type')
            if receptor is not None and receptor != 0:
                syn_spec['receptor_type'] = receptor
            
            # === 7. Size Check für one_to_one ===
            if rule == 'one_to_one' and len(src_pop) != len(tgt_pop):
                return False, f"one_to_one size mismatch: {len(src_pop)} vs {len(tgt_pop)}"
            
            # === 8. Execute Connection! ===
            nest.Connect(src_pop, tgt_pop, conn_spec, syn_spec)
            
            # --- SAVE TO GRAPH MODEL (FIX FOR GUI OVERVIEW) ---
            # Wir müssen die Verbindung auch im Node-Objekt speichern, damit der TreeView sie sieht.
            try:
                src_graph = self.graphs.get(source_info['graph_id'])
                if src_graph:
                    src_node_obj = src_graph.get_node(source_info['node_id'])
                    if src_node_obj:
                        if not hasattr(src_node_obj, 'connections'):
                            src_node_obj.connections = []
                        
                        # Prüfen ob Verbindung schon existiert (um Duplikate zu vermeiden)
                        exists = False
                        for existing in src_node_obj.connections:
                            if existing.get('id') == connection.get('id'):
                                exists = True
                                break
                        
                        if not exists:
                            # Wichtig: Kopie speichern, damit Queue-Referenzen keine Rolle spielen
                            import copy
                            src_node_obj.connections.append(copy.deepcopy(connection))
            except Exception as save_err:
                print(f"Warning: Connection created in NEST but failed to save to Model: {save_err}")
            # --------------------------------------------------

            return True, f"✓ {conn_name}: {len(src_pop)}→{len(tgt_pop)} neurons"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)
        


    def execute_pending_connections(self, connections: List[Dict[str, Any]]) -> Tuple[int, int, List[Dict[str, Any]]]:
        """
        Führt eine Liste von Verbindungen aus und gibt die fehlgeschlagenen Objekte zurück.
        Dies wird vom ConnectionTool benötigt, um die Queue zu aktualisieren.
        """
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
                # Speichere den Fehlergrund direkt im Objekt für die GUI (rotes Highlight/Tooltip)
                conn['error'] = msg
                failed_items.append(conn)
                print(f"  ✗ {conn.get('name', '?')}: {msg}")

        return success_count, fail_count, failed_items
    
    def execute_all(self, connections: List[Dict[str, Any]], 
                   progress_callback=None) -> Tuple[int, int, List[str]]:
        """
        Führt alle Verbindungen aus.
        
        Args:
            connections: Liste von Connection-Dicts
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            (created, failed, error_messages)
        """
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
        """Entfernt alle erstellten Custom-Models (optional)."""
        # NEST unterstützt kein direktes Löschen von Modellen,
        # aber wir können die Liste für Debugging nutzen
        self._created_models.clear()


class ConnectionQueueWidget(QWidget):
    """Widget zur Anzeige und Verwaltung der Connection-Queue."""
    
    connectionSelected = pyqtSignal(int)  # index
    connectionRemoved = pyqtSignal(int)   # index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.connections: List[Dict] = []
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("📋 Connection Queue")
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
        
        # Status
        self.status_label = QLabel("0 connections")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)
    
    def add_connection(self, conn: Dict):
        """Fügt eine Verbindung hinzu."""
        self.connections.append(conn)
        
        # Create list item
        name = conn.get('name', f"Connection {len(self.connections)}")
        src = conn.get('source', {})
        tgt = conn.get('target', {})
        
        label = f"{name}: G{src.get('graph_id',0)}N{src.get('node_id',0)}P{src.get('pop_id',0)} → " \
                f"G{tgt.get('graph_id',0)}N{tgt.get('node_id',0)}P{tgt.get('pop_id',0)}"
        
        item = QListWidgetItem(label)
        
        # Color based on synapse type
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
    
    # Collect all connections from all nodes in all graphs
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
    
    # Recreate each connection
    for conn in all_connections:
        conn_name = conn.get('name', f"Connection_{conn['id']}")
        source = conn['source']
        target = conn['target']
        params = conn['params']
        
        try:
            # Get source and target populations
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
            
            # Check populations exist
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
            
            # Validate and execute
            conn_spec, syn_spec, warnings = validate_connection_params(params)
            
            # Special check for one_to_one
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
                
                # Populate with NEST neurons
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
    
    # 1. Reset NEST kernel
    if verbose:
        print("\nResetting NEST kernel...")
    nest.ResetKernel()
    
    # 2. Re-enable structural plasticity if requested
    if enable_structural_plasticity:
        try:
            nest.EnableStructuralPlasticity()
            if verbose:
                print("✓ Structural plasticity enabled")
        except Exception as e:
            if verbose:
                print(f"Could not enable structural plasticity: {e}")
            enable_structural_plasticity = False
    
    # 3. Repopulate all nodes
    total_pops = repopulate_all_graphs(graphs, verbose=verbose)
    
    # 4. Recreate all connections
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
                
                # Count by synapse model
                model = params.get('synapse_model', 'static_synapse')
                summary['by_synapse_model'][model] = summary['by_synapse_model'].get(model, 0) + 1
                
                # Count by rule
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
    # Rule-specific params
    indegree: int = None,
    outdegree: int = None,
    N: int = None,
    p: float = None,
    # Synapse-specific params
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
    
    # Add rule-specific params if provided
    if indegree is not None:
        params['indegree'] = indegree
    if outdegree is not None:
        params['outdegree'] = outdegree
    if N is not None:
        params['N'] = N
    if p is not None:
        params['p'] = p
    
    # Add synapse-specific params
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
    """Prüft ob SYNAPSE_MODELS nur echte NEST Synapse-Modelle enthält."""
    import nest
    
    nest_synapses = set(nest.SYNAPSE_MODELS)
    
    
    invalid = []
    valid = []
    
    for model in SYNAPSE_MODELS.keys():
        if model in nest_synapses:
            valid.append(model)
            print(f"  ✅ {model}")
        else:
            invalid.append(model)
            print(f"{model} - NOT A VALID SYNAPSE MODEL!")
    
    print(f"\nValid: {len(valid)} | Invalid: {len(invalid)}")
    
    if invalid:
        print("\nREMOVE THESE FROM SYNAPSE_MODELS:")
        for m in invalid:
            print(f'    "{m}": {{}},')
    
    return invalid






from pathlib import Path
from datetime import datetime




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
        
        available_models = ["iaf_psc_alpha", "iaf_psc_exp", "iaf_psc_delta"]
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
        self.save_btn = QPushButton("💾 Save Graph")
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
        
        # Load Button
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
        
        # Buttons erstellen
        self.btn_start = QPushButton("Simulate")
        self.btn_start.setStyleSheet(start_style)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet(stop_style)
        
        self.btn_save = QPushButton("💾 Save Graphs")
        self.btn_save.setStyleSheet(io_style)
    
        self.btn_load = QPushButton("📂 Load Graphs")
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
            node_data = {
                'id': node.id,
                'name': node.name,
                'graph_id': node.graph_id if hasattr(node, 'graph_id') else 0,
                'parameters': _clean_params(node.parameters) if hasattr(node, 'parameters') else {},
                'center_of_mass': list(node.center_of_mass) if hasattr(node, 'center_of_mass') else [0,0,0],
                'types': node.types if hasattr(node, 'types') else [],
                'neuron_models': node.neuron_models if hasattr(node, 'neuron_models') else [],
                'distribution': list(node.distribution) if hasattr(node, 'distribution') and node.distribution else [],
                'connections': _serialize_connections(node.connections) if hasattr(node, 'connections') else [],
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
            
            if 'center_of_mass' in nd:
                params['center_of_mass'] = np.array(nd['center_of_mass'])
            
            parent_id = nd.get('parent_id')
            parent = graph.get_node(parent_id) if parent_id is not None else None
            is_root = (nd['id'] == 0 or parent_id is None)
            
            new_node = graph.create_node(
                parameters=params,
                other=parent,
                is_root=is_root,
                auto_build=False
            )
            
            if nd.get('positions'):
                new_node.positions = [np.array(pos) for pos in nd['positions']]
                new_node.center_of_mass = np.array(nd['center_of_mass'])
        
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
    """
    Linker Bereich im Tool-Menü: Wählt aus, WO das Gerät angeschlossen wird.
    """
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
    """
    Konfigurationsseite für ein spezifisches Gerät.
    Generiert Parameter-Felder basierend auf dem Device-Typ und parst Eingaben.
    """
    deviceCreated = pyqtSignal(dict) # Sendet Daten zurück an ToolsWidget

    def __init__(self, device_label, device_type, graph_list, parent=None):
        super().__init__(parent)
        self.device_type = device_type
        self.device_label = device_label
        self.graph_list = graph_list
        self.param_widgets = {}
        
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- LINKE SEITE: Target Selector ---
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0,0,0,0)
        
        self.target_selector = DeviceTargetSelector(self.graph_list)
        left_layout.addWidget(self.target_selector)
        left_layout.addStretch()
        
        # --- RECHTE SEITE: Parameter ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        
        # Header
        header = QLabel(f"Configure {self.device_label}")
        header.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {self._get_color()};")
        right_layout.addWidget(header)
        
        # Formular
        form_group = QGroupBox("Device Parameters")
        self.form_layout = QFormLayout(form_group)
        
        self._build_parameters()
        
        right_layout.addWidget(form_group)
        right_layout.addStretch()
        
        # Create Button
        self.btn_create = QPushButton(f"✚ Create {self.device_label}")
        self.btn_create.setMinimumHeight(45)
        self.btn_create.setStyleSheet(f"background-color: {self._get_color()}; color: white; font-weight: bold;")
        self.btn_create.clicked.connect(self.create_device)
        right_layout.addWidget(self.btn_create)
        
        main_layout.addWidget(left_container, 1)
        main_layout.addWidget(right_container, 2)

    def _get_color(self):
        if "generator" in self.device_type: return "#FF9800" # Orange
        if "recorder" in self.device_type or "meter" in self.device_type: return "#E91E63" # Pink
        return "#999"

    def _build_parameters(self):
        """Erstellt Felder basierend auf NEST Dokumentation + Verbindungsparameter."""
        
        # --- A. DEVICE PARAMETERS ---
        self.form_layout.addRow(QLabel("--- Device Settings ---"))
        
        # Standards für alle
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

        # --- B. CONNECTION PARAMETERS (NEU) ---
        self.form_layout.addRow(QLabel("--- Connection Settings ---"))
        
        # Standard-Gewicht höher setzen für Generatoren, damit man was sieht
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
        """Hilfsfunktion: String '1, 2, 3' -> Liste [1.0, 2.0, 3.0]"""
        if not text.strip():
            return []
        try:
            # Entferne Klammern falls User [1,2] eingibt
            cleaned = text.replace('[', '').replace(']', '')
            parts = cleaned.split(',')
            return [dtype(p.strip()) for p in parts if p.strip()]
        except Exception as e:
            print(f"Parsing error for input '{text}': {e}")
            return []

    
    def create_device(self):
        target = self.target_selector.get_selection()
        if target['graph_id'] is None or target['node_id'] is None or target['pop_id'] is None:
            print("Error: No target selected!")
            return
            
        # Parameter extrahieren und trennen
        device_params = {}
        conn_params = {}
        
        for key, widget in self.param_widgets.items():
            val = None
            
            # Value holen
            if isinstance(widget, QDoubleSpinBox):
                val = widget.value()
            elif isinstance(widget, QLineEdit):
                text_val = widget.text()
                # Listen Parsing
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
            
            # Sortieren: Ist es Connection-Parameter oder Device-Parameter?
            if key.startswith("conn_"):
                # "conn_weight" -> "weight"
                clean_key = key.replace("conn_", "")
                conn_params[clean_key] = val
            else:
                if val is not None: # Leere Strings ignorieren? Ggf. anpassen
                    device_params[key] = val
        
        # Datenpaket schnüren
        device_data = {
            "model": self.device_type,
            "target": target,
            "params": device_params,      # Geht an nest.Create
            "conn_params": conn_params    # Geht an nest.Connect
        }
        
        self.deviceCreated.emit(device_data)



class ToolsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.graph_list = [] # Wird von CleanAlpha gesetzt/aktualisiert
        self.button_map = {}
        self.init_ui()
        
    def update_graphs(self, graph_list):
        """Wird von CleanAlpha aufgerufen, wenn sich Graphen ändern."""
        self.graph_list = graph_list
        # Refresh aktiven Selector
        current_widget = self.config_stack.currentWidget()
        if isinstance(current_widget, DeviceConfigPage):
            current_widget.target_selector.graph_list = graph_list
            current_widget.target_selector.refresh()
            
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # === MENÜ ===
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
        
        # Startseite
        info_page = QLabel("Select a device from the left menu.")
        info_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_page.setStyleSheet("color: #666;")
        self.config_stack.addWidget(info_page)
        
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
            
            # Page erstellen
            # Wir übergeben self.graph_list (Referenz)
            page = DeviceConfigPage(label, model, self.graph_list)
            page.deviceCreated.connect(self.on_device_created)
            
            idx = self.config_stack.addWidget(page)
            btn.clicked.connect(lambda checked, i=idx, b=btn: self.switch_tool(i, b))
            self.button_map[btn] = idx
            menu_layout.addWidget(btn)
            
        menu_layout.addStretch()
        
        main_layout.addWidget(menu_container)
        main_layout.addWidget(self.config_stack)
        
    def switch_tool(self, index, clicked_btn):
        for btn in self.button_map:
            btn.setChecked(btn == clicked_btn)
        self.config_stack.setCurrentIndex(index)
        
        # Refresh data on switch
        widget = self.config_stack.widget(index)
        if isinstance(widget, DeviceConfigPage):
            widget.target_selector.graph_list = self.graph_list
            widget.target_selector.refresh()

    def on_device_created(self, data):
        """
        Logik zum Erstellen des Devices in NEST und Speichern im Graph-Objekt.
        Nutzt jetzt User-definierte Gewichte.
        """
        print(f"\nCreating Device: {data['model']}")
        
        target = data['target']
        graph_id = target['graph_id']
        node_id = target['node_id']
        pop_id = target['pop_id']
        
        # 1. Node finden
        target_graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not target_graph: return
        target_node = next((n for n in target_graph.node_list if n.id == node_id), None)
        if not target_node: return
        
        if not hasattr(target_node, 'devices'):
            target_node.devices = []
            
        # 2. NEST Device erstellen
        try:
            device = nest.Create(data['model'], params=data['params'])
            print(f"  ✓ NEST ID: {device}")
            
            # 3. Verbinden mit User-Parametern
            pop_nest = target_node.population[pop_id]
            
            # Synapse Spec bauen
            conn_params = data.get('conn_params', {})
            syn_spec = {
                'weight': float(conn_params.get('weight', 1.0)),
                'delay': max(float(conn_params.get('delay', 1.0)), 0.1)
            }
            
            if "generator" in data['model']:
                nest.Connect(device, pop_nest, syn_spec=syn_spec)
                print(f"  ✓ Connected Generator -> Population (W={syn_spec['weight']}, D={syn_spec['delay']})")
            else:
                # Recorder: Gewicht ist hier meist irrelevant, aber Delay zählt
                nest.Connect(pop_nest, device, syn_spec=syn_spec)
                print(f"  ✓ Connected Population -> Recorder (D={syn_spec['delay']})")
                
            # 4. Speichern im Datenmodell
            device_record = {
                "id": len(target_node.devices),
                "model": data['model'],
                "target_pop_id": pop_id,
                "params": data['params'],
                "conn_params": conn_params # Speichern für späteres Laden/Anzeigen
            }
            target_node.devices.append(device_record)
            print(f"  ✓ Saved to Node Model (Total devices: {len(target_node.devices)})")
            
            # Feedback
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Success", f"{data['model']} created!\nWeight: {syn_spec['weight']}")
            
        except Exception as e:
            print(f"Error creating device: {e}")
            import traceback
            traceback.print_exc()



class StructuresWidget(QWidget):
    """
    Zeigt Buttons für biologische Kortex-Regionen an.
    Bei Klick wird ein Signal mit den Modellen und Verteilungen gesendet.
    """
    # Signal sendet: (Region Name, Liste der Modelle, Liste der Wahrscheinlichkeiten)
    structureSelected = pyqtSignal(str, list, list)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header_lbl = QLabel("🧠 CORTICAL STRUCTURES")
        header_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #E91E63; border-bottom: 2px solid #E91E63; padding-bottom: 5px;")
        layout.addWidget(header_lbl)
        
        info = QLabel("Select a region to auto-generate a node patch (10x10x10)\nwith biological neuron distributions.")
        info.setStyleSheet("color: #888; font-style: italic; margin-bottom: 10px;")
        layout.addWidget(info)

        # Scroll Area für die Regionen
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        
        container = QWidget()
        self.grid_layout = QGridLayout(container)
        self.grid_layout.setSpacing(10)
        
        # --- DATEN AUS NEURON_TOOLBOX LADEN ---
        # Wir müssen die Daten transponieren: Von Model->Region zu Region->ModelList
        
        # 1. Dictionary vorbereiten: RegionKey -> {Model: Prob}
        region_data = {r_key: {} for r_key in region_names.values()}
        
        for model, region_map in distributions.items():
            for r_key, prob in region_map.items():
                if prob > 0:
                    region_data[r_key][model] = prob

        # 2. Buttons erstellen
        row, col = 0, 0
        for display_name, r_key in region_names.items():
            if r_key not in region_data or not region_data[r_key]:
                continue
            
            # Modelle und Probs extrahieren
            models_probs = region_data[r_key]
            model_list = list(models_probs.keys())
            prob_list = list(models_probs.values())
            
            # Normalisieren auf 1.0
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
            
            # Tooltip bauen
            tooltip_txt = f"<b>{display_name}</b><br>Contains:<br>"
            for m, p in zip(model_list, prob_list):
                tooltip_txt += f"• {m}: {p*100:.1f}%<br>"
            btn.setToolTip(tooltip_txt)
            
            # Signal verbinden (Nutze Default-Args um Closure-Problem zu vermeiden)
            btn.clicked.connect(lambda checked, n=display_name, m=model_list, p=prob_list: 
                              self.structureSelected.emit(n, m, p))
            
            self.grid_layout.addWidget(btn, row, col)
            
            col += 1
            if col > 1: # 2 Spalten
                col = 0
                row += 1
        
        scroll.setWidget(container)
        layout.addWidget(scroll)









class FlowFieldWidget(QWidget):
    """
    Visualisiert die Vektorfelder (Flow Fields) NUR für den aktuell ausgewählten Node.
    Performance-Optimiert: Berechnet Glyphen nur on-demand für einen Node.
    """
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.target_graph_id = None
        self.target_node_id = None
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # PyVista Setup
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

    def set_target_node(self, graph_id, node_id):
        """Setzt den Fokus auf einen spezifischen Node und rendert neu."""
        if self.target_graph_id == graph_id and self.target_node_id == node_id:
            return # Keine Änderung
            
        self.target_graph_id = graph_id
        self.target_node_id = node_id
        
        # Nur neu bauen, wenn wir sichtbar sind, sonst reicht das Setzen der IDs
        if self.isVisible():
            self.build_scene()

    def build_scene(self):
        self.plotter.clear()
        self.plotter.renderer.AddViewProp(self.text_actor) 
        
        if not self.graph_list:
            self.text_actor.SetInput("No graphs available.")
            self.plotter.render()
            return

        if self.target_graph_id is None or self.target_node_id is None:
            self.text_actor.SetInput("Select a node in 'Graph Overview' to inspect Flow Field.")
            self.plotter.render()
            return

        # Finde Graph und Node
        target_graph = next((g for g in self.graph_list if g.graph_id == self.target_graph_id), None)
        if not target_graph:
            self.text_actor.SetInput(f"Graph {self.target_graph_id} not found.")
            self.plotter.render()
            return
            
        target_node = next((n for n in target_graph.node_list if n.id == self.target_node_id), None)
        if not target_node:
            self.text_actor.SetInput(f"Node {self.target_node_id} not found in Graph.")
            self.plotter.render()
            return

        self.text_actor.SetInput(f"Flow Field: {target_node.name} (Graph {target_graph.graph_id})")

        if not hasattr(target_node, 'positions') or not target_node.positions:
            return

        params = target_node.parameters
        encoded_per_type = params.get("encoded_polynoms_per_type", [])
        poly_gen = PolynomGenerator(n=params.get('polynom_max_power', 5))

        # --- FIX: Legende nur einmal anzeigen ---
        show_legend = True 

        for pop_idx, positions in enumerate(target_node.positions):
            if positions is None or len(positions) == 0:
                continue
            
            f1, f2, f3 = None, None, None
            if pop_idx < len(encoded_per_type):
                try:
                    funcs = poly_gen.decode_multiple(encoded_per_type[pop_idx])
                    if len(funcs) == 3:
                        f1, f2, f3 = funcs
                except Exception as e:
                    print(f"Error decoding polys: {e}")
            
            if f1 is None: continue 

            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]
            
            u = f1(x, y, z)
            v = f2(x, y, z)
            w = f3(x, y, z)
            
            vectors = np.column_stack((u, v, w))
            magnitudes = np.linalg.norm(vectors, axis=1)
            safe_mags = np.where(magnitudes == 0, 1, magnitudes)
            unit_vectors = vectors / safe_mags[:, None]
            geom_scale = np.clip(magnitudes, 0, 2.0)
            
            pdata = pv.PolyData(positions)
            pdata['vectors'] = unit_vectors
            pdata['mag'] = magnitudes
            pdata['scale'] = geom_scale
            
            arrows = pdata.glyph(orient='vectors', scale='scale', factor=0.8)
            
            # Hier wenden wir den Fix an: show_scalar_bar nur wenn show_legend True ist
            self.plotter.add_mesh(
                arrows, 
                scalars='mag', 
                cmap='jet', 
                opacity=0.9,
                show_scalar_bar=show_legend,  # <--- WICHTIG
                scalar_bar_args={'title': 'Flow Magnitude', 'color': 'white'}
            )
            
            # Nach dem ersten Hinzufügen deaktivieren wir weitere Legenden
            show_legend = False 
            
            self.plotter.add_mesh(
                pv.PolyData(positions), 
                color="#444444", 
                point_size=4, 
                render_points_as_spheres=True
            )

        self.plotter.reset_camera()
        self.plotter.update()

class SimulationDashboardWidget(QWidget):
    """
    Zentrale Steuereinheit für die Simulation.
    Zeigt Statistiken, Pie-Charts und Konfigurationen vor dem Start.
    """
    requestStartSimulation = pyqtSignal(float) # sendet duration
    requestOpenSpectator = pyqtSignal()
    requestShowResults = pyqtSignal()

    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- HEADER ---
        header = QLabel("SIMULATION DASHBOARD")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF9800; letter-spacing: 2px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        # --- CONTENT AREA (Grid) ---
        content_layout = QHBoxLayout()
        
        # LINKS: Konfiguration & Stats
        left_col = QVBoxLayout()
        
        # 1. Config Group
        config_group = QGroupBox("⚙️ Configuration")
        config_layout = QFormLayout(config_group)
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 1000000.0)
        self.duration_spin.setValue(1000.0)
        self.duration_spin.setSuffix(" ms")
        self.duration_spin.setStyleSheet("font-size: 14px; padding: 5px;")
        
        self.input_placeholder = QLineEdit("No external input file selected")
        self.input_placeholder.setReadOnly(True)
        self.input_placeholder.setStyleSheet("color: #888; font-style: italic;")
        
        config_layout.addRow("Simulation Duration:", self.duration_spin)
        config_layout.addRow("External Input:", self.input_placeholder)
        
        left_col.addWidget(config_group)
        
        # 2. Stats Group
        stats_group = QGroupBox("📊 Network Summary")
        self.stats_layout = QFormLayout(stats_group)
        self.lbl_nodes = QLabel("0")
        self.lbl_pops = QLabel("0")
        self.lbl_neurons = QLabel("0")
        self.lbl_conns = QLabel("0")
        self.lbl_devices = QLabel("0")
        
        # Style für Zahlen
        number_style = "color: #00E5FF; font-weight: bold; font-size: 12px;"
        self.lbl_nodes.setStyleSheet(number_style)
        self.lbl_pops.setStyleSheet(number_style)
        self.lbl_neurons.setStyleSheet(number_style)
        self.lbl_conns.setStyleSheet(number_style)
        self.lbl_devices.setStyleSheet(number_style)

        self.stats_layout.addRow("Total Nodes:", self.lbl_nodes)
        self.stats_layout.addRow("Populations:", self.lbl_pops)
        self.stats_layout.addRow("Total Neurons:", self.lbl_neurons)
        self.stats_layout.addRow("Connections:", self.lbl_conns)
        self.stats_layout.addRow("Devices:", self.lbl_devices)
        
        left_col.addWidget(stats_group)
        left_col.addStretch()
        
        content_layout.addLayout(left_col, 1)

        # RECHTS: Pie Chart
        right_col = QVBoxLayout()
        chart_group = QGroupBox("Neuron Distribution")
        chart_layout = QVBoxLayout(chart_group)
        
        self.figure = Figure(figsize=(4, 4), facecolor='#2b2b2b')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #2b2b2b;")
        chart_layout.addWidget(self.canvas)
        
        right_col.addWidget(chart_group)
        content_layout.addLayout(right_col, 1)

        main_layout.addLayout(content_layout)

        # --- ACTIONS AREA ---
        action_container = QWidget()
        action_layout = QVBoxLayout(action_container)
        action_layout.setSpacing(10)
        
        # Start Button
        self.btn_start = QPushButton("▶ START SIMULATION")
        self.btn_start.setMinimumHeight(60)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                font-size: 18px; 
                border-radius: 8px;
                border-bottom: 4px solid #2E7D32;
            }
            QPushButton:hover { background-color: #66BB6A; margin-top: 2px; border-bottom: 2px solid #2E7D32; }
            QPushButton:pressed { background-color: #2E7D32; margin-top: 4px; border-bottom: none; }
        """)
        self.btn_start.clicked.connect(self._on_start_clicked)
        
        # Live Spectator
        self.btn_spectator = QPushButton("👁️ Open Live Spectator")
        self.btn_spectator.setMinimumHeight(40)
        self.btn_spectator.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; color: white; font-weight: bold; font-size: 14px; border-radius: 5px;
            }
            QPushButton:hover { background-color: #42A5F5; }
        """)
        self.btn_spectator.clicked.connect(self.requestOpenSpectator.emit)

        # Results (Disabled)
        self.btn_results = QPushButton("📊 Results (No Data)")
        self.btn_results.setMinimumHeight(40)
        self.btn_results.setEnabled(False)
        self.btn_results.setStyleSheet("""
            QPushButton {
                background-color: #444; color: #888; font-weight: bold; border-radius: 5px; border: 1px solid #555;
            }
            QPushButton:enabled {
                background-color: #9C27B0; color: white;
            }
        """)
        self.btn_results.clicked.connect(self.requestShowResults.emit)

        action_layout.addWidget(self.btn_start)
        action_layout.addWidget(self.btn_spectator)
        action_layout.addWidget(self.btn_results)
        
        main_layout.addWidget(action_container)

    def _on_start_clicked(self):
        duration = self.duration_spin.value()
        self.requestStartSimulation.emit(duration)

    def refresh_data(self):
        """Berechnet Statistiken neu und zeichnet das Pie Chart."""
        total_nodes = 0
        total_pops = 0
        total_neurons = 0
        total_conns = 0
        total_devices = 0
        
        model_counts = {}

        for graph in self.graph_list:
            total_nodes += len(graph.node_list)
            
            for node in graph.node_list:
                # Connections
                if hasattr(node, 'connections'):
                    total_conns += len(node.connections)
                
                # Devices
                if hasattr(node, 'devices'):
                    total_devices += len(node.devices)
                
                # Populations & Neurons
                if hasattr(node, 'population') and node.population:
                    for i, nest_pop in enumerate(node.population):
                        if nest_pop:
                            count = len(nest_pop)
                            total_neurons += count
                            total_pops += 1
                            
                            # Model name
                            model_name = "unknown"
                            if hasattr(node, 'neuron_models') and i < len(node.neuron_models):
                                model_name = node.neuron_models[i]
                            
                            model_counts[model_name] = model_counts.get(model_name, 0) + count

        # Update Labels
        self.lbl_nodes.setText(str(total_nodes))
        self.lbl_pops.setText(str(total_pops))
        self.lbl_neurons.setText(str(total_neurons))
        self.lbl_conns.setText(str(total_conns))
        self.lbl_devices.setText(str(total_devices))
        
        # Update Chart
        self._plot_pie_chart(model_counts)

    def _plot_pie_chart(self, counts):
        self.figure.clear()
        
        if not counts:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='white')
            ax.axis('off')
            self.canvas.draw()
            return

        labels = list(counts.keys())
        sizes = list(counts.values())
        
        # Farben holen
        colors = [neuron_colors.get(l, "#888888") for l in labels]
        
        ax = self.figure.add_subplot(111)
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops=dict(color="white")
        )
        
        # Style Anpassungen für Dark Mode
        for text in texts:
            text.set_color("#cccccc")
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_color("black") 
            autotext.set_weight("bold")
            
        self.figure.patch.set_facecolor('#2b2b2b') # Hintergrund des Plots
        
        self.canvas.draw()

class LiveSpectatorWindow(QMainWindow):
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Spectator (PyQtGraph)")
        self.resize(1000, 800)
        self.graph_list = graph_list
        
        self.sim_running = False
        self.current_time = 0.0
        self.update_interval = 25.0  # ms pro Schritt
        self.timer_interval = 20     # ms (50 FPS target)
        
        self.all_positions = None
        self.neuron_colors = None
        self.neuron_activity = None
        self.gid_to_idx = {}
        self.global_recorder = None
        
        self.init_ui()
        self.build_scene_data()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # 1. View
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 60
        self.view.setBackgroundColor('#111111') # Soft Black
        layout.addWidget(self.view, 1)
        
        # 2. Controls
        controls = QHBoxLayout()
        
        self.btn_play = QPushButton("▶ START")
        self.btn_play.setStyleSheet("background-color: #4CAF50; font-weight: bold; padding: 10px;")
        self.btn_play.clicked.connect(self.toggle_simulation)
        controls.addWidget(self.btn_play)
        
        self.lbl_stats = QLabel("Waiting...")
        self.lbl_stats.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 14px; margin-left: 20px;")
        controls.addWidget(self.lbl_stats)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)

    def build_scene_data(self):
        print("Building Spectator Scene...")
        self.view.clear()
        
        pos_list = []
        gids_list = []
        
        for graph in self.graph_list:
            for node in graph.node_list:
                if hasattr(node, 'population') and node.population:
                    for i, pop in enumerate(node.population):
                        if pop:
                            if hasattr(node, 'positions') and i < len(node.positions):
                                coords = node.positions[i]
                                try:
                                    status = nest.GetStatus(pop, 'global_id')
                                    if isinstance(status, tuple): gids = status
                                    else: gids = list(status)
                                    
                                    if len(coords) == len(gids):
                                        pos_list.append(coords)
                                        gids_list.extend(gids)
                                    else:
                                        min_len = min(len(coords), len(gids))
                                        pos_list.append(coords[:min_len])
                                        gids_list.extend(gids[:min_len])
                                        
                                except Exception as e:
                                    print(f"Warning skipping pop {i}: {e}")

        if not pos_list:
            if hasattr(self, 'lbl_stats'):
                self.lbl_stats.setText("No neurons found.")
            return

        self.all_positions = np.vstack(pos_list)
        num_neurons = len(self.all_positions)
        
        self.gid_to_idx = {gid: i for i, gid in enumerate(gids_list)}
        
        try:
            if self.global_recorder is None:
                self.global_recorder = nest.Create("spike_recorder")
                nest.Connect(list(self.gid_to_idx.keys()), self.global_recorder)
                print(f"Spectator connected to {num_neurons} neurons.")
        except Exception as e:
            print(f"Recorder Error: {e}")
        
        self.neuron_colors = np.zeros((num_neurons, 4), dtype=np.float32)
        self.neuron_colors[:, 0] = 0.0 
        self.neuron_colors[:, 1] = 0.2
        self.neuron_colors[:, 2] = 0.8
        self.neuron_colors[:, 3] = 0.6 
        
        self.neuron_activity = np.zeros(num_neurons, dtype=np.float32)

        self.scatter = gl.GLScatterPlotItem(
            pos=self.all_positions,
            color=self.neuron_colors,
            size=5, 
            pxMode=True 
        )
        

        self.scatter.setGLOptions('translucent') 
        
        self.view.addItem(self.scatter)

    def toggle_simulation(self):
        if self.sim_running:
            self.sim_running = False
            self.btn_play.setText("▶ RESUME")
            self.btn_play.setStyleSheet("background-color: #4CAF50; font-weight: bold; padding: 10px;")
            self.timer.stop()
        else:
            self.sim_running = True
            self.btn_play.setText("⏸ PAUSE")
            self.btn_play.setStyleSheet("background-color: #FF9800; font-weight: bold; padding: 10px;")
            self.timer.start(self.timer_interval)

    def simulation_step(self):
        if not self.sim_running: return
        
        try:
            # 1. NEST simulieren
            nest.Simulate(self.update_interval)
            self.current_time += self.update_interval
            
            # 2. Spikes holen
            events = nest.GetStatus(self.global_recorder, "events")[0]
            senders = events["senders"]
            
            # WICHTIG: Recorder leeren für nächsten Step!
            nest.SetStatus(self.global_recorder, {'n_events': 0})
            
            # 3. Aktivität berechnen
            # Decay (Abklingen alter Aktivität)
            self.neuron_activity *= 0.85 
            
            spike_count = len(senders)
            if spike_count > 0:
                # Map GIDs to Indices
                indices = [self.gid_to_idx[gid] for gid in senders if gid in self.gid_to_idx]
                if indices:
                    # Setze Aktivität auf 1.0 (Feuern)
                    self.neuron_activity[indices] = 1.0
            
            # 4. Farben updaten (Vektorisiert)
            # Inaktiv (Blau) -> Aktiv (Weiß/Gelb)
            # R: 0 -> 1
            self.neuron_colors[:, 0] = self.neuron_activity
            # G: 0.2 -> 1.0
            self.neuron_colors[:, 1] = 0.2 + (self.neuron_activity * 0.8)
            # B: 0.8 -> 0.2
            self.neuron_colors[:, 2] = 0.8 - (self.neuron_activity * 0.6)
            # A: 0.4 -> 1.0
            self.neuron_colors[:, 3] = 0.4 + (self.neuron_activity * 0.6)
            
            self.scatter.setData(color=self.neuron_colors)
            
            # 5. Stats anzeigen
            self.lbl_stats.setText(f"T: {self.current_time:.0f}ms | Spikes: {spike_count}")
            
        except Exception as e:
            print(f"Live Sim Error: {e}")
            self.stop_simulation()

    def closeEvent(self, event):
        self.sim_running = False
        self.timer.stop()
        event.accept()
