import sys
from PyQt6.QtWidgets import QApplication,QDialog,QAbstractSpinBox,QDialogButtonBox,QListWidget,QSlider, QMainWindow,QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtWidgets import QSizePolicy,QListWidgetItem, QFrame,QPushButton,QLabel,QGroupBox, QStackedWidget, QToolBar, QMenu, QGridLayout, QStackedLayout,QTreeWidgetItemIterator
from PyQt6.QtGui import QColor, QPalette, QAction,QIcon,QBrush,QMatrix4x4
from PyQt6.QtCore import QSize, Qt, pyqtSignal,QTimer
import code
import os
import shutil
import time
import pyvista as pv
import scipy.ndimage as ndimage
from io import StringIO
from PyQt6 import QtGui
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QSpinBox, QDoubleSpinBox, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt
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
from PyQt6.QtCore import QEvent
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QVector3D
from PyQt6.QtGui import QMatrix4x4, QVector4D
from PyQt6.QtCore import QPoint
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
from PyQt6.QtWidgets import QScrollArea, QInputDialog, QLayout, QWidgetItem
from PyQt6.QtCore import QRect, QPoint

# Multi-Compartment Neuron Models - require explicit receptor_type
MC_MODELS = {'iaf_cond_alpha_mc', 'cm_default', 'cm_main', 'iaf_cond_beta_mc'}

def safe_get_model(population, default='unknown'):
    """Safely get model name from a NEST population."""
    try:
        if population is None or len(population) == 0:
            return default
        return nest.GetStatus(population, 'model')[0]
    except Exception:
        return default

def safe_get_status(population, key=None):
    """Safely get status from a NEST population. Returns None on error."""
    try:
        if population is None or len(population) == 0:
            return None
        if key:
            return nest.GetStatus(population, key)
        return nest.GetStatus(population)
    except Exception:
        return None

def get_receptor_type_for_model(model_name: str, excitatory: bool = True) -> int:
    """
    Returns appropriate receptor_type for multi-compartment models.
    For standard models returns 0 (default).
    
    iaf_cond_alpha_mc receptors:
        1: soma_exc, 2: soma_inh, 3: prox_exc, 4: prox_inh, 5: dist_exc, 6: dist_inh
    """
    if model_name in MC_MODELS:
        return 1 if excitatory else 2  # soma_exc or soma_inh
    return 0

def get_mc_recordables(model_name: str) -> list:
    """
    Returns appropriate recordables for multi-compartment models.
    Standard models use ['V_m'], MC models use compartment-specific variables.
    """
    if model_name in MC_MODELS:
        return ['V_m_s']  # Soma membrane potential
    return ['V_m']


class FlowLayout(QLayout):

    def __init__(self, parent=None, margin=5, spacing=5):
        super().__init__(parent)
        self._margin = margin
        self._spacing = spacing
        self._items = []
        self._cached_height = 0
        
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
    
    def addItem(self, item):
        self._items.append(item)
        self.invalidate()
    
    def addWidget(self, widget):
        item = QWidgetItem(widget)
        self.addItem(item)
    
    def removeWidget(self, widget):
        for i, item in enumerate(self._items):
            if item.widget() == widget:
                self._items.pop(i)
                self.invalidate()
                break
    
    def count(self):
        return len(self._items)
    
    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None
    
    def takeAt(self, index):
        if 0 <= index < len(self._items):
            item = self._items.pop(index)
            self.invalidate()
            return item
        return None
    
    def expandingDirections(self):
        return Qt.Orientation(0)
    
    def hasHeightForWidth(self):
        return True
    
    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)
    
    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)
    
    def sizeHint(self):
        return self.minimumSize()
    
    def minimumSize(self):
        size = QSize(0, 0)
        for item in self._items:
            if item.widget() and item.widget().isVisible():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size
    
    def _do_layout(self, rect, test_only):
        margins = self.contentsMargins()
        effective_rect = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0
        row_start_x = effective_rect.x()
        
        for item in self._items:
            widget = item.widget()
            if widget is None or not widget.isVisible():
                continue
            
            item_size = item.sizeHint()
            if item_size.isEmpty():
                item_size = widget.sizeHint()
            if item_size.isEmpty():
                item_size = QSize(60, 70) 
            if x + item_size.width() > effective_rect.right() + 1 and x > row_start_x:
                x = row_start_x
                y = y + line_height + self._spacing
                line_height = 0
            
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item_size))
            
            x = x + item_size.width() + self._spacing
            line_height = max(line_height, item_size.height())
        
        final_height = y + line_height - rect.y() + margins.bottom()
        self._cached_height = max(final_height, 10)
        
        if not test_only and self.parentWidget():
            parent = self.parentWidget()
            parent.setMinimumHeight(self._cached_height)
        
        return self._cached_height


current_graph_metadata = []
graph_parameters = {}
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

# Tool-spezifische Key-Mappings für geteilte Parameter
# Mappt generische Namen auf tool-spezifische Widget-Keys
TOOL_PARAM_KEYS = {
    'CCW': {
        'n_neurons': 'ccw_n_neurons',
        'radius': 'ccw_radius',
        'tool_neuron_model': 'ccw_tool_neuron_model'
    },
    'Blob': {
        'n_neurons': 'blob_n_neurons',
        'radius': 'blob_radius',
        'tool_neuron_model': 'blob_tool_neuron_model'
    },
    'Cone': {
        'n_neurons': 'cone_n_neurons',
        'tool_neuron_model': 'cone_tool_neuron_model'
    },
    'Grid': {
        'tool_neuron_model': 'grid_tool_neuron_model'
    },
    'custom': {}
}

__all__ = [
    'SYNAPSE_MODELS',
    "DeviceConfigPage",
    "AnalysisDashboard",
    "LiveConnectionController",
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
        r_outer = float(params.get('mask_radius', params.get('radius', params.get('outer_radius', 1.0))))
        r_inner = float(params.get('mask_inner_radius', params.get('inner_radius', params.get('inner', 0.0))))
        
        if mask_type in ('spherical', 'sphere'):
            return nest.CreateMask('spherical', {'radius': r_outer})
            
        elif mask_type in ('rectangular', 'box'):
            s = r_outer 
            return nest.CreateMask('rectangular', {
                'lower_left': [-s, -s, -s],
                'upper_right': [s, s, s]
            })
            
        elif mask_type == 'doughnut':
            m_outer = nest.CreateMask('spherical', {'radius': r_outer})
            
            if r_inner > 0:
                m_inner = nest.CreateMask('spherical', {'radius': r_inner})
                return m_outer & ~m_inner
            else:
                return m_outer
            
        elif mask_type == 'elliptical':
             major = float(params.get('major_axis', 1.0))
             minor = float(params.get('minor_axis', 0.5))
             return nest.CreateMask('elliptical', {'major_axis': major, 'minor_axis': minor})

    except Exception as e:
        print(f"Mask Creation Error: {e}")
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
    def __init__(self, param_name, default_value=0.0, min_val=-1000000.0, max_val=1000000.0, decimals=2):
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
    def __init__(self, param_name, default_value=0, min_val=-1000000, max_val=1000000):
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

        except FileNotFoundError:
            print("Warning: functional_models.json not found")
            self.all_models = {}

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        model_layout = QHBoxLayout()
        model_label = QLabel("Neuron Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.all_models.keys()))
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_container.setLayout(self.params_layout)
        self.params_scroll.setWidget(self.params_container)
        layout.addWidget(self.params_scroll)

        self.save_button = QPushButton("Save Parameters")
        self.save_button.clicked.connect(self.save_parameters)
        layout.addWidget(self.save_button)

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
        
        self.add_section("Auto-Instrumentation")
        self.add_bool_field("auto_spike_recorder", "Auto-Record Spikes (Raster)", default=False)
        self.add_bool_field("auto_multimeter", "Auto-Record Voltage (V_m)", default=False)

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
        self.tool_stack.addWidget(self._create_panel_custom()); self.tool_panels['custom'] = 0
        self.tool_stack.addWidget(self._create_panel_ccw()); self.tool_panels['CCW'] = 1
        self.tool_stack.addWidget(self._create_panel_blob()); self.tool_panels['Blob'] = 2
        self.tool_stack.addWidget(self._create_panel_cone()); self.tool_panels['Cone'] = 3
        self.tool_stack.addWidget(self._create_panel_grid()); self.tool_panels['Grid'] = 4

    def _create_panel_custom(self):
        panel = QWidget(); layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Wave Function Collapse", styleSheet="color:#4CAF50; font-weight:bold;"))
        self._add_field_to_layout(layout, "old", "Use WFC (Multitype)", field_type="bool", default=False)
        self._add_field_to_layout(layout, "grid_size", "Grid Size", field_type="vector3_int")
        self._add_field_to_layout(layout, "sparsity_factor", "Sparsity", field_type="float", default=0.9)
        self._add_field_to_layout(layout, "num_steps", "Flow Steps", field_type="int", default=8)
        self._add_field_to_layout(layout, "dt", "dt", field_type="float", default=0.01)
        layout.addStretch(); return panel

    def _create_panel_ccw(self):
        panel = QWidget(); layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Ring Attractor", styleSheet="color:#FF9800; font-weight:bold;"))
        self._add_model_selector(layout, key="ccw_tool_neuron_model")
        self._add_neuron_edit_button(layout)
        self._add_field_to_layout(layout, "ccw_n_neurons", "Count", field_type="int", default=100)
        self._add_field_to_layout(layout, "ccw_radius", "Radius", field_type="float", default=5.0)
        self._add_field_to_layout(layout, "bidirectional", "Bidirectional", field_type="bool", default=False)
        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        self._add_field_to_layout(layout, "ccw_weight_ex", "Weight", field_type="float", default=30.0)
        self._add_field_to_layout(layout, "k", "Inhibition (k)", field_type="float", default=10.0)
        layout.addStretch(); return panel

    def _create_panel_blob(self):
        panel = QWidget(); layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Random Blob", styleSheet="color:#9C27B0; font-weight:bold;"))
        self._add_model_selector(layout, key="blob_tool_neuron_model"); self._add_neuron_edit_button(layout)
        self._add_field_to_layout(layout, "blob_n_neurons", "Count", field_type="int", default=100)
        self._add_field_to_layout(layout, "blob_radius", "Radius", field_type="float", default=5.0)
        layout.addStretch(); return panel

    def _create_panel_cone(self):
        panel = QWidget(); layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Cone / Column", styleSheet="color:#E91E63; font-weight:bold;"))
        self._add_model_selector(layout, key="cone_tool_neuron_model"); self._add_neuron_edit_button(layout)
        self._add_field_to_layout(layout, "cone_n_neurons", "Count", field_type="int", default=500)
        self._add_field_to_layout(layout, "radius_bottom", "R Bottom", field_type="float", default=5.0)
        self._add_field_to_layout(layout, "radius_top", "R Top", field_type="float", default=1.0)
        self._add_field_to_layout(layout, "height", "Height", field_type="float", default=10.0)
        layout.addStretch(); return panel

    def _create_panel_grid(self):
        panel = QWidget(); layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("2D Grid", styleSheet="color:#00BCD4; font-weight:bold;"))
        self._add_model_selector(layout, key="grid_tool_neuron_model")
        self._add_field_to_layout(layout, "grid_side_length", "Side Length", field_type="int", default=10)
        layout.addStretch(); return panel

    def _add_field_to_layout(self, layout, key, label, field_type="float", min_val=None, max_val=None, default=None):
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{label}:", styleSheet="font-weight:bold; min-width:100px;"))
        if field_type == "int":
            w = QSpinBox(); w.setRange(min_val or 0, max_val or 100000); w.setValue(default or 0)
        elif field_type == "float":
            w = QDoubleSpinBox(); w.setRange(min_val or -1000, max_val or 1000); w.setValue(default or 0.0)
        elif field_type == "bool":
            w = QCheckBox(); w.setChecked(default or False)
        elif field_type == "vector3_int":
            w = None; ws = []
            for p in ["X","Y","Z"]: s=QSpinBox(); s.setRange(1,1000); s.setPrefix(p); s.setValue(default or 10); s.valueChanged.connect(self.on_change); row.addWidget(s); ws.append(s)
            self.widgets[key] = {'type': 'vector3_int', 'widgets': ws}; layout.addLayout(row); return
            
        if w:
            if hasattr(w, 'valueChanged'): w.valueChanged.connect(self.on_change)
            if hasattr(w, 'stateChanged'): w.stateChanged.connect(self.on_change)
            row.addWidget(w)
            if field_type=="bool": row.addStretch()
            self.widgets[key] = {'type': field_type, 'widget': w}
        layout.addLayout(row)

    def on_tool_changed(self):
        self.tool_stack.setCurrentIndex(self.tool_panels.get(self.tool_combo.currentData(), 0))
        self.on_change()

    def add_section(self, title):
        self.content_layout.addWidget(QLabel(title, styleSheet="font-weight:bold; color:#2196F3; margin-top:10px; border-bottom:1px solid #2196F3;"))

    def add_text_field(self, key, label, parent=None):
        l = parent or self.content_layout; r=QHBoxLayout(); w=QLineEdit(); w.textChanged.connect(self.on_change)
        r.addWidget(QLabel(f"{label}:", styleSheet="font-weight:bold;")); r.addWidget(w); l.addLayout(r)
        self.widgets[key] = {'type':'text', 'widget':w}

    def add_int_field(self, key, label, min_val=0, parent=None):
        l = parent or self.content_layout; r=QHBoxLayout(); w=QSpinBox(); w.setRange(min_val, 10000); w.valueChanged.connect(self.on_change)
        r.addWidget(QLabel(f"{label}:", styleSheet="font-weight:bold;")); r.addWidget(w); l.addLayout(r)
        self.widgets[key] = {'type':'int', 'widget':w}

    def add_float_field(self, key, label, min_val=-1000, max_val=1000, parent=None):
        l = parent or self.content_layout; r=QHBoxLayout(); w=QDoubleSpinBox(); w.setRange(min_val, max_val); w.valueChanged.connect(self.on_change)
        r.addWidget(QLabel(f"{label}:", styleSheet="font-weight:bold;")); r.addWidget(w); l.addLayout(r)
        self.widgets[key] = {'type':'float', 'widget':w}

    def add_bool_field(self, key, label, default=False, parent=None):
        l = parent or self.content_layout; r=QHBoxLayout(); w=QCheckBox(); w.setChecked(default); w.stateChanged.connect(self.on_change)
        r.addWidget(QLabel(f"{label}:", styleSheet="font-weight:bold;")); r.addWidget(w); r.addStretch(); l.addLayout(r)
        self.widgets[key] = {'type':'bool', 'widget':w}

    def add_vector3_field(self, key, label, parent=None):
        l = parent or self.content_layout; r=QHBoxLayout(); ws=[]
        r.addWidget(QLabel(f"{label}:", styleSheet="font-weight:bold;"))
        for p in ["X","Y","Z"]: s=QDoubleSpinBox(); s.setRange(-1000,1000); s.setPrefix(p); s.valueChanged.connect(self.on_change); r.addWidget(s); ws.append(s)
        l.addLayout(r); self.widgets[key] = {'type':'vector3', 'widgets':ws}

    def on_change(self):
        if self.auto_save: self.paramsChanged.emit(self.get_current_params())

    def set_population_count(self, count):
        self.num_populations = count
        while self.probability_layout.count(): item=self.probability_layout.takeAt(0); item.widget().deleteLater() if item.widget() else None
        if 'probability_vector' in self.widgets: del self.widgets['probability_vector']
        
        if count == 0: self.probability_layout.addWidget(QLabel("No populations"))
        else:
            ws = []
            for i in range(count):
                r=QHBoxLayout(); s=QDoubleSpinBox(); s.setRange(0,1); s.setValue(1.0/count); s.valueChanged.connect(self.on_change)
                r.addWidget(QLabel(f"Pop {i+1}:")); r.addWidget(s); self.probability_layout.addLayout(r); ws.append(s)
            self.widgets['probability_vector'] = {'type':'prob_list', 'widgets':ws}

    def get_current_params(self):
        res = {}
        
        # Hole aktuelles Tool
        current_tool = self.tool_combo.currentData() or 'custom'
        tool_mapping = TOOL_PARAM_KEYS.get(current_tool, {})
        # Inverses Mapping: tool-spezifischer Key -> generischer Key
        inverse_mapping = {v: k for k, v in tool_mapping.items()}
        
        for k, info in self.widgets.items():
            t = info['type']; w = info.get('widget')
            
            # Bestimme den Output-Key (generischer Name falls gemappt)
            output_key = inverse_mapping.get(k, k)
            
            if t=='text': res[output_key]=w.text()
            elif t=='int' or t=='float': res[output_key]=w.value()
            elif t=='bool': res[output_key]=w.isChecked()
            elif t=='combo': res[output_key]=w.currentData()
            elif t=='combo_text': res[output_key]=w.currentText()
            elif t in ['vector3', 'vector3_int', 'prob_list']: res[output_key]=[s.value() for s in info['widgets']]
        
        if 'center_of_mass' in res: res['m'] = res['center_of_mass']
        sx=res.get('stretch_x',1); sy=res.get('stretch_y',1); sz=res.get('stretch_z',1)
        res['transform_matrix'] = [[sx,0,0],[0,sy,0],[0,0,sz]]
        return res

    def load_data(self, data):
        self.auto_save = False
        self.node_data = data
        
        # Setze zuerst das Tool, damit wir das richtige Mapping haben
        if 'tool_type' in data: 
            self.tool_stack.setCurrentIndex(self.tool_panels.get(data['tool_type'], 0))
            # Update auch die ComboBox
            idx = self.tool_combo.findData(data['tool_type'])
            if idx >= 0:
                self.tool_combo.setCurrentIndex(idx)
        
        # Hole Mapping für das aktuelle Tool
        current_tool = data.get('tool_type', 'custom')
        tool_mapping = TOOL_PARAM_KEYS.get(current_tool, {})
        
        # Falls tool_neuron_model nicht gesetzt ist, aber neuron_models existiert, übertrage den Wert
        if 'tool_neuron_model' not in data and 'neuron_models' in data:
            nm = data['neuron_models']
            if isinstance(nm, list) and nm:
                data['tool_neuron_model'] = nm[0]
        
        for k, info in self.widgets.items():
            # Prüfe ob dieser Widget-Key zu einem generischen Key gemappt werden soll
            # Falls ja, hole den Wert vom generischen Key
            inverse_mapping = {v: k_gen for k_gen, v in tool_mapping.items()}
            data_key = inverse_mapping.get(k, k)  # Generischer Key oder originaler Key
            
            if data_key not in data: continue
            v = data[data_key]; t = info['type']; w = info.get('widget')
            if t=='text': w.setText(str(v))
            elif t in ['int', 'float']: w.setValue(v)
            elif t=='bool': w.setChecked(bool(v))
            elif t=='combo': idx=w.findData(v); w.setCurrentIndex(idx) if idx>=0 else None
            elif t=='combo_text': w.setCurrentText(str(v))
            elif t in ['vector3', 'vector3_int', 'prob_list']:
                for i, s in enumerate(info['widgets']):
                    if i < len(v): s.setValue(v[i])
        self.auto_save = True

    def _add_neuron_edit_button(self, layout):
        btn = QPushButton("⚙️ Edit Neurons"); btn.clicked.connect(self._jump_to_neuron_editor)
        layout.addWidget(btn)
    def _jump_to_neuron_editor(self):
        # Traversiere die Parent-Hierarchie um das QTabWidget zu finden
        widget = self
        while widget is not None:
            if hasattr(widget, 'setCurrentIndex'):
                widget.setCurrentIndex(2)
                return
            if hasattr(widget, 'parent') and callable(widget.parent):
                widget = widget.parent()
            else:
                break
        print("⚠ Could not find tab widget to switch to neuron editor")


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
    itemVisibilityChanged = pyqtSignal(dict, bool)
    requestDeviceDeletion = pyqtSignal(dict)
    requestLiveWeightChange = pyqtSignal(dict, float)
    COLOR_GRAPH_BG = "#000000"; COLOR_GRAPH_FG = "#87CEEB"; COLOR_NODE_BG = "#8B0000"
    COLOR_NODE_FG = "#FFFF00"; COLOR_POP_BG = "#424242"; COLOR_POP_FG = "#FFFFFF"
    COLOR_CONN_BG = "#aaaa00"; COLOR_CONN_FG = "#841414"; COLOR_DEVICE_BG = "#4A148C"; COLOR_DEVICE_FG = "#E0E0E0"
    
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
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #FFFF00;")
        header.addWidget(title)
        header.addStretch()
        
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setFixedSize(28, 28)
        self.refresh_btn.clicked.connect(self.update_tree)
        header.addWidget(self.refresh_btn)
        
        self.expand_btn = QPushButton("⊕"); self.expand_btn.setFixedSize(28, 28); self.expand_btn.clicked.connect(self._expand_all)
        header.addWidget(self.expand_btn)
        self.collapse_btn = QPushButton("⊖"); self.collapse_btn.setFixedSize(28, 28); self.collapse_btn.clicked.connect(self._collapse_all)
        header.addWidget(self.collapse_btn)
        layout.addLayout(header)
        
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Element", "Details"])
        self.tree.setColumnWidth(0, 240)
        self.tree.setStyleSheet("background-color: #1e1e1e; color: #e0e0e0; border: 1px solid #444;")
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        
        layout.addWidget(self.tree)
        
        self.status_label = QLabel("No graphs")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #aaaaaa; font-size: 11px; padding: 5px;")
        layout.addWidget(self.status_label)

        self.tree.setItemsExpandable(True)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Element (Check to Hide/Show)", "Details"])
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.itemChanged.connect(self._on_item_changed)
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
            
            action_del = QAction(f"Delete (Rebuild)", self)
            action_del.triggered.connect(lambda: self.requestConnectionDeletion.emit(conn_data))
            menu.addAction(action_del)
            
            action_sever = QAction(f"Sever Connection (Live, w=0)", self)
            action_sever.triggered.connect(lambda: self.requestLiveWeightChange.emit(conn_data, 0.0))
            menu.addAction(action_sever)
            
            orig_weight = conn_data.get('params', {}).get('weight', 1.0)
            action_restore = QAction(f"Restore Weight (Live, w={orig_weight})", self)
            action_restore.triggered.connect(lambda: self.requestLiveWeightChange.emit(conn_data, float(orig_weight)))
            menu.addAction(action_restore)
            
            conn_data = data.get('connection')
            conn_name = conn_data.get('name', 'Connection')
            
            action_del = QAction(f"Delete (Rebuild)", self)
            action_del.triggered.connect(lambda: self.requestConnectionDeletion.emit(conn_data))
            menu.addAction(action_del)
            
            action_sever = QAction(f"Sever Connection (Live, w=0)", self)
            action_sever.triggered.connect(lambda: self.requestLiveWeightChange.emit(conn_data, 0.0))
            menu.addAction(action_sever)
            
            orig_weight = conn_data.get('params', {}).get('weight', 1.0)
            action_restore = QAction(f"Restore Weight (Live, w={orig_weight})", self)
            action_restore.triggered.connect(lambda: self.requestLiveWeightChange.emit(conn_data, float(orig_weight)))
            menu.addAction(action_restore)
            
        elif item_type == 'device':
            dev_data = data.get('device')
            model = dev_data.get('model', 'Device')
            dev_id = dev_data.get('id', '?')
            
            action_info = QAction(f"ℹ Device: {model}", self)
            action_info.setEnabled(False)
            menu.addAction(action_info)
            
            action_del = QAction(f"🗑️ Delete Device #{dev_id}", self)
            action_del.triggered.connect(lambda: self.requestDeviceDeletion.emit(dev_data))
            menu.addAction(action_del)

        menu.exec(self.tree.viewport().mapToGlobal(position))


    def update_tree(self):
        self.tree.clear()
        
        if not self.graph_list:
            self.status_label.setText("No graphs loaded")
            self.status_label.setVisible(True)
            return
        
        self.status_label.setVisible(False) 
        
        total_nodes = 0
        total_pops = 0
        total_conns = 0
        total_devs = 0
        
        for graph in self.graph_list:
            graph_item = QTreeWidgetItem(self.tree)
            g_name = getattr(graph, 'graph_name', f'Graph_{graph.graph_id}')
            graph_item.setText(0, g_name)
            graph_item.setText(1, f"ID: {graph.graph_id}")
            graph_item.setBackground(0, QBrush(QColor(self.COLOR_GRAPH_BG)))
            graph_item.setForeground(0, QBrush(QColor(self.COLOR_GRAPH_FG)))
            
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
                        self._create_device_item(dev, pop_item, graph.graph_id, node.id, pop_idx)

        self.status_label.setText(f"{len(self.graph_list)} Graphs | {total_nodes} Nodes | {total_pops} Pops | {total_conns} Conns")
        self.status_label.setVisible(True)
        
    
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
    
    def _create_node_item(self, graph, node, parent):
        item = QTreeWidgetItem(parent)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, Qt.CheckState.Checked)
        
        name = getattr(node, 'name', f"Node_{node.id}")
        item.setText(0, f"Node {node.id}: {name}")
        item.setBackground(0, QBrush(QColor(self.COLOR_NODE_BG)))
        item.setForeground(0, QBrush(QColor(self.COLOR_NODE_FG)))
        
        item.setData(0, Qt.ItemDataRole.UserRole, {'type': 'node', 'graph_id': graph.graph_id, 'node_id': node.id})
        item.setExpanded(True)
        return item
    
    def _on_item_changed(self, item, column):
        if column == 0:
            data = item.data(0, Qt.ItemDataRole.UserRole)
            
            if data is None:
                return
            
            visible = (item.checkState(0) == Qt.CheckState.Checked)
            self.itemVisibilityChanged.emit(data, visible)

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
        pops = []
        if hasattr(node, 'types'):
            for i, t in enumerate(node.types):
                model = node.neuron_models[i] if hasattr(node, 'neuron_models') and i < len(node.neuron_models) else 'iaf_psc_alpha'
                count = 0
                if hasattr(node, 'population') and i < len(node.population) and node.population[i]:
                    count = len(node.population[i])
                pops.append({'model': model, 'n_neurons': count})
        elif hasattr(node, 'population'):
             for i, p in enumerate(node.population):
                 pops.append({'model': 'unknown', 'n_neurons': len(p) if p else 0})
        return pops
    
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
        self.noise_slider.setRange(0, 200)
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
        
        twin_btn = QPushButton("Twin")
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
        
        node_idx = len(self.node_list)
        node_btn = QPushButton(f"Node {node_idx + 1}")
        node_btn.setMinimumHeight(50)
        node_btn.clicked.connect(lambda checked=False, idx=node_idx: self.select_node(idx))
        self.node_list_layout.addWidget(node_btn)
        
        import copy
        node_params = copy.deepcopy(node_parameters1)
        node_params['id'] = node_idx
        node_params['m'] = [0.0, 0.0, 0.0]
        node_params['center_of_mass'] = [0.0, 0.0, 0.0]
        node_params['displacement'] = [0.0, 0.0, 0.0]
        
        safe_name = name.replace(" ", "_").replace("/", "-")
        node_params['name'] = f"{safe_name}_{node_idx}"
        node_params['grid_size'] = [10, 10, 10]
        node_params['probability_vector'] = probs
        node_params['sparsity_factor'] = 0.85
        
        node_btn.setText(f"Node {node_idx + 1}: {node_params['name']}")
        
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
            
        self.node_list.append({
            'params': node_params,
            'populations': populations,
            'button': node_btn,
            'original_node': None
        })
        
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
        
        import copy
        new_node_data = copy.deepcopy(data_to_copy)
        
        current_pos = source_node['params'].get('center_of_mass', [0,0,0])
        old_id = source_node['params'].get('id')
        
        dlg = PositionDialog(current_pos, self)
        if dlg.exec():
            new_pos = dlg.get_position()
            
            import copy
            import time
            
            
            new_node_data['original_node'] = None
            
            new_idx = len(self.node_list)
            new_node_data['params']['id'] = new_idx
            new_node_data['params']['name'] = f"{source_node['params'].get('name', 'Node')}_Twin"
            
            new_node_data['params']['m'] = list(new_pos)
            new_node_data['params']['center_of_mass'] = list(new_pos)
            new_node_data['params']['old_center_of_mass'] = list(new_pos)
            
            source_conns = source_node['params'].get('connections', [])
            twin_conns = []
            
            for i, conn in enumerate(source_conns):
                src_id = conn['source'].get('node_id')
                tgt_id = conn['target'].get('node_id')
                
                if src_id == old_id and tgt_id == old_id:
                    new_conn = copy.deepcopy(conn)
                    
                    new_conn['source']['node_id'] = new_idx
                    new_conn['target']['node_id'] = new_idx
                    
                    new_conn['id'] = int(time.time() * 10000) + i
                    new_conn['name'] = f"{conn.get('name', 'conn')}_Twin"
                    
                    twin_conns.append(new_conn)
            
            new_node_data['params']['connections'] = twin_conns
            
            node_btn = QPushButton(f"Node {new_idx + 1}: {new_node_data['params']['name']}")
            node_btn.setMinimumHeight(50)
            node_btn.clicked.connect(lambda checked=False, idx=new_idx: self.select_node(idx))
            self.node_list_layout.addWidget(node_btn)
            
            new_node_data['button'] = node_btn
            
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
            elif tool_type in ('Blob', 'Cone', 'CCW', 'Grid'):
                # Bei Tool-Typen: tool_neuron_model hat Priorität über existierende Populations
                selected_model = node['params'].get('tool_neuron_model', None)
                model_changed = False
                
                if selected_model:
                    # Prüfe ob sich das Modell geändert hat
                    if populations:
                        old_models = [pop.get('model') for pop in populations]
                        model_changed = any(m != selected_model for m in old_models)
                    neuron_models = [selected_model] * max(1, len(populations))
                else:
                    neuron_models = [pop['model'] for pop in populations] if populations else ['iaf_psc_alpha']
                
                types = list(range(len(populations))) if populations else [0]
                
                encoded_polynoms_per_type = []
                for pop in populations:
                    poly_dict = pop.get('polynomials', None)
                    if poly_dict and all(k in poly_dict for k in ['x', 'y', 'z']):
                        encoded_polynoms_per_type.append([poly_dict['x'], poly_dict['y'], poly_dict['z']])
                    else:
                        encoded_polynoms_per_type.append([])
                if not encoded_polynoms_per_type:
                    encoded_polynoms_per_type = [[]]
                
                num_pops = len(neuron_models)
                prob_vec = node['params'].get('probability_vector', [])
                if not prob_vec or len(prob_vec) != num_pops:
                    prob_vec = [1.0 / num_pops] * num_pops if num_pops > 0 else [1.0]
                
                # Bei Modelländerung alte Parameter verwerfen
                if model_changed:
                    pop_nest_params = [{}] * len(neuron_models)
                else:
                    pop_nest_params = [pop.get('params', {}) for pop in populations] if populations else [{}]
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
                'auto_spike_recorder': node['params'].get('auto_spike_recorder', False),
                'auto_multimeter': node['params'].get('auto_multimeter', False),
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
        
        if len(node['populations']) <= 1:
            QMessageBox.warning(self, "Cannot Delete", "Cannot delete the last population of a node.")
            return
        
        prob_vec = node['params'].get('probability_vector', [])
        deleted_prob = 0.0
        if len(prob_vec) > self.current_pop_idx:
            deleted_prob = prob_vec.pop(self.current_pop_idx)
        
        if prob_vec and deleted_prob > 0:
            add_each = deleted_prob / len(prob_vec)
            node['params']['probability_vector'] = [p + add_each for p in prob_vec]
        
        node['populations'].pop(self.current_pop_idx)
        
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
            
            try: node['button'].clicked.disconnect()  # May not be connected
            except RuntimeError: pass  # No connections
            node['button'].clicked.connect(lambda checked=False, idx=i: self.select_node(idx))

        if self.node_list:
            new_idx = max(0, self.current_node_idx - 1)
            self.select_node(new_idx)
        else:
            self.current_node_idx = None
            self.editor_stack.setCurrentIndex(0)
            self.remove_node_btn.setEnabled(False)
            
            while self.pop_list_layout.count():
                item = self.pop_list_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
    def select_node(self, node_idx):
        if node_idx < 0 or node_idx >= len(self.node_list): return
        
        self.save_current_population_params()
        self.current_node_idx = node_idx
        self.current_pop_idx = None
        
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
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        
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
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        
        while self.pop_list_layout.count():
            item = self.pop_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        
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
        
        name_disp_layout = QHBoxLayout()
        
        name_disp_layout.addWidget(QLabel("Graph Name:"))
        self.graph_name_input = QLineEdit()
        self.graph_name_input.setPlaceholderText("Edit graph name...")
        name_disp_layout.addWidget(self.graph_name_input, stretch=2)
        
        line = QFrame(); line.setFrameShape(QFrame.Shape.VLine); line.setStyleSheet("color: #555;")
        name_disp_layout.addWidget(line)
        
        name_disp_layout.addWidget(QLabel("Global Offset:"))
        
        self.spin_g_x = QDoubleSpinBox(); self.spin_g_x.setRange(-1e5, 1e5); self.spin_g_x.setPrefix("X: "); self.spin_g_x.setDecimals(1)
        self.spin_g_y = QDoubleSpinBox(); self.spin_g_y.setRange(-1e5, 1e5); self.spin_g_y.setPrefix("Y: "); self.spin_g_y.setDecimals(1)
        self.spin_g_z = QDoubleSpinBox(); self.spin_g_z.setRange(-1e5, 1e5); self.spin_g_z.setPrefix("Z: "); self.spin_g_z.setDecimals(1)
        
        style_disp = "background-color: #222; color: #00E5FF; font-weight: bold; border: 1px solid #444;"
        for s in [self.spin_g_x, self.spin_g_y, self.spin_g_z]:
            s.setStyleSheet(style_disp)
            s.setFixedWidth(80)
            name_disp_layout.addWidget(s)
            
        main_layout.addLayout(name_disp_layout)
        
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
        
        twin_btn = QPushButton("Twin")
        twin_btn.setToolTip("Clone selected node as a NEW node at new position")
        twin_btn.clicked.connect(self.create_twin_node)
        twin_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        node_buttons_layout.addWidget(twin_btn)

        self.remove_node_btn = QPushButton("Remove")
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
        
        self.remove_pop_btn = QPushButton("DEL")
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
        
        delete_btn = QPushButton("Delete Graph")
        delete_btn.setMinimumHeight(50)
        delete_btn.clicked.connect(self.delete_graph)
        delete_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        bottom_layout.addWidget(delete_btn)
        
        save_btn = QPushButton("SAVE CHANGES & REBUILD")
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

        node_data_wrapper = self.node_list[target_node_idx]
        connections = node_data_wrapper['params'].get('connections', [])
        new_conns = [c for c in connections if c.get('id') != conn_id]
        node_data_wrapper['params']['connections'] = new_conns
        
        if node_data_wrapper.get('original_node'):
            src_node_obj = node_data_wrapper['original_node']
            if hasattr(src_node_obj, 'connections'):
                src_node_obj.connections = [c for c in src_node_obj.connections if c.get('id') != conn_id]
            
            tgt_gid = conn_data['target']['graph_id']
            tgt_nid = conn_data['target']['node_id']
            tgt_node_obj = None
            target_graph = next((g for g in self.graph_list if g.graph_id == tgt_gid), None)
            
            if target_graph:
                tgt_node_obj = target_graph.get_node(tgt_nid)
            
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
        
        if 'button' in node_data:
            self.node_list_layout.removeWidget(node_data['button'])
            node_data['button'].deleteLater()
            
        if self.current_graph and node_data.get('original_node'):
            original_node = node_data['original_node']
            self.current_graph.remove_node(original_node)
            
        for i, node in enumerate(self.node_list):
            node['params']['id'] = i
            old_name = node['params'].get('name', '')
            if old_name.startswith("Node_"):
                node['params']['name'] = f"Node_{i}"
            node['button'].setText(f"Node {i + 1}: {node['params']['name']}")
            try: node['button'].clicked.disconnect()  # May not be connected
            except RuntimeError: pass  # No connections
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
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()

    def remove_population(self):
        if self.current_node_idx is None or self.current_pop_idx is None:
            return
            
        node_wrapper = self.node_list[self.current_node_idx]
        
        if len(node_wrapper['populations']) <= 1:
            QMessageBox.warning(self, "Cannot Delete", "Cannot delete the last population of a node.")
            return
        
        prob_vec = node_wrapper['params'].get('probability_vector', [])
        deleted_prob = 0.0
        if len(prob_vec) > self.current_pop_idx:
            deleted_prob = prob_vec.pop(self.current_pop_idx)
        
        if prob_vec and deleted_prob > 0:
            add_each = deleted_prob / len(prob_vec)
            node_wrapper['params']['probability_vector'] = [p + add_each for p in prob_vec]
        
        node_wrapper['populations'].pop(self.current_pop_idx)
        
        if node_wrapper.get('original_node'):
            node_obj = node_wrapper['original_node']
            new_conns = []
            targets_to_check = set()
            if hasattr(node_obj, 'connections'):
                for conn in node_obj.connections:
                    if conn['source']['pop_id'] == self.current_pop_idx:
                        tgt_gid = conn['target']['graph_id']
                        tgt_nid = conn['target']['node_id']
                        targets_to_check.add((tgt_gid, tgt_nid))
                        continue
                    if conn['source']['pop_id'] > self.current_pop_idx:
                        conn['source']['pop_id'] -= 1
                    if (conn['target']['graph_id'] == node_obj.graph_id and
                        conn['target']['node_id'] == node_obj.id and
                        conn['target']['pop_id'] > self.current_pop_idx):
                        conn['target']['pop_id'] -= 1
                    new_conns.append(conn)
                
                node_obj.connections = new_conns
                for (gid, nid) in targets_to_check:
                    tgt_graph = next((g for g in self.graph_list if g.graph_id == gid), None)
                    if tgt_graph:
                        tgt_node = tgt_graph.get_node(nid)
                        if tgt_node:
                            node_obj.remove_neighbor_if_isolated(tgt_node)
        
        self.update_population_list()
        
        num_pops = len(node_wrapper['populations'])
        self.node_param_widget.set_population_count(num_pops)
        self.node_param_widget.load_data(node_wrapper['params'])
        
        if num_pops > 0:
            new_idx = max(0, self.current_pop_idx - 1)
            self.select_population(new_idx)
        else:
            self.current_pop_idx = None
            self.remove_pop_btn.setEnabled(False)
            self.editor_stack.setCurrentIndex(1)

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
        if not self.graph_list or index < 0: return
        
        graph_id = self.graph_selector.currentData()
        if graph_id is None: return
        
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
        if not self.current_graph: return
        
        print(f"\nLoading Graph {self.current_graph_id}")
        
        self.current_node_idx = None
        self.current_pop_idx = None
        
        stored_offset = getattr(self.current_graph, 'global_offset', [0.0, 0.0, 0.0])
        self.spin_g_x.setValue(stored_offset[0])
        self.spin_g_y.setValue(stored_offset[1])
        self.spin_g_z.setValue(stored_offset[2])
        
        self.node_list.clear()
        
        while self.node_list_layout.count():
            item = self.node_list_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        
        while self.pop_list_layout.count():
            item = self.pop_list_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        
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
                if nest_pop is None or len(nest_pop) == 0: continue
                
                model = safe_get_model(nest_pop, 'unknown')
                params = {}
                try:
                    status = safe_get_status(nest_pop)
                    if status:
                        first_neuron = status[0]
                        param_keys = ['V_m', 'E_L', 'tau_m', 'C_m', 'V_th', 't_ref', 'V_reset', 'I_e', 'tau_syn_ex', 'tau_syn_in']
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
                            polynomials = {'x': poly_list[0], 'y': poly_list[1], 'z': poly_list[2]}
                
                populations.append({'model': model, 'params': params, 'polynomials': polynomials})
        
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
        if node_idx < 0 or node_idx >= len(self.node_list): return
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
                while len(current_probs) < num_pops: current_probs.append(0.0)
                if len(current_probs) > num_pops: current_probs = current_probs[:num_pops]
                params['probability_vector'] = current_probs
                
                self.node_param_widget.auto_save = False
                self.node_param_widget.load_data(params)
                self.node_param_widget.auto_save = True
            self.node_list[self.current_node_idx]['params'] = params
    
    def add_population(self):
        if self.current_node_idx is None: return
        self.save_current_population_params()
        node = self.node_list[self.current_node_idx]
        pop_idx = len(node['populations'])
        
        default_polynomials = {
            'x': generate_biased_polynomial(axis_idx=0, max_degree=2),
            'y': generate_biased_polynomial(axis_idx=1, max_degree=2),
            'z': generate_biased_polynomial(axis_idx=2, max_degree=2)
        }
        node['populations'].append({'model': 'iaf_psc_alpha', 'params': {}, 'polynomials': default_polynomials})
        
        num_pops = len(node['populations'])
        self.node_param_widget.set_population_count(num_pops)
        node['params']['probability_vector'] = [1.0/num_pops] * num_pops
        self.node_param_widget.load_data(node['params'])
        
        self.update_population_list()
        self.select_population(pop_idx)
    
    def update_population_list(self):
        while self.pop_list_layout.count():
            item = self.pop_list_layout.takeAt(0)
            if item.widget(): item.widget().setParent(None)
        if self.current_node_idx is None: return
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
        self.remove_pop_btn.setEnabled(True)

    def create_twin_node(self):
        import copy
        if self.current_node_idx is None: return
        self.save_node_params(self.node_param_widget.get_current_params())
        self.save_current_population_params()
        
        source_node = self.node_list[self.current_node_idx]
        clean_populations = []
        if 'populations' in source_node:
            for pop in source_node['populations']:
                clean_pop = {'model': pop.get('model'), 'params': pop.get('params', {}), 'polynomials': pop.get('polynomials', {})}
                clean_populations.append(clean_pop)
        
        data_to_copy = {'params': source_node['params'], 'populations': clean_populations}
        new_node_data = copy.deepcopy(data_to_copy)
        
        current_pos = source_node['params'].get('center_of_mass', [0,0,0])
        old_id = source_node['params'].get('id')
        
        dlg = PositionDialog(current_pos, self)
        if dlg.exec():
            new_pos = dlg.get_position()
            if 'button' in new_node_data: del new_node_data['button']
            new_node_data['original_node'] = None
            new_idx = len(self.node_list)
            old_name = source_node['params'].get('name', 'Node')
            new_name = f"{old_name}_Twin"
            
            new_node_data['params']['id'] = new_idx
            new_node_data['params']['name'] = new_name
            new_node_data['params']['m'] = list(new_pos)
            new_node_data['params']['center_of_mass'] = list(new_pos)
            new_node_data['params']['old_center_of_mass'] = list(new_pos)
            
            source_conns = source_node['params'].get('connections', [])
            twin_conns = []
            max_conn_id = 0
            for node in self.node_list:
                for c in node['params'].get('connections', []):
                    if isinstance(c.get('id'), int): max_conn_id = max(max_conn_id, c.get('id'))
            
            for i, conn in enumerate(source_conns):
                src_id = conn['source'].get('node_id')
                tgt_id = conn['target'].get('node_id')
                if src_id == old_id and tgt_id == old_id:
                    new_conn = copy.deepcopy(conn)
                    new_conn['source']['node_id'] = new_idx
                    new_conn['target']['node_id'] = new_idx
                    max_conn_id += 1
                    new_conn['id'] = max_conn_id
                    new_conn['name'] = f"{conn.get('name', 'conn')}_Twin"
                    if 'error' in new_conn: del new_conn['error']
                    twin_conns.append(new_conn)
            
            new_node_data['params']['connections'] = twin_conns
            node_btn = QPushButton(f"Node {new_idx + 1} (NEW TWIN)")
            node_btn.setMinimumHeight(50)
            node_btn.clicked.connect(lambda checked=False, idx=new_idx: self.select_node(idx))
            self.node_list_layout.addWidget(node_btn)
            new_node_data['button'] = node_btn
            self.node_list.append(new_node_data)
            self.select_node(new_idx)



    def _build_node_params(self, node_idx, node_data):

        import copy
        
        raw_params = node_data['params']
        populations = node_data['populations']
        tool_type = raw_params.get('tool_type', 'custom')

        if tool_type != 'custom' and not populations:
            selected_model = raw_params.get('tool_neuron_model', 'iaf_psc_alpha')
            neuron_models = [selected_model]
            types = [0]
            encoded_polynoms_per_type = [[]]
            prob_vec = [1.0]
            pop_nest_params = [{}]
        elif tool_type in ('Blob', 'Cone', 'CCW', 'Grid'):
            # Bei Tool-Typen: tool_neuron_model hat Priorität über existierende Populations
            selected_model = raw_params.get('tool_neuron_model', None)
            model_changed = False
            
            if selected_model:
                # Prüfe ob sich das Modell geändert hat
                if populations:
                    old_models = [pop.get('model') for pop in populations]
                    model_changed = any(m != selected_model for m in old_models)
                # Wenn tool_neuron_model gesetzt ist, überschreibe alle Populations damit
                neuron_models = [selected_model] * max(1, len(populations))
            else:
                # Fallback auf existierende Populations
                neuron_models = [pop['model'] for pop in populations] if populations else ['iaf_psc_alpha']
            
            # Setze die anderen erforderlichen Variablen
            types = list(range(len(populations))) if populations else [0]
            encoded_polynoms_per_type = []
            for pop in populations:
                poly_dict = pop.get('polynomials', None)
                if poly_dict and all(k in poly_dict for k in ['x', 'y', 'z']):
                    encoded_polynoms_per_type.append([poly_dict['x'], poly_dict['y'], poly_dict['z']])
                else:
                    encoded_polynoms_per_type.append([])
            if not encoded_polynoms_per_type:
                encoded_polynoms_per_type = [[]]
            
            prob_vec = raw_params.get('probability_vector', [])
            if not prob_vec or len(prob_vec) != len(neuron_models):
                prob_vec = [1.0 / len(neuron_models)] * len(neuron_models)
            
            # WICHTIG: Bei Modelländerung die alten Parameter verwerfen (sie passen nicht zum neuen Modell)
            if model_changed:
                pop_nest_params = [{}] * len(neuron_models)
            else:
                pop_nest_params = [pop.get('params', {}) for pop in populations] if populations else [{}]
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
            
            prob_vec = raw_params.get('probability_vector', [])
            if not prob_vec or len(prob_vec) != len(populations):
                if len(populations) > 0:
                    prob_vec = [1.0 / len(populations)] * len(populations)
                else:
                    prob_vec = []
            
            pop_nest_params = [pop.get('params', {}) for pop in populations]

        old_com = raw_params.get('old_center_of_mass', None)
        if old_com is None and node_data.get('original_node'):
            orig = node_data['original_node']
            if hasattr(orig, 'old_center_of_mass'):
                val = getattr(orig, 'old_center_of_mass', None)
                if isinstance(val, np.ndarray): val = val.tolist()
                old_com = val
            if old_com is None and hasattr(orig, 'center_of_mass'):
                val = orig.center_of_mass
                if isinstance(val, np.ndarray): val = val.tolist()
                old_com = val
        
        if old_com is None:
            old_com = raw_params.get('center_of_mass', [0.0, 0.0, 0.0])

        sx = float(raw_params.get('stretch_x', 1.0))
        sy = float(raw_params.get('stretch_y', 1.0))
        sz = float(raw_params.get('stretch_z', 1.0))
        transform_matrix = [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, sz]]

        existing_connections = []
        original_node = node_data.get('original_node')
        
        if original_node:
            if hasattr(original_node, 'connections') and original_node.connections:
                existing_connections = [copy.deepcopy(c) for c in original_node.connections]
            elif hasattr(original_node, 'parameters') and 'connections' in original_node.parameters:
                existing_connections = [copy.deepcopy(c) for c in original_node.parameters['connections']]
        
        if not existing_connections:
            existing_connections = raw_params.get('connections', [])

        existing_devices = []
        if original_node:
            source_list = []
            if hasattr(original_node, 'devices') and original_node.devices:
                source_list = original_node.devices
            elif hasattr(original_node, 'parameters') and 'devices' in original_node.parameters:
                source_list = original_node.parameters['devices']
            
            for d in source_list:
                d_safe = d.copy()
                if 'runtime_gid' in d_safe:
                    d_safe['runtime_gid'] = None
                existing_devices.append(copy.deepcopy(d_safe))
        
        if not existing_devices:
            existing_devices = raw_params.get('devices', [])

        for dev in existing_devices:
            if 'runtime_gid' in dev: dev['runtime_gid'] = None

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
            'auto_spike_recorder': raw_params.get('auto_spike_recorder', False),
            'auto_multimeter': raw_params.get('auto_multimeter', False),
            
            'devices': existing_devices,
            'connections': existing_connections,

            'ccw_syn_model': raw_params.get('ccw_syn_model', 'static_synapse'),
            'ccw_weight_ex': float(raw_params.get('ccw_weight_ex', 30.0)),
            'ccw_delay_ex': float(raw_params.get('ccw_delay_ex', 1.0)),
            'k': float(raw_params.get('k', 10.0)),
            'bidirectional': bool(raw_params.get('bidirectional', False)),
            'n_neurons': int(raw_params.get('n_neurons', 100)),
            'radius': float(raw_params.get('radius', 5.0)),
            'radius_top': float(raw_params.get('radius_top', 1.0)),
            'radius_bottom': float(raw_params.get('radius_bottom', 5.0)),
            'height': float(raw_params.get('height', 10.0)),
            'grid_side_length': int(raw_params.get('grid_side_length', 10)),
            'm': raw_params.get('center_of_mass', [0.0, 0.0, 0.0]),
            'center_of_mass': raw_params.get('center_of_mass', [0.0, 0.0, 0.0]),
            'displacement': raw_params.get('displacement', [0.0, 0.0, 0.0]),
            'displacement_factor': float(raw_params.get('displacement_factor', 1.0)),
            'rot_theta': float(raw_params.get('rot_theta', 0.0)),
            'rot_phi': float(raw_params.get('rot_phi', 0.0)),
            'transform_matrix': transform_matrix,
            'stretch_x': sx, 'stretch_y': sy, 'stretch_z': sz,
            'old_center_of_mass': old_com,
            'grid_size': raw_params.get('grid_size', [10, 10, 10]),
            'dt': float(raw_params.get('dt', 0.01)),
            'old': raw_params.get('old', True),
            'num_steps': int(raw_params.get('num_steps', 8)),
            'sparse_holes': int(raw_params.get('sparse_holes', 0)),
            'sparsity_factor': float(raw_params.get('sparsity_factor', 0.9)),
            'polynom_max_power': int(raw_params.get('polynom_max_power', 5)),
            'conn_prob': [],
            'field': None,
            'coefficients': None
        }
    
    
    
    
    def save_current_population_params(self):
        if self.current_node_idx is not None and self.current_pop_idx is not None:
            if self.current_node_idx < 0 or self.current_node_idx >= len(self.node_list): return
            node = self.node_list[self.current_node_idx]
            if self.current_pop_idx < 0 or self.current_pop_idx >= len(node['populations']): return
            pop = node['populations'][self.current_pop_idx]
            if self.pop_param_widget.current_model:
                pop['model'] = self.pop_param_widget.current_model
                pop['params'] = {k: w.get_value() for k, w in self.pop_param_widget.parameter_widgets.items()}
                if 'button' in pop: pop['button'].setText(f"Pop {self.current_pop_idx+1}: {pop['model']}")
    
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

    def delete_graph(self):
        if not self.current_graph: return
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(self, 'Delete Graph', f"Really delete '{self.current_graph.graph_name}'?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.graph_list.remove(self.current_graph)
            global _nest_simulation_has_run
            import nest
            nest.ResetKernel()
            _nest_simulation_has_run = False
            for graph in self.graph_list:
                for node in graph.node_list: node.populate_node()
            self.current_graph = None
            self.current_graph_id = None
            self.refresh_graph_list()
            self.graphUpdated.emit(-1)
    def _node_was_edited(self, node_data):

        original = node_data.get('original_node')
        
        if not original:
            return True, True, True
        
        any_change = False
        polynomials_changed = False
        structural_changed = False
        
        if not hasattr(original, 'parameters'):
            return True, True, True
        
        old_params = original.parameters
        new_params = node_data['params']
        
        structural_keys = [
            'grid_size', 'num_steps', 'sparsity_factor', 'sparse_holes',
            'dt', 'displacement', 'displacement_factor',
            'rot_theta', 'rot_phi', 'polynom_max_power',
            'stretch_x', 'stretch_y', 'stretch_z',
            'n_neurons', 'radius', 'radius_top', 'radius_bottom', 'height', 'grid_side_length' 
        ]
        
        for key in structural_keys:
            old_val = old_params.get(key)
            new_val = new_params.get(key)
            
            if old_val != new_val:
                any_change = True
                structural_changed = True

        old_com = np.array(original.center_of_mass)
        new_com = np.array(new_params.get('center_of_mass', [0,0,0]))
        if not np.allclose(old_com, new_com, atol=1e-6):
            any_change = True
        
        old_pop_count = len(original.population) if hasattr(original, 'population') and original.population else 0
        new_pop_count = len(node_data['populations'])
        
        if old_pop_count != new_pop_count:
            any_change = True
            structural_changed = True
        
        old_models = original.neuron_models if hasattr(original, 'neuron_models') else []
        new_models = [pop['model'] for pop in node_data['populations']]
        if old_models != new_models:
            any_change = True
            structural_changed = True 

        if 'population_nest_params' in old_params:
            old_nest_params = old_params['population_nest_params']
            new_nest_params = [pop.get('params', {}) for pop in node_data['populations']]
            
            if len(old_nest_params) == len(new_nest_params):
                for i, (old_p, new_p) in enumerate(zip(old_nest_params, new_nest_params)):
                    if old_p != new_p:
                        any_change = True
                        structural_changed = True 

        old_prob = old_params.get('probability_vector', [])
        new_prob = new_params.get('probability_vector', [])
        if old_prob != new_prob:
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
                any_change = True
                polynomials_changed = True
                structural_changed = True
            else:
                if old_polys != new_polys:
                    any_change = True
                    polynomials_changed = True
                    structural_changed = True

        return any_change, polynomials_changed, structural_changed
    
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
        
        gx = self.spin_g_x.value()
        gy = self.spin_g_y.value()
        gz = self.spin_g_z.value()
        graph_global_offset = np.array([gx, gy, gz])
        
        self.current_graph.global_offset = [gx, gy, gz]
        
        print(f"\n=== Saving changes to Graph {self.current_graph_id} (Global Offset: {graph_global_offset}) ===")
        
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
            
            local_disp = np.array(node_params.get('displacement', [0.0, 0.0, 0.0]))
            
            total_displacement = local_disp + graph_global_offset
            
            node_params['displacement'] = total_displacement.tolist()
            
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

                print(f"  Node {node_idx}: Rebuild due to potential offset shift.")
                new_node.build()
                
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

        self.combo_graph = QComboBox()
        self.combo_node = QComboBox()
        self.combo_pop = QComboBox()
        
        for c in [self.combo_graph, self.combo_node, self.combo_pop]:
            c.setStyleSheet("background-color: #1e1e1e; border: 1px solid #555; padding: 2px;")
            c.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.combo_graph.currentIndexChanged.connect(self.on_graph_changed)
        self.combo_node.currentIndexChanged.connect(self.on_node_changed)

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
                    model = safe_get_model(pop, "empty")
                    self.combo_pop.addItem(f"Pop {i}: {model}", i)

    def get_selection(self):
        g = self.combo_graph.currentData()
        n = self.combo_node.currentData()
        p = self.combo_pop.currentData()
        return g, n, p


class ConnectionTool(QWidget):
    connectionsCreated = pyqtSignal()
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
        self.targets_layout.addStretch()
        
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
        self.weight_spin.setRange(-1000000, 1000000)
        self.weight_spin.setValue(1.0)
        self.weight_spin.setDecimals(3)
        self.weight_spin.setPrefix("W: ")
        
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 10000)
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
        
        self.create_all_btn = QPushButton("CREATE IN NEST")
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
                model = safe_get_model(pop, "empty")
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
                'source': {'graph_id': s_gid, 'node_id': s_nid, 'pop_id': s_pid},
                'target': {'graph_id': t_gid, 'node_id': t_nid, 'pop_id': t_pid},
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
        if success > 0: self.connectionsCreated.emit()

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
            if param_name in ['weight', 'delay']: continue
            
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


class BlinkingNetworkWidget(QWidget):
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        self.plotter.enable_anti_aliasing()
        self.layout.addWidget(self.plotter)
        
        self.neuron_mesh = None
        self.edge_mesh = None
        
        self.activity = None
        self.edge_intensities = None
        self.base_colors = None
        
        self.connected_mask = None
        
        self.base_opacity = 0.2
        
        self.connectivity_map = []
        self.population_ranges = {}
        
        self.simulation_running = False
        self.is_active = True
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        
        self.build_scene()
        self.timer.start(30)

    def set_base_opacity(self, value):
        pass

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
        self.neuron_mesh.point_data["display_color"] = self.base_colors
        
        self.plotter.add_mesh(
            self.neuron_mesh, scalars="display_color", rgb=True,
            point_size=6, render_points_as_spheres=True,
            ambient=0.3, diffuse=0.8
        )
        
        lines_data = []
        current_edge_idx = 0
        VISUAL_EDGE_LIMIT = 200
        
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
            
            colors = ["#333333", "#555555", "#FFD700", "#FFFFFF"]
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

    def stop_simulation(self):
        self.simulation_running = False

    def animate(self):
        if (not self.isVisible() or not self.is_active or
            self.neuron_mesh is None or not hasattr(self, 'plotter')):
            return
        
        try:
            self.activity *= 0.92
            
            if self.edge_intensities is not None:
                diff = self.edge_intensities - self.base_opacity
                self.edge_intensities = self.base_opacity + (diff * 0.88)

            
            if len(self.connectivity_map) > 0:
                import random
                n_flash = max(1, int(len(self.connectivity_map) * 0.05))
                flash_conns = random.sample(self.connectivity_map, n_flash)
                
                for conn in flash_conns:
                    self.edge_intensities[conn['edge_slice']] = 1.0
                    
                    targets = conn['tgt_indices']
                    if len(targets) > 0:
                        n_hits = max(1, int(len(targets) * 0.2))
                        hit_idx = targets[np.random.choice(len(targets), size=n_hits)]
                        self.activity[hit_idx] += 0.3

            self.activity = np.clip(self.activity, 0, 1.2)
            
            act_norm = np.clip(self.activity, 0, 1)[:, None]
            target_color = np.array([1.0, 1.0, 1.0])
            display_colors = self.base_colors * (1.0 - act_norm * 0.6) + target_color * (act_norm * 0.6)
            
            self.neuron_mesh.point_data["display_color"] = display_colors
            
            if self.edge_mesh:
                self.edge_mesh.cell_data["glow"] = self.edge_intensities

            if self.plotter.render_window:
                self.plotter.update()
            
        except Exception:
            pass

    def closeEvent(self, event):
        self.is_active = False
        self.simulation_running = False
        if hasattr(self, 'timer'): self.timer.stop()
        if hasattr(self, 'plotter'): self.plotter.close()
        super().closeEvent(event)


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
        except Exception:
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
        import copy
        import time
        
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

            
            try:
                target_model = nest.GetStatus(tgt_pop, 'model')[0]
            except Exception:
                target_model = 'unknown'
            
            if 'ht_neuron' in str(target_model):
                rec_map = {1: 'AMPA', 2: 'NMDA', 3: 'GABA_A', 4: 'GABA_B'}
                current_rec = params.get('receptor_type', 0)
                if isinstance(current_rec, int) and current_rec in rec_map:
                    params['receptor_type'] = rec_map[current_rec]
                elif current_rec == 0:
                    params.pop('receptor_type', None)
            
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
            else:
                # Auto-detect MC models
                try:
                    target_model = nest.GetStatus(tgt_pop[0], 'model')[0]
                    auto_receptor = get_receptor_type_for_model(target_model, excitatory=(weight >= 0))
                    if auto_receptor > 0:
                        syn_spec['receptor_type'] = auto_receptor
                except Exception: pass  # Model may not support receptor_type
            
            if rule == 'one_to_one' and len(src_pop) != len(tgt_pop):
                return False, f"one_to_one size mismatch: {len(src_pop)} vs {len(tgt_pop)}"
            
            nest.Connect(src_pop, tgt_pop, conn_spec, syn_spec)
            
            try:
                src_graph = self.graphs.get(source_info['graph_id'])
                tgt_graph = self.graphs.get(target_info['graph_id'])

                if src_graph and tgt_graph:
                    src_node_obj = src_graph.get_node(source_info['node_id'])
                    tgt_node_obj = tgt_graph.get_node(target_info['node_id'])

                    if src_node_obj and tgt_node_obj:
                        
                        if not hasattr(src_node_obj, 'connections'):
                            src_node_obj.connections = []
                        
                        exists = False
                        for existing in src_node_obj.connections:
                            if existing.get('id') == connection.get('id'):
                                exists = True
                                break
                        
                        if not exists:
                            src_node_obj.connections.append(copy.deepcopy(connection))
                            
                        src_node_obj.add_neighbor(tgt_node_obj)
                    
                return True, f"✓ {conn_name} created"

            except Exception as save_err:
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
        
        header = QLabel("Connection Queue")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)
        
        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        
        self.remove_btn = QPushButton("Remove")
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
            print(f"  ✗ {conn_name}: {str(e)}")
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
    
    import os
    output_dir = "Simulation_History"
    os.makedirs(output_dir, exist_ok=True)
    
    nest.ResetKernel()
    nest.SetKernelStatus({
        "data_path": output_dir,      
        "overwrite_files": True,      
        "print_time": False,
        "resolution": 0.1
    })

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
        
        self.btn_start = QPushButton("RUN SIMULATION")
        self.btn_start.setStyleSheet(start_style)
        
        self.btn_stop = QPushButton("STOP SIMULATION")
        self.btn_stop.setStyleSheet(stop_style)
        
        self.btn_save = QPushButton("SAVE GRAPHS")
        self.btn_save.setStyleSheet(io_style)
    
        self.btn_load = QPushButton("LOAD GRAPHS")
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
        if hasattr(nest, 'NodeCollection') and isinstance(obj, nest.NodeCollection):
            return obj.tolist()
        return super().default(obj)

def _clean_params(params: dict) -> dict:
    clean = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (np.integer, np.int64, np.int32)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            clean[k] = float(v)
        elif hasattr(nest, 'NodeCollection') and isinstance(v, nest.NodeCollection):
            clean[k] = v.tolist()
        elif isinstance(v, dict):
            clean[k] = _clean_params(v)
        elif isinstance(v, list):
            clean_list = []
            for x in v:
                if isinstance(x, np.ndarray):
                    clean_list.append(x.tolist())
                elif isinstance(x, (np.floating, np.float64, np.float32)):
                    clean_list.append(float(x))
                elif isinstance(x, (np.integer, np.int64, np.int32)):
                    clean_list.append(int(x))
                elif isinstance(x, dict):
                    clean_list.append(_clean_params(x))
                elif hasattr(nest, 'NodeCollection') and isinstance(x, nest.NodeCollection):
                    clean_list.append(x.tolist())
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
        self.syn_param_widgets = {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        header = QLabel("EDIT CONNECTION")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: #673AB7; padding: 5px; border-bottom: 2px solid #673AB7;")
        layout.addWidget(header)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content_widget = QWidget()
        self.form_layout = QFormLayout(content_widget)
        self.form_layout.setSpacing(10)
        
        self.name_input = QLineEdit()
        self.form_layout.addRow("Name:", self.name_input)
        
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(-1e6, 1e6)
        self.weight_spin.setDecimals(3)
        self.weight_spin.setPrefix("W: ")
        self.form_layout.addRow("Weight:", self.weight_spin)
        
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 10000.0)
        self.delay_spin.setDecimals(2)
        self.delay_spin.setSuffix(" ms")
        self.delay_spin.setPrefix("D: ")
        self.form_layout.addRow("Delay:", self.delay_spin)
        
        self.syn_model_combo = QComboBox()
        self.syn_model_combo.addItems(sorted(SYNAPSE_MODELS.keys()))
        self.syn_model_combo.currentTextChanged.connect(self._on_model_changed)
        self.form_layout.addRow("Model:", self.syn_model_combo)
        
        self.rule_combo = QComboBox()
        self.rule_combo.addItems(["all_to_all", "fixed_indegree", "fixed_outdegree", "fixed_total_number", "pairwise_bernoulli", "one_to_one"])
        self.rule_combo.currentTextChanged.connect(self._on_rule_changed)
        self.form_layout.addRow("Rule:", self.rule_combo)
        
        self.rule_param_container = QWidget()
        self.rule_param_layout = QFormLayout(self.rule_param_container)
        self.rule_param_layout.setContentsMargins(0,0,0,0)
        self.form_layout.addRow(self.rule_param_container)
        
        self.spin_indegree = QSpinBox(); self.spin_indegree.setRange(0, int(1e9))
        self.spin_outdegree = QSpinBox(); self.spin_outdegree.setRange(0, int(1e9))
        self.spin_N = QSpinBox(); self.spin_N.setRange(0, int(1e9))
        self.spin_p = QDoubleSpinBox(); self.spin_p.setRange(0.0, 1.0); self.spin_p.setDecimals(4)
        
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); line.setStyleSheet("color: #555;")
        self.form_layout.addRow(line)
        self.form_layout.addRow(QLabel("<b>Dynamic Parameters:</b>"))
        
        self.dyn_container = QWidget()
        self.dyn_layout = QVBoxLayout(self.dyn_container)
        self.dyn_layout.setContentsMargins(0,0,0,0)
        self.form_layout.addRow(self.dyn_container)
        
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        btn_layout = QHBoxLayout()
        self.btn_apply = QPushButton("UPDATE CONNECTION")
        self.btn_apply.setMinimumHeight(40)
        self.btn_apply.setStyleSheet("background-color: #673AB7; color: white; font-weight: bold; border-radius: 4px;")
        self.btn_apply.clicked.connect(self.apply_changes)
        btn_layout.addWidget(self.btn_apply)
        
        layout.addLayout(btn_layout)
        
    def load_data(self, conn_data):
        self.current_conn_data = conn_data
        params = conn_data.get('params', {})
        
        self.name_input.setText(conn_data.get('name', ''))
        self.weight_spin.setValue(float(params.get('weight', 1.0)))
        self.delay_spin.setValue(float(params.get('delay', 1.0)))
        
        model = params.get('synapse_model', 'static_synapse')
        idx = self.syn_model_combo.findText(model)
        if idx >= 0:
            self.syn_model_combo.setCurrentIndex(idx)
        else:
            self.syn_model_combo.addItem(model)
            self.syn_model_combo.setCurrentText(model)
            
        rule = params.get('rule', 'all_to_all')
        idx_r = self.rule_combo.findText(rule)
        if idx_r >= 0: self.rule_combo.setCurrentIndex(idx_r)
        
        if 'indegree' in params: self.spin_indegree.setValue(int(params['indegree']))
        if 'outdegree' in params: self.spin_outdegree.setValue(int(params['outdegree']))
        if 'N' in params: self.spin_N.setValue(int(params['N']))
        if 'p' in params: self.spin_p.setValue(float(params['p']))
        
        for param_name, widget in self.syn_param_widgets.items():
            if param_name in params:
                val = params[param_name]
                try:
                    if isinstance(widget, DoubleInputField): widget.spinbox.setValue(float(val))
                    elif isinstance(widget, IntegerInputField): widget.spinbox.setValue(int(val))
                except (ValueError, TypeError): pass

    def _on_rule_changed(self, rule_name):
        while self.rule_param_layout.count():
            item = self.rule_param_layout.takeAt(0)
            if item.widget(): item.widget().setParent(None)
            
        if rule_name == 'fixed_indegree':
            self.rule_param_layout.addRow("Indegree:", self.spin_indegree)
        elif rule_name == 'fixed_outdegree':
            self.rule_param_layout.addRow("Outdegree:", self.spin_outdegree)
        elif rule_name == 'fixed_total_number':
            self.rule_param_layout.addRow("Total N:", self.spin_N)
        elif 'bernoulli' in rule_name:
            self.rule_param_layout.addRow("Prob (p):", self.spin_p)

    def _on_model_changed(self, model_name):
        while self.dyn_layout.count():
            item = self.dyn_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.syn_param_widgets.clear()
        
        if model_name not in SYNAPSE_MODELS:
            return

        model_def = SYNAPSE_MODELS[model_name]
        params_def = model_def.get('params', {})
        
        for param_key, info in params_def.items():
            if param_key in ['weight', 'delay']: continue
            
            p_type = info.get('type', 'float')
            default = info.get('default', 0)
            
            widget = None
            if p_type == 'float':
                widget = DoubleInputField(param_key, default_value=float(default))
            elif p_type == 'int':
                widget = IntegerInputField(param_key, default_value=int(default))
                
            if widget:
                self.dyn_layout.addWidget(widget)
                self.syn_param_widgets[param_key] = widget

    def apply_changes(self):
        if self.current_conn_data is None: return
        
        self.current_conn_data['name'] = self.name_input.text()
        params = self.current_conn_data['params']
        
        params['weight'] = self.weight_spin.value()
        params['delay'] = self.delay_spin.value()
        params['synapse_model'] = self.syn_model_combo.currentText()
        
        rule = self.rule_combo.currentText()
        params['rule'] = rule
        
        if rule == 'fixed_indegree': params['indegree'] = self.spin_indegree.value()
        elif rule == 'fixed_outdegree': params['outdegree'] = self.spin_outdegree.value()
        elif rule == 'fixed_total_number': params['N'] = self.spin_N.value()
        elif 'bernoulli' in rule: params['p'] = self.spin_p.value()
        
        for param_name, widget in self.syn_param_widgets.items():
            params[param_name] = widget.get_value()
            
        print(f"Connection '{self.current_conn_data['name']}' updated locally. (Model: {params['synapse_model']})")
        

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
                    model = safe_get_model(pop, "empty")
                    self.combo_pop.addItem(f"Pop {i}: {model}", i)

    def get_selection(self):
        return {
            'graph_id': self.combo_graph.currentData(),
            'node_id': self.combo_node.currentData(),
            'pop_id': self.combo_pop.currentData()
        }


class DeviceConfigPage(QWidget):
    deviceCreated = pyqtSignal(dict)
    deviceUpdated = pyqtSignal(dict, dict)

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
        
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0,0,0,0)
        
        self.target_selector = DeviceTargetSelector(self.graph_list)
        left_layout.addWidget(self.target_selector)
        left_layout.addStretch()
        
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

        self.btn_create = QPushButton(f"✚ Create {self.device_label}")
        self.btn_create.setMinimumHeight(45)
        self.btn_create.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_create.setStyleSheet(style_create)
        self.btn_create.clicked.connect(self.create_device)
        right_layout.addWidget(self.btn_create)

        self.btn_save = QPushButton(f"Save Changes & Reset")
        self.btn_save.setMinimumHeight(45)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setStyleSheet(style_save)
        self.btn_save.clicked.connect(self.update_device)
        self.btn_save.setVisible(False)
        right_layout.addWidget(self.btn_save)
        
        self.btn_cancel = QPushButton("Cancel Edit")
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.setStyleSheet(style_cancel)
        self.btn_cancel.clicked.connect(self.reset_to_create_mode)
        self.btn_cancel.setVisible(False)
        right_layout.addWidget(self.btn_cancel)

        main_layout.addWidget(left_container, 1)
        main_layout.addWidget(right_container, 2)

    def load_device_data(self, device_data):
        print(f"Loading data into editor for: {device_data.get('model')}")
        self.current_edit_device = device_data
        
        self.header_label.setText(f"EDIT MODE: {self.device_label} (ID: {device_data.get('id')})")
        self.header_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: #4CAF50; border-bottom: 2px solid #4CAF50;")
        
        self.btn_create.setVisible(False)
        self.btn_save.setVisible(True)
        self.btn_cancel.setVisible(True)
        
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
                    except (ValueError, TypeError): pass
                elif isinstance(widget, QLineEdit):
                    if isinstance(val, list):
                        widget.setText(", ".join(map(str, val)))
                    else:
                        widget.setText(str(val))

    def reset_to_create_mode(self):
        self.current_edit_device = None
        
        self.header_label.setText(f"Configure {self.device_label}")
        self.header_label.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {self._get_color()};")
        
        self.btn_create.setVisible(True)
        self.btn_save.setVisible(False)
        self.btn_cancel.setVisible(False)
        

    def _gather_data(self):
        target = self.target_selector.get_selection()
        if target['graph_id'] is None:
            print("No target selected!")
            return None

        device_params = {}
        conn_params = {}
        
        periodic_config = {}
        
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
            elif key.startswith("gen_p_"):
                periodic_config[key] = val
            else:
                if val is not None:
                    device_params[key] = val
        
        if "spike_generator" in self.device_type:
            p_int = periodic_config.get("gen_p_interval", 0.0)
            p_start = periodic_config.get("gen_p_start", 0.0)
            p_stop = periodic_config.get("gen_p_stop", 0.0)
            
            manual_times = device_params.get("spike_times", [])
            
            if p_int > 0.0001 and p_stop > p_start:
                generated_times = np.arange(p_start, p_stop, p_int).tolist()
                print(f"Auto-Generated {len(generated_times)} spikes (Interval {p_int}ms)")
                
                all_times = sorted(list(set(manual_times + generated_times)))
                device_params["spike_times"] = all_times
                
                if "spike_multiplicities" in device_params and device_params["spike_multiplicities"]:
                    del device_params["spike_multiplicities"]

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

    def _get_color(self):
        if "generator" in self.device_type: return "#FF9800"
        if "recorder" in self.device_type or "meter" in self.device_type: return "#E91E63"
        return "#999"

    def _build_parameters(self):
        self.form_layout.addRow(QLabel("--- Device Settings ---"))
        
        if "recorder" in self.device_type or "meter" in self.device_type:
             self._add_param("start", 0.0, "Start Time (ms)")
             self._add_param("stop", 10000.0, "Stop Time (ms)")
        
        if "poisson_generator" in self.device_type:
            self._add_param("rate", 1000.0, "Rate (Hz)")
            self._add_param("start", 0.0, "Start Time (ms)")
            self._add_param("stop", 10000.0, "Stop Time (ms)")

        elif "noise_generator" in self.device_type:
            self._add_param("mean", 0.0, "Mean (pA)")
            self._add_param("std", 200.0, "Std Dev (pA)")
            self._add_param("dt", 1.0, "Time Step (ms)")
            self._add_param("start", 0.0, "Start (ms)")

        elif "dc_generator" in self.device_type:
            self._add_param("amplitude", 100.0, "Amplitude (pA)")
            self._add_param("start", 0.0, "Start (ms)")
            self._add_param("stop", 10000.0, "Stop (ms)")

        elif "ac_generator" in self.device_type:
            self._add_param("amplitude", 50.0, "Amplitude (pA)")
            self._add_param("frequency", 10.0, "Frequency (Hz)")
            self._add_param("phase", 0.0, "Phase (deg)")
            self._add_param("offset", 0.0, "Offset (pA)")
            
        elif "step_current_generator" in self.device_type:
            self._add_text_param("amplitude_times", "100.0, 300.0, 500.0", "Times (ms)")
            self._add_text_param("amplitude_values", "100.0, 0.0, -50.0", "Currents (pA)")
            
        elif "spike_generator" in self.device_type:
             self.form_layout.addRow(QLabel("<b>Manual Spikes:</b>"))
             self._add_text_param("spike_times", "10.0, 20.5", "Exact Times (ms)")
             self._add_text_param("spike_multiplicities", "", "Multiplicities")
             
             self.form_layout.addRow(QLabel("<b>Periodic Pattern (Optional):</b>"))
             self._add_param("gen_p_start", 0.0, "Start (ms)")
             self._add_param("gen_p_stop", 1000.0, "Stop (ms)")
             self._add_param("gen_p_interval", 0.0, "Interval (ms, 0=Off)")
             self.form_layout.addRow(QLabel("<small style='color:#888'>If Interval > 0, spikes are auto-generated.</small>"))
             
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
        except (ValueError, TypeError): return []


class ToolsWidget(QWidget):
    deviceAdded = pyqtSignal()
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
        
        self.help_page = DeviceHelpWidget()
        self.config_stack.addWidget(self.help_page)
        
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
        current = self.config_stack.currentWidget()
        if isinstance(current, DeviceConfigPage):
            current.reset_to_create_mode()

        for btn in self.button_map:
            btn.setChecked(btn == clicked_btn)
            
        self.config_stack.setCurrentIndex(index)
        
        widget = self.config_stack.widget(index)
        if isinstance(widget, DeviceConfigPage):
            widget.target_selector.graph_list = self.graph_list
            widget.target_selector.refresh()
            widget.reset_to_create_mode()

    def reset_view(self):
        self.config_stack.setCurrentIndex(0)
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
        
        existing_devs = target_node.parameters['devices']
        if existing_devs:
            int_ids = [d.get('id', 0) for d in existing_devs if isinstance(d.get('id'), int)]
            if int_ids:
                current_max_id = max(int_ids)
                next_dev_id = current_max_id + 1
            else:
                next_dev_id = 0
        else:
            next_dev_id = 0
            
        device_record = {
            "id": next_dev_id,
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
            
            self.reset_view()
            self.deviceAdded.emit()
            
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
        
        lbl_icon = QLabel("🛠️")
        lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_icon.setStyleSheet("font-size: 48px; margin-bottom: 10px;")
        layout.addWidget(lbl_icon)
        
        lbl_title = QLabel("Device Toolbox Guide")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #FF9800; font-size: 18px; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(lbl_title)
        
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
            
            <h3 style='color: #E91E63;'> Recorders & Meters (Output)</h3>
            <ul>
                <li><b>Spike Recorder:</b> Records firing times (spikes) from neurons. Essential for Raster Plots.</li>
                <li><b>Voltmeter / Multimeter:</b> Records continuous membrane potentials (V_m). connect to neurons to see analog traces.</li>
            </ul>

            <h3 style='color: #FF9800;'>Generators (Input)</h3>
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


class StructuresWidget(QWidget):

    structureSelected = pyqtSignal(str, list, list)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #2b2b2b; border-bottom: 1px solid #444;")
        hl = QVBoxLayout(header_frame)
        hl.setContentsMargins(10, 10, 10, 10)
        
        header_lbl = QLabel("CORTICAL STRUCTURES")
        header_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: #E91E63;")
        hl.addWidget(header_lbl)
        
        info = QLabel("Select a region to auto-generate a node patch.")
        info.setStyleSheet("color: #888; font-style: italic;")
        hl.addWidget(info)
        
        layout.addWidget(header_frame)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background-color: #1e1e1e;")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        container.setStyleSheet("background-color: #1e1e1e;")
        
        self.items_layout = QHBoxLayout(container)
        self.items_layout.setSpacing(5)
        self.items_layout.setContentsMargins(10, 10, 10, 10)
        self.items_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        region_data = {r_key: {} for r_key in region_names.values()}
        
        for model, region_map in distributions.items():
            for r_key, prob in region_map.items():
                if prob > 0:
                    region_data[r_key][model] = prob

        for display_name, r_key in region_names.items():
            if r_key not in region_data or not region_data[r_key]:
                continue
            
            models_probs = region_data[r_key]
            model_list = list(models_probs.keys())
            prob_list = list(models_probs.values())
            
            total = sum(prob_list)
            if total > 0:
                prob_list = [p/total for p in prob_list]
            
            btn = QPushButton(display_name)
            btn.setMinimumHeight(60)
            btn.setFixedWidth(160)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #37474F;
                    color: white;
                    font-weight: bold;
                    border: 1px solid #455A64;
                    border-radius: 4px;
                    font-size: 11px;
                    text-align: center;
                    padding: 5px;
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
            
            self.items_layout.addWidget(btn)
        
        scroll.setWidget(container)
        layout.addWidget(scroll)


class FlowFieldWidget(QWidget):
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.target_graph_id = None
        self.target_node_id = None
        
        self.dt = 0.05
        self.current_step = 0
        self.active_populations_data = []
        
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(5, 5, 5, 5)
        toolbar.setSpacing(5)
        
        btn_reset = QPushButton("↺ Reset")
        btn_reset.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        btn_reset.clicked.connect(self.reset_positions)
        toolbar.addWidget(btn_reset)
        
        btn_prev10 = QPushButton("<< -10")
        btn_prev10.setStyleSheet("background-color: #5D4037; color: white; font-weight: bold; padding: 5px;")
        btn_prev10.clicked.connect(lambda: self.perform_step(-10))
        toolbar.addWidget(btn_prev10)

        btn_prev = QPushButton("< -1")
        btn_prev.setStyleSheet("background-color: #444; color: white; font-weight: bold; padding: 5px;")
        btn_prev.clicked.connect(lambda: self.perform_step(-1))
        toolbar.addWidget(btn_prev)
        
        self.lbl_step = QLabel("Step: 0")
        self.lbl_step.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_step.setFixedWidth(80)
        self.lbl_step.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 12px; border: 1px solid #444; border-radius: 4px; padding: 2px;")
        toolbar.addWidget(self.lbl_step)
        
        btn_next = QPushButton("+1 >")
        btn_next.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        btn_next.clicked.connect(lambda: self.perform_step(1))
        toolbar.addWidget(btn_next)

        btn_next10 = QPushButton(">> +10")
        btn_next10.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 5px;")
        btn_next10.clicked.connect(lambda: self.perform_step(10))
        toolbar.addWidget(btn_next10)
        
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 1.0)
        self.dt_spin.setSingleStep(0.01)
        self.dt_spin.setValue(0.05)
        self.dt_spin.setPrefix("dt: ")
        self.dt_spin.setDecimals(3)
        self.dt_spin.valueChanged.connect(self.update_dt)
        toolbar.addWidget(self.dt_spin)
        
        self.layout.addLayout(toolbar)
        
        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        self.plotter.add_axes()
        
        self.layout.addWidget(self.plotter)
        
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
        has_valid_flow = False

        for pop_idx, positions in enumerate(target_node.positions):
            if positions is None or len(positions) == 0:
                continue
            
            funcs = []
            
            if pop_idx < len(encoded_per_type):
                data_block = encoded_per_type[pop_idx]
                
                if isinstance(data_block, (list, tuple)) and len(data_block) == 3:
                    try:
                        funcs = poly_gen.decode_multiple(data_block)
                    except Exception as e:
                        print(f"Error decoding polys for pop {pop_idx}: {e}")
                        funcs = []
            
            if not funcs or len(funcs) != 3:
                continue
            
            f1, f2, f3 = funcs
            if not (callable(f1) and callable(f2) and callable(f3)):
                continue
                
            has_valid_flow = True

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

        if not has_valid_flow:
            self.text_actor.SetInput(f"{target_node.name}: No Flow Field Data (Static Structure)")

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
        if not self.active_populations_data:
            return

        direction = 1 if n_steps > 0 else -1
        iterations = abs(n_steps)

        
        for data in self.active_populations_data:
            pos = data['current_pos']
            f1, f2, f3 = data['funcs']
            
            for _ in range(iterations):
                x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
                u = f1(x, y, z)
                v = f2(x, y, z)
                w = f3(x, y, z)
                
                velocity = np.column_stack((u, v, w))
                
                pos = pos + (direction * velocity * self.dt)
            
            data['current_pos'] = pos
            
            data['points_mesh'].points = pos
            
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


      

class SafeGLViewWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.opts['antialias'] = True
        self.opts['viewport'] = None
    
    def paintGL(self, *args, **kwargs):
        if not self.isVisible() or not self.isValid():
            return
        try:
            self.makeCurrent()
            super().paintGL(*args, **kwargs)
        except Exception:
            pass

class SimulationViewWidget(QWidget):
    sigSpeedChanged = pyqtSignal(int)

    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        
        self.scene_loaded = False
        self.active_tool = None
        self.is_paused = True
        
        self.all_points = None
        self.base_colors = None
        self.current_colors = None
        self.global_ids = None
        self.gid_to_idx = {}
        
        self.anim_rg = None
        self.anim_b = None
        self.heat = None
        
        self.electrodes = {}
        self.next_electrode_id = 1
        
        self.stim_params = {'radius': 5.0, 'weight': 1000.0, 'delay': 1.0}
        self.conn_params = {
            'weight': 50.0, 'delay': 1.0, 'radius': 5.0,
            'rule': 'all_to_all', 'p': 0.1, 'indegree': 10,
            'outdegree': 10, 'N': 100,
            'synapse_model': 'static_synapse',
            'allow_autapses': False, 'allow_multapses': True
        }
        
        self.conn_source_gids = []
        self.conn_source_indices = []
        self.temp_connections = []

        self.point_size = 8.0
        self.decay_flash = 0.60
        self.decay_tail = 0.90
        self.decay_heat = 0.80
        
        self.view = None
        self.scatter_item = None
        
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.update_animation)
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.view = SafeGLViewWidget()
        self.view.opts['distance'] = 40
        self.view.setBackgroundColor('#050505')
        gz = gl.GLGridItem(); gz.translate(0, 0, -10); self.view.addItem(gz)
        
        self._original_mouse_press = self.view.mousePressEvent
        self.view.mousePressEvent = self.on_mouse_press
        
        layout.addWidget(self.view, 85)

        self.toolbar_container = QWidget()
        self.toolbar_container.setStyleSheet("background-color: #232323; border-top: 2px solid #444;")
        
        self.toolbar_layout = QHBoxLayout(self.toolbar_container)
        self.toolbar_layout.setContentsMargins(5, 5, 5, 5)
        self.toolbar_layout.setSpacing(10)
        
        left_panel = QWidget()
        left_panel.setStyleSheet("background-color: transparent;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        
        row1 = QHBoxLayout()
        self.btn_load = QPushButton("LOAD")
        self.btn_load.clicked.connect(self.load_scene)
        self.btn_load.setStyleSheet("background-color: #1565C0; color: white; font-weight: bold; border-radius: 4px; padding: 4px;")
        
        self.chk_visual_only = QCheckBox("Visual Only")
        self.chk_visual_only.setToolTip("Disables Data Tab updates for maximum performance")
        self.chk_visual_only.setStyleSheet("color: #FFD700; font-weight: bold; margin-left: 5px;")
        self.chk_visual_only.setChecked(False)

        self.btn_clear = QPushButton("✕")
        self.btn_clear.setFixedWidth(30)
        self.btn_clear.clicked.connect(self.unload_scene)
        self.btn_clear.setStyleSheet("background-color: #444; color: white; border-radius: 4px;")
        
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setToolTip("Animation Delay")
        self.slider_speed.setRange(0, 200); self.slider_speed.setValue(0)
        self.slider_speed.valueChanged.connect(self.sigSpeedChanged.emit)
        
        row1.addWidget(self.btn_load)
        row1.addWidget(self.chk_visual_only)
        row1.addWidget(self.btn_clear)
        row1.addWidget(self.slider_speed)
        left_layout.addLayout(row1)
        
        style_tool = """
            QPushButton {
                background-color: #333; color: #bbb; font-weight: bold; font-size: 11px;
                border: 1px solid #555; border-radius: 4px; padding: 6px;
            }
            QPushButton:checked {
                background-color: #FFEB3B; color: black; border: 1px solid #FBC02D;
            }
            QPushButton:hover { background-color: #444; }
        """
        style_conn = style_tool.replace("#FFEB3B", "#FF9800").replace("#FBC02D", "#F57C00")
        
        tools_row = QHBoxLayout()
        self.btn_tool_inject = QPushButton("ELECTRODE")
        self.btn_tool_inject.setCheckable(True)
        self.btn_tool_inject.clicked.connect(lambda: self.set_active_tool('injector'))
        self.btn_tool_inject.setStyleSheet(style_tool)
        
        self.btn_tool_connect = QPushButton("CONNECT")
        self.btn_tool_connect.setCheckable(True)
        self.btn_tool_connect.clicked.connect(lambda: self.set_active_tool('connector'))
        self.btn_tool_connect.setStyleSheet(style_conn)
        
        self.btn_tool_monitor = QPushButton(" ")
        self.btn_tool_monitor.setCheckable(True)
        self.btn_tool_monitor.clicked.connect(lambda: self.set_active_tool('monitor'))
        self.btn_tool_monitor.setStyleSheet(style_tool)
        
        tools_row.addWidget(self.btn_tool_inject)
        tools_row.addWidget(self.btn_tool_connect)
        tools_row.addWidget(self.btn_tool_monitor)
        left_layout.addLayout(tools_row)
        
        self.lbl_tool_status = QLabel("Mode: View Only")
        self.lbl_tool_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_tool_status.setStyleSheet("color: #777; font-size: 10px; font-style: italic;")
        left_layout.addWidget(self.lbl_tool_status)
        left_layout.addStretch()
        
        self.toolbar_layout.addWidget(left_panel, 20)

        center_panel = QWidget()
        center_panel.setStyleSheet("background-color: #2a2a2a; border-radius: 4px; border: 1px solid #333;")
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(5, 5, 5, 5)
        
        self.settings_stack = QStackedWidget()
        
        self.settings_stack.addWidget(QLabel("Select a tool...", alignment=Qt.AlignmentFlag.AlignCenter))
        
        self.injector_settings = QWidget()
        self.settings_stack.addWidget(self.injector_settings)

        self.connector_settings = QWidget()
        cl = QGridLayout(self.connector_settings)
        cl.setContentsMargins(2, 2, 2, 2); cl.setSpacing(5)
        
        self.lbl_conn_radius = QLabel(f"Rad: {self.conn_params['radius']:.1f}")
        self.lbl_conn_radius.setStyleSheet("color: #FF9800; font-weight:bold;")
        
        self.slider_conn_radius = QSlider(Qt.Orientation.Horizontal)
        self.slider_conn_radius.setRange(1, 200)
        self.slider_conn_radius.setValue(int(self.conn_params['radius'] * 10))
        
        self.slider_conn_radius.valueChanged.connect(self._update_global_radius)
        
        self.combo_conn_rule = QComboBox(); self.combo_conn_rule.addItems(["all_to_all", "pairwise_bernoulli"])
        self.combo_conn_rule.currentTextChanged.connect(self._on_conn_rule_changed)
        
        cl.addWidget(self.lbl_conn_radius, 0, 0)
        cl.addWidget(self.slider_conn_radius, 0, 1)
        cl.addWidget(self.combo_conn_rule, 0, 2)
        
        self.spin_conn_weight = QDoubleSpinBox(); self.spin_conn_weight.setRange(-1e4, 1e4); self.spin_conn_weight.setValue(50.0); self.spin_conn_weight.setPrefix("W: ")
        self.spin_conn_delay = QDoubleSpinBox(); self.spin_conn_delay.setRange(0.1, 1e3); self.spin_conn_delay.setValue(1.0); self.spin_conn_delay.setPrefix("D: ")
        
        self.conn_rule_params_widget = QWidget()
        self.conn_rule_params_layout = QFormLayout(self.conn_rule_params_widget)
        self.conn_rule_params_layout.setContentsMargins(0,0,0,0)
        self.spin_conn_indegree = QSpinBox(); self.spin_conn_outdegree = QSpinBox(); self.spin_conn_N = QSpinBox(); self.spin_conn_p = QDoubleSpinBox()

        cl.addWidget(self.spin_conn_weight, 1, 0)
        cl.addWidget(self.spin_conn_delay, 1, 1)
        cl.addWidget(self.conn_rule_params_widget, 1, 2)
        
        self.chk_autapses = QCheckBox("Aut"); self.chk_multapses = QCheckBox("Mult"); self.chk_multapses.setChecked(True)
        self.lbl_conn_status = QLabel(""); self.lbl_conn_status.setStyleSheet("color: #4CAF50;")
        
        chk_box = QHBoxLayout(); chk_box.setContentsMargins(0,0,0,0)
        chk_box.addWidget(self.chk_autapses); chk_box.addWidget(self.chk_multapses)
        
        cl.addLayout(chk_box, 2, 0, 1, 2)
        cl.addWidget(self.lbl_conn_status, 2, 2)
        
        self.settings_stack.addWidget(self.connector_settings)
        center_layout.addWidget(self.settings_stack)
        
        self.toolbar_layout.addWidget(center_panel, 30)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_elec = QLabel("ACTIVE ELECTRODES")
        lbl_elec.setStyleSheet("color: #666; font-weight: bold; font-size: 10px; margin-bottom: 2px;")
        right_layout.addWidget(lbl_elec)
        
        self.elec_scroll = QScrollArea()
        self.elec_scroll.setWidgetResizable(True)
        self.elec_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.elec_scroll.setStyleSheet("""
            QScrollArea { background-color: #1a1a1a; border-radius: 4px; border: 1px solid #333; }
            QScrollBar:vertical { width: 8px; background: #1a1a1a; }
            QScrollBar::handle:vertical { background: #444; border-radius: 4px; }
        """)

        self.elec_container = QWidget()
        self.elec_container.setStyleSheet("background-color: transparent;")
        self.elec_grid = QGridLayout(self.elec_container)
        self.elec_grid.setContentsMargins(5, 5, 5, 5)
        self.elec_grid.setSpacing(5)
        self.elec_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        self._electrode_widgets = []

        self.elec_scroll.setWidget(self.elec_container)
        self.elec_scroll.resizeEvent = self._on_elec_scroll_resize
        
        right_layout.addWidget(self.elec_scroll)
        
        btn_clear_all = QPushButton("Clear All Electrodes")
        btn_clear_all.clicked.connect(self.restore_injectors)
        btn_clear_all.setStyleSheet("background: transparent; color: #555; font-size: 9px; text-align: right; padding-right: 5px;")
        right_layout.addWidget(btn_clear_all)

        self.toolbar_layout.addWidget(right_panel, 50)

        layout.addWidget(self.toolbar_container, 15)
        
        self.create_injector_ui(self.injector_settings, self.elec_scroll)

    def update_time_display(self, t): pass
    def update_button_styles(self): pass
    def _update_global_radius(self, value):
        r = value / 10.0
        
        self.stim_params['radius'] = r
        self.conn_params['radius'] = r
        
        if hasattr(self, 'lbl_radius'):
            self.lbl_radius.setText(f"Radius: {r:.1f} mm")
            
        if hasattr(self, 'lbl_conn_radius'):
            self.lbl_conn_radius.setText(f"Rad: {r:.1f}")
            
        
        if hasattr(self, 'slider_radius') and self.slider_radius.value() != value:
            self.slider_radius.blockSignals(True)
            self.slider_radius.setValue(value)
            self.slider_radius.blockSignals(False)
            
        if hasattr(self, 'slider_conn_radius') and self.slider_conn_radius.value() != value:
            self.slider_conn_radius.blockSignals(True)
            self.slider_conn_radius.setValue(value)
            self.slider_conn_radius.blockSignals(False)

    def start_rendering(self):
        if not self.scene_loaded: return
        self.is_paused = False
        if not self.render_timer.isActive():
            self.render_timer.start(30)

    def stop_rendering_safe(self):
        self.is_paused = True
        self.render_timer.stop()

    def update_animation(self):
        if not self.isVisible() or not self.scene_loaded or not self.scatter_item:
            return
            
        try:
            self.anim_rg *= self.decay_tail; self.anim_b *= self.decay_flash; self.heat *= self.decay_heat
            if np.max(self.anim_rg) < 0.05 and np.max(self.anim_b) < 0.05: return
            
            new_c = self.base_colors.copy()
            
            mask = self.anim_rg > 0.05
            if np.any(mask):
                val = self.anim_rg[mask, np.newaxis]
                new_c[mask, 0:3] = np.minimum(1.0, new_c[mask, 0:3] + val)
                new_c[mask, 3] = np.minimum(1.0, 0.4 + val.flatten())
            
            mask_b = self.anim_b > 0.05
            if np.any(mask_b):
                val_b = self.anim_b[mask_b, np.newaxis]
                new_c[mask_b, 2] = np.minimum(1.0, new_c[mask_b, 2] + val_b.flatten())
                new_c[mask_b, 0:2] *= 0.5
                new_c[mask_b, 3] = 1.0
                
            self.scatter_item.setData(color=new_c)
            
        except Exception:
            pass

    def load_scene(self):
        if self.scene_loaded: self.unload_scene()
        points = []; colors = []; gids = []
        
        for graph in self.graph_list:
            for node in graph.node_list:
                if hasattr(node, 'positions') and node.positions:
                    for i, clust in enumerate(node.positions):
                        if clust is None or len(clust) == 0: continue
                        
                        model = node.neuron_models[i] if hasattr(node, 'neuron_models') and i < len(node.neuron_models) else "unknown"
                        rgb = mcolors.to_rgba(neuron_colors.get(model, "#ffffff"), alpha=0.6)
                        
                        points.append(clust)
                        colors.append(np.tile(rgb, (len(clust), 1)))
                        
                        ids_found = False
                        if hasattr(node, 'population') and len(node.population) > i and node.population[i]:
                            try:
                                ids = nest.GetStatus(node.population[i], 'global_id')
                                
                                if len(ids) == len(clust):
                                    gids.append(np.array(ids))
                                    ids_found = True
                                else:
                                    pass
                                    
                            except Exception:
                                pass

                        if not ids_found:
                            gids.append(np.zeros(len(clust), dtype=int))

        if not points: return
        
        self.all_points = np.vstack(points)
        self.base_colors = np.vstack(colors)
        self.current_colors = self.base_colors.copy()
        self.global_ids = np.concatenate(gids)
        self.gid_to_idx = {gid: i for i, gid in enumerate(self.global_ids) if gid > 0}
        
        N = len(self.all_points)
        self.anim_rg = np.zeros(N); self.anim_b = np.zeros(N); self.heat = np.zeros(N)
        
        try:
            self.scatter_item = gl.GLScatterPlotItem(pos=self.all_points, color=self.current_colors, size=self.point_size, pxMode=True)
            self.scatter_item.setGLOptions('translucent')
            
            if self.view.isValid():
                self.view.makeCurrent()
                self.view.addItem(self.scatter_item)
                self.scene_loaded = True
                self.start_rendering()
                print(f"Scene Loaded: {N} neurons.")
            else:
                print("GL View not ready.")
        except Exception as e:
            print(f"OpenGL Init Error: {e}")
            self.scatter_item = None

    def unload_scene(self):
        self.stop_rendering_safe()
        if self.scatter_item:
            try:
                if self.view.isValid():
                    self.view.makeCurrent()
                    self.view.removeItem(self.scatter_item)
            except Exception: pass  # Item may already be removed
        self.scatter_item = None; self.scene_loaded = False

    def feed_spikes(self, gids):
        if not self.scene_loaded or not gids: return
        indices = [self.gid_to_idx[gid] for gid in gids if gid in self.gid_to_idx]
        if indices:
            self.anim_rg[indices] = 1.0;
            self.heat[indices] += 1.0

    def _update_speed_label(self, value): pass
    def _update_radius(self, value):
        r = value / 10.0
        self.stim_params['radius'] = r
        self.lbl_radius.setText(f"Radius: {r:.1f} mm")
    
    def _update_conn_radius(self, value):
        r = value / 10.0
        self.conn_params['radius'] = r
        self.lbl_conn_radius.setText(f"Radius: {r:.1f} mm")
        
    def _update_gen_type_ui(self, index):
        if not hasattr(self, 'combo_gen_type'): return
        
        gen_type = self.combo_gen_type.currentData()
        is_poisson = (gen_type == 'poisson_generator')
        is_spike = (gen_type == 'spike_generator')
        
        if hasattr(self, 'lbl_rate'): self.lbl_rate.setVisible(is_poisson)
        if hasattr(self, 'spin_inj_rate'): self.spin_inj_rate.setVisible(is_poisson)
        
        if hasattr(self, 'spike_gen_container'): 
            self.spike_gen_container.setVisible(is_spike)
            if is_spike:
                self._update_spike_mode_ui(self.combo_spike_mode.currentIndex())
        
    def _on_conn_rule_changed(self, rule):
        while self.conn_rule_params_layout.count():
            item = self.conn_rule_params_layout.takeAt(0)
            if item.widget(): item.widget().setParent(None)
        self.conn_params['rule'] = rule
        if rule == "fixed_indegree": self.conn_rule_params_layout.addRow("Indegree:", self.spin_conn_indegree)
        elif rule == "fixed_outdegree": self.conn_rule_params_layout.addRow("Outdegree:", self.spin_conn_outdegree)
        elif rule == "fixed_total_number": self.conn_rule_params_layout.addRow("Total N:", self.spin_conn_N)
        elif rule == "pairwise_bernoulli": self.conn_rule_params_layout.addRow("Probability:", self.spin_conn_p)
    
    def _get_neurons_in_radius(self, center_idx):
        center_pos = self.all_points[center_idx]
        radius = self.conn_params['radius']
        dists = np.linalg.norm(self.all_points - center_pos, axis=1)
        neighbor_indices = np.where(dists <= radius)[0]
        raw_gids = self.global_ids[neighbor_indices]
        valid_mask = raw_gids > 0
        return neighbor_indices[valid_mask], raw_gids[valid_mask]

    def set_active_tool(self, tool_name):
        if self.active_tool == tool_name:
            self.active_tool = None
            self.btn_tool_inject.setChecked(False)
            self.btn_tool_connect.setChecked(False)
            self.btn_tool_monitor.setChecked(False)
            
            self.settings_stack.setCurrentIndex(0)
            self.lbl_tool_status.setText("Mode: View Only")
        else:
            self.active_tool = tool_name
            self.btn_tool_inject.setChecked(tool_name == 'injector')
            self.btn_tool_connect.setChecked(tool_name == 'connector')
            self.btn_tool_monitor.setChecked(tool_name == 'monitor')
            
            if tool_name == 'injector':
                self.settings_stack.setCurrentIndex(1)
                self.lbl_tool_status.setText("Click in 3D View to ADD Injector")
            elif tool_name == 'connector':
                self.settings_stack.setCurrentIndex(2)
                self.lbl_tool_status.setText("Click Source -> Click Target")
            elif tool_name == 'monitor':
                self.settings_stack.setCurrentIndex(1)
                self.lbl_tool_status.setText("Click to RECORD Spikes")
            
            self.conn_source_gids = []
            self.lbl_conn_status.setText("")

    def project_point_to_screen(self, vec3):
        view_matrix = self.view.viewMatrix()
        w = self.view.width()
        h = self.view.height()
        # Fallunterscheidung für verschiedene PyQtGraph-Versionen
        try:
            proj_matrix = self.view.projectionMatrix()
        except TypeError:
            proj_matrix = self.view.projectionMatrix(region=None, viewport=(0, 0, w, h))
        mvp = proj_matrix * view_matrix
        screen_vec = mvp.map(QVector3D(vec3[0], vec3[1], vec3[2]))
        x = (screen_vec.x() + 1.0) * w / 2.0
        y = (1.0 - screen_vec.y()) * h / 2.0
        return x, y

    def on_mouse_press(self, event):
        if not self.active_tool or not self.scene_loaded or self.all_points is None:
            self._original_mouse_press(event)
            return

        mx, my = event.pos().x(), event.pos().y()
        min_dist = 30.0
        hit_gid = None; hit_idx = None
        
        for i, pt in enumerate(self.all_points):
            sx, sy = self.project_point_to_screen(pt)
            dist = np.sqrt((sx - mx)**2 + (sy - my)**2)
            if dist < min_dist:
                min_dist = dist
                hit_gid = self.global_ids[i]
                hit_idx = i
                break
        
        if hit_gid is not None:
            self.handle_tool_click(hit_gid, hit_idx)
            self.view.mousePos = event.position() if hasattr(event, 'position') else event.pos()
            event.accept()
        else:
            self._original_mouse_press(event)

    def create_injector_at(self, center_pos, target_gids, neighbor_indices):
        try:
            gen_type = self.combo_gen_type.currentData()
            w = self.stim_params['weight']; d = self.stim_params['delay']
            
            if gen_type == 'poisson_generator':
                sg = nest.Create('poisson_generator', params={'rate': 0.0})
            else:
                sg = nest.Create('spike_generator')
            
            targets_sorted = sorted(target_gids.tolist())
            target_nc = nest.NodeCollection(targets_sorted)
            
            # Group targets by model to handle MC models correctly
            model_groups = {}
            for gid in targets_sorted:
                try:
                    model = nest.GetStatus(nest.NodeCollection([gid]), 'model')[0]
                    if model not in model_groups:
                        model_groups[model] = []
                    model_groups[model].append(gid)
                except Exception:
                    if 'unknown' not in model_groups:
                        model_groups['unknown'] = []
                    model_groups['unknown'].append(gid)
            
            # Connect each group with appropriate receptor_type
            total_connected = 0
            for model, gids in model_groups.items():
                syn_spec = {'weight': w, 'delay': d}
                receptor = get_receptor_type_for_model(model, excitatory=(w >= 0))
                if receptor > 0:
                    syn_spec['receptor_type'] = receptor
                
                try:
                    nest.Connect(sg, nest.NodeCollection(gids), conn_spec={'rule': 'all_to_all'}, syn_spec=syn_spec)
                    total_connected += len(gids)
                except Exception as e:
                    print(f"  Skipping {len(gids)} {model} neurons: {e}")
            
            gen_gid = sg.tolist()[0]
            elec_id = self.next_electrode_id; self.next_electrode_id += 1
            letter = InjectorButton.get_next_letter()
            
            item = InjectorButton(elec_id, gen_gid, len(target_gids), letter, gen_type)
            item.triggerClicked.connect(self.trigger_electrode)
            item.removeClicked.connect(self.remove_electrode)
            
            self._add_electrode_widget(item)
            
            self.electrodes[elec_id] = {
                'gid': gen_gid,
                'widget': item,
                'targets': neighbor_indices,
                'center': center_pos.tolist(),
                'gen_type': gen_type,
                'letter': letter
            }
            self.lbl_tool_status.setText(f"Injector {letter} added!")
            print(f"Created {gen_type} {letter} connected to {total_connected}/{len(target_gids)} neurons.")
            
        except Exception as e:
            print(f"Injection err: {e}")
            import traceback
            traceback.print_exc()

    def handle_tool_click(self, center_gid, center_idx):
        print(f"Interaction at GID: {center_gid}")
        
        if self.active_tool == 'injector':
            center_pos = self.all_points[center_idx]
            radius = self.stim_params['radius']
            dists = np.linalg.norm(self.all_points - center_pos, axis=1)
            neighbor_indices = np.where(dists <= radius)[0]
            raw_target_gids = self.global_ids[neighbor_indices]
            if len(raw_target_gids) == 0: return
            self.create_injector_at(center_pos, raw_target_gids, neighbor_indices)

        elif self.active_tool == 'connector':
            neighbor_indices, neighbor_gids = self._get_neurons_in_radius(center_idx)
            if len(neighbor_gids) == 0:
                self.lbl_tool_status.setText("No neurons in radius!")
                return
            
            if len(self.conn_source_gids) == 0:
                self.conn_source_gids = neighbor_gids.tolist()
                self.conn_source_indices = neighbor_indices.tolist()
                
                self.anim_b[neighbor_indices] = 3.0
                
                self.lbl_tool_status.setText(f"Source: {len(self.conn_source_gids)} neurons. Click Target.")
                self.lbl_conn_status.setText(f"✓ {len(self.conn_source_gids)} source neurons selected")
            else:
                target_gids = neighbor_gids.tolist()
                target_indices = neighbor_indices.tolist()
                try:
                    src_sorted = sorted(self.conn_source_gids)
                    tgt_sorted = sorted(target_gids)
                    
                    src_collection = nest.NodeCollection(src_sorted)
                    
                    conn_spec = self._build_conn_spec()
                    base_syn_spec = self._build_syn_spec()
                    weight = base_syn_spec.get('weight', 1.0)
                    
                    # Group targets by model
                    model_groups = {}
                    for gid in tgt_sorted:
                        try:
                            model = nest.GetStatus(nest.NodeCollection([gid]), 'model')[0]
                            if model not in model_groups:
                                model_groups[model] = []
                            model_groups[model].append(gid)
                        except Exception:
                            if 'unknown' not in model_groups:
                                model_groups['unknown'] = []
                            model_groups['unknown'].append(gid)
                    
                    # Connect to each model group with appropriate receptor_type
                    total_connected = 0
                    for model, gids in model_groups.items():
                        syn_spec = base_syn_spec.copy()
                        receptor = get_receptor_type_for_model(model, excitatory=(weight >= 0))
                        if receptor > 0:
                            syn_spec['receptor_type'] = receptor
                        
                        try:
                            nest.Connect(src_collection, nest.NodeCollection(gids), conn_spec, syn_spec)
                            total_connected += len(gids)
                        except Exception as e:
                            print(f"  Skipping {len(gids)} {model} targets: {e}")
                    
                    self.anim_rg[target_indices] = 5.0
                    
                    n_src = len(self.conn_source_gids)
                    n_tgt = len(target_gids)
                    print(f"🔗 Connected {n_src} -> {total_connected}/{n_tgt} neurons")
                    self.lbl_tool_status.setText(f"Connected {n_src}→{total_connected}! Next Source?")
                    self.lbl_conn_status.setText(f"✓ {n_src} → {total_connected}")
                    
                except Exception as e:
                    print(f"Connection Error: {e}")
                    self.lbl_conn_status.setText(f"✗ {e}")
                
                self.conn_source_gids = []
                self.conn_source_indices = []
             
        elif self.active_tool == 'monitor':
             center_pos = self.all_points[center_idx]
             radius = self.stim_params['radius']
             dists = np.linalg.norm(self.all_points - center_pos, axis=1)
             neighbor_indices = np.where(dists <= radius)[0]
             raw_target_gids = self.global_ids[neighbor_indices]
             
             if len(raw_target_gids) == 0: return
             
             try:
                 target_node = None
                 target_pop_idx = 0
                 
                 found = False
                 curr_idx = 0
                 for graph in self.graph_list:
                     for node in graph.node_list:
                         if hasattr(node, 'positions'):
                             for pop_i, pos in enumerate(node.positions):
                                 if pos is None: continue
                                 size = len(pos)
                                 if curr_idx <= center_idx < curr_idx + size:
                                     target_node = node
                                     target_pop_idx = pop_i
                                     found = True
                                     break
                                 curr_idx += size
                         if found: break
                     if found: break
                 
                 if not target_node:
                     print("Error: Could not identify source node for monitor.")
                     return

                 rec = nest.Create("spike_recorder")
                 nest.SetStatus(rec, {"label": "manual_monitor", "record_to": "memory"})
                 
                 targets_sorted = sorted(raw_target_gids.tolist())
                 nest.Connect(nest.NodeCollection(targets_sorted), rec)
                 
                 rec_gid = rec.tolist()[0]
                 
                 if not hasattr(target_node, 'devices'): target_node.devices = []
                 if 'devices' not in target_node.parameters: target_node.parameters['devices'] = []
                 
                 existing = target_node.parameters['devices']
                 next_dev_id = (max([d.get('id', 0) for d in existing]) + 1) if existing else 9900
                 
                 device_record = {
                    "id": next_dev_id,
                    "model": "spike_recorder",
                    "target_pop_id": target_pop_idx,
                    "params": {"label": "manual_monitor"},
                    "conn_params": {"weight": 1.0, "delay": 1.0},
                    "runtime_gid": rec,
                    "is_manual_monitor": True
                 }
                 
                 target_node.parameters['devices'].append(device_record)
                 target_node.devices.append(device_record)
                 
                 elec_id = self.next_electrode_id; self.next_electrode_id += 1
                 letter = f"M{elec_id}"
                 
                 item = InjectorButton(elec_id, rec_gid, len(targets_sorted), letter, "recorder")
                 item.set_active(True)
                 item.removeClicked.connect(self.remove_electrode)
                 
                 self._add_electrode_widget(item)
                 self.electrodes[elec_id] = {'gid': rec_gid, 'widget': item, 'targets': neighbor_indices, 'gen_type': 'recorder', 'letter': letter}
                 
                 self.lbl_tool_status.setText(f"Monitor {letter} added. Check Data Tab!")
                 print(f"Manual Monitor added to Node {target_node.id}. Please Refresh Data Tab.")
                 
             except Exception as e:
                 print(f"Monitor Error: {e}")


    def create_injector_ui(self, settings_widget, grid_area_widget=None):
        if isinstance(settings_widget, QVBoxLayout): il = settings_widget
        else:
            if settings_widget.layout() is None: il = QVBoxLayout(settings_widget)
            else: il = settings_widget.layout()
                
        il.setContentsMargins(5, 5, 5, 5); il.setSpacing(6)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Typ:", styleSheet="color: #aaa;"))
        self.combo_gen_type = QComboBox()
        self.combo_gen_type.addItems(["Spike Generator", "Poisson Generator"])
        self.combo_gen_type.setItemData(0, "spike_generator")
        self.combo_gen_type.setItemData(1, "poisson_generator")
        self.combo_gen_type.setStyleSheet("background-color: #333; color: white; padding: 3px;")
        self.combo_gen_type.currentIndexChanged.connect(self._update_gen_type_ui)
        type_row.addWidget(self.combo_gen_type)
        il.addLayout(type_row)

        self.lbl_radius = QLabel(f"Radius: {self.stim_params['radius']:.1f} mm")
        self.lbl_radius.setStyleSheet("color: #FF9800; font-weight:bold;")
        
        self.slider_radius = QSlider(Qt.Orientation.Horizontal)
        self.slider_radius.setRange(1, 200)
        self.slider_radius.setValue(int(self.stim_params['radius'] * 10))
        
        self.slider_radius.valueChanged.connect(self._update_global_radius)
        
        il.addWidget(self.lbl_radius)
        il.addWidget(self.slider_radius)

        param_grid = QGridLayout()
        
        param_grid.addWidget(QLabel("Weight:"), 0, 0)
        self.spin_inj_weight = QDoubleSpinBox()
        self.spin_inj_weight.setRange(-1e5, 1e5); self.spin_inj_weight.setValue(self.stim_params['weight'])
        self.spin_inj_weight.setSuffix(" pA")
        self.spin_inj_weight.valueChanged.connect(lambda v: self.stim_params.update({'weight': v}))
        param_grid.addWidget(self.spin_inj_weight, 0, 1)

        param_grid.addWidget(QLabel("Delay:"), 1, 0)
        self.spin_inj_delay = QDoubleSpinBox()
        self.spin_inj_delay.setRange(0.1, 1e3); self.spin_inj_delay.setValue(self.stim_params['delay'])
        self.spin_inj_delay.setSuffix(" ms")
        self.spin_inj_delay.valueChanged.connect(lambda v: self.stim_params.update({'delay': v}))
        param_grid.addWidget(self.spin_inj_delay, 1, 1)

        self.lbl_rate = QLabel("Rate:")
        self.spin_inj_rate = QDoubleSpinBox()
        self.spin_inj_rate.setRange(0.1, 1e4); self.spin_inj_rate.setValue(100.0); self.spin_inj_rate.setSuffix(" Hz")
        
        param_grid.addWidget(self.lbl_rate, 2, 0)
        param_grid.addWidget(self.spin_inj_rate, 2, 1)
        
        il.addLayout(param_grid)
        
        self.spike_gen_container = QWidget()
        self.spike_gen_container.setStyleSheet("background-color: #2a2a2a; border-radius: 4px; padding: 5px;")
        spike_layout = QVBoxLayout(self.spike_gen_container)
        spike_layout.setContentsMargins(5, 5, 5, 5)
        spike_layout.setSpacing(4)
        
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:", styleSheet="color: #FF9800; font-weight: bold;"))
        self.combo_spike_mode = QComboBox()
        self.combo_spike_mode.addItems(["Single Shot", "Regular Train", "Burst", "Custom Times"])
        self.combo_spike_mode.setStyleSheet("background-color: #333; color: white;")
        self.combo_spike_mode.currentIndexChanged.connect(self._update_spike_mode_ui)
        mode_row.addWidget(self.combo_spike_mode)
        spike_layout.addLayout(mode_row)
        
        self.spike_train_params = QWidget()
        train_grid = QGridLayout(self.spike_train_params)
        train_grid.setContentsMargins(0, 5, 0, 0)
        train_grid.setSpacing(3)
        
        self.lbl_spike_start = QLabel("Start:")
        self.spin_spike_start = QDoubleSpinBox()
        self.spin_spike_start.setRange(0.0, 1e6)
        self.spin_spike_start.setValue(10.0)
        self.spin_spike_start.setSuffix(" ms")
        self.spin_spike_start.setToolTip("First spike time (relative to current sim time)")
        train_grid.addWidget(self.lbl_spike_start, 0, 0)
        train_grid.addWidget(self.spin_spike_start, 0, 1)
        
        self.lbl_spike_interval = QLabel("Interval:")
        self.spin_spike_interval = QDoubleSpinBox()
        self.spin_spike_interval.setRange(0.1, 1e4)
        self.spin_spike_interval.setValue(10.0)
        self.spin_spike_interval.setSuffix(" ms")
        self.spin_spike_interval.setToolTip("Inter-spike interval")
        train_grid.addWidget(self.lbl_spike_interval, 1, 0)
        train_grid.addWidget(self.spin_spike_interval, 1, 1)
        
        self.lbl_spike_count = QLabel("Count:")
        self.spin_spike_count = QSpinBox()
        self.spin_spike_count.setRange(1, 1000)
        self.spin_spike_count.setValue(10)
        self.spin_spike_count.setToolTip("Number of spikes")
        train_grid.addWidget(self.lbl_spike_count, 2, 0)
        train_grid.addWidget(self.spin_spike_count, 2, 1)
        
        self.lbl_burst_spikes = QLabel("Spikes/Burst:")
        self.spin_burst_spikes = QSpinBox()
        self.spin_burst_spikes.setRange(2, 50)
        self.spin_burst_spikes.setValue(5)
        self.spin_burst_spikes.setToolTip("Number of spikes per burst")
        train_grid.addWidget(self.lbl_burst_spikes, 3, 0)
        train_grid.addWidget(self.spin_burst_spikes, 3, 1)
        
        self.lbl_burst_isi = QLabel("Burst ISI:")
        self.spin_burst_isi = QDoubleSpinBox()
        self.spin_burst_isi.setRange(0.5, 100.0)
        self.spin_burst_isi.setValue(2.0)
        self.spin_burst_isi.setSuffix(" ms")
        self.spin_burst_isi.setToolTip("Inter-spike interval within burst")
        train_grid.addWidget(self.lbl_burst_isi, 4, 0)
        train_grid.addWidget(self.spin_burst_isi, 4, 1)
        
        spike_layout.addWidget(self.spike_train_params)
        
        self.custom_times_container = QWidget()
        custom_layout = QVBoxLayout(self.custom_times_container)
        custom_layout.setContentsMargins(0, 5, 0, 0)
        custom_layout.addWidget(QLabel("Spike Times (comma-separated):", styleSheet="color: #aaa; font-size: 10px;"))
        self.txt_custom_times = QLineEdit()
        self.txt_custom_times.setPlaceholderText("e.g. 10, 20, 30, 50, 100")
        self.txt_custom_times.setStyleSheet("background-color: #333; color: white; padding: 4px;")
        custom_layout.addWidget(self.txt_custom_times)
        spike_layout.addWidget(self.custom_times_container)
        
        self.lbl_spike_preview = QLabel("Preview: --")
        self.lbl_spike_preview.setStyleSheet("color: #888; font-size: 9px; font-style: italic;")
        self.lbl_spike_preview.setWordWrap(True)
        spike_layout.addWidget(self.lbl_spike_preview)
        
        self.spin_spike_start.valueChanged.connect(self._update_spike_preview)
        self.spin_spike_interval.valueChanged.connect(self._update_spike_preview)
        self.spin_spike_count.valueChanged.connect(self._update_spike_preview)
        self.spin_burst_spikes.valueChanged.connect(self._update_spike_preview)
        self.spin_burst_isi.valueChanged.connect(self._update_spike_preview)
        self.txt_custom_times.textChanged.connect(self._update_spike_preview)
        self.combo_spike_mode.currentIndexChanged.connect(self._update_spike_preview)
        
        il.addWidget(self.spike_gen_container)
        il.addStretch()

        self._update_gen_type_ui(0)
        self._update_spike_mode_ui(0)
    
    def _update_spike_mode_ui(self, index):
        mode = self.combo_spike_mode.currentText()
        
        self.lbl_spike_start.setVisible(True)
        self.spin_spike_start.setVisible(True)
        
        is_single = (mode == "Single Shot")
        is_train = (mode == "Regular Train")
        is_burst = (mode == "Burst")
        is_custom = (mode == "Custom Times")
        
        self.lbl_spike_interval.setVisible(is_train or is_burst)
        self.spin_spike_interval.setVisible(is_train or is_burst)
        self.lbl_spike_count.setVisible(is_train or is_burst)
        self.spin_spike_count.setVisible(is_train or is_burst)
        
        self.lbl_burst_spikes.setVisible(is_burst)
        self.spin_burst_spikes.setVisible(is_burst)
        self.lbl_burst_isi.setVisible(is_burst)
        self.spin_burst_isi.setVisible(is_burst)
        
        self.custom_times_container.setVisible(is_custom)
        self.spike_train_params.setVisible(not is_custom)
        
        self.lbl_spike_start.setVisible(not is_custom)
        self.spin_spike_start.setVisible(not is_custom)
        
        self._update_spike_preview()
    
    def _update_spike_preview(self):
        times = self._calculate_spike_times(preview=True)
        if times:
            if len(times) > 8:
                preview = ", ".join(f"{t:.1f}" for t in times[:4]) + f" ... ({len(times)} spikes)"
            else:
                preview = ", ".join(f"{t:.1f}" for t in times)
            self.lbl_spike_preview.setText(f"Preview: {preview} ms")
        else:
            self.lbl_spike_preview.setText("Preview: --")
    
    def _calculate_spike_times(self, preview=False):
        mode = self.combo_spike_mode.currentText()
        
        if preview:
            base_time = 0.0
        else:
            try:
                base_time = nest.GetKernelStatus().get('time', 0.0)
            except Exception:
                base_time = 0.0
        
        start = self.spin_spike_start.value()
        
        if mode == "Single Shot":
            return [base_time + start]
        
        elif mode == "Regular Train":
            interval = self.spin_spike_interval.value()
            count = self.spin_spike_count.value()
            return [base_time + start + i * interval for i in range(count)]
        
        elif mode == "Burst":
            interval = self.spin_spike_interval.value()
            count = self.spin_spike_count.value()
            spikes_per_burst = self.spin_burst_spikes.value()
            burst_isi = self.spin_burst_isi.value()
            
            times = []
            for burst_idx in range(count):
                burst_start = base_time + start + burst_idx * interval
                for spike_idx in range(spikes_per_burst):
                    times.append(burst_start + spike_idx * burst_isi)
            return times
        
        elif mode == "Custom Times":
            try:
                text = self.txt_custom_times.text().strip()
                if not text:
                    return []
                times = [base_time + float(t.strip()) for t in text.split(",") if t.strip()]
                return sorted(times)
            except (ValueError, TypeError): return []
        
        return []

    def trigger_electrode(self, gid):
        try:
            elec_data = None
            for e in self.electrodes.values():
                if e['gid'] == gid: 
                    elec_data = e
                    break
            
            if not elec_data:
                print(f"Electrode with GID {gid} not found")
                return
                
            gen_type = elec_data.get('gen_type', 'spike_generator')
            letter = elec_data.get('letter', '?')
            widget = elec_data.get('widget')
            sg_node = nest.NodeCollection([gid])
            
            if gen_type == 'poisson_generator':
                current_rate = nest.GetStatus(sg_node, 'rate')[0]
                if current_rate > 0:
                    nest.SetStatus(sg_node, {'rate': 0.0})
                    if widget:
                        widget.set_active(False)
                    print(f"Injector {letter}: OFF")
                else:
                    r = self.spin_inj_rate.value()
                    nest.SetStatus(sg_node, {'rate': r})
                    if widget:
                        widget.set_active(True)
                    print(f"Injector {letter}: ON ({r} Hz)")
            else:
                spike_times = self._calculate_spike_times(preview=False)
                
                if not spike_times:
                    print(f"Injector {letter}: No spike times defined!")
                    return
                
                current_time = nest.GetKernelStatus().get('time', 0.0)
                future_times = [t for t in spike_times if t > current_time]
                
                if not future_times:
                    min_time = current_time + 1.0
                    future_times = [min_time + (t - min(spike_times)) for t in spike_times]
                
                nest.SetStatus(sg_node, {'spike_times': future_times})
                
                if widget:
                    widget.set_active(True)
                    duration = min(500, 100 + len(future_times) * 20)
                    QTimer.singleShot(duration, lambda: widget.set_active(False))
                
                mode = self.combo_spike_mode.currentText()
                print(f"⚡ Injector {letter} [{mode}]: {len(future_times)} spikes scheduled")
                print(f"   Times: {future_times[:5]}{'...' if len(future_times) > 5 else ''}")
                
        except Exception as e: 
            print(f"Fire Error: {e}")
            import traceback
            traceback.print_exc()

    def _add_electrode_widget(self, widget):
        self._electrode_widgets.append(widget)
        self._rearrange_electrode_grid()
    
    def _remove_electrode_widget(self, widget):
        if widget in self._electrode_widgets:
            self._electrode_widgets.remove(widget)
        self._rearrange_electrode_grid()
    
    def _rearrange_electrode_grid(self):
        while self.elec_grid.count():
            item = self.elec_grid.takeAt(0)
        
        if not self._electrode_widgets:
            return
        
        container_width = self.elec_scroll.viewport().width() - 20  
        button_width = 65  
        cols = max(1, container_width // button_width)
        
        for i, widget in enumerate(self._electrode_widgets):
            row = i // cols
            col = i % cols
            self.elec_grid.addWidget(widget, row, col)
        
        rows = (len(self._electrode_widgets) + cols - 1) // cols
        min_height = rows * 80 + 10
        self.elec_container.setMinimumHeight(min_height)
    
    def _on_elec_scroll_resize(self, event):
        QScrollArea.resizeEvent(self.elec_scroll, event)
        self._rearrange_electrode_grid()

    def remove_electrode(self, elec_id):
        if elec_id in self.electrodes:
            data = self.electrodes.pop(elec_id)
            widget = data['widget']
            self._remove_electrode_widget(widget)
            widget.deleteLater()

    def restore_injectors(self):
        self.electrodes.clear()
        for widget in self._electrode_widgets:
            widget.deleteLater()
        self._electrode_widgets.clear()
        
        while self.elec_grid.count():
            item = self.elec_grid.takeAt(0)
        
        self.next_electrode_id = 1
        InjectorButton.reset_counter()
        self.conn_source_gids = []
        if hasattr(self, 'lbl_conn_status'): self.lbl_conn_status.setText("")

    def _build_conn_spec(self):
        rule = self.combo_conn_rule.currentText()
        spec = {'rule': rule, 'allow_autapses': self.chk_autapses.isChecked(), 'allow_multapses': self.chk_multapses.isChecked()}
        if rule == "fixed_indegree": spec['indegree'] = self.spin_conn_indegree.value()
        elif rule == "fixed_outdegree": spec['outdegree'] = self.spin_conn_outdegree.value()
        elif rule == "fixed_total_number": spec['N'] = self.spin_conn_N.value()
        elif rule == "pairwise_bernoulli": spec['p'] = self.spin_conn_p.value()
        return spec
    
    def _build_syn_spec(self):
        return {'synapse_model': 'static_synapse', 'weight': self.spin_conn_weight.value(), 'delay': max(self.spin_conn_delay.value(), 0.1)}


class InjectorButton(QWidget):
    triggerClicked = pyqtSignal(int)
    removeClicked = pyqtSignal(int)
    
    _letter_counter = 0
    
    COLOR_OFF = "#1565C0"       
    COLOR_OFF_BORDER = "#64B5F6"
    COLOR_ON = "#FDD835"       
    COLOR_ON_BORDER = "#FBC02D"
    
    @staticmethod
    def get_next_letter():
        idx = InjectorButton._letter_counter
        InjectorButton._letter_counter += 1
        result = ""
        while True:
            result = chr(65 + (idx % 26)) + result
            idx = idx // 26 - 1
            if idx < 0: break
        return result
    
    @staticmethod
    def reset_counter():
        InjectorButton._letter_counter = 0

    def __init__(self, electrode_id, generator_gid, target_count, letter, gen_type, parent=None):
        super().__init__(parent)
        self.electrode_id = electrode_id
        self.generator_gid = generator_gid
        self.letter = letter
        self.gen_type = gen_type
        self.is_active = False  
        self.setFixedSize(60, 75)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        self.btn = QPushButton(self.letter)
        self.btn.setFixedSize(50, 50)
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        
        if gen_type == 'poisson_generator':
            self.btn.setToolTip(f"Poisson Generator (Toggle ON/OFF)\nTargets: {target_count}\nRight-click to remove")
        elif gen_type == 'recorder':
            self.btn.setToolTip(f"Spike Recorder\nTargets: {target_count}\nRight-click to remove")
        else:
            self.btn.setToolTip(f"Spike Generator (Single Shot)\nTargets: {target_count}\nRight-click to remove")
        
        self._apply_style()
        
        self.btn.clicked.connect(self._on_click)
        self.btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.btn.customContextMenuRequested.connect(self._on_right_click)
        
        layout.addWidget(self.btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        lbl = QLabel(f"{target_count}n")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #888; font-size: 9px;")
        layout.addWidget(lbl)
    
    def _apply_style(self):
        if self.is_active:
            bg = self.COLOR_ON
            border = self.COLOR_ON_BORDER
            text_color = "black"
        else:
            bg = self.COLOR_OFF
            border = self.COLOR_OFF_BORDER
            text_color = "white"
        
        self.btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: {text_color};
                font-weight: bold; 
                font-size: 18px;
                border: 2px solid {border};
                border-radius: 8px;
            }}
            QPushButton:hover {{ 
                background-color: {border}; 
                color: black; 
            }}
            QPushButton:pressed {{ 
                background-color: white; 
            }}
        """)
    
    def set_active(self, active):
        self.is_active = active
        self._apply_style()
    
    def toggle(self):
        self.is_active = not self.is_active
        self._apply_style()
        return self.is_active
    
    def _on_click(self):
        self.triggerClicked.emit(self.generator_gid)
    
    def _on_right_click(self, pos):
        self.removeClicked.emit(self.electrode_id)
    
    def sizeHint(self):
        return QSize(60, 75)


class ElectrodeItem(QWidget):
    triggerClicked = pyqtSignal(int)
    removeClicked = pyqtSignal(int)
    paramsChanged = pyqtSignal(int, float, float)

    def __init__(self, electrode_id, generator_gid, target_count, init_weight=1000.0, init_delay=1.0, parent=None):
        super().__init__(parent)
        self.electrode_id = electrode_id
        self.generator_gid = generator_gid
        
        self.setStyleSheet("""
            QFrame#MainFrame {
                background-color: #2b2b2b; 
                border: 1px solid #444; 
                border-radius: 4px;
            }
            QLabel { color: #ddd; font-size: 11px; }
            QDoubleSpinBox {
                background-color: #1e1e1e; color: #00E5FF;
                border: 1px solid #555; border-radius: 2px;
                padding: 1px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        
        self.frame = QFrame()
        self.frame.setObjectName("MainFrame")
        fl = QVBoxLayout(self.frame)
        fl.setContentsMargins(6, 6, 6, 6)
        fl.setSpacing(4)
        
        header = QHBoxLayout()
        lbl_id = QLabel(f"<b style='color:#FF9800'>#{electrode_id}</b> <span style='color:#777'>({target_count} targets)</span>")
        header.addWidget(lbl_id)
        header.addStretch()
        
        btn_del = QPushButton("×")
        btn_del.setFixedSize(16, 16)
        btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_del.setStyleSheet("color: #F44336; border: none; font-weight: bold;")
        btn_del.clicked.connect(lambda: self.removeClicked.emit(self.electrode_id))
        header.addWidget(btn_del)
        fl.addLayout(header)
        
        param_layout = QHBoxLayout()
        
        self.spin_w = QDoubleSpinBox()
        self.spin_w.setRange(-10000, 10000)
        self.spin_w.setValue(init_weight)
        self.spin_w.setDecimals(1)
        self.spin_w.setSuffix(" pA")
        self.spin_w.setToolTip("Synaptic Weight")
        self.spin_w.setFixedWidth(70)
        self.spin_w.valueChanged.connect(self.on_params_changed)
        
        self.spin_d = QDoubleSpinBox()
        self.spin_d.setRange(0.1, 1000)
        self.spin_d.setValue(init_delay)
        self.spin_d.setDecimals(1)
        self.spin_d.setSuffix(" ms")
        self.spin_d.setToolTip("Delay")
        self.spin_d.setFixedWidth(60)
        self.spin_d.valueChanged.connect(self.on_params_changed)
        
        param_layout.addWidget(QLabel("W:"))
        param_layout.addWidget(self.spin_w)
        param_layout.addWidget(QLabel("D:"))
        param_layout.addWidget(self.spin_d)
        fl.addLayout(param_layout)
        
        self.btn_fire = QPushButton("⚡ FIRE (1ms)")
        self.btn_fire.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_fire.setStyleSheet("""
            QPushButton {
                background-color: #E65100; color: white; font-weight: bold;
                border: 1px solid #FF9800; border-radius: 3px; padding: 4px;
            }
            QPushButton:hover { background-color: #FF9800; color: black; }
            QPushButton:pressed { background-color: #FFF; border: none; }
        """)
        self.btn_fire.clicked.connect(lambda: self.triggerClicked.emit(self.generator_gid))
        fl.addWidget(self.btn_fire)
        
        layout.addWidget(self.frame)

    def on_params_changed(self):
        self.paramsChanged.emit(self.electrode_id, self.spin_w.value(), self.spin_d.value())


    


class AnalysisDashboard(QWidget):
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        menu_frame = QFrame()
        menu_frame.setFixedWidth(200)
        menu_frame.setStyleSheet("background-color: #232323; border-right: 1px solid #444;")
        menu_layout = QVBoxLayout(menu_frame)
        menu_layout.setContentsMargins(10, 20, 10, 10)
        menu_layout.setSpacing(10)
        
        lbl_title = QLabel("DATA ANALYSIS")
        lbl_title.setStyleSheet("color: #bbb; font-weight: bold; font-size: 12px; letter-spacing: 1px; border-bottom: 1px solid #444; padding-bottom: 5px;")
        menu_layout.addWidget(lbl_title)
        
        self.btn_graph_info = QPushButton("Graph Info")
        self.btn_time_series = QPushButton("Time Series")
        self.btn_tool_inspector = QPushButton("Tool Inspector")
        self.btn_terminal = QPushButton("Terminal")
        
        self.buttons = [
            self.btn_graph_info,
            self.btn_time_series,
            self.btn_tool_inspector,
            self.btn_terminal
        ]
        
        for i, btn in enumerate(self.buttons):
            btn.setCheckable(True)
            btn.setStyleSheet(self._get_btn_style())
            btn.clicked.connect(lambda checked, idx=i: self.switch_view(idx))
            menu_layout.addWidget(btn)
        
        menu_layout.addStretch()
        
        self.content_stack = QStackedWidget()
        
        self.graph_info = GraphInfoWidget(self.graph_list)
        self.content_stack.addWidget(self.graph_info)
        
        self.time_series = TimeSeriesPlotWidget(self.graph_list)
        self.content_stack.addWidget(self.time_series)
        
        self.tool_inspector = ToolInspectorWidget(self.graph_list)
        self.content_stack.addWidget(self.tool_inspector)

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
        
        self.switch_view(0)

    def refresh_all_tabs(self):
        if hasattr(self, 'graph_info'):
            self.graph_info.refresh_combo()
            
        if hasattr(self, 'time_series'):
            self.time_series.refresh_graphs()
            
        
    def switch_view(self, index):
        self.content_stack.setCurrentIndex(index)
        
        for i, btn in enumerate(self.buttons):
            btn.setChecked(i == index)
            
        if index == 0:
            self.graph_info.refresh_combo()
        elif index == 1:
            self.time_series.refresh_graphs()

    def _get_btn_style(self):
        return """
            QPushButton {
                text-align: left; padding: 10px; border: none;
                background-color: transparent; color: #aaa; font-weight: bold;
            }
            QPushButton:hover { background-color: #2c2c2c; color: white; }
            QPushButton:checked { background-color: #1976D2; color: white; border-left: 4px solid #64B5F6; }
        """


class SimulationDashboardWidget(QWidget):
    requestStartSimulation = pyqtSignal(float)
    requestStopSimulation = pyqtSignal()
    sigDurationChanged = pyqtSignal(float)
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
        
        left_col = QVBoxLayout()
        
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout(config_group)
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 10000000.0)
        self.duration_spin.setValue(1000.0)
        self.duration_spin.setSuffix(" ms")
        self.duration_spin.setStyleSheet("font-size: 14px; padding: 5px; color: #00E5FF; font-weight: bold;")
        self.duration_spin.valueChanged.connect(self.sigDurationChanged.emit)
        self.input_placeholder = QLineEdit("No external input file selected")
        self.input_placeholder.setReadOnly(True)
        self.input_placeholder.setStyleSheet("color: #888; font-style: italic;")
        
        config_layout.addRow("Duration:", self.duration_spin)
        config_layout.addRow("Input:", self.input_placeholder)
        
        left_col.addWidget(config_group)
        
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

        action_container = QWidget()
        action_layout = QHBoxLayout(action_container)
        action_layout.setSpacing(15)
        
        self.btn_start = QPushButton("RUN HEADLESS")
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
        self.btn_export = QPushButton("EXPORT TO PYTHON")
        self.btn_export.setStyleSheet("background-color: #009688; color: white; font-weight: bold; padding: 10px;")
        self.btn_export.clicked.connect(self.export_script_dialog)
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setMinimumHeight(60)
        self.btn_stop.setEnabled(False)
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
        
        self.btn_reset = QPushButton("RESET KERNEL")
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
        self.btn_export = QPushButton("EXPORT")
        self.btn_export.setMinimumHeight(60)
        self.btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background-color: #009688; color: white; font-weight: bold; font-size: 16px; 
                border-radius: 8px; border-bottom: 4px solid #00695C;
            }
            QPushButton:hover { background-color: #26A69A; margin-top: 2px; border-bottom: 2px solid #00695C; }
            QPushButton:pressed { background-color: #00695C; margin-top: 4px; border-bottom: none; }
        """)
        self.btn_export.clicked.connect(self.export_script_dialog)
        # ----------------------------------------

        action_layout.addWidget(self.btn_start, 2)
        action_layout.addWidget(self.btn_stop, 1)
        action_layout.addWidget(self.btn_reset, 1)
        action_layout.addWidget(self.btn_export, 1) 
        main_layout.addWidget(action_container)

        

    def _on_start_clicked(self):
        duration = self.duration_spin.value()
        self.requestStartSimulation.emit(duration)
    def export_script_dialog(self):
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Select Destination Folder for Export"
        )
        
        if not parent_dir:
            return

        default_name = f"Neuroticks_Project_{datetime.now().strftime('%Y%m%d_%H%M')}"
        project_name, ok = QInputDialog.getText(
            self, "Project Name", "Enter name for the export folder:", text=default_name
        )
        
        if ok and project_name:
            safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '_', '-')).strip()
            if not safe_name:
                safe_name = "Unnamed_Project"

            exporter = ScriptExporter(self.graph_list)
            duration = self.duration_spin.value()
            
            success = exporter.export_project(parent_dir, safe_name, duration=duration)
            
            if success:
                full_path = os.path.join(parent_dir, safe_name)
                QMessageBox.information(self, "Export Successful", 
                    f"Project exported successfully to:\n{full_path}\n\n"
                    "Files created:\n"
                    "- run.py\n"
                    "- graphs.json\n"
                    "- neuron_toolbox.py")
            else:
                QMessageBox.critical(self, "Export Failed", 
                    "Could not export the project. Check console for errors.")
    def set_ui_locked(self, locked):
        self.btn_start.setEnabled(not locked)
        self.btn_reset.setEnabled(not locked)
        self.duration_spin.setEnabled(not locked)
        
        self.btn_stop.setEnabled(locked)
        
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


    def update_duration_from_external(self, val):
        self.duration_spin.blockSignals(True)
        self.duration_spin.setValue(val)
        self.duration_spin.blockSignals(False)


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




class ScriptExporter:
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def export_project(self, parent_dir, folder_name, duration=1000.0, threads=1):
        # 1. Create Project Directory
        base_path = os.path.join(parent_dir, folder_name)
        try:
            os.makedirs(base_path, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory: {e}")
            return False

        # 2. Define File Paths
        json_filename = "graphs.json"
        script_filename = "run.py"
        toolbox_filename = "neuron_toolbox.py"
        models_filename = "functional_models.json"
        
        json_path = os.path.join(base_path, json_filename)
        script_path = os.path.join(base_path, script_filename)
        toolbox_dest_path = os.path.join(base_path, toolbox_filename)
        models_dest_path = os.path.join(base_path, models_filename)

        # 3. Serialize and Write JSON
        project_data = self._serialize_graphs()
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                # Local import to ensure encoder is available
                from WidgetLib import NumpyEncoder 
                json.dump(project_data, f, cls=NumpyEncoder, indent=2)
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False

        # 4. Copy Auxiliary Files (Toolbox & Models)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Copy neuron_toolbox.py
        try:
            src_path = "neuron_toolbox.py"
            if not os.path.exists(src_path):
                src_path = os.path.join(current_dir, "neuron_toolbox.py")

            if os.path.exists(src_path):
                shutil.copy(src_path, toolbox_dest_path)
            else:
                print(f"Warning: neuron_toolbox.py not found.")
        except Exception as e:
            print(f"Error copying toolbox: {e}")

        # Copy functional_models.json
        try:
            mod_path = "functional_models.json"
            if not os.path.exists(mod_path):
                mod_path = os.path.join(current_dir, "functional_models.json")
            
            if os.path.exists(mod_path):
                shutil.copy(mod_path, models_dest_path)
            else:
                print("Warning: functional_models.json not found.")
        except Exception as e:
            print(f"Error copying models json: {e}")

        # 5. Generate and Write Python Script
        script_content = self._get_template(json_filename, duration, threads)
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
        except Exception as e:
            print(f"Error saving script: {e}")
            return False

        print(f"Export successful to: {base_path}")
        return True

    def _serialize_graphs(self):
        from WidgetLib import _clean_params, _serialize_connections
        
        data = {'graphs': []}
        for graph in self.graph_list:
            nodes_data = []
            for node in graph.node_list:
                positions = [pos.tolist() if isinstance(pos, np.ndarray) else list(pos) 
                             for pos in node.positions] if hasattr(node, 'positions') and node.positions else []
                
                devices = []
                source_devs = getattr(node, 'devices', [])
                if not source_devs and 'devices' in node.parameters:
                    source_devs = node.parameters['devices']

                for d in source_devs:
                    d_copy = d.copy()
                    if 'runtime_gid' in d_copy: del d_copy['runtime_gid']
                    if 'params' in d_copy: d_copy['params'] = _clean_params(d_copy['params'])
                    devices.append(d_copy)

                safe_params = _clean_params(node.parameters) if hasattr(node, 'parameters') else {}
                
                node_data = {
                    'id': node.id,
                    'name': getattr(node, 'name', f"Node_{node.id}"),
                    'graph_id': graph.graph_id,
                    'neuron_models': node.neuron_models,
                    'positions': positions,
                    'devices': devices,
                    'connections': _serialize_connections(node.connections) if hasattr(node, 'connections') else [],
                    'population_nest_params': [
                        _clean_params(p) if p else {} 
                        for p in getattr(node, 'population_nest_params', [])
                    ],
                    'tool_type': safe_params.get('tool_type', 'custom'),
                    'parameters': safe_params
                }
                
                if hasattr(node, 'center_of_mass'):
                    node_data['center_of_mass'] = list(node.center_of_mass) if isinstance(node.center_of_mass, np.ndarray) else node.center_of_mass
                if hasattr(node, 'types'):
                    node_data['types'] = list(node.types)

                nodes_data.append(node_data)
            
            data['graphs'].append({
                'graph_id': graph.graph_id,
                'graph_name': getattr(graph, 'graph_name', f'Graph_{graph.graph_id}'),
                'max_nodes': graph.max_nodes,
                'init_position': list(graph.init_position) if hasattr(graph, 'init_position') else [0,0,0],
                'polynom_max_power': getattr(graph, 'polynom_max_power', 5),
                'polynom_decay': getattr(graph, 'polynom_decay', 0.8),
                'nodes': nodes_data
            })
        return data
    def _get_template(self, json_filename, duration, threads):
        return f"""
import nest
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from neuron_toolbox import Graph, Node, PolynomGenerator

# === CONFIGURATION ===
JSON_FILENAME = "{json_filename}"
SIM_DURATION = {duration}
DUMP_INTERVAL = 2000.0
OUTPUT_DIR = Path("Headless_Runs")

class NumpyNestEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'): return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.integer)): return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def load_and_build():
    print("Initializing NEST...")
    nest.ResetKernel()
    nest.SetKernelStatus({{
        'resolution': 0.1, 
        'print_time': False, 
        'local_num_threads': {threads}  
    }})
    nest.EnableStructuralPlasticity()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, JSON_FILENAME)
    
    if not os.path.exists(json_path):
        print(f"CRITICAL ERROR: JSON file not found: {{json_path}}")
        sys.exit(1)

    with open(json_path, 'r') as f:
        data = json.load(f)

    graphs = {{}}
    print(f"Loading {{len(data.get('graphs', []))}} graphs...")
    
    for g_data in data.get('graphs', []):
        g_id = g_data['graph_id']
        graph = Graph(
            graph_name=g_data.get('graph_name', f'Graph_{{g_id}}'),
            graph_id=g_id,
            max_nodes=g_data.get('max_nodes', 100),
            position=g_data.get('init_position', [0, 0, 0]),
            polynom_max_power=g_data.get('polynom_max_power', 5),
            polynom_decay=g_data.get('polynom_decay', 0.8)
        )

        sorted_nodes = sorted(g_data.get('nodes', []), key=lambda x: x.get('id', 0))
        for nd in sorted_nodes:
            params = nd.get('parameters', {{}}).copy()
            devs = nd.get('devices', [])
            
            # === FIX: population_nest_params laden ===
            pop_nest_params = nd.get('population_nest_params', [])
            
            params.update({{
                'id': nd['id'],
                'name': nd['name'],
                'graph_id': g_id,
                'connections': nd.get('connections', []),
                'devices': devs, 
                'neuron_models': nd.get('neuron_models', ['iaf_psc_alpha']),
                'types': nd.get('types', [0]),
                'population_nest_params': pop_nest_params  # FIX: NEST-Parameter übergeben
            }})
            
            if 'positions' in nd and nd['positions']:
                params['positions'] = [np.array(p) for p in nd['positions']]

            node = graph.create_node(parameters=params, is_root=(nd['id']==0), auto_build=False)
            
            if 'center_of_mass' in nd: node.center_of_mass = np.array(nd['center_of_mass'])
            if 'positions' in params: node.positions = params['positions']
            
            node.connections = nd.get('connections', [])
            node.devices = devs
            
            # === FIX: population_nest_params auch direkt am Node setzen ===
            node.population_nest_params = pop_nest_params
            
            node.populate_node()
        
        graphs[g_id] = graph

    all_graphs_map = {{g.graph_id: g for g in graphs.values()}}
    
    print("Building Connections...")
    for graph in graphs.values():
        graph.build_connections(external_graphs=all_graphs_map)

    return graphs

def cache_active_recorders(graphs):
    \"\"\"
    Erstellt eine flache Liste aller aktiven Recorder.
    Wrappt IDs direkt in NodeCollection für NEST 3.x Kompatibilität.
    \"\"\"
    active_recorders = []
    
    for graph in graphs.values():
        for node in graph.node_list:
            if not hasattr(node, 'devices'): continue
            
            for dev in node.devices:
                gid_col = dev.get('runtime_gid')
                if gid_col is None: continue
                
                # 1. GID extrahieren
                try:
                    if hasattr(gid_col, 'tolist'): gid_int = int(gid_col.tolist()[0])
                    elif isinstance(gid_col, list): gid_int = int(gid_col[0])
                    else: gid_int = int(gid_col)
                except (ValueError, TypeError, IndexError):
                    continue

                model = dev.get('model', '')
                if 'recorder' in model or 'meter' in model:
                    try:
                        # 2. FIX: NodeCollection erstellen und cachen
                        node_handle = nest.NodeCollection([gid_int])
                        
                        active_recorders.append({{
                            'handle': node_handle,
                            'graph_id': graph.graph_id,
                            'node_id': node.id,
                            'device_id': str(dev['id']),
                            'model': model
                        }})
                    except Exception as e:
                        print(f"Error caching device {{gid_int}}: {{e}}")
    
    print(f"Cached {{len(active_recorders)}} active recording devices.")
    
    if len(active_recorders) == 0:
        print("⚠ WARNING: No spike_recorders or multimeters found!")
        print("  → Simulation will run but NO DATA will be collected.")
        print("  → Add devices with 'spike_recorder' or 'multimeter' model to record data.")
    
    return active_recorders

def collect_data_optimized(active_recorders):
    \"\"\"
    Iteriert nur über die vorgefertigte Liste von Recordern.
    \"\"\"
    snapshot = {{}}
    total_events = 0
    
    for rec in active_recorders:
        try:
            # Daten holen (Handle ist schon NodeCollection)
            status = nest.GetStatus(rec['handle'])[0]
            n_events = status.get('n_events', 0)
            
            if n_events > 0:
                events = status.get('events', {{}})
                
                # Struktur initialisieren falls nötig
                gid, nid = rec['graph_id'], rec['node_id']
                if gid not in snapshot: snapshot[gid] = {{}}
                if nid not in snapshot[gid]: snapshot[gid][nid] = {{}}
                
                # Numpy Arrays zu Listen konvertieren
                clean_events = {{k: v.tolist() if isinstance(v, np.ndarray) else list(v) 
                              for k, v in events.items()}}
                
                snapshot[gid][nid][rec['device_id']] = {{
                    'model': rec['model'],
                    'events': clean_events,
                    'n_events': n_events
                }}
                
                total_events += n_events
                
                # Speicher leeren
                nest.SetStatus(rec['handle'], {{'n_events': 0}})
                
        except Exception as e:
            print(f"Read error on device {{rec['device_id']}}: {{e}}")

    return snapshot

def main():
    run_name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Simulation START. Output: {{run_dir}}")
    
    # 1. Aufbauen
    graphs = load_and_build()
    
    # 2. Cachen (Optimierung)
    recorders = cache_active_recorders(graphs)
    
    current_time = 0.0
    dump_count = 0
    
    print(f"Simulating {{SIM_DURATION}} ms...")
    
    while current_time < SIM_DURATION:
        step = min(DUMP_INTERVAL, SIM_DURATION - current_time)
        nest.Simulate(step)
        current_time += step
        
        # 3. Optimiertes Sammeln
        data = collect_data_optimized(recorders)
        
        filename = f"step_{{dump_count:04d}}_{{int(current_time)}}ms.json"
        filepath = run_dir / filename
        
        # Auch leere Files speichern um den Timestamp zu garantieren.
        with open(filepath, 'w') as f:
            dump_obj = {{
                'time': current_time,
                'graphs': data
            }}
            json.dump(dump_obj, f, cls=NumpyNestEncoder)
        
        if data:
            print(f"  Step {{dump_count}}: {{current_time}}ms -> Dumped active data")
        else:
            print(f"  Step {{dump_count}}: {{current_time}}ms -> (no spikes)")
            
        dump_count += 1

    print("Simulation FINISHED.")

if __name__ == "__main__":
    main()
"""




class GraphInfoWidget(QWidget):
    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        sel_layout = QHBoxLayout()
        lbl = QLabel("Select Graph:")
        lbl.setStyleSheet("color: white;")
        sel_layout.addWidget(lbl)
        
        self.combo = QComboBox()
        self.combo.setStyleSheet("background-color: #333; color: white; border: 1px solid #555;")
        self.combo.currentIndexChanged.connect(self.generate_report)
        sel_layout.addWidget(self.combo)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("background-color: #1e1e1e; border: none;")
        
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background-color: #1e1e1e;")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(15)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
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

    def _create_section(self, title, color="#FF9800"):
        frame = QFrame()
        frame.setStyleSheet(f"background-color: #2b2b2b; border: 1px solid #444; border-radius: 5px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 10, 15, 15)
        layout.setSpacing(8)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px; border: none; padding-bottom: 5px;")
        layout.addWidget(lbl_title)
        
        return frame, layout

    def _add_info_row(self, layout, label, value, value_color="#00E5FF"):
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setStyleSheet("color: #888; font-size: 12px; border: none;")
        lbl.setFixedWidth(180)
        
        val = QLabel(str(value))
        val.setStyleSheet(f"color: {value_color}; font-size: 12px; font-weight: bold; border: none;")
        val.setWordWrap(True)
        
        row.addWidget(lbl)
        row.addWidget(val, 1)
        layout.addLayout(row)

    def generate_report(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget(): 
                item.widget().deleteLater()
        
        graph = self.combo.currentData()
        if not graph: 
            return

        total_nodes = len(graph.node_list)
        total_conns = sum(len(n.connections) for n in graph.node_list if hasattr(n, 'connections'))
        
        gen_frame, gen_layout = self._create_section("GENERAL INFORMATION", "#00E5FF")
        self._add_info_row(gen_layout, "Graph Name", getattr(graph, 'graph_name', 'Unknown'))
        self._add_info_row(gen_layout, "Graph ID", graph.graph_id)
        self._add_info_row(gen_layout, "Total Nodes", total_nodes)
        self._add_info_row(gen_layout, "Total Connections", total_conns)
        self.content_layout.addWidget(gen_frame)
        
        pop_frame, pop_layout = self._create_section("POPULATION DISTRIBUTION", "#E91E63")
        
        model_counts = {}
        total_neurons = 0
        for node in graph.node_list:
            if hasattr(node, 'neuron_models') and node.neuron_models:
                for model in node.neuron_models:
                    model_counts[model] = model_counts.get(model, 0) + 1
            if hasattr(node, 'positions') and node.positions:
                for pos_list in node.positions:
                    if pos_list is not None:
                        total_neurons += len(pos_list)
        
        self._add_info_row(pop_layout, "Total Neuron Count", total_neurons, "#4CAF50")
        self._add_info_row(pop_layout, "Unique Models", len(model_counts))
        
        if model_counts:
            lbl_dist = QLabel("Model Distribution:")
            lbl_dist.setStyleSheet("color: #888; font-size: 12px; border: none; margin-top: 5px;")
            pop_layout.addWidget(lbl_dist)
            
            for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
                row = QHBoxLayout()
                model_lbl = QLabel(f"  • {model}")
                model_lbl.setStyleSheet("color: #ccc; font-size: 11px; border: none;")
                count_lbl = QLabel(f"{count} population(s)")
                count_lbl.setStyleSheet("color: #FF9800; font-size: 11px; border: none;")
                row.addWidget(model_lbl)
                row.addStretch()
                row.addWidget(count_lbl)
                pop_layout.addLayout(row)
        
        self.content_layout.addWidget(pop_frame)
        
        dev_frame, dev_layout = self._create_section("RECORDING DEVICES", "#9C27B0")
        
        device_counts = {}
        total_devices = 0
        for node in graph.node_list:
            if hasattr(node, 'devices') and node.devices:
                for dev in node.devices:
                    model = dev.get('model', 'unknown')
                    device_counts[model] = device_counts.get(model, 0) + 1
                    total_devices += 1
        
        self._add_info_row(dev_layout, "Total Devices", total_devices, "#4CAF50")
        
        if device_counts:
            for dev_type, count in sorted(device_counts.items(), key=lambda x: -x[1]):
                row = QHBoxLayout()
                dev_lbl = QLabel(f"  • {dev_type}")
                dev_lbl.setStyleSheet("color: #ccc; font-size: 11px; border: none;")
                count_lbl = QLabel(f"{count}x")
                count_lbl.setStyleSheet("color: #FF9800; font-size: 11px; border: none;")
                row.addWidget(dev_lbl)
                row.addStretch()
                row.addWidget(count_lbl)
                dev_layout.addLayout(row)
        else:
            no_dev = QLabel("  No devices found")
            no_dev.setStyleSheet("color: #666; font-style: italic; font-size: 11px; border: none;")
            dev_layout.addWidget(no_dev)
        
        self.content_layout.addWidget(dev_frame)
        
        if total_nodes > 0:
            node_frame, node_layout = self._create_section("NODE DETAILS", "#FF9800")
            
            for node in graph.node_list:
                node_box = QFrame()
                node_box.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; border-radius: 3px; margin: 2px;")
                nb_layout = QVBoxLayout(node_box)
                nb_layout.setContentsMargins(10, 8, 10, 8)
                nb_layout.setSpacing(4)
                
                header = QLabel(f"Node {node.id}: {node.name}")
                header.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 12px; border: none;")
                nb_layout.addWidget(header)
                
                details = []
                
                if hasattr(node, 'center_of_mass'):
                    com = node.center_of_mass
                    details.append(f"Position: [{com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}]")
                
                pop_count = 0
                neuron_count = 0
                if hasattr(node, 'positions') and node.positions:
                    pop_count = len([p for p in node.positions if p is not None])
                    neuron_count = sum(len(p) for p in node.positions if p is not None)
                details.append(f"Populations: {pop_count}  |  Neurons: {neuron_count}")
                
                if hasattr(node, 'neuron_models') and node.neuron_models:
                    models_str = ", ".join(node.neuron_models[:3])
                    if len(node.neuron_models) > 3:
                        models_str += f" (+{len(node.neuron_models)-3} more)"
                    details.append(f"Models: {models_str}")
                
                if hasattr(node, 'connections'):
                    details.append(f"Connections: {len(node.connections)}")
                
                dev_count = len(node.devices) if hasattr(node, 'devices') else 0
                if dev_count > 0:
                    details.append(f"Devices: {dev_count}")
                
                for detail in details:
                    d_lbl = QLabel(detail)
                    d_lbl.setStyleSheet("color: #888; font-size: 11px; border: none; padding-left: 10px;")
                    nb_layout.addWidget(d_lbl)
                
                node_layout.addWidget(node_box)
            
            self.content_layout.addWidget(node_frame)
        
        hist_frame, hist_layout = self._create_section("SIMULATION HISTORY", "#4CAF50")
        
        total_runs = 0
        total_recordings = 0
        for node in graph.node_list:
            if hasattr(node, 'results') and 'history' in node.results:
                for run in node.results['history']:
                    total_runs += 1
                    if 'devices' in run:
                        total_recordings += len(run['devices'])
        
        self._add_info_row(hist_layout, "Simulation Runs", total_runs)
        self._add_info_row(hist_layout, "Total Recordings", total_recordings)
        
        self.content_layout.addWidget(hist_frame)
        
        self.content_layout.addStretch()


class TimeSeriesPlotWidget(QWidget):
    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list
        self.current_data = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444; border-radius: 4px;")
        ctrl_layout = QHBoxLayout(ctrl_frame)
        
        style_combo = "background-color: #1e1e1e; color: white; border: 1px solid #555; padding: 4px;"
        style_lbl = "color: #ddd; font-weight: bold;"
        
        self.combo_graph = QComboBox(); self.combo_graph.setStyleSheet(style_combo)
        self.combo_node = QComboBox(); self.combo_node.setStyleSheet(style_combo)
        self.combo_device = QComboBox(); self.combo_device.setStyleSheet(style_combo)
        
        self.combo_analysis = QComboBox()
        self.combo_analysis.setStyleSheet("background-color: #004d40; color: white; border: 1px solid #00695c; padding: 4px; font-weight: bold;")
        
        self.btn_plot = QPushButton("ANALYZE")
        self.btn_plot.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_plot.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; border-radius: 3px; padding: 6px 20px;")
        
        ctrl_layout.addWidget(QLabel("Graph:", styleSheet=style_lbl))
        ctrl_layout.addWidget(self.combo_graph)
        ctrl_layout.addWidget(QLabel("Node:", styleSheet=style_lbl))
        ctrl_layout.addWidget(self.combo_node)
        ctrl_layout.addWidget(QLabel("Device:", styleSheet=style_lbl))
        ctrl_layout.addWidget(self.combo_device, 1)
        ctrl_layout.addWidget(QLabel("Tool:", styleSheet=style_lbl))
        ctrl_layout.addWidget(self.combo_analysis, 1)
        ctrl_layout.addWidget(self.btn_plot)
        
        layout.addWidget(ctrl_frame)
        
        self.stack = QStackedWidget()
        
        self.pg_view = pg.GraphicsLayoutWidget()
        self.pg_view.setBackground('w')
        self.stack.addWidget(self.pg_view)
        
        self.gl_view = SafeGLViewWidget()
        self.gl_view.opts['distance'] = 200
        self.gl_view.setBackgroundColor('k')
        self.stack.addWidget(self.gl_view)
        
        self.fig = Figure(figsize=(5, 4), facecolor='white')
        self.mpl_canvas = FigureCanvas(self.fig)
        self.stack.addWidget(self.mpl_canvas)
        
        layout.addWidget(self.stack)
        
        self.combo_graph.currentIndexChanged.connect(self.on_graph_changed)
        self.combo_node.currentIndexChanged.connect(self.on_node_changed)
        self.combo_device.currentIndexChanged.connect(self.on_device_changed)
        self.btn_plot.clicked.connect(self.run_analysis)
        
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
        self.combo_device.blockSignals(True)
        self.combo_device.clear()
        node = self.combo_node.currentData()
        if not node: return
        
        known_devices = {}
        
        if hasattr(node, 'devices'):
            for d in node.devices:
                known_devices[d.get('id')] = d.get('model', 'unknown')
                
        if hasattr(node, 'results') and 'history' in node.results:
            for entry in node.results['history']:
                if 'devices' in entry:
                    for did_str, info in entry['devices'].items():
                        try:
                            did = int(did_str)
                            if did not in known_devices:
                                known_devices[did] = info.get('model', 'unknown')
                        except (ValueError, TypeError): pass
        
        def safe_sort_key(k):
            try:
                return (0, int(k))
            except (ValueError, TypeError):
                return (1, str(k))

        for did in sorted(list(known_devices.keys()), key=safe_sort_key):
            model = known_devices[did]
            if "recorder" in model or "meter" in model:
                self.combo_device.addItem(f"#{did}: {model}", (did, model))
                
        self.combo_device.blockSignals(False)
        self.on_device_changed()

    def on_device_changed(self):
        self.combo_analysis.clear()
        data = self.combo_device.currentData()
        if not data: return
        
        did, model = data
        
        if "spike_recorder" in model:
            self.combo_analysis.addItems([
                "Raster Plot (2D)",
                "Population Activity (Histogram)",
                "Firing Rate (Line)",
                "ISI Distribution (Histogram)",
                "Spike Matrix (Heatmap)"
            ])
        elif "meter" in model or "multimeter" in model:
            self.combo_analysis.addItems([
                "Membrane Potential (Traces)",
                "V_m Heatmap"
            ])

    def _fetch_data(self):
        node = self.combo_node.currentData()
        dev_data_raw = self.combo_device.currentData()
        if not node or not dev_data_raw: return None
        
        dev_id, model = dev_data_raw
        
        if not hasattr(node, 'results') or 'history' not in node.results:
            return None
            
        merged = {'times': [], 'senders': [], 'V_m': [], 'model': model, 'dev_id': dev_id}
        
        for run in node.results['history']:
            if 'devices' not in run: continue
            
            dev_entry = run['devices'].get(dev_id) or run['devices'].get(str(dev_id))
            if not dev_entry or 'data' not in dev_entry: continue
            
            d = dev_entry['data']
            
            if 'times' in d: merged['times'].extend(d['times'])
            if 'senders' in d: merged['senders'].extend(d['senders'])
            if 'V_m' in d: merged['V_m'].extend(d['V_m'])
            
        if merged['times']: merged['times'] = np.array(merged['times'])
        if merged['senders']: merged['senders'] = np.array(merged['senders'])
        if merged['V_m']: merged['V_m'] = np.array(merged['V_m'])
        
        return merged

    def run_analysis(self):
        data = self._fetch_data()
        if not data or len(data['times']) == 0:
            print("No data found for this device.")
            return

        mode = self.combo_analysis.currentText()
        
        self.pg_view.clear()
        self.gl_view.items = []
        self.fig.clear()
        
        if "3D" in mode:
            self.stack.setCurrentIndex(1)
            self._plot_3d(data, mode)
        elif "Matplotlib" in mode:
            self.stack.setCurrentIndex(2)
            self._plot_mpl(data, mode)
        else:
            self.stack.setCurrentIndex(0)
            self._plot_2d(data, mode)

    
    def _plot_2d(self, data, mode):
        times = data['times']
        
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        
        if mode == "Raster Plot (2D)":
            senders = data['senders'][sort_idx]
            p = self.pg_view.addPlot(title=f"Spike Raster (#{data['dev_id']})")
            scatter = pg.ScatterPlotItem(x=times, y=senders, size=3, pen=None, brush='k')
            p.addItem(scatter)
            p.setLabel('bottom', 'Time', 'ms')
            p.setLabel('left', 'Neuron ID')

        elif mode == "Population Activity (Histogram)":
            p = self.pg_view.addPlot(title="Population Activity (PSTH)")
            y, x = np.histogram(times, bins=100)
            curve = pg.BarGraphItem(x=x[:-1], height=y, width=(x[1]-x[0]), brush='b')
            p.addItem(curve)
            p.setLabel('bottom', 'Time', 'ms')
            p.setLabel('left', 'Spike Count')

        elif mode == "Firing Rate (Line)":
            bin_size = 10.0
            bins = np.arange(times.min(), times.max() + bin_size, bin_size)
            counts, _ = np.histogram(times, bins=bins)
            
            unique_senders = len(np.unique(data['senders'])) if len(data['senders']) > 0 else 1
            rate = counts / (bin_size / 1000.0) / unique_senders
            
            p = self.pg_view.addPlot(title="Population Firing Rate")
            p.plot(bins[:-1], rate, pen=pg.mkPen('r', width=2))
            p.setLabel('bottom', 'Time', 'ms')
            p.setLabel('left', 'Rate', 'Hz')

        elif mode == "ISI Distribution (Histogram)":
            senders = data['senders']
            isis = []
            for nid in np.unique(senders):
                st = np.sort(times[senders == nid])
                if len(st) > 1:
                    isis.extend(np.diff(st))
            
            if isis:
                p = self.pg_view.addPlot(title="Inter-Spike Intervals (ISI)")
                y, x = np.histogram(isis, bins=50)
                bg = pg.BarGraphItem(x=x[:-1], height=y, width=(x[1]-x[0]), brush='g')
                p.addItem(bg)
                p.setLabel('bottom', 'ISI', 'ms')
            else:
                self.pg_view.addLabel("Not enough spikes for ISI")

        elif mode == "Membrane Potential (Traces)":
            vals = data['V_m'][sort_idx]
            
            senders = data.get('senders')
            if senders is not None:
                senders = senders[sort_idx]
                unique_ids = np.unique(senders)[:10]
                
                p = self.pg_view.addPlot(title="Membrane Potential (First 10 Neurons)")
                p.addLegend()
                
                colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
                for i, uid in enumerate(unique_ids):
                    mask = senders == uid
                    t_u = times[mask]
                    v_u = vals[mask]
                    p.plot(t_u, v_u, pen=pg.mkPen(colors[i%len(colors)], width=1.5), name=f"N{uid}")
                
                p.setLabel('bottom', 'Time', 'ms')
                p.setLabel('left', 'Voltage', 'mV')

        elif "Heatmap" in mode:
            self._plot_heatmap_2d(data, mode)

    def _plot_heatmap_2d(self, data, mode):
        times = data['times']
        
        if "Spike" in mode:
            senders = data['senders']
            t_bins = 50
            n_bins = max(10, len(np.unique(senders)))
            
            hist, x_edges, y_edges = np.histogram2d(times, senders, bins=[t_bins, n_bins])
            
            p = self.pg_view.addPlot(title="Spike Density Heatmap")
            img = pg.ImageItem(hist)
            img.setLookupTable(pg.colormap.get('inferno').getLookupTable())
            p.addItem(img)
            p.setLabel('bottom', 'Time Bin')
            p.setLabel('left', 'Neuron Bin')
            
            bar = pg.ColorBarItem(values=(0, np.max(hist)), colorMap='inferno')
            bar.setImageItem(img)
            
        elif "V_m" in mode:
            senders = data['senders']
            vals = data['V_m']
            
            u_ids = np.unique(senders)
            u_ids.sort()
            
            common_time = np.linspace(times.min(), times.max(), 200)
            matrix = np.zeros((len(u_ids), len(common_time)))
            
            for i, uid in enumerate(u_ids):
                mask = senders == uid
                t_n = times[mask]
                v_n = vals[mask]
                s = np.argsort(t_n)
                if len(s) > 1:
                    matrix[i, :] = np.interp(common_time, t_n[s], v_n[s])
            
            p = self.pg_view.addPlot(title="Membrane Potential Heatmap")
            img = pg.ImageItem(matrix)
            img.setLookupTable(pg.colormap.get('viridis').getLookupTable())
            p.addItem(img)
            p.setLabel('bottom', 'Time')
            p.setLabel('left', 'Neuron Index')


    def _plot_3d(self, data, mode):
        times = data['times']
        
        self.gl_view.setCameraPosition(distance=100, elevation=30, azimuth=45)
        
        g = gl.GLGridItem()
        g.scale(10, 10, 1)
        self.gl_view.addItem(g)
        
        if mode == "3D Spike Cloud":
            senders = data['senders']
            x = (times - times.min()) / (np.ptp(times) + 1e-6) * 100 - 50
            y = (senders - senders.min()) / (np.ptp(senders) + 1e-6) * 100 - 50
            z = np.zeros_like(x)
            
            pos = np.column_stack((x, y, z))
            
            colors = np.zeros((len(pos), 4))
            colors[:, 0] = (x + 50) / 100.0
            colors[:, 2] = 1.0 - ((x + 50) / 100.0)
            colors[:, 1] = 0.5
            colors[:, 3] = 0.8
            
            sp = gl.GLScatterPlotItem(pos=pos, color=colors, size=5, pxMode=True)
            self.gl_view.addItem(sp)
            
        elif mode == "3D Activity Surface" or "3D V_m" in mode:
            senders = data['senders']
            vals = data['V_m'] if "V_m" in mode else np.ones_like(times)
            
            t_bins = 50
            n_bins = 50
            
            if "V_m" in mode:
                 pass
                 self._plot_3d(data, "3D Spike Cloud")
                 return
            else:
                hist, _, _ = np.histogram2d(times, senders, bins=[t_bins, n_bins])
                
                z = hist * 5.0
                
                surface = gl.GLSurfacePlotItem(z=z, shader='heightColor', computeNormals=False, smooth=False)
                surface.shader()['colorMap'] = np.array([0,0,1,1, 0,1,0,1, 1,1,0,1, 1,0,0,1])
                surface.translate(-t_bins/2, -n_bins/2, 0)
                surface.scale(2, 2, 1)
                self.gl_view.addItem(surface)


class PythonConsoleWidget(QWidget):
    def __init__(self, context_vars=None, parent=None):
        super().__init__(parent)
        self.local_vars = context_vars if context_vars is not None else {}
        self.history = []
        self.history_idx = 0
        
        self.init_ui()
        
        self.interpreter = code.InteractiveConsole(self.local_vars)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
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
        self.input_line.installEventFilter(self)

        layout.addWidget(self.output_view)
        layout.addWidget(self.input_line)
        
        self.write_output("Python Console initialized.\nContext available: 'nest', 'graph_list', 'np'\n")

    def write_output(self, text):
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

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = mystdout = StringIO()
        sys.stderr = mystderr = StringIO()

        try:
            try:
                result = eval(command, self.local_vars)
                if result is not None:
                    print(repr(result))
            except SyntaxError:
                self.interpreter.runsource(command)
            except Exception:
                self.interpreter.runsource(command)
                
        except Exception:
            self.interpreter.runsource(command)
        
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        output = mystdout.getvalue()
        error = mystderr.getvalue()

        if output: self.write_output(output)
        if error: self.write_output(error)

    def eventFilter(self, obj, event):
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
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
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
        sb = self.log_output.verticalScrollBar()
        sb.setValue(sb.maximum())


    def run_inspector(self):
        self.log_output.clear()
        self.log(">>> Starting Inspector...")
        
        tools_found = 0
        
        for graph in self.graph_list:
            
            for node in graph.node_list:
                
                devices = getattr(node, 'devices', [])
                if not devices and hasattr(node, 'parameters') and 'devices' in node.parameters:
                    devices = node.parameters['devices']
                
                for tool in devices:
                    
                    
                    pass
                    
                    
                    model = tool.get('model', 'unknown')
                    node_name = getattr(node, 'name', f"Node_{node.id}")
                    
                    tools_found += 1

        self.log(f"\n>>> Done. Iterated over {tools_found} tools/devices.")


class NodeLiveGroup(QGroupBox):
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
        
        self.rate_plot = pg.PlotWidget()
        self.rate_plot.setBackground(None)
        self.rate_plot.setMaximumHeight(50)
        self.rate_plot.hideAxis('left')
        self.rate_plot.hideAxis('bottom')
        
        c = QColor(color_hex)
        self.rate_curve = self.rate_plot.plot(pen=pg.mkPen(c, width=2), fillLevel=0, brush=pg.mkBrush(c.red(), c.green(), c.blue(), 40))
        self.layout.addWidget(self.rate_plot)
        
        self.btn_toggle = QPushButton("▼ Details")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.setStyleSheet("text-align: left; color: #666; background: transparent; border: none; font-size: 10px;")
        self.btn_toggle.toggled.connect(self._toggle)
        self.layout.addWidget(self.btn_toggle)
        
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
        pass


class NodeLiveGroup(QGroupBox):
    def __init__(self, graph_id, node_id, node_name, model_name, color, parent=None):
        super().__init__(parent)
        self.graph_id = graph_id
        self.node_id = node_id
        self.node_name = node_name
        
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
        
        self.sum_plot = pg.PlotWidget()
        self.sum_plot.setBackground('#121212')
        self.sum_plot.setMaximumHeight(80)
        self.sum_plot.hideAxis('left')
        self.sum_plot.setLabel('bottom', 'Spike Rate (Population)', units='')
        self.sum_plot.getAxis('bottom').setHeight(15)
        
        self.rate_curve = self.sum_plot.plot(pen=pg.mkPen(color, width=2, fillLevel=0, brush=(50,50,50,100)))
        self.layout.addWidget(self.sum_plot)
        
        self.btn_toggle = QPushButton("▼ Show 0 Devices")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(True)
        self.btn_toggle.setStyleSheet("""
            QPushButton {
                text-align: left; background: #222; color: #888; border: none; padding: 4px;
            }
            QPushButton:hover { background: #333; color: white; }
            QPushButton:checked { color: #00E5FF; }
        """)
        self.btn_toggle.toggled.connect(self.toggle_details)
        self.layout.addWidget(self.btn_toggle)
        
        self.details_container = QWidget()
        self.details_layout = QVBoxLayout(self.details_container)
        self.details_layout.setContentsMargins(0,0,0,0)
        self.details_layout.setSpacing(5)
        self.layout.addWidget(self.details_container)
        
        self.spike_times_buffer = []
        self.plot_widgets = []

    def add_device_widget(self, widget):
        self.details_layout.addWidget(widget)
        self.plot_widgets.append(widget)
        self.btn_toggle.setText(f"▼ Show {len(self.plot_widgets)} Devices")

    def toggle_details(self, checked):
        self.details_container.setVisible(checked)
        arrow = "▼" if checked else "▶"
        self.btn_toggle.setText(f"{arrow} Show {len(self.plot_widgets)} Devices")

    def update_sum_plot(self, time_window=100.0):
        new_spikes = []
        for w in self.plot_widgets:
            if "spike_recorder" in w.device_type:
                if w.times:
                    last_t = w.times[-1]
                    cutoff = last_t - time_window
                    
                    
                    
                    y, x = np.histogram(w.times, bins=50)
                    pass
        
        self.sum_plot.setTitle("Aggregate Activity (ToDo)", size='8pt')



class LivePlotWidget(QWidget):
    closed = pyqtSignal()

    def __init__(self, unique_key, device_type, params, node_info, view_window=1000.0, buffer_window=5000.0, parent=None):
        super().__init__(parent)
        self.unique_key = unique_key
        self.nest_gid = None
        self.device_type = device_type
        self.node_color = QColor(node_info.get('color', '#FFFFFF'))
        
        self.view_window = view_window      
        self.buffer_window = buffer_window  
        
        self.downsample_step = 10 
        self.max_traces = 5
        
        self.data_x = np.array([]) 
        self.data_y = np.array([])
        
        self.traces = {} 
        
        self._init_ui(params, node_info)

    def set_view_window(self, window_ms):
        self.view_window = float(window_ms)
        if self.buffer_window < self.view_window:
            self.buffer_window = self.view_window

    def set_buffer_window(self, window_ms):
        self.buffer_window = float(window_ms)
        if self.view_window > self.buffer_window:
            self.view_window = self.buffer_window

    def _init_ui(self, params, node_info):
        self.setStyleSheet(f"background-color: #1e1e1e; border: 1px solid #333; border-left: 4px solid {self.node_color.name()}; border-radius: 4px;")
        l = QVBoxLayout(self); l.setContentsMargins(0,0,0,0); l.setSpacing(0)
        
        h = QWidget(); h.setStyleSheet("background-color: #252525; border-bottom: 1px solid #333;")
        hl = QHBoxLayout(h); hl.setContentsMargins(5,2,5,2)
        
        dev_label = params.get('label', self.device_type)
        title_str = f"<b>{node_info.get('name')}</b>: {dev_label}"
        
        hl.addWidget(QLabel(title_str, styleSheet=f"color: #eee;"))
        hl.addStretch()
        
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 100); self.step_spin.setValue(10)
        self.step_spin.setToolTip("Downsampling Step")
        self.step_spin.setFixedWidth(40)
        self.step_spin.setStyleSheet("background: #333; color: #00E5FF; border: none;")
        
        if "spike" not in self.device_type:
            hl.addWidget(QLabel("Step:", styleSheet="color:#888; font-size:10px;"))
            hl.addWidget(self.step_spin)
        
        b = QPushButton("×"); b.setFixedSize(20,20)
        b.setStyleSheet("background: transparent; color: #888; border: none; font-weight: bold;")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(self.close_and_flush)
        hl.addWidget(b); l.addWidget(h)
        
        self.pw = pg.PlotWidget()
        self.pw.setBackground('#121212')
        self.pw.showGrid(x=True, y=True, alpha=0.2)
        self.pw.setMouseEnabled(x=True, y=True)
        self.pw.getAxis('bottom').setLabel('Time', units='ms')
        self.pw.setDownsampling(mode='peak') 
        self.pw.setClipToView(True)
        l.addWidget(self.pw)
        
        if "spike" in self.device_type:
            self.pw.setLabel('left', 'Neuron ID')
            self.scatter = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(self.node_color), symbol='s', pxMode=True)
            self.pw.addItem(self.scatter)
            self.is_analog = False
        else:
            record_from = params.get('record_from', ['V_m'])
            if isinstance(record_from, str): record_from = [record_from]
            self.value_key = record_from[0] if record_from else 'V_m'
            self.pw.setLabel('left', self.value_key)
            self.is_analog = True

    def update_data(self, events, current_time=None):
        if not events: return
        
        new_times = events.get('times', [])
        new_senders = events.get('senders', [])
        
        if not hasattr(new_times, '__len__') or len(new_times) == 0: return

        arr_new_times = np.array(new_times) if isinstance(new_times, list) else new_times
        arr_new_senders = np.array(new_senders) if isinstance(new_senders, list) else new_senders


        cutoff_time = 0.0
        if current_time is not None:
            cutoff_time = current_time - self.buffer_window

        if not self.is_analog:
            self.data_x = np.append(self.data_x, arr_new_times)
            self.data_y = np.append(self.data_y, arr_new_senders)
            
            if len(self.data_x) > 0 and self.data_x[0] < cutoff_time:
                cut_idx = np.searchsorted(self.data_x, cutoff_time)
                if cut_idx > 0:
                    self.data_x = self.data_x[cut_idx:]
                    self.data_y = self.data_y[cut_idx:]
            
            self.scatter.setData(x=self.data_x, y=self.data_y)

        else:
            values = events.get(self.value_key, [])
            arr_new_values = np.array(values) if isinstance(values, list) else values
            
            if len(arr_new_values) != len(arr_new_times): return
            
            step = self.step_spin.value()
            unique_senders = np.unique(arr_new_senders)
            active_traces = 0
            
            for sender in unique_senders:
                if sender not in self.traces and len(self.traces) >= self.max_traces:
                    continue
                
                mask = (arr_new_senders == sender)
                t_chunk = arr_new_times[mask][::step]
                v_chunk = arr_new_values[mask][::step]
                
                if len(t_chunk) == 0: continue

                if sender not in self.traces:
                    c = self.node_color
                    import random
                    noise = random.randint(-30, 30)
                    r = max(0, min(255, c.red() + noise))
                    g = max(0, min(255, c.green() + noise))
                    b = max(0, min(255, c.blue() + noise))
                    curve = self.pw.plot(pen=pg.mkPen(QColor(r,g,b), width=1.5), name=f"N{sender}")
                    self.traces[sender] = {'x': np.array([]), 'y': np.array([]), 'curve': curve}
                
                trace = self.traces[sender]
                trace['x'] = np.append(trace['x'], t_chunk)
                trace['y'] = np.append(trace['y'], v_chunk)
                
                if len(trace['x']) > 0 and trace['x'][0] < cutoff_time:
                    cut_idx = np.searchsorted(trace['x'], cutoff_time)
                    if cut_idx > 0:
                        trace['x'] = trace['x'][cut_idx:]
                        trace['y'] = trace['y'][cut_idx:]
                
                trace['curve'].setData(trace['x'], trace['y'])
                active_traces += 1
                if active_traces >= self.max_traces: break

    def set_view_range_to_window(self, current_time):
        min_t = max(0, current_time - self.view_window)
        self.pw.setXRange(min_t, current_time, padding=0)

    def close_and_flush(self):
        self.clear_data()
        self.closed.emit()

    def clear_data(self):
        self.data_x = np.array([])
        self.data_y = np.array([])
        if hasattr(self, 'scatter') and self.scatter: 
            self.scatter.clear()
        if hasattr(self, 'traces'):
            for t in self.traces.values():
                t['x'] = np.array([]); t['y'] = np.array([]); t['curve'].clear()
            self.traces.clear()

class LiveDataDashboard(QWidget):
    MODE_POP_ACTIVITY = 0
    MODE_DEVICE_DETAIL = 1
    
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.active_plots_map = {}
        self.current_view_mode = self.MODE_POP_ACTIVITY
        self.data_cache = {}
        
        self.view_window = 1000.0 
        self.buffer_window = 5000.0 
        
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        left_container = QWidget()
        left_container.setFixedWidth(280)
        left_container.setStyleSheet("background-color: #232323; border-right: 1px solid #444;")
        left_layout = QVBoxLayout(left_container)
        
        left_layout.addWidget(QLabel("DATA SOURCE & MODE", styleSheet="color:#00E5FF; font-weight:bold; margin-top:5px;"))

        mode_group = QFrame()
        mode_group.setStyleSheet("background-color: #1a1a1a; border-radius: 4px; padding: 2px;")
        mode_l = QHBoxLayout(mode_group); mode_l.setSpacing(1); mode_l.setContentsMargins(0,0,0,0)
        
        self.btn_mode_pop = QPushButton("Rate (Pop)")
        self.btn_mode_detail = QPushButton("Detail (Device)")
        self.mode_btns = [self.btn_mode_pop, self.btn_mode_detail]
        
        for i, btn in enumerate(self.mode_btns):
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.clicked.connect(lambda checked, idx=i: self.set_view_mode(idx))
        
        mode_l.addWidget(self.btn_mode_pop)
        mode_l.addWidget(self.btn_mode_detail)
        left_layout.addWidget(mode_group)
        
        win_group = QGroupBox("Time Windows")
        win_l = QFormLayout(win_group)
        
        self.chk_scroll = QCheckBox("Auto-Scroll"); self.chk_scroll.setChecked(True)
        win_l.addRow(self.chk_scroll)
        
        self.spin_view = QDoubleSpinBox()
        self.spin_view.setRange(10, 1e6); self.spin_view.setValue(self.view_window)
        self.spin_view.setSuffix(" ms")
        self.spin_view.setToolTip("Zoom: How much time is visible on screen.")
        self.spin_view.valueChanged.connect(self._on_view_changed)
        win_l.addRow("View (Zoom):", self.spin_view)
        
        self.spin_buffer = QDoubleSpinBox()
        self.spin_buffer.setRange(10, 1e6); self.spin_buffer.setValue(self.buffer_window)
        self.spin_buffer.setSuffix(" ms")
        self.spin_buffer.setStyleSheet("color: #FF9800;")
        self.spin_buffer.setToolTip("Buffer: How long data is kept in memory before deleting.")
        self.spin_buffer.valueChanged.connect(self._on_buffer_changed)
        win_l.addRow("History (RAM):", self.spin_buffer)
        
        left_layout.addWidget(win_group)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        self.tree.itemClicked.connect(self.on_tree_item_clicked)
        left_layout.addWidget(self.tree)
        
        btn_scan = QPushButton("↻ Refresh List"); btn_scan.clicked.connect(self.scan_for_devices)
        left_layout.addWidget(btn_scan)
        btn_clear = QPushButton("🗑 Close All Plots"); btn_clear.clicked.connect(self.close_all_plots)
        left_layout.addWidget(btn_clear)
        
        layout.addWidget(left_container)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background-color: #121212;")
        
        self.plot_container = QWidget()
        self.plot_container.setStyleSheet("background-color: #121212;")
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.plot_layout.setSpacing(10)
        
        self.scroll.setWidget(self.plot_container)
        layout.addWidget(self.scroll)
        
        self.set_view_mode(0)

    def _on_view_changed(self, val):
        self.view_window = val
        if self.view_window > self.buffer_window:
            self.spin_buffer.setValue(self.view_window) 
        for w in self.active_plots_map.values():
            if hasattr(w, 'set_view_window'): w.set_view_window(val)

    def _on_buffer_changed(self, val):
        self.buffer_window = val
        if self.buffer_window < self.view_window:
            self.spin_view.setValue(self.buffer_window) 
        for w in self.active_plots_map.values():
            if hasattr(w, 'set_buffer_window'): w.set_buffer_window(val)

    def scan_for_devices(self):
        self.tree.clear()
        
        active_gid_map = {}
        
        for graph in self.graph_list:
            g_item = QTreeWidgetItem(self.tree)
            g_item.setText(0, getattr(graph, 'graph_name', f"Graph {graph.graph_id}"))
            g_item.setExpanded(True)
            
            for node in graph.node_list:
                devs = getattr(node, 'devices', [])
                if not devs and hasattr(node, 'parameters') and 'devices' in node.parameters: 
                    devs = node.parameters['devices']
                
                relevant_devs = [d for d in devs if any(x in d.get('model', '') for x in ['recorder', 'meter'])]
                if not relevant_devs: continue
                
                n_item = QTreeWidgetItem(g_item)
                n_item.setText(0, node.name)
                n_item.setExpanded(True)
                
                pop_map = {}
                for d in relevant_devs:
                    pid = d.get('target_pop_id', 0)
                    if pid not in pop_map: pop_map[pid] = []
                    pop_map[pid].append(d)
                    
                for pid, d_list in pop_map.items():
                    model = node.neuron_models[pid] if hasattr(node, 'neuron_models') and pid < len(node.neuron_models) else 'unknown'
                    
                    neuron_count = 0
                    if hasattr(node, 'positions') and node.positions and pid < len(node.positions):
                        if node.positions[pid] is not None:
                            neuron_count = len(node.positions[pid])
                    
                    p_item = QTreeWidgetItem(n_item)
                    p_item.setText(0, f"Pop {pid}: {model}")
                    p_item.setForeground(0, QBrush(QColor("#FFD700")))
                    
                    pop_key = (graph.graph_id, node.id, pid)
                    pop_info = {
                        'type': 'pop', 
                        'key': pop_key, 
                        'info': {'name': f"{node.name}_P{pid}", 'color': neuron_colors.get(model, '#fff')}, 
                        'neuron_count': neuron_count,
                        'recorders': []
                    }
                    
                    for dev in d_list:
                        dev_model = dev.get('model','')
                        dev_id = dev.get('id', '?')
                        
                        gid = dev.get('runtime_gid')
                        if hasattr(gid, 'tolist'): gid = gid.tolist()[0] if gid else None
                        elif isinstance(gid, list): gid = gid[0] if gid else None
                        
                        dev_unique_key = (graph.graph_id, node.id, dev_id)
                        active_gid_map[dev_unique_key] = gid
                        
                        dev_data = {
                            'type': 'dev', 
                            'key': dev_unique_key, 
                            'nest_gid': gid, 
                            'params': dev.get('params',{}), 
                            'model': dev_model, 
                            'info': {'name': f"{node.name} #{dev_id}", 'color': neuron_colors.get(model, '#fff')}
                        }
                        
                        if "spike" in dev_model: 
                            pop_info['recorders'].append(dev_data)
                            
                        d_item = QTreeWidgetItem(p_item)
                        d_item.setText(0, f"{'⚡' if 'spike' in dev_model else 'DATA'} {dev_model} #{dev_id}")
                        d_item.setData(0, Qt.ItemDataRole.UserRole, dev_data)
                        
                        if "meter" in dev_model: d_item.setForeground(0, QBrush(QColor("#00E5FF")))
                        else: d_item.setForeground(0, QBrush(QColor("#FFEB3B")))
                        
                    p_item.setData(0, Qt.ItemDataRole.UserRole, pop_info)

        for ukey, widget in self.active_plots_map.items():
            
            if isinstance(widget, LivePlotWidget):
                dev_lookup_key = ukey[:3]
                
                if dev_lookup_key in active_gid_map:
                    new_gid = active_gid_map[dev_lookup_key]
                    if new_gid is not None:
                        widget.nest_gid = new_gid

            elif isinstance(widget, PopulationRatePlot):
                for rec_data in widget.recorders_data:
                    d_key = rec_data.get('key')
                    if d_key in active_gid_map:
                        rec_data['nest_gid'] = active_gid_map[d_key]
                        
    def on_tree_item_clicked(self, item, col):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data: return
        dtype = data.get('type'); base_key = data.get('key')
        if self.current_view_mode == self.MODE_POP_ACTIVITY:
            target_pop_data = None
            if dtype == 'pop': target_pop_data = data
            elif dtype == 'dev' and "spike" in data.get('model',''):
                parent = item.parent()
                if parent: target_pop_data = parent.data(0, Qt.ItemDataRole.UserRole)
            if target_pop_data:
                ukey = (*target_pop_data['key'], 'rate')
                if ukey not in self.active_plots_map: self._add_pop_rate_plot(target_pop_data, ukey)
        else:
            if dtype == 'dev':
                ukey = (*base_key, 'detail')
                if ukey not in self.active_plots_map: self._add_dev_plot(data, ukey)

    def _add_pop_rate_plot(self, data, ukey):
        recorders = data.get('recorders', [])
        if not recorders: return
        info = data['info']
        neuron_count = data.get('neuron_count', 100)
        w = PopulationRatePlot(str(ukey), info['name'], neuron_count, info['color'], view_window=self.view_window, buffer_window=self.buffer_window, parent=self.plot_container)
        w.recorders_data = recorders
        w.closed.connect(lambda: self._on_widget_closed(ukey))
        self.plot_layout.addWidget(w); self.active_plots_map[ukey] = w; self.plot_container.adjustSize()

    def _add_dev_plot(self, data, ukey):
        gid = data.get('nest_gid')
        if gid is None: return
        w = LivePlotWidget(ukey, data['model'], data['params'], data['info'], view_window=self.view_window, buffer_window=self.buffer_window, parent=self.plot_container)
        w.setMinimumHeight(200); w.nest_gid = gid
        if gid in self.data_cache:
            cache = self.data_cache[gid]
            hist = {'times': cache['times'], 'senders': cache['senders']}
            for k,v in cache['values'].items(): hist[k] = v
            w.update_data(hist)
        w.closed.connect(lambda: self._on_widget_closed(ukey))
        self.plot_layout.addWidget(w); self.active_plots_map[ukey] = w

    def _on_widget_closed(self, key):
        if key in self.active_plots_map:
            w = self.active_plots_map.pop(key); w.setParent(None); w.deleteLater()

    def process_incoming_data(self, data_snapshot, current_time):
        
        for gid, events in data_snapshot.items():
            if gid not in self.data_cache: 
                self.data_cache[gid] = {'times':[], 'senders':[], 'values':{}}
            if 'times' in events: 
                self.data_cache[gid]['times'].extend(events['times'])
            if 'senders' in events: 
                self.data_cache[gid]['senders'].extend(events['senders'])
            for k,v in events.items():
                if k not in ['times','senders']:
                    if k not in self.data_cache[gid]['values']: 
                        self.data_cache[gid]['values'][k] = []
                    self.data_cache[gid]['values'][k].extend(v)
        
        if not self.isVisible():
            return
            
        for ukey, w in self.active_plots_map.items():
            target_gids = []
            if isinstance(w, PopulationRatePlot):
                for d in w.recorders_data: 
                    if d['nest_gid']: target_gids.append(d['nest_gid'])
            elif isinstance(w, LivePlotWidget):
                if w.nest_gid: target_gids.append(w.nest_gid)
            
            events_list = []
            for gid in target_gids:
                if gid in data_snapshot: events_list.append(data_snapshot[gid])
            
            if events_list:
                if isinstance(w, PopulationRatePlot): w.update_from_recorders(events_list, current_time)
                else: w.update_data(events_list[0], current_time)
            
            if self.chk_scroll.isChecked():
                if hasattr(w, 'set_view_range_to_window'):
                    w.set_view_range_to_window(current_time)

    def close_all_plots(self):
        for k in list(self.active_plots_map.keys()): self._on_widget_closed(k)
    def clear_all_data(self):
        self.data_cache.clear()
        for w in self.active_plots_map.values(): w.clear_data()
    def get_all_data(self): return self.data_cache
    def reload_and_center(self): self.scan_for_devices()
    def set_view_mode(self, idx):
        self.current_view_mode = idx
        for i, btn in enumerate(self.mode_btns):
            btn.setChecked(i == idx)
            style = "background-color: #2196F3; color: white; border: none; font-weight: bold;" if i == idx else "background-color: #333; color: #888; border: none;"
            btn.setStyleSheet(style)
    def _get_mode_btn_style(self, active):
        return f"background-color: {'#2196F3' if active else '#333'}; color: {'white' if active else '#888'}; font-weight: bold; border: none;"


class PopulationRatePlot(QWidget):
    closed = pyqtSignal()

    def __init__(self, plot_key, title, neuron_count, color_hex, view_window=1000.0, buffer_window=5000.0, resolution=None, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        
        self.plot_key = plot_key
        self.neuron_count = max(1, neuron_count)
        
        self.view_window = view_window
        self.buffer_window = buffer_window
        
        if resolution is None:
            try:
                self.resolution = nest.resolution
            except Exception:
                self.resolution = 0.1
        else:
            self.resolution = resolution
        
        self.view_window = view_window
        self.buffer_window = buffer_window
        
        self.time_bins = []
        self.rate_values = []
        self.recorders_data = [] 
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0); self.layout.setSpacing(0)
        self.setStyleSheet(f"border: 1px solid #444; border-left: 4px solid {color_hex}; background-color: #121212; border-radius: 4px;")
        
        header = QWidget(); header.setStyleSheet("background-color: #222; border-bottom: 1px solid #333;")
        hl = QHBoxLayout(header); hl.setContentsMargins(5,2,5,2)
        lbl = QLabel(f"{title} (Rate)"); lbl.setStyleSheet("color: white; font-weight: bold;")
        hl.addWidget(lbl); hl.addStretch()
        btn_close = QPushButton("×"); btn_close.setFixedSize(20,20)
        btn_close.setStyleSheet("background: transparent; color: #888; border: none; font-weight: bold;")
        btn_close.clicked.connect(self.closed.emit)
        hl.addWidget(btn_close)
        self.layout.addWidget(header)
        
        self.pw = pg.PlotWidget()
        self.pw.setBackground(None)
        self.pw.showGrid(x=True, y=True, alpha=0.3)
        self.pw.setLabel('left', 'Firing Rate', units='Hz')
        self.pw.setLabel('bottom', 'Time', units='ms')
        self.pw.setMouseEnabled(x=True, y=True)
        
        c = QColor(color_hex)
        self.curve = self.pw.plot(pen=pg.mkPen(c, width=2), fillLevel=0, brush=pg.mkBrush(c.red(), c.green(), c.blue(), 50))
        self.layout.addWidget(self.pw)

    def set_view_window(self, ms):
        self.view_window = float(ms)
        if self.buffer_window < self.view_window: self.buffer_window = self.view_window

    def set_buffer_window(self, ms):
        self.buffer_window = float(ms)
        if self.view_window > self.buffer_window: self.view_window = self.buffer_window

    def update_from_recorders(self, events_list, current_time):
        total_spikes = 0
        for ev in events_list:
            if 'times' in ev: total_spikes += len(ev['times'])
        
        hz = total_spikes / (self.neuron_count * self.resolution)
        
        self.time_bins.append(current_time)
        self.rate_values.append(hz)
        
        cutoff = current_time - self.buffer_window
        while len(self.time_bins) > 0 and self.time_bins[0] < cutoff:
            self.time_bins.pop(0)
            self.rate_values.pop(0)
            
        self.curve.setData(self.time_bins, self.rate_values)

    def set_view_range_to_window(self, current_time):
        min_t = max(0, current_time - self.view_window)
        self.pw.setXRange(min_t, current_time, padding=0)
        
    def clear_data(self):
        self.time_bins = []; self.rate_values = []; self.curve.clear()



class LiveConnectionController:
    @staticmethod
    def set_weight(graph_list, conn_data, new_weight):
        try:
            src_info = conn_data.get('source', {})
            tgt_info = conn_data.get('target', {})
            
            src_pop = LiveConnectionController._find_population(graph_list, src_info)
            tgt_pop = LiveConnectionController._find_population(graph_list, tgt_info)
            
            if src_pop is None or tgt_pop is None:
                print("⚠ Live Update Failed: Source or Target population not found.")
                return False

            conns = nest.GetConnections(src_pop, tgt_pop)
            
            if not conns:
                print("⚠ Live Update: No active NEST connections found between these populations.")
                return False
            
            nest.SetStatus(conns, {'weight': float(new_weight)})
            print(f"⚡ Live Update: Set weight to {new_weight} for {len(conns)} connections.")
            return True
            
        except Exception as e:
            print(f"Live Update Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def _find_population(graph_list, info):
        gid = info.get('graph_id')
        nid = info.get('node_id')
        pid = info.get('pop_id')
        
        graph = next((g for g in graph_list if g.graph_id == gid), None)
        if not graph: return None
        
        node = graph.get_node(nid)
        if not node or not hasattr(node, 'population'): return None
        
        if pid < len(node.population):
            return node.population[pid]
        return None