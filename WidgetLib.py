import sys
from PyQt6.QtWidgets import QApplication, QMainWindow,QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtWidgets import QPushButton,QLabel, QStackedWidget, QToolBar, QMenu, QGridLayout, QStackedLayout
from PyQt6.QtGui import QColor, QPalette, QAction,QIcon,QBrush
from PyQt6.QtCore import QSize, Qt, pyqtSignal,QTimer
import pyvista as pv
from pyvistaqt import QtInteractor
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.colors as mcolors
from datetime import datetime
from pathlib import Path
from neuron_toolbox import *
from WidgetLib import *
from PyQt6.QtWidgets import (
    QLineEdit,      # Text-Eingabefeld (einzeilig)
    QTextEdit,      # Text-Eingabefeld (mehrzeilig)
    QSpinBox,       # Integer-Input mit +/- Buttons
    QDoubleSpinBox, # Float-Input mit +/- Buttons
    QComboBox,      # Dropdown-Menü
    QCheckBox,      # Checkbox (an/aus)
    QRadioButton,   # Radio Button (eine Option aus Gruppe)
    QSlider,        # Schieberegler
    QDial,          # Drehregler
    QPushButton,    # Button
    QFileDialog,    # Datei-Auswahl Dialog
    QColorDialog,   # Farb-Auswahl Dialog
    QTreeWidget,    
    QTreeWidgetItem 
)
from PyQt6.QtWidgets import QScrollArea, QInputDialog
current_graph_metadata = []
graph_parameters = {}     #graph_id 
next_graph_id = 0



synapse_models = {
    "bernoulli_synapse": {
        "p_transmit": {
            "default": 1.0,
            "type": "float",
            "unit": "dimensionless",
            "min": 0.0,
            "max": 1.0,
            "constraint": "range",
            "description": "Transmission probability"
        }
    },
    "clopath_synapse": {
        "tau_x": {
            "default": 10.0,
            "type": "float",
            "unit": "ms",
            "min": 0.1,
            "max": 100.0,
            "constraint": "positive",
            "description": "Time constant of the trace of the presynaptic spike train"
        },
        "Wmax": {
            "default": 10.0,
            "type": "float",
            "unit": "pA",
            "min": 0.0,
            "max": 100.0,
            "constraint": "non-negative",
            "description": "Maximum allowed weight"
        },
        "Wmin": {
            "default": 0.0,
            "type": "float",
            "unit": "pA",
            "min": 0.0,
            "max": 100.0,
            "constraint": "non-negative",
            "description": "Minimum allowed weight"
        }
    },
    "cont_delay_synapse": {},
    "diffusion_connection": {},
    "gap_junction": {},
    "quantal_stp_synapse": {
        "U": {
            "default": 0.5,
            "type": "float",
            "unit": "dimensionless",
            "min": 0.0,
            "max": 1.0,
            "constraint": "range",
            "description": "Maximal fraction of available resources"
        },
        "u": {
            "default": 0.5,
            "type": "float",
            "unit": "dimensionless",
            "min": 0.0,
            "max": 1.0,
            "constraint": "range",
            "description": "Available fraction of resources"
        },
        "n": {
            "default": 1,
            "type": "integer",
            "unit": "dimensionless",
            "min": 1,
            "max": 100,
            "constraint": "positive",
            "description": "Total number of release sites"
        },
        "a": {
            "default": 1,
            "type": "integer",
            "unit": "dimensionless",
            "min": 1,
            "max": 100,
            "constraint": "positive",
            "description": "Number of available release sites"
        },
        "tau_fac": {
            "default": 0.0,
            "type": "float",
            "unit": "ms",
            "min": 0.0,
            "max": 1000.0,
            "constraint": "non-negative",
            "description": "Time constant for facilitation"
        },
        "tau_rec": {
            "default": 800.0,
            "type": "float",
            "unit": "ms",
            "min": 0.1,
            "max": 1000.0,
            "constraint": "positive",
            "description": "Time constant for depression"
        }
    },
    "rate_connection_delayed": {},
    "rate_connection_instantaneous": {},
    "static_synapse": {},
    "stdp_dopamine_synapse": {},
    "stdp_facetshw_synapse_hom": {},
    "stdp_gg_synapse": {},
    "stdp_pl_synapse_hom": {},
    "stdp_synapse": {
        "tau_plus": {
            "default": 20.0,
            "type": "float",
            "unit": "ms",
            "min": 0.1,
            "max": 100.0,
            "constraint": "positive",
            "description": "Time constant of STDP window, potentiation"
        },
        "lambda": {
            "default": 0.01,
            "type": "float",
            "unit": "dimensionless",
            "min": 0.0,
            "max": 1.0,
            "constraint": "non-negative",
            "description": "Step size"
        },
        "alpha": {
            "default": 1.0,
            "type": "float",
            "unit": "dimensionless",
            "min": 0.0,
            "max": 10.0,
            "constraint": "non-negative",
            "description": "Asymmetry parameter"
        },
        "mu_plus": {
            "default": 1.0,
            "type": "float",
            "unit": "dimensionless",
            "min": 0.0,
            "max": 10.0,
            "constraint": "non-negative",
            "description": "Weight dependence exponent, potentiation"
        },
        "mu_minus": {
            "default": 1.0,
            "type": "float",
            "unit": "dimensionless",
            "min": 0.0,
            "max": 10.0,
            "constraint": "non-negative",
            "description": "Weight dependence exponent, depression"
        },
        "Wmax": {
            "default": 100.0,
            "type": "float",
            "unit": "pA",
            "min": 0.0,
            "max": 1000.0,
            "constraint": "non-negative",
            "description": "Maximum allowed weight"
        }
    },
    "stdp_synapse_hom": {},
    "stdp_triplet_synapse": {},
    "stdp_triplet_synapse_hom": {},
    "tsodyks2_synapse": {},
    "tsodyks_synapse": {
        "U": {
            "default": 0.5,
            "type": "float",
            "unit": "dimensionless",
            "min": 0.0,
            "max": 1.0,
            "constraint": "range",
            "description": "Parameter determining the increase in u with each spike"
        },
        "tau_psc": {
            "default": 3.0,
            "type": "float",
            "unit": "ms",
            "min": 0.1,
            "max": 100.0,
            "constraint": "positive",
            "description": "Time constant of synaptic current"
        },
        "tau_fac": {
            "default": 0.0,
            "type": "float",
            "unit": "ms",
            "min": 0.0,
            "max": 1000.0,
            "constraint": "non-negative",
            "description": "Time constant for facilitation"
        },
        "tau_rec": {
            "default": 800.0,
            "type": "float",
            "unit": "ms",
            "min": 0.1,
            "max": 1000.0,
            "constraint": "positive",
            "description": "Time constant for depression"
        },
        "x": {
            "default": 1.0,
            "type": "float",
            "unit": "dimensionless",
            "min": 0.0,
            "max": 1.0,
            "constraint": "range",
            "description": "Initial fraction of synaptic vesicles in the readily releasable pool"
        },
        "y": {
            "default": 0.0,
            "type": "float",
            "unit": "dimensionless",
            "min": 0.0,
            "max": 1.0,
            "constraint": "range",
            "description": "Initial fraction of synaptic vesicles in the synaptic cleft"
        },
        "u": {
            "default": 0.5,
            "type": "float",
            "unit": "dimensionless",
            "min": 0.0,
            "max": 1.0,
            "constraint": "range",
            "description": "Initial release probability of synaptic vesicles"
        }
    },
    "vogels_sprekeler_synapse": {}
}




# WidgetLib.py - Ganz oben nach den Imports

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
            # ... bleibt gleich
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
        
        self.add_section("Basic Info")
        self.add_text_field("name", "Name")
        self.add_int_field("id", "ID", min_val=0, max_val=1000)
        
        self.add_section("Position & Center")
        #self.add_vector3_field("m", "Center (m)")
        self.add_vector3_field("center_of_mass", "Center of Mass")
        self.add_vector3_field("displacement", "Displacement")
        self.add_float_field("displacement_factor", "Displacement Factor", min_val=0.0, max_val=10.0)
        
        self.add_section("Rotation")
        self.add_float_field("rot_theta", "Theta (°)", min_val=-360.0, max_val=360.0)
        self.add_float_field("rot_phi", "Phi (°)", min_val=-360.0, max_val=360.0)
        
        self.add_section("Grid Settings")
        self.add_vector3_int_field("grid_size", "Grid Size")
        self.add_float_field("sparsity_factor", "Sparsity Factor", min_val=0.0, max_val=1.0)
        self.add_int_field("sparse_holes", "Sparse Holes", min_val=0, max_val=1000)
        
        self.add_section("Algorithm")
        self.add_int_field("num_steps", "WFC Steps", min_val=1, max_val=100)
        self.add_int_field("polynom_max_power", "Max Polynomial Power", min_val=1, max_val=20)
        self.add_float_field("dt", "Time Step (dt)", min_val=0.001, max_val=1.0)
        self.add_bool_field("old", "Use Old Algorithm")
        
        self.add_section("Probability Vector (Population Weights)")
        self.probability_container = QWidget()
        self.probability_layout = QVBoxLayout(self.probability_container)
        self.content_layout.addWidget(self.probability_container)
        
        self.content_layout.addStretch()
        
    def add_section(self, title):
        label = QLabel(title)
        label.setStyleSheet("font-weight: bold; font-size: 13px; color: #2196F3; margin-top: 10px; border-bottom: 2px solid #2196F3; padding-bottom: 5px;")
        self.content_layout.addWidget(label)
    
    def add_text_field(self, key, label):
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        edit = QLineEdit()
        edit.textChanged.connect(self.on_change)
        row.addWidget(lbl)
        row.addWidget(edit)
        self.content_layout.addLayout(row)
        self.widgets[key] = {'type': 'text', 'widget': edit}
    
    def add_int_field(self, key, label, min_val=0, max_val=10000):
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.valueChanged.connect(self.on_change)
        row.addWidget(lbl)
        row.addWidget(spin)
        self.content_layout.addLayout(row)
        self.widgets[key] = {'type': 'int', 'widget': spin}
    
    def add_float_field(self, key, label, min_val=-1000.0, max_val=1000.0):
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
        self.content_layout.addLayout(row)
        self.widgets[key] = {'type': 'float', 'widget': spin}
    
    def add_bool_field(self, key, label):
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        check = QCheckBox()
        check.stateChanged.connect(self.on_change)
        row.addWidget(lbl)
        row.addWidget(check)
        row.addStretch()
        self.content_layout.addLayout(row)
        self.widgets[key] = {'type': 'bool', 'widget': check}
    
    def add_vector3_field(self, key, label):
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        
        x_spin = QDoubleSpinBox()
        x_spin.setRange(-1000.0, 1000.0)
        x_spin.setDecimals(3)
        x_spin.setPrefix("X: ")
        x_spin.valueChanged.connect(self.on_change)
        
        y_spin = QDoubleSpinBox()
        y_spin.setRange(-1000.0, 1000.0)
        y_spin.setDecimals(3)
        y_spin.setPrefix("Y: ")
        y_spin.valueChanged.connect(self.on_change)
        
        z_spin = QDoubleSpinBox()
        z_spin.setRange(-1000.0, 1000.0)
        z_spin.setDecimals(3)
        z_spin.setPrefix("Z: ")
        z_spin.valueChanged.connect(self.on_change)
        
        # ❌ LÖSCHE DIESE GANZE SEKTION:
        # if key == "center_of_mass":
        #     x_spin.valueChanged.connect(lambda v: self.sync_to_m('x', v))
        #     y_spin.valueChanged.connect(lambda v: self.sync_to_m('y', v))
        #     z_spin.valueChanged.connect(lambda v: self.sync_to_m('z', v))
        # elif key == "m":
        #     x_spin.valueChanged.connect(lambda v: self.sync_to_com('x', v))
        #     y_spin.valueChanged.connect(lambda v: self.sync_to_com('y', v))
        #     z_spin.valueChanged.connect(lambda v: self.sync_to_com('z', v))
        
        row.addWidget(lbl)
        row.addWidget(x_spin)
        row.addWidget(y_spin)
        row.addWidget(z_spin)
        self.content_layout.addLayout(row)
        
        self.widgets[key] = {'type': 'vector3', 'widgets': [x_spin, y_spin, z_spin]}
    
    def add_vector3_int_field(self, key, label):
        row = QHBoxLayout()
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(150)
        lbl.setStyleSheet("font-weight: bold;")
        
        x_spin = QSpinBox()
        x_spin.setRange(1, 1000)
        x_spin.setPrefix("X: ")
        x_spin.valueChanged.connect(self.on_change)
        
        y_spin = QSpinBox()
        y_spin.setRange(1, 1000)
        y_spin.setPrefix("Y: ")
        y_spin.valueChanged.connect(self.on_change)
        
        z_spin = QSpinBox()
        z_spin.setRange(1, 1000)
        z_spin.setPrefix("Z: ")
        z_spin.valueChanged.connect(self.on_change)
        
        row.addWidget(lbl)
        row.addWidget(x_spin)
        row.addWidget(y_spin)
        row.addWidget(z_spin)
        self.content_layout.addLayout(row)
        
        self.widgets[key] = {'type': 'vector3_int', 'widgets': [x_spin, y_spin, z_spin]}
    
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
            elif wtype == 'int':
                result[key] = info['widget'].value()
            elif wtype == 'float':
                result[key] = info['widget'].value()
            elif wtype == 'bool':
                result[key] = info['widget'].isChecked()
            elif wtype == 'vector3':
                result[key] = [s.value() for s in info['widgets']]
            elif wtype == 'vector3_int':
                result[key] = [s.value() for s in info['widgets']]
            elif wtype == 'prob_list':
                result[key] = [s.value() for s in info['widgets']]
        
        # ✅ AUTO-SYNC: m = center_of_mass
        if 'center_of_mass' in result:
            result['m'] = result['center_of_mass'].copy()
        
        result['transform_matrix'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        return result
    
    def load_data(self, data):
        self.auto_save = False
        self.node_data = data
        
        for key, info in self.widgets.items():
            if key not in data:
                continue
            
            value = data[key]
            wtype = info['type']
            
            if wtype == 'text':
                info['widget'].setText(str(value))
            elif wtype == 'int':
                info['widget'].setValue(int(value))
            elif wtype == 'float':
                info['widget'].setValue(float(value))
            elif wtype == 'bool':
                info['widget'].setChecked(bool(value))
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
        
        self.auto_save = True

class PolynomialTrioWidget(QWidget):
    
    def __init__(self, pop_name="Population"):
        super().__init__()
        self.pop_name = pop_name
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        header = QLabel(f"Flow Polynomials for {self.pop_name}")
        header.setStyleSheet("font-weight: bold; font-size: 13px; color: #2196F3; padding: 5px;")
        layout.addWidget(header)
        
        from PyQt6.QtWidgets import QTabWidget
        self.tabs = QTabWidget()
        
        self.poly_x = PolynomialBuilderWidget()
        self.poly_y = PolynomialBuilderWidget()
        self.poly_z = PolynomialBuilderWidget()
        
        self.tabs.addTab(self.poly_x, "X Direction")
        self.tabs.addTab(self.poly_y, "Y Direction")
        self.tabs.addTab(self.poly_z, "Z Direction")
        
        layout.addWidget(self.tabs)
    
    def get_polynomials(self):
        return {
            'x': self.poly_x.get_data(),
            'y': self.poly_y.get_data(),
            'z': self.poly_z.get_data()
        }
    
    def load_polynomials(self, poly_dict):
        if 'x' in poly_dict:
            self.poly_x.load_data(poly_dict['x'])
        if 'y' in poly_dict:
            self.poly_y.load_data(poly_dict['y'])
        if 'z' in poly_dict:
            self.poly_z.load_data(poly_dict['z'])



class PolynomialManagerWidget(QWidget):
    polynomialsChanged = pyqtSignal(int, list)  # Signal: (node_idx, polynomials)
    
    def __init__(self):
        super().__init__()
        self.population_polynomials = {}  # {pop_idx: PolynomialTrioWidget}
        self.current_node_idx = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        header = QLabel("POLYNOMIAL FLOW FIELDS")
        header.setStyleSheet("font-weight: bold; font-size: 16px; color: #FF9800; padding: 10px;")
        layout.addWidget(header)
        
        info = QLabel("Define flow field polynomials for each population (X, Y, Z directions)")
        info.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(info)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        scroll.setWidget(self.content)
        layout.addWidget(scroll)
        
        # ✅ Apply Button
        self.apply_btn = QPushButton("✓ Apply Polynomial Changes")
        self.apply_btn.setMinimumHeight(50)
        self.apply_btn.setStyleSheet(
            "background-color: #FF5722; color: white; font-weight: bold; font-size: 14px;"
        )
        self.apply_btn.clicked.connect(self.apply_changes)
        layout.addWidget(self.apply_btn)
    
    def set_populations(self, population_list, node_idx=None):
        """Lädt Populationen und deren Polynome"""
        self.current_node_idx = node_idx
        
        # Clear existing widgets
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.population_polynomials.clear()
        
        if not population_list:
            placeholder = QLabel("No populations defined. Add populations first.")
            placeholder.setStyleSheet("color: #999; font-style: italic; padding: 20px;")
            self.content_layout.addWidget(placeholder)
            return
        
        for i, pop in enumerate(population_list):
            pop_name = f"Population {i+1}: {pop['model']}"
            trio = PolynomialTrioWidget(pop_name)
            
            # Load existing polynomials if available
            if 'polynomials' in pop and pop['polynomials']:
                trio.load_polynomials(pop['polynomials'])
            
            self.content_layout.addWidget(trio)
            self.population_polynomials[i] = trio
            
            # Add separator between populations
            if i < len(population_list) - 1:
                separator = QLabel()
                separator.setStyleSheet("background-color: #DDD; min-height: 2px; max-height: 2px;")
                self.content_layout.addWidget(separator)
        
        self.content_layout.addStretch()
    
    def apply_changes(self):
        """Emits signal with updated polynomials"""
        if self.current_node_idx is None:
            print("⚠ No node selected!")
            return
        
        all_polynomials = self.get_all_polynomials()
        self.polynomialsChanged.emit(self.current_node_idx, all_polynomials)
        print(f"✅ Applied polynomial changes to Node {self.current_node_idx}")
    
    def get_all_polynomials(self):
        """Returns list of polynomial dicts for all populations"""
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
    """
    Hierarchische Übersicht aller Graphen:
    📊 Graph 0: Name
    ├── 🟡 Node 0: NodeName
    │   ├── 🟠 Pop 0: iaf_psc_alpha [1000 neurons]
    │   │   ├── → N1P0 (static_synapse, w=1.50)
    │   │   └── ↻ Self (static_synapse, w=0.80)
    │   └── 🟠 Pop 1: iaf_psc_delta [500 neurons]
    └── 🟡 Node 1: ...
    """
    
    # Signals
    node_selected = pyqtSignal(int, int)  # graph_id, node_id
    population_selected = pyqtSignal(int, int, int)  # graph_id, node_id, pop_id
    connection_selected = pyqtSignal(dict)  # connection dict
    
    # =========================================================================
    # FARB-KONSTANTEN (leicht anpassbar)
    # =========================================================================
    
    # Graph: Blau
    COLOR_GRAPH_BG = "#2196F3"      # Blau
    COLOR_GRAPH_FG = "#111111"      # Weiß
    
    # Node: Gelb
    COLOR_NODE_BG = "#FFC107"       # Amber/Gelb (besser lesbar als reines Gelb)
    COLOR_NODE_FG = "#000000"       # Schwarz
    
    # Population: Orange
    COLOR_POP_BG = "#FF9800"        # Orange
    COLOR_POP_FG = "#000000"        # Schwarz
    
    # Connections
    COLOR_CONN_NORMAL = "#7B1FA2"   # Lila (für normale Connections)
    COLOR_CONN_SELF = "#E65100"     # Dunkelorange (für Self-Connections)
    
    def __init__(self, parent=None, graph_list=None):
        super().__init__(parent)
        self.main_window = parent
        self.graph_list = graph_list if graph_list is not None else []
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # === HEADER ===
        header = QHBoxLayout()
        title = QLabel("📊 Graph Overview")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")
        header.addWidget(title)
        header.addStretch()
        
        # Buttons
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setFixedSize(28, 28)
        self.refresh_btn.setToolTip("Refresh")
        self.refresh_btn.clicked.connect(self.update_tree)
        header.addWidget(self.refresh_btn)
        
        self.expand_btn = QPushButton("⊕")
        self.expand_btn.setFixedSize(28, 28)
        self.expand_btn.setToolTip("Expand All")
        self.expand_btn.clicked.connect(self._expand_all)
        header.addWidget(self.expand_btn)
        
        self.collapse_btn = QPushButton("⊖")
        self.collapse_btn.setFixedSize(28, 28)
        self.collapse_btn.setToolTip("Collapse All")
        self.collapse_btn.clicked.connect(self._collapse_all)
        header.addWidget(self.collapse_btn)
        
        layout.addLayout(header)
        
        # === TREE WIDGET ===
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Element", "Details"])
        self.tree.setColumnWidth(0, 240)
        self.tree.setColumnWidth(1, 200)
        self.tree.setAlternatingRowColors(False)  # Wir machen eigene Farben
        self.tree.setAnimated(True)
        self.tree.setIndentation(20)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        # === HELLES THEME ===
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #FAFAFA;
                color: #333333;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 12px;
            }
            QTreeWidget::item {
                padding: 4px 2px;
                border-bottom: 1px solid #EEEEEE;
                min-height: 24px;
            }
            QTreeWidget::item:selected {
                background-color: #BBDEFB;
                color: #000000;
            }
            QTreeWidget::item:hover {
                background-color: #E3F2FD;
            }
            QHeaderView::section {
                background-color: #E0E0E0;
                color: #333333;
                padding: 6px;
                border: none;
                border-bottom: 2px solid #BDBDBD;
                font-weight: bold;
            }
            QTreeWidget::branch {
                background-color: #FAFAFA;
            }
        """)
        
        layout.addWidget(self.tree)
        
        # === STATUS ===
        self.status_label = QLabel("No graphs")
        self.status_label.setStyleSheet("color: #666; font-size: 11px; padding: 3px;")
        layout.addWidget(self.status_label)
    
    def update_tree(self):
        """Aktualisiert die Baum-Anzeige mit allen Graphen."""
        self.tree.clear()
        
        if not self.graph_list:
            self.status_label.setText("No graphs loaded")
            return
        
        total_nodes = 0
        total_pops = 0
        total_conns = 0
        
        for graph in self.graph_list:
            # === GRAPH ITEM ===
            graph_item = QTreeWidgetItem(self.tree)
            graph_name = getattr(graph, 'graph_name', f'Graph_{graph.graph_id}')
            graph_item.setText(0, f"📊 {graph_name}")
            graph_item.setText(1, f"ID: {graph.graph_id} | {len(graph.node_list)} nodes")
            
            # Graph Styling
            self._style_item(graph_item, self.COLOR_GRAPH_BG, self.COLOR_GRAPH_FG, bold=True)
            
            graph_item.setData(0, Qt.ItemDataRole.UserRole, {
                'type': 'graph', 
                'graph_id': graph.graph_id
            })
            graph_item.setExpanded(True)
            
            # === NODES ===
            for node in graph.node_list:
                total_nodes += 1
                node_item = self._create_node_item(graph, node, graph_item)
                
                # === POPULATIONS ===
                populations = self._get_populations(node)
                for pop_idx, pop_info in enumerate(populations):
                    total_pops += 1
                    pop_item = self._create_population_item(graph, node, pop_idx, pop_info, node_item)
                    
                    # === CONNECTIONS ===
                    connections = self._get_connections_for_pop(node, pop_idx)
                    for conn in connections:
                        total_conns += 1
                        self._create_connection_item(conn, pop_item)
        
        # Status aktualisieren
        self.status_label.setText(
            f"📊 {len(self.graph_list)} graphs | 🟡 {total_nodes} nodes | 🟠 {total_pops} pops | → {total_conns} conns"
        )
    
    def _style_item(self, item, bg_color, fg_color, bold=False):
        """Wendet Hintergrund- und Schriftfarbe auf ein Item an."""
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
        """Erstellt TreeWidget Item für einen Node."""
        item = QTreeWidgetItem(parent_item)
        
        # Icon basierend auf Root-Status
        is_root = not hasattr(node, 'parent') or node.parent is None
        icon = "🔵" if is_root else "🟡"
        
        node_name = getattr(node, 'name', f'Node_{node.id}')
        item.setText(0, f"{icon} Node {node.id}: {node_name}")
        
        # Details
        n_pops = len(node.types) if hasattr(node, 'types') and node.types else 0
        n_conns = len(node.connections) if hasattr(node, 'connections') and node.connections else 0
        item.setText(1, f"{n_pops} pops | {n_conns} connections")
        
        # Node Styling (Gelb)
        self._style_item(item, self.COLOR_NODE_BG, self.COLOR_NODE_FG, bold=False)
        
        # UserData
        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'node',
            'graph_id': graph.graph_id,
            'node_id': node.id
        })
        
        item.setExpanded(True)
        return item
    
    def _create_population_item(self, graph, node, pop_idx, pop_info, parent_item):
        """Erstellt TreeWidget Item für eine Population."""
        item = QTreeWidgetItem(parent_item)
        
        model = pop_info.get('model', 'unknown')
        n_neurons = pop_info.get('n_neurons', '?')
        
        item.setText(0, f"    🟠 Pop {pop_idx}: {model}")
        item.setText(1, f"[{n_neurons} neurons]")
        
        # Population Styling (Orange)
        self._style_item(item, self.COLOR_POP_BG, self.COLOR_POP_FG, bold=False)
        
        # UserData
        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'population',
            'graph_id': graph.graph_id,
            'node_id': node.id,
            'pop_id': pop_idx,
            'info': pop_info
        })
        
        return item
    
    def _create_connection_item(self, conn, parent_item):
        """Erstellt TreeWidget Item für eine Connection."""
        item = QTreeWidgetItem(parent_item)
        
        source = conn.get('source', {})
        target = conn.get('target', {})
        params = conn.get('params', {})
        
        # Self-Connection Check
        is_self = (
            source.get('graph_id') == target.get('graph_id') and
            source.get('node_id') == target.get('node_id') and
            source.get('pop_id') == target.get('pop_id')
        )
        
        if is_self:
            icon = "↻"
            target_str = "Self"
            color = self.COLOR_CONN_SELF
        else:
            icon = "→"
            # Format: G0N1P0 oder N1P0 wenn gleicher Graph
            if source.get('graph_id') != target.get('graph_id'):
                target_str = f"G{target.get('graph_id', '?')}N{target.get('node_id', '?')}P{target.get('pop_id', '?')}"
            else:
                target_str = f"N{target.get('node_id', '?')}P{target.get('pop_id', '?')}"
            color = self.COLOR_CONN_NORMAL
        
        # Synapse Info
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
        
        # Connection Styling (kein Hintergrund, nur Textfarbe)
        fg_brush = QBrush(QColor(color))
        for col in range(2):
            item.setForeground(col, fg_brush)
        
        # Kleinere, kursive Schrift
        font = item.font(0)
        font.setPointSize(max(9, font.pointSize() - 1))
        font.setItalic(True)
        item.setFont(0, font)
        item.setFont(1, font)
        
        # UserData
        item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'connection',
            'connection': conn
        })
        
        return item
    
    def _get_populations(self, node):
        """Extrahiert Population-Infos aus einem Node."""
        populations = []
        
        types = node.types if hasattr(node, 'types') and node.types else []
        models = node.neuron_models if hasattr(node, 'neuron_models') and node.neuron_models else []
        
        for i, t in enumerate(types):
            n_neurons = 0
            
            # Aus positions zählen
            if hasattr(node, 'positions') and node.positions and i < len(node.positions):
                pos = node.positions[i]
                if pos is not None and hasattr(pos, '__len__'):
                    n_neurons = len(pos)
            
            # Fallback: distribution
            if n_neurons == 0 and hasattr(node, 'distribution') and node.distribution:
                if i < len(node.distribution):
                    n_neurons = int(node.distribution[i])
            
            # Fallback: NEST population
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
        """
        Findet ALLE Connections die von dieser Population ausgehen.
        
        WICHTIG: Prüft node.connections UND durchsucht alle anderen Nodes
        nach Connections die auf diese Population zeigen.
        """
        if not hasattr(node, 'connections') or not node.connections:
            return []
        
        result = []
        
        for conn in node.connections:
            source = conn.get('source', {})
            # Connection gehört zu dieser Population wenn source.pop_id == pop_idx
            if source.get('node_id') == node.id and source.get('pop_id') == pop_idx:
                result.append(conn)
        
        return result
    
    def _expand_all(self):
        """Expandiert alle Items."""
        self.tree.expandAll()
    
    def _collapse_all(self):
        """Kollabiert alle außer Graph-Roots."""
        self.tree.collapseAll()
        # Graph-Roots offen lassen
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setExpanded(True)
    
    def _on_item_clicked(self, item, column):
        """Handler für Klick."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        
        item_type = data.get('type')
        
        if item_type == 'node':
            self.node_selected.emit(data['graph_id'], data['node_id'])
            print(f"[GraphOverview] Selected: Graph {data['graph_id']}, Node {data['node_id']}")
        elif item_type == 'population':
            self.population_selected.emit(data['graph_id'], data['node_id'], data['pop_id'])
            print(f"[GraphOverview] Selected: Graph {data['graph_id']}, Node {data['node_id']}, Pop {data['pop_id']}")
        elif item_type == 'connection':
            self.connection_selected.emit(data['connection'])
            print(f"[GraphOverview] Selected Connection: {data['connection'].get('name', 'unnamed')}")
    
    def _on_item_double_clicked(self, item, column):
        """Handler für Doppelklick."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        
        item_type = data.get('type')
        
        if item_type == 'node':
            print(f"[GraphOverview] Double-click: Graph {data['graph_id']}, Node {data['node_id']}")
        elif item_type == 'connection':
            conn = data['connection']
            print(f"[GraphOverview] Double-click Connection: {conn.get('name', 'unnamed')}")
            # Hier könnte man einen Edit-Dialog öffnen
            
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
    polynomial_saved = pyqtSignal(dict)

    def __init__(self, max_power=5, default_decay=0.5):
        super().__init__()
        self.max_power = max_power
        self.default_decay = default_decay
        self.var_map = {0: 'x', 1: 'y', 2: 'z'}
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        settings_layout = QHBoxLayout()
        
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 20)
        self.spin_n.setValue(self.max_power)
        self.spin_n.setPrefix("Max Power (n): ")
        
        self.spin_decay = QDoubleSpinBox()
        self.spin_decay.setRange(0.0, 5.0)
        self.spin_decay.setSingleStep(0.1)
        self.spin_decay.setValue(self.default_decay)
        self.spin_decay.setPrefix("Decay: ")
        
        settings_layout.addWidget(self.spin_n)
        settings_layout.addWidget(self.spin_decay)
        settings_layout.addStretch()
        
        layout.addLayout(settings_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Variable", "Power", "Coefficient", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

        btn_add = QPushButton("Add Term (+)")
        btn_add.clicked.connect(self.add_term_row)
        layout.addWidget(btn_add)

        self.lbl_preview = QLabel("f(x,y,z) = 0")
        self.lbl_preview.setStyleSheet("font-family: Consolas; font-weight: bold; color: #333; padding: 10px; background: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(self.lbl_preview)

        btn_save = QPushButton("Generate Dictionary")
        btn_save.setStyleSheet("font-weight: bold; background-color: #4caf50; color: white; padding: 8px;")
        btn_save.clicked.connect(self.save_polynomial)
        layout.addWidget(btn_save)

        self.add_term_row()

    def add_term_row(self, var_idx=0, power=1, coeff=1.0):
        row = self.table.rowCount()
        self.table.insertRow(row)

        cmb_var = QComboBox()
        cmb_var.addItems(["x", "y", "z"])
        cmb_var.setCurrentIndex(var_idx)
        cmb_var.currentIndexChanged.connect(self.update_preview)
        self.table.setCellWidget(row, 0, cmb_var)

        sb_power = QSpinBox()
        sb_power.setRange(0, 100) 
        sb_power.setValue(power)
        sb_power.valueChanged.connect(self.update_preview)
        self.table.setCellWidget(row, 1, sb_power)

        sb_coeff = QDoubleSpinBox()
        sb_coeff.setRange(-1000.0, 1000.0)
        sb_coeff.setSingleStep(0.1)
        sb_coeff.setDecimals(3)
        sb_coeff.setValue(coeff)
        sb_coeff.valueChanged.connect(self.update_preview)
        self.table.setCellWidget(row, 2, sb_coeff)

        btn_del = QPushButton("X")
        btn_del.setStyleSheet("color: red; font-weight: bold;")
        btn_del.clicked.connect(lambda: self.remove_row(btn_del))
        self.table.setCellWidget(row, 3, btn_del)
        
        self.update_preview()

    def remove_row(self, btn_widget):
        index = self.table.indexAt(btn_widget.pos())
        if index.isValid():
            self.table.removeRow(index.row())
            self.update_preview()

    def get_data(self):

        indices = []
        coefficients = []
        
        rows = self.table.rowCount()
        for r in range(rows):
            cmb_var = self.table.cellWidget(r, 0)
            sb_power = self.table.cellWidget(r, 1)
            sb_coeff = self.table.cellWidget(r, 2)
            
            var_idx = cmb_var.currentIndex()
            power = sb_power.value()
            coeff = sb_coeff.value()
            
            indices.append([var_idx, power])
            coefficients.append(coeff)
            
        return {
            'indices': indices,
            'coefficients': coefficients,
            'n': self.spin_n.value(),
            'decay': self.spin_decay.value()
        }

    def load_data(self, encoded_polynom):
        """
        Lädt ein bestehendes Dictionary in das Widget.
        """
        self.table.setRowCount(0) 
        
        indices = encoded_polynom.get('indices', [])
        coeffs = encoded_polynom.get('coefficients', [])
        
        if 'n' in encoded_polynom: self.spin_n.setValue(encoded_polynom['n'])
        if 'decay' in encoded_polynom: self.spin_decay.setValue(encoded_polynom['decay'])
        
        for (var_idx, power), coeff in zip(indices, coeffs):
            self.add_term_row(var_idx, power, coeff)
            
        self.update_preview()

    def update_preview(self):
        data = self.get_data()
        indices = data['indices']
        coeffs = data['coefficients']
        
        terms = []
        for (v_idx, p), c in zip(indices, coeffs):
            if abs(c) < 1e-9: continue # Skip near zero
            
            var_char = self.var_map.get(v_idx, '?')
            
            sign = "+ " if c >= 0 else "- "
            abs_c = abs(c)
            c_str = f"{abs_c:.2f}" if abs(abs_c - 1.0) > 1e-9 or p == 0 else "" 
            
            if p == 0:
                term_str = f"{c_str if c_str else '1.0'}"
            elif p == 1:
                term_str = f"{c_str}{var_char}"
            else:
                term_str = f"{c_str}{var_char}^{p}"
                
            terms.append((sign, term_str))
            
        full_str = ""
        for i, (sign, t) in enumerate(terms):
            if i == 0:
                if sign == "- ": full_str += "-" + t
                else: full_str += t
            else:
                full_str += f" {sign}{t}"
                
        if not full_str:
            full_str = "0"
            
        self.lbl_preview.setText(f"f(x,y,z) = {full_str}")

    def save_polynomial(self):
        data = self.get_data()
        print(f"Generated Polynomial Data: {data}")
        self.polynomial_saved.emit(data)


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
        
        # === HEADER ===
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Graph Name:"))
        self.graph_name_input = QLineEdit()
        self.graph_name_input.setPlaceholderText("e.g., Visual Cortex")
        header_layout.addWidget(self.graph_name_input)
        main_layout.addLayout(header_layout)
        
        # === MAIN CONTENT ===
        content_layout = QHBoxLayout()
        
        # === NODE COLUMN ===
        node_col = QVBoxLayout()
        node_col.addWidget(QLabel("NODES", alignment=Qt.AlignmentFlag.AlignCenter))
        
        node_scroll = QScrollArea()
        node_scroll.setWidgetResizable(True)
        self.node_list_widget = QWidget()
        self.node_list_layout = QVBoxLayout(self.node_list_widget)
        node_scroll.setWidget(self.node_list_widget)
        node_col.addWidget(node_scroll)
        
        # ✅ NUR ADD NODE BUTTON (kein Remove!)
        add_node_btn = QPushButton("+ Add Node")
        add_node_btn.clicked.connect(self.add_node)
        add_node_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        node_col.addWidget(add_node_btn)
        
        content_layout.addLayout(node_col, 2)
        
        # === POPULATION COLUMN ===
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
        
        # === EDITOR COLUMN ===
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
        
        # === BOTTOM BUTTONS ===
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
    
    def add_population(self):
        """Fügt Population zum aktuellen Node hinzu"""
        if self.current_node_idx is None:
            return
        
        self.save_current_population_params()
        
        node = self.node_list[self.current_node_idx]
        pop_idx = len(node['populations'])
        
        default_polynomials = {
            'x': generate_random_polynomial(max_degree=2, num_terms=3),
            'y': generate_random_polynomial(max_degree=2, num_terms=3),
            'z': generate_random_polynomial(max_degree=2, num_terms=3)
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
        
        # ✅ NEU: Übergebe Node-Index
        self.polynom_manager.set_populations(node['populations'], self.current_node_idx)
        self.editor_stack.setCurrentIndex(3)
    
    def on_polynomials_changed(self, node_idx, polynomials):
        """✅ NEU: Aktualisiere Polynome im Node"""
        if node_idx >= len(self.node_list):
            return
        
        node = self.node_list[node_idx]
        
        # Update polynomials in node data
        for i, poly_dict in enumerate(polynomials):
            if i < len(node['populations']):
                node['populations'][i]['polynomials'] = poly_dict
        
        print(f"✅ Updated polynomials for Node {node_idx}")
    
    def create_graph(self):
        global next_graph_id, graph_parameters
        
        if not self.graph_name_input.text():
            print("ERROR: Graph name required!")
            return
        
        if not self.node_list:
            print("ERROR: Add at least one node!")
            return
        
        for i, node in enumerate(self.node_list):
            if not node['populations']:
                print(f"ERROR: Node {i+1} has no populations! Please add populations to all nodes.")
                return
            
            prob_vec = node['params'].get('probability_vector', [])
            total_prob = sum(prob_vec)
            if abs(total_prob - 1.0) > 0.01:
                print(f"ERROR: Node {i+1} probability vector sums to {total_prob:.2f}, must be 1.0!")
                return
        
        # ✅ FIX: Save current node params BEFORE creating graph!
        if self.current_node_idx is not None:
            print(f"[DEBUG] Saving current node {self.current_node_idx} params before create...")
            current_params = self.node_param_widget.get_current_params()
            if 'center_of_mass' in current_params:
                current_params['m'] = current_params['center_of_mass'].copy()
            self.node_list[self.current_node_idx]['params'] = current_params
            print(f"[DEBUG] Node {self.current_node_idx} center_of_mass: {current_params.get('center_of_mass')}")
        
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
            populations = node['populations']
            
            # ✅ DEBUG: Check was wirklich in params steht
            print(f"[DEBUG] Node {node_idx} params:")
            print(f"  center_of_mass: {node['params'].get('center_of_mass')}")
            print(f"  m: {node['params'].get('m')}")
            print(f"  grid_size: {node['params'].get('grid_size')}")
            
            neuron_models = [pop['model'] for pop in populations]
            types = list(range(len(populations)))
            
            encoded_polynoms_per_type = []
            for pop in populations:
                poly_dict = pop.get('polynomials', None)
                if poly_dict and all(k in poly_dict for k in ['x', 'y', 'z']):
                    encoded_polynoms_per_type.append([poly_dict['x'], poly_dict['y'], poly_dict['z']])
                else:
                    encoded_polynoms_per_type.append([])
            
            prob_vec = node['params'].get('probability_vector', [])
            if not prob_vec and len(populations) > 0:
                prob_vec = [1.0/len(populations)] * len(populations)
            
            node_params = {
                'grid_size': node['params'].get('grid_size', [10, 10, 10]),
                'm': node['params'].get('center_of_mass', [0.0, 0.0, 0.0]),
                'center_of_mass': node['params'].get('center_of_mass', [0.0, 0.0, 0.0]),
                'displacement': node['params'].get('displacement', [0.0, 0.0, 0.0]),
                'displacement_factor': node['params'].get('displacement_factor', 1.0),
                'rot_theta': node['params'].get('rot_theta', 0.0),
                'rot_phi': node['params'].get('rot_phi', 0.0),
                'transform_matrix': node['params'].get('transform_matrix', [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                'dt': node['params'].get('dt', 0.01),
                'old': node['params'].get('old', True),
                'num_steps': node['params'].get('num_steps', 8),
                'sparse_holes': node['params'].get('sparse_holes', 0),
                'sparsity_factor': node['params'].get('sparsity_factor', 0.9),
                'probability_vector': prob_vec,
                'polynom_max_power': node['params'].get('polynom_max_power', 5),
                'name': node['params'].get('name', f'Node_{node_idx}'),
                'id': node_idx,
                'neuron_models': neuron_models,
                'types': types,
                'distribution': prob_vec,
                'encoded_polynoms_per_type': encoded_polynoms_per_type,
                'graph_id': graph_id,
                'grid_size': node['params'].get('grid_size', [10, 10, 10]),
                'm': node['params'].get('center_of_mass', [0.0, 0.0, 0.0]),
                'center_of_mass': node['params'].get('center_of_mass', [0.0, 0.0, 0.0]),
                'population_nest_params': [pop.get('params', {}) for pop in populations],
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
        
        print(f"✅ Graph '{self.graph_name_input.text()}' created successfully!")
        self.graphCreated.emit(graph_id)
        self.reset()
    def add_node(self):
        node_idx = len(self.node_list)
        node_btn = QPushButton(f"Node {node_idx + 1}")
        node_btn.setMinimumHeight(50)
        # ✅ Fix: Use default argument
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


    
    
    def save_node_params(self, params):
        """Auto-Save Node Params"""
        if self.current_node_idx is not None:
            # ✅ Sync: m = center_of_mass
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
        
        #  my Highlight <3
        node = self.node_list[self.current_node_idx]
        for i, pop in enumerate(node['populations']):
            if i == pop_idx:
                pop['button'].setStyleSheet("background-color: #2196F3; color: white;")
            else:
                pop['button'].setStyleSheet("")
        
        pop = node['populations'][pop_idx]
        if pop['params']:
            self.pop_param_widget.model_combo.setCurrentText(pop['model'])
            # TODO: Params in Widgets laden
        
        self.editor_stack.setCurrentIndex(2)
    
    def save_current_population_params(self):
        """Speichert aktuelle Population-Parameter"""
        # ✅ KRITISCH: Bounds-Check hinzufügen!
        if self.current_node_idx is not None and self.current_pop_idx is not None:
            # Check ob Index noch gültig ist (nach Graph-Wechsel könnte Liste neu sein)
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
        # Save current pop params before reset
        self.save_current_population_params()
        
        self.graph_name_input.clear()
        self.node_list.clear()
        
        while self.node_list_layout.count():
            item = self.node_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        while self.pop_list_layout.count():
            item = self.pop_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.current_node_idx = None
        self.current_pop_idx = None
        self.editor_stack.setCurrentIndex(0)
        self.add_pop_btn.setEnabled(False)




class EditGraphWidget(QWidget):
    """Widget zum Bearbeiten existierender Graphs"""
    
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
        
        # === GRAPH SELECTOR ===
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select Graph to Edit:"))
        
        self.graph_selector = QComboBox()
        self.graph_selector.currentIndexChanged.connect(self.on_graph_selected)
        selector_layout.addWidget(self.graph_selector)
        
        refresh_btn = QPushButton("🔄 Refresh List")
        refresh_btn.clicked.connect(self.refresh_graph_list)
        selector_layout.addWidget(refresh_btn)
        
        main_layout.addLayout(selector_layout)
        
        # === GRAPH NAME ===
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Graph Name:"))
        self.graph_name_input = QLineEdit()
        self.graph_name_input.setPlaceholderText("Edit graph name...")
        name_layout.addWidget(self.graph_name_input)
        main_layout.addLayout(name_layout)
        
        # === MAIN CONTENT ===
        content_layout = QHBoxLayout()
        
        # === NODE COLUMN ===
        node_col = QVBoxLayout()
        node_col.addWidget(QLabel("NODES", alignment=Qt.AlignmentFlag.AlignCenter))
        
        node_scroll = QScrollArea()
        node_scroll.setWidgetResizable(True)
        self.node_list_widget = QWidget()
        self.node_list_layout = QVBoxLayout(self.node_list_widget)
        node_scroll.setWidget(self.node_list_widget)
        node_col.addWidget(node_scroll)
        
        # ✅ ADD + REMOVE BUTTONS
        node_buttons_layout = QHBoxLayout()
        
        add_node_btn = QPushButton("+ Add")
        add_node_btn.clicked.connect(self.add_node)
        add_node_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        node_buttons_layout.addWidget(add_node_btn)
        
        self.remove_node_btn = QPushButton("🗑️ Remove")
        self.remove_node_btn.clicked.connect(self.remove_node)
        self.remove_node_btn.setEnabled(False)
        self.remove_node_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        node_buttons_layout.addWidget(self.remove_node_btn)
        
        node_col.addLayout(node_buttons_layout)
        
        content_layout.addLayout(node_col, 2)
        
        # === POPULATION COLUMN ===
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
        
        # === EDITOR COLUMN ===
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
        
        editor_col.addWidget(self.editor_stack)
        
        content_layout.addLayout(editor_col, 6)
        
        main_layout.addLayout(content_layout)
        
        # === BOTTOM BUTTONS ===
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
    def remove_node(self):
        """Entfernt ausgewählten Node – sicher und ohne Crash"""
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
            # 1. Button entfernen
            removed_node = self.node_list.pop(self.current_node_idx)
            if 'button' in removed_node:
                removed_node['button'].deleteLater()

            # 2. ✅ KRITISCH: Alle Buttons neu verbinden mit korrekten Indizes
            for i, node in enumerate(self.node_list):
                if 'button' in node:
                    # Disconnect old connections
                    try:
                        node['button'].clicked.disconnect()
                    except:
                        pass
                    
                    # Reconnect with correct index using default argument trick
                    node['button'].clicked.connect(lambda checked=False, idx=i: self.select_node(idx))
                    
                    # Update button text
                    name = node['params'].get('name', 'Node')
                    is_new = node.get('original_node') is None
                    suffix = " (NEW)" if is_new else ""
                    node['button'].setText(f"Node {i+1}{suffix}: {name}")

            # 3. UI zurücksetzen
            self.current_node_idx = None
            self.current_pop_idx = None
            self.editor_stack.setCurrentIndex(0)
            self.remove_node_btn.setEnabled(len(self.node_list) > 1)
            self.add_pop_btn.setEnabled(False)

            # Population-Liste leeren
            while self.pop_list_layout.count():
                item = self.pop_list_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            print(f"✅ Node '{node_name}' removed from UI (will be deleted on save)")




    def refresh_graph_list(self):
        """Aktualisiert Dropdown mit verfügbaren Graphs"""
        
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
        
        self.graph_selector.blockSignals(False)
        
        if self.graph_selector.count() > 0 and self.graph_list:
            self.on_graph_selected(0)
    
    def on_graph_selected(self, index):
        """Lädt ausgewählten Graph in Editor"""
        
        if not self.graph_list or index < 0:
            return
        
        graph_id = self.graph_selector.currentData()
        if graph_id is None:
            return
        
        # Find graph
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
        """Lädt alle Daten des aktuellen Graphs"""
        if not self.current_graph:
            return
        
        print(f"\n=== Loading Graph {self.current_graph_id} ===")
        
        # ✅ KRITISCH: Indices zurücksetzen BEVOR wir irgendwas machen!
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
        
        print(f"✅ Loaded {len(self.node_list)} nodes")
        
        # Select first node
        if self.node_list:
            self.select_node(0)
    
    def load_node_from_graph(self, node):
        """Lädt einen Node aus dem Graph"""
        node_idx = len(self.node_list)
        
        # Create button
        node_btn = QPushButton(f"Node {node_idx + 1}: {node.name}")
        node_btn.setMinimumHeight(50)
        # ✅ Fix: Use default argument to capture current value
        node_btn.clicked.connect(lambda checked=False, idx=node_idx: self.select_node(idx))
        self.node_list_layout.addWidget(node_btn)
        
        # Extract populations from NEST
        populations = []
        if hasattr(node, 'population') and node.population:
            for pop_idx, nest_pop in enumerate(node.population):
                if nest_pop is None or len(nest_pop) == 0:
                    continue
                
                # Get model
                model = nest.GetStatus(nest_pop, 'model')[0]
                
                # Get params (nur settable)
                params = {}
                try:
                    status = nest.GetStatus(nest_pop)
                    if status:
                        first_neuron = status[0]
                        # Standard LIF params
                        param_keys = ['V_m', 'E_L', 'tau_m', 'C_m', 'V_th', 
                                     't_ref', 'V_reset', 'I_e', 'tau_syn_ex', 'tau_syn_in']
                        for key in param_keys:
                            if key in first_neuron:
                                params[key] = first_neuron[key]
                except Exception as e:
                    print(f"⚠️  Could not extract params from pop {pop_idx}: {e}")
                
                # Get polynomials (from node.parameters if available)
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
        
        # Store node data
        node_data = {
            'params': node.parameters.copy() if hasattr(node, 'parameters') else {},
            'populations': populations,
            'button': node_btn,
            'original_node': node  # Reference zum Original
        }
        
        # Update params with current state
        node_data['params'].update({
            'id': node.id,
            'name': node.name,
            'center_of_mass': node.center_of_mass.tolist(),
            'm': node.center_of_mass.tolist(),
            'probability_vector': node.parameters.get('probability_vector', [])
        })
        
        self.node_list.append(node_data)
        
        print(f"  Loaded Node {node.id}: {node.name} ({len(populations)} populations)")
    
    def add_node(self):
        """Fügt neuen Node hinzu"""
        node_idx = len(self.node_list)
        node_btn = QPushButton(f"Node {node_idx + 1} (NEW)")
        node_btn.setMinimumHeight(50)
        # ✅ Fix: Use default argument
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
            'original_node': None  # Marker für neuen Node
        })
        
        self.select_node(node_idx)
    
    def select_node(self, node_idx):
        """Wählt Node aus und lädt seine Daten"""
        # ✅ Safety: Bounds check
        if node_idx < 0 or node_idx >= len(self.node_list):
            print(f"⚠️  Invalid node index {node_idx} (max: {len(self.node_list)-1})")
            return
        
        self.save_current_population_params()
        
        self.current_node_idx = node_idx
        self.current_pop_idx = None
        
        # Highlight
        for i, node in enumerate(self.node_list):
            if i == node_idx:
                node['button'].setStyleSheet("background-color: #2196F3; color: white;")
            else:
                node['button'].setStyleSheet("")
        self.remove_node_btn.setEnabled(len(self.node_list) > 1)

        
        # Load params
        num_pops = len(self.node_list[node_idx]['populations'])
        self.node_param_widget.set_population_count(num_pops)
        self.node_param_widget.load_data(self.node_list[node_idx]['params'])
        self.editor_stack.setCurrentIndex(1)
        
        self.update_population_list()
        self.add_pop_btn.setEnabled(True)
    
    def save_node_params(self, params):
        """Auto-Save Node Params"""
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
        """Fügt Population hinzu"""
        if self.current_node_idx is None:
            return
        
        self.save_current_population_params()
        
        node = self.node_list[self.current_node_idx]
        pop_idx = len(node['populations'])
        
        default_polynomials = {
            'x': generate_random_polynomial(max_degree=2, num_terms=3),
            'y': generate_random_polynomial(max_degree=2, num_terms=3),
            'z': generate_random_polynomial(max_degree=2, num_terms=3)
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
    
    def update_population_list(self):
        """Update Population List"""
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
        """Wählt Population aus"""
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
        """Speichert aktuelle Population-Parameter"""
        # ✅ KRITISCH: Bounds-Check BEVOR wir auf node_list zugreifen!
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
        """Öffnet Polynomial Editor"""
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
        """Speichert Änderungen zurück zum Graph"""
        if not self.current_graph:
            print("ERROR: No graph selected!")
            return
        
        if not self.graph_name_input.text():
            print("ERROR: Graph name required!")
            return
        
        # Validate
        for i, node in enumerate(self.node_list):
            if not node['populations']:
                print(f"ERROR: Node {i+1} has no populations!")
                return
        
        self.save_current_population_params()
        
        # Save polynomials
        if hasattr(self, '_last_polynomial_node_idx') and self._last_polynomial_node_idx is not None:
            last_node = self.node_list[self._last_polynomial_node_idx]
            all_polynomials = self.polynom_manager.get_all_polynomials()
            for i, poly_dict in enumerate(all_polynomials):
                if i < len(last_node['populations']):
                    last_node['populations'][i]['polynomials'] = poly_dict
        
        print(f"\n=== Saving changes to Graph {self.current_graph_id} ===")
        
        # Update graph name
        self.current_graph.graph_name = self.graph_name_input.text()
        
        import nest
        import numpy as np
        
        # Analyse der Änderungen
        edited_analysis = [self._node_was_edited(n) for n in self.node_list]
        nodes_with_structural_changes = [
            (n, i) for i, (n, (_, _, structural)) in enumerate(zip(self.node_list, edited_analysis)) 
            if structural
        ]
        
        # Indizes für schnelles Lookup
        structural_change_indices = {idx for _, idx in nodes_with_structural_changes}

        print("✏️  Applying changes - NEST reset required")
        nest.ResetKernel()
        
        self.current_graph.node_list.clear()
        self.current_graph.nodes = 0
        self.current_graph._next_id = 0
        
        for node_idx, node_data in enumerate(self.node_list):
            node_params = self._build_node_params(node_idx, node_data)
            original_node = node_data.get('original_node')
            
            # Node erstellen (ohne auto_build)
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
            
            # ================================
            # ✅ KORRIGIERTE ENTSCHEIDUNGSLOGIK
            # ================================
            has_structural_change = node_idx in structural_change_indices
            
            # Prüfen, ob das Original überhaupt valide Positionen hat
            original_has_points = False
            if original_node and hasattr(original_node, 'positions') and original_node.positions:
                # Prüfen ob mindestens ein Cluster Punkte enthält
                original_has_points = any(len(c) > 0 for c in original_node.positions if c is not None)

            # REBUILD ERZWINGEN wenn:
            # 1. Struktur geändert wurde (Grid, WFC Parameter etc.)
            # 2. ODER das Original keine Punkte hatte (Leere Hülle/Fehlerhafter Build)
            if has_structural_change or not original_has_points:
                reason = "Structural change" if has_structural_change else "Original positions empty/invalid"
                print(f"  Node {node_idx}: REBUILD ({reason})")
                try:
                    new_node.build()  # WFC neu ausführen
                except Exception as e:
                    print(f"  ❌ Build failed for Node {node_idx}: {e}")
            
            elif original_node:
                # Nur hier ist es sicher zu kopieren oder zu translatieren
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
            
            # Populate NEST (erstellt Neuronen basierend auf positions)
            new_node.populate_node()
        
        # Andere Graphen repopulieren (da NEST reset wurde)
        for graph in self.graph_list:
            if graph.graph_id == self.current_graph_id:
                continue
            for node in graph.node_list:
                node.populate_node()
        
        print(f"✅ Graph '{self.current_graph.graph_name}' updated!")
        
        self.graphUpdated.emit(-1)
        self.refresh_graph_list()

    def _node_was_edited(self, node_data):
        """
        Prüft ob Node-Parameter sich geändert haben.
        Returns: (any_change, polynomials_changed, structural_changed)
        
        structural_changed = Änderungen die WFC rebuild erfordern:
            - grid_size, num_steps, sparsity, dt, displacement, rotation, etc.
        """
        original = node_data.get('original_node')
        if not original:
            return False, False, False  # Neuer Node
        
        any_change = False
        polynomials_changed = False
        structural_changed = False
        
        if not hasattr(original, 'parameters'):
            return False, False, False
        
        old_params = original.parameters
        new_params = node_data['params']
        
        # ================================
        # STRUCTURAL CHANGES (require rebuild)
        # ================================
        structural_keys = [
            'grid_size', 'num_steps', 'sparsity_factor', 'sparse_holes',
            'dt', 'displacement', 'displacement_factor', 
            'rot_theta', 'rot_phi', 'polynom_max_power'
        ]
        
        for key in structural_keys:
            old_val = old_params.get(key)
            new_val = new_params.get(key)
            
            if old_val != new_val:
                print(f"    Structural: {key} changed: {old_val} → {new_val}")
                any_change = True
                structural_changed = True
        
        # ================================
        # CENTER OF MASS (translation only)
        # ================================
        old_com = np.array(original.center_of_mass)
        new_com = np.array(new_params['center_of_mass'])
        if not np.allclose(old_com, new_com, atol=1e-6):
            print(f"    COM changed: {old_com} → {new_com}")
            any_change = True
            # structural_changed bleibt False (nur Translation nötig)
        
        # ================================
        # POPULATION CHANGES
        # ================================
        old_pop_count = len(original.population) if hasattr(original, 'population') and original.population else 0
        new_pop_count = len(node_data['populations'])
        if old_pop_count != new_pop_count:
            print(f"    Population count changed: {old_pop_count} → {new_pop_count}")
            any_change = True
            structural_changed = True  # Neue Populations = rebuild
        
        # ================================
        # NEURON MODELS
        # ================================
        old_models = original.neuron_models if hasattr(original, 'neuron_models') else []
        new_models = [pop['model'] for pop in node_data['populations']]
        if old_models != new_models:
            print(f"    Neuron models changed")
            any_change = True
            structural_changed = True
        
        # ================================
        # POPULATION PARAMETERS
        # ================================
        if 'population_nest_params' in old_params:
            old_nest_params = old_params['population_nest_params']
            new_nest_params = [pop.get('params', {}) for pop in node_data['populations']]
            for i, (old_p, new_p) in enumerate(zip(old_nest_params, new_nest_params)):
                if old_p != new_p:
                    print(f"    Population {i} NEST parameters changed")
                    any_change = True
                    # NEST params ändern sich ohne rebuild (nur re-populate)
        
        # ================================
        # PROBABILITY VECTOR
        # ================================
        old_prob = old_params.get('probability_vector', [])
        new_prob = new_params.get('probability_vector', [])
        if old_prob != new_prob:
            print(f"    Probability vector changed")
            any_change = True
            structural_changed = True  # Distribution ändert WFC
        
        # ================================
        # POLYNOMIALS
        # ================================
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
        """Helper: Baut node_params dict aus UI-Daten"""
        populations = node_data['populations']
        
        neuron_models = [pop['model'] for pop in populations]
        types = list(range(len(populations)))
        
        # Polynomials extrahieren
        encoded_polynoms_per_type = []
        for pop in populations:
            poly_dict = pop.get('polynomials', None)
            if poly_dict and all(k in poly_dict for k in ['x', 'y', 'z']):
                encoded_polynoms_per_type.append([poly_dict['x'], poly_dict['y'], poly_dict['z']])
            else:
                encoded_polynoms_per_type.append([])
        
        # Probability Vector
        prob_vec = node_data['params'].get('probability_vector', [])
        if not prob_vec and len(populations) > 0:
            prob_vec = [1.0/len(populations)] * len(populations)
        
        return {
            'grid_size': node_data['params'].get('grid_size', [10, 10, 10]),
            'm': node_data['params'].get('center_of_mass', [0.0, 0.0, 0.0]),
            'center_of_mass': node_data['params'].get('center_of_mass', [0.0, 0.0, 0.0]),
            'displacement': node_data['params'].get('displacement', [0.0, 0.0, 0.0]),
            'displacement_factor': node_data['params'].get('displacement_factor', 1.0),
            'rot_theta': node_data['params'].get('rot_theta', 0.0),
            'rot_phi': node_data['params'].get('rot_phi', 0.0),
            'transform_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'dt': node_data['params'].get('dt', 0.01),
            'old': node_data['params'].get('old', True),
            'num_steps': node_data['params'].get('num_steps', 8),
            'sparse_holes': node_data['params'].get('sparse_holes', 0),
            'sparsity_factor': node_data['params'].get('sparsity_factor', 0.9),
            'probability_vector': prob_vec,
            'polynom_max_power': node_data['params'].get('polynom_max_power', 5),
            'name': node_data['params'].get('name', f'Node_{node_idx}'),
            'id': node_idx,
            'neuron_models': neuron_models,
            'types': types,
            'distribution': prob_vec,
            'encoded_polynoms_per_type': encoded_polynoms_per_type,
            'graph_id': self.current_graph_id,
            'population_nest_params': [pop.get('params', {}) for pop in populations],
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
            # Remove from list
            self.graph_list.remove(self.current_graph)
            print(f"✅ Graph {self.current_graph_id} deleted")
            
            # ✅ Reset NEST
            import nest
            nest.ResetKernel()
            print("🔄 NEST Kernel reset")
            
            # ✅ Repopulate remaining graphs (positions bleiben!)
            for graph in self.graph_list:
                print(f"  Repopulating Graph {graph.graph_id}: {graph.graph_name}")
                for node in graph.node_list:
                    node.populate_node()  # ✅ Nutzt existierende positions
            
            self.current_graph = None
            self.current_graph_id = None
            self.refresh_graph_list()
            self.graphUpdated.emit(-1)


# WidgetLib.py - Ganz am Ende hinzufügen/ersetzen

class ConnectionTool(QWidget):
    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list
        self.connections = []  # Temporäre Liste zum Bauen
        self.next_conn_id = 0
        self.current_conn_idx = None
        self.syn_param_widgets = {} # Speicher für dynamische Widgets
        self.init_ui()
    
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # === COLUMN 1: SOURCE & TARGET ===
        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("SOURCE", alignment=Qt.AlignmentFlag.AlignCenter))
        
        # Source Graph
        source_graph_layout = QHBoxLayout()
        source_graph_layout.addWidget(QLabel("Graph:"))
        self.source_graph_combo = QComboBox()
        self.source_graph_combo.currentIndexChanged.connect(self.on_source_graph_changed)
        source_graph_layout.addWidget(self.source_graph_combo)
        left_col.addLayout(source_graph_layout)
        
        # Source Node
        source_node_layout = QHBoxLayout()
        source_node_layout.addWidget(QLabel("Node:"))
        self.source_node_combo = QComboBox()
        self.source_node_combo.currentIndexChanged.connect(self.on_source_node_changed)
        source_node_layout.addWidget(self.source_node_combo)
        left_col.addLayout(source_node_layout)
        
        # Source Population
        source_pop_layout = QHBoxLayout()
        source_pop_layout.addWidget(QLabel("Population:"))
        self.source_pop_combo = QComboBox()
        source_pop_layout.addWidget(self.source_pop_combo)
        left_col.addLayout(source_pop_layout)
        
        left_col.addSpacing(20)
        
        # Target Section
        left_col.addWidget(QLabel("TARGET", alignment=Qt.AlignmentFlag.AlignCenter))
        
        # Target Graph
        target_graph_layout = QHBoxLayout()
        target_graph_layout.addWidget(QLabel("Graph:"))
        self.target_graph_combo = QComboBox()
        self.target_graph_combo.currentIndexChanged.connect(self.on_target_graph_changed)
        target_graph_layout.addWidget(self.target_graph_combo)
        left_col.addLayout(target_graph_layout)
        
        # Target Node
        target_node_layout = QHBoxLayout()
        target_node_layout.addWidget(QLabel("Node:"))
        self.target_node_combo = QComboBox()
        self.target_node_combo.currentIndexChanged.connect(self.on_target_node_changed)
        target_node_layout.addWidget(self.target_node_combo)
        left_col.addLayout(target_node_layout)
        
        # Target Population
        target_pop_layout = QHBoxLayout()
        target_pop_layout.addWidget(QLabel("Population:"))
        self.target_pop_combo = QComboBox()
        target_pop_layout.addWidget(self.target_pop_combo)
        left_col.addLayout(target_pop_layout)
        
        left_col.addStretch()
        main_layout.addLayout(left_col, 2)
        
        # === COLUMN 2: PARAMETERS ===
        middle_col = QVBoxLayout()
        middle_col.addWidget(QLabel("CONNECTION PARAMETERS", alignment=Qt.AlignmentFlag.AlignCenter))
        
        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_widget = QWidget()
        self.params_layout = QVBoxLayout(params_widget) # Referenz speichern für dynamische Felder
        
        # Connection Name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Connection Name:"))
        self.conn_name_input = QLineEdit()
        self.conn_name_input.setPlaceholderText("Auto-generated if empty")
        name_layout.addWidget(self.conn_name_input)
        self.params_layout.addLayout(name_layout)
        
        # Connection Rule
        rule_layout = QHBoxLayout()
        rule_layout.addWidget(QLabel("Rule:"))
        self.rule_combo = QComboBox()
        self.rule_combo.addItems([
            "all_to_all",
            "one_to_one",
            "fixed_indegree",
            "fixed_outdegree",
            "fixed_total_number",
            "pairwise_bernoulli",
            "pairwise_bernoulli_on_source",
            "pairwise_bernoulli_on_target",
            "symmetric_pairwise_bernoulli"
        ])
        self.rule_combo.currentTextChanged.connect(self.on_rule_changed)
        rule_layout.addWidget(self.rule_combo)
        self.params_layout.addLayout(rule_layout)
        
        # === DYNAMIC RULE PARAMETERS ===
        
        # Indegree
        self.indegree_layout = QHBoxLayout()
        self.indegree_layout.addWidget(QLabel("Indegree:"))
        self.indegree_spin = QSpinBox()
        self.indegree_spin.setRange(1, 10000)
        self.indegree_spin.setValue(100)
        self.indegree_layout.addWidget(self.indegree_spin)
        self.params_layout.addLayout(self.indegree_layout)
        
        # Outdegree
        self.outdegree_layout = QHBoxLayout()
        self.outdegree_layout.addWidget(QLabel("Outdegree:"))
        self.outdegree_spin = QSpinBox()
        self.outdegree_spin.setRange(1, 10000)
        self.outdegree_spin.setValue(100)
        self.outdegree_layout.addWidget(self.outdegree_spin)
        self.params_layout.addLayout(self.outdegree_layout)
        
        # Total Number
        self.total_num_layout = QHBoxLayout()
        self.total_num_layout.addWidget(QLabel("Total Number:"))
        self.total_num_spin = QSpinBox()
        self.total_num_spin.setRange(1, 100000)
        self.total_num_spin.setValue(1000)
        self.total_num_layout.addWidget(self.total_num_spin)
        self.params_layout.addLayout(self.total_num_layout)
        
        # Probability
        self.prob_layout = QHBoxLayout()
        self.prob_layout.addWidget(QLabel("Probability:"))
        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setValue(0.1)
        self.prob_spin.setSingleStep(0.05)
        self.prob_spin.setDecimals(4)
        self.prob_layout.addWidget(self.prob_spin)
        self.params_layout.addLayout(self.prob_layout)
        
        # === STANDARD SYNAPSE PARAMETERS ===
        
        # Weight
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(QLabel("Weight:"))
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(-1000.0, 1000.0)
        self.weight_spin.setValue(1.0)
        self.weight_spin.setDecimals(3)
        weight_layout.addWidget(self.weight_spin)
        self.params_layout.addLayout(weight_layout)
        
        # Delay
        delay_layout = QHBoxLayout()
        delay_layout.addWidget(QLabel("Delay (ms):"))
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 100.0)
        self.delay_spin.setValue(1.0)
        self.delay_spin.setDecimals(2)
        delay_layout.addWidget(self.delay_spin)
        self.params_layout.addLayout(delay_layout)
        
        # Synapse Model Selection
        syn_layout = QHBoxLayout()
        syn_layout.addWidget(QLabel("Synapse Model:"))
        self.syn_model_combo = QComboBox()
        # Hole Keys aus der globalen synapse_models Variable
        self.syn_model_combo.addItems(sorted(synapse_models.keys()))
        self.syn_model_combo.currentTextChanged.connect(self.on_synapse_model_changed)
        syn_layout.addWidget(self.syn_model_combo)
        self.params_layout.addLayout(syn_layout)
        
        # === DYNAMIC SYNAPSE PARAMETER CONTAINER ===
        self.dynamic_syn_params_container = QWidget()
        self.dynamic_syn_params_layout = QVBoxLayout(self.dynamic_syn_params_container)
        self.dynamic_syn_params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.addWidget(self.dynamic_syn_params_container)
        
        # Allow Autapses
        autapses_layout = QHBoxLayout()
        self.allow_autapses_check = QCheckBox("Allow Autapses (self-connections)")
        self.allow_autapses_check.setChecked(True)
        autapses_layout.addWidget(self.allow_autapses_check)
        self.params_layout.addLayout(autapses_layout)
        
        # Allow Multapses
        multapses_layout = QHBoxLayout()
        self.allow_multapses_check = QCheckBox("Allow Multapses (multiple connections)")
        self.allow_multapses_check.setChecked(True)
        multapses_layout.addWidget(self.allow_multapses_check)
        self.params_layout.addLayout(multapses_layout)
        
        self.params_layout.addStretch()
        params_scroll.setWidget(params_widget)
        middle_col.addWidget(params_scroll)
        
        main_layout.addLayout(middle_col, 3)
        
        # === COLUMN 3: CONNECTIONS LIST ===
        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("CONNECTIONS", alignment=Qt.AlignmentFlag.AlignCenter))
        
        # Connections List
        conn_scroll = QScrollArea()
        conn_scroll.setWidgetResizable(True)
        self.conn_list_widget = QWidget()
        self.conn_list_layout = QVBoxLayout(self.conn_list_widget)
        conn_scroll.setWidget(self.conn_list_widget)
        right_col.addWidget(conn_scroll)
        
        # Buttons
        btn_layout = QVBoxLayout()
        
        self.add_conn_btn = QPushButton("+ Add Connection")
        self.add_conn_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.add_conn_btn.clicked.connect(self.add_connection)
        btn_layout.addWidget(self.add_conn_btn)
        
        self.delete_conn_btn = QPushButton("🗑️ Delete Connection")
        self.delete_conn_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
        self.delete_conn_btn.clicked.connect(self.delete_connection)
        self.delete_conn_btn.setEnabled(False)
        btn_layout.addWidget(self.delete_conn_btn)
        
        self.create_all_btn = QPushButton("🚀 CREATE ALL CONNECTIONS")
        self.create_all_btn.setMinimumHeight(60)
        self.create_all_btn.setStyleSheet(
            "background-color: #FF5722; color: white; font-weight: bold; font-size: 16px;"
        )
        self.create_all_btn.clicked.connect(self.create_all_connections)
        btn_layout.addWidget(self.create_all_btn)
        
        right_col.addLayout(btn_layout)
        main_layout.addLayout(right_col, 2)
        
        # Initial setup
        self.refresh_graph_list()
        self.on_rule_changed(self.rule_combo.currentText())
        # Trigger initial parameter load for default synapse model
        if self.syn_model_combo.count() > 0:
            self.on_synapse_model_changed(self.syn_model_combo.currentText())
    
    def on_rule_changed(self, rule):
        """Zeigt/versteckt Parameter je nach Rule"""
        self.hide_layout(self.indegree_layout)
        self.hide_layout(self.outdegree_layout)
        self.hide_layout(self.total_num_layout)
        self.hide_layout(self.prob_layout)
        
        if rule == "fixed_indegree":
            self.show_layout(self.indegree_layout)
        elif rule == "fixed_outdegree":
            self.show_layout(self.outdegree_layout)
        elif rule == "fixed_total_number":
            self.show_layout(self.total_num_layout)
        elif rule in ["pairwise_bernoulli", "pairwise_bernoulli_on_source", 
                      "pairwise_bernoulli_on_target", "symmetric_pairwise_bernoulli"]:
            self.show_layout(self.prob_layout)
            
    def on_synapse_model_changed(self, model_name):
        """Erstellt dynamisch Eingabefelder basierend auf dem Synapsenmodell"""
        
        # 1. Alte Widgets entfernen
        while self.dynamic_syn_params_layout.count():
            item = self.dynamic_syn_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.syn_param_widgets.clear()
        
        if model_name not in synapse_models:
            return

        params = synapse_models[model_name]
        if not params:
            return

        # Label für den Bereich
        header = QLabel(f"Parameters for {model_name}:")
        header.setStyleSheet("font-weight: bold; color: #555; margin-top: 5px;")
        self.dynamic_syn_params_layout.addWidget(header)

        # 2. Neue Widgets erstellen
        for param_name, info in params.items():
            p_type = info.get('type', 'float')
            p_default = info.get('default', 0.0)
            p_min = info.get('min')
            p_max = info.get('max')
            
            widget = None
            
            if p_type == 'float':
                # Fallback falls min/max None sind
                safe_min = p_min if p_min is not None else 0.0
                safe_max = p_max if p_max is not None else 10000.0
                
                widget = DoubleInputField(
                    param_name, 
                    default_value=float(p_default),
                    min_val=float(safe_min),
                    max_val=float(safe_max)
                )
            elif p_type == 'integer':
                safe_min = p_min if p_min is not None else 0
                safe_max = p_max if p_max is not None else 10000
                
                widget = IntegerInputField(
                    param_name,
                    default_value=int(p_default),
                    min_val=int(safe_min),
                    max_val=int(safe_max)
                )
            
            if widget:
                self.dynamic_syn_params_layout.addWidget(widget)
                self.syn_param_widgets[param_name] = widget

    def hide_layout(self, layout):
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.setVisible(False)
    
    def show_layout(self, layout):
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.setVisible(True)
    
    def refresh_graph_list(self):
        self.source_graph_combo.clear()
        self.target_graph_combo.clear()
        
        for graph in self.graph_list:
            graph_name = getattr(graph, 'graph_name', f'Graph {graph.graph_id}')
            self.source_graph_combo.addItem(f"{graph_name} (ID: {graph.graph_id})", graph.graph_id)
            self.target_graph_combo.addItem(f"{graph_name} (ID: {graph.graph_id})", graph.graph_id)
        
        if len(self.graph_list) > 0:
            self.on_source_graph_changed(0)
            self.on_target_graph_changed(0)
    
    def on_source_graph_changed(self, index):
        self.source_node_combo.clear()
        self.source_pop_combo.clear()
        
        if index < 0: return
        graph_id = self.source_graph_combo.currentData()
        if graph_id is None: return
        
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph: return
        
        for node in graph.node_list:
            node_name = getattr(node, 'name', f'Node {node.id}')
            self.source_node_combo.addItem(f"{node_name} (ID: {node.id})", node.id)
        
        if len(graph.node_list) > 0:
            self.on_source_node_changed(0)
    
    def on_target_graph_changed(self, index):
        self.target_node_combo.clear()
        self.target_pop_combo.clear()
        
        if index < 0: return
        graph_id = self.target_graph_combo.currentData()
        if graph_id is None: return
        
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph: return
        
        for node in graph.node_list:
            node_name = getattr(node, 'name', f'Node {node.id}')
            self.target_node_combo.addItem(f"{node_name} (ID: {node.id})", node.id)
        
        if len(graph.node_list) > 0:
            self.on_target_node_changed(0)
    
    def on_source_node_changed(self, index):
        self.source_pop_combo.clear()
        graph_id = self.source_graph_combo.currentData()
        node_id = self.source_node_combo.currentData()
        
        if graph_id is None or node_id is None: return
        
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph: return
        
        node = next((n for n in graph.node_list if n.id == node_id), None)
        if not node: return
        
        for i, pop in enumerate(node.population):
            model = node.parameters['neuron_models'][i] if i < len(node.parameters['neuron_models']) else 'unknown'
            self.source_pop_combo.addItem(f"Pop {i}: {model} ({len(pop)} neurons)", i)
    
    def on_target_node_changed(self, index):
        self.target_pop_combo.clear()
        graph_id = self.target_graph_combo.currentData()
        node_id = self.target_node_combo.currentData()
        
        if graph_id is None or node_id is None: return
        
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph: return
        
        node = next((n for n in graph.node_list if n.id == node_id), None)
        if not node: return
        
        for i, pop in enumerate(node.population):
            model = node.parameters['neuron_models'][i] if i < len(node.parameters['neuron_models']) else 'unknown'
            self.target_pop_combo.addItem(f"Pop {i}: {model} ({len(pop)} neurons)", i)
    def _execute_nest_connections(self):
        """
        Erstellt NEST Connections für alle Connections in self.connections.
        Wird aufgerufen NACH Speicherung zu Nodes, VOR Interface-Clear.
        """
        
        if not self.connections:
            return 0
        
        print("\n🔗 Executing NEST Connections...")
        print("-" * 50)
        
        successful = 0
        failed = 0
        
        for conn in self.connections:
            try:
                # === SOURCE POPULATION ===
                source_graph = next(
                    (g for g in self.graph_list if g.graph_id == conn['source']['graph_id']), 
                    None
                )
                if not source_graph:
                    raise ValueError(f"Source graph {conn['source']['graph_id']} not found")
                
                source_node = next(
                    (n for n in source_graph.node_list if n.id == conn['source']['node_id']), 
                    None
                )
                if not source_node:
                    raise ValueError(f"Source node {conn['source']['node_id']} not found")
                
                if conn['source']['pop_id'] >= len(source_node.population):
                    raise ValueError(f"Source population {conn['source']['pop_id']} out of range")
                
                source_pop = source_node.population[conn['source']['pop_id']]
                
                # === TARGET POPULATION ===
                target_graph = next(
                    (g for g in self.graph_list if g.graph_id == conn['target']['graph_id']), 
                    None
                )
                if not target_graph:
                    raise ValueError(f"Target graph {conn['target']['graph_id']} not found")
                
                target_node = next(
                    (n for n in target_graph.node_list if n.id == conn['target']['node_id']), 
                    None
                )
                if not target_node:
                    raise ValueError(f"Target node {conn['target']['node_id']} not found")
                
                if conn['target']['pop_id'] >= len(target_node.population):
                    raise ValueError(f"Target population {conn['target']['pop_id']} out of range")
                
                target_pop = target_node.population[conn['target']['pop_id']]
                
                # === CONNECTION SPEC ===
                params = conn['params']
                conn_spec = {'rule': params['rule']}
                
                # Rule-spezifische Parameter
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
                
                # === SYNAPSE SPEC ===
                syn_spec = {
                    'synapse_model': params.get('synapse_model', 'static_synapse'),
                    'weight': params.get('weight', 1.0),
                    'delay': params.get('delay', 1.0)
                }
                
                # ✅ Zusätzliche Synapse-Parameter (z.B. tau_plus, U, tau_rec, etc.)
                excluded_keys = {
                    'rule', 'indegree', 'outdegree', 'N', 'p',
                    'weight', 'delay', 'synapse_model',
                    'allow_autapses', 'allow_multapses'
                }
                
                for key, value in params.items():
                    if key not in excluded_keys:
                        syn_spec[key] = value
                
                # === NEST CONNECT ===
                nest.Connect(
                    source_pop,
                    target_pop,
                    conn_spec=conn_spec,
                    syn_spec=syn_spec
                )
                
                successful += 1
                print(f"  ✅ {conn['name']}")
                print(f"     {len(source_pop)} → {len(target_pop)} neurons")
                
            except Exception as e:
                failed += 1
                print(f"  ❌ FAILED: {conn['name']}")
                print(f"     Error: {e}")
        
        print("-" * 50)
        print(f"✅ NEST: {successful} successful, {failed} failed")
        
        return successful
    def add_connection(self):
        """Fügt Connection zur temporären Liste hinzu - MIT VOLLSTÄNDIGER VALIDATION"""
        
        # ===== 1. SOURCE & TARGET SAMMELN =====
        source_graph_id = self.source_graph_combo.currentData()
        source_node_id = self.source_node_combo.currentData()
        source_pop_id = self.source_pop_combo.currentData()
        
        target_graph_id = self.target_graph_combo.currentData()
        target_node_id = self.target_node_combo.currentData()
        target_pop_id = self.target_pop_combo.currentData()
        
        if source_graph_id is None or target_graph_id is None:
            print("⚠ Please select source and target!")
            return
        
        # ===== 2. CONNECTION NAME =====
        conn_name = self.conn_name_input.text().strip()
        if not conn_name:
            conn_name = f"Conn_{self.next_conn_id}: G{source_graph_id}N{source_node_id}P{source_pop_id} → G{target_graph_id}N{target_node_id}P{target_pop_id}"
        
        # ===== 3. RULE-SPEZIFISCHE PARAMETER =====
        rule = self.rule_combo.currentText()
        rule_params = {}
        
        if rule == "fixed_indegree":
            rule_params['indegree'] = self.indegree_spin.value()
        elif rule == "fixed_outdegree":
            rule_params['outdegree'] = self.outdegree_spin.value()
        elif rule == "fixed_total_number":
            rule_params['N'] = self.total_num_spin.value()
        elif rule in ["pairwise_bernoulli", "pairwise_bernoulli_on_source", 
                    "pairwise_bernoulli_on_target", "symmetric_pairwise_bernoulli"]:
            rule_params['p'] = self.prob_spin.value()
        
        # ===== 4. WEIGHT & DELAY VALIDATION =====
        weight = self.weight_spin.value()
        delay = self.delay_spin.value()
        
        # ✅ CRITICAL: Delay muss >= NEST resolution sein
        try:
            import nest
            min_delay = nest.resolution  # Default: 0.1 ms
            if delay < min_delay:
                print(f"⚠️  Delay {delay} ms < NEST resolution {min_delay} ms!")
                print(f"   → Adjusting to {min_delay} ms")
                delay = min_delay
                self.delay_spin.setValue(delay)  # Update UI
        except Exception as e:
            print(f"⚠️  Could not validate delay against NEST: {e}")
            if delay < 0.1:
                print(f"   → Using minimum 0.1 ms")
                delay = 0.1
        
        # Optional: Weight warning (nicht kritisch, aber hilfreich)
        if abs(weight) < 1e-6:
            print(f"⚠️  Warning: Weight {weight} is very small (near zero)")
        
        # ===== 5. SYNAPSE MODEL & PARAMETER VALIDATION =====
        syn_model = self.syn_model_combo.currentText()
        
        # Sammle raw parameters von UI
        raw_syn_params = {}
        for param_name, widget in self.syn_param_widgets.items():
            raw_syn_params[param_name] = widget.get_value()
        
        # ✅ VALIDATE gegen synapse_models dict
        validated_syn_params = {}
        
        if syn_model in synapse_models:
            model_spec = synapse_models[syn_model]
            
            for param_name, value in raw_syn_params.items():
                # Check ob Parameter für dieses Modell gültig ist
                if param_name not in model_spec:
                    print(f"⚠️  Parameter '{param_name}' not valid for {syn_model}, skipping")
                    continue
                
                param_info = model_spec[param_name]
                original_value = value
                
                # ✅ MIN Constraint
                if 'min' in param_info and param_info['min'] is not None:
                    if value < param_info['min']:
                        print(f"⚠️  {param_name}={value} < min={param_info['min']}, clamping")
                        value = param_info['min']
                
                # ✅ MAX Constraint
                if 'max' in param_info and param_info['max'] is not None:
                    if value > param_info['max']:
                        print(f"⚠️  {param_name}={value} > max={param_info['max']}, clamping")
                        value = param_info['max']
                
                # ✅ Constraint-Typ spezifisch
                constraint = param_info.get('constraint')
                if constraint == 'positive' and value <= 0:
                    print(f"⚠️  {param_name}={value} must be positive, using default")
                    value = param_info.get('default', 0.1)
                elif constraint == 'non-negative' and value < 0:
                    print(f"⚠️  {param_name}={value} must be non-negative, clamping to 0")
                    value = 0.0
                elif constraint == 'range':
                    # Already handled by min/max above
                    pass
                
                validated_syn_params[param_name] = value
                
                # Update UI wenn geclampt wurde
                if abs(value - original_value) > 1e-9:
                    if param_name in self.syn_param_widgets:
                        widget = self.syn_param_widgets[param_name]
                        if hasattr(widget, 'spinbox'):
                            widget.spinbox.blockSignals(True)
                            widget.spinbox.setValue(value)
                            widget.spinbox.blockSignals(False)
            
            # Zeige Validation Summary
            if validated_syn_params:
                print(f"✓ Validated {len(validated_syn_params)} synapse parameters for {syn_model}")
            else:
                print(f"ℹ️  No additional synapse parameters for {syn_model}")
        
        else:
            # Unbekanntes Modell - nimm was da ist (mit Warnung)
            print(f"⚠️  Unknown synapse model '{syn_model}', parameters not validated")
            validated_syn_params = raw_syn_params
        
        # ===== 6. BUILD FINAL CONNECTION DICT =====
        connection = {
            'id': self.next_conn_id,
            'name': conn_name,
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
                **rule_params,
                'weight': weight,  # ✅ Validated
                'delay': delay,    # ✅ Validated against NEST resolution
                'synapse_model': syn_model,
                'allow_autapses': self.allow_autapses_check.isChecked(),
                'allow_multapses': self.allow_multapses_check.isChecked(),
                **validated_syn_params  # ✅ Validated synapse params
            }
        }
        
        # ===== 7. OPTIONAL: POPULATION SIZE CHECK =====
        # Warne wenn indegree/outdegree größer als Population
        if rule == "fixed_indegree" and 'indegree' in rule_params:
            target_graph = next((g for g in self.graph_list if g.graph_id == target_graph_id), None)
            if target_graph:
                target_node = next((n for n in target_graph.node_list if n.id == target_node_id), None)
                if target_node and hasattr(target_node, 'population'):
                    if target_pop_id < len(target_node.population):
                        target_pop = target_node.population[target_pop_id]
                        if len(target_pop) > 0:  # Check nur wenn bereits populiert
                            if rule_params['indegree'] > len(target_pop):
                                print(f"⚠️  Warning: Indegree {rule_params['indegree']} > target pop size {len(target_pop)}")
                                print(f"   Connection may fail when executed!")
        
        elif rule == "fixed_outdegree" and 'outdegree' in rule_params:
            source_graph = next((g for g in self.graph_list if g.graph_id == source_graph_id), None)
            if source_graph:
                source_node = next((n for n in source_graph.node_list if n.id == source_node_id), None)
                if source_node and hasattr(source_node, 'population'):
                    if source_pop_id < len(source_node.population):
                        source_pop = source_node.population[source_pop_id]
                        if len(source_pop) > 0:
                            if rule_params['outdegree'] > len(source_pop):
                                print(f"⚠️  Warning: Outdegree {rule_params['outdegree']} > source pop size {len(source_pop)}")
                                print(f"   Connection may fail when executed!")
        
        # ===== 8. ADD TO LIST =====
        self.connections.append(connection)
        self.next_conn_id += 1
        
        print(f"\n✅ Connection added: {conn_name}")
        print(f"   Rule: {rule}, Weight: {weight}, Delay: {delay} ms")
        if validated_syn_params:
            print(f"   Synapse params: {list(validated_syn_params.keys())}")
        
        self.update_connection_list()
        self.reset_interface()
    
    def create_all_connections(self):
        """✅ Schreibt Connections zu Source-Nodes UND erstellt NEST Connections"""
        if not self.connections:
            print("⚠ No connections to create!")
            return
        
        print("CREATING ALL CONNECTIONS")
        
        print("\nStoring to source nodes...")
        successful_storage = []
        failed_storage = []
        
        for conn in self.connections:
            try:
                source_graph_id = conn['source']['graph_id']
                source_node_id = conn['source']['node_id']
                
                graph = next((g for g in self.graph_list if g.graph_id == source_graph_id), None)
                if not graph:
                    raise ValueError(f"Graph {source_graph_id} not found!")
                
                node = next((n for n in graph.node_list if n.id == source_node_id), None)
                if not node:
                    raise ValueError(f"Node {source_node_id} not found!")
                
                if not hasattr(node, 'connections'):
                    node.connections = []
                
                node.connections.append(conn)
                successful_storage.append(conn)
                
                print(f"  ✅ {conn['name']} → Node {source_node_id}")
                
            except Exception as e:
                print(f"  ❌ FAILED: {conn['name']}: {e}")
                failed_storage.append(conn)
        
        # === SCHRITT 2: Erstelle NEST Connections (NUR für erfolgreich gespeicherte) ===
        if successful_storage:
            # Temporär nur erfolgreiche Connections in self.connections
            original_connections = self.connections
            self.connections = successful_storage
            
            num_created = self._execute_nest_connections()
            
            # Restore
            self.connections = original_connections
        
        # === SCHRITT 3: Entscheide was passiert ===
        print("\n" + "="*70)
        
        if failed_storage:
            print(f"❌ {len(failed_storage)} connections FAILED to store")
            print("   Keeping failed connections in list for retry")
            print("="*70 + "\n")
            
            self.connections = failed_storage  # Nur failed behalten
            self.update_connection_list()
        else:
            print(f"✅ All {len(successful_storage)} connections stored & executed!")
            print("="*70 + "\n")
            
            # ✅ Clear Interface nur wenn ALLE erfolgreich
            self.connections.clear()
            self.next_conn_id = 0
            self.current_conn_idx = None
            self.update_connection_list()
            self.reset_interface()
            
            print("🧹 Interface cleared - ready for new connections\n")
    
    def delete_connection(self):
        if self.current_conn_idx is None: return
        
        conn = self.connections[self.current_conn_idx]
        print(f"\n🗑️ Deleting: {conn['name']}\n")
        
        del self.connections[self.current_conn_idx]
        self.current_conn_idx = None
        self.delete_conn_btn.setEnabled(False)
        
        self.update_connection_list()
    
    def update_connection_list(self):
        while self.conn_list_layout.count():
            item = self.conn_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for i, conn in enumerate(self.connections):
            btn = QPushButton(conn['name'])
            btn.setMinimumHeight(40)
            btn.clicked.connect(lambda checked, idx=i: self.select_connection(idx))
            self.conn_list_layout.addWidget(btn)
        
        self.conn_list_layout.addStretch()
    
    def select_connection(self, idx):
        self.current_conn_idx = idx
        self.delete_conn_btn.setEnabled(True)
        
        for i in range(self.conn_list_layout.count() - 1):
            widget = self.conn_list_layout.itemAt(i).widget()
            if widget:
                if i == idx:
                    widget.setStyleSheet("background-color: #2196F3; color: white;")
                else:
                    widget.setStyleSheet("")
        
        conn = self.connections[idx]
        print(f"\n📍 Selected: {conn['name']}")
    
    def refresh(self):
        self.refresh_graph_list()
    
    def reset_interface(self):
        self.conn_name_input.clear()
        self.rule_combo.setCurrentIndex(0)
        self.indegree_spin.setValue(100)
        self.outdegree_spin.setValue(100)
        self.total_num_spin.setValue(1000)
        self.prob_spin.setValue(0.1)
        self.weight_spin.setValue(1.0)
        self.delay_spin.setValue(1.0)
        
        # Reset Synapse Model Combo to a default safe state (e.g. static_synapse if exists)
        idx = self.syn_model_combo.findText("static_synapse")
        if idx >= 0:
            self.syn_model_combo.setCurrentIndex(idx)
        elif self.syn_model_combo.count() > 0:
            self.syn_model_combo.setCurrentIndex(0)
            
        self.allow_autapses_check.setChecked(True)
        self.allow_multapses_check.setChecked(True)
        
        if self.source_graph_combo.count() > 0:
            self.source_graph_combo.setCurrentIndex(0)
        if self.target_graph_combo.count() > 0:
            self.target_graph_combo.setCurrentIndex(0)

"""
for graph in self.graph_list:
    for node in graph.node_list:
        if hasattr(node, 'connections'):
            for conn in node.connections:
                #  NEST Connections
                pass



"""
class BlinkingNetworkWidget(QWidget):
    """
    Visualisiert das Netzwerk mit echten Connections aus node.connections.
    Zeigt Neuronen als Punkte und Verbindungen als blinkende Linien.
    """
    
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # PyVista Setup
        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        self.layout.addWidget(self.plotter)
        
        # State Arrays
        self.neuron_mesh = None
        self.edge_mesh = None
        self.neuron_intensities = None
        self.edge_intensities = None
        self.base_colors = None
        
        # Connection tracking
        self.edge_connection_map = []  # Maps edge index to connection info
        
        # Timer Setup
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.is_active = True # Flag hinzufügen
        # Initial Build
        self.build_scene()
        
        # Start Animation (ca. 30 FPS)
        self.timer.start(33)

    def build_scene(self):
        """Baut die Szene mit echten Connections."""
        self.plotter.clear()
        
        if not self.graph_list:
            return
        
        # =================================================================
        # 1. NEURONEN SAMMELN (wie vorher)
        # =================================================================
        all_points = []
        all_colors = []
        
        # Mapping: (graph_id, node_id, pop_id) -> (start_idx, end_idx) in points array
        population_indices = {}
        current_idx = 0
        
        for graph in self.graph_list:
            for node in graph.node_list:
                if not hasattr(node, 'positions') or not node.positions:
                    continue
                    
                for pop_idx, cluster in enumerate(node.positions):
                    if cluster is None or len(cluster) == 0:
                        continue
                    
                    # Model & Color
                    model = node.neuron_models[pop_idx] if pop_idx < len(node.neuron_models) else "unknown"
                    hex_c = neuron_colors.get(model, "#ffffff")
                    rgb = mcolors.to_rgb(hex_c)
                    
                    # Track indices
                    start_idx = current_idx
                    end_idx = current_idx + len(cluster)
                    population_indices[(graph.graph_id, node.id, pop_idx)] = (start_idx, end_idx)
                    current_idx = end_idx
                    
                    all_points.append(cluster)
                    all_colors.extend([rgb] * len(cluster))
        
        if not all_points:
            print("[BlinkingNetwork] No points to visualize")
            return

        points = np.vstack(all_points)
        n_points = len(points)
        self.base_colors = np.array(all_colors)
        
        print(f"[BlinkingNetwork] {n_points} neurons from {len(population_indices)} populations")
        
        # =================================================================
        # 2. NEURONEN MESH
        # =================================================================
        self.neuron_mesh = pv.PolyData(points)
        self.neuron_intensities = np.random.uniform(0.3, 1.0, n_points)
        self.neuron_mesh.point_data["display_color"] = self.base_colors * self.neuron_intensities[:, None]
        
        self.plotter.add_mesh(
            self.neuron_mesh,
            scalars="display_color",
            rgb=True,
            point_size=6,
            render_points_as_spheres=True,
            ambient=0.5
        )
        
        # =================================================================
        # 3. ECHTE CONNECTIONS SAMMELN
        # =================================================================
        all_connections = []
        
        for graph in self.graph_list:
            for node in graph.node_list:
                if not hasattr(node, 'connections') or not node.connections:
                    continue
                
                for conn in node.connections:
                    all_connections.append(conn)
        
        print(f"[BlinkingNetwork] Found {len(all_connections)} connections")
        
        if not all_connections:
            # Keine Connections → keine Kanten
            self.edge_mesh = None
            self.edge_intensities = None
            self.plotter.reset_camera()
            return
        
        # =================================================================
        # 4. KANTEN AUS CONNECTIONS ERSTELLEN
        # =================================================================
        lines_data = []  # [(start_point_idx, end_point_idx), ...]
        self.edge_connection_map = []  # Track which connection each edge belongs to
        
        max_edges_per_connection = 30  # Begrenzt Anzahl pro Connection
        
        for conn_idx, conn in enumerate(all_connections):
            source = conn.get('source', {})
            target = conn.get('target', {})
            
            source_key = (source.get('graph_id'), source.get('node_id'), source.get('pop_id'))
            target_key = (target.get('graph_id'), target.get('node_id'), target.get('pop_id'))
            
            # Prüfe ob Source und Target existieren
            if source_key not in population_indices:
                print(f"  ⚠ Source {source_key} not found for connection '{conn.get('name', '?')}'")
                continue
            if target_key not in population_indices:
                print(f"  ⚠ Target {target_key} not found for connection '{conn.get('name', '?')}'")
                continue
            
            src_start, src_end = population_indices[source_key]
            tgt_start, tgt_end = population_indices[target_key]
            
            src_size = src_end - src_start
            tgt_size = tgt_end - tgt_start
            
            if src_size == 0 or tgt_size == 0:
                continue
            
            # Anzahl der zu zeichnenden Kanten berechnen
            # Basierend auf Connection-Rule und Populationsgröße
            params = conn.get('params', {})
            rule = params.get('rule', 'all_to_all')
            
            if rule == 'one_to_one':
                # 1:1 Verbindungen
                n_edges = min(src_size, tgt_size, max_edges_per_connection)
                for i in range(n_edges):
                    src_idx = src_start + (i % src_size)
                    tgt_idx = tgt_start + (i % tgt_size)
                    lines_data.append((src_idx, tgt_idx))
                    self.edge_connection_map.append(conn_idx)
                    
            elif rule == 'fixed_indegree':
                indegree = params.get('indegree', 10)
                n_edges = min(indegree * min(tgt_size, 5), max_edges_per_connection)
                for _ in range(n_edges):
                    src_idx = src_start + np.random.randint(0, src_size)
                    tgt_idx = tgt_start + np.random.randint(0, tgt_size)
                    lines_data.append((src_idx, tgt_idx))
                    self.edge_connection_map.append(conn_idx)
                    
            elif rule == 'fixed_outdegree':
                outdegree = params.get('outdegree', 10)
                n_edges = min(outdegree * min(src_size, 5), max_edges_per_connection)
                for _ in range(n_edges):
                    src_idx = src_start + np.random.randint(0, src_size)
                    tgt_idx = tgt_start + np.random.randint(0, tgt_size)
                    lines_data.append((src_idx, tgt_idx))
                    self.edge_connection_map.append(conn_idx)
                    
            elif rule == 'pairwise_bernoulli':
                p = params.get('p', 0.1)
                # Zeige proportional zur Wahrscheinlichkeit
                n_edges = min(int(p * src_size * tgt_size * 0.1), max_edges_per_connection)
                n_edges = max(5, n_edges)  # Mindestens 5 Kanten
                for _ in range(n_edges):
                    src_idx = src_start + np.random.randint(0, src_size)
                    tgt_idx = tgt_start + np.random.randint(0, tgt_size)
                    lines_data.append((src_idx, tgt_idx))
                    self.edge_connection_map.append(conn_idx)
                    
            else:  # all_to_all, fixed_total_number, etc.
                # Zeige repräsentative Stichprobe
                n_edges = min(max_edges_per_connection, src_size * tgt_size)
                for _ in range(n_edges):
                    src_idx = src_start + np.random.randint(0, src_size)
                    tgt_idx = tgt_start + np.random.randint(0, tgt_size)
                    lines_data.append((src_idx, tgt_idx))
                    self.edge_connection_map.append(conn_idx)
            
            print(f"  ✓ {conn.get('name', f'Conn_{conn_idx}')}: {len([e for i, e in enumerate(self.edge_connection_map) if self.edge_connection_map[i] == conn_idx])} edges")
        
        if not lines_data:
            print("[BlinkingNetwork] No valid edges to draw")
            self.edge_mesh = None
            self.plotter.reset_camera()
            return
        
        # =================================================================
        # 5. EDGE MESH ERSTELLEN
        # =================================================================
        n_edges = len(lines_data)
        print(f"[BlinkingNetwork] Drawing {n_edges} edges total")
        
        # PyVista Line Format: [2, idx_a, idx_b, 2, idx_a, idx_b, ...]
        lines = []
        for src_idx, tgt_idx in lines_data:
            lines.extend([2, src_idx, tgt_idx])
        lines = np.array(lines)
        
        self.edge_mesh = pv.PolyData(points, lines=lines)
        
        # Edge Intensität
        self.edge_intensities = np.zeros(n_edges)
        self.edge_mesh.cell_data["activity"] = self.edge_intensities
        
        # Colormap: Schwarz → Gelb (für normale) oder Orange (für Self-Connections)
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "activity_cmap", ["black", "yellow", "white"], N=256
        )
        
        self.plotter.add_mesh(
            self.edge_mesh,
            scalars="activity",
            cmap=custom_cmap,
            opacity="linear",
            line_width=1.5,
            show_scalar_bar=False,
            use_transparency=True
        )
        
        self.plotter.reset_camera()
        print("[BlinkingNetwork] Scene built successfully!")

    def animate(self):
        """Animiert Neuronen und Connections."""
        if self.neuron_mesh is None:
            return
        if not self.isVisible() or not self.is_active or self.neuron_mesh is None:
            return
        try:
            # ... (Dein Animations-Code hier) ...
            # Am Ende:
            self.plotter.update()
        except Exception:
            # Fang OpenGL Fehler ab, wenn das Fenster schon halb zu ist
            self.timer.stop()

    def closeEvent(self, event):
        """Sicheres Aufräumen"""
        self.is_active = False
        if self.timer.isActive():
            self.timer.stop()
        super().closeEvent(event)

    def stop_animation(self):
        """Explizite Stop-Methode für das Main Window"""
        self.is_active = False
        self.timer.stop()
        # =================================================================
        # 1. NEURONEN BLINKEN (subtil)
        # =================================================================
        noise = np.random.normal(0, 0.05, len(self.neuron_intensities))
        self.neuron_intensities = np.clip(self.neuron_intensities + noise, 0.3, 1.0)
        
        new_colors = self.base_colors * self.neuron_intensities[:, None]
        self.neuron_mesh.point_data["display_color"] = new_colors
        
        # =================================================================
        # 2. CONNECTIONS FEUERN (Spiking)
        # =================================================================
        if self.edge_mesh is not None and self.edge_intensities is not None:
            n_edges = len(self.edge_intensities)
            
            # Decay: Alle Kanten werden langsam dunkler
            self.edge_intensities *= 0.92
            
            # Random Spikes: ~3% der Kanten feuern pro Frame
            n_fire = max(1, int(n_edges * 0.03))
            if n_fire > 0 and n_edges > 0:
                fire_indices = np.random.choice(n_edges, min(n_fire, n_edges), replace=False)
                self.edge_intensities[fire_indices] = 1.0
            
            self.edge_mesh.cell_data["activity"] = self.edge_intensities
        
        self.plotter.update()
























def create_nest_connections(graph_list):

    
    total_connections_created = 0
    
    print("\n" + "="*70)
    print("🔗 CREATING NEST CONNECTIONS")
    print("="*70)
    
    # Iteriere durch alle Graphs
    for graph in self.graph_list:
        graph_id = graph.graph_id
        print(f"\n📊 Processing Graph {graph_id}...")
        
        # Iteriere durch alle Nodes im Graph
        for node in graph.node_list:
            node_id = node.id
            
            # Prüfe ob Node Connections hat
            if not hasattr(node, 'connections') or not node.connections:
                continue
            
            print(f"  🔷 Node {node_id} has {len(node.connections)} connection(s)")
            
            # Iteriere durch alle Connections des Nodes
            for conn in node.connections:
                try:
                    # === SOURCE ===
                    source_graph_id = conn['source']['graph_id']
                    source_node_id = conn['source']['node_id']
                    source_pop_id = conn['source']['pop_id']
                    
                    # Finde Source Graph
                    source_graph = next((g for g in graph_list if g.graph_id == source_graph_id), None)
                    if not source_graph:
                        print(f"    ⚠ Source Graph {source_graph_id} not found!")
                        continue
                    
                    # Finde Source Node
                    source_node = next((n for n in source_graph.node_list if n.id == source_node_id), None)
                    if not source_node:
                        print(f"    ⚠ Source Node {source_node_id} not found!")
                        continue
                    
                    # Hole Source Population
                    if source_pop_id >= len(source_node.population):
                        print(f"    ⚠ Source Population {source_pop_id} out of range!")
                        continue
                    source_population = source_node.population[source_pop_id]
                    
                    # === TARGET ===
                    target_graph_id = conn['target']['graph_id']
                    target_node_id = conn['target']['node_id']
                    target_pop_id = conn['target']['pop_id']
                    
                    # Finde Target Graph
                    target_graph = next((g for g in graph_list if g.graph_id == target_graph_id), None)
                    if not target_graph:
                        print(f"    ⚠ Target Graph {target_graph_id} not found!")
                        continue
                    
                    # Finde Target Node
                    target_node = next((n for n in target_graph.node_list if n.id == target_node_id), None)
                    if not target_node:
                        print(f"    ⚠ Target Node {target_node_id} not found!")
                        continue
                    
                    # Hole Target Population
                    if target_pop_id >= len(target_node.population):
                        print(f"    ⚠ Target Population {target_pop_id} out of range!")
                        continue
                    target_population = target_node.population[target_pop_id]
                    
                    # === CONNECTION PARAMETERS ===
                    params = conn['params']
                    
                    # Baue NEST Connection Dict
                    conn_spec = {'rule': params['rule']}
                    
                    # Füge rule-spezifische Parameter hinzu
                    if 'indegree' in params:
                        conn_spec['indegree'] = params['indegree']
                    if 'outdegree' in params:
                        conn_spec['outdegree'] = params['outdegree']
                    if 'N' in params:
                        conn_spec['N'] = params['N']
                    if 'p' in params:
                        conn_spec['p'] = params['p']
                    
                    # Autapses und Multapses
                    conn_spec['allow_autapses'] = params.get('allow_autapses', True)
                    conn_spec['allow_multapses'] = params.get('allow_multapses', True)
                    
                    # Synapse Spec
                    syn_spec = {
                        'synapse_model': params.get('synapse_model', 'static_synapse'),
                        'weight': params.get('weight', 1.0),
                        'delay': params.get('delay', 1.0)
                    }
                    
                    # === NEST CONNECT ===
                    nest.Connect(
                        source_population,
                        target_population,
                        conn_spec=conn_spec,
                        syn_spec=syn_spec
                    )
                    
                    total_connections_created += 1
                    
                    print(f"    ✅ {conn['name']}")
                    print(f"       {len(source_population)} → {len(target_population)} neurons")
                    print(f"       Rule: {params['rule']}, Weight: {params['weight']}, Delay: {params['delay']}ms")
                
                except Exception as e:
                    print(f"    ❌ Error creating connection {conn.get('name', 'unknown')}: {e}")
                    import traceback
                    traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"✅ Total NEST connections created: {total_connections_created}")
    print("="*70 + "\n")
    
    return total_connections_created




"""
Connection Logic Module for NEST Neural Simulation GUI
======================================================

This module provides two main functionalities:
1. Small Function: Create connections via GUI and store them in node.connections
2. Large Function: Recreate all stored connections after nest.ResetKernel()

Author: Generated for Neuroticks Project
"""

import nest
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import copy

# =============================================================================
# SYNAPSE MODELS DICTIONARY (Reference for validation)
# =============================================================================

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

# Connection rules supported by NEST
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


# =============================================================================
# CONNECTION DATA STRUCTURE
# =============================================================================

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
    """
    Create a standardized connection dictionary.
    
    Args:
        connection_id: Unique ID for this connection
        name: Human-readable name
        source_*: Source identifiers (graph, node, population)
        target_*: Target identifiers (graph, node, population)
        rule: Connection rule (all_to_all, one_to_one, etc.)
        synapse_model: NEST synapse model name
        weight: Synaptic weight
        delay: Synaptic delay in ms
        allow_autapses: Allow self-connections
        allow_multapses: Allow multiple connections between same pair
        **extra_params: Additional synapse-specific parameters
    
    Returns:
        Dictionary with all connection parameters
    """
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


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_delay(delay: float, resolution: float = None) -> float:
    """
    Validate and adjust delay to meet NEST requirements.
    
    Args:
        delay: Requested delay in ms
        resolution: NEST kernel resolution (if None, queries nest.resolution)
    
    Returns:
        Valid delay value (>= resolution)
    """
    if resolution is None:
        try:
            resolution = nest.resolution
        except:
            resolution = 0.1  # Default fallback
    
    if delay < resolution:
        print(f"⚠ Delay {delay}ms < resolution {resolution}ms, adjusting to {resolution}ms")
        return resolution
    return delay


def validate_connection_params(params: Dict[str, Any]) -> Tuple[Dict, Dict, List[str]]:
    """
    Validate and split connection parameters into conn_spec and syn_spec.
    
    Args:
        params: Combined connection parameters
    
    Returns:
        Tuple of (conn_spec, syn_spec, warnings)
    """
    warnings = []
    
    rule = params.get('rule', 'all_to_all')
    synapse_model = params.get('synapse_model', 'static_synapse')
    
    # Build conn_spec
    conn_spec = {'rule': rule}
    
    if rule in CONNECTION_RULES:
        rule_params = CONNECTION_RULES[rule]['params']
        for p in rule_params:
            if p in params:
                conn_spec[p] = params[p]
    
    # Handle rule-specific required params
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
    
    # Build syn_spec
    syn_spec = {'synapse_model': synapse_model}
    
    # Always include weight and delay
    syn_spec['weight'] = params.get('weight', 1.0)
    syn_spec['delay'] = validate_delay(params.get('delay', 1.0))
    if 'receptor_type' in params:
        syn_spec['receptor_type'] = int(params['receptor_type'])
    # Add synapse-specific parameters
    if synapse_model in SYNAPSE_MODELS:
        model_params = SYNAPSE_MODELS[synapse_model]
        for param_name, param_info in model_params.items():
            if param_name in params and param_name not in ['weight', 'delay']:
                syn_spec[param_name] = params[param_name]
    
    # Check for extra unknown parameters (might be model-specific)
    known_params = {'rule', 'synapse_model', 'weight', 'delay', 
                    'allow_autapses', 'allow_multapses', 'indegree', 
                    'outdegree', 'N', 'p', 'receptor_type'}
    if synapse_model in SYNAPSE_MODELS:
        known_params.update(SYNAPSE_MODELS[synapse_model].keys())
    
    for key in params:
        if key not in known_params:
            # Pass through unknown params to syn_spec (might be valid for this model)
            syn_spec[key] = params[key]
            warnings.append(f"Unknown parameter '{key}' passed to syn_spec")
    
    return conn_spec, syn_spec, warnings


# =============================================================================
# SMALL FUNCTION: Execute connections from GUI and store in node.connections
# =============================================================================

class ConnectionExecutor:
    """
    Handles connection creation from GUI.
    
    This is the "small function" that:
    1. Adds connection parameters to node.connections
    2. Executes the NEST connection
    """
    
    def __init__(self, graphs: Dict[int, Any]):
        """
        Initialize with reference to all graphs.
        
        Args:
            graphs: Dictionary mapping graph_id to Graph objects
        """
        self.graphs = graphs
        self._connection_counter = 0
    
    def _get_next_connection_id(self) -> int:
        """Generate unique connection ID."""
        self._connection_counter += 1
        return self._connection_counter
    
    def _get_population(self, graph_id: int, node_id: int, pop_id: int):
        """
        Retrieve NEST population from graph/node/pop indices.
        
        Returns:
            NEST NodeCollection or None if not found
        """
        if graph_id not in self.graphs:
            print(f"✗ Graph {graph_id} not found")
            return None
        
        graph = self.graphs[graph_id]
        node = graph.get_node(node_id)
        
        if node is None:
            print(f"✗ Node {node_id} not found in Graph {graph_id}")
            return None
        
        if not node.population:
            print(f"✗ Node {node_id} has no populations (not populated yet?)")
            return None
        
        if pop_id >= len(node.population):
            print(f"✗ Population {pop_id} not found in Node {node_id} (has {len(node.population)} pops)")
            return None
        
        pop = node.population[pop_id]
        if pop is None or (hasattr(pop, '__len__') and len(pop) == 0):
            print(f"✗ Population {pop_id} in Node {node_id} is empty")
            return None
        
        return pop
    
    def _execute_single_connection(self, connection: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Execute a single NEST connection.
        
        Args:
            connection: Connection dictionary with source, target, params
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        source_info = connection['source']
        target_info = connection['target']
        params = connection['params']
        
        # Get populations
        source_pop = self._get_population(
            source_info['graph_id'],
            source_info['node_id'],
            source_info['pop_id']
        )
        target_pop = self._get_population(
            target_info['graph_id'],
            target_info['node_id'],
            target_info['pop_id']
        )
        
        if source_pop is None or target_pop is None:
            return False, "Source or target population not found"
        
        # Validate and split parameters
        conn_spec, syn_spec, warnings = validate_connection_params(params)
        
        for w in warnings:
            print(f"  ⚠ {w}")
        
        # Special handling for one_to_one rule
        if conn_spec['rule'] == 'one_to_one':
            if len(source_pop) != len(target_pop):
                return False, f"one_to_one requires equal sizes ({len(source_pop)} vs {len(target_pop)})"
        
        # Execute NEST connection
        try:
            nest.Connect(source_pop, target_pop, conn_spec, syn_spec)
            
            # Count created connections
            conns = nest.GetConnections(source_pop, target_pop)
            n_created = len(conns) if conns else 0
            
            return True, f"Created {n_created} synapses"
        
        except Exception as e:
            return False, f"NEST error: {str(e)}"
    
    def add_and_execute_connection(
        self,
        source_graph_id: int,
        source_node_id: int,
        source_pop_id: int,
        target_graph_id: int,
        target_node_id: int,
        target_pop_id: int,
        params: Dict[str, Any],
        connection_name: str = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Add connection to node.connections AND execute in NEST.
        
        This is the main method called when "Create Connections" button is clicked.
        
        Args:
            source_*: Source identifiers
            target_*: Target identifiers  
            params: Connection parameters (rule, weight, delay, synapse_model, etc.)
            connection_name: Optional name for the connection
        
        Returns:
            Tuple of (success, message, connection_dict or None)
        """
        # Generate connection ID and name
        conn_id = self._get_next_connection_id()
        if connection_name is None:
            connection_name = f"Connection_{conn_id}"
        
        # Create connection dictionary
        connection = create_connection_dict(
            connection_id=conn_id,
            name=connection_name,
            source_graph_id=source_graph_id,
            source_node_id=source_node_id,
            source_pop_id=source_pop_id,
            target_graph_id=target_graph_id,
            target_node_id=target_node_id,
            target_pop_id=target_pop_id,
            **params
        )
        
        # Get source node to store connection
        if source_graph_id not in self.graphs:
            return False, f"Graph {source_graph_id} not found", None
        
        source_node = self.graphs[source_graph_id].get_node(source_node_id)
        if source_node is None:
            return False, f"Node {source_node_id} not found", None
        
        # Check for duplicate connection (same source->target with same params)
        for existing in source_node.connections:
            if (existing['source'] == connection['source'] and 
                existing['target'] == connection['target'] and
                existing['params'] == connection['params']):
                return False, "Duplicate connection already exists", None
        
        # Execute the connection in NEST
        success, message = self._execute_single_connection(connection)
        
        if success:
            # Store in node.connections
            source_node.connections.append(connection)
            print(f"✓ {connection_name}: {message}")
            return True, message, connection
        else:
            print(f"✗ {connection_name}: {message}")
            return False, message, None
    
    def execute_pending_connections(
        self,
        pending_connections: List[Dict[str, Any]]
    ) -> Tuple[int, int, List[Dict]]:
        """
        Execute a list of pending connections (e.g., from GUI queue).
        
        Args:
            pending_connections: List of connection parameter dicts
        
        Returns:
            Tuple of (successful_count, failed_count, failed_connections)
        """
        successful = 0
        failed = 0
        failed_list = []
        
        for conn_params in pending_connections:
            success, msg, conn = self.add_and_execute_connection(
                source_graph_id=conn_params['source_graph_id'],
                source_node_id=conn_params['source_node_id'],
                source_pop_id=conn_params['source_pop_id'],
                target_graph_id=conn_params['target_graph_id'],
                target_node_id=conn_params['target_node_id'],
                target_pop_id=conn_params['target_pop_id'],
                params=conn_params.get('params', {}),
                connection_name=conn_params.get('name')
            )
            
            if success:
                successful += 1
            else:
                failed += 1
                failed_list.append({**conn_params, 'error': msg})
        
        return successful, failed, failed_list


# =============================================================================
# LARGE FUNCTION: Recreate all connections after nest.ResetKernel()
# =============================================================================

def create_nest_connections_from_stored(
    graphs: Dict[int, Any],
    verbose: bool = True
) -> Tuple[int, int, List[Dict]]:
    """
    Recreate ALL stored connections from all graphs after nest.ResetKernel().
    
    This is the "large function" that should be called wherever 
    nest.ResetKernel() is used to maintain consistency.
    
    Args:
        graphs: Dictionary mapping graph_id to Graph objects
        verbose: Print detailed progress
    
    Returns:
        Tuple of (total_created, total_failed, failed_connections_list)
    """
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
            print(f"\n📊 Graph {graph_id} ({graph.graph_name}):")
        
        for node in graph.node_list:
            if node.connections:
                if verbose:
                    print(f"  📦 Node {node.id} ({node.name}): {len(node.connections)} connections")
                
                for conn in node.connections:
                    all_connections.append(conn)
    
    if not all_connections:
        if verbose:
            print("\n⚠ No stored connections found")
        return 0, 0, []
    
    if verbose:
        print(f"\n🔗 Total connections to recreate: {len(all_connections)}")
        print("-"*60)
    
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
        print("-"*60)
        print(f"✓ Created: {total_created} | ✗ Failed: {total_failed}")
        print("="*60 + "\n")
    
    return total_created, total_failed, failed_connections


def repopulate_all_graphs(graphs: Dict[int, Any], verbose: bool = True) -> int:
    """
    Repopulate all nodes in all graphs (create NEST neurons).
    
    Called after nest.ResetKernel() to recreate all neuron populations.
    
    Args:
        graphs: Dictionary mapping graph_id to Graph objects
        verbose: Print progress
    
    Returns:
        Total number of populations created
    """
    total_pops = 0
    
    if verbose:
        print("\n" + "="*60)
        print("REPOPULATING ALL GRAPHS")
        print("="*60)
    
    for graph_id, graph in graphs.items():
        if verbose:
            print(f"\n📊 Graph {graph_id} ({graph.graph_name}):")
        
        for node in graph.node_list:
            try:
                # Build positions if not already built
                if not node.positions or all(len(c) == 0 for c in node.positions):
                    if verbose:
                        print(f"  🔧 Building Node {node.id}...")
                    node.build()
                
                # Populate with NEST neurons
                if verbose:
                    print(f"  🧠 Populating Node {node.id}...")
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
        print("="*60 + "\n")
    
    return total_pops


def safe_nest_reset_and_repopulate(
    graphs: Dict[int, Any],
    enable_structural_plasticity: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Safely reset NEST kernel and recreate all neurons and connections.
    
    This is a wrapper function that should be used INSTEAD of bare
    nest.ResetKernel() to maintain consistency.
    
    Args:
        graphs: Dictionary mapping graph_id to Graph objects
        enable_structural_plasticity: Re-enable structural plasticity after reset
        verbose: Print detailed progress
    
    Returns:
        Dictionary with statistics:
        {
            'populations_created': int,
            'connections_created': int,
            'connections_failed': int,
            'failed_connections': list,
            'structural_plasticity': bool
        }
    """
    if verbose:
        print("\n" + "#"*60)
        print("# SAFE NEST RESET AND REPOPULATE")
        print("#"*60)
    
    # 1. Reset NEST kernel
    if verbose:
        print("\n🔄 Resetting NEST kernel...")
    nest.ResetKernel()
    
    # 2. Re-enable structural plasticity if requested
    if enable_structural_plasticity:
        try:
            nest.EnableStructuralPlasticity()
            if verbose:
                print("✓ Structural plasticity enabled")
        except Exception as e:
            if verbose:
                print(f"⚠ Could not enable structural plasticity: {e}")
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
        print("\n" + "#"*60)
        print("# RESET COMPLETE")
        print(f"# Populations: {total_pops}")
        print(f"# Connections: {created} created, {failed} failed")
        print("#"*60 + "\n")
    
    return result


# =============================================================================
# UTILITY FUNCTIONS FOR INTEGRATION
# =============================================================================

def get_all_connections_summary(graphs: Dict[int, Any]) -> Dict[str, Any]:
    """
    Get a summary of all stored connections across all graphs.
    
    Args:
        graphs: Dictionary mapping graph_id to Graph objects
    
    Returns:
        Summary dictionary
    """
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
    """
    Clear all stored connections from all nodes.
    
    Args:
        graphs: Dictionary mapping graph_id to Graph objects
        verbose: Print progress
    
    Returns:
        Number of connections cleared
    """
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
    """
    Export all connections to a serializable list of dicts.
    
    Useful for saving/loading connection configurations.
    
    Args:
        graphs: Dictionary mapping graph_id to Graph objects
    
    Returns:
        List of connection dictionaries
    """
    all_connections = []
    
    for graph_id, graph in graphs.items():
        for node in graph.node_list:
            for conn in (node.connections or []):
                # Deep copy to avoid reference issues
                all_connections.append(copy.deepcopy(conn))
    
    return all_connections


def import_connections_from_dict(
    graphs: Dict[int, Any],
    connections: List[Dict],
    clear_existing: bool = True
) -> int:
    """
    Import connections from a list of dicts.
    
    Args:
        graphs: Dictionary mapping graph_id to Graph objects
        connections: List of connection dictionaries
        clear_existing: Clear existing connections before import
    
    Returns:
        Number of connections imported
    """
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


# =============================================================================
# INTEGRATION HELPER: For use in GUI ConnectionTool
# =============================================================================

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
    """
    Create a connection dict from GUI widget values.
    
    This is a convenience function that maps GUI inputs to the
    standardized connection dictionary format.
    
    Args:
        source_*, target_*: Population identifiers
        rule: Connection rule name
        synapse_model: Synapse model name
        weight, delay: Basic synapse parameters
        allow_autapses, allow_multapses: Connection options
        indegree, outdegree, N, p: Rule-specific parameters
        **synapse_params: Additional synapse-specific parameters
    
    Returns:
        Connection parameter dictionary (ready for ConnectionExecutor)
    """
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




class ConnectionToolExtended(QWidget):
    """
    Extended ConnectionTool with full connection logic integration.
    
    Signals:
        connection_created: Emitted when a connection is successfully created
        connection_failed: Emitted when a connection fails
    """
    
    connection_created = pyqtSignal(dict)  # Emits the connection dict
    connection_failed = pyqtSignal(str)    # Emits error message
    
    def __init__(self, parent=None, graphs=None):
        super().__init__(parent)
        self.graphs = graphs if graphs is not None else {}
        self.pending_connections = []  # Queue of connections to create
        self.connection_executor = None  # Will be initialized when graphs are set
        
        self._init_ui()
    
    def set_graphs(self, graphs):
        """Set the graphs dictionary and initialize executor."""
        self.graphs = graphs
        self.connection_executor = ConnectionExecutor(graphs)
        self._update_graph_combos()
    
    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        
        # === SOURCE SELECTION ===
        source_group = QGroupBox("Source")
        source_layout = QFormLayout(source_group)
        
        self.source_graph_combo = QComboBox()
        self.source_graph_combo.currentIndexChanged.connect(self._on_source_graph_changed)
        source_layout.addRow("Graph:", self.source_graph_combo)
        
        self.source_node_combo = QComboBox()
        self.source_node_combo.currentIndexChanged.connect(self._on_source_node_changed)
        source_layout.addRow("Node:", self.source_node_combo)
        
        self.source_pop_combo = QComboBox()
        source_layout.addRow("Population:", self.source_pop_combo)
        
        layout.addWidget(source_group)
        
        # === TARGET SELECTION ===
        target_group = QGroupBox("Target")
        target_layout = QFormLayout(target_group)
        
        self.target_graph_combo = QComboBox()
        self.target_graph_combo.currentIndexChanged.connect(self._on_target_graph_changed)
        target_layout.addRow("Graph:", self.target_graph_combo)
        
        self.target_node_combo = QComboBox()
        self.target_node_combo.currentIndexChanged.connect(self._on_target_node_changed)
        target_layout.addRow("Node:", self.target_node_combo)
        
        self.target_pop_combo = QComboBox()
        target_layout.addRow("Population:", self.target_pop_combo)
        
        layout.addWidget(target_group)
        
        # === CONNECTION PARAMETERS ===
        params_group = QGroupBox("Connection Parameters")
        params_layout = QFormLayout(params_group)
        
        # Connection Rule
        self.rule_combo = QComboBox()
        self.rule_combo.addItems([
            'all_to_all', 'one_to_one', 'fixed_indegree', 
            'fixed_outdegree', 'fixed_total_number', 'pairwise_bernoulli'
        ])
        self.rule_combo.currentTextChanged.connect(self._on_rule_changed)
        params_layout.addRow("Rule:", self.rule_combo)
        
        # Rule-specific parameters (dynamic)
        self.rule_params_frame = QFrame()
        self.rule_params_layout = QFormLayout(self.rule_params_frame)
        params_layout.addRow(self.rule_params_frame)
        
        # Initialize rule-specific widgets
        self._init_rule_specific_widgets()
        
        layout.addWidget(params_group)
        
        # === SYNAPSE PARAMETERS ===
        synapse_group = QGroupBox("Synapse Parameters")
        synapse_layout = QFormLayout(synapse_group)
        self.receptor_spin = QSpinBox()
        self.receptor_spin.setRange(0, 100)
        self.receptor_spin.setValue(0)
        self.receptor_spin.setToolTip("0 for standard PSC models.\n1 (Ex), 2 (Inh) for conductance models.")
        synapse_layout.addRow("Receptor ID:", self.receptor_spin)
        # Synapse Model
        self.synapse_combo = QComboBox()
        self.synapse_combo.addItems([
            'static_synapse', 'stdp_synapse', 'stdp_synapse_hom',
            'tsodyks_synapse', 'tsodyks2_synapse', 'stdp_dopamine_synapse',
            'bernoulli_synapse', 'clopath_synapse', 'vogels_sprekeler_synapse'
        ])
        self.synapse_combo.currentTextChanged.connect(self._on_synapse_changed)
        synapse_layout.addRow("Model:", self.synapse_combo)
        
        # Weight
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(-10000.0, 10000.0)
        self.weight_spin.setValue(1.0)
        self.weight_spin.setDecimals(3)
        synapse_layout.addRow("Weight:", self.weight_spin)
        
        # Delay
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 1000.0)
        self.delay_spin.setValue(1.0)
        self.delay_spin.setDecimals(2)
        self.delay_spin.setSuffix(" ms")
        synapse_layout.addRow("Delay:", self.delay_spin)
        
        # Allow autapses
        self.autapses_check = QCheckBox()
        self.autapses_check.setChecked(False)
        synapse_layout.addRow("Allow Autapses:", self.autapses_check)
        
        # Allow multapses
        self.multapses_check = QCheckBox()
        self.multapses_check.setChecked(True)
        synapse_layout.addRow("Allow Multapses:", self.multapses_check)
        
        # Synapse-specific params frame
        self.synapse_params_frame = QFrame()
        self.synapse_params_layout = QFormLayout(self.synapse_params_frame)
        synapse_layout.addRow(self.synapse_params_frame)
        
        layout.addWidget(synapse_group)
        
        # === PENDING CONNECTIONS LIST ===
        pending_group = QGroupBox("Pending Connections")
        pending_layout = QVBoxLayout(pending_group)
        
        self.pending_list = QListWidget()
        pending_layout.addWidget(self.pending_list)
        
        # Add/Remove buttons
        btn_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("Add to Queue")
        self.add_btn.clicked.connect(self._add_to_pending)
        btn_layout.addWidget(self.add_btn)
        
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._remove_from_pending)
        btn_layout.addWidget(self.remove_btn)
        
        pending_layout.addLayout(btn_layout)
        layout.addWidget(pending_group)
        
        # === CREATE CONNECTIONS BUTTON ===
        self.create_btn = QPushButton("🔗 Create Connections")
        self.create_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.create_btn.clicked.connect(self._create_all_connections)
        layout.addWidget(self.create_btn)
        
        # === STATUS ===
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
    
    def _init_rule_specific_widgets(self):
        """Initialize widgets for rule-specific parameters."""
        # Indegree (for fixed_indegree)
        self.indegree_spin = QSpinBox()
        self.indegree_spin.setRange(1, 10000)
        self.indegree_spin.setValue(10)
        
        # Outdegree (for fixed_outdegree)
        self.outdegree_spin = QSpinBox()
        self.outdegree_spin.setRange(1, 10000)
        self.outdegree_spin.setValue(10)
        
        # N (for fixed_total_number)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 1000000)
        self.n_spin.setValue(100)
        
        # p (for pairwise_bernoulli)
        self.p_spin = QDoubleSpinBox()
        self.p_spin.setRange(0.0, 1.0)
        self.p_spin.setValue(0.1)
        self.p_spin.setDecimals(4)
        
        self._on_rule_changed(self.rule_combo.currentText())
    
    def _on_rule_changed(self, rule):
        """Update UI when connection rule changes."""
        # Clear existing widgets
        while self.rule_params_layout.count():
            item = self.rule_params_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        
        # Add relevant widgets
        if rule == 'fixed_indegree':
            self.rule_params_layout.addRow("Indegree:", self.indegree_spin)
        elif rule == 'fixed_outdegree':
            self.rule_params_layout.addRow("Outdegree:", self.outdegree_spin)
        elif rule == 'fixed_total_number':
            self.rule_params_layout.addRow("N (total):", self.n_spin)
        elif rule in ['pairwise_bernoulli', 'symmetric_pairwise_bernoulli']:
            self.rule_params_layout.addRow("Probability:", self.p_spin)
    
    def _on_synapse_changed(self, synapse_model):
        """Update UI when synapse model changes."""
        # Clear existing widgets
        while self.synapse_params_layout.count():
            item = self.synapse_params_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        
        # Add synapse-specific widgets (simplified - expand as needed)
        if synapse_model == 'stdp_synapse':
            tau_plus = QDoubleSpinBox()
            tau_plus.setRange(0.1, 1000.0)
            tau_plus.setValue(20.0)
            tau_plus.setObjectName('tau_plus')
            self.synapse_params_layout.addRow("tau_plus (ms):", tau_plus)
            
            Wmax = QDoubleSpinBox()
            Wmax.setRange(0.0, 10000.0)
            Wmax.setValue(100.0)
            Wmax.setObjectName('Wmax')
            self.synapse_params_layout.addRow("Wmax:", Wmax)
        
        elif synapse_model in ['tsodyks_synapse', 'tsodyks2_synapse']:
            U = QDoubleSpinBox()
            U.setRange(0.0, 1.0)
            U.setValue(0.5)
            U.setObjectName('U')
            self.synapse_params_layout.addRow("U:", U)
            
            tau_rec = QDoubleSpinBox()
            tau_rec.setRange(0.0, 10000.0)
            tau_rec.setValue(800.0)
            tau_rec.setObjectName('tau_rec')
            self.synapse_params_layout.addRow("tau_rec (ms):", tau_rec)
    
    def _update_graph_combos(self):
        """Update graph selection combos."""
        self.source_graph_combo.clear()
        self.target_graph_combo.clear()
        
        for graph_id, graph in self.graphs.items():
            name = f"Graph {graph_id}: {graph.graph_name}"
            self.source_graph_combo.addItem(name, graph_id)
            self.target_graph_combo.addItem(name, graph_id)
    
    def _on_source_graph_changed(self, index):
        """Update node combo when source graph changes."""
        self.source_node_combo.clear()
        
        if index < 0:
            return
        
        graph_id = self.source_graph_combo.currentData()
        if graph_id in self.graphs:
            graph = self.graphs[graph_id]
            for node in graph.node_list:
                self.source_node_combo.addItem(
                    f"Node {node.id}: {node.name}", 
                    node.id
                )
    
    def _on_source_node_changed(self, index):
        """Update population combo when source node changes."""
        self.source_pop_combo.clear()
        
        if index < 0:
            return
        
        graph_id = self.source_graph_combo.currentData()
        node_id = self.source_node_combo.currentData()
        
        if graph_id in self.graphs:
            node = self.graphs[graph_id].get_node(node_id)
            if node and node.population:
                for i, pop in enumerate(node.population):
                    n_neurons = len(pop) if pop else 0
                    self.source_pop_combo.addItem(
                        f"Pop {i} ({n_neurons} neurons)",
                        i
                    )
    
    def _on_target_graph_changed(self, index):
        """Update node combo when target graph changes."""
        self.target_node_combo.clear()
        
        if index < 0:
            return
        
        graph_id = self.target_graph_combo.currentData()
        if graph_id in self.graphs:
            graph = self.graphs[graph_id]
            for node in graph.node_list:
                self.target_node_combo.addItem(
                    f"Node {node.id}: {node.name}",
                    node.id
                )
    
    def _on_target_node_changed(self, index):
        """Update population combo when target node changes."""
        self.target_pop_combo.clear()
        
        if index < 0:
            return
        
        graph_id = self.target_graph_combo.currentData()
        node_id = self.target_node_combo.currentData()
        
        if graph_id in self.graphs:
            node = self.graphs[graph_id].get_node(node_id)
            if node and node.population:
                for i, pop in enumerate(node.population):
                    n_neurons = len(pop) if pop else 0
                    self.target_pop_combo.addItem(
                        f"Pop {i} ({n_neurons} neurons)",
                        i
                    )
    
    def _get_current_params(self):
        """Get all current parameters from UI."""
        params = {
            'rule': self.rule_combo.currentText(),
            'synapse_model': self.synapse_combo.currentText(),
            'weight': self.weight_spin.value(),
            'delay': self.delay_spin.value(),
            'allow_autapses': self.autapses_check.isChecked(),
            'allow_multapses': self.multapses_check.isChecked(),
            'receptor_type': self.receptor_spin.value(),
        }
        
        # Rule-specific params
        rule = params['rule']
        if rule == 'fixed_indegree':
            params['indegree'] = self.indegree_spin.value()
        elif rule == 'fixed_outdegree':
            params['outdegree'] = self.outdegree_spin.value()
        elif rule == 'fixed_total_number':
            params['N'] = self.n_spin.value()
        elif rule in ['pairwise_bernoulli', 'symmetric_pairwise_bernoulli']:
            params['p'] = self.p_spin.value()
        
        # Synapse-specific params
        for i in range(self.synapse_params_layout.count()):
            item = self.synapse_params_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                name = widget.objectName()
                if name and isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    params[name] = widget.value()
        
        return params
    
    def _add_to_pending(self):
        """Add current configuration to pending connections queue."""
        source_graph = self.source_graph_combo.currentData()
        source_node = self.source_node_combo.currentData()
        source_pop = self.source_pop_combo.currentData()
        
        target_graph = self.target_graph_combo.currentData()
        target_node = self.target_node_combo.currentData()
        target_pop = self.target_pop_combo.currentData()
        
        if any(x is None for x in [source_graph, source_node, source_pop,
                                    target_graph, target_node, target_pop]):
            QMessageBox.warning(self, "Incomplete", 
                "Please select source and target populations.")
            return
        
        params = self._get_current_params()
        
        conn_dict = {
            'source_graph_id': source_graph,
            'source_node_id': source_node,
            'source_pop_id': source_pop,
            'target_graph_id': target_graph,
            'target_node_id': target_node,
            'target_pop_id': target_pop,
            'params': params,
            'name': f"G{source_graph}N{source_node}P{source_pop}→G{target_graph}N{target_node}P{target_pop}"
        }
        
        self.pending_connections.append(conn_dict)
        
        # Add to list widget
        item = QListWidgetItem(
            f"{conn_dict['name']} ({params['rule']}, {params['synapse_model']})"
        )
        self.pending_list.addItem(item)
        
        self.status_label.setText(f"Added to queue ({len(self.pending_connections)} pending)")
    
    def _remove_from_pending(self):
        """Remove selected connection from pending queue."""
        row = self.pending_list.currentRow()
        if row >= 0:
            self.pending_list.takeItem(row)
            self.pending_connections.pop(row)
            self.status_label.setText(f"Removed ({len(self.pending_connections)} pending)")
    
    def _create_all_connections(self):
        """
        Create all pending connections - this is the SMALL FUNCTION.
        
        1. Adds each connection to node.connections
        2. Executes the NEST connection
        """
        if not self.pending_connections:
            QMessageBox.information(self, "No Connections", 
                "No pending connections to create.")
            return
        
        if self.connection_executor is None:
            self.connection_executor = ConnectionExecutor(self.graphs)
        
        # Execute all pending connections
        successful, failed, failed_list = self.connection_executor.execute_pending_connections(
            self.pending_connections
        )
        
        # Clear pending queue
        self.pending_connections.clear()
        self.pending_list.clear()
        
        # Show result
        msg = f"Created {successful} connections"
        if failed > 0:
            msg += f", {failed} failed"
            for f in failed_list:
                print(f"Failed: {f['name']} - {f.get('error', 'Unknown error')}")
        
        self.status_label.setText(msg)
        
        if successful > 0:
            self.connection_created.emit({'created': successful, 'failed': failed})
        
        if failed > 0:
            QMessageBox.warning(self, "Some Connections Failed",
                f"{failed} connection(s) failed. Check console for details.")








def validate_synapse_models():
    """Prüft ob synapse_models nur echte NEST Synapse-Modelle enthält."""
    import nest
    
    # Hole alle echten NEST Synapse-Modelle
    nest_synapses = set(nest.synapse_models)
    
    # Prüfe jedes Model in deinem Dict
    print("\n=== SYNAPSE MODEL VALIDATION ===")
    
    invalid = []
    valid = []
    
    for model in synapse_models.keys():
        if model in nest_synapses:
            valid.append(model)
            print(f"  ✅ {model}")
        else:
            invalid.append(model)
            print(f"  ❌ {model} - NOT A VALID SYNAPSE MODEL!")
    
    print(f"\nValid: {len(valid)} | Invalid: {len(invalid)}")
    
    if invalid:
        print("\n⚠️  REMOVE THESE FROM synapse_models:")
        for m in invalid:
            print(f'    "{m}": {{}},')
    
    return invalid



"""
Graph I/O Module - Simple Save/Load for Neuroticks
===================================================

Super simple JSON save/load für Graphs mit allen Node-Parametern.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


# =============================================================================
# NUMPY ENCODER
# =============================================================================


# =============================================================================
# TESTS
# =============================================================================

def run_tests():
    """Führt alle Tests aus."""
    print("\n" + "="*60)
    print("GRAPH I/O TESTS")
    print("="*60)
    
    passed = 0
    failed = 0
    
    # Test 1: NumpyEncoder
    print("\n🧪 Test 1: NumpyEncoder")
    try:
        data = {
            'arr': np.array([1.0, 2.0, 3.0]),
            'int': np.int64(42),
            'float': np.float64(3.14),
        }
        json_str = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(json_str)
        
        assert parsed['arr'] == [1.0, 2.0, 3.0]
        assert parsed['int'] == 42
        assert abs(parsed['float'] - 3.14) < 0.01
        
        print("   ✅ PASSED")
        passed += 1
    except AssertionError as e:
        print(f"   ❌ FAILED: {e}")
        failed += 1
    
    # Test 2: _clean_params
    print("\n🧪 Test 2: _clean_params")
    try:
        params = {
            'm': np.array([1, 2, 3]),
            'grid_size': [np.int64(10), np.int64(10), np.int64(10)],
            'nested': {'val': np.float64(0.5)}
        }
        clean = _clean_params(params)
        
        assert clean['m'] == [1, 2, 3]
        assert all(isinstance(x, int) for x in clean['grid_size'])
        assert isinstance(clean['nested']['val'], float)
        
        # Muss JSON-serialisierbar sein
        json.dumps(clean)
        
        print("   ✅ PASSED")
        passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        failed += 1
    
    # Test 3: Connection Serialization
    print("\n🧪 Test 3: Connection Serialization")
    try:
        conns = [{
            'id': 1,
            'name': 'TestConn',
            'source': {'graph_id': 0, 'node_id': 0, 'pop_id': 0},
            'target': {'graph_id': 0, 'node_id': 0, 'pop_id': 0},
            'params': {'weight': np.float64(1.5), 'delay': 1.0}
        }]
        
        serialized = _serialize_connections(conns)
        
        assert len(serialized) == 1
        assert serialized[0]['params']['weight'] == 1.5
        
        # JSON test
        json.dumps(serialized)
        
        print("   ✅ PASSED")
        passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        failed += 1
    
    # Test 4: Full JSON Round-Trip (Mock)
    print("\n🧪 Test 4: JSON Round-Trip")
    try:
        import tempfile
        import os
        
        mock_data = {
            'meta': {'version': '1.0'},
            'graph': {
                'graph_id': 0,
                'graph_name': 'TestGraph',
                'max_nodes': 2,
                'init_position': [0, 0, 0],
                'polynom_max_power': 5,
                'polynom_decay': 0.8
            },
            'nodes': [
                {
                    'id': 0, 'name': 'Node_0', 'graph_id': 0,
                    'parameters': {'m': [0, 0, 0], 'types': [0]},
                    'center_of_mass': [0.1, 0.2, 0.3],
                    'types': [0], 'neuron_models': ['iaf_psc_alpha'],
                    'distribution': [], 'connections': [],
                    'parent_id': None, 'next_ids': [], 'prev_ids': [],
                    'positions': [[[0, 0, 0], [1, 1, 1]]]
                }
            ]
        }
        
        # Write
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_data, f)
            temp_path = f.name
        
        # Read
        with open(temp_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['graph']['graph_name'] == 'TestGraph'
        assert len(loaded['nodes']) == 1
        
        os.unlink(temp_path)
        
        print("   ✅ PASSED")
        passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        failed += 1
    
    # Test 5: Self-Connection Structure
    print("\n🧪 Test 5: Self-Connection Structure")
    try:
        self_conn = {
            'id': 1,
            'name': 'Self_Conn',
            'source': {'graph_id': 0, 'node_id': 0, 'pop_id': 0},
            'target': {'graph_id': 0, 'node_id': 0, 'pop_id': 0},  # Same as source!
            'params': {'rule': 'all_to_all', 'weight': 1.0}
        }
        
        assert self_conn['source'] == self_conn['target']
        
        serialized = _serialize_connections([self_conn])
        assert serialized[0]['source'] == serialized[0]['target']
        
        print("   ✅ PASSED")
        passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        failed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


def generate_node_parameters_list(n_nodes=5, 
                                   n_types=5, 
                                   vary_polynoms=True,
                                   vary_types_per_node=True,
                                   safe_mode=True,      
                                   max_power=2,
                                   max_coeff=0.8,
                                   graph_id=0,
                                   add_self_connections=False,  # ✅ NEU
                                   self_conn_probability=0.3):  # ✅ NEU
    """
    Generiert Parameter-Liste für Nodes mit optionalem Safe Mode
    
    Args:
        n_nodes: Anzahl Nodes
        n_types: Anzahl Types pro Node
        vary_polynoms: Ob Polynome variiert werden
        vary_types_per_node: Ob Type-Anzahl variiert
        safe_mode: Verhindert Overflow in Polynomen
        max_power: Maximale Potenz (0-2 empfohlen)
        max_coeff: Maximaler Koeffizient
        graph_id: Graph ID für alle Nodes
        add_self_connections: Ob Self-Connections erstellt werden sollen  # ✅ NEU
        self_conn_probability: Wahrscheinlichkeit für Self-Connection pro Node  # ✅ NEU
    """
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
            "connections": [],  # ✅ Initialisiere connections
        }
        
        # ✅ NEU: Self-Connections hinzufügen
        if add_self_connections and np.random.random() < self_conn_probability:
            # Für jeden Type eine Self-Connection erstellen
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
                        'node_id': i,  # ✅ Auf sich selbst!
                        'pop_id': pop_id
                    },
                    'params': {
                        'rule': 'fixed_indegree',
                        'indegree': np.random.randint(1, 5),
                        'synapse_model': 'static_synapse',
                        'weight': np.random.uniform(0.5, 2.0),
                        'delay': 1.0,
                        'allow_autapses': False,  # Keine Selbst-Synapsen auf Neuron-Ebene
                        'allow_multapses': True
                    }
                }
                params['connections'].append(self_conn)
        
        # Polynome generieren
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


# =============================================================================
# SAVE/LOAD BUTTON WIDGET
# =============================================================================

class SaveLoadWidget(QWidget):
    """Widget mit Save/Load Buttons für Graphs."""
    
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
        self.save_btn.clicked.connect(self.save_graph)
        
        # Load Button
        self.load_btn = QPushButton("📂 Load Graph")
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
        self.load_btn.clicked.connect(self.load_graph)
        
        layout.addWidget(self.save_btn)
        layout.addWidget(self.load_btn)
        layout.addStretch()
    
    def save_graph(self):
        """Speichert den aktuell ausgewählten Graph."""
        
        if not hasattr(self.main_window, 'graph_list') or not self.main_window.graph_list:
            QMessageBox.warning(self, "No Graph", "Kein Graph zum Speichern vorhanden!")
            return
        
        current_idx = getattr(self.main_window, 'current_graph_idx', 0)
        if current_idx >= len(self.main_window.graph_list):
            current_idx = 0
        
        graph = self.main_window.graph_list[current_idx]
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Graph speichern",
            f"{graph.graph_name}.json",
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
    
    def load_graph(self):
        """Lädt einen Graph aus einer JSON-Datei."""
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Graph laden", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # Graph-Klasse holen
        Graph = self.Graph_class
        if Graph is None and hasattr(self.main_window, 'Graph'):
            Graph = self.main_window.Graph
        
        if Graph is None:
            QMessageBox.critical(self, "Fehler", "Graph-Klasse nicht verfügbar!")
            return
        
        graph = load_graph(filepath, Graph)
        
        if graph:
            if hasattr(self.main_window, 'graph_list'):
                self.main_window.graph_list.append(graph)
            
            if hasattr(self.main_window, 'update_graph_list'):
                self.main_window.update_graph_list()
            if hasattr(self.main_window, 'update_3d_view'):
                self.main_window.update_3d_view()
            
            QMessageBox.information(self, "Geladen", 
                f"Graph '{graph.graph_name}' erfolgreich geladen!\nNodes: {len(graph.node_list)}")
        else:
            QMessageBox.critical(self, "Fehler", "Fehler beim Laden des Graphs!")


# =============================================================================
# EINFACHE INTEGRATION FUNKTION
# =============================================================================

def add_save_load_buttons(main_window, Graph_class=None, target_layout=None):
    """
    Fügt Save/Load Buttons zu MainWindow hinzu.
    
    Args:
        main_window: Das MainWindow
        Graph_class: Die Graph-Klasse für Load
        target_layout: Optional - Ziel-Layout für die Buttons
    
    Beispiel:
        add_save_load_buttons(self, Graph, self.bottom_right_layout)
    """
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
    print("✅ Save/Load Buttons hinzugefügt")
    return widget





class SimulationControlWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Styles
        btn_style = """
            QPushButton {
                border-radius: 5px;
                font-weight: bold;
                padding: 15px;
                font-size: 14px;
            }
        """
        
        start_style = btn_style + "QPushButton { background-color: #4CAF50; color: white; }"
        stop_style = btn_style + "QPushButton { background-color: #F44336; color: white; }"
        io_style = btn_style + "QPushButton { background-color: #2196F3; color: white; }"
        
        # Buttons
        self.btn_start = QPushButton("Start Simulation")
        self.btn_start.setStyleSheet(start_style)
        
        self.btn_stop = QPushButton("Stop Simulation")
        self.btn_stop.setStyleSheet(stop_style)
        
        self.btn_save = QPushButton("Save Graphs")
        self.btn_save.setStyleSheet(io_style)
    
        self.btn_load = QPushButton("Load Graphs")
        self.btn_load.setStyleSheet(io_style)
        
        # Layout (2 Zeilen, 2 Spalten)
        layout.addWidget(self.btn_start, 0, 0)
        layout.addWidget(self.btn_stop, 0, 1)
        layout.addWidget(self.btn_save, 1, 0)
        layout.addWidget(self.btn_load, 1, 1)


class NumpyEncoder(json.JSONEncoder):
    """Konvertiert numpy types zu JSON-kompatiblen Python types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return super().default(obj)


def _clean_params(params: dict) -> dict:
    """Bereinigt Parameters für JSON (numpy → list/float/int)."""
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
    """Serialisiert Connections für JSON."""
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
    """
    Speichert einen Graph als JSON.
    
    Args:
        graph: Graph-Objekt
        filepath: Ziel-Pfad (.json)
    
    Returns:
        True bei Erfolg
    """
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
        
        print(f"✅ Saved: {graph.graph_name} ({len(graph.node_list)} nodes) → {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Save failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_graph(filepath: str, Graph_class):
    """
    Lädt einen Graph aus JSON.
    
    Args:
        filepath: JSON-Datei Pfad
        Graph_class: Graph-Klasse zum Instanziieren
    
    Returns:
        Graph-Objekt oder None
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        g_data = data['graph']
        nodes_data = sorted(data['nodes'], key=lambda x: x['id'])
        
        print(f"📂 Loading: {g_data['graph_name']} ({len(nodes_data)} nodes)...")
        
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
        
        print(f"✅ Loaded: {graph.graph_name} ({len(graph.node_list)} nodes)")
        return graph
        
    except Exception as e:
        print(f"❌ Load failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# ENDE GRAPH I/O BLOCK
# =============================================================================
