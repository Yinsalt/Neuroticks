import sys
from PyQt6.QtWidgets import QApplication, QMainWindow,QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtWidgets import QPushButton,QLabel, QStackedWidget, QToolBar, QMenu, QGridLayout, QStackedLayout
from PyQt6.QtGui import QColor, QPalette, QAction,QIcon
from PyQt6.QtCore import QSize, Qt, pyqtSignal,QTimer
import pyvista as pv
from pyvistaqt import QtInteractor
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.colors as mcolors
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
"description": "Time constant of STDP window, potentiation (tau_minus defined in postsynaptic neuron)"
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
"description": "Asymmetry parameter (scales depressing increments as alpha*lambda)"
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
"tsunoda_synapse": {},
"vogels_sprekeler_synapse": {},
"astrocyte_lr_1994": {},
"sic_connection": {}
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

    
    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list
        
        layout = QVBoxLayout()
        
        header = QLabel("GRAPH OVERVIEW")
        header.setStyleSheet("font-weight: bold; font-size: 14px; color: orange;")
        layout.addWidget(header)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Graphs & Nodes")
        layout.addWidget(self.tree)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.update_tree)
        layout.addWidget(refresh_btn)
        
        self.setLayout(layout)
        
        self.update_tree()
    
    def update_tree(self):
        self.tree.clear()
        
        if not self.graph_list:
            placeholder = QTreeWidgetItem(self.tree)
            placeholder.setText(0, "No graphs available")
            return
        
        for graph in self.graph_list:
            graph_name = getattr(graph, 'graph_name', f'Graph {graph.graph_id}')
            
            graph_item = QTreeWidgetItem(self.tree)
            graph_item.setText(0, f"{graph_name} ({len(graph.node_list)} Nodes)")
            
            for node in graph.node_list:
                node_item = QTreeWidgetItem(graph_item)
                
                total_neurons = 0
                if hasattr(node, 'positions') and node.positions:
                    total_neurons = sum(len(cluster) for cluster in node.positions if len(cluster) > 0)
                
                node_name = getattr(node, 'name', 'Unnamed')
                node_item.setText(0, f"{node_name} (ID: {node.id}) - {total_neurons} neurons")
                
                if hasattr(node, 'parameters') and 'populations' in node.parameters:
                    for i, pop in enumerate(node.parameters['populations']):
                        pop_item = QTreeWidgetItem(node_item)
                        pop_item.setText(0, f"Pop {i+1}: {pop['model']}")
            
            if self.tree.topLevelItemCount() > 0:
                self.tree.topLevelItem(0).setExpanded(True)

################################################################################

node_parameters1 = {
    "grid_size": [10, 10, 10],
    "m": [0.0, 0.0, 0.0],
    "rot_theta": 0.0,
    "rot_phi": 0.0,
    "transform_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "dt": 0.01,
    "old": True,
    "num_steps": 8,
    "sparse_holes": 0,
    "sparsity_factor": 0.9,
    "probability_vector": [0.3, 0.2, 0.4],
    "name": "Node",
    "id": 0,
    "polynom_max_power": 5,
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
        
        old_node_count = len(self.current_graph.node_list)
        new_node_count = len(self.node_list)

        new_nodes = [n for n in self.node_list if n.get('original_node') is None]

        # ✅ UPDATED: 3 Werte!
        edited_analysis = [self._node_was_edited(n) for n in self.node_list]
        edited_nodes = [n for n, (changed, _, _) in zip(self.node_list, edited_analysis) if changed]
        nodes_with_structural_changes = [
            (n, i) for i, (n, (_, _, structural)) in enumerate(zip(self.node_list, edited_analysis)) 
            if structural
        ]

        removed = new_node_count < old_node_count

        print(f"  Analysis: {len(new_nodes)} new, {len(edited_nodes)} edited, "
            f"{len(nodes_with_structural_changes)} structural changes, {removed=}")
        
        # ================================
        # FALL 1: NUR NEUE NODES
        # → Kein Reset, nur neue builden
        # ================================
        if not new_nodes and not edited_nodes and not removed:
            print("No structural changes – skipping rebuild")
            return
        if new_nodes and not edited_nodes and not removed:
            print("📝 Adding new nodes only (no NEST reset)")
            
            for node_idx, node_data in enumerate(self.node_list):
                # Skip existing nodes
                if node_data.get('original_node') is not None:
                    continue
                
                node_params = self._build_node_params(node_idx, node_data)
                
                # ✅ Finde Parent
                if node_idx == 0:
                    parent = None
                    is_root = True
                else:
                    # Parent ist der vorherige Node in UI-Liste
                    # Kann alt (original_node) ODER neu sein (gerade erstellt)
                    parent = self.current_graph.node_list[node_idx - 1]
                    is_root = False
                
                new_node = self.current_graph.create_node(
                    parameters=node_params,
                    other=parent,
                    is_root=is_root,
                    auto_build=True  # ✅ WFC + populate
                )
                new_node.populate_node()
                
                print(f"  ✅ Added Node {node_idx}: {new_node.name}")
        
        # ================================
        # FALL 2: EDITS oder REMOVES
        # → Reset + Rebuild mit Translation
        # ================================
        else:
            print("✏️  Nodes edited/removed - NEST reset required")
            nest.ResetKernel()
            
            self.current_graph.node_list.clear()
            self.current_graph.nodes = 0
            self.current_graph._next_id = 0
            
            # ✅ Build dict für schnelles Lookup
            structural_change_indices = {idx for _, idx in nodes_with_structural_changes}
            
            for node_idx, node_data in enumerate(self.node_list):
                node_params = self._build_node_params(node_idx, node_data)
                original_node = node_data.get('original_node')
                
                # Create node WITHOUT auto_build
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
                # ✅ DECISION LOGIC
                # ================================
                has_structural_change = node_idx in structural_change_indices
                
                if has_structural_change:
                    # ✅ Structural change → COMPLETE REBUILD
                    print(f"  Node {node_idx}: REBUILD (structural changes detected)")
                    new_node.build()  # WFC mit neuen Parametern
                    
                elif original_node and hasattr(original_node, 'positions') and original_node.positions:
                    # Kein structural change → Check COM für Translation
                    old_com = np.array(original_node.center_of_mass)
                    new_com = np.array(node_params['center_of_mass'])
                    delta = new_com - old_com
                    
                    if np.any(np.abs(delta) > 1e-6):
                        # COM geändert → Translation
                        print(f"  Node {node_idx}: Translation by {delta}")
                        new_node.positions = []
                        for pos_cluster in original_node.positions:
                            if pos_cluster is not None and len(pos_cluster) > 0:
                                translated = pos_cluster + delta
                                new_node.positions.append(translated)
                            else:
                                new_node.positions.append(np.array([]))
                    else:
                        # Nichts strukturelles geändert → Positionen kopieren
                        print(f"  Node {node_idx}: Reusing positions (no structural changes)")
                        new_node.positions = [cluster.copy() for cluster in original_node.positions]
                    
                    new_node.center_of_mass = new_com
                else:
                    # Neuer Node → WFC
                    print(f"  Node {node_idx}: NEW - generating positions")
                    new_node.build()
                
                # ✅ Populate NEST
                new_node.populate_node()
            
            # Repopulate other graphs
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
        params_layout = QVBoxLayout(params_widget)
        
        # Connection Name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Connection Name:"))
        self.conn_name_input = QLineEdit()
        self.conn_name_input.setPlaceholderText("Auto-generated if empty")
        name_layout.addWidget(self.conn_name_input)
        params_layout.addLayout(name_layout)
        
        # ✅ Connection Rule - ALLE NEST RULES
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
        params_layout.addLayout(rule_layout)
        
        # === DYNAMIC PARAMETERS ===
        
        # Indegree
        self.indegree_layout = QHBoxLayout()
        self.indegree_layout.addWidget(QLabel("Indegree:"))
        self.indegree_spin = QSpinBox()
        self.indegree_spin.setRange(1, 10000)
        self.indegree_spin.setValue(100)
        self.indegree_layout.addWidget(self.indegree_spin)
        params_layout.addLayout(self.indegree_layout)
        
        # Outdegree
        self.outdegree_layout = QHBoxLayout()
        self.outdegree_layout.addWidget(QLabel("Outdegree:"))
        self.outdegree_spin = QSpinBox()
        self.outdegree_spin.setRange(1, 10000)
        self.outdegree_spin.setValue(100)
        self.outdegree_layout.addWidget(self.outdegree_spin)
        params_layout.addLayout(self.outdegree_layout)
        
        # Total Number
        self.total_num_layout = QHBoxLayout()
        self.total_num_layout.addWidget(QLabel("Total Number:"))
        self.total_num_spin = QSpinBox()
        self.total_num_spin.setRange(1, 100000)
        self.total_num_spin.setValue(1000)
        self.total_num_layout.addWidget(self.total_num_spin)
        params_layout.addLayout(self.total_num_layout)
        
        # Probability
        self.prob_layout = QHBoxLayout()
        self.prob_layout.addWidget(QLabel("Probability:"))
        self.prob_spin = QDoubleSpinBox()
        self.prob_spin.setRange(0.0, 1.0)
        self.prob_spin.setValue(0.1)
        self.prob_spin.setSingleStep(0.05)
        self.prob_spin.setDecimals(4)
        self.prob_layout.addWidget(self.prob_spin)
        params_layout.addLayout(self.prob_layout)
        
        # Weight
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(QLabel("Weight:"))
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(-1000.0, 1000.0)
        self.weight_spin.setValue(1.0)
        self.weight_spin.setDecimals(3)
        weight_layout.addWidget(self.weight_spin)
        params_layout.addLayout(weight_layout)
        
        # Delay
        delay_layout = QHBoxLayout()
        delay_layout.addWidget(QLabel("Delay (ms):"))
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 100.0)
        self.delay_spin.setValue(1.0)
        self.delay_spin.setDecimals(2)
        delay_layout.addWidget(self.delay_spin)
        params_layout.addLayout(delay_layout)
        
        # Synapse Model - ERWEITERT
        syn_layout = QHBoxLayout()
        syn_layout.addWidget(QLabel("Synapse Model:"))
        self.syn_model_combo = QComboBox()
        self.syn_model_combo.addItems([
            "static_synapse",
            "stdp_synapse",
            "stdp_synapse_hom",
            "stdp_pl_synapse_hom",
            "stdp_triplet_synapse",
            "tsodyks_synapse",
            "tsodyks2_synapse",
            "quantal_stp_synapse",
            "vogels_sprekeler_synapse",
            "clopath_synapse",
            "gap_junction"
        ])
        syn_layout.addWidget(self.syn_model_combo)
        params_layout.addLayout(syn_layout)
        
        # Allow Autapses
        autapses_layout = QHBoxLayout()
        self.allow_autapses_check = QCheckBox("Allow Autapses (self-connections)")
        self.allow_autapses_check.setChecked(True)
        autapses_layout.addWidget(self.allow_autapses_check)
        params_layout.addLayout(autapses_layout)
        
        # Allow Multapses
        multapses_layout = QHBoxLayout()
        self.allow_multapses_check = QCheckBox("Allow Multapses (multiple connections)")
        self.allow_multapses_check.setChecked(True)
        multapses_layout.addWidget(self.allow_multapses_check)
        params_layout.addLayout(multapses_layout)
        
        params_layout.addStretch()
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
        
        # ✅ CREATE ALL - Schreibt zu Source Nodes
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
        print(f"[ConnectionTool] Refreshing... graph_list has {len(self.graph_list)} graphs")
        
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
        
        if index < 0:
            return
        
        graph_id = self.source_graph_combo.currentData()
        if graph_id is None:
            return
        
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph:
            return
        
        for node in graph.node_list:
            node_name = getattr(node, 'name', f'Node {node.id}')
            self.source_node_combo.addItem(f"{node_name} (ID: {node.id})", node.id)
        
        if len(graph.node_list) > 0:
            self.on_source_node_changed(0)
    
    def on_target_graph_changed(self, index):
        self.target_node_combo.clear()
        self.target_pop_combo.clear()
        
        if index < 0:
            return
        
        graph_id = self.target_graph_combo.currentData()
        if graph_id is None:
            return
        
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph:
            return
        
        for node in graph.node_list:
            node_name = getattr(node, 'name', f'Node {node.id}')
            self.target_node_combo.addItem(f"{node_name} (ID: {node.id})", node.id)
        
        if len(graph.node_list) > 0:
            self.on_target_node_changed(0)
    
    def on_source_node_changed(self, index):
        self.source_pop_combo.clear()
        
        graph_id = self.source_graph_combo.currentData()
        node_id = self.source_node_combo.currentData()
        
        if graph_id is None or node_id is None:
            return
        
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph:
            return
        
        node = next((n for n in graph.node_list if n.id == node_id), None)
        if not node:
            return
        
        for i, pop in enumerate(node.population):
            model = node.parameters['neuron_models'][i] if i < len(node.parameters['neuron_models']) else 'unknown'
            self.source_pop_combo.addItem(f"Pop {i}: {model} ({len(pop)} neurons)", i)
    
    def on_target_node_changed(self, index):
        self.target_pop_combo.clear()
        
        graph_id = self.target_graph_combo.currentData()
        node_id = self.target_node_combo.currentData()
        
        if graph_id is None or node_id is None:
            return
        
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph:
            return
        
        node = next((n for n in graph.node_list if n.id == node_id), None)
        if not node:
            return
        
        for i, pop in enumerate(node.population):
            model = node.parameters['neuron_models'][i] if i < len(node.parameters['neuron_models']) else 'unknown'
            self.target_pop_combo.addItem(f"Pop {i}: {model} ({len(pop)} neurons)", i)
    
    def add_connection(self):
        """Fügt Connection zur temporären Liste hinzu"""
        source_graph_id = self.source_graph_combo.currentData()
        source_node_id = self.source_node_combo.currentData()
        source_pop_id = self.source_pop_combo.currentData()
        
        target_graph_id = self.target_graph_combo.currentData()
        target_node_id = self.target_node_combo.currentData()
        target_pop_id = self.target_pop_combo.currentData()
        
        if source_graph_id is None or target_graph_id is None:
            print("⚠ Please select source and target!")
            return
        
        conn_name = self.conn_name_input.text().strip()
        if not conn_name:
            conn_name = f"Conn_{self.next_conn_id}: G{source_graph_id}N{source_node_id}P{source_pop_id} → G{target_graph_id}N{target_node_id}P{target_pop_id}"
        
        # Rule-spezifische Parameter
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
                'weight': self.weight_spin.value(),
                'delay': self.delay_spin.value(),
                'synapse_model': self.syn_model_combo.currentText(),
                'allow_autapses': self.allow_autapses_check.isChecked(),
                'allow_multapses': self.allow_multapses_check.isChecked()
            }
        }
        
        self.connections.append(connection)
        self.next_conn_id += 1
        
        print(f"✅ Connection added to temp list: {conn_name}")
        
        self.update_connection_list()
        self.reset_interface()
    
    def create_all_connections(self):
        """✅ Schreibt alle Connections zu ihren Source-Nodes"""
        if not self.connections:
            print("⚠ No connections to create!")
            return
        
        print("\n" + "="*70)
        print("🚀 CREATING ALL CONNECTIONS - Writing to Source Nodes")
        print("="*70)
        
        for conn in self.connections:
            # Finde Source Graph & Node
            source_graph_id = conn['source']['graph_id']
            source_node_id = conn['source']['node_id']
            
            graph = next((g for g in self.graph_list if g.graph_id == source_graph_id), None)
            if not graph:
                print(f"⚠ Graph {source_graph_id} not found!")
                continue
            
            node = next((n for n in graph.node_list if n.id == source_node_id), None)
            if not node:
                print(f"⚠ Node {source_node_id} not found in Graph {source_graph_id}!")
                continue
            
            # ✅ Schreibe Connection zu Node
            if not hasattr(node, 'connections'):
                node.connections = []
            
            node.connections.append(conn)
            
            print(f"✅ {conn['name']}")
            print(f"   → Stored in Graph {source_graph_id}, Node {source_node_id}")
            print(f"   → Node now has {len(node.connections)} connection(s)")
        
        print("="*70)
        print(f"✅ All {len(self.connections)} connections written to source nodes!")
        print("="*70 + "\n")
        
        # ✅ Clear Interface
        self.connections.clear()
        self.next_conn_id = 0
        self.current_conn_idx = None
        self.update_connection_list()
        self.reset_interface()
        
        print("🧹 Interface cleared and ready for new connections\n")
    
    def delete_connection(self):
        if self.current_conn_idx is None:
            return
        
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
        print(f"   Source: G{conn['source']['graph_id']} N{conn['source']['node_id']} P{conn['source']['pop_id']}")
        print(f"   Target: G{conn['target']['graph_id']} N{conn['target']['node_id']} P{conn['target']['pop_id']}\n")
    
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
        self.syn_model_combo.setCurrentIndex(0)
        self.allow_autapses_check.setChecked(True)
        self.allow_multapses_check.setChecked(True)
        
        if self.source_graph_combo.count() > 0:
            self.source_graph_combo.setCurrentIndex(0)
        if self.target_graph_combo.count() > 0:
            self.target_graph_combo.setCurrentIndex(0)
"""
for graph in graph_list:
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
        self.layout.addWidget(self.plotter)
        
        # State Arrays
        self.neuron_mesh = None
        self.edge_mesh = None
        self.neuron_intensities = None
        self.edge_intensities = None
        self.base_colors = None
        
        # Timer Setup
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        
        # Initial Build
        self.build_scene()
        
        # Start Animation (ca. 30 FPS)
        self.timer.start(33)

    def build_scene(self):
        self.plotter.clear()
        
        all_points = []
        all_colors = []
        
        # 1. Punkte sammeln
        # import matplotlib.colors as mcolors # Nicht hier, besser oben im File
        for graph in self.graph_list:
            for node in graph.node_list:
                for i, cluster in enumerate(node.positions):
                    if len(cluster) == 0: continue
                    
                    model = node.neuron_models[i] if i < len(node.neuron_models) else "unknown"
                    hex_c = neuron_colors.get(model, "#ffffff")
                    rgb = mcolors.to_rgb(hex_c)
                    
                    all_points.append(cluster)
                    all_colors.extend([rgb] * len(cluster))
        
        if not all_points:
            return

        points = np.vstack(all_points)
        n_points = len(points)
        self.base_colors = np.array(all_colors)
        
        # --- A. NEURONEN MESH (Punkte) ---
        self.neuron_mesh = pv.PolyData(points)
        self.neuron_intensities = np.random.uniform(0, 1, n_points)
        self.neuron_mesh.point_data["display_color"] = self.base_colors * self.neuron_intensities[:, None]
        
        self.plotter.add_mesh(
            self.neuron_mesh,
            scalars="display_color",
            rgb=True,
            point_size=8,
            render_points_as_spheres=True,
            ambient=0.5
        )
        
        # --- B. RANDOM KANTEN MESH (Linien) ---
        n_edges = int(n_points * 2.5) 
        
        # Indizes
        idx_a = np.random.randint(0, n_points, n_edges)
        idx_b = np.random.randint(0, n_points, n_edges)
        
        # PyVista Line Format
        lines = np.column_stack((np.full(n_edges, 2), idx_a, idx_b)).flatten()
        
        # FIX 1: Lines direkt im Konstruktor (Verhindert Topologie-Fehler)
        self.edge_mesh = pv.PolyData(points, lines=lines)
        
        # Edge Intensität array erstellen
        self.edge_intensities = np.zeros(n_edges)
        self.edge_mesh.cell_data["activity"] = self.edge_intensities
        
        # FIX 2: Explizite Colormap erstellen (Verhindert n_colors Fehler)
        # Wir erstellen eine kontinuierliche Map von Schwarz nach Gelb mit 256 Stufen
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("black_yellow", ["black", "yellow"], N=256)
        
        self.plotter.add_mesh(
            self.edge_mesh,
            scalars="activity",
            cmap=custom_cmap,     # Übergabe des Objekts statt der Liste
            opacity="linear",     # Jetzt passen 256 Opacity-Werte auf 256 Farben
            line_width=2,
            show_scalar_bar=False,
            use_transparency=True
        )
        
        self.plotter.reset_camera()

    def animate(self):
        if self.neuron_mesh is None: return
        
        # 1. Neuronen Blinken (Chaotisch)
        # Wir ändern die Helligkeit zufällig, aber smooth
        noise = np.random.normal(0, 0.1, len(self.neuron_intensities))
        self.neuron_intensities = np.clip(self.neuron_intensities + noise, 0.2, 1.0)
        
        # Farbe updaten (Base Color * Intensität)
        new_colors = self.base_colors * self.neuron_intensities[:, None]
        self.neuron_mesh.point_data["display_color"] = new_colors
        
        # 2. Kanten Blitzen (Spiking)
        # A) Decay: Alles wird langsam transparent
        self.edge_intensities *= 0.90 
        
        # B) Random Spikes: Zufällige Kanten auf 1.0 setzen
        # Wahrscheinlichkeit, dass eine Kante feuert
        n_edges = len(self.edge_intensities)
        n_fire = int(n_edges * 0.02) # 2% feuern pro Frame
        if n_fire > 0:
            fire_indices = np.random.choice(n_edges, n_fire, replace=False)
            self.edge_intensities[fire_indices] = 1.0
            
        self.edge_mesh.cell_data["activity"] = self.edge_intensities
        
        self.plotter.update()
