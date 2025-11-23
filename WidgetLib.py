import sys
from PyQt6.QtWidgets import QApplication, QMainWindow,QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtWidgets import QPushButton,QLabel, QStackedWidget, QToolBar, QMenu, QGridLayout, QStackedLayout
from PyQt6.QtGui import QColor, QPalette, QAction,QIcon
from PyQt6.QtCore import QSize, Qt, pyqtSignal
import pyvista as pv
from pyvistaqt import QtInteractor
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
import numpy as np
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
        self.add_vector3_field("m", "Center (m)")
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
        
        if key == "center_of_mass":
            x_spin.valueChanged.connect(lambda v: self.sync_to_m('x', v))
            y_spin.valueChanged.connect(lambda v: self.sync_to_m('y', v))
            z_spin.valueChanged.connect(lambda v: self.sync_to_m('z', v))
        elif key == "m":
            x_spin.valueChanged.connect(lambda v: self.sync_to_com('x', v))
            y_spin.valueChanged.connect(lambda v: self.sync_to_com('y', v))
            z_spin.valueChanged.connect(lambda v: self.sync_to_com('z', v))
        
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
    
    def sync_to_m(self, axis, value):
        if 'm' in self.widgets:
            idx = {'x': 0, 'y': 1, 'z': 2}[axis]
            self.widgets['m']['widgets'][idx].blockSignals(True)
            self.widgets['m']['widgets'][idx].setValue(value)
            self.widgets['m']['widgets'][idx].blockSignals(False)
    
    def sync_to_com(self, axis, value):
        if 'center_of_mass' in self.widgets:
            idx = {'x': 0, 'y': 1, 'z': 2}[axis]
            self.widgets['center_of_mass']['widgets'][idx].blockSignals(True)
            self.widgets['center_of_mass']['widgets'][idx].setValue(value)
            self.widgets['center_of_mass']['widgets'][idx].blockSignals(False)
    
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
class PolynomialManagerWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.population_polynomials = {}  # {pop_idx: PolynomialTrioWidget}
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
    
    def set_populations(self, population_list):
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
            
            if 'polynomials' in pop and pop['polynomials']:
                trio.load_polynomials(pop['polynomials'])
            
            self.content_layout.addWidget(trio)
            self.population_polynomials[i] = trio
            
            if i < len(population_list) - 1:
                separator = QLabel()
                separator.setStyleSheet("background-color: #DDD; min-height: 2px; max-height: 2px;")
                self.content_layout.addWidget(separator)
        
        self.content_layout.addStretch()
    
    def get_all_polynomials(self):
        return [trio.get_polynomials() for trio in self.population_polynomials.values()]

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
    
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Graph Name:"))
        self.graph_name_input = QLineEdit()
        self.graph_name_input.setPlaceholderText("e.g., Visual Cortex")
        header_layout.addWidget(self.graph_name_input)
        main_layout.addLayout(header_layout)
        
        content_layout = QHBoxLayout()
        
        node_col = QVBoxLayout()
        node_col.addWidget(QLabel("NODES", alignment=Qt.AlignmentFlag.AlignCenter))
        
        node_scroll = QScrollArea()
        node_scroll.setWidgetResizable(True)
        self.node_list_widget = QWidget()
        self.node_list_layout = QVBoxLayout(self.node_list_widget)
        node_scroll.setWidget(self.node_list_widget)
        node_col.addWidget(node_scroll)
        
        add_node_btn = QPushButton("+ Add Node")
        add_node_btn.clicked.connect(self.add_node)
        add_node_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        node_col.addWidget(add_node_btn)
        
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
        
        self.polynom_manager.set_populations(node['populations'])
        self.editor_stack.setCurrentIndex(3)
    
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
        node_btn.clicked.connect(lambda: self.select_node(node_idx))
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
        """Wählt einen Node aus"""
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
        if self.current_node_idx is not None and self.current_pop_idx is not None:
            node = self.node_list[self.current_node_idx]
            if self.current_pop_idx < len(node['populations']):
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
