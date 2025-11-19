import sys
from PyQt6.QtWidgets import QApplication, QMainWindow,QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtWidgets import QPushButton, QStackedWidget, QToolBar, QMenu, QGridLayout, QStackedLayout
from PyQt6.QtGui import QColor, QPalette, QAction
from PyQt6.QtCore import QSize, Qt, pyqtSignal
import pyvista as pv
from pyvistaqt import QtInteractor
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from neuron_toolbox import *
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
)
from PyQt6.QtWidgets import QScrollArea, QInputDialog



class Window(QWidget):
    def __init__(self):
        super().__init__(self)




class Color(QWidget):
    def __init__(self, color):
        super().__init__()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)

class DoubleInputField(QWidget):
    def __init__(self, param_name, default_value=0.0, min_val=0.0, max_val=100.0, decimals=2):
        super().__init__()
        
        self.param_name = param_name
        
        # Layout
        layout = QHBoxLayout()
        
        # Label
        self.label = QLabel(f"{param_name}:")
        
        # DoubleSpinBox
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





from PyQt6.QtWidgets import QScrollArea, QInputDialog


class PercentageInputField(QWidget):
    def __init__(self, param_name, default_value=0.0):
        super().__init__()
        
        self.param_name = param_name
        
        layout = QHBoxLayout()
        
        self.label = QLabel(f"{param_name}:")
        
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(0.0, 100.0)
        self.spinbox.setDecimals(2)
        self.spinbox.setSuffix(" %")
        self.spinbox.setValue(default_value)
        
        layout.addWidget(self.label)
        layout.addWidget(self.spinbox)
        self.setLayout(layout)
    
    def get_value(self):
        return self.spinbox.value()

















class polynomEditorTool(QWidget):
    def __init__(self, graph_list=None):
        super().__init__()
        
        self.graph_list = graph_list if graph_list else []
        self.current_polynomial_widgets = {}
        
        # Main Layout 30:70 (Selection : Polynomial Editor)
        self.main_layout = QHBoxLayout()
        
        # === LINKE SEITE: Population Selection (30%) ===
        self.selection_widget = QWidget()
        self.selection_layout = QVBoxLayout()
        self.selection_widget.setLayout(self.selection_layout)
        
        selection_label = QLabel("SELECT POPULATION")
        selection_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.selection_layout.addWidget(selection_label)
        
        self.graph_select = DropdownField("Graph", self.get_graph_names(), default_index=0)
        self.node_select = DropdownField("Node", ["Select Graph first"], default_index=0)
        self.population_select = DropdownField("Population", ["Select Node first"], default_index=0)
        
        self.graph_select.combobox.currentTextChanged.connect(self.on_graph_changed)
        self.node_select.combobox.currentTextChanged.connect(self.on_node_changed)
        self.population_select.combobox.currentTextChanged.connect(self.on_population_changed)
        
        self.selection_layout.addWidget(self.graph_select)
        self.selection_layout.addWidget(self.node_select)
        self.selection_layout.addWidget(self.population_select)
        
        # Current Polynomial Info Display
        info_label = QLabel("CURRENT POLYNOMIAL")
        info_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 20px;")
        self.selection_layout.addWidget(info_label)
        
        self.current_poly_display = QTextEdit()
        self.current_poly_display.setReadOnly(True)
        self.current_poly_display.setMaximumHeight(200)
        self.current_poly_display.setPlaceholderText("No polynomial set")
        self.selection_layout.addWidget(self.current_poly_display)
        
        self.selection_layout.addStretch()
        
        # === RECHTE SEITE: Polynomial Editor (70%) ===
        self.editor_widget = QWidget()
        self.editor_scroll = QScrollArea()
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setWidget(self.editor_widget)
        
        self.editor_layout = QVBoxLayout()
        
        editor_title = QLabel("POLYNOMIAL EDITOR")
        editor_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.editor_layout.addWidget(editor_title)
        
        # Polynomial Type Selection
        self.poly_type = DropdownField(
            "Polynomial Type",
            ["linear", "quadratic", "cubic", "quartic", "custom"],
            default_index=0
        )
        self.poly_type.combobox.currentTextChanged.connect(self.on_poly_type_changed)
        self.editor_layout.addWidget(self.poly_type)
        
        # Anzahl Variablen
        self.num_vars = IntegerInputField("Number of Variables", default_value=2, min_val=1, max_val=10)
        self.num_vars.spinbox.valueChanged.connect(self.on_num_vars_changed)
        self.editor_layout.addWidget(self.num_vars)
        
        # Container für Koeffizienten
        coeff_label = QLabel("COEFFICIENTS")
        coeff_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 10px;")
        self.editor_layout.addWidget(coeff_label)
        
        self.coeff_container_widget = QWidget()
        self.coeff_container_layout = QVBoxLayout()
        self.coeff_container_widget.setLayout(self.coeff_container_layout)
        self.editor_layout.addWidget(self.coeff_container_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate Polynomial Matrix")
        self.generate_btn.clicked.connect(self.generate_polynomial_matrix)
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setStyleSheet("background-color: rgba(0, 100, 255, 100); font-weight: bold;")
        
        self.apply_btn = QPushButton("Apply to Population")
        self.apply_btn.clicked.connect(self.apply_to_population)
        self.apply_btn.setMinimumHeight(40)
        self.apply_btn.setStyleSheet("background-color: rgba(0, 255, 0, 100); font-weight: bold;")
        
        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.apply_btn)
        self.editor_layout.addLayout(button_layout)
        
        # Matrix Display
        matrix_label = QLabel("GENERATED COEFFICIENT MATRIX")
        matrix_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 10px;")
        self.editor_layout.addWidget(matrix_label)
        
        self.matrix_display = QTextEdit()
        self.matrix_display.setReadOnly(True)
        self.matrix_display.setMaximumHeight(250)
        self.matrix_display.setPlaceholderText("Click 'Generate Polynomial Matrix' to see coefficient matrix")
        self.editor_layout.addWidget(self.matrix_display)
        
        self.editor_layout.addStretch()
        self.editor_widget.setLayout(self.editor_layout)
        
        # Add to Main Layout
        self.main_layout.addWidget(self.selection_widget, 30)
        self.main_layout.addWidget(self.editor_scroll, 70)
        self.setLayout(self.main_layout)
        
        # Initial setup
        self.current_poly_matrix = None
        self.on_poly_type_changed(self.poly_type.get_value())
    
    def get_graph_names(self):
        if not self.graph_list:
            return ["No graphs available"]
        return [f"Graph {graph.graph_id}" for graph in self.graph_list]
    
    def get_graph_by_name(self, graph_name):
        try:
            graph_id = int(graph_name.split()[1])
            for graph in self.graph_list:
                if graph.graph_id == graph_id:
                    return graph
        except:
            pass
        return None
    
    def on_graph_changed(self, graph_name):
        graph = self.get_graph_by_name(graph_name)
        if graph:
            node_names = [f"Node {i+1}" for i in range(len(graph.node_list))]
            self.node_select.combobox.clear()
            self.node_select.combobox.addItems(node_names if node_names else ["No nodes"])
    
    def on_node_changed(self, node_name):
        graph = self.get_graph_by_name(self.graph_select.get_value())
        if graph:
            try:
                node_id = int(node_name.split()[1]) - 1
                node = graph.node_list[node_id]
                pop_names = [f"Pop {i+1}" for i in range(len(node.populations))]
                self.population_select.combobox.clear()
                self.population_select.combobox.addItems(pop_names if pop_names else ["No populations"])
            except:
                pass
    
    def on_population_changed(self, pop_name):
        """Lädt aktuelles Polynom der Population"""
        graph = self.get_graph_by_name(self.graph_select.get_value())
        if graph:
            try:
                node_id = int(self.node_select.get_value().split()[1]) - 1
                pop_id = int(pop_name.split()[1]) - 1
                node = graph.node_list[node_id]
                
                # Zeige aktuelles Polynom falls vorhanden
                if hasattr(node, 'flow_functions') and pop_id < len(node.flow_functions):
                    flow_func = node.flow_functions[pop_id]
                    self.current_poly_display.setText(f"Flow Function:\n{flow_func}")
                else:
                    self.current_poly_display.setText("No polynomial set for this population")
            except:
                self.current_poly_display.setText("Error loading population data")
    
    def on_poly_type_changed(self, poly_type):
        """Ändert die Anzahl der Koeffizientenfelder basierend auf Typ"""
        type_to_degree = {
            'linear': 1,
            'quadratic': 2,
            'cubic': 3,
            'quartic': 4,
            'custom': 3  # default für custom
        }
        degree = type_to_degree.get(poly_type, 2)
        self.create_coefficient_fields(degree, self.num_vars.get_value())
    
    def on_num_vars_changed(self):
        """Neu-erstellt Koeffizientenfelder mit neuer Variablenanzahl"""
        poly_type = self.poly_type.get_value()
        type_to_degree = {
            'linear': 1,
            'quadratic': 2,
            'cubic': 3,
            'quartic': 4,
            'custom': 3
        }
        degree = type_to_degree.get(poly_type, 2)
        self.create_coefficient_fields(degree, self.num_vars.get_value())
    
    def create_coefficient_fields(self, degree, num_vars):
        """Erstellt Eingabefelder für alle Koeffizienten"""
        # Lösche alte Felder
        while self.coeff_container_layout.count():
            item = self.coeff_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.current_polynomial_widgets.clear()
        
        # Konstante
        const_widget = DoubleInputField("Constant (c0)", default_value=0.0, min_val=-100.0, max_val=100.0, decimals=4)
        self.coeff_container_layout.addWidget(const_widget)
        self.current_polynomial_widgets['c0'] = const_widget
        
        # Lineare Terme
        for i in range(num_vars):
            widget = DoubleInputField(f"x{i+1} coefficient", default_value=0.0, min_val=-100.0, max_val=100.0, decimals=4)
            self.coeff_container_layout.addWidget(widget)
            self.current_polynomial_widgets[f'x{i+1}'] = widget
        
        # Höhere Grade
        if degree >= 2:
            # Quadratische Terme
            for i in range(num_vars):
                widget = DoubleInputField(f"x{i+1}² coefficient", default_value=0.0, min_val=-100.0, max_val=100.0, decimals=4)
                self.coeff_container_layout.addWidget(widget)
                self.current_polynomial_widgets[f'x{i+1}^2'] = widget
            
            # Kreuzterme
            for i in range(num_vars):
                for j in range(i+1, num_vars):
                    widget = DoubleInputField(f"x{i+1}*x{j+1} coefficient", default_value=0.0, min_val=-100.0, max_val=100.0, decimals=4)
                    self.coeff_container_layout.addWidget(widget)
                    self.current_polynomial_widgets[f'x{i+1}*x{j+1}'] = widget
        
        if degree >= 3:
            # Kubische Terme (vereinfacht - nur x^3 Terme)
            for i in range(num_vars):
                widget = DoubleInputField(f"x{i+1}³ coefficient", default_value=0.0, min_val=-100.0, max_val=100.0, decimals=4)
                self.coeff_container_layout.addWidget(widget)
                self.current_polynomial_widgets[f'x{i+1}^3'] = widget
    
    def generate_polynomial_matrix(self):
        """Generiert Koeffizientenmatrix aus Eingaben"""
        num_vars = self.num_vars.get_value()
        poly_type = self.poly_type.get_value()
        
        # Sammle Koeffizienten
        coeffs = {}
        for key, widget in self.current_polynomial_widgets.items():
            coeffs[key] = widget.get_value()
        
        # Erstelle Matrix-Representation
        # Format: [konstante, [lineare], [[quadratische]], ...]
        matrix = {
            'constant': coeffs.get('c0', 0.0),
            'linear': [coeffs.get(f'x{i+1}', 0.0) for i in range(num_vars)],
            'quadratic': [],
            'cubic': []
        }
        
        # Quadratische Matrix
        if poly_type in ['quadratic', 'cubic', 'quartic', 'custom']:
            quad_matrix = np.zeros((num_vars, num_vars))
            for i in range(num_vars):
                quad_matrix[i, i] = coeffs.get(f'x{i+1}^2', 0.0)
            
            for i in range(num_vars):
                for j in range(i+1, num_vars):
                    val = coeffs.get(f'x{i+1}*x{j+1}', 0.0)
                    quad_matrix[i, j] = val / 2  # Symmetrisch aufteilen
                    quad_matrix[j, i] = val / 2
            
            matrix['quadratic'] = quad_matrix.tolist()
        
        # Kubische Terme (diagonal)
        if poly_type in ['cubic', 'quartic', 'custom']:
            cubic_diag = [coeffs.get(f'x{i+1}^3', 0.0) for i in range(num_vars)]
            matrix['cubic'] = cubic_diag
        
        self.current_poly_matrix = matrix
        
        # Zeige Matrix
        display_text = f"POLYNOMIAL REPRESENTATION\n"
        display_text += f"={'='*40}\n\n"
        display_text += f"Constant: {matrix['constant']:.4f}\n\n"
        display_text += f"Linear coefficients:\n{np.array(matrix['linear'])}\n\n"
        
        if matrix['quadratic']:
            display_text += f"Quadratic matrix:\n{np.array(matrix['quadratic'])}\n\n"
        
        if matrix['cubic']:
            display_text += f"Cubic coefficients:\n{np.array(matrix['cubic'])}\n\n"
        
        # Polynomial als Formel
        display_text += f"\nPOLYNOMIAL FORMULA:\n"
        display_text += f"f(x) = {matrix['constant']:.4f}"
        
        for i, coeff in enumerate(matrix['linear']):
            if abs(coeff) > 1e-6:
                display_text += f" + {coeff:.4f}*x{i+1}"
        
        self.matrix_display.setText(display_text)
        
        print(f"\n=== POLYNOMIAL MATRIX GENERATED ===")
        print(json.dumps(matrix, indent=2))
        print("===================================\n")
    
    def apply_to_population(self):
        """Wendet generiertes Polynom auf ausgewählte Population an"""
        if self.current_poly_matrix is None:
            print("ERROR: Generate polynomial matrix first!")
            return
        
        graph = self.get_graph_by_name(self.graph_select.get_value())
        if not graph:
            print("ERROR: No graph selected")
            return
        
        try:
            node_id = int(self.node_select.get_value().split()[1]) - 1
            pop_id = int(self.population_select.get_value().split()[1]) - 1
            node = graph.node_list[node_id]
            
            # Update flow function für diese Population
            if not hasattr(node, 'flow_functions'):
                node.flow_functions = []
            
            while len(node.flow_functions) <= pop_id:
                node.flow_functions.append(None)
            
            node.flow_functions[pop_id] = self.current_poly_matrix.copy()
            
            # Update Display
            self.current_poly_display.setText(
                f"Applied Polynomial:\n"
                f"Constant: {self.current_poly_matrix['constant']}\n"
                f"Linear: {self.current_poly_matrix['linear']}\n"
                f"Quadratic: {'Yes' if self.current_poly_matrix['quadratic'] else 'No'}\n"
                f"Cubic: {'Yes' if self.current_poly_matrix['cubic'] else 'No'}"
            )
            
            print(f"\n=== POLYNOMIAL APPLIED ===")
            print(f"Graph {graph.graph_id}, Node {node_id+1}, Population {pop_id+1}")
            print(f"Polynomial: {json.dumps(self.current_poly_matrix, indent=2)}")
            print("==========================\n")
            
        except Exception as e:
            print(f"ERROR applying polynomial: {e}")





class connectionTool(QWidget):
    def __init__(self, graph_list=None):
        super().__init__()
        
        # Graph-Liste vom MainWindow
        self.graph_list = graph_list if graph_list else []
        self.connection_list = []
        self.current_connection_id = None
        self.current_parameter_widgets = {}
        
        # Mask Types für Spatial Connections
        self.mask_types = {
            'none': {},
            'circular': {
                'radius': {'type': 'float', 'default': 5.0, 'min': 0.1, 'max': 100.0}
            },
            'box': {
                'lower_left_x': {'type': 'float', 'default': -5.0, 'min': -100.0, 'max': 100.0},
                'lower_left_y': {'type': 'float', 'default': -5.0, 'min': -100.0, 'max': 100.0},
                'upper_right_x': {'type': 'float', 'default': 5.0, 'min': -100.0, 'max': 100.0},
                'upper_right_y': {'type': 'float', 'default': 5.0, 'min': -100.0, 'max': 100.0}
            },
            'elliptical': {
                'major_axis': {'type': 'float', 'default': 10.0, 'min': 0.1, 'max': 100.0},
                'minor_axis': {'type': 'float', 'default': 5.0, 'min': 0.1, 'max': 100.0}
            }
        }
        
        # Kernel Types für Distance-dependent Probability
        self.kernel_types = {
            'constant': {
                'p': {'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 1.0}
            },
            'gaussian': {
                'p_center': {'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 1.0},
                'sigma': {'type': 'float', 'default': 2.5, 'min': 0.1, 'max': 50.0}
            },
            'exponential': {
                'a': {'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 10.0},
                'c': {'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 10.0},
                'tau': {'type': 'float', 'default': 2.5, 'min': 0.1, 'max': 50.0}
            },
            'linear': {
                'a': {'type': 'float', 'default': -0.1, 'min': -10.0, 'max': 10.0},
                'c': {'type': 'float', 'default': 1.0, 'min': -10.0, 'max': 10.0}
            }
        }
        
        # Weight Distribution Types
        self.weight_types = {
            'constant': {
                'value': {'type': 'float', 'default': 1.0, 'min': -100.0, 'max': 100.0}
            },
            'gaussian': {
                'mean': {'type': 'float', 'default': 1.0, 'min': -100.0, 'max': 100.0},
                'sigma': {'type': 'float', 'default': 0.5, 'min': 0.0, 'max': 50.0}
            },
            'exponential_distance': {
                'a': {'type': 'float', 'default': 1.0, 'min': -10.0, 'max': 10.0},
                'c': {'type': 'float', 'default': 1.0, 'min': -10.0, 'max': 10.0},
                'tau': {'type': 'float', 'default': 2.5, 'min': 0.1, 'max': 50.0}
            },
            'linear_distance': {
                'a': {'type': 'float', 'default': -0.1, 'min': -10.0, 'max': 10.0},
                'c': {'type': 'float', 'default': 1.0, 'min': -100.0, 'max': 100.0}
            }
        }
        
        # Delay Distribution Types
        self.delay_types = {
            'constant': {
                'value': {'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 100.0}
            },
            'gaussian': {
                'mean': {'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 100.0},
                'sigma': {'type': 'float', 'default': 0.1, 'min': 0.0, 'max': 10.0}
            },
            'linear_distance': {
                'a': {'type': 'float', 'default': 0.02, 'min': 0.0, 'max': 10.0},
                'c': {'type': 'float', 'default': 0.1, 'min': 0.0, 'max': 100.0}
            }
        }
        
        # Connection Rules
        self.connection_rules = {
            'all_to_all': {},
            'one_to_one': {},
            'fixed_indegree': {'indegree': {'type': 'integer', 'default': 100, 'min': 1, 'max': 10000}},
            'fixed_outdegree': {'outdegree': {'type': 'integer', 'default': 100, 'min': 1, 'max': 10000}},
            'fixed_total_number': {'N': {'type': 'integer', 'default': 1000, 'min': 1, 'max': 100000}},
            'pairwise_bernoulli': {}  # Probability wird durch Kernel definiert
        }
        
        # Synapse Models (ohne weight/delay - die kommen aus Distributionen)
        self.synapse_models = {
            'static_synapse': {},
            'stdp_synapse': {
                'alpha': {'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 10.0},
                'lambda': {'type': 'float', 'default': 0.01, 'min': 0.0, 'max': 1.0},
                'mu_plus': {'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 10.0},
                'mu_minus': {'type': 'float', 'default': 1.0, 'min': 0.0, 'max': 10.0},
                'tau_plus': {'type': 'float', 'default': 20.0, 'min': 0.1, 'max': 100.0}
            },
            'tsodyks_synapse': {
                'U': {'type': 'float', 'default': 0.5, 'min': 0.0, 'max': 1.0},
                'tau_rec': {'type': 'float', 'default': 800.0, 'min': 0.0, 'max': 10000.0},
                'tau_fac': {'type': 'float', 'default': 0.0, 'min': 0.0, 'max': 10000.0}
            }
        }
        
        # Main Layout 50:50
        self.main_layout = QHBoxLayout()
        
        # Linke Seite - Connection Liste (50%)
        self.connection_list_widget = QWidget()
        self.connection_list_layout = QVBoxLayout()
        self.connection_list_widget.setLayout(self.connection_list_layout)
        
        self.connection_scroll = QScrollArea()
        self.connection_scroll.setWidgetResizable(True)
        self.connection_scroll.setWidget(self.connection_list_widget)
        
        self.add_connection_btn = QPushButton("Add Connection")
        self.add_connection_btn.clicked.connect(self.add_connection)
        self.connection_list_layout.addWidget(self.add_connection_btn)
        self.connection_list_layout.addStretch()
        
        # Create Connections Button
        self.create_connections_btn = QPushButton("Create All Connections")
        self.create_connections_btn.clicked.connect(self.create_all_connections)
        self.create_connections_btn.setMinimumHeight(80)
        self.create_connections_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.connection_list_layout.addWidget(self.create_connections_btn)
        
        # Rechte Seite - Parameter Editor (50%)
        self.param_edit_widget = QWidget()
        self.param_edit_scroll = QScrollArea()
        self.param_edit_scroll.setWidgetResizable(True)
        self.param_edit_scroll.setWidget(self.param_edit_widget)
        
        self.param_edit_layout = QVBoxLayout()
        
        # SOURCE Section
        source_label = QLabel("SOURCE")
        source_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.param_edit_layout.addWidget(source_label)
        
        self.source_graph = DropdownField("Graph", self.get_graph_names(), default_index=0)
        self.source_node = DropdownField("Node", ["Select Graph first"], default_index=0)
        self.source_population = DropdownField("Population", ["Select Node first"], default_index=0)
        
        self.source_graph.combobox.currentTextChanged.connect(self.on_source_graph_changed)
        self.source_node.combobox.currentTextChanged.connect(self.on_source_node_changed)
        
        self.param_edit_layout.addWidget(self.source_graph)
        self.param_edit_layout.addWidget(self.source_node)
        self.param_edit_layout.addWidget(self.source_population)
        
        # TARGET Section
        target_label = QLabel("TARGET")
        target_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.param_edit_layout.addWidget(target_label)
        
        self.target_graph = DropdownField("Graph", self.get_graph_names(), default_index=0)
        self.target_node = DropdownField("Node", ["Select Graph first"], default_index=0)
        self.target_population = DropdownField("Population", ["Select Node first"], default_index=0)
        
        self.target_graph.combobox.currentTextChanged.connect(self.on_target_graph_changed)
        self.target_node.combobox.currentTextChanged.connect(self.on_target_node_changed)
        
        self.param_edit_layout.addWidget(self.target_graph)
        self.param_edit_layout.addWidget(self.target_node)
        self.param_edit_layout.addWidget(self.target_population)
        
        # SPATIAL PARAMETERS
        spatial_label = QLabel("SPATIAL PARAMETERS")
        spatial_label.setStyleSheet("font-weight: bold; font-size: 14px; color: blue;")
        self.param_edit_layout.addWidget(spatial_label)
        
        # Mask Type
        self.mask_type = DropdownField("Mask Type", list(self.mask_types.keys()), default_index=0)
        self.mask_type.combobox.currentTextChanged.connect(self.on_mask_type_changed)
        self.param_edit_layout.addWidget(self.mask_type)
        
        # Container für Mask Parameter
        self.mask_params_widget = QWidget()
        self.mask_params_layout = QVBoxLayout()
        self.mask_params_widget.setLayout(self.mask_params_layout)
        self.param_edit_layout.addWidget(self.mask_params_widget)
        
        # Kernel Type (für Probability)
        self.kernel_type = DropdownField("Probability Kernel", list(self.kernel_types.keys()), default_index=0)
        self.kernel_type.combobox.currentTextChanged.connect(self.on_kernel_type_changed)
        self.param_edit_layout.addWidget(self.kernel_type)
        
        # Container für Kernel Parameter
        self.kernel_params_widget = QWidget()
        self.kernel_params_layout = QVBoxLayout()
        self.kernel_params_widget.setLayout(self.kernel_params_layout)
        self.param_edit_layout.addWidget(self.kernel_params_widget)
        
        # Connection Rule
        conn_rule_label = QLabel("CONNECTION RULE")
        conn_rule_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.param_edit_layout.addWidget(conn_rule_label)
        
        self.conn_rule = DropdownField("Rule", list(self.connection_rules.keys()), default_index=0)
        self.conn_rule.combobox.currentTextChanged.connect(self.on_conn_rule_changed)
        self.param_edit_layout.addWidget(self.conn_rule)
        
        # Container für Connection Rule Parameter
        self.conn_params_widget = QWidget()
        self.conn_params_layout = QVBoxLayout()
        self.conn_params_widget.setLayout(self.conn_params_layout)
        self.param_edit_layout.addWidget(self.conn_params_widget)
        
        # SYNAPSE PARAMETERS
        syn_label = QLabel("SYNAPSE PARAMETERS")
        syn_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.param_edit_layout.addWidget(syn_label)
        
        # Synapse Model
        self.syn_model = DropdownField("Synapse Model", list(self.synapse_models.keys()), default_index=0)
        self.syn_model.combobox.currentTextChanged.connect(self.on_syn_model_changed)
        self.param_edit_layout.addWidget(self.syn_model)
        
        # Container für Synapse Parameter
        self.syn_params_widget = QWidget()
        self.syn_params_layout = QVBoxLayout()
        self.syn_params_widget.setLayout(self.syn_params_layout)
        self.param_edit_layout.addWidget(self.syn_params_widget)
        
        # Weight Distribution
        self.weight_type = DropdownField("Weight Distribution", list(self.weight_types.keys()), default_index=0)
        self.weight_type.combobox.currentTextChanged.connect(self.on_weight_type_changed)
        self.param_edit_layout.addWidget(self.weight_type)
        
        # Container für Weight Parameter
        self.weight_params_widget = QWidget()
        self.weight_params_layout = QVBoxLayout()
        self.weight_params_widget.setLayout(self.weight_params_layout)
        self.param_edit_layout.addWidget(self.weight_params_widget)
        
        # Delay Distribution
        self.delay_type = DropdownField("Delay Distribution", list(self.delay_types.keys()), default_index=0)
        self.delay_type.combobox.currentTextChanged.connect(self.on_delay_type_changed)
        self.param_edit_layout.addWidget(self.delay_type)
        
        # Container für Delay Parameter
        self.delay_params_widget = QWidget()
        self.delay_params_layout = QVBoxLayout()
        self.delay_params_widget.setLayout(self.delay_params_layout)
        self.param_edit_layout.addWidget(self.delay_params_widget)
        
        self.param_edit_layout.addStretch()
        self.param_edit_widget.setLayout(self.param_edit_layout)
        
        # Add to Main Layout
        self.main_layout.addWidget(self.connection_scroll, 50)
        self.main_layout.addWidget(self.param_edit_scroll, 50)
        self.setLayout(self.main_layout)
        
        # Initial parameter widgets erstellen
        self.on_mask_type_changed(self.mask_type.get_value())
        self.on_kernel_type_changed(self.kernel_type.get_value())
        self.on_conn_rule_changed(self.conn_rule.get_value())
        self.on_syn_model_changed(self.syn_model.get_value())
        self.on_weight_type_changed(self.weight_type.get_value())
        self.on_delay_type_changed(self.delay_type.get_value())
    
    def get_graph_names(self):
        """Erstellt Liste von Graph-Namen aus graph_list"""
        if not self.graph_list:
            return ["No graphs available"]
        return [f"Graph {graph.graph_id}" for graph in self.graph_list]
    
    def get_graph_by_name(self, graph_name):
        """Findet Graph-Objekt anhand des Namens"""
        try:
            graph_id = int(graph_name.split()[1])
            for graph in self.graph_list:
                if graph.graph_id == graph_id:
                    return graph
        except:
            pass
        return None
    
    def on_source_graph_changed(self, graph_name):
        """Update Node-Liste wenn Graph gewählt wird"""
        graph = self.get_graph_by_name(graph_name)
        if graph:
            node_names = [f"Node {i+1}" for i in range(len(graph.node_list))]
            self.source_node.combobox.clear()
            self.source_node.combobox.addItems(node_names if node_names else ["No nodes"])
    
    def on_source_node_changed(self, node_name):
        """Update Population-Liste wenn Node gewählt wird"""
        graph = self.get_graph_by_name(self.source_graph.get_value())
        if graph:
            try:
                node_id = int(node_name.split()[1]) - 1
                node = graph.node_list[node_id]
                pop_names = [f"Pop {i+1}" for i in range(len(node.populations))]
                self.source_population.combobox.clear()
                self.source_population.combobox.addItems(pop_names if pop_names else ["No populations"])
            except:
                pass
    
    def on_target_graph_changed(self, graph_name):
        """Update Node-Liste wenn Graph gewählt wird"""
        graph = self.get_graph_by_name(graph_name)
        if graph:
            node_names = [f"Node {i+1}" for i in range(len(graph.node_list))]
            self.target_node.combobox.clear()
            self.target_node.combobox.addItems(node_names if node_names else ["No nodes"])
    
    def on_target_node_changed(self, node_name):
        """Update Population-Liste wenn Node gewählt wird"""
        graph = self.get_graph_by_name(self.target_graph.get_value())
        if graph:
            try:
                node_id = int(node_name.split()[1]) - 1
                node = graph.node_list[node_id]
                pop_names = [f"Pop {i+1}" for i in range(len(node.populations))]
                self.target_population.combobox.clear()
                self.target_population.combobox.addItems(pop_names if pop_names else ["No populations"])
            except:
                pass
    
    def on_mask_type_changed(self, mask_type):
        """Erstellt Parameter-Widgets für Mask"""
        self.create_dynamic_params(mask_type, self.mask_types, self.mask_params_layout, "mask_")
    
    def on_kernel_type_changed(self, kernel_type):
        """Erstellt Parameter-Widgets für Kernel"""
        self.create_dynamic_params(kernel_type, self.kernel_types, self.kernel_params_layout, "kernel_")
    
    def on_conn_rule_changed(self, rule_name):
        """Erstellt Parameter-Widgets für Connection Rule"""
        self.create_dynamic_params(rule_name, self.connection_rules, self.conn_params_layout, "conn_")
    
    def on_syn_model_changed(self, model_name):
        """Erstellt Parameter-Widgets für Synapse Model"""
        self.create_dynamic_params(model_name, self.synapse_models, self.syn_params_layout, "syn_")
    
    def on_weight_type_changed(self, weight_type):
        """Erstellt Parameter-Widgets für Weight Distribution"""
        self.create_dynamic_params(weight_type, self.weight_types, self.weight_params_layout, "weight_")
    
    def on_delay_type_changed(self, delay_type):
        """Erstellt Parameter-Widgets für Delay Distribution"""
        self.create_dynamic_params(delay_type, self.delay_types, self.delay_params_layout, "delay_")
    
    def create_dynamic_params(self, selection, param_dict, layout, prefix):
        """Generische Funktion zum Erstellen dynamischer Parameter-Widgets"""
        # Lösche alte Widgets
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Entferne alte Einträge mit diesem Prefix
        keys_to_remove = [k for k in self.current_parameter_widgets.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del self.current_parameter_widgets[key]
        
        # Erstelle neue Widgets
        if selection in param_dict:
            params = param_dict[selection]
            for param_name, param_info in params.items():
                widget = self.create_param_widget(param_name, param_info)
                if widget:
                    layout.addWidget(widget)
                    self.current_parameter_widgets[f"{prefix}{param_name}"] = widget
    
    def create_param_widget(self, param_name, param_info):
        """Erstellt Widget basierend auf Parameter-Typ"""
        param_type = param_info['type']
        default = param_info['default']
        min_val = param_info.get('min', 0.0)
        max_val = param_info.get('max', 100.0)
        
        if param_type == 'float':
            return DoubleInputField(
                param_name,
                default_value=default,
                min_val=min_val,
                max_val=max_val,
                decimals=3
            )
        elif param_type == 'integer':
            return IntegerInputField(
                param_name,
                default_value=default,
                min_val=int(min_val),
                max_val=int(max_val)
            )
        return None
    
    def add_connection(self):
        """Erstellt neue Connection mit aktuellen Parametern"""
        # WICHTIG: Parameter SOFORT sammeln
        conn_params = {}
        mask_params = {}
        kernel_params = {}
        syn_params = {}
        weight_params = {}
        delay_params = {}
        
        for key, widget in list(self.current_parameter_widgets.items()):
            try:
                if key.startswith("conn_"):
                    conn_params[key[5:]] = widget.get_value()
                elif key.startswith("mask_"):
                    mask_params[key[5:]] = widget.get_value()
                elif key.startswith("kernel_"):
                    kernel_params[key[7:]] = widget.get_value()
                elif key.startswith("syn_"):
                    syn_params[key[4:]] = widget.get_value()
                elif key.startswith("weight_"):
                    weight_params[key[7:]] = widget.get_value()
                elif key.startswith("delay_"):
                    delay_params[key[6:]] = widget.get_value()
            except RuntimeError:
                pass
        
        conn_data = {
            "source": {
                "graph": self.source_graph.get_value(),
                "node": self.source_node.get_value(),
                "population": self.source_population.get_value()
            },
            "target": {
                "graph": self.target_graph.get_value(),
                "node": self.target_node.get_value(),
                "population": self.target_population.get_value()
            },
            "mask_type": self.mask_type.get_value(),
            "mask_params": mask_params,
            "kernel_type": self.kernel_type.get_value(),
            "kernel_params": kernel_params,
            "conn_rule": self.conn_rule.get_value(),
            "conn_params": conn_params,
            "syn_model": self.syn_model.get_value(),
            "syn_params": syn_params,
            "weight_type": self.weight_type.get_value(),
            "weight_params": weight_params,
            "delay_type": self.delay_type.get_value(),
            "delay_params": delay_params
        }
        
        # Erstelle Button für Connection
        conn_name = f"{conn_data['source']['graph']}:{conn_data['source']['node']}:{conn_data['source']['population']} → {conn_data['target']['graph']}:{conn_data['target']['node']}:{conn_data['target']['population']}"
        
        conn_btn = QPushButton(conn_name)
        conn_btn.setMinimumHeight(50)
        conn_id = len(self.connection_list)
        conn_btn.clicked.connect(lambda checked, cid=conn_id: self.select_connection(cid))
        
        # Insert vor den Buttons
        self.connection_list_layout.insertWidget(len(self.connection_list), conn_btn)
        
        conn_data["button"] = conn_btn
        conn_data["name"] = conn_name
        self.connection_list.append(conn_data)
        
        print(f"Connection added: {conn_name}")
        print(f"  Mask: {conn_data['mask_type']}, Kernel: {conn_data['kernel_type']}")
        print(f"  Weight: {conn_data['weight_type']}, Delay: {conn_data['delay_type']}")
    
    def select_connection(self, conn_id):
        """Lädt Connection-Parameter in die Eingabefelder"""
        self.current_connection_id = conn_id
        conn_data = self.connection_list[conn_id]
        print(f"Connection selected: {conn_data['name']}")
    
    def create_all_connections(self):
        """Gibt alle Connections als Liste zurück"""
        connections_output = []
        
        for conn_data in self.connection_list:
            conn_dict = {
                "source": conn_data["source"].copy(),
                "target": conn_data["target"].copy(),
                "mask_type": conn_data["mask_type"],
                "mask_params": conn_data["mask_params"].copy(),
                "kernel_type": conn_data["kernel_type"],
                "kernel_params": conn_data["kernel_params"].copy(),
                "conn_rule": conn_data["conn_rule"],
                "conn_params": conn_data["conn_params"].copy(),
                "syn_model": conn_data["syn_model"],
                "syn_params": conn_data["syn_params"].copy(),
                "weight_type": conn_data["weight_type"],
                "weight_params": conn_data["weight_params"].copy(),
                "delay_type": conn_data["delay_type"],
                "delay_params": conn_data["delay_params"].copy()
            }
            connections_output.append(conn_dict)
        
        print("\n=== ALL SPATIAL CONNECTIONS ===")
        print(json.dumps(connections_output, indent=2))
        print("================================\n")
        
        return connections_output

class GraphEditorTool(QWidget):
    # Signal, um dem MainWindow zu sagen: "Graph hat sich geändert, bitte neu plotten"
    graphUpdated = pyqtSignal()

    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list
        
        # Modelle laden (Logik aus graphCreationTool übernommen)
        self.functional_models = {}
        self.non_functional_models = {}
        self.all_models = {}
        self.read_json()
        
        self.current_graph = None
        self.current_node = None
        self.current_pop_idx = None
        
        self.current_parameter_widgets = {}
        
        # === LAYOUT ===
        self.main_layout = QVBoxLayout()
        
        # 1. TOP: Graph Selector
        self.top_bar = QHBoxLayout()
        self.graph_select = DropdownField("Select Graph", [], default_index=0)
        self.graph_select.combobox.currentTextChanged.connect(self.on_graph_selected)
        self.refresh_btn = QPushButton("Refresh List")
        self.refresh_btn.clicked.connect(self.refresh_graph_list)
        
        self.top_bar.addWidget(self.graph_select)
        self.top_bar.addWidget(self.refresh_btn)
        self.main_layout.addLayout(self.top_bar)
        
        # 2. BODY: Split View (Links: Listen, Rechts: Editor)
        self.body_layout = QHBoxLayout()
        
        # --- LINKS: Node & Population Listen ---
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout()
        self.left_panel.setLayout(self.left_layout)
        
        # Node Liste
        self.left_layout.addWidget(QLabel("NODES"))
        self.node_scroll = QScrollArea()
        self.node_scroll.setWidgetResizable(True)
        self.node_list_widget = QWidget()
        self.node_list_layout = QVBoxLayout()
        self.node_list_widget.setLayout(self.node_list_layout)
        self.node_scroll.setWidget(self.node_list_widget)
        self.left_layout.addWidget(self.node_scroll, 50)
        
        # Population Liste
        self.left_layout.addWidget(QLabel("POPULATIONS"))
        self.pop_scroll = QScrollArea()
        self.pop_scroll.setWidgetResizable(True)
        self.pop_list_widget = QWidget()
        self.pop_list_layout = QVBoxLayout()
        self.pop_list_widget.setLayout(self.pop_list_layout)
        self.pop_scroll.setWidget(self.pop_list_widget)
        self.left_layout.addWidget(self.pop_scroll, 50)
        
        self.body_layout.addWidget(self.left_panel, 30)
        
        # --- RECHTS: Stacked Editor (Node / Population) ---
        self.right_stack = QStackedWidget()
        
        # PAGE 0: Node Editor
        self.node_edit_page = QWidget()
        self.node_edit_layout = QVBoxLayout()
        self.node_edit_page.setLayout(self.node_edit_layout)
        
        self.node_edit_layout.addWidget(QLabel("EDIT NODE PROPERTIES"))
        
        self.node_neuron_count = IntegerInputField("Neuron Count (Target)", min_val=1, max_val=50000)
        self.node_pos_x = DoubleInputField("Pos X", min_val=-100, max_val=100)
        self.node_pos_y = DoubleInputField("Pos Y", min_val=-100, max_val=100)
        self.node_pos_z = DoubleInputField("Pos Z", min_val=-100, max_val=100)
        
        self.node_edit_layout.addWidget(self.node_neuron_count)
        self.node_edit_layout.addWidget(self.node_pos_x)
        self.node_edit_layout.addWidget(self.node_pos_y)
        self.node_edit_layout.addWidget(self.node_pos_z)
        
        # Rebuild Button
        self.rebuild_node_btn = QPushButton("Apply Changes & Rebuild Node")
        self.rebuild_node_btn.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 10px;")
        self.rebuild_node_btn.clicked.connect(self.apply_node_changes)
        self.node_edit_layout.addWidget(self.rebuild_node_btn)
        self.node_edit_layout.addStretch()
        
        self.right_stack.addWidget(self.node_edit_page)
        
        # PAGE 1: Population Editor
        self.pop_edit_page = QWidget()
        self.pop_edit_scroll = QScrollArea()
        self.pop_edit_scroll.setWidgetResizable(True)
        self.pop_content = QWidget()
        self.pop_layout = QVBoxLayout()
        self.pop_content.setLayout(self.pop_layout)
        self.pop_edit_scroll.setWidget(self.pop_content)
        
        self.pop_main_layout = QVBoxLayout()
        self.pop_main_layout.addWidget(QLabel("EDIT POPULATION PARAMETERS"))
        self.pop_main_layout.addWidget(self.pop_edit_scroll)
        
        self.save_pop_btn = QPushButton("Save Parameters")
        self.save_pop_btn.clicked.connect(self.save_population_changes)
        self.pop_main_layout.addWidget(self.save_pop_btn)
        
        self.pop_edit_page.setLayout(self.pop_main_layout)
        self.right_stack.addWidget(self.pop_edit_page)
        
        self.body_layout.addWidget(self.right_stack, 70)
        self.main_layout.addLayout(self.body_layout)
        
        self.setLayout(self.main_layout)
        
        # Init
        self.refresh_graph_list()

    def read_json(self):
        """Lädt Modelle (identisch zu graphCreationTool)"""
        try:
            with open("functional_models.json") as f:
                self.functional_models = json.load(f)
        except:
            self.functional_models = {}
        try:
            with open("non_functional_models.json") as f:
                self.non_functional_models = json.load(f)
        except:
            self.non_functional_models = {}
        self.all_models = {**self.functional_models, **self.non_functional_models}

    def refresh_graph_list(self):
        self.graph_select.combobox.blockSignals(True)
        self.graph_select.combobox.clear()
        if not self.graph_list:
            self.graph_select.combobox.addItem("No Graphs available")
        else:
            for g in self.graph_list:
                self.graph_select.combobox.addItem(f"Graph {g.graph_id} ({g.nodes} Nodes)")
        self.graph_select.combobox.blockSignals(False)
        
        # Automatisch den ersten wählen
        if self.graph_list:
            self.on_graph_selected(self.graph_select.combobox.currentText())

    def on_graph_selected(self, text):
        if not self.graph_list: return
        try:
            # Extrahiere ID aus String "Graph 0 (...)"
            g_id = int(text.split()[1])
            for g in self.graph_list:
                if g.graph_id == g_id:
                    self.current_graph = g
                    self.load_nodes_list()
                    break
        except:
            pass

    def load_nodes_list(self):
        # Clear Node List
        while self.node_list_layout.count():
            child = self.node_list_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        
        # Clear Pop List
        self.clear_pop_list()
        
        if not self.current_graph: return

        for node in self.current_graph.node_list:
            btn = QPushButton(f"ID {node.id}: {node.name}")
            btn.clicked.connect(lambda checked, n=node: self.on_node_selected(n))
            self.node_list_layout.addWidget(btn)
        
        self.node_list_layout.addStretch()

    def clear_pop_list(self):
        while self.pop_list_layout.count():
            child = self.pop_list_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

    def on_node_selected(self, node):
        self.current_node = node
        self.current_pop_idx = None
        
        # 1. Fülle Node Editor Werte
        # Achtung: Node parameters können unterschiedlich strukturiert sein
        # Wir schauen in node.parameters dictionary
        p = node.parameters
        
        # Hole Werte sicher, falls Keys fehlen
        neuron_count = 0
        # Wir versuchen die Neuronenzahl zu schätzen oder aus Params zu holen
        if "neuron_count" in p: # Falls wir das im CreationTool gesetzt haben
             neuron_count = p["neuron_count"]
        elif hasattr(node, 'population') and node.population:
             neuron_count = len(node.population)
        
        self.node_neuron_count.spinbox.setValue(neuron_count)
        
        pos = node.center_of_mass
        self.node_pos_x.spinbox.setValue(pos[0])
        self.node_pos_y.spinbox.setValue(pos[1])
        self.node_pos_z.spinbox.setValue(pos[2])
        
        self.right_stack.setCurrentIndex(0) # Zeige Node Editor
        
        # 2. Fülle Population Liste
        self.clear_pop_list()
        
        # node.types enthält Indizes, node.neuron_models die Namen
        if "populations" in p:
            # Falls wir die Struktur aus dem CreationTool noch im dict haben
            for i, pop_data in enumerate(p["populations"]):
                name = pop_data.get('name', f"Pop {i}")
                btn = QPushButton(name)
                btn.clicked.connect(lambda checked, idx=i: self.on_pop_selected(idx))
                self.pop_list_layout.addWidget(btn)
        else:
            # Fallback, falls Node nicht über GUI erstellt wurde
            for i, model in enumerate(node.neuron_models):
                btn = QPushButton(f"Pop {i}: {model}")
                btn.clicked.connect(lambda checked, idx=i: self.on_pop_selected(idx))
                self.pop_list_layout.addWidget(btn)
                
        self.pop_list_layout.addStretch()

    def on_pop_selected(self, idx):
        self.current_pop_idx = idx
        
        # Hole Daten aus dem parameters dict des Nodes
        # Strukturannahme: node.parameters['populations'][idx]['params']
        if "populations" in self.current_node.parameters:
            pop_data = self.current_node.parameters["populations"][idx]
            model = pop_data["model"]
            params = pop_data["params"]
            self.build_param_editor(model, params)
            self.right_stack.setCurrentIndex(1)
        else:
            print("Warning: Cannot edit populations for nodes created outside GUI (missing param struct)")

    def build_param_editor(self, model_name, current_values):
        # Layout leeren
        while self.pop_layout.count():
            child = self.pop_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        
        self.current_parameter_widgets = {}
        
        # Hole Parameter-Definitionen aus JSON
        if model_name not in self.all_models:
            self.pop_layout.addWidget(QLabel(f"No parameter definition for {model_name}"))
            return

        model_def = self.all_models[model_name]
        
        for param_key, param_info in model_def.items():
            val = current_values.get(param_key, param_info['default'])
            p_type = param_info['type']
            
            widget = None
            if p_type == 'float':
                widget = DoubleInputField(param_key, default_value=val, min_val=-10000, max_val=10000)
            elif p_type == 'integer':
                widget = IntegerInputField(param_key, default_value=val, min_val=0, max_val=100000)
            elif p_type == 'boolean':
                widget = CheckboxField(param_key, default_checked=val) # Annahme CheckboxField existiert
            
            if widget:
                self.pop_layout.addWidget(widget)
                self.current_parameter_widgets[param_key] = widget
        
        self.pop_layout.addStretch()

    def save_population_changes(self):
        if not self.current_node or self.current_pop_idx is None: return
        
        # Werte aus Widgets lesen
        new_params = {}
        for key, widget in self.current_parameter_widgets.items():
            new_params[key] = widget.get_value()
        
        # In Node speichern
        self.current_node.parameters["populations"][self.current_pop_idx]["params"] = new_params
        print(f"Updated params for Node {self.current_node.id}, Pop {self.current_pop_idx}")
        
        # Hinweis: NEST Parameter müssen oft via SetStatus gesetzt werden.
        # Wenn der Node schon "gebaut" ist (population list existiert),
        # müssten wir hier eigentlich nest.SetStatus aufrufen.
        # Fürs erste aktualisieren wir nur das Config-Dict für den nächsten Rebuild.

    def apply_node_changes(self):
        """Aktualisiert Position/Größe und baut den Node neu"""
        if not self.current_node: return
        
        # 1. Update Parameters
        new_count = self.node_neuron_count.get_value()
        new_pos = np.array([
            self.node_pos_x.get_value(),
            self.node_pos_y.get_value(),
            self.node_pos_z.get_value()
        ])
        
        self.current_node.parameters["neuron_count"] = new_count
        self.current_node.parameters["m"] = new_pos
        
        # Update Center of Mass im Objekt selbst
        self.current_node.center_of_mass = new_pos
        
        # 2. Grid neu berechnen (Logik aus MainWindow.process_created_graph)
        sparsity = self.current_node.parameters.get("sparsity_factor", 0.8)
        needed_volume = new_count / (1.0 - sparsity)
        side_length = int(np.ceil(needed_volume ** (1/3))) + 2
        
        self.current_node.parameters["grid_size"] = [side_length, side_length, side_length]
        
        print(f"Rebuilding Node {self.current_node.id} at {new_pos} with Grid {side_length}^3...")
        
        # 3. Rebuild Action
        try:
            # build() führt WFC aus und setzt self.positions neu
            self.current_node.build() 
            # populate_node() erstellt NEST Nodes (Achtung: Alte NEST Nodes werden nicht automatisch gelöscht!)
            # TODO: Alte NEST IDs sauber aufräumen falls nötig
            self.current_node.populate_node()
            
            print("Rebuild successful.")
            
            # 4. Signal an Mainwindow zum Neuzeichnen
            self.graphUpdated.emit()
            
        except Exception as e:
            print(f"Error rebuilding node: {e}")


class graphCreationTool(QWidget):
    graphCreated = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self.functional_models = {}
        self.non_functional_models = {}
        self.all_models = {}
        self.read_json()
        print(f"Loaded {len(self.all_models)} models")
        
        self.current_node = None
        self.current_population_id = None
        self.current_parameter_widgets = {}
        self.current_percentage_widgets = {}
        
        # Node Liste (15%) mit Create Graph Button unten
        self.node_list = []
        self.node_list_widget = QWidget()
        self.node_list_layout = QVBoxLayout()
        self.node_list_widget.setLayout(self.node_list_layout)
        
        self.node_scroll = QScrollArea()
        self.node_scroll.setWidgetResizable(True)
        self.node_scroll.setWidget(self.node_list_widget)
        
        self.node_list_add_btn = QPushButton("Add Node")
        self.node_list_add_btn.clicked.connect(self.add_node)
        self.node_list_layout.addWidget(self.node_list_add_btn)
        self.node_list_layout.addStretch()
        
        # CREATE GRAPH BUTTON - FIXIERT UNTEN
        self.create_graph_btn = QPushButton("Create Graph")
        self.create_graph_btn.clicked.connect(self.create_graph)
        self.create_graph_btn.setMinimumHeight(80)
        self.create_graph_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.node_list_layout.addWidget(self.create_graph_btn)
        
        # Population Liste (25%)
        self.population_list_widget = QWidget()
        self.population_layout = QVBoxLayout()
        self.population_list_widget.setLayout(self.population_layout)
        
        self.pop_scroll = QScrollArea()
        self.pop_scroll.setWidgetResizable(True)
        self.pop_scroll.setWidget(self.population_list_widget)
        
        self.add_pop_btn = QPushButton("Add Population")
        self.add_pop_btn.clicked.connect(self.add_population)
        self.add_pop_btn.setEnabled(False)
        self.population_layout.addWidget(self.add_pop_btn)
        self.population_layout.addStretch()
        
        # Rechte Seite - STACKED für Node/Population Menüs (60%)
        self.right_side_stack = QStackedWidget()
        
        # Node Edit Menu
        self.node_edit_widget = QWidget()
        self.node_edit_scroll = QScrollArea()
        self.node_edit_scroll.setWidgetResizable(True)
        self.node_edit_scroll.setWidget(self.node_edit_widget)
        
        self.node_edit_layout = QVBoxLayout()
        
        # Neuron Count
        self.node_neuron_count = IntegerInputField("neuron_count", default_value=100, min_val=1, max_val=10000)
        self.node_neuron_count.spinbox.valueChanged.connect(self.save_current_node_data)
        self.node_edit_layout.addWidget(self.node_neuron_count)
        
        # Center of Mass - X, Y, Z
        self.node_com_x = DoubleInputField("center_x", default_value=0.0, min_val=-100.0, max_val=100.0, decimals=3)
        self.node_com_y = DoubleInputField("center_y", default_value=0.0, min_val=-100.0, max_val=100.0, decimals=3)
        self.node_com_z = DoubleInputField("center_z", default_value=0.0, min_val=-100.0, max_val=100.0, decimals=3)
        
        self.node_com_x.spinbox.valueChanged.connect(self.save_current_node_data)
        self.node_com_y.spinbox.valueChanged.connect(self.save_current_node_data)
        self.node_com_z.spinbox.valueChanged.connect(self.save_current_node_data)
        
        self.node_edit_layout.addWidget(self.node_com_x)
        self.node_edit_layout.addWidget(self.node_com_y)
        self.node_edit_layout.addWidget(self.node_com_z)
        
        # Container für dynamische % Felder
        self.percentage_widget = QWidget()
        self.percentage_layout = QVBoxLayout()
        self.percentage_widget.setLayout(self.percentage_layout)
        self.node_edit_layout.addWidget(self.percentage_widget)
        
        self.node_edit_layout.addStretch()
        self.node_edit_widget.setLayout(self.node_edit_layout)
        
        # Population Edit Menu - DYNAMISCH
        self.pop_edit_widget = QWidget()
        self.pop_edit_scroll = QScrollArea()
        self.pop_edit_scroll.setWidgetResizable(True)
        self.pop_edit_scroll.setWidget(self.pop_edit_widget)
        
        self.pop_edit_layout = QVBoxLayout()
        
        # Model Selector (immer oben)
        self.pop_model = DropdownField("neuron_model", list(self.all_models.keys()), default_index=0)
        self.pop_model.combobox.currentTextChanged.connect(self.on_model_changed)
        self.pop_edit_layout.addWidget(self.pop_model)
        
        # Container für dynamische Parameter
        self.dynamic_params_widget = QWidget()
        self.dynamic_params_layout = QVBoxLayout()
        self.dynamic_params_widget.setLayout(self.dynamic_params_layout)
        self.pop_edit_layout.addWidget(self.dynamic_params_widget)
        
        self.pop_edit_layout.addStretch()
        self.pop_edit_widget.setLayout(self.pop_edit_layout)
        
        # Add beide zu Stack
        self.right_side_stack.addWidget(self.node_edit_scroll)  # Index 0
        self.right_side_stack.addWidget(self.pop_edit_scroll)   # Index 1
        
        # Main Layout 15:25:60
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.node_scroll, 15)
        self.main_layout.addWidget(self.pop_scroll, 25)
        self.main_layout.addWidget(self.right_side_stack, 60)
        self.setLayout(self.main_layout)
    
    def read_json(self):
        """Lädt beide JSON-Dateien"""
        try:
            with open("functional_models.json") as f:
                self.functional_models = json.load(f)
        except:
            print("functional_models.json not found")
            self.functional_models = {}
        
        try:
            with open("non_functional_models.json") as f:
                self.non_functional_models = json.load(f)
        except:
            print("non_functional_models.json not found")
            self.non_functional_models = {}
        
        self.all_models = {**self.functional_models, **self.non_functional_models}
    
    def get_default_params(self, model_name):
        """Erstellt dict mit default-Werten für ein Modell"""
        if model_name not in self.all_models:
            return {}
        
        model_params = self.all_models[model_name]
        defaults = {}
        
        for param_name, param_info in model_params.items():
            defaults[param_name] = param_info['default']
        
        return defaults
    
    def create_percentage_fields(self):
        """Erstellt % Felder für alle Populationen des aktuellen Nodes"""
        while self.percentage_layout.count():
            item = self.percentage_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.current_percentage_widgets.clear()
        
        if not self.current_node:
            return
        
        for i, pop_data in enumerate(self.current_node['populations']):
            percentage = pop_data.get('percentage', 0.0)
            widget = PercentageInputField(f"{pop_data['name']}", default_value=percentage)
            widget.spinbox.valueChanged.connect(self.save_current_node_data)
            self.percentage_layout.addWidget(widget)
            self.current_percentage_widgets[i] = widget
        
        if self.current_node['populations']:
            total = self.get_total_percentage()
            label = QLabel(f"Total: {total:.2f}%")
            if abs(total - 100.0) < 0.01:
                label.setStyleSheet("color: green; font-weight: bold;")
            else:
                label.setStyleSheet("color: orange; font-weight: bold;")
            self.percentage_layout.addWidget(label)
    
    def get_total_percentage(self):
        """Berechnet die Summe aller % Werte"""
        total = 0.0
        for widget in self.current_percentage_widgets.values():
            total += widget.get_value()
        return total
    
    def create_parameter_widgets(self, model_name):
        """Erstellt dynamisch Widgets basierend auf Modell-Parametern"""
        while self.dynamic_params_layout.count():
            item = self.dynamic_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.current_parameter_widgets.clear()
        
        if model_name not in self.all_models:
            return
        
        model_params = self.all_models[model_name]
        
        for param_name, param_info in model_params.items():
            param_type = param_info['type']
            default = param_info['default']
            min_val = param_info.get('min', 0.0)
            max_val = param_info.get('max', 100.0)
            
            widget = None
            
            if param_type == 'float':
                widget = DoubleInputField(
                    param_name,
                    default_value=default,
                    min_val=min_val if min_val is not None else -1000.0,
                    max_val=max_val if max_val is not None else 1000.0,
                    decimals=3
                )
                widget.spinbox.valueChanged.connect(self.save_current_population_data)
                
            elif param_type == 'integer':
                widget = IntegerInputField(
                    param_name,
                    default_value=default,
                    min_val=int(min_val) if min_val is not None else 0,
                    max_val=int(max_val) if max_val is not None else 10000
                )
                widget.spinbox.valueChanged.connect(self.save_current_population_data)
                
            elif param_type == 'boolean':
                widget = CheckboxField(param_name, default_checked=default)
                widget.checkbox.stateChanged.connect(self.save_current_population_data)
            
            if widget:
                self.dynamic_params_layout.addWidget(widget)
                self.current_parameter_widgets[param_name] = widget
        
        print(f"Created {len(self.current_parameter_widgets)} parameter widgets for {model_name}")
    
    def on_model_changed(self, model_name):
        """Wird aufgerufen wenn Modell gewechselt wird"""
        print(f"Model changed to: {model_name}")
        self.create_parameter_widgets(model_name)
        
        if self.current_node and self.current_population_id is not None:
            pop_data = self.current_node['populations'][self.current_population_id]
            pop_data['model'] = model_name
            pop_data['params'] = self.get_default_params(model_name)
            pop_data['name'] = f"Pop {self.current_population_id + 1}: {model_name}"
            
            self.update_population_view_safe()
            
            print(f"Reset params to defaults for {model_name}")
    
    def add_node(self):
        neuron_count, ok = QInputDialog.getInt(
            self, 
            "Node erstellen", 
            "Wie viele Neuronen?",
            value=100,
            min=1,
            max=10000
        )
        
        if ok:
            node_btn = QPushButton(f"Node {len(self.node_list) + 1} ({neuron_count} neurons)")
            node_btn.setMinimumHeight(50)
            node_id = len(self.node_list)
            node_btn.clicked.connect(lambda checked, nid=node_id: self.select_node(nid))
            
            # Insert BEFORE Create Graph button (count - 2)
            self.node_list_layout.insertWidget(self.node_list_layout.count() - 2, node_btn)
            
            node_data = {
                "button": node_btn,
                "neuron_count": neuron_count,
                "center_of_mass": [0.0, 0.0, 0.0],  # [x, y, z]
                "populations": []
            }
            self.node_list.append(node_data)
            self.validate_all_buttons_safe()
    
    def add_population(self):
        if self.current_node is None:
            return
        
        first_model = list(self.all_models.keys())[0] if self.all_models else "default"
        pop_number = len(self.current_node['populations']) + 1
        
        pop_data = {
            "name": f"Pop {pop_number}: {first_model}",
            "model": first_model,
            "params": self.get_default_params(first_model),
            "percentage": 0.0
        }
        self.current_node['populations'].append(pop_data)
        self.update_population_view_safe()
        self.create_percentage_fields()
    
    def select_node(self, node_id):
        self.current_node = self.node_list[node_id]
        self.current_population_id = None
        print(f"Node selected: {self.current_node['button'].text()}")
        
        self.right_side_stack.setCurrentIndex(0)
        self.load_node_data(self.current_node)
        
        self.update_population_view_safe()
        self.add_pop_btn.setEnabled(True)
    
    def load_node_data(self, node_data):
        """Lädt Node-Parameter in die Eingabefelder"""
        self.node_neuron_count.spinbox.valueChanged.disconnect()
        self.node_com_x.spinbox.valueChanged.disconnect()
        self.node_com_y.spinbox.valueChanged.disconnect()
        self.node_com_z.spinbox.valueChanged.disconnect()
        
        self.node_neuron_count.spinbox.setValue(node_data['neuron_count'])
        
        com = node_data.get('center_of_mass', [0.0, 0.0, 0.0])
        self.node_com_x.spinbox.setValue(com[0])
        self.node_com_y.spinbox.setValue(com[1])
        self.node_com_z.spinbox.setValue(com[2])
        
        self.node_neuron_count.spinbox.valueChanged.connect(self.save_current_node_data)
        self.node_com_x.spinbox.valueChanged.connect(self.save_current_node_data)
        self.node_com_y.spinbox.valueChanged.connect(self.save_current_node_data)
        self.node_com_z.spinbox.valueChanged.connect(self.save_current_node_data)
        
        self.create_percentage_fields()
    
    def save_current_node_data(self):
        """Speichert aktuelle Node-Werte"""
        if self.current_node:
            self.current_node['neuron_count'] = self.node_neuron_count.get_value()
            self.current_node['center_of_mass'] = [
                self.node_com_x.get_value(),
                self.node_com_y.get_value(),
                self.node_com_z.get_value()
            ]
            
            for i, widget in self.current_percentage_widgets.items():
                self.current_node['populations'][i]['percentage'] = widget.get_value()
            
            node_id = self.node_list.index(self.current_node)
            self.current_node['button'].setText(f"Node {node_id + 1} ({self.current_node['neuron_count']} neurons)")
            
            self.create_percentage_fields()
            self.validate_all_buttons_safe()
            
            print(f"Node saved: neurons={self.current_node['neuron_count']}, COM={self.current_node['center_of_mass']}, Total %: {self.get_total_percentage()}")
    
    def update_population_view_safe(self):
        """SICHERE Version - löscht alte Buttons ohne Crash"""
        while self.population_layout.count() > 2:
            item = self.population_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if self.current_node:
            for i, pop_data in enumerate(self.current_node['populations']):
                pop_btn = QPushButton(pop_data['name'])
                pop_btn.setMinimumHeight(50)
                pop_btn.clicked.connect(lambda checked, pid=i: self.select_population(pid))
                self.population_layout.insertWidget(i, pop_btn)
        
        self.validate_all_buttons_safe()
    
    def select_population(self, pop_id):
        if self.current_node:
            self.current_population_id = pop_id
            pop_data = self.current_node['populations'][pop_id]
            print(f"Population selected: {pop_data['name']}, Model: {pop_data['model']}")
            
            self.right_side_stack.setCurrentIndex(1)
            self.load_population_data(pop_data)
    
    def load_population_data(self, pop_data):
        """Lädt Population-Parameter und erstellt entsprechende Widgets"""
        model_name = pop_data['model']
        params = pop_data['params']
        
        self.pop_model.combobox.currentTextChanged.disconnect()
        self.pop_model.combobox.setCurrentText(model_name)
        self.pop_model.combobox.currentTextChanged.connect(self.on_model_changed)
        
        self.create_parameter_widgets(model_name)
        
        for param_name, widget in self.current_parameter_widgets.items():
            if param_name in params:
                value = params[param_name]
                
                if isinstance(widget, (IntegerInputField, DoubleInputField)):
                    widget.spinbox.valueChanged.disconnect()
                    widget.spinbox.setValue(value)
                    widget.spinbox.valueChanged.connect(self.save_current_population_data)
                    
                elif isinstance(widget, CheckboxField):
                    widget.checkbox.stateChanged.disconnect()
                    widget.checkbox.setChecked(value)
                    widget.checkbox.stateChanged.connect(self.save_current_population_data)
    
    def save_current_population_data(self):
        """Speichert aktuelle Parameterwerte in die Population"""
        if self.current_node and self.current_population_id is not None:
            pop_data = self.current_node['populations'][self.current_population_id]
            
            params = {}
            for param_name, widget in self.current_parameter_widgets.items():
                params[param_name] = widget.get_value()
            
            pop_data['params'] = params
            pop_data['model'] = self.pop_model.get_value()
            
            self.validate_all_buttons_safe()
            
            print(f"Saved {pop_data['name']}: Model={pop_data['model']}, Params={len(params)} parameters")
    
    def validate_node(self, node_data):
        """Prüft ob ein Node valide ist"""
        if not node_data['populations']:
            return False
        
        total = sum(pop.get('percentage', 0.0) for pop in node_data['populations'])
        if abs(total - 100.0) > 0.01:
            return False
        
        return True
    
    def validate_population(self, pop_data):
        """Prüft ob eine Population valide ist"""
        if not pop_data.get('params'):
            return False
        
        model_name = pop_data['model']
        if model_name not in self.all_models:
            return False
        
        expected_params = set(self.all_models[model_name].keys())
        actual_params = set(pop_data['params'].keys())
        
        return expected_params == actual_params
    
    def validate_all_buttons_safe(self):
        """SICHERE Validierung - kein Crash bei gelöschten Buttons"""
        for node_data in self.node_list:
            try:
                is_valid = self.validate_node(node_data)
                button = node_data.get('button')
                
                if button and not button.isHidden():
                    if is_valid:
                        button.setStyleSheet("background-color: rgba(0, 255, 0, 50);")
                    else:
                        button.setStyleSheet("background-color: rgba(255, 165, 0, 50);")
            except RuntimeError:
                pass
        
        if self.current_node:
            for i in range(self.population_layout.count() - 2):
                try:
                    widget = self.population_layout.itemAt(i).widget()
                    if widget and i < len(self.current_node['populations']):
                        pop_data = self.current_node['populations'][i]
                        pop_valid = self.validate_population(pop_data)
                        
                        if pop_valid:
                            widget.setStyleSheet("background-color: rgba(0, 255, 0, 50);")
                        else:
                            widget.setStyleSheet("background-color: rgba(255, 165, 0, 50);")
                except (RuntimeError, AttributeError):
                    pass
    
    def create_graph(self):
        """Erstellt Graph-Dict aus allen Nodes und Populationen"""
        graph_data = {"nodes": []}
        all_valid = True
        
        for node_data in self.node_list:
            if not self.validate_node(node_data):
                all_valid = False
                print(f"Node {node_data['button'].text()} ist nicht valide!")
                continue
            
            node_dict = {
                "neuron_count": node_data['neuron_count'],
                "center_of_mass": node_data['center_of_mass'],
                "populations": []
            }
            
            for pop_data in node_data['populations']:
                if not self.validate_population(pop_data):
                    all_valid = False
                    print(f"Population {pop_data['name']} ist nicht valide!")
                    continue
                
                pop_dict = {
                    "name": pop_data['name'],
                    "model": pop_data['model'],
                    "percentage": pop_data.get('percentage', 0.0),
                    "params": pop_data['params'].copy()
                }
                node_dict['populations'].append(pop_dict)
            
            graph_data['nodes'].append(node_dict)
        
        if all_valid:
            print("\nGraph-Structure:")
            print(json.dumps(graph_data, indent=2))
            
            # 2. HIER DAS SIGNAL EMITTIEREN (statt nur print)
            self.graphCreated.emit(graph_data) 

            # RESET INTERFACE nach erfolgreicher Erstellung
            self.reset_interface()
        else:
            print("\n!!!invalid node parametrization...!!!\n")
        
        return graph_data # Hier Aufruf der Ersteller funktion. PERFEKT.

    def reset_interface(self):
        """Setzt das gesamte Interface auf Startzustand zurück"""
        print("Resetting interface to initial state...")
        
        # 1. Lösche alle Node-Buttons
        for node_data in self.node_list:
            try:
                button = node_data.get('button')
                if button:
                    button.deleteLater()
            except RuntimeError:
                pass
        
        # 2. Leere Node-Liste
        self.node_list.clear()
        
        # 3. Lösche alle Population-Buttons
        while self.population_layout.count() > 2:  # Behalte Add-Button und Stretch
            item = self.population_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 4. Lösche alle Parameter-Widgets
        while self.dynamic_params_layout.count():
            item = self.dynamic_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 5. Lösche alle Percentage-Widgets
        while self.percentage_layout.count():
            item = self.percentage_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 6. Reset aktuelle Auswahl
        self.current_node = None
        self.current_population_id = None
        self.current_parameter_widgets.clear()
        self.current_percentage_widgets.clear()
        
        # 7. Disable Add Population Button
        self.add_pop_btn.setEnabled(False)
        
        # 8. Reset Node Edit Fields auf Defaults
        self.node_neuron_count.spinbox.valueChanged.disconnect()
        self.node_com_x.spinbox.valueChanged.disconnect()
        self.node_com_y.spinbox.valueChanged.disconnect()
        self.node_com_z.spinbox.valueChanged.disconnect()
        
        self.node_neuron_count.spinbox.setValue(100)
        self.node_com_x.spinbox.setValue(0.0)
        self.node_com_y.spinbox.setValue(0.0)
        self.node_com_z.spinbox.setValue(0.0)
        
        self.node_neuron_count.spinbox.valueChanged.connect(self.save_current_node_data)
        self.node_com_x.spinbox.valueChanged.connect(self.save_current_node_data)
        self.node_com_y.spinbox.valueChanged.connect(self.save_current_node_data)
        self.node_com_z.spinbox.valueChanged.connect(self.save_current_node_data)
        
        # 9. Zeige Node Edit Screen (Index 0)
        self.right_side_stack.setCurrentIndex(0)
        
        print("Interface reset complete!")

































class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuroticks")
        self.resize(1920, 1080)






        # meine Widgets



        self.btn = self.create_bottom_button_menu()
        self.graph_list = [createGraph(max_nodes=2,graph_id=0)]# bestimme wie viele nodes initial erstellt werden
        self.graph_vis = self.create_graph_visualization()
        self.node_vis = self.create_neuron_visualization()

        self.vis_widget_list = []
        
        self.vis_widget_list.append(self.node_vis)
        self.vis_widget_list.append(self.graph_vis)

        self.vis_btn_grp = self.create_vis_button_menu()


        self.main_layout = QVBoxLayout()
        self.button_layout = QHBoxLayout() # Für später. Buttons in Reihe
        





        self.stacked = self.visualization(self.vis_widget_list)
        self.stacked.setCurrentIndex(0) 



        # aktive variablen, zb selection
        self.selected_graph = self.graph_list[0]

        ####### POLYNOM EDITOR
        self.polynom_editor = polynomEditorTool(graph_list=self.graph_list)
        self.graph_creator = graphCreationTool()
        self.graph_creator.graphCreated.connect(self.process_created_graph)
        self.graph_editor = GraphEditorTool(self.graph_list)
        # Wenn im Editor was geändert wird -> Neu Plotten
        self.graph_editor.graphUpdated.connect(self.refresh_visualization)

        #Erstellungsmenü für connections, nodes, graphen usw
        self.creation_functionalities = [self.graph_creator,##################### HIER
                                         self.polynom_editor,
                                         self.graph_editor,
                                         Color("blue"),
                                         connectionTool(),
                                         Color("yellow"),
                                         Color("orange"),
                                         Color("red"),
                                         Color("pink"),
                                         Color("grey")]



        self.edit_menu = self.creation_menu(self.creation_functionalities)











        # Top Layout (60% der Höhe)
        self.layout_top = QHBoxLayout()
        self.layout_top_left = QHBoxLayout()
        self.scene_menu = self.vis_btn_grp # scene buttons
        #self.create_bottom_button_menu()

        self.plot_scene = self.stacked # scene
        self.layout_top_left.addWidget(self.scene_menu,1)
        self.layout_top_left.addWidget(self.plot_scene,9)
        self.layout_top.addLayout(self.layout_top_left, 7)  # 70% der Breite - PYVISTA
        self.layout_top.addWidget(Color("orange"), 3)  # 30% der Breite - NODE LIST
        
        # Bottom Layout (40% der Höhe)
        self.layout_bottom = QHBoxLayout()
        
        # Bottom Left (70% der Breite)
        self.layout_bottom_left = QVBoxLayout()
        self.layout_bottom_left.addWidget(self.edit_menu, 9)    # 90% der Höhe - Population Creation
        self.layout_bottom_left.addWidget(Color("green"), 1)   # 10% der Höhe - Ladebalken
        
        

        # Bottom Right (30% der Breite)
        self.layout_bottom_right = QHBoxLayout()
        self.layout_bottom_right.addWidget(self.btn, 7)    # 70% der Breite - Buttons
        self.layout_bottom_right.addWidget(Color("green"), 3)  # 30% der Breite - graph plot/static pictures
        
        self.layout_bottom.addLayout(self.layout_bottom_left, 7)
        self.layout_bottom.addLayout(self.layout_bottom_right, 3)
        
        self.main_layout.addLayout(self.layout_top, 3)      # 60% der Höhe
        self.main_layout.addLayout(self.layout_bottom, 2)   # 40% der Höhe
        
        self.widget = QWidget()
        self.widget.setLayout(self.main_layout)
        self.setCentralWidget(self.widget)
        self.plot_points_of_graphs()        # Endgültige visualisierung
        self.plot_graphs()
    

    def create_graph_visualization(self):
        """Erstellt die Graph-Skeleton-Visualisierung"""
        plotter = QtInteractor(self)
        plotter.set_background('black')
        return plotter
    
    def refresh_visualization(self):
        self.plot_points_of_graphs()
        self.plot_graphs()
        self.stacked.setCurrentIndex(0)

    def create_neuron_visualization(self):
        """Erstellt die Neuron-Points-Visualisierung"""
        plotter = QtInteractor(self)
        plotter.set_background('black')
        return plotter


    def create_vis_button_menu(self):
        # Container Widget
        btn_widget = QWidget()
        btn_layout = QVBoxLayout()  # oder QHBoxLayout für horizontal
        
        # Buttons erstellen
        btn1 = QPushButton("Neurons")
        btn2 = QPushButton("Graph")
        btn3 = QPushButton("Simulation")
        btn4 = QPushButton("other stuff")
    
        btn1.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        btn2.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        #btn3.clicked.connect(lambda: self.stacked.setCurrentIndex(2))
        #btn4.clicked.connect(lambda: self.stacked.setCurrentIndex(3))
        # Zum Layout hinzufügen
        btn_layout.addWidget(btn1)
        btn_layout.addWidget(btn2)
        btn_layout.addWidget(btn3)
        btn_layout.addWidget(btn4)
        
        btn_widget.setLayout(btn_layout)
        return btn_widget


    def create_bottom_button_menu(self):
        # Container Widget
        btn_widget = QWidget()
        btn_layout = QVBoxLayout()  # oder QHBoxLayout für horizontal
        
        # Buttons erstellen
        btn1 = QPushButton("Create New Graph")
        btn2 = QPushButton("Flow Field Editor")
        btn3 = QPushButton("Graph Editor Tool")
        btn4 = QPushButton("Edit Node")
        btn5 = QPushButton("New Population Connection")# hier auf jeden fall nodes miteinander verbinden
        btn6 = QPushButton("Delete Connection")
        btn7 = QPushButton("Add Blob")
        btn8 = QPushButton("Add Sensors")

        btn9 = QPushButton("Delete Graph")
        btn10 = QPushButton("Delete Node")

        btn1.clicked.connect(lambda: self.edit_menu.setCurrentIndex(0))
        btn2.clicked.connect(lambda: self.edit_menu.setCurrentIndex(1))
        btn3.clicked.connect(lambda: self.edit_menu.setCurrentIndex(2))
        btn4.clicked.connect(lambda: self.edit_menu.setCurrentIndex(3))
        btn5.clicked.connect(lambda: self.edit_menu.setCurrentIndex(4))
        btn6.clicked.connect(lambda: self.edit_menu.setCurrentIndex(5))
        btn7.clicked.connect(lambda: self.edit_menu.setCurrentIndex(6))
        btn8.clicked.connect(lambda: self.edit_menu.setCurrentIndex(7))
        btn9.clicked.connect(lambda: self.edit_menu.setCurrentIndex(8))
        btn10.clicked.connect(lambda: self.edit_menu.setCurrentIndex(9))
        # Zum Layout hinzufügen
        btn_layout.addWidget(btn1)
        btn_layout.addWidget(btn2)
        btn_layout.addWidget(btn3)
        btn_layout.addWidget(btn4)
        btn_layout.addWidget(btn5)
        btn_layout.addWidget(btn6)
        btn_layout.addWidget(btn7)
        btn_layout.addWidget(btn8)
        btn_layout.addWidget(btn9)
        btn_layout.addWidget(btn10)
        
        btn_widget.setLayout(btn_layout)
        return btn_widget
    


    def creation_menu(self,widget_list):
        stacked = QStackedWidget()
        for w in widget_list:
            stacked.addWidget(w)
        return stacked

    def visualization(self,widget_list):
        stacked = QStackedWidget()
        for w in widget_list:
            stacked.addWidget(w)
        return stacked
    
    def plot_points_of_graphs(self):
        self.node_vis.clear()
        
        legend_entries = []  # [(label, color), ...]
        used_types = set()  # Track welche Typen vorkommen
        
        for graph in self.graph_list:
            for node in graph.node_list:
                for i, pts in enumerate(node.positions):
                    
                    # --- FIX START: Prüfen ob Punkte existieren ---
                    if pts is None or len(pts) == 0:
                        continue
                    # --- FIX ENDE ---

                    # Sicherstellen, dass wir nicht out of range laufen, falls models fehlen
                    if i < len(node.neuron_models):
                        neuron_type = node.neuron_models[i]
                    else:
                        neuron_type = "unknown"

                    color = neuron_colors.get(neuron_type, "#FFFFFF")
                    
                    # Nur einmal pro Typ zur Legende hinzufügen
                    if neuron_type not in used_types:
                        legend_entries.append([neuron_type, color])
                        used_types.add(neuron_type)
                    
                    point_cloud = pv.PolyData(pts)
                    self.node_vis.add_mesh(
                        point_cloud,
                        color=color,
                        point_size=10
                    )
        
        # Legende hinzufügen (nur wenn Einträge vorhanden)
        if legend_entries:
            self.node_vis.add_legend(
                legend_entries,
                size=(0.2, 0.2),
                loc='upper right'
            )
        
        self.node_vis.update()
    
    
    def plot_graphs(self):
        self.graph_vis.clear()
        
        # Feste Farbpalette - gut unterscheidbare Farben
        color_palette = [
            [1.0, 0.0, 0.0],   # Rot
            [0.0, 1.0, 0.0],   # Grün  
            [0.0, 0.0, 1.0],   # Blau
            [1.0, 1.0, 0.0],   # Gelb
            [1.0, 0.0, 1.0],   # Magenta
            [0.0, 1.0, 1.0],   # Cyan
            [1.0, 0.5, 0.0],   # Orange
            [0.5, 0.0, 1.0],   # Lila
        ]
        
        for graph in self.graph_list:
            # Farbe aus Palette wählen
            color_idx = graph.graph_id % len(color_palette)
            graph_color = color_palette[color_idx]
            
            print(f"Graph {graph.graph_id}: Color {graph_color}")  # Debug
            
            # Plot Nodes als Spheres
            for node in graph.node_list:
                sphere = pv.Sphere(
                    radius=0.15, 
                    center=node.center_of_mass
                )
                self.graph_vis.add_mesh(
                    sphere,
                    color=graph_color,
                    opacity=0.8
                )
            
            # Plot Connections (Edges)
            for node in graph.node_list:
                for next_node in node.next:
                    start_pos = np.array(node.center_of_mass)
                    end_pos = np.array(next_node.center_of_mass)
                    
                    line = pv.Line(start_pos, end_pos)
                    self.graph_vis.add_mesh(
                        line,
                        color=graph_color,
                        line_width=3
                    )
        
        # Legende mit tatsächlich verwendeten Graphen
        legend_entries = []
        for graph in self.graph_list:
            color_idx = graph.graph_id % len(color_palette)
            legend_entries.append([f"Graph {graph.graph_id}", color_palette[color_idx]])
        
        if legend_entries:
            self.graph_vis.add_legend(
                legend_entries,
                size=(0.2, 0.2),
                loc='upper right'
            )
        
        self.graph_vis.reset_camera()
        self.graph_vis.update()

    def process_created_graph(self, graph_data):
        """
        Übersetzt die GUI-Daten in echte Node-Objekte der neuron_toolbox
        und aktualisiert die Visualisierung.
        """
        print("Processing graph data via Neuron Toolbox...")
        
        # Wir nutzen den ersten existierenden Graphen oder erstellen einen neuen Wrapper
        if not self.graph_list:
            self.graph_list = [Graph(graph_id=0)]
        
        active_graph = self.graph_list[0]
        
        for node_data in graph_data['nodes']:
            # 1. BERECHNUNG DER GRID-GRÖßE
            # Wave Function Collapse füllt nicht alles. Wir schätzen die nötige Grid-Größe.
            # Annahme: Sparsity factor ca. 0.8 (20% gefüllt) -> Wir brauchen 5x mehr Platz.
            # Formel: (NeuronCount / (1 - Sparsity))^(1/3)
            target_count = node_data['neuron_count']
            sparsity = 0.8 # Standard aus toolbox
            needed_volume = target_count / (1.0 - sparsity)
            side_length = int(np.ceil(needed_volume ** (1/3)))
            # Sicherheitsaufschlag, damit es nicht zu eng wird
            side_length += 2 
            
            grid_size = [side_length, side_length, side_length]
            
            # 2. DATEN FÜR WFC VORBEREITEN
            types = []
            probs = []
            models = []
            
            # Sortieren nach ID um Konsistenz zu sichern
            for i, pop in enumerate(node_data['populations']):
                types.append(i) # Typ ist einfach der Index
                probs.append(pop['percentage'] / 100.0) # % in 0.0-1.0
                models.append(pop['model'])
            
            # 3. NODE PARAMETER ZUSAMMENBAUEN
            # Wir mappen die GUI-Daten auf das params-Dict der Node-Klasse
            node_params = {
                "id": len(active_graph.node_list),
                "name": f"Node_{len(active_graph.node_list)}",
                "m": node_data['center_of_mass'], # Position aus GUI
                "grid_size": grid_size,
                "neuron_models": models,
                "types": types,
                "probability_vector": probs,
                "sparsity_factor": sparsity,
                "sparse_holes": 0,
                "dt": 0.01,
                "old": False, # Schnellerer WFC Algorithmus
                "plot_clusters": False # Wir plotten selbst in PyVista
            }
            
            # 4. NODE ERSTELLEN UND BAUEN (WFC ausführen)
            # Hier arbeitet die neuron_toolbox!
            print(f"Building Node with Grid {grid_size} for {target_count} neurons...")
            try:
                new_node = active_graph.create_node(parameters=node_params, auto_build=True)
                print(f"Node {new_node.id} created successfully with {len(new_node.population)} neurons.")
            except Exception as e:
                print(f"CRITICAL ERROR building node: {e}")

        # 5. VISUALISIERUNG UPDATE
        self.plot_points_of_graphs()
        self.plot_graphs()
        
        # Wechsel zur 3D Ansicht
        self.stacked.setCurrentIndex(0)


                    

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
