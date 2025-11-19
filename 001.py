import sys
from PyQt6.QtWidgets import QApplication, QMainWindow,QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtWidgets import QPushButton, QStackedWidget, QToolBar, QMenu, QGridLayout, QStackedLayout
from PyQt6.QtGui import QColor, QPalette, QAction
from PyQt6.QtCore import QSize, Qt
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


class graphCreationTool(QWidget):
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

            
            # RESET INTERFACE nach erfolgreicher Erstellung
            self.reset_interface()
        else:
            print("\n!!!invalid node parametrization. green ones have passed all checks, organge ones did not. !!!\n")
        
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



        #Erstellungsmenü für connections, nodes, graphen usw
        self.creation_functionalities = [graphCreationTool(),##################### HIER
                                         Color("white"),
                                         Color("purple"),
                                         Color("blue"),
                                         Color("green"),
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
        btn1 = QPushButton("Add Graph")
        btn2 = QPushButton("Add Node")
        btn3 = QPushButton("Edit Graph")
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
                    neuron_type = node.neuron_models[i]
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
        
        # Legende hinzufügen
        self.node_vis.add_legend(
            legend_entries,
            size=(0.2, 0.2),  # Größe der Legende (relativ)
            loc='upper right'  # oder 'lower left', 'upper left', etc.
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


                    

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
