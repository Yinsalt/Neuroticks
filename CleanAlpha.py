import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSlider, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QStackedWidget, QMessageBox, QProgressBar,
    QGridLayout, QFileDialog,QDoubleSpinBox,QTreeWidgetItemIterator
)
import WidgetLib
import pyqtgraph.dockarea as dock
from PyQt6.QtGui import QColor, QPalette, QAction
from PyQt6.QtCore import Qt,QSize
import numpy as np
import time
from CustomExtension import CustomTabWidget
import vtk
import pyvista as pv
from pyvistaqt import QtInteractor
from neuron_toolbox import *
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from WidgetLib import *
from WidgetLib import _clean_params, _serialize_connections, NumpyEncoder, BlinkingNetworkWidget, FlowFieldWidget, StructuresWidget, LiveConnectionController
import re
import shutil
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import vtk
nest.EnableStructuralPlasticity()
nest.set_verbosity("M_ERROR")


def apply_dark_mode(app):
    app.setStyle("Fusion")

    dark_palette = QPalette()
    
    dark_color = QColor(45, 45, 45)
    disabled_color = QColor(127, 127, 127)
    
    dark_palette.setColor(QPalette.ColorRole.Window, dark_color)
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, dark_color)
    
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    
    dark_palette.setColor(QPalette.ColorRole.Button, dark_color)
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_color)
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_color)
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_color)

    app.setPalette(dark_palette)

    app.setStyleSheet("""
        QToolTip { 
            color: #ffffff; 
            background-color: #2a82da; 
            border: 1px solid white; 
        }
        QMainWindow {
            background-color: #2b2b2b;
        }
        QGroupBox {
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            color: white;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
            background-color: #1e1e1e;
            color: white;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 2px;
        }
        QComboBox {
            background-color: #1e1e1e;
            color: white;
            border: 1px solid #555;
            padding: 3px;
        }
        QComboBox QAbstractItemView {
            background-color: #1e1e1e;
            color: white;
            selection-background-color: #2a82da;
        }
        QTabWidget::pane { 
            border: 1px solid #444; 
        }
        QTabBar::tab {
            background: #353535;
            color: #b1b1b1;
            padding: 8px 15px;
            border: 1px solid #444;
            border-bottom-color: #444; 
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #505050;
            color: white;
            border-bottom-color: #505050;
        }
        QHeaderView::section {
            background-color: #353535;
            color: white;
            padding: 4px;
            border: 1px solid #444;
        }
        QTableWidget {
            gridline-color: #444;
        }
        QTableWidget::item {
            color: white;
        }
        QScrollArea {
            border: none;
        }
        /* Buttons global */
        QPushButton {
            background-color: #444;
            color: white;
            border: 1px solid #555;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #555;
        }
        QPushButton:pressed {
            background-color: #333;
        }
        QPushButton:disabled {
            background-color: #333;
            color: #777;
        }
    """)


graph_list = []


neuron_colors = {
    'iaf_psc_alpha':            '#00FF88',
    'iaf_psc_exp':              '#00FF00',
    'iaf_psc_delta':            '#00CC66',
    'iaf_psc_alpha_multisynapse': '#00FFAA',
    'iaf_psc_exp_multisynapse': '#00FF44',

    'iaf_cond_alpha':           '#0088FF',
    'iaf_cond_exp':             '#00FFFF',
    'iaf_cond_beta':            '#0088CC',
    'iaf_cond_alpha_mc':        '#00AAFF',

    'aeif_cond_alpha':          '#FF00FF',
    'aeif_cond_exp':            '#FF00AA',
    'aeif_psc_alpha':           '#CC00FF',
    'aeif_psc_exp':             '#FF33FF',
    'aeif_cond_beta_multisynapse': '#AA00FF',

    'hh_psc_alpha':             '#FF4400',
    'hh_cond_exp_traub':        '#FF0000',
    'hh_cond_beta_gap_traub':   '#CC0000',
    'hh_psc_alpha_gap':         '#FF2200',

    'gif_cond_exp':             '#FFFF00',
    'gif_psc_exp':              '#FFCC00',
    'gif_cond_exp_multisynapse': '#FFEE00',
    'glif_cond':                '#FFD700',
    'glif_psc':                 '#FFAA00',

    'izhikevich':               '#FF8800',
    'mat2_psc_exp':             '#FF6600',
    'amat2_psc_exp':            '#FF9933',
    'ht_neuron':                '#FF33AA',
    'pp_psc_delta':             '#FF0066',
    'siegert_neuron':           '#00FFCC',

    'parrot_neuron':            '#888888',
    'parrot_neuron_ps':         '#666666',
    'mcculloch_pitts_neuron':   '#444444',
    'unknown':                  '#FFFFFF',
    'default':                  '#8888FF'
}


def convert_widget_to_graph_format(graph_id):
    global graph_parameters
    
    if graph_id not in graph_parameters:
        raise ValueError(f"Graph {graph_id} not found in graph_parameters")
    
    data = graph_parameters[graph_id]
    converted = {
        'graph_name': data.get('graph_name', f'Graph_{graph_id}'),
        'graph_id': graph_id,
        'color': data.get('color', '0xffffff'),
        'parameter_list': data.get('parameter_list', []),
        'max_nodes': len(data.get('parameter_list', []))
    }
    
    del graph_parameters[graph_id]
    return converted

class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            background-color: #000000; 
            border: 1px solid #333;
        """)
        self.setMinimumSize(80, 80)
        self._updating = False
        
    def resizeEvent(self, event):
        if self._updating:
            return super().resizeEvent(event)
        
        self._updating = True
        width = event.size().width()
        if self.height() != width:
            self.setFixedHeight(width)
        self._updating = False
        super().resizeEvent(event)
    
    def sizeHint(self):
        return QSize(150, 150)
    
    def minimumSizeHint(self):
        return QSize(80, 80)


def create_graph_from_widget(graph_id):
    global graph_list
    
    converted = convert_widget_to_graph_format(graph_id)
    
    graph = Graph(
        graph_name=converted['graph_name'],
        graph_id=converted['graph_id'],
        parameter_list=converted['parameter_list'],
        max_nodes=converted['max_nodes']
    )
    
    for i, node_params in enumerate(converted['parameter_list']):
        if i == 0:
            graph.create_node(parameters=node_params, is_root=True, auto_build=True)
        else:
            user_pos = node_params.get('center_of_mass', [0.0, 0.0, 0.0])
            has_explicit_position = not np.allclose(user_pos, [0.0, 0.0, 0.0])
            
            if has_explicit_position:
                graph.create_node(parameters=node_params, auto_build=True)
            else:
                graph.create_node(
                    parameters=node_params,
                    other=graph.node_list[i-1],
                    auto_build=True
                )
    
    graph_list.append(graph)
    return graph


class Color(QWidget):
    def __init__(self, color):
        super().__init__()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)
def get_neuron_color(model_name):
        return neuron_colors.get(model_name, neuron_colors['default'])


class StatusBarWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #2E7D32;
                font-weight: bold;
                padding: 2px 5px;
            }
        """)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;     
                border-radius: 3px;
                text-align: center;
                background-color: #333;     
                color: white;               
            }
            QProgressBar::chunk {
                background-color: #4CAF50; 
                border-radius: 2px;
            }
        """)
        self.progress_bar.setVisible(False)
        
        layout.addWidget(self.status_label, 8)
        layout.addWidget(self.progress_bar, 2)
    
    def set_status(self, message, color="#2E7D32"):
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-weight: bold;
                padding: 2px 5px;
            }}
        """)
    
    def set_progress(self, value, maximum=100, show=True):
        if show:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(maximum)
            self.progress_bar.setValue(value)
        else:
            self.progress_bar.setVisible(False)
    
    def reset(self):
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #2E7D32;
                font-weight: bold;
                padding: 2px 5px;
            }
        """)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
    
    def show_error(self, message):
        self.set_status(f"{message}", color="#D32F2F")
        self.progress_bar.setVisible(False)
    
    def show_success(self, message):
        self.set_status(f"{message}", color="#2E7D32")
        self.progress_bar.setVisible(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuroticks")
        self.setMinimumSize(1280, 720)
        self.resize(1920, 1080)
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.active_graphs = {}
        self.graphs = self.active_graphs
        self.structural_plasticity_enabled = True
        self.hidden_nodes = set()
        self.hidden_graphs = set()
        self.create_menubar()
        self.setup_ui()
        self.graph_builder.polynom_manager.polynomialsChanged.connect(self.rebuild_node_with_new_polynomials)


    def create_menubar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Project", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_project_dialog)
        file_menu.addAction(new_action)
        
        save_action = QAction("Save Project", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_all_graphs_dialog)
        file_menu.addAction(save_action)
        
        load_action = QAction("Load Project", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_all_graphs_dialog)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        settings_menu = menubar.addMenu("⚙️ Settings")
        nest_menu = menubar.addMenu("⚙️ NEST Settings")
        self.action_auto_save = QAction("Auto-Save Live History on Reset", self)
        self.action_auto_save.setCheckable(True)
        self.action_auto_save.setChecked(True)
        settings_menu.addAction(self.action_auto_save)
        self.plasticity_action = QAction("🧠 Structural Plasticity", self)
        self.plasticity_action.setCheckable(True)
        self.plasticity_action.setChecked(self.structural_plasticity_enabled)
        self.plasticity_action.triggered.connect(self.toggle_structural_plasticity)
        nest_menu.addAction(self.plasticity_action)
        
        nest_menu.addSeparator()
        
        reset_action = QAction("🔄 Reset NEST Kernel", self)
        reset_action.triggered.connect(self.manual_nest_reset)
        nest_menu.addAction(reset_action)
        
        view_menu = menubar.addMenu("👁️ View")
        refresh_action = QAction("🔄 Refresh Visualizations", self)
        refresh_action.triggered.connect(self.update_visualizations)
        view_menu.addAction(refresh_action)


    def init_connection_system(self):

        self.connection_executor = ConnectionExecutor(self.graphs)
        
        if hasattr(self, 'connection_tool'):
            self.connection_tool.set_graphs(self.graphs)
    def new_project_dialog(self):

        if graph_list:
            reply = QMessageBox.question(
                self,
                "New Project",
                "Do you want to save your changes before starting a new project?\n\n"
                "If you choose 'Discard', all current graphs and settings will be lost.",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Cancel:
                return
            
            if reply == QMessageBox.StandardButton.Save:
                self.save_all_graphs_dialog()
        
        self.reset_application()

    def reset_application(self):
        self.status_bar.set_status("Cleaning up for new project...", color="#FF5722")
        QApplication.processEvents()
        
        try:
            global graph_list
            graph_list.clear()
            
            WidgetLib.next_graph_id = 0
            WidgetLib.graph_parameters.clear()
            if hasattr(self, 'tools_widget'):
                self.tools_widget.update_graphs(graph_list)
  
            
            print("Resetting NEST Kernel...")
            nest.ResetKernel()
            if self.structural_plasticity_enabled:
                nest.EnableStructuralPlasticity()
            
            self.active_graphs.clear()
            if hasattr(self, 'simulation_view') and self.simulation_view:
                self.simulation_view._initialized = False
            if hasattr(self, 'graph_builder'):
                self.graph_builder.reset()
            
            if hasattr(self, 'graph_editor'):
                self.graph_editor.current_graph = None
                self.graph_editor.current_graph_id = None
                self.graph_editor.node_list.clear()

                self.graph_editor.graph_name_input.clear()
                self.graph_editor.refresh_graph_list()
                
                while self.graph_editor.node_list_layout.count():
                    item = self.graph_editor.node_list_layout.takeAt(0)
                    if item.widget(): item.widget().deleteLater()
                while self.graph_editor.pop_list_layout.count():
                    item = self.graph_editor.pop_list_layout.takeAt(0)
                    if item.widget(): item.widget().deleteLater()
            
            if hasattr(self, 'connection_tool'):
                self.connection_tool.connections.clear()
                self.connection_tool.next_conn_id = 0
                self.connection_tool.update_connection_list()
                self.connection_tool.refresh()
            
            self.update_visualizations()
            
            if hasattr(self, 'graph_overview'):
                self.graph_overview.update_tree()
                
            if hasattr(self, 'blink_widget'):
                self.blink_widget.build_scene()
            
            self.status_bar.show_success("New Project created!")
            print("Application reset successful.")
            
        except Exception as e:
            self.status_bar.show_error(f"Error during reset: {e}")
            print(f"Reset Error: {e}")
            import traceback
            traceback.print_exc()
    def safe_reset_kernel(self, enable_structural_plasticity=False):
         
        if hasattr(self, 'status_bar'):
            self.status_bar.set_status("Resetting NEST kernel...", color="#FF5722")
            QApplication.processEvents()
        
        stats = safe_nest_reset_and_repopulate(
            self.active_graphs,
            enable_structural_plasticity=enable_structural_plasticity,
            verbose=True
        )
        
        msg = (f"Reset complete: {stats['populations_created']} populations, "
               f"{stats['connections_created']} connections")
        
        if hasattr(self, 'status_bar'):
            self.status_bar.show_success(msg)
        
        if hasattr(self, 'update_visualizations'):
            self.update_visualizations()
        
        return stats


    def rebuild_node_with_new_polynomials(self, node_idx, polynomials):
        if not self.active_graphs:
            print("No active graphs to rebuild")
            return
        
        graph = list(self.active_graphs.values())[-1]
        
        if node_idx >= len(graph.node_list):
            print(f"⚠ Node {node_idx} not found in graph")
            return
        
        node = graph.node_list[node_idx]
        
        print(f"Rebuilding Node {node_idx} with new polynomials...")
        
        encoded_polynoms_per_type = []
        for poly_dict in polynomials:
            if poly_dict and all(k in poly_dict for k in ['x', 'y', 'z']):
                encoded_polynoms_per_type.append([poly_dict['x'], poly_dict['y'], poly_dict['z']])
            else:
                encoded_polynoms_per_type.append([])
        
        node.parameters['encoded_polynoms_per_type'] = encoded_polynoms_per_type
        
        node.build()
        
        nest.ResetKernel()
        for g in self.active_graphs.values():
            for n in g.node_list:
                n.populate_node()
        
        self.update_visualizations()
        print(f"Node {node_idx} rebuilt successfully!")
    def toggle_structural_plasticity(self, enabled):
                
        if enabled:
            self.safe_reset_kernel(enable_structural_plasticity=True)
        else:
            self.safe_reset_kernel(enable_structural_plasticity=False)
    
    def on_graph_created(self, graph):

        self.graphs[graph.graph_id] = graph
        
        if hasattr(self, 'connection_tool'):
            self.connection_tool.set_graphs(self.graphs)
    
    def on_graph_deleted(self, graph_id):
        if graph_id in self.graphs:
            del self.graphs[graph_id]
        
        if hasattr(self, 'connection_tool'):
            self.connection_tool.set_graphs(self.graphs)
    
    def get_connection_summary(self):
        return get_all_connections_summary(self.graphs)
    
    def export_connections(self, filepath):
        import json
        
        connections = export_connections_to_dict(self.graphs)
        with open(filepath, 'w') as f:
            json.dump(connections, f, indent=2)
        
        return len(connections)
    
    def import_connections(self, filepath, clear_existing=True):
        import json
        
        with open(filepath, 'r') as f:
            connections = json.load(f)
        
        return import_connections_from_dict(self.graphs, connections, clear_existing)
    
    def manual_nest_reset(self):
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            'Reset NEST Kernel',
            'This will reset NEST and repopulate all graphs.\n\nContinue?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.status_bar.set_status("Resetting NEST...", color="#1976D2")
            self.status_bar.set_progress(0)
            QApplication.processEvents()
            
            print("\nManual NEST Reset...")
            nest.ResetKernel()
            self.status_bar.set_progress(30)
            QApplication.processEvents()
            
            if self.structural_plasticity_enabled:
                nest.EnableStructuralPlasticity()
            else:
                nest.DisableStructuralPlasticity()
            
            if graph_list:
                total_graphs = len(graph_list)
                for i, graph in enumerate(graph_list):
                    self.status_bar.set_status(f"Repopulating graph {i+1}/{total_graphs}...", color="#1976D2")
                    self.status_bar.set_progress(30 + int(50 * (i+1) / total_graphs))
                    QApplication.processEvents()
                    
                    for node in graph.node_list:
                        node.populate_node()
                
                self.status_bar.set_status("Updating visualizations...", color="#1976D2")
                self.status_bar.set_progress(90)
                QApplication.processEvents()
                
                self.update_visualizations()
                self.graph_overview.update_tree()

            if hasattr(self, 'simulation_view'):
                self.simulation_view.restore_injectors()
            
            self.status_bar.show_success("NEST Reset complete!")
    

    def verify_nest_populations(self):
        print("VERIFYING NEST POPULATIONS")
        
        all_ok = True
        
        for graph in graph_list:
            graph_name = getattr(graph, 'graph_name', f'Graph {graph.graph_id}')
            print(f"\n{graph_name} (ID: {graph.graph_id})")
            
            for node in graph.node_list:
                node_name = getattr(node, 'name', 'Unnamed')
                
                if not hasattr(node, 'population') or not node.population:
                    print(f" {node_name}: No NEST population")
                    all_ok = False
                    continue
                
                try:
                    total_neurons = 0
                    for pop_idx, nest_pop in enumerate(node.population):
                        if nest_pop is None or len(nest_pop) == 0:
                            continue
                        
                        actual_pos = nest.GetPosition(nest_pop)
                        neuron_count = len(nest_pop)
                        total_neurons += neuron_count
                        
                        if len(actual_pos) != neuron_count:
                            print(f"{node_name} Pop{pop_idx}: Position mismatch")
                            all_ok = False
                        else:
                            print(f"{node_name} Pop{pop_idx}: {neuron_count} neurons spatial")
                    
                    if total_neurons == 0:
                        print(f"{node_name}: Total 0 neurons")
                        all_ok = False
                        
                except Exception as e:
                    print(f"{node_name}: {type(e).__name__}: {e}")
                    all_ok = False
        
        report = "All verified" if all_ok else "Issues found"
        
        
        return all_ok, report
    
    
    def update_visualizations(self):
        print("VERIFYING NEST POPULATIONS")
        
        success, report = self.verify_nest_populations()
        
        if not success:
            print("\nWARNING: Some nodes have issues with NEST populations!")
        else:
            print("\nAll NEST populations verified successfully!")
        
        self.plot_neuron_points()
        self.plot_graph_skeleton()
        
        if hasattr(self, 'blink_widget'):
            print("Rebuilding Simulation Scene...")
            self.blink_widget.build_scene()


    def setup_ui(self):
        central_widget = QWidget()
        central_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        switch_bar = QWidget()
        switch_bar.setStyleSheet("background-color: #1a1a1a; border-bottom: 2px solid #333;")
        switch_bar.setFixedHeight(50)
        switch_layout = QHBoxLayout(switch_bar)
        switch_layout.setContentsMargins(10, 5, 10, 5)
        switch_layout.setSpacing(10)
        
        self.view_switch_style_active = "QPushButton { background-color: #2196F3; color: white; font-weight: bold; border: none; border-radius: 4px; padding: 10px 30px; font-size: 14px; } QPushButton:hover { background-color: #42A5F5; }"
        self.view_switch_style_inactive = "QPushButton { background-color: #333; color: #aaa; font-weight: bold; border: 1px solid #555; border-radius: 4px; padding: 10px 30px; font-size: 14px; } QPushButton:hover { background-color: #444; color: white; }"
        
        self.btn_view_editor = QPushButton(" EDITOR")
        self.btn_view_simulation = QPushButton(" SIMULATION")
        self.btn_view_data = QPushButton(" DATA")
        self.btn_view_custom = QPushButton(" EXTENSION")
        
        for btn in [self.btn_view_editor, self.btn_view_simulation, self.btn_view_data, self.btn_view_custom]:
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self.view_switch_style_inactive)

        self.btn_view_editor.clicked.connect(lambda: self._switch_main_view(0))
        self.btn_view_simulation.clicked.connect(lambda: self._switch_main_view(1))
        self.btn_view_data.clicked.connect(lambda: self._switch_main_view(2))
        self.btn_view_custom.clicked.connect(lambda: self._switch_main_view(3))
        
        self.view_buttons = [self.btn_view_editor, self.btn_view_simulation, self.btn_view_data, self.btn_view_custom]
        
        switch_layout.addWidget(self.btn_view_editor)
        switch_layout.addWidget(self.btn_view_simulation)
        switch_layout.addWidget(self.btn_view_data)
        switch_layout.addWidget(self.btn_view_custom)
        switch_layout.addStretch()

        self.global_sim_control = QWidget()
        self.global_sim_control.setStyleSheet("background-color: transparent;")
        gsc_layout = QHBoxLayout(self.global_sim_control)
        gsc_layout.setContentsMargins(0, 0, 0, 0)
        gsc_layout.setSpacing(8)
        
        lbl_step = QLabel("Res:"); lbl_step.setStyleSheet("color:#888;")
        self.global_step_spin = QDoubleSpinBox()
        self.global_step_spin.setRange(0.1, 1000); self.global_step_spin.setValue(25.0); self.global_step_spin.setSuffix(" ms")
        self.global_step_spin.setFixedWidth(70); self.global_step_spin.setStyleSheet("background:#333; color:#00E5FF; border:1px solid #555;")
        
        lbl_dur = QLabel("Target:"); lbl_dur.setStyleSheet("color:#888;")
        self.global_duration_spin = QDoubleSpinBox()
        self.global_duration_spin.setRange(0, 1e7); self.global_duration_spin.setValue(1000.0); self.global_duration_spin.setSuffix(" ms")
        self.global_duration_spin.setFixedWidth(90); self.global_duration_spin.setStyleSheet("background:#333; color:#00E5FF; border:1px solid #555;")
        
        self.global_time_label = QLabel("T: 0.0 ms")
        self.global_time_label.setFixedWidth(100)
        self.global_time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.global_time_label.setStyleSheet("background-color: #000; color: #00FF00; font-family: Consolas; font-size: 14px; font-weight: bold; border: 1px solid #444; padding: 2px;")
        
        btn_style = "font-weight: bold; border-radius: 3px; padding: 5px 10px;"
        
        self.global_btn_start = QPushButton("▶")
        self.global_btn_start.setStyleSheet(f"{btn_style} background-color: #2E7D32; color: white;")
        self.global_btn_start.clicked.connect(self._global_start)
        
        self.global_btn_pause = QPushButton("⏸")
        self.global_btn_pause.setStyleSheet(f"{btn_style} background-color: #FBC02D; color: black;")
        self.global_btn_pause.clicked.connect(self._global_pause)
        
        self.global_btn_reset = QPushButton("↺")
        self.global_btn_reset.setStyleSheet(f"{btn_style} background-color: #E65100; color: white;")
        self.global_btn_reset.clicked.connect(self._global_reset)
        
        gsc_layout.addWidget(lbl_step)
        gsc_layout.addWidget(self.global_step_spin)
        gsc_layout.addWidget(lbl_dur)
        gsc_layout.addWidget(self.global_duration_spin)
        gsc_layout.addWidget(self.global_time_label)
        gsc_layout.addWidget(self.global_btn_start)
        gsc_layout.addWidget(self.global_btn_pause)
        gsc_layout.addWidget(self.global_btn_reset)
        
        switch_layout.addWidget(self.global_sim_control)
        main_layout.addWidget(switch_bar)
        
        self.main_stack = QStackedWidget()
        self.editor_widget = self._create_editor_widget()
        self.simulation_view = SimulationViewWidget(graph_list, self)
        self.live_dashboard = LiveDataDashboard(graph_list, self)
        self.custom_tab = CustomTabWidget(graph_list, self)
        self.data_view = self._create_data_view()
        
        self.main_stack.addWidget(self.editor_widget)
        self.main_stack.addWidget(self.simulation_view)
        self.main_stack.addWidget(self.live_dashboard)
        self.main_stack.addWidget(self.custom_tab)
        
        main_layout.addWidget(self.main_stack)

        self.sim_dashboard.sigDurationChanged.connect(self.global_duration_spin.setValue)
        self.global_duration_spin.valueChanged.connect(self.sim_dashboard.update_duration_from_external)
        
        self.update_visualizations()
        self.init_simulation_timer()
        self._switch_main_view(0)


    def _create_data_view(self):
        self.data_dashboard = AnalysisDashboard(graph_list)
        return self.data_dashboard
    
    def _switch_main_view(self, index):
        self.main_stack.setCurrentIndex(index)
        
        for i, btn in enumerate(self.view_buttons):
            if i == index:
                btn.setStyleSheet(self.view_switch_style_active)
            else:
                btn.setStyleSheet(self.view_switch_style_inactive)
        
        if index == 1:
            if hasattr(self, 'simulation_view') and self.simulation_view.scene_loaded:
                self.simulation_view.start_rendering()
        else:
            if hasattr(self, 'simulation_view'):
                self.simulation_view.stop_rendering_safe()
                
        if index == 2:
            if hasattr(self, 'live_dashboard'):
                self.live_dashboard.reload_and_center()
            
            if hasattr(self, 'data_dashboard'):
                self.data_dashboard.refresh_all_tabs()
                
        if index == 3:
            if hasattr(self, 'custom_tab'):
                self.custom_tab.on_tab_active()
    
    


    def _create_editor_widget(self):
        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)
        
        top_layout = QHBoxLayout()
        top_left = self.create_top_left()
        self.graph_overview = GraphOverviewWidget(self, graph_list=graph_list)
        top_layout.addLayout(top_left, 7)
        top_layout.addWidget(self.graph_overview, 3)
        
        bottom_layout = QHBoxLayout()
        bottom_left = self.create_bottom_left()
        bottom_right = self.create_bottom_right()
        
        bottom_layout.addLayout(bottom_left, 7)
        bottom_layout.addLayout(bottom_right, 3)
        
        editor_layout.addLayout(top_layout, 3)
        editor_layout.addLayout(bottom_layout, 2)
        
        return editor
  
    


    def create_top_left(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        scene_menu = QWidget()
        scene_menu.setStyleSheet("background-color: #2b2b2b;")
        
        scene_layout = QVBoxLayout()
        scene_layout.setContentsMargins(0, 0, 0, 0)
        scene_layout.setSpacing(0)
        
        nav_style = """
            QPushButton {
                background-color: #333;
                color: #aaa;
                border: 1px solid #444;
                border-radius: 0px;
                font-weight: bold;
                text-align: center;
                padding: 5px;
            }
            QPushButton:hover { 
                background-color: #444; 
                color: white;
                border: 1px solid #2196F3; 
            }
            QPushButton:checked { 
                background-color: #555; 
                color: white;
                border: 1px solid #2196F3;
                border-left: 4px solid #2196F3; 
            }
        """
        
        btn_neurons = QPushButton("Neurons")
        btn_graph = QPushButton("Graph")
        
        self.firing_patterns_container = QWidget()
        fp_layout = QVBoxLayout(self.firing_patterns_container)
        fp_layout.setContentsMargins(0, 0, 0, 0)
        fp_layout.setSpacing(0)
        
        btn_sim = QPushButton("Firing\nPatterns")
        
        self.slider_wrapper = QWidget()
        self.slider_wrapper.setStyleSheet("background-color: #3a3a3a; border-left: 4px solid #FFD700;")
        self.slider_wrapper.setVisible(False)
        slider_layout = QVBoxLayout(self.slider_wrapper)
        slider_layout.setContentsMargins(10, 5, 10, 10)
        
        lbl_slider = QLabel("Edge Opacity")
        lbl_slider.setStyleSheet("color: #ddd; font-size: 10px; font-weight: normal; border: none;")
        lbl_slider.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(30)
        self.opacity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #555;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #FFD700;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
        """)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        
        slider_layout.addWidget(lbl_slider)
        slider_layout.addWidget(self.opacity_slider)
        
        fp_layout.addWidget(btn_sim)
        fp_layout.addWidget(self.slider_wrapper)
        
        btn_flow = QPushButton("Positional\nFlowfield")
        btn_simulation = QPushButton("Simulation")
        btn_other = QPushButton("Other")
        
        all_nav_buttons = [btn_neurons, btn_graph, btn_sim, btn_flow, btn_simulation, btn_other]
        for btn in all_nav_buttons:
            btn.setStyleSheet(nav_style)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            btn.setCheckable(True)
        
        buttons_container = QWidget()
        buttons_layout = QVBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(1)
        
        buttons_layout.addWidget(btn_neurons, 1)
        buttons_layout.addWidget(btn_graph, 1)
        buttons_layout.addWidget(self.firing_patterns_container, 1)
        buttons_layout.addWidget(btn_flow, 1)
        buttons_layout.addWidget(btn_simulation, 1)
        buttons_layout.addWidget(btn_other, 1)
        
        self.preview_widget = PreviewWidget()
        
        scene_layout.addWidget(buttons_container, 1)
        scene_layout.addWidget(self.preview_widget, 0)
        
        scene_menu.setLayout(scene_layout)
        
        self.vis_stack = QStackedWidget()
        
        self.neuron_plotter = self.create_neuron_visualization()
        self.graph_plotter = self.create_graph_visualization()
        self.blink_widget = BlinkingNetworkWidget(graph_list)
        self.flow_widget = FlowFieldWidget(graph_list)
        
        self.vis_stack.addWidget(self.neuron_plotter)
        self.vis_stack.addWidget(self.graph_plotter)
        self.vis_stack.addWidget(self.blink_widget)
        self.vis_stack.addWidget(self.flow_widget)
        self.sim_dashboard = SimulationDashboardWidget(graph_list)
        
        self.sim_dashboard.requestStartSimulation.connect(self.start_headless_simulation)
        self.sim_dashboard.requestStopSimulation.connect(self.stop_headless_simulation)
        self.sim_dashboard.requestResetKernel.connect(self.manual_nest_reset)
        
        self.vis_stack.addWidget(self.sim_dashboard)
        self.vis_stack.addWidget(Color("darkorange"))
        
        self.nav_buttons = [btn_neurons, btn_graph, btn_sim, btn_flow, btn_simulation, btn_other]
        
        btn_neurons.clicked.connect(lambda: self._switch_view(0))
        btn_graph.clicked.connect(lambda: self._switch_view(1))
        btn_sim.clicked.connect(lambda: self._switch_view(2, sim_mode=True))
        btn_flow.clicked.connect(lambda: self._switch_view(3))
        btn_simulation.clicked.connect(lambda: self._switch_view(4))
        btn_other.clicked.connect(lambda: self._switch_view(5))
        
        btn_neurons.setChecked(True)
        
        layout.addWidget(scene_menu, 1)
        layout.addWidget(self.vis_stack, 9)
        
        return layout
        
    
    def start_headless_simulation(self, duration):
        print(f"\n>>> STARTING HEADLESS SIMULATION ({duration} ms) <<<")
        
        self.sim_dashboard.set_ui_locked(True)
        self.simulation_view.setEnabled(False)
        self.btn_view_simulation.setEnabled(False)
        
        self.status_bar.set_status("HEADLESS SIMULATION RUNNING...", color="#E65100")
        self.status_bar.set_progress(0)
        
        self.headless_target_time = duration
        self.headless_current_time = 0.0
        
        self._ensure_spike_recorders()
        
        try:
            kernel_time = nest.GetKernelStatus().get('time', 0.0)
            self.headless_target_time += kernel_time
        except: pass

        self.headless_step_size = 50.0
        self.headless_timer = QTimer()
        self.headless_timer.timeout.connect(self.headless_loop_step)
        self.headless_timer.start(0)

    def headless_loop_step(self):
        try:
            current_time = nest.GetKernelStatus().get('time', 0.0)
            
            remaining = self.headless_target_time - current_time
            
            if remaining <= 0.0001:
                self.finish_headless_simulation()
                return

            step_to_take = min(self.headless_step_size, remaining)

            nest.Simulate(step_to_take)
            
            current_time = nest.GetKernelStatus().get('time', 0.0)
            
            prog = int((current_time % 1000) / 10)
            self.status_bar.set_progress(prog)
            
        except Exception as e:
            self.headless_timer.stop()
            self.stop_headless_simulation(error_msg=str(e))

    def stop_headless_simulation(self, error_msg=None):
        if hasattr(self, 'headless_timer'):
            self.headless_timer.stop()
            
        print(">>> Headless Simulation Stopped.")
        
        if error_msg:
            self.status_bar.show_error(f"Headless Error: {error_msg}")
            QMessageBox.critical(self, "Simulation Error", error_msg)
        else:
            self.status_bar.set_status("Simulation stopped by user.", "#D32F2F")
            
        self._restore_ui_after_headless()
        self.collect_simulation_results(0)

    def finish_headless_simulation(self):
        self.headless_timer.stop()
        print(">>> Headless Simulation Finished.")
        self.status_bar.show_success("Headless Simulation Complete!")
        
        self.collect_simulation_results(0)
        self._restore_ui_after_headless()

    def _restore_ui_after_headless(self):
        self.sim_dashboard.set_ui_locked(False)
        self.simulation_view.setEnabled(True)
        self.btn_view_simulation.setEnabled(True)
        self.status_bar.set_progress(100)

    


    def reset_from_live_dashboard(self, keep_data):
        print("Live Dashboard Reset triggered.")
        self.sim_timer.stop()
        
        if self.action_auto_save.isChecked():
            self.archive_live_data()
            
        self.reset_and_restart()
        self.simulation_view.update_time_display(0.0)


    def archive_live_data(self):
        print("Archiving Live Data history...")
        try:
            data = self.live_dashboard.get_all_data()
            if not data:
                print("No data to save.")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"history_live_{timestamp}.json"
            
            Path("sim_history").mkdir(exist_ok=True)
            filepath = Path("sim_history") / filename
            
            with open(filepath, 'w') as f:
                json.dump(data, f, cls=NumpyEncoder, indent=2)
                
            self.status_bar.show_success(f"History saved: {filename}")
            print(f"Saved history to {filepath}")
            
        except Exception as e:
            print(f"Error archiving data: {e}")
            self.status_bar.show_error(f"Save Error: {e}")

    
        
    def _switch_view(self, index, sim_mode=False):
        
        self.vis_stack.setCurrentIndex(index)
        
        if index == 2:
            self.slider_wrapper.setVisible(True)
        else:
            self.slider_wrapper.setVisible(False)
            
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        
        if index == 2:
            self.blink_widget.build_scene()
            if sim_mode:
                self.blink_widget.start_simulation()
            else:
                self.blink_widget.stop_simulation()
        else:
            self.blink_widget.stop_simulation()
            
        if index == 3:
            if hasattr(self, 'flow_widget'):
                if self.flow_widget.target_node_id is None and graph_list:
                    first_graph = graph_list[0]
                    if first_graph.node_list:
                        first_node = first_graph.node_list[0]
                        self.flow_widget.set_target_node(first_graph.graph_id, first_node.id)
                
                self.flow_widget.build_scene()
        if index == 4:
            self.sim_dashboard.refresh_data()

    def create_neuron_visualization(self):
        plotter = QtInteractor(self)
        plotter.set_background('black')
        plotter.add_axes()
        return plotter
    
    def _on_opacity_changed(self, value):
        opacity = value / 100.0
        if hasattr(self, 'blink_widget'):
            self.blink_widget.set_base_opacity(opacity)

    def create_graph_visualization(self):
        plotter = QtInteractor(self)
        plotter.set_background('black')
        plotter.add_axes()
        return plotter
    def on_graph_updated(self, graph_id):
        if graph_id == -1:
            self.status_bar.set_status("Graph deleted, refreshing...", color="#1976D2")
            if hasattr(self, 'tools_widget'):
                self.tools_widget.update_graphs(graph_list)
        else:
            print(f"Graph {graph_id} updated, refreshing...")
            self.status_bar.set_status(f"Graph {graph_id} updated, refreshing...", color="#1976D2")
        
        self.status_bar.set_progress(50)
        QApplication.processEvents()
        
        self.update_visualizations()
        self.graph_overview.update_tree()
        self.connection_tool.refresh()
        self.status_bar.show_success("Changes saved!")

        QApplication.processEvents()

    def create_bottom_left(self):
        layout = QVBoxLayout()
        
        self.tool_stack = QStackedWidget()

        self.graph_builder = GraphCreatorWidget()
        self.graph_builder.graphCreated.connect(self.on_graph_created)
        self.tool_stack.addWidget(self.graph_builder)
        
        self.graph_editor = EditGraphWidget(graph_list=graph_list)
        self.graph_editor.graphUpdated.connect(self.on_graph_updated)
        self.tool_stack.addWidget(self.graph_editor)
        
        self.connection_tool = ConnectionTool(graph_list)
        
        self.connection_tool.connectionsCreated.connect(self.graph_overview.update_tree)
        self.connection_tool.connectionsCreated.connect(self.update_visualizations)
        
        self.tool_stack.addWidget(self.connection_tool)
        
        self.structures_widget = StructuresWidget()
        self.structures_widget.structureSelected.connect(self.on_structure_selected)
        self.tool_stack.addWidget(self.structures_widget)
        
        self.tools_widget = ToolsWidget()
        self.tools_widget.update_graphs(graph_list)
        self.tools_widget.deviceAdded.connect(self.graph_overview.update_tree)
        self.tools_widget.deviceAdded.connect(self.update_visualizations)
        
        for i in range(self.tools_widget.config_stack.count()):
            widget = self.tools_widget.config_stack.widget(i)
            if isinstance(widget, DeviceConfigPage):
                widget.deviceUpdated.connect(self.handle_device_update)

        self.tool_stack.addWidget(self.tools_widget)
        self.status_bar = StatusBarWidget()
        
        layout.addWidget(self.tool_stack, 9)
        layout.addWidget(self.status_bar, 1)
        
        self.graph_overview.node_selected.connect(self._on_overview_node_selected)
        self.graph_overview.device_selected.connect(self.on_device_tree_click)
        self.graph_overview.population_selected.connect(self._on_overview_pop_selected)
        self.graph_overview.connection_selected.connect(self._on_overview_conn_selected)
        self.graph_overview.requestConnectionCreation.connect(self.open_connection_tool_for_node)
        self.graph_overview.requestConnectionDeletion.connect(self.delete_connection_wrapper)
        
        self.graph_overview.requestDeviceDeletion.connect(self.delete_device_wrapper)
        self.graph_overview.requestLiveWeightChange.connect(self.on_live_weight_change)

        return layout
    
    def on_device_tree_click(self, device_data):
        self.tool_stack.setCurrentIndex(4)
        self.tools_widget.open_device_editor(device_data)
        self.status_bar.set_status(f"Editing Device: {device_data.get('model')}", "#FF9800")

    def handle_device_update(self, old_data, new_data):
        print("\n=== UPDATING DEVICE (In-Place) ===")
        
        if hasattr(self, 'sim_timer'):
            self.sim_timer.stop()
        self.live_recorders = []

        self.status_bar.set_status("Updating Device Params & Rebuilding...", "#FF5722")
        QApplication.processEvents()

        target = old_data.get('target', {})
        gid = target.get('graph_id')
        nid = target.get('node_id')
        old_id = old_data.get('id')
        
        target_graph = next((g for g in graph_list if g.graph_id == gid), None)
        
        updated = False
        
        if target_graph:
            node = target_graph.get_node(nid)
            if node:
                if hasattr(node, 'parameters') and 'devices' in node.parameters:
                    for dev_conf in node.parameters['devices']:
                        if str(dev_conf.get('id')) == str(old_id):
                            dev_conf['params'] = new_data['params']
                            dev_conf['conn_params'] = new_data.get('conn_params', {})
                            dev_conf['model'] = new_data.get('model', dev_conf['model'])
                            
                            dev_conf['runtime_gid'] = None
                            updated = True
                            print(f"  ✓ Device config {old_id} updated in-place.")
                            break
        
        if not updated:
            print("Error: Could not find device to update in parameters.")
            self.status_bar.show_error("Update failed: Device not found.")
            return

        print("Resetting Kernel...")
        nest.ResetKernel()
        if self.structural_plasticity_enabled:
            nest.EnableStructuralPlasticity()
            
        self.rebuild_all_graphs(reset_nest=False, verbose=False)
        
        self.update_visualizations()
        self.graph_overview.update_tree()
        self.status_bar.show_success("Device updated successfully!")


    def delete_connection_wrapper(self, conn_data):
        from PyQt6.QtWidgets import QMessageBox
        
        name = conn_data.get('name', 'Connection')
        reply = QMessageBox.question(
            self,
            "Delete Connection",
            f"Are you sure you want to delete '{name}'?\n\nThis will trigger a graph rebuild.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.tool_stack.setCurrentIndex(1)
            self.graph_editor.delete_connection_by_data(conn_data)
            self.status_bar.show_success(f"Connection '{name}' deleted.")
            self.update_visualizations()

    def _on_overview_node_selected(self, graph_id, node_id):
        
        self.tool_stack.setCurrentIndex(1)
        self.graph_editor.select_node_by_id(graph_id, node_id)
        
        if hasattr(self, 'flow_widget'):
            self.flow_widget.set_target_node(graph_id, node_id)
    def open_connection_tool_for_node(self, graph_id, node_id, pop_id=0):

        print(f"Context Menu Action: Setting Source to Graph {graph_id}, Node {node_id}, Pop {pop_id}")
        
        self.tool_stack.setCurrentIndex(2)
        
        self.connection_tool.refresh()
        
        self.connection_tool.set_source(graph_id, node_id, pop_id)
        
        self.status_bar.set_status(f"Source preset: G{graph_id} N{node_id} P{pop_id}", color="#2196F3")


    def _on_overview_pop_selected(self, graph_id, node_id, pop_id):
        self.tool_stack.setCurrentIndex(1)
        self.graph_editor.select_population_by_ids(graph_id, node_id, pop_id)


    def _on_overview_conn_selected(self, connection_data):
        self.tool_stack.setCurrentIndex(1)
        self.graph_editor.load_connection_editor(connection_data)


    
    def on_structure_selected(self, name, models, probs):
        
        self.tool_stack.setCurrentIndex(0)
        
        self.graph_builder.load_structure_preset(name, models, probs, grid_size=[10, 10, 10])
        
        self.status_bar.show_success(f"Preset '{name}' loaded into Graph Creator.")
        print(f"Loaded Structure: {name} with {len(models)} populations.")
    


    def closeEvent(self, event):
        try:
            vtk.vtkObject.GlobalWarningDisplayOff()
            
            if hasattr(self, 'simulation_view') and self.simulation_view:
                self.simulation_view.sim_running = False
                if hasattr(self.simulation_view, 'timer'):
                    self.simulation_view.timer.stop()
            
            if hasattr(self, 'blink_widget') and self.blink_widget:
                self.blink_widget.is_active = False
                self.blink_widget.simulation_running = False
                if hasattr(self.blink_widget, 'timer'):
                    self.blink_widget.timer.stop()
            
            if hasattr(self, 'vis_stack'):
                for i in range(self.vis_stack.count()):
                    widget = self.vis_stack.widget(i)
                    if widget:
                        if hasattr(widget, 'timer'):
                            widget.timer.stop()
                        if hasattr(widget, 'plotter'):
                            try:
                                widget.plotter.close()
                            except:
                                pass

            for plotter_name in ['neuron_plotter', 'graph_plotter']:
                if hasattr(self, plotter_name):
                    plotter = getattr(self, plotter_name)
                    if plotter:
                        try:
                            plotter.close()
                        except:
                            pass
            
            if hasattr(self, 'blink_widget') and self.blink_widget:
                if hasattr(self.blink_widget, 'plotter'):
                    try:
                        self.blink_widget.plotter.close()
                    except:
                        pass
                        
        except Exception as e:
            print(f"Cleanup error (harmless on exit): {e}")
        
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(50, self._update_plotters_after_resize)
    
    def _update_plotters_after_resize(self):
        try:
            for plotter_name in ['neuron_plotter', 'graph_plotter']:
                if hasattr(self, plotter_name):
                    plotter = getattr(self, plotter_name)
                    if plotter and hasattr(plotter, 'render_window') and plotter.render_window:
                        plotter.update()
            
            if hasattr(self, 'blink_widget') and self.blink_widget:
                if hasattr(self.blink_widget, 'plotter'):
                    if self.blink_widget.plotter and self.blink_widget.plotter.render_window:
                        self.blink_widget.plotter.update()
        except:
            pass

    def create_bottom_right(self):
        main_container = QWidget()
        main_container.setStyleSheet("background-color: #232323; border-left: 1px solid #444;")
        
        main_layout = QHBoxLayout(main_container)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        base_btn_style = """
            QPushButton {
                background-color: #2b2b2b;
                color: #B0BEC5;
                border: 1px solid #37474F;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
                text-align: center;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #37474F;
                color: white;
            }
            QPushButton:pressed {
                background-color: #102027;
            }
        """

        active_btn_style = """
            QPushButton {
                background-color: #1c242b;
                color: #00E5FF;
                border: 2px solid #00E5FF; /* Neon Rand */
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
                text-align: center;
                margin: 2px;
            }
        """

        action_btn_style = """
            QPushButton {
                background-color: #3E2723; color: #D7CCC8;
                border: 1px solid #5D4037; border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background-color: #4E342E; border: 1px solid #8D6E63; color: white; }
        """
        
        io_btn_style = """
            QPushButton {
                background-color: #263238; color: #ECEFF1;
                border: 1px solid #455A64; border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background-color: #37474F; border: 1px solid #607D8B; color: white; }
        """

        def create_header(text):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #757575; font-weight: bold; font-size: 11px; letter-spacing: 2px; margin-bottom: 5px;")
            lbl.setFixedHeight(20)
            return lbl

        col1_layout = QVBoxLayout()
        col1_layout.setSpacing(5)
        col1_layout.addWidget(create_header("NAVIGATION"))
        
        self.nav_btns = []
        
        nav_items = [
            ("Network Builder", 0),
            ("Graph Inspector", 1),
            ("Connectivity", 2),
            ("Population Examples", 3),
            ("Instrumentation", 4)
        ]

        def on_nav_click(target_idx, clicked_btn):
            self.tool_stack.setCurrentIndex(target_idx)
            
            for btn in self.nav_btns:
                if btn == clicked_btn:
                    btn.setStyleSheet(active_btn_style)
                else:
                    btn.setStyleSheet(base_btn_style)

        for label, idx in nav_items:
            btn = QPushButton(label)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            
            btn.setStyleSheet(base_btn_style)
            
            btn.clicked.connect(lambda checked, i=idx, b=btn: on_nav_click(i, b))
            
            self.nav_btns.append(btn)
            col1_layout.addWidget(btn)

        if self.nav_btns:
            self.nav_btns[0].setStyleSheet(active_btn_style)

        col2_layout = QVBoxLayout()
        col2_layout.setSpacing(5)
        col2_layout.addWidget(create_header("KERNEL"))
        
        ops_items = [
            ("Import Sub-Graph", self.merge_graphs_dialog),
            ("Refresh Connectivity", self.reconnect_network),
            ("Reinstantiate Network", self.rebuild_all_graphs)
        ]
        
        for label, func in ops_items:
            btn = QPushButton(label)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            btn.setStyleSheet(action_btn_style)
            btn.clicked.connect(func)
            col2_layout.addWidget(btn)

        col3_layout = QVBoxLayout()
        col3_layout.setSpacing(5)
        col3_layout.addWidget(create_header("PROJECT"))
        
        io_items = [
            ("Save Project", self.save_all_graphs_dialog),
            ("Load Project", self.load_all_graphs_dialog)
        ]
        
        for label, func in io_items:
            btn = QPushButton(label)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            btn.setStyleSheet(io_btn_style)
            btn.clicked.connect(func)
            col3_layout.addWidget(btn)

        main_layout.addLayout(col1_layout, 4)
        main_layout.addLayout(col2_layout, 3)
        main_layout.addLayout(col3_layout, 2)

        wrapper_layout = QHBoxLayout()
        wrapper_layout.setContentsMargins(0,0,0,0)
        wrapper_layout.addWidget(main_container)
        
        return wrapper_layout
    def delete_device_wrapper(self, device_data):
        from PyQt6.QtWidgets import QMessageBox
        
        model = device_data.get('model', 'Device')
        dev_id = device_data.get('id')
        
        reply = QMessageBox.question(
            self,
            "Delete Device",
            f"Really delete {model} (ID: {dev_id})?\n\nThis will reset the kernel.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.status_bar.set_status("Deleting device & Resetting...", "#FF5722")
            QApplication.processEvents()
            
            target = device_data.get('target', {})
            gid = target.get('graph_id')
            nid = target.get('node_id')
            
            target_graph = next((g for g in graph_list if g.graph_id == gid), None)
            if target_graph:
                node = target_graph.get_node(nid)
                if node:
                    if hasattr(node, 'devices'):
                        node.devices = [d for d in node.devices if str(d.get('id')) != str(dev_id)]
                    
                    if hasattr(node, 'parameters') and 'devices' in node.parameters:
                        node.parameters['devices'] = [d for d in node.parameters['devices'] if str(d.get('id')) != str(dev_id)]
            
            self.manual_nest_reset()
            
            self.graph_overview.update_tree()
            self.status_bar.show_success(f"Device {dev_id} deleted.")

    def reconnect_network(self):

        if not graph_list:
            self.status_bar.show_error("No graphs to reconnect!")
            return

        reply = QMessageBox.question(
            self,
            'Reconnect Network',
            'Reset NEST kernel and restore all connections?\n\n'
            '• Positions: KEPT (No WFC)\n'
            '• Neurons: Re-created\n'
            '• Connections: Re-connected',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return


        self.status_bar.set_status("Resetting NEST & Reconnecting...", color="#00BCD4")
        self.status_bar.set_progress(10)
        QApplication.processEvents()

        try:
            graphs_dict = {g.graph_id: g for g in graph_list}

            stats = safe_nest_reset_and_repopulate(
                graphs_dict,
                enable_structural_plasticity=self.structural_plasticity_enabled,
                verbose=True
            )

            self.status_bar.set_status("Refreshing Views...", color="#00BCD4")
            self.status_bar.set_progress(90)
            QApplication.processEvents()

            self.update_visualizations()
            self.connection_tool.refresh()
            
            if hasattr(self, 'simulation_view'):
                self.simulation_view.restore_injectors()

            if hasattr(self, 'blink_widget'):
                self.blink_widget.build_scene()

            msg = (f"Reconnected! Created {stats['populations_created']} populations "
                   f"and {stats['connections_created']} connections.")
            
            if stats['connections_failed'] > 0:
                msg += f" (WARNING: {stats['connections_failed']} connections failed!)"
                self.status_bar.set_status(f"⚠ {msg}", color="#FF9800")
                QMessageBox.warning(self, "Reconnect Warning", msg)
            else:
                self.status_bar.show_success(msg)
            
            print(f"{msg}")

        except Exception as e:
            self.status_bar.show_error(f"Reconnect failed: {e}")
            print(f"Reconnect failed: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Reconnect Error", str(e))
    
    def on_graph_created(self, graph_id):
        try:
            self.status_bar.set_status("Creating graph...", color="#1976D2")
            self.status_bar.set_progress(0, maximum=100)
            QApplication.processEvents()
        
            graph = create_graph_from_widget(graph_id)
            self.active_graphs[graph_id] = graph
            self.status_bar.set_progress(30)
            QApplication.processEvents()

            total_nodes = len(graph.node_list)
            for i, node in enumerate(graph.node_list):
                self.status_bar.set_status(f"Building node {i+1}/{total_nodes}...", color="#1976D2")
                self.status_bar.set_progress(30 + int(30 * (i+1) / total_nodes))
                QApplication.processEvents()
                
                if not hasattr(node, 'positions') or node.positions is None:
                    node.build()
                node.populate_node()

            self.status_bar.set_status("Updating visualizations...", color="#1976D2")
            self.status_bar.set_progress(80)
            QApplication.processEvents()
            
            self.update_visualizations()
            self.graph_overview.update_tree()
            
            self.connection_tool.refresh()
            
            self.graph_editor.refresh_graph_list()
            if hasattr(self, 'tools_widget'):
                self.tools_widget.update_graphs(graph_list)

            self.status_bar.show_success(f"Graph '{graph.graph_name}' created!")
            QApplication.processEvents()
            
        except Exception as e:
            self.status_bar.show_error(f"Failed to create graph: {e}")
            print(f"Error creating graph: {e}")
            import traceback
            traceback.print_exc()
    def on_visibility_changed(self, data, visible):
        dtype = data.get('type')
        
        if dtype == 'node':
            key = (data['graph_id'], data['node_id'])
            if visible:
                if key in self.hidden_nodes: self.hidden_nodes.remove(key)
            else:
                self.hidden_nodes.add(key)
                
        elif dtype == 'graph':
            gid = data['graph_id']
            if visible:
                if gid in self.hidden_graphs: self.hidden_graphs.remove(gid)
            else:
                self.hidden_graphs.add(gid)
        
        self.plot_neuron_points()
        self.plot_graph_skeleton()


    def plot_neuron_points(self):
        self.neuron_plotter.clear()
        legend_entries = []
        used_types = set()
        
        for graph in graph_list:
            if graph.graph_id in self.hidden_graphs:
                continue
                
            for node in graph.node_list:
                if (graph.graph_id, node.id) in self.hidden_nodes:
                    continue
                
                for i, pts in enumerate(node.positions):
                    if pts is None or len(pts) == 0: continue
                    neuron_type = node.neuron_models[i] if i < len(node.neuron_models) else "unknown"
                    color = get_neuron_color(neuron_type)
                    
                    if neuron_type not in used_types:
                        legend_entries.append([neuron_type, color])
                        used_types.add(neuron_type)
                        
                    point_cloud = pv.PolyData(pts)
                    self.neuron_plotter.add_mesh(point_cloud, color=color, point_size=10)
                    
        if legend_entries:
            self.neuron_plotter.add_legend(legend_entries, size=(0.12, 0.12), loc='upper right')
        self.neuron_plotter.update()
        
   


    def plot_graph_skeleton(self):
        self.graph_plotter.clear()
        
        self.highlighted_actor = None
        self.original_color = None
        self.original_opacity = None
        
        self.skeleton_info_map = {}
        
        self.tooltip_actor = vtk.vtkTextActor()
        self.tooltip_actor.GetTextProperty().SetFontSize(16)
        self.tooltip_actor.GetTextProperty().SetColor(1.0, 1.0, 0.0)
        self.tooltip_actor.GetTextProperty().SetFontFamilyToArial()
        self.tooltip_actor.GetTextProperty().BoldOn()
        self.tooltip_actor.GetTextProperty().ShadowOn()
        self.tooltip_actor.SetVisibility(False)
        self.graph_plotter.renderer.AddViewProp(self.tooltip_actor)
        
        node_map = {}
        for graph in graph_list:
            for node in graph.node_list:
                node_map[(graph.graph_id, node.id)] = node

        import random

        for graph in graph_list:
            cmap = plt.get_cmap("tab20")
            rgba = cmap(graph.graph_id % 20)
            graph_color = rgba[:3]
            
            for node in graph.node_list:
                center = np.array(node.center_of_mass)
                
                sphere = pv.Sphere(radius=0.5, center=center)
                node_actor = self.graph_plotter.add_mesh(
                    sphere,
                    color=graph_color,
                    opacity=0.8,
                    smooth_shading=True,
                    pickable=True
                )
                
                n_pops = len(node.population) if hasattr(node, 'population') and node.population else 0
                info_text = f"NODE: {node.name} (ID: {node.id})\nGraph: {graph.graph_id}\nPops: {n_pops}"
                self.skeleton_info_map[node_actor] = info_text
                
                has_devices = False
                if hasattr(node, 'devices') and node.devices:
                    has_devices = True
                elif hasattr(node, 'parameters') and node.parameters.get('devices'):
                    has_devices = True
                
                if has_devices:
                    aura = pv.Sphere(radius=0.9, center=center)
                    self.graph_plotter.add_mesh(
                        aura,
                        color="#FF9800",
                        opacity=0.15,
                        pickable=False,
                        smooth_shading=True
                    )

                if hasattr(node, 'connections') and node.connections:
                    for conn in node.connections:
                        try:
                            target = conn.get('target', {})
                            tgt_gid = int(target.get('graph_id'))
                            tgt_nid = int(target.get('node_id'))
                            
                            target_node = node_map.get((tgt_gid, tgt_nid))
                            
                            if target_node:
                                conn_name = conn.get('name', 'Connection')
                                params = conn.get('params', {})
                                weight = float(params.get('weight', 0.0))
                                
                                if weight > 0:
                                    edge_color = "#FF3333"
                                elif weight < 0:
                                    edge_color = "#3366FF"
                                else:
                                    edge_color = "white"
                                
                                start = np.array(node.center_of_mass)
                                end = np.array(target_node.center_of_mass)
                                
                                conn_info_text = (f"CONN: {conn_name}\n"
                                                  f"{node.name} -> {target_node.name}\n"
                                                  f"Weight: {weight}")

                                if node == target_node:
                                    torus = pv.ParametricTorus(ringradius=1.2, crosssectionradius=0.08)
                                    torus.rotate_x(45)
                                    torus.translate(start)
                                    
                                    actor_self = self.graph_plotter.add_mesh(
                                        torus, color=edge_color, opacity=0.9, pickable=True
                                    )
                                    self.skeleton_info_map[actor_self] = conn_info_text + " (Self)"
                                
                                else:
                                    mid_point = (start + end) / 2.0
                                    
                                    seed = node.id * 100 + target_node.id
                                    random.seed(seed)
                                    
                                    offset = np.array([
                                        random.uniform(-3, 3),
                                        random.uniform(-3, 3),
                                        random.uniform(1, 4)
                                    ])
                                    control_point = mid_point + offset
                                    
                                    points = np.array([start, control_point, end])
                                    spline = pv.Spline(points, n_points=20)
                                    
                                    actor_line = self.graph_plotter.add_mesh(
                                        spline,
                                        color=edge_color,
                                        line_width=2.0,
                                        render_lines_as_tubes=True,
                                        pickable=False
                                    )
                                    
                                    hitbox = spline.tube(radius=0.4)
                                    actor_hitbox = self.graph_plotter.add_mesh(
                                        hitbox,
                                        color=edge_color,
                                        opacity=0.0,
                                        pickable=True
                                    )
                                    self.skeleton_info_map[actor_hitbox] = conn_info_text
                                    
                                    last_pt = spline.points[-1]
                                    prev_pt = spline.points[-3]
                                    direction = last_pt - prev_pt
                                    norm = np.linalg.norm(direction)
                                    if norm > 0: direction /= norm
                                    
                                    cone_pos = end - (direction * 0.5)
                                    
                                    arrow_head = pv.Cone(
                                        center=cone_pos,
                                        direction=direction,
                                        height=0.6,
                                        radius=0.25,
                                        resolution=12
                                    )
                                    
                                    actor_head = self.graph_plotter.add_mesh(
                                        arrow_head,
                                        color=edge_color,
                                        pickable=False
                                    )

                        except Exception as e:
                            print(f"Error plotting connection skeleton: {e}")

        try:
            self.graph_plotter.iren.remove_observer(self._observer_tag)
        except:
            pass
        self._observer_tag = self.graph_plotter.iren.add_observer(
            "MouseMoveEvent", self._on_skeleton_hover
        )
        
        self.graph_plotter.reset_camera()
        self.graph_plotter.update()


    def update_simulation_speed(self, value):
        
        delay = int((100 - value) * 2)
        
        if hasattr(self, 'sim_timer'):
            self.sim_timer.setInterval(delay)
    def update_simulation_target(self, new_duration):
        if hasattr(self, 'sim_timer') and self.sim_timer.isActive():
            self.sim_target_time = self.current_nest_time + new_duration
            print(f"  -> Target updated to {self.sim_target_time}ms")
        else:
            self.sim_target_time = new_duration
    def _on_skeleton_hover(self, interactor, event):

        x, y = interactor.GetEventPosition()
        
        picker = vtk.vtkPropPicker()
        picker.Pick(x, y, 0, self.graph_plotter.renderer)
        actor = picker.GetActor()
        
        if self.highlighted_actor and self.highlighted_actor != actor:
            try:
                prop = self.highlighted_actor.GetProperty()
                prop.SetColor(self.original_color)
                prop.SetOpacity(self.original_opacity)
            except AttributeError:
                pass
            self.highlighted_actor = None
            self.tooltip_actor.SetVisibility(False)

        if actor and actor in self.skeleton_info_map:
            if self.highlighted_actor != actor:
                self.highlighted_actor = actor
                prop = actor.GetProperty()
                
                self.original_color = prop.GetColor()
                self.original_opacity = prop.GetOpacity()
                
                prop.SetColor(1.0, 1.0, 1.0)
                prop.SetOpacity(1.0)
                prop.SetLineWidth(3.0)
                
                text_content = self.skeleton_info_map[actor]
                self.tooltip_actor.SetInput(text_content)
                self.tooltip_actor.SetVisibility(True)
            
            self.tooltip_actor.SetPosition(x + 15, y + 10)
            
        else:
            self.tooltip_actor.SetVisibility(False)
        
        interactor.Render()


    def open_live_spectator(self):

        self._switch_main_view(1)


    def run_nest_simulation(self, duration):
        print(f"\n>>> STARTING LIVE SIMULATION (Duration: {duration} ms) <<<")
        self.status_bar.set_status("Simulation running...", color="#FF9800")
        
        self._ensure_spike_recorders()
        
        step_size = 25.0
        current_time = 0.0
        
        self._switch_main_view(1)
        
        try:
            while current_time < duration:
                nest.Simulate(step_size)
                current_time += step_size
                
                
                QApplication.processEvents()
                
                prog = int((current_time / duration) * 100)
                self.status_bar.set_progress(prog)
                
                if hasattr(self.simulation_view, 'is_paused') and self.simulation_view.is_paused:
                    break

            self.collect_simulation_results(duration)
            self.status_bar.show_success("Simulation finished.")
            self.sim_dashboard.btn_results.setEnabled(True)
            
        except Exception as e:
            self.status_bar.show_error(f"Simulation Error: {e}")
            print(f"Error: {e}")
            import traceback; traceback.print_exc()

    def _ensure_spike_recorders(self):
        self.live_recorders = []
        
        non_spiking = [
            'siegert_neuron', 'mcculloch_pitts_neuron',
            'rate_neuron_ipn', 'rate_neuron_opn', 'gif_pop_psc_exp',
            'ht_neuron'
        ]
        
        for graph in graph_list:
            for node in graph.node_list:
                if hasattr(node, 'population'):
                    for pop in node.population:
                        if pop is None: continue
                        
                        try:
                            model = nest.GetStatus(pop, 'model')[0]
                            if model in non_spiking:
                                continue
                                
                            rec = nest.Create("spike_recorder")
                            
                            nest.SetStatus(rec, {"record_to": "memory"})
                            
                            nest.Connect(pop, rec)
                            
                            self.live_recorders.append(rec)
                            
                        except Exception as e:
                            print(f"Warning creating live recorder: {e}")

    def init_simulation_timer(self):
        self.sim_timer = QTimer()
        self.sim_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.sim_timer.timeout.connect(self.on_sim_timer_timeout)
        
        self.sim_mode = 'continuous'
        self.sim_target_time = 0.0
        self.current_nest_time = 0.0
        
        self.last_ui_update_realtime = 0.0
        
        if hasattr(self, 'simulation_view'):
            self.simulation_view.sigSpeedChanged.connect(self.update_simulation_speed)
        

    def _global_start(self):
        step = self.global_step_spin.value()
        duration = self.global_duration_spin.value()
        self.start_continuous_simulation(step, duration)
        self._update_global_button_state('running')
    
    def _global_step(self):
        step = self.global_step_spin.value()
        self.step_simulation(step)
    
    def _global_pause(self):
        self.pause_simulation()
        self._update_global_button_state('paused')
    
    def _global_stop(self):
        self.stop_simulation()
        self._update_global_button_state('stopped')
    
    def _global_reset(self):
        self.reset_and_restart()
        self._update_global_button_state('stopped')
        self.global_time_label.setText("T: 0.0 ms")
    
    def _update_global_button_state(self, state):
        base = "font-weight: bold; border-radius: 3px; padding: 5px 10px;"
        
        if state == 'running':
            self.global_btn_start.setEnabled(False)
            self.global_btn_start.setStyleSheet(f"{base} background-color: #1B5E20; color: #666;")
            self.global_btn_pause.setEnabled(True)
            self.global_btn_pause.setStyleSheet(f"{base} background-color: #FBC02D; color: black;")
        
        elif state == 'paused' or state == 'stopped':
            self.global_btn_start.setEnabled(True)
            self.global_btn_start.setStyleSheet(f"{base} background-color: #2E7D32; color: white;")
            self.global_btn_pause.setEnabled(False)
            self.global_btn_pause.setStyleSheet(f"{base} background-color: #555; color: #888;")
    
    def update_global_time_display(self, time_ms=None):
        if not hasattr(self, 'global_time_label'): return
        
        if time_ms is None:
            try: time_ms = nest.GetKernelStatus().get('time', 0.0)
            except: time_ms = 0.0

        self.global_time_label.setText(f"{float(time_ms):.1f} ms")
        self.global_time_label.repaint()
    def update_simulation_speed(self, slider_value):
        if hasattr(self, 'sim_timer'):
            self.sim_timer.setInterval(slider_value)


    def start_continuous_simulation(self, step_size, max_duration):
        step_val = self.global_step_spin.value()
        duration_val = self.global_duration_spin.value()
        
        self.sim_mode = 'continuous'
        self.sim_step_size = step_val
        
        try:
            stat = nest.GetKernelStatus()
            self.current_nest_time = stat.get('time', 0.0)
        except:
            self.current_nest_time = 0.0
            
        print(f"Starting Continuous Run from {self.current_nest_time} ms")

        if duration_val > 0:
            self.sim_target_time = self.current_nest_time + duration_val
            print(f"  -> Target set to {self.sim_target_time:.1f}ms (+{duration_val}ms)")
        else:
            self.sim_target_time = float('inf')
            print("  -> Infinite Run (Duration = 0)")

        self.status_bar.set_status("Running...", "#2E7D32")
        self._ensure_spike_recorders()

        self.sim_timer.setSingleShot(False)
        
        if hasattr(self, 'simulation_view'):
            self.simulation_view.start_rendering()
            
        self.sim_timer.start(0)


    def step_simulation(self, step_size):
        print(f"Executing Single Step ({step_size}ms)")
        self.sim_mode = 'step'
        self.sim_step_size = step_size
        
        self.sim_timer.setSingleShot(True)
        self._ensure_spike_recorders()
        self.sim_timer.start(0)


    def pause_simulation(self):
        print("Simulation Paused")
        self.sim_timer.stop()
        self.status_bar.set_status("Paused", "#FBC02D")
        
        if hasattr(self, 'simulation_view'):
            self.simulation_view.stop_rendering_safe()
        
        self._update_global_button_state('paused')

    def stop_simulation(self):
        print("Simulation Stopped")
        self.sim_timer.stop()
        self.collect_simulation_results(0)
        self.status_bar.set_status("Stopped", "#D32F2F")
        
        if hasattr(self, 'simulation_view'):
            self.simulation_view.stop_rendering_safe()
            
        self._update_global_button_state('stopped')

    


    def on_live_weight_change(self, conn_data, new_weight):
        success = LiveConnectionController.set_weight(graph_list, conn_data, new_weight)
        
        if success:
            src_gid = conn_data['source']['graph_id']
            src_nid = conn_data['source']['node_id']
            conn_id = conn_data.get('id')
            
            target_graph = next((g for g in graph_list if g.graph_id == src_gid), None)
            if target_graph:
                node = target_graph.get_node(src_nid)
                if node and hasattr(node, 'connections'):
                    for c in node.connections:
                        if c.get('id') == conn_id:
                            c['params']['weight'] = new_weight
                            if new_weight == 0.0:
                                if "(Severed)" not in c.get('name', ''):
                                    c['name'] = f"{c.get('name','Conn')} (Severed)"
                            else:
                                c['name'] = c.get('name', 'Conn').replace(" (Severed)", "")
                            break
            
            self.graph_overview.update_tree()
            self.status_bar.show_success(f"Connection weight set to {new_weight} (Live).")
        else:
            self.status_bar.show_error("Live update failed (Simulation running?)")


    def on_sim_timer_timeout(self):
        try:
            if self.sim_mode == 'continuous' and self.sim_target_time != float('inf'):
                remaining = self.sim_target_time - self.current_nest_time
                if remaining <= 0.0001:
                    self.pause_simulation()
                    self.update_global_time_display(self.current_nest_time)
                    self.status_bar.show_success(f"Target reached ({self.current_nest_time:.1f}ms).")
                    return
                step_to_take = min(self.sim_step_size, remaining)
            else:
                step_to_take = self.sim_step_size

            nest.Simulate(step_to_take)
            
            self.current_nest_time = nest.GetKernelStatus().get('time', 0.0)
            
            self.update_global_time_display(self.current_nest_time)
            self._distribute_simulation_data()
            QApplication.processEvents()
            
        except Exception as e:
            self.sim_timer.stop()
            self.status_bar.show_error(f"Sim Error: {e}")
            print(f"Sim Error: {e}")
            import traceback; traceback.print_exc()


    def reset_and_restart(self, duration=None):
        print("\n=== RESETTING SIMULATION ===")
        if hasattr(self, 'sim_timer'):
            self.sim_timer.stop()
        
        self.current_nest_time = 0.0
        self.sim_target_time = 0.0

        
        self.current_nest_time = 0.0
        self.sim_target_time = 0.0
        self.sim_mode = 'step'
        
        try:
            self.status_bar.set_status("Resetting Kernel & Network...", "#E65100")
            QApplication.processEvents()
            
            if hasattr(self, 'simulation_view'):
                if hasattr(self.simulation_view, 'stop_rendering_safe'):
                    self.simulation_view.stop_rendering_safe()
                
                self.simulation_view.update_time_display(0.0)
                self.simulation_view.is_paused = True
                self.simulation_view.update_button_styles()

            nest.ResetKernel()
            
            if self.structural_plasticity_enabled:
                nest.EnableStructuralPlasticity()
            self.current_nest_time = 0.0
            self.update_global_time_display(0.0)
            self._update_global_button_state('stopped')
            
            print("=== RESET COMPLETE: TIME 0.0 ===")
            for graph in graph_list:
                for node in graph.node_list:
                    if not hasattr(node, 'positions') or not node.positions:
                        node.build()
                    node.populate_node()
            
            graphs_dict = {g.graph_id: g for g in graph_list}
            create_nest_connections_from_stored(graphs_dict, verbose=False)
            
            if self.main_stack.currentIndex() == 1 and hasattr(self, 'simulation_view'):
                self.simulation_view.load_scene()
                self.simulation_view.restore_injectors()
            
            self.live_recorders = []
            self._ensure_spike_recorders()
            
            if hasattr(self, 'live_dashboard'):
                self.live_dashboard.clear_all_data()
                
                QTimer.singleShot(200, self.live_dashboard.scan_for_devices)
                
            self.update_global_time_display(0.0)
            self._update_global_button_state('stopped')
            print("=== RESET COMPLETE: TIME 0.0 ===")
            
        except Exception as e:
            self.status_bar.show_error(f"Reset Failed: {e}")
            print(f"Reset Error: {e}")
            import traceback; traceback.print_exc()
 


    def _distribute_simulation_data(self):
        try:
            k_stat = nest.GetKernelStatus()
            sim_time = k_stat.get('time', k_stat.get('biological_time', 0.0))
        except:
            sim_time = 0.0

        if hasattr(self, 'simulation_view') and self.simulation_view.isVisible():
            if hasattr(self, 'live_recorders') and self.live_recorders:
                visual_spikes_flat = []
                
                for rec in self.live_recorders:
                    try:
                        st = nest.GetStatus(rec)[0]
                        
                        if st.get('n_events', 0) > 0:
                            events = st.get('events', {})
                            if 'senders' in events:
                                senders = events['senders']
                                if hasattr(senders, 'tolist'):
                                    visual_spikes_flat.extend(senders.tolist())
                                else:
                                    visual_spikes_flat.extend(senders)
                            
                            nest.SetStatus(rec, {'n_events': 0})
                    except Exception:
                        pass

                if visual_spikes_flat:
                    self.simulation_view.feed_spikes(visual_spikes_flat)

        
        live_data_snapshot = {}
        has_dashboard_data = False
        
        graphs = getattr(self, 'active_graphs', {}).values()
        if not graphs: graphs = graph_list

        for graph in graphs:
            for node in graph.node_list:
                if not hasattr(node, 'devices') or not node.devices:
                    continue
                
                if "history" not in node.results:
                    node.results["history"] = []
                
                step_record = {
                    "time": sim_time,
                    "devices": {}
                }
                data_in_step = False

                for dev in node.devices:
                    gid = dev.get('runtime_gid')
                    model = dev.get('model', '')
                    dev_id = dev.get('id')
                    
                    if gid is None: continue
                    
                    if "recorder" in model or "meter" in model:
                        try:
                            nest_handle = gid
                            dict_key = None
                            
                            if hasattr(gid, 'tolist'):
                                vals = gid.tolist()
                                dict_key = vals[0] if vals else None
                            elif isinstance(gid, (list, tuple)):
                                dict_key = gid[0] if gid else None
                            else:
                                dict_key = gid
                                nest_handle = [gid]
                                
                            if dict_key is None: continue

                            status = nest.GetStatus(nest_handle)[0]
                            if status.get('n_events', 0) > 0:
                                events = status.get('events', {})
                                
                                clean_events = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in events.items()}
                                
                                step_record["devices"][str(dev_id)] = {
                                    "type": model,
                                    "events": clean_events
                                }
                                data_in_step = True
                                
                                live_data_snapshot[dict_key] = events
                                has_dashboard_data = True
                                
                                nest.SetStatus(nest_handle, {'n_events': 0})
                                
                        except Exception:
                            pass
                
                if data_in_step:
                    node.results["history"].append(step_record)

        if has_dashboard_data and hasattr(self, 'live_dashboard'):
            self.live_dashboard.process_incoming_data(live_data_snapshot, sim_time)


    def rebuild_all_graphs(
            self,
            target_graphs=None,
            reset_nest: bool = True,
            rebuild_positions: bool = False,
            enable_structural_plasticity: bool = False,
            status_callback=None,
            verbose: bool = True
        ) -> Dict[str, Any]:
            
            graphs_to_process = graph_list
            
            if isinstance(target_graphs, list):
                graphs_to_process = target_graphs
            
            stats = {
                'graphs_rebuilt': 0,
                'nodes_rebuilt': 0,
                'populations_created': 0,
                'errors': []
            }
            
            if not graphs_to_process:
                if verbose:
                    print("No graphs to rebuild")
                return stats
            
            if verbose:

                print("REBUILD ALL GRAPHS")

            
            if reset_nest:
                if verbose:
                    print("\nResetting NEST kernel...")
                
                if hasattr(self, 'status_bar'):
                    self.status_bar.set_status("Resetting NEST kernel...", color="#FF5722")
                    self.status_bar.set_progress(5)
                    QApplication.processEvents()

                nest.ResetKernel()
                
                if enable_structural_plasticity:
                    try:
                        nest.EnableStructuralPlasticity()
                        if verbose:
                            print("✓ Structural plasticity enabled")
                    except Exception as e:
                        if verbose:
                            print(f"Could not enable structural plasticity: {e}")
            
            total_graphs = len(graphs_to_process)
            
            for g_idx, graph in enumerate(graphs_to_process):
                graph_name = getattr(graph, 'graph_name', f'Graph_{graph.graph_id}')
                
                if verbose:
                    print(f"\n[{g_idx+1}/{total_graphs}] Rebuilding '{graph_name}' (ID: {graph.graph_id})")
                
                if hasattr(self, 'status_bar'):
                    progress = 10 + int(80 * g_idx / total_graphs)
                    self.status_bar.set_status(f"Rebuilding {graph_name}...", color="#1976D2")
                    self.status_bar.set_progress(progress)
                    QApplication.processEvents()
                
                for n_idx, node in enumerate(graph.node_list):
                    node_name = getattr(node, 'name', f'Node_{node.id}')
                    
                    try:
                        if rebuild_positions:
                            if verbose:
                                print(f"Building Node {node.id}: {node_name}...")
                            node.build()
                        else:
                            if not hasattr(node, 'positions') or not node.positions:
                                node.build()
                            elif all(len(p) == 0 for p in node.positions if p is not None):
                                node.build()
                        
                        if verbose:
                            print(f"Populating Node {node.id}: {node_name}...")
                        node.populate_node()
                        
                        if hasattr(node, 'population') and node.population:
                            n_pops = len([p for p in node.population if p is not None and len(p) > 0])
                            stats['populations_created'] += n_pops
                        
                        stats['nodes_rebuilt'] += 1
                        
                    except Exception as e:
                        error_msg = f"Node {node.id} ({node_name}): {str(e)}"
                        stats['errors'].append(error_msg)
                        if verbose:
                            print(f" Error: {e}")
                
                stats['graphs_rebuilt'] += 1
            
            if verbose:
                print(f"   Rebuild complete!")
                print(f"   Graphs: {stats['graphs_rebuilt']}")
                print(f"   Nodes: {stats['nodes_rebuilt']}")
                print(f"   Populations: {stats['populations_created']}")
            
            if hasattr(self, 'status_bar'):
                self.status_bar.show_success(f"Rebuild complete! ({stats['populations_created']} pops created)")
            
            self.update_visualizations()
            self.update_visualizations()
        
            if hasattr(self, 'simulation_view'):
                self.simulation_view.restore_injectors()
            return stats
    
    def start_simulation(self):
        print("Simulation started (Not implemented yet)")
        self.status_bar.set_status("Simulation running...", color="#4CAF50")

    def stop_simulation(self):
        print("Simulation stopped")
        self.status_bar.set_status("Simulation stopped", color="#F44336")

    def save_all_graphs_dialog(self):
        if not graph_list:
            QMessageBox.warning(self, "Save Error", "No graphs to save!")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "JSON Files (*.json);;All Files (*)"
        )

        if filepath:
            if not filepath.endswith('.json'):
                filepath += '.json'
            
            try:
                project_data = {
                    'meta': {
                        'version': '2.0',
                        'type': 'neuroticks_project',
                        'timestamp': str(np.datetime64('now'))
                    },
                    'graphs': []
                }

                for graph in graph_list:
                    nodes_data = []
                    for node in graph.node_list:
                        
                        cleaned_devices = []
                        if hasattr(node, 'devices'):
                            for dev in node.devices:
                                dev_copy = dev.copy()
                                if 'runtime_gid' in dev_copy:
                                    del dev_copy['runtime_gid']
                                if 'params' in dev_copy:
                                    dev_copy['params'] = _clean_params(dev_copy['params'])
                                cleaned_devices.append(dev_copy)

                        safe_params = node.parameters.copy() if hasattr(node, 'parameters') else {}
                        if 'devices' in safe_params:
                            del safe_params['devices']

                        node_data = {
                            'id': node.id,
                            'name': node.name,
                            'graph_id': graph.graph_id,
                            'parameters': _clean_params(safe_params),
                            'positions': [pos.tolist() if isinstance(pos, np.ndarray) else list(pos)
                                          for pos in node.positions] if node.positions else [],
                            'center_of_mass': list(node.center_of_mass),
                            'connections': _serialize_connections(node.connections),
                            'devices': cleaned_devices
                        }
                        nodes_data.append(node_data)

                    graph_data = {
                        'graph_id': graph.graph_id,
                        'graph_name': getattr(graph, 'graph_name', f'Graph_{graph.graph_id}'),
                        'max_nodes': graph.max_nodes,
                        'init_position': list(graph.init_position),
                        'polynom_max_power': graph.polynom_max_power,
                        'nodes': nodes_data
                    }
                    project_data['graphs'].append(graph_data)

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(project_data, f, cls=NumpyEncoder, indent=2)
                
                self.status_bar.show_success(f"Project saved to {filepath}")
                print(f"Project saved: {len(graph_list)} graphs.")

            except Exception as e:
                self.status_bar.show_error(f"Save failed: {e}")
                print(f"Save failed: {e}")
                import traceback
                traceback.print_exc()

    def load_all_graphs_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "JSON Files (*.json);;All Files (*)"
        )

        if not filepath:
            return

        try:
            self.status_bar.set_status("Loading project...", color="#2196F3")
            self.status_bar.set_progress(10)
            QApplication.processEvents()

            with open(filepath, 'r', encoding='utf-8') as f:
                project_data = json.load(f)

            if 'graphs' not in project_data:
                raise ValueError("Invalid file format: 'graphs' key missing")

            print("\nResetting Environment for Load...")
            nest.ResetKernel()
            
            global graph_list
            graph_list.clear()
            
            WidgetLib.next_graph_id = 0
            WidgetLib.graph_parameters.clear()

            total_graphs = len(project_data['graphs'])
            
            for i, g_data in enumerate(project_data['graphs']):
                self.status_bar.set_status(f"Building Graph {i+1}/{total_graphs}...")
                self.status_bar.set_progress(20 + int(40 * (i/total_graphs)))
                QApplication.processEvents()

                gid = g_data['graph_id']
                
                if gid >= WidgetLib.next_graph_id:
                    WidgetLib.next_graph_id = gid + 1

                graph = Graph(
                    graph_name=g_data.get('graph_name', 'LoadedGraph'),
                    graph_id=gid,
                    parameter_list=[],
                    polynom_max_power=g_data.get('polynom_max_power', 5),
                    position=g_data.get('init_position', [0,0,0]),
                    max_nodes=g_data.get('max_nodes', 100)
                )
                
                for nd in g_data['nodes']:
                    params = nd['parameters'].copy()
                    params['id'] = nd['id']
                    params['name'] = nd['name']
                    params['graph_id'] = nd['graph_id']
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
                    
                    new_node.populate_node()

                graph_list.append(graph)
                print(f"Graph '{graph.graph_name}' loaded (ID: {gid}). Next ID set to: {WidgetLib.next_graph_id}")


            if hasattr(self, 'tools_widget'):
                self.tools_widget.update_graphs(graph_list)
                
            self.status_bar.set_status("Recreating Connections...")
            self.status_bar.set_progress(80)
            QApplication.processEvents()

            graphs_dict = {g.graph_id: g for g in graph_list}
            
            created, failed, _ = create_nest_connections_from_stored(graphs_dict, verbose=True)

            self.status_bar.set_status("Refreshing View...")
            self.status_bar.set_progress(95)
            QApplication.processEvents()

            self.update_visualizations()
            self.graph_overview.update_tree()
            self.connection_tool.refresh()
            self.graph_editor.refresh_graph_list()
            self.update_visualizations()
            self.graph_overview.update_tree()
            self.connection_tool.refresh()
            self.graph_editor.refresh_graph_list()

            if hasattr(self, 'graph_builder'):
                self.graph_builder.reset()

            msg = f"Loaded {total_graphs} graphs. Connections: {created} created, {failed} failed."
            self.status_bar.show_success("Project loaded successfully!")
            print(f"\n{msg}")
            QMessageBox.information(self, "Load Complete", msg)

        except Exception as e:
            self.status_bar.show_error(f"Load failed: {e}")
            print(f"Load failed: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", str(e))


    def merge_graphs_dialog(self):
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Merge Graphs (Add to existing)", "",
            "JSON Files (*.json);;All Files (*)"
        )

        if not filepath:
            return

        try:
            self.status_bar.set_status("Loading graphs to merge...", color="#9C27B0")
            self.status_bar.set_progress(5)
            QApplication.processEvents()

            with open(filepath, 'r', encoding='utf-8') as f:
                project_data = json.load(f)

            graphs_to_load = []
            
            if 'graphs' in project_data:
                graphs_to_load = project_data['graphs']
                print(f"Found project with {len(graphs_to_load)} graph(s)")
            elif 'graph' in project_data:
                single_graph = project_data['graph'].copy()
                single_graph['nodes'] = project_data.get('nodes', [])
                graphs_to_load = [single_graph]
                print("Found single graph format")
            else:
                raise ValueError("Invalid file format: no 'graph' or 'graphs' key")

            if not graphs_to_load:
                QMessageBox.warning(self, "Empty File", "No graphs found in file!")
                return

            if graph_list:
                max_existing_id = max(g.graph_id for g in graph_list)
                id_offset = max_existing_id + 1
            else:
                id_offset = 0
            
            print(f"MERGE GRAPHS")
            print(f"   Existing graphs: {len(graph_list)}")
            print(f"   Graphs to merge: {len(graphs_to_load)}")
            print(f"   ID offset: {id_offset}")

            self.status_bar.set_status(f"Merging {len(graphs_to_load)} graph(s)...", color="#9C27B0")
            self.status_bar.set_progress(15)
            QApplication.processEvents()


            id_mapping = {}
            for g_data in graphs_to_load:
                old_id = g_data.get('graph_id', 0)
                new_id = old_id + id_offset
                id_mapping[old_id] = new_id
            
            print(f"\nID Mapping:")
            for old, new in id_mapping.items():
                print(f"   Graph {old} → Graph {new}")

            total_graphs = len(graphs_to_load)
            merged_graphs = []
            total_nodes = 0
            total_connections = 0

            for i, g_data in enumerate(graphs_to_load):
                self.status_bar.set_status(f"Building Graph {i+1}/{total_graphs}...")
                self.status_bar.set_progress(20 + int(50 * (i / total_graphs)))
                QApplication.processEvents()

                old_graph_id = g_data.get('graph_id', i)
                new_graph_id = id_mapping[old_graph_id]
                
                graph_name = g_data.get('graph_name', f'MergedGraph_{new_graph_id}')
                print(f"\n[{i+1}/{total_graphs}] {graph_name}")
                print(f"   Old ID: {old_graph_id} → New ID: {new_graph_id}")

                graph = Graph(
                    graph_name=graph_name,
                    graph_id=new_graph_id,
                    parameter_list=[],
                    polynom_max_power=g_data.get('polynom_max_power', 5),
                    position=g_data.get('init_position', [0, 0, 0]),
                    max_nodes=g_data.get('max_nodes', 100)
                )

                nodes_data = g_data.get('nodes', [])
                nodes_data = sorted(nodes_data, key=lambda x: x.get('id', 0))
                
                for nd in nodes_data:
                    params = nd.get('parameters', {}).copy()
                    
                    params['id'] = nd.get('id', 0)
                    params['name'] = nd.get('name', f"Node_{nd.get('id', 0)}")
                    params['graph_id'] = new_graph_id
                    
                    if 'center_of_mass' in nd:
                        params['center_of_mass'] = np.array(nd['center_of_mass'])

                    original_connections = nd.get('connections', [])
                    adjusted_connections = []
                    
                    for conn in original_connections:
                        new_conn = {
                            'id': conn.get('id'),
                            'name': conn.get('name'),
                            'source': conn.get('source', {}).copy(),
                            'target': conn.get('target', {}).copy(),
                            'params': conn.get('params', {}).copy()
                        }
                        
                        if 'source' in new_conn and new_conn['source']:
                            old_src_gid = new_conn['source'].get('graph_id')
                            if old_src_gid in id_mapping:
                                new_conn['source']['graph_id'] = id_mapping[old_src_gid]
                        
                        if 'target' in new_conn and new_conn['target']:
                            old_tgt_gid = new_conn['target'].get('graph_id')
                            if old_tgt_gid in id_mapping:
                                new_conn['target']['graph_id'] = id_mapping[old_tgt_gid]
                        
                        adjusted_connections.append(new_conn)
                        total_connections += 1
                    
                    params['connections'] = adjusted_connections
                    
                    is_root = (nd.get('id', 0) == 0)
                    parent_id = nd.get('parent_id')
                    parent = graph.get_node(parent_id) if parent_id is not None else None
                    
                    new_node = graph.create_node(
                        parameters=params,
                        other=parent,
                        is_root=is_root,
                        auto_build=False
                    )
                    
                    if nd.get('positions'):
                        new_node.positions = [np.array(pos) for pos in nd['positions']]
                        if 'center_of_mass' in nd:
                            new_node.center_of_mass = np.array(nd['center_of_mass'])
                    
                    total_nodes += 1
                    
                    n_conns = len(adjusted_connections)
                    print(f" ✓ Node {nd.get('id')}: {nd.get('name', 'unnamed')} ({n_conns} connections adjusted)")

                for nd in nodes_data:
                    node = graph.get_node(nd.get('id'))
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

                print(f"   Populating NEST neurons...")
                for node in graph.node_list:
                    if not hasattr(node, 'positions') or not node.positions:
                        node.build()
                    elif all(len(p) == 0 for p in node.positions if p is not None):
                        node.build()
                    
                    node.populate_node()

                graph_list.append(graph)
                merged_graphs.append(graph)
                print(f"   Graph '{graph_name}' merged successfully!")
            if hasattr(self, 'tools_widget'):
                self.tools_widget.update_graphs(graph_list)
            self.status_bar.set_status("Creating NEST connections...", color="#9C27B0")
            self.status_bar.set_progress(80)
            QApplication.processEvents()

            print(f"\nCreating NEST connections for merged graphs...")
            
            conn_created = 0
            conn_failed = 0
            
            for graph in merged_graphs:
                for node in graph.node_list:
                    if not hasattr(node, 'connections') or not node.connections:
                        continue
                    
                    for conn in node.connections:
                        try:
                            source = conn.get('source', {})
                            target = conn.get('target', {})
                            params = conn.get('params', {})
                            
                            src_graph = next((g for g in graph_list if g.graph_id == source.get('graph_id')), None)
                            if not src_graph:
                                raise ValueError(f"Source graph {source.get('graph_id')} not found")
                            
                            src_node = next((n for n in src_graph.node_list if n.id == source.get('node_id')), None)
                            if not src_node:
                                raise ValueError(f"Source node {source.get('node_id')} not found")
                            
                            if source.get('pop_id') >= len(src_node.population):
                                raise ValueError(f"Source pop {source.get('pop_id')} out of range")
                            
                            src_pop = src_node.population[source.get('pop_id')]
                            if src_pop is None or len(src_pop) == 0:
                                raise ValueError("Source population empty")
                            
                            tgt_graph = next((g for g in graph_list if g.graph_id == target.get('graph_id')), None)
                            if not tgt_graph:
                                raise ValueError(f"Target graph {target.get('graph_id')} not found")
                            
                            tgt_node = next((n for n in tgt_graph.node_list if n.id == target.get('node_id')), None)
                            if not tgt_node:
                                raise ValueError(f"Target node {target.get('node_id')} not found")
                            
                            if target.get('pop_id') >= len(tgt_node.population):
                                raise ValueError(f"Target pop {target.get('pop_id')} out of range")
                            
                            tgt_pop = tgt_node.population[target.get('pop_id')]
                            if tgt_pop is None or len(tgt_pop) == 0:
                                raise ValueError("Target population empty")
                            
                            rule = params.get('rule', 'all_to_all')
                            conn_spec = {'rule': rule}
                            
                            if 'indegree' in params:
                                conn_spec['indegree'] = params['indegree']
                            if 'outdegree' in params:
                                conn_spec['outdegree'] = params['outdegree']
                            if 'N' in params:
                                conn_spec['N'] = params['N']
                            if 'p' in params:
                                conn_spec['p'] = params['p']
                            
                            conn_spec['allow_autapses'] = params.get('allow_autapses', False)
                            conn_spec['allow_multapses'] = params.get('allow_multapses', True)
                            
                            syn_spec = {
                                'synapse_model': params.get('synapse_model', 'static_synapse'),
                                'weight': params.get('weight', 1.0),
                                'delay': max(params.get('delay', 1.0), nest.resolution)
                            }
                            
                            if rule == 'one_to_one' and len(src_pop) != len(tgt_pop):
                                raise ValueError(f"one_to_one: size mismatch ({len(src_pop)} vs {len(tgt_pop)})")
                            
                            nest.Connect(src_pop, tgt_pop, conn_spec, syn_spec)
                            conn_created += 1
                            
                        except Exception as e:
                            conn_failed += 1
                            print(f"   ⚠ Connection failed: {conn.get('name', '?')}: {e}")

            print(f"\n   Connections: {conn_created} created, {conn_failed} failed")

            self.status_bar.set_status("Refreshing view...", color="#9C27B0")
            self.status_bar.set_progress(95)
            QApplication.processEvents()

            self.update_visualizations()
            self.graph_overview.update_tree()
            self.connection_tool.refresh()
            self.graph_editor.refresh_graph_list()
            self.blink_widget.build_scene()

            print(f"MERGE COMPLETE!")
            print(f"   Graphs merged: {len(merged_graphs)}")
            print(f"   Nodes added: {total_nodes}")
            print(f"   Connections: {conn_created} created, {conn_failed} failed")
            print(f"   Total graphs now: {len(graph_list)}")
            if graph_list:
                current_max_id = max(g.graph_id for g in graph_list)
                if current_max_id >= WidgetLib.next_graph_id:
                    WidgetLib.next_graph_id = current_max_id + 1
            
            if hasattr(self, 'graph_builder'):
                self.graph_builder.reset()
            self.status_bar.show_success(f"Merged {len(merged_graphs)} graph(s)!")
            
            QMessageBox.information(
                self, "Merge Complete",
                f"Successfully merged:\n\n"
                f"• {len(merged_graphs)} graph(s)\n"
                f"• {total_nodes} node(s)\n"
                f"• {conn_created} connection(s) created\n"
                f"• {conn_failed} connection(s) failed\n\n"
                f"New Graph IDs: {[g.graph_id for g in merged_graphs]}\n"
                f"Total graphs now: {len(graph_list)}"
            )

        except Exception as e:
            self.status_bar.show_error(f"Merge failed: {e}")
            print(f"Merge failed: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Merge Error", str(e))
    def collect_simulation_results(self, duration):
        print(f"\n>>> COLLECTING & FLUSHING DATA <<<")
        timestamp = datetime.now().isoformat()
        
        has_data = False
        
        for graph in graph_list:
            for node in graph.node_list:
                if "history" not in node.results:
                    node.results["history"] = []
                
                run_data = {
                    "timestamp": timestamp,
                    "devices": {}
                }
                
                if not hasattr(node, 'devices') or not node.devices:
                    continue

                for dev in node.devices:
                    gid = dev.get('runtime_gid')
                    dev_id = dev.get('id')
                    model = dev.get('model', '')
                    
                    if gid is None: continue
                    
                    if "recorder" in model or "meter" in model:
                        try:
                            status = nest.GetStatus(gid)[0]
                            events = status.get('events', {})
                            
                            n_events = status.get('n_events', 0)
                            if n_events == 0:
                                continue
                                
                            cleaned_events = {}
                            for k, v in events.items():
                                if isinstance(v, np.ndarray):
                                    cleaned_events[k] = v.tolist()
                                else:
                                    cleaned_events[k] = v
                            
                            run_data["devices"][dev_id] = {
                                "model": model,
                                "type": "recorder",
                                "data": cleaned_events
                            }
                            has_data = True
                            
                            nest.SetStatus(gid, {'n_events': 0})
                            print(f"  -> {model} (ID {dev_id}): Collected & Flushed {n_events} events.")
                            
                        except Exception as e:
                            print(f"Error collecting {model}: {e}")

                if run_data["devices"]:
                    node.results["history"].append(run_data)
        
        if has_data:
            print(">>> Data collection complete.")
            if hasattr(self, 'status_bar'):
                self.status_bar.show_success("Simulation data collected & flushed.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_mode(app)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
