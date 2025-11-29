import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSlider, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QStackedWidget, QMessageBox, QProgressBar,
    QGridLayout, QFileDialog
)
import WidgetLib
import pyqtgraph.dockarea as dock
from PyQt6.QtGui import QColor, QPalette, QAction
from PyQt6.QtCore import Qt
import numpy as np
import time
import vtk
import pyvista as pv
from pyvistaqt import QtInteractor
from neuron_toolbox import *
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from WidgetLib import *
from WidgetLib import _clean_params, _serialize_connections, NumpyEncoder,BlinkingNetworkWidget,FlowFieldWidget,StructuresWidget
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
    """Simple colored placeholder widget."""
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
        
        # Status Label (left)
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
        """Update status text with color"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-weight: bold;
                padding: 2px 5px;
            }}
        """)
    
    def set_progress(self, value, maximum=100, show=True):
        """Update progress bar"""
        if show:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(maximum)
            self.progress_bar.setValue(value)
        else:
            self.progress_bar.setVisible(False)
    
    def reset(self):
        """Reset to ready state"""
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
        """Show error message"""
        self.set_status(f"{message}", color="#D32F2F")
        self.progress_bar.setVisible(False)
    
    def show_success(self, message):
        """Show success message"""
        self.set_status(f"{message}", color="#2E7D32")
        self.progress_bar.setVisible(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuroticks")
        self.resize(1920, 1080)
        self.active_graphs = {}
        self.structural_plasticity_enabled = True 
        
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
        self.action_auto_save.setChecked(True) # Standardmäßig an
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
            if hasattr(self, 'tools_widget'):
                self.tools_widget.update_graphs(graph_list)
            WidgetLib.graph_parameters.clear()
            WidgetLib.next_graph_id = 0
            
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
            self.status_bar.showMessage("Resetting NEST kernel...")
        
        stats = safe_nest_reset_and_repopulate(
            self.graphs,
            enable_structural_plasticity=enable_structural_plasticity,
            verbose=True
        )
        
        msg = (f"Reset complete: {stats['populations_created']} populations, "
               f"{stats['connections_created']} connections")
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(msg, 5000)
        
        if hasattr(self, 'update_visualization'):
            self.update_visualization()
        
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
        """
        Called when a graph is deleted.
        Ensures connection system is updated.
        """
        if graph_id in self.graphs:
            del self.graphs[graph_id]
        
        if hasattr(self, 'connection_tool'):
            self.connection_tool.set_graphs(self.graphs)
    
    def get_connection_summary(self):
        """Get summary of all connections."""
        return get_all_connections_summary(self.graphs)
    
    def export_connections(self, filepath):
        """Export all connections to JSON file."""
        import json
        
        connections = export_connections_to_dict(self.graphs)
        with open(filepath, 'w') as f:
            json.dump(connections, f, indent=2)
        
        return len(connections)
    
    def import_connections(self, filepath, clear_existing=True):
        """Import connections from JSON file."""
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
        
        # --- SWITCH BAR ---
        switch_bar = QWidget()
        switch_bar.setStyleSheet("background-color: #1a1a1a; border-bottom: 2px solid #333;")
        switch_bar.setFixedHeight(50) 
        switch_layout = QHBoxLayout(switch_bar)
        switch_layout.setContentsMargins(10, 5, 10, 5)
        switch_layout.setSpacing(10)
        
        self.view_switch_style_active = """
            QPushButton {
                background-color: #2196F3; color: white; font-weight: bold;
                border: none; border-radius: 4px; padding: 10px 40px; font-size: 14px;
            }
            QPushButton:hover { background-color: #42A5F5; }
        """
        self.view_switch_style_inactive = """
            QPushButton {
                background-color: #333; color: #aaa; font-weight: bold;
                border: 1px solid #555; border-radius: 4px; padding: 10px 40px; font-size: 14px;
            }
            QPushButton:hover { background-color: #444; color: white; }
        """
        
        self.btn_view_editor = QPushButton(" EDITOR")
        self.btn_view_simulation = QPushButton(" SIMULATION")
        self.btn_view_data = QPushButton(" DATA")
        
        self.btn_view_editor.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_view_simulation.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_view_data.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.btn_view_editor.setStyleSheet(self.view_switch_style_active)
        self.btn_view_simulation.setStyleSheet(self.view_switch_style_inactive)
        self.btn_view_data.setStyleSheet(self.view_switch_style_inactive)

        self.btn_view_editor.clicked.connect(lambda: self._switch_main_view(0))
        self.btn_view_simulation.clicked.connect(lambda: self._switch_main_view(1))
        self.btn_view_data.clicked.connect(lambda: self._switch_main_view(2))
        
        self.view_buttons = [self.btn_view_editor, self.btn_view_simulation, self.btn_view_data]
        
        switch_layout.addWidget(self.btn_view_editor)
        switch_layout.addWidget(self.btn_view_simulation)
        switch_layout.addWidget(self.btn_view_data)
        switch_layout.addStretch()
        
        lbl_title = QLabel("NEUROTICKS")
        lbl_title.setStyleSheet("color: #666; font-weight: bold; font-size: 12px; letter-spacing: 3px;")
        switch_layout.addWidget(lbl_title)
        
        main_layout.addWidget(switch_bar)
        
        # --- STACK ---
        self.main_stack = QStackedWidget()
        self.main_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) 
        
        self.editor_widget = self._create_editor_widget()
        self.main_stack.addWidget(self.editor_widget)

        # --- SIMULATION VIEW SETUP ---
        self.simulation_view = SimulationViewWidget(graph_list, self)
        
        # FIX: Korrekte Signalnamen verwenden (sigPauseSimulation, sigStopSimulation)
        self.simulation_view.sigPauseSimulation.connect(lambda: self.collect_simulation_results(duration=0))
        self.simulation_view.sigStopSimulation.connect(lambda: self.collect_simulation_results(duration=0))
        
        self.main_stack.addWidget(self.simulation_view)
        self.live_dashboard = LiveDataDashboard(graph_list, self)
        self.live_dashboard.sigStartLive.connect(self.start_continuous_simulation)
        self.live_dashboard.sigPauseLive.connect(self.pause_simulation)
        
        # WICHTIG: Neuer Reset Slot
        self.live_dashboard.sigResetLive.connect(self.reset_from_live_dashboard)
        
        self.main_stack.addWidget(self.live_dashboard)
        self.data_view = self._create_data_view()
        self.main_stack.addWidget(self.data_view)
        
        main_layout.addWidget(self.main_stack)
        
        self.update_visualizations()
        
        # Initialisiert den Timer und verbindet die Steuersignale (Start/Step)
        self.init_simulation_timer()

    def _create_data_view(self):
        self.data_dashboard = AnalysisDashboard(graph_list) 
        return self.data_dashboard
    
    def _switch_main_view(self, index):
        old_index = self.main_stack.currentIndex()
        
        # Index 1 ist der Simulation View
        if old_index == 1:
            print("Verlasse Simulation View -> Cleanup GL...")
            if hasattr(self.simulation_view, 'cleanup_gl_context'):
                self.simulation_view.cleanup_gl_context()
            
        self.main_stack.setCurrentIndex(index)
        
        # Buttons Styling update...
        for i, btn in enumerate(self.view_buttons):
            if i == index:
                btn.setStyleSheet(self.view_switch_style_active)
            else:
                btn.setStyleSheet(self.view_switch_style_inactive)
    
    



    def _create_editor_widget(self):
        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)
        
        # Top section (60%)
        top_layout = QHBoxLayout()
        top_left = self.create_top_left()
        self.graph_overview = GraphOverviewWidget(self, graph_list=graph_list)        
        top_layout.addLayout(top_left, 7)
        top_layout.addWidget(self.graph_overview, 3)
        
        # Bottom section (40%)
        bottom_layout = QHBoxLayout()
        bottom_left = self.create_bottom_left()
        bottom_right = self.create_bottom_right()
        
        bottom_layout.addLayout(bottom_left, 7)
        bottom_layout.addLayout(bottom_right, 3)
        
        # Assemble
        editor_layout.addLayout(top_layout, 3)
        editor_layout.addLayout(bottom_layout, 2)
        
        return editor
    
    def create_top_left(self):
        """Visualization area with scene selector."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Scene menu buttons container
        scene_menu = QWidget()
        scene_menu.setStyleSheet("background-color: #2b2b2b;")
        
        scene_layout = QVBoxLayout()
        scene_layout.setContentsMargins(0, 0, 0, 0)
        scene_layout.setSpacing(1)
        
        # Button Style
        nav_style = """
            QPushButton {
                background-color: #444;
                color: white;
                border: none;
                border-radius: 0px;
                font-weight: bold;
                text-align: left;
                padding-left: 15px;
                min-height: 40px;
            }
            QPushButton:hover { background-color: #555; border-left: 4px solid #2196F3; }
            QPushButton:checked { background-color: #666; border-left: 4px solid #2196F3; }
        """
        
        btn_neurons = QPushButton("Neurons")
        btn_graph = QPushButton("Graph")
        
        self.firing_patterns_container = QWidget()
        fp_layout = QVBoxLayout(self.firing_patterns_container)
        fp_layout.setContentsMargins(0, 0, 0, 0)
        fp_layout.setSpacing(0)
        
        btn_sim = QPushButton("Firing Patterns")
        
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
        
        btn_flow = QPushButton("Positional Flowfield")
        btn_simulation = QPushButton("Simulation")
        btn_other = QPushButton("Other")
        
        for btn in [btn_neurons, btn_graph, btn_flow, btn_simulation, btn_other]:
            btn.setStyleSheet(nav_style)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        btn_sim.setStyleSheet(nav_style)
        
        scene_layout.addWidget(btn_neurons)
        scene_layout.addWidget(btn_graph)
        scene_layout.addWidget(self.firing_patterns_container) 
        scene_layout.addWidget(btn_flow)
        scene_layout.addWidget(btn_simulation)
        scene_layout.addWidget(btn_other)
            
        scene_layout.addStretch()
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
        # HEADLESS SIGNALE VERBINDEN
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
        
        for btn in self.nav_buttons:
            btn.setCheckable(True)
        
        btn_neurons.setChecked(True)
        
        layout.addWidget(scene_menu, 1)
        layout.addWidget(self.vis_stack, 9)
        
        return layout
    def start_headless_simulation(self, duration):
        """
        Startet die Simulation ohne grafische Updates.
        Sperrt den SimulationView Tab.
        """
        print(f"\n>>> STARTING HEADLESS SIMULATION ({duration} ms) <<<")
        
        # 1. UI Sperren & Feedback
        self.sim_dashboard.set_ui_locked(True)
        self.simulation_view.setEnabled(False) # Simulation Tab ausgrauen
        self.btn_view_simulation.setEnabled(False) # Tab Wechsel verhindern
        
        self.status_bar.set_status("HEADLESS SIMULATION RUNNING...", color="#E65100")
        self.status_bar.set_progress(0)
        
        # 2. Parameter vorbereiten
        self.headless_target_time = duration
        self.headless_current_time = 0.0
        
        # Wir müssen Recorder sicherstellen, sonst gibt es keine Ergebnisse
        self._ensure_spike_recorders()
        
        # NEST Zeit holen (falls schon vorher lief)
        try:
            kernel_time = nest.GetKernelStatus().get('time', 0.0)
            self.headless_target_time += kernel_time # Relative Dauer addieren
        except: pass

        # 3. Timer starten (Chunked Simulation, damit GUI nicht einfriert)
        # Wir simulieren in 50ms Blöcken, damit der Stop-Button reagiert
        self.headless_step_size = 50.0 
        self.headless_timer = QTimer()
        self.headless_timer.timeout.connect(self.headless_loop_step)
        self.headless_timer.start(0) # Sofort feuern

    def headless_loop_step(self):
        try:
            # 1. Simulieren
            nest.Simulate(self.headless_step_size)
            
            # 2. Zeit prüfen
            current_time = nest.GetKernelStatus().get('time', 0.0)
            
            # 3. Progress Update
            # (Berechnung relativ zum Start schwer, hier einfach visual feedback)
            # Wir lassen den Balken einfach pulsieren oder laufen
            prog = int((current_time % 1000) / 10) 
            self.status_bar.set_progress(prog)
            
            # 4. Abbruchbedingung
            if current_time >= self.headless_target_time:
                self.finish_headless_simulation()
                
        except Exception as e:
            self.headless_timer.stop()
            self.stop_headless_simulation(error_msg=str(e))

    def stop_headless_simulation(self, error_msg=None):
        """Manueller Stop oder Fehler."""
        if hasattr(self, 'headless_timer'):
            self.headless_timer.stop()
            
        print(">>> Headless Simulation Stopped.")
        
        if error_msg:
            self.status_bar.show_error(f"Headless Error: {error_msg}")
            QMessageBox.critical(self, "Simulation Error", error_msg)
        else:
            self.status_bar.set_status("Simulation stopped by user.", "#D32F2F")
            
        self._restore_ui_after_headless()
        # Daten sammeln bis hierhin
        self.collect_simulation_results(0)

    def finish_headless_simulation(self):
        """Reguläres Ende."""
        self.headless_timer.stop()
        print(">>> Headless Simulation Finished.")
        self.status_bar.show_success("Headless Simulation Complete!")
        
        self.collect_simulation_results(0)
        self._restore_ui_after_headless()

    def _restore_ui_after_headless(self):
        """Stellt UI-Zustand wieder her."""
        self.sim_dashboard.set_ui_locked(False)
        self.simulation_view.setEnabled(True) # Simulation Tab wieder aktiv
        self.btn_view_simulation.setEnabled(True)
        self.status_bar.set_progress(100)

    



    def reset_from_live_dashboard(self, keep_data):
        print("Live Dashboard Reset triggered.")
        self.sim_timer.stop()
        
        # Save History (wenn gewünscht)
        if self.action_auto_save.isChecked():
            self.archive_live_data()
            
        # Reset aufrufen
        self.reset_and_restart()



    def archive_live_data(self):
        """Speichert aktuelle Live-Daten in eine JSON Datei."""
        print("Archiving Live Data history...")
        try:
            data = self.live_dashboard.get_all_data()
            if not data:
                print("No data to save.")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"history_live_{timestamp}.json"
            
            # Ordner erstellen
            Path("sim_history").mkdir(exist_ok=True)
            filepath = Path("sim_history") / filename
            
            with open(filepath, 'w') as f:
                json.dump(data, f, cls=NumpyEncoder, indent=2)
                
            self.status_bar.show_success(f"History saved: {filename}")
            print(f"Saved history to {filepath}")
            
        except Exception as e:
            print(f"Error archiving data: {e}")
            self.status_bar.show_error(f"Save Error: {e}")

    # Update in 'on_sim_timer_timeout':
    def on_sim_timer_timeout(self):
        try:
            # 1. Zeit prüfen (nur bei Continuous Modus relevant)
            if self.sim_mode == 'continuous' and self.sim_target_time > 0:
                try:
                    current_time = nest.GetKernelStatus().get('time', 0.0)
                    if current_time >= self.sim_target_time:
                        self.pause_simulation()
                        self.status_bar.show_success(f"Finished (Time: {current_time}ms)")
                        if hasattr(self, 'simulation_view'):
                            self.simulation_view.action_stop()
                        return
                except: pass

            # 2. NEST Simulieren (ein Zeitschritt)
            nest.Simulate(self.sim_step_size)
            
            # 3. Daten verteilen (3D & 2D)
            self._distribute_simulation_data()
            
            # 4. GUI lebendig halten (wichtig bei hoher Last)
            QApplication.processEvents()
            
        except Exception as e:
            # Bei Fehler: Timer stoppen
            self.sim_timer.stop()
            if hasattr(self, 'simulation_view'):
                self.simulation_view.is_paused = True
            
            error_msg = str(e)
            
            # --- FIX: InvalidNodeCollection stummschalten ---
            # Das passiert erwartungsgemäß, wenn während der Simulation ein Reset gedrückt wird.
            if "InvalidNodeCollection" in error_msg:
                return # Silent exit, kein Popup
            # ------------------------------------------------
            
            print(f"CRITICAL SIM ERROR: {error_msg}")
            
            if "NumericalInstability" in error_msg:
                self.status_bar.show_error("Numerical Instability! Reset required.")
                QMessageBox.critical(self, "Crash", "Simulation crashed (Numerical Instability).\nPlease RESET.")
            elif "Prepare called twice" in error_msg:
                self.status_bar.show_error("Kernel corrupted. Reset required.")
            else:
                self.status_bar.show_error(f"Sim Error: {e}")
        
    def _switch_view(self, index, sim_mode=False):
        """Switches view and controls simulation state."""
        
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
        """Creates graph skeleton plotter."""
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

        # Index 0: Builder
        self.graph_builder = GraphCreatorWidget()
        self.graph_builder.graphCreated.connect(self.on_graph_created)
        self.tool_stack.addWidget(self.graph_builder)
        
        # Index 1: Editor
        self.graph_editor = EditGraphWidget(graph_list=graph_list)
        self.graph_editor.graphUpdated.connect(self.on_graph_updated)
        self.tool_stack.addWidget(self.graph_editor)
        
        # Index 2: Connections
        self.connection_tool = ConnectionTool(graph_list)        
        self.tool_stack.addWidget(self.connection_tool)
        
        # Index 3: Structures 
        self.structures_widget = StructuresWidget()
        self.structures_widget.structureSelected.connect(self.on_structure_selected)
        self.tool_stack.addWidget(self.structures_widget)
        
        # Index 4: Tools (Hier müssen wir sicherstellen, dass wir an das Signal kommen)
        self.tools_widget = ToolsWidget()
        self.tools_widget.update_graphs(graph_list)
        
        # WICHTIG: deviceUpdated mit MainWindow verbinden für Update-Logik
        # HINWEIS: deviceCreated ist bereits in ToolsWidget.init_ui() verbunden!
        for i in range(self.tools_widget.config_stack.count()):
            widget = self.tools_widget.config_stack.widget(i)
            if isinstance(widget, DeviceConfigPage):
                # NUR deviceUpdated hier verbinden (deviceCreated läuft über ToolsWidget)
                widget.deviceUpdated.connect(self.handle_device_update)

        self.tool_stack.addWidget(self.tools_widget)        
        self.status_bar = StatusBarWidget()
        
        layout.addWidget(self.tool_stack, 9)
        layout.addWidget(self.status_bar, 1)
        
        # Signale verbinden
        self.graph_overview.node_selected.connect(self._on_overview_node_selected)
        # ... andere Signale ...
        
        # NEU: Wenn man im Tree auf ein Device klickt
        self.graph_overview.device_selected.connect(self.on_device_tree_click)
        
        self.graph_overview.node_selected.connect(self._on_overview_node_selected)
        self.graph_overview.population_selected.connect(self._on_overview_pop_selected)
        self.graph_overview.connection_selected.connect(self._on_overview_conn_selected)
        self.graph_overview.requestConnectionCreation.connect(self.open_connection_tool_for_node)
        self.graph_overview.requestConnectionDeletion.connect(self.delete_connection_wrapper)
        return layout
    
    def on_device_tree_click(self, device_data):
        # 1. Zum Tools Tab wechseln
        self.tool_stack.setCurrentIndex(4) 
        # 2. Editor öffnen
        self.tools_widget.open_device_editor(device_data)
        self.status_bar.set_status(f"Editing Device: {device_data.get('model')}", "#FF9800")

    def handle_device_update(self, old_data, new_data):
        """
        Löscht das alte Device, resettet NEST und erstellt das neue Device.
        """
        print("\n=== UPDATING DEVICE ===")
        
        # --- FIX: Simulation stoppen bevor wir resetten! ---
        if hasattr(self, 'sim_timer'):
            self.sim_timer.stop()
        self.live_recorders = [] # Referenzen auf alte GIDs löschen
        # ---------------------------------------------------

        self.status_bar.set_status("Updating Device & Resetting Kernel...", "#FF5722")
        QApplication.processEvents()

        # 1. Target Infos holen
        target = old_data.get('target', {})
        gid = target.get('graph_id')
        nid = target.get('node_id')
        
        old_id = old_data.get('id')
        if old_id is None:
            print("Error: Old device has no ID. Cannot update.")
            return
        
        found_and_deleted = False
        
        # 2. Graph und Node finden & Device löschen
        target_graph = next((g for g in graph_list if g.graph_id == gid), None)
        
        if target_graph:
            node = target_graph.get_node(nid)
            if node:
                def filter_devs(dev_list, target_id):
                    return [d for d in dev_list if str(d.get('id')) != str(target_id)]

                if hasattr(node, 'devices'):
                    len_before = len(node.devices)
                    node.devices = filter_devs(node.devices, old_id)
                    if len(node.devices) < len_before:
                        print(f"  ✓ Deleted device {old_id} from node.devices")
                        found_and_deleted = True

                if hasattr(node, 'parameters') and 'devices' in node.parameters:
                    len_before = len(node.parameters['devices'])
                    node.parameters['devices'] = filter_devs(node.parameters['devices'], old_id)
                    if len(node.parameters['devices']) < len_before:
                        print(f"  ✓ Deleted device {old_id} from node.parameters['devices']")
                        found_and_deleted = True
        
        # 3. NEST Reset
        nest.ResetKernel()
        if self.structural_plasticity_enabled:
            nest.EnableStructuralPlasticity()
            
        print("Kernel Reset complete.")
        
        # 4. Welt neu aufbauen
        for graph in graph_list:
            for node in graph.node_list:
                if not hasattr(node, 'positions') or not node.positions:
                    node.build()
                elif all(len(p) == 0 for p in node.positions if p is not None):
                    node.build()
                node.populate_node() 
        
        # 5. Das NEUE Device erstellen
        global _nest_simulation_has_run
        _nest_simulation_has_run = False
        
        self.tools_widget.on_device_created(new_data)
        
        # 6. GUI Refresh
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
        """Handle node selection from tree view."""
        
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
        
        # Layout des Containers
        main_layout = QHBoxLayout(main_container)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15) 

        # --- CSS STYLES ---
        # Basis-Stil für alle großen Buttons (füllend, dunkel)
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

        # Stil für den AKTIVEN Button (Neon-Blau, leuchtend)
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

        # Akzent-Stile für Aktionen (Col 2 & 3)
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

        # ==========================================
        # COLUMN 1: NAVIGATION (Full Height)
        # ==========================================
        col1_layout = QVBoxLayout()
        col1_layout.setSpacing(5)
        col1_layout.addWidget(create_header("NAVIGATION"))
        
        # Liste der Navigations-Buttons
        self.nav_btns = []
        
        # Definition der Tabs
        nav_items = [
            ("Network Builder", 0),
            ("Graph Inspector", 1),
            ("Connectivity", 2),
            ("Structure Lib", 3),
            ("Instrumentation", 4)
        ]

        def on_nav_click(target_idx, clicked_btn):
            # 1. Stack umschalten
            self.tool_stack.setCurrentIndex(target_idx)
            
            # 2. Styles updaten (Neon-Effekt umschalten)
            for btn in self.nav_btns:
                if btn == clicked_btn:
                    btn.setStyleSheet(active_btn_style)
                else:
                    btn.setStyleSheet(base_btn_style)

        for label, idx in nav_items:
            btn = QPushButton(label)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Füllt den Raum
            
            # Standardmäßig inaktiv
            btn.setStyleSheet(base_btn_style)
            
            # Klick-Verbindung (Lambda fixiert idx und btn)
            btn.clicked.connect(lambda checked, i=idx, b=btn: on_nav_click(i, b))
            
            self.nav_btns.append(btn)
            col1_layout.addWidget(btn)

        # Den ersten Button initial als aktiv markieren
        if self.nav_btns:
            self.nav_btns[0].setStyleSheet(active_btn_style)

        # ==========================================
        # COLUMN 2: KERNEL OPERATIONS
        # ==========================================
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

        # ==========================================
        # COLUMN 3: PROJECT I/O
        # ==========================================
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

        # --- Layout Assembly ---
        # Verteilung: Col 1 ist am wichtigsten (Nav), daher etwas breiter
        main_layout.addLayout(col1_layout, 4)
        main_layout.addLayout(col2_layout, 3)
        main_layout.addLayout(col3_layout, 2)

        wrapper_layout = QHBoxLayout() 
        wrapper_layout.setContentsMargins(0,0,0,0)
        wrapper_layout.addWidget(main_container)
        
        return wrapper_layout

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
                self.simulation_view.restore_injectors() # Generatoren wiederherstellen
            # ------------------

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
    """
    def update_visualizations(self):
        self.plot_neuron_points()
        self.plot_graph_skeleton()
    """
    def plot_neuron_points(self):
        self.neuron_plotter.clear()
        legend_entries = []
        used_types = set()
        for graph in graph_list:
            for node in graph.node_list:
                for i, pts in enumerate(node.positions):
                    if pts is None or len(pts) == 0:
                        continue
                    neuron_type = node.neuron_models[i] if i < len(node.neuron_models) else "unknown"
                    color = get_neuron_color(neuron_type)  # ← OHNE self. !
                    if neuron_type not in used_types:
                        legend_entries.append([neuron_type, color])
                        used_types.add(neuron_type)
                    point_cloud = pv.PolyData(pts)
                    self.neuron_plotter.add_mesh(
                        point_cloud,
                        color=color,
                        point_size=10
                    )
        if legend_entries:
            self.neuron_plotter.add_legend(legend_entries, size=(0.12, 0.12), loc='upper right')
        self.neuron_plotter.reset_camera()
        self.neuron_plotter.update()
        
   


    def plot_graph_skeleton(self):

        self.graph_plotter.clear()
        
        self.highlighted_actor = None
        self.original_color = None
        self.original_opacity = None
        
        self.skeleton_info_map = {}
        
        cmap = plt.get_cmap("tab20")
        
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

        for graph in graph_list:
            rgba = cmap(graph.graph_id % 20)
            graph_color = rgba[:3]
            graph_name = getattr(graph, 'graph_name', f'Graph {graph.graph_id}')
            
            for node in graph.node_list:
                center = np.array(node.center_of_mass)
                sphere = pv.Sphere(radius=0.4, center=center)
                
                actor = self.graph_plotter.add_mesh(
                    sphere,
                    color=graph_color,
                    opacity=0.6,
                    smooth_shading=True,
                    pickable=True
                )
                
                info_text = f"NODE: {node.name} (ID: {node.id})"
                self.skeleton_info_map[actor] = info_text
                
                if hasattr(node, 'connections') and node.connections:
                    for conn in node.connections:
                        try:
                            target = conn.get('target', {})
                            tgt_gid = int(target.get('graph_id'))
                            tgt_nid = int(target.get('node_id'))
                            
                            target_node = node_map.get((tgt_gid, tgt_nid))
                            
                            if target_node:
                                conn_name = conn.get('name', 'Connection')
                                info_text = f"CONN: {conn_name}\nFrom: {node.name} -> To: {target_node.name}"

                                start = np.array(node.center_of_mass)
                                end = np.array(target_node.center_of_mass)

                                if node == target_node:
                                    torus = pv.ParametricTorus(ringradius=0.6, crosssectionradius=0.05) 
                                    torus.translate([0, 0, 0.5])
                                    torus.translate(start)
                                    
                                    conn_actor = self.graph_plotter.add_mesh(
                                        torus,
                                        color=graph_color,
                                        opacity=0.8,
                                        pickable=True
                                    )
                                    self.skeleton_info_map[conn_actor] = info_text + " (Self)"

                                else:
                                    direction_vec = end - start
                                    dist = np.linalg.norm(direction_vec)
                                    
                                    if dist > 0.1:
                                        direction_norm = direction_vec / dist
                                        
                                        offset_start = 0.4
                                        offset_end = 0.4
                                        arrow_start = start + (direction_norm * offset_start)
                                        
                                        raw_length = dist - (offset_start + offset_end)
                                        arrow_length = max(0.1, raw_length)

                                        arrow = pv.Arrow(
                                            start=arrow_start, 
                                            direction=direction_norm, 
                                            scale=arrow_length,
                                            tip_length=min(0.25, 0.4 / arrow_length), 
                                            tip_radius=0.08,  
                                            shaft_radius=0.02  
                                        )
                                        
                                        conn_actor = self.graph_plotter.add_mesh(
                                            arrow, 
                                            color=graph_color, 
                                            opacity=0.8,
                                            pickable=True,
                                            smooth_shading=True
                                        )
                                        self.skeleton_info_map[conn_actor] = info_text

                        except Exception as e:
                            print(f"Error plotting conn: {e}")

        try:
            self.graph_plotter.iren.remove_observer(self._observer_tag)
        except:
            pass
        self._observer_tag = self.graph_plotter.iren.add_observer(
            "MouseMoveEvent", self._on_skeleton_hover
        )
        
        self.graph_plotter.update()


    
    
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
        
        # 1. Sicherstellen, dass alle Popualtionen Spike Recorder haben
        # Das ist entscheidend für die Visualisierung!
        self._ensure_spike_recorders()
        
        # 2. Simulation Step-by-Step
        step_size = 25.0 # ms (40 FPS)
        current_time = 0.0
        
        # Umschalten auf Simulation Tab, damit man was sieht
        self._switch_main_view(1) 
        
        try:
            while current_time < duration:
                # NEST simulieren
                nest.Simulate(step_size)
                current_time += step_size
                
                # Spikes einsammeln und visualisieren
                
                # GUI Update erzwingen
                QApplication.processEvents()
                
                # Progress Update
                prog = int((current_time / duration) * 100)
                self.status_bar.set_progress(prog)
                
                # Abbrechen falls User Stop gedrückt hat (Check im Widget)
                if hasattr(self.simulation_view, 'is_paused') and self.simulation_view.is_paused:
                    break

            # Abschluss
            self.collect_simulation_results(duration)
            self.status_bar.show_success("Simulation finished.")
            self.sim_dashboard.btn_results.setEnabled(True)
            
        except Exception as e:
            self.status_bar.show_error(f"Simulation Error: {e}")
            print(f"Error: {e}")
            import traceback; traceback.print_exc()

    def _ensure_spike_recorders(self):
        """Verbindet temporäre Spike Recorder mit allen Neuronen für die Live-View."""
        self.live_recorders = []
        for graph in graph_list:
            for node in graph.node_list:
                if hasattr(node, 'population'):
                    for pop in node.population:
                        if pop:
                            rec = nest.Create("spike_recorder")
                            nest.Connect(pop, rec)
                            self.live_recorders.append(rec)



    # === SIMULATION CONTROL LOOP ===

    def init_simulation_timer(self):
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.on_sim_timer_timeout)
        
        # Modus-Flag: 'continuous' oder 'step'
        self.sim_mode = 'continuous' 
        self.sim_target_time = 0.0
        
        if hasattr(self, 'simulation_view'):
            self.simulation_view.sigStartContinuous.connect(self.start_continuous_simulation)
            self.simulation_view.sigStepSimulation.connect(self.step_simulation)
            self.simulation_view.sigPauseSimulation.connect(self.pause_simulation)
            self.simulation_view.sigStopSimulation.connect(self.stop_simulation)
            self.simulation_view.sigResetSimulation.connect(self.reset_and_restart)

    def start_continuous_simulation(self, step_size, max_duration):
        print(f"Starting Continuous Run (Step: {step_size}ms, Target: {max_duration}ms)")
        self.sim_mode = 'continuous'
        self.sim_step_size = step_size
        self.sim_target_time = max_duration
        
        self.sim_timer.setSingleShot(False) # Timer läuft endlos
        
        # Prüfen ob wir schon am Ende sind (falls nicht resetet wurde)
        try:
            curr = nest.GetKernelStatus().get('time', 0.0)
            if self.sim_target_time > 0 and curr >= self.sim_target_time:
                print("Target time reached. Please Reset.")
                self.status_bar.show_error("Target time reached. Please Reset.")
                return
        except: pass

        self._ensure_spike_recorders()
        self.sim_timer.start(0) # 0ms = so schnell wie möglich

    def step_simulation(self, step_size):
        print(f"Executing Single Step ({step_size}ms)")
        self.sim_mode = 'step'
        self.sim_step_size = step_size
        
        self.sim_timer.setSingleShot(True) # Timer feuert nur EINMAL
        self._ensure_spike_recorders()
        self.sim_timer.start(0)

    def pause_simulation(self):
        print("Simulation Paused")
        self.sim_timer.stop()
        self.status_bar.set_status("Paused", "#FBC02D")

    def stop_simulation(self):
        print("Simulation Stopped")
        self.sim_timer.stop()
        self.collect_simulation_results(0)
        self.status_bar.set_status("Stopped", "#D32F2F")

    def reset_and_restart(self, duration=None):
        """
        Kompletter Reset des Kernels und Neuaufbau.
        Argument 'duration' ist optional und wird ignoriert (kein Auto-Start mehr).
        """
        print("\n=== RESETTING SIMULATION ===")
        self.sim_timer.stop()
        
        try:
            self.status_bar.set_status("Resetting Kernel & Network...", "#E65100")
            QApplication.processEvents()
            
            # 1. Visualisierung stoppen
            if hasattr(self, 'simulation_view'):
                self.simulation_view.stop_rendering_safe()

            # 2. NEST Reset
            nest.ResetKernel()
            if self.structural_plasticity_enabled:
                nest.EnableStructuralPlasticity()
            
            # 3. Netz neu bauen
            for graph in graph_list:
                for node in graph.node_list:
                    if not hasattr(node, 'positions') or not node.positions:
                        node.build()
                    node.populate_node()
            
            # 4. Verbindungen
            graphs_dict = {g.graph_id: g for g in graph_list}
            create_nest_connections_from_stored(graphs_dict, verbose=False)
            
            # 5. Visualisierung updaten
            if self.main_stack.currentIndex() == 1 and hasattr(self, 'simulation_view'):
                self.simulation_view.load_scene() 
                self.simulation_view.restore_injectors()
            
            # 6. Recorder
            self.live_recorders = [] 
            self._ensure_spike_recorders()
            
            # 7. Live Dashboard Rescan
            if hasattr(self, 'live_dashboard'):
                QTimer.singleShot(200, self.live_dashboard.scan_for_devices)

            self.status_bar.show_success("Reset Done. Ready.")
            print("=== RESET COMPLETE: READY ===")
            
        except Exception as e:
            self.status_bar.show_error(f"Reset Failed: {e}")
            print(f"Reset Error: {e}")
            import traceback; traceback.print_exc()



    def on_sim_timer_timeout(self):
        try:
            # 1. Zeit prüfen (bei Continuous)
            if self.sim_mode == 'continuous' and self.sim_target_time > 0:
                try:
                    current_time = nest.GetKernelStatus().get('time', 0.0)
                    if current_time >= self.sim_target_time:
                        self.pause_simulation()
                        self.status_bar.show_success(f"Finished (Time: {current_time}ms)")
                        if hasattr(self, 'simulation_view'):
                            self.simulation_view.action_stop()
                        return
                except: pass

            # 2. NEST Simulieren
            nest.Simulate(self.sim_step_size)
            
            self._distribute_simulation_data()
            
            QApplication.processEvents()

        except Exception as e:
            self.sim_timer.stop()
            if hasattr(self, 'simulation_view'):
                self.simulation_view.is_paused = True
            
            error_msg = str(e)
            
            # --- FIX: InvalidNodeCollection stummschalten ---
            # Das passiert, wenn ein Reset durchgeführt wurde, während der Timer noch lief.
            if "InvalidNodeCollection" in error_msg:
                print("Simulation loop stopped due to Kernel Reset (InvalidNodeCollection). This is expected.")
                self.live_recorders = [] # Liste bereinigen
                return # Silent exit, kein Popup für den User
            # ------------------------------------------------
            
            print(f"CRITICAL SIM ERROR: {error_msg}")
            
            if "NumericalInstability" in error_msg:
                self.status_bar.show_error("Numerical Instability! Reset required.")
                QMessageBox.critical(self, "Crash", "Simulation crashed (Numerical Instability).\nPlease RESET.")
            elif "Prepare called twice" in error_msg:
                self.status_bar.show_error("Kernel corrupted. Reset required.")
            else:
                self.status_bar.show_error(f"Sim Error: {e}")
                
    def _distribute_simulation_data(self):
        # 1. 3D View (Temp Recorder)
        if hasattr(self, 'live_recorders'):
            all_spikes = []
            for rec in self.live_recorders:
                try:
                    ev = nest.GetStatus(rec, 'events')[0]
                    if len(ev['times']) > 0:
                        all_spikes.extend(ev['senders'])
                        nest.SetStatus(rec, {'n_events': 0})
                except: pass
            
            if all_spikes and hasattr(self, 'simulation_view'):
                self.simulation_view.feed_spikes(all_spikes)

        # 2. Live Dashboard (User Recorder)
        if hasattr(self, 'live_dashboard'):
            self.live_dashboard.update_plots()
    
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
        
        # --- NEUER HOOK ---
            if hasattr(self, 'simulation_view'):
                self.simulation_view.restore_injectors()
        # ------------------
            return stats
    
    def start_simulation(self):###########################################################
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
            
            global next_graph_id
            next_graph_id = 0

            total_graphs = len(project_data['graphs'])
            
            for i, g_data in enumerate(project_data['graphs']):
                self.status_bar.set_status(f"Building Graph {i+1}/{total_graphs}...")
                self.status_bar.set_progress(20 + int(40 * (i/total_graphs)))
                QApplication.processEvents()

                graph = Graph(
                    graph_name=g_data.get('graph_name', 'LoadedGraph'),
                    graph_id=g_data['graph_id'],
                    parameter_list=[],
                    polynom_max_power=g_data.get('polynom_max_power', 5),
                    position=g_data.get('init_position', [0,0,0]),
                    max_nodes=g_data.get('max_nodes', 100)
                )
                
                next_graph_id = max(next_graph_id, g_data['graph_id'] + 1)

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
                print(f"Graph '{graph.graph_name}' loaded and populated.")


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
        """
        Sammelt Daten aus allen Recordern, speichert sie in history und LEERT die Recorder.
        """
        print(f"\n>>> COLLECTING & FLUSHING DATA <<<")
        timestamp = datetime.now().isoformat()
        
        has_data = False
        
        for graph in graph_list:
            for node in graph.node_list:
                if "history" not in node.results:
                    node.results["history"] = []
                
                # Container für diesen Snapshot
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
                    
                    # Nur Recorder interessieren uns hier für Daten
                    if "recorder" in model or "meter" in model:
                        try:
                            # 1. Daten holen
                            status = nest.GetStatus(gid)[0]
                            events = status.get('events', {})
                            
                            # Prüfen ob Daten da sind
                            n_events = status.get('n_events', 0)
                            if n_events == 0:
                                continue
                                
                            # 2. Daten bereinigen (numpy -> list)
                            cleaned_events = {}
                            for k, v in events.items():
                                if isinstance(v, np.ndarray):
                                    cleaned_events[k] = v.tolist()
                                else:
                                    cleaned_events[k] = v
                            
                            # 3. Speichern
                            run_data["devices"][dev_id] = {
                                "model": model,
                                "type": "recorder",
                                "data": cleaned_events
                            }
                            has_data = True
                            
                            # 4. PUFFER LEEREN (WICHTIG!)
                            nest.SetStatus(gid, {'n_events': 0})
                            print(f"  -> {model} (ID {dev_id}): Collected & Flushed {n_events} events.")
                            
                        except Exception as e:
                            print(f"Error collecting {model}: {e}")

                if run_data["devices"]:
                    node.results["history"].append(run_data)
        
        if has_data:
            print(">>> Data collection complete.")
            # Statusbar update
            if hasattr(self, 'status_bar'):
                self.status_bar.show_success("Simulation data collected & flushed.")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_mode(app)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
