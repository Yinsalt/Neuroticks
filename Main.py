import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSlider, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QStackedWidget, QMessageBox, QProgressBar,
    QGridLayout, QFileDialog,QDoubleSpinBox,QTreeWidgetItemIterator,QInputDialog,
    QFrame, QDialog, QScrollArea, QDialogButtonBox
)
from ExtraTab import ExtraTabWidget
import WidgetLib
# OtherTab is a user-editable sandbox in OtherTab.py. If the file is
# missing or has an error, fall back to a tiny placeholder so the app
# still starts.
try:
    from OtherTab import OtherTabWidget
    _OTHER_TAB_OK = True
except Exception as _other_err:
    print(f"OtherTab.py not loadable ({_other_err}); using placeholder.")
    _OTHER_TAB_OK = False
import pyqtgraph.dockarea as dock
from PyQt6.QtGui import QColor, QPalette, QAction
from PyQt6.QtCore import Qt,QSize
import numpy as np

import time
from CustomExtension import CustomTabWidget
import vtk
import pyvista as pv
from pyvistaqt import QtInteractor

# Retina-Test-Tab (optional - fails gracefully if retina modules missing)
try:
    from RetinaTestTab import RetinaTestTabWidget
    HAS_RETINA_TEST_TAB = True
except Exception as _retina_tab_err:
    HAS_RETINA_TEST_TAB = False
    print(f"[Main] RetinaTestTab nicht geladen: {_retina_tab_err}")
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

def _invalidate_nest_refs(graphs):
    """Remove auto-devices and null runtime_gids before ResetKernel to prevent stale NodeCollection access."""
    for graph in graphs:
        for node in graph.node_list:
            if hasattr(node, 'devices'):
                node.devices = [d for d in node.devices if not d.get('is_auto', False)]
                for d in node.devices:
                    d['runtime_gid'] = None
            if hasattr(node, 'parameters') and 'devices' in node.parameters:
                node.parameters['devices'] = [d for d in node.parameters['devices'] if not d.get('is_auto', False)]
                for d in node.parameters['devices']:
                    d['runtime_gid'] = None
            if hasattr(node, 'population'):
                node.population = []


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
        
        file_menu.addSeparator()
        
        # ── LOAD (kernel reset, replaces current session) ───────────
        load_action = QAction("Load Graph", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_all_graphs_dialog)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        # ── SAVE (current session → file) ────────────────────────────
        save_fat_action = QAction("Save as Fat Graph", self)
        save_fat_action.setShortcut("Ctrl+S")
        save_fat_action.setStatusTip(
            "Save with positions. Larger file, exact reproduction."
        )
        save_fat_action.triggered.connect(self.save_all_graphs_dialog)
        file_menu.addAction(save_fat_action)
        
        save_thin_action = QAction("Save as Thin Graph", self)
        save_thin_action.setShortcut("Ctrl+Shift+S")
        save_thin_action.setStatusTip(
            "Save without positions (or WFC-only). Loader regenerates "
            "via build() — frees you from position-constraint editing."
        )
        save_thin_action.triggered.connect(self.save_thin_graphs_dialog)
        file_menu.addAction(save_thin_action)
        
        file_menu.addSeparator()
        
        # ── IMPORT (additive, no kernel reset) ───────────────────────
        import_fat_action = QAction("Import Fat Subgraph…", self)
        import_fat_action.setShortcut("Ctrl+I")
        import_fat_action.setStatusTip(
            "Append a graph file (with positions) to the current session. "
            "Full feature support (anisotropic / spatial / distance-dep). "
            "Auto-shifts graph_ids on collision."
        )
        import_fat_action.triggered.connect(self.merge_graphs_dialog)
        file_menu.addAction(import_fat_action)
        
        import_thin_action = QAction("Import Thin Subgraph…", self)
        import_thin_action.setShortcut("Ctrl+Shift+I")
        import_thin_action.setStatusTip(
            "Append a thin graph file (positions empty → rebuilt via "
            "build()) to the current session. Toolbox-based loader."
        )
        import_thin_action.triggered.connect(self.load_thin_graphs_dialog)
        file_menu.addAction(import_thin_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        settings_menu = menubar.addMenu("Settings")
        nest_menu = menubar.addMenu("NEST Settings")
        self.action_auto_save = QAction("Auto-Save Live History on Reset", self)
        self.action_auto_save.setCheckable(True)
        self.action_auto_save.setChecked(True)
        settings_menu.addAction(self.action_auto_save)
        self.plasticity_action = QAction("Structural Plasticity", self)
        self.plasticity_action.setCheckable(True)
        self.plasticity_action.setChecked(self.structural_plasticity_enabled)
        self.plasticity_action.triggered.connect(self.toggle_structural_plasticity)
        nest_menu.addAction(self.plasticity_action)
        
        nest_menu.addSeparator()
        
        reset_action = QAction("Reset NEST Kernel", self)
        reset_action.triggered.connect(self.manual_nest_reset)
        nest_menu.addAction(reset_action)
        
        view_menu = menubar.addMenu("View")
        refresh_action = QAction("Refresh Visualizations", self)
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
        if hasattr(self, 'sim_timer'):
            self.sim_timer.stop()
        if hasattr(self, 'simulation_view'):
            self.simulation_view.stop_rendering_safe()
        
        try:
            global graph_list
            graph_list.clear()
            
            WidgetLib.next_graph_id = 0
            WidgetLib.graph_parameters.clear()
            if hasattr(self, 'tools_widget'):
                self.tools_widget.update_graphs(graph_list)
  
            
            print("Resetting NEST Kernel...")
            self.live_recorders = []
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
        if hasattr(self, 'sim_timer'):
            self.sim_timer.stop()
        if hasattr(self, 'simulation_view'):
            self.simulation_view.stop_rendering_safe()
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
        
        # Invalidate stale refs before ResetKernel
        if hasattr(self, 'sim_timer'):
            self.sim_timer.stop()
        self.live_recorders = []
        _invalidate_nest_refs(list(self.active_graphs.values()))
        
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

            # RetinaTestTab informieren dass seine NodeCollections jetzt tot sind
            # (und den virtual_graph aus graph_list entfernen BEVOR repopulate laeuft).
            if hasattr(self, 'retina_test_tab') and self.retina_test_tab is not None:
                try:
                    self.retina_test_tab.on_nest_kernel_reset()
                except Exception as _e:
                    print(f"retina_test_tab.on_nest_kernel_reset error: {_e}")

            # Invalidate stale NodeCollection refs before ResetKernel
            if hasattr(self, 'sim_timer'):
                self.sim_timer.stop()
            self.live_recorders = []
            _invalidate_nest_refs(graph_list)
            
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
        
        self.btn_view_editor = QPushButton("EDITOR")
        self.btn_view_simulation = QPushButton("INTERACTIVE")
        self.btn_view_data = QPushButton("LIVE PLOTS")
        self.btn_view_custom = QPushButton("HISTORY")
        self.btn_view_extra = QPushButton("FUSION TOOL") 
        self.btn_view_eye = QPushButton("EYE")
        for btn in [self.btn_view_editor, self.btn_view_simulation, self.btn_view_data, self.btn_view_custom, self.btn_view_extra, self.btn_view_eye]: 
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self.view_switch_style_inactive)

        self.btn_view_editor.clicked.connect(lambda: self._switch_main_view(0))
        self.btn_view_simulation.clicked.connect(lambda: self._switch_main_view(1))
        self.btn_view_data.clicked.connect(lambda: self._switch_main_view(2))
        self.btn_view_custom.clicked.connect(lambda: self._switch_main_view(3))
        self.btn_view_extra.clicked.connect(lambda: self._switch_main_view(4))
        self.btn_view_eye.clicked.connect(lambda: self._switch_main_view(5))
        
        self.view_buttons = [
            self.btn_view_editor, 
            self.btn_view_simulation, 
            self.btn_view_data, 
            self.btn_view_custom,
            self.btn_view_extra,
            self.btn_view_eye,
        ]
        
        switch_layout.addWidget(self.btn_view_editor)
        switch_layout.addWidget(self.btn_view_simulation)
        switch_layout.addWidget(self.btn_view_data)
        switch_layout.addWidget(self.btn_view_custom)
        switch_layout.addWidget(self.btn_view_extra) 
        switch_layout.addWidget(self.btn_view_eye)

        self.global_sim_control = QWidget()
        self.global_sim_control.setStyleSheet("background-color: transparent;")
        gsc_layout = QHBoxLayout(self.global_sim_control)
        gsc_layout.setContentsMargins(0, 0, 0, 0)
        gsc_layout.setSpacing(8)
        
        self.global_btn_save_data = QPushButton("Save Data")
        self.global_btn_save_data.setCursor(Qt.CursorShape.PointingHandCursor)
        self.global_btn_save_data.setStyleSheet("""
            QPushButton { background-color: #00897B; color: white; font-weight: bold; border: 1px solid #00695C; border-radius: 3px; padding: 5px 10px; }
            QPushButton:hover { background-color: #26A69A; }
        """)
        self.global_btn_save_data.clicked.connect(self.manual_save_data_dialog)
        gsc_layout.addWidget(self.global_btn_save_data)

        lbl_step = QLabel("Res:"); lbl_step.setStyleSheet("color:#888;")
        self.global_step_spin = QDoubleSpinBox()
        self.global_step_spin.setRange(0.1, 1000); self.global_step_spin.setValue(25.0); self.global_step_spin.setSuffix(" ms")
        self.global_step_spin.setFixedWidth(70); self.global_step_spin.setStyleSheet("background:#333; color:#00E5FF; border:1px solid #555;")
        
        self.global_time_label = QLabel("0.0 ms")
        self.global_time_label.setFixedWidth(180)
        self.global_time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.global_time_label.setStyleSheet("background-color: #000; color: #00FF00; font-family: Consolas; font-size: 14px; font-weight: bold; border: 1px solid #444; padding: 2px;")
        
        btn_style = "font-weight: bold; border-radius: 3px; padding: 5px 10px;"
        
        self.global_btn_start = QPushButton("▶")
        self.global_btn_start.setToolTip("Run Continuous")
        self.global_btn_start.setStyleSheet(f"{btn_style} background-color: #2E7D32; color: white;")
        self.global_btn_start.clicked.connect(self._global_start)
        
        self.global_btn_pause = QPushButton("⏸")
        self.global_btn_pause.setStyleSheet(f"{btn_style} background-color: #FBC02D; color: black;")
        self.global_btn_pause.clicked.connect(self._global_pause)
        
        self.global_btn_reset = QPushButton("↺")
        self.global_btn_reset.setToolTip("Reset Kernel & Visuals")
        self.global_btn_reset.setStyleSheet(f"{btn_style} background-color: #E65100; color: white;")
        self.global_btn_reset.clicked.connect(self._global_reset)
        
        gsc_layout.addWidget(lbl_step)
        gsc_layout.addWidget(self.global_step_spin)
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
        self.extra_tab = ExtraTabWidget(graph_list, self) 
        self.data_view = self._create_data_view() 
        self.main_stack.addWidget(self.editor_widget)   # Index 0
        self.main_stack.addWidget(self.simulation_view) # Index 1
        self.main_stack.addWidget(self.live_dashboard)  # Index 2
        self.main_stack.addWidget(self.custom_tab)      # Index 3
        self.main_stack.addWidget(self.extra_tab)       # Index 4 

        # Index 5: Retina-Test-Tab (falls ladbar)
        if HAS_RETINA_TEST_TAB:
            self.retina_test_tab = RetinaTestTabWidget(graph_list, main_window=self)
            self.retina_test_tab.requestVizRefresh.connect(self._on_retina_viz_refresh)
            self.retina_test_tab.retinaBuilt.connect(self._on_retina_viz_refresh)
            self.retina_test_tab.retinaDestroyed.connect(self._on_retina_viz_refresh)
            self.main_stack.addWidget(self.retina_test_tab)
        else:
            self.retina_test_tab = None
            # Placeholder falls RetinaTestTab nicht geladen werden konnte
            _placeholder = QLabel("EYE tab unavailable - retina_main.py not found in PYTHONPATH")
            _placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            _placeholder.setStyleSheet("color: #888; font-size: 16px;")
            self.main_stack.addWidget(_placeholder)
        
        main_layout.addWidget(self.main_stack)

        self.update_visualizations()
        self.init_simulation_timer()
        self._update_global_button_state('stopped')  # FIX: Ensure buttons are in stopped state on startup
        self._switch_main_view(0)

    def _create_data_view(self):
        self.data_dashboard = AnalysisDashboard(graph_list)
        return self.data_dashboard
    def manual_save_data_dialog(self):
        default_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filename, ok = QInputDialog.getText(self, "Save Data", "Filename (without .json):", text=default_name)
        
        if ok and filename:
            if not filename.endswith('.json'):
                filename += ".json"
            self.archive_live_data(filename)





    def archive_live_data(self, custom_filename=None):

        print("Archiving Simulation Data...")
        try:
            raw_live_data = self.live_dashboard.get_all_data()
            
            if not raw_live_data:
                self.status_bar.show_error("No live data found to save.")
                return

            if custom_filename:
                filename = custom_filename
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"history_live_{timestamp}.json"
            
            if not filename.endswith('.json'):
                filename += '.json'
            
            output_dir = Path("Simulation_History")
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / filename

            history_data = {
                'meta': {
                    'version': '3.0',
                    'type': 'neuroticks_history',
                    'timestamp': str(datetime.now()),
                },
                'graphs': []
            }

            total_measurements = 0

            for graph in graph_list:
                # Skip virtuelle Graphen (RetinaTestTab) im History-Export
                if getattr(graph, 'is_virtual', False):
                    continue
                g_data = {
                    'graph_id': graph.graph_id,
                    'graph_name': getattr(graph, 'graph_name', f'Graph_{graph.graph_id}'),
                    'nodes': []
                }

                for node in graph.node_list:
                    n_data = {
                        'id': node.id,
                        'name': node.name,
                        'neuron_models': getattr(node, 'neuron_models', []),
                        'types': getattr(node, 'types', []),
                        'devices': [],
                        'results': {
                            'history': []
                        }
                    }
                    
                    all_devices = getattr(node, 'devices', [])
                    if not all_devices and hasattr(node, 'parameters') and 'devices' in node.parameters:
                        all_devices = node.parameters['devices']
                    
                    run_entry = {'devices': {}}
                    
                    for dev in all_devices:
                        dev_id = dev.get('id', 0)
                        dev_model = dev.get('model', 'unknown')
                        runtime_gid = dev.get('runtime_gid')
                        
                        dev_config = dev.copy()
                        if 'runtime_gid' in dev_config:
                            del dev_config['runtime_gid']
                        if 'params' in dev_config:
                            dev_config['params'] = _clean_params(dev_config['params'])
                        n_data['devices'].append(dev_config)
                        
                        search_gid = None
                        if runtime_gid is not None:
                            try:
                                if hasattr(runtime_gid, 'tolist'):
                                    search_gid = runtime_gid.tolist()[0]
                                elif isinstance(runtime_gid, list):
                                    search_gid = runtime_gid[0]
                                else:
                                    search_gid = int(runtime_gid)
                            except:
                                pass
                        
                        if search_gid is not None and search_gid in raw_live_data:
                            rec_data = raw_live_data[search_gid]
                            
                            clean_data = {}
                            for k, v in rec_data.items():
                                if k == 'values' and isinstance(v, dict):
                                    for vk, vv in v.items():
                                        clean_data[vk] = vv.tolist() if hasattr(vv, 'tolist') else list(vv) if hasattr(vv, '__iter__') and not isinstance(vv, str) else vv
                                else:
                                    clean_data[k] = v.tolist() if hasattr(v, 'tolist') else v
                            
                            run_entry['devices'][str(dev_id)] = {
                                'model': dev_model,
                                'data': clean_data
                            }
                            total_measurements += 1
                    
                    if run_entry['devices']:
                        n_data['results']['history'].append(run_entry)
                    
                    g_data['nodes'].append(n_data)
                
                history_data['graphs'].append(g_data)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, cls=NumpyEncoder, indent=2)
                
            self.status_bar.show_success(f"Saved {total_measurements} device recordings to {filename}")
            print(f"Saved history with {total_measurements} device recordings.")
            
            if hasattr(self, 'custom_tab'):
                self.custom_tab.refresh_history_list()
            
        except Exception as e:
            print(f"Error archiving data: {e}")
            import traceback
            traceback.print_exc()
            self.status_bar.show_error(f"Save Error: {e}")
    
    
    
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

        if index == 4:
            if hasattr(self, 'extra_tab'):
                self.extra_tab.on_tab_active()

        if index == 5:
            if hasattr(self, 'retina_test_tab') and self.retina_test_tab is not None:
                try:
                    self.retina_test_tab.on_tab_activated()
                except Exception as e:
                    print(f"retina_test_tab.on_tab_activated error: {e}")


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
        # create_graph_visualization() sets self.graph_plotter (the
        # QtInteractor) internally and returns the wrapper widget that
        # contains the header + plotter + detail panel. We store the
        # wrapper separately so we don't overwrite self.graph_plotter.
        self.graph_skeleton_wrapper = self.create_graph_visualization()
        self.blink_widget = BlinkingNetworkWidget(graph_list)
        self.flow_widget = FlowFieldWidget(graph_list)
        
        self.vis_stack.addWidget(self.neuron_plotter)
        self.vis_stack.addWidget(self.graph_skeleton_wrapper)
        self.vis_stack.addWidget(self.blink_widget)
        self.vis_stack.addWidget(self.flow_widget)
        self.sim_dashboard = SimulationDashboardWidget(graph_list)
        
        self.sim_dashboard.requestStartSimulation.connect(self.start_headless_simulation)
        self.sim_dashboard.requestStopSimulation.connect(self.stop_headless_simulation)
        self.sim_dashboard.requestResetKernel.connect(self.manual_nest_reset)
        
        self.vis_stack.addWidget(self.sim_dashboard)
        if _OTHER_TAB_OK:
            self.other_tab = OtherTabWidget(graph_list=graph_list)
        else:
            self.other_tab = QLabel(
                "OtherTab.py konnte nicht geladen werden — "
                "siehe Konsole für Details."
            )
            self.other_tab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.other_tab.setStyleSheet(
                "background: #2a1a1a; color: #ff8888; font-size: 14px;"
            )
        self.vis_stack.addWidget(self.other_tab)
        
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

        print(f"\n>>> STARTING ROBUST HEADLESS SIMULATION ({duration} ms) <<<")
        
        self.sim_dashboard.set_ui_locked(True)
        self.simulation_view.setEnabled(False)
        self.btn_view_simulation.setEnabled(False)
        
        self.status_bar.set_status("HEADLESS SIMULATION RUNNING...", color="#E65100")
        self.status_bar.set_progress(0)
        QApplication.processEvents()
        
        if hasattr(self, 'blink_widget') and self.blink_widget:
            self.blink_widget.stop_simulation()
            if hasattr(self.blink_widget, 'timer'): self.blink_widget.timer.stop()
            
        if hasattr(self, 'simulation_view') and self.simulation_view:
            self.simulation_view.stop_rendering_safe()

        self._ensure_spike_recorders()
        
        try:
            kernel_time = self._get_nest_time()
            self.headless_target_time = kernel_time + duration
            self.headless_start_time = kernel_time
        except: 
            self.headless_target_time = duration
            self.headless_start_time = 0.0


        min_chunk = 50.0
        max_chunk = 500.0
        dynamic_chunk = duration / 50.0 
        self.headless_step_size = max(min_chunk, min(max_chunk, dynamic_chunk))
        
        print(f"   -> Optimization: Chunk size set to {self.headless_step_size:.1f} ms")

        self.headless_timer = QTimer()
        self.headless_timer.timeout.connect(self.headless_loop_step)
        self.headless_timer.start(0)

    def headless_loop_step(self):
        try:
            current_time = self._get_nest_time()
            
            remaining = self.headless_target_time - current_time
            
            if remaining <= 0.001:
                self.finish_headless_simulation()
                return

            step_to_take = min(self.headless_step_size, remaining)

            self._simulate_step_and_update(step_to_take)

            total_duration = self.headless_target_time - self.headless_start_time
            if total_duration > 0:
                elapsed = current_time - self.headless_start_time + step_to_take
                prog = int((elapsed / total_duration) * 100)
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
            
        self.collect_simulation_results(0)
        self._restore_ui_after_headless()

    def finish_headless_simulation(self):
        self.headless_timer.stop()
        
        final_time = self._get_nest_time()
        self.update_global_time_display(final_time)
        
        print(f">>> Headless Simulation Finished at {final_time:.1f} ms.")
        self.status_bar.set_status("Collecting Data...", "#1976D2")
        QApplication.processEvents()
        
        self.collect_simulation_results(0)
        
        self.status_bar.show_success(f"Headless Run Complete ({final_time:.1f} ms). Check Data Tab.")
        self._restore_ui_after_headless()

    def _restore_ui_after_headless(self):
        self.sim_dashboard.set_ui_locked(False)
        self.simulation_view.setEnabled(True)
        self.btn_view_simulation.setEnabled(True)
        self.status_bar.set_progress(100)
        
        if self.main_stack.currentIndex() == 1: 
             if hasattr(self, 'blink_widget') and self.blink_widget:
                 self.blink_widget.start_simulation()
                 if hasattr(self.blink_widget, 'timer'): self.blink_widget.timer.start(30)
    


    def reset_from_live_dashboard(self, keep_data):
        print("Live Dashboard Reset triggered.")
        self.sim_timer.stop()
        
        if self.action_auto_save.isChecked():
            self.archive_live_data()
            
        self.reset_and_restart()
        self.simulation_view.update_time_display(0.0)


    

    
        
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

    def _on_retina_viz_refresh(self):
        """Triggert Redraw der Live-Viz nach Retina-Build oder Frame-Tick.

        Die BlinkingNetworkWidget aktualisiert sich selbst via Timer, aber bei
        strukturellen Aenderungen (Retina neu gebaut / zerstoert) muss build_scene()
        erneut laufen, damit die virtual_graph-Nodes mitgerendert werden.
        """
        try:
            if hasattr(self, 'blink_widget') and self.blink_widget is not None:
                # Nur bei strukturellen Aenderungen rebuild - nicht bei jedem Frame-Tick
                # Der Caller (retinaBuilt/Destroyed) signalisiert das.
                if self.sender() is not None and self.sender().__class__.__name__ == 'RetinaTestTabWidget':
                    # Signal name checken via sender()... einfacher: immer rebuild
                    pass
                self.blink_widget.build_scene()
            # Neuron- und Graph-Plotter ebenfalls aktualisieren wenn gerade sichtbar
            if self.vis_stack.currentIndex() == 0:
                self.plot_neuron_points()
            elif self.vis_stack.currentIndex() == 1:
                self.plot_graph_skeleton()
        except Exception as e:
            print(f"viz refresh error: {e}")

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
        """Returns a wrapper widget: graph plotter with a top header bar
        (label-toggle) and an overlay detail-panel. The actual plotter
        sits at self.graph_plotter; helpers below access it directly."""
        wrapper = QWidget()
        v = QVBoxLayout(wrapper)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # Header bar with the label-toggle
        header = QWidget()
        header.setStyleSheet("background: #1a1a1a; border-bottom: 1px solid #444;")
        header.setFixedHeight(34)
        h = QHBoxLayout(header)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(8)

        title = QLabel("Graph Skeleton")
        title.setStyleSheet("color: #FFD700; font-weight: bold;")
        h.addWidget(title)
        h.addStretch()

        self.graph_label_toggle = QPushButton("Labels: OFF")
        self.graph_label_toggle.setCheckable(True)
        self.graph_label_toggle.setChecked(False)
        self.graph_label_toggle.setFixedWidth(110)
        self.graph_label_toggle.setStyleSheet("""
            QPushButton { background: #333; color: #ccc; border: 1px solid #555;
                          border-radius: 3px; padding: 3px 8px; font-size: 11px; }
            QPushButton:checked { background: #2E7D32; color: white;
                                  border: 1px solid #4CAF50; }
        """)
        self.graph_label_toggle.toggled.connect(self._on_graph_labels_toggled)
        h.addWidget(self.graph_label_toggle)

        v.addWidget(header)

        # The actual PyVista plotter
        self.graph_plotter = QtInteractor(wrapper)
        self.graph_plotter.set_background('black')
        self.graph_plotter.add_axes()
        v.addWidget(self.graph_plotter, 1)

        # Floating detail panel — sits on top of the plotter, hidden until
        # a node/connection is clicked. Click-outside closes it.
        self.graph_detail_panel = QFrame(wrapper)
        self.graph_detail_panel.setStyleSheet("""
            QFrame {
                background: rgba(20, 20, 20, 235);
                border: 2px solid #FFD700;
                border-radius: 6px;
            }
            QLabel { color: #eee; }
        """)
        self.graph_detail_panel.setMinimumWidth(320)
        self.graph_detail_panel.setMaximumWidth(420)
        dp_layout = QVBoxLayout(self.graph_detail_panel)
        dp_layout.setContentsMargins(12, 10, 12, 10)
        dp_layout.setSpacing(6)

        dp_header = QHBoxLayout()
        self.graph_detail_title = QLabel("Details")
        self.graph_detail_title.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #FFD700;"
        )
        dp_header.addWidget(self.graph_detail_title)
        dp_header.addStretch()
        close_btn = QPushButton("×")
        close_btn.setFixedSize(22, 22)
        close_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #ccc; "
            "border: none; font-size: 16px; font-weight: bold; } "
            "QPushButton:hover { color: #FF3333; }"
        )
        close_btn.clicked.connect(self._hide_graph_detail_panel)
        dp_header.addWidget(close_btn)
        dp_layout.addLayout(dp_header)

        self.graph_detail_text = QLabel("")
        self.graph_detail_text.setWordWrap(True)
        self.graph_detail_text.setTextFormat(Qt.TextFormat.RichText)
        self.graph_detail_text.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px; color: #ddd;"
        )
        self.graph_detail_text.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        dp_layout.addWidget(self.graph_detail_text)

        self.graph_detail_panel.hide()
        # Make wrapper own the panel so we can position it absolutely later
        self._graph_skeleton_wrapper = wrapper

        return wrapper
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
        
        # Connection Editor — separate page for editing existing connections.
        # Routed to from the graph-overview when the user clicks an arrow.
        # Trigger network rebuild on update so NEST-side reflects the
        # changed synapse model / weight / spatial spec.
        self.connection_editor = ConnectionEditorWidget(graph_list=graph_list)
        self.connection_editor.connectionUpdated.connect(self._on_connection_edited)
        self.tool_stack.addWidget(self.connection_editor)
        
        # The "Population Examples" tab now hosts an Examples-folder browser.
        # The old StructuresWidget (cortical regions) is still instantiated
        # for backward compat (callers might use it via self.structures_widget)
        # but is not added to the tool_stack — its slot belongs to examples.
        self.structures_widget = StructuresWidget()
        self.structures_widget.structureSelected.connect(self.on_structure_selected)
        self.examples_widget = _ExamplesPage(main_window=self)
        self.tool_stack.addWidget(self.examples_widget)
        
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
        self.set_tool_page(5)
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
        # Invalidate stale refs before ResetKernel
        _invalidate_nest_refs(graph_list)
        
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
            self.set_tool_page(1)
            self.graph_editor.delete_connection_by_data(conn_data)
            self.status_bar.show_success(f"Connection '{name}' deleted.")
            self.update_visualizations()

    def set_tool_page(self, index):
        """Zentrale Methode zum Wechseln der Tool-Seite mit Navigation-Sync."""
        self.tool_stack.setCurrentIndex(index)
        
        # Synchronisiere Navigation-Buttons
        if hasattr(self, 'nav_btns') and self.nav_btns:
            for i, btn in enumerate(self.nav_btns):
                if i == index:
                    btn.setStyleSheet(self.nav_active_style)
                else:
                    btn.setStyleSheet(self.nav_base_style)

    def _on_overview_node_selected(self, graph_id, node_id):
        
        self.set_tool_page(1)
        self.graph_editor.select_node_by_id(graph_id, node_id)
        
        if hasattr(self, 'flow_widget'):
            self.flow_widget.set_target_node(graph_id, node_id)
    def open_connection_tool_for_node(self, graph_id, node_id, pop_id=0):

        print(f"Context Menu Action: Setting Source to Graph {graph_id}, Node {node_id}, Pop {pop_id}")
        
        self.set_tool_page(2)
        
        self.connection_tool.refresh()
        
        self.connection_tool.set_source(graph_id, node_id, pop_id)
        
        self.status_bar.set_status(f"Source preset: G{graph_id} N{node_id} P{pop_id}", color="#2196F3")


    def _on_overview_pop_selected(self, graph_id, node_id, pop_id):
        self.set_tool_page(1)
        self.graph_editor.select_population_by_ids(graph_id, node_id, pop_id)


    def _on_overview_conn_selected(self, connection_data):
        # Open the standalone Connection Editor on tool-page 3.
        # Used to dispatch into Graph Inspector — that path got too crowded
        # so editing existing connections lives on its own page now.
        self.set_tool_page(3)
        self.connection_editor.load_connection(connection_data)
    
    def _on_connection_edited(self, conn_data):
        """Called when ConnectionEditor.apply_changes runs. Triggers a
        rebuild so NEST reflects the new synapse model / weight / spatial
        spec. Connection data is already mutated in place — we just need
        to invalidate the existing NEST projections and recreate them."""
        try:
            self.status_bar.set_status(
                f"Rebuilding network after connection edit: {conn_data.get('name','?')}",
                color="#673AB7",
            )
            QApplication.processEvents()
            # Full rebuild path — same one used after device updates.
            _invalidate_nest_refs(graph_list)
            nest.ResetKernel()
            if getattr(self, 'structural_plasticity_enabled', False):
                try: nest.EnableStructuralPlasticity()
                except Exception: pass
            self.rebuild_all_graphs(reset_nest=False, verbose=False)
            self.update_visualizations()
            self.graph_overview.update_tree()
            self.status_bar.show_success("Connection updated.")
        except Exception as e:
            print(f"Rebuild after connection edit failed: {e}")
            self.status_bar.show_error("Rebuild failed; see console.")


    
    def on_structure_selected(self, name, models, probs):
        
        self.set_tool_page(0)
        
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

        self.nav_base_style = """
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

        self.nav_active_style = """
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
            ("Connection Creator", 2),
            ("Connection Editor", 3),
            ("Population Examples", 4),
            ("Instrumentation", 5)
        ]

        for label, idx in nav_items:
            btn = QPushButton(label)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            
            btn.setStyleSheet(self.nav_base_style)
            
            btn.clicked.connect(lambda checked, i=idx: self.set_tool_page(i))
            
            self.nav_btns.append(btn)
            col1_layout.addWidget(btn)

        if self.nav_btns:
            self.nav_btns[0].setStyleSheet(self.nav_active_style)

        col2_layout = QVBoxLayout()
        col2_layout.setSpacing(5)
        col2_layout.addWidget(create_header("KERNEL"))
        
        ops_items = [
            ("Refresh Connectivity", self.reconnect_network),
            ("Reinstantiate Network", self.rebuild_all_graphs)
        ]
        
        for label, func in ops_items:
            btn = QPushButton(label)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            btn.setStyleSheet(action_btn_style)
            btn.clicked.connect(func)
            col2_layout.addWidget(btn)

        # ── GRAPH I/O column ─────────────────────────────────────────
        # Aufgespalten in 5 explizite Aktionen damit der Mental Model
        # klar ist:
        #   IMPORT (additiv)    : append an laufende Session, kein
        #                         Kernel-Reset, IDs werden automatisch
        #                         geshiftet wenn Kollision droht.
        #   LOAD                : ersetzt Session komplett (Kernel-Reset).
        #   SAVE THIN/FAT       : aktuelle Session → Datei.
        # Thin = ohne (oder nur WFC) Positionen, build() rekonstruiert.
        # Fat  = mit Positionen, exakte Reproduktion.
        col3_layout = QVBoxLayout()
        col3_layout.setSpacing(5)
        col3_layout.addWidget(create_header("GRAPH I/O"))
        
        # Distinct styles damit der User die drei Aktionsklassen optisch
        # unterscheidet (Import = additiv, Load = destruktiv, Save = output)
        import_btn_style = """
            QPushButton {
                background-color: #1B5E20; color: #C8E6C9;
                border: 1px solid #2E7D32; border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2E7D32; border: 1px solid #66BB6A; color: white; }
        """
        load_btn_style = """
            QPushButton {
                background-color: #4A148C; color: #E1BEE7;
                border: 1px solid #6A1B9A; border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background-color: #6A1B9A; border: 1px solid #BA68C8; color: white; }
        """
        # io_btn_style is already defined above (used for Save buttons here)
        
        graph_io_items = [
            # (label, callback, style, tooltip)
            ("Import Thin Subgraph", self.load_thin_graphs_dialog, import_btn_style,
             "Append a thin graph (positions empty → rebuilt). Toolbox loader."),
            ("Import Fat Subgraph",  self.merge_graphs_dialog,      import_btn_style,
             "Append a fat graph (with positions). Full feature support."),
            ("Load Graph",           self.load_all_graphs_dialog,   load_btn_style,
             "Replace current session with a project file (kernel reset)."),
            ("Save as Thin Graph",   self.save_thin_graphs_dialog,  io_btn_style,
             "Save current session without positions (or WFC-only auto)."),
            ("Save as Fat Graph",    self.save_all_graphs_dialog,   io_btn_style,
             "Save current session with positions."),
        ]
        
        for label, func, style, tip in graph_io_items:
            btn = QPushButton(label)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            btn.setStyleSheet(style)
            btn.setToolTip(tip)
            btn.clicked.connect(func)
            col3_layout.addWidget(btn)

        # Stretch-Faktoren angepasst: GRAPH I/O hat jetzt 5 Buttons,
        # bekommt etwas mehr Platz.
        main_layout.addLayout(col1_layout, 4)
        main_layout.addLayout(col2_layout, 2)
        main_layout.addLayout(col3_layout, 4)

        wrapper_layout = QHBoxLayout()
        wrapper_layout.setContentsMargins(0,0,0,0)
        wrapper_layout.addWidget(main_container)
        
        return wrapper_layout
    


    def save_all_graphs_dialog(self):
        if not graph_list:
            QMessageBox.warning(self, "Save Error", "No graphs to save!")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save as Fat Graph", "", "JSON Files (*.json);;All Files (*)"
        )

        if filepath:
            if not filepath.endswith('.json'):
                filepath += '.json'
            
            try:
                project_data = {
                    'meta': {
                        'version': '2.1',
                        'type': 'neuroticks_project',
                        'timestamp': str(np.datetime64('now'))
                    },
                    'graphs': []
                }

                for graph in graph_list:
                    # Skip virtuelle Graphen (z.B. RetinaTestTab): nicht exportieren
                    if getattr(graph, 'is_virtual', False):
                        continue
                    nodes_data = []
                    for node in graph.node_list:
                        
                        cleaned_devices = []
                        source_devices = getattr(node, 'devices', [])
                        if not source_devices and 'devices' in node.parameters:
                            source_devices = node.parameters['devices']

                        for dev in source_devices:
                            dev_copy = dev.copy()
                            if 'runtime_gid' in dev_copy: del dev_copy['runtime_gid']
                            if 'params' in dev_copy: dev_copy['params'] = _clean_params(dev_copy['params'])
                            cleaned_devices.append(dev_copy)

                        safe_params = node.parameters.copy() if hasattr(node, 'parameters') else {}
                        if 'devices' in safe_params: del safe_params['devices']

                        parent_id = None
                        if hasattr(node, 'parent') and node.parent is not None:
                            parent_id = node.parent.id

                        node_data = {
                            'id': node.id,
                            'name': node.name,
                            'graph_id': graph.graph_id,
                            'parameters': _clean_params(safe_params),
                            
                            'positions': [pos.tolist() if isinstance(pos, np.ndarray) else list(pos)
                                          for pos in node.positions] if hasattr(node, 'positions') and node.positions else [],
                            'center_of_mass': list(node.center_of_mass) if hasattr(node, 'center_of_mass') else [0,0,0],
                            
                            'connections': _serialize_connections(node.connections) if hasattr(node, 'connections') else [],
                            'devices': cleaned_devices,
                            
                            'types': node.types if hasattr(node, 'types') else [],
                            'neuron_models': node.neuron_models if hasattr(node, 'neuron_models') else [],
                            'distribution': list(node.distribution) if hasattr(node, 'distribution') and node.distribution else [],
                            
                            'parent_id': parent_id,
                            'next_ids': [n.id for n in node.next] if hasattr(node, 'next') else [],
                            'prev_ids': [n.id for n in node.prev] if hasattr(node, 'prev') else [],
                        }
                        nodes_data.append(node_data)

                    graph_data = {
                        'graph_id': graph.graph_id,
                        'graph_name': getattr(graph, 'graph_name', f'Graph_{graph.graph_id}'),
                        'max_nodes': graph.max_nodes,
                        'init_position': list(graph.init_position) if hasattr(graph, 'init_position') else [0,0,0],
                        'polynom_max_power': graph.polynom_max_power if hasattr(graph, 'polynom_max_power') else 5,
                        'polynom_decay': graph.polynom_decay if hasattr(graph, 'polynom_decay') else 0.8, 
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

            # FIX: Auch den Editor-State aktualisieren, sonst kommt das Device beim naechsten
            # save_changes zurueck (_build_node_params bevorzugt raw_params)
            if hasattr(self, 'graph_editor') and self.graph_editor.current_graph_id == gid:
                for nd in self.graph_editor.node_list:
                    if nd['params'].get('id') == nid:
                        devs = nd['params'].get('devices', [])
                        nd['params']['devices'] = [d for d in devs if str(d.get('id')) != str(dev_id)]
                        break

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
        
        self.skeleton_info_map = {}
        self.skeleton_label_actors = []  # actors that toggle with the label button
        self.highlighted_actor = None
        self.original_color = None
        self.original_opacity = None
        
        self.tooltip_actor = vtk.vtkTextActor()
        self.tooltip_actor.GetTextProperty().SetFontSize(14)
        self.tooltip_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0) 
        self.tooltip_actor.GetTextProperty().SetBackgroundColor(0.0, 0.0, 0.0) 
        self.tooltip_actor.GetTextProperty().SetBackgroundOpacity(0.7)
        self.tooltip_actor.GetTextProperty().SetFontFamilyToArial()
        self.tooltip_actor.GetTextProperty().BoldOn()
        self.tooltip_actor.SetVisibility(False)
        self.graph_plotter.renderer.AddViewProp(self.tooltip_actor)
        
        node_map = {}
        for graph in graph_list:
            if graph.graph_id in self.hidden_graphs: continue
            for node in graph.node_list:
                node_map[(graph.graph_id, node.id)] = node

        import random
        import matplotlib.pyplot as plt

        labels_visible = (
            self.graph_label_toggle.isChecked()
            if hasattr(self, 'graph_label_toggle') else False
        )

        for graph in graph_list:
            if graph.graph_id in self.hidden_graphs: continue

            cmap = plt.get_cmap("tab20")
            rgba = cmap(graph.graph_id % 20)
            graph_node_color = rgba[:3] 

            # Optional: graph-name label, placed above the highest node of
            # the graph. Only added when label-toggle is on.
            if labels_visible and graph.node_list:
                node_centers = np.array([
                    nd.center_of_mass for nd in graph.node_list
                    if (graph.graph_id, nd.id) not in self.hidden_nodes
                ])
                if len(node_centers) > 0:
                    top_pt = node_centers.mean(axis=0).copy()
                    top_pt[2] = node_centers[:, 2].max() + 1.6
                    g_name = getattr(graph, 'graph_name', f'Graph_{graph.graph_id}')
                    try:
                        actor = self.graph_plotter.add_point_labels(
                            np.array([top_pt]),
                            [g_name],
                            font_size=14,
                            text_color="#FFD700",
                            point_color=None, point_size=0,
                            shape=None, shape_opacity=0.0,
                            show_points=False, always_visible=True,
                            pickable=False,
                            name=f"glabel_g{graph.graph_id}"
                        )
                        self.skeleton_label_actors.append(actor)
                    except Exception:
                        pass
            
            for node in graph.node_list:
                if (graph.graph_id, node.id) in self.hidden_nodes: continue

                center = np.array(node.center_of_mass)
                
                sphere = pv.Sphere(radius=0.6, center=center)
                node_actor = self.graph_plotter.add_mesh(
                    sphere,
                    color=graph_node_color,
                    opacity=0.9,
                    smooth_shading=True,
                    pickable=True
                )
                
                n_pops = len(node.population) if hasattr(node, 'population') and node.population else 0
                dev_count = len(node.devices) if hasattr(node, 'devices') else 0
                if not dev_count and 'devices' in node.parameters: dev_count = len(node.parameters['devices'])

                # Sum neuron count from populations (also if some are None)
                total_neurons = 0
                if hasattr(node, 'population') and node.population:
                    for pop in node.population:
                        if pop is not None:
                            try:
                                total_neurons += len(pop)
                            except (TypeError, AttributeError):
                                pass

                info_text = (f"NODE: {node.name}\n"
                             f"ID: {node.id} (Graph {graph.graph_id})\n"
                             f"Populations: {n_pops}\n"
                             f"Neurons: {total_neurons}\n"
                             f"Devices: {dev_count}\n"
                             f"Pos: {center.round(2)}")
                # Build a richer info dict for click-panel + plain text for hover
                self.skeleton_info_map[node_actor] = {
                    'kind': 'node',
                    'short': info_text,
                    'detail_html': self._format_node_detail_html(
                        graph, node, n_pops, total_neurons, dev_count, center
                    ),
                    'title': f"Node: {node.name}",
                }

                # Permanent label only when toggle is on. When off, no label
                # actor is created at all (cleaner than visibility hacks).
                if labels_visible:
                    label_text = f"{node.name}\n({total_neurons}n)"
                    label_pos = center + np.array([0.0, 0.0, 0.85])
                    try:
                        lbl_actor = self.graph_plotter.add_point_labels(
                            np.array([label_pos]), [label_text],
                            font_size=11,
                            text_color="white",
                            point_color=None,
                            point_size=0,
                            shape=None,
                            shape_opacity=0.0,
                            show_points=False,
                            always_visible=True,
                            pickable=False,
                            name=f"label_g{graph.graph_id}_n{node.id}"
                        )
                        self.skeleton_label_actors.append(lbl_actor)
                    except Exception:
                        try:
                            lbl_actor = self.graph_plotter.add_point_labels(
                                np.array([label_pos]), [label_text],
                                font_size=11, text_color="white",
                                show_points=False, always_visible=True
                            )
                            self.skeleton_label_actors.append(lbl_actor)
                        except Exception:
                            pass

                if dev_count > 0:
                    aura = pv.Sphere(radius=0.9, center=center)
                    self.graph_plotter.add_mesh(aura, color="#FFD700", opacity=0.15, pickable=False)

                if hasattr(node, 'connections') and node.connections:
                    # Group connections by (target_graph, target_node) so we
                    # can fan multiple connections out across distinct
                    # control-points instead of overlapping a single spline.
                    pair_buckets = {}
                    for conn in node.connections:
                        tgt_id = (
                            int(conn.get('target', {}).get('graph_id', -1)),
                            int(conn.get('target', {}).get('node_id', -1)),
                        )
                        pair_buckets.setdefault(tgt_id, []).append(conn)

                    for (tgt_gid, tgt_nid), conns_to_pair in pair_buckets.items():
                        if (tgt_gid, tgt_nid) not in node_map: continue
                        if tgt_gid in self.hidden_graphs: continue
                        if (tgt_gid, tgt_nid) in self.hidden_nodes: continue
                        target_node = node_map[(tgt_gid, tgt_nid)]

                        for conn_idx, conn in enumerate(conns_to_pair):
                            try:
                                self._draw_single_connection(
                                    graph, node, target_node, conn,
                                    conn_idx, len(conns_to_pair),
                                    tgt_gid, tgt_nid,
                                )
                            except Exception as e:
                                print(f"Error plotting connection: {e}")

        try:
            self.graph_plotter.iren.remove_observer(self._observer_tag)
        except: pass
        
        self._observer_tag = self.graph_plotter.iren.add_observer(
            "MouseMoveEvent", self._on_skeleton_hover
        )
        # Click handler for the detail panel
        try:
            self.graph_plotter.iren.remove_observer(self._click_observer_tag)
        except: pass
        self._click_observer_tag = self.graph_plotter.iren.add_observer(
            "LeftButtonPressEvent", self._on_skeleton_click
        )

        self.graph_plotter.reset_camera()
        self.graph_plotter.update()

    def _draw_single_connection(self, graph, node, target_node, conn,
                                 conn_idx, total_for_pair, tgt_gid, tgt_nid):
        """Draws ONE connection as its own spline+arrowhead. Multiple
        connections between the same node pair fan out across distinct
        control-points so they don't overlap. Each gets its own pickable
        hitbox so hover/click identifies that specific connection."""
        params = conn.get('params', {})
        weight = float(params.get('weight', 1.0))
        conn_name = conn.get('name', 'Connection')

        if weight > 0:
            edge_color = "#FF3333"
            conn_type = "Excitatory (+)"
        elif weight < 0:
            edge_color = "#3366FF"
            conn_type = "Inhibitory (-)"
        else:
            edge_color = "#888888"
            conn_type = "Silent (0)"

        start = np.array(node.center_of_mass)
        end = np.array(target_node.center_of_mass)

        delay_val = params.get('delay', '?')
        rule = params.get('rule', '?')
        p_val = params.get('p', None)
        src_pop = conn.get('source', {}).get('pop_id', '?')
        tgt_pop = conn.get('target', {}).get('pop_id', '?')
        synapse_model = params.get('synapse_model', '?')

        conn_short = (f"CONNECTION: {conn_name}\n"
                      f"Type: {conn_type}\n"
                      f"From: {node.name} (G{graph.graph_id} N{node.id} P{src_pop})\n"
                      f"To:   {target_node.name} (G{tgt_gid} N{tgt_nid} P{tgt_pop})\n"
                      f"Weight: {weight}\n"
                      f"Delay:  {delay_val} ms\n"
                      f"Model: {synapse_model}\n"
                      f"Rule:  {rule}" + (f" (p={p_val})" if p_val is not None else ""))

        conn_detail_html = self._format_connection_detail_html(
            conn_name, conn_type, edge_color,
            graph, node, src_pop,
            target_node, tgt_gid, tgt_nid, tgt_pop,
            weight, delay_val, synapse_model, rule, p_val,
            params, conn_idx, total_for_pair,
        )
        click_payload = {
            'kind': 'connection',
            'short': conn_short,
            'detail_html': conn_detail_html,
            'title': f"Connection: {conn_name}",
        }

        if node == target_node:
            # Self-loop: ring on top of the node, slight tilt per
            # conn_idx so multiple self-loops are visible.
            tilt_deg = 90 + (conn_idx - (total_for_pair - 1) / 2) * 10
            torus = pv.ParametricTorus(ringradius=0.7, crosssectionradius=0.07)
            try:
                torus.rotate_x(tilt_deg, inplace=True)
                torus.translate(start + np.array([0.0, 0.0, 0.65 + 0.15 * conn_idx]),
                                inplace=True)
            except TypeError:
                torus = torus.rotate_x(tilt_deg)
                torus = torus.translate(
                    start + np.array([0.0, 0.0, 0.65 + 0.15 * conn_idx])
                )
            actor_self = self.graph_plotter.add_mesh(
                torus, color=edge_color, opacity=0.85, pickable=True
            )
            self.skeleton_info_map[actor_self] = click_payload
            return

        # Fan-out: each connection gets a distinct control-point so multiple
        # parallel connections don't overlap. Offset is index-based, not
        # random, so re-renders are stable.
        mid_point = (start + end) / 2.0
        # Build an axis perpendicular to the edge direction to fan along
        edge_dir = end - start
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-6:
            edge_dir_n = np.array([1.0, 0.0, 0.0])
        else:
            edge_dir_n = edge_dir / edge_len
        # Pick a stable perp axis — cross with world-up, fall back to X
        world_up = np.array([0.0, 0.0, 1.0])
        perp = np.cross(edge_dir_n, world_up)
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(edge_dir_n, np.array([1.0, 0.0, 0.0]))
        perp = perp / max(np.linalg.norm(perp), 1e-6)

        # Spread angle around the mid-point, centered around 0
        if total_for_pair == 1:
            spread = np.array([0.0, 0.0, 1.2])  # single conn: small upward bow
        else:
            theta = (conn_idx / max(total_for_pair - 1, 1)) * np.pi - np.pi / 2
            radius_lift = 1.5 + 0.4 * total_for_pair
            spread = perp * np.cos(theta) * radius_lift + world_up * (
                np.sin(theta) * radius_lift * 0.4 + 0.6
            )
        control_point = mid_point + spread

        points = np.array([start, control_point, end])
        spline = pv.Spline(points, n_points=20)

        self.graph_plotter.add_mesh(
            spline, color=edge_color, line_width=2.0,
            render_lines_as_tubes=True, pickable=False
        )

        # Invisible thicker tube for picking
        hitbox = spline.tube(radius=0.25)
        actor_hitbox = self.graph_plotter.add_mesh(
            hitbox, color=edge_color, opacity=0.0, pickable=True
        )
        self.skeleton_info_map[actor_hitbox] = click_payload

        # Arrow head at the target end, offset along spline tangent
        last_pt = spline.points[-1]
        prev_pt = spline.points[-3]
        direction = last_pt - prev_pt
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        cone_pos = end - (direction * 0.8)
        arrow_head = pv.Cone(
            center=cone_pos, direction=direction,
            height=0.5, radius=0.2, resolution=12
        )
        self.graph_plotter.add_mesh(arrow_head, color=edge_color, pickable=False)

    def _format_node_detail_html(self, graph, node, n_pops, total_neurons,
                                  dev_count, center):
        g_name = getattr(graph, 'graph_name', f'Graph_{graph.graph_id}')
        rows = [
            f"<b>Graph:</b> {g_name} (id={graph.graph_id})",
            f"<b>Node:</b> {node.name} (id={node.id})",
            f"<b>Populations:</b> {n_pops}",
            f"<b>Total neurons:</b> {total_neurons}",
            f"<b>Devices:</b> {dev_count}",
            f"<b>Center of mass:</b> [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]",
        ]
        # Per-pop neuron model + count
        if hasattr(node, 'neuron_models') and node.neuron_models:
            rows.append("<br><b>Per-population:</b>")
            for i, model in enumerate(node.neuron_models):
                pop_n = 0
                if hasattr(node, 'population') and i < len(node.population):
                    p = node.population[i]
                    if p is not None:
                        try:
                            pop_n = len(p)
                        except Exception:
                            pop_n = 0
                rows.append(f"&nbsp;&nbsp;Pop {i}: {model} ({pop_n}n)")
        n_outgoing = len(node.connections) if hasattr(node, 'connections') else 0
        rows.append(f"<br><b>Outgoing connections:</b> {n_outgoing}")
        return "<br>".join(rows)

    def _format_connection_detail_html(self, conn_name, conn_type, edge_color,
                                        graph, src_node, src_pop,
                                        tgt_node, tgt_gid, tgt_nid, tgt_pop,
                                        weight, delay_val, synapse_model, rule,
                                        p_val, params, conn_idx, total_for_pair):
        g_name = getattr(graph, 'graph_name', f'Graph_{graph.graph_id}')
        type_html = f"<span style='color:{edge_color};'>{conn_type}</span>"
        rows = [
            f"<b>Name:</b> {conn_name}",
            f"<b>Type:</b> {type_html}",
            "",
            f"<b>From:</b> {src_node.name} (Pop {src_pop})",
            f"&nbsp;&nbsp;<i>Graph: {g_name} (id={graph.graph_id}), Node id={src_node.id}</i>",
            f"<b>To:</b> {tgt_node.name} (Pop {tgt_pop})",
            f"&nbsp;&nbsp;<i>Graph id={tgt_gid}, Node id={tgt_nid}</i>",
            "",
            f"<b>Weight:</b> {weight}",
            f"<b>Delay:</b> {delay_val} ms",
            f"<b>Synapse model:</b> {synapse_model}",
            f"<b>Rule:</b> {rule}" + (f" (p={p_val})" if p_val is not None else ""),
        ]
        # Append any additional params not covered above
        extras = {
            k: v for k, v in (params or {}).items()
            if k not in ('weight', 'delay', 'synapse_model', 'rule', 'p',
                         'allow_autapses', 'allow_multapses', 'receptor_type',
                         'use_spatial')
        }
        if extras:
            rows.append("")
            rows.append("<b>Extra synapse params:</b>")
            for k, v in extras.items():
                rows.append(f"&nbsp;&nbsp;{k} = {v}")
        if total_for_pair > 1:
            rows.append("")
            rows.append(
                f"<i>Connection {conn_idx + 1} of {total_for_pair} between this pair</i>"
            )
        return "<br>".join(rows)

    def _on_graph_labels_toggled(self, checked):
        """Re-renders the skeleton with or without labels."""
        if hasattr(self, 'graph_label_toggle'):
            self.graph_label_toggle.setText("Labels: ON" if checked else "Labels: OFF")
        # Easiest path: rebuild the skeleton (cheap for normal graph counts).
        if hasattr(self, 'graph_plotter') and hasattr(self, 'plot_graph_skeleton'):
            try:
                self.plot_graph_skeleton()
            except Exception as e:
                print(f"label toggle rebuild error: {e}")

    def _on_skeleton_click(self, interactor, event):
        """Click on a node/connection → show the detail panel. Click on
        empty space → hide it."""
        x, y = interactor.GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(x, y, 0, self.graph_plotter.renderer)
        actor = picker.GetActor()
        if actor and actor in self.skeleton_info_map:
            payload = self.skeleton_info_map[actor]
            if isinstance(payload, dict):
                self._show_graph_detail_panel(
                    payload.get('title', 'Details'),
                    payload.get('detail_html', payload.get('short', '')),
                )
            else:
                # Backward-compat: old code stored plain strings
                self._show_graph_detail_panel("Details", str(payload))
        else:
            self._hide_graph_detail_panel()

    def _show_graph_detail_panel(self, title, html_body):
        if not hasattr(self, 'graph_detail_panel'):
            return
        self.graph_detail_title.setText(title)
        self.graph_detail_text.setText(html_body)
        # Position the panel in the top-right corner of the wrapper
        wrapper = self._graph_skeleton_wrapper
        self.graph_detail_panel.adjustSize()
        pw = self.graph_detail_panel.sizeHint().width()
        # Anchor 12px from the right edge, 50px below the header
        x = max(10, wrapper.width() - pw - 12)
        y = 46
        self.graph_detail_panel.setGeometry(x, y, pw, self.graph_detail_panel.sizeHint().height())
        self.graph_detail_panel.raise_()
        self.graph_detail_panel.show()

    def _hide_graph_detail_panel(self):
        if hasattr(self, 'graph_detail_panel'):
            self.graph_detail_panel.hide()
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
                if self.original_color: prop.SetColor(self.original_color)
                if self.original_opacity: prop.SetOpacity(self.original_opacity)
            except AttributeError: pass
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
                
                payload = self.skeleton_info_map[actor]
                # Tooltip uses the short plain-text form. Click-panel uses
                # the html form. For backward-compat, accept plain strings.
                if isinstance(payload, dict):
                    text_content = payload.get('short', '')
                else:
                    text_content = str(payload)
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
                self._simulate_step_and_update(step_size)
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
        for graph in graph_list:
            # Skip virtuelle Graphs (RetinaTestTab hat eigene Recorder)
            if getattr(graph, 'is_virtual', False):
                continue
            for node in graph.node_list:
                if hasattr(node, 'devices'):
                    for dev in node.devices:
                        model = dev.get('model', '')
                        if 'recorder' in model or 'meter' in model:
                            gid = dev.get('runtime_gid')
                            if gid is not None:
                                # Keep as NodeCollection for nest.GetStatus compatibility
                                if not hasattr(gid, 'tolist') and not isinstance(gid, list):
                                    # It's a raw int - wrap it
                                    try:
                                        gid = nest.NodeCollection([int(gid)])
                                    except Exception:
                                        continue
                                if gid not in self.live_recorders:
                                    self.live_recorders.append(gid)
        non_spiking = [
            'siegert_neuron', 'mcculloch_pitts_neuron',
            'rate_neuron_ipn', 'rate_neuron_opn', 'gif_pop_psc_exp',
            'ht_neuron'
        ]
        
        for graph in graph_list:
            for node in graph.node_list:
                if hasattr(node, 'population'):
                    for pop_idx, pop in enumerate(node.population):
                        if pop is None: continue
                        
                        try:
                            model = nest.GetStatus(pop, 'model')[0]
                            if model in non_spiking:
                                continue
                            

                            if not hasattr(node, 'devices'): node.devices = []
                            if 'devices' not in node.parameters: node.parameters['devices'] = []
                            
                            existing = [d for d in node.devices if d.get('target_pop_id') == pop_idx and d.get('is_auto', False)]
                            
                            if existing:
                                rec_entry = existing[0]
                                rec = rec_entry.get('runtime_gid')

                                try:
                                    nest.GetStatus(rec) 
                                except:
                                    rec = None
                                    rec_entry['runtime_gid'] = None
                                
                                if rec:
                                    self.live_recorders.append(rec)
                                    continue

                            rec = nest.Create("spike_recorder")
                            nest.SetStatus(rec, {"record_to": "memory"})
                            nest.Connect(pop, rec)
                            self.live_recorders.append(rec)
                            
                            auto_device_entry = {
                                "id": f"auto_{graph.graph_id}_{node.id}_{pop_idx}", 
                                "model": "spike_recorder",
                                "target_pop_id": pop_idx,
                                "params": {"label": "auto_visual_recorder"},
                                "conn_params": {},
                                "runtime_gid": rec,
                                "is_auto": True 
                            }
                            
                            # FIX: Doppel-Append vermeiden bei shared list reference
                            if 'devices' not in node.parameters:
                                node.parameters['devices'] = []
                            node.parameters['devices'].append(auto_device_entry)
                            if node.devices is not node.parameters['devices']:
                                node.devices.append(auto_device_entry)
                            
                        except Exception as e:
                            print(f"Warning creating live recorder: {e}")
                            
        if hasattr(self, 'live_dashboard'):
            QTimer.singleShot(100, self.live_dashboard.scan_for_devices)

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
        self.start_continuous_simulation(step, max_duration=0)
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
        self.global_time_label.setText("0.0 ms")
    
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
            time_ms = self._get_nest_time()

        self.global_time_label.setText(f"{float(time_ms):.1f} ms")
        # Forciere UI-Repaint: setText + repaint() reicht in PyQt6 nicht
        # immer wenn der main-thread durch nest.Simulate blockiert wird.
        # processEvents() gibt Qt explizit Zeit zum painten.
        self.global_time_label.update()
        self.global_time_label.repaint()
        QApplication.processEvents()
    
    @staticmethod
    def _get_nest_time():
        """Liest die aktuelle NEST-Simulationszeit. NEST 3.x hat den Wert
        unter 'biological_time'; 'time' ist deprecated/leer und gibt 0.0
        zurück, was den Timer einfrieren ließ."""
        try:
            stat = nest.GetKernelStatus()
            return float(stat.get('biological_time', stat.get('time', 0.0)) or 0.0)
        except Exception:
            return 0.0
    
    def _simulate_step_and_update(self, step):
        """Wrapper um nest.Simulate(step) der den globalen Timer mit der
        akkumulierten Sim-Zeit aktualisiert. Jeder Pfad in Neuroticks der
        NEST live laufen lässt (continuous-loop, live-run, headless) sollte
        diesen Helper benutzen — sonst bleibt das global_time_label hängen.
        Reset auf 0 passiert ohnehin in reset_and_restart()."""
        nest.Simulate(float(step))
        # NEST-Zeit auslesen + lokale Akkumulation als Backup
        local_t = float(getattr(self, 'current_nest_time', 0.0)) + float(step)
        nest_t = self._get_nest_time()
        self.current_nest_time = max(local_t, nest_t)
        self.update_global_time_display(self.current_nest_time)
        return self.current_nest_time
    def update_simulation_speed(self, slider_value):
        if hasattr(self, 'sim_timer'):
            self.sim_timer.setInterval(slider_value)


    def start_continuous_simulation(self, step_size, max_duration=None):

        step_val = self.global_step_spin.value()
        
        self.sim_mode = 'continuous'
        self.sim_step_size = step_val
        
        self.current_nest_time = self._get_nest_time()
        
        # Initiales Label-Update damit User sieht von wo wir starten
        self.update_global_time_display(self.current_nest_time)
            
        print(f"Starting Continuous Run from {self.current_nest_time} ms")

        self.sim_target_time = float('inf')
        
        if max_duration is not None and max_duration > 0:
             self.sim_target_time = self.current_nest_time + max_duration

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

            self._simulate_step_and_update(step_to_take)
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

            # RetinaTestTab informieren BEVOR ResetKernel, damit es seinen State clearen
            # und den virtual_graph aus graph_list entfernen kann.
            if hasattr(self, 'retina_test_tab') and self.retina_test_tab is not None:
                try:
                    self.retina_test_tab.on_nest_kernel_reset()
                except Exception as _e:
                    print(f"retina_test_tab.on_nest_kernel_reset error: {_e}")

            # Invalidate stale NodeCollection refs before ResetKernel
            _invalidate_nest_refs(graph_list)

            nest.ResetKernel()
            
            if self.structural_plasticity_enabled:
                nest.EnableStructuralPlasticity()
            self.current_nest_time = 0.0
            self.update_global_time_display(0.0)
            self._update_global_button_state('stopped')
            
            print("RESET COMPLETE: TIME 0.0")
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

        visual_only = False
        if hasattr(self, 'simulation_view') and self.simulation_view.chk_visual_only.isChecked():
            visual_only = True

        all_events_cache = {}
        recorders_to_clear = []
        
        if hasattr(self, 'live_recorders') and self.live_recorders:
            for rec in self.live_recorders:
                try:
                    status = nest.GetStatus(rec)[0]
                    n_events = status.get('n_events', 0)
                    if n_events > 0:
                        events = status.get('events', {})
                        events_copy = {}
                        for k, v in events.items():
                            if hasattr(v, 'copy'):
                                events_copy[k] = v.copy()
                            elif hasattr(v, 'tolist'):
                                events_copy[k] = np.array(v)  
                            else:
                                events_copy[k] = v
                        
                        nest_id = rec.tolist()[0] if hasattr(rec, 'tolist') else int(rec[0]) if isinstance(rec, list) else int(rec)
                        all_events_cache[nest_id] = events_copy
                        recorders_to_clear.append(rec)
                except Exception:
                    pass

        if hasattr(self, 'simulation_view') and self.simulation_view.isVisible():
            visual_spikes_flat = []
            for nest_id, events in all_events_cache.items():
                if 'senders' in events:
                    senders = events['senders']
                    if hasattr(senders, 'tolist'):
                        visual_spikes_flat.extend(senders.tolist())
                    else:
                        visual_spikes_flat.extend(list(senders))
            
            if visual_spikes_flat:
                self.simulation_view.feed_spikes(visual_spikes_flat)

        QApplication.processEvents()


        for rec in recorders_to_clear:
            try:
                nest.SetStatus(rec, {'n_events': 0})
            except:
                pass

        if visual_only:
            return

        live_data_snapshot = {}
        has_dashboard_data = False
        
        graphs = getattr(self, 'active_graphs', {}).values()
        if not graphs: 
            graphs = graph_list

        process_counter = 0

        for graph in graphs:
            # Skip virtuelle Graphs (RetinaTestTab bewirtschaftet eigene Recorder)
            if getattr(graph, 'is_virtual', False):
                continue
            for node in graph.node_list:
                process_counter += 1
                if process_counter % 10 == 0:
                    QApplication.processEvents()

                if not hasattr(node, 'devices') or not node.devices:
                    continue

                # Safety-Net: results-Attribut sicherstellen
                if not hasattr(node, 'results') or node.results is None:
                    node.results = {}
                if "history" not in node.results:
                    node.results["history"] = []
                
                step_record = {
                    "time": sim_time,
                    "devices": {}
                }
                data_in_step = False

                for dev in node.devices:
                    gid = dev.get('runtime_gid')
                    dev_id = dev.get('id')
                    model = dev.get('model', '')
                    
                    if gid is None: 
                        continue
                    
                    try:
                        nest_id = gid
                        if isinstance(gid, list):
                            nest_id = gid[0]
                        elif hasattr(gid, 'tolist'):
                            nest_id = gid.tolist()[0]
                    except Exception:
                        continue  # Stale NodeCollection → skip

                    if "recorder" in model or "meter" in model:
                        if nest_id in all_events_cache:
                            events = all_events_cache[nest_id]
                            
                            clean_events = {}
                            for k, v in events.items():
                                if hasattr(v, 'tolist'): 
                                    clean_events[k] = v.tolist()
                                else: 
                                    clean_events[k] = list(v) if hasattr(v, '__iter__') else v
                            
                            step_record["devices"][str(dev_id)] = {
                                "type": model,
                                "events": clean_events
                            }
                            data_in_step = True
                            
                            live_data_snapshot[nest_id] = events
                            has_dashboard_data = True
                
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

                # Invalidate stale NodeCollection refs before ResetKernel
                if hasattr(self, 'sim_timer'):
                    self.sim_timer.stop()
                self.live_recorders = []
                _invalidate_nest_refs(graphs_to_process)

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




    def fixed_save_all_graphs_dialog(self):
        """Korrigierte Version der save_all_graphs_dialog Methode."""
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
                        'version': '2.1', 
                        'type': 'neuroticks_project',
                        'timestamp': str(np.datetime64('now'))
                    },
                    'graphs': []
                }

                for graph in graph_list:
                    # Skip virtuelle Graphen (z.B. RetinaTestTab): nicht exportieren
                    if getattr(graph, 'is_virtual', False):
                        continue
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
                            'devices': cleaned_devices,
                            'types': node.types if hasattr(node, 'types') else [],
                            'neuron_models': node.neuron_models if hasattr(node, 'neuron_models') else [],
                            'distribution': list(node.distribution) if hasattr(node, 'distribution') and node.distribution else [],
                            'parent_id': node.parent.id if hasattr(node, 'parent') and node.parent else None,
                            'next_ids': [n.id for n in node.next] if hasattr(node, 'next') else [],
                            'prev_ids': [n.id for n in node.prev] if hasattr(node, 'prev') else [],
                        }
                        nodes_data.append(node_data)

                    graph_data = {
                        'graph_id': graph.graph_id,
                        'graph_name': getattr(graph, 'graph_name', f'Graph_{graph.graph_id}'),
                        'max_nodes': graph.max_nodes,
                        'init_position': list(graph.init_position),
                        'polynom_max_power': graph.polynom_max_power,
                        'polynom_decay': graph.polynom_decay if hasattr(graph, 'polynom_decay') else 0.8, 
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
            self, "Load Graph", "", "JSON Files (*.json);;All Files (*)"
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
            
            # Stop simulation and invalidate stale refs before ResetKernel
            global graph_list
            if hasattr(self, 'sim_timer'):
                self.sim_timer.stop()
            self.live_recorders = []
            _invalidate_nest_refs(graph_list)
            
            nest.ResetKernel()
            
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
                    
                    if 'types' in nd and 'types' not in params:
                        params['types'] = nd['types']
                    if 'neuron_models' in nd and 'neuron_models' not in params:
                        params['neuron_models'] = nd['neuron_models']

                    if 'devices' in nd:
                        params['devices'] = nd['devices']
                    
                    if 'center_of_mass' in nd:
                        params['center_of_mass'] = np.array(nd['center_of_mass'])
                    
                    # FIX: Ensure population_nest_params is correctly loaded
                    # Priority: nd['parameters']['population_nest_params'] > nd['population_nest_params']
                    if 'population_nest_params' not in params or not params['population_nest_params']:
                        # Check if it's at top level (older format)
                        if 'population_nest_params' in nd and nd['population_nest_params']:
                            params['population_nest_params'] = nd['population_nest_params']
                    
                    # Debug output for parameter loading
                    pop_params = params.get('population_nest_params', [])
                    if pop_params:
                        print(f"  Loading Node {nd['id']} ({nd['name']}): {len(pop_params)} pop_nest_params")
                        for idx, p in enumerate(pop_params):
                            if p:
                                print(f"    Pop {idx}: V_th={p.get('V_th', 'N/A')}, C_m={p.get('C_m', 'N/A')}, t_ref={p.get('t_ref', 'N/A')}")
                    
                    new_node = graph.create_node(
                        parameters=params,
                        is_root=(nd['id']==0),
                        auto_build=False
                    )
                    
                    # NUR Struktur + Positionen hier setzen — populate_node
                    # läuft am Ende zentral via rebuild_all_graphs(reset_nest=
                    # False). Damit verhalten sich Load-Graph und
                    # Reinstantiate-Network exakt gleich, und es gibt
                    # keine doppelten NEST-Populationen.
                    saved_positions = nd.get('positions')
                    if saved_positions:
                        new_node.positions = [np.array(pos) for pos in saved_positions]
                    # else: positions bleiben leer — rebuild_all_graphs ruft
                    # build() für jeden Node mit leerer self.positions auf.
                    
                    if nd.get('distribution'):
                        new_node.distribution = nd['distribution']
                    
                    # population_nest_params auf den Node syncen — das sieht
                    # populate_node() später als Fallback wenn parameters
                    # leer sind.
                    new_node.population_nest_params = params.get('population_nest_params', [])

                graph_list.append(graph)
                print(f"Graph '{graph.graph_name}' loaded (ID: {gid}).")


            print("Reconstructing graph topology...")
            # Build a flat lookup over ALL graphs so we can resolve
            # cross-graph targets when populating prev/next from
            # node.connections. Without this, prev only contains
            # same-graph predecessors and Node.remove() can't clean up
            # cross-graph links → dangling refs after delete.
            global_lookup = {}
            for g_obj in graph_list:
                for n_obj in g_obj.node_list:
                    global_lookup[(g_obj.graph_id, n_obj.id)] = n_obj
            
            for g_data in project_data['graphs']:
                gid = g_data['graph_id']
                current_graph_obj = next((g for g in graph_list if g.graph_id == gid), None)
                
                if not current_graph_obj:
                    continue
                
                node_map = {n.id: n for n in current_graph_obj.node_list}

                for nd in g_data['nodes']:
                    node_id = nd['id']
                    node_obj = node_map.get(node_id)
                    
                    if not node_obj: continue

                    parent_id = nd.get('parent_id')
                    if parent_id is not None and parent_id in node_map:
                        node_obj.parent = node_map[parent_id]


                    for nid in nd.get('next_ids', []):
                        if nid in node_map:
                            target_node = node_map[nid]
                            if target_node not in node_obj.next:
                                node_obj.next.append(target_node)
                            if node_obj not in target_node.prev:
                                target_node.prev.append(node_obj)

                    for pid in nd.get('prev_ids', []):
                        if pid in node_map:
                            source_node = node_map[pid]
                            if source_node not in node_obj.prev:
                                node_obj.prev.append(source_node)
                            if node_obj not in source_node.next:
                                source_node.next.append(node_obj)
                    
                    # Cross-graph backref pass: every connection in this
                    # node's connections list represents an edge to
                    # (target_graph_id, target_node_id). Resolve via the
                    # global lookup and call add_neighbor so target.prev
                    # knows about us. add_neighbor is idempotent.
                    for conn in (node_obj.connections or []):
                        try:
                            tgt = conn.get('target', {}) or {}
                            tgt_key = (tgt.get('graph_id'), tgt.get('node_id'))
                            tgt_node = global_lookup.get(tgt_key)
                            if tgt_node is not None:
                                node_obj.add_neighbor(tgt_node)
                        except Exception:
                            pass


            # ─── ZENTRALER POPULATE-PASS via rebuild_all_graphs ───────
            # Statt populate_node inline im Main-Loop (was bei thin-Files
            # mit leeren positions[] für Cone/Blob/CCW/Grid leere
            # Populationen erzeugte) delegieren wir an die identische
            # Logik die der "Reinstantiate Network" Button nutzt:
            #   1. Pro Node: build() falls self.positions leer
            #   2. populate_node() für alle
            # Kernel ist bereits oben mit nest.ResetKernel() gewipt,
            # deshalb reset_nest=False (sonst wären die Topologie-Refs
            # nochmal invalidiert).
            self.status_bar.set_status("Building NEST populations...", color="#FF9800")
            self.status_bar.set_progress(70)
            QApplication.processEvents()
            
            rebuild_stats = self.rebuild_all_graphs(
                target_graphs=graph_list,
                reset_nest=False,
                rebuild_positions=False,
                verbose=True,
            )
            print(f"[Load] Rebuild summary: {rebuild_stats['nodes_rebuilt']} nodes, "
                  f"{rebuild_stats['populations_created']} populations, "
                  f"{len(rebuild_stats['errors'])} errors")
            if rebuild_stats['errors']:
                for err in rebuild_stats['errors'][:10]:
                    print(f"  ✗ {err}")

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



    # ════════════════════════════════════════════════════════════════
    #  THIN-MODE LOADING / SAVING
    #  Append-only Loader für Graphen aus 'thin' JSONs (positions
    #  optional). Kein Kernel-Reset, kein Pos-Constraint — Graphen
    #  werden additiv an die laufende Session angehängt. graph_ids
    #  werden automatisch um einen Offset verschoben falls eine
    #  Kollision mit existierenden Graphen droht. Connection-Targets
    #  innerhalb der geladenen Menge werden mit-remappt; cross-graph-
    #  refs auf bereits geladene Graphen bleiben gültig.
    # ════════════════════════════════════════════════════════════════

    def load_thin_graphs_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Thin Graph(s)", "", "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return
        
        global graph_list
        try:
            self.status_bar.set_status("Loading thin graph(s)...", color="#2196F3")
            self.status_bar.set_progress(10)
            QApplication.processEvents()
            
            # Peek: welche graph_ids will das File belegen?
            with open(filepath, 'r', encoding='utf-8') as f:
                preview = json.load(f)
            
            if 'graphs' in preview:
                wanted_ids = [int(g.get('graph_id', i)) for i, g in enumerate(preview['graphs'])]
            elif 'graph' in preview:
                wanted_ids = [int(preview['graph'].get('graph_id', 0))]
            else:
                wanted_ids = [0]
            
            existing_ids = {g.graph_id for g in graph_list}
            collision = bool(existing_ids & set(wanted_ids))
            
            if collision and existing_ids:
                # Schiebe alle gewollten IDs um (max(existing)+1 - min(wanted))
                # nach oben — bewahrt relative Abstände, vermeidet Kollisionen.
                offset = (max(existing_ids) + 1) - min(wanted_ids)
            else:
                offset = 0
            
            print(f"\n[ThinLoad] file wants graph_ids={wanted_ids}, "
                  f"existing={sorted(existing_ids)}, offset={offset}")
            
            self.status_bar.set_progress(25)
            QApplication.processEvents()
            
            # Toolbox macht den Heavy-Lifting Teil: Nodes erstellen,
            # Positionen (build() oder direkt aus JSON), populate_node()
            # für NEST-Populationen + Auto-Devices, Topologie restaurieren.
            # build_connections=False weil wir gleich die WidgetLib-Routine
            # nehmen für vollen Feature-Support (anisotropic / spatial /
            # distance-dependent).
            # populate=False: NEST-Populationen erstellen wir gleich
            # zentral via rebuild_all_graphs — die selbe Routine die der
            # "Reinstantiate Network" Button benutzt. Damit verhalten sich
            # Import-Thin und Reinstantiate-Network exakt gleich.
            new_graphs = Graph.load_all_from_json(
                filepath,
                populate=False,
                build_connections=False,
                id_offset=offset,
                verbose=True,
            )
            
            self.status_bar.set_progress(60)
            QApplication.processEvents()
            
            # Anhängen + WidgetLib-Counter mit-tracken
            for g in new_graphs:
                graph_list.append(g)
                if g.graph_id >= WidgetLib.next_graph_id:
                    WidgetLib.next_graph_id = g.graph_id + 1
            
            # ─── ZENTRALER POPULATE-PASS via rebuild_all_graphs ───────
            # Gleiche Routine wie der "Reinstantiate Network" Button.
            # reset_nest=False, weil wir append-only laden — wir wollen
            # die existierende Session nicht wegwerfen. Pro neuem Node:
            #   1. build() falls self.positions leer (thin-mode)
            #   2. populate_node() erstellt NEST-Populationen
            self.status_bar.set_status("Building NEST populations...", color="#FF9800")
            self.status_bar.set_progress(70)
            QApplication.processEvents()
            
            rebuild_stats = self.rebuild_all_graphs(
                target_graphs=new_graphs,
                reset_nest=False,
                rebuild_positions=False,
                verbose=True,
            )
            print(f"[ThinLoad] Rebuild summary: {rebuild_stats['nodes_rebuilt']} nodes, "
                  f"{rebuild_stats['populations_created']} populations, "
                  f"{len(rebuild_stats['errors'])} errors")
            if rebuild_stats['errors']:
                for err in rebuild_stats['errors'][:10]:
                    print(f"  ✗ {err}")
            
            self.status_bar.set_status("Wiring connections (new graphs only)...")
            self.status_bar.set_progress(80)
            QApplication.processEvents()
            
            # Connections: NUR für die neu geladenen Graphen aufbauen
            # (existierende sind schon im Kernel). Lookups gehen aber
            # gegen die VOLLE Registry — sonst können Cross-Graph-
            # Connections in die existierende Session nicht aufgelöst
            # werden.
            from WidgetLib import validate_connection_params
            full_registry = {g.graph_id: g for g in graph_list}
            created = 0
            failed = 0
            
            for ng in new_graphs:
                for node in ng.node_list:
                    for conn in (node.connections or []):
                        try:
                            src = conn.get('source', {}) or {}
                            tgt = conn.get('target', {}) or {}
                            sg = full_registry.get(int(src.get('graph_id', -1)))
                            tg = full_registry.get(int(tgt.get('graph_id', -1)))
                            if sg is None or tg is None:
                                failed += 1
                                continue
                            sn = sg.get_node(int(src.get('node_id', -1)))
                            tn = tg.get_node(int(tgt.get('node_id', -1)))
                            if sn is None or tn is None:
                                failed += 1
                                continue
                            sp_id = int(src.get('pop_id', 0))
                            tp_id = int(tgt.get('pop_id', 0))
                            if (sp_id >= len(sn.population) or tp_id >= len(tn.population)
                                or sn.population[sp_id] is None or tn.population[tp_id] is None):
                                failed += 1
                                continue
                            sp = sn.population[sp_id]
                            tp = tn.population[tp_id]
                            if len(sp) == 0 or len(tp) == 0:
                                failed += 1
                                continue
                            
                            cs, ss, _w = validate_connection_params(conn.get('params', {}))
                            if cs.get('rule') == 'one_to_one' and len(sp) != len(tp):
                                print(f"  ✗ {conn.get('name', '?')}: one_to_one size mismatch "
                                      f"({len(sp)} vs {len(tp)})")
                                failed += 1
                                continue
                            nest.Connect(sp, tp, cs, ss)
                            created += 1
                        except Exception as e:
                            print(f"  ✗ Connection {conn.get('name', '?')}: {e}")
                            failed += 1
            
            self.status_bar.set_status("Refreshing view...")
            self.status_bar.set_progress(95)
            QApplication.processEvents()
            
            if hasattr(self, 'tools_widget'):
                self.tools_widget.update_graphs(graph_list)
            self.update_visualizations()
            self.graph_overview.update_tree()
            self.connection_tool.refresh()
            self.graph_editor.refresh_graph_list()
            if hasattr(self, 'graph_builder'):
                self.graph_builder.reset()
            
            offset_msg = f" (id_offset={offset})" if offset else ""
            msg = (f"Thin-loaded {len(new_graphs)} graph(s){offset_msg}. "
                   f"Connections: {created} created, {failed} failed.")
            self.status_bar.show_success("Thin graph(s) loaded.")
            print(f"\n{msg}")
            QMessageBox.information(self, "Load Complete", msg)
        
        except Exception as e:
            self.status_bar.show_error(f"Thin load failed: {e}")
            print(f"Thin load failed: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", str(e))

    def save_thin_graphs_dialog(self):
        if not graph_list:
            QMessageBox.warning(self, "Save Error", "No graphs to save!")
            return
        
        # Drei Optionen: Auto (WFC-only), Always-Include, Never-Include.
        # Auto ist der intelligente Default — speichert Positionen NUR
        # für tool_type='custom' (WFC, nicht-deterministisch). Tools wie
        # Cone/Blob/CCW werden auf load() frisch regeneriert (deterministisch
        # aus den Parametern), sparen also Filesize ohne Verlust.
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Save Thin Graph(s)")
        msg.setText("How to handle positions?")
        msg.setInformativeText(
            "• AUTO  = save positions only for WFC-custom nodes (the only "
            "non-deterministic case). Cone/Blob/CCW get rebuilt from "
            "parameters on load. Recommended.\n"
            "• YES   = always save positions (preserves every realization, "
            "largest file).\n"
            "• NO    = never save positions (smallest file, but stochastic "
            "WFC realizations will differ on reload)."
        )
        btn_auto = msg.addButton("Auto (WFC only)", QMessageBox.ButtonRole.AcceptRole)
        btn_yes  = msg.addButton("Always Include", QMessageBox.ButtonRole.YesRole)
        btn_no   = msg.addButton("Thin (No Positions)", QMessageBox.ButtonRole.NoRole)
        msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is btn_auto:
            include_positions = 'auto'
        elif clicked is btn_yes:
            include_positions = True
        elif clicked is btn_no:
            include_positions = False
        else:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Thin Graph(s)", "", "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return
        if not filepath.endswith('.json'):
            filepath += '.json'
        
        # Falls nur ein Graph: schreibe single-graph Format (kompatibel
        # mit WidgetLib.load_graph). Mehrere Graphen → Projekt-Format.
        try:
            real_graphs = [g for g in graph_list if not getattr(g, 'is_virtual', False)]
            if not real_graphs:
                QMessageBox.warning(self, "Save Error", "No real graphs to save (all virtual).")
                return
            
            if len(real_graphs) == 1:
                real_graphs[0].to_json(filepath, include_positions=include_positions, verbose=True)
            else:
                # Projekt-Format: serialisiere jeden Graph einzeln, packe in graphs[].
                # Für 'auto' wird per-Node entschieden (siehe to_json).
                project_data = {
                    'meta': {
                        'version': 'thin-1.0',
                        'type': 'neuroticks_thin_project',
                        'include_positions': include_positions,
                        'timestamp': str(np.datetime64('now')),
                    },
                    'graphs': []
                }
                for g in real_graphs:
                    nodes_data = []
                    for node in g.node_list:
                        if include_positions == 'auto':
                            tt = (node.parameters or {}).get('tool_type', 'custom')
                            include_pos_for_this = (tt == 'custom')
                        else:
                            include_pos_for_this = bool(include_positions)
                        nodes_data.append(g._serialize_node(
                            node,
                            include_positions=include_pos_for_this,
                            include_devices=True,
                            include_connections=True,
                        ))
                    project_data['graphs'].append({
                        'graph_id': int(g.graph_id),
                        'graph_name': g.graph_name,
                        'max_nodes': int(g.max_nodes) if g.max_nodes else len(nodes_data),
                        'init_position': (g.init_position.tolist()
                                          if isinstance(g.init_position, np.ndarray)
                                          else list(g.init_position)),
                        'polynom_max_power': int(g.polynom_max_power),
                        'polynom_decay': float(g.polynom_decay),
                        'nodes': nodes_data,
                    })
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(project_data, f, indent=2)
                tag = ("auto" if include_positions == 'auto'
                       else "with positions" if include_positions
                       else "thin (no positions)")
                print(f"Thin project saved: {len(real_graphs)} graphs → {filepath}  [{tag}]")
            
            self.status_bar.show_success(f"Thin graph(s) saved to {filepath}")
        except Exception as e:
            self.status_bar.show_error(f"Save failed: {e}")
            print(f"Thin save failed: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Save Error", str(e))



    # ═══════════════════════════════════════════════════════════════
    #  EXAMPLES — Examples/ folder browser
    # ═══════════════════════════════════════════════════════════════

    def _examples_dir(self):
        """Return Path to the Examples/ folder. Create it if missing."""
        import os
        from pathlib import Path
        # Sit next to Main.py so it's portable across installs
        here = Path(os.path.dirname(os.path.abspath(__file__)))
        d = here / "Examples"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Examples dir create failed: {e}")
        return d
    
    def _scan_examples(self):
        """
        Scan Examples/ for *.json files. Returns one entry per file:
            (filepath, display_label, graph_count)
        Files that don't parse or contain no graphs are skipped.
        Multi-graph files import as a bundle so cross-graph connections
        between them stay intact.
        """
        import json
        from pathlib import Path
        d = self._examples_dir()
        results = []
        if not d.exists():
            return results
        for fp in sorted(d.glob("*.json")):
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            graphs = data.get('graphs')
            if isinstance(graphs, list) and graphs:
                results.append((str(fp), fp.stem, len(graphs)))
            elif 'graph' in data:
                # Single-graph format
                results.append((str(fp), fp.stem, 1))
        return results
    
    def _refresh_examples_buttons(self):
        """Tells the Examples page to re-scan the folder and rebuild buttons."""
        if hasattr(self, 'examples_widget') and self.examples_widget is not None:
            self.examples_widget.rebuild()
    
    def _on_example_clicked(self, filepath, display_label, graph_count):
        """Ask for a displacement, then import the entire file (all graphs
        with their cross-graph connections preserved) into the scene."""
        import os
        title = os.path.basename(filepath)
        dlg = _DisplacementDialog(
            f"{title} — {graph_count} graph(s)",
            parent=self,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        dx, dy, dz = dlg.get_displacement()
        try:
            self._import_example_file(filepath, np.array([dx, dy, dz], dtype=float))
        except Exception as e:
            self.status_bar.show_error(f"Example import failed: {e}")
            print(f"Example import failed: {e}")
            import traceback; traceback.print_exc()
    
    def _import_example_file(self, filepath, displacement):
        """
        Import ALL graphs in a file as a bundle, shifted by `displacement`.
        Cross-graph connections between graphs in the same file stay intact
        because every old graph_id is rewritten consistently to its new id.
        Cross-graph connections that pointed at IDs not contained in the
        file are left as-is (they would be dangling anyway and get pruned
        on reset).
        """
        import json
        import os
        with open(filepath, 'r', encoding='utf-8') as f:
            project_data = json.load(f)
        
        graphs_to_load = project_data.get('graphs', [])
        if not graphs_to_load and 'graph' in project_data:
            g_only = project_data['graph'].copy()
            g_only['nodes'] = project_data.get('nodes', [])
            graphs_to_load = [g_only]
        if not graphs_to_load:
            raise ValueError(f"No graphs in {os.path.basename(filepath)}")
        
        # Allocate consecutive new IDs for every graph in the file
        next_id = max((g.graph_id for g in graph_list), default=-1) + 1
        id_mapping = {}
        for i, g_data in enumerate(graphs_to_load):
            old_id = int(g_data.get('graph_id', i))
            id_mapping[old_id] = next_id + i
        
        self.status_bar.set_status(
            f"Importing {len(graphs_to_load)} graph(s) from "
            f"{os.path.basename(filepath)}...",
            color="#2E7D32",
        )
        QApplication.processEvents()
        
        new_graphs = []
        for g_data in graphs_to_load:
            old_id = int(g_data.get('graph_id', 0))
            new_graph_id = id_mapping[old_id]
            graph_name = g_data.get('graph_name', f'Imported_{new_graph_id}')
            
            # Shifted init position
            init_pos = np.array(g_data.get('init_position', [0, 0, 0]), dtype=float) + displacement
            
            graph = Graph(
                graph_name=graph_name,
                graph_id=new_graph_id,
                parameter_list=[],
                polynom_max_power=g_data.get('polynom_max_power', 5),
                polynom_decay=g_data.get('polynom_decay', 0.8),
                position=init_pos.tolist(),
                max_nodes=g_data.get('max_nodes', 100),
            )
            
            nodes_data = sorted(g_data.get('nodes', []), key=lambda x: x.get('id', 0))
            
            for nd in nodes_data:
                params = nd.get('parameters', {}).copy()
                params['id'] = nd.get('id', 0)
                params['name'] = nd.get('name', f"Node_{params['id']}")
                params['graph_id'] = new_graph_id
                
                if 'center_of_mass' in nd:
                    params['center_of_mass'] = (
                        np.array(nd['center_of_mass'], dtype=float) + displacement
                    )
                elif 'center_of_mass' in params:
                    params['center_of_mass'] = (
                        np.array(params['center_of_mass'], dtype=float) + displacement
                    )
                
                shifted_positions = None
                if nd.get('positions'):
                    shifted_positions = [
                        np.array(p, dtype=float) + displacement
                        for p in nd['positions']
                    ]
                
                # Rewrite graph_ids in connection source/target via id_mapping.
                # IDs not in the mapping (refs to graphs outside the file)
                # are kept as-is — they may be dangling and prune_dangling
                # will catch them on next reset.
                adjusted_connections = []
                for conn in nd.get('connections', []):
                    new_conn = copy.deepcopy(conn)
                    if 'source' in new_conn and new_conn['source']:
                        old_src = new_conn['source'].get('graph_id')
                        if old_src in id_mapping:
                            new_conn['source']['graph_id'] = id_mapping[old_src]
                    if 'target' in new_conn and new_conn['target']:
                        old_tgt = new_conn['target'].get('graph_id')
                        if old_tgt in id_mapping:
                            new_conn['target']['graph_id'] = id_mapping[old_tgt]
                    adjusted_connections.append(new_conn)
                params['connections'] = adjusted_connections
                
                cleaned_devices = []
                for dev in nd.get('devices', []):
                    d_copy = copy.deepcopy(dev)
                    d_copy['runtime_gid'] = None
                    cleaned_devices.append(d_copy)
                params['devices'] = cleaned_devices
                
                is_root = (nd.get('id', 0) == 0)
                parent_id = nd.get('parent_id')
                parent = graph.get_node(parent_id) if parent_id is not None else None
                
                new_node = graph.create_node(
                    parameters=params,
                    other=parent,
                    is_root=is_root,
                    auto_build=False,
                )
                
                if shifted_positions is not None:
                    new_node.positions = shifted_positions
                    if 'center_of_mass' in params:
                        new_node.center_of_mass = np.array(params['center_of_mass'], dtype=float)
                
                new_node.devices = cleaned_devices
                new_node.connections = adjusted_connections
                
                if 'neuron_models' in nd:
                    new_node.neuron_models = nd['neuron_models']
                if 'types' in nd:
                    new_node.types = nd['types']
                    new_node.population = [None] * len(nd['types'])
            
            # Intra-graph prev/next
            for nd in nodes_data:
                node = graph.get_node(nd.get('id'))
                if not node: continue
                for nid in nd.get('next_ids', []):
                    nxt = graph.get_node(nid)
                    if nxt and nxt not in node.next: node.next.append(nxt)
                for pid in nd.get('prev_ids', []):
                    prv = graph.get_node(pid)
                    if prv and prv not in node.prev: node.prev.append(prv)
            
            # Populate NEST
            for node in graph.node_list:
                if not hasattr(node, 'positions') or not node.positions:
                    node.build()
                elif all(len(p) == 0 for p in node.positions if p is not None):
                    node.build()
                node.populate_node()
            
            graph_list.append(graph)
            new_graphs.append(graph)
        
        WidgetLib.next_graph_id = max(g.graph_id for g in graph_list) + 1
        
        # Cross-graph backref pass over EVERY graph (so backrefs work
        # both within the imported bundle and to existing graphs).
        global_lookup = {(g.graph_id, n.id): n for g in graph_list for n in g.node_list}
        for g in new_graphs:
            for node in g.node_list:
                for conn in node.connections or []:
                    tgt = conn.get('target', {}) or {}
                    tn = global_lookup.get((tgt.get('graph_id'), tgt.get('node_id')))
                    if tn is not None:
                        node.add_neighbor(tn)
        
        # Build NEST connections for all newly imported graphs
        graphs_dict = {g.graph_id: g for g in graph_list}
        executor = ConnectionExecutor(graphs_dict)
        all_new_conns = []
        for g in new_graphs:
            for n in g.node_list:
                if hasattr(n, 'connections'):
                    all_new_conns.extend(n.connections)
        success, fail, _ = executor.execute_all(all_new_conns)
        
        # Refresh UI
        if hasattr(self, 'tools_widget'):
            self.tools_widget.update_graphs(graph_list)
        self.update_visualizations()
        self.graph_overview.update_tree()
        if hasattr(self, 'connection_tool'):
            self.connection_tool.refresh()
        if hasattr(self, 'graph_editor'):
            self.graph_editor.refresh_graph_list()
        if hasattr(self, 'blink_widget'):
            self.blink_widget.build_scene()
        
        graph_names = ", ".join(g.graph_name for g in new_graphs)
        self.status_bar.show_success(
            f"Imported {len(new_graphs)} graph(s) [{graph_names}] — "
            f"{success} conns, {fail} failed"
        )

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
            elif 'graph' in project_data:
                single_graph = project_data['graph'].copy()
                single_graph['nodes'] = project_data.get('nodes', [])
                graphs_to_load = [single_graph]
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
            
            print(f"MERGE GRAPHS (Offset: {id_offset})")

            self.status_bar.set_status(f"Merging {len(graphs_to_load)} graph(s)...", color="#9C27B0")
            self.status_bar.set_progress(15)
            QApplication.processEvents()

            id_mapping = {}
            for g_data in graphs_to_load:
                old_id = g_data.get('graph_id', 0)
                new_id = old_id + id_offset
                id_mapping[old_id] = new_id
            
            merged_graphs = []
            total_nodes = 0
            total_connections = 0

            for i, g_data in enumerate(graphs_to_load):
                self.status_bar.set_status(f"Building Graph {i+1}/{len(graphs_to_load)}...")
                QApplication.processEvents()

                old_graph_id = g_data.get('graph_id', i)
                new_graph_id = id_mapping[old_graph_id]
                
                graph_name = g_data.get('graph_name', f'MergedGraph_{new_graph_id}')
                
                graph = Graph(
                    graph_name=graph_name,
                    graph_id=new_graph_id,
                    parameter_list=[],
                    polynom_max_power=g_data.get('polynom_max_power', 5),
                    polynom_decay=g_data.get('polynom_decay', 0.8), 
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
                        new_conn = copy.deepcopy(conn)
                        
                        if 'source' in new_conn and new_conn['source']:
                            old_src = new_conn['source'].get('graph_id')
                            if old_src in id_mapping: new_conn['source']['graph_id'] = id_mapping[old_src]
                        
                        if 'target' in new_conn and new_conn['target']:
                            old_tgt = new_conn['target'].get('graph_id')
                            if old_tgt in id_mapping: new_conn['target']['graph_id'] = id_mapping[old_tgt]
                        
                        adjusted_connections.append(new_conn)
                        total_connections += 1
                    
                    params['connections'] = adjusted_connections
                    

                    devices_data = nd.get('devices', [])

                    cleaned_devices = []
                    for dev in devices_data:
                        d_copy = copy.deepcopy(dev)
                        d_copy['runtime_gid'] = None 
                        cleaned_devices.append(d_copy)
                    
                    params['devices'] = cleaned_devices


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
                    
                    new_node.devices = cleaned_devices
                    new_node.connections = adjusted_connections
                    
                    if 'neuron_models' in nd:
                        new_node.neuron_models = nd['neuron_models']
                    if 'types' in nd:
                        new_node.types = nd['types']
                        new_node.population = [None] * len(nd['types'])

                    total_nodes += 1

                for nd in nodes_data:
                    node = graph.get_node(nd.get('id'))
                    if not node: continue
                    for next_id in nd.get('next_ids', []):
                        nxt = graph.get_node(next_id)
                        if nxt and nxt not in node.next: node.next.append(nxt)
                    for prev_id in nd.get('prev_ids', []):
                        prv = graph.get_node(prev_id)
                        if prv and prv not in node.prev: node.prev.append(prv)

                print(f"   Populating NEST neurons for {graph_name}...")
                for node in graph.node_list:
                    if not hasattr(node, 'positions') or not node.positions:
                        node.build()
                    elif all(len(p) == 0 for p in node.positions if p is not None):
                        node.build()
                    
                    node.populate_node()

                graph_list.append(graph)
                merged_graphs.append(graph)

            if hasattr(self, 'tools_widget'):
                self.tools_widget.update_graphs(graph_list)
            
            if graph_list:
                WidgetLib.next_graph_id = max(g.graph_id for g in graph_list) + 1

            self.status_bar.set_status("Creating NEST connections...", color="#9C27B0")
            QApplication.processEvents()

            conn_created = 0
            conn_failed = 0
            
            graphs_dict = {g.graph_id: g for g in graph_list}
            executor = ConnectionExecutor(graphs_dict)
            
            all_new_conns = []
            for g in merged_graphs:
                for n in g.node_list:
                    if hasattr(n, 'connections'):
                        all_new_conns.extend(n.connections)
            
            success, fail, _ = executor.execute_all(all_new_conns)
            conn_created = success
            conn_failed = fail

            # Refresh UI
            self.status_bar.set_status("Refreshing view...", color="#9C27B0")
            self.update_visualizations()
            self.graph_overview.update_tree()
            self.connection_tool.refresh()
            self.graph_editor.refresh_graph_list()
            
            if hasattr(self, 'blink_widget'):
                self.blink_widget.build_scene()

            msg = f"Merged {len(merged_graphs)} graphs. Devices restored. {conn_created} connections created."
            self.status_bar.show_success(msg)
            print(msg)
            
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





from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        return super().default(obj)


















class _ExamplesPage(QWidget):
    """
    The "Population Examples" tab: lists every *.json file inside the
    Examples/ folder. Each file is a button — clicking opens a sub-list
    of all graphs the file contains, then a displacement dialog, then
    the import.
    
    The actual scanning and importing logic lives on MainWindow
    (_scan_examples, _import_example_file). This widget is just the UI.
    """
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main = main_window
        self._build_ui()
        self.rebuild()
    
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #2b2b2b; border-bottom: 1px solid #444;")
        hl = QVBoxLayout(header_frame)
        hl.setContentsMargins(10, 10, 10, 10)
        title = QLabel("EXAMPLES")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #66BB6A;")
        hl.addWidget(title)
        info = QLabel(
            "Click a file to import its graphs into the current scene. "
            "You'll be asked for a 3D displacement so the new graph "
            "doesn't overlap existing ones."
        )
        info.setStyleSheet("color: #888; font-style: italic;")
        info.setWordWrap(True)
        hl.addWidget(info)
        # Folder path + rescan button
        bottom_row = QHBoxLayout()
        self._path_label = QLabel("")
        self._path_label.setStyleSheet("color: #666; font-size: 10px; font-family: monospace;")
        bottom_row.addWidget(self._path_label, 1)
        rescan_btn = QPushButton("⟳ Rescan")
        rescan_btn.setFixedHeight(24)
        rescan_btn.setStyleSheet("""
            QPushButton { background: #263238; color: #ECEFF1;
                          border: 1px solid #455A64; border-radius: 4px;
                          padding: 2px 10px; font-size: 11px; }
            QPushButton:hover { background: #37474F; border: 1px solid #607D8B; }
        """)
        rescan_btn.clicked.connect(self.rebuild)
        bottom_row.addWidget(rescan_btn)
        hl.addLayout(bottom_row)
        outer.addWidget(header_frame)
        
        # Scrollable list of example files
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background-color: #1e1e1e;")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._container = QWidget()
        self._container.setStyleSheet("background-color: #1e1e1e;")
        self._inner_layout = QVBoxLayout(self._container)
        self._inner_layout.setSpacing(4)
        self._inner_layout.setContentsMargins(10, 10, 10, 10)
        self._inner_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._container)
        outer.addWidget(scroll, 1)
    
    def rebuild(self):
        """Re-scans the folder and rebuilds the file-button list."""
        # Wipe
        while self._inner_layout.count():
            item = self._inner_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        
        try:
            self._path_label.setText(str(self.main._examples_dir()))
        except Exception:
            self._path_label.setText("Examples/")
        
        entries = self.main._scan_examples()
        if not entries:
            hint = QLabel(
                f"<i>No example files yet.</i><br><br>"
                f"Drop <code>*.json</code> files into:<br>"
                f"<b>{self.main._examples_dir()}</b><br><br>"
                f"Each file imports as a complete bundle (all graphs + connections)."
            )
            hint.setTextFormat(Qt.TextFormat.RichText)
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #888; font-size: 12px; padding: 20px;")
            hint.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            self._inner_layout.addWidget(hint)
            return
        
        # One button per file. The whole file is imported as a bundle so
        # cross-graph connections between graphs in the same file stay
        # intact. Files that contain a single graph still work.
        import os
        for fp, label, count in entries:
            btn = QPushButton(label)
            btn.setMinimumHeight(48)
            btn.setMinimumWidth(220)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #1B5E20; color: #C8E6C9;
                    border: 1px solid #2E7D32; border-radius: 6px;
                    padding: 8px 14px; font-weight: bold; font-size: 13px;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #2E7D32; border: 1px solid #66BB6A;
                    color: white;
                }
            """)
            count_word = "graph" if count == 1 else "graphs"
            btn.setToolTip(
                f"Import {os.path.basename(fp)} ({count} {count_word}) "
                f"as a complete bundle."
            )
            # Show the count subtly inside the button text
            btn.setText(f"{label}    [{count} {count_word}]")
            btn.clicked.connect(
                lambda checked, p=fp, lab=label, c=count:
                self.main._on_example_clicked(p, lab, c)
            )
            self._inner_layout.addWidget(btn)
        
        self._inner_layout.addStretch()


class _DisplacementDialog(QDialog):
    """Tiny modal dialog: asks the user for a 3D offset (dx, dy, dz)
    before importing an example graph. Returns (dx, dy, dz) on accept."""
    
    def __init__(self, graph_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Import: {graph_name}")
        self.setModal(True)
        self.setMinimumWidth(360)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(10)
        
        info = QLabel(
            f"Where to place <b>{graph_name}</b>?<br>"
            "<i style='color:#888'>The graph will be shifted by this offset "
            "so it doesn't overlap existing graphs.</i>"
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Three spinboxes
        self.dx = QDoubleSpinBox(); self.dx.setRange(-1e6, 1e6); self.dx.setDecimals(3); self.dx.setSingleStep(1.0); self.dx.setPrefix("dx = ")
        self.dy = QDoubleSpinBox(); self.dy.setRange(-1e6, 1e6); self.dy.setDecimals(3); self.dy.setSingleStep(1.0); self.dy.setPrefix("dy = ")
        self.dz = QDoubleSpinBox(); self.dz.setRange(-1e6, 1e6); self.dz.setDecimals(3); self.dz.setSingleStep(1.0); self.dz.setPrefix("dz = ")
        for w in (self.dx, self.dy, self.dz):
            w.setFixedHeight(28)
        row = QHBoxLayout()
        row.addWidget(self.dx); row.addWidget(self.dy); row.addWidget(self.dz)
        layout.addLayout(row)
        
        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
    
    def get_displacement(self):
        return self.dx.value(), self.dy.value(), self.dz.value()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_mode(app)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())