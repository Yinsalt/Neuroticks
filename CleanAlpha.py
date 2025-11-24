import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QMessageBox, QProgressBar
)
from PyQt6.QtGui import QColor, QPalette, QAction
from PyQt6.QtCore import Qt
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from neuron_toolbox import *
from WidgetLib import *

# ===== NEST INITIALIZATION =====
# Structural Plasticity wird über Menüleiste gesteuert (default: enabled)
nest.EnableStructuralPlasticity()

# ===== GLOBAL STATE =====
graph_list = []

# ===== NEURON COLORS =====

neuron_colors = {
    # === Klassische Leaky Integrate-and-Fire (grün-blau) ===
    'iaf_psc_alpha':            '#00FF88',  # knalliges Hellgrün
    'iaf_psc_exp':              '#00FF00',  # Lime
    'iaf_psc_delta':            '#00CC66',
    'iaf_psc_alpha_multisynapse': '#00FFAA',
    'iaf_psc_exp_multisynapse': '#00FF44',

    # === Conductance-based IAF (blau-türkis) ===
    'iaf_cond_alpha':           '#0088FF',  # kräftiges Blau
    'iaf_cond_exp':             '#00FFFF',  # Cyan
    'iaf_cond_beta':            '#0088CC',
    'iaf_cond_alpha_mc':        '#00AAFF',

    # === Adaptive Exponential (AdEx) – Lila/Magenta ===
    'aeif_cond_alpha':          '#FF00FF',  # Magenta (der Klassiker)
    'aeif_cond_exp':            '#FF00AA',
    'aeif_psc_alpha':           '#CC00FF',
    'aeif_psc_exp':             '#FF33FF',
    'aeif_cond_beta_multisynapse': '#AA00FF',

    # === Hodgkin-Huxley Familie – Rot/Orange ===
    'hh_psc_alpha':             '#FF4400',  # kräftiges Orange-Rot
    'hh_cond_exp_traub':        '#FF0000',  # Feuerrot
    'hh_cond_beta_gap_traub':   '#CC0000',
    'hh_psc_alpha_gap':         '#FF2200',

    # === GIF & GLIF (stochastisch) – Gelb/Gold ===
    'gif_cond_exp':             '#FFFF00',  # leuchtendes Gelb
    'gif_psc_exp':              '#FFCC00',
    'gif_cond_exp_multisynapse': '#FFEE00',
    'glif_cond':                '#FFD700',  # Gold
    'glif_psc':                 '#FFAA00',

    # === Sonderlinge & Exoten ===
    'izhikevich':               '#FF8800',  # kräftiges Orange
    'mat2_psc_exp':             '#FF6600',
    'amat2_psc_exp':            '#FF9933',
    'ht_neuron':                '#FF33AA',  # Pink
    'pp_psc_delta':             '#FF0066',
    'siegert_neuron':           '#00FFCC',

    # === Fallback ===
    'parrot_neuron':            '#888888',  # Grau
    'parrot_neuron_ps':         '#666666',
    'mcculloch_pitts_neuron':   '#444444',
    'unknown':                  '#FFFFFF',  # Weiß (sichtbar auf schwarzem Hintergrund)
    'default':                  '#8888FF'   # falls mal gar nix passt
}


# ===== CONVERTER =====
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
    """Creates Graph object from graph_parameters."""
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
            # ✅ Prüfe ob User explizite Position gesetzt hat
            user_pos = node_params.get('center_of_mass', [0.0, 0.0, 0.0])
            has_explicit_position = not np.allclose(user_pos, [0.0, 0.0, 0.0])
            
            if has_explicit_position:
                # Keine automatische Positionierung → kein `other`!
                graph.create_node(parameters=node_params, auto_build=True)
            else:
                # Standard: relativ zum vorherigen Node
                graph.create_node(
                    parameters=node_params, 
                    other=graph.node_list[i-1], 
                    auto_build=True
                )
    
    graph_list.append(graph)
    return graph


# ===== PLACEHOLDER WIDGET =====
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


# ===== STATUS BAR WIDGET =====
class StatusBarWidget(QWidget):
    """Professional status bar with text + progress bar"""
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
        
        # Progress Bar (right)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
                background-color: #f0f0f0;
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
        self.set_status(f"❌ {message}", color="#D32F2F")
        self.progress_bar.setVisible(False)
    
    def show_success(self, message):
        """Show success message"""
        self.set_status(f"✅ {message}", color="#2E7D32")
        self.progress_bar.setVisible(False)


# ===== MAIN WINDOW =====
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuroticks")
        self.resize(1920, 1080)
        self.active_graphs = {}
        self.structural_plasticity_enabled = True  # Default: enabled (wie in Zeile 14)
        self.create_menubar()
        self.setup_ui()
        self.graph_builder.polynom_manager.polynomialsChanged.connect(self.rebuild_node_with_new_polynomials)
    def create_menubar(self):
        """Erstellt die Menüleiste mit NEST-Einstellungen"""
        menubar = self.menuBar()
        
        # NEST Menu
        nest_menu = menubar.addMenu("⚙️ NEST Settings")
        
        # Structural Plasticity Toggle
        self.plasticity_action = QAction("🧠 Structural Plasticity", self)
        self.plasticity_action.setCheckable(True)
        self.plasticity_action.setChecked(self.structural_plasticity_enabled)
        self.plasticity_action.triggered.connect(self.toggle_structural_plasticity)
        nest_menu.addAction(self.plasticity_action)
        
        nest_menu.addSeparator()
        
        # Manual Reset
        reset_action = QAction("🔄 Reset NEST Kernel", self)
        reset_action.triggered.connect(self.manual_nest_reset)
        nest_menu.addAction(reset_action)
        
        # View Menu
        view_menu = menubar.addMenu("👁️ View")
        
        # Placeholder für später
        refresh_action = QAction("🔄 Refresh Visualizations", self)
        refresh_action.triggered.connect(self.update_visualizations)
        view_menu.addAction(refresh_action)
    def rebuild_node_with_new_polynomials(self, node_idx, polynomials):
        """✅ NEU: Rebuildet Node mit neuen Polynomen"""
        # Finde den entsprechenden Graph (hier: letzter erstellter)
        if not self.active_graphs:
            print("⚠ No active graphs to rebuild")
            return
        
        # Nimm den letzten Graph (oder implementiere Graph-Selection)
        graph = list(self.active_graphs.values())[-1]
        
        if node_idx >= len(graph.node_list):
            print(f"⚠ Node {node_idx} not found in graph")
            return
        
        node = graph.node_list[node_idx]
        
        print(f"🔄 Rebuilding Node {node_idx} with new polynomials...")
        
        # Update encoded_polynoms_per_type
        encoded_polynoms_per_type = []
        for poly_dict in polynomials:
            if poly_dict and all(k in poly_dict for k in ['x', 'y', 'z']):
                encoded_polynoms_per_type.append([poly_dict['x'], poly_dict['y'], poly_dict['z']])
            else:
                encoded_polynoms_per_type.append([])
        
        node.parameters['encoded_polynoms_per_type'] = encoded_polynoms_per_type
        
        # Rebuild positions
        node.build()
        
        # Recreate NEST neurons
        nest.ResetKernel()  # ⚠ Achtung: Löscht ALLE Neuronen!
        for g in self.active_graphs.values():
            for n in g.node_list:
                n.populate_node()
        
        # Update visualization
        self.update_visualizations()
        print(f"✅ Node {node_idx} rebuilt successfully!")
    def toggle_structural_plasticity(self, checked):
        """Toggle Structural Plasticity mit NEST Reset"""
        self.structural_plasticity_enabled = checked
        
        print("\n" + "="*70)
        if checked:
            self.status_bar.set_status("🧠 Enabling Structural Plasticity...", color="#1976D2")
            print("🧠 ENABLING Structural Plasticity")
            nest.EnableStructuralPlasticity()
        else:
            self.status_bar.set_status("🚫 Disabling Structural Plasticity...", color="#1976D2")
            print("🚫 DISABLING Structural Plasticity")
            nest.DisableStructuralPlasticity()
        
        self.status_bar.set_progress(20)
        QApplication.processEvents()
        
        # Reset NEST
        self.status_bar.set_status("🔄 Resetting NEST Kernel...", color="#1976D2")
        print("🔄 Resetting NEST Kernel...")
        nest.ResetKernel()
        self.status_bar.set_progress(40)
        QApplication.processEvents()
        
        # Repopulate all graphs
        if graph_list:
            print(f"📊 Repopulating {len(graph_list)} graphs...")
            total_graphs = len(graph_list)
            
            for i, graph in enumerate(graph_list):
                self.status_bar.set_status(f"📊 Repopulating graph {i+1}/{total_graphs}...", color="#1976D2")
                self.status_bar.set_progress(40 + int(40 * (i+1) / total_graphs))
                QApplication.processEvents()
                
                print(f"  • Graph {graph.graph_id}: {graph.graph_name}")
                for node in graph.node_list:
                    node.populate_node()
            
            # Update visualizations
            self.status_bar.set_status("📊 Updating visualizations...", color="#1976D2")
            self.status_bar.set_progress(90)
            QApplication.processEvents()
            
            self.update_visualizations()
            self.graph_overview.update_tree()
            print("✅ All graphs repopulated successfully!")
            
            self.status_bar.show_success("Plasticity toggled successfully!")
        else:
            self.status_bar.show_success("Plasticity toggled (no graphs)")
        
        print("="*70 + "\n")
    
    def manual_nest_reset(self):
        """Manueller NEST Reset"""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            'Reset NEST Kernel',
            'This will reset NEST and repopulate all graphs.\n\nContinue?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.status_bar.set_status("🔄 Resetting NEST...", color="#1976D2")
            self.status_bar.set_progress(0)
            QApplication.processEvents()
            
            print("\n🔄 Manual NEST Reset...")
            nest.ResetKernel()
            self.status_bar.set_progress(30)
            QApplication.processEvents()
            
            # Re-apply plasticity setting
            if self.structural_plasticity_enabled:
                nest.EnableStructuralPlasticity()
            else:
                nest.DisableStructuralPlasticity()
            
            # Repopulate all
            if graph_list:
                total_graphs = len(graph_list)
                for i, graph in enumerate(graph_list):
                    self.status_bar.set_status(f"📊 Repopulating graph {i+1}/{total_graphs}...", color="#1976D2")
                    self.status_bar.set_progress(30 + int(50 * (i+1) / total_graphs))
                    QApplication.processEvents()
                    
                    for node in graph.node_list:
                        node.populate_node()
                
                self.status_bar.set_status("📊 Updating visualizations...", color="#1976D2")
                self.status_bar.set_progress(90)
                QApplication.processEvents()
                
                self.update_visualizations()
                self.graph_overview.update_tree()
            
            print("✅ Reset complete!\n")
            self.status_bar.show_success("NEST Reset complete!")
    

    def verify_nest_populations(self):
        print("VERIFYING NEST POPULATIONS")
        
        all_ok = True
        
        for graph in graph_list:
            graph_name = getattr(graph, 'graph_name', f'Graph {graph.graph_id}')
            print(f"\n📊 {graph_name} (ID: {graph.graph_id})")
            print("-" * 70)
            
            for node in graph.node_list:
                node_name = getattr(node, 'name', 'Unnamed')
                
                if not hasattr(node, 'population') or not node.population:
                    print(f"  ❌ {node_name}: No NEST population")
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
                            print(f"  ⚠️  {node_name} Pop{pop_idx}: Position mismatch")
                            all_ok = False
                        else:
                            print(f"  ✅ {node_name} Pop{pop_idx}: {neuron_count} neurons spatial")
                    
                    if total_neurons == 0:
                        print(f"  ❌ {node_name}: Total 0 neurons")
                        all_ok = False
                        
                except Exception as e:
                    print(f"  ❌ {node_name}: {type(e).__name__}: {e}")
                    all_ok = False
        
        report = "✅ All verified" if all_ok else "⚠️  Issues found"
        
        print("\n" + "="*70)
        print(report)
        print("="*70 + "\n")
        
        return all_ok, report  # ✅ Jetzt zwei Werte!
    
    
    def update_visualizations(self):
        
        print("VERIFYING NEST POPULATIONS")
        
        success, report = self.verify_nest_populations()
        
        if not success:
            print("\nWARNING: Some nodes have issues with NEST populations!")
            print("Check the report above for details.")
        else:
            print("\nAll NEST populations verified successfully!")
        
        
        # Now plot
        self.plot_neuron_points()
        self.plot_graph_skeleton()


    def setup_ui(self):
        main_layout = QVBoxLayout()
        
        # Top section (60%)
        top_layout = QHBoxLayout()
        top_left = self.create_top_left()
        self.graph_overview = GraphOverviewWidget(graph_list=graph_list)
        
        top_layout.addLayout(top_left, 7)
        top_layout.addWidget(self.graph_overview, 3)
        
        # Bottom section (40%)
        bottom_layout = QHBoxLayout()
        bottom_left = self.create_bottom_left()
        bottom_right = self.create_bottom_right()
        
        bottom_layout.addLayout(bottom_left, 7)
        bottom_layout.addLayout(bottom_right, 3)
        
        # Assemble
        main_layout.addLayout(top_layout, 3)
        main_layout.addLayout(bottom_layout, 2)
        
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        
        # Initial plot
        self.update_visualizations()
    
    def create_top_left(self):
        """Visualization area with scene selector."""
        layout = QHBoxLayout()
        
        # Scene menu buttons
        scene_menu = QWidget()
        scene_layout = QVBoxLayout()
        
        btn_neurons = QPushButton("Neurons")
        btn_graph = QPushButton("Graph")
        btn_sim = QPushButton("Simulation")
        btn_other = QPushButton("Other")
        
        scene_layout.addWidget(btn_neurons)
        scene_layout.addWidget(btn_graph)
        scene_layout.addWidget(btn_sim)
        scene_layout.addWidget(btn_other)
        scene_menu.setLayout(scene_layout)
        
        # Visualization stack
        self.vis_stack = QStackedWidget()
        
        # Create PyVista widgets
        self.neuron_plotter = self.create_neuron_visualization()
        self.graph_plotter = self.create_graph_visualization()
        

        self.blink_widget = BlinkingNetworkWidget(graph_list)

        self.vis_stack.addWidget(self.neuron_plotter)  # Index 0
        self.vis_stack.addWidget(self.graph_plotter)   # Index 1
        self.vis_stack.addWidget(self.blink_widget)        
        self.vis_stack.addWidget(Color("darkorange"))  # Index 3
        
        # Buttons verbinden
        btn_neurons.clicked.connect(lambda: self.vis_stack.setCurrentIndex(0))
        btn_graph.clicked.connect(lambda: self.vis_stack.setCurrentIndex(1))
        btn_sim.clicked.connect(lambda: self.vis_stack.setCurrentIndex(2))
        
        # Button für "Other" zeigt jetzt das Blinken
        btn_other.clicked.connect(lambda: self.vis_stack.setCurrentIndex(3))
        # Wichtig: Wenn man auf den Tab wechselt, sollte man evtl. neu bauen, 
        # falls Graphen geändert wurden. Quick fix:
        btn_other.clicked.connect(self.blink_widget.build_scene)
        
        layout.addWidget(scene_menu, 1)
        layout.addWidget(self.vis_stack, 9)
        
        return layout
    
    def create_neuron_visualization(self):
        """Creates neuron point cloud plotter."""
        plotter = QtInteractor(self)
        plotter.set_background('black')
        return plotter
    
    def create_graph_visualization(self):
        """Creates graph skeleton plotter."""
        plotter = QtInteractor(self)
        plotter.set_background('black')
        return plotter
    def on_graph_updated(self, graph_id):
        """Callback wenn Graph bearbeitet wurde"""
        if graph_id == -1:  # Delete wurde ausgeführt
            print("Graph deleted, refreshing all...")
            self.status_bar.set_status("🗑️ Graph deleted, refreshing...", color="#1976D2")
        else:
            print(f"Graph {graph_id} updated, refreshing...")
            self.status_bar.set_status(f"✏️ Graph {graph_id} updated, refreshing...", color="#1976D2")
        
        self.status_bar.set_progress(50)
        QApplication.processEvents()
        
        self.update_visualizations()
        self.graph_overview.update_tree()
        print(f"[DEBUG] Calling connection_tool.refresh_graph_list()...")
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
        
        # ✅ ConnectionTool bekommt Referenz zur globalen graph_list
        self.connection_tool = ConnectionTool(graph_list)        
        self.tool_stack.addWidget(self.connection_tool)
        
        self.tool_stack.addWidget(Color("purple"))
        
        self.status_bar = StatusBarWidget()
        
        layout.addWidget(self.tool_stack, 9)
        layout.addWidget(self.status_bar, 1)
            
        return layout
    def closeEvent(self, event):
        """Proper cleanup beim Schließen"""
        print("\nCleaning up...")
        
        try:
            # Clear PyVista plotters
            if hasattr(self, 'neuron_plotter'):
                self.neuron_plotter.clear()
                self.neuron_plotter.close()
            
            if hasattr(self, 'graph_plotter'):
                self.graph_plotter.clear()
                self.graph_plotter.close()
            
            print("PyVista cleanup done")
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        event.accept()

    def create_bottom_right(self):
        """Tool selector buttons and plots."""
        layout = QHBoxLayout()
        
        # Tool buttons
        btn_widget = QWidget()
        btn_layout = QVBoxLayout()
        
        btn_create = QPushButton("Create Graph")
        btn_flow = QPushButton("Graph Editor")
        btn_edit = QPushButton("Additional Structures")
        btn_connect = QPushButton("Connections")
        btn_rebuild = QPushButton("Rebuild")
        btn_rebuild.setStyleSheet("background-color: #FF5722; color: white; font-weight: bold;")
        btn_create.clicked.connect(lambda: self.tool_stack.setCurrentIndex(0))
        btn_flow.clicked.connect(lambda: self.tool_stack.setCurrentIndex(1))
        btn_edit.clicked.connect(lambda: self.tool_stack.setCurrentIndex(2))
        btn_connect.clicked.connect(lambda: self.tool_stack.setCurrentIndex(3))
        btn_rebuild.clicked.connect(self.rebuild_all_graphs)        
        btn_layout.addWidget(btn_create)
        btn_layout.addWidget(btn_flow)
        btn_layout.addWidget(btn_edit)
        btn_layout.addWidget(btn_connect)
        btn_layout.addWidget(btn_rebuild)
        btn_widget.setLayout(btn_layout)
        
        # Plots
        plots = Color("lightgreen")
        
        layout.addWidget(btn_widget, 7)
        layout.addWidget(plots, 3)
        
        return layout
    
    def on_graph_created(self, graph_id):
        try:
            self.status_bar.set_status("🔨 Creating graph...", color="#1976D2")
            self.status_bar.set_progress(0, maximum=100)
            QApplication.processEvents()
        
            graph = create_graph_from_widget(graph_id)
            self.active_graphs[graph_id] = graph
            print(f"[DEBUG] Graph created! Total graphs in graph_list: {len(graph_list)}")
            self.status_bar.set_progress(30)
            QApplication.processEvents()

            total_nodes = len(graph.node_list)
            for i, node in enumerate(graph.node_list):
                self.status_bar.set_status(f"🧠 Building node {i+1}/{total_nodes}...", color="#1976D2")
                self.status_bar.set_progress(30 + int(30 * (i+1) / total_nodes))
                QApplication.processEvents()
                
                if not hasattr(node, 'positions') or node.positions is None:
                    node.build()              
                node.populate_node()

            self.status_bar.set_status("📊 Updating visualizations...", color="#1976D2")
            self.status_bar.set_progress(80)
            QApplication.processEvents()
            
            self.update_visualizations()
            self.graph_overview.update_tree()
            
            # ✅ FIX: refresh() statt refresh_graph_list()
            self.connection_tool.refresh()
            
            self.status_bar.show_success(f"Graph '{graph.graph_name}' created!")
            QApplication.processEvents()
            print(f"[DEBUG] Before refresh: ConnectionTool sees {len(self.connection_tool.graph_list)} graphs")
            self.connection_tool.refresh()  # ✅ Jetzt sollte es funktionieren
            print(f"[DEBUG] After refresh: Dropdowns updated")
            
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
        
        color_palette = [
            [1.0, 0.0, 0.0],   # Red
            [0.0, 1.0, 0.0],   # Green
            [0.0, 0.0, 1.0],   # Blue
            [1.0, 1.0, 0.0],   # Yellow
            [1.0, 0.0, 1.0],   # Magenta
            [0.0, 1.0, 1.0],   # Cyan
            [1.0, 0.5, 0.0],   # Orange
            [0.5, 0.0, 1.0],   # Purple
        ]
        
        legend_entries = []
        
        for graph in graph_list:
            color_idx = graph.graph_id % len(color_palette)
            graph_color = color_palette[color_idx]
            
            for node in graph.node_list:
                sphere = pv.Sphere(
                    radius=0.15,
                    center=node.center_of_mass
                )
                self.graph_plotter.add_mesh(
                    sphere,
                    color=graph_color,
                    opacity=0.8
                )
            
            for node in graph.node_list:
                for next_node in node.next:
                    start_pos = np.array(node.center_of_mass)
                    end_pos = np.array(next_node.center_of_mass)
                    
                    line = pv.Line(start_pos, end_pos)
                    self.graph_plotter.add_mesh(
                        line,
                        color=graph_color,
                        line_width=3
                    )
            
            graph_name = getattr(graph, 'graph_name', f'Graph {graph.graph_id}')
            legend_entries.append([graph_name, graph_color])
        
        if legend_entries:
            self.graph_plotter.add_legend(
                legend_entries,
                size=(0.12, 0.12),
                loc='upper right'
            )
        
        self.graph_plotter.reset_camera()
        self.graph_plotter.update() 

    def rebuild_all_graphs(self):
        
        print("REBUILD NOT FINISHED")
        

# ===== MAIN =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
