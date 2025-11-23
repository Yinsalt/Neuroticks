import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget
)
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtCore import Qt
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from neuron_toolbox import *
from WidgetLib import *

# ===== GLOBAL STATE =====
graph_list = []

# ===== NEURON COLORS =====
neuron_colors = {
    'iaf_psc_alpha': '#FF0000',
    'iaf_psc_exp': '#00FF00',
    'iaf_cond_alpha': '#0000FF',
    'iaf_cond_exp': '#FFFF00',
    'aeif_cond_alpha': '#FF00FF',
    'aeif_cond_exp': '#00FFFF',
    'hh_psc_alpha': '#FF8800',
    'hh_cond_exp_traub': '#8800FF',
    'unknown': '#FFFFFF'
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
            graph.create_node(parameters=node_params, other=graph.node_list[i-1], auto_build=True)
    
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


# ===== MAIN WINDOW =====
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuroticks")
        self.resize(1920, 1080)
        self.active_graphs = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize UI layout."""
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
        
        self.vis_stack.addWidget(self.neuron_plotter)  # Index 0
        self.vis_stack.addWidget(self.graph_plotter)   # Index 1
        self.vis_stack.addWidget(Color("darkgreen"))   # Index 2
        self.vis_stack.addWidget(Color("darkorange"))  # Index 3
        
        btn_neurons.clicked.connect(lambda: self.vis_stack.setCurrentIndex(0))
        btn_graph.clicked.connect(lambda: self.vis_stack.setCurrentIndex(1))
        btn_sim.clicked.connect(lambda: self.vis_stack.setCurrentIndex(2))
        btn_other.clicked.connect(lambda: self.vis_stack.setCurrentIndex(3))
        
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
    
    def create_bottom_left(self):
        """Edit menu with tools."""
        layout = QVBoxLayout()
        
        # Tool stack
        self.tool_stack = QStackedWidget()
        
        self.graph_builder = GraphCreatorWidget()
        self.graph_builder.graphCreated.connect(self.on_graph_created)
        self.tool_stack.addWidget(self.graph_builder)
        
        self.tool_stack.addWidget(Color("purple"))
        self.tool_stack.addWidget(Color("purple"))
        self.tool_stack.addWidget(Color("purple"))
        
        # Progress bar
        progress = Color("green")
        
        layout.addWidget(self.tool_stack, 9)
        layout.addWidget(progress, 1)
        
        return layout
    
    def create_bottom_right(self):
        """Tool selector buttons and plots."""
        layout = QHBoxLayout()
        
        # Tool buttons
        btn_widget = QWidget()
        btn_layout = QVBoxLayout()
        
        btn_create = QPushButton("Create Graph")
        btn_flow = QPushButton("Flow Editor")
        btn_edit = QPushButton("Graph Editor")
        btn_connect = QPushButton("Connections")
        
        btn_create.clicked.connect(lambda: self.tool_stack.setCurrentIndex(0))
        btn_flow.clicked.connect(lambda: self.tool_stack.setCurrentIndex(1))
        btn_edit.clicked.connect(lambda: self.tool_stack.setCurrentIndex(2))
        btn_connect.clicked.connect(lambda: self.tool_stack.setCurrentIndex(3))
        
        btn_layout.addWidget(btn_create)
        btn_layout.addWidget(btn_flow)
        btn_layout.addWidget(btn_edit)
        btn_layout.addWidget(btn_connect)
        btn_widget.setLayout(btn_layout)
        
        # Plots
        plots = Color("lightgreen")
        
        layout.addWidget(btn_widget, 7)
        layout.addWidget(plots, 3)
        
        return layout
    
    def on_graph_created(self, graph_id):
        try:
            graph = create_graph_from_widget(graph_id)
            self.active_graphs[graph_id] = graph
            graph_list.append(graph)          # WICHTIG: global graph_list pflegen!

            # Jetzt alle Nodes nochmal "nachbauen" falls Flow den CoM verändert hat
            for node in graph.node_list:
                if not hasattr(node, 'positions') or node.positions is None:
                    node.build()              # sicherstellen, dass build wirklich durchgelaufen ist
                node.populate_node()

            # Jetzt erst visualisieren!
            self.update_visualizations()
            self.graph_overview.update_tree()

        except Exception as e:
            print(f"Error creating graph: {e}")
            import traceback
            traceback.print_exc()
    
    def update_visualizations(self):
        self.plot_neuron_points()
        self.plot_graph_skeleton()
    
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
                    color = neuron_colors.get(neuron_type, "#FFFFFF")
                    
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
            self.neuron_plotter.add_legend(
                legend_entries,
                size=(0.2, 0.2),
                loc='upper right'
            )
        
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
                size=(0.2, 0.2),
                loc='upper right'
            )
        
        self.graph_plotter.reset_camera()
        self.graph_plotter.update() 


# ===== MAIN =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
