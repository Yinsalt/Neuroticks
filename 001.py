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









class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuroticks")
        self.resize(1920, 1080)






        # meine Widgets
        self.btn = self.create_bottom_button_menu()
        self.graph_list = [createGraph(max_nodes=2,graph_id=0), createGraph(max_nodes=3,graph_id=1),createGraph(max_nodes=4,graph_id=2)]# bestimme wie viele nodes initial erstellt werden
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
        self.creation_functionalities = [Color("black"),
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