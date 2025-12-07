

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QApplication, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QScrollArea, QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt
from neuron_toolbox import Graph, Node

from WidgetLib import AnalysisDashboard


class CustomTabWidget(QWidget):
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.live_graph_list = graph_list
        
        self.history_dir = Path("Simulation_History")
        
        self.selected_file_path = None
        self.file_buttons = []
        self.reconstructed_graphs = [] 
        
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        header_frame = QFrame()
        header_frame.setFixedHeight(60)
        header_frame.setStyleSheet("background-color: #2b2b2b; border-bottom: 2px solid #444;")
        header_layout = QHBoxLayout(header_frame)
        lbl_title = QLabel("SIMULATION HISTORY BROWSER")
        lbl_title.setStyleSheet("color: #FFD700; font-size: 18px; font-weight: bold; letter-spacing: 2px;")
        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        self.layout.addWidget(header_frame)

        body_widget = QWidget()
        body_layout = QHBoxLayout(body_widget)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        sidebar_container = QWidget()
        sidebar_container.setFixedWidth(300)
        sidebar_container.setStyleSheet("background-color: #232323; border-right: 1px solid #444;")
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)

        lbl_hist = QLabel("SAVED SNAPSHOTS")
        lbl_hist.setStyleSheet("color: #888; font-weight: bold; font-size: 12px; margin-bottom: 5px;")
        sidebar_layout.addWidget(lbl_hist)

        self.hist_scroll = QScrollArea()
        self.hist_scroll.setWidgetResizable(True)
        self.hist_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.hist_scroll.setStyleSheet("background: transparent;")
        
        self.hist_list_widget = QWidget()
        self.hist_list_layout = QVBoxLayout(self.hist_list_widget)
        self.hist_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.hist_list_layout.setSpacing(2)
        self.hist_scroll.setWidget(self.hist_list_widget)
        sidebar_layout.addWidget(self.hist_scroll)
        
        btn_layout = QVBoxLayout()
        self.btn_delete = QPushButton(" Delete Selected")
        self.btn_delete.setEnabled(False)
        self.btn_delete.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete.setStyleSheet("background-color: #b71c1c; color: white; padding: 6px; font-weight: bold; border-radius: 4px;")
        self.btn_delete.clicked.connect(self.delete_selected_file)
        btn_layout.addWidget(self.btn_delete)

        btn_refresh = QPushButton("↻ Refresh List")
        btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_refresh.setStyleSheet("background-color: #444; color: white; padding: 6px; border-radius: 4px;")
        btn_refresh.clicked.connect(self.refresh_history_list)
        btn_layout.addWidget(btn_refresh)
        sidebar_layout.addLayout(btn_layout)
        body_layout.addWidget(sidebar_container)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_placeholder = QLabel("Select a history file to load analysis tools.")
        self.lbl_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_placeholder.setStyleSheet("color: #555; font-size: 16px; font-style: italic;")
        self.content_layout.addWidget(self.lbl_placeholder)

        body_layout.addWidget(self.content_area)
        self.layout.addWidget(body_widget)

        self.refresh_history_list()

    def refresh_history_list(self):
        self.selected_file_path = None
        self.btn_delete.setEnabled(False)
        self.file_buttons = []
        
        while self.hist_list_layout.count():
            child = self.hist_list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not self.history_dir.exists():
            self.history_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            files = sorted(list(self.history_dir.glob("*.json")), reverse=True)
            if not files:
                self.hist_list_layout.addWidget(QLabel("No snapshots found.", styleSheet="color: #666;"))
                return

            for file_path in files:
                filename = file_path.name
                try:
                    ts_str = filename.replace("history_live_", "").replace(".json", "")
                    dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                    display = dt.strftime("%d.%m.%Y  %H:%M:%S")
                except:
                    display = filename

                btn = QPushButton(f"{display}")
                btn.setCheckable(True)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setProperty("file_path", file_path)
                btn.clicked.connect(lambda c, b=btn: self.on_file_selected(b))
                self.style_button(btn, False)
                
                self.file_buttons.append(btn)
                self.hist_list_layout.addWidget(btn)

        except Exception as e:
            self.hist_list_layout.addWidget(QLabel(f"Error: {e}", styleSheet="color:red;"))

    def style_button(self, btn, selected):
        if selected:
            btn.setStyleSheet("text-align:left; background-color:#2196F3; color:white; border:1px solid #64B5F6; border-radius:3px; padding:8px; font-weight:bold;")
        else:
            btn.setStyleSheet("text-align:left; background-color:#333; color:#ccc; border:1px solid #444; border-radius:3px; padding:8px;")

    def on_file_selected(self, clicked_btn):
        self.selected_file_path = clicked_btn.property("file_path")
        for btn in self.file_buttons:
            is_target = (btn == clicked_btn)
            btn.setChecked(is_target)
            self.style_button(btn, is_target)
        self.btn_delete.setEnabled(True)
        
        self.load_and_show_data(self.selected_file_path)

    def load_and_show_data(self, filepath):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            
        lbl_loading = QLabel("Loading Snapshot...")
        lbl_loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_loading.setStyleSheet("color: #FF9800; font-size: 14px;")
        self.content_layout.addWidget(lbl_loading)
        QApplication.processEvents()
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            version = data.get('meta', {}).get('version', '1.0')
            print(f"Loading history file version {version}")
            
            self.reconstructed_graphs = []
            
            if 'graphs' in data:
                for g_data in data['graphs']:
                    g = self._reconstruct_graph(g_data)
                    self.reconstructed_graphs.append(g)
            
            if 'measurements' in data and data['measurements']:
                print("Detected old format (v2.x) - converting measurements...")
                self._convert_old_measurements(data['measurements'])
            
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            
            if self.reconstructed_graphs:
                dashboard = AnalysisDashboard(self.reconstructed_graphs)
                self.content_layout.addWidget(dashboard)
                
                total_nodes = sum(len(g.node_list) for g in self.reconstructed_graphs)
                total_devices = 0
                for g in self.reconstructed_graphs:
                    for n in g.node_list:
                        if hasattr(n, 'results') and 'history' in n.results:
                            for run in n.results['history']:
                                total_devices += len(run.get('devices', {}))
                
                print(f"Loaded {len(self.reconstructed_graphs)} graphs, {total_nodes} nodes, {total_devices} device recordings.")
            else:
                err_lbl = QLabel("No graph data found in file.")
                err_lbl.setStyleSheet("color: #F44336;")
                self.content_layout.addWidget(err_lbl)
            
        except Exception as e:
            print(f"Load error: {e}")
            import traceback
            traceback.print_exc()
            
            if self.content_layout.count() > 0:
                item = self.content_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            err_lbl = QLabel(f"Error loading file:\n{str(e)}")
            err_lbl.setStyleSheet("color: #F44336;")
            err_lbl.setWordWrap(True)
            self.content_layout.addWidget(err_lbl)

    def _reconstruct_graph(self, g_data):
        graph = Graph(
            graph_name=g_data.get('graph_name', 'HistGraph'),
            graph_id=g_data.get('graph_id', 0),
            parameter_list=[],
            max_nodes=len(g_data.get('nodes', []))
        )
        
        nodes_data = sorted(g_data.get('nodes', []), key=lambda x: x.get('id', 0))
        
        for nd in nodes_data:
            params = nd.get('parameters', {}).copy()
            params['id'] = nd.get('id', 0)
            params['name'] = nd.get('name', 'Node')
            params['graph_id'] = g_data.get('graph_id', 0)
            
            new_node = graph.create_node(
                parameters=params, 
                is_root=(nd.get('id', 0) == 0), 
                auto_build=False
            )
            
            new_node.devices = nd.get('devices', [])
            new_node.neuron_models = nd.get('neuron_models', [])
            new_node.types = nd.get('types', [])
            
            if 'results' in nd:
                new_node.results = nd['results']
            else:
                new_node.results = {'history': []}
            
        return graph

    def _convert_old_measurements(self, measurements):

        import ast
        
        node_map = {}
        for g in self.reconstructed_graphs:
            for n in g.node_list:
                node_map[(g.graph_id, n.id)] = n
                if not hasattr(n, 'results'):
                    n.results = {}
                if 'history' not in n.results:
                    n.results['history'] = []
                if not n.results['history']:
                    n.results['history'].append({'devices': {}})
        
        for key_str, rec_data in measurements.items():
            try:
                key_tuple = ast.literal_eval(key_str)
                
                g_id = key_tuple[1]
                n_id = key_tuple[3]
                pop_id = key_tuple[5]
                dev_type = key_tuple[7]
                
                node = node_map.get((g_id, n_id))
                if node:
                    dev_id = str(pop_id)  
                    for d in node.devices:
                        if d.get('model') == dev_type and d.get('target_pop_id') == pop_id:
                            dev_id = str(d.get('id', pop_id))
                            break
                    
                    history_entry = node.results['history'][0]
                    history_entry['devices'][dev_id] = {
                        'model': dev_type,
                        'data': rec_data
                    }
                    
            except Exception as e:
                print(f"Error converting old measurement {key_str[:50]}...: {e}")

    def delete_selected_file(self):
        if not self.selected_file_path:
            return
        
        reply = QMessageBox.question(
            self, 
            "Delete", 
            f"Delete {self.selected_file_path.name}?", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(self.selected_file_path)
                self.refresh_history_list()
                
                while self.content_layout.count():
                    item = self.content_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                
                self.lbl_placeholder = QLabel("Select a history file to load analysis tools.")
                self.lbl_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.lbl_placeholder.setStyleSheet("color: #555; font-size: 16px; font-style: italic;")
                self.content_layout.addWidget(self.lbl_placeholder)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def on_tab_active(self):
        self.refresh_history_list()