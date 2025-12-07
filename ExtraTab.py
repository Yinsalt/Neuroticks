import sys
import json
import numpy as np
import copy
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, 
    QPushButton, QFileDialog, QSplitter, QSizePolicy, QMessageBox,
    QComboBox, QCheckBox, QTabWidget, QScrollArea, QFormLayout, 
    QDoubleSpinBox, QSpinBox, QListWidget, QListWidgetItem, QGridLayout, QLineEdit,
    QGroupBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QBrush

from WidgetLib import (
    GraphOverviewWidget, 
    GraphInfoWidget, 
    ConnectionTool, 
    ToolsWidget,
    DeviceConfigPage,
    DeviceTargetSelector, 
    ConnectionTargetRow,  
    SYNAPSE_MODELS,
    NumpyEncoder,     
    _clean_params,
    _serialize_connections,
    DoubleInputField,  
    IntegerInputField  
)
from neuron_toolbox import Graph, Node

pg.setConfigOption('background', '#151515')
pg.setConfigOption('foreground', '#d0d0d0')
pg.setConfigOptions(antialias=True)

class SewingInspector(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("background-color: #1e1e1e; border: none;")
        
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background-color: #1e1e1e;")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_layout.setSpacing(10)
        
        self.lbl_placeholder = QLabel("Select a Graph or Node to inspect.")
        self.lbl_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_placeholder.setStyleSheet("color: #555; font-style: italic; font-size: 14px; margin-top: 50px;")
        self.content_layout.addWidget(self.lbl_placeholder)
        
        self.scroll.setWidget(self.content_widget)
        layout.addWidget(self.scroll)

    def clear_content(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def inspect_graph(self, graph):
        self.clear_content()
        title = QLabel(f"GRAPH: {getattr(graph, 'graph_name', 'Unnamed')}")
        title.setStyleSheet("color: #00E5FF; font-size: 18px; font-weight: bold; border-bottom: 2px solid #00E5FF; padding-bottom: 5px;")
        self.content_layout.addWidget(title)
        
        n_nodes = len(graph.node_list)
        n_conns = sum(len(getattr(n, 'connections', [])) for n in graph.node_list)
        
        stats_grid = QGridLayout()
        stats_grid.addWidget(QLabel("Graph ID:"), 0, 0)
        stats_grid.addWidget(QLabel(str(graph.graph_id), styleSheet="color: white; font-weight: bold;"), 0, 1)
        stats_grid.addWidget(QLabel("Total Nodes:"), 1, 0)
        stats_grid.addWidget(QLabel(str(n_nodes), styleSheet="color: white; font-weight: bold;"), 1, 1)
        stats_grid.addWidget(QLabel("Total Connections:"), 2, 0)
        stats_grid.addWidget(QLabel(str(n_conns), styleSheet="color: white; font-weight: bold;"), 2, 1)
        
        stats_box = QFrame()
        stats_box.setStyleSheet("background-color: #2b2b2b; border-radius: 5px; padding: 10px;")
        stats_box.setLayout(stats_grid)
        self.content_layout.addWidget(stats_box)
        self.content_layout.addStretch()

    def inspect_node(self, node):
        self.clear_content()
        title = QLabel(f"NODE: {node.name}")
        title.setStyleSheet("color: #E040FB; font-size: 18px; font-weight: bold; border-bottom: 2px solid #E040FB; padding-bottom: 5px;")
        self.content_layout.addWidget(title)
        
        self.content_layout.addWidget(QLabel("Populations:", styleSheet="color: #FF9800; font-weight: bold; margin-top: 10px;"))
        models = getattr(node, 'neuron_models', [])
        for i, m in enumerate(models):
            lbl = QLabel(f"  Pop {i}: {m}")
            lbl.setStyleSheet("color: white; background-color: #333; padding: 5px; border-radius: 3px;")
            self.content_layout.addWidget(lbl)
            
        conns = getattr(node, 'connections', [])
        self.content_layout.addWidget(QLabel(f"Connections (Source): {len(conns)}", styleSheet="color: #4CAF50; font-weight: bold; margin-top: 10px;"))
        for c in conns:
            tgt = c.get('target', {})
            name = c.get('name', 'conn')
            lbl = QLabel(f"  → G{tgt.get('graph_id')} N{tgt.get('node_id')} P{tgt.get('pop_id')} ({name})")
            lbl.setStyleSheet("color: #ccc;")
            self.content_layout.addWidget(lbl)
            
        devs = getattr(node, 'devices', [])
        self.content_layout.addWidget(QLabel(f"Devices: {len(devs)}", styleSheet="color: #E91E63; font-weight: bold; margin-top: 10px;"))
        for d in devs:
            lbl = QLabel(f"  🛠 {d.get('model')} (#{d.get('id')})")
            lbl.setStyleSheet("color: #ccc;")
            self.content_layout.addWidget(lbl)
            
        self.content_layout.addStretch()



class SewingConnectionTool(QWidget):
    connectionChanged = pyqtSignal()
    
    def __init__(self, graph_list):
        super().__init__()
        self.graph_list = graph_list
        self.current_source_node = None
        self.current_source_graph_id = None
        self.editing_connection_idx = None
        
        self.syn_param_widgets = {} 
        
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        
        left_frame = QFrame(); left_frame.setFixedWidth(280)
        left_frame.setStyleSheet("background-color: #232323; border-right: 1px solid #444;")
        left_layout = QVBoxLayout(left_frame)
        
        left_layout.addWidget(QLabel("SOURCE SELECTION", styleSheet="color: #00E5FF; font-weight: bold; font-size: 12px;"))
        self.lbl_source_info = QLabel("No Node Selected")
        self.lbl_source_info.setStyleSheet("color: #eee; font-weight: bold; border: 1px solid #555; padding: 5px; background: #333;")
        left_layout.addWidget(self.lbl_source_info)
        
        form_src = QFormLayout()
        self.combo_src_pop = QComboBox()
        form_src.addRow("Source Pop:", self.combo_src_pop)
        left_layout.addLayout(form_src)
        
        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("EXISTING CONNECTIONS", styleSheet="color: #bbb; font-weight: bold; font-size: 11px;"))
        
        self.conn_list = QListWidget()
        self.conn_list.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        self.conn_list.itemClicked.connect(self.load_connection_to_edit)
        left_layout.addWidget(self.conn_list)
        
        btn_new = QPushButton("+ NEW CONNECTION")
        btn_new.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 5px;")
        btn_new.clicked.connect(self.reset_form)
        left_layout.addWidget(btn_new)
        
        btn_del = QPushButton("Delete Selected")
        btn_del.setStyleSheet("background-color: #D32F2F; color: white; padding: 5px;")
        btn_del.clicked.connect(self.delete_selected_connection)
        left_layout.addWidget(btn_del)
        
        layout.addWidget(left_frame)

        right_frame = QWidget()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background-color: transparent;")
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(10)
        
        target_group = QGroupBox("Target Selection")
        target_group.setStyleSheet("QGroupBox { border: 1px solid #FF9800; font-weight: bold; margin-top: 10px; } QGroupBox::title { color: #FF9800; }")
        t_form = QFormLayout(target_group)
        
        self.combo_tgt_graph = QComboBox(); self.combo_tgt_graph.currentIndexChanged.connect(self.on_tgt_graph_changed)
        self.combo_tgt_node = QComboBox(); self.combo_tgt_node.currentIndexChanged.connect(self.on_tgt_node_changed)
        self.combo_tgt_pop = QComboBox()
        
        t_form.addRow("Graph:", self.combo_tgt_graph)
        t_form.addRow("Node:", self.combo_tgt_node)
        t_form.addRow("Pop:", self.combo_tgt_pop)
        scroll_layout.addWidget(target_group)
        
        name_layout = QHBoxLayout()
        self.edit_name = QLineEdit(); self.edit_name.setPlaceholderText("Connection Name")
        name_layout.addWidget(QLabel("Name:"))
        name_layout.addWidget(self.edit_name)
        scroll_layout.addLayout(name_layout)
        
        self.tabs = QTabWidget()
        self.tab_spatial = QWidget()
        self._init_spatial_tab()
        self.tabs.addTab(self.tab_spatial, "🌍 Spatial")
        
        self.tab_topo = QWidget()
        self._init_topological_tab()
        self.tabs.addTab(self.tab_topo, "🕸️ Topological")
        
        scroll_layout.addWidget(self.tabs)
        
        syn_group = QGroupBox("Synapse Properties")
        syn_layout = QFormLayout(syn_group)
        
        self.combo_synapse = QComboBox()
        self.combo_synapse.addItems(sorted(SYNAPSE_MODELS.keys()))
        self.combo_synapse.currentTextChanged.connect(self.on_synapse_model_changed)
        syn_layout.addRow("Model:", self.combo_synapse)
        
        wd_layout = QHBoxLayout()
        self.spin_weight = QDoubleSpinBox(); self.spin_weight.setRange(-1e6, 1e6); self.spin_weight.setValue(1.0); self.spin_weight.setPrefix("W: ")
        self.spin_delay = QDoubleSpinBox(); self.spin_delay.setRange(0.1, 1000.0); self.spin_delay.setValue(1.0); self.spin_delay.setPrefix("D: ")
        self.spin_delay.setSuffix(" ms")
        wd_layout.addWidget(self.spin_weight)
        wd_layout.addWidget(self.spin_delay)
        syn_layout.addRow("Base Params:", wd_layout)
        
        self.dynamic_syn_params_container = QWidget()
        self.dynamic_syn_params_layout = QVBoxLayout(self.dynamic_syn_params_container)
        self.dynamic_syn_params_layout.setContentsMargins(0,0,0,0)
        syn_layout.addRow(self.dynamic_syn_params_container)
        
        opts_layout = QHBoxLayout()
        self.allow_autapses_check = QCheckBox("Autapses")
        self.allow_multapses_check = QCheckBox("Multapses"); self.allow_multapses_check.setChecked(True)
        self.receptor_spin = QSpinBox(); self.receptor_spin.setRange(0, 255); self.receptor_spin.setPrefix("Receptor: ")
        
        opts_layout.addWidget(self.allow_autapses_check)
        opts_layout.addWidget(self.allow_multapses_check)
        opts_layout.addWidget(self.receptor_spin)
        syn_layout.addRow(opts_layout)
        
        scroll_layout.addWidget(syn_group)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        right_layout.addWidget(scroll)
        
        btn_container = QWidget()
        btn_l = QVBoxLayout(btn_container)
        btn_l.setContentsMargins(10, 5, 10, 10)
        
        self.btn_save = QPushButton("SAVE CONNECTION")
        self.btn_save.setMinimumHeight(50)
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")
        self.btn_save.clicked.connect(self.save_connection)
        
        btn_l.addWidget(self.btn_save)
        right_layout.addWidget(btn_container)
        
        layout.addWidget(right_frame)
        
        self.on_synapse_model_changed("static_synapse")

    def _init_spatial_tab(self):
        layout = QVBoxLayout(self.tab_spatial)
        layout.addWidget(QLabel("Connects neurons based on spatial positions.", styleSheet="color: #AAA; font-style: italic;"))
        
        form = QFormLayout()
        self.mask_type_combo = QComboBox()
        self.mask_type_combo.addItems(["sphere", "box", "doughnut"])
        form.addRow("Mask Shape:", self.mask_type_combo)
        
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.01, 1000.0); self.radius_spin.setValue(0.5)
        form.addRow("Outer Radius/Size:", self.radius_spin)
        
        self.inner_radius_spin = QDoubleSpinBox()
        self.inner_radius_spin.setRange(0.0, 1000.0); self.inner_radius_spin.setValue(0.0)
        form.addRow("Inner Radius:", self.inner_radius_spin)
        layout.addLayout(form)
        
        dist_layout = QHBoxLayout()
        self.dist_dep_check = QCheckBox("Scale Weight by Distance")
        self.dist_dep_check.toggled.connect(lambda c: [self.dist_factor_spin.setEnabled(c), self.dist_offset_spin.setEnabled(c)])
        
        self.dist_factor_spin = QDoubleSpinBox(); self.dist_factor_spin.setEnabled(False); self.dist_factor_spin.setValue(1.0)
        self.dist_offset_spin = QDoubleSpinBox(); self.dist_offset_spin.setEnabled(False); self.dist_offset_spin.setValue(0.0)
        
        dist_layout.addWidget(self.dist_dep_check); dist_layout.addWidget(QLabel("Factor:")); dist_layout.addWidget(self.dist_factor_spin)
        dist_layout.addWidget(QLabel("Offset:")); dist_layout.addWidget(self.dist_offset_spin)
        layout.addLayout(dist_layout)
        
        prob_layout = QHBoxLayout()
        self.spatial_prob_spin = QDoubleSpinBox(); self.spatial_prob_spin.setRange(0.0, 1.0); self.spatial_prob_spin.setValue(1.0)
        prob_layout.addWidget(QLabel("Probability (p):")); prob_layout.addWidget(self.spatial_prob_spin)
        layout.addLayout(prob_layout)
        layout.addStretch()

    def _init_topological_tab(self):
        layout = QVBoxLayout(self.tab_topo)
        self.rule_combo = QComboBox()
        self.rule_combo.addItems(["all_to_all", "fixed_indegree", "fixed_outdegree", "fixed_total_number", "pairwise_bernoulli", "one_to_one"])
        self.rule_combo.currentTextChanged.connect(self.on_rule_changed)
        
        layout.addWidget(QLabel("Connection Rule:"))
        layout.addWidget(self.rule_combo)
        
        self.topo_params_widget = QWidget(); self.topo_params_layout = QFormLayout(self.topo_params_widget)
        layout.addWidget(self.topo_params_widget)
        
        self.indegree_spin = QSpinBox(); self.indegree_spin.setRange(1, 100000); self.indegree_spin.setValue(10)
        self.outdegree_spin = QSpinBox(); self.outdegree_spin.setRange(1, 100000); self.outdegree_spin.setValue(10)
        self.total_num_spin = QSpinBox(); self.total_num_spin.setRange(1, 1000000); self.total_num_spin.setValue(100)
        self.topo_prob_spin = QDoubleSpinBox(); self.topo_prob_spin.setRange(0, 1); self.topo_prob_spin.setValue(0.1)
        
        self.on_rule_changed("all_to_all")
        layout.addStretch()

    def on_rule_changed(self, rule):
        while self.topo_params_layout.count():
            item = self.topo_params_layout.takeAt(0)
            if item.widget(): item.widget().setParent(None)
        
        if rule == "fixed_indegree": self.topo_params_layout.addRow("Indegree:", self.indegree_spin)
        elif rule == "fixed_outdegree": self.topo_params_layout.addRow("Outdegree:", self.outdegree_spin)
        elif rule == "fixed_total_number": self.topo_params_layout.addRow("Total Connections:", self.total_num_spin)
        elif "bernoulli" in rule: self.topo_params_layout.addRow("Probability:", self.topo_prob_spin)

    def on_synapse_model_changed(self, model_name):
        while self.dynamic_syn_params_layout.count():
            item = self.dynamic_syn_params_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        
        self.syn_param_widgets.clear()
        if model_name not in SYNAPSE_MODELS: return
        
        params = SYNAPSE_MODELS[model_name]
        for param_name, info in params.items():
            if param_name in ['weight', 'delay']: continue
            
            p_type = info.get('type', 'float'); p_default = info.get('default', 0.0)
            widget = None
            if p_type == 'float': widget = DoubleInputField(param_name, default_value=float(p_default))
            elif p_type == 'integer': widget = IntegerInputField(param_name, default_value=int(p_default))
            
            if widget:
                self.dynamic_syn_params_layout.addWidget(widget)
                self.syn_param_widgets[param_name] = widget

    def _get_current_params(self):
        is_spatial = (self.tabs.currentIndex() == 0)
        params = {
            'synapse_model': self.combo_synapse.currentText(),
            'weight': self.spin_weight.value(),
            'delay': self.spin_delay.value(),
            'allow_autapses': self.allow_autapses_check.isChecked(),
            'allow_multapses': self.allow_multapses_check.isChecked(),
            'receptor_type': self.receptor_spin.value(),
            'use_spatial': is_spatial
        }
        
        if is_spatial:
            params.update({
                'rule': 'pairwise_bernoulli', 
                'p': self.spatial_prob_spin.value(), 
                'mask_type': self.mask_type_combo.currentText(), 
                'mask_radius': self.radius_spin.value(), 
                'mask_inner_radius': self.inner_radius_spin.value(), 
                'distance_dependent_weight': self.dist_dep_check.isChecked(), 
                'dist_factor': self.dist_factor_spin.value(), 
                'dist_offset': self.dist_offset_spin.value()
            })
        else:
            rule = self.rule_combo.currentText()
            params['rule'] = rule
            if rule == 'fixed_indegree': params['indegree'] = self.indegree_spin.value()
            elif rule == 'fixed_outdegree': params['outdegree'] = self.outdegree_spin.value()
            elif rule == 'fixed_total_number': params['N'] = self.total_num_spin.value()
            elif 'bernoulli' in rule: params['p'] = self.topo_prob_spin.value()
            
        for name, widget in self.syn_param_widgets.items(): 
            params[name] = widget.get_value()
            
        return params

    def refresh_combos(self):
        self.combo_tgt_graph.blockSignals(True); self.combo_tgt_graph.clear()
        for g in self.graph_list:
            name = getattr(g, 'graph_name', f"Graph {g.graph_id}")
            self.combo_tgt_graph.addItem(f"{name} (ID: {g.graph_id})", g.graph_id)
        self.combo_tgt_graph.blockSignals(False); self.on_tgt_graph_changed()

    def on_tgt_graph_changed(self):
        self.combo_tgt_node.blockSignals(True); self.combo_tgt_node.clear()
        gid = self.combo_tgt_graph.currentData()
        graph = next((g for g in self.graph_list if g.graph_id == gid), None)
        if graph:
            for node in graph.node_list: self.combo_tgt_node.addItem(f"{node.name} (ID: {node.id})", node.id)
        self.combo_tgt_node.blockSignals(False); self.on_tgt_node_changed()

    def on_tgt_node_changed(self):
        self.combo_tgt_pop.clear()
        gid = self.combo_tgt_graph.currentData(); nid = self.combo_tgt_node.currentData()
        graph = next((g for g in self.graph_list if g.graph_id == gid), None)
        if graph:
            node = graph.get_node(nid)
            if node:
                models = getattr(node, 'neuron_models', [])
                if not models and hasattr(node, 'types'): models = ["unknown"] * len(node.types)
                for i, m in enumerate(models): self.combo_tgt_pop.addItem(f"Pop {i}: {m}", i)


    def set_source(self, graph_id, node_id):
        self.refresh_combos()
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph: return
        node = graph.get_node(node_id)
        if not node: return
        
        self.current_source_node = node
        self.current_source_graph_id = graph_id
        self.lbl_source_info.setText(f"Source: {node.name} (G{graph_id})")
        
        self.combo_src_pop.clear()
        models = getattr(node, 'neuron_models', [])
        if not models and hasattr(node, 'types'): models = ["unknown"] * len(node.types)
        for i, m in enumerate(models): self.combo_src_pop.addItem(f"Pop {i}: {m}", i)
            
        self.refresh_connection_list()
        self.reset_form()

    def refresh_connection_list(self):
        self.conn_list.clear()
        if not self.current_source_node: return
        
        conns = getattr(self.current_source_node, 'connections', [])
        if not conns and hasattr(self.current_source_node, 'parameters'):
             conns = self.current_source_node.parameters.get('connections', [])
             self.current_source_node.connections = conns 

        for i, c in enumerate(conns):
            tgt = c.get('target', {}); name = c.get('name', f'Conn_{i}')
            icon = "🌍" if c.get('params', {}).get('use_spatial', False) else "🕸️"
            item = QListWidgetItem(f"{icon} {name} -> G{tgt.get('graph_id')}N{tgt.get('node_id')}P{tgt.get('pop_id')}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.conn_list.addItem(item)

    def load_connection_to_edit(self, item):
        idx = item.data(Qt.ItemDataRole.UserRole)
        self.editing_connection_idx = idx
        
        conns = getattr(self.current_source_node, 'connections', [])
        if idx < len(conns):
            c = conns[idx]
            self.btn_save.setText("UPDATE CONNECTION")
            self.btn_save.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
            
            src = c.get('source', {}); tgt = c.get('target', {}); params = c.get('params', {})
            
            idx_s = self.combo_src_pop.findData(src.get('pop_id'))
            if idx_s >= 0: self.combo_src_pop.setCurrentIndex(idx_s)
            
            idx_tg = self.combo_tgt_graph.findData(tgt.get('graph_id'))
            if idx_tg >= 0: self.combo_tgt_graph.setCurrentIndex(idx_tg); self.on_tgt_graph_changed()
            
            idx_tn = self.combo_tgt_node.findData(tgt.get('node_id'))
            if idx_tn >= 0: self.combo_tgt_node.setCurrentIndex(idx_tn); self.on_tgt_node_changed()
            
            idx_tp = self.combo_tgt_pop.findData(tgt.get('pop_id'))
            if idx_tp >= 0: self.combo_tgt_pop.setCurrentIndex(idx_tp)
            
            self.edit_name.setText(c.get('name', ''))
            self.spin_weight.setValue(params.get('weight', 1.0))
            self.spin_delay.setValue(params.get('delay', 1.0))
            self.allow_autapses_check.setChecked(params.get('allow_autapses', False))
            self.allow_multapses_check.setChecked(params.get('allow_multapses', True))
            self.receptor_spin.setValue(params.get('receptor_type', 0))
            
            model = params.get('synapse_model', 'static_synapse')
            self.combo_synapse.setCurrentText(model)
            
            if params.get('use_spatial', False):
                self.tabs.setCurrentIndex(0)
                self.mask_type_combo.setCurrentText(params.get('mask_type', 'sphere'))
                self.radius_spin.setValue(params.get('mask_radius', 0.5))
                self.inner_radius_spin.setValue(params.get('mask_inner_radius', 0.0))
                self.dist_dep_check.setChecked(params.get('distance_dependent_weight', False))
                self.dist_factor_spin.setValue(params.get('dist_factor', 1.0))
                self.dist_offset_spin.setValue(params.get('dist_offset', 0.0))
                self.spatial_prob_spin.setValue(params.get('p', 1.0))
            else:
                self.tabs.setCurrentIndex(1)
                rule = params.get('rule', 'all_to_all')
                self.rule_combo.setCurrentText(rule)
                if 'indegree' in params: self.indegree_spin.setValue(params['indegree'])
                if 'outdegree' in params: self.outdegree_spin.setValue(params['outdegree'])
                if 'N' in params: self.total_num_spin.setValue(params['N'])
                if 'p' in params: self.topo_prob_spin.setValue(params['p'])
            
            for name, widget in self.syn_param_widgets.items():
                if name in params:
                    widget.set_value(params[name])

    def reset_form(self):
        self.editing_connection_idx = None
        self.btn_save.setText("SAVE NEW CONNECTION")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.edit_name.clear()
        self.spin_weight.setValue(10.0)
        self.conn_list.clearSelection()

    def save_connection(self):
        if not self.current_source_node: return
        
        params = self._get_current_params()
        
        conn_data = {
            'id': np.random.randint(100000, 999999), 
            'name': self.edit_name.text() or "Sewed_Connection",
            'source': {
                'graph_id': self.current_source_graph_id, 
                'node_id': self.current_source_node.id, 
                'pop_id': self.combo_src_pop.currentData()
            },
            'target': {
                'graph_id': self.combo_tgt_graph.currentData(), 
                'node_id': self.combo_tgt_node.currentData(), 
                'pop_id': self.combo_tgt_pop.currentData()
            },
            'params': params
        }
        
        if not hasattr(self.current_source_node, 'connections'):
            self.current_source_node.connections = []
        
        if self.editing_connection_idx is not None:
            old_id = self.current_source_node.connections[self.editing_connection_idx].get('id')
            conn_data['id'] = old_id
            self.current_source_node.connections[self.editing_connection_idx] = conn_data
        else:
            self.current_source_node.connections.append(conn_data)
        
        self.current_source_node.parameters['connections'] = self.current_source_node.connections
        
        self.refresh_connection_list()
        self.reset_form()
        self.connectionChanged.emit()

    def delete_selected_connection(self):
        row = self.conn_list.currentRow()
        if row >= 0:
            item = self.conn_list.item(row)
            idx = item.data(Qt.ItemDataRole.UserRole)
            self.current_source_node.connections.pop(idx)
            self.current_source_node.parameters['connections'] = self.current_source_node.connections
            self.refresh_connection_list()
            self.reset_form()
            self.connectionChanged.emit()


class SewingDeviceTool(ToolsWidget):
    def __init__(self):
        super().__init__()
        
        self.layout().addWidget(QLabel("----------------"))
        self.btn_delete = QPushButton("🗑 Delete Current Device")
        self.btn_delete.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold;")
        self.btn_delete.clicked.connect(self.delete_current_device)
        self.layout().addWidget(self.btn_delete)

        for i in range(self.config_stack.count()):
            widget = self.config_stack.widget(i)
            if isinstance(widget, DeviceConfigPage):
                try: widget.deviceCreated.disconnect() 
                except: pass
                try: widget.deviceUpdated.disconnect() 
                except: pass
                
                widget.deviceCreated.connect(self.on_device_created_offline)
                widget.deviceUpdated.connect(self.on_device_updated_offline)
                
                widget.btn_create.setText(f"Attach {widget.device_label}")
                style = widget.btn_create.styleSheet()
                widget.btn_create.setStyleSheet(style.replace("#E91E63", "#00838F").replace("#FF9800", "#00838F"))
                
                old_sel = widget.target_selector
                layout = old_sel.parentWidget().layout()
                layout.removeWidget(old_sel); old_sel.deleteLater()
                
                new_sel = SewingDeviceTargetSelector(self.graph_list)
                layout.insertWidget(0, new_sel)
                widget.target_selector = new_sel

    def set_selection(self, graph_id, node_id):
        for i in range(self.config_stack.count()):
            widget = self.config_stack.widget(i)
            if isinstance(widget, DeviceConfigPage):
                widget.target_selector.set_selection(graph_id, node_id)

    def update_graphs(self, graph_list):
        self.graph_list = graph_list
        for i in range(self.config_stack.count()):
            widget = self.config_stack.widget(i)
            if isinstance(widget, DeviceConfigPage):
                widget.target_selector.graph_list = graph_list
                widget.target_selector.refresh()

    def on_device_created_offline(self, data):
        self._save_device(data, is_update=False)

    def on_device_updated_offline(self, old_data, new_data):
        dev_id = old_data.get('id')
        new_data['id'] = dev_id
        self._save_device(new_data, is_update=True)

    def _save_device(self, data, is_update=False):
        target = data['target']
        graph_id = target['graph_id']
        node_id = target['node_id']
        
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if not graph: return
        node = graph.get_node(node_id)
        if not node: return
        
        if not hasattr(node, 'devices') or node.devices is None: 
            node.devices = node.parameters.get('devices', [])
        if 'devices' not in node.parameters: node.parameters['devices'] = []
        
        device_entry = {
            "id": data.get('id') if is_update else np.random.randint(10000, 99999),
            "model": data['model'],
            "target_pop_id": target['pop_id'],
            "params": data['params'],
            "conn_params": data.get('conn_params', {}),
            "runtime_gid": None
        }
        
        if is_update:
            found = False
            for i, d in enumerate(node.devices):
                if str(d.get('id')) == str(device_entry['id']):
                    node.devices[i] = device_entry
                    found = True
                    break
            if not found: node.devices.append(device_entry)
            
            node.parameters['devices'] = node.devices
            print(f"[Sewing] Updated {data['model']} (ID {device_entry['id']})")
        else:
            node.devices.append(device_entry)
            node.parameters['devices'] = node.devices
            print(f"[Sewing] Attached {data['model']} to G{graph_id} N{node_id}")
            
        self.deviceAdded.emit(); self.reset_view()

    def delete_current_device(self):
        current_page = self.config_stack.currentWidget()
        if isinstance(current_page, DeviceConfigPage) and current_page.current_edit_device:
            dev_data = current_page.current_edit_device
            dev_id = dev_data.get('id')
            target = dev_data.get('target', {})
            graph = next((g for g in self.graph_list if g.graph_id == target.get('graph_id')), None)
            if graph:
                node = graph.get_node(target.get('node_id'))
                if node:
                    node.devices = [d for d in node.devices if str(d.get('id')) != str(dev_id)]
                    node.parameters['devices'] = node.devices
                    
            print(f"[Sewing] Deleted device {dev_id}")
            self.deviceAdded.emit()
            self.reset_view()

class SewingDeviceTargetSelector(QWidget):
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        layout = QFormLayout(self)
        self.combo_graph = QComboBox()
        self.combo_node = QComboBox()
        self.combo_pop = QComboBox()
        layout.addRow("Graph:", self.combo_graph)
        layout.addRow("Node:", self.combo_node)
        layout.addRow("Population:", self.combo_pop)
        self.combo_graph.currentIndexChanged.connect(self.on_graph_changed)
        self.combo_node.currentIndexChanged.connect(self.on_node_changed)
        self.refresh()

    def refresh(self):
        self.combo_graph.blockSignals(True); self.combo_graph.clear()
        for g in self.graph_list: self.combo_graph.addItem(f"{getattr(g,'graph_name','G')} ({g.graph_id})", g.graph_id)
        self.combo_graph.blockSignals(False); self.on_graph_changed()

    def on_graph_changed(self):
        self.combo_node.blockSignals(True); self.combo_node.clear()
        gid = self.combo_graph.currentData()
        g = next((x for x in self.graph_list if x.graph_id==gid), None)
        if g: 
            for n in g.node_list: self.combo_node.addItem(f"{n.name} ({n.id})", n.id)
        self.combo_node.blockSignals(False); self.on_node_changed()

    def on_node_changed(self):
        self.combo_pop.clear()
        gid = self.combo_graph.currentData(); nid = self.combo_node.currentData()
        g = next((x for x in self.graph_list if x.graph_id==gid), None)
        if g:
            n = g.get_node(nid)
            if n:
                mods = getattr(n, 'neuron_models', [])
                if not mods and hasattr(n,'types'): mods = ["unk"]*len(n.types)
                for i, m in enumerate(mods): self.combo_pop.addItem(f"Pop {i}: {m}", i)

    def get_selection(self):
        return {'graph_id': self.combo_graph.currentData(), 'node_id': self.combo_node.currentData(), 'pop_id': self.combo_pop.currentData()}

    def set_selection(self, graph_id, node_id):
        idx_g = self.combo_graph.findData(graph_id)
        if idx_g >= 0:
            self.combo_graph.setCurrentIndex(idx_g)
            self.on_graph_changed()
            
            idx_n = self.combo_node.findData(node_id)
            if idx_n >= 0:
                self.combo_node.setCurrentIndex(idx_n)
                self.on_node_changed()


class SewingToolsPanel(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; background-color: #222; }
            QTabBar::tab { background: #333; color: #aaa; padding: 8px 15px; }
            QTabBar::tab:selected { background: #444; color: white; border-top: 2px solid #00E5FF; }
        """)
        
        self.combined_list = []
        
        self.inspector = SewingInspector()
        self.addTab(self.inspector, "Inspector")
        
        self.connector = SewingConnectionTool(graph_list=[]) 
        self.addTab(self.connector, "Sewing (Connect)")
        
        self.devices = SewingDeviceTool() 
        self.addTab(self.devices, "Instruments")

    def update_all_graphs(self, left_list, right_list):
        self.combined_list = left_list + right_list
        self.connector.graph_list = self.combined_list
        self.connector.refresh_combos()
        self.devices.update_graphs(self.combined_list)

    def handle_selection(self, data):
        dtype = data.get('type')
        gid = data.get('graph_id')
        nid = data.get('node_id')
        
        graph = next((g for g in self.combined_list if g.graph_id == gid), None)
        
        if dtype == 'graph' and graph:
            self.setCurrentIndex(0)
            self.inspector.inspect_graph(graph)
            
        elif dtype == 'node' and graph:
            node = graph.get_node(nid)
            if node:
                self.setCurrentIndex(0)
                self.inspector.inspect_node(node)
                
                self.connector.set_source(gid, nid)
                
                self.devices.set_selection(gid, nid)
        
        elif dtype == 'connection' and graph:
            conn_data = data.get('connection')
            node = graph.get_node(nid)
            if node and conn_data:
                self.setCurrentIndex(1) 
                self.connector.set_source(gid, nid) 
                self.connector.load_connection_to_edit(self.connector.conn_list.currentItem()) 
                self.connector.select_connection_by_data_idx(conn_data) 

        elif dtype == 'device':
            device_data = data.get('device')
            if device_data:
                self.setCurrentIndex(2)
                self.devices.open_device_editor(device_data)


class SewingGraphView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(300) 
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        toolbar = QHBoxLayout()
        self.btn_relayout = QPushButton("⟳ Reset Layout")
        self.btn_relayout.setStyleSheet("background-color: #333; color: white; border: 1px solid #555; padding: 4px;")
        self.btn_relayout.clicked.connect(self.reset_and_start_animation)
        toolbar.addWidget(self.btn_relayout)
        
        self.chk_physics = QCheckBox("Physics Active")
        self.chk_physics.setChecked(True); self.chk_physics.toggled.connect(self.toggle_physics)
        toolbar.addWidget(self.chk_physics); toolbar.addStretch()
        self.layout.addLayout(toolbar)

        self.view = pg.GraphicsLayoutWidget()
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setBackground('#121212')
        
        self.layout.addWidget(self.view)
        self.plot = self.view.addPlot(); self.plot.setAspectLocked(); self.plot.hideAxis('left'); self.plot.hideAxis('bottom')
        self.graph_item = pg.GraphItem(); self.plot.addItem(self.graph_item)
        
        self.nodes_pos = np.array([]); self.nodes_adj = np.array([]); self.node_colors = []; self.node_symbols = []; self.node_groups = np.array([]) 
        self.node_map = {}; self.timer = QTimer(); self.timer.timeout.connect(self.step_layout); self.steps_remaining = 0

    def toggle_physics(self, active):
        if active:
            if self.steps_remaining <= 0: self.steps_remaining = 100 
            self.timer.start(20)
        else: self.timer.stop()

    def update_data(self, left_list, right_list):
        self.timer.stop(); self.node_map = {}; positions = []; colors = []; symbols = []; adjacency = []; groups = []; current_idx = 0
        
        def add_nodes(g_list, brush, sym, grp, x_off):
            nonlocal current_idx
            for graph in g_list:
                for node in graph.node_list:
                    self.node_map[(graph.graph_id, node.id)] = current_idx
                    pos = np.array([node.center_of_mass[0], node.center_of_mass[1]]) if hasattr(node, 'center_of_mass') else np.array([0.0,0.0])
                    pos += np.random.normal(scale=10.0, size=2) + [x_off, 0]
                    positions.append(pos); colors.append(brush); symbols.append(sym); groups.append(grp); current_idx += 1

        add_nodes(left_list, pg.mkBrush('#00E5FF'), 'o', 0, -300.0)
        add_nodes(right_list, pg.mkBrush('#E040FB'), 's', 1, +300.0)
        
        if not positions: self.graph_item.setData(pos=np.array([]), adj=np.array([])); return
        
        self.nodes_pos = np.array(positions, dtype=float); self.node_colors = colors; self.node_symbols = symbols; self.node_groups = np.array(groups, dtype=int)
        
        all_graphs = left_list + right_list
        for graph in all_graphs:
            for node in graph.node_list:
                src_idx = self.node_map.get((graph.graph_id, node.id))
                if src_idx is None: continue
                conns = getattr(node, 'connections', [])
                if not conns and hasattr(node, 'parameters'): conns = node.parameters.get('connections', [])
                for c in conns:
                    tgt = c.get('target', {})
                    t_gid = int(tgt.get('graph_id')) if tgt.get('graph_id') is not None else -1
                    t_nid = int(tgt.get('node_id')) if tgt.get('node_id') is not None else -1
                    tgt_idx = self.node_map.get((t_gid, t_nid))
                    if tgt_idx is not None: adjacency.append([src_idx, tgt_idx])

        self.nodes_adj = np.array(adjacency, dtype=int) if adjacency else np.array([], dtype=int).reshape(0, 2)
        self.update_view()
        if self.chk_physics.isChecked(): self.start_layout_animation()

    def update_view(self):
        try: self.graph_item.setData(pos=self.nodes_pos, adj=self.nodes_adj, pen=pg.mkPen(color=(255, 255, 255, 40), width=1), size=12, symbol=self.node_symbols, symbolBrush=self.node_colors, pxMode=True)
        except: pass

    def start_layout_animation(self): self.steps_remaining = 100; self.timer.start(20)
    def reset_and_start_animation(self):
        if len(self.nodes_pos) == 0: return
        for i, group in enumerate(self.node_groups):
            base_x = -300.0 if group == 0 else 300.0
            self.nodes_pos[i] = [base_x + np.random.uniform(-50, 50), np.random.uniform(-50, 50)]
        self.start_layout_animation()

    def step_layout(self):
        if self.steps_remaining <= 0 or len(self.nodes_pos) == 0: self.timer.stop(); return
        if np.any(np.isnan(self.nodes_pos)): self.nodes_pos = np.nan_to_num(self.nodes_pos); self.timer.stop(); return
        
        pos = self.nodes_pos
        k = 40.0
        force_rep = 3000.0
        force_att = 0.05 
        gravity = 0.05
        dt = 0.4
        max_speed = 40.0
        
        targets = np.zeros_like(pos)
        targets[self.node_groups == 0] = [-300.0, 0.0]
        targets[self.node_groups == 1] = [+300.0, 0.0]
        
        disp = (targets - pos) * gravity
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(delta**2, axis=2)) + 1.0
        
        repulsion = np.minimum(force_rep / (dist**2), 100.0)
        disp += np.sum((delta / dist[:, :, np.newaxis]) * repulsion[:, :, np.newaxis], axis=1)
        
        if len(self.nodes_adj) > 0:
            for i, j in self.nodes_adj:
                delta_vec = pos[i] - pos[j]
                dist_val = np.linalg.norm(delta_vec)
                if dist_val > 0.1:
                    d_vec = (delta_vec / dist_val) * ((dist_val**2 / k) * force_att)
                    disp[i] -= d_vec
                    disp[j] += d_vec
                    
        lengths = np.linalg.norm(disp, axis=1)
        mask = lengths > max_speed
        if np.any(mask): disp[mask] *= (max_speed / lengths[mask])[:, np.newaxis]
        
        self.nodes_pos += disp * dt * 0.9
        self.update_view()
        
        if np.max(lengths) < 0.2: self.steps_remaining = 0
        else: self.steps_remaining -= 1


class ExtraTabWidget(QWidget):
    def __init__(self, main_graph_list, parent=None):
        super().__init__(parent)
        self.main_simulation_graphs = main_graph_list 
        self.left_graphs = []; self.right_graphs = []
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self); main_layout.setContentsMargins(0, 0, 0, 0); main_layout.setSpacing(2)
        
        left_container = QWidget(); left_container.setStyleSheet("background-color: #232323; border-right: 1px solid #444;")
        self.left_layout = QVBoxLayout(left_container); self.left_layout.setContentsMargins(5, 5, 5, 5)
        self.left_layout.addWidget(self._create_header("SOURCE A (CYAN)"))
        btn_load_left = QPushButton("Load A"); btn_load_left.setStyleSheet(self._get_btn_style("#00838F"))
        btn_load_left.clicked.connect(lambda: self.load_graph_file(self.left_graphs, self.left_overview, "A"))
        self.left_layout.addWidget(btn_load_left)
        self.left_overview = GraphOverviewWidget(parent=self, graph_list=self.left_graphs)
        self.left_layout.addWidget(self.left_overview)
        
        center_container = QWidget(); center_container.setStyleSheet("background-color: #1e1e1e;")
        center_layout = QVBoxLayout(center_container)
        canvas_header = QFrame(); canvas_header.setStyleSheet("background-color: #2b2b2b; border-bottom: 1px solid #444;")
        ch_layout = QHBoxLayout(canvas_header); ch_lbl = QLabel("SEWING CANVAS"); ch_lbl.setStyleSheet("color: #ccc; font-weight: bold;")
        ch_layout.addWidget(ch_lbl); center_layout.addWidget(canvas_header)
        
        self.canvas = SewingGraphView(self)
        self.tools_panel = SewingToolsPanel(self)
        

        self.tools_panel.connector.connectionChanged.connect(self.refresh_all_views)
        self.tools_panel.devices.deviceAdded.connect(self.refresh_all_views)
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #444; }")
        
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.tools_panel)
        splitter.setStretchFactor(0, 1) 
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([700, 300]) 
        
        center_layout.addWidget(splitter)
        
        disp_frame = QFrame(); disp_layout = QHBoxLayout(disp_frame); disp_layout.setContentsMargins(5,5,5,5)
        disp_layout.addWidget(QLabel("Disp B (X/Y/Z):", styleSheet="color: #E040FB; font-weight:bold;"))
        self.spin_disp_x = QDoubleSpinBox(); self.spin_disp_x.setRange(-1e6, 1e6); self.spin_disp_x.setValue(0.0); self.spin_disp_x.setDecimals(1)
        self.spin_disp_y = QDoubleSpinBox(); self.spin_disp_y.setRange(-1e6, 1e6); self.spin_disp_y.setValue(0.0)
        self.spin_disp_z = QDoubleSpinBox(); self.spin_disp_z.setRange(-1e6, 1e6); self.spin_disp_z.setValue(0.0)
        disp_layout.addWidget(self.spin_disp_x); disp_layout.addWidget(self.spin_disp_y); disp_layout.addWidget(self.spin_disp_z)
        center_layout.addWidget(disp_frame)

        self.btn_merge = QPushButton("MERGE & SAVE PROJECT")
        self.btn_merge.setMinimumHeight(50)
        self.btn_merge.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_merge.setStyleSheet("background-color: #6A1B9A; color: white; font-weight: bold; font-size: 16px; border-top: 2px solid #8E24AA;")
        self.btn_merge.clicked.connect(self.merge_and_save_project)
        center_layout.addWidget(self.btn_merge)

        right_container = QWidget(); right_container.setStyleSheet("background-color: #232323; border-left: 1px solid #444;")
        self.right_layout = QVBoxLayout(right_container); self.right_layout.setContentsMargins(5, 5, 5, 5)
        self.right_layout.addWidget(self._create_header("SOURCE B (MAGENTA)"))
        btn_load_right = QPushButton("Load B"); btn_load_right.setStyleSheet(self._get_btn_style("#AD1457"))
        btn_load_right.clicked.connect(lambda: self.load_graph_file(self.right_graphs, self.right_overview, "B"))
        self.right_layout.addWidget(btn_load_right)
        self.right_overview = GraphOverviewWidget(parent=self, graph_list=self.right_graphs)
        self.right_layout.addWidget(self.right_overview)

        main_layout.addWidget(left_container, 15); main_layout.addWidget(center_container, 70); main_layout.addWidget(right_container, 15)

        self.left_overview.tree.itemClicked.connect(self._on_tree_clicked)
        self.right_overview.tree.itemClicked.connect(self._on_tree_clicked)
        
        self.left_overview.requestDeviceDeletion.connect(self.delete_device_from_overview)
        self.right_overview.requestDeviceDeletion.connect(self.delete_device_from_overview)
        self.left_overview.requestConnectionDeletion.connect(self.delete_connection_from_overview)
        self.right_overview.requestConnectionDeletion.connect(self.delete_connection_from_overview)

    def refresh_all_views(self):
        self.left_overview.update_tree()
        self.right_overview.update_tree()
        self.canvas.update_data(self.left_graphs, self.right_graphs)

    def delete_device_from_overview(self, device_data):
        try:
            dev_id = device_data.get('id')
            target = device_data.get('target', {})
            gid = target.get('graph_id')
            nid = target.get('node_id')
            
            all_graphs = self.left_graphs + self.right_graphs
            graph = next((g for g in all_graphs if g.graph_id == gid), None)
            if graph:
                node = graph.get_node(nid)
                if node:
                    node.devices = [d for d in node.devices if str(d.get('id')) != str(dev_id)]
                    node.parameters['devices'] = node.devices
                    
            self.refresh_all_views()
            
        except Exception as e:
            print(f"Delete Error: {e}")

    def delete_connection_from_overview(self, conn_data):
        try:
            conn_id = conn_data.get('id')
            src = conn_data.get('source', {})
            gid = src.get('graph_id')
            nid = src.get('node_id')
            
            all_graphs = self.left_graphs + self.right_graphs
            graph = next((g for g in all_graphs if g.graph_id == gid), None)
            if graph:
                node = graph.get_node(nid)
                if node:
                    conns = getattr(node, 'connections', [])
                    node.connections = [c for c in conns if c.get('id') != conn_id]
                    node.parameters['connections'] = node.connections
                    
            self.refresh_all_views()
            
        except Exception as e:
            print(f"Delete Error: {e}")

    def _on_tree_clicked(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data: self.tools_panel.handle_selection(data)

    def _create_header(self, text):
        lbl = QLabel(text); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setStyleSheet("color: #bbb; font-weight: bold; font-size: 11px; letter-spacing: 1px; padding: 5px;")
        return lbl

    def _get_btn_style(self, color):
        return f"QPushButton {{ background-color: {color}; color: white; font-weight: bold; border: none; border-radius: 4px; padding: 8px; }} QPushButton:hover {{ background-color: white; color: {color}; }}"

    def load_graph_file(self, target_list, overview_widget, source_name):
        filepath, _ = QFileDialog.getOpenFileName(self, f"Load into {source_name}", "", "JSON Files (*.json);;All Files (*)")
        if not filepath: return
        try:
            with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
            loaded = []
            if "graphs" in data:
                for g_data in data["graphs"]: loaded.append(self._reconstruct_graph(g_data, g_data.get('nodes', [])))
            elif "graph" in data: loaded.append(self._reconstruct_graph(data["graph"], data.get("nodes", [])))
            else: QMessageBox.warning(self, "Error", "Unknown JSON format."); return

            if loaded:
                off = 1000000 if source_name == "A" else 2000000
                off += len(target_list) * 1000; id_map = {}
                for i, g in enumerate(loaded): old = g.graph_id; g.graph_id = off + i; id_map[old] = g.graph_id
                for g in loaded:
                    for n in g.node_list:
                        n.graph_id = g.graph_id
                        if not hasattr(n, 'connections'): n.connections = n.parameters.get('connections', [])
                        for c in n.connections:
                            s=c['source']; t=c['target']
                            if s['graph_id'] in id_map: s['graph_id'] = id_map[s['graph_id']]
                            if t['graph_id'] in id_map: t['graph_id'] = id_map[t['graph_id']]
                
                target_list.extend(loaded)
                overview_widget.update_tree()
                self.canvas.update_data(self.left_graphs, self.right_graphs)
                self.tools_panel.update_all_graphs(self.left_graphs, self.right_graphs)
                print(f"[Sewing] Loaded {len(loaded)} graphs into {source_name}.")
        except Exception as e: print(f"Load Error: {e}"); import traceback; traceback.print_exc()

    def _reconstruct_graph(self, graph_meta, nodes_data):
        decay = graph_meta.get('polynom_decay', 0.8)
        if isinstance(decay, list): decay = decay[0] if decay else 0.8
        decay = float(decay)
        power = graph_meta.get('polynom_max_power', 5)
        if isinstance(power, list): power = power[0] if power else 5
        power = int(power)
        init_pos = graph_meta.get('init_position', [0, 0, 0])
        if not isinstance(init_pos, list):
             try: init_pos = init_pos.tolist()
             except: init_pos = [0,0,0]
             
        graph = Graph(graph_name=graph_meta.get('graph_name', 'Loaded'), graph_id=graph_meta.get('graph_id', 0), parameter_list=[], polynom_max_power=power, polynom_decay=decay, position=init_pos, max_nodes=len(nodes_data))
        nodes_data = sorted(nodes_data, key=lambda x: x['id'])
        for nd in nodes_data:
            params = nd.get('parameters', {}).copy()
            params.update({'id': nd['id'], 'name': nd['name'], 'graph_id': nd.get('graph_id', graph.graph_id)})
            if 'center_of_mass' in nd: params['center_of_mass'] = np.array(nd['center_of_mass'])
            if 'm' in nd: params['m'] = np.array(nd['m'])
            
            params['connections'] = nd.get('connections') if nd.get('connections') is not None else params.get('connections', [])
            params['devices'] = nd.get('devices') if nd.get('devices') is not None else params.get('devices', [])

            new_node = graph.create_node(parameters=params, is_root=(nd['id'] == 0), auto_build=False)
            if nd.get('positions'): new_node.positions = [np.array(pos) for pos in nd['positions']]
            if 'center_of_mass' in nd: new_node.center_of_mass = np.array(nd['center_of_mass'])
            
            if 'types' in nd: new_node.population = [None] * len(nd['types'])
            if 'neuron_models' in nd: new_node.neuron_models = nd['neuron_models']
            
            new_node.connections = params['connections']
            new_node.devices = params['devices']
        return graph

    def merge_and_save_project(self):
        all_graphs = self.left_graphs + self.right_graphs
        if not all_graphs: QMessageBox.warning(self, "Error", "No graphs to save."); return

        filepath, _ = QFileDialog.getSaveFileName(self, "Save Merged Project", "", "JSON Files (*.json)")
        if not filepath: return
        if not filepath.endswith('.json'): filepath += '.json'

        id_map = {g.graph_id: i for i, g in enumerate(all_graphs)}
        disp_vec = np.array([self.spin_disp_x.value(), self.spin_disp_y.value(), self.spin_disp_z.value()])
        
        project_data = {'meta': {'type': 'sewed_project'}, 'graphs': []}
        
        for g in all_graphs:
            new_gid = id_map[g.graph_id]
            is_from_b = (g in self.right_graphs)
            
            g_data = {
                'graph_id': new_gid, 'graph_name': g.graph_name, 'max_nodes': g.max_nodes,
                'init_position': list(g.init_position) if hasattr(g, 'init_position') else [0,0,0], 
                'polynom_max_power': g.polynom_max_power,
                'polynom_decay': getattr(g, 'polynom_decay', 0.8),
                'nodes': []
            }
            
            for node in g.node_list:
                final_conns = []
                source_conns = getattr(node, 'connections', [])
                if not source_conns and 'connections' in node.parameters: source_conns = node.parameters['connections']

                for conn in source_conns:
                    c = copy.deepcopy(conn)
                    s_gid = c['source']['graph_id']; t_gid = c['target']['graph_id']
                    if s_gid in id_map: c['source']['graph_id'] = id_map[s_gid]
                    if t_gid in id_map: c['target']['graph_id'] = id_map[t_gid]
                    final_conns.append(c)
                
                
                final_devs = []
                source_devs = getattr(node, 'devices', [])
                if not source_devs and 'devices' in node.parameters: source_devs = node.parameters['devices']

                for d in source_devs:
                    dc = copy.deepcopy(d)
                    if 'runtime_gid' in dc: del dc['runtime_gid']
                    if 'params' in dc: dc['params'] = _clean_params(dc['params'])
                    final_devs.append(dc)

                safe_p = _clean_params(node.parameters.copy())
                safe_p['connections'] = final_conns
                safe_p['devices'] = final_devs
                if 'graph_id' in safe_p: safe_p['graph_id'] = new_gid
                
                final_com = node.center_of_mass
                final_positions = node.positions
                final_m = node.parameters.get('m', [0,0,0])
                
                if is_from_b:
                    if isinstance(final_com, np.ndarray): final_com += disp_vec
                    else: final_com = np.array(final_com) + disp_vec
                    
                    if final_positions:
                        final_positions = [p + disp_vec if isinstance(p, np.ndarray) else p for p in final_positions]
                    
                    if isinstance(final_m, list): final_m = np.array(final_m)
                    final_m += disp_vec

                if isinstance(final_com, np.ndarray): safe_p['center_of_mass'] = final_com.tolist()
                else: safe_p['center_of_mass'] = final_com
                    
                if isinstance(final_m, np.ndarray): safe_p['m'] = final_m.tolist()
                else: safe_p['m'] = final_m

                n_data = {
                    'id': node.id, 'name': node.name, 'graph_id': new_gid,
                    'parameters': safe_p, 'connections': final_conns, 'devices': final_devs,
                    'center_of_mass': safe_p['center_of_mass'],
                    'positions': [p.tolist() if isinstance(p, np.ndarray) else list(p) for p in final_positions] if final_positions else [],
                    'neuron_models': node.neuron_models if hasattr(node, 'neuron_models') else [],
                    'types': node.types if hasattr(node, 'types') else [],
                    'parent_id': node.parent.id if hasattr(node, 'parent') and node.parent else None
                }
                g_data['nodes'].append(n_data)
            project_data['graphs'].append(g_data)

        try:
            with open(filepath, 'w') as f: json.dump(project_data, f, cls=NumpyEncoder, indent=2)
            QMessageBox.information(self, "Success", f"Merged project saved with {len(all_graphs)} graphs.")
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def on_tab_active(self): pass