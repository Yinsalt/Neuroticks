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
    QGroupBox, QMenu
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QBrush

from WidgetLib import (
    GraphOverviewWidget, 
    GraphInfoWidget, 
    ConnectionTool,
    ConnectionEditorWidget,
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

# Helper für da_source.graph_id-Remapping beim Load — sitzt im
# graph_factory, weil server.py das beim Multi-Agent-Load auch braucht.
# Wir nutzen die selbe Funktion, damit beide Code-Pfade konsistent bleiben.
try:
    from graph_factory import _remap_da_source_in_blueprint
except ImportError:
    # Fallback wenn graph_factory nicht verfügbar (sollte nie passieren
    # in einem laufenden Neuroticks-Setup).
    def _remap_da_source_in_blueprint(blueprint_data, id_offset, file_graph_ids):
        if id_offset == 0:
            return 0
        n = 0
        for g in blueprint_data.get('graphs', []):
            for nd in g.get('nodes', []):
                for storage in (nd.get('connections', []) or [],
                                 nd.get('parameters', {}).get('connections', []) or []):
                    for conn in storage:
                        da = conn.get('da_source')
                        if not isinstance(da, dict): continue
                        gid = da.get('graph_id')
                        if gid is None: continue
                        try: gid_i = int(gid)
                        except (TypeError, ValueError): continue
                        if gid_i in file_graph_ids:
                            da['graph_id'] = gid_i + id_offset
                            n += 1
        return n

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





class OfflineConnectionTargetRow(ConnectionTargetRow):
    """ConnectionTargetRow für offline geladene Graphen.

    Der echte ConnectionTargetRow iteriert für die Pop-Combo über
    node.population — was bei populate=False eine leere Liste ist.
    Hier nehmen wir stattdessen node.types / node.neuron_models als
    Quelle, weil die im thin-JSON immer mitgespeichert sind.
    """

    def on_node_changed(self):
        self.combo_pop.clear()
        graph_id = self.combo_graph.currentData()
        node_id = self.combo_node.currentData()

        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if graph is None:
            return
        node = next((n for n in graph.node_list if n.id == node_id), None)
        if node is None:
            return

        types = list(getattr(node, 'types', []) or [])
        models = list(getattr(node, 'neuron_models', []) or [])
        live_pops = list(getattr(node, 'population', []) or [])
        n_pops = max(len(types), len(models), len(live_pops))
        for i in range(n_pops):
            m = models[i] if i < len(models) else 'unknown'
            self.combo_pop.addItem(f"Pop {i}: {m}", i)


class OfflineConnectionTool(ConnectionTool):
    """ConnectionTool für offline-Graphen (Sewing/Fusion).

    Im Sewing-Tool sind die Graphen reine Datenobjekte ohne NEST-Build —
    populate=False, build_connections=False beim Load. Der Apply-Button
    vom echten ConnectionTool versucht über ConnectionExecutor in NEST
    zu verkabeln, was hier zwangsläufig fehlschlägt (keine NEST-Pops
    vorhanden). Diese Subklasse persistiert die queue'ten Connections
    nur ans Datenmodell (node.connections + node.parameters['connections']),
    ohne NEST anzufassen.

    Außerdem werden die Pop-Combos (Source + Targets) aus types/
    neuron_models gefüllt statt aus node.population — letzteres ist bei
    populate=False leer.

    Alle anderen Features — Spatial-Modes, Topological-Rules, DA-Source
    Editor, STDP-Synapsenparameter — kommen 1:1 aus der Parent-Klasse.
    """

    # ─── Pop-Combos: Source ───────────────────────────────────────────
    def on_source_node_changed(self, index):
        self.source_pop_combo.clear()
        if index < 0:
            return
        graph_id = self.source_graph_combo.currentData()
        node_id = self.source_node_combo.currentData()
        graph = next((g for g in self.graph_list if g.graph_id == graph_id), None)
        if graph is None:
            return
        node = next((n for n in graph.node_list if n.id == node_id), None)
        if node is None:
            return
        types = list(getattr(node, 'types', []) or [])
        models = list(getattr(node, 'neuron_models', []) or [])
        live_pops = list(getattr(node, 'population', []) or [])
        n_pops = max(len(types), len(models), len(live_pops))
        for i in range(n_pops):
            m = models[i] if i < len(models) else 'unknown'
            self.source_pop_combo.addItem(f"Pop {i}: {m}", i)

    # ─── Target-Rows nutzen die Offline-Subklasse ─────────────────────
    def add_target_row(self):
        idx = len(self.target_rows)
        row = OfflineConnectionTargetRow(self.graph_list, idx, self)
        row.removeClicked.connect(self.remove_target_row)
        self.target_rows.append(row)
        self.targets_layout.insertWidget(idx, row)
        QTimer.singleShot(100, lambda: self.scroll_targets.verticalScrollBar().setValue(
            self.scroll_targets.verticalScrollBar().maximum()
        ))

    # ─── Apply: persistiert offline statt NEST-Verkabelung ────────────
    def create_all_connections(self):
        if not self.connections:
            return
        graphs_dict = {g.graph_id: g for g in self.graph_list}
        success = 0
        failed = 0
        for conn in list(self.connections):
            s = conn.get('source', {}) or {}
            graph = graphs_dict.get(s.get('graph_id'))
            if graph is None:
                failed += 1
                continue
            node = graph.get_node(s.get('node_id'))
            if node is None:
                failed += 1
                continue
            if not hasattr(node, 'connections') or node.connections is None:
                node.connections = []
            # deepcopy damit die Queue im Tool unabhängig vom persistierten
            # State bleibt — sonst würde späteres Editing der Queue rückwirkend
            # das gespeicherte Connection-Dict mutieren.
            conn_copy = copy.deepcopy(conn)
            node.connections.append(conn_copy)
            # Beide Storages syncen (parameters['connections'] wird vom
            # Loader und vom Save-Pfad parallel benutzt).
            if hasattr(node, 'parameters') and node.parameters is not None:
                node.parameters['connections'] = node.connections
            success += 1

        # Erfolgreich persistierte Connections aus der Queue entfernen.
        self.connections = []
        self.update_connection_list()
        if failed > 0:
            self.status_label.setText(
                f"Persisted {success} connections.  {failed} failed (graph/node not found)."
            )
        else:
            self.status_label.setText(f"Persisted {success} connections to graph data.")

        if success > 0:
            self.connectionsCreated.emit()


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
    """Tools-Panel im Sewing-Workspace.

    Nutzt jetzt die echten WidgetLib-Klassen:
      - OfflineConnectionTool (= echter ConnectionTool, persistiert offline)
      - ConnectionEditorWidget (existing-connection editor, identisch zum
        Main-Tool)
    Damit kriegt das Sewing-Tool automatisch alle Verbindungsregeln,
    DA-Source-Bindings, Spatial-Modes und STDP-Parameter, die der echte
    Connection-Workflow auch hat — kein Drift mehr zwischen den beiden UIs.
    """

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

        # Echter ConnectionTool (mit Spatial-Modes, DA-source, STDP-Editor).
        # Offline-Variante persistiert nur in node.connections, kein NEST.
        self.connector = OfflineConnectionTool(graph_list=self.combined_list)
        self.addTab(self.connector, "Connect")

        # Bearbeitungs-Tab — derselbe Editor wie im Hauptprogramm.
        self.editor = ConnectionEditorWidget(graph_list=self.combined_list)
        self.addTab(self.editor, "Edit")

        self.devices = SewingDeviceTool()
        self.addTab(self.devices, "Instruments")

    def update_all_graphs(self, left_list, right_list):
        self.combined_list = left_list + right_list
        # In-place graph_list assignment auf den Tools — sie haben die
        # Liste als Reference, also reicht Re-Bind + Refresh.
        self.connector.graph_list = self.combined_list
        self.connector.refresh_graph_list()
        self.editor.graph_list = self.combined_list
        # Wenn der Editor gerade eine Connection geöffnet hat, Re-Resolve
        # auf den neuen graph_list damit der live_ref nicht stale wird.
        if getattr(self.editor, 'current_conn_ref', None) is not None:
            try:
                self.editor.load_connection(self.editor.current_conn_ref)
            except Exception:
                self.editor.clear()
        self.devices.update_graphs(self.combined_list)

    def handle_selection(self, data):
        """Tree-Click — NUR Inspector-Update.

        Das Connect-Panel bleibt bewusst unangetastet, damit der User
        unabhängig von der Tree-Selektion an Verbindungen arbeiten kann.
        Source/Target werden nur auf Right-Click → Context-Menu gesetzt
        (siehe ExtraTabWidget._on_tree_context_menu).
        """
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
                # Devices-Tab kriegt weiterhin die Selection — sonst weiß
                # er nicht wo das nächste Device hin soll.
                self.devices.set_selection(gid, nid)

        elif dtype == 'population' and graph:
            # Pop-Click → Inspector-Update auf Node-Ebene
            node = graph.get_node(nid)
            if node:
                self.setCurrentIndex(0)
                self.inspector.inspect_node(node)
                self.devices.set_selection(gid, nid)

        elif dtype == 'connection' and graph:
            conn_data = data.get('connection')
            if conn_data:
                # Existierende Connection → in den Edit-Tab laden.
                self.setCurrentIndex(2)
                self.editor.load_connection(conn_data)

        elif dtype == 'device':
            device_data = data.get('device')
            if device_data:
                self.setCurrentIndex(3)
                self.devices.open_device_editor(device_data)

    # ─── Programmatic API für Right-Click → Source/Target ─────────────
    def use_as_source(self, gid, nid, pid):
        """Vom Tree-Context-Menu: setzt Source des Connect-Tools."""
        self.setCurrentIndex(1)
        try:
            self.connector.set_source(gid, nid, pid)
        except Exception as e:
            print(f"[Sewing] use_as_source({gid},{nid},{pid}): {e}")

    def add_as_target(self, gid, nid, pid):
        """Vom Tree-Context-Menu: hängt Target an die nächste freie Row.
        Wenn alle bestehenden Rows belegt sind, wird eine neue erstellt."""
        self.setCurrentIndex(1)
        rows = self.connector.target_rows
        target_row = None
        for r in rows:
            try:
                g, n, p = r.get_selection()
            except Exception:
                continue
            if None in (g, n, p):
                target_row = r
                break
        if target_row is None:
            # Alle Rows belegt → neue erzeugen
            self.connector.add_target_row()
            target_row = self.connector.target_rows[-1]

        # Combos in der Reihenfolge G → N → P setzen (Cascade über
        # currentIndexChanged füllt das jeweils nächste Combo).
        idx_g = target_row.combo_graph.findData(gid)
        if idx_g >= 0:
            target_row.combo_graph.setCurrentIndex(idx_g)
        idx_n = target_row.combo_node.findData(nid)
        if idx_n >= 0:
            target_row.combo_node.setCurrentIndex(idx_n)
        idx_p = target_row.combo_pop.findData(pid)
        if idx_p >= 0:
            target_row.combo_pop.setCurrentIndex(idx_p)


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
        

        self.tools_panel.connector.connectionsCreated.connect(self.refresh_all_views)
        self.tools_panel.editor.connectionUpdated.connect(self.refresh_all_views)
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

        # Right-Click Context-Menu für Use-as-Source / Add-as-Target.
        # Tree-Click selbst füllt das Connect-Panel NICHT mehr (entkoppelt) —
        # der Connect-Tool ist immer unabhängig vom Tree bedienbar.
        for tree in (self.left_overview.tree, self.right_overview.tree):
            tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            tree.customContextMenuRequested.connect(
                lambda pos, t=tree: self._on_tree_context_menu(t, pos)
            )

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

    def _on_tree_context_menu(self, tree, pos):
        """Right-Click auf einem Tree-Item → Kontextmenü mit Aktionen
        die in den Connect-Tab durchschlagen. Entkoppelt vom normalen
        Click damit der Connect-Tool unabhängig vom Tree bedient werden
        kann.
        """
        item = tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        dtype = data.get('type')
        gid = data.get('graph_id')
        nid = data.get('node_id')

        # Kontextmenü hat nur Sinn für Node und Population. Connection und
        # Device haben eigene Edit-Wege, Graph-Item hat nichts zum Verbinden.
        if dtype not in ('node', 'population') or gid is None or nid is None:
            return

        # Pop-Item: pop_id steht im data-dict.
        # Node-Item: pop_id default 0 (User kann's danach im Combo ändern).
        pid = data.get('pop_id', 0) if dtype == 'population' else 0
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            pid = 0

        # Label: bei population zeigen wir Pop-Index, bei Node nicht.
        label_target = (f"G{gid}.N{nid}.P{pid}" if dtype == 'population'
                        else f"G{gid}.N{nid} (Pop 0)")

        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #2b2b2b; color: #ddd; border: 1px solid #555; }"
            "QMenu::item:selected { background-color: #1976D2; color: white; }"
        )
        act_src = menu.addAction(f"Use as Source  →  {label_target}")
        act_tgt = menu.addAction(f"Add as Target  →  {label_target}")
        menu.addSeparator()
        act_inspect = menu.addAction("Inspect")

        chosen = menu.exec(tree.viewport().mapToGlobal(pos))
        if chosen is act_src:
            self.tools_panel.use_as_source(gid, nid, pid)
        elif chosen is act_tgt:
            self.tools_panel.add_as_target(gid, nid, pid)
        elif chosen is act_inspect:
            self.tools_panel.handle_selection(data)

    def _create_header(self, text):
        lbl = QLabel(text); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setStyleSheet("color: #bbb; font-weight: bold; font-size: 11px; letter-spacing: 1px; padding: 5px;")
        return lbl

    def _get_btn_style(self, color):
        return f"QPushButton {{ background-color: {color}; color: white; font-weight: bold; border: none; border-radius: 4px; padding: 8px; }} QPushButton:hover {{ background-color: white; color: {color}; }}"

    def load_graph_file(self, target_list, overview_widget, source_name):
        filepath, _ = QFileDialog.getOpenFileName(
            self, f"Load into {source_name}", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Single-graph Files in das 'graphs'-Format wrappen, damit der
            # universelle Loader sie verarbeiten kann.
            if 'graphs' in data:
                blueprint = data
            elif 'graph' in data:
                blueprint = {
                    'graphs': [{
                        **data['graph'],
                        'nodes': data.get('nodes', []),
                    }]
                }
            else:
                QMessageBox.warning(self, "Error", "Unknown JSON format.")
                return

            # ID-Offset wählen: max(existing_graph_ids) + 1.
            #
            # Damit kollidieren neu geladene Graphen weder mit den schon-
            # geladenen aus der anderen Source noch mit Re-Loads in dieselbe
            # Source. Das Format passt zur angepassten Graph-Klasse —
            # Graph.load_all_from_json(id_offset=N) addiert N auf jede
            # graph_id im File und remappt source/target.graph_id in den
            # Connections automatisch mit. Wir ergänzen das durch
            # _remap_da_source_in_blueprint für die top-level da_source-
            # Felder der Dopa-Synapsen, die der Loader nicht kennt.
            existing_gids = [
                g.graph_id for g in (self.left_graphs + self.right_graphs)
            ]
            off = (max(existing_gids) + 1) if existing_gids else 0

            # Vor dem Load: da_source.graph_id mit demselben Offset remappen.
            # Graph.load_all_from_json kümmert sich um source/target.graph_id,
            # nicht aber um da_source — das muss extern gemacht werden, sonst
            # zeigen Dopa-Synapsen nach dem Sewing auf falsche/inexistente
            # Pops.
            file_graph_ids = {
                int(g.get('graph_id', 0)) for g in blueprint.get('graphs', [])
            }
            n_da = _remap_da_source_in_blueprint(blueprint, off, file_graph_ids)

            # Kanonischer Loader — populate=False weil wir nur die
            # Datenstruktur brauchen (kein NEST-Build), build_connections=False
            # weil wir nicht in NEST verkabeln. Funktioniert mit thin
            # (positions=[]) und fat (positions vorhanden) Format gleich.
            loaded = Graph.load_all_from_json(
                blueprint,
                populate=False,
                build_connections=False,
                id_offset=off,
                verbose=False,
            )

            if not loaded:
                QMessageBox.warning(self, "Error", "Loader returned no graphs.")
                return

            target_list.extend(loaded)
            overview_widget.update_tree()
            self.canvas.update_data(self.left_graphs, self.right_graphs)
            self.tools_panel.update_all_graphs(self.left_graphs, self.right_graphs)

            n_conns = sum(
                len(getattr(n, 'connections', []) or [])
                for g in loaded for n in g.node_list
            )
            print(f"[Sewing] Loaded {len(loaded)} graphs into {source_name}  "
                  f"(id_offset={off}, {n_conns} connections, {n_da} da_source remapped).")

        except Exception as e:
            print(f"Load Error: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", str(e))

    def merge_and_save_project(self):
        """Speichert beide Source-Listen in EIN merged thin-1.0 Projekt.

        Wichtige Eigenschaften:
          - graph_ids werden auf 0..N-1 kompaktiert (sonst sähen sie nach
            den Sewing-Offsets aus: 1_000_000+).
          - source/target/da_source.graph_id werden konsistent mit-remappt
            damit Cross-Graph-Connections und VT-Bindings nach dem Reload
            stimmen.
          - next_ids/prev_ids werden via Graph._serialize_node deduplictiert
            (gleicher Code wie der Main-Save-Pfad → keine Drift).
          - B-Graphen kriegen Displacement-Offset auf init_position +
            center_of_mass + parameters['m']. Positions werden nicht
            gespeichert (Thin/NO) — der Loader baut sie via populate_node()
            beim nächsten Open neu.
        """
        all_graphs = self.left_graphs + self.right_graphs
        if not all_graphs:
            QMessageBox.warning(self, "Error", "No graphs to save.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Merged Project", "", "JSON Files (*.json)"
        )
        if not filepath:
            return
        if not filepath.endswith('.json'):
            filepath += '.json'

        # Compact final IDs: 0..N-1 (Sewing-Offsets verschwinden im File).
        id_map = {g.graph_id: i for i, g in enumerate(all_graphs)}

        disp_vec = np.array([
            self.spin_disp_x.value(),
            self.spin_disp_y.value(),
            self.spin_disp_z.value(),
        ], dtype=float)
        has_disp = bool(np.any(disp_vec != 0.0))

        project_data = {
            'meta': {
                'version': 'thin-1.0',
                'type': 'neuroticks_thin_project',
                'include_positions': False,
                'source': 'sewing_tool',
            },
            'graphs': [],
        }

        n_dopa_total = 0

        for g in all_graphs:
            new_gid = id_map[g.graph_id]
            is_from_b = (g in self.right_graphs)

            # Graph-Header analog zu Graph.to_json()
            init_pos = (
                g.init_position.tolist() if isinstance(g.init_position, np.ndarray)
                else list(g.init_position) if hasattr(g, 'init_position') else [0.0, 0.0, 0.0]
            )
            if is_from_b and has_disp:
                init_pos = (np.array(init_pos, dtype=float) + disp_vec).tolist()

            g_data = {
                'graph_id': new_gid,
                'graph_name': g.graph_name,
                'max_nodes': int(g.max_nodes) if g.max_nodes else len(g.node_list),
                'init_position': init_pos,
                'polynom_max_power': int(g.polynom_max_power),
                'polynom_decay': float(getattr(g, 'polynom_decay', 0.8)),
                'nodes': [],
            }

            for node in g.node_list:
                # ★ Kanonische Serialisierung — selbe Funktion wie Main-Save.
                # Schreibt da_source mit, deduplictiert next_ids/prev_ids,
                # macht numpy→python Konvertierung sauber.
                nd = g._serialize_node(
                    node,
                    include_positions=False,   # Thin/NO
                    include_devices=True,
                    include_connections=True,
                )

                # graph_id im Node + parameters auf neue ID
                nd['graph_id'] = new_gid
                params = nd.get('parameters', {})
                if isinstance(params, dict):
                    params['graph_id'] = new_gid

                # source/target/da_source.graph_id in Connections remappen
                for c in nd.get('connections', []) or []:
                    if not isinstance(c, dict):
                        continue
                    src = c.get('source')
                    tgt = c.get('target')
                    if isinstance(src, dict) and src.get('graph_id') in id_map:
                        src['graph_id'] = id_map[src['graph_id']]
                    if isinstance(tgt, dict) and tgt.get('graph_id') in id_map:
                        tgt['graph_id'] = id_map[tgt['graph_id']]
                    da = c.get('da_source')
                    if isinstance(da, dict):
                        if da.get('graph_id') in id_map:
                            da['graph_id'] = id_map[da['graph_id']]
                        n_dopa_total += 1

                # B-side Displacement
                if is_from_b and has_disp:
                    com = nd.get('center_of_mass')
                    if com is not None:
                        nd['center_of_mass'] = (np.array(com, dtype=float) + disp_vec).tolist()
                    if isinstance(params, dict):
                        m_val = params.get('m')
                        if m_val is not None:
                            params['m'] = (np.array(m_val, dtype=float) + disp_vec).tolist()

                g_data['nodes'].append(nd)

            project_data['graphs'].append(g_data)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, cls=NumpyEncoder, indent=2)
            n_nodes = sum(len(g['nodes']) for g in project_data['graphs'])
            n_conns = sum(
                len(nd.get('connections', []) or [])
                for g in project_data['graphs']
                for nd in g['nodes']
            )
            QMessageBox.information(
                self, "Success",
                f"Merged project saved.\n\n"
                f"Graphs:  {len(all_graphs)}\n"
                f"Nodes:   {n_nodes}\n"
                f"Connections:  {n_conns}\n"
                f"Dopamine connections (da_source preserved):  {n_dopa_total}"
            )
            print(f"[Sewing] Saved {filepath}: {len(all_graphs)} graphs, "
                  f"{n_nodes} nodes, {n_conns} conns, {n_dopa_total} dopa.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))

    def on_tab_active(self): pass