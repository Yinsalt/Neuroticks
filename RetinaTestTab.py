"""
RetinaTestTab.py
================

Test-Tab im "Other"-Slot der Neuroticks-UI.

Funktionen:
  - Baut eine *nicht-exportierbare* Retina als Pre-Population.
  - Nimmt ein Video als Input (Datei).
  - Jeder Ganglion-Ausgang (6 Stück) kann mit einem beliebigen existierenden
    Graph/Node/Population verbunden werden.
  - Die Retina-Neuronen erscheinen im Live-Tab wie jede andere Population.
  - Play spult das Video in Echtzeit ab, feedet Frames in die Retina, und
    ruft dazwischen nest.Simulate() mit dem passenden Zeitfenster auf.

Integration in bestehenden Code (minimal-invasiv):
  - Erstellt einen VirtualRetinaGraph der in graph_list eingehaengt wird,
    aber ein is_virtual=True Flag traegt.
  - Bestehende Viz-Widgets iterieren ueber graph_list und sehen die Retina
    automatisch ohne Aenderung.
  - save_all_graphs_dialog muss (aussen) is_virtual-Graphen filtern.
"""

from __future__ import annotations

import os
import traceback
from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QGroupBox, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QProgressBar, QMessageBox, QSizePolicy, QScrollArea,
    QCheckBox, QSplitter, QTabWidget, QTextEdit, QApplication
)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import nest

# Retina-Imports - koennen optional fehlschlagen wenn der Pfad nicht stimmt
try:
    from retina_main import (
        Retina,
        GANGLION_POPS,
    )
    from retina_scales import get_config, list_scales, list_variants
    HAS_RETINA = True
except Exception as e:
    HAS_RETINA = False
    _RETINA_IMPORT_ERROR = str(e)
    GANGLION_POPS = []


# ===========================================================================
#  VIRTUAL GRAPH ADAPTER
# ===========================================================================
#
# Wir wollen die Retina-Populationen im Live-Tab sehen wie normale Neuronen.
# Die Live-Widgets iterieren alle ueber graph_list und greifen auf
# node.positions, node.neuron_models, node.id etc. zu.
#
# Statt alle Widgets anzufassen, bauen wir einen duck-typed Adapter:
# eine Graph-aehnliche Klasse mit is_virtual=True, deren node_list aus
# VirtualRetinaNode-Objekten besteht.

NEURON_MODEL_BY_POP = {
    # Mapping Retina-Pop-Name -> NEST-Modell-String (fuer Farbwahl im Live-Viz)
    # Fallback ist iaf_psc_alpha, das hat eine Default-Farbe.
    'L_foveal':                          'iaf_cond_alpha',
    'M_foveal':                          'iaf_cond_alpha',
    'L_peripheral':                      'iaf_cond_alpha',
    'M_peripheral':                      'iaf_cond_alpha',
    'S_peripheral':                      'iaf_cond_alpha',
    'rods':                              'iaf_cond_alpha',
    'horizontal_foveal':                 'iaf_cond_alpha',
    'horizontal_peripheral':             'iaf_cond_alpha',
    'midget_ON_bipolar_foveal':          'iaf_cond_alpha',
    'midget_OFF_bipolar_foveal':         'iaf_cond_alpha',
    'midget_ON_bipolar_peripheral':      'iaf_cond_alpha',
    'midget_OFF_bipolar_peripheral':     'iaf_cond_alpha',
    'parasol_ON_bipolar':                'iaf_cond_alpha',
    'parasol_OFF_bipolar':               'iaf_cond_alpha',
    'amacrine_foveal':                   'aeif_cond_alpha',
    'amacrine_peripheral':               'aeif_cond_alpha',
    'midget_ON_ganglion_foveal':         'aeif_cond_alpha',
    'midget_OFF_ganglion_foveal':        'aeif_cond_alpha',
    'midget_ON_ganglion_peripheral':     'aeif_cond_alpha',
    'midget_OFF_ganglion_peripheral':    'aeif_cond_alpha',
    'parasol_ON_ganglion':               'aeif_cond_alpha',
    'parasol_OFF_ganglion':              'aeif_cond_alpha',
}


class VirtualRetinaNode:
    """Duck-typed Node-Ersatz fuer eine Retina-Population."""

    def __init__(self, node_id, name, positions_array, neuron_model,
                 nest_pop, center_of_mass):
        self.id = node_id
        self.name = name
        # positions: list[np.ndarray] - eine Liste pro Pop (Live-Viz erwartet das)
        self.positions = [positions_array] if positions_array is not None and len(positions_array) else []
        self.neuron_models = [neuron_model]
        self.types = [0]
        self.center_of_mass = np.array(center_of_mass, dtype=float) if center_of_mass is not None else np.zeros(3)
        self.distribution = []
        self.connections = []
        self.devices = []
        self.next = []
        self.prev = []
        self.parent = None
        self.parameters = {
            'id': node_id,
            'name': name,
            'neuron_models': [neuron_model],
            'types': [0],
            'connections': [],
            'devices': [],
        }
        # Mit population emulieren wir die NodeCollection-Struktur von normal Nodes
        self.population = [nest_pop] if nest_pop is not None else []
        self.nest_connections = []
        self.nest_references = {}
        # Main.py's _distribute_simulation_data erwartet node.results.
        # Auch wenn VirtualRetinaNode keine Devices hat, wird es hier sicherheitshalber
        # vorbereitet (der Reset-Loop oder _ensure_spike_recorders kann naemlich
        # doch was dran haengen).
        self.results = {'history': []}

    # -- No-ops fuer Kompatibilitaet mit dem Reset-Flow in Main.py --
    # Main.reset_and_restart() ruft auf jedem Node build() und populate_node().
    # Der Virtual-Node hat nichts zu bauen (Retina lebt im nest-Kernel bereits),
    # aber die Methoden muessen existieren damit der Reset nicht crasht.
    def build(self, *args, **kwargs):
        return self

    def populate_node(self, *args, **kwargs):
        return self

    def connect(self, *args, **kwargs):
        pass

    def verify_and_report(self, *args, **kwargs):
        pass

    def instantiate_devices(self, *args, **kwargs):
        pass

    def add_neighbor(self, other):
        pass

    def remove_neighbor_if_isolated(self, other):
        pass


class VirtualRetinaGraph:
    """Duck-typed Graph-Ersatz fuer die Retina. Wird in graph_list eingehaengt."""

    def __init__(self, graph_id, retina: "Retina"):
        self.graph_id = graph_id
        self.graph_name = "__RetinaTest__"
        self.is_virtual = True   # <--- das Flag fuer Save-Filter
        self.node_list = []
        self.nodes = 0
        self._next_id = 0
        self.max_nodes = 100
        self.polynom_max_power = 5
        self.polynom_decay = 0.8
        self.global_offset = [0, 0, 0]
        self._retina_ref = retina
        self._build_nodes(retina)

    def _build_nodes(self, retina):
        positions = retina.get_positions()
        counts = retina.get_counts()

        node_id = 0
        for pop_name, pop_nodecol in retina.populations.items():
            pts = positions.get(pop_name, None)
            if pts is None or (hasattr(pts, '__len__') and len(pts) == 0):
                continue
            pts_arr = np.asarray(pts, dtype=float)
            # COM aus Punktwolke
            com = pts_arr.mean(axis=0) if len(pts_arr) > 0 else np.zeros(3)
            model = NEURON_MODEL_BY_POP.get(pop_name, 'iaf_psc_alpha')
            node = VirtualRetinaNode(
                node_id=node_id,
                name=pop_name,
                positions_array=pts_arr,
                neuron_model=model,
                nest_pop=pop_nodecol,
                center_of_mass=com,
            )
            self.node_list.append(node)
            node_id += 1
        self._next_id = node_id
        self.nodes = len(self.node_list)

    def get_node(self, node_id):
        for n in self.node_list:
            if n.id == node_id:
                return n
        return None

    # Kompatibilitaet: wenn save_changes() und aehnliches aus WidgetLib diese
    # Methoden callt, brechen wir nicht sondern tun nichts.
    def remove_node(self, node):
        pass

    def build_connections(self, external_graphs=None):
        pass

    def create_node(self, **kwargs):
        raise RuntimeError("VirtualRetinaGraph is non-editable")


# ===========================================================================
#  VIDEO READER
# ===========================================================================

class VideoFrameReader:
    """Sehr duenner Wrapper um cv2.VideoCapture, der LMS+Intensity liefert.

    BGR -> normalized [0,1] RGB wird direkt als LMS verwendet
    (die Retina nimmt die Kanaele als L/M/S — fuer Test-Zwecke reicht das,
     eine echte LMS-Konversion waere eine Hunt-Pointer-Estevez Matrix).
    """

    def __init__(self, path, target_resolution=(64, 64)):
        if not HAS_CV2:
            raise RuntimeError("OpenCV (cv2) nicht installiert — fuer Video-Input benoetigt.")
        self.path = path
        self.target_resolution = target_resolution
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Konnte Video nicht oeffnen: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.current_idx = 0

    def seek(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.current_idx = frame_idx

    def read_next(self):
        """-> (lms (H,W,3) float32 in [0,1], intensity (H,W) float32 in [0,1], bgr_preview (H,W,3) uint8) or None"""
        ret, bgr = self.cap.read()
        if not ret:
            return None
        self.current_idx += 1
        # Zu Ziel-Aufloesung resampeln
        h, w = self.target_resolution
        bgr_small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lms = rgb  # Annaeherung: RGB als LMS verwenden
        intensity = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
        return lms, intensity, bgr

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ===========================================================================
#  MAIN TAB WIDGET
# ===========================================================================

class RetinaTestTabWidget(QWidget):
    """Tab-Inhalt fuer den 'Other'-Tab: Retina-Testing."""

    # Signale die Main.py mitbekommen kann
    retinaBuilt = pyqtSignal()
    retinaDestroyed = pyqtSignal()
    requestVizRefresh = pyqtSignal()     # nach Frame-Tick neu zeichnen

    VIRTUAL_GRAPH_ID = 99999             # in der Regel weit oberhalb der normalen IDs

    def __init__(self, graph_list_ref, main_window=None, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list_ref    # Referenz, KEINE Kopie
        self.main_window = main_window

        # State
        self.retina: Optional["Retina"] = None
        self.feeder = None
        self.virtual_graph: Optional[VirtualRetinaGraph] = None
        self.output_spike_recorders: Dict[str, Any] = {}
        self.registered_connections: List[Tuple[str, Any, Any]] = []

        self.video_reader: Optional[VideoFrameReader] = None

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)
        self._playing = False

        self._init_ui()
        self._refresh_target_selectors()

    # --------------------------------------------------------------------
    #  UI
    # --------------------------------------------------------------------

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Header
        header = QLabel("Retina Test Object  ·  non-exportable pre-population")
        header.setStyleSheet("color: #FFD700; font-size: 16px; font-weight: bold;"
                             " padding: 4px; border-bottom: 1px solid #444;")
        outer.addWidget(header)

        if not HAS_RETINA:
            err = QLabel(
                f"Retina-Module konnten nicht geladen werden:\n{_RETINA_IMPORT_ERROR}"
                f"\n\nStell sicher dass retina_main.py und retina_scales.py im PYTHONPATH sind."
            )
            err.setStyleSheet("color: #F44336; padding: 20px;")
            err.setWordWrap(True)
            outer.addWidget(err)
            return

        # Top row: Config | Video | Status
        top_row = QHBoxLayout()
        top_row.addWidget(self._build_config_group(),  2)
        top_row.addWidget(self._build_video_group(),   3)
        top_row.addWidget(self._build_status_group(),  2)
        outer.addLayout(top_row, 0)

        # Mittelteil: Connection-Tabelle
        outer.addWidget(self._build_connection_group(), 3)

        # Play-Controls
        outer.addWidget(self._build_playback_group(), 0)

        # Log
        outer.addWidget(self._build_log_group(), 1)

    def _build_config_group(self) -> QGroupBox:
        gb = QGroupBox("1. Retina Config")
        gb.setStyleSheet("QGroupBox { color: #4FC3F7; font-weight: bold; }")
        form = QFormLayout(gb)
        form.setContentsMargins(8, 12, 8, 8)

        self.cmb_scale = QComboBox()
        for s, desc in list_scales():
            self.cmb_scale.addItem(f"{s}  —  {desc}", s)
        self.cmb_scale.setCurrentIndex(0)

        self.cmb_variant = QComboBox()
        for v, desc in list_variants():
            self.cmb_variant.addItem(f"{v}  —  {desc}", v)
        self.cmb_variant.setCurrentIndex(0)

        pos_row = QHBoxLayout()
        self.spin_px = QDoubleSpinBox(); self.spin_px.setRange(-1000, 1000); self.spin_px.setValue(-15.0)
        self.spin_py = QDoubleSpinBox(); self.spin_py.setRange(-1000, 1000); self.spin_py.setValue(0.0)
        self.spin_pz = QDoubleSpinBox(); self.spin_pz.setRange(-1000, 1000); self.spin_pz.setValue(0.0)
        for w in (self.spin_px, self.spin_py, self.spin_pz):
            w.setSingleStep(1.0)
            pos_row.addWidget(w)
        pos_wrapper = QWidget(); pos_wrapper.setLayout(pos_row)

        self.btn_build = QPushButton("Build Retina")
        self.btn_build.setStyleSheet("background-color: #2E7D32; color: white; "
                                     "padding: 6px; font-weight: bold;")
        self.btn_build.clicked.connect(self._on_build_clicked)

        self.btn_destroy = QPushButton("Destroy")
        self.btn_destroy.setStyleSheet("background-color: #C62828; color: white; "
                                       "padding: 6px; font-weight: bold;")
        self.btn_destroy.clicked.connect(self._on_destroy_clicked)
        self.btn_destroy.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_build)
        btn_row.addWidget(self.btn_destroy)
        btn_wrap = QWidget(); btn_wrap.setLayout(btn_row)

        form.addRow("Scale:",    self.cmb_scale)
        form.addRow("Variant:",  self.cmb_variant)
        form.addRow("Position:", pos_wrapper)
        form.addRow(btn_wrap)
        return gb

    def _build_video_group(self) -> QGroupBox:
        gb = QGroupBox("2. Video Input")
        gb.setStyleSheet("QGroupBox { color: #81C784; font-weight: bold; }")
        v = QVBoxLayout(gb)
        v.setContentsMargins(8, 12, 8, 8)

        file_row = QHBoxLayout()
        self.edt_video = QLineEdit()
        self.edt_video.setPlaceholderText("Pfad zum Video...")
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        file_row.addWidget(self.edt_video, 1)
        file_row.addWidget(self.btn_browse, 0)
        v.addLayout(file_row)

        info_row = QHBoxLayout()
        self.lbl_video_info = QLabel("— no video loaded —")
        self.lbl_video_info.setStyleSheet("color: #aaa; font-size: 11px;")
        info_row.addWidget(self.lbl_video_info)
        v.addLayout(info_row)

        # Mini-Preview
        self.preview_label = QLabel()
        self.preview_label.setFixedHeight(90)
        self.preview_label.setStyleSheet("background: #111; border: 1px solid #333;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setText("preview")
        v.addWidget(self.preview_label)

        opt_form = QFormLayout()
        self.spin_target_w = QSpinBox(); self.spin_target_w.setRange(32, 512); self.spin_target_w.setValue(64)
        self.spin_target_h = QSpinBox(); self.spin_target_h.setRange(32, 512); self.spin_target_h.setValue(64)
        res_row = QHBoxLayout()
        res_row.addWidget(self.spin_target_w); res_row.addWidget(QLabel("×")); res_row.addWidget(self.spin_target_h)
        res_wrap = QWidget(); res_wrap.setLayout(res_row)
        opt_form.addRow("Resample to:", res_wrap)

        self.chk_loop = QCheckBox("Loop video")
        self.chk_loop.setChecked(True)
        opt_form.addRow(self.chk_loop)
        v.addLayout(opt_form)
        return gb

    def _build_status_group(self) -> QGroupBox:
        gb = QGroupBox("Status")
        gb.setStyleSheet("QGroupBox { color: #FFB74D; font-weight: bold; }")
        v = QVBoxLayout(gb)
        v.setContentsMargins(8, 12, 8, 8)
        self.lbl_state = QLabel("State: idle")
        self.lbl_neuron_count = QLabel("Neurons: —")
        self.lbl_connections = QLabel("Connections: 0")
        self.lbl_frame = QLabel("Frame: —")
        self.lbl_nest_t = QLabel("NEST t: 0.0 ms")
        for l in (self.lbl_state, self.lbl_neuron_count, self.lbl_connections,
                  self.lbl_frame, self.lbl_nest_t):
            l.setStyleSheet("color: #E0E0E0; font-family: monospace; font-size: 11px;")
            v.addWidget(l)
        v.addStretch(1)
        return gb

    def _build_connection_group(self) -> QGroupBox:
        gb = QGroupBox("3. Output Connections  ·  Retina Ganglia → Graph Population")
        gb.setStyleSheet("QGroupBox { color: #CE93D8; font-weight: bold; }")
        v = QVBoxLayout(gb)
        v.setContentsMargins(8, 12, 8, 8)

        hint = QLabel(
            "Pro Retina-Ausgang mindestens 1 Target-Zeile. Per '+' weitere Ziele "
            "fuer denselben Ausgang hinzufuegen, per '×' entfernen. "
            "'Connect All' verbindet alle Zeilen mit gueltigem Ziel."
        )
        hint.setStyleSheet("color: #aaa; font-size: 11px;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        # Scrollable Container fuer die Pop-Sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        self._pop_sections_container = QWidget()
        self._pop_sections_container.setStyleSheet("background: transparent;")
        sections_layout = QVBoxLayout(self._pop_sections_container)
        sections_layout.setContentsMargins(0, 0, 0, 0)
        sections_layout.setSpacing(6)

        # State-Container: pro Pop-Name ein Dict mit GroupBox, Rows-Layout, Liste der Rows
        # self._pop_sections[pop_name] = {
        #    'group': QGroupBox,
        #    'rows_layout': QVBoxLayout,
        #    'rows': List[_TargetRow],
        # }
        self._pop_sections: Dict[str, Dict[str, Any]] = {}

        for pop_name in GANGLION_POPS:
            section = self._build_pop_section(pop_name)
            sections_layout.addWidget(section['group'])
            self._pop_sections[pop_name] = section
            # Start mit einer Target-Zeile pro Pop
            self._add_target_row(pop_name)

        sections_layout.addStretch(1)
        scroll.setWidget(self._pop_sections_container)
        v.addWidget(scroll, 1)

        # Globale Action-Buttons
        row = QHBoxLayout()
        self.btn_connect_all = QPushButton("Connect All")
        self.btn_connect_all.setStyleSheet("background-color: #1565C0; color: white; padding: 4px 16px;")
        self.btn_connect_all.clicked.connect(self._on_connect_all)

        self.btn_disconnect_all = QPushButton("Disconnect All (rebuild)")
        self.btn_disconnect_all.setStyleSheet("background-color: #EF6C00; color: white; padding: 4px 16px;")
        self.btn_disconnect_all.clicked.connect(self._on_disconnect_all)

        self.btn_refresh_targets = QPushButton("⟳ Refresh target list")
        self.btn_refresh_targets.clicked.connect(self._refresh_target_selectors)

        row.addWidget(self.btn_connect_all)
        row.addWidget(self.btn_disconnect_all)
        row.addStretch(1)
        row.addWidget(self.btn_refresh_targets)
        v.addLayout(row)

        return gb

    def _build_pop_section(self, pop_name: str) -> Dict[str, Any]:
        """Erstellt die GroupBox fuer einen Ganglion-Ausgang."""
        group = QGroupBox(pop_name)
        group.setStyleSheet(
            "QGroupBox { color: #FFD54F; font-weight: bold; border: 1px solid #444; "
            "border-radius: 4px; margin-top: 8px; padding-top: 8px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
        )
        vbox = QVBoxLayout(group)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(3)

        # Rows-Container (VBox) - hier kommen die einzelnen Target-Zeilen rein
        rows_container = QWidget()
        rows_layout = QVBoxLayout(rows_container)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(2)
        vbox.addWidget(rows_container)

        # + Button zum Hinzufuegen weiterer Targets
        btn_add = QPushButton(f"+ weiteres Ziel fuer {pop_name}")
        btn_add.setStyleSheet(
            "background-color: #37474F; color: #B0BEC5; padding: 3px; "
            "font-size: 10px; border: 1px dashed #546E7A; border-radius: 2px;"
        )
        btn_add.clicked.connect(lambda _, p=pop_name: self._add_target_row(p))
        vbox.addWidget(btn_add)

        return {
            'group': group,
            'rows_layout': rows_layout,
            'rows': [],
        }

    def _add_target_row(self, pop_name: str) -> '_TargetRow':
        """Fuegt eine neue Target-Zeile zum gegebenen Ganglion-Pop hinzu."""
        section = self._pop_sections.get(pop_name)
        if section is None:
            return None
        row = _TargetRow(pop_name=pop_name, parent_tab=self)
        row.connectRequested.connect(self._on_target_row_connect)
        row.removeRequested.connect(self._on_target_row_remove)
        section['rows'].append(row)
        section['rows_layout'].addWidget(row)
        # Initiale Befuellung der Graph-Auswahl
        row.populate_graphs(self.graph_list)
        return row

    def _on_target_row_remove(self, row: '_TargetRow'):
        pop_name = row.pop_name
        section = self._pop_sections.get(pop_name)
        if section is None:
            return
        # Mindestens eine Zeile pro Pop beibehalten - sonst wird's verwirrend
        if len(section['rows']) <= 1:
            self._log(f"Mindestens eine Target-Zeile pro Retina-Pop erforderlich.",
                      color="#FFA726")
            return
        section['rows'].remove(row)
        section['rows_layout'].removeWidget(row)
        row.setParent(None)
        row.deleteLater()

    def _iter_all_rows(self):
        """Iteriert ueber alle Target-Rows, gibt (pop_name, row)-Tupel zurueck."""
        for pop_name, section in self._pop_sections.items():
            for row in section['rows']:
                yield pop_name, row

    def _build_playback_group(self) -> QGroupBox:
        gb = QGroupBox("4. Playback")
        gb.setStyleSheet("QGroupBox { color: #FF8A65; font-weight: bold; }")
        h = QHBoxLayout(gb)
        h.setContentsMargins(8, 12, 8, 8)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setStyleSheet("background-color: #388E3C; color: white; "
                                    "padding: 8px 20px; font-size: 14px; font-weight: bold;")
        self.btn_play.clicked.connect(self._on_play_clicked)

        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setStyleSheet("background-color: #F9A825; color: white; "
                                     "padding: 8px 20px; font-size: 14px; font-weight: bold;")
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        self.btn_pause.setEnabled(False)

        self.btn_rewind = QPushButton("⏮ Rewind")
        self.btn_rewind.clicked.connect(self._on_rewind_clicked)

        self.spin_fps_override = QDoubleSpinBox()
        self.spin_fps_override.setRange(0, 120)
        self.spin_fps_override.setValue(0)
        self.spin_fps_override.setSingleStep(1)
        self.spin_fps_override.setSpecialValueText("(video default)")

        self.spin_sim_per_frame = QDoubleSpinBox()
        self.spin_sim_per_frame.setRange(1, 500)
        self.spin_sim_per_frame.setValue(33.3)
        self.spin_sim_per_frame.setSingleStep(1)

        self.progress = QProgressBar()
        self.progress.setMinimum(0); self.progress.setMaximum(100); self.progress.setValue(0)

        h.addWidget(self.btn_play)
        h.addWidget(self.btn_pause)
        h.addWidget(self.btn_rewind)
        h.addWidget(QLabel("FPS override:"))
        h.addWidget(self.spin_fps_override)
        h.addWidget(QLabel("ms/Frame:"))
        h.addWidget(self.spin_sim_per_frame)
        h.addWidget(self.progress, 1)
        return gb

    def _build_log_group(self) -> QGroupBox:
        gb = QGroupBox("Log")
        gb.setStyleSheet("QGroupBox { color: #90A4AE; font-weight: bold; }")
        v = QVBoxLayout(gb)
        v.setContentsMargins(8, 12, 8, 8)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background-color: #0D1117; color: #C9D1D9; "
                                   "font-family: monospace; font-size: 11px;")
        self.txt_log.setMaximumHeight(120)
        v.addWidget(self.txt_log)
        return gb

    # --------------------------------------------------------------------
    #  LOGGING
    # --------------------------------------------------------------------

    def _log(self, msg, color="#C9D1D9"):
        self.txt_log.append(f'<span style="color:{color}">{msg}</span>')
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())
        print(f"[RetinaTest] {msg}")

    # --------------------------------------------------------------------
    #  TARGETS (Graph/Node/Pop-Selector in der Tabelle)
    # --------------------------------------------------------------------

    def _refresh_target_selectors(self):
        """Fuellt alle Graph-Dropdowns aller Target-Zeilen mit den aktuellen Graphen."""
        if not hasattr(self, '_pop_sections'):
            return
        for pop_name, row in self._iter_all_rows():
            row.populate_graphs(self.graph_list)

    # --------------------------------------------------------------------
    #  BUILD / DESTROY
    # --------------------------------------------------------------------

    def _on_build_clicked(self):
        if self.retina is not None:
            QMessageBox.information(self, "Already built",
                                    "Retina existiert bereits. Erst 'Destroy', dann neu bauen.")
            return
        try:
            scale = self.cmb_scale.currentData()
            variant = self.cmb_variant.currentData()
            pos = (self.spin_px.value(), self.spin_py.value(), self.spin_pz.value())
            self._log(f"Baue Retina  scale={scale}  variant={variant}  @ {pos}", color="#4FC3F7")
            QApplication.processEvents()

            params, neuron_params, feeder_cfg = get_config(scale, variant)
            params = deepcopy(params)
            neuron_params = deepcopy(neuron_params)
            feeder_cfg = deepcopy(feeder_cfg)
            params['origin'] = pos
            # Input-Resolution auf Target-Groesse vom Video anpassen.
            # Wichtig: input_resolution lebt im FEEDER-Config, nicht in params.
            tw = self.spin_target_w.value(); th = self.spin_target_h.value()
            feeder_cfg['input_resolution'] = (th, tw)

            self.retina = Retina(params=params, neuron_params=neuron_params, verbose=False)
            self.retina.build()
            self.retina.connect()
            self.feeder = self.retina.create_input_feeder(feeder_cfg)

            # Spike-Recorder fuer jede Ganglion-Pop (fuer Live-Stats)
            for name, pop in self.retina.get_output_populations().items():
                sr = nest.Create('spike_recorder')
                nest.Connect(pop, sr)
                self.output_spike_recorders[name] = sr

            # Virtual Graph fuer Live-Viz-Integration
            self.virtual_graph = VirtualRetinaGraph(self.VIRTUAL_GRAPH_ID, self.retina)
            if self.virtual_graph not in self.graph_list:
                self.graph_list.append(self.virtual_graph)

            total_n = sum(self.retina.get_counts().values())
            self.lbl_neuron_count.setText(f"Neurons: {total_n:,}")
            self.lbl_state.setText("State: built")
            self.btn_build.setEnabled(False)
            self.btn_destroy.setEnabled(True)
            self._log(f"✓ Retina bereit: {total_n:,} Neuronen, "
                      f"{len(self.retina.get_output_populations())} Ganglion-Pops", color="#81C784")

            self._refresh_target_selectors()
            self.retinaBuilt.emit()
            self.requestVizRefresh.emit()
        except Exception as e:
            self._log(f"✗ Build fehlgeschlagen: {e}", color="#EF5350")
            traceback.print_exc()
            self.retina = None
            self.feeder = None

    def _on_destroy_clicked(self):
        """Baut die Retina ab. Achtung: nest.ResetKernel() vernichtet sie mit."""
        self._on_pause_clicked()
        if self.virtual_graph is not None and self.virtual_graph in self.graph_list:
            self.graph_list.remove(self.virtual_graph)
        self.virtual_graph = None
        self.retina = None
        self.feeder = None
        self.output_spike_recorders.clear()
        self.registered_connections.clear()
        self.lbl_state.setText("State: idle")
        self.lbl_neuron_count.setText("Neurons: —")
        self.lbl_connections.setText("Connections: 0")
        # Connected-Flag aller Target-Zeilen zuruecksetzen
        if hasattr(self, '_pop_sections'):
            for pop_name, row in self._iter_all_rows():
                row.set_connected(False)
        self.btn_build.setEnabled(True)
        self.btn_destroy.setEnabled(False)
        self._log("Retina zerstoert (Python-seitig). "
                  "NEST-Populationen bleiben bis ResetKernel aktiv.", color="#FFA726")
        self.retinaDestroyed.emit()
        self.requestVizRefresh.emit()

    # --------------------------------------------------------------------
    #  CONNECTIONS
    # --------------------------------------------------------------------

    def _on_target_row_connect(self, row: '_TargetRow'):
        """Handler fuer den Connect-Button einer einzelnen Target-Zeile."""
        if self.retina is None:
            QMessageBox.warning(self, "No Retina", "Erst 'Build Retina' druecken.")
            return

        gid = row.get_graph_id()
        nid = row.get_node_id()
        pop_idx = row.get_pop_idx()
        weight = row.get_weight()
        delay = row.get_delay()
        pop_name = row.pop_name

        if gid is None or nid is None:
            self._log(f"[{pop_name}] kein gueltiges Ziel", color="#EF5350")
            return

        target_graph = next((g for g in self.graph_list if g.graph_id == gid), None)
        if target_graph is None:
            self._log(f"[{pop_name}] Graph {gid} nicht gefunden", color="#EF5350")
            return
        target_node = next((n for n in target_graph.node_list if n.id == nid), None)
        if target_node is None:
            self._log(f"[{pop_name}] Node {nid} nicht gefunden", color="#EF5350")
            return

        src_pop = self.retina.populations.get(pop_name, None)
        if src_pop is None or len(src_pop) == 0:
            self._log(f"[{pop_name}] Retina-Pop leer", color="#EF5350")
            return

        # Validitaet der Retina-NodeCollection pruefen
        try:
            nest.GetStatus(src_pop[:1])
        except Exception:
            self._log(f"⚠ Retina-NodeCollection invalidiert (ResetKernel?). "
                      "Clear State und neu bauen.", color="#EF5350")
            self.on_nest_kernel_reset()
            return

        # Target-Pops sammeln
        if pop_idx is None:
            target_pops = [p for p in getattr(target_node, 'population', []) if p is not None]
        else:
            if pop_idx < len(target_node.population) and target_node.population[pop_idx] is not None:
                target_pops = [target_node.population[pop_idx]]
            else:
                self._log(f"[{pop_name}] Pop {pop_idx} in target node leer", color="#EF5350")
                return

        try:
            for tp in target_pops:
                nest.Connect(
                    src_pop, tp,
                    conn_spec={'rule': 'pairwise_bernoulli', 'p': 0.1,
                               'allow_autapses': False, 'allow_multapses': True},
                    syn_spec={'weight': weight, 'delay': delay,
                              'synapse_model': 'static_synapse'},
                )
                self.registered_connections.append((pop_name, src_pop, tp))
            row.set_connected(True)
            self.lbl_connections.setText(f"Connections: {len(self.registered_connections)}")
            self._log(f"✓ {pop_name} -> G{gid}.N{nid}.{pop_idx}  w={weight}  d={delay}",
                      color="#81C784")
        except Exception as e:
            self._log(f"✗ Connect fehlgeschlagen [{pop_name}]: {e}", color="#EF5350")
            traceback.print_exc()

    def _on_connect_all(self):
        """Verbindet alle Target-Zeilen mit gueltigem Ziel, die noch nicht connected sind."""
        for pop_name, row in self._iter_all_rows():
            if row.is_connected():
                continue
            if row.get_graph_id() is not None and row.get_node_id() is not None:
                self._on_target_row_connect(row)

    def _on_disconnect_all(self):
        """Rebuild-only: in NEST kann man nicht selektiv disconnecten. Wir loeschen nur
        die Python-Liste und setzen den UI-State zurueck. Der naechste ResetKernel baut
        sauber neu."""
        self.registered_connections.clear()
        for pop_name, row in self._iter_all_rows():
            row.set_connected(False)
        self.lbl_connections.setText("Connections: 0 (pending ResetKernel)")
        self._log("Connections aus Python-State entfernt. "
                  "Fuer echten Reset: NEST-Kernel reset ueber Main.", color="#FFA726")

    # --------------------------------------------------------------------
    #  VIDEO
    # --------------------------------------------------------------------

    def _on_browse_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Video auswaehlen", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        if not path:
            return
        self.edt_video.setText(path)
        self._try_open_video(path)

    def _try_open_video(self, path):
        try:
            if self.video_reader is not None:
                self.video_reader.release()
                self.video_reader = None
            target = (self.spin_target_h.value(), self.spin_target_w.value())
            self.video_reader = VideoFrameReader(path, target_resolution=target)
            info = (f"{os.path.basename(path)}  |  "
                    f"{self.video_reader.width}×{self.video_reader.height}  |  "
                    f"{self.video_reader.fps:.1f} fps  |  "
                    f"{self.video_reader.n_frames} frames")
            self.lbl_video_info.setText(info)
            self._show_preview_frame()
            self._log(f"Video geladen: {info}", color="#4FC3F7")
        except Exception as e:
            self._log(f"✗ Video laden fehlgeschlagen: {e}", color="#EF5350")

    def _show_preview_frame(self):
        if self.video_reader is None:
            return
        self.video_reader.seek(0)
        triple = self.video_reader.read_next()
        if triple is None:
            return
        _, _, bgr = triple
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_label.width(), self.preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)
        self.video_reader.seek(0)

    # --------------------------------------------------------------------
    #  PLAYBACK
    # --------------------------------------------------------------------

    def _on_play_clicked(self):
        if self.retina is None:
            QMessageBox.warning(self, "No Retina", "Erst 'Build Retina' druecken.")
            return
        if self.video_reader is None:
            QMessageBox.warning(self, "No Video", "Erst ein Video laden.")
            return

        fps = self.spin_fps_override.value() or self.video_reader.fps
        interval_ms = max(1, int(round(1000.0 / fps)))
        self._playing = True
        self.play_timer.start(interval_ms)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self._log(f"▶ Play @ {fps:.1f} fps  ({interval_ms} ms/Tick, "
                  f"{self.spin_sim_per_frame.value():.1f} ms NEST/Frame)",
                  color="#81C784")

        # Auto-switch zum INTERACTIVE-Tab + globale Simulation starten,
        # damit der User die Live-Aktivitaet sieht
        if self.main_window is not None:
            try:
                # Zum INTERACTIVE-Tab (main_stack index 1) switchen
                if hasattr(self.main_window, '_switch_main_view'):
                    self.main_window._switch_main_view(1)
                # Globalen Sim-Timer starten (falls nicht schon laeuft) - dieser
                # macht nest.Simulate() im Hintergrund. Wir machen das NICHT mehr
                # selbst im Play-Tick (siehe _on_play_tick), sondern feeden nur
                # Frames synchron zum Video-FPS.
                sim_timer = getattr(self.main_window, 'sim_timer', None)
                if sim_timer is not None and not sim_timer.isActive():
                    if hasattr(self.main_window, '_global_start'):
                        self.main_window._global_start()
            except Exception as e:
                self._log(f"Auto-switch/start warning: {e}", color="#FFA726")

    def _on_pause_clicked(self):
        self._playing = False
        self.play_timer.stop()
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        # Globalen Sim-Timer ebenfalls pausieren, damit nicht in die Leere simuliert wird
        if self.main_window is not None:
            try:
                if hasattr(self.main_window, '_global_pause'):
                    sim_timer = getattr(self.main_window, 'sim_timer', None)
                    if sim_timer is not None and sim_timer.isActive():
                        self.main_window._global_pause()
            except Exception:
                pass

    def _on_rewind_clicked(self):
        if self.video_reader is not None:
            self.video_reader.seek(0)
            self._show_preview_frame()
            self.progress.setValue(0)
            self.lbl_frame.setText("Frame: 0")

    def _on_play_tick(self):
        if not self._playing or self.retina is None or self.video_reader is None:
            return

        triple = self.video_reader.read_next()
        if triple is None:
            if self.chk_loop.isChecked():
                self.video_reader.seek(0)
                triple = self.video_reader.read_next()
                if triple is None:
                    self._on_pause_clicked()
                    return
            else:
                self._on_pause_clicked()
                return
        lms, intensity, bgr = triple

        try:
            self.feeder.feed(lms, intensity)
        except Exception as e:
            self._log(f"feed() Fehler: {e}", color="#EF5350")
            self._on_pause_clicked()
            return

        # Falls der globale Sim-Timer (Main.py) bereits laeuft, macht der das
        # nest.Simulate() selbst. Wir feeden nur Frames. Andernfalls simulieren
        # wir hier direkt (standalone-Modus oder wenn der User den globalen
        # Sim nicht starten will).
        global_sim_running = False
        if self.main_window is not None:
            st = getattr(self.main_window, 'sim_timer', None)
            if st is not None and st.isActive():
                global_sim_running = True

        if not global_sim_running:
            sim_ms = float(self.spin_sim_per_frame.value())
            try:
                nest.Simulate(sim_ms)
            except Exception as e:
                self._log(f"Simulate() Fehler: {e}", color="#EF5350")
                self._on_pause_clicked()
                return

        try:
            self.lbl_nest_t.setText(f"NEST t: {nest.GetKernelStatus('biological_time'):.1f} ms")
        except Exception:
            pass

        self.lbl_frame.setText(f"Frame: {self.video_reader.current_idx}")
        if self.video_reader.n_frames > 0:
            pct = int(100 * self.video_reader.current_idx / max(1, self.video_reader.n_frames))
            self.progress.setValue(pct)

        # Preview aktualisieren (jedes 5. Frame reicht optisch)
        if self.video_reader.current_idx % 5 == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.preview_label.width(), self.preview_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setPixmap(pix)

        self.requestVizRefresh.emit()

    # --------------------------------------------------------------------
    #  HOOKS fuer Main.py
    # --------------------------------------------------------------------

    def is_retina_active(self) -> bool:
        return self.retina is not None

    def on_tab_activated(self):
        """Wird von Main.py aufgerufen wenn der Tab angezeigt wird."""
        self._refresh_target_selectors()

    def on_nest_kernel_reset(self):
        """Externer Hook: wird von Main.py aufgerufen wenn ein ResetKernel stattgefunden hat.

        Nach einem ResetKernel sind alle NEST-NodeCollections der Retina tot.
        Die Python-Referenzen zeigen ins Leere, jeder nest.Connect() wuerde
        'InvalidNodeCollection' werfen. Wir muessen den internen State clearen,
        damit der User eine neue Retina bauen kann.
        """
        if self.retina is None and self.virtual_graph is None:
            return   # Bereits clean
        self._on_pause_clicked()
        if self.virtual_graph is not None and self.virtual_graph in self.graph_list:
            try:
                self.graph_list.remove(self.virtual_graph)
            except ValueError:
                pass
        self.virtual_graph = None
        self.retina = None
        self.feeder = None
        self.output_spike_recorders.clear()
        self.registered_connections.clear()
        self.lbl_state.setText("State: idle (kernel reset)")
        self.lbl_neuron_count.setText("Neurons: —")
        self.lbl_connections.setText("Connections: 0")
        for pop_name, row in self._iter_all_rows():
            row.set_connected(False)
        self.btn_build.setEnabled(True)
        self.btn_destroy.setEnabled(False)
        self._log("⚠ NEST-Kernel wurde von aussen resettet. "
                  "Retina-State gecleart. Build neu druecken fuer eine neue Retina.",
                  color="#FFA726")
        self.retinaDestroyed.emit()


# ===========================================================================
#  TARGET ROW WIDGET
# ===========================================================================
#
# Eine einzelne Ziel-Zeile fuer einen Retina-Output. Eine Pop kann beliebig
# viele davon haben (dynamisch per + Button).

class _TargetRow(QWidget):
    connectRequested = pyqtSignal(object)   # self
    removeRequested  = pyqtSignal(object)   # self

    def __init__(self, pop_name: str, parent_tab: 'RetinaTestTabWidget', parent=None):
        super().__init__(parent)
        self.pop_name = pop_name
        self.parent_tab = parent_tab
        self._connected = False
        self._init_ui()

    def _init_ui(self):
        h = QHBoxLayout(self)
        h.setContentsMargins(2, 2, 2, 2)
        h.setSpacing(4)

        self.cmb_graph = QComboBox()
        self.cmb_graph.setMinimumWidth(130)
        self.cmb_graph.currentIndexChanged.connect(self._on_graph_changed)

        self.cmb_node = QComboBox()
        self.cmb_node.setMinimumWidth(130)
        self.cmb_node.currentIndexChanged.connect(self._on_node_changed)

        self.cmb_pop = QComboBox()
        self.cmb_pop.setMinimumWidth(100)

        self.sp_w = QDoubleSpinBox()
        self.sp_w.setRange(-1000, 1000); self.sp_w.setValue(1.0); self.sp_w.setSingleStep(0.1)
        self.sp_w.setFixedWidth(60); self.sp_w.setPrefix("w ")

        self.sp_d = QDoubleSpinBox()
        self.sp_d.setRange(0.1, 100); self.sp_d.setValue(1.0); self.sp_d.setSingleStep(0.1)
        self.sp_d.setFixedWidth(60); self.sp_d.setPrefix("d ")

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setStyleSheet("background-color: #1976D2; color: white; padding: 2px 8px;")
        self.btn_connect.clicked.connect(lambda: self.connectRequested.emit(self))

        self.btn_remove = QPushButton("×")
        self.btn_remove.setFixedWidth(22)
        self.btn_remove.setStyleSheet(
            "QPushButton { background-color: #5D4037; color: #FFAB91; border: none; "
            "border-radius: 2px; font-weight: bold; } "
            "QPushButton:hover { background-color: #D32F2F; color: white; }"
        )
        self.btn_remove.clicked.connect(lambda: self.removeRequested.emit(self))

        h.addWidget(QLabel("→"))
        h.addWidget(self.cmb_graph, 1)
        h.addWidget(self.cmb_node, 1)
        h.addWidget(self.cmb_pop, 1)
        h.addWidget(self.sp_w)
        h.addWidget(self.sp_d)
        h.addWidget(self.btn_connect)
        h.addWidget(self.btn_remove)

    # -- Public getters for the tab --
    def get_graph_id(self):
        return self.cmb_graph.currentData()

    def get_node_id(self):
        return self.cmb_node.currentData()

    def get_pop_idx(self):
        return self.cmb_pop.currentData()

    def get_weight(self) -> float:
        return float(self.sp_w.value())

    def get_delay(self) -> float:
        return max(0.1, float(self.sp_d.value()))

    def is_connected(self) -> bool:
        return self._connected

    def set_connected(self, connected: bool):
        self._connected = connected
        if connected:
            self.btn_connect.setText("✓ Connected")
            self.btn_connect.setStyleSheet("background-color: #2E7D32; color: white; padding: 2px 8px;")
        else:
            self.btn_connect.setText("Connect")
            self.btn_connect.setStyleSheet("background-color: #1976D2; color: white; padding: 2px 8px;")

    # -- Dropdown-Befuellung --
    def populate_graphs(self, graph_list):
        """Befuellt das Graph-Dropdown mit allen nicht-virtuellen Graphen."""
        was_blocked = self.cmb_graph.blockSignals(True)
        prev = self.cmb_graph.currentData()
        self.cmb_graph.clear()
        self.cmb_graph.addItem("— select graph —", None)
        for g in graph_list:
            if getattr(g, 'is_virtual', False):
                continue
            self.cmb_graph.addItem(f"{getattr(g, 'graph_name', 'Graph')} (id={g.graph_id})",
                                   g.graph_id)
        if prev is not None:
            for i in range(self.cmb_graph.count()):
                if self.cmb_graph.itemData(i) == prev:
                    self.cmb_graph.setCurrentIndex(i); break
        self.cmb_graph.blockSignals(was_blocked)
        self._on_graph_changed()

    def _on_graph_changed(self):
        gid = self.cmb_graph.currentData()
        was_blocked = self.cmb_node.blockSignals(True)
        prev_nid = self.cmb_node.currentData()
        self.cmb_node.clear()
        self.cmb_node.addItem("— select node —", None)
        if gid is not None and self.parent_tab is not None:
            graph = next((g for g in self.parent_tab.graph_list if g.graph_id == gid), None)
            if graph is not None:
                for node in graph.node_list:
                    self.cmb_node.addItem(f"{getattr(node, 'name', 'Node')} (id={node.id})",
                                          node.id)
        if prev_nid is not None:
            for i in range(self.cmb_node.count()):
                if self.cmb_node.itemData(i) == prev_nid:
                    self.cmb_node.setCurrentIndex(i); break
        self.cmb_node.blockSignals(was_blocked)
        self._on_node_changed()

    def _on_node_changed(self):
        gid = self.cmb_graph.currentData()
        nid = self.cmb_node.currentData()
        was_blocked = self.cmb_pop.blockSignals(True)
        self.cmb_pop.clear()
        self.cmb_pop.addItem("— all pops —", None)
        if gid is not None and nid is not None and self.parent_tab is not None:
            graph = next((g for g in self.parent_tab.graph_list if g.graph_id == gid), None)
            if graph is not None:
                node = next((n for n in graph.node_list if n.id == nid), None)
                if node is not None and hasattr(node, 'population'):
                    for i, pop in enumerate(node.population):
                        if pop is None:
                            continue
                        try:
                            sz = len(pop)
                        except Exception:
                            sz = '?'
                        self.cmb_pop.addItem(f"Pop {i} ({sz}n)", i)
        self.cmb_pop.blockSignals(was_blocked)


# ===========================================================================
#  STANDALONE ENTRY POINT
# ===========================================================================
#
# Startet das Widget eigenstaendig ohne Neuroticks. Dann gibt es natuerlich
# keine realen Graph-Targets zum Verbinden (graph_list ist leer), aber:
#   - Retina bauen
#   - Video laden
#   - Play/Pause
#   - Spike-Stats pro Output-Pop
# funktionieren alle. Nuetzlich zum Parametertunen und Video-Testing.
#
# Aufruf:   python3 RetinaTestTab.py
#           python3 RetinaTestTab.py --video /pfad/zu/video.mp4
# ---------------------------------------------------------------------------

def _standalone_main():
    import argparse
    import sys
    import nest as _nest

    parser = argparse.ArgumentParser(description="Retina-Test standalone")
    parser.add_argument('--video', type=str, default=None,
                        help='Pfad zu einem Video-File (optional, kann auch im UI geladen werden)')
    parser.add_argument('--scale', type=str, default='tiny',
                        help='Retina scale (tiny/small/medium/large)')
    parser.add_argument('--variant', type=str, default='default',
                        help='Retina variant')
    parser.add_argument('--resolution', type=float, default=0.1,
                        help='NEST resolution (ms)')
    parser.add_argument('--threads', type=int, default=1,
                        help='NEST local_num_threads')
    args = parser.parse_args()

    _nest.ResetKernel()
    _nest.SetKernelStatus({
        'resolution': args.resolution,
        'print_time': False,
        'local_num_threads': args.threads,
    })

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    # Dark palette damit's zum Hauptprogramm passt
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,         QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText,     QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base,           QColor(20, 20, 20))
    palette.setColor(QPalette.ColorRole.AlternateBase,  QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.Text,           QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button,         QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText,     QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Highlight,      QColor(38, 79, 120))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    # Standalone: leere graph_list -> Connection-Targets sind leer,
    # aber alle anderen Features funktionieren.
    standalone_graph_list = []

    container = QWidget()
    container.setWindowTitle("Retina Standalone Test")
    container.resize(1400, 900)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    widget = RetinaTestTabWidget(standalone_graph_list, main_window=None)
    layout.addWidget(widget)

    if args.video:
        widget.edt_video.setText(args.video)
        widget._try_open_video(args.video)

    # Set default Scale/Variant
    for i in range(widget.cmb_scale.count()):
        if widget.cmb_scale.itemData(i) == args.scale:
            widget.cmb_scale.setCurrentIndex(i); break
    for i in range(widget.cmb_variant.count()):
        if widget.cmb_variant.itemData(i) == args.variant:
            widget.cmb_variant.setCurrentIndex(i); break

    container.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    _standalone_main()
