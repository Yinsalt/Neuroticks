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
        BIPOLAR_POPS,
    )
    from retina_scales import get_config, list_scales, list_variants
    HAS_RETINA = True
except Exception as e:
    HAS_RETINA = False
    _RETINA_IMPORT_ERROR = str(e)
    GANGLION_POPS = []
    BIPOLAR_POPS = []


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


# ===========================================================================
#  CANONICAL RETINA → LGN MAPPING
# ===========================================================================
#
# These tables mirror retina_factory.connect_retina_to_lgn — the same code
# path server.py uses for headless multi-agent runs. Keeping them here lets
# the eye tab apply *exactly* that wiring when a matching LGN graph is
# present, so the build path matches between live debugging and server
# deployment.
#
# Format: (lgn_node_name, indegree, weight, delay)
#
# fixed_indegree is preferred over pairwise_bernoulli because it stays
# stable across retina scales: every LGN neuron always gets the same
# number of inputs regardless of how big the source pop is. At small
# scales the indegree caps to (n_src - 1).

LGN_GRAPH_NAME = 'LGN_RELAY_POPS'

SERVER_RETINA_TO_LGN = {
    # Foveal Parvo: tight 1:1-ish focus, ~30 cones per midget relay
    'midget_ON_ganglion_foveal':       ('LGN_PARVO_ON_FOVEAL',      30, 1.2, 1.5),
    'midget_OFF_ganglion_foveal':      ('LGN_PARVO_OFF_FOVEAL',     30, 1.2, 1.5),
    # Peripheral Parvo: more convergence, larger RFs
    'midget_ON_ganglion_peripheral':   ('LGN_PARVO_ON_PERIPHERAL',  60, 0.8, 1.5),
    'midget_OFF_ganglion_peripheral':  ('LGN_PARVO_OFF_PERIPHERAL', 60, 0.8, 1.5),
    # Magno: parasol converges ~30-80 per magno relay; 15 is biologically
    # reasonable and scales cleanly without rejection-sampler issues at
    # small scales (server.py comment).
    'parasol_ON_ganglion':             ('LGN_MAGNO_ON',             15, 1.5, 1.5),
    'parasol_OFF_ganglion':            ('LGN_MAGNO_OFF',            15, 1.5, 1.5),
    # Konio: smaller fan-in because S-cones are sparse
    'konio_ganglion_peripheral':       ('LGN_KONIOCELLULAR',        40, 1.0, 1.5),
}


# Connection rules exposed in the per-row "Advanced" panel. fixed_indegree
# is the server-canonical default; the others are there for experimentation.
CONNECTION_RULES = (
    'fixed_indegree',
    'fixed_outdegree',
    'pairwise_bernoulli',
    'all_to_all',
    'one_to_one',
)


class VirtualRetinaNode:
    """Duck-typed Node replacement for a retina population."""

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
    """Duck-typed Graph replacement for the retina. Gets registered in graph_list."""

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
    """Thin wrapper around cv2.VideoCapture, returns LMS + intensity.

    BGR is decoded, normalized to [0,1] RGB, and used directly as LMS.
    For test purposes that's fine — a real LMS conversion would need a
    Hunt-Pointer-Estevez matrix, but the retina just consumes three
    channels and treats them as L/M/S regardless.
    """

    def __init__(self, path, target_resolution=(64, 64)):
        if not HAS_CV2:
            raise RuntimeError("OpenCV (cv2) not installed — required for video input.")
        self.path = path
        self.target_resolution = target_resolution
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {path}")
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
        # Resample to target resolution
        h, w = self.target_resolution
        bgr_small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lms = rgb  # approximation: use RGB as LMS
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
    """Eye tab: retina testing and video-driven feed-in."""

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
        # Diagnostic recorders on the BIPOLAR stage so we can detect
        # whether the bipolar→ganglion intra-retina connection is
        # firing or not. Critical for telling "retina internal problem"
        # apart from "downstream visualization problem".
        self.bipolar_spike_recorders: Dict[str, Any] = {}
        self.registered_connections: List[Tuple[str, Any, Any]] = []

        self.video_reader: Optional[VideoFrameReader] = None

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)
        self._playing = False
        # Accumulator-architecture state, set by _on_play_clicked from FPS.
        # _frame_bio_ms     = bio time per video frame (1000/fps)
        # _frame_bio_accum  = bio time elapsed since current frame was fed
        # _current_frame_fed = whether the current frame has been fed yet
        self._frame_bio_ms = None
        self._frame_bio_accum = 0.0
        self._current_frame_fed = False
        self._last_bgr = None
        self._last_preview_idx = -1

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
                f"Retina modules could not be loaded:\n{_RETINA_IMPORT_ERROR}"
                f"\n\nMake sure retina_main.py and retina_scales.py are on the PYTHONPATH."
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
        # Scale change → suggest matching input resolution. Doesn't lock the
        # spinner: user can still override afterwards. Without this, picking
        # scale=3 (which expects 128×128) and leaving the resolution at the
        # default 64×64 silently downsamples every frame and the foveal
        # cones get a fraction of the data they were tuned for.
        self.cmb_scale.currentIndexChanged.connect(self._on_scale_changed)

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

    def _on_scale_changed(self, _idx: int):
        """When the user picks a scale, update the input-resolution spinners
        to the recommended power-of-2 for that scale. The user can still
        override afterwards if they want a custom resolution."""
        scale = self.cmb_scale.currentData()
        if scale is None:
            return
        try:
            from retina_scales import _RESOLUTION_FOR_SCALE, _resolve_scale
            s_int = _resolve_scale(scale)
            h, w = _RESOLUTION_FOR_SCALE[s_int]
        except Exception:
            return
        # Spinners may not exist yet if this fires during early init.
        if hasattr(self, 'spin_target_w') and hasattr(self, 'spin_target_h'):
            self.spin_target_w.blockSignals(True)
            self.spin_target_h.blockSignals(True)
            self.spin_target_w.setValue(int(w))
            self.spin_target_h.setValue(int(h))
            self.spin_target_w.blockSignals(False)
            self.spin_target_h.blockSignals(False)
            # If a video is already open, re-open it with the new target so
            # subsequent reads decode at the right size.
            if self.video_reader is not None:
                path = self.edt_video.text().strip()
                if path:
                    self._try_open_video(path)

    def _build_video_group(self) -> QGroupBox:
        gb = QGroupBox("2. Video Input")
        gb.setStyleSheet("QGroupBox { color: #81C784; font-weight: bold; }")
        v = QVBoxLayout(gb)
        v.setContentsMargins(8, 12, 8, 8)

        file_row = QHBoxLayout()
        self.edt_video = QLineEdit()
        self.edt_video.setPlaceholderText("Path to video file…")
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

        # Mini-preview
        self.preview_label = QLabel()
        self.preview_label.setFixedHeight(90)
        self.preview_label.setStyleSheet("background: #111; border: 1px solid #333;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setText("preview")
        v.addWidget(self.preview_label)

        opt_form = QFormLayout()
        # Range matches retina_main.VALID_INPUT_RESOLUTIONS (powers of 2 up
        # to 16384). Spinner is plain int — non-power-of-2 values get
        # snapped to the nearest valid resolution at build time.
        self.spin_target_w = QSpinBox()
        self.spin_target_w.setRange(32, 16384)
        self.spin_target_w.setValue(64)
        self.spin_target_h = QSpinBox()
        self.spin_target_h.setRange(32, 16384)
        self.spin_target_h.setValue(64)
        self.spin_target_w.setToolTip(
            "Frame width fed to the retina. Powers of 2 only "
            "(32, 64, 128, 256, 512, 1024…). Other values get snapped."
        )
        self.spin_target_h.setToolTip(self.spin_target_w.toolTip())
        # Reload video when resolution changes so the next read decodes at
        # the new size (otherwise frames still come in at the old target).
        self.spin_target_w.editingFinished.connect(self._on_target_resolution_changed)
        self.spin_target_h.editingFinished.connect(self._on_target_resolution_changed)
        res_row = QHBoxLayout()
        res_row.addWidget(self.spin_target_w)
        res_row.addWidget(QLabel("×"))
        res_row.addWidget(self.spin_target_h)
        res_wrap = QWidget(); res_wrap.setLayout(res_row)
        opt_form.addRow("Resample to:", res_wrap)

        self.chk_loop = QCheckBox("Loop video")
        self.chk_loop.setChecked(True)
        opt_form.addRow(self.chk_loop)
        v.addLayout(opt_form)
        return gb

    def _on_target_resolution_changed(self):
        """Spinner edit committed — re-open the video at the new resolution."""
        if self.video_reader is None:
            return
        path = self.edt_video.text().strip()
        if path:
            self._try_open_video(path)

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
            "Each retina output needs at least 1 target row. Use '+' to add more "
            "targets for the same output, '×' to remove. "
            "'Connect All' wires every row that has a valid target."
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

        # Global action buttons
        row = QHBoxLayout()
        self.btn_connect_all = QPushButton("Connect All")
        self.btn_connect_all.setStyleSheet("background-color: #1565C0; color: white; padding: 4px 16px;")
        self.btn_connect_all.clicked.connect(self._on_connect_all)

        self.btn_disconnect_all = QPushButton("Disconnect All (rebuild)")
        self.btn_disconnect_all.setStyleSheet("background-color: #EF6C00; color: white; padding: 4px 16px;")
        self.btn_disconnect_all.clicked.connect(self._on_disconnect_all)

        # Quick-apply the server-canonical wiring (fixed_indegree with the
        # exact indegree/weight/delay values from retina_factory). Greyed
        # out when the LGN graph isn't loaded — there's nothing to map to.
        self.btn_server_match = QPushButton("⚡ Apply Server Config (1:1)")
        self.btn_server_match.setStyleSheet(
            "background-color: #00838F; color: white; padding: 4px 16px; font-weight: bold;"
        )
        self.btn_server_match.setToolTip(
            "Set every retina pop's first target row to the exact wiring "
            "server.py uses: fixed_indegree, server-canonical indegree/weight/"
            "delay per pop. Requires the LGN_RELAY_POPS graph to be loaded."
        )
        self.btn_server_match.clicked.connect(self._apply_server_match)

        self.btn_refresh_targets = QPushButton("⟳ Refresh target list")
        self.btn_refresh_targets.clicked.connect(self._refresh_target_selectors)

        row.addWidget(self.btn_connect_all)
        row.addWidget(self.btn_disconnect_all)
        row.addWidget(self.btn_server_match)
        row.addStretch(1)
        row.addWidget(self.btn_refresh_targets)
        v.addLayout(row)

        return gb

    def _build_pop_section(self, pop_name: str) -> Dict[str, Any]:
        """Build the GroupBox for a single ganglion output."""
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
        btn_add = QPushButton(f"+ another target for {pop_name}")
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
            self._log(f"At least one target row per retina pop is required.",
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

        # Diagnostic snapshot button: prints cumulative spike counts per
        # bipolar and ganglion population to the log. Useful any time
        # during play to check whether the retina is firing internally.
        self.btn_diag = QPushButton("🔬 Status")
        self.btn_diag.setStyleSheet("background-color: #455A64; color: white; "
                                    "padding: 8px 14px; font-size: 13px;")
        self.btn_diag.setToolTip("Print cumulative spike counts of all "
                                  "bipolar and ganglion pops")
        self.btn_diag.clicked.connect(self._on_diag_clicked)

        self.spin_fps_override = QDoubleSpinBox()
        self.spin_fps_override.setRange(0, 120)
        self.spin_fps_override.setValue(0)
        self.spin_fps_override.setSingleStep(1)
        self.spin_fps_override.setSpecialValueText("(video default)")
        # When FPS changes, update the derived bio-time label live so the
        # user sees what one frame becomes in NEST time.
        self.spin_fps_override.valueChanged.connect(self._update_bio_per_frame_label)

        # NEST bio-time per frame is DERIVED from FPS, not user-set.
        # Bio-time/frame = 1000/fps, so 30fps → 33.33 ms/frame, 60fps → 16.67.
        # This keeps NEST simulation rate locked to video real-time playback,
        # which is the only sane mapping for a video → retina test.
        self.lbl_bio_per_frame = QLabel("(no video)")
        self.lbl_bio_per_frame.setStyleSheet("color: #B0BEC5; font-family: monospace;")
        self.lbl_bio_per_frame.setMinimumWidth(110)

        self.progress = QProgressBar()
        self.progress.setMinimum(0); self.progress.setMaximum(100); self.progress.setValue(0)

        h.addWidget(self.btn_play)
        h.addWidget(self.btn_pause)
        h.addWidget(self.btn_rewind)
        h.addWidget(self.btn_diag)
        h.addWidget(QLabel("FPS override:"))
        h.addWidget(self.spin_fps_override)
        h.addWidget(QLabel("→ NEST:"))
        h.addWidget(self.lbl_bio_per_frame)
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
        """Populate every target row's graph dropdown with the current graphs."""
        if not hasattr(self, '_pop_sections'):
            return
        for pop_name, row in self._iter_all_rows():
            row.populate_graphs(self.graph_list)

    def _check_server_compat(self):
        """Detect whether the loaded graph_list contains an LGN_RELAY_POPS
        graph with all the node names required for server-canonical wiring.

        Returns:
          (compatible: bool, lgn_graph: Graph or None,
           found_nodes: dict[retina_pop_name -> Node],
           missing_lgn_nodes: list[str])
        """
        lgn_graph = next(
            (g for g in self.graph_list
             if getattr(g, 'graph_name', '') == LGN_GRAPH_NAME
             and not getattr(g, 'is_virtual', False)),
            None
        )
        if lgn_graph is None:
            return False, None, {}, []

        nodes_by_name = {getattr(n, 'name', ''): n for n in lgn_graph.node_list}
        found = {}
        missing = []
        for retina_pop, (lgn_name, _, _, _) in SERVER_RETINA_TO_LGN.items():
            n = nodes_by_name.get(lgn_name)
            if n is None or not getattr(n, 'population', None):
                missing.append(lgn_name)
                continue
            found[retina_pop] = n

        compatible = (len(missing) == 0)
        return compatible, lgn_graph, found, missing

    def _apply_default_targets(self):
        """Pre-populate every retina pop's first target row with the
        corresponding LGN node (and the server-canonical fixed_indegree
        config) IF the LGN graph is loaded with all expected node names.
        Connections are NOT created here — only the UI is filled in.

        If the LGN graph is missing or some node names don't match, all
        target dropdowns stay on '— select graph —' and the user picks
        targets manually with whatever rule they prefer.
        """
        if not hasattr(self, '_pop_sections'):
            return

        compatible, lgn_graph, found_nodes, missing = self._check_server_compat()

        if lgn_graph is None:
            self._log(
                f"Server-match: no '{LGN_GRAPH_NAME}' graph loaded — "
                f"target selection stays empty. Pick targets and a "
                f"connection rule manually via the row's '▸' panel.",
                color="#9E9E9E",
            )
            self.btn_server_match.setEnabled(False)
            return

        if not compatible:
            self._log(
                f"Server-match: '{LGN_GRAPH_NAME}' is loaded but "
                f"{len(missing)} expected LGN node(s) are missing: "
                f"{', '.join(missing)}. Use the row '▸' panel to pick "
                f"targets and rules manually.",
                color="#FFA726",
            )
            self.btn_server_match.setEnabled(False)
            return

        # All 7 ganglion → LGN mappings resolve. Pre-fill every section's
        # FIRST row with the canonical target + server config. Additional
        # user rows (added via '+') stay untouched.
        n_set = 0
        for pop_name, section in self._pop_sections.items():
            mapping = SERVER_RETINA_TO_LGN.get(pop_name)
            target_node = found_nodes.get(pop_name)
            if mapping is None or target_node is None:
                continue
            _, indegree, weight, delay = mapping
            if not section['rows']:
                continue
            first_row = section['rows'][0]
            ok = first_row.set_target(lgn_graph.graph_id, target_node.id, pop_idx=0)
            if ok:
                first_row.apply_server_defaults(indegree, weight, delay)
                n_set += 1

        self.btn_server_match.setEnabled(True)
        self._log(
            f"Server-match: {n_set}/{len(SERVER_RETINA_TO_LGN)} retina pops "
            f"pre-set to {LGN_GRAPH_NAME} 1:1 (fixed_indegree, "
            f"server-canonical weights). Press 'Connect All' to wire.",
            color="#26A69A",
        )

    def _apply_server_match(self):
        """Force-apply server-canonical config to every first-row target
        regardless of current settings. Useful after the user has poked at
        rules and wants to reset to the server config."""
        # _apply_default_targets is the same logic — just call it.
        # It overwrites whatever's in the first row.
        self._apply_default_targets()

    # --------------------------------------------------------------------
    #  BUILD / DESTROY
    # --------------------------------------------------------------------

    def _on_build_clicked(self):
        if self.retina is not None:
            QMessageBox.information(self, "Already built",
                                    "Retina already exists. Press 'Destroy' first, then rebuild.")
            return
        try:
            scale = self.cmb_scale.currentData()
            variant = self.cmb_variant.currentData()
            pos = (self.spin_px.value(), self.spin_py.value(), self.spin_pz.value())
            self._log(f"Building retina  scale={scale}  variant={variant}  @ {pos}", color="#4FC3F7")
            QApplication.processEvents()

            params, neuron_params, feeder_cfg = get_config(scale, variant)
            params = deepcopy(params)
            neuron_params = deepcopy(neuron_params)
            feeder_cfg = deepcopy(feeder_cfg)
            params['origin'] = pos

            # Resolve target resolution. The retina's photoreceptor positions
            # are tuned to a specific resolution per scale — feeding a
            # mismatched resolution still works (the feeder resizes
            # internally), but coverage of the foveola degrades. So we snap
            # the spinner values to the nearest power-of-2 in
            # VALID_INPUT_RESOLUTIONS, and warn if the user is far from the
            # scale's recommended size.
            tw_raw = int(self.spin_target_w.value())
            th_raw = int(self.spin_target_h.value())
            tw = self._snap_to_power_of_2(tw_raw)
            th = self._snap_to_power_of_2(th_raw)
            if tw != tw_raw or th != th_raw:
                self._log(
                    f"Resolution snapped to power of 2: "
                    f"{tw_raw}×{th_raw} → {tw}×{th}",
                    color="#FFA726",
                )
                # Reflect the snap in the spinners so the user sees what's used.
                self.spin_target_w.blockSignals(True)
                self.spin_target_h.blockSignals(True)
                self.spin_target_w.setValue(tw)
                self.spin_target_h.setValue(th)
                self.spin_target_w.blockSignals(False)
                self.spin_target_h.blockSignals(False)

            # Compare to scale recommendation, warn if very different.
            try:
                from retina_scales import _RESOLUTION_FOR_SCALE, _resolve_scale
                rec_h, rec_w = _RESOLUTION_FOR_SCALE[_resolve_scale(scale)]
                if abs(tw - rec_w) > rec_w or abs(th - rec_h) > rec_h:
                    self._log(
                        f"Note: scale={scale} is tuned for {rec_w}×{rec_h}, "
                        f"using {tw}×{th} — feeder will resize but foveal "
                        f"coverage may be reduced.",
                        color="#FFA726",
                    )
            except Exception:
                pass

            feeder_cfg['input_resolution'] = (th, tw)

            self.retina = Retina(params=params, neuron_params=neuron_params, verbose=False)
            self.retina.build()
            self.retina.connect()
            self.feeder = self.retina.create_input_feeder(feeder_cfg)

            # Spike-recorder per ganglion pop (downstream output)
            for name, pop in self.retina.get_output_populations().items():
                sr = nest.Create('spike_recorder')
                nest.Connect(pop, sr)
                self.output_spike_recorders[name] = sr

            # Spike-recorder per bipolar pop (diagnostic — to verify the
            # bipolar→ganglion intra-retina connection is actually firing).
            # If bipolars fire but ganglia don't, the internal wiring
            # is the bug. If both fire, the issue is downstream.
            for bip_name in BIPOLAR_POPS:
                if bip_name not in self.retina.populations:
                    continue
                bip_pop = self.retina.populations[bip_name]
                if bip_pop is None or len(bip_pop) == 0:
                    continue
                sr = nest.Create('spike_recorder')
                nest.Connect(bip_pop, sr)
                self.bipolar_spike_recorders[bip_name] = sr

            # Virtual graph for live-viz integration
            self.virtual_graph = VirtualRetinaGraph(self.VIRTUAL_GRAPH_ID, self.retina)
            if self.virtual_graph not in self.graph_list:
                self.graph_list.append(self.virtual_graph)

            total_n = sum(self.retina.get_counts().values())
            self.lbl_neuron_count.setText(f"Neurons: {total_n:,}")
            self.lbl_state.setText("State: built")
            self.btn_build.setEnabled(False)
            self.btn_destroy.setEnabled(True)
            self._log(f"✓ Retina ready: {total_n:,} neurons, "
                      f"{len(self.retina.get_output_populations())} ganglion pops", color="#81C784")

            # If a video is already loaded, reload it at the new target so
            # subsequent reads decode at the resolution the retina expects.
            if self.video_reader is not None:
                path = self.edt_video.text().strip()
                if path:
                    self._try_open_video(path)

            self._refresh_target_selectors()
            self._apply_default_targets()
            self.retinaBuilt.emit()
            self.requestVizRefresh.emit()
        except Exception as e:
            self._log(f"✗ Build failed: {e}", color="#EF5350")
            traceback.print_exc()
            self.retina = None
            self.feeder = None

    @staticmethod
    def _snap_to_power_of_2(value: int) -> int:
        """Snap an integer to the nearest power of 2 in [32, 16384].
        Matches retina_main.VALID_INPUT_RESOLUTIONS."""
        valid = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
        if value <= valid[0]:
            return valid[0]
        if value >= valid[-1]:
            return valid[-1]
        # Find the closest by absolute distance.
        return min(valid, key=lambda v: abs(v - value))

    def _on_destroy_clicked(self):
        """Baut die Retina ab. Achtung: nest.ResetKernel() vernichtet sie mit."""
        self._on_pause_clicked()
        if self.virtual_graph is not None and self.virtual_graph in self.graph_list:
            self.graph_list.remove(self.virtual_graph)
        self.virtual_graph = None
        self.retina = None
        self.feeder = None
        self.output_spike_recorders.clear()
        self.bipolar_spike_recorders.clear()
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
        self._log("Retina destroyed (Python-side). "
                  "NEST populations remain alive until ResetKernel.", color="#FFA726")
        self.retinaDestroyed.emit()
        self.requestVizRefresh.emit()

    # --------------------------------------------------------------------
    #  CONNECTIONS
    # --------------------------------------------------------------------

    def _on_target_row_connect(self, row: '_TargetRow'):
        """Handler for the Connect button on a single target row."""
        if self.retina is None:
            QMessageBox.warning(self, "No Retina", "Press 'Build Retina' first.")
            return

        gid = row.get_graph_id()
        nid = row.get_node_id()
        pop_idx = row.get_pop_idx()
        weight = row.get_weight()
        delay = row.get_delay()
        pop_name = row.pop_name

        if gid is None or nid is None:
            self._log(f"[{pop_name}] no valid target", color="#EF5350")
            return

        target_graph = next((g for g in self.graph_list if g.graph_id == gid), None)
        if target_graph is None:
            self._log(f"[{pop_name}] graph {gid} not found", color="#EF5350")
            return
        target_node = next((n for n in target_graph.node_list if n.id == nid), None)
        if target_node is None:
            self._log(f"[{pop_name}] node {nid} not found", color="#EF5350")
            return

        src_pop = self.retina.populations.get(pop_name, None)
        if src_pop is None or len(src_pop) == 0:
            self._log(f"[{pop_name}] retina pop is empty", color="#EF5350")
            return

        # Validate the retina NodeCollection — invalidated by ResetKernel
        try:
            nest.GetStatus(src_pop[:1])
        except Exception:
            self._log(f"⚠ Retina NodeCollection invalidated (ResetKernel?). "
                      "Clear state and rebuild.", color="#EF5350")
            self.on_nest_kernel_reset()
            return

        # Collect target pops
        if pop_idx is None:
            target_pops = [p for p in getattr(target_node, 'population', []) if p is not None]
        else:
            if pop_idx < len(target_node.population) and target_node.population[pop_idx] is not None:
                target_pops = [target_node.population[pop_idx]]
            else:
                self._log(f"[{pop_name}] pop {pop_idx} in target node is empty", color="#EF5350")
                return

        try:
            rule = row.get_rule()
            syn_model = row.get_synapse_model()

            for tp in target_pops:
                # Build conn_spec based on the rule chosen in the row's
                # advanced panel. Rule-specific parameters come from the
                # spinboxes that get_rule() / get_indegree() / get_probability()
                # expose.
                if rule == 'fixed_indegree':
                    n_src = len(src_pop)
                    indeg = row.get_indegree()
                    eff_indeg = min(indeg, max(1, n_src - 1))
                    conn_spec = {
                        'rule': 'fixed_indegree',
                        'indegree': eff_indeg,
                        'allow_autapses': False,
                        'allow_multapses': False,
                    }
                    if eff_indeg < indeg:
                        self._log(
                            f"  [{pop_name}] indegree capped {indeg} → {eff_indeg} "
                            f"(only {n_src} source neurons available)",
                            color="#FFA726",
                        )
                elif rule == 'fixed_outdegree':
                    n_tgt = len(tp)
                    outdeg = row.get_indegree()  # same spinbox, semantically outdegree here
                    eff_outdeg = min(outdeg, max(1, n_tgt - 1))
                    conn_spec = {
                        'rule': 'fixed_outdegree',
                        'outdegree': eff_outdeg,
                        'allow_autapses': False,
                        'allow_multapses': False,
                    }
                elif rule == 'pairwise_bernoulli':
                    conn_spec = {
                        'rule': 'pairwise_bernoulli',
                        'p': row.get_probability(),
                        'allow_autapses': False,
                        'allow_multapses': True,
                    }
                elif rule == 'one_to_one':
                    if len(src_pop) != len(tp):
                        self._log(
                            f"  [{pop_name}] one_to_one needs equal pop sizes "
                            f"(src={len(src_pop)}, tgt={len(tp)}) — skipping",
                            color="#EF5350",
                        )
                        continue
                    conn_spec = {'rule': 'one_to_one'}
                else:  # all_to_all
                    conn_spec = {
                        'rule': 'all_to_all',
                        'allow_autapses': False,
                    }

                nest.Connect(
                    src_pop, tp,
                    conn_spec=conn_spec,
                    syn_spec={'weight': weight, 'delay': delay,
                              'synapse_model': syn_model},
                )
                self.registered_connections.append((pop_name, src_pop, tp))
            row.set_connected(True)
            self.lbl_connections.setText(f"Connections: {len(self.registered_connections)}")
            self._log(
                f"✓ {pop_name} → G{gid}.N{nid}.{pop_idx}  "
                f"[{rule}, w={weight}, d={delay}, syn={syn_model}]",
                color="#81C784",
            )
        except Exception as e:
            self._log(f"✗ Connect failed [{pop_name}]: {e}", color="#EF5350")
            traceback.print_exc()

    def _on_connect_all(self):
        """Connect every target row that has a valid target and isn't already wired."""
        for pop_name, row in self._iter_all_rows():
            if row.is_connected():
                continue
            if row.get_graph_id() is not None and row.get_node_id() is not None:
                self._on_target_row_connect(row)

    def _on_disconnect_all(self):
        """Rebuild-only: NEST has no selective disconnect. We just clear the
        Python-side list and reset the UI state. The next ResetKernel will
        rebuild cleanly."""
        self.registered_connections.clear()
        for pop_name, row in self._iter_all_rows():
            row.set_connected(False)
        self.lbl_connections.setText("Connections: 0 (pending ResetKernel)")
        self._log("Connections cleared from Python state. "
                  "For an actual reset: trigger NEST kernel reset via Main.", color="#FFA726")

    # --------------------------------------------------------------------
    #  VIDEO
    # --------------------------------------------------------------------

    def _on_browse_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "",
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
            self._update_bio_per_frame_label()
            self._log(f"Video loaded: {info}", color="#4FC3F7")
        except Exception as e:
            self._log(f"✗ Video load failed: {e}", color="#EF5350")

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

    def _get_effective_fps(self) -> Optional[float]:
        """Return the playback FPS we'd actually use, or None if no video.
        Override > 0 takes precedence over the video's native FPS.
        Used by both the Play handler and the derived bio-time label.
        """
        if self.video_reader is None:
            return None
        override = float(self.spin_fps_override.value())
        if override > 0:
            return override
        return float(self.video_reader.fps)

    def _update_bio_per_frame_label(self):
        """Refresh the '→ NEST: XX.XX ms/frame' label. Called whenever
        FPS-affecting state changes (video load, FPS override edit).
        Shows the actual rounded value Simulate() will see, since NEST
        only accepts multiples of resolution."""
        fps = self._get_effective_fps()
        if fps is None or fps <= 0:
            self.lbl_bio_per_frame.setText("(no video)")
            self.lbl_bio_per_frame.setStyleSheet(
                "color: #B0BEC5; font-family: monospace;"
            )
            return
        raw = 1000.0 / fps
        try:
            res = float(nest.GetKernelStatus('resolution'))
        except Exception:
            res = 0.1
        if res > 0:
            snapped = max(1, int(round(raw / res))) * res
        else:
            snapped = raw
        self.lbl_bio_per_frame.setText(f"{snapped:.2f} ms/frame")
        self.lbl_bio_per_frame.setStyleSheet(
            "color: #81C784; font-family: monospace; font-weight: bold;"
        )

    def _on_play_clicked(self):
        if self.retina is None:
            QMessageBox.warning(self, "No Retina", "Press 'Build Retina' first.")
            return
        if self.video_reader is None:
            QMessageBox.warning(self, "No Video",
                                "Load a video first — Eye-Tab only fires "
                                "while a video is being played back.")
            return

        # Refuse if the global sim_timer is running. NEST cannot have two
        # Simulate() in flight at once → 'Prepare called twice'.
        if self.main_window is not None:
            global_st = getattr(self.main_window, 'sim_timer', None)
            if global_st is not None and global_st.isActive():
                QMessageBox.warning(
                    self, "Global Sim is running",
                    "The top-bar continuous simulation is currently running.\n\n"
                    "Pause or stop the global sim first, then press Play here."
                )
                return

        fps = self._get_effective_fps()
        if fps is None or fps <= 0:
            QMessageBox.warning(self, "Bad FPS",
                                f"Cannot derive frame rate (got {fps}).")
            return

        # ── Pre-Play setup ────────────────────────────────────────────
        # Trigger the INTERACTIVE-tab LOAD button equivalent so the new
        # retina population gets registered in the 3D scatter view —
        # load_scene() rebuilds gid_to_idx from the current graph_list.
        # We deliberately do NOT touch the top-bar step spinner: that is
        # the user's "neural microscope" master clock and it dictates
        # every nest.Simulate() chunk in this Eye-Tab. Read-only here.
        if self.main_window is not None:
            try:
                sv = getattr(self.main_window, 'simulation_view', None)
                if sv is not None:
                    sv.load_scene()
            except Exception as e:
                self._log(f"[warn] simulation_view.load_scene: {e}",
                          color="#FFA726")

        # ── Frame duration ────────────────────────────────────────────
        # 1000/fps in bio-time. Pure float, no resolution snapping —
        # the accumulator compares fractionally.
        self._frame_bio_ms = 1000.0 / fps

        # Sanity: log the master step so user can see what they signed
        # up for. Read fresh; never cached, never overwritten.
        try:
            res = float(nest.GetKernelStatus('resolution'))
        except Exception:
            res = 0.1
        if (self.main_window is not None
                and hasattr(self.main_window, 'global_step_spin')):
            master_step = float(self.main_window.global_step_spin.value())
        else:
            master_step = res

        # Reset accumulator state for a fresh Play session.
        self._frame_bio_accum = 0.0
        self._current_frame_fed = False
        self._last_preview_idx = -1
        self._last_bgr = None

        # Tick frequency: 100Hz real-time, decoupled from video FPS.
        # Frame advances are driven by the accumulator, not by wall clock.
        tick_interval_ms = 10

        self._playing = True
        self.play_timer.start(tick_interval_ms)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self._log(
            f"▶ Play @ {fps:.2f} fps  (frame={self._frame_bio_ms:.2f}ms bio, "
            f"top-bar step={master_step:.2f}ms, res={res}ms)",
            color="#81C784",
        )

        # Spike recorders + render timer for the live 3D view.
        if self.main_window is not None:
            try:
                if hasattr(self.main_window, '_ensure_spike_recorders'):
                    self.main_window._ensure_spike_recorders()
                sv = getattr(self.main_window, 'simulation_view', None)
                if sv is not None and sv.scene_loaded:
                    sv.start_rendering()
            except Exception as e:
                self._log(f"[warn] recorder/render setup: {e}", color="#FFA726")

        # Switch to INTERACTIVE so the user actually sees activity.
        if self.main_window is not None:
            try:
                if hasattr(self.main_window, '_switch_main_view'):
                    self.main_window._switch_main_view(1)
            except Exception as e:
                self._log(f"Auto-switch warning: {e}", color="#FFA726")

    def _on_pause_clicked(self):
        self._playing = False
        self.play_timer.stop()
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)

    def _on_rewind_clicked(self):
        if self.video_reader is not None:
            self.video_reader.seek(0)
            self._show_preview_frame()
            self.progress.setValue(0)
            self.lbl_frame.setText("Frame: 0")

    def _on_play_tick(self):
        """One tick:
          - if accumulator says "frame done" → load + feed next frame
          - simulate exactly one resolution-step (chunk), accumulate
          - pump visualization
          - update UI
        Frame advance is purely accumulator-driven: as soon as
        accumulator >= frame_bio_ms, advance regardless of how much
        was overshot.
        """
        if not self._playing or self.retina is None or self.video_reader is None:
            return

        # Bail if global sim_timer became active mid-play.
        if self.main_window is not None:
            st = getattr(self.main_window, 'sim_timer', None)
            if st is not None and st.isActive():
                self._log(
                    "✗ Global sim_timer started — Eye-Tab Play auto-paused.",
                    color="#EF5350",
                )
                self._on_pause_clicked()
                return

        # ── Frame advance ─────────────────────────────────────────────
        if not self._current_frame_fed:
            triple = self.video_reader.read_next()
            if triple is None:
                if self.chk_loop.isChecked():
                    self.video_reader.seek(0)
                    triple = self.video_reader.read_next()
                if triple is None:
                    self._on_pause_clicked()
                    return
            lms, intensity, bgr = triple
            try:
                self.feeder.feed(lms, intensity)
            except Exception as e:
                self._log(f"feed() error: {e}", color="#EF5350")
                import traceback; traceback.print_exc()
                self._on_pause_clicked()
                return
            self._current_frame_fed = True
            self._frame_bio_accum = 0.0
            self._last_bgr = bgr

        # ── Simulate one chunk = step from top-bar (snapped to res) ───
        # Read live each tick so the user can adjust mid-playback. NEST
        # requires the sim time to be a multiple of resolution.
        try:
            res = float(nest.GetKernelStatus('resolution'))
        except Exception:
            res = 0.1
        if (self.main_window is not None
                and hasattr(self.main_window, 'global_step_spin')):
            requested = float(self.main_window.global_step_spin.value())
        else:
            requested = res
        n_steps = max(1, int(round(requested / res)))
        chunk = n_steps * res

        try:
            nest.Simulate(chunk)
        except Exception as e:
            self._log(f"Simulate() error: {e}", color="#EF5350")
            self._on_pause_clicked()
            return
        self._frame_bio_accum += chunk

        # ── Pump visualization ────────────────────────────────────────
        if self.main_window is not None:
            try:
                if hasattr(self.main_window, '_distribute_simulation_data'):
                    self.main_window._distribute_simulation_data()
                if hasattr(self.main_window, 'current_nest_time'):
                    self.main_window.current_nest_time = (
                        nest.GetKernelStatus('biological_time')
                    )
                    if hasattr(self.main_window, 'update_global_time_display'):
                        self.main_window.update_global_time_display(
                            self.main_window.current_nest_time
                        )
            except Exception:
                pass

        # ── Frame done check (pure accumulator) ───────────────────────
        if self._frame_bio_accum >= self._frame_bio_ms:
            self._current_frame_fed = False  # Next tick will feed next frame

        # ── UI updates ────────────────────────────────────────────────
        try:
            self.lbl_nest_t.setText(
                f"NEST t: {nest.GetKernelStatus('biological_time'):.1f} ms"
            )
        except Exception:
            pass

        self.lbl_frame.setText(f"Frame: {self.video_reader.current_idx}")
        if self.video_reader.n_frames > 0:
            pct = int(100 * self.video_reader.current_idx
                      / max(1, self.video_reader.n_frames))
            self.progress.setValue(pct)

        # Preview pixmap: only when we've moved to a new frame (not per
        # chunk; the BGR doesn't change within a frame anyway).
        if (self._last_preview_idx != self.video_reader.current_idx
                and self._last_bgr is not None):
            self._last_preview_idx = self.video_reader.current_idx
            try:
                rgb = cv2.cvtColor(self._last_bgr, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(
                    self.preview_label.width(), self.preview_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.preview_label.setPixmap(pix)
            except Exception:
                pass

        self.requestVizRefresh.emit()

    def _on_diag_clicked(self):
        """User-triggered status snapshot. Shows cumulative spike counts
        for all bipolar and ganglion pops since the recorders were
        created (i.e. since Build). Pause-safe — works whether playing
        or stopped, as long as the retina is built."""
        if self.retina is None:
            self._log("Status: no retina built", color="#FFA726")
            return
        try:
            t_now = float(nest.GetKernelStatus('biological_time'))
        except Exception:
            t_now = -1.0

        self._log(f"━━ Status snapshot @ NEST t={t_now:.1f}ms ━━",
                  color="#80DEEA")

        # Bipolar pops (intra-retina, signals the bipolar→ganglion path)
        if self.bipolar_spike_recorders:
            total_bip = 0
            for name, rec in self.bipolar_spike_recorders.items():
                try:
                    n = int(nest.GetStatus(rec, 'n_events')[0])
                except Exception:
                    n = 0
                total_bip += n
                short = name.replace('_bipolar', '_bip')
                self._log(f"  bip · {short}: {n} spikes total",
                          color="#90CAF9")
            self._log(f"  bip · TOTAL: {total_bip} spikes",
                      color="#42A5F5")

        # Ganglion pops (downstream output, what LGN would receive)
        if self.output_spike_recorders:
            total_gang = 0
            for name, rec in self.output_spike_recorders.items():
                try:
                    n = int(nest.GetStatus(rec, 'n_events')[0])
                except Exception:
                    n = 0
                total_gang += n
                short = name.replace('_ganglion', '_g')
                self._log(f"  gang· {short}: {n} spikes total",
                          color="#A5D6A7")
            self._log(f"  gang· TOTAL: {total_gang} spikes",
                      color="#66BB6A")

            # Critical diagnosis: if bipolars are firing but ganglia
            # aren't (more than a tiny fraction), the bipolar→ganglion
            # internal connection is the bug.
            if self.bipolar_spike_recorders:
                ratio = (total_gang / max(1, total_bip))
                if total_bip > 100 and total_gang < total_bip * 0.001:
                    self._log(
                        f"  ⚠ DIAGNOSIS: bipolars fire ({total_bip}) but "
                        f"ganglia almost silent ({total_gang}). "
                        f"Bipolar→Ganglion connection is the problem.",
                        color="#EF5350",
                    )
                else:
                    self._log(
                        f"  · gang/bip ratio = {ratio:.4f}",
                        color="#90A4AE",
                    )
        else:
            self._log("  no recorders found", color="#FFA726")

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
        self.bipolar_spike_recorders.clear()
        self.registered_connections.clear()
        self.lbl_state.setText("State: idle (kernel reset)")
        self.lbl_neuron_count.setText("Neurons: —")
        self.lbl_connections.setText("Connections: 0")
        for pop_name, row in self._iter_all_rows():
            row.set_connected(False)
        self.btn_build.setEnabled(True)
        self.btn_destroy.setEnabled(False)
        self._log("⚠ NEST-Kernel wurde von aussen resettet. "
                  "Retina state cleared. Press Build again to create a new retina.",
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
        # Two-row layout: main controls always visible, advanced panel
        # collapsible. The advanced panel exposes the connection rule and
        # rule-specific parameters (indegree / p), plus the synapse model.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(2)

        # ── Top row: target combos + weight/delay + buttons ──
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)

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

        # ▸ / ▾ toggle. Stays small so it doesn't dominate the row.
        self.btn_advanced = QPushButton("▸")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setFixedWidth(22)
        self.btn_advanced.setToolTip("Show / hide advanced connection options")
        self.btn_advanced.setStyleSheet(
            "QPushButton { background-color: #37474F; color: #B0BEC5; border: none; "
            "border-radius: 2px; } "
            "QPushButton:checked { background-color: #455A64; color: #ECEFF1; }"
        )
        self.btn_advanced.toggled.connect(self._on_advanced_toggled)

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

        top.addWidget(QLabel("→"))
        top.addWidget(self.cmb_graph, 1)
        top.addWidget(self.cmb_node, 1)
        top.addWidget(self.cmb_pop, 1)
        top.addWidget(self.sp_w)
        top.addWidget(self.sp_d)
        top.addWidget(self.btn_advanced)
        top.addWidget(self.btn_connect)
        top.addWidget(self.btn_remove)
        outer.addLayout(top)

        # ── Advanced panel (collapsed by default) ──
        self.advanced_panel = QFrame()
        self.advanced_panel.setStyleSheet(
            "QFrame { background-color: #263238; border-left: 3px solid #455A64; "
            "border-radius: 2px; }"
        )
        self.advanced_panel.setVisible(False)
        adv_h = QHBoxLayout(self.advanced_panel)
        adv_h.setContentsMargins(8, 4, 6, 4)
        adv_h.setSpacing(8)

        # Connection rule
        self.cmb_rule = QComboBox()
        self.cmb_rule.setMinimumWidth(140)
        for r in CONNECTION_RULES:
            self.cmb_rule.addItem(r, r)
        self.cmb_rule.setCurrentText('fixed_indegree')
        self.cmb_rule.currentIndexChanged.connect(self._on_rule_changed)

        # Indegree (used by fixed_indegree / fixed_outdegree)
        self.sp_indegree = QSpinBox()
        self.sp_indegree.setRange(1, 10000)
        self.sp_indegree.setValue(15)
        self.sp_indegree.setFixedWidth(70)
        self.sp_indegree.setPrefix("k=")
        self.sp_indegree.setToolTip("Inputs per target neuron (fixed_indegree) "
                                    "or outputs per source neuron (fixed_outdegree)")

        # Probability (used by pairwise_bernoulli)
        self.sp_p = QDoubleSpinBox()
        self.sp_p.setRange(0.0, 1.0)
        self.sp_p.setValue(0.1)
        self.sp_p.setSingleStep(0.05)
        self.sp_p.setDecimals(3)
        self.sp_p.setFixedWidth(70)
        self.sp_p.setPrefix("p=")
        self.sp_p.setToolTip("Connection probability per (source, target) pair")

        # Synapse model
        self.cmb_synapse = QComboBox()
        self.cmb_synapse.setMinimumWidth(140)
        # Common ones are enough here — full synapse-model editor lives in
        # the main connection tool; this is a Quick wiring helper.
        for s in ('static_synapse', 'stdp_synapse', 'stdp_pl_synapse',
                  'tsodyks2_synapse', 'bernoulli_synapse'):
            self.cmb_synapse.addItem(s, s)

        adv_h.addWidget(QLabel("rule:"))
        adv_h.addWidget(self.cmb_rule)
        adv_h.addWidget(self.sp_indegree)
        adv_h.addWidget(self.sp_p)
        adv_h.addWidget(QLabel("syn:"))
        adv_h.addWidget(self.cmb_synapse)
        adv_h.addStretch(1)
        outer.addWidget(self.advanced_panel)

        # Initialize visibility for rule-specific spinners
        self._on_rule_changed()

    def _on_advanced_toggled(self, checked: bool):
        self.btn_advanced.setText("▾" if checked else "▸")
        self.advanced_panel.setVisible(checked)

    def _on_rule_changed(self):
        rule = self.cmb_rule.currentData() or 'all_to_all'
        # indegree spinner only makes sense for fixed_indegree / fixed_outdegree
        self.sp_indegree.setVisible(rule in ('fixed_indegree', 'fixed_outdegree'))
        # probability spinner only matters for pairwise_bernoulli
        self.sp_p.setVisible(rule == 'pairwise_bernoulli')

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

    def get_rule(self) -> str:
        return self.cmb_rule.currentData() or 'all_to_all'

    def get_indegree(self) -> int:
        return int(self.sp_indegree.value())

    def get_probability(self) -> float:
        return float(self.sp_p.value())

    def get_synapse_model(self) -> str:
        return self.cmb_synapse.currentData() or 'static_synapse'

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

    def apply_server_defaults(self, indegree: int, weight: float, delay: float):
        """Set rule + indegree + weight + delay to the server-canonical values
        for this row's pop. Called from _apply_server_match()."""
        self.sp_w.setValue(float(weight))
        self.sp_d.setValue(float(delay))
        idx = self.cmb_rule.findData('fixed_indegree')
        if idx >= 0:
            self.cmb_rule.setCurrentIndex(idx)
        self.sp_indegree.setValue(int(indegree))
        # Reflect rule-specific visibility
        self._on_rule_changed()

    # -- Dropdown-Befuellung --
    def populate_graphs(self, graph_list):
        """Populate the graph dropdown with all non-virtual graphs."""
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

    def set_target(self, gid, nid, pop_idx=None) -> bool:
        """Programmatically set the three target combos. Cascade fires
        through currentIndexChanged: cmb_graph._on_graph_changed populates
        cmb_node, cmb_node._on_node_changed populates cmb_pop. When the
        index we want is already current, setCurrentIndex doesn't emit a
        signal, so the cascade has to be triggered manually in that case.

        Returns True if graph + node both resolved.
        """
        # Graph
        graph_ok = False
        for i in range(self.cmb_graph.count()):
            if self.cmb_graph.itemData(i) == gid:
                if self.cmb_graph.currentIndex() == i:
                    self._on_graph_changed()
                else:
                    self.cmb_graph.setCurrentIndex(i)
                graph_ok = True
                break
        if not graph_ok:
            return False

        # Node — cmb_node was just populated by the graph cascade
        node_ok = False
        for i in range(self.cmb_node.count()):
            if self.cmb_node.itemData(i) == nid:
                if self.cmb_node.currentIndex() == i:
                    self._on_node_changed()
                else:
                    self.cmb_node.setCurrentIndex(i)
                node_ok = True
                break
        if not node_ok:
            return False

        # Pop (optional)
        if pop_idx is not None:
            for i in range(self.cmb_pop.count()):
                if self.cmb_pop.itemData(i) == pop_idx:
                    self.cmb_pop.setCurrentIndex(i)
                    break

        return True

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
