# CustomExtension.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt

class CustomTabWidget(QWidget):
    def __init__(self, graph_list, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list
        self.init_ui()

    def init_ui(self):
        # Haupt-Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(20)

        # --- Header Bereich ---
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #2b2b2b; border-bottom: 2px solid #444;")
        header_layout = QHBoxLayout(header_frame)
        
        lbl_title = QLabel("CUSTOM EXTENSION")
        lbl_title.setStyleSheet("color: #FFD700; font-size: 18px; font-weight: bold; letter-spacing: 2px;")
        
        header_layout.addWidget(lbl_title)
        header_layout.addStretch()
        
        self.layout.addWidget(header_frame)

        # --- Content Area (Blank Canvas) ---
        # Wir nutzen eine ScrollArea, falls dein Inhalt größer wird
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background-color: #1e1e1e; border-radius: 6px;")
        
        self.content_container = QWidget()
        self.content_container.setStyleSheet("background-color: #1e1e1e;")
        self.content_layout = QVBoxLayout(self.content_container)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # ... HIER DEINE WIDGETS EINFÜGEN ...
        
        # Beispiel: Platzhalter Text
        info_lbl = QLabel("This is your blank canvas.\nImport your own scripts here.")
        info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_lbl.setStyleSheet("color: #666; font-style: italic; font-size: 14px; margin-top: 50px;")
        self.content_layout.addWidget(info_lbl)
        
        # Beispiel: Debug Button
        btn_debug = QPushButton("Print Graph Data to Console")
        btn_debug.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_debug.setFixedWidth(250)
        btn_debug.setStyleSheet("""
            QPushButton {
                background-color: #333; color: white; border: 1px solid #555; 
                padding: 8px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #444; border-color: #FFD700; }
        """)
        btn_debug.clicked.connect(self.debug_print_graphs)
        self.content_layout.addWidget(btn_debug, alignment=Qt.AlignmentFlag.AlignCenter)

        scroll.setWidget(self.content_container)
        self.layout.addWidget(scroll)

    def debug_print_graphs(self):
        """Beispiel: Zugriff auf die globale Graph-Liste."""
        print(f"\n[CustomTab] Accessing Data...")
        print(f"Total Graphs: {len(self.graph_list)}")
        for g in self.graph_list:
            print(f" - ID: {g.graph_id}, Name: {getattr(g, 'graph_name', 'N/A')}, Nodes: {len(g.node_list)}")

    def on_tab_active(self):
        """Wird aufgerufen, wenn der Tab sichtbar wird (für Refresh-Logik)."""
        print("[CustomTab] Activated / Refreshed.")
