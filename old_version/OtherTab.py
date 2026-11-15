"""
OtherTab.py — placeholder/sandbox for user widgets.

This file is loaded by Main.py for the "Other" navigation button. Edit
this file freely to build your own widget — it lives outside the main
codebase so refactors of Main.py / WidgetLib.py won't touch it.

The expected interface is a class `OtherTabWidget` with `__init__(self,
graph_list=None, parent=None)`. That's it. Whatever you put inside is
shown when the user clicks "Other".
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout,
                              QPushButton, QFrame, QTextEdit)
from PyQt6.QtCore import Qt


class OtherTabWidget(QWidget):
    """Placeholder. Replace the body with whatever you need."""

    def __init__(self, graph_list=None, parent=None):
        super().__init__(parent)
        self.graph_list = graph_list if graph_list is not None else []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(15)

        title = QLabel("Other — Sandbox")
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #FFD700;"
        )
        outer.addWidget(title)

        info = QLabel(
            "This tab loads from <code>OtherTab.py</code> in the project root.\n\n"
            "Edit that file freely — it's outside Main.py / WidgetLib.py and "
            "won't be touched by future refactors of the main code.\n\n"
            "The class <code>OtherTabWidget</code> with signature\n"
            "<code>__init__(self, graph_list=None, parent=None)</code> is "
            "the only contract."
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setStyleSheet("color: #ccc; font-size: 13px; line-height: 1.4;")
        outer.addWidget(info)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #444;")
        outer.addWidget(sep)

        # Tiny demo: list current graphs so user sees the data is wired up
        graphs_label = QLabel(
            f"<b>Currently loaded:</b> {len(self.graph_list)} graph(s)"
        )
        graphs_label.setStyleSheet("color: #aaa;")
        outer.addWidget(graphs_label)

        if self.graph_list:
            details = QTextEdit()
            details.setReadOnly(True)
            details.setStyleSheet(
                "background: #1e1e1e; color: #ddd; border: 1px solid #444; "
                "font-family: Consolas, monospace; font-size: 12px;"
            )
            lines = []
            for g in self.graph_list:
                name = getattr(g, 'graph_name', f'Graph_{getattr(g, "graph_id", "?")}')
                gid = getattr(g, 'graph_id', '?')
                n_nodes = len(getattr(g, 'node_list', []))
                lines.append(f"  - {name} (id={gid}): {n_nodes} node(s)")
            details.setPlainText("\n".join(lines))
            details.setMaximumHeight(180)
            outer.addWidget(details)

        outer.addStretch()

        # Hint footer
        hint = QLabel(
            "Tip: anything you build here has access to <code>self.graph_list</code> "
            "(live references to all current graphs). Use it for quick experiments, "
            "ad-hoc analysis tools, custom plots, etc."
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.RichText)
        hint.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        outer.addWidget(hint)
