#!/usr/bin/env python3
"""
quantum_glasses.py
==================
Quantum Gate Visualizer — Python Edition
-----------------------------------------
An interactive single-qubit quantum gate visualizer.
Apply gates from a palette, watch the qubit state sweep across
a real-time 3-D Bloch sphere, inspect amplitudes and probabilities,
and build visual quantum circuits — all in pure Python.

Author : Priyam Dixit
Stack  : Python · NumPy · Matplotlib · Tkinter
Gates  : X, Y, Z, H, S, S†, T, T†, Rx(θ), Ry(θ), Rz(θ), P(φ)

Run    : python quantum_glasses.py
Deps   : pip install numpy matplotlib
"""

from __future__ import annotations

import sys
import math
import random
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
from mpl_toolkits.mplot3d.art3d import Line3D

# ══════════════════════════════════════════════════════════════════════════════
# Design tokens
# ══════════════════════════════════════════════════════════════════════════════
BG_BASE    = "#050c1a"
BG_SURFACE = "#0a1528"
BG_PANEL   = "#08111f"
TEXT_PRI   = "#e2e8f0"
TEXT_SEC   = "#8a9bbf"
TEXT_MUTED = "#4b6180"

COL_PAULI    = "#f472b6"
COL_CLIFFORD = "#38bdf8"
COL_ROTATION = "#fbbf24"
COL_PHASE    = "#34d399"
COL_SPECIAL  = "#a78bfa"

CATEGORY_COLORS = {
    "pauli":    COL_PAULI,
    "clifford": COL_CLIFFORD,
    "rotation": COL_ROTATION,
    "phase":    COL_PHASE,
}

FONT_TITLE = ("Helvetica", 15, "bold")
FONT_LABEL = ("Helvetica", 10)
FONT_MONO  = ("Courier New", 12, "bold")
FONT_SMALL = ("Helvetica", 9)

INV_SQRT2 = 1.0 / math.sqrt(2)


# ══════════════════════════════════════════════════════════════════════════════
# Gate Library
# ══════════════════════════════════════════════════════════════════════════════
class GateLibrary:
    """
    Provides gate matrices (as NumPy arrays) and educational metadata
    for all 12 supported single-qubit gates.
    """

    # Static (non-parameterised) gate matrices
    STATIC: dict[str, np.ndarray] = {
        "X":     np.array([[0, 1],  [1,  0]],             dtype=complex),
        "Y":     np.array([[0, -1j],[1j, 0]],             dtype=complex),
        "Z":     np.array([[1, 0],  [0, -1]],             dtype=complex),
        "H":     np.array([[1, 1],  [1, -1]], dtype=complex) * INV_SQRT2,
        "S":     np.array([[1, 0],  [0, 1j]],             dtype=complex),
        "S_DAG": np.array([[1, 0],  [0, -1j]],            dtype=complex),
        "T":     np.array([[1, 0],  [0, np.exp(1j * np.pi / 4)]],  dtype=complex),
        "T_DAG": np.array([[1, 0],  [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
    }

    # Educational metadata for the info panel
    INFO: dict[str, dict] = {
        "X": {
            "name": "Pauli-X  (NOT Gate)", "category": "pauli", "label": "X",
            "desc": (
                "The quantum NOT gate. Flips |0⟩ to |1⟩ and vice versa. "
                "Equivalent to a 180° rotation around the X-axis of the Bloch sphere."
            ),
            "matrix": "[ [0, 1],\n  [1, 0] ]",
            "effects": "|0⟩ → |1⟩\n|1⟩ → |0⟩",
            "analogy": "Like flipping a coin — heads instantly becomes tails.",
        },
        "Y": {
            "name": "Pauli-Y", "category": "pauli", "label": "Y",
            "desc": (
                "Combines a bit-flip and a phase-flip simultaneously. "
                "180° rotation around the Y-axis."
            ),
            "matrix": "[ [ 0, -i],\n  [+i,  0] ]",
            "effects": "|0⟩ → i|1⟩\n|1⟩ → -i|0⟩",
            "analogy": "A spin that flips both state and phase at once.",
        },
        "Z": {
            "name": "Pauli-Z  (Phase Flip)", "category": "pauli", "label": "Z",
            "desc": (
                "Leaves |0⟩ unchanged but adds a −1 phase to |1⟩. "
                "180° rotation around the Z-axis."
            ),
            "matrix": "[ [1,  0],\n  [0, -1] ]",
            "effects": "|0⟩ → |0⟩\n|1⟩ → −|1⟩",
            "analogy": "Like a mirror — one side unchanged, the other flips sign.",
        },
        "H": {
            "name": "Hadamard", "category": "clifford", "label": "H",
            "desc": (
                "Creates superposition! Puts a qubit into an equal mix of |0⟩ and |1⟩. "
                "The most important gate in quantum computing."
            ),
            "matrix": "(1/√2) · [ [1,  1],\n           [1, -1] ]",
            "effects": "|0⟩ → |+⟩ = (|0⟩+|1⟩)/√2\n|1⟩ → |−⟩ = (|0⟩−|1⟩)/√2",
            "analogy": "Spinning a coin — it is heads AND tails simultaneously.",
        },
        "S": {
            "name": "S Gate  (√Z)", "category": "clifford", "label": "S",
            "desc": "Applies a 90° (π/2) phase to |1⟩. Two S gates equal one Z gate.",
            "matrix": "[ [1, 0],\n  [0, i] ]",
            "effects": "|0⟩ → |0⟩\n|1⟩ → i|1⟩",
            "analogy": "A quarter-turn of the phase — rotating a clock by 90°.",
        },
        "S_DAG": {
            "name": "S† Dagger", "category": "clifford", "label": "S†",
            "desc": "Inverse of S. Applies a −90° (−π/2) phase to |1⟩.",
            "matrix": "[ [1,  0],\n  [0, -i] ]",
            "effects": "|0⟩ → |0⟩\n|1⟩ → -i|1⟩",
            "analogy": "Rotate the clock backwards by 90°.",
        },
        "T": {
            "name": "T Gate  (π/8)", "category": "clifford", "label": "T",
            "desc": (
                "Applies a 45° (π/4) phase to |1⟩. "
                "Non-Clifford gate critical for universal quantum computation."
            ),
            "matrix": "[ [1,        0       ],\n  [0, e^(iπ/4) ] ]",
            "effects": "|0⟩ → |0⟩\n|1⟩ → e^(iπ/4)|1⟩",
            "analogy": "The secret ingredient that enables algorithms impossible without it.",
        },
        "T_DAG": {
            "name": "T† Dagger", "category": "clifford", "label": "T†",
            "desc": "Inverse of T. Applies a −45° (−π/4) phase to |1⟩.",
            "matrix": "[ [1,         0        ],\n  [0, e^(-iπ/4) ] ]",
            "effects": "|0⟩ → |0⟩\n|1⟩ → e^(-iπ/4)|1⟩",
            "analogy": "Reverses the T gate rotation by 45°.",
        },
        "RX": {
            "name": "Rx(θ)  X-Rotation", "category": "rotation", "label": "Rx",
            "desc": "Rotates the Bloch sphere by angle θ around the X-axis. At θ=180° equals X.",
            "matrix": "[ [cos(θ/2),   -i·sin(θ/2)],\n  [-i·sin(θ/2), cos(θ/2)  ] ]",
            "effects": "|0⟩ → cos(θ/2)|0⟩ - i·sin(θ/2)|1⟩\n|1⟩ → -i·sin(θ/2)|0⟩ + cos(θ/2)|1⟩",
            "analogy": "Tilting a globe around the X-axis by θ degrees.",
        },
        "RY": {
            "name": "Ry(θ)  Y-Rotation", "category": "rotation", "label": "Ry",
            "desc": "Rotates the Bloch sphere by θ around the Y-axis. Real-valued — no imaginary phase.",
            "matrix": "[ [cos(θ/2), -sin(θ/2)],\n  [sin(θ/2),  cos(θ/2)] ]",
            "effects": "|0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩\n|1⟩ → -sin(θ/2)|0⟩ + cos(θ/2)|1⟩",
            "analogy": "A pure real rotation — like tilting a globe north-south.",
        },
        "RZ": {
            "name": "Rz(θ)  Z-Rotation", "category": "rotation", "label": "Rz",
            "desc": "Rotates the Bloch sphere by θ around the Z-axis. Only affects relative phase.",
            "matrix": "[ [e^(-iθ/2), 0        ],\n  [0,          e^(iθ/2)] ]",
            "effects": "|0⟩ → e^(-iθ/2)|0⟩\n|1⟩ → e^(iθ/2)|1⟩",
            "analogy": "Advancing a clock hand — changes phase without changing probabilities.",
        },
        "P": {
            "name": "Phase Gate P(φ)", "category": "phase", "label": "P",
            "desc": (
                "Applies an arbitrary phase φ to |1⟩; leaves |0⟩ unchanged. "
                "Generalises Z (φ=180°), S (φ=90°), T (φ=45°)."
            ),
            "matrix": "[ [1, 0      ],\n  [0, e^(iφ) ] ]",
            "effects": "|0⟩ → |0⟩\n|1⟩ → e^(iφ)|1⟩",
            "analogy": "A universal phase knob — dial in any angle.",
        },
    }

    @classmethod
    def get_matrix(cls, gate_id: str, angle_deg: float = 0.0) -> np.ndarray:
        """Return the gate's 2×2 unitary matrix. Parameterised gates use angle_deg."""
        if gate_id in cls.STATIC:
            return cls.STATIC[gate_id]

        t = math.radians(angle_deg)
        if gate_id == "RX":
            return np.array([
                [math.cos(t / 2),              -1j * math.sin(t / 2)],
                [-1j * math.sin(t / 2),          math.cos(t / 2)    ],
            ], dtype=complex)
        if gate_id == "RY":
            return np.array([
                [math.cos(t / 2),  -math.sin(t / 2)],
                [math.sin(t / 2),   math.cos(t / 2)],
            ], dtype=complex)
        if gate_id == "RZ":
            return np.array([
                [np.exp(-1j * t / 2),  0],
                [0,                    np.exp(1j * t / 2)],
            ], dtype=complex)
        if gate_id == "P":
            return np.array([
                [1,  0],
                [0,  np.exp(1j * t)],
            ], dtype=complex)

        raise ValueError(f"Unknown gate: {gate_id}")

    @classmethod
    def get_info(cls, gate_id: str) -> Optional[dict]:
        """Return metadata dict for a gate, or None if unknown."""
        return cls.INFO.get(gate_id)


# ══════════════════════════════════════════════════════════════════════════════
# Quantum State
# ══════════════════════════════════════════════════════════════════════════════
class QuantumState:
    """
    Single-qubit pure quantum state |ψ⟩ = α|0⟩ + β|1⟩.

    Internally stored as a complex NumPy vector [α, β].
    All gate applications are followed by re-normalisation.
    """

    def __init__(self) -> None:
        self._vec: np.ndarray = np.array([1.0 + 0j, 0.0 + 0j])  # |0⟩

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def alpha(self) -> complex:
        """Amplitude of |0⟩."""
        return complex(self._vec[0])

    @property
    def beta(self) -> complex:
        """Amplitude of |1⟩."""
        return complex(self._vec[1])

    @property
    def prob0(self) -> float:
        """Probability of measuring |0⟩."""
        return float(abs(self._vec[0]) ** 2)

    @property
    def prob1(self) -> float:
        """Probability of measuring |1⟩."""
        return float(abs(self._vec[1]) ** 2)

    @property
    def bloch_vector(self) -> tuple[float, float, float]:
        """
        Bloch sphere coordinates (x, y, z) from the density matrix.

        x = 2·Re(α*·β)   ↔  ⟨X⟩
        y = 2·Im(α*·β)   ↔  ⟨Y⟩
        z = |α|² − |β|²  ↔  ⟨Z⟩
        """
        a, b = self._vec
        x = 2.0 * float(np.real(np.conj(a) * b))
        y = 2.0 * float(np.imag(np.conj(a) * b))
        z = float(abs(a) ** 2 - abs(b) ** 2)
        return x, y, z

    @property
    def ket_label(self) -> str:
        """Human-readable ket label for common Bloch sphere states."""
        eps = 1e-3
        if abs(self.prob0 - 1.0) < eps:    return "|0⟩"
        if abs(self.prob1 - 1.0) < eps:    return "|1⟩"
        if abs(abs(self.alpha) - INV_SQRT2) < eps:
            phase = np.angle(self.beta) - np.angle(self.alpha)
            phase = (phase + 2 * math.pi) % (2 * math.pi)
            if phase < eps:                return "|+⟩"
            if abs(phase - math.pi) < 0.05: return "|−⟩"
            if abs(phase - math.pi / 2) < 0.05: return "|+i⟩"
            if abs(phase - 3 * math.pi / 2) < 0.05: return "|−i⟩"
        return "|ψ⟩"

    # ── Mutations ────────────────────────────────────────────────────────────

    def apply(self, matrix: np.ndarray) -> None:
        """Apply a 2×2 unitary gate matrix and re-normalise."""
        self._vec = matrix @ self._vec
        norm = np.linalg.norm(self._vec)
        if norm > 1e-12:
            self._vec /= norm

    def reset(self) -> None:
        """Reset to ground state |0⟩."""
        self._vec = np.array([1.0 + 0j, 0.0 + 0j])

    def collapse(self, outcome: int) -> None:
        """Collapse to a basis state post-measurement."""
        if outcome == 0:
            self._vec = np.array([1.0 + 0j, 0.0 + 0j])
        else:
            self._vec = np.array([0.0 + 0j, 1.0 + 0j])

    def measure(self) -> int:
        """
        Perform a projective measurement.
        Returns 0 or 1 weighted by Born-rule probabilities,
        then collapses the state to the measurement result.
        """
        outcome = 0 if random.random() < self.prob0 else 1
        self.collapse(outcome)
        return outcome

    @staticmethod
    def _fmt_complex(z: complex) -> str:
        sign = "+" if z.imag >= 0 else "−"
        return f"{z.real:+.4f}  {sign}  {abs(z.imag):.4f}i"


# ══════════════════════════════════════════════════════════════════════════════
# Circuit History
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class GateRecord:
    """A single gate application record stored in the circuit."""
    gate_id:  str
    label:    str
    angle:    float = 0.0
    category: str   = "clifford"


class CircuitHistory:
    """
    Ordered list of applied gates with full undo/redo support.
    Uses snapshot-based history (list-of-lists).
    """

    def __init__(self) -> None:
        self._gates:   list[GateRecord] = []
        self._history: list[list[GateRecord]] = [[]]
        self._index:   int = 0

    @property
    def gates(self) -> list[GateRecord]:
        return list(self._gates)

    def add(self, record: GateRecord) -> None:
        """Append a gate and save a history snapshot."""
        self._gates = [*self._gates, record]
        self._save()

    def remove(self, index: int) -> None:
        """Remove gate at index and save snapshot."""
        self._gates = [g for i, g in enumerate(self._gates) if i != index]
        self._save()

    def clear(self) -> None:
        """Remove all gates."""
        if not self._gates:
            return
        self._gates = []
        self._save()

    def undo(self) -> bool:
        """Move back in history. Returns True if successful."""
        if self._index <= 0:
            return False
        self._index -= 1
        self._gates = list(self._history[self._index])
        return True

    def redo(self) -> bool:
        """Move forward in history. Returns True if successful."""
        if self._index >= len(self._history) - 1:
            return False
        self._index += 1
        self._gates = list(self._history[self._index])
        return True

    def can_undo(self) -> bool:
        return self._index > 0

    def can_redo(self) -> bool:
        return self._index < len(self._history) - 1

    def _save(self) -> None:
        self._history = self._history[: self._index + 1]
        self._history.append(list(self._gates))
        self._index = len(self._history) - 1


# ══════════════════════════════════════════════════════════════════════════════
# Bloch Sphere + Circuit Renderer  (Matplotlib)
# ══════════════════════════════════════════════════════════════════════════════
class QuantumRenderer:
    """
    Matplotlib-based renderer embedded in the Tkinter window.

    Layout
    ------
    • Top  (ax3d)  : 3-D Bloch sphere
    • Bottom (ax2d) : 2-D circuit wire diagram
    """

    def __init__(self, parent_frame: tk.Frame) -> None:
        # ── Create figure ──────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(5.6, 6.2), facecolor=BG_BASE)
        gs = self.fig.add_gridspec(
            2, 1, height_ratios=[5.5, 1],
            hspace=0.06, left=0.02, right=0.98,
            top=0.97, bottom=0.02,
        )
        self.ax3d: plt.Axes = self.fig.add_subplot(gs[0], projection="3d")
        self.ax2d: plt.Axes = self.fig.add_subplot(gs[1])

        self._style_axes()

        # ── Embed in Tkinter ───────────────────────────────────────────────
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        widget = self.canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)

        # Initial render at |0⟩ (north pole)
        self.update(0.0, 0.0, 1.0, [])

    # ── Style ─────────────────────────────────────────────────────────────

    def _style_axes(self) -> None:
        """Apply dark theme to both axes."""
        self.ax3d.set_facecolor(BG_BASE)
        self.ax3d.grid(False)
        for pane in (self.ax3d.xaxis.pane, self.ax3d.yaxis.pane, self.ax3d.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("none")
        self.ax3d.set_axis_off()

        self.ax2d.set_facecolor(BG_SURFACE)
        for spine in self.ax2d.spines.values():
            spine.set_color(TEXT_MUTED)
        self.ax2d.tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)

    # ── Public update ─────────────────────────────────────────────────────

    def update(
        self,
        bx: float, by: float, bz: float,
        gates: list[GateRecord],
    ) -> None:
        """Redraw both subplots with the new Bloch vector and gate list."""
        self._draw_bloch(bx, by, bz)
        self._draw_circuit(gates)
        self.canvas.draw()

    # ── Bloch sphere ──────────────────────────────────────────────────────

    def _draw_bloch(self, bx: float, by: float, bz: float) -> None:
        ax = self.ax3d
        ax.cla()
        ax.set_facecolor(BG_BASE)
        ax.set_axis_off()

        # Sphere wireframe
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(sx, sy, sz, color=COL_CLIFFORD, alpha=0.07, linewidth=0.4)

        # Equator (z=0 circle)
        eq = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(eq), np.sin(eq), np.zeros_like(eq),
                color=COL_CLIFFORD, alpha=0.35, linewidth=1.0)

        # Coordinate axes
        ax.plot([-1.15, 1.15], [0, 0], [0, 0], color=COL_PAULI,    alpha=0.55, linewidth=1.0)
        ax.plot([0, 0], [-1.15, 1.15], [0, 0], color=COL_PHASE,    alpha=0.55, linewidth=1.0)
        ax.plot([0, 0], [0, 0], [-1.15, 1.15], color=COL_CLIFFORD, alpha=0.55, linewidth=1.0)

        # Axis labels
        kw = dict(ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text( 0,    0,    1.30, "|0⟩",  color=COL_CLIFFORD, **kw)
        ax.text( 0,    0,   -1.30, "|1⟩",  color=COL_CLIFFORD, **kw)
        ax.text( 1.30, 0,    0,    "|+⟩",  color=COL_PAULI,    **kw)
        ax.text(-1.30, 0,    0,    "|−⟩",  color=COL_PAULI,    **kw)
        ax.text( 0,    1.30, 0,    "|+i⟩", color=COL_PHASE,    **kw)
        ax.text( 0,   -1.30, 0,    "|−i⟩", color=COL_PHASE,    **kw)

        # State vector arrow
        ax.quiver(
            0, 0, 0, bx, by, bz,
            color=COL_SPECIAL, linewidth=2.5,
            arrow_length_ratio=0.18, alpha=0.95,
        )

        # Dashed projection onto equatorial plane
        ax.plot([bx, bx], [by, by], [bz, 0],
                color=COL_SPECIAL, alpha=0.25, linewidth=1, linestyle="--")
        ax.plot([0, bx], [0, by], [0, 0],
                color=COL_SPECIAL, alpha=0.25, linewidth=1, linestyle="--")

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_box_aspect([1, 1, 1])

    # ── Circuit diagram ───────────────────────────────────────────────────

    def _draw_circuit(self, gates: list[GateRecord]) -> None:
        ax = self.ax2d
        ax.cla()
        ax.set_facecolor(BG_SURFACE)
        ax.set_axis_off()

        y = 0.5
        n = len(gates)
        total_w = max(1.0, 0.15 + n * 0.16 + 0.15)
        ax.set_xlim(0, total_w)
        ax.set_ylim(0, 1)

        # Wire line
        ax.axhline(y=y, color=COL_CLIFFORD, alpha=0.30, linewidth=1.0)

        # |0⟩ label on left
        ax.text(0.02, y, "q[0] |0⟩ ─", color=TEXT_MUTED,
                fontsize=8, va="center", fontfamily="monospace")

        # Gate blocks
        x_start = 0.16
        for i, gate in enumerate(gates):
            x = x_start + i * 0.16
            color = CATEGORY_COLORS.get(gate.category, COL_CLIFFORD)

            # Box
            rect = plt.Rectangle(
                (x - 0.065, y - 0.28), 0.13, 0.56,
                facecolor=color + "22",
                edgecolor=color, linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(rect)

            # Label
            ax.text(
                x, y, gate.label,
                ha="center", va="center",
                color=color, fontsize=8, fontweight="bold",
                fontfamily="monospace", zorder=4,
            )

        # End marker
        end_x = x_start + n * 0.16 + 0.02
        ax.text(end_x, y, "─✦", color=COL_CLIFFORD, fontsize=9,
                va="center", alpha=0.6)


# ══════════════════════════════════════════════════════════════════════════════
# Main Application Window
# ══════════════════════════════════════════════════════════════════════════════
class QuantumGlassesApp:
    """
    Main Tkinter application.

    Layout
    ------
    Header  ──  title, current state ket, undo/redo/reset controls
    Body    ──  3-column grid
      Left   : gate palette (Pauli, Clifford, Rotation, Phase, Measure)
      Center : embedded Matplotlib canvas (Bloch sphere + circuit)
      Right  : state info (amplitudes, probabilities, gate info card)
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._setup_window()
        self._setup_theme()

        # ── Core components ────────────────────────────────────────────────
        self.state   = QuantumState()
        self.circuit = CircuitHistory()

        # ── Build UI ───────────────────────────────────────────────────────
        self._build_header()
        self._build_body()
        self._build_gate_palette()
        self._build_state_panel()

        # ── Initial render ─────────────────────────────────────────────────
        self._refresh_all()
        self.ui.show_gate_info("H")

        # ── Keyboard shortcuts ─────────────────────────────────────────────
        shortcuts = {"x": "X", "y": "Y", "z": "Z", "h": "H", "s": "S", "t": "T"}
        for key, gate_id in shortcuts.items():
            root.bind(key, lambda e, g=gate_id: self._apply_gate(g))
            root.bind(key.upper(), lambda e, g=gate_id: self._apply_gate(g))
        root.bind("<Control-z>", lambda e: self._undo())
        root.bind("<Control-y>", lambda e: self._redo())
        root.bind("<Control-Z>", lambda e: self._redo())  # Ctrl+Shift+Z

    # ── Window & theme ─────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.root.title("⚛  Quantum Glasses — Python Edition")
        self.root.configure(bg=BG_BASE)
        self.root.geometry("1100x720")
        self.root.minsize(900, 640)
        self.root.resizable(True, True)

    def _setup_theme(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".",            background=BG_BASE,    foreground=TEXT_PRI)
        style.configure("TFrame",       background=BG_BASE)
        style.configure("TLabel",       background=BG_BASE,    foreground=TEXT_PRI)
        style.configure("TEntry",       background=BG_SURFACE, foreground=TEXT_PRI,
                        fieldbackground=BG_SURFACE, insertcolor=TEXT_SEC)
        style.configure("TScrollbar",   background=BG_SURFACE)

    # ── Header ─────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        hdr = tk.Frame(self.root, bg=BG_PANEL, height=54)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)

        # Brand
        brand = tk.Frame(hdr, bg=BG_PANEL)
        brand.pack(side=tk.LEFT, padx=16, pady=8)
        tk.Label(brand, text="⚛  Quantum Glasses", bg=BG_PANEL,
                 fg=COL_CLIFFORD, font=("Helvetica", 16, "bold")).pack(anchor="w")
        tk.Label(brand, text="Python · NumPy · Matplotlib · Tkinter",
                 bg=BG_PANEL, fg=TEXT_MUTED, font=("Helvetica", 8)).pack(anchor="w")

        # State badge
        mid = tk.Frame(hdr, bg=BG_PANEL)
        mid.pack(side=tk.LEFT, expand=True)
        tk.Label(mid, text="Current state:", bg=BG_PANEL,
                 fg=TEXT_MUTED, font=FONT_SMALL).pack(side=tk.LEFT, padx=(0, 4))
        self.ket_label_var = tk.StringVar(value="|0⟩")
        tk.Label(mid, textvariable=self.ket_label_var, bg=BG_PANEL,
                 fg=COL_SPECIAL, font=("Courier New", 16, "bold")).pack(side=tk.LEFT)

        # Controls
        ctrl = tk.Frame(hdr, bg=BG_PANEL)
        ctrl.pack(side=tk.RIGHT, padx=16, pady=10)

        self.btn_undo = self._mk_ctrl_btn(ctrl, "⬅ Undo",  self._undo)
        self.btn_redo = self._mk_ctrl_btn(ctrl, "Redo ➡",  self._redo)
        self._mk_ctrl_btn(ctrl, "⟳ Reset |0⟩", self._reset, danger=True)

        self.btn_undo.pack(side=tk.LEFT, padx=3)
        self.btn_redo.pack(side=tk.LEFT, padx=3)
        ctrl.winfo_children()[-1].pack(side=tk.LEFT, padx=(6, 0))

    def _mk_ctrl_btn(self, parent, text, cmd, danger=False) -> tk.Button:
        fg = "#f9a8d4" if danger else TEXT_SEC
        abg = "#3a0020" if danger else "#1a2a40"
        return tk.Button(
            parent, text=text, command=cmd,
            bg=BG_SURFACE, fg=fg,
            activebackground=abg, activeforeground=TEXT_PRI,
            relief=tk.FLAT, bd=0, padx=10, pady=4,
            font=("Helvetica", 10), cursor="hand2",
        )

    # ── Body (3-column) ────────────────────────────────────────────────────

    def _build_body(self) -> None:
        body = tk.Frame(self.root, bg=BG_BASE)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Left panel
        self.left_panel = tk.Frame(body, bg=BG_PANEL, width=220)
        self.left_panel.grid(row=0, column=0, sticky="ns", padx=(0, 7))
        self.left_panel.grid_propagate(False)

        # Center panel
        center = tk.Frame(body, bg=BG_BASE)
        center.grid(row=0, column=1, sticky="nsew", padx=0)
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        self.renderer = QuantumRenderer(center)

        # Right panel
        self.right_panel = tk.Frame(body, bg=BG_PANEL, width=260)
        self.right_panel.grid(row=0, column=2, sticky="ns", padx=(7, 0))
        self.right_panel.grid_propagate(False)

        self.ui = _UIPanels(self.left_panel, self.right_panel)

    # ── Gate palette (delegates to _UIPanels) ──────────────────────────────

    def _build_gate_palette(self) -> None:
        self.ui.build_palette(
            on_gate=self._apply_gate,
            on_measure=self._measure,
            get_rot_angle=lambda: self._angle_var.get(),
            get_phase_angle=lambda: self._phi_var.get(),
        )
        self._angle_var: tk.StringVar = self.ui.rot_angle_var
        self._phi_var:   tk.StringVar = self.ui.phase_angle_var

    def _build_state_panel(self) -> None:
        self.ui.build_state_panel(on_info_hover=lambda g: self.ui.show_gate_info(g))

    # ── Gate application ───────────────────────────────────────────────────

    def _apply_gate(self, gate_id: str, angle_deg: float = 0.0) -> None:
        """Apply gate, record in circuit, refresh display."""
        matrix = GateLibrary.get_matrix(gate_id, angle_deg)
        self.state.apply(matrix)

        info  = GateLibrary.get_info(gate_id)
        label = info["label"] if info else gate_id
        if gate_id in ("RX", "RY", "RZ", "P"):
            label = f"{label}({int(angle_deg)}°)"
        cat = info["category"] if info else "clifford"

        self.circuit.add(GateRecord(gate_id, label, angle_deg, cat))
        self.ui.show_gate_info(gate_id)
        self._refresh_all()

    def _undo(self, _event=None) -> None:
        if self.circuit.undo():
            self._replay_circuit()
            self._refresh_all()

    def _redo(self, _event=None) -> None:
        if self.circuit.redo():
            self._replay_circuit()
            self._refresh_all()

    def _reset(self) -> None:
        self.state.reset()
        self.circuit.clear()
        self._refresh_all()

    def _measure(self) -> None:
        """Simulate a projective measurement with Born-rule probabilities."""
        p0, p1 = self.state.prob0, self.state.prob1
        outcome = self.state.measure()
        self.circuit.clear()
        self._refresh_all()

        lbl  = "|0⟩" if outcome == 0 else "|1⟩"
        prob = p0 if outcome == 0 else p1
        messagebox.showinfo(
            "⚡  Measurement Result",
            f"Outcome:  {lbl}\n\n"
            f"P(|0⟩) = {p0 * 100:.2f}%\n"
            f"P(|1⟩) = {p1 * 100:.2f}%\n\n"
            f"The qubit collapsed to {lbl} with probability {prob * 100:.2f}%.",
            parent=self.root,
        )

    def _replay_circuit(self) -> None:
        """Recompute state from scratch by replaying all recorded gates."""
        self.state.reset()
        for gate in self.circuit.gates:
            matrix = GateLibrary.get_matrix(gate.gate_id, gate.angle)
            self.state.apply(matrix)

    def _refresh_all(self) -> None:
        """Push latest state to every UI element and the Matplotlib canvas."""
        bx, by, bz = self.state.bloch_vector

        # Bloch sphere + circuit
        self.renderer.update(bx, by, bz, self.circuit.gates)

        # Ket label in header
        self.ket_label_var.set(self.state.ket_label)

        # State info panel
        self.ui.update_state(self.state)

        # Undo/redo button states
        self.btn_undo.config(state=tk.NORMAL if self.circuit.can_undo() else tk.DISABLED)
        self.btn_redo.config(state=tk.NORMAL if self.circuit.can_redo() else tk.DISABLED)


# ══════════════════════════════════════════════════════════════════════════════
# UI Panels helper (keeps QuantumGlassesApp tidy)
# ══════════════════════════════════════════════════════════════════════════════
class _UIPanels:
    """
    Builds and owns all Tkinter sub-widgets for the left (palette)
    and right (state info) panels.
    """

    def __init__(self, left: tk.Frame, right: tk.Frame) -> None:
        self._left  = left
        self._right = right
        self.rot_angle_var   = tk.StringVar(value="90")
        self.phase_angle_var = tk.StringVar(value="45")

    # ── Gate palette ───────────────────────────────────────────────────────

    def build_palette(
        self, on_gate, on_measure,
        get_rot_angle, get_phase_angle,
    ) -> None:
        p = self._left
        self._section_header(p, "Gate Palette")

        # ── Pauli ──
        self._cat_label(p, "Pauli Gates", COL_PAULI)
        row = tk.Frame(p, bg=BG_PANEL)
        row.pack(fill=tk.X, padx=8, pady=(0, 6))
        for g in ("X", "Y", "Z"):
            self._gate_btn(row, g, COL_PAULI, lambda gid=g: on_gate(gid)).pack(
                side=tk.LEFT, padx=3, pady=2, expand=True, fill=tk.X)

        # ── Clifford ──
        self._sep(p)
        self._cat_label(p, "Clifford Gates", COL_CLIFFORD)
        row2 = tk.Frame(p, bg=BG_PANEL)
        row2.pack(fill=tk.X, padx=8, pady=(0, 4))
        for g in ("H", "S", "S†", "T", "T†"):
            gid = "S_DAG" if g == "S†" else ("T_DAG" if g == "T†" else g)
            self._gate_btn(row2, g, COL_CLIFFORD, lambda gid=gid: on_gate(gid)).pack(
                side=tk.LEFT, padx=2, pady=2, expand=True, fill=tk.X)

        # ── Rotation ──
        self._sep(p)
        self._cat_label(p, "Rotation Gates", COL_ROTATION)
        self._angle_row(p, "θ", self.rot_angle_var, COL_ROTATION)
        row3 = tk.Frame(p, bg=BG_PANEL)
        row3.pack(fill=tk.X, padx=8, pady=(0, 6))
        for g in ("Rx", "Ry", "Rz"):
            gid = g.upper()
            self._gate_btn(row3, g, COL_ROTATION, lambda gid=gid: on_gate(
                gid, float(get_rot_angle() or 90))).pack(
                side=tk.LEFT, padx=3, pady=2, expand=True, fill=tk.X)

        # ── Phase ──
        self._sep(p)
        self._cat_label(p, "Phase Gate", COL_PHASE)
        self._angle_row(p, "φ", self.phase_angle_var, COL_PHASE)
        self._gate_btn(
            p, "P(φ)", COL_PHASE,
            lambda: on_gate("P", float(get_phase_angle() or 45)),
        ).pack(fill=tk.X, padx=8, pady=(0, 6))

        # ── Measure ──
        self._sep(p)
        tk.Button(
            p, text="⚡  Measure", command=on_measure,
            bg="#1a0f30", fg=COL_SPECIAL,
            activebackground="#2a1050", activeforeground="#e9d5ff",
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT, bd=0, padx=10, pady=10, cursor="hand2",
        ).pack(fill=tk.X, padx=8, pady=(6, 4))

        # ── Shortcuts ──
        self._sep(p)
        tk.Label(p, text="Shortcuts: X Y Z H S T   ·  Ctrl+Z/Y  undo/redo",
                 bg=BG_PANEL, fg=TEXT_MUTED, font=("Helvetica", 8),
                 wraplength=200, justify=tk.LEFT).pack(padx=8, pady=4, anchor="w")

    def _section_header(self, parent, text: str) -> None:
        tk.Label(parent, text=text, bg=BG_PANEL, fg=COL_CLIFFORD,
                 font=("Helvetica", 12, "bold")).pack(pady=(10, 6), padx=8, anchor="w")

    def _cat_label(self, parent, text: str, color: str) -> None:
        f = tk.Frame(parent, bg=BG_PANEL)
        f.pack(fill=tk.X, padx=8, pady=(4, 2))
        tk.Frame(f, bg=color, width=6, height=6).pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(f, text=text, bg=BG_PANEL, fg=TEXT_MUTED,
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)

    def _sep(self, parent) -> None:
        tk.Frame(parent, bg=TEXT_MUTED, height=1).pack(fill=tk.X, padx=8, pady=4)

    def _gate_btn(self, parent, label: str, color: str, cmd) -> tk.Button:
        bg = BG_SURFACE
        btn = tk.Button(
            parent, text=label, command=cmd,
            bg=bg, fg=color,
            activebackground=color + "33",
            activeforeground=color,
            font=("Courier New", 11, "bold"),
            relief=tk.FLAT, bd=0, padx=6, pady=6, cursor="hand2",
        )
        btn.bind("<Enter>", lambda e, b=btn, c=color: b.config(bg=c + "25"))
        btn.bind("<Leave>", lambda e, b=btn:          b.config(bg=bg))
        return btn

    def _angle_row(self, parent, sym: str, var: tk.StringVar, color: str) -> None:
        f = tk.Frame(parent, bg=BG_PANEL)
        f.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(f, text=f"{sym} =", bg=BG_PANEL, fg=color,
                 font=("Courier New", 11, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        e = tk.Entry(f, textvariable=var, width=6, bg=BG_SURFACE, fg=TEXT_PRI,
                     insertbackground=TEXT_SEC, relief=tk.FLAT,
                     font=("Courier New", 11))
        e.pack(side=tk.LEFT)
        tk.Label(f, text="°", bg=BG_PANEL, fg=TEXT_SEC,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=2)

    # ── State info panel ───────────────────────────────────────────────────

    def build_state_panel(self, on_info_hover=None) -> None:
        p = self._right
        tk.Label(p, text="Quantum State  |ψ⟩", bg=BG_PANEL,
                 fg=COL_CLIFFORD, font=("Helvetica", 12, "bold")).pack(
                     pady=(10, 6), padx=10, anchor="w")

        # Amplitude display
        self._amp_frame = tk.Frame(p, bg=BG_PANEL)
        self._amp_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        self._alpha_var = tk.StringVar(value="1.0000 + 0.0000i")
        self._beta_var  = tk.StringVar(value="0.0000 + 0.0000i")

        for label_text, var in (("α · |0⟩", self._alpha_var),
                                  ("β · |1⟩", self._beta_var)):
            tk.Label(self._amp_frame, text=label_text, bg=BG_PANEL,
                     fg=TEXT_MUTED, font=("Helvetica", 9)).pack(anchor="w")
            tk.Label(self._amp_frame, textvariable=var, bg=BG_PANEL,
                     fg=TEXT_PRI, font=("Courier New", 10)).pack(anchor="w", pady=(0, 4))

        # Probabilities
        tk.Frame(p, bg=TEXT_MUTED, height=1).pack(fill=tk.X, padx=10, pady=2)
        tk.Label(p, text="Measurement Probabilities", bg=BG_PANEL,
                 fg=COL_CLIFFORD, font=("Helvetica", 11, "bold")).pack(
                     pady=(8, 4), padx=10, anchor="w")

        self._prob0_var = tk.StringVar(value="100.00%")
        self._prob1_var = tk.StringVar(value="  0.00%")

        self._prob_canvas = tk.Canvas(p, bg=BG_PANEL, height=52,
                                       highlightthickness=0)
        self._prob_canvas.pack(fill=tk.X, padx=10, pady=(0, 6))

        # Gate info
        tk.Frame(p, bg=TEXT_MUTED, height=1).pack(fill=tk.X, padx=10, pady=2)
        tk.Label(p, text="Gate Information", bg=BG_PANEL,
                 fg=COL_CLIFFORD, font=("Helvetica", 11, "bold")).pack(
                     pady=(8, 4), padx=10, anchor="w")

        self._info_text = tk.Text(
            p, bg=BG_SURFACE, fg=TEXT_SEC,
            font=("Helvetica", 9), relief=tk.FLAT,
            wrap=tk.WORD, state=tk.DISABLED,
            padx=6, pady=6, height=14,
            insertbackground=TEXT_SEC,
        )
        self._info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Tag colours for info text
        self._info_text.tag_configure("heading", foreground=COL_CLIFFORD,
                                       font=("Helvetica", 10, "bold"))
        self._info_text.tag_configure("matrix",  foreground=TEXT_SEC,
                                       font=("Courier New", 9))
        self._info_text.tag_configure("effects", foreground=COL_PHASE,
                                       font=("Courier New", 9, "bold"))
        self._info_text.tag_configure("analogy", foreground=COL_SPECIAL,
                                       font=("Helvetica", 9, "italic"))
        self._info_text.tag_configure("label",   foreground=TEXT_MUTED,
                                       font=("Helvetica", 8, "bold"))

    def update_state(self, state: QuantumState) -> None:
        """Update amplitude text, probability bars, and ket label."""
        def _fmt(z: complex) -> str:
            sign = "+" if z.imag >= 0 else "−"
            return f"{z.real:+.4f}  {sign}  {abs(z.imag):.4f}i"

        self._alpha_var.set(_fmt(state.alpha))
        self._beta_var.set(_fmt(state.beta))
        self._draw_prob_bars(state.prob0, state.prob1)

    def _draw_prob_bars(self, p0: float, p1: float) -> None:
        c = self._prob_canvas
        c.delete("all")
        W = c.winfo_width() or 220
        BAR_H = 16
        TRACK_W = W - 80

        for i, (label, p, color) in enumerate((
            ("|0⟩", p0, COL_CLIFFORD),
            ("|1⟩", p1, COL_PAULI),
        )):
            y = 4 + i * 26
            c.create_text(4, y + BAR_H // 2, text=f"P({label})",
                          anchor="w", fill=TEXT_MUTED, font=("Courier New", 9))
            c.create_rectangle(54, y, 54 + TRACK_W, y + BAR_H,
                               fill=BG_SURFACE, outline=TEXT_MUTED, width=0)
            fill_w = max(2, int(p * TRACK_W))
            c.create_rectangle(54, y, 54 + fill_w, y + BAR_H,
                               fill=color, outline="")
            c.create_text(58 + TRACK_W, y + BAR_H // 2,
                          text=f"{p * 100:.1f}%",
                          anchor="w", fill=TEXT_PRI,
                          font=("Courier New", 9, "bold"))

    def show_gate_info(self, gate_id: str) -> None:
        """Populate the info text widget with gate metadata."""
        info = GateLibrary.get_info(gate_id)
        t    = self._info_text
        t.config(state=tk.NORMAL)
        t.delete("1.0", tk.END)

        if not info:
            t.insert(tk.END, "Hover over a gate to see its properties.",
                     "analogy")
            t.config(state=tk.DISABLED)
            return

        color = CATEGORY_COLORS.get(info["category"], COL_CLIFFORD)

        t.insert(tk.END, f"  {info['name']}\n", "heading")
        t.insert(tk.END, f"Category: {info['category']}\n\n", "label")

        t.insert(tk.END, "Description\n", "label")
        t.insert(tk.END, info["desc"] + "\n\n")

        t.insert(tk.END, "Matrix\n", "label")
        t.insert(tk.END, info["matrix"] + "\n\n", "matrix")

        t.insert(tk.END, "Effects\n", "label")
        t.insert(tk.END, info["effects"] + "\n\n", "effects")

        t.insert(tk.END, "💡  " + info["analogy"], "analogy")

        t.config(state=tk.DISABLED)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    try:
        import numpy  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"Missing dependency: {exc}\n"
            "Install with:  pip install numpy matplotlib"
        )

    root = tk.Tk()
    app  = QuantumGlassesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
