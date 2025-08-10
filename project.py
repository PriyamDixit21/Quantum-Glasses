# Quantum-Glasses
"""
quantum_glasses.py

Tkinter GUI + Qiskit backend to visualize single-qubit operations on a Bloch sphere.
Features:
 - Buttons: X, Y, Z, H, S
 - Rotation gates: Rx, Ry, Rz (angle in degrees from an Entry box)
 - Reset (to |0>) and Measure (show probabilities)
 - Bloch sphere visualization opens in a separate Matplotlib window

Author: Re-implementation (original code)
"""

import sys
import math
import tkinter as tk
from tkinter import messagebox

# Try imports and give helpful error if missing
try:
    import numpy as np
    from qiskit.quantum_info import Statevector, Operator
    from qiskit.visualization import plot_bloch_multivector
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing required packages. Please run:\n  pip install -r requirements.txt")
    print("Error:", e)
    sys.exit(1)


# ---------- Quantum helper functions ----------
def state_from_vector(vec):
    """Ensure normalization and return Statevector"""
    vec = np.asarray(vec, dtype=complex)
    if np.linalg.norm(vec) == 0:
        vec = np.array([1.0 + 0j, 0.0 + 0j])
    vec = vec / np.linalg.norm(vec)
    return Statevector(vec)


# Pauli and common gates as numpy arrays
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)


def rx_matrix(theta_rad):
    """Rx(θ) matrix"""
    return np.cos(theta_rad / 2) * np.eye(2) - 1j * np.sin(theta_rad / 2) * X


def ry_matrix(theta_rad):
    """Ry(θ) matrix"""
    return np.cos(theta_rad / 2) * np.eye(2) - 1j * np.sin(theta_rad / 2) * Y


def rz_matrix(theta_rad):
    """Rz(θ) matrix"""
    return np.cos(theta_rad / 2) * np.eye(2) - 1j * np.sin(theta_rad / 2) * Z


# ---------- Application ----------
class QuantumGlassesApp:
    def __init__(self, master):
        self.master = master
        master.title("Quantum Glasses")
        master.geometry("520x420")
        master.resizable(False, False)

        # Initial quantum state |0>
        self.state = state_from_vector([1, 0])

        # UI fonts/colors (simple)
        label_font = ("Arial", 12, "bold")
        small_font = ("Arial", 10)

        # --- Top frame: display current state amplitudes and Bloch button ---
        top = tk.Frame(master)
        top.pack(pady=8)

        tk.Label(top, text="Quantum Glasses", font=("Arial", 18, "bold")).pack()

        info_frame = tk.Frame(master)
        info_frame.pack(pady=6)

        tk.Label(info_frame, text="Amplitude α (|0>):", font=small_font).grid(row=0, column=0, sticky="w")
        self.alpha_var = tk.StringVar(value=self._alpha_text())
        self.alpha_entry = tk.Entry(info_frame, textvariable=self.alpha_var, width=28, state="readonly")
        self.alpha_entry.grid(row=0, column=1, padx=6, pady=2)

        tk.Label(info_frame, text="Amplitude β (|1>):", font=small_font).grid(row=1, column=0, sticky="w")
        self.beta_var = tk.StringVar(value=self._beta_text())
        self.beta_entry = tk.Entry(info_frame, textvariable=self.beta_var, width=28, state="readonly")
        self.beta_entry.grid(row=1, column=1, padx=6, pady=2)

        # Bloch button
        bloch_btn = tk.Button(master, text="Show Bloch Sphere", command=self.show_bloch)
        bloch_btn.pack(pady=6)

        # --- Buttons frame ---
        buttons_frame = tk.Frame(master)
        buttons_frame.pack(pady=6, fill="x")

        # First row (Paulis + H + S)
        row1 = tk.Frame(buttons_frame)
        row1.pack(pady=4)
        tk.Button(row1, text="X", width=8, command=lambda: self.apply_gate(X, "X")).pack(side="left", padx=4)
        tk.Button(row1, text="Y", width=8, command=lambda: self.apply_gate(Y, "Y")).pack(side="left", padx=4)
        tk.Button(row1, text="Z", width=8, command=lambda: self.apply_gate(Z, "Z")).pack(side="left", padx=4)
        tk.Button(row1, text="H", width=8, command=lambda: self.apply_gate(H, "H")).pack(side="left", padx=4)
        tk.Button(row1, text="S", width=8, command=lambda: self.apply_gate(S, "S")).pack(side="left", padx=4)

        # Second row (rotations with angle)
        row2 = tk.Frame(buttons_frame)
        row2.pack(pady=6)
        tk.Label(row2, text="Angle (deg):", font=small_font).pack(side="left", padx=(0, 6))
        self.angle_entry = tk.Entry(row2, width=8)
        self.angle_entry.insert(0, "90")  # default
        self.angle_entry.pack(side="left", padx=(0, 8))

        tk.Button(row2, text="Rx", width=8, command=lambda: self.apply_rotation("rx")).pack(side="left", padx=4)
        tk.Button(row2, text="Ry", width=8, command=lambda: self.apply_rotation("ry")).pack(side="left", padx=4)
        tk.Button(row2, text="Rz", width=8, command=lambda: self.apply_rotation("rz")).pack(side="left", padx=4)

        # Third row (reset, measure)
        row3 = tk.Frame(buttons_frame)
        row3.pack(pady=6)
        tk.Button(row3, text="Reset", width=12, command=self.reset_state).pack(side="left", padx=8)
        tk.Button(row3, text="Measure", width=12, command=self.measure_state).pack(side="left", padx=8)

        # Status bar / small notes
        notes = tk.Label(master, text="Tip: Click gates to apply. Use angle box for rotations.", font=("Arial", 9))
        notes.pack(side="bottom", pady=6)

        # update UI
        self.update_amplitude_display()

    # --- UI helpers ---
    def _alpha_text(self):
        a = self.state.data[0]
        return f"{a.real:.4f} + {a.imag:.4f}i"

    def _beta_text(self):
        b = self.state.data[1]
        return f"{b.real:.4f} + {b.imag:.4f}i"

    def update_amplitude_display(self):
        self.alpha_var.set(self._alpha_text())
        self.beta_var.set(self._beta_text())

    # --- Quantum operations ---
    def apply_gate(self, matrix, name=""):
        """Apply a 2x2 matrix to the current state"""
        try:
            op = Operator(matrix)
            new_sv = self.state.evolve(op)
            self.state = new_sv
            self.update_amplitude_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply gate {name}: {e}")

    def apply_rotation(self, kind):
        ang_text = self.angle_entry.get()
        try:
            ang_deg = float(ang_text)
        except ValueError:
            messagebox.showerror("Invalid angle", "Please enter a numeric angle in degrees.")
            return
        theta = math.radians(ang_deg)
        if kind == "rx":
            mat = rx_matrix(theta)
        elif kind == "ry":
            mat = ry_matrix(theta)
        elif kind == "rz":
            mat = rz_matrix(theta)
        else:
            return
        self.apply_gate(mat, name=kind.upper() + f"({ang_deg}°)")

    def reset_state(self):
        self.state = state_from_vector([1, 0])
        self.update_amplitude_display()
        messagebox.showinfo("Reset", "State reset to |0>")

    def measure_state(self):
        # Get probabilities
        probs = self.state.probabilities_dict()
        # Format nicely
        p0 = probs.get("0", 0.0)
        p1 = probs.get("1", 0.0)
        msg = f"P(|0>) = {p0:.4f}\nP(|1>) = {p1:.4f}"
        messagebox.showinfo("Measurement probabilities", msg)

    def show_bloch(self):
        # Use qiskit's plot_bloch_multivector
        try:
            fig = plot_bloch_multivector(self.state)
            fig.suptitle("Bloch Sphere — Current State", fontsize=12)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to render Bloch sphere: {e}")


# ---------- Run ----------
def main():
    root = tk.Tk()
    app = QuantumGlassesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
