# ⚛ Quantum Glasses

> **An interactive, browser-based quantum gate visualizer** — apply gates, watch your qubit evolve on a real-time 3-D Bloch sphere, and understand quantum computing intuitively.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue?style=flat-square&logo=github)](https://priyamdixit21.github.io/Quantum-Glasses/)
[![GitHub](https://img.shields.io/badge/Source-GitHub-181717?style=flat-square&logo=github)](https://github.com/PriyamDixit21/Quantum-Glasses)
[![Tech](https://img.shields.io/badge/Built%20with-HTML%20·%20CSS%20·%20JavaScript-informational?style=flat-square)]()
[![No Dependencies](https://img.shields.io/badge/Dependencies-None-success?style=flat-square)]()

---

## 🎯 What is Quantum Glasses?

Quantum Glasses is a **zero-install, zero-dependency** web application for visualizing single-qubit quantum gate operations.  

Pick a gate → click → watch the qubit state arrow sweep across the Bloch sphere in real time. Every transformation is backed by correct complex-number linear algebra computed entirely in the browser.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌐 **Live 3-D Bloch Sphere** | Drag to rotate, scroll to zoom. Front/back grid lines, labelled axes, glowing state vector |
| 🧱 **Visual Circuit Builder** | Gate blocks appear on a wire as you apply them; click × to remove any gate |
| ⏪ **Undo / Redo** | Full history stack — replays circuit from \|0⟩ to compute exact state |
| 📊 **Live Probability Bars** | Animated bars showing P(\|0⟩) and P(\|1⟩) update with every gate |
| 🔢 **Amplitude Display** | Complex amplitudes α and β shown to 4 decimal places |
| 📖 **Gate Info Cards** | Hover any gate — see its matrix, plain-English description, and real-world analogy |
| ⚡ **Measurement Collapse** | Probabilistic outcome reveal with dramatic modal animation |
| ⌨️ **Keyboard Shortcuts** | `X Y Z H S T` keys apply gates; `Ctrl+Z/Y` undo/redo |

---

## 🎛️ Supported Gates

### Pauli Gates
| Symbol | Name | Effect |
|--------|------|--------|
| **X** | Pauli-X / NOT | Flips \|0⟩↔\|1⟩ — 180° rotation around X-axis |
| **Y** | Pauli-Y | Bit-flip + phase-flip — 180° around Y-axis |
| **Z** | Pauli-Z | Phase flip — 180° around Z-axis |

### Clifford Gates
| Symbol | Name | Effect |
|--------|------|--------|
| **H** | Hadamard | Creates superposition: \|0⟩→\|+⟩, \|1⟩→\|−⟩ |
| **S** | Phase (√Z) | 90° phase on \|1⟩ |
| **S†** | S-dagger | −90° phase on \|1⟩ (inverse of S) |
| **T** | π/8 gate | 45° phase on \|1⟩ — enables universal computation |
| **T†** | T-dagger | −45° phase on \|1⟩ |

### Parameterised Gates
| Symbol | Name | Parameter |
|--------|------|-----------|
| **Rx(θ)** | X-axis rotation | θ in degrees |
| **Ry(θ)** | Y-axis rotation | θ in degrees |
| **Rz(θ)** | Z-axis rotation | θ in degrees |
| **P(φ)** | Arbitrary phase | φ in degrees |

---

## 🚀 Getting Started

**No installation required.** Just open the file in a browser:

```bash
git clone https://github.com/your-username/Quantum-Glasses.git
cd Quantum-Glasses
open index.html          # macOS
# or drag index.html into your browser
```

That's it — no `npm install`, no Python server, no build step.

---

## 🧠 How It Works

### Quantum Math (Pure JavaScript)
- Qubit state `ψ = α|0⟩ + β|1⟩` stored as two complex numbers `{re, im}`  
- Gate matrices are 2×2 complex arrays applied via matrix-vector multiplication  
- State is always re-normalised after each gate  
- Bloch coordinates computed as: `x = 2·Re(α*β)`, `y = 2·Im(α*β)`, `z = |α|²−|β|²`

### Bloch Sphere Renderer
- Custom 3-D renderer on an HTML5 `<canvas>` — **no WebGL, no libraries**  
- Orthographic projection with mouse-drag-controlled azimuth/elevation angles  
- Front/back grid line split: front solid, back dashed — correct depth perception  
- State vector arrow animates with ease-in-out interpolation

### Circuit History
- Every gate apply / remove / undo creates a snapshot of the gate list  
- State is always recomputed by **replaying gates from |0⟩** — guarantees correctness  

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Structure | HTML5 (semantic, ARIA-accessible) |
| Styling | Vanilla CSS (glassmorphism, CSS custom properties, animations) |
| Logic | Vanilla JavaScript ES6 (modular IIFE / class pattern) |
| Fonts | Google Fonts — Space Grotesk + JetBrains Mono |
| Rendering | HTML5 Canvas 2D API |
| Dependencies | **None** |

---

## 📂 Project Structure

```
Quantum-Glasses/
├── index.html          # App shell — 3-column layout
├── style.css           # Design system (dark space theme, glassmorphism)
├── js/
│   ├── quantum.js      # Complex number & quantum math engine
│   ├── gates.js        # Gate matrices, categories, metadata
│   ├── bloch.js        # 3-D Bloch sphere canvas renderer
│   ├── circuit.js      # Circuit builder & undo/redo
│   ├── ui.js           # DOM update manager
│   └── app.js          # Main coordinator (event wiring)
├── README.md
└── project.py          # Original Python/Qiskit prototype (reference only)
```

---

## 🔬 Quantum Correctness Checks

| Test | Expected | ✓ |
|------|----------|---|
| H on \|0⟩ | P(\|0⟩) = P(\|1⟩) = 50% | ✅ |
| X on \|0⟩ | \|1⟩, P(\|1⟩) = 100% | ✅ |
| H·H on \|0⟩ | Back to \|0⟩ | ✅ |
| Rx(180°) on \|0⟩ | Equivalent to X | ✅ |
| Z on \|+⟩ | \|−⟩ | ✅ |
| Rz(90°) on \|0⟩ | z-axis unchanged | ✅ |

---

## 👤 Author

**Priyam Dixit**  
Built as part of a quantum computing course — demonstrating Pauli, Clifford, rotation, and phase gates through interactive visualization.

---

## 📄 License

MIT — free to use, modify, and share.
