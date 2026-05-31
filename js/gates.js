/**
 * gates.js
 * Quantum gate definitions: matrices, categories, and educational metadata.
 * Depends on: quantum.js (QM namespace)
 */
'use strict';

const GATES = (() => {
  const { c, cexp } = QM;
  const I    = c(1,  0);
  const ZERO = c(0,  0);
  const INV2 = 1 / Math.sqrt(2);

  // ─── Static Gate Matrices [m00, m01, m10, m11] ──────────────────────────────

  const STATIC = {
    X:     [ZERO,       I,          I,          ZERO],
    Y:     [ZERO,       c(0, -1),   c(0, 1),    ZERO],
    Z:     [I,          ZERO,       ZERO,        c(-1, 0)],
    H:     [c(INV2),    c(INV2),    c(INV2),    c(-INV2)],
    S:     [I,          ZERO,       ZERO,        c(0,  1)],
    S_DAG: [I,          ZERO,       ZERO,        c(0, -1)],
    T:     [I,          ZERO,       ZERO,        cexp(Math.PI / 4)],
    T_DAG: [I,          ZERO,       ZERO,        cexp(-Math.PI / 4)],
  };

  // ─── Parameterised Rotation Matrices ────────────────────────────────────────

  function rxMatrix(theta) {
    const cos  = c(Math.cos(theta / 2));
    const isin = c(0, -Math.sin(theta / 2));
    return [cos, isin, isin, cos];
  }

  function ryMatrix(theta) {
    const cos  = c(Math.cos(theta / 2));
    const psin = c( Math.sin(theta / 2));
    const nsin = c(-Math.sin(theta / 2));
    return [cos, nsin, psin, cos];
  }

  function rzMatrix(theta) {
    return [cexp(-theta / 2), ZERO, ZERO, cexp(theta / 2)];
  }

  function pMatrix(phi) {
    return [I, ZERO, ZERO, cexp(phi)];
  }

  // ─── Gate Educational Metadata ──────────────────────────────────────────────

  const INFO = {
    X: {
      id: 'X', name: 'Pauli-X  (NOT Gate)', symbol: 'X', category: 'pauli',
      description:
        'The quantum NOT gate. Flips |0⟩ to |1⟩ and vice versa. ' +
        'Rotates the Bloch sphere 180° around the X-axis.',
      matrixText: '[ [0, 1],\n  [1, 0] ]',
      effect0: '|0⟩  →  |1⟩',
      effect1: '|1⟩  →  |0⟩',
      analogy: '🎲 Like flipping a coin — heads becomes tails instantly.'
    },
    Y: {
      id: 'Y', name: 'Pauli-Y', symbol: 'Y', category: 'pauli',
      description:
        'Combines a bit-flip and a phase-flip simultaneously. ' +
        'Rotates the Bloch sphere 180° around the Y-axis.',
      matrixText: '[ [0, −i],\n  [i,  0] ]',
      effect0: '|0⟩  →  i|1⟩',
      effect1: '|1⟩  →  −i|0⟩',
      analogy: '🌀 A spin that flips both the state and its phase.'
    },
    Z: {
      id: 'Z', name: 'Pauli-Z  (Phase Flip)', symbol: 'Z', category: 'pauli',
      description:
        'Leaves |0⟩ unchanged but adds a −1 phase to |1⟩. ' +
        'Rotates the Bloch sphere 180° around the Z-axis.',
      matrixText: '[ [1,  0],\n  [0, −1] ]',
      effect0: '|0⟩  →  |0⟩',
      effect1: '|1⟩  →  −|1⟩',
      analogy: '🪞 Like a mirror reflection — one side unchanged, the other flips sign.'
    },
    H: {
      id: 'H', name: 'Hadamard', symbol: 'H', category: 'clifford',
      description:
        'Creates superposition! Transforms a classical 0 or 1 into an equal ' +
        'quantum superposition of both. The most important gate in quantum computing.',
      matrixText: '(1/√2) · [ [1, 1],\n            [1, −1] ]',
      effect0: '|0⟩  →  |+⟩ = (|0⟩ + |1⟩)/√2',
      effect1: '|1⟩  →  |−⟩ = (|0⟩ − |1⟩)/√2',
      analogy: '🎲 Spinning a coin — both heads AND tails simultaneously until measured.'
    },
    S: {
      id: 'S', name: 'S Gate  (√Z)', symbol: 'S', category: 'clifford',
      description:
        'Applies a 90° (π/2) phase to |1⟩. Equivalent to √Z. ' +
        'Two S gates applied sequentially equal one Z gate.',
      matrixText: '[ [1, 0],\n  [0, i] ]',
      effect0: '|0⟩  →  |0⟩',
      effect1: '|1⟩  →  i|1⟩',
      analogy: '🕐 A quarter-turn of the phase — like rotating a clock by 90°.'
    },
    S_DAG: {
      id: 'S_DAG', name: 'S† Dagger', symbol: 'S†', category: 'clifford',
      description:
        'The inverse of the S gate. Applies a −90° (−π/2) phase to |1⟩. ' +
        'S† · S = Identity.',
      matrixText: '[ [1,  0],\n  [0, −i] ]',
      effect0: '|0⟩  →  |0⟩',
      effect1: '|1⟩  →  −i|1⟩',
      analogy: '🕐 Rotate the clock backwards by 90°. Undoes the S gate.'
    },
    T: {
      id: 'T', name: 'T Gate  (π/8)', symbol: 'T', category: 'clifford',
      description:
        'Applies a 45° (π/4) phase to |1⟩. This non-Clifford gate is ' +
        'critical for universal quantum computation.',
      matrixText: '[ [1,      0       ],\n  [0, e^(iπ/4) ] ]',
      effect0: '|0⟩  →  |0⟩',
      effect1: '|1⟩  →  e^(iπ/4)|1⟩',
      analogy: '🔑 The "secret ingredient" — enables algorithms impossible without it.'
    },
    T_DAG: {
      id: 'T_DAG', name: 'T† Dagger', symbol: 'T†', category: 'clifford',
      description:
        'The inverse of the T gate. Applies a −45° (−π/4) phase to |1⟩.',
      matrixText: '[ [1,       0        ],\n  [0, e^(−iπ/4) ] ]',
      effect0: '|0⟩  →  |0⟩',
      effect1: '|1⟩  →  e^(−iπ/4)|1⟩',
      analogy: '↩ Reverses the T gate rotation by 45°.'
    },
    RX: {
      id: 'RX', name: 'Rx(θ) — X Rotation', symbol: 'Rx', category: 'rotation',
      description:
        'Rotates the Bloch sphere by angle θ around the X-axis. ' +
        'At θ = 180° it equals the X gate.',
      matrixText:
        '[ [cos(θ/2),   −i·sin(θ/2) ],\n  [−i·sin(θ/2), cos(θ/2)   ] ]',
      effect0: '|0⟩  →  cos(θ/2)|0⟩ − i·sin(θ/2)|1⟩',
      effect1: '|1⟩  →  −i·sin(θ/2)|0⟩ + cos(θ/2)|1⟩',
      analogy: '🌐 Tilting a globe east-west by θ degrees around the x-axis pole.'
    },
    RY: {
      id: 'RY', name: 'Ry(θ) — Y Rotation', symbol: 'Ry', category: 'rotation',
      description:
        'Rotates the Bloch sphere by angle θ around the Y-axis. ' +
        'Real-valued matrix — no imaginary phase. At θ = 180° equals Y (up to phase).',
      matrixText:
        '[ [cos(θ/2), −sin(θ/2) ],\n  [sin(θ/2),  cos(θ/2) ] ]',
      effect0: '|0⟩  →  cos(θ/2)|0⟩ + sin(θ/2)|1⟩',
      effect1: '|1⟩  →  −sin(θ/2)|0⟩ + cos(θ/2)|1⟩',
      analogy: '🌍 A pure real rotation — like tilting a globe north-south.'
    },
    RZ: {
      id: 'RZ', name: 'Rz(θ) — Z Rotation', symbol: 'Rz', category: 'rotation',
      description:
        'Rotates the Bloch sphere by angle θ around the Z-axis. ' +
        'Only affects the relative phase between |0⟩ and |1⟩.',
      matrixText:
        '[ [e^(−iθ/2), 0         ],\n  [0,          e^(iθ/2) ] ]',
      effect0: '|0⟩  →  e^(−iθ/2)|0⟩',
      effect1: '|1⟩  →  e^(iθ/2)|1⟩',
      analogy: '🕰 Advancing a clock hand — changes phase without changing state.'
    },
    P: {
      id: 'P', name: 'Phase Gate P(φ)', symbol: 'P', category: 'phase',
      description:
        'Applies an arbitrary phase φ to |1⟩. Leaves |0⟩ unchanged. ' +
        'Generalises Z (φ=180°), S (φ=90°), and T (φ=45°).',
      matrixText: '[ [1, 0      ],\n  [0, e^(iφ) ] ]',
      effect0: '|0⟩  →  |0⟩',
      effect1: '|1⟩  →  e^(iφ)|1⟩',
      analogy: '🎛 A universal phase knob — set any angle to get Z, S, T, or anything between.'
    }
  };

  // ─── Category Colours (CSS-ready hex) ───────────────────────────────────────

  const CATEGORY_COLORS = {
    pauli:    '#f472b6',   // Pink
    clifford: '#38bdf8',   // Cyan
    rotation: '#fbbf24',   // Amber
    phase:    '#34d399',   // Emerald
    special:  '#a78bfa',   // Violet
  };

  // ─── Public API ──────────────────────────────────────────────────────────────

  function getMatrix(gateId, angleDeg = 0) {
    if (STATIC[gateId]) return STATIC[gateId];
    const rad = angleDeg * Math.PI / 180;
    switch (gateId) {
      case 'RX': return rxMatrix(rad);
      case 'RY': return ryMatrix(rad);
      case 'RZ': return rzMatrix(rad);
      case 'P':  return pMatrix(rad);
      default:   throw new Error(`Unknown gate: ${gateId}`);
    }
  }

  function getInfo(gateId) {
    return INFO[gateId] || null;
  }

  return { getMatrix, getInfo, INFO, CATEGORY_COLORS };
})();
