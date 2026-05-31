/**
 * app.js
 * Main application coordinator.
 * Wires together: quantum state ↔ BlochSphere ↔ CircuitBuilder ↔ UIManager.
 *
 * Depends on: quantum.js, gates.js, bloch.js, circuit.js, ui.js
 */
'use strict';

(function () {

  // ─── State ───────────────────────────────────────────────────────────────────
  /** Current qubit state: [alpha, beta] (complex pair). Always normalised. */
  let qState = [QM.c(1, 0), QM.c(0, 0)];   // |0⟩

  // ─── Initialise components ───────────────────────────────────────────────────
  const bloch   = new BlochSphere(document.getElementById('bloch-canvas'));
  const circuit = new CircuitBuilder(document.getElementById('circuit-gates'));
  const ui      = new UIManager();

  // ─── Circuit → state replay ──────────────────────────────────────────────────
  /**
   * Recompute qState by replaying all gates from |0⟩.
   * Called whenever the circuit gate list changes.
   */
  function replayState(gates) {
    qState = [QM.c(1, 0), QM.c(0, 0)];
    for (const g of gates) {
      const mat = GATES.getMatrix(g.id, g.angle || 0);
      qState    = QM.applyGate(qState, mat);
    }
    bloch.setState(QM.toBloch(qState));
    ui.updateState(qState);
    ui.updateUndoRedo(circuit.canUndo(), circuit.canRedo());
  }

  // Wire the circuit's change callback
  circuit.onGatesChanged = replayState;

  // ─── Initial render ──────────────────────────────────────────────────────────
  ui.updateState(qState);
  bloch.setState(QM.toBloch(qState));
  ui.updateUndoRedo(false, false);
  ui.showGateInfo('H');   // show Hadamard info on load

  // ─── Apply a gate ────────────────────────────────────────────────────────────
  function applyGate(gateId, angleDeg = 0) {
    const info  = GATES.getInfo(gateId);
    let   label = info ? info.symbol : gateId;

    if (['RX', 'RY', 'RZ'].includes(gateId)) {
      label += `(${angleDeg}°)`;
    } else if (gateId === 'P') {
      label += `(${angleDeg}°)`;
    }

    circuit.addGate({
      id:       gateId,
      label,
      angle:    angleDeg,
      category: info ? info.category : 'clifford'
    });

    ui.showToast(`Applied ${label}`, 'success');
    _flashBtn(gateId);
  }

  function _flashBtn(gateId) {
    const btn = document.querySelector(`[data-gate="${gateId}"]`);
    if (!btn) return;
    btn.classList.add('gate-btn--flash');
    setTimeout(() => btn.classList.remove('gate-btn--flash'), 450);
  }

  // ─── Gate buttons ─────────────────────────────────────────────────────────────
  document.querySelectorAll('.gate-btn').forEach(btn => {
    const gateId = btn.dataset.gate;

    // Hover → show gate info in side panel
    btn.addEventListener('mouseenter', () => ui.showGateInfo(gateId));

    // Click → apply gate
    btn.addEventListener('click', () => {
      let angle = 0;
      if (['RX', 'RY', 'RZ'].includes(gateId)) {
        angle = parseFloat(document.getElementById('rot-angle').value) || 90;
      } else if (gateId === 'P') {
        angle = parseFloat(document.getElementById('phase-angle').value) || 45;
      }
      applyGate(gateId, angle);
    });
  });

  // ─── Undo / Redo ──────────────────────────────────────────────────────────────
  document.getElementById('btn-undo').addEventListener('click', () => {
    circuit.undo();
    ui.showToast('Undid last gate', 'info');
  });

  document.getElementById('btn-redo').addEventListener('click', () => {
    circuit.redo();
    ui.showToast('Redid gate', 'info');
  });

  // ─── Reset ────────────────────────────────────────────────────────────────────
  document.getElementById('btn-reset').addEventListener('click', () => {
    circuit.clear();          // fires onGatesChanged → replayState([]) → |0⟩
    ui.showToast('Reset to |0⟩', 'info');
  });

  // ─── Clear circuit ────────────────────────────────────────────────────────────
  document.getElementById('btn-clear').addEventListener('click', () => {
    circuit.clear();
    ui.showToast('Circuit cleared', 'info');
  });

  // ─── Measure ──────────────────────────────────────────────────────────────────
  document.getElementById('btn-measure').addEventListener('click', () => {
    const p0      = QM.prob0(qState);
    const p1      = QM.prob1(qState);
    const outcome = Math.random() < p0 ? 0 : 1;

    // Collapse to basis state
    const collapsed = outcome === 0
      ? [QM.c(1, 0), QM.c(0, 0)]
      : [QM.c(0, 0), QM.c(1, 0)];

    // Show modal first (before altering state)
    ui.showMeasurementModal(outcome, p0, p1);

    // Clear circuit silently (don't retrigger replay with wrong state)
    circuit.clearSilent();

    // Update manually
    qState = collapsed;
    bloch.setState(QM.toBloch(qState));
    ui.updateState(qState);
    ui.updateUndoRedo(false, false);
  });

  // ─── View presets ─────────────────────────────────────────────────────────────
  document.querySelectorAll('.view-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      bloch.setView(
        parseFloat(btn.dataset.az  || '-0.55'),
        parseFloat(btn.dataset.el  ||  '0.38')
      );
    });
  });

  // ─── Keyboard shortcuts ───────────────────────────────────────────────────────
  const SHORTCUTS = { x:'X', y:'Y', z:'Z', h:'H', s:'S', t:'T' };

  document.addEventListener('keydown', e => {
    if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;

    // Ctrl/Cmd + Z  →  undo
    if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key === 'z') {
      e.preventDefault();
      circuit.undo();
      ui.showToast('Undid last gate', 'info');
      return;
    }
    // Ctrl/Cmd + Shift + Z  or  Ctrl/Cmd + Y  →  redo
    if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.shiftKey && e.key === 'z'))) {
      e.preventDefault();
      circuit.redo();
      ui.showToast('Redid gate', 'info');
      return;
    }
    // Single-key gate shortcuts (no modifiers)
    if (!e.ctrlKey && !e.metaKey && !e.altKey) {
      const gateId = SHORTCUTS[e.key.toLowerCase()];
      if (gateId) applyGate(gateId);
    }
  });

  console.log('⚛  Quantum Glasses — ready.');
})();
