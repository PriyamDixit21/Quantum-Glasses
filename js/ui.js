/**
 * ui.js
 * UI manager — updates DOM elements in response to quantum state changes,
 * shows gate info cards, measurement modal, and toast notifications.
 *
 * Depends on: quantum.js (QM), gates.js (GATES)
 */
'use strict';

class UIManager {

  constructor() {
    // ── Amplitude display ────────────────────────────────────────────────────
    this._alphaDisplay = document.getElementById('alpha-display');
    this._betaDisplay  = document.getElementById('beta-display');
    this._alphaBar     = document.getElementById('alpha-bar');
    this._betaBar      = document.getElementById('beta-bar');

    // ── Probability display ──────────────────────────────────────────────────
    this._prob0Pct = document.getElementById('prob0-pct');
    this._prob1Pct = document.getElementById('prob1-pct');
    this._prob0Bar = document.getElementById('prob0-bar');
    this._prob1Bar = document.getElementById('prob1-bar');

    // ── Ket label ────────────────────────────────────────────────────────────
    this._headerKet = document.getElementById('header-ket');

    // ── Gate info panel ──────────────────────────────────────────────────────
    this._gateInfoBody = document.getElementById('gate-info-body');

    // ── Measure modal ────────────────────────────────────────────────────────
    this._measureModal   = document.getElementById('measure-modal');
    this._measureOutcome = document.getElementById('measure-outcome');
    this._modalProb0     = document.getElementById('modal-prob0');
    this._modalProb1     = document.getElementById('modal-prob1');
    document.getElementById('measure-continue').addEventListener('click', () => {
      this._measureModal.classList.add('hidden');
    });
    // Also close by clicking backdrop
    this._measureModal.addEventListener('click', e => {
      if (e.target === this._measureModal) this._measureModal.classList.add('hidden');
    });

    // ── Toast ────────────────────────────────────────────────────────────────
    this._toast        = document.getElementById('toast');
    this._toastTimer   = null;

    // Show idle state in gate info
    this._showIdleGateInfo();
  }

  // ─── State display ──────────────────────────────────────────────────────────

  /**
   * Update all state-related UI elements.
   * @param {[complex, complex]} state  Normalised qubit state vector
   */
  updateState(state) {
    const [a, b] = state;
    const p0 = QM.prob0(state);
    const p1 = QM.prob1(state);

    // Amplitude values
    this._alphaDisplay.textContent = QM.cformat(a);
    this._betaDisplay.textContent  = QM.cformat(b);

    // Amplitude magnitude bars (reflect probability, not raw amplitude)
    this._alphaBar.style.width = `${(p0 * 100).toFixed(1)}%`;
    this._betaBar.style.width  = `${(p1 * 100).toFixed(1)}%`;

    // Probability bars + percentage text
    const p0pct = (p0 * 100).toFixed(2);
    const p1pct = (p1 * 100).toFixed(2);
    this._prob0Bar.style.width  = `${p0pct}%`;
    this._prob1Bar.style.width  = `${p1pct}%`;
    this._prob0Pct.textContent  = `${p0pct}%`;
    this._prob1Pct.textContent  = `${p1pct}%`;

    // Ket label in header
    this._headerKet.textContent = QM.ketLabel(state);
  }

  // ─── Undo/Redo button states ───────────────────────────────────────────────

  updateUndoRedo(canUndo, canRedo) {
    document.getElementById('btn-undo').disabled = !canUndo;
    document.getElementById('btn-redo').disabled = !canRedo;
  }

  // ─── Gate info panel ───────────────────────────────────────────────────────

  showGateInfo(gateId) {
    const info = GATES.getInfo(gateId);
    if (!info) { this._showIdleGateInfo(); return; }

    const color = GATES.CATEGORY_COLORS[info.category] || '#38bdf8';

    this._gateInfoBody.innerHTML = `
      <div class="gate-info-card">
        <div class="gi-header">
          <span class="gi-symbol" style="color:${color};border-color:color-mix(in srgb,${color} 50%,transparent)">
            ${info.symbol}
          </span>
          <div>
            <div class="gi-name">${info.name}</div>
            <div class="gi-cat" style="color:${color}">${info.category}</div>
          </div>
        </div>

        <p class="gi-desc">${info.description}</p>

        <div class="gi-matrix">
          <div class="gi-matrix-label">Matrix</div>
          <pre class="gi-matrix-code">${info.matrixText}</pre>
        </div>

        <div class="gi-effects">
          <div class="gi-effect">${info.effect0}</div>
          <div class="gi-effect">${info.effect1}</div>
        </div>

        <div class="gi-analogy">
          <span>${info.analogy}</span>
        </div>
      </div>
    `;
  }

  _showIdleGateInfo() {
    this._gateInfoBody.innerHTML = `
      <div class="gi-idle">
        <div class="gi-idle-icon">🔬</div>
        <p>Hover over any gate to explore its quantum properties</p>
      </div>
    `;
  }

  // ─── Measurement modal ─────────────────────────────────────────────────────

  showMeasurementModal(outcome, p0, p1) {
    const label = outcome === 0 ? '|0⟩' : '|1⟩';
    this._measureOutcome.textContent = label;
    this._measureOutcome.className   =
      'measure-outcome-val ' + (outcome === 0 ? 'outcome--zero' : 'outcome--one');
    this._modalProb0.textContent = `${(p0 * 100).toFixed(2)}%`;
    this._modalProb1.textContent = `${(p1 * 100).toFixed(2)}%`;

    this._measureModal.classList.remove('hidden');
    // Re-trigger animation
    void this._measureModal.offsetWidth;
  }

  // ─── Toast notifications ───────────────────────────────────────────────────

  /**
   * Show a brief toast notification.
   * @param {string} message
   * @param {'success'|'info'|'warn'} type
   */
  showToast(message, type = 'info') {
    const t = this._toast;
    t.textContent = message;
    t.className   = `toast toast--${type} toast--visible`;

    if (this._toastTimer) clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => {
      t.classList.remove('toast--visible');
    }, 2600);
  }
}
