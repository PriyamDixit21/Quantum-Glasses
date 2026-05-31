/**
 * circuit.js
 * Quantum circuit builder — maintains the ordered list of applied gates,
 * renders them as coloured blocks on a wire, and provides undo/redo.
 *
 * Depends on: gates.js (GATES.CATEGORY_COLORS)
 */
'use strict';

class CircuitBuilder {
  /**
   * @param {HTMLElement} container  The element where gate blocks are rendered.
   */
  constructor(container) {
    this.container = container;

    /** @type {Array<{id:string, label:string, angle:number, category:string}>} */
    this.gates = [];

    // Undo/redo history — array of gate-list snapshots
    this._history      = [[]];
    this._historyIndex = 0;

    /**
     * Called after any gate-list change with the new gate array.
     * @type {((gates: Array) => void) | null}
     */
    this.onGatesChanged = null;
  }

  // ─── Public API ─────────────────────────────────────────────────────────────

  /**
   * Append a gate to the circuit.
   * @param {{ id, label, angle, category }} gateData
   */
  addGate(gateData) {
    this.gates = [...this.gates, gateData];
    this._saveSnapshot();
    this._notify();
  }

  /**
   * Remove the gate at `index` and replay.
   * @param {number} index
   */
  removeGate(index) {
    this.gates = this.gates.filter((_, i) => i !== index);
    this._saveSnapshot();
    this._notify();
  }

  /** Remove all gates. */
  clear() {
    if (this.gates.length === 0) return;
    this.gates = [];
    this._saveSnapshot();
    this._notify();
  }

  /** Like clear() but does NOT fire onGatesChanged (used after measurement). */
  clearSilent() {
    this.gates = [];
    this._saveSnapshot();
    this._render();
  }

  undo() {
    if (this._historyIndex > 0) {
      this._historyIndex--;
      this.gates = [...this._history[this._historyIndex]];
      this._notify();
    }
  }

  redo() {
    if (this._historyIndex < this._history.length - 1) {
      this._historyIndex++;
      this.gates = [...this._history[this._historyIndex]];
      this._notify();
    }
  }

  canUndo() { return this._historyIndex > 0; }
  canRedo() { return this._historyIndex < this._history.length - 1; }

  // ─── Private helpers ────────────────────────────────────────────────────────

  _saveSnapshot() {
    // Truncate any redo history
    this._history      = this._history.slice(0, this._historyIndex + 1);
    this._history.push([...this.gates]);
    this._historyIndex = this._history.length - 1;
  }

  _notify() {
    this._render();
    if (this.onGatesChanged) this.onGatesChanged(this.gates);
  }

  _render() {
    this.container.innerHTML = '';
    const colors = GATES.CATEGORY_COLORS;

    this.gates.forEach((gate, index) => {
      const block = document.createElement('div');
      block.className = 'circuit-gate-block';
      block.style.setProperty('--gate-c', colors[gate.category] || '#38bdf8');
      block.title = `${gate.label} — click × to remove`;

      const sym = document.createElement('span');
      sym.className   = 'gate-block-sym';
      sym.textContent = gate.label;

      const rm = document.createElement('button');
      rm.className   = 'gate-block-rm';
      rm.textContent = '×';
      rm.title       = 'Remove this gate';
      rm.setAttribute('aria-label', `Remove ${gate.label}`);
      rm.addEventListener('click', e => {
        e.stopPropagation();
        this.removeGate(index);
      });

      block.appendChild(sym);
      block.appendChild(rm);
      this.container.appendChild(block);
    });
  }
}
