/**
 * quantum.js
 * Pure JavaScript quantum computing math engine.
 * No external dependencies — runs entirely in the browser.
 *
 * State vector: [alpha, beta] where each is a complex number {re, im}
 * Invariant:    |alpha|² + |beta|² ≈ 1  (normalized)
 */
'use strict';

const QM = (() => {

  // ─── Complex Number Arithmetic ──────────────────────────────────────────────

  /** Create a complex number */
  function c(re, im = 0) { return { re, im }; }

  function cadd(a, b)  { return c(a.re + b.re, a.im + b.im); }
  function csub(a, b)  { return c(a.re - b.re, a.im - b.im); }
  function cmul(a, b)  {
    return c(
      a.re * b.re - a.im * b.im,
      a.re * b.im + a.im * b.re
    );
  }
  function cconj(a)    { return c(a.re, -a.im); }
  function cabs(a)     { return Math.sqrt(a.re * a.re + a.im * a.im); }
  function cabs2(a)    { return a.re * a.re + a.im * a.im; }
  function cscale(a,s) { return c(a.re * s, a.im * s); }

  /** e^(i·theta) = cos(theta) + i·sin(theta) */
  function cexp(theta) { return c(Math.cos(theta), Math.sin(theta)); }

  /** Format complex number for display */
  function cformat(a, digits = 4) {
    const absIm = Math.abs(a.im);
    const sign  = a.im >= 0 ? '+' : '−';
    return `${a.re.toFixed(digits)} ${sign} ${absIm.toFixed(digits)}i`;
  }

  // ─── 2×2 Complex Matrix ──────────────────────────────────────────────────────
  // Represented as a flat array [m00, m01, m10, m11]  → [[m00,m01],[m10,m11]]

  /** Multiply 2×2 matrix M by column vector v=[v0,v1] */
  function matMulVec(M, v) {
    const [m00, m01, m10, m11] = M;
    const [v0,  v1]            = v;
    return [
      cadd(cmul(m00, v0), cmul(m01, v1)),
      cadd(cmul(m10, v0), cmul(m11, v1))
    ];
  }

  // ─── Quantum State Operations ────────────────────────────────────────────────

  /** Normalize state vector to unit length */
  function normalize(state) {
    const [a, b] = state;
    const norm   = Math.sqrt(cabs2(a) + cabs2(b));
    if (norm < 1e-12) return [c(1, 0), c(0, 0)];
    return [cscale(a, 1 / norm), cscale(b, 1 / norm)];
  }

  /** Apply a 2×2 gate matrix to the qubit state, return normalized new state */
  function applyGate(state, matrix) {
    return normalize(matMulVec(matrix, state));
  }

  /** Probability of measuring |0⟩ */
  function prob0(state) { return cabs2(state[0]); }

  /** Probability of measuring |1⟩ */
  function prob1(state) { return cabs2(state[1]); }

  // ─── Bloch Sphere Coordinates ────────────────────────────────────────────────
  //
  //   x = 2·Re(α*·β)       ← ⟨X⟩
  //   y = 2·Im(α*·β)       ← ⟨Y⟩
  //   z = |α|² − |β|²     ← ⟨Z⟩
  //
  // These satisfy x²+y²+z² ≤ 1  (equality for pure states)

  function toBloch(state) {
    const [a, b] = state;
    const aConj  = cconj(a);
    const ab     = cmul(aConj, b);   // α*·β
    return {
      x: 2 * ab.re,
      y: 2 * ab.im,
      z: cabs2(a) - cabs2(b)
    };
  }

  // ─── Ket Label ───────────────────────────────────────────────────────────────
  //   Recognises the six cardinal Bloch sphere states + generic |ψ⟩

  function ketLabel(state) {
    const [a, b] = state;
    const eps    = 0.001;
    const INV2   = 1 / Math.sqrt(2);

    if (cabs2(a) > 1 - eps)              return '|0⟩';
    if (cabs2(b) > 1 - eps)              return '|1⟩';

    if (Math.abs(cabs(a) - INV2) < eps && Math.abs(cabs(b) - INV2) < eps) {
      // Relative phase between α and β
      const phase = Math.atan2(b.im, b.re) - Math.atan2(a.im, a.re);
      const p     = ((phase % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      if (p < eps || Math.abs(p - 2 * Math.PI) < eps) return '|+⟩';
      if (Math.abs(p - Math.PI)       < 0.05)          return '|−⟩';
      if (Math.abs(p - Math.PI / 2)   < 0.05)          return '|+i⟩';
      if (Math.abs(p - 3 * Math.PI / 2) < 0.05)        return '|−i⟩';
    }
    return '|ψ⟩';
  }

  // ─── Public API ──────────────────────────────────────────────────────────────
  return {
    c, cadd, csub, cmul, cconj, cabs, cabs2, cscale, cexp, cformat,
    matMulVec, normalize, applyGate,
    prob0, prob1, toBloch, ketLabel
  };
})();
