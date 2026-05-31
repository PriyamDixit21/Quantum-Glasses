/**
 * bloch.js
 * Interactive 3-D Bloch sphere renderer — pure HTML5 Canvas, no WebGL.
 *
 * Features
 * --------
 *  • Orthographic projection with mouse / touch drag to rotate view
 *  • Scroll-to-zoom
 *  • Front/back grid lines drawn with different opacity
 *  • Smooth state-vector arrow animation (ease-in-out lerp)
 *  • HiDPI / Retina support via devicePixelRatio
 *
 * Depends on: quantum.js (QM)
 */
'use strict';

class BlochSphere {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');

    // ── View angles (radians) ──────────────────────────────────────────────────
    this.azimuth   = -0.55;   // rotation around vertical (Y) axis
    this.elevation =  0.38;   // tilt around horizontal (X) axis

    // ── Layout (computed in _resize) ──────────────────────────────────────────
    this.scale  = 120;
    this.cx     = 0;
    this.cy     = 0;
    this._cssW  = 0;
    this._cssH  = 0;

    // ── Current & animated Bloch vector ──────────────────────────────────────
    this.bloch      = { x: 0, y: 0, z: 1 };  // |0⟩ = north pole
    this.animFrom   = { x: 0, y: 0, z: 1 };
    this.animTarget = { x: 0, y: 0, z: 1 };
    this.animT      = 1;   // 1 = finished

    // ── Zoom ──────────────────────────────────────────────────────────────────
    this.zoom = 1.0;

    // ── Mouse / touch drag ───────────────────────────────────────────────────
    this.dragging  = false;
    this.lastMouse = { x: 0, y: 0 };

    this._resize();
    this._bindEvents();
    this._loop();
  }

  // ─── Public: update target Bloch vector (triggers animation) ───────────────
  setState(blochVec) {
    this.animFrom   = { ...this.bloch };
    this.animTarget = { ...blochVec };
    this.animT      = 0;
  }

  // ─── Public: jump view to preset angle ────────────────────────────────────
  setView(azimuth, elevation) {
    this.azimuth   = azimuth;
    this.elevation = elevation;
  }

  // ─── Resize handling ───────────────────────────────────────────────────────
  _resize() {
    const parent = this.canvas.parentElement;
    const rect   = parent.getBoundingClientRect();
    const size   = Math.min(rect.width, rect.height, 460) - 8;
    const dpr    = window.devicePixelRatio || 1;

    this.canvas.width        = size * dpr;
    this.canvas.height       = size * dpr;
    this.canvas.style.width  = size + 'px';
    this.canvas.style.height = size + 'px';

    // Absolute transform (does NOT accumulate on repeated calls)
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    this._cssW  = size;
    this._cssH  = size;
    this.cx     = size / 2;
    this.cy     = size / 2;
    this.scale  = size * 0.37 * this.zoom;
  }

  // ─── Mouse / Touch event binding ──────────────────────────────────────────
  _bindEvents() {
    const cvs = this.canvas;

    // Mouse drag
    cvs.addEventListener('mousedown', e => {
      this.dragging  = true;
      this.lastMouse = { x: e.clientX, y: e.clientY };
      cvs.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', e => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastMouse.x;
      const dy = e.clientY - this.lastMouse.y;
      this.azimuth   += dx * 0.012;
      this.elevation  = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.elevation + dy * 0.012));
      this.lastMouse  = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener('mouseup', () => {
      this.dragging    = false;
      cvs.style.cursor = 'grab';
    });

    // Scroll-to-zoom
    cvs.addEventListener('wheel', e => {
      e.preventDefault();
      this.zoom   = Math.max(0.5, Math.min(2.0, this.zoom - e.deltaY * 0.001));
      this.scale  = this._cssW * 0.37 * this.zoom;
    }, { passive: false });

    // Touch drag
    cvs.addEventListener('touchstart', e => {
      this.dragging  = true;
      this.lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }, { passive: true });
    cvs.addEventListener('touchmove', e => {
      if (!this.dragging) return;
      e.preventDefault();
      const dx = e.touches[0].clientX - this.lastMouse.x;
      const dy = e.touches[0].clientY - this.lastMouse.y;
      this.azimuth   += dx * 0.012;
      this.elevation  = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.elevation + dy * 0.012));
      this.lastMouse  = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }, { passive: false });
    cvs.addEventListener('touchend', () => { this.dragging = false; });

    // Resize
    window.addEventListener('resize', () => this._resize());
    cvs.style.cursor = 'grab';
  }

  // ─── 3-D → 2-D projection ─────────────────────────────────────────────────
  _project(x, y, z) {
    const az = this.azimuth, el = this.elevation;
    // Rotate around Y-axis (azimuth)
    const x1 =  Math.cos(az) * x + Math.sin(az) * z;
    const y1  = y;
    const z1  = -Math.sin(az) * x + Math.cos(az) * z;
    // Rotate around X-axis (elevation)
    const x2  = x1;
    const y2  =  Math.cos(el) * y1 - Math.sin(el) * z1;
    const z2  =  Math.sin(el) * y1 + Math.cos(el) * z1;
    return {
      sx:    this.cx + x2 * this.scale,
      sy:    this.cy - y2 * this.scale,
      depth: z2   // positive = front-facing
    };
  }

  // ─── Build N points along a circle in 3-D ─────────────────────────────────
  //   normal: unit vector perpendicular to the circle's plane
  //   center: 3-D centre point
  //   radius: circle radius
  _circlePoints(normal, center, radius, N = 72) {
    const [nx, ny, nz] = normal;
    // Find two orthogonal basis vectors in the plane
    let u;
    if (Math.abs(nx) < 0.9) {
      u = [0, nz, -ny];
    } else {
      u = [nz, 0, -nx];
    }
    const uLen = Math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2);
    u = u.map(v => v / uLen);
    // v = n × u  (cross product)
    const v = [
      ny * u[2] - nz * u[1],
      nz * u[0] - nx * u[2],
      nx * u[1] - ny * u[0]
    ];

    const pts = [];
    for (let i = 0; i <= N; i++) {
      const t = (2 * Math.PI * i) / N;
      pts.push([
        center[0] + radius * (Math.cos(t) * u[0] + Math.sin(t) * v[0]),
        center[1] + radius * (Math.cos(t) * u[1] + Math.sin(t) * v[1]),
        center[2] + radius * (Math.cos(t) * u[2] + Math.sin(t) * v[2])
      ]);
    }
    return pts;
  }

  // ─── Draw a 3-D curve, front/back split ───────────────────────────────────
  _drawCurveSplit(pts3D, frontColor, backColor, lineWidth = 1) {
    const ctx       = this.ctx;
    const projected = pts3D.map(([x, y, z]) => this._project(x, y, z));

    // Back pass — dashed
    ctx.save();
    ctx.strokeStyle = backColor;
    ctx.lineWidth   = lineWidth;
    ctx.setLineDash([3, 6]);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < projected.length - 1; i++) {
      const p1 = projected[i], p2 = projected[i + 1];
      if ((p1.depth + p2.depth) / 2 < 0) {
        if (!started) { ctx.moveTo(p1.sx, p1.sy); started = true; }
        ctx.lineTo(p2.sx, p2.sy);
      } else { started = false; }
    }
    ctx.stroke();

    // Front pass — solid
    ctx.strokeStyle = frontColor;
    ctx.setLineDash([]);
    ctx.beginPath();
    started = false;
    for (let i = 0; i < projected.length - 1; i++) {
      const p1 = projected[i], p2 = projected[i + 1];
      if ((p1.depth + p2.depth) / 2 >= 0) {
        if (!started) { ctx.moveTo(p1.sx, p1.sy); started = true; }
        ctx.lineTo(p2.sx, p2.sy);
      } else { started = false; }
    }
    ctx.stroke();
    ctx.restore();
  }

  // ─── Ease-in-out and linear interp ────────────────────────────────────────
  _ease(t) { return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; }
  _lerp(a, b, t) {
    return {
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
      z: a.z + (b.z - a.z) * t
    };
  }

  // ─── Animation loop ────────────────────────────────────────────────────────
  _loop() {
    const ANIM_DUR = 0.45; // seconds
    let   last     = 0;

    const tick = ts => {
      const dt = Math.min((ts - last) / 1000, 0.1);
      last      = ts;

      if (this.animT < 1) {
        this.animT  = Math.min(1, this.animT + dt / ANIM_DUR);
        this.bloch  = this._lerp(this.animFrom, this.animTarget, this._ease(this.animT));
      }

      this._draw();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // ─── Master draw ──────────────────────────────────────────────────────────
  _draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this._cssW, this._cssH);

    this._drawBackground();
    this._drawGrid();
    this._drawAxes();
    this._drawSphereShell();
    this._drawStateVector();
    this._drawLabels();
  }

  _drawBackground() {
    const { cx, cy, scale, ctx } = this;
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, scale * 1.6);
    grad.addColorStop(0,   'rgba(56, 189, 248, 0.05)');
    grad.addColorStop(0.5, 'rgba(99, 102, 241, 0.03)');
    grad.addColorStop(1,   'transparent');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, scale * 1.6, 0, Math.PI * 2);
    ctx.fill();
  }

  _drawSphereShell() {
    const { ctx, cx, cy, scale } = this;

    // 3-D depth gradient (lighter top-left = simulated light source)
    const grad = ctx.createRadialGradient(
      cx - scale * 0.18, cy - scale * 0.22, scale * 0.04,
      cx, cy, scale
    );
    grad.addColorStop(0,   'rgba(120, 190, 255, 0.10)');
    grad.addColorStop(0.45,'rgba(56,  89,  148, 0.05)');
    grad.addColorStop(1,   'rgba(5,   12,   26, 0.55)');

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, scale, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // Rim glow
    ctx.lineWidth   = 1.5;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
    ctx.stroke();
    ctx.lineWidth   = 5;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
    ctx.stroke();
    ctx.restore();
  }

  _drawGrid() {
    // Latitude circles (z = constant)
    const latZ = [-0.71, -0.38, 0, 0.38, 0.71];
    for (const z0 of latZ) {
      const r    = Math.sqrt(Math.max(0, 1 - z0 * z0));
      const pts  = this._circlePoints([0, 0, 1], [0, 0, z0], r);
      const isEq = Math.abs(z0) < 0.01;
      this._drawCurveSplit(pts,
        isEq ? 'rgba(99,179,237,0.55)' : 'rgba(99,102,241,0.35)',
        isEq ? 'rgba(99,179,237,0.18)' : 'rgba(99,102,241,0.11)',
        isEq ? 1.4 : 0.7
      );
    }

    // Longitude (meridian) great circles
    const phis = [0, Math.PI / 4, Math.PI / 2, 3 * Math.PI / 4];
    for (const phi of phis) {
      const normal = [-Math.sin(phi), Math.cos(phi), 0];
      const pts    = this._circlePoints(normal, [0, 0, 0], 1);
      this._drawCurveSplit(pts,
        'rgba(99,102,241,0.30)',
        'rgba(99,102,241,0.09)',
        0.7
      );
    }
  }

  _drawAxes() {
    // Z-axis (blue) — |0⟩ north, |1⟩ south
    this._drawAxisSegment([0,0,-1.18], [0,0, 1.18], '#38bdf8', '#1a5c82');
    // X-axis (pink)
    this._drawAxisSegment([-1.18,0,0], [1.18,0,0], '#f472b6', '#7a2b56');
    // Y-axis (emerald)
    this._drawAxisSegment([0,-1.18,0], [0, 1.18,0], '#34d399', '#186644');
  }

  _drawAxisSegment(from3D, to3D, frontColor, backColor) {
    const ctx  = this.ctx;
    const pF   = this._project(...from3D);
    const pT   = this._project(...to3D);
    const mid  = (pF.depth + pT.depth) / 2;
    ctx.save();
    ctx.lineWidth   = 1.1;
    ctx.strokeStyle = mid >= 0 ? frontColor : backColor;
    ctx.setLineDash(mid >= 0 ? [] : [4, 6]);
    ctx.beginPath();
    ctx.moveTo(pF.sx, pF.sy);
    ctx.lineTo(pT.sx, pT.sy);
    ctx.stroke();
    ctx.restore();
  }

  _drawStateVector() {
    const ctx   = this.ctx;
    const { cx, cy, bloch } = this;
    const tip   = this._project(bloch.x, bloch.y, bloch.z);

    // Dashed projection onto equatorial plane
    const eq  = this._project(bloch.x, bloch.y, 0);
    ctx.save();
    ctx.strokeStyle = 'rgba(167,139,250,0.22)';
    ctx.lineWidth   = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(tip.sx, tip.sy);
    ctx.lineTo(eq.sx, eq.sy);
    ctx.stroke();

    // Vertical drop from equatorial projection to sphere axis
    const axisEq = this._project(0, 0, 0);
    ctx.beginPath();
    ctx.moveTo(eq.sx, eq.sy);
    ctx.lineTo(axisEq.sx, axisEq.sy);
    ctx.stroke();
    ctx.restore();

    // Glowing arrow stem
    const grad = ctx.createLinearGradient(cx, cy, tip.sx, tip.sy);
    grad.addColorStop(0,   'rgba(167,139,250,0.25)');
    grad.addColorStop(0.6, 'rgba(167,139,250,0.9)');
    grad.addColorStop(1,   '#c4b5fd');

    ctx.save();
    ctx.shadowColor = '#a78bfa';
    ctx.shadowBlur  = 18;
    ctx.strokeStyle = grad;
    ctx.lineWidth   = 2.8;
    ctx.setLineDash([]);
    ctx.lineCap     = 'round';
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(tip.sx, tip.sy);
    ctx.stroke();

    // Arrow head
    const ang  = Math.atan2(tip.sy - cy, tip.sx - cx);
    const hLen = 13;
    ctx.fillStyle = '#c4b5fd';
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.moveTo(tip.sx, tip.sy);
    ctx.lineTo(
      tip.sx - hLen * Math.cos(ang - 0.38),
      tip.sy - hLen * Math.sin(ang - 0.38)
    );
    ctx.lineTo(
      tip.sx - hLen * Math.cos(ang + 0.38),
      tip.sy - hLen * Math.sin(ang + 0.38)
    );
    ctx.closePath();
    ctx.fill();

    // Tip glow dot
    ctx.shadowBlur = 20;
    ctx.shadowColor = '#c4b5fd';
    ctx.fillStyle   = '#ffffff';
    ctx.beginPath();
    ctx.arc(tip.sx, tip.sy, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  _drawLabels() {
    const ctx = this.ctx;
    const labels = [
      { pos: [0,  0,  1.25], text: '|0⟩',  color: '#60cdff' },
      { pos: [0,  0, -1.25], text: '|1⟩',  color: '#60cdff' },
      { pos: [ 1.25, 0, 0],  text: '|+⟩',  color: '#f9a8d4' },
      { pos: [-1.25, 0, 0],  text: '|−⟩',  color: '#f9a8d4' },
      { pos: [0,  1.25, 0],  text: '|+i⟩', color: '#6ee7b7' },
      { pos: [0, -1.25, 0],  text: '|−i⟩', color: '#6ee7b7' },
    ];

    ctx.save();
    ctx.font         = "bold 12.5px 'Space Grotesk', sans-serif";
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    for (const { pos, text, color } of labels) {
      const { sx, sy } = this._project(...pos);
      ctx.fillStyle    = color;
      // Subtle halo for readability
      ctx.shadowColor  = 'rgba(5,12,26,0.9)';
      ctx.shadowBlur   = 6;
      ctx.fillText(text, sx, sy);
    }
    ctx.restore();
  }
}
