"""
GA Mini — a didactic, runnable Python sketch of Generator Architecture on the 12,288 lattice.

Now with a **Hello World** default **and a Transformer‑ish sector enabled by default**.

What you get:

* A 48x256 (12,288) tile and a tiny GA block that minimizes a sectorized action S[psi]
* Sectors: data fidelity (deblur), conservation fairness (C768 means), smoothness (Laplacian), page zero-mean (simple gauge)
* **Transformer‑ish sector (ON by default)** — a fixed, normalized **affinity operator A** (attention‑like, low‑rank per page) with energy `0.5‖(I−A)ψ‖²`
* Receipts (BHIC): R96 histogram/entropy/checksum, C768 means/vars + rotation invariance check, Klein probes (toy), Φ‑roundtrip (toy), β‑ledger
* **Hello World (default)**: zero tile → β=0 under P‑Core (no Klein requirement) → prints a tiny TPT‑lite
* Accept predicate: beta==0 under crush (i.e., each sector residual below tolerance)

Notes:

* This is a pedagogical approximation. It preserves the _shape_ of GA: tile, sectors, action, receipts, β.
* R96 here uses byte%96 over an 8‑bit quantization for clarity; production uses pair‑normalized evaluation with oscillators.
* Φ/NF‑Lift are identity for the demo (round‑trip acceptance shows the hook).
* The Transformer‑ish sector builds a **per‑page low‑rank attention operator** from the measurement `y` (not learned at runtime): rows of `A` are normalized affinities to a small set of **anchor positions**.

Usage: python ga_mini.py # runs self-tests, then Hello World (no flags)
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List

# ------------------------------
# Tile geometry and helpers
# ------------------------------

PAGES, BYTES = 48, 256
N = PAGES * BYTES

Idx = Tuple[int, int]

def lin_index(p: int, b: int) -> int:
    return p * BYTES + b

def residue_classes() -> np.ndarray:
    # Residues for C768 = 3 * 256 along linear index (mod 3)
    idx = np.arange(N)
    return (idx % 3).reshape(PAGES, BYTES)

RESIDUES = residue_classes()

# ------------------------------
# Simple torus convolution via FFT (wrap-around)
# ------------------------------

def fft_conv2_torus(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    H, W = x.shape
    kh, kw = kernel.shape
    
    # Center the kernel at (0,0) in frequency domain (wrap)
    k = np.zeros_like(x)
    k[:kh, :kw] = kernel
    k = np.roll(k, shift=-(kh//2), axis=0)
    k = np.roll(k, shift=-(kw//2), axis=1)
    
    X = np.fft.rfftn(x)
    K = np.fft.rfftn(k)
    Y = X * K
    y = np.fft.irfftn(Y, s=x.shape)
    return np.real(y)

# 5x5 binomial blur kernel (separable [1,4,6,4,1]/16)
_BLUR_1D = np.array([1, 4, 6, 4, 1], dtype=float)
_BLUR_1D = _BLUR_1D / _BLUR_1D.sum()
_BLUR_2D = np.outer(_BLUR_1D, _BLUR_1D)

# ------------------------------
# Receipts: R96, C768, Klein, Φ (toy), β ledger
# ------------------------------

def quantize_u8(x: np.ndarray) -> np.ndarray:
    xnorm = x - x.min()
    rng = xnorm.max() or 1.0
    q = np.clip((255.0 * xnorm / rng).round(), 0, 255).astype(np.uint8)
    return q

def r96_receipt(x: np.ndarray) -> Dict[str, Any]:
    q = quantize_u8(x)
    counts = np.bincount(q.flatten() % 96, minlength=96)
    entropy = -np.sum(counts * np.log(counts + 1e-10)) / N
    checksum = int(np.sum(q)) % (2**32)
    return {"counts": counts, "entropy": entropy, "checksum": checksum}

def c768_receipt(x: np.ndarray) -> Dict[str, Any]:
    means = []
    vars_ = []
    for r in range(3):
        mask = (RESIDUES == r)
        if mask.any():
            vals = x[mask]
            means.append(float(np.mean(vals)))
            vars_.append(float(np.var(vals)))
        else:
            means.append(0.0)
            vars_.append(0.0)
    
    # Rotation invariance check (toy): compare page 0 vs page 1 means
    rotation_invariant = abs(means[0] - means[1]) < 0.1
    
    return {"means": means, "vars": vars_, "rotationInvariant": rotation_invariant}

def klein_probes(x: np.ndarray) -> np.ndarray:
    # Toy Klein probes: 192 bits from random sampling
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    probes = np.zeros(192, dtype=bool)
    for i in range(192):
        p = rng.integers(0, PAGES)
        b = rng.integers(0, BYTES)
        probes[i] = x[p, b] > 0.0
    return probes

def phi_roundtrip_accept(x: np.ndarray, tol: float = 0.0) -> bool:
    # Toy Φ: identity round-trip should always accept
    return True

# ------------------------------
# Sectors: data fidelity, conservation, smoothness, gauge, attention
# ------------------------------

def blur_op(x: np.ndarray) -> np.ndarray:
    return fft_conv2_torus(x, _BLUR_2D)

def data_fidelity_grad(psi: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    # L2 data fidelity: 0.5 * ||blur(psi) - y||^2
    y_pred = blur_op(psi)
    residual = y_pred - y
    loss = 0.5 * np.sum(residual**2)
    
    # Gradient via adjoint blur
    grad = fft_conv2_torus(residual, _BLUR_2D)
    return loss, grad

def conservation_grad(psi: np.ndarray) -> Tuple[float, np.ndarray]:
    # C768 conservation: 0.5 * sum_r (mean_r - target)^2
    target = 0.0
    loss = 0.0
    grad = np.zeros_like(psi)
    
    for r in range(3):
        mask = (RESIDUES == r)
        if mask.any():
            mean_r = np.mean(psi[mask])
            diff = mean_r - target
            loss += 0.5 * diff**2
            grad[mask] += diff / np.sum(mask)
    
    return loss, grad

def smooth_grad(psi: np.ndarray) -> Tuple[float, np.ndarray]:
    # Laplacian smoothness: 0.5 * ||L psi||^2
    # Simple 5-point stencil
    L = np.zeros_like(psi)
    L[1:-1, 1:-1] = 4*psi[1:-1, 1:-1] - psi[2:, 1:-1] - psi[:-2, 1:-1] - psi[1:-1, 2:] - psi[1:-1, :-2]
    
    loss = 0.5 * np.sum(L**2)
    grad = L  # Self-adjoint
    return loss, grad

def gauge_grad(psi: np.ndarray) -> Tuple[float, np.ndarray]:
    # Page zero-mean gauge: 0.5 * sum_p (mean_p)^2
    page_means = np.mean(psi, axis=1)
    loss = 0.5 * np.sum(page_means**2)
    
    grad = np.zeros_like(psi)
    for p in range(PAGES):
        grad[p, :] = page_means[p] / BYTES
    
    return loss, grad

def attn_loss_grad(psi: np.ndarray, attn_ctx: Dict[str, Any]) -> Tuple[float, np.ndarray]:
    # Transformer-like attention: 0.5 * ||(I - A) psi||^2
    A = attn_ctx["A"]
    I_minus_A = np.eye(A.shape[0]) - A
    
    # Apply per page
    loss = 0.0
    grad = np.zeros_like(psi)
    
    for p in range(PAGES):
        psi_p = psi[p, :]
        residual = I_minus_A @ psi_p
        loss += 0.5 * np.sum(residual**2)
        grad[p, :] = I_minus_A.T @ residual
    
    return loss, grad

def build_attention_context(y: np.ndarray) -> Dict[str, Any]:
    # Build per-page low-rank attention operator from measurement y
    # Anchor positions: every 32nd byte
    anchors = np.arange(0, BYTES, 32)
    n_anchors = len(anchors)
    
    A = np.zeros((BYTES, BYTES))
    
    for p in range(PAGES):
        # Get measurement values at anchor positions
        anchor_vals = y[p, anchors]
        
        # Build affinity matrix (normalized)
        for i in range(BYTES):
            for j, anchor_idx in enumerate(anchors):
                # Simple affinity based on distance and value similarity
                dist = abs(i - anchor_idx) / BYTES
                val_sim = 1.0 / (1.0 + abs(y[p, i] - anchor_vals[j]))
                A[i, anchor_idx] = val_sim * np.exp(-dist)
        
        # Normalize rows
        row_sums = A.sum(axis=1, keepdims=True)
        A = A / (row_sums + 1e-10)
    
    return {"A": A}

# ------------------------------
# Configuration and main GA block
# ------------------------------

@dataclass
class Sectors:
    lambda_data: float = 1.0
    lambda_cons: float = 0.2
    lambda_smooth: float = 0.15
    lambda_gauge: float = 0.05
    lambda_attn: float = 0.4

@dataclass
class GAConfig:
    steps: int = 100
    step_size: float = 0.1
    sectors: Sectors = field(default_factory=Sectors)

@dataclass
class BHIC:
    r96: Dict[str, Any]
    c768: Dict[str, Any]
    klein: np.ndarray
    phi_accept: bool
    beta: Dict[str, float]
    
    def accept(self, eps: float = 1e-3, require_klein: bool = True) -> bool:
        # Accept if all beta components below tolerance
        beta_ok = all(abs(v) <= eps for v in self.beta.values())
        
        # Klein requirement (optional)
        klein_ok = not require_klein or self.klein.all()
        
        return beta_ok and klein_ok and self.phi_accept

def ga_block(y: np.ndarray, cfg: GAConfig) -> Tuple[np.ndarray, BHIC]:
    # Initialize
    psi = np.copy(y)
    
    # Build attention context if enabled
    attn_ctx = None
    if cfg.sectors.lambda_attn > 0:
        attn_ctx = build_attention_context(y)
    
    # Gradient descent
    for step in range(cfg.steps):
        L = 0.0
        G = np.zeros_like(psi)
        
        # Data fidelity
        l_data, g_data = data_fidelity_grad(psi, y)
        L += cfg.sectors.lambda_data * l_data
        G += cfg.sectors.lambda_data * g_data
        
        # Conservation
        l_cons, g_cons = conservation_grad(psi)
        L += cfg.sectors.lambda_cons * l_cons
        G += cfg.sectors.lambda_cons * g_cons
        
        # Smoothness
        l_smooth, g_smooth = smooth_grad(psi)
        L += cfg.sectors.lambda_smooth * l_smooth
        G += cfg.sectors.lambda_smooth * g_smooth
        
        # Gauge
        l_gauge, g_gauge = gauge_grad(psi)
        L += cfg.sectors.lambda_gauge * l_gauge
        G += cfg.sectors.lambda_gauge * g_gauge
        
        # Attention
        if attn_ctx is not None:
            l_attn, g_attn = attn_loss_grad(psi, attn_ctx)
            L += cfg.sectors.lambda_attn * l_attn
            G += cfg.sectors.lambda_attn * g_attn
        
        psi -= cfg.step_size * G  # gradient descent step
    
    # Project (toy): identity
    x_out = psi
    
    # Receipts
    r96 = r96_receipt(x_out)
    c768 = c768_receipt(x_out)
    klein = klein_probes(x_out)
    phi_ok = phi_roundtrip_accept(x_out, tol=0.0)
    
    # β-ledger: sector residuals at solution
    l_data, _ = data_fidelity_grad(x_out, y)
    l_cons, _ = conservation_grad(x_out)
    l_smooth, _ = smooth_grad(x_out)
    l_gauge, _ = gauge_grad(x_out)
    beta = {
        "data": float(l_data),
        "cons": float(l_cons),
        "smooth": float(l_smooth),
        "gauge": float(l_gauge),
    }
    if attn_ctx is not None:
        l_attn, _ = attn_loss_grad(x_out, attn_ctx)
        beta["attn"] = float(l_attn)
    
    return x_out, BHIC(r96=r96, c768=c768, klein=klein, phi_accept=phi_ok, beta=beta)

# ------------------------------
# Hello World (deterministic accept under P-Core semantics)
# ------------------------------

def hello_world() -> None:
    """Minimal, deterministic demo: zero tile ⇒ β=0; accept ignoring Klein.
    Prints a tiny TPT-lite and exits 0.
    """
    y = np.zeros((PAGES, BYTES), dtype=float)
    x_hat, bhic = ga_block(y, GAConfig(steps=1, step_size=0.7))
    accepted = bhic.accept(eps=0.0, require_klein=False)  # P-Core: no Klein requirement
    
    tpt_lite = {
        "tile": f"{PAGES}x{BYTES}",
        "profile": "P-Core(hello)",
        "beta": bhic.beta,
        "c768": {"rotationInvariant": bhic.c768["rotationInvariant"]},
        "phi": {"accept": bhic.phi_accept},
        "accepted": accepted,
    }
    
    print("Hello, GA (12,288)! 👋")
    print("TPT-lite:")
    print(json.dumps(tpt_lite, indent=2))

# ------------------------------
# Demo: Deblur + Verify on the 12,288 tile (kept for developers)
# ------------------------------

def demo(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    
    # Ground truth tile (synthetic): few blobs + stripes
    truth = np.zeros((PAGES, BYTES), dtype=float)
    for p in range(0, PAGES, 6):
        truth[p, :] += np.sin(np.linspace(0, 4*np.pi, BYTES))
    
    for _ in range(120):
        i = rng.integers(0, PAGES)
        j = rng.integers(0, BYTES)
        truth[i, j] += rng.uniform(1.0, 3.0)
    
    truth = fft_conv2_torus(truth, np.ones((3,3))/9.0)  # smooth blobs
    
    # Measurement: blur + noise
    y = blur_op(truth)
    y += 0.02 * rng.standard_normal(size=y.shape)
    
    cfg = GAConfig(steps=250, step_size=0.7, sectors=Sectors(
        lambda_data=1.0, lambda_cons=0.2, lambda_smooth=0.15, lambda_gauge=0.05, lambda_attn=0.4
    ))
    
    x_hat, bhic = ga_block(y, cfg)
    
    # Acceptance under crush (toy): beta components below tol and receipts ok
    eps = 1e-3
    accepted = bhic.accept(eps)
    
    print("=== GA Mini Demo (12,288 tile) ===")
    print(f"Tile: {PAGES}x{BYTES}; Steps: {cfg.steps}; Step size: {cfg.step_size}")
    print("Beta ledger:")
    for k, v in bhic.beta.items():
        print(f"  {k:7s}: {v:.6f}")
    print(f"Receipts: rotationInvariant={bhic.c768['rotationInvariant']} klein_all={bhic.klein.all()} phi={bhic.phi_accept}")
    print(f"R96 entropy: {bhic.r96['entropy']:.3f}  checksum: {bhic.r96['checksum']}")
    print(f"ACCEPT (beta<=eps & receipts): {accepted}")
    
    # Minimal Sufficient Witness (MSW) sketch
    msw = {
        "r96": {"counts": bhic.r96["counts"].tolist(), "entropy": bhic.r96["entropy"]},
        "c768": {"means": bhic.c768["means"], "vars": bhic.c768["vars"], "rotationInvariant": bhic.c768["rotationInvariant"]},
        "klein": bhic.klein.astype(int).tolist(),
        "phi": {"accept": bhic.phi_accept},
        "beta": bhic.beta,
        "meta": {"profile": "P-Full", "tile": f"{PAGES}x{BYTES}", "tol": eps}
    }
    print("Witness fields (keys):", list(msw.keys()))

# ------------------------------
# Self-tests (quick, no heavy optimization)
# ------------------------------

def run_tests() -> None:
    print("=== Running GA Mini self-tests ===")
    
    # Test 1: GAConfig should not raise due to mutable default
    cfg1 = GAConfig()
    cfg2 = GAConfig()
    assert cfg1.sectors is not cfg2.sectors, "GAConfig.sectors should use default_factory to avoid shared state"
    
    # Test 2: receipts shapes and invariants on random field
    x = np.random.default_rng(0).standard_normal((PAGES, BYTES))
    r96 = r96_receipt(x)
    assert r96["counts"].shape == (96,), "R96 counts must have length 96"
    assert int(r96["counts"].sum()) == N, "R96 counts must sum to 12,288"
    
    c768 = c768_receipt(x)
    assert len(c768["means"]) == 3 and len(c768["vars"]) == 3, "C768 must have 3 means/vars"
    
    klein = klein_probes(x)
    assert klein.shape == (192,), "Klein probes must be 192 bits"
    assert phi_roundtrip_accept(x), "Φ round-trip should accept identity"
    
    # Test 3: GA block runs with tiny steps and returns BHIC (attn enabled by default)
    y = np.random.default_rng(1).standard_normal((PAGES, BYTES))
    x_out, bhic = ga_block(y, GAConfig(steps=2, step_size=0.5))
    assert x_out.shape == (PAGES, BYTES), "ga_block output shape must be 48x256"
    assert "attn" in bhic.beta and bhic.beta["attn"] >= 0.0, "β should include attention term by default"
    
    # Test 4: Hello World acceptance should pass under P-Core semantics
    y0 = np.zeros((PAGES, BYTES), dtype=float)
    _, bhic0 = ga_block(y0, GAConfig(steps=1, step_size=0.7))
    assert bhic0.accept(0.0, require_klein=False), "Hello World should accept with require_klein=False"
    assert not bhic0.accept(0.0, require_klein=True), "Hello World should not require Klein in P-Core"
    
    # Test 5: Attention sector wiring (shape + β["attn"] present when enabled explicitly)
    cfg_attn = GAConfig(steps=2, step_size=0.4, sectors=Sectors(
        lambda_data=0.0, lambda_cons=0.0, lambda_smooth=0.0, lambda_gauge=0.0, lambda_attn=0.7
    ))
    x_out2, bhic2 = ga_block(y, cfg_attn)
    assert "attn" in bhic2.beta and bhic2.beta["attn"] >= 0.0, "Attention sector β missing or invalid"
    
    # Test 6: Disabling attention removes the attn β entry
    cfg_no_attn = GAConfig(steps=2, step_size=0.4, sectors=Sectors(
        lambda_attn=0.0, lambda_data=0.0, lambda_cons=0.0, lambda_smooth=0.0, lambda_gauge=0.0
    ))
    _, bhic3 = ga_block(y, cfg_no_attn)
    assert "attn" not in bhic3.beta, "β should not include attention when lambda_attn=0"
    
    print("All self-tests passed.\n")

if __name__ == "__main__":
    # Run quick tests first so users see a green check quickly
    run_tests()
    
    # Always run Hello World by default (no flags)
    hello_world()
