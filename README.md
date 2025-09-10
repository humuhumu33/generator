"""
GA Mini — a didactic, runnable Python sketch of Generator Architecture on the 12,288 lattice.

Now with a **Hello World** default **and a Transformer‑ish sector enabled by default**.

What you get:
- A 48x256 (12,288) tile and a tiny GA block that minimizes a sectorized action S[psi]
- Sectors: data fidelity (deblur), conservation fairness (C768 means), smoothness (Laplacian), page zero-mean (simple gauge)
- **Transformer‑ish sector (ON by default)** — a fixed, normalized **affinity operator A** (attention‑like, low‑rank per page) with energy `0.5‖(I−A)ψ‖²`
- Receipts (BHIC): R96 histogram/entropy/checksum, C768 means/vars + rotation invariance check, Klein probes (toy), Φ‑roundtrip (toy), β‑ledger
- **Hello World (default)**: zero tile → β=0 under P‑Core (no Klein requirement) → prints a tiny TPT‑lite
- Accept predicate: beta==0 under crush (i.e., each sector residual below tolerance)

Notes:
- This is a pedagogical approximation. It preserves the *shape* of GA: tile, sectors, action, receipts, β.
- R96 here uses byte%96 over an 8‑bit quantization for clarity; production uses pair‑normalized evaluation with oscillators.
- Φ/NF‑Lift are identity for the demo (round‑trip acceptance shows the hook).
- The Transformer‑ish sector builds a **per‑page low‑rank attention operator** from the measurement `y` (not learned at runtime): rows of `A` are normalized affinities to a small set of **anchor positions**.

Usage:
  python ga_mini.py        # runs self-tests, then Hello World (no flags)
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
    classes = (q % 96).astype(np.int32)
    counts = np.bincount(classes.ravel(), minlength=96).astype(np.int64)
    probs = counts / counts.sum() if counts.sum() else np.zeros_like(counts, dtype=float)
    eps = 1e-12
    entropy = float(-(probs * np.log2(probs + eps)).sum())
    checksum = int(counts.sum() % (2**32))  # toy checksum
    return {"counts": counts, "entropy": entropy, "checksum": checksum}


def c768_receipt(x: np.ndarray) -> Dict[str, Any]:
    # Means/vars per residue (mod 3 along linear index)
    means, vars_ = [], []
    for r in (0, 1, 2):
        mask = (RESIDUES == r)
        vals = x[mask]
        means.append(float(vals.mean()))
        vars_.append(float(vals.var()))
    # Rotation invariance proxy: invariance under byte rotation by 1
    rot = np.roll(x, shift=1, axis=1)
    rot_ok = np.allclose(sorted(means), sorted([rot[RESIDUES == r].mean() for r in (0,1,2)]), atol=1e-6)
    return {"means": means, "vars": vars_, "rotationInvariant": bool(rot_ok)}


def klein_probes(x: np.ndarray, n_bits: int = 192) -> np.ndarray:
    # Toy parity probes on repeated 2x2 windows (V4 spirit)
    # Build bits by hashing local 2x2 sums modulo 2 across positions
    bits = np.zeros(n_bits, dtype=bool)
    H, W = x.shape
    step_h = max(1, H // (n_bits // 4))
    step_w = max(1, W // 4)
    k = 0
    for i in range(0, H-1, step_h):
        for j in range(0, W-1, step_w):
            s = x[i:i+2, j:j+2].sum()
            bits[k % n_bits] = (int(abs(s)) & 1) == 1
            k += 1
            if k >= n_bits:
                return bits
    return bits


def phi_roundtrip_accept(x: np.ndarray, tol: float = 0.0) -> bool:
    # Toy: NF-Lift + projection are identity; round-trip diff == 0 implies accept
    rt = x  # identity
    return bool(np.allclose(rt, x, atol=tol))

@dataclass
class BHIC:
    r96: Dict[str, Any]
    c768: Dict[str, Any]
    klein: np.ndarray  # 192 bits
    phi_accept: bool
    beta: Dict[str, float]  # sector residuals

    def accept(self, eps: float = 1e-6, *, require_klein: bool = True) -> bool:
        """Acceptance under crush: β components <= eps and required receipts pass.
        If require_klein=False (P-Core/hello), Klein bits are not part of the predicate.
        """
        beta_zero = all(v <= eps for v in self.beta.values())
        base = self.c768["rotationInvariant"] and self.phi_accept
        return bool(beta_zero and (base and (self.klein.all() if require_klein else True)))

# ------------------------------
# Sectors and Action
# ------------------------------
@dataclass
class Sectors:
    lambda_data: float = 1.0
    lambda_cons: float = 0.1
    lambda_smooth: float = 0.1
    lambda_gauge: float = 0.05
    # Transformer-ish (attention-like) coupling (ENABLED BY DEFAULT)
    lambda_attn: float = 0.4

@dataclass
class AttnParams:
    m: int = 32         # anchors per page (low-rank)
    sigma: float = 0.5  # feature (value) bandwidth
    gamma: float = 16.0 # positional bandwidth (bytes)

@dataclass
class GAConfig:
    steps: int = 200
    step_size: float = 0.8  # tuned for this toy
    tol_beta: float = 1e-6
    sectors: Sectors = field(default_factory=Sectors)        # avoid mutable default
    attn: AttnParams = field(default_factory=AttnParams)     # params for attention sector

# ------------------------------
# Transformer-ish affinity operator A (low-rank per page)
# ------------------------------
# We build A from measurement y (fixed during a block): for each page, choose m anchor
# positions and compute normalized affinities combining *content* similarity and *positional* proximity.
# Then A acts as: (Aψ)_j = Σ_k W[j,k] · ψ[anchor_k], where rows of W sum to 1.

AttnCtx = List[Dict[str, Any]]  # per-page: {"anchors": np.ndarray[int], "W": np.ndarray[bytes x m]}


def build_attn_context(y: np.ndarray, params: AttnParams) -> AttnCtx:
    ctx: AttnCtx = []
    anchors = np.linspace(0, BYTES-1, params.m, dtype=int)
    jj = np.arange(BYTES).reshape(-1, 1)  # column for broadcasting
    for p in range(PAGES):
        ypage = y[p, :]
        ya = ypage[anchors].reshape(1, -1)  # [1, m]
        # Positional proximity
        pos = np.exp(-((jj - anchors.reshape(1, -1)) ** 2) / (2.0 * params.gamma ** 2))  # [256, m]
        # Content (value) similarity
        val = np.exp(-((ypage.reshape(-1, 1) - ya) ** 2) / (2.0 * params.sigma ** 2))      # [256, m]
        W = pos * val
        # Normalize rows to sum 1 (fallback to uniform if a row is all zeros)
        row_sum = W.sum(axis=1, keepdims=True)
        zero_rows = (row_sum == 0)
        if np.any(zero_rows):
            W[zero_rows[:, 0], :] = 1.0
            row_sum = W.sum(axis=1, keepdims=True)
        W = W / row_sum
        ctx.append({"anchors": anchors.copy(), "W": W.astype(float)})
    return ctx


def attn_apply_M(psi: np.ndarray, ctx: AttnCtx) -> np.ndarray:
    out = np.zeros_like(psi)
    for p in range(PAGES):
        W = ctx[p]["W"]                    # [256, m]
        a = ctx[p]["anchors"]              # [m]
        v = psi[p, a]                       # [m]
        out[p, :] = W @ v                   # [256]
    return out


def attn_apply_MT(u: np.ndarray, ctx: AttnCtx) -> np.ndarray:
    # Apply transpose of A: M^T u = S (W^T u)
    out = np.zeros_like(u)
    for p in range(PAGES):
        W = ctx[p]["W"]                    # [256, m]
        a = ctx[p]["anchors"]              # [m]
        coeffs = W.T @ u[p, :]              # [m]
        out[p, a] += coeffs                 # scatter to anchor positions
    return out


def attn_loss_grad(psi: np.ndarray, ctx: AttnCtx) -> Tuple[float, np.ndarray]:
    # E_attn = 0.5 * || (I - A) psi ||^2; grad = (I - A)^T ((I - A) psi) = r - A^T r
    Mpsi = attn_apply_M(psi, ctx)
    r = psi - Mpsi
    loss = 0.5 * float(np.vdot(r, r).real)
    grad = r - attn_apply_MT(r, ctx)
    return loss, grad

# ------------------------------
# Gradients of other sector terms
# ------------------------------

def blur_op(x: np.ndarray) -> np.ndarray:
    return fft_conv2_torus(x, _BLUR_2D)


def data_fidelity_grad(psi: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    # 0.5 ||H psi - y||^2; grad = H^T(H psi - y) = H(H psi - y)
    r = blur_op(psi) - y
    grad = blur_op(r)
    loss = 0.5 * float(np.vdot(r, r).real)
    return loss, grad


def smooth_grad(psi: np.ndarray) -> Tuple[float, np.ndarray]:
    # 0.5 * ||∇psi||^2 via 2D Laplacian (torus)
    lap = (
        np.roll(psi, 1, 0) + np.roll(psi, -1, 0) +
        np.roll(psi, 1, 1) + np.roll(psi, -1, 1) - 4 * psi
    )
    # Energy ~ 0.5 * <psi, -lap>
    loss = 0.5 * float(np.vdot(psi, -lap).real)
    grad = -lap
    return loss, grad


def conservation_grad(psi: np.ndarray) -> Tuple[float, np.ndarray]:
    # Encourage equal means across residues r=0,1,2
    means = [psi[RESIDUES == r].mean() for r in (0,1,2)]
    mu = float(np.mean(means))
    loss = sum((float(m) - mu) ** 2 for m in means)
    # Distribute gradient: d/dx of (m_r - mu)^2 ≈ 2*(m_r - mu)/|R_r|
    grad = np.zeros_like(psi)
    for r in (0,1,2):
        mask = (RESIDUES == r)
        dr = 2.0 * (float(means[r]) - mu) / mask.sum()
        grad[mask] += dr
    # Centering term for mu cancels when summing over residues (approximation)
    return float(loss), grad


def gauge_grad(psi: np.ndarray) -> Tuple[float, np.ndarray]:
    # Simple page-level zero-mean gauge: penalize per-page mean^2
    loss = 0.0
    grad = np.zeros_like(psi)
    for p in range(PAGES):
        page = psi[p, :]
        m = float(page.mean())
        loss += m * m
        grad[p, :] += 2.0 * m / BYTES
    return float(loss), grad

# ------------------------------
# GA Block
# ------------------------------

def ga_block(y: np.ndarray, cfg: GAConfig) -> Tuple[np.ndarray, BHIC]:
    psi = y.copy()  # NF-Lift (toy): start from measured boundary
    s = cfg.sectors

    # Precompute attention context if used
    attn_ctx: AttnCtx | None = None
    if s.lambda_attn > 0.0:
        attn_ctx = build_attn_context(y, cfg.attn)

    for _ in range(cfg.steps):
        L, G = 0.0, np.zeros_like(psi)

        l_data, g_data = data_fidelity_grad(psi, y)
        L += s.lambda_data * l_data
        G += s.lambda_data * g_data

        l_cons, g_cons = conservation_grad(psi)
        L += s.lambda_cons * l_cons
        G += s.lambda_cons * g_cons

        l_smooth, g_smooth = smooth_grad(psi)
        L += s.lambda_smooth * l_smooth
        G += s.lambda_smooth * g_smooth

        l_gauge, g_gauge = gauge_grad(psi)
        L += s.lambda_gauge * l_gauge
        G += s.lambda_gauge * g_gauge

        if attn_ctx is not None:
            l_attn, g_attn = attn_loss_grad(psi, attn_ctx)
            L += s.lambda_attn * l_attn
            G += s.lambda_attn * g_attn

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
    cfg_attn = GAConfig(steps=2, step_size=0.4, sectors=Sectors(lambda_data=0.0, lambda_cons=0.0,
                                                               lambda_smooth=0.0, lambda_gauge=0.0,
                                                               lambda_attn=0.7))
    x_out2, bhic2 = ga_block(y, cfg_attn)
    assert "attn" in bhic2.beta and bhic2.beta["attn"] >= 0.0, "Attention sector β missing or invalid"

    # Test 6: Disabling attention removes the attn β entry
    cfg_no_attn = GAConfig(steps=2, step_size=0.4, sectors=Sectors(lambda_attn=0.0, lambda_data=0.0,
                                                                  lambda_cons=0.0, lambda_smooth=0.0,
                                                                  lambda_gauge=0.0))
    _, bhic3 = ga_block(y, cfg_no_attn)
    assert "attn" not in bhic3.beta, "β should not include attention when lambda_attn=0"

    print("All self-tests passed.\n")

if __name__ == "__main__":
    # Run quick tests first so users see a green check quickly
    run_tests()
    # Always run Hello World by default (no flags)
    hello_world()
