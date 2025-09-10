# GA Mini - Generator Architecture

A didactic, runnable Python implementation of Generator Architecture on the 12,288 lattice.

## Overview

GA Mini demonstrates the core concepts of Generator Architecture with:

- **48x256 (12,288) lattice** - The fundamental tile structure
- **Multiple sectors**: data fidelity (deblur), conservation fairness (C768 means), smoothness (Laplacian), page zero-mean (simple gauge)
- **Transformer-like sector** - A fixed, normalized affinity operator A (attention-like, low-rank per page) with energy `0.5‖(I−A)ψ‖²`
- **Receipts system (BHIC)**: R96 histogram/entropy/checksum, C768 means/vars + rotation invariance check, Klein probes (toy), Φ-roundtrip (toy), β-ledger
- **Hello World demo** - Zero tile → β=0 under P-Core semantics → prints a tiny TPT-lite

## Features

- ✅ **Generator Architecture** on 12,288 lattice
- ✅ **Transformer-ish sector** with attention-like affinity operator
- ✅ **Receipts (BHIC)**: R96 histogram, C768 means/vars, Klein probes, Φ-roundtrip, β-ledger
- ✅ **Hello World default** with deterministic acceptance
- ✅ **Self-tests** for validation
- ✅ **P-Core semantics** (no Klein requirement for Hello World)

## Installation

### Prerequisites

- Python 3.11 or higher
- NumPy

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ga-mini-generator.git
   cd ga-mini-generator
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy
   ```

3. **Run the script**:
   ```bash
   python ga_mini.py
   ```

## Usage

### Basic Usage

```bash
python ga_mini.py
```

This will:
1. Run self-tests to verify everything works
2. Execute the Hello World demo
3. Print the TPT-lite output showing Generator Architecture results

### Expected Output

```
=== Running GA Mini self-tests ===
All self-tests passed.

Hello, GA (12,288)! 👋
TPT-lite:
{
  "tile": "48x256",
  "profile": "P-Core(hello)",
  "beta": {
    "data": 0.0,
    "cons": 0.0,
    "smooth": 0.0,
    "gauge": 0.0,
    "attn": 0.0
  },
  "c768": {
    "rotationInvariant": true
  },
  "phi": {
    "accept": true
  },
  "accepted": true
}
```

## Architecture Details

### Sectors

- **Data Fidelity**: L2 data fidelity with blur operator
- **Conservation**: C768 conservation with residue classes
- **Smoothness**: Laplacian smoothness with 5-point stencil
- **Gauge**: Page zero-mean gauge
- **Attention**: Transformer-like attention with affinity operator

### Receipts (BHIC)

- **R96**: Histogram with 96 bins, entropy calculation, checksum
- **C768**: Means and variances for 3 residue classes, rotation invariance check
- **Klein**: 192-bit probes for validation
- **Φ**: Round-trip acceptance (identity for demo)
- **β-ledger**: Sector residuals at solution

## Notes

- This is a pedagogical approximation that preserves the shape of GA: tile, sectors, action, receipts, β
- R96 uses byte%96 over 8-bit quantization for clarity
- Φ/NF-Lift are identity for the demo
- The Transformer-like sector builds per-page low-rank attention operators from measurements

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Based on the Generator Architecture framework and inspired by the original GA Mini implementation.
