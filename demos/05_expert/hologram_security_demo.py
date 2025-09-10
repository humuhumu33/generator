"""
Presence Security Lattice Demo - Non-trivial run showing β movement and receipts under load

This demonstrates the full Generator Architecture as a presence security system:
- Lattice as environment (48x256 = 12,288 presence points)
- Sectors as security layers (data fidelity, conservation, smoothness, gauge, attention)
- Receipts as cryptographic proof of presence validation
"""

import numpy as np
import json
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from hologram_generator_mini import ga_block, GAConfig, Sectors, PAGES, BYTES

def create_presence_scenario():
    """Create a realistic presence security scenario"""
    print("🔒 PRESENCE SECURITY LATTICE SCENARIO")
    print("=" * 50)
    print("Creating a realistic security environment...")
    
    # Simulate a security scenario with multiple presence zones
    np.random.seed(42)  # Reproducible results
    
    # Ground truth: Multiple security zones with different threat levels
    ground_truth = np.zeros((PAGES, BYTES))
    
    # Zone 1: High security area (strong presence signal)
    ground_truth[10:20, 50:100] = 2.0
    
    # Zone 2: Medium security area
    ground_truth[25:35, 150:200] = 1.0
    
    # Zone 3: Low security area (weak presence)
    ground_truth[5:15, 200:250] = 0.5
    
    # Add some noise and interference (realistic measurements)
    noise = np.random.normal(0, 0.3, ground_truth.shape)
    interference = np.random.exponential(0.1, ground_truth.shape)
    
    # Corrupted measurements (what the sensors actually see)
    measurements = ground_truth + noise + interference
    
    print(f"   ✅ Created {PAGES}x{BYTES} = {PAGES*BYTES:,} presence points")
    print(f"   🎯 3 security zones with different threat levels")
    print(f"   📡 Added realistic noise and interference")
    print(f"   📊 Signal range: {measurements.min():.3f} to {measurements.max():.3f}")
    
    return ground_truth, measurements

def run_security_processing(measurements):
    """Run GA processing with security-focused configuration"""
    print(f"\n🛡️ SECURITY LAYER PROCESSING")
    print("=" * 40)
    
    # Security-optimized configuration
    config = GAConfig(
        steps=100,  # More steps for thorough processing
        step_size=0.15,
        sectors=Sectors(
            lambda_data=1.2,    # High data fidelity (trust measurements)
            lambda_cons=0.3,    # Moderate conservation (energy preservation)
            lambda_smooth=0.8,  # High smoothness (reduce false positives)
            lambda_gauge=0.2,   # Moderate gauge (normalize baselines)
            lambda_attn=0.6     # High attention (focus on patterns)
        )
    )
    
    print(f"   🔧 Security Configuration:")
    print(f"      - Processing steps: {config.steps}")
    print(f"      - Data fidelity: {config.sectors.lambda_data} (trust sensors)")
    print(f"      - Smoothness: {config.sectors.lambda_smooth} (reduce noise)")
    print(f"      - Attention: {config.sectors.lambda_attn} (pattern detection)")
    print(f"      - Conservation: {config.sectors.lambda_cons} (energy balance)")
    print(f"      - Gauge: {config.sectors.lambda_gauge} (baseline correction)")
    
    # Process the security data
    start_time = time.time()
    reconstructed, bhic = ga_block(measurements, config)
    processing_time = time.time() - start_time
    
    print(f"   ⏱️ Processing time: {processing_time:.2f} seconds")
    print(f"   📊 Processed {PAGES*BYTES:,} presence points")
    
    return reconstructed, bhic, processing_time

def analyze_security_results(ground_truth, measurements, reconstructed, bhic):
    """Analyze security processing results"""
    print(f"\n📊 SECURITY ANALYSIS RESULTS")
    print("=" * 35)
    
    # Calculate security metrics
    noise_error = np.mean((measurements - ground_truth)**2)
    reconstruction_error = np.mean((reconstructed - ground_truth)**2)
    improvement = ((noise_error - reconstruction_error) / noise_error) * 100
    
    # Security-specific metrics
    false_positive_rate = np.mean((reconstructed > 0.5) & (ground_truth < 0.5))
    false_negative_rate = np.mean((reconstructed < 0.5) & (ground_truth > 0.5))
    detection_accuracy = 1.0 - (false_positive_rate + false_negative_rate)
    
    print(f"   🎯 SECURITY METRICS:")
    print(f"      - Signal improvement: {improvement:.1f}%")
    print(f"      - Detection accuracy: {detection_accuracy:.1%}")
    print(f"      - False positive rate: {false_positive_rate:.1%}")
    print(f"      - False negative rate: {false_negative_rate:.1%}")
    
    print(f"\n   🔍 SECTOR PERFORMANCE (β values - lower is better):")
    for sector, value in bhic.beta.items():
        if value < 0.01:
            status = "✅ EXCELLENT"
        elif value < 0.1:
            status = "✅ GOOD"
        elif value < 1.0:
            status = "⚠️ MODERATE"
        else:
            status = "❌ NEEDS ATTENTION"
        print(f"      - {sector:6s}: {value:8.4f} {status}")
    
    print(f"\n   🧾 RECEIPTS (Cryptographic Proof):")
    print(f"      - R96 entropy: {bhic.r96['entropy']:.3f} (data complexity)")
    print(f"      - R96 checksum: {bhic.r96['checksum']} (integrity check)")
    print(f"      - C768 rotation invariant: {bhic.c768['rotationInvariant']} (stability)")
    print(f"      - C768 means: {[f'{m:.3f}' for m in bhic.c768['means']]} (residue classes)")
    print(f"      - Klein probes: {bhic.klein.sum()}/192 bits set (validation)")
    print(f"      - Φ round-trip: {bhic.phi_accept} (consistency check)")
    
    return improvement, detection_accuracy

def show_presence_zones(ground_truth, measurements, reconstructed):
    """Show the presence zones and how they were processed"""
    print(f"\n🔍 PRESENCE ZONE ANALYSIS")
    print("=" * 30)
    
    # Analyze each zone
    zones = [
        (10, 20, 50, 100, "High Security Zone"),
        (25, 35, 150, 200, "Medium Security Zone"),
        (5, 15, 200, 250, "Low Security Zone")
    ]
    
    for p_start, p_end, b_start, b_end, name in zones:
        gt_zone = ground_truth[p_start:p_end, b_start:b_end]
        meas_zone = measurements[p_start:p_end, b_start:b_end]
        recon_zone = reconstructed[p_start:p_end, b_start:b_end]
        
        gt_mean = np.mean(gt_zone)
        meas_mean = np.mean(meas_zone)
        recon_mean = np.mean(recon_zone)
        
        print(f"   🎯 {name}:")
        print(f"      - Ground truth: {gt_mean:.3f}")
        print(f"      - Measurements: {meas_mean:.3f}")
        print(f"      - Reconstructed: {recon_mean:.3f}")
        print(f"      - Recovery: {((recon_mean - meas_mean) / (gt_mean - meas_mean) * 100):.1f}%")

def security_decision(bhic, detection_accuracy):
    """Make final security decision based on receipts and metrics"""
    print(f"\n🏆 SECURITY DECISION")
    print("=" * 25)
    
    # Security-specific acceptance criteria
    beta_threshold = 0.5  # More lenient for security applications
    accuracy_threshold = 0.8
    
    # Check all criteria
    beta_ok = all(v <= beta_threshold for v in bhic.beta.values())
    accuracy_ok = detection_accuracy >= accuracy_threshold
    receipts_ok = bhic.c768['rotationInvariant'] and bhic.phi_accept
    
    print(f"   🔍 SECURITY CRITERIA:")
    print(f"      - β values ≤ {beta_threshold}: {'✅ PASS' if beta_ok else '❌ FAIL'}")
    print(f"      - Detection accuracy ≥ {accuracy_threshold:.1%}: {'✅ PASS' if accuracy_ok else '❌ FAIL'}")
    print(f"      - Receipts valid: {'✅ PASS' if receipts_ok else '❌ FAIL'}")
    
    # Final decision
    security_approved = beta_ok and accuracy_ok and receipts_ok
    
    print(f"\n   🚨 FINAL SECURITY DECISION:")
    if security_approved:
        print(f"      ✅ SECURITY APPROVED")
        print(f"      🛡️ Presence validation successful")
        print(f"      🔒 System ready for deployment")
    else:
        print(f"      ❌ SECURITY REJECTED")
        print(f"      ⚠️ Additional calibration required")
        print(f"      🔧 Review sector parameters")
    
    return security_approved

def main():
    """Run the complete presence security demonstration"""
    print("🔒 GENERATOR ARCHITECTURE: PRESENCE SECURITY LATTICE")
    print("=" * 60)
    print("Demonstrating β movement and receipts under load")
    print()
    
    # Step 1: Create security scenario
    ground_truth, measurements = create_presence_scenario()
    
    # Step 2: Run security processing
    reconstructed, bhic, processing_time = run_security_processing(measurements)
    
    # Step 3: Analyze results
    improvement, detection_accuracy = analyze_security_results(
        ground_truth, measurements, reconstructed, bhic
    )
    
    # Step 4: Show presence zones
    show_presence_zones(ground_truth, measurements, reconstructed)
    
    # Step 5: Make security decision
    security_approved = security_decision(bhic, detection_accuracy)
    
    # Step 6: Generate security report
    print(f"\n📋 SECURITY REPORT (TPT-Security)")
    print("=" * 35)
    
    security_report = {
        "lattice": f"{PAGES}x{BYTES}",
        "profile": "Presence-Security",
        "processing_time_seconds": round(processing_time, 2),
        "signal_improvement_percent": round(improvement, 1),
        "detection_accuracy": round(detection_accuracy, 3),
        "beta_ledger": {k: round(v, 6) for k, v in bhic.beta.items()},
        "receipts": {
            "r96_entropy": round(bhic.r96['entropy'], 3),
            "r96_checksum": bhic.r96['checksum'],
            "c768_rotation_invariant": bhic.c768['rotationInvariant'],
            "c768_means": [round(m, 3) for m in bhic.c768['means']],
            "klein_probes_set": int(bhic.klein.sum()),
            "phi_roundtrip": bhic.phi_accept
        },
        "security_approved": security_approved,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(json.dumps(security_report, indent=2))
    
    print(f"\n🎉 PRESENCE SECURITY LATTICE DEMONSTRATION COMPLETE")
    print(f"   ✅ Processed {PAGES*BYTES:,} presence points")
    print(f"   ✅ Applied 5 security layers (sectors)")
    print(f"   ✅ Generated cryptographic receipts")
    print(f"   ✅ Made security decision: {'APPROVED' if security_approved else 'REJECTED'}")
    print(f"   ✅ Signal improvement: {improvement:.1f}%")
    print(f"   ✅ Detection accuracy: {detection_accuracy:.1%}")

if __name__ == "__main__":
    main()


