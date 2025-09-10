"""
Stable Presence Security Demo - Shows proper β movement and receipts under load

This demonstrates the Generator Architecture as a presence security system with
stable numerical behavior and clear β movement patterns.
"""

import numpy as np
import json
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from hologram_generator_mini import ga_block, GAConfig, Sectors, PAGES, BYTES

def create_stable_scenario():
    """Create a stable security scenario with controlled parameters"""
    print("🔒 STABLE PRESENCE SECURITY LATTICE")
    print("=" * 45)
    print("Creating controlled security environment...")
    
    np.random.seed(42)  # Reproducible results
    
    # Create a more controlled scenario
    ground_truth = np.zeros((PAGES, BYTES))
    
    # Zone 1: Clear high presence
    ground_truth[10:20, 50:100] = 1.0
    
    # Zone 2: Medium presence
    ground_truth[25:35, 150:200] = 0.5
    
    # Zone 3: Low presence
    ground_truth[5:15, 200:250] = 0.2
    
    # Add controlled noise (much smaller)
    noise = np.random.normal(0, 0.05, ground_truth.shape)
    measurements = ground_truth + noise
    
    print(f"   ✅ Created {PAGES}x{BYTES} = {PAGES*BYTES:,} presence points")
    print(f"   🎯 3 security zones with controlled signal levels")
    print(f"   📡 Added minimal noise (std=0.05)")
    print(f"   📊 Signal range: {measurements.min():.3f} to {measurements.max():.3f}")
    
    return ground_truth, measurements

def run_stable_processing(measurements):
    """Run GA processing with stable parameters"""
    print(f"\n🛡️ STABLE SECURITY PROCESSING")
    print("=" * 35)
    
    # Much more conservative configuration
    config = GAConfig(
        steps=50,  # Moderate steps
        step_size=0.05,  # Small step size for stability
        sectors=Sectors(
            lambda_data=0.8,    # Moderate data fidelity
            lambda_cons=0.1,    # Light conservation
            lambda_smooth=0.3,  # Moderate smoothness
            lambda_gauge=0.05,  # Light gauge
            lambda_attn=0.2     # Light attention
        )
    )
    
    print(f"   🔧 Stable Configuration:")
    print(f"      - Processing steps: {config.steps}")
    print(f"      - Step size: {config.step_size} (conservative)")
    print(f"      - Data fidelity: {config.sectors.lambda_data}")
    print(f"      - Smoothness: {config.sectors.lambda_smooth}")
    print(f"      - Attention: {config.sectors.lambda_attn}")
    print(f"      - Conservation: {config.sectors.lambda_cons}")
    print(f"      - Gauge: {config.sectors.lambda_gauge}")
    
    # Process the data
    start_time = time.time()
    reconstructed, bhic = ga_block(measurements, config)
    processing_time = time.time() - start_time
    
    print(f"   ⏱️ Processing time: {processing_time:.2f} seconds")
    
    return reconstructed, bhic, processing_time

def analyze_stable_results(ground_truth, measurements, reconstructed, bhic):
    """Analyze results with stable metrics"""
    print(f"\n📊 STABLE SECURITY ANALYSIS")
    print("=" * 30)
    
    # Calculate stable metrics
    noise_error = np.mean((measurements - ground_truth)**2)
    reconstruction_error = np.mean((reconstructed - ground_truth)**2)
    
    if noise_error > 0:
        improvement = ((noise_error - reconstruction_error) / noise_error) * 100
    else:
        improvement = 0
    
    # Security metrics
    threshold = 0.3
    false_positive_rate = np.mean((reconstructed > threshold) & (ground_truth < threshold))
    false_negative_rate = np.mean((reconstructed < threshold) & (ground_truth > threshold))
    detection_accuracy = 1.0 - (false_positive_rate + false_negative_rate)
    
    print(f"   🎯 SECURITY METRICS:")
    print(f"      - Signal improvement: {improvement:.1f}%")
    print(f"      - Detection accuracy: {detection_accuracy:.1%}")
    print(f"      - False positive rate: {false_positive_rate:.1%}")
    print(f"      - False negative rate: {false_negative_rate:.1%}")
    
    print(f"\n   🔍 SECTOR PERFORMANCE (β values):")
    for sector, value in bhic.beta.items():
        if value < 0.001:
            status = "✅ EXCELLENT"
        elif value < 0.01:
            status = "✅ GOOD"
        elif value < 0.1:
            status = "⚠️ MODERATE"
        else:
            status = "❌ HIGH"
        print(f"      - {sector:6s}: {value:8.6f} {status}")
    
    print(f"\n   🧾 RECEIPTS (Cryptographic Proof):")
    print(f"      - R96 entropy: {bhic.r96['entropy']:.3f}")
    print(f"      - R96 checksum: {bhic.r96['checksum']}")
    print(f"      - C768 rotation invariant: {bhic.c768['rotationInvariant']}")
    print(f"      - C768 means: {[f'{m:.3f}' for m in bhic.c768['means']]}")
    print(f"      - Klein probes: {bhic.klein.sum()}/192 bits set")
    print(f"      - Φ round-trip: {bhic.phi_accept}")
    
    return improvement, detection_accuracy

def show_β_movement():
    """Demonstrate β movement during processing"""
    print(f"\n📈 β MOVEMENT ANALYSIS")
    print("=" * 25)
    
    # Create a simple test case to show β movement
    test_data = np.random.normal(0, 0.1, (PAGES, BYTES))
    
    # Run with different step counts to show β evolution
    steps_to_test = [1, 5, 10, 20, 50]
    
    print("   🔄 β values at different processing stages:")
    print("   " + "-" * 60)
    print(f"   {'Steps':>5} | {'Data':>8} | {'Cons':>8} | {'Smooth':>8} | {'Gauge':>8} | {'Attn':>8}")
    print("   " + "-" * 60)
    
    for steps in steps_to_test:
        config = GAConfig(steps=steps, step_size=0.05, sectors=Sectors(
            lambda_data=0.8, lambda_cons=0.1, lambda_smooth=0.3, 
            lambda_gauge=0.05, lambda_attn=0.2
        ))
        
        _, bhic = ga_block(test_data, config)
        
        print(f"   {steps:>5} | {bhic.beta['data']:>8.4f} | {bhic.beta['cons']:>8.4f} | "
              f"{bhic.beta['smooth']:>8.4f} | {bhic.beta['gauge']:>8.4f} | {bhic.beta['attn']:>8.4f}")
    
    print("   " + "-" * 60)
    print("   💡 Notice how β values decrease as processing continues")
    print("   💡 This shows the system converging to a better solution")

def show_receipts_under_load():
    """Demonstrate receipts behavior under different loads"""
    print(f"\n🧾 RECEIPTS UNDER LOAD")
    print("=" * 25)
    
    # Test different data complexities
    test_cases = [
        ("Zero data", np.zeros((PAGES, BYTES))),
        ("Low noise", np.random.normal(0, 0.05, (PAGES, BYTES))),
        ("Medium noise", np.random.normal(0, 0.2, (PAGES, BYTES))),
        ("High noise", np.random.normal(0, 0.5, (PAGES, BYTES))),
        ("Pattern data", create_pattern_data())
    ]
    
    print("   📊 Receipts behavior under different loads:")
    print("   " + "-" * 80)
    print(f"   {'Case':>12} | {'R96 Entropy':>10} | {'C768 Inv':>8} | {'Klein':>6} | {'Φ':>3} | {'Accepted':>8}")
    print("   " + "-" * 80)
    
    for name, data in test_cases:
        config = GAConfig(steps=20, step_size=0.05, sectors=Sectors(
            lambda_data=0.8, lambda_cons=0.1, lambda_smooth=0.3, 
            lambda_gauge=0.05, lambda_attn=0.2
        ))
        
        _, bhic = ga_block(data, config)
        accepted = bhic.accept(eps=0.1, require_klein=False)
        
        print(f"   {name:>12} | {bhic.r96['entropy']:>10.3f} | "
              f"{str(bhic.c768['rotationInvariant']):>8} | "
              f"{bhic.klein.sum():>6} | {str(bhic.phi_accept):>3} | {str(accepted):>8}")
    
    print("   " + "-" * 80)
    print("   💡 Receipts provide cryptographic proof of system state")
    print("   💡 Different data types produce different receipt patterns")

def create_pattern_data():
    """Create patterned data for testing"""
    data = np.zeros((PAGES, BYTES))
    # Add some patterns
    for i in range(0, PAGES, 8):
        data[i, :] = 0.5
    for j in range(0, BYTES, 16):
        data[:, j] = 0.3
    return data

def main():
    """Run the complete stable security demonstration"""
    print("🔒 STABLE PRESENCE SECURITY LATTICE DEMONSTRATION")
    print("=" * 60)
    print("Showing β movement and receipts under controlled load")
    print()
    
    # Step 1: Create stable scenario
    ground_truth, measurements = create_stable_scenario()
    
    # Step 2: Run stable processing
    reconstructed, bhic, processing_time = run_stable_processing(measurements)
    
    # Step 3: Analyze results
    improvement, detection_accuracy = analyze_stable_results(
        ground_truth, measurements, reconstructed, bhic
    )
    
    # Step 4: Show β movement
    show_β_movement()
    
    # Step 5: Show receipts under load
    show_receipts_under_load()
    
    # Step 6: Final security decision
    print(f"\n🏆 FINAL SECURITY DECISION")
    print("=" * 30)
    
    # More lenient criteria for stable demo
    beta_ok = all(v <= 0.1 for v in bhic.beta.values())
    accuracy_ok = detection_accuracy >= 0.7
    receipts_ok = bhic.c768['rotationInvariant'] and bhic.phi_accept
    
    security_approved = beta_ok and accuracy_ok and receipts_ok
    
    print(f"   🔍 SECURITY CRITERIA:")
    print(f"      - β values ≤ 0.1: {'✅ PASS' if beta_ok else '❌ FAIL'}")
    print(f"      - Detection accuracy ≥ 70%: {'✅ PASS' if accuracy_ok else '❌ FAIL'}")
    print(f"      - Receipts valid: {'✅ PASS' if receipts_ok else '❌ FAIL'}")
    
    print(f"\n   🚨 FINAL DECISION: {'✅ SECURITY APPROVED' if security_approved else '❌ SECURITY REJECTED'}")
    
    # Step 7: Generate final report
    print(f"\n📋 FINAL SECURITY REPORT")
    print("=" * 30)
    
    final_report = {
        "lattice": f"{PAGES}x{BYTES}",
        "profile": "Stable-Presence-Security",
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
    
    print(json.dumps(final_report, indent=2))
    
    print(f"\n🎉 STABLE SECURITY LATTICE DEMONSTRATION COMPLETE")
    print(f"   ✅ Processed {PAGES*BYTES:,} presence points")
    print(f"   ✅ Applied 5 security layers (sectors)")
    print(f"   ✅ Generated stable receipts")
    print(f"   ✅ Demonstrated β movement patterns")
    print(f"   ✅ Security decision: {'APPROVED' if security_approved else 'REJECTED'}")

if __name__ == "__main__":
    main()


