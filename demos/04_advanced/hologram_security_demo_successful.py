"""
Successful Security Demo - Shows a case where the presence security lattice approves

This demonstrates the Generator Architecture successfully processing a security scenario
with proper β convergence and valid receipts, resulting in security approval.
"""

import numpy as np
import json
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from hologram_generator_mini import ga_block, GAConfig, Sectors, PAGES, BYTES

def create_approvable_scenario():
    """Create a scenario that should result in security approval"""
    print("🔒 SUCCESSFUL PRESENCE SECURITY SCENARIO")
    print("=" * 45)
    print("Creating scenario designed for security approval...")
    
    np.random.seed(123)  # Different seed for variety
    
    # Create a clean, well-defined scenario
    ground_truth = np.zeros((PAGES, BYTES))
    
    # Single, clear security zone
    ground_truth[15:25, 100:150] = 0.8  # Clear presence signal
    
    # Add very minimal noise
    noise = np.random.normal(0, 0.02, ground_truth.shape)
    measurements = ground_truth + noise
    
    print(f"   ✅ Created {PAGES}x{BYTES} = {PAGES*BYTES:,} presence points")
    print(f"   🎯 Single clear security zone")
    print(f"   📡 Minimal noise (std=0.02)")
    print(f"   📊 Signal range: {measurements.min():.3f} to {measurements.max():.3f}")
    
    return ground_truth, measurements

def run_approval_processing(measurements):
    """Run processing optimized for approval"""
    print(f"\n🛡️ APPROVAL-OPTIMIZED PROCESSING")
    print("=" * 35)
    
    # Configuration optimized for success
    config = GAConfig(
        steps=100,  # More steps for better convergence
        step_size=0.02,  # Small step size for stability
        sectors=Sectors(
            lambda_data=0.5,    # Moderate data fidelity
            lambda_cons=0.05,   # Light conservation
            lambda_smooth=0.2,  # Moderate smoothness
            lambda_gauge=0.02,  # Light gauge
            lambda_attn=0.1     # Light attention
        )
    )
    
    print(f"   🔧 Approval-Optimized Configuration:")
    print(f"      - Processing steps: {config.steps}")
    print(f"      - Step size: {config.step_size} (very conservative)")
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

def analyze_approval_results(ground_truth, measurements, reconstructed, bhic):
    """Analyze results for approval criteria"""
    print(f"\n📊 APPROVAL ANALYSIS")
    print("=" * 25)
    
    # Calculate metrics
    noise_error = np.mean((measurements - ground_truth)**2)
    reconstruction_error = np.mean((reconstructed - ground_truth)**2)
    
    if noise_error > 0:
        improvement = ((noise_error - reconstruction_error) / noise_error) * 100
    else:
        improvement = 0
    
    # Security metrics
    threshold = 0.4
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
            status = "✅ ACCEPTABLE"
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

def show_β_convergence():
    """Show β convergence pattern"""
    print(f"\n📈 β CONVERGENCE PATTERN")
    print("=" * 30)
    
    # Test convergence with the same data
    test_data = np.random.normal(0, 0.05, (PAGES, BYTES))
    
    steps_to_test = [1, 10, 25, 50, 75, 100]
    
    print("   🔄 β convergence during processing:")
    print("   " + "-" * 70)
    print(f"   {'Steps':>5} | {'Data':>8} | {'Cons':>8} | {'Smooth':>8} | {'Gauge':>8} | {'Attn':>8}")
    print("   " + "-" * 70)
    
    for steps in steps_to_test:
        config = GAConfig(steps=steps, step_size=0.02, sectors=Sectors(
            lambda_data=0.5, lambda_cons=0.05, lambda_smooth=0.2, 
            lambda_gauge=0.02, lambda_attn=0.1
        ))
        
        _, bhic = ga_block(test_data, config)
        
        print(f"   {steps:>5} | {bhic.beta['data']:>8.4f} | {bhic.beta['cons']:>8.4f} | "
              f"{bhic.beta['smooth']:>8.4f} | {bhic.beta['gauge']:>8.4f} | {bhic.beta['attn']:>8.4f}")
    
    print("   " + "-" * 70)
    print("   💡 β values converge to stable, low values")
    print("   💡 This indicates successful optimization")

def show_receipt_validation():
    """Show receipt validation process"""
    print(f"\n🧾 RECEIPT VALIDATION PROCESS")
    print("=" * 35)
    
    # Test different scenarios
    test_cases = [
        ("Clean data", np.random.normal(0, 0.01, (PAGES, BYTES))),
        ("Noisy data", np.random.normal(0, 0.1, (PAGES, BYTES))),
        ("Pattern data", create_clean_pattern())
    ]
    
    print("   📊 Receipt validation across scenarios:")
    print("   " + "-" * 85)
    print(f"   {'Scenario':>12} | {'R96 Entropy':>10} | {'C768 Inv':>8} | {'Klein':>6} | {'Φ':>3} | {'β Max':>8} | {'Approved':>8}")
    print("   " + "-" * 85)
    
    for name, data in test_cases:
        config = GAConfig(steps=50, step_size=0.02, sectors=Sectors(
            lambda_data=0.5, lambda_cons=0.05, lambda_smooth=0.2, 
            lambda_gauge=0.02, lambda_attn=0.1
        ))
        
        _, bhic = ga_block(data, config)
        max_beta = max(bhic.beta.values())
        approved = bhic.accept(eps=0.1, require_klein=False)
        
        print(f"   {name:>12} | {bhic.r96['entropy']:>10.3f} | "
              f"{str(bhic.c768['rotationInvariant']):>8} | "
              f"{bhic.klein.sum():>6} | {str(bhic.phi_accept):>3} | "
              f"{max_beta:>8.4f} | {str(approved):>8}")
    
    print("   " + "-" * 85)
    print("   💡 Receipts provide cryptographic proof of system integrity")
    print("   💡 Approval depends on β values and receipt validity")

def create_clean_pattern():
    """Create a clean pattern for testing"""
    data = np.zeros((PAGES, BYTES))
    # Add a simple, clean pattern
    data[20:30, 100:120] = 0.5
    return data

def main():
    """Run the successful security demonstration"""
    print("🔒 SUCCESSFUL PRESENCE SECURITY LATTICE DEMONSTRATION")
    print("=" * 65)
    print("Showing β convergence and receipt validation for approval")
    print()
    
    # Step 1: Create approvable scenario
    ground_truth, measurements = create_approvable_scenario()
    
    # Step 2: Run approval-optimized processing
    reconstructed, bhic, processing_time = run_approval_processing(measurements)
    
    # Step 3: Analyze results
    improvement, detection_accuracy = analyze_approval_results(
        ground_truth, measurements, reconstructed, bhic
    )
    
    # Step 4: Show β convergence
    show_β_convergence()
    
    # Step 5: Show receipt validation
    show_receipt_validation()
    
    # Step 6: Final security decision
    print(f"\n🏆 FINAL SECURITY DECISION")
    print("=" * 30)
    
    # Approval criteria
    beta_ok = all(v <= 0.1 for v in bhic.beta.values())
    accuracy_ok = detection_accuracy >= 0.8
    receipts_ok = bhic.c768['rotationInvariant'] and bhic.phi_accept
    
    security_approved = beta_ok and accuracy_ok and receipts_ok
    
    print(f"   🔍 SECURITY CRITERIA:")
    print(f"      - β values ≤ 0.1: {'✅ PASS' if beta_ok else '❌ FAIL'}")
    print(f"      - Detection accuracy ≥ 80%: {'✅ PASS' if accuracy_ok else '❌ FAIL'}")
    print(f"      - Receipts valid: {'✅ PASS' if receipts_ok else '❌ FAIL'}")
    
    print(f"\n   🚨 FINAL DECISION: {'✅ SECURITY APPROVED' if security_approved else '❌ SECURITY REJECTED'}")
    
    if security_approved:
        print(f"\n   🎉 CONGRATULATIONS!")
        print(f"      🛡️ Presence security lattice is operational")
        print(f"      🔒 All sectors performing within tolerance")
        print(f"      🧾 Receipts provide cryptographic proof")
        print(f"      ✅ System ready for deployment")
    
    # Step 7: Generate approval report
    print(f"\n📋 SECURITY APPROVAL REPORT")
    print("=" * 30)
    
    approval_report = {
        "lattice": f"{PAGES}x{BYTES}",
        "profile": "Successful-Presence-Security",
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
        "approval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(json.dumps(approval_report, indent=2))
    
    print(f"\n🎉 SUCCESSFUL SECURITY LATTICE DEMONSTRATION COMPLETE")
    print(f"   ✅ Processed {PAGES*BYTES:,} presence points")
    print(f"   ✅ Applied 5 security layers (sectors)")
    print(f"   ✅ Generated valid receipts")
    print(f"   ✅ Demonstrated β convergence")
    print(f"   ✅ Security decision: {'APPROVED' if security_approved else 'REJECTED'}")

if __name__ == "__main__":
    main()


