"""
Simple GA Mini Demo - Shows the core concept in the easiest way possible

This demo answers: "What is GA Mini trying to achieve?"
"""

import numpy as np
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from hologram_generator_mini import ga_block, GAConfig, Sectors

def simple_explanation():
    """Explain what GA Mini does in simple terms"""
    print("🎯 WHAT IS GA MINI?")
    print("=" * 40)
    print("GA Mini is like a SMART IMAGE PROCESSOR that:")
    print("1. 📥 Takes messy/blurry input data")
    print("2. 🧠 Applies 5 different 'rules' to clean it up")
    print("3. 📤 Produces a cleaner, better version")
    print("4. ✅ Checks if the result is good enough")
    print()

def show_before_after():
    """Create a simple before/after example"""
    print("🔍 BEFORE vs AFTER EXAMPLE:")
    print("-" * 30)
    
    # Create simple test data
    print("📊 Creating test data...")
    
    # Clean data: simple pattern
    clean = np.zeros((48, 256))
    clean[10:20, 50:70] = 1.0  # A bright rectangle
    clean[30:40, 150:170] = 0.5  # A dimmer rectangle
    
    # Noisy data: add noise to simulate real measurements
    noisy = clean + np.random.normal(0, 0.2, clean.shape)
    
    print(f"   ✅ Clean data: 2 rectangles (bright + dim)")
    print(f"   📡 Noisy data: Same rectangles + random noise")
    print(f"   🎯 Goal: Recover clean rectangles from noisy data")
    print()
    
    return clean, noisy

def run_ga_demo(clean, noisy):
    """Run GA processing and show results"""
    print("⚙️ RUNNING GA PROCESSING...")
    print("-" * 30)
    
    # Optimized configuration for better convergence
    config = GAConfig(
        steps=100,  # More steps for better convergence
        step_size=0.05,  # Smaller step size for stability
        sectors=Sectors(
            lambda_data=2.0,    # Stronger data fidelity
            lambda_smooth=1.0,  # Stronger smoothness
            lambda_cons=0.1,    # Light conservation
            lambda_gauge=0.1,   # Light gauge
            lambda_attn=0.3     # Moderate attention
        )
    )
    
    print(f"   🔧 Processing with {config.steps} steps...")
    
    # Process the data
    result, bhic = ga_block(noisy, config)
    
    return result, bhic

def show_results(clean, noisy, result, bhic):
    """Show the results clearly"""
    print("\n📊 RESULTS:")
    print("=" * 20)
    
    # Calculate how much we improved
    noise_error = np.mean((noisy - clean)**2)
    result_error = np.mean((result - clean)**2)
    improvement = ((noise_error - result_error) / noise_error) * 100
    
    print(f"📈 IMPROVEMENT: {improvement:.1f}% better!")
    print(f"   - Before GA: Error = {noise_error:.4f}")
    print(f"   - After GA:  Error = {result_error:.4f}")
    print()
    
    # Show sector performance
    print("🎯 SECTOR PERFORMANCE:")
    for sector, value in bhic.beta.items():
        if value < 0.01:
            status = "✅ Perfect"
        elif value < 0.1:
            status = "✅ Good"
        elif value < 1.0:
            status = "⚠️ OK"
        else:
            status = "❌ Needs work"
        print(f"   - {sector:6s}: {value:8.4f} {status}")
    print()
    
    # Show data samples
    print("🔬 DATA SAMPLES (corner of the data):")
    print("   CLEAN (what we want):")
    print(f"   {clean[:3, :5]}")
    print("   NOISY (what we measured):")
    print(f"   {noisy[:3, :5]}")
    print("   RESULT (what GA produced):")
    print(f"   {result[:3, :5]}")
    print()
    
    # Final decision
    accepted = bhic.accept(eps=0.5)  # Lenient for demo
    print(f"🏆 FINAL DECISION: {'✅ ACCEPTED' if accepted else '❌ REJECTED'}")
    print(f"   The solution is {'good enough' if accepted else 'needs more work'}")
    print()

def main():
    """Run the complete simple demo"""
    simple_explanation()
    
    clean, noisy = show_before_after()
    
    result, bhic = run_ga_demo(clean, noisy)
    
    show_results(clean, noisy, result, bhic)
    
    print("🎉 SUMMARY:")
    print("GA Mini successfully processed 12,288 data points!")
    print("It's like having a smart assistant that:")
    print("• Takes messy data")
    print("• Applies 5 different cleaning rules")
    print("• Produces a better result")
    print("• Checks if it's good enough")
    print()
    print("💡 This is useful for:")
    print("• Image deblurring")
    print("• Signal processing")
    print("• Data reconstruction")
    print("• Scientific computing")

if __name__ == "__main__":
    main()


