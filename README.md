# scRNA-Seq batch effect checker
This is my mini_toolkit to check for batch effect for scRNA-Seq dataset

Step 1: evaluate_batch_effects.py
Step 2: apply_harmony_correction.py

    # Provide recommendation
    Overall batch effect score : 0-1 scale, higher means stronger batch effect
    if batch_effect_score < 0.3:
        print("RECOMMENDATION: Batch correction likely NOT needed")
        print("- Your data shows minimal batch effects")
        print("- Proceed without Harmony or other batch correction methods")
    elif batch_effect_score < 0.6:
        print("RECOMMENDATION: Moderate batch effects detected")
        print("- Try analysis both with and without batch correction")
        print("- Consider using Harmony with default parameters")
    else:
        print("RECOMMENDATION: Strong batch effects detected")
        print("- Batch correction is highly recommended")
        print("- Use Harmony or another batch correction method")
        print("- Consider higher theta parameter in Harmony (e.g., 2.0) for stronger correction")
    
    print("\nSee generated plots for visual assessment of batch effects")
