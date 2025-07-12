#!/usr/bin/env python
"""Script to evaluate tracking results."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import MOTEvaluator


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate tracking results')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory containing ground truth files')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing tracking results')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                       help='Specific sequences to evaluate (default: all)')
    parser.add_argument('--metrics', type=str, nargs='+', 
                       default=['HOTA', 'CLEAR', 'Identity'],
                       help='Metrics to compute')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='IoU threshold for matching')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save results (optional)')
    
    args = parser.parse_args()
    
    # Validate directories
    gt_dir = Path(args.gt_dir)
    results_dir = Path(args.results_dir)
    
    if not gt_dir.exists():
        print(f"Error: Ground truth directory not found: {gt_dir}")
        return 1
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    # Initialize evaluator
    print(f"Initializing evaluator with metrics: {args.metrics}")
    evaluator = MOTEvaluator(
        metrics=args.metrics,
        threshold=args.threshold
    )
    
    # Run evaluation
    print(f"Evaluating results...")
    print(f"GT directory: {gt_dir}")
    print(f"Results directory: {results_dir}")
    
    try:
        metrics = evaluator.evaluate(
            gt_dir=str(gt_dir),
            results_dir=str(results_dir),
            sequences=args.sequences
        )
        
        # Print results
        evaluator.print_results(metrics)
        
        # Save results if output file specified
        if args.output:
            import json
            
            output_file = Path(args.output)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"\nResults saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())