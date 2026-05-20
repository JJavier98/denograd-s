#!/usr/bin/env python3
"""
Demo script: Variance Threshold Compact Sparsification
Tests both post-training and during-training variance-threshold methods
with aggressive settings to demonstrate parameter reduction.
"""

import json
import sys
from pathlib import Path
from src.libs.experiment_runner import run_tabular_experiment, run_ts_experiment

def run_post_training_demo():
    """Post-training variance_threshold_compact with aggressive variance_pct=0.5"""
    print("\n" + "="*70)
    print("DEMO 1: POST-TRAINING VARIANCE_THRESHOLD_COMPACT (variance_pct=0.5)")
    print("="*70)
    
    cfg_post = {
        'artifacts_dir': 'out/demo_variance_threshold',
        'version': 'post_training_aggressive',
        'seed': 42,
        'device': 'cuda:0',
        'dataset': {
            'path': 'data/tabular/house_prices/kc_house_data.csv',
            'target_col': 'price',
            'noise_std': 0.05,
            'batch_size': 128,
            'test_split': 0.2,
            'val_split': 0.1,
        },
        'model': {
            'name': 'dnn',
            'hidden_dims': [128, 64, 32],
            'lr': 1e-3,
            'patience': 3,
            'max_epochs': 15,
        },
        'denograd': {
            'nrr': 1e-3,
            'threshold': 5e-3,
            'max_iters': 30,
            'batch_size': 1024,
        },
        'benchmark': {
            'ridge': {'enabled': True},
            'knn': {'enabled': False},
            'xgboost': {'enabled': False},
            'tabpfn': {'enabled': False},
            'dnn': {'enabled': True, 'max_epochs': 15, 'hidden_dims': [128, 64, 32], 'patience': 3},
        },
        'sparsity': {
            'enabled': True,
            'method': 'variance_threshold_compact',
            'variance_pct': 0.5,  # Aggressive threshold
            'include_bias': False,
        },
    }
    
    try:
        summary = run_tabular_experiment(cfg_post)
        
        # Print comparison
        if summary.get("sparse", {}).get("dense_vs_sparse_comparison", {}).get("machine"):
            comp = summary["sparse"]["dense_vs_sparse_comparison"]["machine"]
            print("\n✓ Dense vs Sparse Comparison:")
            print(f"  Params: {comp['total_params']['dense']} → {comp['total_params']['sparse']} "
                  f"({comp['total_params']['delta_pct']:.1f}%)")
            print(f"  Denoising: {comp['denoising_seconds']['dense']:.1f}s → "
                  f"{comp['denoising_seconds']['sparse']:.1f}s "
                  f"({comp['denoising_seconds']['speedup_x']:.2f}x speedup)")
            
            report = summary["sparse"].get("report", {})
            print(f"\n✓ Sparse Report:")
            print(f"  Method: {report.get('resolved_method')}")
            print(f"  Variance: {report.get('model_variance'):.6f}")
            print(f"  Epsilon:  {report.get('epsilon'):.6f}")
            print(f"  Reduction: {report.get('param_reduction_ratio')*100:.2f}%")
            print("\n[PASS] Post-training demo completed!")
            return True
        else:
            print("[FAIL] No comparison metrics!")
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def run_during_training_demo():
    """During-training gradual_variance_threshold with scheduling"""
    print("\n" + "="*70)
    print("DEMO 2: DURING-TRAINING GRADUAL_VARIANCE_THRESHOLD (0.2→0.5 over epochs)")
    print("="*70)
    
    cfg_during = {
        'artifacts_dir': 'out/demo_variance_threshold',
        'version': 'during_training_gradual',
        'seed': 42,
        'device': 'cuda:0',
        'dataset': {
            'path': 'data/tabular/house_prices/kc_house_data.csv',
            'target_col': 'price',
            'noise_std': 0.05,
            'batch_size': 128,
            'test_split': 0.2,
            'val_split': 0.1,
        },
        'model': {
            'name': 'dnn',
            'hidden_dims': [128, 64, 32],
            'lr': 1e-3,
            'patience': 3,
            'max_epochs': 15,
        },
        'denograd': {
            'nrr': 1e-3,
            'threshold': 5e-3,
            'max_iters': 30,
            'batch_size': 1024,
        },
        'benchmark': {
            'ridge': {'enabled': True},
            'knn': {'enabled': False},
            'xgboost': {'enabled': False},
            'tabpfn': {'enabled': False},
            'dnn': {'enabled': True, 'max_epochs': 15, 'hidden_dims': [128, 64, 32], 'patience': 3},
        },
        'sparsity': {
            'enabled': True,
            'method': 'gradual_variance_threshold',
            'start_variance_pct': 0.2,
            'target_variance_pct': 0.5,
            'include_bias': False,
        },
    }
    
    try:
        summary = run_tabular_experiment(cfg_during)
        
        # Print comparison
        if summary.get("sparse", {}).get("dense_vs_sparse_comparison", {}).get("machine"):
            comp = summary["sparse"]["dense_vs_sparse_comparison"]["machine"]
            print("\n✓ Dense vs Sparse Comparison:")
            print(f"  Params: {comp['total_params']['dense']} → {comp['total_params']['sparse']} "
                  f"({comp['total_params']['delta_pct']:.1f}%)")
            print(f"  Denoising: {comp['denoising_seconds']['dense']:.1f}s → "
                  f"{comp['denoising_seconds']['sparse']:.1f}s "
                  f"({comp['denoising_seconds']['speedup_x']:.2f}x speedup)")
            
            report = summary["sparse"].get("report", {})
            print(f"\n✓ During-Training Report:")
            print(f"  Method: {report.get('method')}")
            print(f"  Start Variance: {report.get('start_variance_pct')}")
            print(f"  Target Variance: {report.get('target_variance_pct')}")
            print(f"  Final Sparsity: {(1-report.get('final_density', 1.0))*100:.1f}%")
            print("\n[PASS] During-training demo completed!")
            return True
        else:
            print("[FAIL] No comparison metrics!")
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("VARIANCE THRESHOLD SPARSIFICATION DEMO")
    print("="*70)
    
    post_ok = run_post_training_demo()
    during_ok = run_during_training_demo()
    
    print("\n" + "="*70)
    print(f"Results: Post-training={'✓' if post_ok else '✗'} | During-training={'✓' if during_ok else '✗'}")
    print("="*70 + "\n")
    
    sys.exit(0 if (post_ok and during_ok) else 1)
