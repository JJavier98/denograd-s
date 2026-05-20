#!/usr/bin/env python3
"""
Quick Validation: Compare updated sparsity methods on house_prices.
Methods: dense, post_training_ratio_priority (ratio=0.2), sparse_on_training (k=0.1)
"""

import json
from pathlib import Path
from src.libs.experiment_runner import run_tabular_experiment

METHODS = [
    {
        'label': 'DENSE (baseline)',
        'config': {
            'sparsity': {'enabled': False}
        }
    },
    {
        'label': 'POST-TRAINING RATIO PRIORITY (ratio=0.2)',
        'config': {
            'sparsity': {
                'enabled': True,
                'method': 'post_training_ratio_priority',
                'ratio': 0.2,
                'layer_priority_strength': 1.0,
                'include_bias': False,
            }
        }
    },
    {
        'label': 'SPARSE ON TRAINING SIGMA (k=0.1)',
        'config': {
            'sparsity': {
                'enabled': True,
                'method': 'sparse_on_training',
                'k': 0.1,
                'layer_priority_strength': 1.0,
                'include_bias': False,
            }
        }
    },
]

def build_config(method_config):
    """Build full experiment config with method-specific sparsity settings"""
    base = {
        'artifacts_dir': 'out',
        'version': 'quick_validation_v1',
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
            'patience': 15,
            'max_epochs': 100,
        },
        'denograd': {
            'nrr': 0.01,
            'threshold': 0.1,
            'max_iters': 150,
            'batch_size': 1024,
        },
        'benchmark': {
            'ridge': {'enabled': True},
            'knn': {'enabled': False},
            'xgboost': {'enabled': True, 'n_estimators': 50, 'max_depth': 6, 'learning_rate': 0.08},
            'tabpfn': {'enabled': False},
            'dnn': {'enabled': True, 'max_epochs': 100, 'hidden_dims': [128, 64, 32], 'patience': 15},
        },
    }
    base.update(method_config)
    return base

def extract_metrics(summary):
    """Extract key metrics from summary"""
    metrics = {
        'dense_improvement_pct': None,
        'sparse_improvement_pct': None,
        'param_reduction_pct': None,
        'time_speedup': None,
        'vram_delta_bytes': None,
    }
    
    # Dense improvement
    if 'evaluation' in summary:
        vals = [v for k, v in summary['evaluation'].items() 
               if k.endswith('_improvement_pct') and isinstance(v, (int, float))]
        if vals:
            metrics['dense_improvement_pct'] = round(sum(vals) / len(vals), 4)
    
    # Sparse metrics
    if 'sparse' in summary:
        if 'evaluation' in summary['sparse']:
            vals = [v for k, v in summary['sparse']['evaluation'].items() 
                   if k.endswith('_improvement_pct') and isinstance(v, (int, float))]
            if vals:
                metrics['sparse_improvement_pct'] = round(sum(vals) / len(vals), 4)
        
        if 'report' in summary['sparse']:
            report = summary['sparse']['report']
            param_red_ratio = report.get('param_reduction_ratio') or report.get('compact_report', {}).get('param_reduction_ratio', 0)
            metrics['param_reduction_pct'] = round((1 - param_red_ratio) * 100, 2)
        
        # Time speedup
        comp = summary['sparse'].get('dense_vs_sparse_comparison', {}).get('machine', {})
        if 'denoising_seconds' in comp:
            metrics['time_speedup'] = round(comp['denoising_seconds'].get('speedup_x', 1.0), 3)
        
        # VRAM delta
        if 'allocated_net_bytes' in comp:
            metrics['vram_delta_bytes'] = comp['allocated_net_bytes'].get('delta_pct', 0.0)
    
    return metrics

def print_results(results):
    """Print formatted comparison table"""
    print("\n" + "="*100)
    print("QUICK VALIDATION: SPARSITY METHODS COMPARISON")
    print("="*100)
    
    print(f"\n{'Method':<40} | {'Dense Impr':<12} | {'Sparse Impr':<12} | {'Param Red':<10} | {'Time 1.0x':<10} | {'VRAM Δ%':<10}")
    print("-"*100)
    
    for method, result in zip(METHODS, results):
        if result['status'] == 'failed':
            print(f"{method['label']:<40} | {'[FAILED]':<12} | {result['error'][:30]:<12} | {'—':<10} | {'—':<10} | {'—':<10}")
        else:
            metrics = result['metrics']
            print(f"{method['label']:<40} | {str(metrics['dense_improvement_pct']):<12} | {str(metrics['sparse_improvement_pct']):<12} | {str(metrics['param_reduction_pct']):<10} | {str(metrics['time_speedup']):<10} | {str(metrics['vram_delta_bytes']):<10}")
    
    print("="*100)
    print("\nLegend:")
    print("  Dense Impr: Average improvement on downstream models (dense backbone)")
    print("  Sparse Impr: Average improvement on downstream models (sparse backbone)")
    print("  Param Red: Parameter reduction percentage (100 - reduction_ratio%)")
    print("  Time 1.0x: Denoising speedup (1.0x = no change, >1.0x = faster sparse)")
    print("  VRAM Δ%: Change in VRAM allocated during denoising (%)")

def main():
    """Run quick validation on all methods"""
    results = []
    
    for i, method in enumerate(METHODS, 1):
        print(f"\n[{i}/{len(METHODS)}] {method['label']}")
        print("-" * 80)
        
        try:
            cfg = build_config(method['config'])
            summary = run_tabular_experiment(cfg)
            metrics = extract_metrics(summary)
            
            results.append({
                'method': method['label'],
                'status': 'success',
                'metrics': metrics,
            })
            
            print(f"✓ Completed - Dense Impr: {metrics['dense_improvement_pct']}, Sparse Impr: {metrics['sparse_improvement_pct']}")
        
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({
                'method': method['label'],
                'status': 'failed',
                'error': str(e),
            })
    
    # Save results
    out_file = Path('out/quick_validation_results.json')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding='utf-8')
    
    # Print summary table
    print_results(results)
    
    print(f"\n✓ Results saved to {out_file}\n")

if __name__ == "__main__":
    main()
