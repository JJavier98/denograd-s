#!/usr/bin/env python3
"""Full experimental matrix with updated sparsity protocol."""

import json
import subprocess
from pathlib import Path
from datetime import datetime

EXPERIMENTS = [
    # TABULAR EXPERIMENTS
    {
        "domain": "tabular",
        "dataset": "house_prices",
        "method": "dense",
        "seed": 42,
        "gpu": 0,
    },
    {
        "domain": "tabular",
        "dataset": "house_prices",
        "method": "post_training_ratio_priority",
        "ratio": 0.2,
        "layer_priority_strength": 1.0,
        "seed": 42,
        "gpu": 0,
    },
    {
        "domain": "tabular",
        "dataset": "house_prices",
        "method": "sparse_on_training",
        "k": 0.1,
        "layer_priority_strength": 1.0,
        "seed": 42,
        "gpu": 0,
    },
    # TIME SERIES EXPERIMENTS
    {
        "domain": "ts",
        "dataset": "daily_climate",
        "method": "dense",
        "seed": 42,
        "gpu": 0,
    },
    {
        "domain": "ts",
        "dataset": "daily_climate",
        "method": "post_training_ratio_priority",
        "ratio": 0.2,
        "layer_priority_strength": 1.0,
        "seed": 42,
        "gpu": 0,
    },
    {
        "domain": "ts",
        "dataset": "daily_climate",
        "method": "sparse_on_training",
        "k": 0.1,
        "layer_priority_strength": 1.0,
        "seed": 42,
        "gpu": 0,
    },
]

def build_tabular_config(exp):
    """Build config for tabular experiment"""
    return {
        'artifacts_dir': 'out',
        'version': f"{exp['domain']}_{exp['dataset']}_{exp['method']}_s{exp['seed']}_gpu{exp['gpu']}",
        'seed': exp['seed'],
        'device': f"cuda:{exp['gpu']}",
        'dataset': {
            'path': f"data/tabular/{exp['dataset']}/kc_house_data.csv",
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
            'knn': {'enabled': True, 'n_neighbors': 7},
            'xgboost': {'enabled': True, 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.08},
            'tabpfn': {'enabled': False},
            'dnn': {'enabled': True, 'max_epochs': 100, 'hidden_dims': [128, 64, 32], 'patience': 15},
        },
        'sparsity': _build_sparsity_cfg(exp),
    }

def build_ts_config(exp):
    """Build config for time series experiment"""
    return {
        'artifacts_dir': 'out',
        'version': f"{exp['domain']}_{exp['dataset']}_{exp['method']}_s{exp['seed']}_gpu{exp['gpu']}",
        'seed': exp['seed'],
        'device': f"cuda:{exp['gpu']}",
        'dataset': {
            'path': f"data/time_series/{exp['dataset']}/DailyDelhiClimateTrain.csv",
            'test_path': f"data/time_series/{exp['dataset']}/DailyDelhiClimateTest.csv",
            'target_col': 'meantemp',
            'noise_std': 0.05,
            'batch_size': 64,
            'seq_len': 96,
            'pred_len': 12,
        },
        'model': {
            'name': 'lstm',
            'input_dim': 4,
            'hidden_dim': 64,
            'num_layers': 2,
            'output_dim': 4,
            'lr': 1e-3,
            'patience': 15,
            'max_epochs': 100,
        },
        'denograd': {
            'nrr': 0.01,
            'threshold': 0.1,
            'max_iters': 150,
            'batch_size': 512,
        },
        'benchmark': {
            'dnn': {'enabled': True, 'hidden_dims': [64, 32], 'max_epochs': 100, 'patience': 15},
            'lstm': {'enabled': True, 'hidden_dim': 64, 'num_layers': 2, 'max_epochs': 100, 'patience': 15},
        },
        'sparsity': _build_sparsity_cfg(exp),
    }

def _build_sparsity_cfg(exp):
    """Build sparsity config based on experiment method"""
    if exp['method'] == 'dense':
        return {'enabled': False}
    elif exp['method'] == 'post_training_ratio_priority':
        return {
            'enabled': True,
            'method': 'post_training_ratio_priority',
            'ratio': exp['ratio'],
            'layer_priority_strength': exp.get('layer_priority_strength', 1.0),
            'include_bias': False,
        }
    elif exp['method'] == 'sparse_on_training':
        return {
            'enabled': True,
            'method': 'sparse_on_training',
            'k': exp['k'],
            'layer_priority_strength': exp.get('layer_priority_strength', 1.0),
            'include_bias': False,
        }
    else:
        raise ValueError(f"Unknown sparsity method: {exp['method']}")

def run_experiment(exp):
    """Execute single experiment and capture results"""
    import sys
    sys.path.insert(0, '/home/jjavier98/denograd-s')
    
    from src.libs.experiment_runner import run_tabular_experiment, run_ts_experiment
    
    print(f"\n{'='*70}")
    print(f"[RUN] {exp['domain'].upper()}: {exp['dataset']} × {exp['method']} (seed={exp['seed']}, gpu={exp['gpu']})")
    print(f"{'='*70}")
    
    try:
        if exp['domain'] == 'tabular':
            cfg = build_tabular_config(exp)
            summary = run_tabular_experiment(cfg)
        else:  # ts
            cfg = build_ts_config(exp)
            summary = run_ts_experiment(cfg)
        
        # Extract key metrics
        result = {
            'exp': exp,
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'dense_avg_improvement': None,
            'sparse_avg_improvement': None,
            'sparse_comparison': None,
        }
        
        # Dense eval
        if 'evaluation' in summary:
            vals = [v for k, v in summary['evaluation'].items() 
                   if k.endswith('_improvement_pct') and isinstance(v, (int, float))]
            result['dense_avg_improvement'] = round(sum(vals) / len(vals), 4) if vals else None
        
        # Sparse eval
        if 'sparse' in summary and 'evaluation' in summary['sparse']:
            vals = [v for k, v in summary['sparse']['evaluation'].items() 
                   if k.endswith('_improvement_pct') and isinstance(v, (int, float))]
            result['sparse_avg_improvement'] = round(sum(vals) / len(vals), 4) if vals else None
        
        # Sparse comparison
        if 'sparse' in summary and 'dense_vs_sparse_comparison' in summary['sparse']:
            result['sparse_comparison'] = summary['sparse']['dense_vs_sparse_comparison'].get('machine')
        
        return result
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {
            'exp': exp,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(e),
        }

def main():
    """Run full experimental matrix"""
    results_file = Path('out/matrix_results.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] Starting experiment...")
        result = run_experiment(exp)
        all_results.append(result)
        
        # Save incrementally
        results_file.write_text(json.dumps(all_results, indent=2), encoding='utf-8')
        print(f"[SAVED] {results_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENTAL MATRIX SUMMARY")
    print(f"{'='*70}")
    
    successful = [r for r in all_results if r['status'] == 'success']
    failed = [r for r in all_results if r['status'] == 'failed']
    
    print(f"✓ Successful: {len(successful)}/{len(all_results)}")
    print(f"✗ Failed: {len(failed)}/{len(all_results)}")
    
    # Group results by method
    by_method = {}
    for r in successful:
        method = r['exp']['method']
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(r)
    
    print(f"\nResults by method:")
    for method, runs in by_method.items():
        avg_sparse_imp = [r['sparse_avg_improvement'] for r in runs if r['sparse_avg_improvement'] is not None]
        avg_dense_imp = [r['dense_avg_improvement'] for r in runs if r['dense_avg_improvement'] is not None]
        
        print(f"  {method}:")
        if avg_dense_imp:
            print(f"    Dense avg improvement: {sum(avg_dense_imp)/len(avg_dense_imp):.4f}")
        if avg_sparse_imp:
            print(f"    Sparse avg improvement: {sum(avg_sparse_imp)/len(avg_sparse_imp):.4f}")
    
    print(f"\n[DONE] Full results saved to {results_file}\n")

if __name__ == "__main__":
    main()
