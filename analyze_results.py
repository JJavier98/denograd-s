#!/usr/bin/env python3
"""
Post-Processing & Analysis: Generate visualizations and reports from experimental results
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import statistics

def load_results(results_file: Path) -> List[Dict[str, Any]]:
    """Load experimental results from JSON"""
    if not results_file.exists():
        return []
    return json.loads(results_file.read_text())

def generate_comparison_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate high-level summary comparing all methods"""
    successful = [r for r in results if r.get('status') == 'success']
    
    summary = {
        'total_runs': len(results),
        'successful': len(successful),
        'failed': len(results) - len(successful),
        'by_method': {},
    }
    
    # Group by method label
    for result in successful:
        method = result['method']
        if method not in summary['by_method']:
            summary['by_method'][method] = {
                'runs': 0,
                'metrics': {
                    'dense_improvement': [],
                    'sparse_improvement': [],
                    'param_reduction': [],
                    'time_speedup': [],
                    'vram_delta': [],
                }
            }
        
        summary['by_method'][method]['runs'] += 1
        metrics = result['metrics']
        
        if metrics['dense_improvement_pct'] is not None:
            summary['by_method'][method]['metrics']['dense_improvement'].append(metrics['dense_improvement_pct'])
        if metrics['sparse_improvement_pct'] is not None:
            summary['by_method'][method]['metrics']['sparse_improvement'].append(metrics['sparse_improvement_pct'])
        if metrics['param_reduction_pct'] is not None:
            summary['by_method'][method]['metrics']['param_reduction'].append(metrics['param_reduction_pct'])
        if metrics['time_speedup'] is not None:
            summary['by_method'][method]['metrics']['time_speedup'].append(metrics['time_speedup'])
        if metrics['vram_delta_bytes'] is not None:
            summary['by_method'][method]['metrics']['vram_delta'].append(metrics['vram_delta_bytes'])
    
    # Compute averages
    for method_data in summary['by_method'].values():
        base_metric_names = ['dense_improvement', 'sparse_improvement', 'param_reduction', 'time_speedup', 'vram_delta']
        for metric_name in base_metric_names:
            values = method_data['metrics'].get(metric_name, [])
            if values:
                method_data['metrics'][f'{metric_name}_avg'] = round(statistics.mean(values), 4)
                method_data['metrics'][f'{metric_name}_stdev'] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
    
    return summary

def generate_markdown_report(summary: Dict[str, Any]) -> str:
    """Generate markdown report of results"""
    report = []
    report.append("# Experimental Results Report\n")
    report.append(f"## Summary")
    report.append(f"- **Total Runs**: {summary['total_runs']}")
    report.append(f"- **Successful**: {summary['successful']}")
    report.append(f"- **Failed**: {summary['failed']}\n")
    
    report.append("## Results by Method\n")
    
    for method, data in summary['by_method'].items():
        report.append(f"### {method}\n")
        report.append(f"- **Runs**: {data['runs']}\n")
        
        metrics = data['metrics']
        for metric_name in ['dense_improvement', 'sparse_improvement', 'param_reduction', 'time_speedup', 'vram_delta']:
            avg_key = f'{metric_name}_avg'
            stdev_key = f'{metric_name}_stdev'
            if avg_key in metrics:
                stdev = metrics.get(stdev_key, 0)
                value = metrics[avg_key]
                report.append(f"- **{metric_name.replace('_', ' ').title()}**: {value} (± {stdev})")
        
        report.append("")
    
    return "\n".join(report)

def main():
    """Generate analysis and reports"""
    results_file = Path('out/quick_validation_results.json')
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run 'python run_quick_validation.py' first to generate results.")
        return
    
    print(f"[LOAD] {results_file}")
    results = load_results(results_file)
    
    print(f"[ANALYZE] {len(results)} total runs")
    summary = generate_comparison_summary(results)
    
    print(f"[GENERATE] Markdown report")
    report = generate_markdown_report(summary)
    
    # Save report
    report_file = Path('out/quick_validation_report.md')
    report_file.write_text(report, encoding='utf-8')
    
    print(f"[SAVED] {report_file}")
    print("\n" + report)

if __name__ == "__main__":
    main()
