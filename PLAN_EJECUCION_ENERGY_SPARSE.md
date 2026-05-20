# Matriz Operativa de Ejecucion - Energy Sparse + Transformers

Este documento traduce el plan canonico a una secuencia ejecutable de trabajo, con orden recomendado, scripts y salidas esperadas.

## Convenciones

- Seeds de campaña: 3 por dataset (recomendado: 42, 52, 62).
- DenoGrad base: eta=0.01, tau=0.1, max_iters=150 (ajustable 100-200).
- Entrenamiento base: 100 epocas, patience=15.
- During-training sparse: sparse_on_training con k=0.1 y layer_priority_strength=1.0.
- Post-training: structured_compact con seleccion energy.
- Fine-tuning: 12 epocas, patience 5, lr_scale 0.1.

## Fase A - Screening inicial (datasets faro)

Objetivo: seleccionar candidatos para ablacion en house_prices y daily_climate.

### A1. Tabular (house_prices)

- Backbones:
  - dnn
  - transformer (FTTransformer)
- Condiciones por backbone:
  - dense
  - sparse_energy (energy_target inicial: 0.95)
- Seeds:
  - 42, 52, 62

### A2. Time series (daily_climate)

- Backbone baseline:
  - lstm
- Backbone transformer temporal con fallback:
  - itransformer -> patchtst -> transformer_vanilla
- Condiciones por backbone:
  - dense
  - sparse_energy (energy_target inicial: 0.95)
- Seeds:
  - 42, 52, 62

### Script recomendado

- Script de ejecucion: src/run_screening_energy_phase1.py

Ejemplo:

```bash
python src/run_screening_energy_phase1.py \
  --seeds 42,52,62 \
  --energy-target 0.95 \
  --version-prefix phase1_screening
```

### Salidas esperadas

- out/meta/summaries/phase1_screening_results.json
- out/meta/summaries/phase1_screening_table.csv

## Fase B - Ablacion de energy target

Objetivo: elegir energy target ganador por dominio y por familia si diverge.

### Grid obligatorio

- Energy targets: 0.80, 0.85, 0.90, 0.95
- Datasets faro:
  - house_prices
  - daily_climate
- Backbones: supervivientes de Fase A
- Seeds: 42, 52, 62

### Salidas esperadas

- out/meta/summaries/ablation_energy_results.json
- out/meta/summaries/ablation_energy_table.csv
- out/meta/summaries/ablation_energy_pareto.json

## Fase C - Campana completa todos los datasets

Objetivo: validar generalizacion del ganador tabular y temporal.

### C1. Tabular completo

- Datasets: todos en data/tabular/*
- Config: mejor backbone + mejor energy target tabular
- Condiciones: dense vs sparse_energy
- Seeds: 42, 52, 62

### C2. Time series completo

- Datasets: todos en data/time_series/*
- Config: mejor backbone + mejor energy target temporal
- Condiciones: dense vs sparse_energy
- Seeds: 42, 52, 62

### Salidas esperadas

- out/meta/summaries/full_campaign_results.json
- out/meta/summaries/full_campaign_table.csv
- out/meta/summaries/final_recommendations.json

## Fase D - Reporting y visualizacion

Objetivo: producir entrega tecnica y ejecutiva.

### Tablas minimas

- Dense vs sparse por dataset y dominio
- Wins/ties/losses
- Seleccion de backbone
- Recomendacion global y por dominio/familia

### Graficas minimas

- Energy target vs benchmark improvement
- Energy target vs SWD
- Energy target vs correlacion
- Dense vs sparse: parametros, MB, inferencia, denoising
- Pareto calidad-coste
- Heatmap dataset x backbone x energy target

## Criterios de decision

1. Benchmark improvement (prioridad alta)
2. SWD y correlacion
3. Eficiencia: params, MB, inferencia, denoising

Si hay divergencia consistente entre familias, se reporta recomendacion por dominio/familia y no una regla universal.

## Notas de alcance

- La sensibilidad al ruido no se repite en esta campana principal (ya cubierta en arXiv:2511.10161).
- Claims de speedup real solo cuando benchmark_inference confirme mejora consistente.
