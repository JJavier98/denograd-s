# Plan Canónico: DenoGrad + Energy Sparse + Transformers

Este es el plan de referencia vigente del repositorio. Sustituye y absorbe los planes anteriores de piloto sparse y de poda híbrida on-training. Solo se conservan aquí las restricciones, parámetros operativos y decisiones que siguen siendo útiles para el estudio actual; los hitos históricos y estados antiguos quedan descartados como guía de ejecución.

## Objetivo

Diseñar, ejecutar y documentar una campaña experimental completa para evaluar si DenoGrad se beneficia de backbones sparse obtenidos mediante un pipeline híbrido:

entrenamiento con máscara -> compactación física estructural -> fine-tuning corto

La comparación principal debe aislar el efecto del backbone sparse sobre el proceso de denoising, manteniendo fijo el benchmark downstream. El foco del estudio no es demostrar que un predictor sparse sea mejor por sí mismo, sino medir si un backbone sparse hace más útil, eficiente y práctico a DenoGrad.

## Hipótesis del estudio

1. El uso de sparsity energy-aware con compactación física real puede mantener o mejorar la utilidad práctica de DenoGrad frente al backbone denso.
2. Parte de esa ventaja puede venir no solo de calidad downstream, sino también de una menor huella en parámetros, tamaño del modelo y coste de inferencia o denoising.
3. El mejor energy target y el mejor backbone pueden no ser universales; el resultado final puede requerir una recomendación por dominio o por familia de modelo.

## Alcance experimental

### Dominios y datasets

- Tabular: ejecutar el estudio completo sobre todos los datasets tabulares del repositorio.
- Series temporales: ejecutar el estudio completo sobre todos los datasets temporales del repositorio.
- Datasets faro para screening y ablación inicial:
  - tabular: house_prices
  - series temporales: daily_climate

### Familias de backbone candidatas

- Tabular:
  - DNN como baseline principal.
  - FTTransformer como baseline Transformer principal.
- Series temporales:
  - LSTM como baseline principal.
  - Transformer temporal con cadena de fallback explícita:
    - iTransformer como opción preferente.
    - PatchTST si iTransformer falla por integración, memoria, convergencia o incompatibilidad del adapter.
    - Transformer Vanilla si iTransformer y PatchTST no son viables.

### Qué queda fuera del alcance principal

- La sensibilidad al ruido no forma parte de esta campaña principal, porque ya está estudiada en el artículo base de DenoGrad, arXiv:2511.10161.
- Los hitos históricos de planes anteriores no se reutilizan como criterio de avance.
- Sparse-from-scratch, xLSTM y variantes complejas de TimeSeriesLibrary quedan fuera de la primera oleada salvo necesidad justificada posterior.

## Configuración operativa de referencia

### DenoGrad

- eta = 0.01 como valor por defecto.
- tau = 0.1 como valor por defecto.
- max_iters en el rango 100-200 con early stopping por umbral.
- Comparaciones dense vs sparse siempre con exactamente los mismos hiperparámetros de DenoGrad.

### Entrenamiento del backbone

- 100 épocas como configuración estándar.
- Early stopping con patience = 15.
- Misma seed, mismo split y mismo protocolo de entrenamiento al comparar dense vs sparse.

### Pipeline sparse canónico

- Método during-training: sparse_on_training.
- Umbral por capa: threshold_l = k * sigma_l.
- k = 0.1 como configuración inicial de referencia.
- layer_priority_strength = 1.0 para hacer la poda más agresiva en capas finales que en capas iniciales.
- Compactación post-training: structured_compact con selección energy-aware.
- Fine-tuning corto sobre modelo compactado:
  - 10-15 épocas.
  - patience corta.
  - nuevo optimizador con LR reducido, recomendado lr / 10 respecto al entrenamiento principal.
  - sin modificar trainer.py; la orquestación debe hacerse desde experiment_runner.

### Restricciones de implementación heredadas y vigentes

- El fine-tuning debe reutilizar Trainer con un nuevo optimizador y el modelo ya podado como punto de partida.
- Deben guardarse artefactos intermedios por fase: modelo masked, modelo compactado y modelo finetuned.
- La compactación debe ser física y confirmar reducción real de parámetros, no solo aparición de ceros en pesos.
- El benchmark downstream debe permanecer fijo en la comparación principal.

## Métricas obligatorias

### Calidad de denoising

- SWD.
- Correlación de Pearson.
- Si la implementación lo permite, preservación de estructura de correlación y análisis cualitativo cuando haya desacople entre benchmark y métricas de denoising.

### Calidad downstream

- Benchmark improvement respecto al baseline noisy.
- MSE, RMSE y MAE cuando estén disponibles.
- Wins, ties y losses frente a dense por dataset.

### Eficiencia

- Número total de parámetros.
- Parámetros no nulos y densidad efectiva cuando aplique.
- Tamaño del modelo en bytes y en formato humano (KB, MB, GB).
- Tiempo de inferencia.
- Tiempo total del proceso de denoising.
- VRAM: allocated, reserved, pico y allocated neta.
- Speedup o slowdown relativo frente a dense cuando esté justificado.

### Métricas estructurales y de trazabilidad

- Número de unidades retenidas o podadas tras compactación.
- param_reduction_ratio real.
- Registro de artefactos por fase y manifiesto por run.

## Visualizaciones obligatorias

- Curvas de ablación energy target vs benchmark improvement.
- Curvas energy target vs SWD.
- Curvas energy target vs correlación.
- Barras dense vs sparse por dataset para benchmark, SWD, correlación, parámetros, MB y tiempo de inferencia.
- Scatter o burbujas tipo Pareto para calidad vs coste.
- Heatmaps dataset x backbone x energy target.
- Gráficas cualitativas clean/noisy/denoised para casos representativos.
- La figura dense_vs_sparse principal debe separar memoria y tiempo, o usar doble eje y de forma clara.
- allocated_net_bytes no debe aparecer en la figura principal; si hace falta, debe quedar en tablas o anexos.

## Organización de resultados

La campaña debe dejar una estructura clara y navegable:

- resultados raw por run,
- agregados del estudio,
- tablas finales,
- figuras finales,
- reportes narrativos,
- manifiesto maestro del estudio.

Debe mantenerse un índice maestro con enlaces a:

- seed,
- config,
- resumen agregado,
- figuras derivadas,
- rutas de artefactos por fase.

## Fases del plan

### Fase 0 — Consolidación de infraestructura

- Formalizar el manifiesto maestro del estudio.
- Normalizar el contrato de resultados por run.
- Asegurar agregación multi-run desde el índice global existente.
- Garantizar persistencia y reutilización segura de checkpoints, datasets noisy y datasets denoised.

### Fase 1 — Integración de backbones Transformer

- Verificar FTTransformer en tabular con paso completo dense y sparse.
- Integrar Transformer temporal con fallback ordenado:
  - iTransformer,
  - PatchTST,
  - Transformer Vanilla.
- Registrar siempre el motivo técnico del fallback si ocurre.

### Fase 2 — Screening de backbones en datasets faro

- Tabular en house_prices: DNN vs FTTransformer.
- Temporal en daily_climate: LSTM vs Transformer temporal adoptado.
- Ejecutar dense y sparse con energy target inicial = 0.95.
- Ejecutar 3 seeds por dataset.
- Selección con este orden:
  1. benchmark improvement,
  2. SWD y correlación,
  3. eficiencia en parámetros, MB e inferencia.

### Fase 3 — Ablación de energy targets

- Ejecutar ablación en house_prices y daily_climate.
- Targets obligatorios:
  - 0.80,
  - 0.85,
  - 0.90,
  - 0.95.
- Ejecutar 3 seeds por combinación.
- Registrar dense vs sparse y todos los deltas asociados.
- Generar frentes de Pareto calidad-eficiencia.

### Fase 4 — Selección de configuración ganadora

- Elegir ganador tabular y ganador temporal.
- Si el mejor backbone o energy target diverge por familia, conservar también una recomendación por familia.
- No forzar una regla universal si los resultados no la sostienen.

### Fase 5 — Campaña completa sobre todos los datasets

- Ejecutar la mejor configuración tabular en todos los datasets tabulares.
- Ejecutar la mejor configuración temporal en todos los datasets temporales.
- Ejecutar 3 seeds por dataset.
- Permitir fallback explícito a un backbone estable cuando haya fallo real de memoria, convergencia o compatibilidad.

### Fase 6 — Tablas comparativas finales

- Tablas dense vs sparse por dataset y dominio.
- Tabla de wins/ties/losses.
- Tabla de selección de backbone.
- Tabla de recomendaciones finales:
  - recomendación global,
  - recomendación por dominio,
  - recomendación por familia si difiere.

### Fase 7 — Gráficas y narrativa visual

- Curvas, heatmaps, scatter de Pareto y paneles cualitativos.
- Figura ejecutiva final de calidad vs coste computacional.
- Presentación explícita de los casos donde Transformer mejora calidad pero no compensa coste, si ocurre.

### Fase 8 — Cierre, trazabilidad y publicación interna

- Reporte ejecutivo.
- Reporte técnico.
- Snapshot de configuración ganadora por dominio.
- Snapshot de mejor configuración por familia si difiere.
- Snapshot del grid completo de ablación.

### Fase 9 — Extensiones recomendadas no bloqueantes

- Robustez estadística: mediana, ranking promedio y pruebas sencillas de significancia o bootstrap cuando compense.
- Impacto de despliegue: análisis de memoria, tamaño y tiempo de inferencia como proxy de viabilidad real.
- Análisis de fallos: documentar datasets o backbones donde sparse empeora de forma clara.
- Contexto bibliográfico: citar explícitamente que la sensibilidad al ruido ya está cubierta en el paper base de DenoGrad.

## Verificación mínima antes de escalar

1. Smoke test dense y sparse en house_prices con DNN y FTTransformer.
2. Smoke test dense y sparse en daily_climate con LSTM y el Transformer temporal finalmente adoptado.
3. Confirmación de que cada run exporta el bundle mínimo de métricas y artefactos.
4. Verificación de igualdad de split, seed de ruido, benchmark downstream y criterio de parada entre dense y sparse.
5. Revisión de frentes de Pareto antes de elegir configuración ganadora.
6. Revisión cualitativa explícita cuando benchmark improvement y SWD o correlación diverjan.

## Decisiones vigentes

- Archivo canónico del plan: este documento.
- Pipeline sparse principal: sparse_on_training + structured_compact + fine-tuning corto.
- Configuración base during-training: k = 0.1 y layer_priority_strength = 1.0.
- Configuración estadística de campaña: 3 seeds por dataset y agregación por media con dispersión.
- Fallback temporal oficial: iTransformer -> PatchTST -> Transformer Vanilla.
- El estudio no repite sensibilidad al ruido; se cita el artículo base de DenoGrad para ese punto.
- Las afirmaciones de speedup real solo se hacen si benchmark_inference las respalda de forma consistente.

## Archivos relevantes

- src/libs/experiment_runner.py
- src/libs/sparsity.py
- src/libs/trainer.py
- src/libs/benchmark.py
- src/libs/evaluation.py
- src/libs/profiling.py
- src/models/__init__.py
- src/models/ft_transformer.py
- src/models/lstm.py
- src/models/TimeSeriesLibrary/models/iTransformer.py
- src/models/TimeSeriesLibrary/models/PatchTST.py
- src/models/TimeSeriesLibrary/models/Transformer.py
- out/meta/indexes/runs.jsonl
- out/meta/summaries
