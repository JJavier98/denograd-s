# Checklist Ejecutiva - Energy Sparse + Transformers

## Fase A - Screening inicial

- [ ] Ejecutar screening tabular en house_prices con dnn y transformer (dense + sparse_energy)
- [ ] Ejecutar screening temporal en daily_climate con lstm y transformer temporal (dense + sparse_energy)
- [ ] Verificar fallback temporal iTransformer -> PatchTST -> Transformer Vanilla
- [ ] Confirmar 3 seeds por experimento (42, 52, 62)
- [ ] Guardar resultados agregados JSON y CSV en out/meta/summaries
- [ ] Validar que dense vs sparse usa mismo split, seed de ruido y benchmark downstream

## Fase B - Ablacion energy target

- [ ] Ejecutar grid energy_target = 0.80, 0.85, 0.90, 0.95
- [ ] Ejecutar grid en house_prices y daily_climate
- [ ] Ejecutar 3 seeds por combinacion
- [ ] Generar resumen agregado por dominio y backbone
- [ ] Generar frente de Pareto calidad-coste
- [ ] Seleccionar candidatos ganadores para campana completa

## Fase C - Campana completa

- [ ] Ejecutar todos los datasets tabulares con ganador tabular (dense + sparse)
- [ ] Ejecutar todos los datasets temporales con ganador temporal (dense + sparse)
- [ ] Aplicar fallback documentado si un backbone falla
- [ ] Consolidar resultados por dataset, dominio y global

## Fase D - Metricas obligatorias

- [ ] Benchmark improvement
- [ ] SWD
- [ ] Correlacion
- [ ] Parametros totales y no nulos
- [ ] Tamano de modelo (bytes y formato humano)
- [ ] Tiempo de inferencia
- [ ] Tiempo de denoising
- [ ] VRAM (allocated, reserved, pico, allocated neta)

## Fase E - Visualizacion y reporte

- [ ] Tabla dense vs sparse por dataset
- [ ] Tabla wins/ties/losses
- [ ] Tabla recomendacion global y por dominio/familia
- [ ] Curvas de ablacion energy vs benchmark/SWD/correlacion
- [ ] Graficas dense vs sparse de eficiencia
- [ ] Scatter/burbujas Pareto
- [ ] Heatmap dataset x backbone x energy
- [ ] Reporte ejecutivo
- [ ] Reporte tecnico

## Criterios de cierre

- [ ] Existen recomendaciones finales trazables por dominio
- [ ] Hay evidencia de robustez con 3 seeds por dataset
- [ ] No hay dependencia de hitos historicos de planes antiguos
- [ ] El plan canónico usado es DENOGRAD_SPARSITY_EXPERIMENT_PLAN.md
