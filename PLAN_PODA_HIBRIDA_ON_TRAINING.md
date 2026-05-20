# Plan sustituido

Este archivo ya no es una fuente de verdad independiente.

Su contenido vigente ha sido absorbido por el plan canónico del repositorio:

- [DENOGRAD_SPARSITY_EXPERIMENT_PLAN.md](DENOGRAD_SPARSITY_EXPERIMENT_PLAN.md)

Las restricciones que se conservan allí desde este plan son:

- pipeline híbrido: entrenamiento con máscara -> compactación física -> fine-tuning corto,
- sparse_on_training con k = 0.1,
- prioridad de poda por profundidad con layer_priority_strength = 1.0,
- fine-tuning corto desde experiment_runner reutilizando Trainer,
- reducción real de parámetros como requisito de compactación.
