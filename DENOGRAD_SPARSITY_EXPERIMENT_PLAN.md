# Plan: DenoGrad + Sparsity Pilot

Montar en denograd-s un piloto reproducible que permita comprobar si DenoGrad mejora más cuando el backbone es sparse que cuando es denso. La comparación principal debe aislar el efecto del backbone sparse sobre el denoising, manteniendo fijo el benchmark downstream. La recomendación técnica es híbrida: masking con PyTorch/torch.ao para baseline y sparse-from-scratch, Torch-Pruning para pruning estructural, y torchao solo para 2:4 o block sparsity cuando haya soporte real de hardware.

## Tipos de sparsity

- Masking o pruning no estructural: se ponen ceros en pesos individuales sin cambiar la forma de las capas. Es el punto de partida más barato para comparar hipótesis.
- Pruning estructural: se eliminan unidades completas, canales, neuronas o heads. Reduce la arquitectura efectiva y suele dar ahorro real de memoria y latencia.
- 2:4 o semi-structured sparsity: en cada bloque de 4 pesos, 2 se ponen a cero siguiendo el patrón soportado por algunos kernels acelerados. Es la opción más interesante si el hardware y la librería la soportan.
- Sparse-from-scratch: se entrena un modelo ya sparse desde cero o con parametrización de máscara durante todo el entrenamiento. Es una opción válida, pero no será el primer foco del estudio.

## Steps

### 1. Fase 0 — Infraestructura reproducible base

- Portar o reimplementar en denograd-s las piezas experimentales que ya existen en el repo hermano: trainer, utils, evaluación de cambios, caché de noisy/denoised y patrón de configuración/orquestación.
- Hacer ejecutable el flujo real que hoy está solo esbozado en src/benchmark.py: cargar clean, inyectar ruido, benchmark noisy, entrenar backbone, ejecutar DenoGrad, benchmark sobre denoised y registrar artefactos.
- Definir un esquema único de resultados por run: dominio, dataset, seed, backbone, régimen de sparsity, ratio/patrón, hiperparámetros de DenoGrad, métricas before/after, tiempos y artefactos.
- Definir desde el inicio una política de persistencia de estados intermedios para evitar recomputación: checkpoints de modelos densos y sparse, datasets noisy y denoised, salidas intermedias de DenoGrad, resúmenes de métricas y metadatos de ejecución.

### 2. Fase 1 — Delimitar matriz de modelos y alcance piloto

- Tier A del piloto: TabularDNN, FTTransformerBackbone, DLinearAdapter y MultivariateLSTM.
- Tier B tras validación del piloto: CNN/AutoEncoder para tabular y xLSTM para series temporales.
- Comparación primaria: sparsificar solo el backbone usado por DenoGrad y mantener fijo el benchmark downstream para no confundir calidad de denoising con calidad del predictor.
- Comparación secundaria opcional: sparsificar también los modelos downstream una vez que la hipótesis principal esté validada.
- Matriz recomendada por modelo:
  - TabularDNN: masking, pruning estructural, 2:4 y sparse-from-scratch.
  - FTTransformerBackbone: masking en Linear, pruning estructural de FFN/heads con reglas conservadoras, 2:4 en capas lineales y sparse-from-scratch solo después de estabilizar masking/estructural.
  - DLinearAdapter: masking, estructural, 2:4 y sparse-from-scratch.
  - MultivariateLSTM: masking y sparse-from-scratch en el piloto; pruning estructural de hidden units solo como extensión controlada.
- Excluir del primer piloto el sparse-from-scratch y el pruning estructural de xLSTM y de modelos complejos de TimeSeriesLibrary por coste de integración y riesgo de mezclar demasiadas variables.

### 3. Fase 2 — Capa de abstracción de sparsity

- Introducir una interfaz común por régimen: preparar modelo denso, aplicar masking, aplicar pruning estructural, aplicar 2:4/block, entrenar sparse-from-scratch y exportar metadatos de sparsity.
- Backend recomendado por régimen:
  - Masking baseline: utilidades de pruning de PyTorch / torch.ao para mantener forma y facilitar ablaciones.
  - Estructural: Torch-Pruning con example_inputs, ignored_layers y reglas específicas por familia.
  - 2:4 / block: torchao en módulos lineales compatibles y solo cuando se pueda verificar activación de kernels sparse.
  - Sparse-from-scratch: dejarlo como opción futura del estudio, no como prioridad inicial.
- Añadir un barrido de sensibilidad por capa/modelo antes de lanzar experimentos grandes para identificar ratios seguros.

#### Compatibilidad con RTX 3070 Ti

- Con dos RTX 3070 Ti sí tiene sentido probar 2:4 como línea experimental, porque la arquitectura Ampere puede servir de base para sparsidad semi-estructurada si el stack de software lo soporta.
- Aun así, no hay que asumir speedup garantizado: la aceleración real depende de que la librería, el dtype y los kernels disponibles aprovechen el patrón 2:4.
- En este proyecto, 2:4 se debe tratar como una hipótesis de eficiencia a validar empíricamente, no como una promesa de mejora automática.
- Si la combinación concreta de GPU, PyTorch y torchao no activa kernels sparse, el experimento sigue siendo útil como comparación de calidad y memoria, aunque no como prueba de speedup de hardware.

### 4. Fase 3 — Protocolo experimental y reglas de justicia

- Para cada dataset/modelo/régimen/seed:
  - partir del mismo dataset clean y de la misma realización de ruido;
  - entrenar baseline denso sobre noisy;
  - entrenar o derivar la versión sparse con mismo optimizador, epochs, patience y seed;
  - ejecutar DenoGrad por separado con backbone denso y sparse usando exactamente los mismos hiperparámetros;
  - medir la huella del backbone antes y después de sparsification, tanto en memoria teórica de parámetros como en VRAM ocupada durante carga e inferencia cuando haya CUDA;
  - medir como métrica temporal prioritaria el tiempo del proceso de denoising de DenoGrad con backbone denso y sparse, bajo el mismo batch size y el mismo protocolo de calentamiento y repetición;
  - reconstruir train/val/test de forma consistente para noisy y denoised;
  - ejecutar el mismo benchmark downstream sobre ambas condiciones.
- Reutilizar artefactos persistidos siempre que la firma experimental coincida exactamente: dataset, split, seed, ruido, backbone, régimen sparse, ratio, hiperparámetros de DenoGrad y versión del código.
- Comparar tres niveles:
  - denso vs sparse sin denoising;
  - mejora por DenoGrad con backbone denso vs mejora por DenoGrad con backbone sparse;
  - mejora o paridad por unidad de sparsity, latencia, VRAM y parámetros.
- Usar 2 seeds en el piloto y reservar 3-5 seeds para la expansión completa.

### 4.1 Política de caché y artefactos reutilizables

- Persistir checkpoints de entrenamiento para:
  - backbone denso pre-DenoGrad;
  - backbone sparse post-pruning o sparse-from-scratch;
  - modelos reentrenados o fine-tuned tras sparsification cuando aplique.
- Persistir datasets y tensores intermedios para:
  - dataset clean de referencia usado en cada run;
  - dataset noisy generado con su seed;
  - dataset denoised con backbone denso;
  - dataset denoised con backbone sparse;
  - alineaciones o reconstrucciones temporales necesarias para series temporales.
- Persistir resultados y metadatos para:
  - métricas before/after por fase;
  - tiempos por etapa;
  - VRAM, memoria de parámetros, tamaño de checkpoint y densidad efectiva;
  - hashes o firmas de configuración para invalidar caché cuando cambie algo relevante.
- Separar artefactos por nivel para no contaminar comparaciones:
  - cache de datos;
  - cache de modelos;
  - cache de benchmarking;
  - cache de métricas y figuras.
- Guardar un manifiesto por run con rutas a todos los artefactos producidos para poder reanudar experimentos parciales y reconstruir tablas sin reejecutar todo.
- Permitir reanudación por etapa:
  - si existe checkpoint del backbone, saltar reentrenamiento;
  - si existe salida denoised válida, saltar DenoGrad;
  - si existe benchmark downstream con la misma firma, saltar reevaluación;
  - si existen métricas agregadas y artefactos base, regenerar solo tablas o figuras si hace falta.
- Añadir reglas de invalidación conservadoras: cualquier cambio en arquitectura, sparsity pattern, ratio, seed, split, hiperparámetros de DenoGrad o versión de preprocesado invalida el artefacto dependiente.

### 5. Fase 4 — Paquete de métricas para responder la hipótesis

- Métricas de calidad de denoising usando la referencia clean previa a la inyección sintética de ruido:
  - SWD(clean, noisy) y SWD(clean, denoised), junto con reducción relativa;
  - correlación de Pearson por variable entre clean-vs-noisy y clean-vs-denoised;
  - preservación de estructura de correlación: distancia entre matrices de correlación de clean, noisy y denoised;
  - para series temporales, perfiles de autocorrelación por variable hasta un lag K comparando clean, noisy y denoised;
  - opcionalmente MAE/MSE entre clean y denoised para sanity check.
- Métricas downstream:
  - MSE, RMSE y MAE por modelo;
  - mejora relativa sobre noisy baseline por cada modelo downstream;
  - score agregado por dataset contando wins/ties/losses o media normalizada.
- Métricas de eficiencia:
  - tiempo de entrenamiento, tiempo de denoising, tiempo de benchmark y latencia de inferencia del backbone cuando sea útil como diagnóstico secundario;
  - tiempo del proceso de denoising de DenoGrad antes y después de sparsification, reportado como media, desviación y speedup relativo;
  - tamaño del backbone en memoria: número de parámetros, número de parámetros no nulos, densidad efectiva, tamaño estimado en memoria de pesos y checkpoints;
  - VRAM del backbone antes y después de sparsification: memoria reservada, memoria asignada y pico de memoria durante inferencia o denoising cuando se ejecute en CUDA;
  - MACs/FLOPs estimados;
  - cuando proceda, speedup sparse real frente al denso.
- Métrica de interacción que responde la pregunta principal:
  - DenoGrad Benefit = error benchmark noisy - error benchmark denoised.
  - Sparse Synergy = Benefit con backbone sparse - Benefit con backbone denso.
- Criterio de éxito práctico adicional:
  - si DenoGrad-sparse no mejora de forma clara las métricas downstream pero las mantiene aproximadamente al mismo nivel que DenoGrad-denso, el resultado sigue siendo positivo si se obtiene una reducción sustancial de VRAM, tamaño de backbone y/o tiempo del proceso de denoising.

### 6. Fase 5 — Visualizaciones y reporting

- Gráficas de dataset:
  - overlays clean/noisy/denoised en series temporales;
  - comparaciones marginales o scatter en tabular;
  - heatmaps de correlación y heatmaps de diferencias;
  - barras de mejora en SWD/correlación.
- Gráficas de modelo:
  - curvas sparsity vs mejora downstream;
  - sparsity vs latencia/params/MACs/VRAM;
  - before/after de VRAM y tamaño del backbone para denso vs sparse;
  - before/after del tiempo del proceso de denoising de DenoGrad para denso vs sparse;
  - deltas de beneficio de DenoGrad entre denso y sparse por familia.
- Tablas del estudio:
  - resumen por dataset/modelo/régimen;
  - leaderboard del piloto en Sparse Synergy;
  - tabla de ablación de ratios y seeds.
- Tablas operativas adicionales:
  - inventario de artefactos reutilizados vs recomputados;
  - ahorro acumulado de tiempo gracias a caché, checkpoints y reutilización de datasets denoised.

### 7. Fase 6 — Plan de ejecución del piloto

- Datasets recomendados del piloto:
  - tabular: house_prices y parkinsons;
  - series temporales: daily_climate y microsoft_stock.
- Matriz recomendada del piloto:
  - TabularDNN en ambos datasets tabulares;
  - FTTransformerBackbone en ambos datasets tabulares cuando TabularDNN esté estabilizado;
  - DLinearAdapter y MultivariateLSTM en ambos datasets temporales.
- Regímenes mínimos del piloto:
  - masking por magnitud;
  - pruning estructural;
  - 2:4 acelerable si el hardware lo soporta;
  - sparse-from-scratch solo como fase posterior, no en la primera iteración.
- Ratios recomendados del piloto:
  - dos niveles para masking/estructural;
  - patrón fijo 2:4;
  - una densidad objetivo para sparse-from-scratch equivalente al nivel medio.
- Criterio de paso a expansión:
  - al menos un régimen sparse debe superar al baseline denso en Sparse Synergy tanto en un modelo tabular como en uno temporal, o bien mantener un rendimiento downstream aproximadamente equivalente con mejoras materiales de VRAM, tamaño del backbone o tiempo del proceso de denoising de DenoGrad, sin regresiones severas de estabilidad.

### 8. Fase 7 — Expansión tras piloto

- Añadir CNN/AutoEncoder y opcionalmente xLSTM.
- Escalar a todos los datasets y más seeds.
- Introducir predictors downstream sparse como eje factorial separado si la hipótesis principal ya está respaldada.
- Añadir análisis estadístico y tablas finales orientadas a paper.

## Relevant files

- /home/jjavier98/denograd-s/README.md — objetivo del repo: usar modelos sparse como backbones de DenoGrad.
- /home/jjavier98/denograd-s/src/benchmark.py — esqueleto actual del benchmark tabular/TS, punto central de reutilización.
- /home/jjavier98/denograd-s/src/models/__init__.py — factoría de modelos y wrapper, mejor punto para introducir construcción sparsity-aware.
- /home/jjavier98/denograd-s/src/models/dnn.py — primer candidato para cubrir los cuatro regímenes.
- /home/jjavier98/denograd-s/src/models/ft_transformer.py — candidato Transformer tabular; requiere reglas específicas para pruning de capas lineales/heads.
- /home/jjavier98/denograd-s/src/models/dlinear_adapter.py — objetivo TS lineal, ideal para estructural y 2:4.
- /home/jjavier98/denograd-s/src/models/lstm.py — baseline recurrente; conviene mantener el piloto conservador.
- /home/jjavier98/denograd-s/src/test.py — útil solo para smoke tests, no como orquestador principal.
- /home/jjavier98/denograd-s/data/tabular — datasets tabulares del piloto.
- /home/jjavier98/denograd-s/data/time_series — datasets TS del piloto.

## Verification

1. Smoke test end-to-end en un dataset tabular y uno temporal pequeños: benchmark noisy, entrenamiento backbone, DenoGrad, benchmark denoised, export de métricas y generación de una figura.
2. Validar automáticamente invariantes de justicia experimental: mismos índices de split, misma seed de ruido, mismos hiperparámetros de DenoGrad y mismo benchmark downstream entre comparaciones dense/sparse.
3. Para cada backend de sparsity, ejecutar checks de forma e inferencia antes de lanzar entrenamientos completos y verificar que el output conserva la misma semántica que el baseline denso.
4. Registrar por separado fallos por régimen: graph inválido en pruning estructural, kernel sparse no disponible, divergencia en sparse-from-scratch y OOM/timeouts en benchmarks.
5. Exigir que cada run del piloto produzca bundle completo de artefactos: snapshot de config, JSON/CSV de métricas, tiempos, gráficas, resumen de sparsity y métricas de memoria del backbone y VRAM.
6. Validar que las métricas de memoria y VRAM se capturan con el mismo protocolo en denso y sparse, incluyendo reset de picos de CUDA y mismo tamaño de lote.
7. Antes de escalar a todos los datasets, reproducir el comportamiento de dense vs sparse en una segunda seed y comprobar que la ordenación noisy/denoised sigue siendo coherente.
8. Verificar que la reutilización de caché no mezcla artefactos incompatibles, comprobando la firma experimental completa antes de reusar checkpoints, datasets denoised o resultados downstream.

## Decisions

- La prueba principal es si un backbone sparse hace más útil a DenoGrad, no si un predictor sparse post-denoising es mejor por sí mismo.
- La recomendación técnica es combinar varias librerías: PyTorch/torch.ao para masks y sparse-from-scratch básicos, Torch-Pruning para pruning estructural y torchao solo cuando haya soporte real para 2:4/block.
- El piloto cubre los cuatro regímenes, pero no todos los modelos con la misma profundidad. LSTM entra de forma conservadora y xLSTM queda fuera hasta estabilizar la infraestructura.
- El orden de ataque del estudio es masking primero, pruning estructural después y 2:4 en paralelo como hipótesis de aceleración; sparse-from-scratch queda reservado para una segunda ola.
- Las métricas de calidad deben usar como referencia el dataset clean previo al ruido sintético siempre que esté disponible; comparar noisy contra denoised sin clean es insuficiente para medir denoising.
- El benchmark downstream debe mantenerse fijo en el estudio principal para evitar confundir mejor denoising con mejor compresión del predictor.
- La evaluación de éxito no depende solo de mejorar el benchmark downstream: una paridad razonable con reducciones claras de VRAM, tamaño del backbone o tiempo del proceso de denoising de DenoGrad también cuenta como resultado valioso.
- La métrica temporal principal del estudio es el tiempo total del proceso de denoising de DenoGrad; cualquier otra latencia se considera secundaria salvo que ayude a diagnosticar el comportamiento del backbone sparse.
- El sistema debe guardar y reutilizar todos los estados intermedios relevantes siempre que sea seguro hacerlo, porque reducir recomputación es parte del valor práctico del pipeline y además mejora la trazabilidad de métricas.

## Further considerations

1. Si no hay soporte de kernels 2:4 en la GPU disponible, mantener 2:4 como régimen de precisión/estructura en el piloto y posponer cualquier claim de speedup real.
2. Si el pruning estructural en LSTM resulta frágil, rebajarlo a masking + sparse-from-scratch en el piloto y dejar el pruning recurrente estructural para la expansión.
3. Si el beneficio depende mucho del nivel de ruido, introducir un barrido corto de sensibilidad al ruido antes de escalar a todos los datasets.
