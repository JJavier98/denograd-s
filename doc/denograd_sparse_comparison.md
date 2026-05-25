# Comparativa DenoGrad: Dense vs Sparse Backbone (mediana)

Esta tabla compara DenoGrad con y sin sparse backbone usando **medianas** para reducir la influencia de outliers.

| Dominio | Mejora mediana dense (%) | Mejora mediana sparse (%) | Delta mejora (pp) | Neuronas medianas dense | Neuronas medianas sparse | Reduccion neuronas (%) | Peso mediano dense (KB) | Peso mediano sparse (KB) | Reduccion peso (%) | Tiempo mediano dense (s) | Tiempo mediano sparse (s) | Aceleracion sparse (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tabular | 50.3905 | 51.4109 | +1.0204 | 12801 | 9798 | 23.46 | 50.00 | 38.27 | 23.46 | 26.5887 | 24.5830 | 7.54 |
| Series temporales | 73.4486 | 73.6065 | +0.1580 | 51521 | 33508 | 34.96 | 201.25 | 130.89 | 34.96 | 5.4801 | 5.4016 | 1.43 |
| Global | 57.9145 | 61.8083 | +3.8938 | 51265 | 30430 | 40.64 | 200.25 | 118.87 | 40.64 | 26.0444 | 24.2313 | 6.96 |

## Nota sobre outliers (para redaccion del informe/articulo)

- Tabular: min = -111.3691, max = 79.9725, corridas con mejora positiva = 57/60.
- Series temporales: min = -2349.6443, max = 86.1058, corridas con mejora positiva = 72/78.

Se recomienda reportar estos outliers en el texto del articulo para aportar transparencia y credibilidad, manteniendo la mediana como metrica central de comparacion.