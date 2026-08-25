# ArchaeoSort Dataset Builder

[English](README.md) | **Español**

> Pipeline reproducible para construcción, análisis de calidad y auditoría semántica de datasets de visión artificial aplicada a arqueología.

ArchaeoSort Dataset Builder es una herramienta desarrollada en Python para construir, validar y auditar datasets de imágenes antes de utilizarlos para entrenar modelos de visión artificial.

Forma parte del ecosistema **ArchaeoSort-AI** y está orientado a ingeniería de datos reproducible aplicada a visión artificial arqueológica.

## Características

- Verificación y validación de datasets
- Detección de duplicados
- Análisis de desenfoque
- Análisis de brillo
- Análisis de contraste
- Análisis de resolución
- Análisis de relación de aspecto
- Análisis del balance de clases
- Estadísticas del dataset
- Informes automáticos de calidad
- Embeddings visuales mediante DINOv2
- Indexación de similitud mediante FAISS
- Auditoría semántica del dataset
- Detección de outliers visuales
- Hojas de revisión visual para supervisión humana
- Construcción de datasets train/val/test versionados y reproducibles
- Preprocesado de aislamiento de objeto (elimina fondo y regla de escala)
- Clasificador baseline ResNet18 (transfer learning)
- Pipeline automatizado de extremo a extremo
- Suite de tests con pytest
- Linting y formato con Ruff
- Integración continua mediante GitHub Actions

## Arquitectura del pipeline

```text
Dataset
   |
   v
Validación
   |
   v
Análisis de calidad
   |
   +--> Duplicados
   +--> Desenfoque
   +--> Brillo
   +--> Contraste
   +--> Resolución
   +--> Relación de aspecto
   +--> Balance de clases
   +--> Estadísticas
   |
   v
Embeddings DINOv2
   |
   v
Índice FAISS
   |
   +--> Auditoría semántica
   +--> Detección de outliers
   |
   v
Informes + revisión humana
```

## Interfaz CLI

Para consultar todos los comandos disponibles:

```powershell
uv run --no-sync python builder.py --help
```

Comandos disponibles:

```text
verify
analyze
duplicates
quality
blur
brightness
resolution
contrast
aspect
classes
statistics
report
embeddings
index
search
semantic-audit
pipeline
outlier-review
outliers
```

Para ejecutar el pipeline completo:

```powershell
uv run --no-sync python builder.py pipeline
```

## Build (dataset versionado y reproducible)

`archaeosort build` convierte un dataset de origen organizado por carpetas de clase en un dataset train/val/test versionado, reutilizando el splitter determinista ya existente:

```powershell
uv run archaeosort --dataset <ruta-dataset-origen> build --output <ruta-salida> --train 0.70 --val 0.15 --test 0.15 --seed 42 [--copy] [--force]
```

- `--copy` copia los archivos en vez de usar hardlinks (por defecto se intenta hardlink y se cae a copia si falla).
- `--force` es obligatorio para sobrescribir una carpeta de salida no vacía; sin él, `build` se niega a tocarla.
- El dataset de origen y la carpeta de salida deben ser rutas distintas — `build` lo rechaza, porque `--force` puede borrar la carpeta de salida.

Estructura de salida:

```text
<salida>/
|-- train/<clase>/...
|-- val/<clase>/...
|-- test/<clase>/...
|-- split_report.json   # recuento de imágenes por split y clase, y el modo usado
|-- provenance.json      # dataset origen, semilla, ratios, clases, timestamp, modo
`-- manifest.json         # número de archivos, tamaño y huella SHA-256 determinista
```

El campo `sha256` de `manifest.json` es una huella de contenido: depende solo de las rutas y bytes de los archivos del dataset (lectura por chunks; `manifest.json`, `split_report.json` y `provenance.json` quedan siempre excluidos — son metadatos del build, no contenido del dataset), nunca del reloj — dos builds con el mismo contenido siempre dan la misma huella. `version`/`created` siguen siendo metadatos basados en timestamp.

`archaeosort build` tiene su propia suite de tests (`tests/test_dataset_builder.py`, `tests/test_cli_build.py`) además de los tests de endurecimiento de exporter/versioning (`tests/test_dataset_exporter.py`, `tests/test_dataset_version.py`).

## Clasificador baseline (Fase G)

`archaeosort train-baseline` entrena un clasificador ResNet18 (transfer learning) sobre un dataset train/val/test generado con `build()` y lo evalúa sobre el split de test:

```powershell
uv run archaeosort train-baseline --data <ruta-salida-de-build> --output <ruta-carpeta-resultado> --epochs 15 --patience 3 [--batch-size 16] [--lr 0.001] [--freeze-backbone/--no-freeze-backbone] [--seed 42]
```

Escribe `best_model.pt`, `training_history.json` y `evaluation.json` (un `classification_report` por clase más la matriz de confusión, evaluados sobre el split de test) en `--output`, restaurando el mejor checkpoint según accuracy de validación antes de evaluar (early stopping mediante `--patience`). Usa CUDA automáticamente cuando está disponible, y CPU en caso contrario.

### Recorte de objetos — corrigiendo un problema de shortcut-learning

Un primer entrenamiento sobre las imágenes originales de `stone_figshare` alcanzó un sospechoso **100% de accuracy en test** en solo dos épocas. Se descartó primero la fuga por imágenes duplicadas (`archaeosort duplicates` no encontró ningún grupo de duplicados confirmado entre 5.319 imágenes), y la inspección visual de muestras reveló la causa real: las dos clases se habían fotografiado con montajes de estudio distintos e internamente consistentes (fondos y reglas de escala/etiquetas diferentes). El modelo estaba separando las clases por el estilo fotográfico, no por el contenido del objeto — un caso de manual de sesgo de dataset / shortcut-learning, no un fallo del pipeline de ingeniería de datos en sí.

`archaeosort crop-objects` elimina ese factor de confusión con visión por computador clásica (sin ML, nada que sobreajustar): umbraliza el fondo (mayormente uniforme), localiza el contorno del objeto — priorizando los contornos que no tocan el borde del fotograma, ya que las reglas de escala y etiquetas suelen colocarse en el borde mientras la pieza suele quedar centrada — y recorta a ese cuadro delimitador:

```powershell
uv run archaeosort crop-objects --source <dataset por carpetas de clase> --output <ruta> [--margin 0.06] [--force]
```

Replica la estructura `<clase>/` de origen bajo `--output` y escribe `crop_report.json` (recuento por clase de imágenes recortadas / de baja confianza / ilegibles) para poder auditar después los recortes de baja confianza en vez de darlos por buenos sin más.

Reentrenar sobre el dataset recortado dio un resultado creíble de **95,9% de accuracy en test** (100% de precisión / 84,8% de recall en `natural_rock`, 94,6% de precisión / 100% de recall en `stone_artefact` — coherente con el desbalance de clases del dataset origen, 73%/27%, y no con un atajo), claramente por encima de un baseline anterior, con un dataset más pequeño, que alcanzaba el 83%.

`train-baseline` y `crop-objects` tienen su propia suite de tests (`tests/test_classifier_baseline_layout.py`, `tests/test_cli_train_baseline.py`, `tests/test_object_crop.py`, `tests/test_cli_crop_objects.py`).

## Análisis semántico

El proyecto combina controles tradicionales de calidad de imagen con técnicas modernas de representación visual.

### DINOv2

DINOv2 transforma las imágenes en embeddings que representan información visual y semántica.

### FAISS

FAISS permite realizar búsquedas eficientes de vecinos más próximos sobre los embeddings generados.

Esta combinación permite:

- Buscar imágenes visualmente similares
- Explorar semánticamente el dataset
- Detectar posibles errores de etiquetado
- Localizar muestras anómalas
- Auditar datasets a escala

## Resultados reales

El pipeline ha sido probado sobre un dataset arqueológico compuesto por **5.319 imágenes JPEG**.

| Métrica | Resultado |
| --- | ---: |
| Imágenes analizadas | 5.319 |
| Grupos duplicados confirmados | 0 |
| Imágenes nítidas | 3.537 |
| Imágenes desenfocadas | 1.782 |
| Contraste bajo | 669 |
| Contraste normal | 4.635 |
| Contraste alto | 15 |
| Imágenes oscuras | 1 |
| Brillo normal | 2.794 |
| Imágenes brillantes | 2.524 |
| Outliers visuales | 107 |

Estos valores corresponden al análisis del dataset y no representan métricas de rendimiento de un modelo.

## Informes generados

El sistema genera informes JSON, HTML y evidencias visuales dentro de `reports/`.

```text
reports/
|-- blur.json
|-- brightness.json
|-- contrast.json
|-- duplicates.json
|-- resolution.json
|-- class_balance.json
|-- semantic_audit.json
|-- outliers.json
|-- verify_report.json
|-- report.html
|-- outlier_review/
|   `-- outliers_top30.jpg
`-- semantic_review/
    `-- imágenes de revisión semántica
```

### Revisión visual de outliers

![Revisión visual de outliers](reports/outlier_review/outliers_top30.jpg)

### Revisión semántica

![Ejemplo de revisión semántica](reports/semantic_review/suspicious_001_full.jpg)

## Instalación

Requisitos:

- Python >=3.11,<3.13
- Git
- uv

Instalación de dependencias:

```powershell
uv sync
```

## Desarrollo y calidad

Comprobar el código con Ruff:

```powershell
uv run --no-sync ruff check .
```

Ejecutar los tests:

```powershell
uv run --no-sync pytest -q
```

Estado actual:

```text
26 passed
```

Calcular cobertura:

```powershell
uv run --no-sync pytest --cov=archaeosort_dataset_builder --cov-report=term-missing
```

Cobertura base actual: **21%**.

La cobertura se ampliará progresivamente priorizando los componentes críticos del pipeline.

## Integración continua

El proyecto incorpora GitHub Actions mediante `.github/workflows/ci.yml`.

Cada cambio del repositorio puede validarse automáticamente mediante Ruff, pytest y cobertura.

```text
Push / Pull Request
        |
        v
      Ruff
        |
        v
     Pytest
        |
        v
    Coverage
```

## Documentación

En [`docs/`](docs/) hay documentación adicional del proyecto:

- [`ROADMAP_ARCHAEOSORT.md`](docs/ROADMAP_ARCHAEOSORT.md) — roadmap por fases del sistema ArchaeoSort-AI completo (ingeniería de datos, clasificador, API de inferencia, frontend, Isaac Sim).
- [`ARCHAEOSORT_AGENT_CONTEXT.md`](docs/ARCHAEOSORT_AGENT_CONTEXT.md) — notas técnicas de contexto para asistentes de IA que trabajen en este repo.
- Una memoria técnica en español (`.docx`) con el proyecto explicado en más profundidad.

## Objetivos de ingeniería

- Construcción reproducible de datasets
- Arquitectura modular
- Validación automática de calidad
- Calidad de datos antes del entrenamiento
- Revisión human-in-the-loop
- Análisis semántico mediante visión artificial
- Testing automatizado
- Integración continua

## Roadmap

- API de inferencia (FastAPI) para servir el clasificador baseline
- Frontend de revisión de predicciones (Next.js)
- Soporte Docker y CI/CD para el stack completo
- Integración con active learning
- Dashboard interactivo
- Mejorar la detección semántica de duplicados
- Comparación entre versiones del dataset
- Nuevos modelos de embeddings
- Isaac Sim: datos sintéticos + simulación de cámara/cinta transportadora
- Integración con el sistema completo ArchaeoSort-AI

## Estado del proyecto

**Versión: 0.1.0**

- Análisis de calidad: implementado
- Detección de duplicados: implementada
- Embeddings DINOv2: implementados
- Índice FAISS: implementado
- Auditoría semántica: implementada
- Detección de outliers: implementada
- Pipeline automatizado: implementado
- Construcción reproducible de datasets: implementada
- Preprocesado de aislamiento de objeto: implementado
- Clasificador baseline (ResNet18, 95,9% de accuracy en test sobre el dataset recortado): implementado
- CI: configurada

## Autor

**David Falcon Perez**

Automatización, robótica, inteligencia artificial y visión artificial aplicada.

---

ArchaeoSort Dataset Builder forma parte del proyecto **ArchaeoSort-AI**.
