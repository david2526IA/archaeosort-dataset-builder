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

- Aumentar la cobertura de tests
- Configuración externa del pipeline
- Dashboard interactivo
- Mejorar la detección semántica de duplicados
- Comparación entre versiones del dataset
- Integración con active learning
- Nuevos modelos de embeddings
- Soporte Docker
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
- Tests: 26 pasando
- CI: configurada

## Autor

**David Falcon Perez**

Automatización, robótica, inteligencia artificial y visión artificial aplicada.

---

ArchaeoSort Dataset Builder forma parte del proyecto **ArchaeoSort-AI**.
