# ArchaeoSort — Hoja de ruta y estado funcional
_Generado el 2026-08-24. Carpeta única del proyecto: `archaeosort-dataset-builder`._

## 1. Objetivo final

Construir **ArchaeoSort**, un sistema completo de IA/visión artificial para clasificar automáticamente piezas y fragmentos arqueológicos (artefacto vs. roca natural, y en el futuro categorías más finas), partiendo de fotografías, y exponerlo como:

- una **herramienta de ingeniería de datasets** reproducible y auditable (lo que se está construyendo ahora, `archaeosort-dataset-builder`),
- un **modelo de clasificación** entrenado sobre esos datasets,
- una **API + frontend** donde subir una foto y obtener una predicción,
- y un **proyecto de portfolio** documentado (GitHub, LinkedIn).

## 2. Flujo completo del sistema

```text
IMÁGENES FUENTE (fotos de piezas/rocas)
        │
        ▼
1. VERIFICACIÓN DE INTEGRIDAD        (verify)
2. ANÁLISIS TÉCNICO DE CALIDAD       (analyze, blur, brightness, contrast, resolution, aspect)
3. DUPLICADOS Y BALANCE DE CLASES    (duplicates, classes)
4. EMBEDDINGS SEMÁNTICOS             (DINOv2 → vectores)
5. ÍNDICE + AUDITORÍA SEMÁNTICA      (FAISS → similarity, semantic-audit, outliers)
6. INFORMES                          (JSON + HTML, statistics, report)
        │
        ▼
7. BUILD REPRODUCIBLE                 ← ESTAMOS AQUÍ (siguiente hito)
   → train/ val/ test/ (por clase)
   → split_report.json (qué fue a cada split)
   → provenance.json (con qué parámetros/semilla se generó)
   → manifest.json + huella SHA-256 (versión del dataset)
        │
        ▼
DATASET VERSIONADO Y REPRODUCIBLE
        │
        ▼
8. CLASIFICADOR BASELINE              (entrenar un modelo simple)
9. COMPARATIVA TRANSFER LEARNING      (ResNet/EfficientNet preentrenados)
10. MÉTRICAS Y MATRIZ DE CONFUSIÓN
11. EXPORTACIÓN DEL MODELO
        │
        ▼
12. API DE INFERENCIA (FastAPI)       → sube una imagen, devuelve predicción
13. FRONTEND (Next.js + TypeScript)   → interfaz de usuario, explorador de similares
14. REVISIÓN / ACTIVE LEARNING        → corregir predicciones, reentrenar
15. HISTÓRICO EN POSTGRESQL
16. DOCKER + CI/CD
17. FICHAS DE MODELO/DATASET + MATERIAL DE PORTFOLIO
```

**Rama paralela — Isaac Sim** (aprendizaje + diferencial de portfolio, no bloquea el resto):

```text
                    ┌─ L1. DATOS SINTÉTICOS (Isaac Sim / Omniverse)
                    │   renderiza piezas 3D con luz/ángulo/fondo variables
                    │   → alimenta el dataset real (paso 3-6 de arriba)
DATASET VERSIONADO ─┤
   + MODELO ────────┤
                    └─ L2. SIMULACIÓN DEL PROCESO
                        cámara + cinta transportadora simuladas en Isaac Sim
                        → captura frame → llama al modelo/API → clasifica → separa
                        (demo end-to-end del "clasificador físico" sin hardware real)
```

## 3. Tecnologías que intervienen

**Ya en uso (dataset builder):**
Python 3.12 · uv (gestor de entorno/paquetes) · Typer (CLI) · Pydantic (configuración) · Rich · Pillow / OpenCV / NumPy (procesado de imagen) · Torch / Torchvision / Transformers / timm (embeddings) · FAISS (búsqueda vectorial) · Ruff (lint) · Pytest (tests) · GitHub Actions (CI) · Git.

**Pendientes de introducir según avancemos de fase:**
FastAPI (API de inferencia) · PostgreSQL (histórico) · Next.js + TypeScript + Tailwind + shadcn/ui (frontend) · Framer Motion, Recharts (UI/gráficas) · Docker (contenedores) · Flutter (app móvil, opcional y al final) · **NVIDIA Isaac Sim / Omniverse + USD** (datos sintéticos y simulación cámara+cinta, rama L, opcional/aprendizaje).

## 4. Qué tenemos ya funcionando (confirmado)

- Estructura de proyecto instalable (`pyproject.toml`, `uv.lock`), paquete `archaeosort_dataset_builder`.
- CLI instalable con Typer: `uv run archaeosort --help` expone **19 comandos** (verify, analyze, duplicates, blur, brightness, contrast, resolution, aspect, classes, quality, statistics, report, embeddings, index, search, semantic-audit, outliers, outlier-review, pipeline).
- Configuración centralizada sin rutas hardcodeadas (`--dataset`, `--reports`, `--outputs`).
- Pipeline completo probado sobre un dataset real de 5.319 imágenes (`stone_figshare`: natural_rock 27%, stone_artefact 73%), optimizado de 851,77 s a 309,50 s.
- Módulos de embeddings DINOv2 + índice FAISS + auditoría semántica + detección de outliers, ya existentes.
- `splitter/dataset_splitter.py` — **listo y reutilizable**, es la pieza clave para construir el `build`.
- `exporter/dataset_exporter.py` — existe pero es básico e inseguro (borra el destino sin confirmar).
- `versioning/dataset_version.py` — existe (`sha256_directory`, `create_manifest`, `save_manifest`) pero necesita hashing determinista (por chunks, sin que el propio manifest se contamine).
- 31 tests pasando en el último chequeo confirmado.
- CI en GitHub Actions, documentación bilingüe (README.md / README_ES.md).
- Reorganización de carpetas hecha hoy: todo el proyecto vive ahora en una única carpeta (`archaeosort-dataset-builder`), con el material antiguo de `archaeosort-ai` archivado como referencia en `_legacy_archaeosort-ai/` (modelo ResNet18 ya entrenado con métricas, config de entrenamiento, e implementación alternativa de duplicados con hashing perceptual/BK-tree que nunca se integró) y el catálogo de datasets fusionado en `datasets_repo/`.

**Sobre Isaac Sim:** en `archaeosort-ai` existía una carpeta `isaac_sim/` (`assets/`, `config/`, `scenes/`, `scripts/`), pero estaba completamente vacía — ningún archivo, y ningún documento describía qué se pensaba hacer con ella. No se arrastró en la consolidación porque no había nada que preservar. Confirmaste que la intención es (1) generar datos sintéticos y (2) simular el proceso de identificación con cámara + cinta transportadora, sobre todo para aprender Isaac Sim y añadir un punto diferencial al portfolio. Queda planteada como **Fase L** más abajo, en paralelo al resto — no bloquea el hito de `build`.

**Nota importante:** al instalar el pack de "ahorro de tokens" se sobrescribió por error el `README.md` del proyecto (coincidía el nombre de archivo). Ya lo he restaurado a la última versión guardada en Git (`git show HEAD:README.md`) y he renombrado el README del pack a `README-PACK-IA.md` para que no vuelva a chocar. Si tenías cambios sin commitear en el README que no llegaste a guardar en Git, no he podido recuperarlos desde aquí — revisa el historial de versiones de OneDrive de ese archivo por si acaso.

## 5. Qué falta — dividido en fases, para ir pidiéndolas una a una

Cada fase trae un prompt listo para copiar y pegar a tu agente de código cuando quieras abordarla. Pide una fase completa cada vez, no varias a la vez.

### Fase A — Verificar estado real antes de tocar código
```
Continúa el desarrollo del repo archaeosort-dataset-builder. Antes de escribir nada,
inspecciona el estado real: git status, git log, uv run archaeosort --help,
uv run --no-sync ruff check ., uv run --no-sync pytest -q, y comprueba si existe
src/archaeosort_dataset_builder/builder/dataset_builder.py y si "build" ya está en el CLI.
Repórtame el estado exacto antes de proponer cambios.
```

### Fase B — Motor de `build` (el hito inmediato)
```
Implementa el motor de "archaeosort build" en
src/archaeosort_dataset_builder/builder/dataset_builder.py reutilizando el splitter
existente (dataset_splitter.py) sin reescribirlo. Debe: validar ratios train/val/test,
rechazar dataset de entrada igual al de salida, preparar el directorio de salida de forma
segura (rechazar si no está vacío salvo --force), generar train/val/test por clase,
guardar split_report.json y provenance.json (timestamp, dataset origen, semilla, ratios,
clases, modo copy/hardlink). No toques el CLI todavía.
```

### Fase C — Tests del motor de build
```
Añade tests para el motor de build usando pytest tmp_path con imágenes sintéticas
pequeñas (no el dataset real de 5.319 imágenes). Cubre: ratios que suman 1, ratio
negativo rechazado, origen inexistente, origen sin clases, salida no vacía sin --force,
--force reconstruye, misma semilla es determinista, se generan train/val/test,
se preservan las clases, el total de imágenes de salida coincide con el origen,
split_report.json y provenance.json correctos, origen == destino rechazado.
```

### Fase D — Conectar `build` al CLI
```
Expón el motor de build como comando Typer "archaeosort build" con las opciones
--output, --train, --val, --test, --seed, --copy, --force. Añade tests de CLI.
Verifica con uv run archaeosort build --help y compílalo con py_compile tras el cambio.
```

### Fase E — Endurecer exporter y versioning
```
Revisa exporter/dataset_exporter.py: hazlo seguro por defecto (no borrar el destino sin
--force explícito) y ofrece una API invocable en vez de solo argparse.
Revisa versioning/dataset_version.py: cambia el hash a lectura por chunks (sin cargar
el archivo entero en memoria), y define un contrato determinista para que manifest.json
no se contamine a sí mismo ni dependa de timestamps variables. Añade tests para ambos.
```

### Fase F — Smoke test real y documentación
```
Ejecuta "archaeosort build" contra el dataset real stone_figshare en un directorio de
salida separado (no lo subas a Git). Analiza el dataset generado con el propio
ArchaeoSort. Actualiza README.md y README_ES.md documentando el comando build,
el layout de salida, provenance y manifest. Haz commit final:
"feat: add reproducible versioned dataset build command".
```

### Fase G — Clasificador baseline (siguiente gran bloque, después del build)
```
Con un dataset versionado por "archaeosort build", entrena un clasificador baseline
(ResNet18 u otro backbone ligero via transfer learning) sobre train/val, evalúa en test
con métricas y matriz de confusión, y guarda el modelo exportado. Documenta resultados.
```

### Fase H — API de inferencia (FastAPI)
```
Crea una API FastAPI que cargue el modelo exportado y exponga un endpoint de
predicción a partir de una imagen subida, con validación de entrada y tests.
```

### Fase I — Frontend (Next.js + TypeScript)
```
Crea un frontend Next.js + TypeScript + Tailwind que permita subir una imagen,
llamar a la API de inferencia y mostrar el resultado, con un explorador de imágenes
similares usando el índice FAISS ya existente.
```

### Fase J — Empaquetado y despliegue
```
Dockeriza el backend (FastAPI) y el frontend, define un docker-compose para desarrollo
local, y configura CI/CD (build, test, lint) en GitHub Actions.
```

### Fase K — Revisión activa, histórico y material de portfolio
```
Añade una vista de revisión humana de predicciones (active learning), guarda el
histórico de predicciones/revisiones en PostgreSQL, y prepara fichas de modelo/dataset
(model card / data card) y material para portfolio/LinkedIn.
```

### Fase L — Isaac Sim: datos sintéticos y simulación (opcional, aprendizaje + portfolio)

Carpeta vacía de origen, se empieza desde cero. Se puede abordar en cualquier momento después de tener el `build` (L1) o después del modelo/API (L2) — no depende de G/H si solo quieres aprender la herramienta primero.

**L1 — datos sintéticos:**
```
Quiero aprender NVIDIA Isaac Sim generando datos sintéticos para ArchaeoSort. Ayúdame a
montar una escena mínima en Isaac Sim/Omniverse con 2-3 modelos 3D genéricos (uno como
"artefacto", otro como "roca natural"), variando iluminación, ángulo de cámara y fondo,
y exportando imágenes renderizadas con su etiqueta a una carpeta compatible con el layout
de clases que usa archaeosort-dataset-builder (una subcarpeta por clase). Empieza por lo
más simple que funcione end-to-end, luego lo ampliamos.
```

**L2 — simulación del proceso (cámara + cinta):**
```
Quiero simular en Isaac Sim una cinta transportadora con una cámara fija que capture
piezas pasando por delante, envíe cada frame al modelo/API de ArchaeoSort ya entrenado,
y simule una separación física (por ejemplo mover el objeto a la izquierda o derecha
según la clasificación). Empieza con una escena simple (una cinta, una cámara, un objeto)
y un script que capture un frame, llame a la predicción y mueva el objeto según el
resultado, antes de complicar la escena.
```

## 6. Cómo usar esto

1. Empieza siempre por la **Fase A** al abrir una sesión nueva (verificar antes de tocar).
2. Pide las fases en orden: B → C → D → E → F cierran el hito de `build`.
3. G en adelante son bloques grandes independientes — puedes reordenarlos si cambian tus prioridades, pero G (baseline) es el que desbloquea H, I, K.
4. La Fase L (Isaac Sim) es independiente y opcional — puedes empezarla en paralelo, incluso antes que G, si tu prioridad ahora es aprender la herramienta.
5. Este documento y `ARCHAEOSORT_AGENT_CONTEXT.md` (más detallado, en inglés, con todo el histórico técnico) están guardados en la raíz del proyecto — compártelos con el agente al empezar cada sesión nueva.
