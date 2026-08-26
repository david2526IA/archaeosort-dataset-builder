# ArchaeoSort Frontend (Fase I)

A minimal Next.js + TypeScript + Tailwind frontend for the [ArchaeoSort inference API](../README.md#inference-api-fase-h) (Fase H): upload a photo, get a classification (stone artefact vs. natural rock) with per-class confidence, and browse visually similar images from the dataset via the existing DINOv2/FAISS index.

## Prerequisites

- Node.js 18.18+ and npm.
- A running `archaeosort serve` instance (see the main repo's README) with at least one of `--model` / `--embeddings-dir` — the UI adapts to whichever endpoints are actually available, reported by `GET /health`.

## Setup

```bash
npm install
cp .env.local.example .env.local   # adjust NEXT_PUBLIC_API_BASE_URL if not on the default host/port
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## What it does

- On load, calls `GET /health` and shows whether the classifier and/or the similarity index are available, so the UI is honest about a backend that's only serving one of the two features.
- Uploading (click or drag-and-drop) an image and pressing "Classify" calls `POST /predict` and `POST /similar` in parallel; each one is independently optional — if the backend returns 503 for one (not loaded), the UI simply skips that section instead of failing outright.
- Prediction results show the predicted class, confidence, and a bar per class probability.
- Similarity results render as a small grid of thumbnails, each fetched from `GET /similar-image/{path_id}` (the API resolves `path_id` to a real file on the machine running `archaeosort serve` — the frontend never sees or needs the raw filesystem path).

## Project structure

```text
src/
|-- app/
|   |-- layout.tsx      # root layout, page metadata
|   |-- page.tsx         # landing page
|   `-- globals.css      # Tailwind entry point
|-- components/
|   `-- ClassifierUploader.tsx   # the whole upload/predict/similar UI
`-- lib/
    `-- api.ts            # typed fetch client for the inference API
```

## Development

```bash
npm run lint       # eslint
npx tsc --noEmit   # type-check
npm run build      # production build
```

## Known limitations (honest, not yet done)

- No automated component tests yet (would need a DOM test runner such as Vitest + Testing Library) — verified manually against a real running backend instead for this phase.
- No image-similarity result caching or pagination; `/similar` always asks for a fixed `k` per request.
- Deployment/containerization is Fase J (Docker/CI), not covered here.
