"use client";

import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  getHealth,
  predictImage,
  searchSimilar,
  similarImageUrl,
  type HealthResponse,
  type PredictionResponse,
  type SimilarityResult,
} from "@/lib/api";

type LoadState = "idle" | "loading" | "done" | "error";

export default function ClassifierUploader() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [state, setState] = useState<LoadState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [similarResults, setSimilarResults] = useState<SimilarityResult[] | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((result) => {
        if (!cancelled) setHealth(result);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setHealthError(
            error instanceof ApiError
              ? error.detail || error.message
              : "Could not reach the ArchaeoSort API."
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function handleFileChange(selected: File | null) {
    setFile(selected);
    setPrediction(null);
    setSimilarResults(null);
    setErrorMessage(null);
    setState("idle");

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(selected ? URL.createObjectURL(selected) : null);
  }

  async function handleClassify() {
    if (!file) return;
    setState("loading");
    setErrorMessage(null);
    setPrediction(null);
    setSimilarResults(null);

    const tasks: Promise<void>[] = [];

    if (health?.model_loaded !== false) {
      tasks.push(
        predictImage(file)
          .then(setPrediction)
          .catch((error: unknown) => {
            // Only surface a classifier error if similarity also fails --
            // one feature being unavailable shouldn't block the other.
            if (error instanceof ApiError && error.status === 503) return;
            throw error;
          })
      );
    }

    if (health?.similarity_loaded !== false) {
      tasks.push(
        searchSimilar(file, 6)
          .then((response) => setSimilarResults(response.results))
          .catch((error: unknown) => {
            if (error instanceof ApiError && error.status === 503) return;
            throw error;
          })
      );
    }

    try {
      await Promise.all(tasks);
      setState("done");
    } catch (error) {
      setErrorMessage(
        error instanceof ApiError ? error.detail || error.message : "Something went wrong."
      );
      setState("error");
    }
  }

  const sortedProbabilities = prediction
    ? Object.entries(prediction.probabilities).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div className="flex w-full max-w-2xl flex-col gap-8">
      <ApiStatusBanner health={health} healthError={healthError} />

      <div
        className="flex cursor-pointer flex-col items-center gap-4 rounded-xl border-2 border-dashed border-zinc-300 bg-white p-10 text-center transition-colors hover:border-zinc-400 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:border-zinc-600"
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          const dropped = event.dataTransfer.files?.[0];
          if (dropped) handleFileChange(dropped);
        }}
      >
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element -- arbitrary user-picked image, not an optimizable static asset
          <img
            src={previewUrl}
            alt="Selected specimen"
            className="max-h-64 rounded-lg object-contain"
          />
        ) : (
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Click or drag an image here (a stone artefact or natural rock photo)
          </p>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(event) => handleFileChange(event.target.files?.[0] ?? null)}
        />
      </div>

      <button
        type="button"
        disabled={!file || state === "loading"}
        onClick={handleClassify}
        className="rounded-full bg-zinc-900 px-6 py-3 text-sm font-medium text-white transition-colors enabled:hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-zinc-50 dark:text-zinc-900 dark:enabled:hover:bg-zinc-200"
      >
        {state === "loading" ? "Analyzing…" : "Classify"}
      </button>

      {errorMessage && (
        <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
          {errorMessage}
        </p>
      )}

      {prediction && (
        <section className="flex flex-col gap-3 rounded-xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
            Prediction
          </h2>
          <p className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">
            {prediction.predicted_class}{" "}
            <span className="text-base font-normal text-zinc-500 dark:text-zinc-400">
              ({(prediction.confidence * 100).toFixed(1)}% confidence)
            </span>
          </p>
          <div className="flex flex-col gap-2">
            {sortedProbabilities.map(([className, probability]) => (
              <ProbabilityBar key={className} label={className} value={probability} />
            ))}
          </div>
        </section>
      )}

      {similarResults && similarResults.length > 0 && (
        <section className="flex flex-col gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
            Visually similar images
          </h2>
          <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
            {similarResults.map((result) => (
              // eslint-disable-next-line @next/next/no-img-element -- served dynamically from the API, not a static asset Next can optimize
              <img
                key={result.path_id}
                src={similarImageUrl(result.path_id)}
                alt={result.filename}
                title={`${result.filename} (score ${result.score.toFixed(3)})`}
                className="aspect-square w-full rounded-lg object-cover"
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function ApiStatusBanner({
  health,
  healthError,
}: {
  health: HealthResponse | null;
  healthError: string | null;
}) {
  if (healthError) {
    return (
      <p className="rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:bg-amber-950 dark:text-amber-200">
        Can&apos;t reach the ArchaeoSort API ({healthError}). Make sure{" "}
        <code className="font-mono">archaeosort serve</code> is running and{" "}
        <code className="font-mono">NEXT_PUBLIC_API_BASE_URL</code> points at it.
      </p>
    );
  }
  if (!health) return null;

  const parts: string[] = [];
  parts.push(
    health.model_loaded
      ? `Classifier loaded (${health.device}, classes: ${health.classes?.join(", ")})`
      : "Classifier not loaded"
  );
  parts.push(
    health.similarity_loaded
      ? `Similarity index loaded (${health.similarity_size} images)`
      : "Similarity search not loaded"
  );

  return (
    <p className="rounded-lg bg-zinc-100 px-4 py-3 text-sm text-zinc-600 dark:bg-zinc-900 dark:text-zinc-400">
      {parts.join(" · ")}
    </p>
  );
}

function ProbabilityBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="w-32 shrink-0 text-zinc-600 dark:text-zinc-400">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
        <div
          className="h-full rounded-full bg-zinc-900 dark:bg-zinc-50"
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
      <span className="w-12 shrink-0 text-right text-zinc-500 dark:text-zinc-400">
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}
