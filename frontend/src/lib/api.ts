/**
 * Thin client for the ArchaeoSort inference API (Fase H, FastAPI backend).
 * The API base URL is read from NEXT_PUBLIC_API_BASE_URL (see .env.local.example)
 * so the frontend can point at any running `archaeosort serve` instance
 * without a rebuild-time constant.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  classes: string[] | null;
  device: string | null;
  similarity_loaded: boolean;
  similarity_size: number | null;
}

export interface PredictionResponse {
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface SimilarityResult {
  rank: number;
  score: number;
  path_id: number;
  filename: string;
}

export interface SimilarityResponse {
  results: SimilarityResult[];
}

/** Thrown for any non-2xx API response; `detail` carries the API's own
 * error message (FastAPI's HTTPException `detail` field) when available. */
export class ApiError extends Error {
  status: number;
  detail?: string;

  constructor(status: number, detail?: string) {
    super(detail || `Request failed with status ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function parseErrorDetail(response: Response): Promise<string | undefined> {
  try {
    const body = await response.json();
    return typeof body?.detail === "string" ? body.detail : undefined;
  } catch {
    return undefined;
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorDetail(response));
  }
  return response.json();
}

export async function predictImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorDetail(response));
  }
  return response.json();
}

export async function searchSimilar(file: File, k = 5): Promise<SimilarityResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/similar?k=${k}`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorDetail(response));
  }
  return response.json();
}

/** Direct image URL for a similarity result -- an <img src> can point here. */
export function similarImageUrl(pathId: number): string {
  return `${API_BASE_URL}/similar-image/${pathId}`;
}
