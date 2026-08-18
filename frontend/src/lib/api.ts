export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Category = "malignant" | "benign" | "unknown";
export type RiskLevel = "low" | "high" | "critical" | "unknown";

export interface ClassProbability {
  key: string;
  full_name: string;
  probability: number;
  category: Category;
}

export interface PredictionResponse {
  predicted_key: string;
  predicted_name: string;
  confidence: number;
  category: Category;
  risk_level: RiskLevel;
  description: string;
  typical_action: string;
  probabilities: ClassProbability[];
  gradcam_overlay: string | null;
  model_val_accuracy: number | null;
  disclaimer: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  num_classes: number | null;
  model_val_accuracy: number | null;
  detail: string | null;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/health`, { cache: "no-store" });
  if (!res.ok) throw new ApiError("Health check failed", res.status);
  return res.json();
}

export async function predictImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let detail = `Request failed with status ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore parse failure, use default message
    }
    throw new ApiError(detail, res.status);
  }

  return res.json();
}
