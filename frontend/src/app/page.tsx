"use client";

import { useCallback, useEffect, useState } from "react";
import UploadDropzone from "@/components/UploadDropzone";
import ResultPanel from "@/components/ResultPanel";
import { ApiError, checkHealth, predictImage, type PredictionResponse } from "@/lib/api";

type BackendStatus = "checking" | "ready" | "model_unavailable" | "unreachable";

export default function Home() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [backendDetail, setBackendDetail] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    checkHealth()
      .then((h) => {
        if (cancelled) return;
        setBackendStatus(h.model_loaded ? "ready" : "model_unavailable");
        setBackendDetail(h.detail);
      })
      .catch(() => {
        if (!cancelled) setBackendStatus("unreachable");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleFileSelected = useCallback((selected: File) => {
    setFile(selected);
    setResult(null);
    setError(null);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(selected);
    });
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const prediction = await predictImage(file);
      setResult(prediction);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError("Something went wrong talking to the backend. Is it running?");
      }
    } finally {
      setLoading(false);
    }
  }, [file]);

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col gap-8 px-4 py-12">
      <header className="text-center">
        <h1 className="text-3xl font-bold text-ink-primary-light dark:text-ink-primary-dark">
          Skin Lesion Classifier
        </h1>
        <p className="mt-2 text-sm text-ink-secondary-light dark:text-ink-secondary-dark">
          Upload a close-up photo of a skin lesion. The EfficientNet + ResNet fusion model trained on
          HAM10000 will predict the most likely of 7 diagnosis classes, show a confidence breakdown, and
          highlight the regions it focused on.
        </p>
      </header>

      {backendStatus !== "ready" && backendStatus !== "checking" && (
        <div className="rounded-xl border border-status-critical/30 bg-status-critical/5 p-4 text-sm text-ink-primary-light dark:text-ink-primary-dark">
          {backendStatus === "unreachable" && (
            <>
              <span className="font-medium">Can&apos;t reach the backend.</span> Make sure the FastAPI
              server is running (see backend/README.md) and NEXT_PUBLIC_API_URL points at it.
            </>
          )}
          {backendStatus === "model_unavailable" && (
            <>
              <span className="font-medium">Backend is up, but the model isn&apos;t loaded.</span>{" "}
              {backendDetail ?? "Check the server logs."}
            </>
          )}
        </div>
      )}

      <UploadDropzone onFileSelected={handleFileSelected} disabled={loading} previewUrl={previewUrl} />

      {file && (
        <button
          onClick={handleAnalyze}
          disabled={loading || backendStatus !== "ready"}
          className="w-full rounded-xl bg-brand px-6 py-3 text-sm font-semibold text-white transition-colors hover:bg-brand-dark disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Analyze image"}
        </button>
      )}

      {error && (
        <div className="rounded-xl border border-status-critical/30 bg-status-critical/5 p-4 text-sm text-status-critical">
          {error}
        </div>
      )}

      {result && previewUrl && <ResultPanel result={result} originalPreviewUrl={previewUrl} />}

      <footer className="pb-8 text-center text-xs text-ink-muted">
        Research demo only &middot; not a medical device &middot; always consult a dermatologist.
      </footer>
    </main>
  );
}
