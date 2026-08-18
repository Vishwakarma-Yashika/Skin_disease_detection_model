"use client";

import { useState } from "react";
import type { PredictionResponse, RiskLevel } from "@/lib/api";
import ProbabilityChart from "./ProbabilityChart";

const RISK_BADGE: Record<RiskLevel, { bg: string; text: string; label: string }> = {
  low: { bg: "bg-status-good/10", text: "text-status-good", label: "Low risk" },
  high: { bg: "bg-status-serious/10", text: "text-status-serious", label: "High risk" },
  critical: { bg: "bg-status-critical/10", text: "text-status-critical", label: "Critical risk" },
  unknown: { bg: "bg-ink-muted/10", text: "text-ink-muted", label: "Unknown" },
};

interface Props {
  result: PredictionResponse;
  originalPreviewUrl: string;
}

export default function ResultPanel({ result, originalPreviewUrl }: Props) {
  const [showOverlay, setShowOverlay] = useState(true);
  const badge = RISK_BADGE[result.risk_level as RiskLevel] ?? RISK_BADGE.unknown;
  const riskByKey = Object.fromEntries(
    result.probabilities.map((p) => [p.key, inferRiskFromCategoryAndKey(p.key, p.category)])
  );

  return (
    <div className="flex w-full flex-col gap-6">
      {/* Headline card */}
      <div className="rounded-xl border border-line-hairline-light bg-surface-light p-6 dark:border-line-hairline-dark dark:bg-surface-dark">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-ink-muted">Top prediction</p>
            <h2 className="mt-1 text-2xl font-semibold text-ink-primary-light dark:text-ink-primary-dark">
              {result.predicted_name}
            </h2>
            <p className="mt-1 text-sm text-ink-secondary-light dark:text-ink-secondary-dark">
              {(result.confidence * 100).toFixed(1)}% confidence
              {result.model_val_accuracy != null && (
                <> &middot; model validation accuracy {(result.model_val_accuracy * 100).toFixed(1)}%</>
              )}
            </p>
          </div>
          <span className={`shrink-0 rounded-full px-3 py-1 text-sm font-medium ${badge.bg} ${badge.text}`}>
            {badge.label}
          </span>
        </div>

        <p className="mt-4 text-sm leading-relaxed text-ink-secondary-light dark:text-ink-secondary-dark">
          {result.description}
        </p>
        <p className="mt-3 rounded-lg bg-brand/5 p-3 text-sm text-ink-primary-light dark:text-ink-primary-dark">
          <span className="font-medium">What to do: </span>
          {result.typical_action}
        </p>
      </div>

      {/* Image + Grad-CAM */}
      <div className="rounded-xl border border-line-hairline-light bg-surface-light p-6 dark:border-line-hairline-dark dark:bg-surface-dark">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-ink-primary-light dark:text-ink-primary-dark">
            What the model looked at
          </h3>
          {result.gradcam_overlay && (
            <button
              onClick={() => setShowOverlay((v) => !v)}
              className="rounded-md border border-line-hairline-light px-3 py-1 text-xs font-medium text-ink-secondary-light hover:border-brand hover:text-brand dark:border-line-hairline-dark dark:text-ink-secondary-dark"
            >
              {showOverlay ? "Show original" : "Show heatmap"}
            </button>
          )}
        </div>
        <div className="flex justify-center rounded-lg bg-line-hairline-light/40 p-2 dark:bg-line-hairline-dark/40">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={showOverlay && result.gradcam_overlay ? result.gradcam_overlay : originalPreviewUrl}
            alt={showOverlay ? "Grad-CAM heatmap overlay" : "Uploaded lesion photo"}
            className="max-h-80 rounded-md object-contain"
          />
        </div>
        {!result.gradcam_overlay && (
          <p className="mt-2 text-xs text-ink-muted">Grad-CAM overlay wasn&apos;t available for this prediction.</p>
        )}
        {result.gradcam_overlay && (
          <p className="mt-2 text-xs text-ink-muted">
            Warmer colors show the image regions that most influenced the model&apos;s top prediction.
          </p>
        )}
      </div>

      {/* Probability breakdown */}
      <div className="rounded-xl border border-line-hairline-light bg-surface-light p-6 dark:border-line-hairline-dark dark:bg-surface-dark">
        <ProbabilityChart probabilities={result.probabilities} riskByKey={riskByKey} />
      </div>

      {/* Disclaimer */}
      <div className="rounded-xl border border-status-warning/30 bg-status-warning/5 p-4 text-sm text-ink-secondary-light dark:text-ink-secondary-dark">
        <span className="font-medium text-ink-primary-light dark:text-ink-primary-dark">Important: </span>
        {result.disclaimer}
      </div>
    </div>
  );
}

// The /api/predict response only carries category (malignant/benign) per class,
// not the full risk_level -- approximate a risk tier client-side for the chart's
// status coloring. mel is called out as critical to match backend/app/class_info.py.
function inferRiskFromCategoryAndKey(key: string, category: string): RiskLevel {
  if (category !== "malignant") return "low";
  return key === "mel" ? "critical" : "high";
}
