"use client";

import { useState } from "react";
import type { ClassProbability, RiskLevel } from "@/lib/api";

// Status color per class, driven by clinical risk rather than plain category
// identity -- this is a "how worried should you be" signal, so it uses the
// reserved status palette (good/serious/critical) rather than a categorical hue.
const RISK_TO_STATUS: Record<RiskLevel, { color: string; label: string }> = {
  low: { color: "#0ca30c", label: "Low risk" },
  high: { color: "#ec835a", label: "High risk" },
  critical: { color: "#d03b3b", label: "Critical risk" },
  unknown: { color: "#898781", label: "Unknown" },
};

interface Props {
  probabilities: ClassProbability[];
  riskByKey: Record<string, RiskLevel>;
}

export default function ProbabilityChart({ probabilities, riskByKey }: Props) {
  const [hovered, setHovered] = useState<string | null>(null);
  const sorted = [...probabilities].sort((a, b) => b.probability - a.probability);
  const usedRisks = Array.from(new Set(sorted.map((p) => riskByKey[p.key] ?? "unknown"))) as RiskLevel[];

  return (
    <div className="viz-root w-full">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ink-primary-light dark:text-ink-primary-dark">
          Probability by class
        </h3>
        <div className="flex flex-wrap items-center gap-3">
          {usedRisks
            .filter((r) => r !== "unknown")
            .map((r) => (
              <span key={r} className="flex items-center gap-1.5 text-xs text-ink-secondary-light dark:text-ink-secondary-dark">
                <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: RISK_TO_STATUS[r].color }} />
                {RISK_TO_STATUS[r].label}
              </span>
            ))}
        </div>
      </div>

      <div className="flex flex-col gap-2" role="table" aria-label="Class probability breakdown">
        {sorted.map((p) => {
          const risk = riskByKey[p.key] ?? "unknown";
          const status = RISK_TO_STATUS[risk];
          const pct = p.probability * 100;
          const isHovered = hovered === p.key;
          return (
            <div
              key={p.key}
              role="row"
              className="group relative flex items-center gap-3"
              onMouseEnter={() => setHovered(p.key)}
              onMouseLeave={() => setHovered(null)}
            >
              <div className="w-32 shrink-0 truncate text-xs text-ink-secondary-light dark:text-ink-secondary-dark" title={p.full_name}>
                {p.full_name}
              </div>
              <div className="relative h-3 flex-1 overflow-hidden rounded-full bg-line-hairline-light dark:bg-line-hairline-dark">
                <div
                  className="h-full rounded-full transition-[width] duration-300 ease-out"
                  style={{ width: `${Math.max(pct, 1.5)}%`, backgroundColor: status.color }}
                />
              </div>
              <div className="w-14 shrink-0 text-right text-xs font-medium tabular-nums text-ink-primary-light dark:text-ink-primary-dark">
                {pct.toFixed(1)}%
              </div>
              {isHovered && (
                <div className="pointer-events-none absolute -top-8 left-32 z-10 rounded-md bg-ink-primary-light px-2 py-1 text-xs text-white shadow-md dark:bg-ink-primary-dark dark:text-ink-primary-light">
                  {p.full_name}: {pct.toFixed(2)}% &middot; {status.label}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
