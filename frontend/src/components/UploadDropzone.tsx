"use client";

import { useCallback, useRef, useState } from "react";

interface Props {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
  previewUrl: string | null;
}

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

export default function UploadDropzone({ onFileSelected, disabled, previewUrl }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      setError(null);
      const file = files?.[0];
      if (!file) return;
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError("Please upload a JPEG, PNG, or WebP image.");
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected]
  );

  return (
    <div className="w-full">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload a lesion photo"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (!disabled && (e.key === "Enter" || e.key === " ")) inputRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          if (!disabled) handleFiles(e.dataTransfer.files);
        }}
        className={`relative flex min-h-[280px] w-full flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 text-center transition-colors
          ${isDragging ? "border-brand bg-brand/5" : "border-line-hairline-light dark:border-line-hairline-dark"}
          ${disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:border-brand/60"}
        `}
      >
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt="Uploaded lesion preview"
            className="max-h-64 rounded-lg object-contain"
          />
        ) : (
          <>
            <UploadIcon />
            <p className="text-sm font-medium text-ink-primary-light dark:text-ink-primary-dark">
              Drag and drop a lesion photo, or click to browse
            </p>
            <p className="text-xs text-ink-muted">JPEG, PNG, or WebP &middot; close-up, well-lit, in focus</p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          className="hidden"
          disabled={disabled}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>
      {error && <p className="mt-2 text-sm text-status-critical">{error}</p>}
    </div>
  );
}

function UploadIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" className="text-ink-muted">
      <path
        d="M12 16V4M12 4L7 9M12 4l5 5M4 16v3a2 2 0 002 2h12a2 2 0 002-2v-3"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
