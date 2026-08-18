import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Skin Lesion Classifier",
  description: "Upload a lesion photo and get a research-grade prediction from the HAM10000-trained fusion model.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
