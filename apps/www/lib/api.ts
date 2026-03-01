import { IMAGE_MAX_DIMENSION } from "./constants";
import type { AnalyzeResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

export async function analyzeImage(imageBlob: Blob): Promise<AnalyzeResponse> {
  const resized = await resizeImage(imageBlob);
  const base64 = await blobToBase64(resized);

  const response = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: base64 }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.error ?? `Request failed (${response.status})`);
  }

  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error ?? "Analysis failed");
  }

  return data as AnalyzeResponse;
}

export async function warmLambda(): Promise<void> {
  try {
    await fetch(`${API_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ warm: true }),
    });
  } catch {
    // Silently ignore warm failures
  }
}

async function blobToBase64(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

function resizeImage(blob: Blob, maxDimension = IMAGE_MAX_DIMENSION): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(blob);

    img.onload = () => {
      URL.revokeObjectURL(url);
      const { width, height } = img;

      if (width <= maxDimension && height <= maxDimension) {
        resolve(blob);
        return;
      }

      const scale = maxDimension / Math.max(width, height);
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Failed to get canvas context"));
        return;
      }

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (result) => (result ? resolve(result) : reject(new Error("Failed to resize image"))),
        "image/jpeg",
        0.85,
      );
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load image"));
    };

    img.src = url;
  });
}
