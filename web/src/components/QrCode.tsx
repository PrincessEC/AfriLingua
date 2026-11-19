import { useEffect, useState } from "react";
import QRCode from "qrcode";
import { cn } from "../lib/utils";

/** Client-side QR code (no network). Always dark-on-light for reliable scanning. */
export function QrCode({ value, size = 240, className, label }: { value: string; size?: number; className?: string; label: string }) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    QRCode.toDataURL(value, {
      margin: 1,
      width: size * 2,
      errorCorrectionLevel: "M",
      color: { dark: "#0b0b0d", light: "#ffffff" },
    })
      .then((url) => alive && setSrc(url))
      .catch(() => alive && setSrc(null));
    return () => {
      alive = false;
    };
  }, [value, size]);
  return (
    <div
      className={cn("grid place-items-center rounded-xl bg-white p-3 ring-1 ring-black/5", className)}
      style={{ width: size + 24, height: size + 24 }}
    >
      {src ? (
        <img src={src} width={size} height={size} alt={label} className="block [image-rendering:pixelated]" />
      ) : (
        <div className="size-full animate-pulse rounded-lg bg-black/5" aria-hidden />
      )}
    </div>
  );
}
