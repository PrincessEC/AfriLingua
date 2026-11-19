import { useId } from "react";
import type { LanguageSpec } from "../lib/api";
import { Select } from "./ui";

export function LanguageSelect({
  value,
  onChange,
  languages,
  label,
  allowAuto = false,
  exclude,
  hideLabel = false,
  className,
  id: idProp,
  disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  languages: LanguageSpec[];
  label: string;
  allowAuto?: boolean;
  exclude?: string[];
  hideLabel?: boolean;
  className?: string;
  id?: string;
  disabled?: boolean;
}) {
  const autoId = useId();
  const id = idProp ?? autoId;
  return (
    <div className={className}>
      <label htmlFor={id} className={hideLabel ? "sr-only" : "field-label mb-2 block"}>
        {label}
      </label>
      <Select id={id} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
        {allowAuto && <option value="auto">Detect language</option>}
        {languages
          .filter((l) => !exclude?.includes(l.name))
          .map((l) => (
            <option key={l.name} value={l.name}>
              {l.display_name}
              {l.native_name && l.native_name !== l.display_name ? ` · ${l.native_name}` : ""}
            </option>
          ))}
      </Select>
    </div>
  );
}
