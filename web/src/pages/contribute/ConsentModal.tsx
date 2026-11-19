import { useState } from "react";
import { ShieldCheck } from "lucide-react";
import type { License } from "../../lib/api";
import { useContributor } from "../../lib/contributor";
import { Button, Field, Modal } from "../../components/ui";

export function ConsentModal({ open, onClose, onDecline }: { open: boolean; onClose: () => void; onDecline?: () => void }) {
  const { profile, update } = useContributor();
  const [name, setName] = useState(profile.displayName);
  const [license, setLicense] = useState<License>(profile.license);
  const [cloning, setCloning] = useState(profile.voiceCloningOptIn);
  const [agree, setAgree] = useState(false);
  const firstTime = !profile.consent;

  return (
    <Modal open={open} onClose={onClose} dismissible={!firstTime} title={firstTime ? "Before you contribute" : "Your contributor settings"}>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!agree && firstTime) return;
          update({
            displayName: name.trim(),
            license,
            voiceCloningOptIn: cloning,
            consent: true,
            consentAt: profile.consentAt ?? new Date().toISOString(),
          });
          onClose();
        }}
      >
        <ul className="space-y-3 text-[16px] leading-relaxed text-muted">
          <li className="flex gap-3">
            <ShieldCheck className="mt-0.5 size-5 shrink-0 text-ok" strokeWidth={1.6} aria-hidden />
            <span>
              Your text and recordings are shared under an open licence so anyone can build better tools for these languages.
            </span>
          </li>
          <li className="flex gap-3">
            <ShieldCheck className="mt-0.5 size-5 shrink-0 text-ok" strokeWidth={1.6} aria-hidden />
            <span>Your voice is never used to clone a voice unless you opt in below. That is a separate choice.</span>
          </li>
          <li className="flex gap-3">
            <ShieldCheck className="mt-0.5 size-5 shrink-0 text-ok" strokeWidth={1.6} aria-hidden />
            <span>Everything stays on this machine until you, or the server owner, choose to sync it.</span>
          </li>
        </ul>

        <div className="mt-6 flex flex-col gap-5">
          <Field
            label="Display name"
            htmlFor="contrib-name"
            optional
            help={`Shown on the leaderboard. Leave empty to stay anonymous as ${profile.id}.`}
          >
            <input
              id="contrib-name"
              className="field-input"
              maxLength={40}
              value={name}
              autoComplete="nickname"
              onChange={(e) => setName(e.target.value)}
            />
          </Field>

          <fieldset>
            <legend className="field-label">Licence for your contributions</legend>
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              {(
                [
                  { id: "CC-BY-4.0", title: "CC-BY-4.0", body: "Free to use, with credit to contributors." },
                  { id: "CC0-1.0", title: "CC0", body: "Public domain. No credit required." },
                ] as const
              ).map((o) => (
                <label
                  key={o.id}
                  className={`cursor-pointer rounded-2xl p-4 ring-1 transition-colors has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent ${
                    license === o.id ? "bg-accent-soft ring-accent" : "ring-line-strong hover:bg-sunken"
                  }`}
                >
                  <input
                    type="radio"
                    name="license"
                    className="sr-only"
                    checked={license === o.id}
                    onChange={() => setLicense(o.id)}
                  />
                  <span className="block text-[15px] font-semibold text-fg">
                    {o.title}
                    {o.id === "CC-BY-4.0" && <span className="ml-1.5 font-normal text-subtle">default</span>}
                  </span>
                  <span className="mt-1 block text-[14px] text-muted">{o.body}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <label className="flex items-start gap-3 text-[16px] text-fg">
            <input type="checkbox" className="mt-1 size-4" checked={cloning} onChange={(e) => setCloning(e.target.checked)} />
            <span>
              Also allow my recordings to train synthetic voices.{" "}
              <span className="text-subtle">Optional. You can change this later.</span>
            </span>
          </label>

          {firstTime && (
            <label className="flex items-start gap-3 rounded-2xl bg-sunken p-4 text-[16px] text-fg">
              <input type="checkbox" className="mt-1 size-4" checked={agree} onChange={(e) => setAgree(e.target.checked)} required />
              <span>I agree to share my contributions under the licence above, and confirm they are my own work.</span>
            </label>
          )}
        </div>

        <div className="mt-7 flex justify-end gap-2">
          <Button variant="ghost" onClick={firstTime && onDecline ? onDecline : onClose}>
            {firstTime ? "Not now" : "Cancel"}
          </Button>
          <Button type="submit" disabled={firstTime && !agree}>
            {firstTime ? "Start contributing" : "Save"}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
