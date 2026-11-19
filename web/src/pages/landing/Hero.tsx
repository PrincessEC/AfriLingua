import { motion, useReducedMotion } from "motion/react";
import { Cpu, Plug, ShieldCheck, Users, WifiOff } from "lucide-react";
import { CtaLink } from "../../components/ui";
import { Rings } from "../../components/Brand";
import { LiveCaptionDemo } from "./LiveCaptionDemo";

const ease = [0.16, 1, 0.3, 1] as const;

export function Hero() {
  const reduce = useReducedMotion();
  const enter = (delay: number) =>
    reduce
      ? {}
      : {
          initial: { opacity: 0, y: 22, filter: "blur(8px)" },
          animate: { opacity: 1, y: 0, filter: "blur(0px)" },
          transition: { duration: 0.9, delay, ease },
        };

  return (
    <section className="relative overflow-hidden">
      <Rings className="pointer-events-none absolute -left-40 top-10 hidden size-[640px] text-fg opacity-[0.06] lg:block" count={9} />
      <div className="mx-auto grid max-w-[1240px] items-center gap-12 px-5 pb-16 pt-12 md:px-8 md:pt-16 lg:min-h-[calc(100dvh-5rem)] lg:grid-cols-[1.08fr_1fr] lg:gap-14 lg:pb-20 lg:pt-10">
        <div className="relative">
          <motion.h1
            {...enter(0.05)}
            className="text-[2.6rem] font-semibold leading-[1.02] tracking-[-0.045em] text-fg sm:text-[3.4rem] lg:text-[3.9rem]"
          >
            Real-time interpretation for{" "}
            <span className="relative inline-block sm:whitespace-nowrap">
              African languages.
              <svg
                aria-hidden
                viewBox="0 0 300 14"
                preserveAspectRatio="none"
                className="absolute -bottom-2 left-0 h-3 w-full text-ochre"
              >
                <motion.path
                  d="M3 9 C 60 3, 120 3, 170 7 S 260 12, 297 5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3.5"
                  strokeLinecap="round"
                  initial={reduce ? false : { pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1.1, delay: 0.7, ease: [0.65, 0, 0.35, 1] }}
                />
              </svg>
            </span>
          </motion.h1>
          <motion.p {...enter(0.18)} className="mt-7 max-w-[34rem] text-lg leading-relaxed text-muted md:text-[1.2rem]">
            <strong className="font-semibold text-fg">Offline. On any laptop.</strong> A speaker talks and every phone in the
            room follows along in Yoruba, Igbo or Hausa.
          </motion.p>
          <motion.div {...enter(0.3)} className="mt-9 flex flex-col gap-3 sm:flex-row sm:items-center">
            <CtaLink to="/live">Start interpreting</CtaLink>
            <CtaLink to="/contribute" variant="secondary">
              Contribute your language
            </CtaLink>
          </motion.div>
        </div>

        <motion.div
          initial={reduce ? false : { opacity: 0, y: 30, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 1, delay: 0.25, ease }}
          className="relative lg:-mr-4"
        >
          <LiveCaptionDemo />
        </motion.div>
      </div>
    </section>
  );
}

const PROOF = [
  { icon: WifiOff, title: "Works offline", body: "No internet after setup" },
  { icon: Cpu, title: "CPU-fast", body: "int8 models, no GPU needed" },
  { icon: Users, title: "Community-trained", body: "Speakers improve every release" },
  { icon: Plug, title: "MCP + OpenAI-compatible", body: "Drop into agents and apps" },
  { icon: ShieldCheck, title: "Data stays on your device", body: "Nothing is sent anywhere" },
];

export function ProofStrip() {
  return (
    <section aria-label="Highlights" className="border-y border-line bg-raised/50">
      <ul className="no-scrollbar mx-auto flex max-w-[1240px] snap-x snap-mandatory gap-0 overflow-x-auto px-2 md:grid md:grid-cols-5 md:px-8">
        {PROOF.map(({ icon: Icon, title, body }, i) => (
          <li
            key={title}
            className={`flex min-w-[15rem] snap-start items-start gap-3 px-4 py-6 md:min-w-0 md:px-5 ${i > 0 ? "md:border-l md:border-line" : ""}`}
          >
            <Icon className="mt-0.5 size-5 shrink-0 text-accent-text" strokeWidth={1.5} aria-hidden />
            <div>
              <p className="text-[15px] font-semibold leading-tight text-fg">{title}</p>
              <p className="mt-1 text-[13px] leading-snug text-subtle">{body}</p>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
