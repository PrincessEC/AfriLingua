import { Hero, ProofStrip } from "./landing/Hero";
import { Community, Faq, FinalCta, HowItWorks, LanguagesSection, UseCases } from "./landing/Sections";
import { TryIt } from "./landing/TryIt";
import { Developers } from "./landing/Developers";

export default function Landing() {
  return (
    <>
      <Hero />
      <ProofStrip />
      <HowItWorks />
      <TryIt />
      <UseCases />
      <Community />
      <Developers />
      <LanguagesSection />
      <Faq />
      <FinalCta />
    </>
  );
}
