import { Hero } from "./landing/Hero";
import { Speed } from "./landing/Speed";
import { TryIt } from "./landing/TryIt";
import { Developers } from "./landing/Developers";
import { Train } from "./landing/Train";
import { Community, Faq, FinalCta, LanguagesSection, Products } from "./landing/Sections";

export default function Landing() {
  return (
    <>
      <Hero />
      <Speed />
      <TryIt />
      <Developers />
      <Train />
      <Products />
      <LanguagesSection />
      <Community />
      <Faq />
      <FinalCta />
    </>
  );
}
