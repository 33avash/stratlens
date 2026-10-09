# StratLens labelling guidelines

**Unit:** one passage (a paragraph chunk of up to ~70 words) from a Chairman/CEO letter or the MD&A of an Indian FMCG annual report.
**Task:** assign exactly one theme by the passage's **dominant strategic purpose**, meaning what the company is trying to achieve. Ignore the words it happens to use.
**Provenance:** the 885 labels in `data/labels/labels.csv` were drafted by an AI assistant (Claude) during the build, following these guidelines (`label_source = "ai_drafted"`). They are not yet human-verified; see the review sample in `data/labels/human_review_sample.csv`.

All example passages below are quoted from the reports in the corpus; page numbers are PDF page numbers.

---

## PROD: Product innovation & portfolio
**Definition:** launching, reformulating, premiumising or extending products, and portfolio choices (acquisitions, divestments, entering or exiting categories).
- *"The year saw Dabur enter the Nutraceuticals market with a new brand, Siens by Dabur. The brand is focused on delivering innovative, science-backed formulations made with natural ingredients."* (Dabur FY2025, MD&A, p. 136)
- *"We announced the acquisition of the active-led premium brand Minimalist, launched disruptive innovations in our existing brands like TRESemmé Silk Press, and introduced Liquid IV, Unilever's largest wellbeing brand."* (HUL FY2025, letter, p. 26)

**Don't confuse with:** BRAND, when the launch is only the backdrop to a campaign or ambassador. DIST, when a pack or price point exists to widen reach.

## DIST: Market expansion & distribution
**Definition:** reaching more outlets, villages, geographies or channels; route-to-market; distributor and trade relationships; exports and entry into new countries.
- *"Dabur's retail reach today is amongst the widest in the FMCG industry in India, covering 8.4 million retail outlets, up from 7.9 million a year earlier. Our village coverage also expanded by around 16000 during the year to 1.32 lakh villages."* (Dabur FY2025, MD&A, p. 132)
- *"A decade ago, your Company was predominantly urban-weighted. In 2019, we commenced our RUrban journey, deepening our presence in smaller towns that connect urban and rural India. From 2025, we extended our presence further into rural markets."* (Nestlé FY2026, letter, p. 8)

**Don't confuse with:** DIGI, when the channel in question is e-commerce, quick commerce or D2C specifically. OTHER, for market-share results with no stated reach action.

## DIGI: Digital & e-commerce
**Definition:** e-commerce, quick commerce, D2C, eB2B ordering platforms, digital and influencer marketing as a capability, data, analytics and AI used for growth or consumer engagement.
- *"We have significantly boosted our investments in digital marketing – today 40% of our spends are on digital media. We have over 12,000 influencers whom we collaborate with for our brands."* (HUL FY2025, letter, p. 27)
- *"Ecommerce continues to be our fastest growing channel, with Quick Commerce being the growth engine within this cohort."* (Dabur FY2026, MD&A, p. 196)

**Don't confuse with:** OPS, when the technology serves the supply chain or manufacturing. BRAND, when the passage is about a campaign's message rather than the digital medium. OTHER, for cyber-security and IT risk controls.

## OPS: Operations & supply chain
**Definition:** manufacturing, capacity, procurement, logistics, cost and productivity programmes, commodity and input-cost management, product quality and safety systems.
- *"The factory focused on continuous improvement through Kaizen initiatives, low‑cost automation, and packaging line de‑bottlenecking, leading to higher line speeds and a 3% year‑on‑year productivity improvement."* (Dabur FY2026, MD&A, p. 197)
- *"AI and advanced analytics are transforming our supply chain—making it more responsive, more agile and more resilient. … five of our factories have been recognised as Advanced Fourth Industrial Revolution (4IR) Lighthouses."* (HUL FY2026, letter, p. 10)

**Don't confuse with:** SUST, when a plant initiative is framed around energy, water or emissions. DIGI, when AI is applied to marketing or sales.

## SUST: Sustainability & sourcing
**Definition:** packaging and plastics, water, emissions and energy, waste and circularity, regenerative agriculture, responsible sourcing, and farmer livelihoods tied to sourcing.
- *"We also achieved a 30% reduction in water intensity and maintained our leadership in plastic waste management with 105% plastic waste positivity. Our packaging is now 87% recyclable…"* (Dabur FY2025, letter, p. 20)
- *"We continue to scale regenerative agriculture, now covering over 61,000 hectares. On plastics, we continued efforts to reduce, redesign and recycle packaging."* (HUL FY2026, letter, p. 10)

**Don't confuse with:** OTHER, for social CSR such as education, health camps and diversity. OPS, when the stated goal is cost or efficiency.

## BRAND: Consumer & brand strategy
**Definition:** consumer trends and insight, segmentation, positioning, advertising campaigns, ambassadors, activations, sampling, packaging refreshes for visual identity, and brand-purpose (cause) campaigns.
- *"Dabur signed South superstar Nagarjuna Akkineni as the new brand ambassador of Meswak toothpaste. This collaboration sought to strengthen Dabur Meswak's presence in South India… A new campaign, 'Nothing like Meswak' was rolled out across television screens and digital platforms."* (Dabur FY2025, MD&A, p. 139)
- *"Consumer-first thinking must permeate the value chain, from manufacturing, supply chain to sales and marketing. We listen closely - market by market, city by city, town by town, village by village and turn insight into action."* (Nestlé FY2026, letter, p. 7)

**Don't confuse with:** DIGI, when the digital medium is the point (creators, social-first, AI-personalised media). PROD, when a new product is the subject.

## OTHER: Other / unclear
**Definition:** macro-economy, financial results, dividends, governance, risk frameworks, HR and culture, thanks and greetings, generic statements, fragments, and passages with several levers and no dominant one.
- *"India's economic progress in recent years has been encouraging. Our GDP continues to grow among the fastest globally, supported by structural reforms…"* (HUL FY2026, letter, p. 9)
- *"During the financial year 2024-25, the Company declared a dividend of INR 8 per share, which resulted in total dividend amounting to INR 1,418 Crores."* (Dabur FY2025, MD&A, p. 151)

---

## Overlap and tie-break rules
1. **Dominant purpose wins.** Example: a low-price sachet launched to reach rural buyers → DIST. An AI demand-forecasting tool to cut inventory → OPS, not DIGI.
2. **Results without an action → OTHER.** "Brand X grew double digits / gained 140 bps share" is OTHER unless the passage states the strategic action behind it.
3. **Mixed lists with no dominant lever → OTHER.** Example: "driven by launches, distribution, TV campaign and influencers".
4. **Launch vs campaign.** If the launch is the subject → PROD. If the message, ambassador or film is the subject → BRAND.
5. **Digital medium vs message.** Influencer or creator programmes, "social-first" or "digital-first" approaches, and AI-personalised media → DIGI. A specific creative idea or emotional story that happens to run online → BRAND.
6. **Technology by purpose.** Tools for the sales force or distributors (e.g. sales-force apps) → DIST. eB2B ordering platforms for retailers → DIGI. Supply-chain AI → OPS. IT and cyber security → OTHER.
7. **Plants and sustainability.** A new plant or capacity → OPS. A plant passage whose stated emphasis is energy, emissions or water → SUST.
8. **Portfolio moves.** Acquisitions, divestments, demergers and venture platforms for new brands → PROD, even when the target is a D2C brand.
9. **Consumer trends** (affluence, premiumisation, wellness, value-seeking) → BRAND. Pure macro-economics → OTHER.
10. **Packaging.** Recyclability or less plastic → SUST. Refreshed look or identity → BRAND. Upgraded product format → PROD.
11. **CSR.** Environmental or sourcing livelihoods (beekeeping, farmers) → SUST. Brand-led cause campaigns → BRAND. Other social programmes → OTHER.
12. **Product quality and safety systems** → OPS. Nutrition and on-pack labelling compliance → OTHER.

Rules 5 and 6 caused the most hesitation while drafting. They are the first place to check when reviewing labels.
