# StratLens: 5-minute demo script

**Before you start (2 minutes earlier):** run `streamlit run app.py`, open the app at 1280 px or wider, then
⋮ → **Demo mode** (or open `http://localhost:8501/?demo=1`). It warms every cache, which takes about 15 seconds
the first time, and shows a **Next →** bar at the bottom. Nothing on the path needs the internet. Numbers below
are from the current build (12 reports, model trained 2026-10-10); if you rebuild, read them off the screen instead.

---

### 1. The problem (20 s) · no screen needed
> "An annual report is 200 to 600 pages. The strategy story, meaning what the company says it will focus on, is
> maybe 20 of them, spread across the Chairman's letter and the management discussion. If you want to compare
> six FMCG companies over two years, that is 12 reports and thousands of pages. StratLens reads them for you
> and shows what each company emphasises, how concretely, and what changed, with the passage behind every
> number."

### 2. Home: findings and fingerprint (60 s) · Demo 1/6
- Point at the **pipeline strip**: "Six steps: read, clean, split, classify each passage into a theme, score how
  concrete it is, compare."
- **Three findings** (each card is the insight itself):
  - "Dabur talks about consumer & brand strategy about twice as much as its peer Marico: 34% of its strategy
    passages against 14%."
  - "Britannia doubled its focus on market expansion & distribution, 14% to 30%, and the change is bigger than
    chance."
  - "Dabur writes the most concrete strategy: 41% of its strategy passages name figures, dates or initiatives,
    against 24% on average."
- **Fingerprint:** "Six companies by six themes. Product innovation leads at three of the six." Click **Change**:
  "Coloured cells are real shifts; grey is noise. Only four of the 36 company-theme changes are real."
- Say once: "This measures what a report talks about, not what the company spends."
- **Next →**

### 3. Company: Dabur FY2026 (90 s) · Demo 2/6
- **Brief** (3 sentences): read the first sentence aloud. "The wording can be polished by an AI, but every number
  is inserted from the data." Mention the **PDF brief** button.
- **Benchmark:** "Coloured bar is Dabur, dark grey is its peer Marico, light grey is all six companies. It leans
  furthest from Marico on consumer & brand strategy." Click a bar: its passages appear at the top.
- **What's new:** "21 of Dabur's 172 strategy passages this year have nothing close in meaning in last year's
  report. Most are on brand: new campaigns such as the Marathi-language MediFacts campaign. On the right, what
  was dropped since FY2025."
- **Promises:** "We look for a figure plus a deadline or a commitment. Dabur makes none in FY2026; Godrej makes
  five, for example net zero Scope 1 and 2 emissions by 2035. For last year's promises we show 'mentioned again'
  or 'not mentioned'. We never claim one was achieved."
- **Signature phrases** (one line): "'Water positive enterprise', 'digital-first approach': words Dabur uses that
  its peers don't."
- **Evidence:** open one theme. "Every passage has a page number and 'why this theme': the words that pushed the
  model."
- **Next →**

### 4. Compare: Dabur FY25 → FY26 (45 s) · Demo 3/6
- "Same company, two years. The bars are the change in share; the whiskers are the likely range."
- "Only digital & e-commerce moved by more than chance: 6% to 14% of strategy passages. Everything grey could
  be the luck of which passages ended up in the report."
- Scroll to **Passages on digital & e-commerce**: FY2025 on the left, FY2026 on the right.
- (Optional) Click **HUL vs Godrej**: "Godrej gives sustainability 35% against HUL's 15%. Part of that is
  Godrej folding sustainability into its MD&A, which we flag in Limits."
- **Next →**

### 5. Ask (30 s) · Demo 4/6
- The example "How is Dabur talking about quick commerce?" is already filled in.
- "Retrieval finds the passages by keywords and by meaning; the answer cites them as [1], [2]. Click a
  citation to jump to the passage."
- If the answer box says no stored AI answer: "Offline or without an API key, it still finds the passages;
  only the written summary needs the AI." (Each passage shows its page.)
- **Next →**

### 6. How it works (60 s) · Demo 5/6 and 6/6
- **Pipeline inspector** (pre-selected example: "We aim to double our direct reach to 3 million outlets by
  FY27…"): walk the six numbered steps:
  1. cleaned text;
  2. names and numbers hidden ([NUM], [YEAR]);
  3. word weights, where "reach", "outlets" and "rural" push hardest;
  4. probabilities for the 7 classes;
  5. theme: market expansion & distribution, confidence 0.63;
  6. concreteness 3/5: a figure, a deadline and commitment language.

  "Gemini never labels passages or computes numbers."
- **Next →** to **Accuracy**: "Macro-F1 0.77 on 177 passages the model never saw, against 0.49 for keyword
  matching. A meaning-based embedding model ties with it, so we use the one whose every decision traces back to
  words." Point at the hand-check card (unseen companies and AI-drafted labels against my labels).
- Click **Limits** and read the first two: "It measures emphasis, not spending" and "Writing style reveals the
  company 79% of the time, so some differences are house style."

### 7. Optional: live upload (if time)
- **Add a report** → drop any annual-report PDF named like `acme_FY2026.pdf` → check the detected page ranges →
  **Classify and compare** → it lands on the same Company view, benchmarked against all six companies.
- Takes under a minute for a typical report.

---

**If something goes wrong:** every page works offline and without an API key. If the app was started without
demo mode, use the sidebar: Home → Company (Dabur, FY2026) → Compare (preset "Dabur FY25 → FY26") → Ask
(first example) → How it works.
