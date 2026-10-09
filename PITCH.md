# StratLens: 5-minute pitch script

**Setup before class:** run `streamlit run app.py`, open the Benchmark page in a full-screen browser at 1280 px or wider, zoom 110–125% for the projector, and leave the Analyse reports page open in a second tab. Have `data/raw/HUL_FY2026.pdf` ready to upload. Close other tabs.

| Time | Section | Say | Show |
|---|---|---|---|
| 0:00–0:40 | **Problem** | "FMCG annual reports run to 250–600 pages. The strategy story is maybe 20 of them, buried among accounts, governance and sustainability tables. If you're an analyst comparing HUL, Dabur and Nestlé, how do you see what each one actually *talks about*, and whether that changed this year?" | Benchmark page title and KPI strip |
| 0:40–1:30 | **Approach** | "We restrict to the Chairman/CEO letter and MD&A, the narrative strategy sections, so every company is compared on the same footing. We clean and split them into 1,184 passages. A TF-IDF plus logistic regression classifier sorts each passage into six themes or Other. We trained it on 885 passages labelled under written guidelines. It scores macro-F1 0.78 on held-out text, against 0.49 for a keyword baseline." | Sidebar "About the data"; KPI strip (0.78) |
| 1:30–2:10 | **Fingerprint** | "Each row is one report's strategic narrative, summing to 100%. Dabur is brand- and product-led: over a third of its strategy text is about campaigns and consumers. HUL spreads evenly, and gives sustainability three times the space the others do." Point at the note: "This is disclosure emphasis, not spend or execution, and not a ranking." | Strategy fingerprint heatmap |
| 2:10–2:40 | **YoY shift** | "Year on year, Dabur's biggest move is digital: plus 7.6 points, from 6% to 14% of its narrative. HUL shifts toward brand and distribution. The scale check underneath reminds us that one passage moves a share by about a point, so small shifts are noise." | YoY diverging bars, then hover a bar |
| 2:40–3:10 | **Evidence** | "Every number is traceable. Click Dabur's digital bar…" (click) "…and here are the exact passages, with company, year, section, page and confidence. The highlighted words are the ones that drove the classification." | Click a bar, scroll to Evidence |
| 3:10–3:40 | **Why this label** | Switch to Model audit, then *Why this label?* → *Own text*: type "We expanded our direct reach to 1.5 million outlets and 4,000 new villages." "It's DIST with high confidence, because of *reach*, *outlets* and *villages*. These weights show association, not causation." | Model audit page, local explanation |
| 3:40–4:20 | **Key audit finding** | "The audit's main finding: the model is much weaker on Chairman's letters than on MD&A. Macro-F1 is 0.59 against 0.79. Letters are rhetorical and anecdotal, and they pack several strategies into one sentence. So the letter rows of the benchmark deserve more scepticism, and the review flag catches the least confident passages: accuracy is 52% below the threshold against 87% above." | Subgroup table and key finding note |
| 4:20–4:45 | **Honest limitation** | "The biggest limitation is the labels. They were drafted by an AI assistant following our guidelines, and they are not yet human-verified. We built a blind 150-passage review sample and a kappa script, but until a person labels it, our 0.78 measures agreement with the drafted labels, not with expert judgement. Also: three companies, five reports. Nestlé's FY2025 report is missing." | Label provenance note / Model card |
| 4:45–5:00 | **Close** | "StratLens turns 1,500 pages into a comparable, traceable view of strategic emphasis, with an audit that tells you where not to trust it. Thank you." | Back to Benchmark |

**If the live demo fails:** use the screenshots in `docs/screenshots/` (benchmark, audit, analyse) and narrate the same path.

**Likely questions:**
- *Why not BERT?* With 885 labels, an interpretable linear model is competitive, trains in seconds, and gives the term-level explanations the audit needs.
- *How were the letters made comparable?* By using the same sections for every company, with page ranges confirmed manually and listed in `data/sections.csv`.
- *Is a 7.6 pp shift real?* Dabur's digital passages rose from 11 to 27, a material change in emphasis. Shifts of 2–3 pp are within noise.
