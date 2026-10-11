# StratLens: 5-minute pitch

**Set-up:** run `streamlit run app.py` at 1280 px or wider, zoom 110–125%. Open **Overview**. In a second tab, open **Briefs** with *Dabur India FY2026* generated (it's cached). Keep `docs/screenshots/` as a backup.

| Time | Section | Say | Show |
|---|---|---|---|
| 0:00–0:30 | **Problem** | "An FMCG annual report is 250 to 600 pages, and only about 20 of them are strategy. If you follow HUL, Dabur and Nestlé, how do you see what each one talks about, how concretely, and what changed this year?" | Overview title |
| 0:30–1:30 | **Approach: the NLP** | "We built a nine-step NLP pipeline. We extract the Chairman's letter and MD&A, clean them with regular expressions, split them into sentences, and hide company names and numbers so the model learns topics, not authors. Then TF-IDF turns words into numbers, and a trained classifier sorts each passage into six strategy themes. spaCy's named-entity recognition scores how concrete each passage is." | How it works → *The NLP at each step* table |
| 1:30–2:00 | **Does it work?** | "On 177 passages it never saw, it's right 81% of the time, with a macro-F1 of 0.77 against 0.49 for keyword matching. A modern embedding model scored the same, so we kept the one whose every decision traces back to words." | Evaluation → KPI tiles, Models table |
| 2:00–2:50 | **What we found** | "Dabur leads with brand-building, about a third of its strategy talk. HUL spreads evenly and gives sustainability three times the space of the others. Nestlé leans on distribution. And only one year-on-year shift is real: Dabur's digital share, from 6 to 14%. Everything else is noise." Click a heatmap cell: "every number opens the sentences behind it." | Overview → *What we found*, fingerprint, evidence |
| 2:50–3:20 | **Why this label?** | "Paste any sentence and watch each NLP step: masking, TF-IDF weights, class probabilities, specificity." | How it works → inspector |
| 3:20–3:50 | **Generated brief** | "Briefs turns this into a one-pager. The LLM only writes prose from our computed numbers, and every number it writes is checked against the data." | Briefs → Download PDF |
| 3:50–4:30 | **Key audit finding** | "Hiding names doesn't change accuracy, yet another classifier still guesses the company 79% of the time from masked text. House writing style is a fingerprint, so some cross-company differences may be style, not strategy. Also, when the model is wrong it usually *misses* strategy (15 errors) rather than inventing it (7), so our shares are conservative." | Evaluation → *Key audit findings* |
| 4:30–4:50 | **Honest limitation** | "Our labels were drafted by an AI following written guidelines and haven't been checked by a person, so 0.77 measures agreement with those drafts. And it's three companies, five reports." | Evaluation → Errors → Limitations |
| 4:50–5:00 | **Close** | "StratLens turns 1,500 pages into a traceable view of what companies emphasise, and it tells you where not to trust it." | Overview |

**If something fails:** the app runs without the LLM (template text). Screenshots are in `docs/screenshots/`.

**Likely questions**
- *Why not BERT or a transformer?* We tested MiniLM sentence embeddings. They tied with TF-IDF (−0.01, CI −0.07 to +0.06), and TF-IDF is explainable word by word.
- *Why hide company names?* So the model learns themes rather than who wrote the text. The audit shows style still leaks, and we report that.
- *Does the LLM make things up?* It never labels text or computes numbers. It rewrites computed figures, cites passages, and falls back to template text if a number doesn't match the data.
