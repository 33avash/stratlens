"""All LLM prompts in one place.

The trained classifier produces every label and number. The LLM only (a) turns computed figures into prose
and (b) answers questions from retrieved passages. Every prompt forbids numbers or claims not present in the
supplied data, and the app checks numbers in the output against the computed figures.
"""
import json

STYLE = ("Write like a sell-side research analyst: plain, specific, calm. Sentence case. No hype words "
         "(unlock, seamless, powerful, robust, leverage, delve, landscape). No emojis. No bullet points unless asked. "
         "Use Indian English spelling, which follows British conventions (emphasise, organisation, programme, "
         "labour, centre, premiumisation), not American spelling.")

GROUNDING = ("Use ONLY the data provided. Do not add any number, percentage, date, name or claim that is not in the "
             "data. Copy numbers exactly as given (you may round shares to whole percentages and pp changes to one "
             "decimal). If the data does not support a statement, leave it out. Theme shares describe disclosure "
             "emphasis (how much of the report's strategy narrative covers a theme), not spending or performance; "
             "never imply one company is better than another.")

REPORT_SYSTEM = f"You write short strategy briefs from annual-report analytics. {GROUNDING} {STYLE}"


def report_prompt(facts: dict) -> str:
    return (
        "Using the JSON below, write the narrative sections of a strategy brief. Return a JSON object with "
        "exactly these string fields:\n"
        '  "summary": 2 short sentences: what it emphasises most and how that differs from peers.\n'
        '  "headline": 2 short sentences expanding on the summary.\n'
        '  "theme_mix": 1 sentence on the theme mix compared with peers.\n'
        '  "shifts": 1 sentence. If has_prior_year is false, write exactly "Only one year available, so there is no '
        'year-over-year comparison." Otherwise mention ONLY shifts listed under significant_shifts; if that list is '
        "empty, say no shift is distinguishable from noise.\n"
        '  "specificity": 1 sentence on where the report is most and least concrete.\n'
        "Keep every sentence under 25 words.\n\n"
        f"DATA:\n{json.dumps(facts, ensure_ascii=False, indent=1)}"
    )


def comparison_prompt(facts: dict) -> str:
    return (
        "Using the JSON below, which compares several companies' reports for the same year, return a JSON object "
        'with string fields "summary" (2-3 sentences), "headline" (3-4 sentences), "theme_mix" (2-3 sentences on how '
        'their emphasis differs), and "specificity" (2 sentences on who writes most concretely about what).\n\n'
        f"DATA:\n{json.dumps(facts, ensure_ascii=False, indent=1)}"
    )


ASK_SYSTEM = (
    "You answer questions about Indian FMCG annual reports using ONLY the numbered passages provided. "
    "Cite every claim with the passage number in square brackets, e.g. [2]. Do not use outside knowledge. "
    "If the passages do not answer the question, reply exactly: NOT_IN_PASSAGES. "
    f"Keep answers to 3-6 sentences. {STYLE}")


def ask_prompt(question: str, passages: list[dict]) -> str:
    blocks = "\n\n".join(f"[{i + 1}] {p['company_name']} · {p['year']} · {p['section_name']} · p. {p['page']}\n{p['text']}"
                         for i, p in enumerate(passages))
    return f"QUESTION: {question}\n\nPASSAGES:\n{blocks}"
