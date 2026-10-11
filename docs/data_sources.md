# Data sources

All reports are the companies' statutory annual reports as filed with the National Stock Exchange of India (NSE)
under Regulation 34 of the SEBI Listing Regulations, except where noted. PDFs are not in the repository
(copyrighted, 8–31 MB each); download them into `data/raw/` with the names below, then run
`python -m stratlens.build`. Page ranges used are in `config/section_overrides.csv`.

| File in `data/raw/` | Company | Year | Source |
|---|---|---|---|
| `DABUR_FY2025.pdf` | Dabur India | 2024-25 | Company website (downloaded 2026-10-09; exact URL not recorded). NSE copy: [AR_26811](https://nsearchives.nseindia.com/annual_reports/AR_26811_DABUR_2024_2025_A_09072025180150.pdf) |
| `DABUR_FY2026.pdf` | Dabur India | 2025-26 | [NSE AR_29588](https://nsearchives.nseindia.com/annual_reports/AR_29588_DABUR_2025_2026_A_28311115_06072026183341.pdf) (byte size matches) |
| `HUL_FY2025.pdf` | Hindustan Unilever | 2024-25 | Company website (downloaded 2026-10-09; exact URL not recorded). NSE copy: [AR_26462](https://nsearchives.nseindia.com/annual_reports/AR_26462_HINDUNILVR_2024_2025_A_29052025214241.pdf) |
| `HUL_FY2026.pdf` | Hindustan Unilever | 2025-26 | Company website (downloaded 2026-10-09; exact URL not recorded). NSE copy: [AR_29330](https://nsearchives.nseindia.com/annual_reports/AR_29330_HINDUNILVR_2025_2026_A_21643813_02062026215058.pdf) |
| `nestle_FY2025.pdf` | Nestlé India | 2024-25 | [NSE AR_26487](https://nsearchives.nseindia.com/annual_reports/AR_26487_NESTLEIND_2024_2025_A_03062025002440.pdf) (downloaded 2026-10-11) |
| `NESTLE_FY2026.pdf` | Nestlé India | 2025-26 | [NSE AR_29352](https://nsearchives.nseindia.com/annual_reports/AR_29352_NESTLEIND_2025_2026_A_19722134_05062026154628.pdf) (byte size matches) |
| `marico_FY2025.pdf` | Marico | 2024-25 | [NSE AR_27099](https://nsearchives.nseindia.com/annual_reports/AR_27099_MARICO_2024_2025_U_24072025194510.pdf) (revised filing of 24 Jul 2025; downloaded 2026-10-11) |
| `marico_FY2026.pdf` | Marico | 2025-26 | [NSE AR_29675](https://nsearchives.nseindia.com/annual_reports/AR_29675_MARICO_2025_2026_A_25834811_09072026124936.pdf) (downloaded 2026-10-11) |
| `britannia_FY2025.pdf` | Britannia Industries | 2024-25 | [NSE AR_27040](https://nsearchives.nseindia.com/annual_reports/AR_27040_BRITANNIA_2024_2025_A_19072025234802.pdf) (downloaded 2026-10-11) |
| `britannia_FY2026.pdf` | Britannia Industries | 2025-26 | [NSE AR_32044](https://nsearchives.nseindia.com/annual_reports/AR_32044_BRITANNIA_2025_2026_A_8821120_16092026190351.pdf) (downloaded 2026-10-11) |
| `godrej_FY2025.pdf` | Godrej Consumer Products | 2024-25 | [NSE AR_26977](https://nsearchives.nseindia.com/annual_reports/AR_26977_GODREJCP_2024_2025_A_16072025185902.pdf) (downloaded 2026-10-11) |
| `godrej_FY2026.pdf` | Godrej Consumer Products | 2025-26 | [NSE AR_32043](https://nsearchives.nseindia.com/annual_reports/AR_32043_GODREJCP_2025_2026_A_9497368_16092026154019.pdf) (downloaded 2026-10-11) |

## What is read from each report

Only the Chairman/CEO letter and the MD&A (see `config/section_overrides.csv` for page ranges and reasons).
Report-specific notes:

- **Britannia** publishes no Chairman/CEO letter in either year; only its MD&A is read.
- **Godrej Consumer Products** integrates its MD&A into the integrated report (printed pp. 48–223 in FY2026).
  StratLens reads the business model and risk sections and all three Strategic Pillars (category development;
  funded by radical simplification; people and planet). Only the TCFD climate report is excluded. Pillar 3 covers
  HR, CSR, human rights and environmental data that other companies put in their separate BRSR (not read), so
  Godrej's sustainability share is not like-for-like with its peers. The "letter" for Godrej is the Executive Chairperson's letter plus the MD & CEO interview.
- **Nestlé India** embeds its MD&A in the Board's Report; the business narrative and the risks / MD&A pages are read.
