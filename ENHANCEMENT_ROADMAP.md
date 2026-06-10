# co-president-2026: Post-MVP Enhancement Roadmap

**Document Version**: 1.0  
**Last Updated**: 2026-06-10  
**Status**: Approved for Implementation  
**Scope**: SPEC-12 through SPEC-14 (Phases 1-3) + Future Phases 4-5

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State: MVP Achievements](#current-state-mvp-achievements)
3. [Enhancement Vision](#enhancement-vision)
4. [Architecture Overview](#architecture-overview)
5. [Phase 1: Geographic & Historical Foundation (SPEC-12)](#phase-1-geographic--historical-foundation-spec-12)
6. [Phase 2: Structural Fundamentals (SPEC-13)](#phase-2-structural-fundamentals-spec-13)
7. [Phase 3: Feature Matrix Integration (SPEC-14)](#phase-3-feature-matrix-integration-spec-14)
8. [Phase 4: Digital Signals Integration (Future)](#phase-4-digital-signals-integration-future)
9. [Phase 5: Hierarchical Model Evolution (Future)](#phase-5-hierarchical-model-evolution-future)
10. [Data Sources Inventory](#data-sources-inventory)
11. [Implementation Strategy](#implementation-strategy)
12. [Testing & Validation Framework](#testing--validation-framework)
13. [Integration with Existing MVP](#integration-with-existing-mvp)
14. [Risk Mitigation](#risk-mitigation)
15. [Dependency Graph](#dependency-graph)

---

## 1. Executive Summary

The MVP (SPEC-01 to SPEC-11) successfully demonstrated a **national-level Bayesian polling model** using reverse-time random walks and hierarchical pollster effects. The model predicts 2022 election results within ±5% using only national polls.

### The Enhancement Goal

Evolve from a polling-only temporal model to a **municipal-level hierarchical Bayesian framework** that integrates:

- Historical voting patterns (2002-2022)
- Socioeconomic fundamentals (poverty, demographics, infrastructure)
- Conflict and electoral risk factors
- Real-time digital signals (Google Trends, social media)

### Why This Matters

The research shows that historical municipal vote shares achieve **94% explained variance** (USANTOMAS benchmark). National polls alone cannot capture the geographic heterogeneity of Colombian voting behavior. The existing CEDAE/RNEC datasets already contain voting-station-level data going back to 1958 — the data exists, we just need to integrate it.

### Key Research References

- **Academic benchmark**: USANTOMAS 5 ML models study (Dialnet: `codigo=10404048`) — Feedforward Neural Network with CLR transformation achieved 56-94% explained variance
- **Google Trends validation**: SciELO 2023 study — <2% error on runoff winner using data from 36 hours before election (DOI: `S0012-73532023000100064`)
- **Turnout suppression coefficients**: PMC study (`PMC11929383`) — assassination events reduce turnout by 1.9-9.0 percentage points
- **Twitter emotion corpus**: arXiv `2407.07258` — BERT/GPT-3.5 labeled emotion time-series

---

## 2. Current State: MVP Achievements

### What We Built

| Spec | Module | Purpose |
|------|--------|---------|
| SPEC-01 | Scaffolding | Project structure, dependencies, tooling |
| SPEC-02 | `config.py` | Candidate maps, dates, pollster ratings, hyperparameters |
| SPEC-03 | `data_results.py` | Election results consolidation (Registraduría + MOE) |
| SPEC-04 | `data_polls.py` | Poll loading, cleaning, undecided redistribution |
| SPEC-05 | `aggregation.py` | Baseline weighted polling averages |
| SPEC-06 | `model_round1.py` | Dirichlet-Multinomial + reverse-time RW (1st round) |
| SPEC-07 | `model_runoff_simple.py` | Dirichlet-Multinomial(K=3) runoff model |
| SPEC-08 | `model_runoff_matrix.py` | Full probabilistic pairing matrix |
| SPEC-09 | `validation.py` + `plotting.py` | Backtesting, calibration, rolling forecast |
| SPEC-10 | `__main__.py` | CLI entry point |
| SPEC-11 | `_data_quality.py` | Pollster accuracy monitoring, time decay validation, methodology effect analysis |

### Data Currently Used

| File | Source | Rows | Granularity |
|------|--------|------|-------------|
| `encuestas_2022.csv` | recetas-electorales.com (Nelson Amaya) | 46 | National polls |
| `consultas.csv` | recetas-electorales.com | 65 | Coalition internal polls |
| `MMV_NACIONAL_PRESIDENTE_2022_1v.csv` | Registraduría Nacional | 727,510 | Polling-station-level |
| `MMV_NACIONAL_PRESIDENTE_2022_2v.csv` | Registraduría Nacional | 398,387 | Polling-station-level |
| `moe_vuelta1.csv` | MOE | 11,864 | Municipal-level |
| `moe_vuelta2.csv` | MOE | 5,550 | Municipal-level |

### What We're Missing

- **Geographic granularity**: All predictions are national aggregates
- **Structural predictors**: No socioeconomic or demographic features
- **Historical context**: No use of 2002-2018 voting patterns
- **Real-time signals**: No Google Trends or social media integration
- **Turnout modeling**: No suppression factors from conflict/violence

---

## 3. Enhancement Vision

### The 5-Layer Hierarchical Model

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: BASELINE & HISTORICAL ALIGNMENT                   │
│  - RNEC Historical Results (2002-2022)                      │
│  - Georeferenced DIVIPOLA Voting Booths                     │
│  - Municipal vote share trajectories                        │
│                                                             │
│  SPEC-12: DIVIPOLA + Historical Ingestion                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│  Layer 2: STRUCTURAL DEMOGRAPHICS                           │
│  - DANE CNPV 2018 Census (ethnicity, education, internet)   │
│  - Municipal IPM (Multidimensional Poverty Index)           │
│  - Population projections (2018 to 2022)                    │
│  - MOE Risk Maps / INDEPAZ / PDET / UNODC Coca             │
│                                                             │
│  SPEC-13: Socioeconomic + Risk Ingestion                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│  Layer 3: FEATURE INTEGRATION                               │
│  - Join all components on DIVIPOLA code                     │
│  - Validate data quality across 1,122 municipalities        │
│  - Pivot historical data to wide format                     │
│                                                             │
│  SPEC-14: Build Municipal Feature Matrix                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│  Layer 4: REAL-TIME PREFERENCE SIGNALS (FUTURE)             │
│  - Weighted Poll Aggregations (existing MVP)                │
│  - Google/YouTube Search Trends (pytrends)                  │
│  - Social Media Sentiment (Twitter emotion corpus)          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│  Layer 5: HIERARCHICAL MODEL (FUTURE)                       │
│  - Municipal-level priors from fundamentals                 │
│  - Hierarchical Bayesian posterior updating                 │
│  - Rollup to departmental and national aggregates           │
└─────────────────────────────────────────────────────────────┘
```

### Mathematical Framework

**Municipal Prior (from fundamentals)**:

```
p(Vote_Share_m) ~ Normal(
    α + β1 * Historical_m + β2 * Poverty_m + β3 * Ethnicity_m + β4 * Risk_m,
    σ_municipal
)
```

**Turnout Suppression Model**:

```
T_mt = α + β1 * Historical_Turnout_m + β2 * Violence_Index_m(t-1) + γ * X_m + ε_mt
```

where β2 has been empirically estimated at −0.090 to −0.019 (assassination of general victim, p < 0.01).

**Bayesian Update (polls + digital signals)**:

```
p(μ_m | polls, trends) ∝ p(polls | μ_m) × p(trends | μ_m) × p(μ_m)
```

**Digital Signal Weighting by Internet Penetration**:

```python
digital_weight[m] = base_weight * internet_penetration[m]
historical_weight[m] = 1 - digital_weight[m]
```

In highly connected municipalities, digital signals receive higher weight; in disconnected regions, rely more on historical vote patterns.

---

## 4. Architecture Overview

### New Directory Structure

```
co-president-2026/
├── src/co_president/
│   ├── ingestion/                           # NEW: Data acquisition pipelines
│   │   ├── __init__.py
│   │   ├── ingest_divipola.py               # SPEC-12: Geographic codes
│   │   ├── ingest_historical.py             # SPEC-12: 2002-2022 results
│   │   ├── ingest_socioeconomic.py          # SPEC-13: DANE census + IPM
│   │   ├── ingest_risk.py                   # SPEC-13: MOE risk + INDEPAZ
│   │   └── build_feature_matrix.py          # SPEC-14: Join all components
│   ├── fundamentals/                        # FUTURE: Feature engineering
│   │   ├── __init__.py
│   │   ├── historical_features.py
│   │   ├── demographic_features.py
│   │   └── risk_features.py
│   └── [existing MVP modules unchanged]
├── data/
│   ├── raw/                                 # NEW: Untouched API downloads
│   ├── fundamentals/                        # NEW: Cleaned components
│   │   ├── divipola_master.csv
│   │   ├── historical_results.csv
│   │   ├── socioeconomic.csv
│   │   └── risk_factors.csv
│   ├── processed/                           # NEW: Final joined matrix
│   │   └── municipal_feature_matrix.csv
│   └── [existing MVP data unchanged]
└── tests/
    ├── test_ingestion.py                    # NEW: Ingestion pipeline tests
    └── [existing MVP tests unchanged]
```

### Dependency Additions

```toml
# Add to [project] dependencies in pyproject.toml
"requests>=2.31",              # HTTP client for APIs
"sodapy>=2.2",                 # Socrata API client (datos.gov.co)
"beautifulsoup4>=4.12",        # HTML parsing for static pages
"playwright>=1.40",            # Browser automation for JS-heavy sites
"tenacity>=8.2",               # Exponential backoff for rate limits
"pdfplumber>=0.10",            # PDF table extraction (INDEPAZ, UNODC)
```

---

## 5. Phase 1: Geographic & Historical Foundation (SPEC-12)

**Goal**: Establish the municipal-level geographic foundation and download 20 years of historical election results.

### SPEC-12.1: DIVIPOLA Master Registry

**Objective**: Create a canonical registry of all 1,122 Colombian municipalities with their 5-digit DIVIPOLA codes.

**Data Sources**:
- Primary: `datos.gov.co` dataset `mv2e-prx5` (Divipole Elecciones Territoriales 2023 with geo)
- Fallback: GitHub Gist `b5848316671422b19e19bfca7f8aadcb`
- Validation: FOPEP table (PDF from `fopep.gov.co`)

**Implementation**:

```python
# src/co_president/ingestion/ingest_divipola.py

from tenacity import retry, stop_after_attempt, wait_exponential
from sodapy import Socrata


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_divipola_socrata() -> pd.DataFrame:
    """Fetch DIVIPOLA codes from datos.gov.co via Socrata API."""
    client = Socrata("www.datos.gov.co", None)  # No app token
    results = client.get("mv2e-prx5", limit=2000)
    df = pd.DataFrame.from_records(results)
    # Expected columns: codigo_municipio, nombre_municipio, departamento, the_geom
    return df


def fetch_divipola_github() -> pd.DataFrame:
    """Fallback: Parse GitHub Gist if Socrata fails."""
    url = "https://gist.github.com/b5848316671422b19e19bfca7f8aadcb"
    # Parse the gist content
    ...


def validate_divipola(df: pd.DataFrame) -> pd.DataFrame:
    """Validate: 1,122 municipalities, unique codes, no nulls."""
    assert len(df) >= 1122, f"Expected >=1122 municipalities, got {len(df)}"
    assert df["codigo_municipio"].nunique() == len(df), "Duplicate codes detected"
    assert df["codigo_municipio"].notna().all(), "Null codes detected"
    return df


def build_divipola_master() -> None:
    """Main pipeline: fetch -> validate -> save."""
    df = fetch_divipola_socrata()
    df = validate_divipola(df)
    df.to_csv("data/fundamentals/divipola_master.csv", index=False)
```

**Output Schema**:

```csv
codigo_municipio,nombre_municipio,departamento,latitud,longitud
05001,Medellin,Antioquia,6.2442,-75.5812
05002,Abriaqui,Antioquia,6.6167,-76.0167
...
```

**Acceptance Criteria**:
- Exactly 1,122 municipalities
- All codes are 5-digit integers (zero-padded as strings)
- No duplicate codes
- All department names match official RNEC list

**TDD Steps**:
1. **Red**: Write test that `fetch_divipola_socrata()` returns DataFrame with expected columns
2. **Red**: Write test that `validate_divipola()` raises `AssertionError` on invalid data
3. **Green**: Implement fetching with `sodapy` client + exponential backoff
4. **Green**: Implement GitHub Gist fallback with `requests` + BeautifulSoup
5. **Refactor**: Add logging, error handling, type annotations
6. **Type-check + Lint + Commit**: `pyright src/` must pass, `ruff check src/` must pass

---

### SPEC-12.2: Historical Election Results (2002-2022)

**Objective**: Download municipal-level results for all presidential elections from 2002 to 2022 and compute lagged features.

**Data Sources**:
- Primary: CEDAE database (`cedae.datasketch.co`)
- Secondary: `datos.gov.co` dataset `jhy3-m55z` (Presidenciales 2010-2022)
- Validation: RNEC historical results (`registraduria.gov.co`)

**Implementation**:

```python
# src/co_president/ingestion/ingest_historical.py

ELECTION_YEARS = [2002, 2006, 2010, 2014, 2018, 2022]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_cedae_results(year: int, round_num: int) -> pd.DataFrame:
    """Fetch CEDAE results for a specific year and round."""
    url = f"https://cedae.datasketch.co/api/results/{year}/{round_num}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.DataFrame.from_records(response.json())


def fetch_datos_gov_results() -> pd.DataFrame:
    """Fetch from datos.gov.co Socrata API."""
    client = Socrata("www.datos.gov.co", None)
    results = client.get("jhy3-m55z", limit=100000)
    return pd.DataFrame.from_records(results)


def compute_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical features per municipality.

    For each municipality:
    - Vote share for left candidate (Petro/Polo Democratico) in each year
    - Vote share for right candidate (Uribismo/Centro Democratico) in each year
    - Abstention rate
    - Delta from previous election
    """
    # Identify left/right candidates per election year
    candidate_ideology = {
        2002: {"left": "horacio_serpa", "right": "alvaro_uribe"},
        2006: {"left": "carlos_gaviria", "right": "alvaro_uribe"},
        2010: {"left": "gustavo_petro", "right": "juan_manuel_santos"},
        2014: {"left": "clara_lopez", "right": "oscar_ivan_zuluaga"},
        2018: {"left": "gustavo_petro", "right": "ivan_duque"},
        2022: {"left": "gustavo_petro", "right": "rodolfo_hernandez"},
    }
    ...


def build_historical_matrix() -> None:
    """Main pipeline: fetch all years -> compute features -> save."""
    all_results = []
    for year in ELECTION_YEARS:
        for round_num in [1, 2]:
            try:
                df = fetch_cedae_results(year, round_num)
                df["year"] = year
                df["round"] = round_num
                all_results.append(df)
            except Exception:
                logging.warning(f"Failed to fetch {year} round {round_num}, trying fallback")
                fallback = fetch_datos_gov_results()
                ...

    combined = pd.concat(all_results, ignore_index=True)
    features = compute_lagged_features(combined)
    features.to_csv("data/fundamentals/historical_results.csv", index=False)
```

**Output Schema**:

```csv
codigo_municipio,year,round,candidate,votes,vote_share,abstention_rate
05001,2022,1,gustavo_petro,245678,0.42,0.45
05001,2022,1,rodolfo_hernandez,156789,0.27,0.45
05001,2018,1,ivan_duque,198765,0.38,0.48
...
```

**Acceptance Criteria**:
- All 6 election years (2002, 2006, 2010, 2014, 2018, 2022) present
- Both rounds for each year (where applicable)
- All 1,122 municipalities represented in 2022 data
- Vote shares sum to ~100% (+-2% for rounding)
- Abstention rates calculated correctly

**TDD Steps**:
1. **Red**: Test `fetch_cedae_results()` returns DataFrame with expected columns
2. **Red**: Test `compute_lagged_features()` delta calculations with known inputs
3. **Green**: Implement fetching with `requests` + retry logic
4. **Green**: Implement delayed feature computation with `pandas.groupby`
5. **Refactor**: Add validation, error handling
6. **Type-check + Lint + Commit**

---

## 6. Phase 2: Structural Fundamentals (SPEC-13)

**Goal**: Download socioeconomic, demographic, and risk factors that predict voting behavior.

### SPEC-13.1: Socioeconomic & Demographic Data

**Objective**: Fetch DANE 2018 Census variables and Multidimensional Poverty Index (IPM).

**Data Sources**:
- DANE CNPV 2018: `dane.gov.co` (Censo Nacional de Poblacion y Vivienda)
- DANE IPM Municipal: `dane.gov.co` (Indice de Pobreza Multidimensional)
- DANE Population Projections 2018-2042: `dane.gov.co`
- DANE NBI: `dane.gov.co` (Necesidades Basicas Insatisfechas)

**Key Variables to Extract**:

| Variable | Description | Source | Predictive Value |
|----------|-------------|--------|------------------|
| `pct_afro_colombian` | Percentage Afro-Colombian population | CNPV 2018 | Strongly correlates with Petro support |
| `pct_indigenous` | Percentage indigenous population | CNPV 2018 | Correlates with left voting |
| `pct_rural_disperso` | Percentage rural dispersed population | CNPV 2018 | Correlates with clientelism |
| `years_schooling` | Average years of education | CNPV 2018 | Negative correlate with populism |
| `internet_access_rate` | Percentage with internet access | CNPV 2018 | Used to weight digital signals |
| `ipm_score` | Multidimensional Poverty Index (0-1) | DANE IPM | Strong predictor of Petro support |
| `nbi_rate` | Unsatisfied Basic Needs (%) | DANE NBI | Alternative poverty measure |
| `population_2022` | Projected 2022 population | DANE Projections | Turnout normalization denominator |

**Implementation**:

```python
# src/co_president/ingestion/ingest_socioeconomic.py
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_dane_csv(dane_url: str) -> pd.DataFrame | None:
    """Attempt to fetch a DANE CSV/excel file directly.

    DANE often provides direct download links to CSV or Excel files.
    """
    try:
        response = requests.get(dane_url, timeout=30)
        if response.status_code == 200 and "text/html" not in response.headers.get("Content-Type", ""):
            return _parse_data_file(response)
    except Exception as e:
        logging.warning(f"Direct fetch failed for {dane_url}: {e}")
    return None


def scrape_dane_portal_playwright() -> pd.DataFrame:
    """Fallback: Use Playwright to navigate DANE portal and download census data.

    DANE's website is JavaScript-heavy. Playwright handles dynamic page rendering,
    file download dialogs, and authentication redirects.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to CNPV 2018 microdata catalog
        page.goto("https://microdatos.dane.gov.co/index.php/catalog/643", wait_until="networkidle")

        # Locate and click the download button
        page.click("text=Descargar")

        # Wait for file download dialog
        with page.expect_download() as download_info:
            page.click("text=CSV")

        download = download_info.value
        download.save_as("data/raw/cnpv_2018.csv")
        return pd.read_csv("data/raw/cnpv_2018.csv", encoding="latin-1")


def fetch_ipm_municipal() -> pd.DataFrame:
    """Fetch Multidimensional Poverty Index at the municipal level."""
    # IPM censal URL
    url = "https://www.dane.gov.co/index.php/estadisticas-por-tema/pobreza-y-condiciones-de-vida/pobreza-y-desigualdad/medida-de-pobreza-multidimensional-de-fuente-censal"

    df = fetch_dane_csv(url)
    if df is not None:
        return df

    # Fallback: scrape with BeautifulSoup for Excel download links
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")
    excel_links = [a["href"] for a in soup.select("a[href$='.xlsx']")]
    if excel_links:
        ...
    ...


def calculate_features(census_df: pd.DataFrame, ipm_df: pd.DataFrame,
                       projections_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize census variables and merge with IPM and projections."""
    # 1. Calculate percentages from census counts
    census_df["pct_afro_colombian"] = (
        census_df["poblacion_afrocolombiana"] / census_df["poblacion_total"]
    )
    census_df["pct_rural_disperso"] = (
        census_df["poblacion_rural_dispersa"] / census_df["poblacion_total"]
    )

    # 2. Adjust 2018 census to 2022 electorate using projections
    census_df = census_df.merge(projections_df, on="codigo_municipio", how="left")
    census_df["poblacion_2022"] = census_df["proyeccion_2022"]

    # 3. Merge with IPM
    combined = census_df.merge(ipm_df, on="codigo_municipio", how="left")
    return combined


def build_socioeconomic_matrix() -> None:
    """Main pipeline: fetch census + IPM + projections -> save."""
    census = fetch_dane_csv(DANE_CENSUS_URL) or scrape_dane_portal_playwright()
    ipm = fetch_ipm_municipal()
    projections = fetch_dane_csv(DANE_PROJECTIONS_URL)

    features = calculate_features(census, ipm, projections)
    features.to_csv("data/fundamentals/socioeconomic.csv", index=False)
```

**Output Schema**:

```csv
codigo_municipio,pct_afro_colombian,pct_indigenous,pct_rural_disperso,years_schooling,internet_access_rate,ipm_score,population_2022
05001,0.08,0.001,0.05,10.2,0.78,0.15,2569007
05002,0.45,0.02,0.85,6.5,0.25,0.68,3456
...
```

**Acceptance Criteria**:
- All 1,122 municipalities present
- IPM scores range from 0 to 1
- Internet access rates range from 0 to 1
- Population projections are positive integers
- No nulls in critical columns (IPM, internet, population)

**TDD Steps**:
1. **Red**: Test `fetch_dane_csv()` returns DataFrame or None
2. **Red**: Test `scrape_dane_portal_playwright()` produces valid census DataFrame
3. **Red**: Test `calculate_features()` correctly normalizes percentages
4. **Green**: Implement direct CSV fetch + Playwright fallback
5. **Green**: Implement IPM fetch with BeautifulSoup link scraping
6. **Refactor**: Add structured logging, download caching
7. **Type-check + Lint + Commit**

---

### SPEC-13.2: Electoral Risk & Conflict Data

**Objective**: Fetch risk maps, armed group presence, PDET municipalities, and coca cultivation data.

**Data Sources**:

| Source | URL | Format | Key Data |
|--------|-----|--------|----------|
| MOE Risk Maps | `moe.org.co/datos-electorales/mapas-de-riesgo-electoral/` | CSV/HTML | Risk level per municipality |
| INDEPAZ | `indepaz.org.co` | PDF | Armed group presence in 216+ municipalities |
| PDET | `centralpdet.renovacionterritorio.gov.co` | CSV/HTML | 170 prioritized municipalities |
| UNODC Coca | `unodc.org/colombia` | PDF | Hectares of coca cultivation per municipality |

**Implementation**:

```python
# src/co_president/ingestion/ingest_risk.py


def fetch_moe_risk_maps() -> pd.DataFrame:
    """Fetch MOE electoral risk classification."""
    url = "https://moe.org.co/datos-electorales/mapas-de-riesgo-electoral/"

    # Parse HTML for download links
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find CSV/Excel download links
    download_links = []
    for link in soup.select("a[href$='.csv'], a[href$='.xlsx']"):
        download_links.append(link["href"])

    if download_links:
        df = pd.read_csv(download_links[0])
        return df

    # Fallback: hardcoded risk classifications from published reports
    risk_municipios = {
        "extreme": 49,   # Municipios with extreme risk
        "high": 65,      # Municipios with high risk
        "medium": 17,    # Municipios with medium risk
    }
    ...


def parse_indepaz_pdf(pdf_path: str) -> pd.DataFrame:
    """Parse INDEPAZ PDF to extract armed group presence by municipality."""
    import pdfplumber

    records = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if len(row) >= 2:
                        municipio = row[0]
                        group = row[1]
                        records.append({"municipio": municipio, "armed_group": group})

    df = pd.DataFrame(records)
    df["armed_group_presence"] = 1
    return df.groupby("municipio").agg({"armed_group_presence": "max"}).reset_index()


def fetch_pdet_list() -> pd.DataFrame:
    """Fetch official PDET municipality list (170 prioritized municipalities)."""
    url = "https://centralpdet.renovacionterritorio.gov.co/conoce-los-pdet/"
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")

    municipalities = []
    for item in soup.select(".pdet-municipio"):
        name = item.get_text(strip=True)
        municipalities.append({"nombre_municipio": name, "is_pdet": 1})

    df = pd.DataFrame(municipalities)
    assert len(df) == 170, f"Expected 170 PDET municipalities, got {len(df)}"
    return df


def calculate_risk_features(moe: pd.DataFrame, indepaz: pd.DataFrame,
                            pdet: pd.DataFrame, coca: pd.DataFrame) -> pd.DataFrame:
    """Combine all risk indicators into a single feature set."""
    combined = moe.merge(indepaz, on="codigo_municipio", how="left")
    combined = combined.merge(pdet, on="codigo_municipio", how="left")
    combined = combined.merge(coca, on="codigo_municipio", how="left")

    # Fill NaN for municipalities not in PDET/INDEPAZ lists
    combined["is_pdet"] = combined["is_pdet"].fillna(0).astype(int)
    combined["armed_group_presence"] = combined["armed_group_presence"].fillna(0).astype(int)
    combined["coca_hectares"] = combined["coca_hectares"].fillna(0)

    combined["high_risk_flag"] = combined["risk_level"].isin(["extreme", "high"]).astype(int)
    return combined


def build_risk_matrix() -> None:
    """Main pipeline: fetch all risk sources -> combine -> save."""
    moe = fetch_moe_risk_maps()
    indepaz = parse_indepaz_pdf("data/raw/indepaz_2022.pdf")
    pdet = fetch_pdet_list()
    coca = fetch_unodc_coca()

    features = calculate_risk_features(moe, indepaz, pdet, coca)
    features.to_csv("data/fundamentals/risk_factors.csv", index=False)
```

**Output Schema**:

```csv
codigo_municipio,risk_level,high_risk_flag,armed_group_presence,is_pdet,coca_hectares
05001,low,0,0,0,0
91001,extreme,1,1,1,1250
...
```

**Acceptance Criteria**:
- All 1,122 municipalities present
- Risk levels are one of: extreme, high, medium, low
- Binary flags are 0 or 1
- PDET list contains exactly 170 municipalities
- Coca hectares are non-negative

**TDD Steps**:
1. **Red**: Test `fetch_moe_risk_maps()` returns DataFrame with risk levels
2. **Red**: Test `parse_indepaz_pdf()` returns DataFrame with municipal-level group presence
3. **Red**: Test `fetch_pdet_list()` returns exactly 170 municipalities
4. **Green**: Implement HTML parsing with BeautifulSoup
5. **Green**: Implement PDF extraction with pdfplumber
6. **Green**: Implement feature calculation with proper NaN handling
7. **Refactor**: Add Playwright fallback for JS-heavy pages
8. **Type-check + Lint + Commit**

---

## 7. Phase 3: Feature Matrix Integration (SPEC-14)

**Goal**: Join all fundamental components into a single, validated feature matrix ready for modeling.

### SPEC-14.1: Municipal Feature Matrix Builder

**Implementation**:

```python
# src/co_president/ingestion/build_feature_matrix.py

COMPONENT_FILES = {
    "divipola": "data/fundamentals/divipola_master.csv",
    "historical": "data/fundamentals/historical_results.csv",
    "socioeconomic": "data/fundamentals/socioeconomic.csv",
    "risk": "data/fundamentals/risk_factors.csv",
}


def load_all_components() -> dict[str, pd.DataFrame]:
    """Load all fundamental components from disk."""
    components = {}
    for name, path in COMPONENT_FILES.items():
        components[name] = pd.read_csv(path)
        logging.info(f"Loaded {name}: {len(components[name])} rows, "
                      f"{len(components[name].columns)} columns")
    return components


def validate_component_health(components: dict[str, pd.DataFrame]) -> list[str]:
    """Check each component for basic data quality.

    Returns a list of warnings (empty list = healthy).
    """
    warnings = []
    expected_municipalities = 1122

    for name, df in components.items():
        if "codigo_municipio" not in df.columns:
            warnings.append(f"{name}: missing codigo_municipio column")
            continue

        # Check for null codes
        null_count = df["codigo_municipio"].isna().sum()
        if null_count > 0:
            warnings.append(f"{name}: {null_count} null codigo_municipio values")

        # Check for duplicate codes
        dup_count = df["codigo_municipio"].duplicated().sum()
        if dup_count > 0:
            warnings.append(f"{name}: {dup_count} duplicate codigo_municipio values")

    return warnings


def pivot_historical_wide(historical: pd.DataFrame) -> pd.DataFrame:
    """Pivot historical results to wide format (one row per municipality).

    Before: 1 row per (municipio, year, round, candidate)
    After:  1 row per municipio with columns like vote_share_2022_r1_petro
    """
    historical_pivot = historical.pivot_table(
        index="codigo_municipio",
        columns=["year", "round", "candidate"],
        values="vote_share",
    )
    # Flatten MultiIndex columns
    historical_pivot.columns = [
        f"vote_share_{year}_r{round}_{cand}"
        for year, round_, cand in historical_pivot.columns
    ]
    return historical_pivot.reset_index()


def build_feature_matrix() -> pd.DataFrame:
    """Build the final municipal feature matrix."""
    components = load_all_components()
    warnings = validate_component_health(components)

    if warnings:
        logging.warning(f"Component health warnings: {warnings}")
        # In production, decide whether to proceed or abort

    divipola = components["divipola"]
    historical = components["historical"]
    socioeconomic = components["socioeconomic"]
    risk = components["risk"]

    # Pivot historical to wide format
    historical_wide = pivot_historical_wide(historical)

    # Sequential joins (all LEFT joins anchored on DIVIPOLA)
    matrix = divipola.merge(historical_wide, on="codigo_municipio", how="left")
    matrix = matrix.merge(socioeconomic, on="codigo_municipio", how="left")
    matrix = matrix.merge(risk, on="codigo_municipio", how="left")

    # Final validation
    assert len(matrix) == 1122, f"Matrix has {len(matrix)} rows, expected 1122"
    assert matrix["codigo_municipio"].is_unique, "Matrix has duplicate codes"

    return matrix


def generate_data_dictionary(matrix: pd.DataFrame, output_path: str) -> None:
    """Generate a human-readable data dictionary from the matrix columns."""
    with open(output_path, "w") as f:
        f.write("# Municipal Feature Matrix - Data Dictionary\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Rows: {len(matrix)}, Columns: {len(matrix.columns)}\n\n")

        for col in matrix.columns:
            dtype = matrix[col].dtype
            nulls = matrix[col].isna().sum()
            sample = matrix[col].dropna().iloc[0] if nulls < len(matrix) else "ALL NULL"
            f.write(f"## {col}\n")
            f.write(f"- **Type**: {dtype}\n")
            f.write(f"- **Nulls**: {nulls}/{len(matrix)}\n")
            f.write(f"- **Sample**: {sample}\n\n")


def save_feature_matrix(matrix: pd.DataFrame) -> None:
    """Save the final feature matrix in multiple formats."""
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # CSV for human inspection
    matrix.to_csv("data/processed/municipal_feature_matrix.csv", index=False)

    # Parquet for efficient loading
    matrix.to_parquet("data/processed/municipal_feature_matrix.parquet", index=False)

    # Data dictionary
    generate_data_dictionary(matrix, "data/processed/feature_dictionary.md")

    # Summary statistics
    summary = {
        "municipalities": len(matrix),
        "columns": len(matrix.columns),
        "null_rate": matrix.isna().sum().sum() / (len(matrix) * len(matrix.columns)),
        "memory_mb": matrix.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    with open("data/processed/build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Feature matrix saved: {summary}")
```

**Acceptance Criteria**:
- Exactly 1,122 rows (one per municipality)
- All columns from all components present
- No nulls in critical columns (DIVIPOLA code, 2022 vote shares)
- Data dictionary generated
- Matrix saved in both CSV and Parquet formats
- Build summary JSON with statistics

**TDD Steps**:
1. **Red**: Test `validate_component_health()` returns warnings for known-bad data
2. **Red**: Test `pivot_historical_wide()` produces correct wide-format columns
3. **Red**: Test `generate_data_dictionary()` writes valid markdown
4. **Green**: Implement all functions
5. **Green**: Implement end-to-end `build_feature_matrix()` pipeline
6. **Refactor**: Add progress bars, caching, parallel loading
7. **Type-check + Lint + Commit**

---

## 8. Phase 4: Digital Signals Integration (Future)

**Goal**: Integrate real-time digital signals as model updaters.

### Google Trends Pipeline

```python
# src/co_president/ingestion/ingest_trends.py

from pytrends.request import TrendReq


def fetch_google_trends(
    keywords: list[str],
    timeframe: str = "2022-01-01 2022-06-19",
    geo: str = "CO",
) -> pd.DataFrame:
    """Fetch Google Trends data for candidate names.

    Uses the pytrends unofficial API (no API key required).

    Args:
        keywords: Candidate names to search (e.g., ["Gustavo Petro", "Rodolfo Hernandez"])
        timeframe: Date range in format "YYYY-MM-DD YYYY-MM-DD"
        geo: Geographic region ("CO" for Colombia)

    Returns:
        DataFrame with weekly search interest scores (0-100)
    """
    pytrends = TrendReq(hl="es-CO", tz=300)
    pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo)
    return pytrends.interest_over_time()


def weight_trends_by_internet(
    trends: pd.DataFrame,
    internet_penetration: pd.Series,
) -> pd.DataFrame:
    """Weight trend signals by municipal internet penetration.

    In municipalities with high internet access, digital signals receive
    higher weight. In disconnected regions, historical vote patterns dominate.

    Formula:
        digital_signal_weight[m] = base_weight * internet_penetration[m]
        historical_weight[m] = 1 - digital_signal_weight[m]
    """
    trends_weighted = trends.copy()
    for col in trends.columns:
        if col != "isPartial":
            trends_weighted[col] = trends[col] * internet_penetration
    return trends_weighted
```

### Social Media Sentiment (Twitter Emotion Corpus)

The arXiv paper `2407.07258` provides an emotion-labeled Twitter corpus (`Identification of emotions on Twitter during the 2022 electoral process in Colombia`).

```python
# Future: src/co_president/ingestion/ingest_twitter.py

def fetch_emotion_corpus() -> pd.DataFrame:
    """Fetch emotion-labeled tweet corpus from arXiv supplementary data."""
    url = "https://arxiv.org/src/2407.07258"
    # Download and parse emotion classifications
    ...

def compute_weekly_sentiment(tweets: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly sentiment scores per candidate.

    Uses BERT-identified emotion probabilities to create a
    candidate-level sentiment time series: joy, fear, anger, disgust.
    """
    ...
```

### Key Implementation Notes

- Colombian law prohibits publishing polls in the final week of the campaign, making Google Trends the primary legal real-time signal during that window
- Weight departmental Trends scores by `internet_penetration_rate` from CNPV 2018
- YouTube search category is accessible via the same `pytrends` interface
- The `pytrends` library requires no authentication but respects rate limits

---

## 9. Phase 5: Hierarchical Model Evolution (Future)

**Goal**: Evolve the PyMC model from national temporal to municipal hierarchical.

### Planned Model Architecture

```python
# src/co_president/model_hierarchical.py

import pymc as pm
import numpy as np


def build_hierarchical_model(
    feature_matrix: pd.DataFrame,
    polls: pd.DataFrame,
) -> pm.Model:
    """Build a municipal-level hierarchical Bayesian model.

    Structure:
        1. Municipal priors from fundamentals (historical + socioeconomic + risk)
        2. National polling updates (from existing MVP model)
        3. Digital signal observation model (Google Trends, Twitter)
        4. Posterior predictive at municipal, departmental, and national levels
    """
    M = len(feature_matrix)  # Number of municipalities (1,122)

    with pm.Model() as model:
        # --- Hyperpriors (shared across municipalities) ---
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta_historical = pm.Normal("beta_historical", mu=0, sigma=1)
        beta_poverty = pm.Normal("beta_poverty", mu=0, sigma=1)
        beta_ethnicity = pm.Normal("beta_ethnicity", mu=0, sigma=1)
        beta_risk = pm.Normal("beta_risk", mu=0, sigma=1)

        # --- Municipal-level priors (non-centered) ---
        mu_m_raw = pm.Normal("mu_m_raw", mu=0, sigma=1, shape=M)
        sigma_m = pm.HalfNormal("sigma_m", sigma=1)

        # Municipal prior (from fundamentals)
        baseline = (
            alpha
            + beta_historical * feature_matrix["vote_share_2018_petro"]
            + beta_poverty * feature_matrix["ipm_score"]
            + beta_ethnicity * feature_matrix["pct_afro_colombian"]
            + beta_risk * feature_matrix["high_risk_flag"]
        )
        mu_m = pm.Deterministic("mu_m", baseline + sigma_m * mu_m_raw)

        # --- Temporal evolution (reverse-time random walk) ---
        # eta_rw ~ HalfNormal(0.5)
        # theta[t] ~ Normal(theta[t+1], eta_rw)  (walking backward)

        # --- Observation model (polls) ---
        # DirichletMultinomial with house effects

        # --- Observation model (digital signals) ---
        # Normal likelihood for Google Trends z-scores

        # --- Posterior predictive ---
        # Municipal vote shares rolled up to national totals

    return model
```

### Running Comparison: Bayesian vs. ML Benchmark

```python
# Future: src/co_president/benchmarks/ml_comparison.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

def benchmark_ml_models(feature_matrix: pd.DataFrame, target: str) -> dict:
    """Replicate the USANTOMAS benchmark on our municipal feature matrix.

    Models tested in the original study:
    1. Random Forest
    2. Gradient Boosting
    3. SVM
    4. KNN
    5. Feedforward Neural Network (CLR-transformed)

    Returns:
        dict: {model_name: cross_val_score}
    """
    X = feature_matrix.drop(columns=[target, "codigo_municipio"])
    y = feature_matrix[target]

    # Centered Log-Ratio (CLR) transformation for compositional data
    from sklearn.preprocessing import FunctionTransformer
    from scipy.stats import gmean

    def clr_transform(X):
        return np.log(X / gmean(X, axis=1, keepdims=True))

    models = {
        "random_forest": RandomForestRegressor(n_estimators=100),
        "gradient_boosting": GradientBoostingRegressor(),
        "neural_network": MLPRegressor(hidden_layer_sizes=(64, 32)),
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        results[name] = {"mean_r2": scores.mean(), "std_r2": scores.std()}

    return results
```

---

## 10. Data Sources Inventory

### Primary Data Sources (Phase 1-3)

| Source | URL | Format | Access Method | Priority |
|--------|-----|--------|---------------|----------|
| DIVIPOLA Codes | `datos.gov.co` (mv2e-prx5) | JSON/CSV | `sodapy` API | **Critical** |
| CEDAE Historical | `cedae.datasketch.co` | CSV | `requests` + retry | **Critical** |
| DANE Census 2018 | `microdatos.dane.gov.co` (catalog/643) | CSV | Playwright | **Critical** |
| DANE IPM | `dane.gov.co` | Excel/CSV | Playwright + BS4 | **Critical** |
| DANE Projections | `dane.gov.co` | Excel/CSV | Playwright | **Critical** |
| MOE Risk Maps | `moe.org.co` | CSV | Playwright + BS4 | **High** |
| INDEPAZ Groups | `indepaz.org.co` | PDF | `pdfplumber` | **High** |
| PDET List | `centralpdet.renovacionterritorio.gov.co` | HTML/CSV | BS4 + Playwright | **High** |
| UNODC Coca | `unodc.org/colombia` | PDF | `pdfplumber` | **Medium** |
| DANE NBI | `dane.gov.co` | Excel/CSV | Playwright | **Medium** |

### Secondary Data Sources (Phase 4-5)

| Source | URL | Format | Access Method | Priority |
|--------|-----|--------|---------------|----------|
| Google Trends | `trends.google.com` | JSON | `pytrends` | **High** |
| Twitter Emotions | `arxiv.org/abs/2407.07258` | CSV | `requests` | **Medium** |
| YouTube Trends | `trends.google.com` | JSON | `pytrends` (youtube cat) | **Low** |
| Campaign Finance | `cnecuentasclaras.gov.co` | CSV/HTML | Playwright | **Low** |
| GEIH Labor | `dane.gov.co` | CSV/Excel | Playwright | **Low** |
| LAPOP Survey | `obsdemocracia.org` | CSV | `requests` | **Low** |
| DANE ECP | `microdatos.dane.gov.co` (catalog/730) | CSV | Playwright | **Low** |

### Fallback Strategy

When a primary source is unavailable:

1. **Try alternative URL** from `reference/colombia_predictor_urls.txt` (189 URLs available)
2. **Use Playwright** to navigate the site and interact with dynamic forms/buttons
3. **Use BeautifulSoup** to parse static HTML for download links
4. **Log warning** with the failing URL and continue the pipeline gracefully
5. **Request manual download** only as last resort

---

## 11. Implementation Strategy

### Development Workflow

1. **One SPEC at a time**: Complete SPEC-12 before starting SPEC-13
2. **TDD discipline**: RED -> GREEN -> REFACTOR -> TYPE-CHECK -> LINT -> COMMIT
3. **Incremental validation**: After each SPEC, validate outputs before proceeding
4. **Data file tracking**: Use `git-lfs` or `.gitignore` for large generated files

### Execution Order

| Phase | Week | SPEC | Deliverable |
|-------|------|------|-------------|
| 1 | 1 | SPEC-12.1 | `divipola_master.csv` (1,122 municipalities) |
| 1 | 2 | SPEC-12.2 | `historical_results.csv` (2002-2022, all rounds) |
| 2 | 3 | SPEC-13.1 | `socioeconomic.csv` (census + IPM + projections) |
| 2 | 4 | SPEC-13.2 | `risk_factors.csv` (MOE + INDEPAZ + PDET + coca) |
| 3 | 5 | SPEC-14 | `municipal_feature_matrix.csv` (joined + validated) |
| 3 | 6 | Integration | End-to-end validation, data dictionary, summary stats |

### Quality Gates (before every commit)

```bash
make sync && make install     # Install dependencies
pyright src/                   # Zero type errors
ruff check src/ tests/         # Zero lint errors
pytest tests/test_ingestion.py -v   # All ingestion tests pass
```

---

## 12. Testing & Validation Framework

### Unit Test Structure

```
tests/
└── test_ingestion.py
    ├── test_divipola
    │   ├── test_fetch_divipola_socrata()      # Returns DataFrame with expected columns
    │   ├── test_fetch_divipola_github()        # Fallback works
    │   └── test_validate_divipola()            # Raises on bad data
    ├── test_historical
    │   ├── test_fetch_cedae_results()          # Returns correct year's data
    │   ├── test_compute_lagged_features()      # Delta calculations correct
    │   └── test_pivot_historical_wide()        # Wide format has correct columns
    ├── test_socioeconomic
    │   ├── test_fetch_dane_csv()               # Returns DataFrame or None
    │   ├── test_scrape_playwright()            # Playwright fallback works
    │   └── test_calculate_features()           # Percentages sum correctly
    ├── test_risk
    │   ├── test_fetch_moe_risk_maps()          # Risk levels are valid
    │   ├── test_parse_indepaz_pdf()            # Municipalities extracted
    │   └── test_fetch_pdet_list()              # Exactly 170 municipalities
    └── test_feature_matrix
        ├── test_load_all_components()          # All files load
        ├── test_validate_component_health()    # Warnings on bad data
        └── test_build_feature_matrix()         # Final matrix has 1122 rows
```

### Integration Test

```python
def test_full_ingestion_pipeline():
    """End-to-end test of the entire ingestion pipeline."""
    # 1. Fetch all components
    build_divipola_master()
    build_historical_matrix()
    build_socioeconomic_matrix()
    build_risk_matrix()

    # 2. Build feature matrix
    matrix = build_feature_matrix()

    # 3. Validate
    assert len(matrix) == 1122
    assert matrix["codigo_municipio"].is_unique
    assert matrix["vote_share_2022_r1_gustavo_petro"].notna().all()
    assert matrix["ipm_score"].min() >= 0
    assert matrix["ipm_score"].max() <= 1
```

### Data Quality Checks

```python
def run_data_quality_audit(matrix: pd.DataFrame) -> dict:
    """Comprehensive data quality audit.

    Returns a dictionary with audit results.
    """
    audit = {
        "total_municipalities": len(matrix),
        "missing_depts": [],
        "critical_nulls": [],
        "out_of_range": [],
        "duplicates": matrix["codigo_municipio"].duplicated().sum(),
    }

    # Check for nulls in critical columns
    critical_cols = [
        "codigo_municipio", "vote_share_2022_r1_gustavo_petro",
        "ipm_score", "internet_access_rate",
    ]
    for col in critical_cols:
        null_count = matrix[col].isna().sum()
        if null_count > 0:
            audit["critical_nulls"].append({"column": col, "null_count": null_count})

    # Check value ranges
    if matrix["ipm_score"].max() > 1.0:
        audit["out_of_range"].append("IPM scores exceed 1.0")

    return audit
```

---

## 13. Integration with Existing MVP

### Backward Compatibility

The enhancement **will not break** the existing MVP:

- All existing SPEC-01 to SPEC-11 modules remain **unchanged**
- New ingestion modules are **additive**
- Existing tests continue to **pass**
- MVP model can still run with polling-only data

### Shared Infrastructure

| Component | Existing | Enhanced |
|-----------|----------|----------|
| `config.py` | Candidate maps, hyperparameters | Add feature column names, DIVIPOLA constants |
| `paths.py` | `data/2022-polls/`, `data/2022-presidential-results/` | Add `data/fundamentals/`, `data/processed/` |
| `__main__.py` | `run`, `aggregate`, `validate` commands | Add `ingest` command with subcommands |
| `make check` | `fmt -> lint -> typecheck -> test` | Add ingestion tests |

### CLI Extension (Future)

```python
# In __main__.py

@click.group()
def cli():
    pass

@cli.group()
def ingest():
    """Ingest external data sources."""
    pass

@ingest.command()
def divipola():
    """Build DIVIPOLA master registry."""
    from co_president.ingestion.ingest_divipola import build_divipola_master
    build_divipola_master()
    click.echo("DIVIPOLA master saved.")

@ingest.command()
def historical():
    """Fetch 2002-2022 historical election results."""
    ...

@ingest.command()
def socioeconomic():
    """Fetch DANE census + IPM data."""
    ...

@ingest.command()
def risk():
    """Fetch MOE risk + INDEPAZ + PDET data."""
    ...

@ingest.command()
def all():
    """Run all ingestion pipelines."""
    ...
```

### Gradual Migration Path

```
Phase 1-3: Build fundamentals           [No model changes]
Phase 4:   Add digital signals          [Optional features]
Phase 5:   Build hierarchical model     [Alongside existing model]
Validation: Compare hierarchical vs MVP [Choose or ensemble]
```

---

## 14. Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Socrata API rate limiting | **High** | Medium | `tenacity` exponential backoff, `stop_after_attempt(3)` |
| Website structure changes | **Medium** | High | Playwright fallback, graceful failure, manual download option |
| PDF format inconsistencies | **Medium** | Medium | Multi-strategy extraction (pdfplumber -> camelot -> OCR fallback) |
| Encoding issues (Latin-1 vs UTF-8) | **High** | Low | Try multiple encodings, detect BOM, log warnings |
| Large file downloads (>100MB) | **Low** | Medium | Streaming downloads, chunked processing |
| JavaScript-heavy pages | **Medium** | Medium | Playwright with `wait_until="networkidle"` |

### Data Quality Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Missing municipalities in component | **Low** | **High** | `validate_joins()` checks all 1,122 are preserved |
| Inconsistent DIVIPOLA codes across years | **Medium** | **High** | Canonical registry, cross-validation with FOPEP table |
| Outdated census data (2018 vs 2022) | **Medium** | Medium | Population projections normalize to 2022 electorate |
| Duplicate entries from multiple sources | **Medium** | Medium | Deduplication logic in `build_feature_matrix()` |
| Null values in critical features | **Low** | Medium | `validate_component_health()` flags before building matrix |

### Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scope creep (trying to build all 5 phases at once) | **Medium** | **High** | Strictly follow SPEC-by-SPEC approach |
| API deprecation (DANE changes their portal) | **Low** | **High** | Playwright fallback + manual download path |
| Performance issues with 1,122-row loops | **Low** | Medium | Vectorized pandas operations, avoid Python loops |
| Model convergence with many municipalities | **Low** | **High** | Non-centered parameterization, good initial values |

---

## 15. Dependency Graph

```
  SPEC-01 (Scaffolding)
    |
    SPEC-02 (Config & Constants)
    |
    SPEC-03 (Results Consolidation) + SPEC-04 (Poll Loading)
    |         |
    |         +-- SPEC-05 (Baseline Aggregation)
    |         |
    |         +-- SPEC-06 (Bayesian 1st Round) + SPEC-07 (Runoff) + SPEC-08 (Matrix)
    |                  |
    |                  +-- SPEC-09 (Validation) + SPEC-10 (CLI)
    |                            |
    |                            +-- SPEC-11 (Data Quality Diagnostics)
    |
    +============================================================
    |  ENHANCEMENT PIPELINE (this document)
    |
    +-- SPEC-12.1 (DIVIPOLA Registry)                     [No dependencies on other specs]
    |
    +-- SPEC-12.2 (Historical Results)                    [Depends on SPEC-12.1 for DIVIPOLA codes]
    |
    +-- SPEC-13.1 (Socioeconomic) + SPEC-13.2 (Risk)     [Depends on SPEC-12.1]
    |         |
    +-- SPEC-14 (Feature Matrix)                          [Depends on all SPEC-12 + SPEC-13 outputs]
    |
    +-- Phase 4 (Digital Signals)                         [Optional, depends on SPEC-14]
    |
    +-- Phase 5 (Hierarchical Model)                      [Depends on SPEC-14 + Phase 4]
```

---

## Appendix A: Key Reference URLs from Research

### Electoral Data
| Source | URL |
|--------|-----|
| CEDAE Explore | `cedae.datasketch.co/datos-democracia/resultados-electorales/explora-los-datos/` |
| CEDAE Download | `cedae.datasketch.co/datos-democracia/resultados-electorales/descarga-los-datos/` |
| Datos Abiertos Dashboard | `datos.gov.co` (dataset `jhy3-m55z`) |
| MOE Presidential Book | `moe.org.co/wp-content/uploads/2022/11/2022.11.09-LIBRO-RESULTADOS-ELECTORALES-PRESIDENCIALES-2022.pdf` |
| MOE Digital Edition | `moe.org.co/wp-content/uploads/2022/11/DIGITAL-Resultados-Presidenciales-2022.pdf` |

### Socioeconomic & Demographic
| Source | URL |
|--------|-----|
| DANE CNPV 2018 | `dane.gov.co` (Censo Nacional de Poblacion y Vivienda 2018) |
| DANE IPM Municipal | `dane.gov.co` (Medida de Pobreza Multidimensional de Fuente Censal) |
| DANE Population Projections | `dane.gov.co` (Proyecciones de Poblacion, 2018-2042) |
| DANE NBI | `dane.gov.co` (Necesidades Basicas Insatisfechas) |
| MPPN IPM Colombia | `mppn.org/es/ipm-municipal-colombia/` |
| DNP Social Indicators | `colaboracion.dnp.gov.co/CDT/Desarrollo%20Social/boletin37.pdf` |

### Conflict & Risk
| Source | URL |
|--------|-----|
| MOE Risk Maps | `moe.org.co/datos-electorales/mapas-de-riesgo-electoral/` |
| INDEPAZ Armed Groups | `indepaz.org.co/wp-content/uploads/2022/11/RESUMEN_GRUPOS_2022.pdf` |
| UNODC Colombia | `unodc.org/colombia/en/index.html` |
| PDET Municipalities | `centralpdet.renovacionterritorio.gov.co/conoce-los-pdet/` |
| PMC Turnout Study | `pmc.ncbi.nlm.nih.gov/articles/PMC11929383/` |

### Digital & Social Media
| Source | URL |
|--------|-----|
| Google Trends | `trends.google.com/trending?hl=es-419` |
| Twitter Emotion Corpus | `arxiv.org/abs/2407.07258` |
| Hernandez Sentiment | `revistas.usb.edu.co/index.php/GuillermoOckham/article/view/7462` |
| Kaggle Twitter 2018 | `kaggle.com/datasets/saurabhshahane/colombian-election-campaign-on-twitter-2018` |
| SciELO Google Trends | `scielo.org.co/scielo.php?script=sci_arttext&pid=S0012-73532023000100064` |

### Geographic Reference
| Source | URL |
|--------|-----|
| DIVIPOLA FOPEP | `fopep.gov.co/sheempoo/2019/02/Tabla-Codigos-Dane.pdf` |
| DIVIPOLA GitHub Gist | `gist.github.com/b5848316671422b19e19bfca7f8aadcb` |
| Divipole Territoriales | `datos.gov.co` (dataset `mv2e-prx5`) |

### Academic Benchmarks
| Source | URL |
|--------|-----|
| 5 ML Models (USANTOMAS) | `revistas.usantotomas.edu.co/index.php/estadistica/article/view/11212` |
| 5 ML Models (Dialnet) | `dialnet.unirioja.es/servlet/articulo?codigo=10404048` |
| PPEG Codebook | `ppeg.wzb.eu/www/codebook_ppeg_comb_2025v1.pdf` |
| Freedom in the World | `kaggle.com/datasets/justin2028/freedom-in-the-world-2013-2022` |

---

*This roadmap is synthesized from the MVP_SPECS_GUIDE.md, Colombia_Presidential_Predictor_Research_Reference.md, and colombia_predictor_urls.txt. All data sources are documented in the research reference with complete URLs, formats, and access methods.*
