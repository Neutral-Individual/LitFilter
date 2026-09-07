"""LitFilter — literature review paper screening for researchers."""

from __future__ import annotations

import io
import json
import re
import sys
import time
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from thefuzz import fuzz

REQUIRED_FIELDS = [
    "title",
    "abstract",
    "year",
    "keywords",
    "journal",
    "item_type",
]
OPTIONAL_FIELDS = ["doi"]

FIELD_LABELS = {
    "title": "Title",
    "abstract": "Abstract",
    "year": "Year",
    "keywords": "Keywords",
    "journal": "Journal/Conference Name",
    "item_type": "Item Type",
    "doi": "DOI",
}

COLUMN_ALIASES = {
    "title": [
        "title",
        "article title",
        "document title",
        "paper title",
        "ti",
        "titles",
        "article_title",
        "document_title",
    ],
    "abstract": ["abstract", "ab", "summary", "abstracts"],
    "year": [
        "year",
        "publication year",
        "cover date",
        "py",
        "pub year",
        "date",
        "publication date",
        "year of publication",
        "cover_date",
    ],
    "keywords": [
        "keywords",
        "author keywords",
        "index keywords",
        "de",
        "author keyword",
        "controlled terms",
        "author_keywords",
        "index_keywords",
        "key words",
    ],
    "journal": [
        "source title",
        "journal",
        "publication title",
        "conference name",
        "source",
        "journal/conference name",
        "journal name",
        "conference",
        "publication name",
        "source_title",
        "publication_title",
        "journal title",
        "conference title",
    ],
    "item_type": [
        "document type",
        "item type",
        "type",
        "publication type",
        "article type",
        "document_type",
        "item_type",
    ],
    "doi": [
        "doi",
        "digital object identifier",
        "doi link",
        "dois",
        "digital_object_identifier",
    ],
}

DOMAIN_TERMS = {
    "Robotics": [
        "robot",
        "robotic",
        "robotics",
        "uav",
        "drone",
        "humanoid",
        "manipulator arm",
    ],
    "Medical": [
        "medical",
        "clinical",
        "patient",
        "hospital",
        "surgery",
        "surgical",
        "diagnosis",
        "pathology",
        "radiology",
    ],
    "Healthcare": [
        "healthcare",
        "health care",
        "nursing",
        "ehr",
        "electronic health record",
        "telemedicine",
        "public health",
    ],
    "Nature/Biology": [
        "biology",
        "biological",
        "ecology",
        "ecological",
        "species",
        "genome",
        "genomic",
        "protein",
        "organism",
        "botany",
        "zoology",
    ],
    "Education": [
        "education",
        "classroom",
        "pedagogy",
        "pedagogical",
        "student learning",
        "e-learning",
        "curriculum",
        "teacher training",
    ],
    "Manufacturing": [
        "manufacturing",
        "factory",
        "production line",
        "shop floor",
        "industry 4.0",
        "cnc",
        "additive manufacturing",
    ],
    "Automotive": [
        "automotive",
        "vehicle",
        "autonomous driving",
        "self-driving",
        "powertrain",
        "electric vehicle",
        "adas",
    ],
    "Finance": [
        "finance",
        "financial",
        "banking",
        "stock market",
        "fintech",
        "credit risk",
        "portfolio",
        "cryptocurrency trading",
    ],
    "Supply Chain": [
        "supply chain",
        "logistics",
        "inventory management",
        "warehouse",
        "procurement",
        "demand forecasting",
    ],
    "Computer Networks": [
        "computer network",
        "routing protocol",
        "tcp/ip",
        "wireless sensor",
        "sdn",
        "network topology",
        "packet routing",
        "5g network",
    ],
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "for",
    "in",
    "on",
    "to",
    "with",
    "using",
    "via",
    "from",
    "by",
    "into",
    "over",
    "under",
    "a",
    "vs",
    "based",
    "towards",
    "toward",
    "study",
    "studies",
    "approach",
    "approaches",
}

STEPS = [
    "Upload",
    "Deduplicate",
    "Keywords",
    "AI Screening",
    "Results",
]

BATCH_SIZE = 5
GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_DAILY_TOKEN_LIMIT = 2_000_000
CALL_DELAY_SECONDS = 3.0
MAX_RETRIES = 3
TITLE_SIMILARITY_THRESHOLD = 90


def init_state() -> None:
    defaults = {
        "step": 1,
        "groq_api_key": "",
        "research_topic": "",
        "merge_files": True,
        "df_raw": None,
        "df": None,
        "df_deduped": None,
        "df_filtered": None,
        "df_scored": None,
        "column_mapping": {},
        "mapping_confirmed": False,
        "missing_required": [],
        "total_uploaded": 0,
        "after_dedup": 0,
        "after_keyword_filter": 0,
        "after_domain_filter": 0,
        "after_ai_screening": 0,
        "final_included": 0,
        "duplicates_removed": 0,
        "doi_dupes": 0,
        "title_dupes": 0,
        "inclusion_keywords": "",
        "exclusion_keywords": "",
        "domain_exclusions": [],
        "target_papers": 100,
        "suggested_keywords": [],
        "keyword_suggestions_ready": False,
        "screening_done": False,
        "is_screening": False,
        "screening_index": 0,
        "screening_scores": [],
        "screening_reasons": [],
        "screening_stopped": False,
        "screening_total": 0,
        "screening_tokens_used": 0,
        "groq_first_error": None,
        "score_threshold": 6,
        "upload_names": [],
        "needs_manual_mapping": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_secret_api_key() -> str:
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        return ""


def get_groq_api_key() -> str:
    user_key = (st.session_state.get("groq_api_key") or "").strip()
    if user_key:
        return user_key
    return get_secret_api_key()


def normalize_col_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def auto_map_columns(columns: list[str]) -> dict[str, str | None]:
    normalized = {normalize_col_name(c): c for c in columns}
    mapping: dict[str, str | None] = {}
    used: set[str] = set()
    for field, aliases in COLUMN_ALIASES.items():
        mapping[field] = None
        for alias in aliases:
            if alias in normalized and normalized[alias] not in used:
                mapping[field] = normalized[alias]
                used.add(normalized[alias])
                break
    return mapping


def parse_year(value) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def normalize_doi(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    text = text.replace("doi:", "").strip()
    return text


def normalize_title(value) -> str:
    text = str(value).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_csv_file(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, on_bad_lines="skip")
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8", on_bad_lines="skip", engine="python")


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    out = pd.DataFrame()
    for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
        source = mapping.get(field)
        if source and source in df.columns:
            out[field] = df[source]
        else:
            out[field] = ""
    out["year"] = out["year"].apply(parse_year)
    out["doi"] = out["doi"].apply(normalize_doi)
    for col in ("title", "abstract", "keywords", "journal", "item_type"):
        out[col] = out[col].fillna("").astype(str)
    out["title"] = out["title"].str.strip()
    out = out[out["title"].str.len() > 0].copy()
    out = out.reset_index(drop=True)
    return out


def missing_required_fields(mapping: dict[str, str | None], df: pd.DataFrame | None = None) -> list[str]:
    missing = []
    for field in REQUIRED_FIELDS:
        source = mapping.get(field)
        if not source:
            missing.append(field)
            continue
        if df is not None and source in df.columns:
            series = df[source]
            if series.isna().all() or (series.astype(str).str.strip() == "").all():
                missing.append(field)
    return missing


def parse_keyword_list(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[\n,;]+", text)
    return [p.strip() for p in parts if p.strip()]


def text_contains_any(text: str, phrases: list[str]) -> bool:
    haystack = text.lower()
    return any(p.lower() in haystack for p in phrases)


def suggest_keywords_from_topic(topic: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", topic)
    keywords: list[str] = []
    for token in tokens:
        if token.lower() in STOPWORDS:
            continue
        if len(token) < 3:
            continue
        keywords.append(token)
    phrases = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", topic)
    for phrase in phrases:
        if phrase not in keywords:
            keywords.insert(0, phrase)
    seen: set[str] = set()
    unique: list[str] = []
    for item in keywords:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:12]


def suggest_keywords_with_groq(topic: str, api_key: str) -> list[str] | None:
    try:
        from groq import Groq

        client = Groq(api_key=api_key, timeout=30.0)
        prompt = (
            "Suggest 8-12 concise inclusion keywords for screening academic papers "
            "on this research topic. Return only a JSON array of strings. "
            f"Topic: {topic}"
        )
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        content = response.choices[0].message.content or "[]"
        match = re.search(r"\[.*\]", content, re.DOTALL)
        payload = json.loads(match.group(0) if match else content)
        if isinstance(payload, list):
            return [str(x).strip() for x in payload if str(x).strip()]
    except Exception:
        return None
    return None


def deduplicate_papers(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    work = df.copy()
    work["_year_num"] = pd.to_numeric(work["year"], errors="coerce").fillna(0).astype(int)
    work = work.sort_values("_year_num", ascending=False)

    doi_dupes = 0
    has_doi = work["doi"].astype(str).str.len() > 0
    with_doi = work[has_doi]
    without_doi = work[~has_doi]
    before_doi = len(with_doi)
    with_doi = with_doi.drop_duplicates(subset=["doi"], keep="first")
    doi_dupes = before_doi - len(with_doi)
    work = pd.concat([with_doi, without_doi], ignore_index=True)
    work = work.sort_values("_year_num", ascending=False).reset_index(drop=True)

    work["_norm_title"] = work["title"].apply(normalize_title)
    before_exact = len(work)
    work = work.drop_duplicates(subset=["_norm_title"], keep="first").reset_index(drop=True)
    exact_title_dupes = before_exact - len(work)

    buckets: dict[str, list[tuple[int, str]]] = defaultdict(list)
    keep_mask = [True] * len(work)
    fuzzy_dupes = 0
    for idx, row in work.iterrows():
        title = row["_norm_title"]
        if not title:
            continue
        prefix = title[:4]
        is_dup = False
        for _kept_idx, kept_title in buckets[prefix]:
            if fuzz.ratio(title, kept_title) >= TITLE_SIMILARITY_THRESHOLD:
                is_dup = True
                break
        if is_dup:
            keep_mask[idx] = False
            fuzzy_dupes += 1
        else:
            buckets[prefix].append((idx, title))

    work = work[pd.Series(keep_mask, index=work.index)].copy()
    work = work.drop(columns=["_year_num", "_norm_title"], errors="ignore")
    title_dupes = exact_title_dupes + fuzzy_dupes
    return work.reset_index(drop=True), doi_dupes, title_dupes


def apply_keyword_filters(
    df: pd.DataFrame,
    inclusion: list[str],
    exclusion: list[str],
    domains: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current = df.copy()

    if inclusion:
        mask = current["title"].apply(lambda t: text_contains_any(str(t), inclusion))
        after_incl = current[mask].copy()
    else:
        after_incl = current

    if exclusion:
        def excluded(row) -> bool:
            blob = f"{row['title']} {row['abstract']} {row['keywords']}"
            return text_contains_any(blob, exclusion)

        after_excl = after_incl[~after_incl.apply(excluded, axis=1)].copy()
    else:
        after_excl = after_incl

    if domains:
        terms: list[str] = []
        for domain in domains:
            terms.extend(DOMAIN_TERMS.get(domain, [domain.lower()]))

        def domain_hit(row) -> bool:
            blob = f"{row['title']} {row['abstract']} {row['keywords']}".lower()
            return any(term.lower() in blob for term in terms)

        after_domain = after_excl[~after_excl.apply(domain_hit, axis=1)].copy()
    else:
        after_domain = after_excl

    return after_incl.reset_index(drop=True), after_excl.reset_index(drop=True), after_domain.reset_index(drop=True)


def extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None
    return None


def safe_console_print(message: str, **kwargs) -> None:
    """Print to the terminal without raising on Windows charmap encodings."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    text = str(message).encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(text, **kwargs)


def score_one_paper(client, topic: str, row: pd.Series) -> tuple[int, str]:
    abstract = str(row.get("abstract") or "").strip()
    if not abstract:
        abstract = "(No abstract provided. Score using the title and other fields only.)"

    prompt = (
        "You are a research paper screener. Given the research topic and paper details, "
        "rate this paper's relevance on a scale of 0-10. Return only a JSON object with "
        "two fields: score (integer 0-10) and reason (one sentence). "
        f"Research topic: {topic}. "
        f"Paper title: {row.get('title', '')}. "
        f"Abstract: {abstract}. "
        f"Keywords: {row.get('keywords', '')}. "
        f"Journal: {row.get('journal', '')}. "
        f"Type: {row.get('item_type', '')}"
    )

    last_error = "Unknown error"
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            content = response.choices[0].message.content or ""
            data = extract_json_object(content)
            if not data:
                last_error = "Could not parse model response as JSON"
                safe_console_print(
                    f"[LitFilter] Groq response was not JSON (attempt {attempt + 1}/{MAX_RETRIES}): {content[:500]!r}",
                    flush=True,
                )
                time.sleep(CALL_DELAY_SECONDS)
                continue
            score = int(data.get("score", 0))
            score = max(0, min(10, score))
            reason = str(data.get("reason", "")).strip() or "No reason provided."
            usage = getattr(response, "usage", None)
            tokens_used = int(getattr(usage, "total_tokens", 0) or 0)
            return score, reason, tokens_used
        except Exception as exc:
            name = type(exc).__name__
            last_error = str(exc) or name
            safe_console_print(
                f"[LitFilter] Groq API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {name}: {exc}",
                flush=True,
            )
            wait = CALL_DELAY_SECONDS
            if "rate" in name.lower() or "rate" in last_error.lower() or "429" in last_error:
                wait = 5 * (attempt + 1)
            elif "timeout" in name.lower() or "timeout" in last_error.lower():
                wait = 3 * (attempt + 1)
            time.sleep(wait)

    return 0, f"Skipped after {MAX_RETRIES} retries: {last_error}", 0


def score_paper_batch(
    df: pd.DataFrame,
    start_idx: int,
    batch_size: int,
    topic: str,
    api_key: str,
) -> tuple[list[int], list[str], str | None, int]:
    from groq import Groq

    client = Groq(api_key=api_key, timeout=45.0)
    scores: list[int] = []
    reasons: list[str] = []
    first_error: str | None = None
    batch_tokens: int = 0
    total = len(df)
    end_idx = min(start_idx + batch_size, total)

    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        score, reason, tokens = score_one_paper(client, topic, row)
        if first_error is None and reason.startswith("Skipped after"):
            first_error = reason
            safe_console_print(f"[LitFilter] First Groq screening error: {first_error}", flush=True)
        scores.append(score)
        reasons.append(reason)
        batch_tokens += tokens
        if i < end_idx - 1 or end_idx < total:
            time.sleep(CALL_DELAY_SECONDS)

    return scores, reasons, first_error, batch_tokens


def run_groq_screening(df: pd.DataFrame, topic: str, api_key: str, progress_bar, status) -> pd.DataFrame:
    from groq import Groq

    client = Groq(api_key=api_key, timeout=45.0)
    scored = df.copy()
    scores: list[int] = []
    reasons: list[str] = []
    total = len(scored)
    first_error: str | None = None

    for i, (_, row) in enumerate(scored.iterrows()):
        status.text(f"Scoring paper {i + 1} of {total}…")
        score, reason, _ = score_one_paper(client, topic, row)
        if first_error is None and reason.startswith("Skipped after"):
            first_error = reason
            safe_console_print(f"[LitFilter] First Groq screening error: {first_error}", flush=True)
            st.session_state.groq_first_error = first_error
            st.error(first_error)
        scores.append(score)
        reasons.append(reason)
        if progress_bar is not None:
            progress_bar.progress((i + 1) / total)
        if i < total - 1:
            time.sleep(CALL_DELAY_SECONDS)

    scored["score"] = scores
    scored["reason"] = reasons
    return scored


def estimate_paper_tokens(row: pd.Series, topic: str) -> int:
    abstract = str(row.get("abstract") or "").strip()
    if not abstract:
        abstract = "(No abstract provided. Score using the title and other fields only.)"
    title = str(row.get("title") or "").strip()
    keywords = str(row.get("keywords") or "").strip()
    journal = str(row.get("journal") or "").strip()
    item_type = str(row.get("item_type") or "").strip()
    prompt_text = (
        f"{topic} {title} {abstract} {keywords} {journal} {item_type} "
        "You are a research paper screener. Given the research topic and paper details, "
        "rate this paper's relevance on a scale of 0-10. Return only a JSON object with "
        "two fields: score (integer 0-10) and reason (one sentence)."
    )
    prompt_tokens = int(len(prompt_text) / 3.8)
    return prompt_tokens + 50


def estimate_total_tokens(df: pd.DataFrame, topic: str) -> int:
    if df is None or df.empty:
        return 0
    return sum(estimate_paper_tokens(row, topic) for _, row in df.iterrows())


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Screened Papers") -> bytes:
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    except Exception:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def format_export_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    col_order = ["title", "year", "score", "reason", "journal", "item_type", "abstract", "keywords", "doi"]
    col_labels = {
        "title": "Title",
        "year": "Year",
        "score": "Score",
        "reason": "Reason",
        "journal": "Journal",
        "item_type": "Item Type",
        "abstract": "Abstract",
        "keywords": "Keywords",
        "doi": "DOI",
    }
    present = [c for c in col_order if c in df.columns]
    remaining = [c for c in df.columns if c not in col_order and not str(c).startswith("_")]
    final_cols = present + remaining
    return df[final_cols].rename(columns=col_labels)


def prisma_figure(
    identified: int,
    after_dedup: int,
    after_keyword: int,
    after_domain: int,
    after_ai: int,
    included: int,
) -> go.Figure:
    dup_removed = max(identified - after_dedup, 0)
    kw_removed = max(after_dedup - after_keyword, 0)
    domain_removed = max(after_keyword - after_domain, 0)
    ai_removed = max(after_domain - after_ai, 0)
    thresh_removed = max(after_ai - included, 0)

    boxes = [
        (0.5, 5.6, f"Records identified\nn = {identified}"),
        (0.5, 4.5, f"After deduplication\nn = {after_dedup}"),
        (0.5, 3.4, f"After keyword filter\nn = {after_keyword}"),
        (0.5, 2.3, f"After domain filter\nn = {after_domain}"),
        (0.5, 1.2, f"After AI screening\nn = {after_ai}"),
        (0.5, 0.1, f"Included\nn = {included}"),
    ]
    excludes = [
        (1.55, 5.05, f"Duplicates removed\nn = {dup_removed}"),
        (1.55, 3.95, f"Keyword excluded\nn = {kw_removed}"),
        (1.55, 2.85, f"Domain excluded\nn = {domain_removed}"),
        (1.55, 1.75, f"Not relevant / failed\nn = {ai_removed}"),
        (1.55, 0.65, f"Below score threshold\nn = {thresh_removed}"),
    ]

    fig = go.Figure()
    fig.update_xaxes(range=[-0.1, 2.4], visible=False)
    fig.update_yaxes(range=[-0.4, 6.4], visible=False)
    fig.update_layout(
        height=640,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title="PRISMA flow",
        showlegend=False,
    )

    shapes = []
    annotations = []

    for x, y, text in boxes:
        shapes.append(
            dict(
                type="rect",
                x0=x - 0.38,
                x1=x + 0.38,
                y0=y,
                y1=y + 0.85,
                line=dict(color="#1f4e79", width=2),
                fillcolor="#e8f1fa",
            )
        )
        annotations.append(
            dict(x=x, y=y + 0.42, text=text, showarrow=False, font=dict(size=12, color="#1f4e79"))
        )

    for x, y, text in excludes:
        shapes.append(
            dict(
                type="rect",
                x0=x - 0.38,
                x1=x + 0.38,
                y0=y,
                y1=y + 0.7,
                line=dict(color="#8b1e1e", width=1),
                fillcolor="#fdecea",
            )
        )
        annotations.append(
            dict(x=x, y=y + 0.35, text=text, showarrow=False, font=dict(size=11, color="#8b1e1e"))
        )

    connectors = [
        (0.5, 5.6, 0.5, 5.35),
        (0.5, 4.5, 0.5, 4.25),
        (0.5, 3.4, 0.5, 3.15),
        (0.5, 2.3, 0.5, 2.05),
        (0.5, 1.2, 0.5, 0.95),
    ]
    for x0, y0, x1, y1 in connectors:
        shapes.append(
            dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#333", width=2))
        )

    side_lines = [
        (0.88, 5.4, 1.17, 5.4),
        (0.88, 4.3, 1.17, 4.3),
        (0.88, 3.2, 1.17, 3.2),
        (0.88, 2.1, 1.17, 2.1),
        (0.88, 1.0, 1.17, 1.0),
    ]
    for x0, y0, x1, y1 in side_lines:
        shapes.append(
            dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#8b1e1e", width=1))
        )

    fig.update_layout(shapes=shapes, annotations=annotations)
    return fig


def render_step_nav() -> None:
    cols = st.columns(len(STEPS))
    current = st.session_state.step
    for i, label in enumerate(STEPS, start=1):
        with cols[i - 1]:
            if i == current:
                st.markdown(f"**{i}. {label}**")
            else:
                st.caption(f"{i}. {label}")


def go_to(step: int) -> None:
    st.session_state.step = step
    st.rerun()


@st.dialog("Skip to AI Screening?")
def confirm_skip_dialog(file_obj, topic: str, api_key: str) -> None:
    st.warning(
        "⚠️ **This will skip deduplication and keyword filtering.**\n\n"
        "Make sure your CSV is already cleaned and contains the required columns "
        "(`Title`, `Abstract`, `Year`, `Keywords`, `Journal`, `Item Type`). Continue?"
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Confirm & Proceed", type="primary", use_container_width=True):
            if not topic.strip():
                st.error("Please enter a research paper title / topic statement first.")
                return
            current_key = api_key.strip() or get_groq_api_key()
            if not current_key:
                st.error("Please enter your Groq API key first.")
                return
            try:
                raw = read_csv_file(file_obj)
            except Exception as exc:
                st.error(f"Could not read CSV file: {exc}")
                return
            if raw.empty:
                st.error("The uploaded CSV is empty.")
                return
            mapping = auto_map_columns(list(raw.columns))
            missing = missing_required_fields(mapping, raw)
            if missing:
                missing_labels = [FIELD_LABELS.get(m, m) for m in missing]
                st.error(
                    f"❌ Missing required columns: {', '.join(missing_labels)}.\n\n"
                    f"Required columns needed for AI screening: {', '.join([FIELD_LABELS[f] for f in REQUIRED_FIELDS])}."
                )
                return
            standardized = apply_mapping(raw, mapping)
            if standardized.empty:
                st.error("No valid papers with a title remained in the uploaded CSV.")
                return

            st.session_state.research_topic = topic
            st.session_state.groq_api_key = api_key
            st.session_state.df_raw = standardized
            st.session_state.df = standardized.copy()
            st.session_state.df_deduped = standardized
            st.session_state.df_filtered = standardized
            st.session_state.df_scored = None
            st.session_state.column_mapping = mapping
            st.session_state.mapping_confirmed = True
            st.session_state.missing_required = []
            st.session_state.total_uploaded = len(standardized)
            st.session_state.after_dedup = len(standardized)
            st.session_state.after_keyword_filter = len(standardized)
            st.session_state.after_domain_filter = len(standardized)
            st.session_state.screening_done = False
            st.session_state.is_screening = False
            st.session_state.screening_index = 0
            st.session_state.screening_scores = []
            st.session_state.screening_reasons = []
            st.session_state.screening_stopped = False
            st.session_state.screening_total = len(standardized)
            st.session_state.groq_first_error = None
            go_to(4)
    with c2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


def render_upload() -> None:
    st.header("Step 1 — File upload")
    st.caption("Upload Scopus, IEEE, ACM, or ScienceDirect CSV exports.")

    topic = st.text_area(
        "Research paper title / topic statement",
        value=st.session_state.research_topic,
        placeholder="e.g. Retrieval-augmented generation for systematic literature reviews",
        help="Used as context when Groq scores each paper.",
    )
    st.session_state.research_topic = topic

    has_secret_key = bool(get_secret_api_key())
    if has_secret_key:
        st.info("🔒 **Groq API key:** Using configured key from secrets.")
        with st.expander("Override API key (optional)", expanded=False):
            manual_key = st.text_input(
                "Custom Groq API key",
                type="password",
                placeholder="Enter a different key or leave blank to use secrets",
                value=st.session_state.get("groq_api_key", ""),
                help="Optional: enter a different key if you do not want to use the configured secret.",
            )
            st.session_state.groq_api_key = manual_key.strip()
    else:
        manual_key = st.text_input(
            "Groq API key",
            type="password",
            placeholder="Enter your Groq API key (gsk_...)",
            value=st.session_state.get("groq_api_key", ""),
            help="Your API key is kept in memory during this session and never stored.",
        )
        st.session_state.groq_api_key = manual_key.strip()

    with st.expander("⚡ Already have a cleaned/filtered CSV? Skip to AI Screening", expanded=False):
        st.info(
            "Upload your pre-cleaned CSV here to bypass deduplication and keyword filtering and jump straight to Step 4 (AI Screening)."
        )
        direct_file = st.file_uploader(
            "Upload pre-cleaned CSV",
            type=["csv"],
            key="direct_precleaned_csv",
            help="Must include Title, Abstract, Year, Keywords, Journal, and Item Type columns.",
        )
        if direct_file is not None:
            if st.button("Proceed directly to AI Screening →", type="primary", key="btn_direct_skip"):
                confirm_skip_dialog(direct_file, topic, st.session_state.get("groq_api_key", ""))

    st.divider()

    st.subheader("Standard workflow: Upload database export(s)")
    files = st.file_uploader(
        "CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="You can upload one file or several databases to merge.",
    )

    merge = st.checkbox(
        "Merge multiple CSVs into one master dataset",
        value=st.session_state.merge_files,
    )
    st.session_state.merge_files = merge

    st.markdown(
        "**Required CSV columns:** Title, Abstract, Year, Keywords, "
        "Journal/Conference Name, Item Type. DOI is optional but used for deduplication."
    )

    if not files:
        st.info("Upload at least one CSV to continue.")
        return

    try:
        frames = []
        names = []
        for f in files:
            names.append(f.name)
            frames.append(read_csv_file(f))
    except Exception as exc:
        st.error(f"Could not read a CSV file: {exc}")
        return

    if merge or len(frames) == 1:
        raw = pd.concat(frames, ignore_index=True, sort=False)
    else:
        choice = st.selectbox("Use a single CSV", options=names)
        raw = frames[names.index(choice)]

    if raw.empty:
        st.error("The selected file(s) produced an empty dataset.")
        return

    st.session_state.upload_names = names
    mapping = auto_map_columns(list(raw.columns))
    if st.session_state.column_mapping and set(st.session_state.column_mapping.keys()) >= set(mapping.keys()):
        saved = st.session_state.column_mapping
        for field in mapping:
            if saved.get(field) in list(raw.columns):
                mapping[field] = saved[field]

    missing = missing_required_fields(mapping, raw)
    auto_complete = len(missing) == 0
    st.session_state.needs_manual_mapping = not auto_complete

    if missing:
        st.error(
            "Missing or unmapped required columns: "
            + ", ".join(FIELD_LABELS[m] for m in missing)
            + ". Map your columns below."
        )
    else:
        st.success("All required columns were matched automatically. You can still adjust mapping.")

    st.subheader("Column mapping")
    st.caption("If your CSV headers do not match the expected names, map them here.")
    options = ["— not mapped —"] + list(raw.columns)
    mapped: dict[str, str | None] = {}
    col_a, col_b = st.columns(2)
    fields = REQUIRED_FIELDS + OPTIONAL_FIELDS
    for i, field in enumerate(fields):
        target = col_a if i % 2 == 0 else col_b
        current = mapping.get(field) or "— not mapped —"
        if current not in options:
            current = "— not mapped —"
        with target:
            selected = st.selectbox(
                FIELD_LABELS[field] + (" (optional)" if field in OPTIONAL_FIELDS else ""),
                options=options,
                index=options.index(current),
                key=f"map_{field}",
            )
        mapped[field] = None if selected == "— not mapped —" else selected

    st.session_state.column_mapping = mapped
    missing = missing_required_fields(mapped, raw)
    if missing:
        st.error(
            "Still missing required columns: "
            + ", ".join(FIELD_LABELS[m] for m in missing)
        )

    st.dataframe(raw.head(8), use_container_width=True)
    st.caption(f"{len(raw)} rows · {len(raw.columns)} columns")

    if st.button("Continue to deduplication", type="primary"):
        if not topic.strip():
            st.error("Enter a research title / topic statement.")
            return
        if not get_groq_api_key():
            st.error("Enter a Groq API key, or set GROQ_API_KEY in Streamlit secrets.")
            return
        if missing:
            st.error("Map every required column before continuing.")
            return
        standardized = apply_mapping(raw, mapped)
        if standardized.empty:
            st.error("No papers with a title remained after mapping.")
            return
        st.session_state.df_raw = standardized
        st.session_state.df = standardized.copy()
        st.session_state.df_deduped = None
        st.session_state.df_filtered = None
        st.session_state.df_scored = None
        st.session_state.total_uploaded = len(standardized)
        st.session_state.mapping_confirmed = True
        st.session_state.missing_required = missing
        st.session_state.screening_done = False
        st.session_state.keyword_suggestions_ready = False
        go_to(2)


def render_dedup() -> None:
    st.header("Step 2 — Deduplication")
    df = st.session_state.df_raw
    if df is None or df.empty:
        st.warning("No dataset loaded. Go back to upload.")
        if st.button("Back to upload"):
            go_to(1)
        return

    st.write(f"Papers loaded: **{len(df)}**")
    doi_count = int((df["doi"].astype(str).str.len() > 0).sum())
    if doi_count == 0:
        st.info("No DOI values found. DOI matching will be skipped; titles will be used.")
    else:
        st.write(f"Rows with DOI: **{doi_count}**")

    st.caption(
        "Duplicates are removed by exact DOI, then by title similarity "
        f"({TITLE_SIMILARITY_THRESHOLD}%+). The latest year is kept."
    )

    if st.button("Run deduplication", type="primary"):
        cleaned, doi_dupes, title_dupes = deduplicate_papers(df)
        st.session_state.df = cleaned
        st.session_state.df_deduped = cleaned
        st.session_state.doi_dupes = doi_dupes
        st.session_state.title_dupes = title_dupes
        st.session_state.duplicates_removed = doi_dupes + title_dupes
        st.session_state.after_dedup = len(cleaned)
        st.session_state.screening_done = False

    if st.session_state.after_dedup:
        st.success(
            f"Removed **{st.session_state.duplicates_removed}** duplicates "
            f"({st.session_state.doi_dupes} by DOI, {st.session_state.title_dupes} by title). "
            f"**{st.session_state.after_dedup}** papers remain."
        )
        st.dataframe(
            st.session_state.df[["title", "year", "journal", "item_type"]].head(20),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back"):
            go_to(1)
    with c2:
        if st.button("Continue to keyword filtering", disabled=not st.session_state.after_dedup):
            go_to(3)


def render_keywords() -> None:
    st.header("Step 3 — Keyword filtering")
    df = st.session_state.df_deduped if st.session_state.df_deduped is not None else st.session_state.df
    if df is None or df.empty:
        st.warning("The dataset is empty. Go back and reload or relax earlier filters.")
        if st.button("Back to deduplication"):
            go_to(2)
        return

    topic = st.session_state.research_topic
    if not st.session_state.keyword_suggestions_ready:
        suggestions = suggest_keywords_from_topic(topic)
        api_key = get_groq_api_key()
        if api_key:
            ai_suggestions = suggest_keywords_with_groq(topic, api_key)
            if ai_suggestions:
                suggestions = ai_suggestions
        st.session_state.suggested_keywords = suggestions
        if not st.session_state.inclusion_keywords:
            st.session_state.inclusion_keywords = ", ".join(suggestions)
        st.session_state.keyword_suggestions_ready = True

    if st.session_state.suggested_keywords:
        st.caption("Suggested from your research topic (editable below).")
        st.write(", ".join(st.session_state.suggested_keywords))

    inclusion = st.text_area(
        "Inclusion keywords (paper title must contain at least one)",
        value=st.session_state.inclusion_keywords,
        help="Comma or newline separated. Leave empty to skip inclusion filtering.",
    )
    exclusion = st.text_area(
        "Exclusion keywords (papers containing these are removed)",
        value=st.session_state.exclusion_keywords,
        help="Matched against title, abstract, and keywords.",
    )
    domains = st.multiselect(
        "Domain exclusion",
        options=list(DOMAIN_TERMS.keys()),
        default=st.session_state.domain_exclusions,
        help="Papers matching these domains in title, abstract, or keywords are removed.",
    )
    target = st.number_input(
        "Target number of final papers",
        min_value=1,
        max_value=5000,
        value=int(st.session_state.target_papers),
        step=50,
    )

    st.session_state.inclusion_keywords = inclusion
    st.session_state.exclusion_keywords = exclusion
    st.session_state.domain_exclusions = domains
    st.session_state.target_papers = int(target)

    incl_list = parse_keyword_list(inclusion)
    excl_list = parse_keyword_list(exclusion)
    after_incl, after_excl, after_domain = apply_keyword_filters(df, incl_list, excl_list, domains)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("After dedup", st.session_state.after_dedup or len(df))
    m2.metric("After inclusion keywords", len(after_incl))
    m3.metric("After exclusion keywords", len(after_excl))
    m4.metric("After domain exclusion", len(after_domain))

    if after_domain.empty:
        st.warning(
            "No papers remain after these filters. Relax inclusion, exclusion, or domain rules, "
            "or go back."
        )
    else:
        st.dataframe(
            after_domain[["title", "year", "journal", "item_type"]].head(20),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back"):
            go_to(2)
    with c2:
        if st.button("Continue to AI screening", type="primary", disabled=after_domain.empty):
            st.session_state.df = after_domain
            st.session_state.df_filtered = after_domain
            st.session_state.after_keyword_filter = len(after_excl)
            st.session_state.after_domain_filter = len(after_domain)
            st.session_state.screening_done = False
            st.session_state.is_screening = False
            st.session_state.screening_index = 0
            st.session_state.screening_scores = []
            st.session_state.screening_reasons = []
            st.session_state.screening_stopped = False
            st.session_state.screening_total = len(after_domain)
            go_to(4)


def render_screening() -> None:
    st.header("Step 4 — Groq API screening")
    if st.session_state.get("is_screening", False):
        st.warning("⚠️ **Screening in progress** — do not refresh or close this tab, or you will lose progress.")

    df = st.session_state.df_filtered if st.session_state.df_filtered is not None else st.session_state.df
    if df is None or df.empty:
        st.warning("No papers left to screen. Go back and adjust filters.")
        if st.button("Back to keywords"):
            go_to(3)
        return

    api_key = get_groq_api_key()
    if not api_key:
        st.error("Groq API key is missing. Go back to Step 1 and enter it.")
        if st.button("Back to upload"):
            go_to(1)
        return

    total = len(df)
    st.write(f"Papers to score: **{total}**")
    st.caption(
        f"Model: `{GROQ_MODEL}`. About {CALL_DELAY_SECONDS:.0f}s between calls "
        "(~30 requests/minute). Timeouts retry up to 3 times, then the paper is scored 0."
    )
    missing_abs = int((df["abstract"].astype(str).str.strip() == "").sum())
    if missing_abs:
        st.info(f"{missing_abs} paper(s) have no abstract and will be scored from the title.")

    est_tokens = estimate_total_tokens(df, st.session_state.research_topic)
    if est_tokens > GROQ_DAILY_TOKEN_LIMIT:
        st.warning(
            f"⚠️ **Estimated tokens for this run: ~{est_tokens:,}** across {total:,} API requests. "
            f"This is likely to exceed Groq's free-tier daily limit of {GROQ_DAILY_TOKEN_LIMIT:,} tokens for `{GROQ_MODEL}`. "
            "Consider narrowing your dataset using keyword or domain filters in Step 3."
        )
    else:
        st.info(
            f"ℹ️ **Estimated tokens for this run: ~{est_tokens:,}** across {total:,} API requests. "
            f"Your daily limit is {GROQ_DAILY_TOKEN_LIMIT:,} tokens for `{GROQ_MODEL}`."
        )

    if st.session_state.get("groq_first_error"):
        st.error(st.session_state.groq_first_error)

    if st.session_state.get("is_screening", False):
        current_idx = st.session_state.get("screening_index", 0)
        tokens_used = st.session_state.get("screening_tokens_used", 0)

        # Live request and token usage counters
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Requests made", f"{current_idx} / {total} papers")
        with col_m2:
            st.metric("Tokens used this session", f"~{tokens_used:,}" if tokens_used > 0 else "0")

        progress_val = min(1.0, current_idx / total) if total > 0 else 1.0
        st.progress(progress_val)
        status = st.empty()
        next_batch_end = min(current_idx + BATCH_SIZE, total)
        status.text(f"Scoring papers {current_idx + 1} to {next_batch_end} of {total}…")

        col_start, col_stop = st.columns(2)
        with col_start:
            st.button("Start AI screening", disabled=True)
        with col_stop:
            stop_clicked = st.button("Stop screening", type="secondary", key="stop_screening_btn")

        if stop_clicked:
            st.session_state.is_screening = False
            st.session_state.screening_stopped = True
            scored_count = len(st.session_state.screening_scores)
            if scored_count > 0:
                scored = df.iloc[:scored_count].copy()
                scored["score"] = st.session_state.screening_scores
                scored["reason"] = st.session_state.screening_reasons
                st.session_state.df = scored
                st.session_state.df_scored = scored
                st.session_state.after_ai_screening = len(scored)
                st.session_state.screening_done = True
            else:
                st.session_state.screening_done = False
            st.rerun()

        try:
            batch_scores, batch_reasons, first_err, batch_tok = score_paper_batch(
                df,
                current_idx,
                BATCH_SIZE,
                st.session_state.research_topic,
                api_key,
            )
            if first_err and not st.session_state.get("groq_first_error"):
                st.session_state.groq_first_error = first_err
                st.error(first_err)

            st.session_state.screening_scores.extend(batch_scores)
            st.session_state.screening_reasons.extend(batch_reasons)
            st.session_state.screening_tokens_used = st.session_state.get("screening_tokens_used", 0) + batch_tok
            st.session_state.screening_index = next_batch_end

            # Persist intermediate scored progress in session state after every batch
            intermediate_df = df.iloc[:len(st.session_state.screening_scores)].copy()
            intermediate_df["score"] = st.session_state.screening_scores
            intermediate_df["reason"] = st.session_state.screening_reasons
            st.session_state.df_scored = intermediate_df
            st.session_state.df = intermediate_df

            if st.session_state.screening_index >= total:
                st.session_state.is_screening = False
                st.session_state.screening_stopped = False
                st.session_state.after_ai_screening = len(intermediate_df)
                st.session_state.screening_done = True
            st.rerun()
        except Exception as exc:
            st.session_state.is_screening = False
            st.error(f"Screening stopped: {exc}")
            return

    elif not st.session_state.get("screening_done", False):
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("Start AI screening", type="primary"):
                st.session_state.groq_first_error = None
                st.session_state.is_screening = True
                st.session_state.screening_stopped = False
                st.session_state.screening_index = 0
                st.session_state.screening_scores = []
                st.session_state.screening_reasons = []
                st.session_state.screening_tokens_used = 0
                st.session_state.screening_total = total
                st.session_state.df_scored = None
                st.rerun()
        with col_stop:
            st.button("Stop screening", disabled=True)

    else:
        scored = st.session_state.df_scored if st.session_state.df_scored is not None else st.session_state.df
        total_original = st.session_state.get("screening_total", 0) or (
            len(st.session_state.df_filtered) if st.session_state.df_filtered is not None else len(scored)
        )
        tokens_final = st.session_state.get("screening_tokens_used", 0)
        token_info = f" (Tokens used: ~{tokens_final:,})" if tokens_final > 0 else ""
        if st.session_state.get("screening_stopped", False):
            st.warning(f"Stopped after **{len(scored)}** of **{total_original}** papers.{token_info}")
        else:
            st.success(f"Scored **{len(scored)}** papers.{token_info}")

        if scored is not None and not scored.empty and "score" in scored.columns:
            fig = px.histogram(
                scored,
                x="score",
                nbins=11,
                title="Score distribution",
                range_x=[-0.5, 10.5],
            )
            fig.update_layout(xaxis_title="Relevance score", yaxis_title="Papers", bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)

        if st.button("Score again"):
            st.session_state.screening_done = False
            st.session_state.is_screening = False
            st.session_state.screening_stopped = False
            st.session_state.screening_index = 0
            st.session_state.screening_scores = []
            st.session_state.screening_reasons = []
            st.session_state.screening_tokens_used = 0
            st.session_state.df_scored = None
            if st.session_state.df_filtered is not None:
                st.session_state.df = st.session_state.df_filtered
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", disabled=st.session_state.get("is_screening", False)):
            go_to(3)
    with c2:
        can_continue = (
            st.session_state.get("screening_done", False)
            and not st.session_state.get("is_screening", False)
            and st.session_state.df_scored is not None
            and not st.session_state.df_scored.empty
        )
        if st.button("Continue to results", disabled=not can_continue, type="primary"):
            go_to(5)


def suggested_threshold(df: pd.DataFrame, target: int) -> int:
    if df.empty or "score" not in df.columns:
        return 6
    counts = []
    for t in range(1, 11):
        counts.append(((df["score"] >= t).sum(), t))
    counts.sort(key=lambda x: abs(x[0] - target))
    return counts[0][1]


def render_results() -> None:
    st.header("Step 5 — Results")
    df = st.session_state.df_scored if st.session_state.df_scored is not None else st.session_state.df
    if df is None or df.empty or "score" not in df.columns:
        st.warning("No scored papers to show. Go back and run screening.")
        if st.button("Back to screening"):
            go_to(4)
        return

    hint = suggested_threshold(df, int(st.session_state.target_papers))
    threshold = st.slider(
        "Minimum score threshold",
        min_value=1,
        max_value=10,
        value=int(st.session_state.score_threshold),
        help=f"To approach {st.session_state.target_papers} papers, a threshold near {hint} is typical.",
        key="score_threshold_slider",
    )
    st.session_state.score_threshold = st.session_state.score_threshold_slider

    included = df[df["score"] >= threshold].copy()
    included = included.sort_values(["score", "year"], ascending=[False, False])
    st.session_state.final_included = len(included)

    st.metric("Papers above threshold", len(included), help=f"Target was {st.session_state.target_papers}")

    if included.empty:
        st.warning("No papers meet this threshold. Lower the slider or go back.")
    else:
        display_cols = ["title", "year", "score", "reason", "journal", "item_type"]
        table = included[display_cols].rename(
            columns={
                "title": "Title",
                "year": "Year",
                "score": "Score",
                "reason": "Reason",
                "journal": "Journal",
                "item_type": "Item Type",
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Export & Downloads")

    # 1. Included Papers (Papers meeting threshold)
    st.markdown(f"**Included Papers (Score ≥ {threshold}):** {len(included)} paper(s)")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download Included Papers (CSV)",
            data=dataframe_to_csv_bytes(format_export_df(included)),
            file_name="litfilter_included_papers.csv",
            mime="text/csv",
            disabled=included.empty,
            key="dl_inc_csv",
        )
    with d2:
        st.download_button(
            "Download Included Papers (Excel)",
            data=dataframe_to_excel_bytes(format_export_df(included), sheet_name="Included Papers"),
            file_name="litfilter_included_papers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=included.empty,
            key="dl_inc_excel",
        )

    # 2. Rated Papers (All papers scored by AI)
    rated_df = df if (df is not None and not df.empty and "score" in df.columns) else pd.DataFrame()
    st.markdown(f"**Rated Papers (All scored papers):** {len(rated_df)} paper(s)")
    r1, r2 = st.columns(2)
    with r1:
        st.download_button(
            "Download Rated Papers (CSV)",
            data=dataframe_to_csv_bytes(format_export_df(rated_df)),
            file_name="litfilter_rated_papers.csv",
            mime="text/csv",
            disabled=rated_df.empty,
            key="dl_rated_csv",
        )
    with r2:
        st.download_button(
            "Download Rated Papers (Excel)",
            data=dataframe_to_excel_bytes(format_export_df(rated_df), sheet_name="Rated Papers"),
            file_name="litfilter_rated_papers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=rated_df.empty,
            key="dl_rated_excel",
        )

    # 3. Unrated Papers (Papers not yet scored, e.g. if stopped early)
    all_pool = (
        st.session_state.df_filtered
        if st.session_state.df_filtered is not None
        else st.session_state.df_raw
    )
    scored_len = len(rated_df)
    unrated_df = (
        all_pool.iloc[scored_len:].copy()
        if (all_pool is not None and len(all_pool) > scored_len)
        else pd.DataFrame()
    )
    st.markdown(f"**Unrated Papers (Not yet scored):** {len(unrated_df)} paper(s)")
    if unrated_df.empty:
        st.caption("All papers from the current filter set were scored (no unrated papers remaining).")
    u1, u2 = st.columns(2)
    with u1:
        st.download_button(
            "Download Unrated Papers (CSV)",
            data=dataframe_to_csv_bytes(format_export_df(unrated_df)) if not unrated_df.empty else b"",
            file_name="litfilter_unrated_papers.csv",
            mime="text/csv",
            disabled=unrated_df.empty,
            key="dl_unrated_csv",
        )
    with u2:
        st.download_button(
            "Download Unrated Papers (Excel)",
            data=dataframe_to_excel_bytes(format_export_df(unrated_df), sheet_name="Unrated Papers") if not unrated_df.empty else b"",
            file_name="litfilter_unrated_papers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=unrated_df.empty,
            key="dl_unrated_excel",
        )

    st.subheader("PRISMA numbers")
    identified = st.session_state.total_uploaded
    after_dedup = st.session_state.after_dedup
    after_kw = st.session_state.after_keyword_filter
    after_domain = st.session_state.after_domain_filter
    after_ai = st.session_state.after_ai_screening
    final_n = st.session_state.final_included

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Started with", identified)
    p2.metric("After deduplication", after_dedup)
    p3.metric("After keyword filter", after_kw)
    p4.metric("After AI screening (included)", final_n)

    st.plotly_chart(
        prisma_figure(identified, after_dedup, after_kw, after_domain, after_ai, final_n),
        use_container_width=True,
    )

    if st.button("Back to screening"):
        go_to(4)


def main() -> None:
    st.set_page_config(page_title="LitFilter", page_icon="🔍", layout="wide")
    init_state()

    st.title("LitFilter")
    st.caption("Your literature review, minus the tedious part.")
    render_step_nav()
    st.divider()

    with st.sidebar:
        st.header("Progress")
        st.write(f"Uploaded: {st.session_state.total_uploaded}")
        st.write(f"After dedup: {st.session_state.after_dedup}")
        st.write(f"After keywords: {st.session_state.after_keyword_filter}")
        st.write(f"After domain: {st.session_state.after_domain_filter}")
        st.write(f"After AI screening: {st.session_state.after_ai_screening}")
        st.write(f"Final included: {st.session_state.final_included}")
        if st.button("Start over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    step = st.session_state.step
    if step == 1:
        render_upload()
    elif step == 2:
        render_dedup()
    elif step == 3:
        render_keywords()
    elif step == 4:
        render_screening()
    else:
        render_results()


if __name__ == "__main__":
    main()
