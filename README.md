# LitFilter

LitFilter is a Streamlit app for screening papers in a systematic literature review. You upload bibliographic CSVs (Scopus, IEEE, ACM, ScienceDirect, or similar), remove duplicates, apply keyword and domain filters, then score remaining papers for relevance with Groq (`qwen/qwen3.8-27b`). Results can be thresholded, exported, and summarized with PRISMA-style counts and a simple flow diagram.

## Requirements

- Python 3.10 or later
- A [Groq](https://console.groq.com/) API key (free-tier keys work; the app waits ~2 seconds between calls to stay near 30 requests/minute)

Python packages (see `requirements.txt`):

- `streamlit`
- `pandas`
- `groq`
- `thefuzz`
- `python-Levenshtein`
- `plotly`
- `openpyxl`
- `xlsxwriter`

## Installation

```bash
git clone <your-repo-url>
cd litfilter
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

The app opens in your browser (typically `http://localhost:8501`).

Optional: store the API key in Streamlit secrets instead of typing it each time. Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_key_here"
```

The key in the Step 1 input box is used first. If that box is empty, LitFilter falls back to `st.secrets["GROQ_API_KEY"]`. Do not hardcode keys in `app.py`.

## How to use

The workflow is five steps. Session state keeps counts and datasets as you move forward and back.

1. **Upload**  
   Upload one or more CSVs. Choose whether to merge them into one master table or use a single file. Enter the research title/topic (used as scoring context) and your Groq API key. If headers do not match the expected names, map them in the column-mapping UI. A red error lists any still-missing required fields.

2. **Deduplicate**  
   Exact DOI matches are removed first (skipped if DOI is absent). Titles are then compared with fuzzy matching; pairs at 90%+ similarity are treated as duplicates. The record with the latest year is kept. You see how many DOI and title duplicates were removed.

3. **Keywords**  
   Inclusion keywords are suggested from your topic (and from Groq when the API key works). Papers must contain **at least one** inclusion phrase in the **title**. Exclusion keywords and optional domain tags (Robotics, Medical, Healthcare, Nature/Biology, Education, Manufacturing, Automotive, Finance, Supply Chain, Computer Networks) remove matching papers. Counts update live. Set a target number of final papers (for example 100, 200, or 300); the results page uses it to hint a score threshold. If a filter leaves zero rows, a warning is shown and you can go back. uses 400 tokens to suggest the keywords from the title we are provided 

4. **AI screening**  
   Each paper is sent to Groq with the research topic, title, abstract (or title only if the abstract is missing), keywords, journal, and item type. The model returns a 0–10 score and a one-sentence reason. A progress bar updates during the run. Rate limits and timeouts are retried up to three times; a paper that still fails is kept with score `0`. After scoring, a Plotly histogram shows the score distribution.it uses 300 token limit per paper to score it on the basis of the details which are provided in the csv file. 

5. **Results**  
   A slider sets the minimum score (1–10). The table is sortable: Title, Year, Score, Reason, Journal, Item Type. Download CSV or Excel. PRISMA counts show: started with X → after deduplication Y → after keyword filter Z → after AI screening / included. A Plotly shape diagram illustrates the same flow.

Use **Start over** in the sidebar to clear the session.

## CSV format required

Required fields (names need not match exactly; you can map them):

| Field | Typical source names |
| --- | --- |
| Title | Title, Article Title, Document Title |
| Abstract | Abstract |
| Year | Year, Publication Year, Date |
| Keywords | Keywords, Author Keywords, Index Keywords |
| Journal/Conference Name | Source title, Publication Title, Journal |
| Item Type | Document Type, Item Type, Type |

Optional:

| Field | Typical source names |
| --- | --- |
| DOI | DOI |

**Edge cases**

- Missing abstract: scoring uses the title and other available fields.
- Missing DOI: DOI deduplication is skipped; title similarity is still applied.
- Unexpected column names: use the mapping dropdowns on Step 1.
- Empty file after a filter: warning plus a back button.
- Groq timeout: three retries, then score `0` and a skip reason.

## Deployment

### Streamlit Community Cloud

1. Push this project to a GitHub repository (`app.py`, `requirements.txt`, `README.md`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and deploy the repo. Main file: `app.py`.
3. In the app settings, add a secret:

   ```toml
   GROQ_API_KEY = "your_key_here"
   ```

4. Users can still paste a key on Step 1; secrets are only a fallback.

Do not commit `.streamlit/secrets.toml` or `.env`.

### Hugging Face Spaces

1. Create a new Space: SDK **Streamlit**, Python.
2. Upload `app.py`, `requirements.txt`, and this README.
3. Optionally put YAML front matter at the **top** of `README.md`:

   ```yaml
   ---
   title: LitFilter
   emoji: 📄
   colorFrom: blue
   colorTo: indigo
   sdk: streamlit
   app_file: app.py
   pinned: false
   ---
   ```

4. In the Space **Settings → Variables and secrets**, add `GROQ_API_KEY` as a secret. Streamlit on Spaces reads it via `st.secrets` if you also add it under the Space’s Streamlit secrets, or users can paste the key in the app.

5. Hardware: CPU is enough. Screening is slow by design (~2 seconds per paper) so that free-tier rate limits are less likely to trip.

## Project layout

```
litfilter/
├── app.py              # entire application
├── requirements.txt
├── README.md
└── .gitignore
```
