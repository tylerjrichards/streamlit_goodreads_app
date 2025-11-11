# Product Requirements Document (PRD)

Project: Streamlit Chess / Recommendations App
Author: Jayfetra
Date: 2025-11-11
Repository: https://github.com/Jayfetra/streamlit_goodreads_app

## 1 — Overview (elevator pitch)
A Streamlit application that downloads chess data from chess.com for a given user_id, then displays exploration and analysis in the UI. The app will also call an LLM (ChatGPT/OpenAI) to provide player recommendations based on the downloaded data.

The project is deployed at the domain chesstics.com.

## 2 — Goals & Success Metrics
- Goal:
  Display a user's chess data and provide actionable recommendations to help the player improve.
- Success metrics:
  - App starts locally within 30 seconds on a typical developer machine after environment setup.
  - Data successfully downloads from chess.com for a valid user_id (with sensible timeout/retry behavior).
  - The main dashboard displays the user's core statistics and visualizations.
  - The LLM-based recommendations are produced and shown on the UI.
  - No secrets are committed to the repo; `.env` (or environment variables) is used for keys.

## 3 — Background & Context
This repo contains a Streamlit app and helper scripts to download and explore chess data. The main app file is `chess_streamlit.py`.

## 4 — Users & Personas
- Developer (owner): runs the app locally, iterates on code, and experiments with features.
- Reviewer: TBD (use code owners or assign reviewers per PR).
- Non-technical user: can run the packaged app following the README instructions or use the deployed site on chesstics.com.

## 5 — Key User Flows
1. Developer clones repo, creates an env, installs dependencies, runs:
   ```bash
   streamlit run chess_streamlit.py
   ```
2. User sees the chesstics website, is able to view historical game data, and receives recommendations to help improve their play.


## 6 — Functional Requirements
F1. App loads and renders the main page with no runtime exceptions.
F2. Provide simple filters (by date range, color: black/white, opponent rating ranges, and text search).
F3. Display at least two visualizations (e.g., bar chart of counts, time series of rating or activity).
F4. Export filtered results to an Excel file (uses `openpyxl`).
F5. Read secrets from `.env` using `python-dotenv` (no secrets in repo).
F6. Provide a lightweight smoke-test script that imports core packages and checks basic behaviour.
F7. Provide a settings/config panel to enter and persist the chess.com user_id and optionally an API key for OpenAI (sourced from `.env`).

## 7 — Non-Functional Requirements
N1. Local install should be reproducible via `requirements.txt` or a recommended conda env (conda recommended for Windows).
N2. Keep memory usage modest; app should work with the sample CSV without heavy RAM.
N3. Prefer prebuilt binary packages for heavy deps (pyarrow) to avoid local compilation.
N4. Code must be linted minimally and have clear, small commits for each change.

## 8 — Data Sources & Format
- Primary data will be downloaded from the chess.com API (public endpoints) for a given user_id. Implement sensible rate-limit handling and retries.
- Additional data may come from the `chess-openings-master/` folder (TSV) or local CSVs such as `recommendations_df.csv`.
 
Key fields expected from chess.com (examples):
- game_id, url, pgn, white.username, black.username, time_control, end_time, rated, result, white_rating, black_rating

## 9 — Environment & Dependencies
- Primary runtime: Python 3.10 (recommended) for best binary wheel availability on Windows.
- Recommended environment: conda env `chesstic`.
- Key packages: streamlit, pandas, numpy, pyarrow (via conda-forge), openpyxl, python-dotenv, requests, plotly, seaborn.
- `requirements.txt` exists but prefer a conda-first install to avoid `pyarrow` build issues on Windows.

Quick setup (recommended):
```powershell
conda create -n chesstic python=3.10 -y
conda activate chesstic
conda install -c conda-forge pyarrow pandas numpy streamlit openpyxl -y
python -m pip install -r requirements.txt
streamlit run chess_streamlit.py
```

## 10 — UX / Screen List
- Main Dashboard: summary stats, filters, and a table.
- Item Detail: click an item to see full metadata and any associated visualizations.
- Export: button to export current filtered view to Excel.

## 10.1 — Deployment
- The app is deployed at `chesstics.com`. Document deployment steps (Streamlit Cloud, Docker + host, or VPS) and store deployment config outside of the repo if it contains secrets.

## 10.2 — Key repository files
- `chess_streamlit.py` — main Streamlit app entrypoint
- `chess_com_download.py` — helper script to download chess.com data
- `deepseek_chess.py` — analysis utility
- `recommendations_df.csv` — example dataset used by the app
- `requirements.txt` — pip requirements (keep in sync with conda environment recommendations)

## 11 — Acceptance Criteria
AC1: Running `streamlit run chess_streamlit.py` in the recommended env opens the app and shows the main dashboard.
AC2: Default dataset (sample included) loads and displays; for the sample file (<=10k rows) loading should complete within 5s on a typical dev machine.
AC3: Filter controls correctly filter the table and charts and reflect in exported files.
AC4: Export produces a valid `.xlsx` file matching the filtered table.
AC5: No secrets in the repo; `.env` is included in `.gitignore` and an `.env.example` lists required keys.

## 12 — Risks & Mitigations
- Risk: `pyarrow` build fails on Windows (observed). Mitigation: install `pyarrow` via conda-forge before pip installs.
- Risk: Committing secrets. Mitigation: Add `.env` to `.gitignore`, use `python-dotenv` to load keys.
- Risk: Breaking the app during iterative "vibe coding". Mitigation: use feature branches, small commits, and draft PRs for review.

## 13 — Workflow & "Vibe Coding" Safety Rules
- Always branch: `git switch -c feat/short-description`.
- Commit small, descriptive commits.
- Open a draft PR early for larger changes — mark it as `Draft` in GitHub.
- Do not push directly to `master`/`main`. Use PRs and pull requests to merge.
- Add tests or smoke-test steps for any behavior you change.
- Inform 1 step at a time, and focus on the current task before moving on to the next one. Keep the to do list into maximum 4 task on every chat. 

## 14 — Open Questions / Decisions
- Do we prefer `conda` or stick to `venv` for cross-developer consistency? (Recommendation: conda on Windows.) -> we prefer to use conda. I already run "conda activate myenv" this code to activate the envoronment
- Do we want CI (GitHub Actions) to run smoke tests on PRs? (Nice to have.) -> I don't know anything about CI, but if you think it helps, then yes

## 15 — Next Steps & Owner
- Owner: Jayfetra
- Next steps:
  1. Add `docs/PRD.md` to the repo (this file).
  2. Add `.env.example` to show needed env vars (no secrets).
  3. Create `chesstic` conda env and verify dependencies.
  4. Create a branch for code review: `feat/prd-setup` and open a PR referencing this PRD.

---

Links
- README: `README.md`
- Requirements: `requirements.txt`

Appendix: Quick smoke-test (save as `check_install.py`):
```python
import importlib
pkgs = ['streamlit','pandas','numpy','pyarrow']
for p in pkgs:
    try:
        m = importlib.import_module(p)
        print(p, getattr(m,'__version__','ok'))
    except Exception as e:
        print(p, 'ERROR', e)
```

Vibe-coding note: create a branch, commit often, and open a draft PR before large refactors. This PRD will be versioned in `docs/PRD.md`, so iterate on it in the repo.
