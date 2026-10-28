# India Housing Intelligence — Streamlit

A Streamlit app for exploring Indian housing data, training regression/classification baselines, viewing saved metrics, and estimating prices.

## Files

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `india_housing_prices.csv` — add your dataset here if it is approved for deployment
- `artifacts/` — created at runtime for model files and metrics

## Run locally

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Add `app.py`, `requirements.txt`, and (if appropriate) `india_housing_prices.csv`.
3. Push the files to GitHub.
4. Sign in to https://share.streamlit.io/ with GitHub.
5. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
6. Deploy. Streamlit Community Cloud will build the environment and provide a URL such as `https://your-app-name.streamlit.app`.

If the dataset contains personal, licensed, confidential, or otherwise restricted information, do not commit it to a public repository. Use the app's sidebar uploader or an approved private storage pattern instead.

## Important persistence limitation

The Community Cloud filesystem is not a durable database. Models and `artifacts/metrics.json` saved during an app session can disappear when the app restarts or redeploys. For reliable persistence, train models in a controlled offline pipeline and version model/metrics artifacts in an approved artifact store or database. Avoid committing sensitive model artifacts.

## Modeling notes

- The app can train regression models against `Price_in_Lakhs` (current price).
- It can also create a five-year *simulated* target using a fixed 7% annual appreciation assumption. Metrics against that target only show how well a model reproduces the formula; they do not validate future market prices.
- Classification training is enabled only when the dataset contains `Good_Investment` with recognizable binary labels.
- Confirm the target's definition and check for target leakage before using the results.
- This is a research/decision-support tool, not a property valuation or investment guarantee.
