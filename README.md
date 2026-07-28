# Quant

This repo is for research, analysis, backtests, and derived outputs.

## Data Rule

- Read shared data from the platform-specific Google Drive data folder.
- Write only derived research outputs or project artifacts.
- Do not download or scrape source data from this repo.
- Do not import provider clients from the note repo's `module` directory.
- Use `pandas.read_parquet`, `pandas.read_csv`, `pandas.read_pickle`, or the
  local helpers in `cloud_data.py`.

Source data updates belong in the note repo. `project_paths.py` selects the
default root automatically: `/Users/xinc/GitHub` on macOS and
`C:\Users\user\Documents\GitHub` on Windows. Set `GITHUB_ROOT`,
`QUANT_ROOT`, `DATA_ROOT`, or `DATA_DOWNLOAD_OWNER_ROOT` to override it.

`cloud_data.py` contains the shared path table, local readers, and a lightweight
HTTP guard so research notebooks do not silently refresh remote data.

When a shared source file is missing, update it from note:

```bash
cd "<your GitHub root>/note"
python scripts/data_updates/update_quant_market_data.py
```
