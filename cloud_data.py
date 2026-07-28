import urllib.request

from pathlib import Path

import pandas as pd

from project_paths import DATA_ROOT, NOTE_REPO_ROOT



class QuantRemoteDownloadBlocked(RuntimeError):
    pass


def _inside(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        target = root.resolve()
    except OSError:
        return False
    return resolved == target or target in resolved.parents


def remote_download_allowed() -> bool:
    return _inside(Path.cwd(), NOTE_REPO_ROOT)


def _block_message() -> str:
    return (
        f"Remote data download is blocked in {Path(__file__).resolve().parent}. "
        f"Run data update workflows from {NOTE_REPO_ROOT}, then read "
        f"the synced files under {DATA_ROOT}."
    )


def install_quant_data_guard() -> None:
    try:
        import requests
    except ImportError:
        requests = None

    if requests is not None:
        original_request = requests.sessions.Session.request
        if not getattr(original_request, "_quant_policy_wrapped", False):

            def guarded_request(self, method, url, *args, **kwargs):
                if not remote_download_allowed():
                    raise QuantRemoteDownloadBlocked(_block_message())
                return original_request(self, method, url, *args, **kwargs)

            guarded_request._quant_policy_wrapped = True
            requests.sessions.Session.request = guarded_request

    original_urlopen = urllib.request.urlopen
    if not getattr(original_urlopen, "_quant_policy_wrapped", False):

        def guarded_urlopen(*args, **kwargs):
            if not remote_download_allowed():
                raise QuantRemoteDownloadBlocked(_block_message())
            return original_urlopen(*args, **kwargs)

        guarded_urlopen._quant_policy_wrapped = True
        urllib.request.urlopen = guarded_urlopen


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def require_data_file(path: str | Path, *, updater: str | None = None) -> Path:
    resolved = Path(path)
    if resolved.exists():
        return resolved
    message = f"Shared data file is missing: {resolved}"
    if updater:
        message += f". Update it from the note repo with: {updater}"
    raise FileNotFoundError(message)


def read_frame(
    path: str | Path,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    date_column: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a shared dataframe and optionally filter it by calendar date.

    ``start`` and ``end`` are applied locally after the file is read. When a
    date range is supplied, ``date_column`` defaults to the first available
    column among ``date``, ``Date``, ``Timestamp``, ``datetime``, and ``time``.
    """
    resolved = require_data_file(path)
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(resolved, **kwargs)
    elif suffix == ".csv":
        frame = pd.read_csv(resolved, **kwargs)
    elif suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(resolved, **kwargs)
    else:
        raise ValueError(f"Unsupported shared dataframe format: {resolved}")

    if start is None and end is None:
        return frame

    if date_column is None:
        date_column = next(
            (column for column in ("date", "Date", "Timestamp", "datetime", "time") if column in frame.columns),
            None,
        )

    if date_column is not None:
        if date_column not in frame.columns:
            raise KeyError(f"date column not found: {date_column}")
        dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    elif isinstance(frame.index, pd.DatetimeIndex):
        dates = pd.Series(frame.index.normalize(), index=frame.index)
    else:
        raise ValueError(
            f"Cannot apply start/end filter to {resolved}: pass date_column=... "
            "or use a dataframe with a DatetimeIndex."
        )

    if start is not None:
        frame = frame.loc[dates.ge(pd.Timestamp(start).normalize())]
    if end is not None:
        frame = frame.loc[dates.le(pd.Timestamp(end).normalize())]
    return frame


def manifest_path() -> Path:
    return data_path("_index", "file_manifest.csv")


def dataset_index_path() -> Path:
    return data_path("_index", "dataset_index.csv")


TM_ROOT = data_path("tm")
TRADING_ROOT = data_path("trading")

TW_STOCK_ROOT = data_path("trading", "tw_stock")
TW_STOCK_REFERENCE = TW_STOCK_ROOT / "reference"
TW_STOCK_DISPOSAL = TW_STOCK_ROOT / "disposal"
TW_STOCK_KBAR_ROOT = TW_STOCK_ROOT / "kbar"
TW_STOCK_KBAR_1D = TW_STOCK_KBAR_ROOT / "1D"
TW_STOCK_KBAR_1MIN = TW_STOCK_KBAR_ROOT / "1min"
TW_STOCK_KBAR_ABOVE_MA60 = TW_STOCK_KBAR_1MIN / "above_ma60"
TW_STOCK_KBAR_1H = TW_STOCK_KBAR_ROOT / "1H"
TW_STOCK_DAILY_PRICE = TW_STOCK_KBAR_1D / "daily_stock_price.parquet"
TW_STOCK_TAIEX_DAILY = TW_STOCK_KBAR_1D / "TAIEX.parquet"
TW_STOCK_SECTOR_MAP = TW_STOCK_ROOT / "sector_map.parquet"
TW_STOCK_INDEX_ROOT = TW_STOCK_ROOT / "index"
TW_STOCK_TWII = TW_STOCK_INDEX_ROOT / "twii.pkl"
TW_STOCK_MARGIN_INFO = TW_STOCK_REFERENCE / "margin_info.csv"
TW_STOCK_OVERVIEW = TW_STOCK_REFERENCE / "台股總覽.csv"
TW_STOCK_INSTITUTIONAL_TRADING = TW_STOCK_REFERENCE / "整體市場三大法人買賣表.csv"
TW_STOCK_MARGIN_ROOT = TW_STOCK_ROOT / "margin"
TW_STOCK_OTC_MARGIN_BALANCE = TW_STOCK_MARGIN_ROOT / "otc_margin_balance.parquet"
TW_STOCK_DISPOSAL_INFORMATION = TW_STOCK_DISPOSAL / "disposal_information.pkl"
TW_STOCK_PROCESSED_DISPOSAL = TW_STOCK_DISPOSAL / "processed_disposal_events.csv"
TW_FUTURES_ROOT = data_path("trading", "tw_futures")
TW_FUTURES_REFERENCE = TW_FUTURES_ROOT / "reference"
TW_FUTURES_TX = TW_FUTURES_ROOT / "TX.csv"
TW_FUTURES_TX_AFTER = TW_FUTURES_ROOT / "TX_after.csv"
TW_FUTURES_TX_LEGACY_NOTE = TW_FUTURES_REFERENCE / "TX_legacy_note.csv"
TW_FUTURES_SETTLE_DATE_TX = TW_FUTURES_REFERENCE / "settle_date_TX.csv"
TW_OPTIONS_ROOT = data_path("trading", "tw_options")
TW_OPTIONS_REFERENCE = TW_OPTIONS_ROOT / "reference"
TW_OPTIONS_TICK = TW_OPTIONS_ROOT / "tick"
TW_OPTIONS_TXO = TW_OPTIONS_ROOT / "TXO.pkl"
TW_OPTIONS_TXO_AFTER = TW_OPTIONS_ROOT / "TXO_after.pkl"
TW_OPTIONS_SETTLE_TXO = TW_OPTIONS_REFERENCE / "settle_TXO.csv"
TW_OPTIONS_INSTITUTION_ROOT = TW_OPTIONS_ROOT / "institution_position"
TW_OPTIONS_INSTITUTION_DAY = TW_OPTIONS_INSTITUTION_ROOT / "day.parquet"
TW_OPTIONS_INSTITUTION_NIGHT = TW_OPTIONS_INSTITUTION_ROOT / "night.parquet"

MARKET_DATA_ROOT = data_path("trading", "market_data")
MARKET_TRADINGVIEW_ROOT = MARKET_DATA_ROOT / "tradingview"
MARKET_MOVE_DAILY = MARKET_TRADINGVIEW_ROOT / "TVC" / "MOVE" / "1D_contract-0_extended-0.parquet"
MARKET_SOX_DAILY = MARKET_TRADINGVIEW_ROOT / "TVC" / "SOX" / "1D_contract-0_extended-0.parquet"
MARKET_FEAR_GREED = MARKET_DATA_ROOT / "finmind" / "cnn_fear_greed.parquet"

CRYPTO_COINTEGRATION = data_path("trading", "crypto", "cointegration")
CRYPTO_SPOT = CRYPTO_COINTEGRATION / "spot"
CRYPTO_FUTURES = CRYPTO_COINTEGRATION / "futures"
CRYPTO_FUNDING_RATE = CRYPTO_COINTEGRATION / "funding_rate"
CRYPTO_METADATA = CRYPTO_COINTEGRATION / "metadata"
CRYPTO_CACHE = CRYPTO_COINTEGRATION / "cache"
CRYPTO_OHLCV_1D = data_path("trading", "crypto", "ohlcv_1d")


def read_tx_futures(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Read and normalize the front-month TX day/night rows from shared CSV."""
    frame = read_frame(TW_FUTURES_TX)
    frame = frame.loc[frame["futures_id"].astype(str).eq("TX")].copy()
    frame["contract_date"] = frame["contract_date"].astype(str).str.strip()
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"]).dt.normalize()
    frame = frame.loc[~frame["contract_date"].str.contains("/", na=False)]
    front = frame.groupby(["Timestamp", "trading_session"])["contract_date"].transform("min")
    frame = frame.loc[frame["contract_date"].eq(front)].drop_duplicates(
        ["Timestamp", "trading_session"], keep="last"
    )
    if start is not None:
        frame = frame.loc[frame["Timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame["Timestamp"] <= pd.Timestamp(end)]
    return frame.set_index("Timestamp").sort_index()


install_quant_data_guard()
