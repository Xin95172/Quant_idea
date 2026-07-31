from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import module.plot_func as plot
import plotly.graph_objects as go
from IPython.display import display
from cloud_data import MARKET_FEAR_GREED
from project_paths import NOTE_REPO_ROOT


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters for the active TX day-session strategy."""

    backtest_start_date: str = '2026-03-01'
    move_threshold: float = 0.0001
    sox_threshold: float = 0.0075
    gap_threshold: float = 0.001
    foreign_option_threshold: float = -0.0035
    divergence_threshold: float = -0.0015
    day_long_position: float = 1.0
    day_short_position: float = -1.0
    night_position: float = 0.0


class StrategyEngine:
    """Stateless factor and position logic for the active TX strategy."""

    REQUIRED_COLUMNS = {
        'futures_id', 'Open', 'Close', 'Close_a', 'MOVE_open', 'MOVE_high', 'MOVE_low', 'MOVE_close',
        'SOX_open', 'SOX_close', 'foreign_opt_pos_divergence_a',
    }

    @classmethod
    def calculate_factors(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the frame with all strategy-derived factor columns."""
        missing_columns = cls.REQUIRED_COLUMNS - set(frame.columns)
        if missing_columns:
            raise KeyError(f"strategy data missing columns: {sorted(missing_columns)}")

        df = frame.copy()
        df['SOX_ind'] = ((df['SOX_close'] / df['SOX_open']) - 1).shift(1).ffill()
        df['MOVE_ind'] = (df['MOVE_close'] / df['MOVE_open']) - 1
        df['MOVE_vol'] = ((df['MOVE_high'] / df['MOVE_low']) - 1).shift(1)

        df['3_ma'] = df['Close_a'].rolling(window=3).mean()
        df['divergence'] = (df['Close_a'] / df['3_ma']) - 1
        df['3_ma_v2'] = df['Close'].rolling(window=3).mean()
        df['divergence_v2'] = ((df['Close'] / df['3_ma_v2']) - 1).shift(1)
        df['gap'] = (df['Close_a'] / df['Close'].shift(1)) - 1
        df['gap_v2'] = (df['Open'] / df['Close'].shift(1)) - 1
        return df

    @staticmethod
    def apply_positions(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        """Return active day and night positions for a factor-ready frame."""
        df = frame.loc[frame.index > config.backtest_start_date].copy()
        df.dropna(subset='futures_id', inplace=True)
        df['pos_night'] = config.night_position
        df['pos_day'] = 0.0

        move_below_threshold = df['MOVE_ind'] < config.move_threshold
        sox_below_threshold = df['SOX_ind'] < config.sox_threshold
        foreign_option_bearish = df['foreign_opt_pos_divergence_a'] < config.foreign_option_threshold

        base_gap_supports_long = df['gap'] < config.gap_threshold
        base_divergence_supports_long = df['divergence'] < config.divergence_threshold
        df.loc[move_below_threshold & sox_below_threshold & ~base_gap_supports_long, 'pos_day'] = config.day_long_position
        df.loc[move_below_threshold & ~sox_below_threshold, 'pos_day'] = config.day_long_position
        df.loc[~move_below_threshold & foreign_option_bearish, 'pos_day'] = config.day_short_position
        df.loc[~move_below_threshold & ~foreign_option_bearish & ~base_divergence_supports_long, 'pos_day'] = config.day_long_position

        final_divergence_supports_long = df['divergence_v2'] < config.divergence_threshold
        df.loc[move_below_threshold, 'pos_day'] = config.day_long_position
        df.loc[~move_below_threshold & foreign_option_bearish, 'pos_day'] = config.day_short_position
        df.loc[~move_below_threshold & ~foreign_option_bearish & ~final_divergence_supports_long, 'pos_day'] = config.day_long_position
        return df


class TXAnalyzer:
    """Notebook-facing facade for TX features, diagnostics, and backtesting."""

    # =========================================================================
    # Construction and session preprocessing
    # =========================================================================
    def __init__(self, df: pd.DataFrame, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()
        self.df = self._prepare_session_frame(df)
        self._add_session_return_columns()

    @staticmethod
    def _prepare_session_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Pair day and night rows by their normalized close date."""
        frame = df.copy()
        frame.index = pd.to_datetime(frame.index).normalize()
        day_frame = frame.loc[frame['trading_session'].eq('position')].copy()
        night_frame = frame.loc[frame['trading_session'].eq('after_market')].copy()

        if day_frame.index.duplicated().any() or night_frame.index.duplicated().any():
            raise ValueError('each trading session must contain at most one row per close date')

        return pd.concat([day_frame, night_frame.add_suffix('_a')], axis=1)

    def _add_session_return_columns(self) -> None:
        """Add day/night return, cumulative return, PnL, and cumulative PnL columns."""
        for suffix in ('', '_a'):
            open_column = f'Open{suffix}'
            close_column = f'Close{suffix}'
            return_column = f'daily_ret{suffix}'
            pnl_column = f'daily_pnl{suffix}'
            self.df[return_column] = (self.df[close_column] / self.df[open_column]) - 1
            self.df[f'cum_{return_column}'] = self.df[return_column].cumsum()
            self.df[pnl_column] = self.df[close_column] - self.df[open_column]
            self.df[f'cum_{pnl_column}'] = self.df[pnl_column].cumsum()

    @staticmethod
    def _prepare_return_curves(
        df: pd.DataFrame,
        *,
        sort_by: str,
        return_columns: tuple[str, ...] = ('daily_ret_a', 'daily_ret'),
    ) -> pd.DataFrame:
        """Sort by a factor and add demeaned and cumulative return curves."""
        frame = df.sort_values(by=sort_by).reset_index(drop=True).copy()

        for column in return_columns:
            demeaned_column = f'demeaned_{column}'
            frame[demeaned_column] = frame[column] - frame[column].mean()
            frame[f'cum_{demeaned_column}'] = frame[demeaned_column].cumsum()
            frame[f'cum_{column}'] = frame[column].cumsum()

        return frame
    
    # =========================================================================
    # Configuration, data views, and summary helpers
    # =========================================================================
    def _get_statistics(self, ret_col: pd.Series) -> pd.Series:
        return ret_col.describe()

    @staticmethod
    def _percentile_value(values: pd.Series, percentile: float, side: str) -> float:
        """Return a raw lower or upper percentile value from a non-empty series."""
        if not 0 < percentile <= 100:
            raise ValueError('percentile must be greater than 0 and at most 100')
        if side not in {'low', 'high'}:
            raise ValueError("side must be either 'low' or 'high'")

        clean_values = values.dropna()
        if clean_values.empty:
            raise ValueError('no valid observations are available for this percentile')

        quantile = percentile / 100
        if side == 'high':
            quantile = 1 - quantile
        return float(clean_values.quantile(quantile))

    @classmethod
    def _indicator_value(
        cls,
        values: pd.Series,
        *,
        name: str,
        return_series: bool,
        percentile: float | None,
        side: str,
    ) -> pd.Series | float | None:
        """Apply the shared indicator interface before the charting path."""
        if return_series and percentile is not None:
            raise ValueError('return_series and percentile cannot be used together')
        series = values.rename(name)
        if return_series:
            return series.dropna()
        if percentile is not None:
            return cls._percentile_value(series, percentile, side)
        return None

    def _handle_indicator_output(
        self,
        values: pd.Series,
        *,
        name: str,
        return_series: bool,
        add_to_df: bool,
        percentile: float | None,
        side: str,
    ) -> pd.Series | float | None:
        """Apply the shared indicator output interface and optionally persist a factor."""
        if add_to_df:
            if return_series or percentile is not None:
                raise ValueError('add_to_df cannot be combined with return_series or percentile')
            series = values.rename(name)
            self.merge_features(series.to_frame())
            return series.dropna()
        return self._indicator_value(
            values,
            name=name,
            return_series=return_series,
            percentile=percentile,
            side=side,
        )

    def set_config(self, config: StrategyConfig) -> None:
        """Replace the active strategy configuration for subsequent analysis and backtests."""
        self.config = config

    def session_alignment_report(self) -> pd.Series:
        """Check day and night session pairing under the close-date convention.

        The night session labeled ``D`` closes before the day session labeled
        ``D`` opens, so every available day session must have a same-date night
        session. A night-only date can occur when the following day session is
        cancelled, for example because of a typhoon closure.
        """
        day_available = self.df[['Open', 'Close']].notna().all(axis=1)
        night_available = self.df[['Open_a', 'Close_a']].notna().all(axis=1)
        paired = day_available & night_available
        return pd.Series({
            'total_dates': len(self.df),
            'paired_day_night_dates': int(paired.sum()),
            'day_only_dates': int((day_available & ~night_available).sum()),
            'night_only_dates': int((~day_available & night_available).sum()),
            'missing_both_dates': int((~day_available & ~night_available).sum()),
            'duplicate_dates': int(self.df.index.duplicated().sum()),
            'close_date_alignment_ok': bool(
                not (day_available & ~night_available).any()
                and not self.df.index.duplicated().any()
            ),
        }, name='session_alignment')

    def for_period(
        self,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> "TXAnalyzer":
        """Return an independent analyzer view limited to an inclusive date range."""
        frame = self.df
        if start is not None:
            frame = frame.loc[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame.index <= pd.Timestamp(end)]

        view = object.__new__(TXAnalyzer)
        view.config = self.config
        view.df = frame.copy()
        return view

    @staticmethod
    def split_periods(
        index: pd.Index,
        *,
        start: str | pd.Timestamp | None = None,
        train_ratio: float = 0.6,
        validation_ratio: float = 0.2,
    ) -> dict[str, pd.Timestamp]:
        """Split ordered trading dates into train, optional validation, and test periods."""
        if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
            raise ValueError('train_ratio must be positive; validation_ratio must be non-negative; their sum must be less than 1')

        dates = pd.DatetimeIndex(pd.to_datetime(index)).normalize().unique().sort_values()
        if start is not None:
            dates = dates[dates > pd.Timestamp(start)]
        if len(dates) < 3:
            raise ValueError('at least three trading dates are required to create train, validation, and test periods')

        train_count = max(1, int(len(dates) * train_ratio))
        validation_count = int(len(dates) * validation_ratio)
        if train_count + validation_count >= len(dates):
            raise ValueError('split ratios leave no trading dates for the test period')

        split = {
            'train_end': dates[train_count - 1],
            'test_start': dates[train_count + validation_count],
        }
        if validation_count:
            split['validation_start'] = dates[train_count]
            split['validation_end'] = dates[train_count + validation_count - 1]
        return split
    
    def display_df(self) -> pd.DataFrame:
        return self.df
    
    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        return StrategyEngine.calculate_factors(df)

    def _apply_signals_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        return StrategyEngine.apply_positions(df, self.config)

    def _calculate_metrics(self, returns: pd.Series, point_version: bool = False) -> dict:
        """
        Calculate strategy performance metrics including advanced stats.
        returns: Series of daily returns (percentage if point_version=False, points if True)
        """
        if returns.empty:
            return {}
        returns.dropna(inplace=True)
        
        # 1. Basic Distributions
        total_pnl = returns.sum()
        
        if point_version:
            # Points Mode: Calculation is additive
            cum_ret = returns.cumsum()
            total_ret_val = total_pnl
        else:
            # Percentage Mode: Calculation is compounded
            cum_ret = (1 + returns).cumprod()
            total_ret_val = cum_ret.iloc[-1] - 1 if not cum_ret.empty else 0.0
        
        # 2. Time Statistics
        trading_days = len(returns)
        ann_factor = 252 
        
        # CAGR (Only for percentage mode)
        if not point_version and trading_days > 0:
            cagr = (1 + total_ret_val) ** (ann_factor / trading_days) - 1
        else:
            cagr = 0.0

        # Volatility (Annualized)
        vol_ann = returns.std() * np.sqrt(ann_factor)

        # Sharpe Ratio (Rf=0)
        sharpe = 0.0
        if vol_ann > 0:
            if point_version:
                # For point version, we use simplified annualized profit over volatility
                ann_pnl = (total_pnl / trading_days) * ann_factor if trading_days > 0 else 0
                sharpe = ann_pnl / vol_ann
            else:
                sharpe = returns.mean() / returns.std() * np.sqrt(ann_factor)
        
        # 3. Drawdown Statistics
        running_max = cum_ret.cummax()
        if point_version:
            drawdown = cum_ret - running_max # Points down
        else:
            drawdown = cum_ret / running_max - 1
            
        max_dd = drawdown.min()
        
        # Max DD Duration (Days)
        is_dd = drawdown < 0
        dd_duration = is_dd.astype(int).groupby((is_dd != is_dd.shift()).cumsum()).cumsum()
        max_dd_duration = dd_duration.max() if not dd_duration.empty else 0

        # 4. Trade Statistics (Win Rate, Odds, etc.)
        # Filter non-zero returns to count "Active Trading Days"
        active_rets = returns[returns != 0]
        n_trades = len(active_rets)
        
        if n_trades > 0:
            wins = active_rets[active_rets > 0]
            losses = active_rets[active_rets < 0]
            
            n_win = len(wins)
            win_rate = n_win / n_trades
            
            avg_win = wins.mean() if not wins.empty else 0.0
            avg_loss = losses.mean() if not losses.empty else 0.0
            
            # Profit Factor
            gross_profit = wins.sum()
            gross_loss = abs(losses.sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            # Odds (Avg Win / |Avg Loss|)
            odds = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else 0.0
            
            # Expectancy (Avg Return per trade)
            avg_return = active_rets.mean() # 筆均
            
            # Kelly Criterion
            # W - (1-W)/R  (W: Win Rate, R: Odds)
            if odds > 0:
                kelly = win_rate - (1 - win_rate) / odds
            else:
                kelly = 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            odds = 0.0
            avg_return = 0.0
            kelly = 0.0

        return {
            'Total Return' if not point_version else 'Total PnL': total_ret_val,
            'CAGR': cagr,
            'Volatility': vol_ann,
            'Sharpe': sharpe,
            'Max Drawdown' if not point_version else 'Max Points DD': max_dd,
            'Max DD Duration': max_dd_duration,
            'Profit Factor': profit_factor,
            'Win Rate': win_rate,
            'Odds': odds,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Avg Return (Exp)': avg_return,
            'Kelly': kelly
        }

    def update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace the working frame after an explicit research transformation."""
        self.df = df
        return self.df

    # =========================================================================
    # Feature ingestion and normalization
    # =========================================================================
    def merge_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Left-join date-indexed features onto the strategy frame."""
        features = features.copy()
        features.index = pd.to_datetime(features.index).normalize()
        features = features.loc[~features.index.duplicated(keep='last')]

        df = self.df.copy()
        df.index = pd.to_datetime(df.index).normalize()
        df = df.drop(columns=features.columns.intersection(df.columns))
        self.df = df.join(features, how='left')
        return self.df

    def add_market_ohlc(self, market_df: pd.DataFrame, prefix: str, *, shift: int = 1) -> pd.DataFrame:
        """Add a daily market OHLC series with a consistent factor prefix."""
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = set(required_columns) - set(market_df.columns)
        if missing_columns:
            raise KeyError(f"{prefix} missing OHLC columns: {sorted(missing_columns)}")

        features = market_df[required_columns].copy()
        features.index = pd.to_datetime(features.index).normalize()
        if shift:
            features = features.shift(shift)
        features = features.rename(columns={column: f'{prefix}_{column}' for column in required_columns})
        return self.merge_features(features)

    @staticmethod
    def build_option_signals(
        day_df: pd.DataFrame,
        night_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert institutional option data into day and night strategy signals.

        Supports both the current long-form parquet schema (one row per date,
        institution, and CALL/PUT) and the legacy files retained for notebooks.
        """
        long_form_columns = {
            'date', 'call_put', 'institutional_investor',
            'buy_amount_thousand', 'sell_amount_thousand',
        }

        def build_long_form_signal(frame: pd.DataFrame, suffix: str = '') -> pd.DataFrame:
            if frame.empty:
                return pd.DataFrame(columns=[
                    f'foreign_opt_pos_divergence{suffix}', f'Dealer_Opt_Signal{suffix}',
                ])

            missing_columns = long_form_columns - set(frame.columns)
            if missing_columns:
                raise KeyError(f"option data missing columns: {sorted(missing_columns)}")

            data = frame.copy()
            data['date'] = pd.to_datetime(data['date']).dt.normalize()
            data['call_put'] = data['call_put'].replace({
                '買權': 'CALL', '賣權': 'PUT', 'Call': 'CALL', 'Put': 'PUT',
            }).str.upper()
            data['institution'] = data['institutional_investor'].replace({
                '外資及陸資': 'Foreign', '外資': 'Foreign', 'Foreign': 'Foreign',
                '自營商': 'Dealer', 'Dealer': 'Dealer',
            })
            data = data.loc[data['institution'].isin(['Foreign', 'Dealer'])]
            data['net_amount'] = data['buy_amount_thousand'] - data['sell_amount_thousand']
            data['turnover'] = data['buy_amount_thousand'] + data['sell_amount_thousand']
            pivot = data.pivot_table(
                index='date', columns=['institution', 'call_put'],
                values=['net_amount', 'turnover'], aggfunc='sum', fill_value=0,
            )

            def signal_for(institution: str) -> pd.Series:
                net_call = pivot.get(('net_amount', institution, 'CALL'), pd.Series(0, index=pivot.index))
                net_put = pivot.get(('net_amount', institution, 'PUT'), pd.Series(0, index=pivot.index))
                call_turnover = pivot.get(('turnover', institution, 'CALL'), pd.Series(0, index=pivot.index))
                put_turnover = pivot.get(('turnover', institution, 'PUT'), pd.Series(0, index=pivot.index))
                return (net_call - net_put) / (call_turnover + put_turnover).replace(0, np.nan)

            return pd.DataFrame({
                f'foreign_opt_pos_divergence{suffix}': signal_for('Foreign'),
                f'Dealer_Opt_Signal{suffix}': signal_for('Dealer'),
            })

        if long_form_columns.issubset(day_df.columns) and long_form_columns.issubset(night_df.columns):
            return build_long_form_signal(day_df).join(build_long_form_signal(night_df, '_a'), how='outer')

        # Legacy day/night schemas retained for existing historical notebooks.
        if day_df.empty:
            day_signal = pd.DataFrame(columns=['foreign_opt_pos_divergence', 'Dealer_Opt_Signal'])
        else:
            required_columns = {'date', 'call_put', 'institutional_investors', 'long_deal_amount', 'short_deal_amount'}
            missing_columns = required_columns - set(day_df.columns)
            if missing_columns:
                raise KeyError(f"day option data missing columns: {sorted(missing_columns)}")
            day_data = day_df.copy()
            day_data['date'] = pd.to_datetime(day_data['date']).dt.normalize()
            day_data['call_put'] = day_data['call_put'].replace({'買權': 'CALL', '賣權': 'PUT', 'Call': 'CALL', 'Put': 'PUT'})
            day_data['net_amount'] = day_data['long_deal_amount'] - day_data['short_deal_amount']
            day_data['turnover'] = day_data['long_deal_amount'] + day_data['short_deal_amount']
            pivot = day_data.pivot_table(index='date', columns=['institutional_investors', 'call_put'], values=['net_amount', 'turnover'], aggfunc='sum', fill_value=0)

            def legacy_signal_for(institution: str) -> pd.Series:
                net_call = pivot.get(('net_amount', institution, 'CALL'), pd.Series(0, index=pivot.index))
                net_put = pivot.get(('net_amount', institution, 'PUT'), pd.Series(0, index=pivot.index))
                call_turnover = pivot.get(('turnover', institution, 'CALL'), pd.Series(0, index=pivot.index))
                put_turnover = pivot.get(('turnover', institution, 'PUT'), pd.Series(0, index=pivot.index))
                return (net_call - net_put) / (call_turnover + put_turnover).replace(0, np.nan)

            day_signal = pd.DataFrame({'foreign_opt_pos_divergence': legacy_signal_for('外資'), 'Dealer_Opt_Signal': legacy_signal_for('自營商')})

        if night_df.empty:
            night_signal = pd.DataFrame(columns=['foreign_opt_pos_divergence_a'])
        else:
            required_columns = {'foreign_long_call_amount', 'foreign_short_call_amount', 'foreign_long_put_amount', 'foreign_short_put_amount'}
            missing_columns = required_columns - set(night_df.columns)
            if missing_columns:
                raise KeyError(f"night option data missing columns: {sorted(missing_columns)}")
            night_data = night_df.copy()
            night_data.index = pd.to_datetime(night_data.index).normalize()
            net_call = night_data['foreign_long_call_amount'] - night_data['foreign_short_call_amount']
            net_put = night_data['foreign_long_put_amount'] - night_data['foreign_short_put_amount']
            turnover = night_data[list(required_columns)].sum(axis=1).replace(0, np.nan)
            night_signal = pd.DataFrame({'foreign_opt_pos_divergence_a': (net_call - net_put) / turnover})

        return day_signal.join(night_signal, how='outer')

    def add_option_signals(self, day_df: pd.DataFrame, night_df: pd.DataFrame) -> pd.DataFrame:
        """Build and merge institutional option signals."""
        return self.merge_features(self.build_option_signals(day_df, night_df))

    @staticmethod
    def fetch_option_daily(
        client,
        option_id: str,
        start_date: str,
        end_date: str,
        *,
        pause_seconds: float = 0.5,
    ) -> pd.DataFrame:
        """Download option daily data in yearly batches to avoid API request limits."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        period_starts = pd.date_range(start=start, end=end, freq='YS')
        if start not in period_starts:
            period_starts = period_starts.insert(0, start).sort_values().unique()

        chunks = []
        for period_start in period_starts:
            period_end = min(period_start + pd.offsets.YearEnd(0), end)
            if period_start > period_end:
                continue

            option_part = client.get_option_daily(
                option_id=option_id,
                start_date=period_start,
                end_date=period_end,
                trading_session='all',
            )
            if not option_part.empty:
                chunks.append(option_part)
            if pause_seconds:
                time.sleep(pause_seconds)

        if not chunks:
            raise RuntimeError(f'未下載到 {option_id} 選擇權資料')
        return pd.concat(chunks, ignore_index=True)

    def add_option_iv_skew(
        self,
        option_df: pd.DataFrame,
        settlement_df: pd.DataFrame,
        *,
        iv_calculator,
        risk_free_rate: float = 0.015,
    ) -> pd.DataFrame:
        """Calculate option IV skew by session and merge it into the strategy frame."""
        required_option_columns = {'date', 'trading_session', 'contract_date'}
        missing_option_columns = required_option_columns - set(option_df.columns)
        if missing_option_columns:
            raise KeyError(f"option data missing columns: {sorted(missing_option_columns)}")
        required_settlement_columns = {'contract', 'settle_date'}
        missing_settlement_columns = required_settlement_columns - set(settlement_df.columns)
        if missing_settlement_columns:
            raise KeyError(f"settlement data missing columns: {sorted(missing_settlement_columns)}")

        options = option_df.copy()
        options['date'] = pd.to_datetime(options['date']).dt.normalize()
        spot = self.df[['Close', 'Close_a']].copy()
        spot.index = pd.to_datetime(spot.index).normalize()
        options = options.merge(spot, left_on='date', right_index=True, how='left')
        options['underlying_price'] = np.where(
            options['trading_session'].eq('after_market'), options['Close_a'], options['Close']
        )

        settlements = settlement_df.copy()
        settlements['settle_date'] = pd.to_datetime(settlements['settle_date'])
        iv_df = iv_calculator(
            options,
            model='bs',
            underlying_col='underlying_price',
            risk_free_rate=risk_free_rate,
            shape_options={'group_cols': ['date', 'contract_date', 'trading_session']},
            settlement_df=settlements,
            settlement_contract_col='contract',
            settlement_date_col='settle_date',
        )
        required_iv_columns = {'date', 'trading_session', 'SkewSlope', 'SkewSlope3'}
        missing_iv_columns = required_iv_columns - set(iv_df.columns)
        if missing_iv_columns:
            raise KeyError(f"IV calculation missing columns: {sorted(missing_iv_columns)}")

        skew = iv_df.groupby(['date', 'trading_session'])[['SkewSlope', 'SkewSlope3']].first().unstack('trading_session')
        skew.columns = [
            factor if session == 'position' else f'{factor}_a'
            for factor, session in skew.columns
        ]
        return self.merge_features(skew)

    @staticmethod
    def fetch_fear_greed(timeout: int = 15) -> pd.DataFrame:
        """Read the CNN Fear & Greed history prepared by the note repo."""
        if not MARKET_FEAR_GREED.exists():
            raise FileNotFoundError(
                f'{MARKET_FEAR_GREED} is missing. Update it from '
                f'{NOTE_REPO_ROOT}.'
            )
        return pd.read_parquet(MARKET_FEAR_GREED)

    def add_fear_greed(self, historical_df: pd.DataFrame, latest_df: pd.DataFrame) -> pd.DataFrame:
        """Merge historical and current CNN Fear & Greed observations, then lag one day."""
        historical = historical_df.copy()
        latest = latest_df.copy()
        latest = latest.rename(columns={'score': 'fear_greed', 'rating': 'fear_greed_emotion'})
        columns = ['fear_greed', 'fear_greed_emotion']

        def normalize(frame: pd.DataFrame) -> pd.DataFrame:
            if 'date' in frame.columns:
                frame['date'] = pd.to_datetime(frame['date']).dt.normalize()
                frame = frame.set_index('date')
            else:
                frame.index = pd.to_datetime(frame.index).normalize()
            for column in columns:
                if column not in frame:
                    frame[column] = pd.NA
            return frame[columns].loc[~frame.index.duplicated(keep='last')]

        historical = normalize(historical)
        latest = normalize(latest)
        combined = historical.reindex(historical.index.union(latest.index))
        combined.update(latest)
        combined['fear_greed'] = pd.to_numeric(combined['fear_greed'], errors='coerce')
        return self.merge_features(combined.sort_index().shift(1))

    def feature_status(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Summarize feature availability before analysis or backtesting."""
        if columns is None:
            columns = [
                'MOVE_open', 'SOX_open', 'foreign_opt_pos_divergence_a',
                'SkewSlope', 'fear_greed', 'US_bond_5y',
            ]

        status = []
        for column in columns:
            if column not in self.df:
                status.append({'feature': column, 'available': False, 'missing': len(self.df), 'last_value_date': pd.NaT})
                continue

            values = self.df[column]
            valid_dates = values.dropna().index
            status.append({
                'feature': column,
                'available': True,
                'missing': int(values.isna().sum()),
                'last_value_date': valid_dates.max() if len(valid_dates) else pd.NaT,
            })
        return pd.DataFrame(status).set_index('feature')

    # =========================================================================
    # Return and calendar summaries
    # =========================================================================
    def daily_ret(self):
        return plot.plot(self.df, ly=['cum_daily_ret', 'cum_daily_ret_a'], title='daily_return')

    def monthly_ret(self, mode: str = 'strategy', point_version: bool = False):
        """
        mode: 'strategy' (default), 'benchmark', 'night', 'day'
        """
        df = self.df.copy()
        
        target_col = 'strat_ret'
        label = "Strategy"
        
        if mode == 'strategy':
            # Ensure strategy returns are calculated
            df = self._calculate_factors(df)
            df = self._apply_signals_logic(df)
            if point_version:
                df['strat_ret'] = (df['daily_pnl_a'] * df['pos_night']) + (df['daily_pnl'] * df['pos_day'])
            else:
                df['strat_ret'] = (df['daily_ret_a'] * df['pos_night']) + (df['daily_ret'] * df['pos_day'])
            target_col = 'strat_ret'
            label = "Strategy"
            
        elif mode == 'benchmark':
            if point_version:
                df['benchmark'] = df['daily_pnl_a'] + df['daily_pnl']
            else:
                df['benchmark'] = df['daily_ret_a'] + df['daily_ret']
            target_col = 'benchmark'
            label = "Buy & Hold (Benchmark)"
            
        elif mode == 'night':
            target_col = 'daily_pnl_a' if point_version else 'daily_ret_a'
            label = "Night Session (Raw)"
            
        elif mode == 'day':
            target_col = 'daily_pnl' if point_version else 'daily_ret'
            label = "Day Session (Raw)"
        
        df['year'] = df.index.year
        df['month'] = df.index.month
        
        # Calculate monthly returns (sum for points, compounded for percentages)
        if point_version:
            monthly_df = df.groupby(['year', 'month'])[target_col].sum().reset_index()
            monthly_df['display_val'] = monthly_df[target_col]
            unit = "pts"
        else:
            monthly_df = df.groupby(['year', 'month'])[target_col].apply(lambda x: (1 + x).prod() - 1).reset_index()
            monthly_df['display_val'] = monthly_df[target_col] * 100
            unit = "%"
        
        # Create Pivot Table
        pivot_df = monthly_df.pivot(index='year', columns='month', values='display_val')
        
        # Prepare annotations
        annotations = []
        for y_idx, row in pivot_df.iterrows():
            for x_idx, val in row.items():
                if pd.notna(val):
                    annotations.append(
                        dict(
                            x=x_idx, 
                            y=y_idx, 
                            text=f"{val:.1f}" if point_version else f"{val:.2f}", 
                            showarrow=False, 
                            font=dict(color='black' if abs(val) < (100 if point_version else 5) else 'white')
                        )
                    )
                    
        # Plot Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlGn',
            colorbar=dict(title=f'PnL ({unit})'),
            zmid=0
        ))
        
        fig.update_layout(
            title=f"Monthly P&L Heatmap ({'Points' if point_version else 'Percentage'}) - {label}",
            xaxis_title='Month',
            yaxis_title='Year',
            yaxis=dict(autorange='reversed'), # Make recent years at bottom
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            annotations=annotations
        )
        
        fig.show()

    # =========================================================================
    # Factor diagnostics: position and calendar
    # =========================================================================
    def indicator_position_ret(self, *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        df = self.df.copy()
        df['ind'] = df['daily_ret'].shift(1) + df['daily_ret_a'].shift(1) + df['daily_ret'].shift(2)
        df['ind'] = df['ind'].rolling(window=3).sum()
        result = self._handle_indicator_output(df['ind'], name='position_ret', return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
        if result is not None:
            return result
        df['demeaned_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
        df = df.sort_values(by='daily_ret').reset_index(drop=True)
        df['cum_demeaned_daily_ret_a'] = df['demeaned_daily_ret_a'].cumsum()
        df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
        return plot.plot(df, ly='cum_demeaned_daily_ret_a', x='index', ry = 'daily_ret', sub_ly=['cum_daily_ret_a'])

    def indicator_gap_days(self, after_holiday: bool = False, *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        df = self.df.copy()

        # Calendar
        df.index = pd.to_datetime(df.index)
        df['prev_date'] = df.index.to_series().shift(1)
        df['gap'] = (df.index.to_series() - df['prev_date']).dt.days
        result = self._handle_indicator_output(df['gap'], name='gap_days', return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
        if result is not None:
            return result
        
        if after_holiday:
            # 想看週一 (Post-holiday)
            # Day: 週一日盤 = Current (No shift)
            # Night: 週一夜盤 = 記在週二 (Next Row) -> shift(-1)
            df['daily_ret_a'] = df['daily_ret_a'].shift(-1)
        else:
            # 想看週五 (Pre-holiday)
            # Day: 週五日盤 = 記在週五 (Prev Row) -> shift(1)
            df['daily_ret'] = df['daily_ret'].shift(1)
            # Night: 週五夜盤 = 記在週一 (Current Row) -> No shift
            pass

        df = df.sort_values(by='gap').reset_index(drop=True)
        df['demeaned_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
        df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        df['cum_demeaned_daily_ret_a'] = df['demeaned_daily_ret_a'].cumsum()
        df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
        df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
        df['cum_daily_ret'] = df['daily_ret'].cumsum()


        period = 'after_holiday' if after_holiday else 'before_holiday'
        return plot.plot(df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='gap', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'], title=f'gap_days_{period}')

    # =========================================================================
    # Factor diagnostics: margin and options
    # =========================================================================
    def indicator_maintenance_rate(self, point_version: bool = False, *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        if 'TotalExchangeMarginMaintenance' not in self.df.columns:
            raise ValueError("TotalExchangeMarginMaintenance is not in the DataFrame.")
        temp_df = self.df.copy()
        temp_df['TotalExchangeMarginMaintenance'] = temp_df['TotalExchangeMarginMaintenance'].shift(1)
        result = self._handle_indicator_output(temp_df['TotalExchangeMarginMaintenance'], name='maintenance_rate', return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
        if result is not None:
            return result
        if point_version:
            temp_df['daily_ret_a'] = (temp_df['Close_a'] - temp_df['Open_a'])
            temp_df['daily_ret'] = (temp_df['Close'] - temp_df['Open'])
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        temp_df = temp_df.sort_values(by='TotalExchangeMarginMaintenance').reset_index(drop=True)
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()

        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='TotalExchangeMarginMaintenance', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'], title='margin_maintenance_rate')

    def indicator_margin_utilization(self, *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        """Analyze the prior day's cross-sectional mean margin utilization."""
        column = 'MarginUtilization'
        if column not in self.df.columns:
            raise ValueError(f"{column} is not in the DataFrame.")

        temp_df = self.df.copy()
        temp_df[column] = temp_df[column].shift(1)
        result = self._handle_indicator_output(
            temp_df[column],
            name='margin_utilization',
            return_series=return_series,
            add_to_df=add_to_df,
            percentile=percentile,
            side=side,
        )
        if result is not None:
            return result

        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        temp_df = temp_df.sort_values(by=column).reset_index(drop=True)
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        return plot.plot(
            temp_df,
            ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'],
            ry=column,
            sub_ly=['cum_daily_ret_a', 'cum_daily_ret'],
            title='margin_utilization',
        )

    def indicator_option_iv(self, trading_session: str = 'day', *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        if {'SkewSlope', 'SkewSlope_a'} - set(self.df.columns):
            from cloud_data import TW_OPTIONS_SETTLE_TXO, TW_OPTIONS_TXO, read_frame
            from module.options.option_tools import compute_iv
            self.add_option_iv_skew(read_frame(TW_OPTIONS_TXO), read_frame(TW_OPTIONS_SETTLE_TXO), iv_calculator=compute_iv)
        df = self.df.copy()
        if trading_session == 'day':
            result = self._handle_indicator_output(df['SkewSlope_a'], name='option_iv_day', return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
            if result is not None:
                return result
            df = df.sort_values(by='SkewSlope_a').reset_index(drop=True)
            df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
            df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
            df['cum_daily_ret'] = df['daily_ret'].cumsum()
            return plot.plot(df, ly=['cum_demeaned_daily_ret'], ry='SkewSlope_a', sub_ly=['cum_daily_ret'], title='option_iv_night')
        elif trading_session == 'night':
            df['SkewSlope'] = df['SkewSlope'].shift(1)
            result = self._handle_indicator_output(df['SkewSlope'], name='option_iv_night', return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
            if result is not None:
                return result
            df = df.sort_values(by='SkewSlope').reset_index(drop=True)
            df['demeaned_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
            df['cum_demeaned_daily_ret_a'] = df['demeaned_daily_ret_a'].cumsum()
            df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
            return plot.plot(df, ly=['cum_demeaned_daily_ret_a'], ry='SkewSlope', sub_ly=['cum_daily_ret_a'], title='option_iv_day')
    
    def indicator_opt_position(self, indicator: str = 'foreign_opt_pos_divergence', trading_session: str = 'day', window: int = 1, time_series_analysis: bool = False, *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        # foreign_opt_pos_divergence, Dealer_Opt_Signal
        # signal 代表，每一塊錢中，有多少做多(> 0) / 做空(< 0)
        if {indicator, f'{indicator}_a'} - set(self.df.columns):
            from cloud_data import TW_OPTIONS_INSTITUTION_DAY, TW_OPTIONS_INSTITUTION_NIGHT, read_frame
            self.add_option_signals(read_frame(TW_OPTIONS_INSTITUTION_DAY), read_frame(TW_OPTIONS_INSTITUTION_NIGHT))
        df = self.df.copy()
        if trading_session == 'day':
            raw = df[f'{indicator}_a']
            factor = (raw / raw.rolling(window).mean()) - 1 if window > 1 else raw
            factor_name = f'{indicator}_day' if window == 1 else f'{indicator}_day_div{window}'
            result = self._handle_indicator_output(factor, name=factor_name, return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
            if result is not None:
                return result
            df[factor_name] = factor
            df = df.sort_values(by=factor_name).reset_index(drop=True)
            df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
            df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
            df['cum_daily_ret'] = df['daily_ret'].cumsum()
            return plot.plot(df, ly=['cum_demeaned_daily_ret'], ry=factor_name, sub_ly=['cum_daily_ret'], title=factor_name)
        elif trading_session == 'night':
            df['pos_continue'] = df[indicator] + df[f'{indicator}_a'] + df[f'{indicator}'].shift(1)
            df['pos_continue'] = df['pos_continue'].shift(1)
            result = self._handle_indicator_output(df['pos_continue'], name=f'{indicator}_night', return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
            if result is not None:
                return result
            if time_series_analysis:
                df['signal'] = df['pos_continue'] > 0.012
                df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
                return plot.plot(df, ly=['cum_daily_ret_a'], ry='signal', ry_dashed=False, title=f'option_position_night_{indicator}_time_series')
            df = df.sort_values(by='pos_continue').reset_index(drop=True)
            df['demeaned_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
            df['cum_demeaned_daily_ret_a'] = df['demeaned_daily_ret_a'].cumsum()
            df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
            return plot.plot(df, ly=['cum_demeaned_daily_ret_a'], ry='pos_continue', sub_ly=['cum_daily_ret_a'], title=f'option_position_night_{indicator}')

    # =========================================================================
    # Factor diagnostics: volatility, flows, and MA divergence
    # =========================================================================
    def check_volatility(self, window: int = 20):
        """
        事件變數 + 事件前 window 天波動度分佈（PDF）
        pos_continue_t = foreign_opt_pos_divergence_t + foreign_opt_pos_divergence_a_t + foreign_opt_pos_divergence_{t-1}
        sig_lag_t      = pos_continue_{t-1}
        event_t        = (sig_lag_t >= 0.012)
        """
        df = self.df.copy()
        ret_col = 'daily_ret_a'

        # 訊號與事件
        df['pos_continue'] = df['foreign_opt_pos_divergence'] + df['foreign_opt_pos_divergence_a'] + df['foreign_opt_pos_divergence'].shift(1)
        df['sig_lag'] = df['pos_continue'].shift(1)
        th = 0.012
        df['event'] = (df['sig_lag'] >= th)

        # 波動：事件日前 window 天（不偷看）
        df['vol'] = df[ret_col].rolling(window).std().shift(1)

        # 事件統計（頻率與 run-length）
        event_rate = df['event'].mean()
        runs = (df['event'] != df['event'].shift()).cumsum()
        run_lengths = df.groupby(runs)['event'].agg(['first', 'size'])
        event_runs = run_lengths[run_lengths['first'] == True]['size']
        avg_run = event_runs.mean() if not event_runs.empty else 0
        max_run = event_runs.max() if not event_runs.empty else 0
        first_20 = df.index[df['event']].to_series().head(20)
        print(f"[event] rate: {event_rate:.4f}, avg_run: {avg_run}, max_run: {max_run}")
        if not first_20.empty:
            print("[event] first 20 dates:")
            print(first_20)

        # Sanity check：確認 lag 與報酬無前視
        sample = df[['foreign_opt_pos_divergence', 'foreign_opt_pos_divergence_a', 'pos_continue', 'sig_lag', 'event', ret_col]].head(5)
        print("[sanity] sample (check shifts):")
        print(sample)

        sig_vol = df.loc[df['event'] == True, 'vol'].dropna()
        nonsig_vol = df.loc[df['event'] == False, 'vol'].dropna()

        if sig_vol.empty or nonsig_vol.empty:
            print("[warn] empty group")
            return None

        # PDF
        plot.plot_pdf(sig_vol.to_frame('vol'), col="vol", title=f"event vol (prev {window}d)")
        plot.plot_pdf(nonsig_vol.to_frame('vol'), col="vol", title=f"non-event vol (prev {window}d)")

        # 上漲/下跌機率
        res_rows = []
        for label, mask in [('event', df['event'] == True), ('non_event', df['event'] == False)]:
            sub = df.loc[mask, [ret_col]].dropna()
            if sub.empty:
                p_pos = p_neg = np.nan
                n = 0
                mean_neg = mean_pos = mean_abs_neg = mean_abs_pos = np.nan
            else:
                n = len(sub)
                p_pos = (sub[ret_col] > 0).mean()
                p_neg = (sub[ret_col] < 0).mean()
                neg = sub.loc[sub[ret_col] < 0, ret_col]
                pos = sub.loc[sub[ret_col] > 0, ret_col]
                mean_neg = neg.mean() if not neg.empty else np.nan
                mean_pos = pos.mean() if not pos.empty else np.nan
                mean_abs_neg = neg.abs().mean() if not neg.empty else np.nan
                mean_abs_pos = pos.abs().mean() if not pos.empty else np.nan
            res_rows.append({
                'group': label,
                'p_pos': p_pos, 'p_neg': p_neg, 'n': n,
                'mean_neg': mean_neg, 'mean_pos': mean_pos,
                'mean_abs_neg': mean_abs_neg, 'mean_abs_pos': mean_abs_pos,
            })
        freq_df = pd.DataFrame(res_rows)
        print("=== Up/Down Probability & Magnitude by Event Group ===")
        print(freq_df)

        # 左尾風險：多個門檻的 tail probability
        thresholds = [0.005, 0.01, 0.015]  # 0.5%, 1%, 1.5%（可依需要調整）
        tail_rows = []
        for x in thresholds:
            p_evt = (df.loc[df['event'] == True, ret_col] <= -x).mean()
            p_none = (df.loc[df['event'] == False, ret_col] <= -x).mean()
            tail_rows.append({
                'threshold_x': x,
                'p_event': p_evt,
                'p_nonevent': p_none,
                'diff': p_evt - p_none,
                'ratio': (p_evt / p_none) if p_none not in [0, np.nan] else np.nan,
            })
        tail_df = pd.DataFrame(tail_rows)
        print("=== Tail Probability (ret <= -x) by Event Group ===")
        print(tail_df)

        # 分位數比較（左尾更穩健）
        quants = [0.01, 0.05, 0.10]
        q_rows = []
        for qv in quants:
            evt_val = df.loc[df['event'] == True, ret_col].quantile(qv)
            none_val = df.loc[df['event'] == False, ret_col].quantile(qv)
            q_rows.append({
                'quantile': qv,
                'event_value': evt_val,
                'non_event_value': none_val,
                'diff': evt_val - none_val,
            })
        q_df = pd.DataFrame(q_rows)
        print("=== Quantile Comparison (event vs non_event) ===")
        print(q_df)

        # 連跌（二連跌機率），條件用 event_t
        res_pairs = []
        evt_mask = df['event'] == True
        none_mask = df['event'] == False
        for label, mask in [('event', evt_mask), ('non_event', none_mask)]:
            sub = df.loc[mask, [ret_col]].dropna()
            # 將當期與下一期配對
            pair_down = (sub[ret_col] < 0) & (sub[ret_col].shift(-1) < 0)
            n_pairs = pair_down.notna().sum() - 1  # 有效配對數
            p_2down = pair_down.mean()
            res_pairs.append({'group': label, 'p_2down': p_2down, 'n_pairs': n_pairs})
        pairs_df = pd.DataFrame(res_pairs)
        print("=== Two-day Down Probability (condition on event_t) ===")
        print(pairs_df)

        # conditional downside/upside vol within rolling window
        res_cond = []
        for label, mask in [('event', df['event'] == True), ('non_event', df['event'] == False)]:
            sub = df.loc[mask, ret_col]
            if sub.empty:
                res_cond.append({'group': label, 'downside_std': np.nan, 'upside_std': np.nan, 'ratio': np.nan})
                continue
            down_series = sub.rolling(window).apply(lambda x: x[x < 0].std() if (x < 0).any() else np.nan, raw=False)
            up_series = sub.rolling(window).apply(lambda x: x[x > 0].std() if (x > 0).any() else np.nan, raw=False)
            down_mean = down_series.dropna().mean()
            up_mean = up_series.dropna().mean()
            ratio = (down_mean / up_mean) if up_mean not in [0, np.nan] else np.nan
            res_cond.append({
                'group': label,
                'downside_std': down_mean,
                'upside_std': up_mean,
                'ratio': ratio,
            })
        cond_df = pd.DataFrame(res_cond)
        print("=== Conditional Downside/Upstate Vol (rolling window) ===")
        print(cond_df)

    def indicator_otc_margin_growth(self, *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        """Build and inspect prior-day OTC four-digit-stock margin-balance growth."""
        from cloud_data import TW_STOCK_OTC_MARGIN_BALANCE, read_frame

        start = self.df.index.min() - pd.Timedelta(days=31)
        end = self.df.index.max()
        otc_margin = read_frame(TW_STOCK_OTC_MARGIN_BALANCE, start=start, end=end)
        otc_margin = otc_margin.loc[otc_margin['代號'].str.fullmatch(r'\d{4}', na=False)]
        otc_margin_growth = (
            (otc_margin.groupby('date')['資餘額'].sum()
             / otc_margin.groupby('date')['前資餘額(張)'].sum()) - 1
        ).rename('otc_margin_growth')
        self.merge_features(otc_margin_growth.shift(1).to_frame())

        temp_df = self.df.copy()
        result = self._handle_indicator_output(
            temp_df['otc_margin_growth'],
            name='otc_margin_growth',
            return_series=return_series,
            add_to_df=add_to_df,
            percentile=percentile,
            side=side,
        )
        if result is not None:
            return result
        temp_df = temp_df.dropna(subset=['otc_margin_growth']).sort_values(by='otc_margin_growth').reset_index(drop=True)
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        temp_df['cum_ret'] = (temp_df['daily_ret_a'] + temp_df['daily_ret']).cumsum()
        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='otc_margin_growth', sub_ly=['cum_daily_ret_a', 'cum_daily_ret', 'cum_ret'], title='otc_margin_growth')

    def indicator_otc_margin_growth_divergence(
        self,
        window: int = 20,
        *,
        return_series: bool = False,
        add_to_df: bool = False,
        percentile: float | None = None,
        side: str = 'low',
    ):
        """Analyze OTC margin-growth deviation from its trailing average.

        The factor is the daily aggregate OTC margin-balance growth minus its
        trailing ``window``-day mean. It is lagged one trading day before being
        aligned with TX returns, so the current session never uses same-day
        margin data.
        """
        if window < 2:
            raise ValueError('window must be at least 2')
        from cloud_data import TW_STOCK_OTC_MARGIN_BALANCE, read_frame

        start = self.df.index.min() - pd.Timedelta(days=31)
        end = self.df.index.max()
        otc_margin = read_frame(TW_STOCK_OTC_MARGIN_BALANCE, start=start, end=end)
        otc_margin = otc_margin.loc[otc_margin['代號'].str.fullmatch(r'\d{4}', na=False)]
        otc_margin_growth = (
            (otc_margin.groupby('date')['資餘額'].sum()
             / otc_margin.groupby('date')['前資餘額(張)'].sum()) - 1
        ).rename('otc_margin_growth')
        factor_name = f'otc_margin_growth_divergence_{window}'
        divergence = (
            otc_margin_growth - otc_margin_growth.rolling(window=window).mean()
        ).shift(1).rename(factor_name)

        temp_df = self.df.copy()
        temp_df[factor_name] = divergence
        result = self._handle_indicator_output(
            temp_df[factor_name],
            name=factor_name,
            return_series=return_series,
            add_to_df=add_to_df,
            percentile=percentile,
            side=side,
        )
        if result is not None:
            return result
        temp_df = temp_df.dropna(subset=[factor_name]).sort_values(by=factor_name).reset_index(drop=True)
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        temp_df['cum_ret'] = (temp_df['daily_ret_a'] + temp_df['daily_ret']).cumsum()
        return plot.plot(
            temp_df,
            ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'],
            ry=factor_name,
            sub_ly=['cum_daily_ret_a', 'cum_daily_ret', 'cum_ret'],
            title=factor_name,
        )        

    def indicator_institutional_flow(self, *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        temp_df = self.df.copy()
        temp_df['foreign_inflow'] = (temp_df['Net_Foreign_Investor'] - temp_df['Net_Foreign_Investor'].rolling(window=20).mean()) / temp_df['Net_Foreign_Investor'].rolling(window=20).std()
        temp_df['foreign_inflow'] = temp_df['foreign_inflow'].shift(1)
        result = self._handle_indicator_output(temp_df['foreign_inflow'], name='institutional_flow', return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
        if result is not None:
            return result
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        temp_df = temp_df.sort_values(by='foreign_inflow').reset_index(drop=True)
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        temp_df['cum_ret'] = (temp_df['daily_ret_a'] + temp_df['daily_ret']).cumsum()
        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='foreign_inflow', sub_ly=['cum_daily_ret_a', 'cum_daily_ret', 'cum_ret'])

    def indicator_ma_divergence(
        self,
        window: int = 15,
        *,
        percentile: float | None = None,
        side: str = 'low',
        volatility_window: int | None = None,
        volatility_regime: int | None = None,
        volatility_bins: int = 5,
        return_series: bool = False,
        add_to_df: bool = False,
    ):
        """Plot MA divergence, return its percentile threshold, or return its series.

        When ``percentile`` is provided, the method returns a float instead of
        plotting. Optionally pass a historical-volatility window and regime to
        calculate the divergence threshold only inside that volatility group.
        """
        if window < 2:
            raise ValueError('window must be at least 2')
        has_volatility_condition = volatility_window is not None or volatility_regime is not None
        if has_volatility_condition and (volatility_window is None or volatility_regime is None):
            raise ValueError('volatility_window and volatility_regime must be provided together')
        if volatility_window is not None and volatility_window < 2:
            raise ValueError('volatility_window must be at least 2')
        if volatility_regime is not None and not 1 <= volatility_regime <= volatility_bins:
            raise ValueError('volatility_regime must be between 1 and volatility_bins')
        if volatility_bins < 2:
            raise ValueError('volatility_bins must be at least 2')
        if (return_series or add_to_df) and percentile is not None:
            raise ValueError('return_series or add_to_df cannot be combined with percentile')

        temp_df = self.df.copy()
        temp_df[f'{window}_ma'] = temp_df['Close'].rolling(window=window).mean()
        temp_df['divergence'] = (temp_df['Close'] / temp_df[f'{window}_ma']) - 1
        temp_df['divergence'] = temp_df['divergence'].shift(1)
        temp_df = temp_df.dropna(subset=['divergence'])

        if return_series or add_to_df:
            return self._handle_indicator_output(
                temp_df['divergence'],
                name=f'ma_divergence_{window}',
                return_series=return_series,
                add_to_df=add_to_df,
                percentile=None,
                side=side,
            )

        if percentile is not None:
            divergence = temp_df['divergence']
            if has_volatility_condition:
                temp_df['hist_vol'] = self.df['daily_ret'].rolling(volatility_window).std().shift(1)
                temp_df = temp_df.dropna(subset=['hist_vol'])
                try:
                    temp_df['volatility_regime'] = pd.qcut(
                        temp_df['hist_vol'],
                        q=volatility_bins,
                        labels=False,
                        duplicates='raise',
                    ) + 1
                except ValueError as error:
                    raise ValueError('volatility data cannot be split into the requested bins') from error
                divergence = temp_df.loc[
                    temp_df['volatility_regime'].eq(volatility_regime),
                    'divergence',
                ]
            return self._percentile_value(divergence, percentile, side)

        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()

        temp_df = temp_df.sort_values(by='divergence').reset_index(drop=True)
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='divergence', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'], title=f'{window}ma_divergence')

    # Kept for existing notebook cells written before the indicator was renamed.
    indicator_15ma_divergence = indicator_ma_divergence

    def indicator_hist_vol(
        self,
        window: int = 20,
        *,
        percentile: float | None = None,
        side: str = 'low',
        return_series: bool = False,
        add_to_df: bool = False,
    ):
        """Plot historical volatility, return a percentile threshold, or return its series."""
        if window < 2:
            raise ValueError('window must be at least 2')
        temp_df = self.df.copy()
        temp_df['hist_vol'] = temp_df['daily_ret'].rolling(window=window).std()
        temp_df['hist_vol'] = temp_df['hist_vol'].shift(1)
        temp_df = temp_df.dropna(subset=['hist_vol'])
        result = self._handle_indicator_output(
            temp_df['hist_vol'],
            name=f'hist_vol_{window}',
            return_series=return_series,
            add_to_df=add_to_df,
            percentile=percentile,
            side=side,
        )
        if result is not None:
            return result

        temp_df = self._prepare_return_curves(temp_df, sort_by='hist_vol')
        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='hist_vol', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'], title=f'{window}d_hist_vol')

    def indicator_HL_vol(
        self,
        window: int = 20,
        *,
        percentile: float | None = None,
        side: str = 'low',
        return_series: bool = False,
        add_to_df: bool = False,
    ):
        if window < 2:
            raise ValueError('window must be at least 2')
        temp_df = self.df.copy()
        temp_df['HL_vol'] = temp_df['High'] / temp_df['Low'] - 1
        temp_df['HL_vol_ma'] = temp_df['HL_vol'].rolling(window=window).mean()
        temp_df['HL_vol'] = temp_df['HL_vol'] - temp_df['HL_vol_ma']
        temp_df['HL_vol'] = temp_df['HL_vol'].shift(1)
        temp_df = temp_df.dropna(subset=['HL_vol'])
        result = self._handle_indicator_output(
            temp_df['HL_vol'],
            name=f'HL_vol_{window}',
            return_series=return_series,
            add_to_df=add_to_df,
            percentile=percentile,
            side=side,
        )
        if result is not None:
            return result

        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()

        temp_df = temp_df.sort_values(by='HL_vol').reset_index(drop=True)
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='HL_vol', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'], title=f'{window}d_HL_vol')

    # =========================================================================
    # Factor diagnostics: price structure and macro markets
    # =========================================================================
    def indicator_night_ret(
        self,
        window: int = 3,
        *,
        return_series: bool = False,
        add_to_df: bool = False,
        percentile: float | None = None,
        side: str = 'low',
    ):
        """Plot or return night-session close divergence from its trailing MA."""
        if window < 2:
            raise ValueError('window must be at least 2')
        df = self.df.copy()
        factor_name = 'night_ret_divergence' if window == 3 else f'night_ret_divergence_{window}'
        df['night_ret_ma'] = df['Close_a'].rolling(window).mean()
        df['divergence'] = (df['Close_a'] / df['night_ret_ma']) - 1
        df = df.dropna(subset=['divergence'])
        result = self._handle_indicator_output(df['divergence'], name=factor_name, return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
        if result is not None:
            return result

        df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        df = df.sort_values(by='divergence').reset_index(drop=True)
        df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
        df['cum_daily_ret'] = df['daily_ret'].cumsum()
        return plot.plot(df, ly=['cum_demeaned_daily_ret'], ry='divergence', sub_ly=['cum_daily_ret'], title='night_ret')

    def indicator_night_ret_divergence(
        self,
        window: int = 3,
        *,
        return_series: bool = False,
        add_to_df: bool = False,
        percentile: float | None = None,
        side: str = 'low',
    ):
        """Measure night-session return relative to its trailing mean return.

        ``daily_ret_a`` is the night-session open-to-close return.  Under the
        analyzer's close-date convention, the night session labelled ``t``
        ends before day session ``t`` opens, so this factor needs no extra
        shift when it predicts ``daily_ret[t]``.
        """
        if window < 2:
            raise ValueError('window must be at least 2')

        df = self.df.copy()
        factor_name = f'night_ret_divergence_{window}'
        df[factor_name] = df['daily_ret_a'] - df['daily_ret_a'].rolling(window).mean()
        result = self._handle_indicator_output(
            df[factor_name],
            name=factor_name,
            return_series=return_series,
            add_to_df=add_to_df,
            percentile=percentile,
            side=side,
        )
        if result is not None:
            return result

        df = df.dropna(subset=[factor_name]).sort_values(by=factor_name).reset_index(drop=True)
        df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
        df['cum_daily_ret'] = df['daily_ret'].cumsum()
        return plot.plot(
            df,
            ly=['cum_demeaned_daily_ret'],
            ry=factor_name,
            sub_ly=['cum_daily_ret'],
            title=factor_name,
        )

    def indicator_spread(self, window: int = 5, *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        df = self.df.copy()
        df.dropna(subset=['spread_a'], inplace=True)
        df['next_ret'] = (df['Close_a'].shift(-window) - df['Close_a']) / df['Close_a']
        df['sum_spread'] = df['spread_a'].rolling(window=window).sum()
        df['sum_spread'] = df['sum_spread'].shift(1)
        result = self._handle_indicator_output(df['sum_spread'], name=f'spread_{window}', return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
        if result is not None:
            return result
        # df.dropna(subset=['sum_spread'], inplace=True)
        df.sort_values(by='sum_spread', ignore_index=True, inplace=True)
        df['demeaned_daily_ret_a'] = df['next_ret'] - df['next_ret'].mean()
        df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        df['cum_demeaned_daily_ret_a'] = df['demeaned_daily_ret_a'].cumsum()
        df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
        df['cum_daily_ret_a'] = df['next_ret'].cumsum()
        return plot.plot(df, ly=['cum_demeaned_daily_ret_a'], ry='sum_spread', sub_ly=['cum_daily_ret_a'])

    def indicator_weekday_stats(self, *, return_series: bool = False, add_to_df: bool = False):
        """
        統計並繪製每週各交易日 (Mon-Fri) 的平均報酬率
        """
        import plotly.express as px
        
        df = self.df.copy()
        df['weekday'] = df.index.weekday
        if return_series or add_to_df:
            series = df['weekday'].rename('weekday')
            if add_to_df:
                self.merge_features(series.to_frame())
            return series
        
        # Group by weekday
        weekday_stats = df.groupby('weekday')[['daily_ret', 'daily_ret_a']].mean()
        
        # Rename index for better readability
        weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        weekday_stats.index = weekday_stats.index.map(weekday_map)
        
        # Plot
        fig = px.bar(
            weekday_stats, 
            barmode='group',
            title='Average Return by Weekday',
            labels={'value': 'Avg Return', 'index': 'Weekday', 'variable': 'Session'}
        )
        fig.show()

    def indicator_US_bond(self, indicator: str, *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        import numpy as np
        temp_df = self.df.copy()
        ind_list = [
            'yield_shock', 'yield_divergence', 'yield_presure',
            'cash_crunch', 'near_inversion', 'near_yield_vol'
            ]

        ffill_col = ['US_bond_5y']
        for col in ffill_col:
            if temp_df[col].isna().sum() > 0:
                temp_df[col] = temp_df[col].ffill()

        # 長債
        temp_df['yield_shock'] = temp_df['US_bond_5y'] - temp_df['US_bond_5y'].shift(20)

        temp_df['yield_shock'] = temp_df['yield_shock'].shift(3)

        if indicator not in ind_list:
            raise ValueError(f'unsupported US bond indicator: {indicator}')
        result = self._handle_indicator_output(temp_df[indicator], name=f'us_bond_{indicator}', return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
        if result is not None:
            return result

        temp_df['demean_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demean_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        temp_df['daily_ret_a'] = temp_df['daily_ret_a']
        temp_df['daily_ret'] = temp_df['daily_ret']

        temp_df = temp_df.sort_values(by=indicator).reset_index(drop=True)
        temp_df['cum_demean_daily_ret_a'] = temp_df['demean_daily_ret_a'].cumsum()
        temp_df['cum_demean_daily_ret'] = temp_df['demean_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        return plot.plot(temp_df, ly=['cum_demean_daily_ret_a', 'cum_demean_daily_ret'], ry=indicator, sub_ly=['cum_daily_ret_a', 'cum_daily_ret'], title=f'us_bond_{indicator}')

    def indicator_structural_weakness(self, *, return_series: bool = False, add_to_df: bool = False):
        """
        分析「市場結構轉弱」(Structural Weakness) 對績效的影響
        
        指標定義：
        1. 收盤位置 (CLV): (Close - Low) / (High - Low)
           - CLV < 0.4 代表收盤無力 (收在下半部)
        2. 反彈失敗 (Bounce Failure): 
           - T日上漲 (Ret > 0)
           - 但 T+1日收盤 < T-1日收盤 (漲勢曇花一現，馬上被吞噬)
           
        濾網條件 (Filter): 過去 5 天內
        - CLV < 0.4 的天數 >= 3
        - 下跌天數 (Ret < 0) >= 3
        - 發生過反彈失敗 >= 1 (Optional)
        
        驗證：當濾網觸發 (Signal ON) 時，後續的績效表現是否顯著較差？
        """
        df = self.df.copy()
        df['weakness'] = (df['Close_a'] < df['Close_a'].shift(2))
        if return_series or add_to_df:
            series = df['weakness'].rename('structural_weakness')
            if add_to_df:
                self.merge_features(series.to_frame())
            return series
        df['pos_night'] = np.where(
            df['weakness'].shift(1),
            0,
            1
        )
        df['strat_ret'] = df['daily_ret_a'] * df['pos_night']
        df['cum_strat_ret'] = df['strat_ret'].cumsum()
        df['cum_bnh_ret'] = df['daily_ret_a'].cumsum()
        return plot.plot(df, ly=['cum_strat_ret', 'cum_bnh_ret'])

    def indicator_fear_greed(self, trading_session: str, time_series_analysis: bool = False, *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        if 'fear_greed' not in self.df:
            from cloud_data import MARKET_FEAR_GREED, read_frame
            self.add_fear_greed(read_frame(MARKET_FEAR_GREED), pd.DataFrame())
        df = self.df.copy()

        if trading_session == 'night':
            df['fear_greed'] = df['fear_greed'].shift(1)
        elif trading_session == 'day':
            df['fear_greed'] = df['fear_greed']

        df['delta_fear_greed'] = df['fear_greed'] - df['fear_greed'].shift(1)
        result = self._handle_indicator_output(df['delta_fear_greed'], name=f'fear_greed_{trading_session}', return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
        if result is not None:
            return result
        df.dropna(subset=['daily_ret_a'], inplace=True)
        df['demean_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
        df['demean_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        df = df.sort_values(by='delta_fear_greed').reset_index(drop=False)
        df['cum_demean_daily_ret_a'] = df['demean_daily_ret_a'].cumsum()
        df['cum_demean_daily_ret'] = df['demean_daily_ret'].cumsum()
        df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
        df['cum_daily_ret'] = df['daily_ret'].cumsum()
        if trading_session == 'night':
            return plot.plot(df, ly=['cum_demean_daily_ret_a'], ry='delta_fear_greed', sub_ly=['cum_daily_ret_a'], title='fear_greed_night')
        elif trading_session == 'day':
            return plot.plot(df, ly=['cum_demean_daily_ret'], ry='delta_fear_greed', sub_ly=['cum_daily_ret'], title='fear_greed_day')

    def indicator_move(self, trading_session: str, *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        if 'MOVE_open' not in self.df:
            from cloud_data import MARKET_MOVE_DAILY, read_frame
            self.add_market_ohlc(read_frame(MARKET_MOVE_DAILY), 'MOVE')
        df = self.df.copy()
        # ====== 計算指標 ======


        df['3_ma'] = df['Close_a'].rolling(window=3).mean()
        df['divergence'] = (df['Close_a'] / df['3_ma']) - 1

        df['ind'] = (df['MOVE_close'] / df['MOVE_open']) - 1
        df['MOVE_ma'] = 1
        df['MOVE_divergence'] = (df['MOVE_close'] / df['MOVE_ma']) - 1

        df['gap'] = (df['Close_a'] / df['Close'].shift(1)) - 1

        if trading_session == 'night':
            factor = ((df['MOVE_high'] / df['MOVE_low']) - 1).shift(1)
            factor_name = 'move_vol_night'
        elif trading_session == 'day':
            factor = df['ind']
            factor_name = 'move_return_day'
        else:
            raise ValueError("trading_session must be either 'day' or 'night'")
        result = self._handle_indicator_output(factor, name=factor_name, return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
        if result is not None:
            return result

        df['demean_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
        df['demean_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()

        if trading_session == 'night':
            df['MOVE_vol'] = (df['MOVE_high'] / df['MOVE_low']) - 1
            df['MOVE_vol'] = df['MOVE_vol'].shift(1)
            df['MOVE_divergence'] = df['MOVE_divergence'].shift(1)

            df['ind'] = df['ind'].shift(1)
            df = df.sort_values(by='MOVE_vol').reset_index(drop=True)
            df['cum_demean_daily_ret_a'] = df['demean_daily_ret_a'].cumsum()
            df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
            return plot.plot(df, ly=['cum_demean_daily_ret_a'], ry='MOVE_vol', sub_ly=['cum_daily_ret_a'], title='move_night')

        elif trading_session == 'day':
            df = df.sort_values(by='ind').reset_index(drop=True)
            df['cum_demean_daily_ret'] = df['demean_daily_ret'].cumsum()
            df['cum_daily_ret'] = df['daily_ret'].cumsum()
            return plot.plot(df, ly=['cum_demean_daily_ret'], ry='ind', sub_ly=['cum_daily_ret'], title='move_day')

    def indicator_sox(self, trading_session: str, *, return_series: bool = False, add_to_df: bool = False, percentile: float | None = None, side: str = 'low'):
        if 'SOX_open' not in self.df:
            from cloud_data import MARKET_SOX_DAILY, read_frame
            self.add_market_ohlc(read_frame(MARKET_SOX_DAILY), 'SOX')
        df = self.df.copy()

        df['ind'] = (df['SOX_close'] / df['SOX_open']) - 1
        # df['ind'] = df['ind'].rolling(window=3).mean()
        df['ind'] = df['ind'].shift(1)
        # df['ind'] = (df['SOX_high'] - df['SOX_low']) / df['SOX_close'].shift(1)
        df = df.dropna(subset='ind')

        factor = df['ind'].shift(1) if trading_session == 'night' else df['ind']
        result = self._handle_indicator_output(factor, name=f'sox_{trading_session}', return_series=return_series, add_to_df=add_to_df, percentile=percentile, side=side)
        if result is not None:
            return result

        df['gap'] = (df['Open'] / df['Close'].shift(1)) - 1

        df['demean_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
        df['demean_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        # df.dropna(subset=['ind'], inplace=True)
        if trading_session == 'night':
            df['ind'] = df['ind'].shift(1)
            df = df.sort_values(by='ind').reset_index(drop=True)
            df['cum_demean_daily_ret_a'] = df['demean_daily_ret_a'].cumsum()
            df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
            return plot.plot(df, ly=['cum_demean_daily_ret_a'], ry='ind', sub_ly=['cum_daily_ret_a'], title='sox_night')
        elif trading_session == 'day':
            df = df.sort_values(by='ind').reset_index(drop=True)
            df['cum_demean_daily_ret'] = df['demean_daily_ret'].cumsum()
            df['cum_daily_ret'] = df['daily_ret'].cumsum()
            mean_l = df.loc[df['ind'] < 0.0025, 'daily_ret'].mean()
            mean_r = df.loc[df['ind'] >= 0.0025, 'daily_ret'].mean()
            print(f"SOX ind threshold=0.0025\nmean_l={mean_l:.6f} | mean_r={mean_r:.6f}")
            return plot.plot(df, ly=['cum_demean_daily_ret'], ry='ind', sub_ly=['cum_daily_ret'], title='sox_day')

    # =========================================================================
    # Factor research tools: conditional sorts, signal timelines, and overlap
    # =========================================================================
    @staticmethod
    def fit_factor_thresholds(
        values: pd.Series,
        percentile_range: float | tuple[float, float],
        *,
        condition: pd.Series | None = None,
        condition_percentile_range: float | tuple[float, float] | None = None,
    ) -> pd.Series:
        """Fit fixed raw factor cutoffs on a training sample.

        A scalar percentile means the left-tail interval ``(0, p)``; a tuple
        selects an explicit interval. When ``condition`` and
        ``condition_percentile_range`` are supplied, first select the
        condition interval and then fit the factor cutoffs within it. All
        inputs should be restricted to the training sample before calling.
        """
        def normalize_range(
            raw_range: float | tuple[float, float],
            *,
            parameter_name: str,
        ) -> tuple[float, float]:
            if isinstance(raw_range, tuple):
                lower, upper = map(float, raw_range)
            else:
                lower, upper = 0.0, float(raw_range)
            if not 0 <= lower < upper <= 100:
                raise ValueError(f'{parameter_name} must satisfy 0 <= lower < upper <= 100')
            return lower, upper

        lower_pct, upper_pct = normalize_range(percentile_range, parameter_name='percentile_range')
        if condition is None:
            if condition_percentile_range is not None:
                raise ValueError('condition_percentile_range requires condition')
            selected = values.dropna()
            condition_thresholds = {}
            observations = len(selected)
        else:
            if condition_percentile_range is None:
                raise ValueError('condition requires condition_percentile_range')
            condition_lower_pct, condition_upper_pct = normalize_range(
                condition_percentile_range,
                parameter_name='condition_percentile_range',
            )
            frame = pd.concat({'factor': values, 'condition': condition}, axis=1).dropna()
            if frame.empty:
                raise ValueError('no overlapping non-null observations are available')
            condition_lower = float(frame['condition'].quantile(condition_lower_pct / 100))
            condition_upper = float(frame['condition'].quantile(condition_upper_pct / 100))
            selected = frame.loc[
                frame['condition'].between(condition_lower, condition_upper, inclusive='both'),
                'factor',
            ]
            if selected.empty:
                raise ValueError('the condition percentile range selected no observations')
            condition_thresholds = {
                'condition_lower_percentile': condition_lower_pct,
                'condition_upper_percentile': condition_upper_pct,
                'condition_lower_cutoff': condition_lower,
                'condition_upper_cutoff': condition_upper,
                'observations_before_condition': len(frame),
            }
            observations = len(selected)

        return pd.Series({
            'lower_percentile': lower_pct,
            'upper_percentile': upper_pct,
            'lower_cutoff': float(selected.quantile(lower_pct / 100)),
            'upper_cutoff': float(selected.quantile(upper_pct / 100)),
            **condition_thresholds,
            'observations': observations,
        }, name=values.name or 'factor_thresholds')

    @staticmethod
    def threshold_signal(
        values: pd.Series,
        thresholds: pd.Series,
        *,
        condition: pd.Series | None = None,
    ) -> pd.Series:
        """Build a nullable signal from fixed factor and optional condition cutoffs."""
        required = {'lower_cutoff', 'upper_cutoff'}
        if not required.issubset(thresholds.index):
            raise ValueError(f'thresholds must contain: {sorted(required)}')
        signal = values.between(
            thresholds['lower_cutoff'],
            thresholds['upper_cutoff'],
            inclusive='both',
        )
        condition_required = {'condition_lower_cutoff', 'condition_upper_cutoff'}
        if condition_required.issubset(thresholds.index):
            if condition is None:
                raise ValueError('condition is required by these thresholds')
            signal &= condition.between(
                thresholds['condition_lower_cutoff'],
                thresholds['condition_upper_cutoff'],
                inclusive='both',
            )
            return signal.where(values.notna() & condition.notna())
        if condition is not None:
            raise ValueError('condition was provided but thresholds have no condition cutoffs')
        return signal.where(values.notna())

    @staticmethod
    def signal_overlap_diagnostics(
        signals: pd.DataFrame | dict[str, pd.Series] | None = None,
        *,
        factor_configs: list[tuple[str, pd.Series, float | tuple[float, float]]] | None = None,
        training_end: str | pd.Timestamp | None = None,
        forward_returns: pd.Series | None = None,
        evaluation_start: str | pd.Timestamp | None = None,
        evaluation_end: str | pd.Timestamp | None = None,
        lead_lags: tuple[int, ...] = (-1, 0, 1),
        hac_lags: int = 5,
    ) -> dict[str, pd.DataFrame]:
        """Diagnose whether arbitrary threshold signals describe the same state.

        Pass either a date-indexed boolean ``signals`` matrix, or
        ``factor_configs`` as ``(name, factor_series, percentile_range)``
        tuples plus ``training_end``. In the latter form, thresholds are fit
        only on data through ``training_end`` and signals are built internally.
        The returned frames include thresholds, Phi correlations, Jaccard
        overlap, directed conditional trigger rates, daily lead-lag, and
        joint-return regressions.

        ``forward_returns`` must already be aligned to the horizon being
        predicted; it is not shifted inside this helper.
        """
        if (signals is None) == (factor_configs is None):
            raise ValueError('pass exactly one of signals or factor_configs')

        if factor_configs is not None:
            if training_end is None:
                raise ValueError('training_end is required when using factor_configs')
            return TXAnalyzer._factor_overlap_diagnostics(
                factor_configs,
                training_end=training_end,
                forward_returns=forward_returns,
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
                lead_lags=lead_lags,
                hac_lags=hac_lags,
            )

        signal_frame = pd.DataFrame(signals).copy().sort_index()
        if evaluation_start is not None:
            signal_frame = signal_frame.loc[signal_frame.index >= pd.Timestamp(evaluation_start)]
        if evaluation_end is not None:
            signal_frame = signal_frame.loc[signal_frame.index <= pd.Timestamp(evaluation_end)]
        signal_frame = signal_frame.astype('boolean').dropna(how='any').astype(bool)
        return TXAnalyzer._signal_frame_overlap_diagnostics(
            signal_frame,
            forward_returns=forward_returns,
            lead_lags=lead_lags,
            hac_lags=hac_lags,
        )

    @staticmethod
    def export_diagnostics(
        diagnostics: dict[str, pd.DataFrame],
        output_path: str | Path,
    ) -> Path:
        """Export diagnostic DataFrames to one Excel workbook or a CSV directory.

        Pass an ``.xlsx`` path to create one workbook with one sheet per table.
        Pass a directory path (or a ``.csv`` filename) to create one CSV per
        table; a ``.csv`` filename becomes a directory with the same stem.
        """
        tables = {
            str(name): table
            for name, table in diagnostics.items()
            if isinstance(table, pd.DataFrame)
        }
        if not tables:
            raise ValueError('diagnostics contains no DataFrames to export')

        destination = Path(output_path)
        suffix = destination.suffix.lower()
        if suffix == '.xlsx':
            destination.parent.mkdir(parents=True, exist_ok=True)
            used_sheet_names: set[str] = set()
            try:
                with pd.ExcelWriter(destination, engine='openpyxl') as writer:
                    for name, table in tables.items():
                        sheet_name = name.translate(str.maketrans({char: '_' for char in '[]:*?/\\'}))[:31] or 'table'
                        base_name = sheet_name
                        counter = 1
                        while sheet_name in used_sheet_names:
                            suffix_text = f'_{counter}'
                            sheet_name = f'{base_name[:31 - len(suffix_text)]}{suffix_text}'
                            counter += 1
                        used_sheet_names.add(sheet_name)
                        table.to_excel(writer, sheet_name=sheet_name, index=True)
            except ImportError as error:
                raise ImportError(
                    'Excel export requires openpyxl. Install it with: '
                    '/Users/xinc/.conda/envs/quant/bin/python -m pip install openpyxl'
                ) from error
            return destination

        csv_directory = destination.with_suffix('') if suffix == '.csv' else destination
        csv_directory.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            filename = name.translate(str.maketrans({char: '_' for char in '/\\'}))
            table.to_csv(csv_directory / f'{filename}.csv', index=True)
        return csv_directory

    @staticmethod
    def _filter_regression_design(design: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Drop constant, duplicate, and linearly dependent regressors."""
        kept: list[str] = []
        removed = []
        for name in design.columns:
            candidate = design[name]
            if candidate.nunique(dropna=False) <= 1:
                removed.append({'column': name, 'reason': 'constant'})
                continue
            if any(candidate.equals(design[existing]) for existing in kept):
                removed.append({'column': name, 'reason': 'duplicate'})
                continue
            base = np.ones((len(design), 1))
            if kept:
                base = np.column_stack((base, design[kept].to_numpy(dtype=float)))
            candidate_matrix = np.column_stack((base, candidate.to_numpy(dtype=float)))
            if np.linalg.matrix_rank(candidate_matrix) == np.linalg.matrix_rank(base):
                removed.append({'column': name, 'reason': 'linear_dependent'})
                continue
            kept.append(name)
        return design.loc[:, kept], pd.DataFrame(removed, columns=['column', 'reason'])

    @staticmethod
    def _lead_lag_diagnostics(binary: pd.DataFrame, lead_lags: tuple[int, ...]) -> pd.DataFrame:
        """Measure directional binary-signal overlap relative to each lag's base rate."""
        rows = []
        names = list(binary.columns)
        for source_name in names:
            for target_name in names:
                if source_name == target_name:
                    continue
                source = binary[source_name].astype('boolean')
                target = binary[target_name].astype('boolean')
                for lag in lead_lags:
                    # At source date t, target.shift(lag) reads target at t-lag.
                    # Thus +1 means target fired one trading day before source.
                    lagged = pd.DataFrame({
                        'source': source,
                        'target': target.shift(lag),
                    }).dropna(how='any')
                    source_days = int(lagged['source'].sum())
                    target_rate = lagged['target'].mean()
                    conditional = (
                        lagged.loc[lagged['source'], 'target'].mean()
                        if source_days else np.nan
                    )
                    phi = (
                        lagged['source'].astype(int).corr(lagged['target'].astype(int))
                        if lagged['source'].nunique() > 1 and lagged['target'].nunique() > 1
                        else np.nan
                    )
                    rows.append({
                        'source': source_name,
                        'target': target_name,
                        'lag': lag,
                        'lag_meaning': 'target earlier than source' if lag > 0 else ('same day' if lag == 0 else 'target later than source'),
                        'lag_observations': len(lagged),
                        'source_signal_days': source_days,
                        'target_signal_days': int(lagged['target'].sum()),
                        'target_base_rate': target_rate,
                        'p_target_given_source_at_lag': conditional,
                        'conditional_lift': conditional / target_rate if target_rate and not pd.isna(conditional) else np.nan,
                        'phi': phi,
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def _signal_combination_returns(
        signal_frame: pd.DataFrame,
        forward_returns: pd.Series | None,
    ) -> pd.DataFrame:
        """Summarise none/only/pair/all signal states and their aligned returns."""
        labels = signal_frame.apply(
            lambda row: ' + '.join(row.index[row].tolist()) if row.any() else 'none',
            axis=1,
        ).rename('signal_combination')
        summary = labels.value_counts().rename_axis('signal_combination').to_frame('signal_days')
        summary['signal_share'] = summary['signal_days'] / len(labels)
        if forward_returns is None:
            summary[['return_observations', 'mean_return', 'median_return', 'win_rate']] = np.nan
            return summary

        return_frame = pd.concat([labels, forward_returns.rename('forward_return')], axis=1).dropna(how='any')
        if return_frame.empty:
            summary[['return_observations', 'mean_return', 'median_return', 'win_rate']] = np.nan
            return summary
        grouped = return_frame.groupby('signal_combination')['forward_return']
        return_summary = grouped.agg(
            return_observations='count',
            mean_return='mean',
            median_return='median',
            win_rate=lambda values: (values > 0).mean(),
        )
        return summary.join(return_summary)

    @staticmethod
    def _fit_signal_regression(
        binary: pd.DataFrame,
        forward_returns: pd.Series | None,
        *,
        include_interactions: bool,
        hac_lags: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit HAC OLS after removing non-identifiable signal regressors."""
        if forward_returns is None:
            return pd.DataFrame(), pd.DataFrame(columns=['column', 'reason'])

        design = binary.copy()
        names = list(design.columns)
        if include_interactions:
            for left_position, left_name in enumerate(names):
                for right_name in names[left_position + 1:]:
                    design[f'{left_name} × {right_name}'] = design[left_name] * design[right_name]
        model_frame = pd.concat([design, forward_returns.rename('forward_return')], axis=1).dropna(how='any')
        if model_frame.empty:
            raise ValueError('no common signal and forward-return observations for joint regression')

        filtered_design, removed = TXAnalyzer._filter_regression_design(
            model_frame.drop(columns='forward_return'),
        )
        if filtered_design.empty:
            raise ValueError('joint regression has no non-constant, identifiable regressors')
        if len(model_frame) <= filtered_design.shape[1] + 1:
            raise ValueError('not enough observations for the identifiable joint regression design')

        import statsmodels.api as sm
        fitted = sm.OLS(
            model_frame['forward_return'],
            sm.add_constant(filtered_design, has_constant='add'),
        ).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
        confidence = fitted.conf_int()
        coefficients = pd.DataFrame({
            'coefficient': fitted.params,
            'hac_std_error': fitted.bse,
            't_stat': fitted.tvalues,
            'p_value': fitted.pvalues,
            'ci_lower': confidence[0],
            'ci_upper': confidence[1],
            'observations': int(fitted.nobs),
            'r_squared': fitted.rsquared,
        })
        return coefficients, removed

    @staticmethod
    def _signal_frame_overlap_diagnostics(
        signal_frame: pd.DataFrame,
        *,
        forward_returns: pd.Series | None,
        lead_lags: tuple[int, ...],
        hac_lags: int,
    ) -> dict[str, pd.DataFrame]:
        """Compute state overlap on a complete signal sample and regressions separately."""
        if signal_frame.shape[1] < 2:
            raise ValueError('signals must contain at least two named columns')
        if signal_frame.columns.duplicated().any():
            raise ValueError('signal column names must be unique')
        if hac_lags < 0:
            raise ValueError('hac_lags must be non-negative')
        if signal_frame.empty:
            raise ValueError('no common non-null signal observations in the evaluation period')

        binary = signal_frame.astype(int)
        names = list(binary.columns)
        observations = len(binary)
        pairwise_rows = []
        for left_position, left_name in enumerate(names):
            left = binary[left_name]
            for right_name in names[left_position + 1:]:
                right = binary[right_name]
                left_count, right_count = int(left.sum()), int(right.sum())
                overlap = int((left & right).sum())
                union = int((left | right).sum())
                observed_rate = overlap / observations
                expected_rate = (left_count / observations) * (right_count / observations)
                denominator = np.sqrt(left_count * (observations - left_count) * right_count * (observations - right_count))
                phi = (overlap * observations - left_count * right_count) / denominator if denominator else np.nan
                pairwise_rows.append({
                    'left': left_name, 'right': right_name, 'observations': observations,
                    'left_signal_days': left_count, 'right_signal_days': right_count,
                    'overlap_days': overlap, 'observed_overlap_rate': observed_rate,
                    'expected_overlap_rate': expected_rate,
                    'overlap_lift': observed_rate / expected_rate if expected_rate else np.nan,
                    'jaccard': overlap / union if union else np.nan, 'phi': phi,
                    'p_right_given_left': overlap / left_count if left_count else np.nan,
                    'p_left_given_right': overlap / right_count if right_count else np.nan,
                })
        joint_regression, joint_regression_removed = TXAnalyzer._fit_signal_regression(
            binary, forward_returns, include_interactions=False, hac_lags=hac_lags,
        )
        interaction_regression, interaction_regression_removed = TXAnalyzer._fit_signal_regression(
            binary, forward_returns, include_interactions=True, hac_lags=hac_lags,
        )

        return {
            'signals': signal_frame,
            'phi_correlation': binary.corr(),
            'pairwise_overlap': pd.DataFrame(pairwise_rows),
            'lead_lag': TXAnalyzer._lead_lag_diagnostics(binary, lead_lags),
            'signal_combinations': TXAnalyzer._signal_combination_returns(signal_frame, forward_returns),
            'joint_regression': joint_regression,
            'joint_regression_removed_columns': joint_regression_removed,
            'joint_regression_pairwise_interactions': interaction_regression,
            'joint_regression_pairwise_interactions_removed_columns': interaction_regression_removed,
        }

    @staticmethod
    def _factor_overlap_diagnostics(
        factor_configs: list[tuple[str, pd.Series, float | tuple[float, float]]],
        *,
        training_end: str | pd.Timestamp,
        forward_returns: pd.Series | None,
        evaluation_start: str | pd.Timestamp | None,
        evaluation_end: str | pd.Timestamp | None,
        lead_lags: tuple[int, ...],
        hac_lags: int,
    ) -> dict[str, pd.DataFrame]:
        """Run the common-sample factor/state/return diagnostics."""
        if len(factor_configs) < 2:
            raise ValueError('factor_configs must contain at least two factors')
        if hac_lags < 0:
            raise ValueError('hac_lags must be non-negative')

        factor_map = {}
        percentile_map = {}
        for item in factor_configs:
            if len(item) != 3:
                raise ValueError('each factor config must be (name, factor_series, percentile_range)')
            name, factor, percentile_range = item
            if not isinstance(name, str) or not isinstance(factor, pd.Series):
                raise TypeError('factor configs require a string name and pandas Series')
            if name in factor_map:
                raise ValueError(f'duplicate factor name: {name}')
            factor_map[name] = factor.rename(name)
            percentile_map[name] = percentile_range

        raw_factors = pd.concat(factor_map, axis=1).sort_index()
        thresholds = pd.DataFrame({
            name: TXAnalyzer.fit_factor_thresholds(
                factor.loc[:pd.Timestamp(training_end)],
                percentile_map[name],
            )
            for name, factor in factor_map.items()
        })
        raw_signals = pd.DataFrame({
            name: factor.between(
                thresholds.loc['lower_cutoff', name],
                thresholds.loc['upper_cutoff', name],
                inclusive='both',
            )
            for name, factor in factor_map.items()
        })

        evaluation = raw_factors.copy()
        if evaluation_start is not None:
            evaluation = evaluation.loc[evaluation.index >= pd.Timestamp(evaluation_start)]
        if evaluation_end is not None:
            evaluation = evaluation.loc[evaluation.index <= pd.Timestamp(evaluation_end)]
        evaluation = evaluation.dropna(how='any')
        if evaluation.empty:
            raise ValueError('no common non-null observations in the evaluation period')

        raw_factors = evaluation.loc[:, list(factor_map)]
        signal_frame = raw_signals.reindex(evaluation.index).astype(bool)
        regression_returns = (
            forward_returns.reindex(signal_frame.index)
            if forward_returns is not None else None
        )
        binary = signal_frame.astype(int)
        names = list(binary.columns)
        observations = len(binary)

        pairwise_rows = []
        for left_position, left_name in enumerate(names):
            left = binary[left_name]
            for right_name in names[left_position + 1:]:
                right = binary[right_name]
                left_count, right_count = int(left.sum()), int(right.sum())
                overlap = int((left & right).sum())
                union = int((left | right).sum())
                observed_rate = overlap / observations
                expected_rate = (left_count / observations) * (right_count / observations)
                denominator = np.sqrt(left_count * (observations - left_count) * right_count * (observations - right_count))
                phi = (overlap * observations - left_count * right_count) / denominator if denominator else np.nan
                pairwise_rows.append({
                    'left': left_name,
                    'right': right_name,
                    'observations': observations,
                    'left_signal_days': left_count,
                    'right_signal_days': right_count,
                    'overlap_days': overlap,
                    'observed_overlap_rate': observed_rate,
                    'expected_overlap_rate': expected_rate,
                    'overlap_lift': observed_rate / expected_rate if expected_rate else np.nan,
                    'jaccard': overlap / union if union else np.nan,
                    'phi': phi,
                    'p_right_given_left': overlap / left_count if left_count else np.nan,
                    'p_left_given_right': overlap / right_count if right_count else np.nan,
                })
        joint_regression, joint_regression_removed = TXAnalyzer._fit_signal_regression(
            binary, regression_returns, include_interactions=False, hac_lags=hac_lags,
        )
        interaction_regression, interaction_regression_removed = TXAnalyzer._fit_signal_regression(
            binary, regression_returns, include_interactions=True, hac_lags=hac_lags,
        )

        return {
            'thresholds': thresholds,
            'raw_factors': raw_factors,
            'spearman_correlation': raw_factors.corr(method='spearman'),
            'signals': signal_frame,
            'phi_correlation': binary.corr(),
            'pairwise_overlap': pd.DataFrame(pairwise_rows),
            'lead_lag': TXAnalyzer._lead_lag_diagnostics(binary, lead_lags),
            'signal_combinations': TXAnalyzer._signal_combination_returns(signal_frame, regression_returns),
            'joint_regression': joint_regression,
            'joint_regression_removed_columns': joint_regression_removed,
            'joint_regression_pairwise_interactions': interaction_regression,
            'joint_regression_pairwise_interactions_removed_columns': interaction_regression_removed,
        }

    def conditional_factor_sort(
        self,
        conditions: list[dict],
        sort_factor: pd.Series | str,
        *,
        return_column: str = 'daily_ret',
        plot_return_columns: list[str] | tuple[str, ...] | None = None,
        bin_percentile: float = 5,
        demean_return: bool = False,
        plot_mode: str = 'sorted',
        show_before_sort: bool = False,
        title: str | None = None,
    ) -> pd.DataFrame:
        """Condition on sequential factor-percentile ranges, then sort one factor.

        Each item in ``conditions`` must contain ``factor`` and
        ``percentile_range``. ``factor`` can be a series returned by an
        indicator with ``return_series=True`` or a column name from the
        analyzer frame. Every percentile range is ranked *within the rows
        retained by earlier conditions*. ``sort_factor`` is then ranked within
        the final selected rows and split into equal percentile bins.

        With ``plot_mode='sorted'`` (default), the chart follows the indicator
        convention: after the final factor is sorted, it plots cumulative
        demeaned returns on the main panel, the final factor on the right axis,
        and cumulative raw returns below. With ``plot_mode='bin'``, it plots
        the mean return for each final-factor percentile bin instead.
        ``plot_return_columns`` defaults to ``return_column``. Pass multiple
        columns explicitly when the day and night curves should be shown
        together. Set ``show_before_sort=True`` to first show the selected
        sample sorted by its condition factors, before it is sorted by
        ``sort_factor``.
        The returned table contains final-bin statistics; sequential
        cutoffs and selected dates are available in ``result.attrs`` as
        ``selection_summary`` and ``selected_data``.
        """
        if not conditions:
            raise ValueError('conditions must contain at least one percentile-range condition')
        if return_column not in self.df:
            raise KeyError(f'missing return column: {return_column}')
        if not 0 < bin_percentile <= 50:
            raise ValueError('bin_percentile must be greater than 0 and at most 50')
        if plot_mode not in {'sorted', 'bin'}:
            raise ValueError("plot_mode must be either 'sorted' or 'bin'")
        if plot_return_columns is None:
            plot_return_columns = [return_column]
        else:
            plot_return_columns = list(plot_return_columns)
            if not plot_return_columns:
                raise ValueError('plot_return_columns must contain at least one return column')
            missing_plot_columns = set(plot_return_columns) - set(self.df.columns)
            if missing_plot_columns:
                raise KeyError(f'missing plot return columns: {sorted(missing_plot_columns)}')

        def resolve_factor(factor: pd.Series | str, fallback_name: str) -> tuple[pd.Series, str]:
            if isinstance(factor, str):
                if factor not in self.df:
                    raise KeyError(f'missing factor column: {factor}')
                return self.df[factor].rename(factor), factor
            if not isinstance(factor, pd.Series):
                raise TypeError('each factor must be a pandas Series or an analyzer column name')
            return factor.rename(factor.name or fallback_name), factor.name or fallback_name

        resolved_conditions: list[tuple[str, pd.Series, float, float]] = []
        for index, condition in enumerate(conditions, start=1):
            if not isinstance(condition, dict):
                raise TypeError('each condition must be a dictionary')
            if 'factor' not in condition or 'percentile_range' not in condition:
                raise ValueError("each condition requires 'factor' and 'percentile_range'")
            extra_keys = set(condition) - {'factor', 'percentile_range', 'name'}
            if extra_keys:
                raise ValueError(f'unsupported condition keys: {sorted(extra_keys)}')
            try:
                lower_pct, upper_pct = map(float, condition['percentile_range'])
            except (TypeError, ValueError) as error:
                raise ValueError('percentile_range must be a two-value tuple such as (60, 80)') from error
            if not 0 <= lower_pct < upper_pct <= 100:
                raise ValueError('each percentile_range must satisfy 0 <= lower < upper <= 100')
            series, default_name = resolve_factor(condition['factor'], f'condition_{index}')
            name = str(condition.get('name') or default_name)
            resolved_conditions.append((name, series, lower_pct, upper_pct))

        final_factor, final_factor_name = resolve_factor(sort_factor, 'sort_factor')
        frame = pd.DataFrame({'return': self.df[return_column], 'sort_factor': final_factor})
        for column in plot_return_columns:
            frame[column] = self.df[column]
        for index, (name, series, _, _) in enumerate(resolved_conditions, start=1):
            frame[f'condition_{index}'] = series
        frame = frame.dropna().copy()
        if frame.empty:
            raise ValueError('no overlapping non-null observations are available')

        selection_rows = []
        selected = frame
        for index, (name, _, lower_pct, upper_pct) in enumerate(resolved_conditions, start=1):
            column = f'condition_{index}'
            before_count = len(selected)
            selected = selected.copy()
            lower_cutoff = float(selected[column].quantile(lower_pct / 100))
            upper_cutoff = float(selected[column].quantile(upper_pct / 100))
            selected['percentile'] = selected[column].rank(method='first', pct=True) * 100
            selected = selected.loc[
                (selected['percentile'] > lower_pct) & (selected['percentile'] <= upper_pct)
            ].copy()
            if selected.empty:
                raise ValueError(f"condition '{name}' selected no observations")
            selection_rows.append({
                'step': index,
                'factor': name,
                'percentile_range': f'{lower_pct:g}% to {upper_pct:g}%',
                'lower_cutoff': lower_cutoff,
                'upper_cutoff': upper_cutoff,
                'observations_before': before_count,
                'observations_after': len(selected),
            })
            selected = selected.drop(columns='percentile')

        condition_columns = [f'condition_{index}' for index in range(1, len(resolved_conditions) + 1)]
        selected_before_sort = selected.sort_values(condition_columns).copy()
        selected_before_sort['condition_sort_factor'] = selected_before_sort[condition_columns[-1]]
        selected = selected.sort_values('sort_factor').copy()
        selected['sort_percentile'] = selected['sort_factor'].rank(method='first', pct=True) * 100
        selected['display_return'] = (
            selected['return'] - selected['return'].mean() if demean_return else selected['return']
        )
        bin_edges = np.append(np.arange(0, 100, bin_percentile), 100.0)
        result_rows = []
        for lower_pct, upper_pct in zip(bin_edges[:-1], bin_edges[1:]):
            values = selected.loc[
                (selected['sort_percentile'] > lower_pct) & (selected['sort_percentile'] <= upper_pct)
            ]
            result_rows.append({
                'sort_percentile_bin': f'{lower_pct:g}% to {upper_pct:g}%',
                'lower_percentile': lower_pct,
                'upper_percentile': upper_pct,
                'factor_lower': values['sort_factor'].min(),
                'factor_upper': values['sort_factor'].max(),
                'mean_factor': values['sort_factor'].mean(),
                'mean_return': values['display_return'].mean(),
                'median_return': values['display_return'].median(),
                'P(return<0)': (values['return'] < 0).mean(),
                'observations': len(values),
            })
        result = pd.DataFrame(result_rows).set_index('sort_percentile_bin')
        result.attrs['selection_summary'] = pd.DataFrame(selection_rows).set_index('step')
        result.attrs['selected_data'] = selected.drop(columns='display_return')

        condition_text = ' → '.join(
            f'{row["factor"]} {row["percentile_range"]}' for row in selection_rows
        )
        chart_title = title or final_factor_name

        def plot_sorted_returns(
            chart_frame: pd.DataFrame,
            *,
            right_axis: str,
            plot_title: str,
            note: str,
        ) -> None:
            cumulative_demeaned_columns = []
            cumulative_return_columns = []
            for column in plot_return_columns:
                demeaned_column = f'demeaned_{column}'
                cumulative_demeaned_column = f'cum_{demeaned_column}'
                cumulative_return_column = f'cum_{column}'
                chart_frame[demeaned_column] = chart_frame[column] - chart_frame[column].mean()
                chart_frame[cumulative_demeaned_column] = chart_frame[demeaned_column].cumsum()
                chart_frame[cumulative_return_column] = chart_frame[column].cumsum()
                cumulative_demeaned_columns.append(cumulative_demeaned_column)
                cumulative_return_columns.append(cumulative_return_column)
            plot.plot(
                chart_frame,
                ly=cumulative_demeaned_columns,
                ry=right_axis,
                sub_ly=cumulative_return_columns,
                title=plot_title,
                note=note,
            )

        if show_before_sort:
            plot_sorted_returns(
                selected_before_sort.reset_index(drop=True).copy(),
                right_axis='condition_sort_factor',
                plot_title=f'{chart_title} (after condition sort, before {final_factor_name} sort)',
                note=(
                    f'{condition_text}<br>Condition sort: {resolved_conditions[-1][0]}'
                    f'<br>Sample size: {len(selected_before_sort)}'
                ),
            )

        if plot_mode == 'sorted':
            chart_frame = selected.reset_index(drop=True).copy()
            plot_sorted_returns(
                chart_frame,
                right_axis='sort_factor',
                plot_title=chart_title,
                note=f'{condition_text}<br>Sample size: {len(chart_frame)}',
            )
        else:
            colors = ['#b2182b' if value < 0 else '#2166ac' for value in result['mean_return'].fillna(0)]
            customdata = np.column_stack((
                result['factor_lower'],
                result['factor_upper'],
                result['mean_factor'],
                result['median_return'],
                result['P(return<0)'],
                result['observations'],
            ))
            value_label = f'Demeaned {return_column}' if demean_return else return_column
            fig = go.Figure(
                go.Bar(
                    x=result.index,
                    y=result['mean_return'],
                    marker_color=colors,
                    customdata=customdata,
                    hovertemplate=(
                        f'{final_factor_name} percentile bin: %{{x}}<br>'
                        'Raw factor range: %{customdata[0]:.4%} to %{customdata[1]:.4%}<br>'
                        'Mean raw factor: %{customdata[2]:.4%}<br>'
                        f'Mean {value_label}: %{{y:.3%}}<br>'
                        f'Median {value_label}: %{{customdata[3]:.3%}}<br>'
                        'P(raw return < 0): %{customdata[4]:.1%}<br>'
                        'Observations: %{customdata[5]}'
                        '<extra></extra>'
                    ),
                )
            )
            fig.update_layout(
                title=f'{chart_title}<br><sup>{condition_text} | Sample size: {len(selected)}</sup>',
                template='plotly_white',
                height=520,
                showlegend=False,
            )
            fig.update_xaxes(title_text=f'{final_factor_name} percentile within selected sample')
            fig.update_yaxes(title_text=f'Mean {value_label}', tickformat='.2%')
            fig.add_hline(y=0, line_color='#555555', line_width=1)
            fig.show()
        return result

    def compare_factor_percentiles(
        self,
        factors: dict[str, pd.Series] | pd.Series | str,
        *,
        condition_factor: pd.Series | str | dict[str, pd.Series | str] | None = None,
        return_column: str = 'daily_ret',
        condition_bins: int = 5,
        factor_bin_percentile: float = 20,
        demean_return: bool = False,
        condition_name: str | None = None,
        factor_value_format: str = '.4g',
        condition_value_format: str = '.4g',
        title: str | None = None,
    ) -> pd.DataFrame:
        """Compare arbitrary factors across regimes of another arbitrary factor.

        ``factors`` is either one series/column name or a mapping from display
        names to factor series. Optionally, ``condition_factor`` is split into
        quantile regimes; it can also be a mapping keyed like ``factors`` when
        each row needs a different condition series. Inside every regime each
        tested factor is ranked and split into percentile bins. Omit
        ``condition_factor`` to rank every factor across the full sample.

        ``factor_value_format`` and
        ``condition_value_format`` use Plotly/D3 number formatting in hover
        labels; pass ``'.2%'`` for percentage-valued factors. A heatmap is
        shown and the returned table contains return and raw-factor summaries.
        """
        if return_column not in self.df:
            raise KeyError(f'missing return column: {return_column}')
        if condition_bins < 1:
            raise ValueError('condition_bins must be at least 1')
        if not 0 < factor_bin_percentile <= 50:
            raise ValueError('factor_bin_percentile must be greater than 0 and at most 50')

        def resolve_factor(factor: pd.Series | str, fallback_name: str) -> tuple[pd.Series, str]:
            if isinstance(factor, str):
                if factor not in self.df:
                    raise KeyError(f'missing factor column: {factor}')
                return self.df[factor].rename(factor), factor
            if not isinstance(factor, pd.Series):
                raise TypeError('factors must be pandas Series or analyzer column names')
            return factor.rename(factor.name or fallback_name), factor.name or fallback_name

        if isinstance(factors, dict):
            if not factors:
                raise ValueError('factors must contain at least one factor')
            resolved_factors = {
                str(name): resolve_factor(series, str(name))[0]
                for name, series in factors.items()
            }
        else:
            series, name = resolve_factor(factors, 'factor')
            resolved_factors = {name: series}

        has_condition = condition_factor is not None
        if condition_factor is None:
            condition_bins = 1
            condition_series = pd.Series(0.0, index=self.df.index, name='all observations')
            resolved_conditions = {name: condition_series for name in resolved_factors}
            default_condition_name = 'all observations'
        elif isinstance(condition_factor, dict):
            if set(condition_factor) != set(resolved_factors):
                raise ValueError('condition factor mapping keys must exactly match factors keys')
            resolved_conditions = {
                name: resolve_factor(series, f'{name} condition')[0]
                for name, series in condition_factor.items()
            }
            default_condition_name = 'condition'
        else:
            condition_series, default_condition_name = resolve_factor(condition_factor, 'condition')
            resolved_conditions = {name: condition_series for name in resolved_factors}
        condition_name = condition_name or default_condition_name
        if condition_bins == 1:
            regime_labels = ['all observations']
        else:
            regime_labels = [f'Q{group + 1}' for group in range(condition_bins)]
            regime_labels[0] += ' low'
            regime_labels[-1] += ' high'

        bin_edges = np.append(np.arange(0, 100, factor_bin_percentile), 100.0)
        bin_labels = [f'{lower:g}% to {upper:g}%' for lower, upper in zip(bin_edges[:-1], bin_edges[1:])]
        column_labels = [f'{regime}\n{bin_label}' for regime in regime_labels for bin_label in bin_labels]
        factor_names = list(resolved_factors)
        ma_suffix = 'MA divergence'
        display_factor_names = [
            name[:-len(ma_suffix)].strip()
            if name.endswith(ma_suffix) and name[:-len(ma_suffix)].strip().isdigit()
            else name
            for name in factor_names
        ]
        if len(factor_names) == 1:
            display_factor_names = ['']
            y_axis_title = factor_names[0]
        elif display_factor_names != factor_names and all(name.isdigit() for name in display_factor_names):
            y_axis_title = 'MA window'
        else:
            y_axis_title = 'Tested factor'
        mean_returns = np.full((len(factor_names), len(column_labels)), np.nan)
        median_returns = np.full((len(factor_names), len(column_labels)), np.nan)
        neg_ratios = np.full((len(factor_names), len(column_labels)), np.nan)
        mean_factors = np.full((len(factor_names), len(column_labels)), np.nan)
        mean_conditions = np.full((len(factor_names), len(column_labels)), np.nan)
        observations = np.zeros((len(factor_names), len(column_labels)), dtype=int)

        for row, (factor_name, factor_series) in enumerate(resolved_factors.items()):
            frame = pd.DataFrame({
                'return': self.df[return_column],
                'condition': resolved_conditions[factor_name],
                'factor': factor_series,
            }).dropna()
            if frame.empty:
                continue
            if condition_bins == 1:
                frame['condition_group'] = 0
            else:
                try:
                    frame['condition_group'] = pd.qcut(
                        frame['condition'],
                        q=condition_bins,
                        labels=False,
                        duplicates='raise',
                    )
                except ValueError as error:
                    raise ValueError(
                        f'condition factor cannot be split into {condition_bins} bins for {factor_name}'
                    ) from error
                if frame['condition_group'].nunique() != condition_bins:
                    raise ValueError(
                        f'condition factor cannot create {condition_bins} non-empty bins for {factor_name}'
                    )
            frame['display_return'] = frame['return'] - frame['return'].mean() if demean_return else frame['return']
            for group in range(condition_bins):
                regime = frame.loc[frame['condition_group'].eq(group)].copy()
                if regime.empty:
                    continue
                regime['factor_percentile'] = regime['factor'].rank(method='first', pct=True) * 100
                for bin_index, (lower_pct, upper_pct) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                    column = group * len(bin_labels) + bin_index
                    values = regime.loc[
                        (regime['factor_percentile'] > lower_pct)
                        & (regime['factor_percentile'] <= upper_pct)
                    ]
                    mean_returns[row, column] = values['display_return'].mean()
                    median_returns[row, column] = values['display_return'].median()
                    neg_ratios[row, column] = (values['return'] < 0).mean() if len(values) else np.nan
                    mean_factors[row, column] = values['factor'].mean()
                    mean_conditions[row, column] = values['condition'].mean() if has_condition else np.nan
                    observations[row, column] = len(values)

        customdata = np.stack((mean_factors, mean_conditions, median_returns, neg_ratios, observations), axis=-1)
        column_positions = np.arange(len(column_labels))
        value_label = f'Demeaned {return_column}' if demean_return else return_column
        condition_hover = (
            f'Mean {condition_name}: %{{customdata[1]:{condition_value_format}}}<br>'
            if has_condition
            else ''
        )
        fig = go.Figure(
            go.Heatmap(
                x=column_positions,
                y=display_factor_names,
                z=mean_returns,
                customdata=customdata,
                text=np.tile(column_labels, (len(factor_names), 1)),
                colorscale='RdBu',
                zmid=0,
                xgap=1,
                ygap=1,
                colorbar=dict(title=f'Mean<br>{value_label}', tickformat='.1%'),
                hovertemplate=(
                    f'Factor: {factor_names[0]}<br>%{{text}}<br>'
                    if len(factor_names) == 1
                    else 'Factor: %{y}<br>%{text}<br>'
                ) + (
                    f'Mean raw factor: %{{customdata[0]:{factor_value_format}}}<br>'
                    f'{condition_hover}'
                    f'Mean {value_label}: %{{z:.3%}}<br>'
                    f'Median {value_label}: %{{customdata[2]:.3%}}<br>'
                    'P(raw return < 0): %{customdata[3]:.1%}<br>'
                    'Observations: %{customdata[4]}'
                    '<extra></extra>'
                ),
            )
        )
        for group in range(1, condition_bins):
            fig.add_vline(x=group * len(bin_labels) - 0.5, line_color='#555555', line_width=1)
        fig.update_layout(
            title=title or (
                f'Factor comparison: {value_label} by {condition_name} regime'
                if has_condition
                else f'Factor comparison: {value_label} by factor percentile'
            ),
            template='plotly_white',
            height=max(520, len(factor_names) * 30 + 220),
        )
        fig.update_xaxes(
            title_text=(
                f'{condition_name} regime / factor percentile bin'
                if has_condition
                else 'Factor percentile bin'
            ),
            tickmode='array',
            tickvals=column_positions,
            ticktext=column_labels,
            tickangle=-45,
        )
        fig.update_yaxes(title_text=y_axis_title)
        fig.show()

    def show_factor_signal_timeline(
        self,
        *,
        ma_window: int = 30,
        factor_percentile: float | tuple[float, float] = 15,
        return_column: str = 'daily_ret',
        factor: pd.Series | str | None = None,
        factor_name: str | None = None,
        conditions: list[dict] | None = None,
        position: float = 1.0,
        one_way_cost: float = 0.0,
        thresholds: pd.Series | dict | None = None,
        session: str | None = None,
        show_metrics: bool = True,
    ) -> pd.DataFrame:
        """Plot when a selected factor condition occurs over time.

        By default this calculates MA divergence. Pass ``factor`` as a Series
        (or a column name) to inspect any numeric indicator instead. Pass
        ``conditions`` to filter signal candidates with any factor before
        ranking this factor. Each condition is a dictionary with ``factor``,
        ``percentile_range=(low, high)``, and optional ``name``; conditions are
        applied sequentially. The factor percentile is calculated only after
        all conditions have selected their samples.

        Pass ``factor_percentile=(80, 100)`` to select a closed percentile range.
        Thresholds are fitted as raw quantile cutoffs (or supplied through
        ``thresholds``) and the returned frame contains only signal dates.
        With ``show_metrics=True``, performance is calculated by the same
        position, turnover, and transaction-cost engine as the formal
        threshold backtest.  ``session`` defaults to ``'day'`` for
        ``daily_ret`` and ``'night'`` for ``daily_ret_a``.
        """
        from plotly.subplots import make_subplots

        if factor is None and ma_window < 2:
            raise ValueError('ma_window must be at least 2')
        if not isinstance(position, (int, float, np.number)) or not np.isfinite(position) or position == 0:
            raise ValueError('position must be a finite non-zero number')
        if one_way_cost < 0:
            raise ValueError('one_way_cost must be non-negative')
        if session is None:
            session = {'daily_ret': 'day', 'daily_ret_a': 'night'}.get(return_column)
        if session not in {'day', 'night'}:
            raise ValueError("session must be 'day' or 'night'; specify it for a custom return_column")
        if return_column not in self.df:
            raise KeyError(f"missing return column: {return_column}")
        if conditions is not None and not isinstance(conditions, list):
            raise TypeError('conditions must be a list of condition dictionaries')
        if isinstance(factor_percentile, tuple):
            if len(factor_percentile) != 2:
                raise ValueError('factor_percentile range must contain exactly two values')
            percentile_lower, percentile_upper = map(float, factor_percentile)
            if not 0 <= percentile_lower < percentile_upper <= 100:
                raise ValueError('factor_percentile range must satisfy 0 <= lower < upper <= 100')
            percentile_label = f'{percentile_lower:g}% to {percentile_upper:g}%'
        else:
            percentile_lower = 0.0
            percentile_upper = float(factor_percentile)
            if not 0 < percentile_upper <= 100:
                raise ValueError('factor_percentile must be greater than 0 and at most 100')
            percentile_label = f'<= {percentile_upper:g}%'

        if isinstance(factor, str):
            if factor not in self.df:
                raise KeyError(f'missing factor column: {factor}')
            factor_series = self.df[factor]
        elif factor is None:
            factor_series = ((self.df['Close'] / self.df['Close'].rolling(ma_window).mean()) - 1).shift(1)
        else:
            factor_series = pd.Series(factor).copy()
        factor_label = factor_name or (factor_series.name if factor_series.name else f'{ma_window}MA divergence')

        returns = self.df[return_column]
        return_label = return_column
        factor_values = factor_series
        condition_summary = []
        condition_columns = []
        if conditions:
            frame = pd.DataFrame({'factor': factor_values, 'return': returns})
            for index, condition in enumerate(conditions, start=1):
                if not isinstance(condition, dict):
                    raise TypeError('each condition must be a dictionary')
                if 'factor' not in condition or 'percentile_range' not in condition:
                    raise ValueError("each condition requires 'factor' and 'percentile_range'")
                unknown_keys = set(condition) - {'factor', 'percentile_range', 'name'}
                if unknown_keys:
                    raise ValueError(f'unsupported condition keys: {sorted(unknown_keys)}')
                try:
                    lower, upper = map(float, condition['percentile_range'])
                except (TypeError, ValueError) as error:
                    raise ValueError('condition percentile_range must be a two-value tuple such as (60, 90)') from error
                if not 0 <= lower < upper <= 100:
                    raise ValueError('condition percentile_range must satisfy 0 <= lower < upper <= 100')

                raw_condition_factor = condition['factor']
                if isinstance(raw_condition_factor, str):
                    if raw_condition_factor not in self.df:
                        raise KeyError(f'missing condition factor column: {raw_condition_factor}')
                    condition_series = self.df[raw_condition_factor]
                    default_name = raw_condition_factor
                elif isinstance(raw_condition_factor, pd.Series):
                    condition_series = raw_condition_factor.copy()
                    default_name = condition_series.name or f'condition {index}'
                else:
                    raise TypeError('condition factor must be a pandas Series or an analyzer column name')

                condition_name = str(condition.get('name') or default_name)
                column = f'_condition_{index}'
                frame[column] = condition_series
                condition_columns.append((column, condition_name, lower, upper))

            frame = frame.dropna()
            for column, condition_name, lower, upper in condition_columns:
                frame['condition_percentile'] = frame[column].rank(method='first', pct=True) * 100
                before_count = len(frame)
                frame = frame.loc[
                    frame['condition_percentile'].between(lower, upper, inclusive='both')
                ].copy()
                if frame.empty:
                    raise ValueError(f'condition {condition_name!r} selected no observations')
                condition_summary.append({
                    'condition': condition_name,
                    'percentile_range': f'{lower:g}% to {upper:g}%',
                    'observations_before': before_count,
                    'observations_after': len(frame),
                })

            frame = frame.drop(columns='condition_percentile')
            regime_label = 'within ' + ' / '.join(
                f"{item['condition']} {item['percentile_range']}"
                for item in condition_summary
            )
            percentile_context = 'after conditions'
        else:
            frame = pd.DataFrame({'factor': factor_values, 'return': returns}).dropna()
            regime_label = 'without conditions'
            percentile_context = ''

        # Use the same fixed-cutoff mechanism as ``backtest_threshold_rules``.
        # Ranking is retained only for the chart label; it does not decide a
        # trade, because rank-based ties can disagree with quantile cutoffs.
        if thresholds is None:
            if len(condition_columns) > 1:
                raise ValueError(
                    'formal timeline metrics support at most one condition; '
                    'pass fixed thresholds or use one condition'
                )
            if condition_columns:
                column, _, lower, upper = condition_columns[0]
                fitted_thresholds = self.fit_factor_thresholds(
                    frame['factor'],
                    (percentile_lower, percentile_upper),
                    condition=frame[column],
                    condition_percentile_range=(lower, upper),
                )
                signal_mask = self.threshold_signal(
                    frame['factor'], fitted_thresholds, condition=frame[column]
                ).fillna(False)
            else:
                fitted_thresholds = self.fit_factor_thresholds(
                    frame['factor'], (percentile_lower, percentile_upper)
                )
                signal_mask = self.threshold_signal(frame['factor'], fitted_thresholds).fillna(False)
        else:
            fitted_thresholds = pd.Series(thresholds)
            if len(condition_columns) > 1:
                raise ValueError('fixed-threshold timeline metrics support at most one condition')
            condition_series = frame[condition_columns[0][0]] if condition_columns else None
            signal_mask = self.threshold_signal(
                frame['factor'], fitted_thresholds, condition=condition_series
            ).fillna(False)
        frame['factor_percentile'] = frame['factor'].rank(method='first', pct=True) * 100
        events = frame.loc[signal_mask].copy()
        events.index.name = 'date'
        events.attrs['condition_summary'] = pd.DataFrame(condition_summary)
        customdata = np.column_stack((frame['factor'],))
        hovertemplate = (
            f'Date: %{{x|%Y-%m-%d}}<br>{return_label}: %{{y:.3%}}'
            f'<br>{factor_label}: %{{customdata[0]:.4%}}'
            f'<br>Factor percentile {percentile_context}: %{{text:.1f}}%<extra></extra>'
        )

        positions = pd.DataFrame(0.0, index=self.df.index, columns=['pos_day', 'pos_night'])
        positions.loc[frame.index[signal_mask], f'pos_{session}'] = position
        backtest = self.evaluate(positions, one_way_cost=one_way_cost)
        metrics = self.summarize_result(backtest, return_column='strat_ret')
        events.attrs['metrics'] = metrics
        events.attrs['thresholds'] = fitted_thresholds
        events.attrs['backtest'] = backtest

        monthly_index = pd.date_range(frame.index.min().to_period('M').to_timestamp(), frame.index.max().to_period('M').to_timestamp(), freq='MS')
        monthly_count = events.resample('MS').size().reindex(monthly_index, fill_value=0)
        active_months = int(monthly_count.gt(0).sum())
        gaps = events.index.to_series().diff().dt.days.dropna()
        median_gap = gaps.median() if not gaps.empty else np.nan
        max_gap = gaps.max() if not gaps.empty else np.nan

        max_abs_return = events['return'].abs().max() if not events.empty else 0.0
        color_limit = max(float(max_abs_return), 0.001)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.68, 0.32],
            vertical_spacing=0.1,
            subplot_titles=[f'Signal-day {return_column}', 'Signal count by month'],
        )
        fig.add_trace(
            go.Scatter(
                x=events.index,
                y=events['return'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=events['return'],
                    colorscale='RdBu',
                    cmin=-color_limit,
                    cmax=color_limit,
                    colorbar=dict(title=return_column, tickformat='.1%'),
                    line=dict(color='#222222', width=0.4),
                ),
                customdata=customdata[frame.index.get_indexer(events.index)],
                text=events['factor_percentile'],
                hovertemplate=hovertemplate,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_color='#666666', line_width=1, row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=monthly_count.index,
                y=monthly_count,
                marker_color='#4c78a8',
                hovertemplate='Month: %{x|%Y-%m}<br>Signal days: %{y}<extra></extra>',
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        gap_text = 'n/a' if pd.isna(median_gap) else f'{median_gap:.0f} days'
        max_gap_text = 'n/a' if pd.isna(max_gap) else f'{max_gap:.0f} days'
        fig.update_layout(
            title=(
                f'{factor_label} {percentile_label} {regime_label}: '
                f'{len(events)} signal days | active months {active_months}/{len(monthly_count)} '
                f'| median gap {gap_text} | max gap {max_gap_text}'
            ),
            template='plotly_white',
            height=700,
        )
        fig.update_yaxes(title_text=return_column, tickformat='.2%', row=1, col=1)
        fig.update_yaxes(title_text='Signal days', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.show()
        if show_metrics:
            from IPython.display import display

            display(metrics.to_frame().T)

    def show_factor_signal_overlap(
        self,
        signal_configs: list[dict],
        *,
        return_column: str = 'daily_ret',
        volatility_return_column: str = 'daily_ret',
    ) -> pd.DataFrame:
        """Show overlapping signal dates for arbitrary factors.

        A configuration accepts ``factor`` (a Series or analyzer column name),
        ``factor_name``, ``factor_percentile`` (for example ``15`` or
        ``(80, 100)``), and optional volatility controls: ``volatility_window``,
        ``volatility_regime`` (a bin number or percentile range),
        ``volatility_bins``, and ``volatility_return_column``. Omit ``factor``
        and use the legacy ``ma_window`` / ``divergence_percentile`` settings
        to construct MA divergence. All configured signals must occur on a date
        for it to be included in the returned overlap frame.
        """
        from plotly.subplots import make_subplots

        if len(signal_configs) < 2:
            raise ValueError('signal_configs must contain at least two configurations')
        if return_column not in self.df:
            raise KeyError(f'missing return column: {return_column}')

        signals = pd.DataFrame(index=self.df.index)
        details = pd.DataFrame(index=self.df.index)
        labels: list[str] = []
        summaries = []

        for position, raw_config in enumerate(signal_configs, start=1):
            allowed_keys = {
                'factor', 'factor_name', 'factor_percentile', 'name',
                'ma_window', 'divergence_percentile',
                'volatility_window', 'volatility_regime', 'volatility_bins',
                'volatility_return_column',
            }
            unknown_keys = set(raw_config) - allowed_keys
            if unknown_keys:
                raise ValueError(f'unknown signal configuration keys: {sorted(unknown_keys)}')

            factor = raw_config.get('factor')
            if isinstance(factor, str):
                if factor not in self.df:
                    raise KeyError(f'missing factor column: {factor}')
                factor_series = self.df[factor]
                factor_label = raw_config.get('factor_name') or factor
            elif factor is None:
                ma_window = int(raw_config.get('ma_window', 30))
                if ma_window < 2:
                    raise ValueError('ma_window must be at least 2')
                factor_series = ((self.df['Close'] / self.df['Close'].rolling(ma_window).mean()) - 1).shift(1)
                factor_label = raw_config.get('factor_name') or f'{ma_window}MA divergence'
            elif isinstance(factor, pd.Series):
                factor_series = factor.copy()
                factor_label = raw_config.get('factor_name') or factor.name or f'factor {position}'
            else:
                raise TypeError('factor must be a pandas Series, analyzer column name, or omitted with ma_window')

            factor_percentile = raw_config.get('factor_percentile', raw_config.get('divergence_percentile', 15))
            if isinstance(factor_percentile, tuple):
                if len(factor_percentile) != 2:
                    raise ValueError('factor_percentile range must contain exactly two values')
                factor_lower, factor_upper = map(float, factor_percentile)
            else:
                factor_lower, factor_upper = 0.0, float(factor_percentile)
            if not 0 <= factor_lower < factor_upper <= 100:
                raise ValueError('factor_percentile must satisfy 0 <= lower < upper <= 100')

            volatility_bins = int(raw_config.get('volatility_bins', 1))
            if volatility_bins < 1:
                raise ValueError('volatility_bins must be at least 1')
            frame = pd.DataFrame({'factor': factor_series}).dropna()
            if volatility_bins == 1:
                frame['volatility_group'] = 0
                volatility_label = 'all volatility'
            else:
                volatility_window = int(raw_config.get('volatility_window', 20))
                if volatility_window < 2:
                    raise ValueError('volatility_window must be at least 2')
                volatility_column = raw_config.get('volatility_return_column', volatility_return_column)
                if volatility_column not in self.df:
                    raise KeyError(f'missing volatility return column: {volatility_column}')
                frame['volatility'] = self.df[volatility_column].rolling(volatility_window).std().shift(1)
                frame = frame.dropna()
                frame['volatility_group'] = pd.qcut(frame['volatility'], q=volatility_bins, labels=False, duplicates='drop')
                if frame['volatility_group'].nunique() != volatility_bins:
                    raise ValueError(f'not enough volatility variation to create {volatility_bins} bins for configuration {position}')
                regime = raw_config.get('volatility_regime', volatility_bins)
                if isinstance(regime, tuple):
                    if len(regime) != 2:
                        raise ValueError('volatility_regime range must contain exactly two values')
                    regime_lower, regime_upper = map(float, regime)
                    if not 0 <= regime_lower < regime_upper <= 100:
                        raise ValueError('volatility_regime range must satisfy 0 <= lower < upper <= 100')
                    frame['volatility_percentile'] = frame['volatility'].rank(method='first', pct=True) * 100
                    frame = frame.loc[frame['volatility_percentile'].between(regime_lower, regime_upper, inclusive='both')].copy()
                    frame['volatility_group'] = 0
                    volatility_label = f'vol {regime_lower:g}% to {regime_upper:g}% ({volatility_window}D)'
                else:
                    regime = int(regime)
                    if not 1 <= regime <= volatility_bins:
                        raise ValueError('volatility_regime must be between 1 and volatility_bins')
                    frame = frame.loc[frame['volatility_group'].eq(regime - 1)].copy()
                    volatility_label = f'vol Q{regime} ({volatility_window}D)'

            if frame.empty:
                raise ValueError(f'configuration {position} selected no valid observations')
            frame['factor_percentile'] = frame['factor'].rank(method='first', pct=True) * 100
            signal = frame['factor_percentile'].between(factor_lower, factor_upper, inclusive='both')
            label = str(raw_config.get('name') or factor_label)
            if label in labels:
                label = f'{label} {position}'
            signals[label] = signal.reindex(signals.index, fill_value=False)
            details[f'factor_{position}'] = frame['factor']
            details[f'factor_percentile_{position}'] = frame['factor_percentile']
            if 'volatility' in frame:
                details[f'volatility_{position}'] = frame['volatility']
            labels.append(label)
            summaries.append({
                'signal': label,
                'factor': factor_label,
                'factor_percentile': f'{factor_lower:g}% to {factor_upper:g}%',
                'volatility_condition': volatility_label,
            })

        signals['overlap'] = signals[labels].all(axis=1)
        events = pd.concat([self.df[[return_column]], signals, details], axis=1).loc[signals['overlap']].copy()
        events.index.name = 'date'
        summary = pd.DataFrame(summaries).set_index('signal')
        summary['signal_days'] = [int(signals[label].sum()) for label in labels]
        summary['overlap_rate'] = [len(events) / count if count else np.nan for count in summary['signal_days']]
        events.attrs['signal_summary'] = summary

        monthly_index = pd.date_range(self.df.index.min().to_period('M').to_timestamp(), self.df.index.max().to_period('M').to_timestamp(), freq='MS')
        monthly_counts = {label: signals[label].resample('MS').sum().reindex(monthly_index, fill_value=0) for label in labels}
        overlap_monthly_count = events.resample('MS').size().reindex(monthly_index, fill_value=0)
        gaps = events.index.to_series().diff().dt.days.dropna()
        gap_text = 'n/a' if gaps.empty else f'{gaps.median():.0f} days'
        max_gap_text = 'n/a' if gaps.empty else f'{gaps.max():.0f} days'
        display_labels = [f'{label} ({count})' for label, count in zip(labels, summary['signal_days'])]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.1, subplot_titles=['Signal dates', 'Signal days by month'])
        colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
        for position, label in enumerate(labels):
            dates = signals.index[signals[label]]
            fig.add_trace(go.Scatter(x=dates, y=np.full(len(dates), position), mode='markers', marker=dict(color=colors[position % len(colors)], size=6), hovertemplate=f'{label}<br>Date: %{{x|%Y-%m-%d}}<extra></extra>', showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(x=monthly_counts[label].index, y=monthly_counts[label], name=label, marker_color=colors[position % len(colors)], hovertemplate=f'{label}<br>Month: %{{x|%Y-%m}}<br>Signal days: %{{y}}<extra></extra>'), row=2, col=1)
        fig.add_trace(go.Scatter(x=events.index, y=np.full(len(events), -0.55), mode='markers', marker=dict(color='#111111', size=9, symbol='x'), hovertemplate='Overlap<br>Date: %{{x|%Y-%m-%d}}<extra></extra>', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=overlap_monthly_count.index, y=overlap_monthly_count, name='overlap', marker_color='#111111', hovertemplate='Overlap<br>Month: %{{x|%Y-%m}}<br>Signal days: %{{y}}<extra></extra>'), row=2, col=1)
        rates = ' / '.join(f'{rate:.0%}' for rate in summary['overlap_rate'])
        fig.update_layout(title=f'Signal overlap: {len(events)} days ({rates}) | median gap {gap_text} | max gap {max_gap_text}', template='plotly_white', height=700, barmode='group', legend=dict(orientation='h', x=0, y=0.43, xanchor='left', yanchor='bottom'))
        fig.update_yaxes(title_text='Signal', tickmode='array', tickvals=[-0.55, *range(len(labels))], ticktext=['overlap', *display_labels], range=[-1, len(labels)], row=1, col=1)
        fig.update_yaxes(title_text='Signal days', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.show()

    # =========================================================================
    # Strategy evaluation, risk review, and backtesting
    # =========================================================================
    def evaluate(
        self,
        positions: pd.DataFrame | StrategyConfig | None = None,
        *,
        one_way_cost: float = 0.0,
        config: StrategyConfig | None = None,
        point_version: bool = False,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Evaluate user-supplied day/night positions or the legacy configured strategy.

        ``positions`` must be date-indexed and contain ``pos_day`` and
        ``pos_night``. A position of 1, -1, or 0 means long, short, or flat for
        that session. Passing no positions retains the legacy strategy path.
        """
        if isinstance(positions, StrategyConfig):
            config = positions
            positions = None
        result = (
            self._build_backtest_frame(point_version=point_version, config=config)
            if positions is None
            else self._build_position_backtest_frame(
                positions,
                one_way_cost=one_way_cost,
                point_version=point_version,
            )
        )
        if start is not None:
            result = result.loc[result.index >= pd.Timestamp(start)]
        if end is not None:
            result = result.loc[result.index <= pd.Timestamp(end)]
        return result

    def evaluate_threshold_rules(
        self,
        factors: pd.DataFrame,
        rules: dict[str, dict],
        *,
        one_way_cost: float = 0.0,
    ) -> pd.DataFrame:
        """Backtest fixed factor thresholds for day and/or night sessions.

        Parameters
        ----------
        factors:
            Date-indexed factor table. Its index must be the trading close date.
        rules:
            A mapping with optional ``day`` and ``night`` entries. Each entry
            needs ``factor`` (a column in ``factors``), ``thresholds`` (the
            Series/dict returned by :meth:`fit_factor_thresholds`), and
            ``position`` (normally ``1.0`` or ``-1.0``).  If the thresholds
            include a volatility condition, supply ``condition`` with the
            corresponding factor-column name.

        Returns
        -------
        DataFrame
            Daily factor values, boolean signals, positions, gross/net return,
            transaction cost, and strategy/benchmark equity curves.

        Notes
        -----
        This function never fits thresholds. Fit them on the training set only,
        then pass the fixed values here for validation and test evaluation.
        """
        # Accept the original notebook's factor-name keys as aliases so older
        # notebooks can call this wrapper without first renaming their rules.
        session_aliases = {
            'ma_divergence': 'day',
            'day_divergence': 'day',
            # Night-session factors are known after the night close and before
            # the same-date day open, so their default tradable session is day.
            'night_ret': 'day',
            'night_ret_divergence': 'day',
            'night_divergence': 'day',
        }
        # A mapping key is only a user-facing rule name.  Infer the tradable
        # session from an explicit ``session`` field first, then from the
        # factor name, so names such as ``long_night_ret_divergence`` work.
        normalized_rules = []
        for name, rule in rules.items():
            if not isinstance(rule, dict):
                raise TypeError(f'{name} rule must be a dictionary')
            factor_name = rule.get('factor')
            session = rule.get('session')
            if session is None:
                session = session_aliases.get(factor_name, session_aliases.get(name, name))
            normalized_rules.append((session, name, rule))
        unsupported = {session for session, _, _ in normalized_rules} - {'day', 'night'}
        if unsupported:
            raise ValueError(f"rules only supports 'day' and 'night': {sorted(unsupported)}")
        factor_frame = factors.copy()
        factor_frame.index = pd.to_datetime(factor_frame.index).normalize()
        if factor_frame.index.duplicated().any():
            raise ValueError('factors must contain at most one row per trading date')

        positions = pd.DataFrame(0.0, index=factor_frame.index, columns=['pos_day', 'pos_night'])
        signals = pd.DataFrame(False, index=factor_frame.index, columns=['signal_day', 'signal_night'])
        for session, rule_name, rule in normalized_rules:
            required = {'factor', 'thresholds', 'position'}
            missing = required - set(rule)
            if missing:
                raise ValueError(f"{session} rule missing: {sorted(missing)}")
            factor_name = rule['factor']
            if factor_name not in factor_frame:
                raise KeyError(f"{session} factor not found: {factor_name}")
            thresholds = pd.Series(rule['thresholds'])
            condition_name = rule.get('condition')
            condition = None if condition_name is None else factor_frame[condition_name]
            signal = self.threshold_signal(factor_frame[factor_name], thresholds, condition=condition).fillna(False)
            position = float(rule['position'])
            position_column = f'pos_{session}'
            signal_column = f'signal_{session}'
            overlapping_opposite_position = (
                signal & signals[signal_column] & positions[position_column].ne(position)
            )
            if overlapping_opposite_position.any():
                dates = overlapping_opposite_position.index[overlapping_opposite_position].strftime('%Y-%m-%d').tolist()
                raise ValueError(f"conflicting {session}-session positions on overlapping rules: {dates[:5]}")
            # Multiple same-direction rules are combined with OR: an overlap
            # remains one position, not two units of exposure.
            signals[signal_column] |= signal
            positions.loc[signal, position_column] = position
            signals[f'signal_{rule_name}'] = signal

        return self.evaluate(positions, one_way_cost=one_way_cost).join(factor_frame).join(signals)

    def backtest_threshold_rules(
        self,
        factors: pd.DataFrame,
        rules: dict[str, dict],
        split_dates: dict[str, pd.Timestamp],
        *,
        one_way_cost: float = 0.0,
        title: str = 'Factor threshold backtest: test period',
        plot_period: str = 'test',
        show: bool = True,
    ) -> dict[str, pd.DataFrame | go.Figure | dict[str, go.Figure]]:
        """Run a threshold backtest and present train/test metrics plus the test chart.

        This notebook-facing wrapper combines threshold evaluation, train/test
        performance calculation, and requested equity curves. ``split_dates``
        needs only ``train_end`` and ``test_start``. ``plot_period`` may be
        ``'train'``, ``'test'`` (default), ``'both'``, or ``'none'``.
        """
        required_splits = {'train_end', 'test_start'}
        missing_splits = required_splits - set(split_dates)
        if missing_splits:
            raise ValueError(f"split_dates missing: {sorted(missing_splits)}")
        if plot_period not in {'train', 'test', 'both', 'none'}:
            raise ValueError("plot_period must be 'train', 'test', 'both', or 'none'")

        result = self.evaluate_threshold_rules(factors, rules, one_way_cost=one_way_cost)
        train = result.loc[:split_dates['train_end']].copy()
        test = result.loc[split_dates['test_start']:].copy()
        performance = pd.DataFrame({
            'Train Strategy': self.summarize_result(train, return_column='strat_ret'),
            'Train Benchmark': self.summarize_result(train, return_column='benchmark_ret'),
            'Test Strategy': self.summarize_result(test, return_column='strat_ret'),
            'Test Benchmark': self.summarize_result(test, return_column='benchmark_ret'),
        }).T
        # Benchmark is a passive fully invested comparison, so it has no
        # strategy turnover even though the shared result frame records it.
        performance.loc[['Train Benchmark', 'Test Benchmark'], 'Annual Turnover'] = 0.0
        if show:
            display(performance)
        figures = {}
        if plot_period in {'train', 'both'}:
            figures['train'] = self.plot_equity_curve(
                train,
                title=title.replace('test period', 'training period'),
                show=show,
            )
        if plot_period in {'test', 'both'}:
            figures['test'] = self.plot_equity_curve(test, title=title, show=show)
        return {
            'result': result,
            'train': train,
            'test': test,
            'performance': performance,
            'figures': figures,
        }

    @staticmethod
    def plot_equity_curve(
        result: pd.DataFrame,
        *,
        title: str = 'Factor threshold backtest',
        show: bool = True,
    ) -> go.Figure:
        """Build an interactive strategy-versus-benchmark equity chart."""
        required = {'equity_strat', 'equity_benchmark'}
        missing = required - set(result.columns)
        if missing:
            raise KeyError(f"result missing equity columns: {sorted(missing)}")
        equity = result.loc[:, ['equity_strat', 'equity_benchmark']].copy()
        if equity.empty:
            raise ValueError('cannot plot an empty result')
        # A sliced test period inherits cumulative equity from the full sample.
        # Rebase both series so the comparison always starts at growth-of-$1.
        equity = equity.div(equity.iloc[0])
        fig = go.Figure()
        fig.add_scatter(x=equity.index, y=equity['equity_strat'], mode='lines', name='Strategy (net)')
        fig.add_scatter(x=equity.index, y=equity['equity_benchmark'], mode='lines', name='Buy & hold')
        fig.update_layout(title=title, template='plotly_white', yaxis_title='Growth of $1')
        if show:
            fig.show()
        return fig

    def summarize_result(
        self,
        result: pd.DataFrame,
        *,
        return_column: str = 'strat_ret',
        point_version: bool = False,
    ) -> pd.Series:
        """Summarize an ``evaluate()`` result or an explicitly sliced validation period."""
        if return_column not in result:
            raise KeyError(f"result missing return column: {return_column}")
        summary = pd.Series(self._calculate_metrics(result[return_column].copy(), point_version=point_version))
        if 'turnover' in result and len(result):
            summary['Annual Turnover'] = result['turnover'].fillna(0).mean() * 252
        return summary

    def _build_position_backtest_frame(
        self,
        positions: pd.DataFrame,
        *,
        one_way_cost: float,
        point_version: bool,
    ) -> pd.DataFrame:
        """Apply a date-indexed position matrix to the paired session returns."""
        required = {'pos_day', 'pos_night'}
        missing = required - set(positions.columns)
        if missing:
            raise KeyError(f"positions missing columns: {sorted(missing)}")
        if one_way_cost < 0:
            raise ValueError('one_way_cost must be non-negative')

        position_frame = positions.loc[:, ['pos_day', 'pos_night']].copy()
        position_frame.index = pd.to_datetime(position_frame.index).normalize()
        if position_frame.index.duplicated().any():
            raise ValueError('positions must contain at most one row per trading date')
        df = self.df.join(position_frame, how='left')
        df[['pos_day', 'pos_night']] = df[['pos_day', 'pos_night']].fillna(0.0)

        df['turnover'] = (
            df['pos_day'].diff().abs().fillna(df['pos_day'].abs())
            + df['pos_night'].diff().abs().fillna(df['pos_night'].abs())
        )
        if point_version:
            df['gross_ret'] = df['pos_day'] * df['daily_pnl'] + df['pos_night'] * df['daily_pnl_a']
            df['benchmark_ret'] = df['Close'].diff()
        else:
            df['gross_ret'] = (
                (1 + df['pos_day'] * df['daily_ret'])
                * (1 + df['pos_night'] * df['daily_ret_a']) - 1
            )
            # Continuous Buy & Hold: previous trading-day day close to the
            # current day close. This includes overnight and session-opening
            # gaps, unlike multiplying two intraday open-to-close returns.
            df['benchmark_ret'] = df['Close'].pct_change()
        df['cost'] = df['turnover'] * one_way_cost
        df['strat_ret'] = df['gross_ret'] - df['cost']
        if point_version:
            df['equity_strat'] = df['strat_ret'].fillna(0).cumsum()
            df['equity_benchmark'] = df['benchmark_ret'].fillna(0).cumsum()
        else:
            df['equity_strat'] = (1 + df['strat_ret'].fillna(0)).cumprod()
            df['equity_benchmark'] = (1 + df['benchmark_ret'].fillna(0)).cumprod()
        df['cum_strat'] = df['equity_strat'] - 1
        df['cum_bnh'] = df['equity_benchmark'] - 1
        return df

    def _build_backtest_frame(
        self,
        point_version: bool,
        config: StrategyConfig | None = None,
    ) -> pd.DataFrame:
        """Build the result frame without rendering charts or writing files."""
        active_config = config or self.config
        df = StrategyEngine.calculate_factors(self.df)
        df = StrategyEngine.apply_positions(df, active_config)

        if point_version:
            df['daily_pnl_a'] = df['Open'] - df['Open_a']
            df['daily_pnl'] = df['Open_a'].shift(-1) - df['Open']
            df['strat_ret'] = (df['daily_pnl_a'] * df['pos_night']) + (df['daily_pnl'] * df['pos_day'])
            df['benchmark_ret'] = df['daily_pnl']
        else:
            df['daily_ret_a'] = (df['Open'] / df['Open_a']) - 1
            df['daily_ret'] = (df['Open_a'].shift(-1) / df['Open']) - 1
            df['strat_ret'] = (df['daily_ret_a'] * df['pos_night']) + (df['daily_ret'] * df['pos_day'])
            df['benchmark_ret'] = df['daily_ret_a']

        df['cum_strat'] = df['strat_ret'].cumsum()
        df['cum_bnh'] = df['benchmark_ret'].cumsum()
        return df

    def check_risk_events(self, filter_tech_signal: bool = False) -> pd.DataFrame:
        """Return a log that explains each active day-session position."""
        df = StrategyEngine.apply_positions(StrategyEngine.calculate_factors(self.df), self.config)
        move_below_threshold = df['MOVE_ind'] < self.config.move_threshold
        foreign_option_bearish = df['foreign_opt_pos_divergence_a'] < self.config.foreign_option_threshold
        divergence_supports_long = df['divergence_v2'] < self.config.divergence_threshold

        log = pd.DataFrame(index=df.index)
        log['Factor'] = np.select(
            [move_below_threshold, foreign_option_bearish, ~divergence_supports_long],
            ['MOVE', 'Foreign option positioning', 'Day divergence'],
            default='No active signal',
        )
        log['Value'] = np.select(
            [move_below_threshold, foreign_option_bearish, ~divergence_supports_long],
            [df['MOVE_ind'], df['foreign_opt_pos_divergence_a'], df['divergence_v2']],
            default=np.nan,
        )
        log['Action'] = np.select(
            [df['pos_day'].gt(0), df['pos_day'].lt(0)],
            ['Long day session', 'Short day session'],
            default='Flat',
        )
        log['Tech_Signal'] = np.where(df['pos_day'].gt(0), 'Buy', 'Neutral')
        log['Divergence'] = df['divergence_v2']
        log.index.name = 'Date'
        log = log.reset_index()

        if filter_tech_signal:
            log = log.loc[log['Action'].ne('Flat')]
        return log

    def backtest(
        self,
        risk_log: bool = False,
        point_version: bool = False,
        config: StrategyConfig | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        *,
        positions: pd.DataFrame | None = None,
        one_way_cost: float = 0.0,
    ):
        """Run a supplied position matrix or the legacy configured strategy."""
        df = self.evaluate(
            positions,
            config=config,
            one_way_cost=one_way_cost,
            point_version=point_version,
            start=start,
            end=end,
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / 'backtest.csv', index=True)
        # ===============================================================
        # 4. 顯示績效統計 (Performance Metrics)
        # ===============================================================
        print(f"=== Performance Metrics ({'Points' if point_version else 'Percentage'}) ===")
        # Strategy Metrics
        strat_metrics = self._calculate_metrics(df['strat_ret'], point_version=point_version)
        # Benchmark Metrics
        bnh_metrics = self._calculate_metrics(df['benchmark_ret'], point_version=point_version)
        
        # Combine into DataFrame
        metrics_df = pd.DataFrame([strat_metrics, bnh_metrics], index=['Strategy', 'Benchmark']).T
        
        # Formatting function
        def format_metrics(val, name):
            if isinstance(val, (int, float)):
                if not point_version and ('Return' in name or 'CAGR' in name or 'Volatility' in name or 'Drawdown' in name or 'Win Rate' in name or 'Avg' in name):
                     return f"{val*100:.2f}%"
                elif point_version:
                    if 'Win Rate' in name:
                         return f"{val*100:.2f}%"
                    elif 'Total PnL' in name or 'Points DD' in name or 'Avg' in name:
                        return f"{val:.2f} pts"
                    else:
                        return f"{val:.2f}"
                elif not point_version:
                    if 'Duration' in name:
                        return f"{int(val)} days"
                    else:
                        return f"{val:.2f}"
            return val

        # Apply formatting
        formatted_df = metrics_df.copy().astype(object)
        for idx in formatted_df.index:
            formatted_df.loc[idx] = formatted_df.loc[idx].apply(lambda x: format_metrics(x, idx))

        display(formatted_df.T)
        

        # ===============================================================
        # 5. 顯示風控事件 (Risk Events)
        # ===============================================================
        if risk_log and positions is None:
            print("=== Risk Events Log (Top 20) ===")
            risk_events = self.check_risk_events()
            if not risk_events.empty:
                display(risk_events.tail(20)) # 顯示最近 20 筆，避免洗版
            else:
                print("No risk events triggered.")

        return plot.plot(df, ly=['cum_strat', 'cum_bnh'], ry_dashed=False)

    # =========================================================================
    # Distribution and robustness views
    # =========================================================================
    def show_factor_distributions(self, factors: list = None):
        """
        顯示策略使用之各項指標因子分佈情形 (Histograms & Statistics)
        factors: 指定要分析的因子清單 (預設為常用因子)
        """
        from plotly.subplots import make_subplots
        
        df = self.df.copy()
        # 確保因子已計算
        df = self._calculate_factors(df)
        
        # 定義主要因子 (若未指定 fetch default)
        if factors is None:
            factors = ['foreign_opt_pos_divergence_a', 'MOVE_ind', 'SOX_ind', 'divergence']
        
        # 移除沒有該 flag 的情形 (例如 holiday 需要計算)
        valid_factors = [f for f in factors if f in df.columns]

        if not valid_factors:
            print("No valid factors found to plot.")
            return

        # 1. 統計數據
        print("=== Factor Statistics ===")
        display(df[valid_factors].describe())
        
        # 2. 繪圖 (2x2 Grid)
        rows = 2
        cols = 2
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=valid_factors)
        
        for i, col in enumerate(valid_factors):
            row = (i // cols) + 1
            c = (i % cols) + 1
            
            # 過濾 NaN
            series = df[col].dropna()
            
            fig.add_trace(
                go.Histogram(x=series, name=col, nbinsx=100, histnorm='probability'),
                row=row, col=c
            )
            
        fig.update_layout(
            title_text="Strategy Factors Distribution", 
            showlegend=False,
            height=700,
        )
        fig.show()

    def show_performance_distributions(self, rolling_window: int = 126):
        """
        顯示策略績效指標的分佈 (Rolling Metrics & Returns Distribution)
        rolling_window: 滾動窗口天數 (預設 126 天約半年)
        包含所有指標: CAGR, Volatility, Sharpe, Max Drawdown, Max DD Duration, 
        Profit Factor, Win Rate, Odds, Avg Win, Avg Loss, Avg Return (Exp), Kelly
        """
        from plotly.subplots import make_subplots
        from tqdm import tqdm
        
        df = self.df.copy()
        
        # 自動執行回測邏輯以取得報酬率 (若尚未計算)
        if 'strat_ret' not in df.columns:
            df = self._calculate_factors(df)
            df = self._apply_signals_logic(df)
            df['strat_ret'] = (df['daily_ret_a'] * df['pos_night']) + (df['daily_ret'] * df['pos_day'])

        # 1. 準備滾動數據
        daily_rets = df['strat_ret'].dropna()
        n_samples = len(daily_rets)
        
        if n_samples < rolling_window:
            print(f"Not enough data for rolling window {rolling_window}. Samples: {n_samples}")
            return

        rolling_metrics = []
        
        # 使用迴圈計算 (雖然較慢但最準確，且能重用 _calculate_metrics 邏輯)
        # 為了效率，先取得 numpy array
        dates = daily_rets.index.tolist()
        
        print(f"Calculating rolling metrics for {n_samples - rolling_window + 1} windows...")
        for i in tqdm(range(n_samples - rolling_window + 1), desc="Rolling Metrics"):
            window_slice = daily_rets.iloc[i : i+rolling_window]
            # 呼叫既有的計算函數 (回傳的是 raw float dict)
            metrics = self._calculate_metrics(window_slice)
            # 加上日期標籤 (用窗口最後一天)
            metrics['Date'] = dates[i+rolling_window-1]
            rolling_metrics.append(metrics)
            
        r_df = pd.DataFrame(rolling_metrics).set_index('Date')
        
        # 移除 Total Return (因為是 Rolling 的，Total Return 只是該期間的報酬，用 CAGR 或 Avg Return 可能較好，但這裡還是會有)
        # 這裡根據需求顯示 12 個指標
        target_metrics = [
            'CAGR', 'Volatility', 'Sharpe', 
            'Max Drawdown', 'Max DD Duration', 'Profit Factor', 
            'Win Rate', 'Odds', 'Kelly',
            'Avg Win', 'Avg Loss', 'Avg Return (Exp)'
        ]
        
        # 2. 統計摘要
        print(f"=== Rolling Performance Statistics (Window: {rolling_window} days) ===")
        display(r_df[target_metrics].describe())

        # 3. 繪圖 (4x3 Grid)
        rows = 4
        cols = 3
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=target_metrics)
        
        for i, metric in enumerate(target_metrics):
            if metric not in r_df.columns:
                continue
                
            row = (i // cols) + 1
            c = (i % cols) + 1
            
            # 過濾 inf / nan
            series = r_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            
            # 根據指標設定不同顏色 (綠色好/紅色壞 的概念，或統一)
            color = '#1f77b4' # default blue
            if metric in ['Max Drawdown', 'Volatility', 'Avg Loss', 'Max DD Duration']:
                color = '#d62728' # red for risk/less is better
            elif metric in ['Sharpe', 'CAGR', 'Profit Factor', 'Win Rate', 'Kelly']:
                color = '#2ca02c' # green for good
                
            fig.add_trace(
                go.Histogram(x=series, name=metric, nbinsx=50, histnorm='probability', marker_color=color),
                row=row, col=c
            )
            
            # Add mean line annotation (optional/cluttered)
        
        fig.update_layout(
            title_text=f"Rolling Strategy Performance Distributions (Window: {rolling_window} days)", 
            showlegend=False,
            height=1000,
        )
        fig.show()
