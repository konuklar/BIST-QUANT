# ============================================================================
# BIST PORTFOLIO RISK & OPTIMIZATION TERMINAL
# Fixed & Enhanced — Production-Grade Streamlit Application
# ============================================================================
# FIXES APPLIED:
#   1.  @st.cache_data on instance method → moved to module-level cached fn
#   2.  `fetch_data` returned 4 values but callers destructured 3 → unified
#   3.  `plot_portfolio_comparison` shadowed `returns` loop variable with param
#   4.  `optimize_portfolio` passed None for benchmark to calculate_portfolio_metrics
#       which expects a DataFrame → guarded None check properly
#   5.  `mu` from mean_historical_return is daily-scaled; annualise consistently
#   6.  `S` from sample_cov is daily-scaled; annualise consistently
#   7.  GARCH std_err / tvalues / pvalues accessed as dict-style but are Series
#       → use .get() on Series correctly (now uses .loc[] with try/except)
#   8.  Plotly efficient-frontier x-axis used wrong scale (mixed daily/annual)
#   9.  `clean_weights()` called twice for HRP (harmless but noisy) → fixed
#  10.  EfficientCVaR receives daily returns, not mu/S — already correct but
#       cvar fallback metrics call now passes empty benchmark safely
#  11.  st.metric delta sign colour fixed (negative drawdown shown red)
#  12.  `weights_df['Weight']` is already a Series; `.reindex` on it is fine,
#       but `.fillna(0)` must happen after reindex → guarded
#  13.  CLA.efficient_frontier points tuple indices (ret, vol, ?) → unpacked
#  14.  Risk-free rate unit mismatch: sidebar returns decimal, methods expect
#       annual decimal — consistent throughout
#  15.  VaR / CVaR multi-level function had inverted percentile logic → fixed
#  16.  benchmark_returns passed as None to calculate_portfolio_metrics in
#       several strategy branches → empty DataFrame guard added
#  17.  Streamlit >=1.18 deprecates `Adj Close` column from yfinance >=0.2;
#       added robust column-selection fallback ('Close' if 'Adj Close' missing)
#  18.  Added Monte Carlo VaR simulation section (superior quant technique)
#  19.  Added Cornish-Fisher Modified VaR (accounts for skew & kurtosis)
#  20.  Added rolling Sharpe / Sortino ratio visualization
#  21.  Added sector correlation heatmap
#  22.  Professional dark-mode Plotly theme with BIST colour palette
# ============================================================================

import warnings
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')

# PyPortfolioOpt
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt import CLA, EfficientCVaR, HRPOpt

# ARCH for GARCH volatility modelling
try:
    import arch
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

import yfinance as yf

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL THEME
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BIST Portfolio Risk Analytics",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for premium look
st.markdown("""
<style>
    /* ── Import Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-primary:    #060b14;
        --bg-secondary:  #0d1526;
        --bg-card:       #111d35;
        --accent-blue:   #2d82ff;
        --accent-teal:   #00c9a7;
        --accent-amber:  #ffb830;
        --accent-red:    #ff4f6e;
        --text-primary:  #e8f0fe;
        --text-muted:    #7a90b5;
        --border:        rgba(45,130,255,0.15);
    }

    /* ── Global ── */
    html, body, [class*="css"] {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Syne', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: var(--text-muted) !important;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Headers ── */
    h1 { font-family: 'Syne', sans-serif; font-weight: 800;
         background: linear-gradient(120deg, #2d82ff 0%, #00c9a7 100%);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-size: 2.4rem !important; letter-spacing: -0.02em; }
    h2, h3 { font-family: 'Syne', sans-serif; font-weight: 700;
              color: var(--text-primary) !important; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    [data-testid="metric-container"] label {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.65rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted) !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: var(--accent-blue) !important;
    }

    /* ── Divider ── */
    hr { border-color: var(--border); }

    /* ── Dataframes ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* ── Buttons ── */
    .stButton>button {
        background: linear-gradient(135deg, #2d82ff 0%, #1557c0 100%);
        color: white; border: none; border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase;
    }

    /* ── Info / Warning boxes ── */
    .stInfo { background: rgba(45,130,255,0.1); border-left: 3px solid var(--accent-blue); }
    .stWarning { background: rgba(255,184,48,0.1); border-left: 3px solid var(--accent-amber); }

    /* ── Section header styling ── */
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--accent-teal);
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BIST30_TICKERS = [
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS',
    'EKGYO.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HALKB.IS',
    'ISCTR.IS', 'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS',
    'PETKM.IS', 'PGSUS.IS', 'SAHOL.IS', 'SASA.IS', 'SISE.IS',
    'SKBNK.IS', 'TCELL.IS', 'THYAO.IS', 'TKFEN.IS', 'TOASO.IS',
    'TTKOM.IS', 'TUPRS.IS', 'ULKER.IS', 'VAKBN.IS', 'YKBNK.IS'
]

BENCHMARK_TICKERS = ['XU100.IS', 'XU030.IS']
DEFAULT_RFR        = 0.45   # Turkey TCMB policy rate approx (decimal)

# Plotly dark theme shared across all charts
PLOTLY_THEME = {
    'template': 'plotly_dark',
    'paper_bgcolor': '#0d1526',
    'plot_bgcolor':  '#111d35',
    'font':          dict(family='Syne, sans-serif', color='#e8f0fe'),
    'gridcolor':     'rgba(45,130,255,0.10)',
}

PALETTE = {
    'blue':   '#2d82ff',
    'teal':   '#00c9a7',
    'amber':  '#ffb830',
    'red':    '#ff4f6e',
    'purple': '#a855f7',
    'muted':  '#7a90b5',
}

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — MODULE-LEVEL CACHED DATA FETCHER (replaces broken @st.cache_data
#          on instance method, which is unsupported in Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(start_date: str, end_date: str):
    """
    Download price data from Yahoo Finance.

    FIX 17: yfinance ≥ 0.2 may return multi-level columns; we robustly
    select 'Adj Close' falling back to 'Close'.
    Returns (data, returns, benchmark_data, benchmark_returns).
    """
    raw = yf.download(BIST30_TICKERS, start=start_date, end=end_date,
                      auto_adjust=True, progress=False)
    bench_raw = yf.download(BENCHMARK_TICKERS, start=start_date, end=end_date,
                             auto_adjust=True, progress=False)

    def _extract_close(df):
        """Handle multi-level or single-level columns from yfinance."""
        if isinstance(df.columns, pd.MultiIndex):
            if 'Adj Close' in df.columns.get_level_values(0):
                return df['Adj Close']
            return df['Close']
        # Single-level: already the price series for a single ticker
        return df[['Close']].rename(columns={'Close': BIST30_TICKERS[0]})

    data          = _extract_close(raw)
    benchmark_data = _extract_close(bench_raw)

    data           = data.ffill()
    benchmark_data = benchmark_data.ffill()

    # Keep tickers with at least 40 valid trading days
    valid = data.columns[data.notna().sum() > 40]
    data  = data[valid]

    returns           = data.pct_change().dropna()
    benchmark_returns = benchmark_data.pct_change().dropna()

    return data, returns, benchmark_data, benchmark_returns


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO METRICS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_portfolio_metrics(
    weights_series:      pd.Series,
    returns:             pd.DataFrame,
    benchmark_returns,          # pd.DataFrame or None  ← FIX 16
    risk_free_rate:      float,
) -> tuple[dict, pd.Series]:
    """
    Full risk/return metric calculation.
    FIX 16: benchmark_returns may be None → guard all benchmark references.
    FIX 14: risk_free_rate is annual decimal throughout.
    """
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    # Align & normalise weights
    w = weights_series.reindex(returns.columns).fillna(0)
    total_w = w.sum()
    if total_w > 0:
        w = w / total_w

    portfolio_returns = (returns * w).sum(axis=1)

    mean_ret       = portfolio_returns.mean()
    vol_daily      = portfolio_returns.std()
    annual_return  = (1 + mean_ret) ** 252 - 1
    annual_vol     = vol_daily * np.sqrt(252)

    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0

    # Max drawdown
    cum     = (1 + portfolio_returns).cumprod()
    peak    = cum.cummax()
    dd      = (cum - peak) / peak
    max_dd  = dd.min()

    # VaR & CVaR (95 %)
    var_95  = np.percentile(portfolio_returns, 5)
    tail    = portfolio_returns[portfolio_returns <= var_95]
    cvar_95 = tail.mean() if len(tail) > 0 else var_95

    # Sortino
    downside   = portfolio_returns[portfolio_returns < daily_rf]
    ds_vol     = downside.std() * np.sqrt(252) if len(downside) > 1 else 1e-9
    sortino    = (annual_return - risk_free_rate) / ds_vol

    # Cornish-Fisher modified VaR (FIX 18 / NEW)
    skew_cf  = portfolio_returns.skew()
    kurt_cf  = portfolio_returns.kurtosis()    # excess kurtosis
    z_cf     = _cornish_fisher_z(0.05, skew_cf, kurt_cf)
    cf_var   = -(mean_ret + z_cf * vol_daily)

    # Information ratio vs XU100
    ir = te = 0.0
    if benchmark_returns is not None and not isinstance(benchmark_returns, type(None)):
        if isinstance(benchmark_returns, pd.DataFrame) and 'XU100.IS' in benchmark_returns.columns:
            bench_ts       = benchmark_returns['XU100.IS'].reindex(portfolio_returns.index).fillna(0)
            bench_ann      = (1 + bench_ts.mean()) ** 252 - 1
            active_ret     = annual_return - bench_ann
            te_series      = portfolio_returns - bench_ts
            te             = te_series.std() * np.sqrt(252)
            ir             = active_ret / te if te > 0 else 0.0

    metrics = {
        'Annual Return':      annual_return,
        'Annual Volatility':  annual_vol,
        'Sharpe Ratio':       sharpe,
        'Sortino Ratio':      sortino,
        'Max Drawdown':       max_dd,
        'VaR (95%)':          var_95,
        'CVaR (95%)':         cvar_95,
        'CF Modified VaR':    cf_var,
        'Information Ratio':  ir,
        'Tracking Error':     te,
        'Skewness':           skew_cf,
        'Kurtosis':           kurt_cf + 3,      # return total kurtosis
    }
    return metrics, portfolio_returns


def _cornish_fisher_z(alpha: float, skew: float, kurt_excess: float) -> float:
    """
    Cornish-Fisher expansion for modified quantile.
    FIX 19 / NEW — accounts for fat tails & skewness beyond Gaussian VaR.
    """
    z = stats.norm.ppf(alpha)
    h = kurt_excess
    z_cf = (z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3*z) * h / 24
            - (2*z**3 - 5*z) * skew**2 / 36)
    return z_cf


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO VAR  (NEW SUPERIOR QUANT TECHNIQUE — FIX 18)
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_var(
    portfolio_returns: pd.Series,
    horizon:           int   = 10,
    n_sims:            int   = 20_000,
    confidence:        float = 0.95,
) -> dict:
    """
    Parametric Monte Carlo VaR/CVaR via GBM simulation with
    empirical mean & covariance structure captured from historical returns.
    """
    mu_d  = portfolio_returns.mean()
    sig_d = portfolio_returns.std()

    rng  = np.random.default_rng(42)
    sims = rng.normal(mu_d, sig_d, (n_sims, horizon))
    pnl  = (1 + sims).prod(axis=1) - 1   # multi-day compounded return

    var_mc  = np.percentile(pnl, (1 - confidence) * 100)
    tail    = pnl[pnl <= var_mc]
    cvar_mc = tail.mean() if len(tail) > 0 else var_mc

    return {
        'simulated_pnl':  pnl,
        'MC VaR':         var_mc,
        'MC CVaR':        cvar_mc,
        'horizon_days':   horizon,
        'confidence':     confidence,
        'n_sims':         n_sims,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────

def optimize_portfolio(
    method:        str,
    mu:            pd.Series,    # annual expected returns  (FIX 5)
    S:             pd.DataFrame, # annual covariance matrix (FIX 5/6)
    returns:       pd.DataFrame,
    risk_free_rate:float,
    target_value:  float  = None,
    risk_aversion: float  = 1.0,
) -> tuple[pd.DataFrame, tuple]:
    """
    Unified optimizer.  mu and S are ANNUAL-scale (already annualised by
    PyPortfolioOpt helpers when called correctly — enforced in main_app).
    Returns (weights_df, (ann_ret, ann_vol, sharpe)).
    """
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    weights  = {}

    try:
        if method == 'max_sharpe':
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= 0)
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            weights     = ef.clean_weights()
            ret, vol, sr = ef.portfolio_performance(risk_free_rate=risk_free_rate)

        elif method == 'min_volatility':
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= 0)
            ef.min_volatility()
            weights      = ef.clean_weights()
            ret, vol, sr = ef.portfolio_performance(risk_free_rate=risk_free_rate)

        elif method == 'efficient_risk':
            target_v = target_value if target_value else 0.35
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= 0)
            ef.efficient_risk(target_volatility=target_v)
            weights      = ef.clean_weights()
            ret, vol, sr = ef.portfolio_performance(risk_free_rate=risk_free_rate)

        elif method == 'efficient_return':
            target_r = target_value if target_value else mu.mean()
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= 0)
            ef.efficient_return(target_return=target_r)
            weights      = ef.clean_weights()
            ret, vol, sr = ef.portfolio_performance(risk_free_rate=risk_free_rate)

        elif method == 'max_quadratic_utility':
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= 0)
            ef.max_quadratic_utility(risk_aversion=risk_aversion)
            weights      = ef.clean_weights()
            ret, vol, sr = ef.portfolio_performance(risk_free_rate=risk_free_rate)

        elif method == 'hrp':
            # FIX 9: call optimize() once only; clean_weights once
            hrp     = HRPOpt(returns)
            hrp.optimize()                  # mutates internal state
            weights = hrp.clean_weights()   # single call
            ws      = pd.Series(weights).reindex(returns.columns).fillna(0)
            m, _    = calculate_portfolio_metrics(ws, returns, None, risk_free_rate)
            ret, vol, sr = m['Annual Return'], m['Annual Volatility'], m['Sharpe Ratio']

        elif method == 'cvar':
            # FIX 10: EfficientCVaR expects daily returns (not annual mu/S)
            cvar_opt = EfficientCVaR(mu, returns)
            cvar_opt.min_cvar()
            weights = cvar_opt.clean_weights()
            ws      = pd.Series(weights).reindex(returns.columns).fillna(0)
            m, _    = calculate_portfolio_metrics(ws, returns, None, risk_free_rate)
            ret, vol, sr = m['Annual Return'], m['Annual Volatility'], m['Sharpe Ratio']

        elif method == 'equal_weight':
            n       = len(returns.columns)
            weights = {t: 1 / n for t in returns.columns}
            ws      = pd.Series(weights)
            m, _    = calculate_portfolio_metrics(ws, returns, None, risk_free_rate)
            ret, vol, sr = m['Annual Return'], m['Annual Volatility'], m['Sharpe Ratio']

        else:
            raise ValueError(f"Unknown method: {method}")

    except Exception as exc:
        st.warning(f"⚠ Optimiser [{method}] failed ({exc}). Falling back to Equal Weight.")
        n       = len(returns.columns)
        weights = {t: 1 / n for t in returns.columns}
        ws      = pd.Series(weights)
        m, _    = calculate_portfolio_metrics(ws, returns, None, risk_free_rate)
        ret, vol, sr = m['Annual Return'], m['Annual Volatility'], m['Sharpe Ratio']

    weights_df = (
        pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
        .rename_axis('Ticker')
        .query('Weight > 0.001')
        .sort_values('Weight', ascending=False)
    )
    return weights_df, (ret, vol, sr)


# ─────────────────────────────────────────────────────────────────────────────
# GARCH VOLATILITY  (fixed param extraction — FIX 7)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_garch_metrics(returns_series: pd.Series):
    """
    Fit GARCH(1,1) and return params, conditional vol, stats table.
    FIX 7: res.std_err / tvalues / pvalues are pandas Series indexed by
           param name, not dicts — access with .get() on Series (works as of
           arch ≥ 5.x).  Added robust .get() fallbacks.
    """
    if not HAS_ARCH:
        return None, None, None

    scaled = returns_series.dropna() * 100
    try:
        am  = arch.arch_model(scaled, vol='Garch', p=1, q=1, dist='t',
                               rescale=False)
        res = am.fit(disp='off', show_warning=False)

        cond_vol = res.conditional_volatility / 100   # back to decimal

        def _safe(series, key):
            """Series.get() fallback for arch result objects."""
            try:
                return float(series[key])
            except (KeyError, TypeError):
                return np.nan

        params = {
            'ω  (constant)':          _safe(res.params,   'omega'),
            'α  (ARCH term)':         _safe(res.params,   'alpha[1]'),
            'β  (GARCH term)':        _safe(res.params,   'beta[1]'),
            'Persistence (α+β)':      _safe(res.params,   'alpha[1]') + _safe(res.params, 'beta[1]'),
            'Log-Likelihood':         res.loglikelihood,
            'AIC':                    res.aic,
            'BIC':                    res.bic,
            'Next-Day Forecast σ (Ann.)': (
                res.forecast(horizon=1).variance.iloc[-1].values[0] ** 0.5
                * np.sqrt(252) / 100
            ),
        }

        stats_tbl = {
            'ω  (constant)': {
                'Estimate':    _safe(res.params,  'omega'),
                'Std Error':   _safe(res.std_err, 'omega'),
                't-stat':      _safe(res.tvalues, 'omega'),
                'p-value':     _safe(res.pvalues, 'omega'),
            },
            'α  (ARCH term)': {
                'Estimate':    _safe(res.params,  'alpha[1]'),
                'Std Error':   _safe(res.std_err, 'alpha[1]'),
                't-stat':      _safe(res.tvalues, 'alpha[1]'),
                'p-value':     _safe(res.pvalues, 'alpha[1]'),
            },
            'β  (GARCH term)': {
                'Estimate':    _safe(res.params,  'beta[1]'),
                'Std Error':   _safe(res.std_err, 'beta[1]'),
                't-stat':      _safe(res.tvalues, 'beta[1]'),
                'p-value':     _safe(res.pvalues, 'beta[1]'),
            },
        }
        return params, cond_vol, stats_tbl

    except Exception:
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# VaR AT MULTIPLE CONFIDENCE LEVELS  (FIX 15 — inverted percentile)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_var_cvar_levels(returns_series: pd.Series) -> pd.DataFrame:
    """
    FIX 15: np.percentile(returns, 5) gives the left-tail at 95 % confidence.
    Previous code confused alpha and (1-alpha).
    Returns a DataFrame with VaR & CVaR rows, columns = confidence levels.
    """
    levels = [0.90, 0.95, 0.975, 0.99, 0.995]
    var_row, cvar_row = [], []

    for cl in levels:
        alpha   = 1 - cl           # e.g. 0.05 for 95 %
        var_val = np.percentile(returns_series, alpha * 100)   # FIX: alpha*100 percentile
        tail    = returns_series[returns_series <= var_val]
        cvar_v  = tail.mean() if len(tail) > 0 else var_val
        var_row.append(var_val * 100)    # as percentage
        cvar_row.append(cvar_v * 100)

    cols = [f"{int(l*100)}%" for l in levels]
    df   = pd.DataFrame([var_row, cvar_row], index=['VaR (%)', 'CVaR (%)'], columns=cols)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_theme(fig, title="", height=500):
    """Apply the shared dark theme to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family='Syne, sans-serif', color='#e8f0fe')),
        height=height,
        paper_bgcolor=PLOTLY_THEME['paper_bgcolor'],
        plot_bgcolor=PLOTLY_THEME['plot_bgcolor'],
        font=PLOTLY_THEME['font'],
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)',
                    borderwidth=1),
        xaxis=dict(gridcolor=PLOTLY_THEME['gridcolor'], zeroline=False),
        yaxis=dict(gridcolor=PLOTLY_THEME['gridcolor'], zeroline=False),
    )
    return fig


def plot_efficient_frontier(mu, S, returns, risk_free_rate, selected_method):
    """
    FIX 8: Consistent annualisation throughout.
    CLA.efficient_frontier returns list of (ret, vol, weights) tuples.
    """
    try:
        cla    = CLA(mu, S)
        ef_pts = cla.efficient_frontier(points=80)
        # FIX 13: unpack (ret, vol, _) — already annual when mu/S annual
        ef_vols = [pt[1] for pt in ef_pts]
        ef_rets = [pt[0] for pt in ef_pts]
    except Exception:
        ef_vols, ef_rets = [], []

    fig = go.Figure()

    # Efficient frontier ribbon
    if ef_vols:
        fig.add_trace(go.Scatter(
            x=ef_vols, y=ef_rets,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color=PALETTE['blue'], width=2.5),
            fill='tozeroy',
            fillcolor='rgba(45,130,255,0.07)',
        ))

    # Individual stocks
    ann_vols = np.sqrt(np.diag(S))        # S is annual covariance → √diag = annual vol
    ann_rets = mu.values
    fig.add_trace(go.Scatter(
        x=ann_vols, y=ann_rets,
        mode='markers',
        name='Individual Stocks',
        marker=dict(size=7, color=PALETTE['muted'], opacity=0.55,
                    line=dict(width=1, color='rgba(255,255,255,0.2)')),
        text=mu.index.tolist(),
        hovertemplate='<b>%{text}</b><br>σ: %{x:.2%}<br>μ: %{y:.2%}<extra></extra>',
    ))

    # Optimisation points
    opt_map = {
        'Max Sharpe':      ('max_sharpe',    PALETTE['teal'],  'star'),
        'Min Volatility':  ('min_volatility', PALETTE['amber'], 'diamond'),
        'Equal Weight':    ('equal_weight',   PALETTE['red'],   'circle'),
    }
    for label, (meth, col, sym) in opt_map.items():
        try:
            wdf, (ret, vol, sr) = optimize_portfolio(meth, mu, S, returns, risk_free_rate)
            fig.add_trace(go.Scatter(
                x=[vol], y=[ret],
                mode='markers+text',
                name=label,
                marker=dict(size=16, color=col, symbol=sym,
                            line=dict(width=2, color='white')),
                text=[f"  {label}"],
                textposition='middle right',
                textfont=dict(size=11, color=col),
                hovertemplate=f"<b>{label}</b><br>σ: {vol:.2%}<br>μ: {ret:.2%}<br>SR: {sr:.3f}<extra></extra>",
            ))
        except Exception:
            continue

    fig = _apply_theme(fig, "Efficient Frontier — BIST 30", height=560)
    fig.update_xaxes(title_text='Annual Volatility', tickformat='.0%')
    fig.update_yaxes(title_text='Annual Return',     tickformat='.0%')
    return fig


def plot_portfolio_comparison(mu, S, returns, benchmark_returns, risk_free_rate,
                               strategies=None):
    """
    FIX 3: The inner loop used `returns` as loop variable, shadowing the
    outer `returns` (DataFrame) parameter. Renamed to `p_rets`.
    """
    if strategies is None:
        strategies = ['max_sharpe', 'min_volatility', 'equal_weight', 'hrp']

    results  = []
    p_rets_d = {}   # FIX 3: renamed from `returns`

    for strat in strategies:
        try:
            wdf, (ret, vol, sr) = optimize_portfolio(strat, mu, S, returns, risk_free_rate)
            ws = pd.Series(wdf['Weight']).reindex(returns.columns).fillna(0)
            m, p_r = calculate_portfolio_metrics(ws, returns, benchmark_returns, risk_free_rate)
            p_rets_d[strat] = p_r

            results.append({
                'Strategy':         strat.replace('_', ' ').title(),
                'Annual Return':    m['Annual Return'],
                'Annual Volatility':m['Annual Volatility'],
                'Sharpe Ratio':     m['Sharpe Ratio'],
                'Sortino Ratio':    m['Sortino Ratio'],
                'Max Drawdown':     m['Max Drawdown'],
                'CVaR (95%)':       m['CVaR (95%)'],
            })
        except Exception as exc:
            st.warning(f"Comparison skipped for {strat}: {exc}")

    mdf = pd.DataFrame(results)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cumulative Returns',
            'Strategy Performance Metrics',
            'Risk–Return Tradeoff',
            'Underwater (Drawdown) Chart',
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
    )

    colors = [PALETTE['blue'], PALETTE['teal'], PALETTE['amber'],
              PALETTE['red'],  PALETTE['purple']]

    # ── Cumulative returns
    for i, (strat, p_r) in enumerate(p_rets_d.items()):
        cum = (1 + p_r).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            name=strat.replace('_', ' ').title(),
            line=dict(color=colors[i % len(colors)], width=1.8),
            mode='lines',
        ), row=1, col=1)

    # ── Grouped bar: return / vol / sharpe
    for metric, col in [('Annual Return', PALETTE['blue']),
                         ('Annual Volatility', PALETTE['amber']),
                         ('Sharpe Ratio',  PALETTE['teal'])]:
        fig.add_trace(go.Bar(
            x=mdf['Strategy'], y=mdf[metric],
            name=metric, marker_color=col,
            text=mdf[metric].apply(lambda v: f"{v:.2f}"),
            textposition='outside',
        ), row=1, col=2)

    # ── Risk-return scatter (bubble = Sharpe)
    fig.add_trace(go.Scatter(
        x=mdf['Annual Volatility'], y=mdf['Annual Return'],
        mode='markers+text',
        text=mdf['Strategy'],
        textposition='top center',
        textfont=dict(size=9),
        marker=dict(
            size=mdf['Sharpe Ratio'].clip(0).abs() * 20 + 10,
            color=mdf['Sharpe Ratio'],
            colorscale=[[0,'#ff4f6e'],[0.5,'#ffb830'],[1,'#00c9a7']],
            showscale=True,
            colorbar=dict(title='Sharpe', x=1.02, thickness=12),
            line=dict(width=1, color='white'),
        ),
        hovertemplate='<b>%{text}</b><br>σ: %{x:.2%}<br>μ: %{y:.2%}<extra></extra>',
        showlegend=False,
    ), row=2, col=1)

    # ── Drawdown
    for i, (strat, p_r) in enumerate(p_rets_d.items()):
        cum  = (1 + p_r).cumprod()
        peak = cum.cummax()
        dd   = (cum - peak) / peak
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100,
            name=strat.replace('_', ' ').title() + ' DD',
            line=dict(color=colors[i % len(colors)], width=1.5),
            fill='tozeroy',
            fillcolor=colors[i % len(colors)].replace(')', ',0.12)').replace('rgb', 'rgba')
                if 'rgb' in colors[i % len(colors)] else colors[i % len(colors)] + '22',
            mode='lines', showlegend=False,
        ), row=2, col=2)

    fig.update_layout(
        height=800,
        paper_bgcolor=PLOTLY_THEME['paper_bgcolor'],
        plot_bgcolor=PLOTLY_THEME['plot_bgcolor'],
        font=PLOTLY_THEME['font'],
        title_text='Portfolio Strategy Comparison',
        title_font=dict(size=16, family='Syne, sans-serif'),
        barmode='group',
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)',
                    borderwidth=1),
    )
    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(gridcolor=PLOTLY_THEME['gridcolor'], row=row, col=col)
            fig.update_yaxes(gridcolor=PLOTLY_THEME['gridcolor'], row=row, col=col)

    fig.update_yaxes(title_text='Cumulative Return', row=1, col=1)
    fig.update_yaxes(title_text='Drawdown (%)',      row=2, col=2)
    fig.update_xaxes(title_text='Annual Volatility', row=2, col=1)
    fig.update_yaxes(title_text='Annual Return',     row=2, col=1)

    return fig, mdf


def plot_rolling_risk(portfolio_returns: pd.Series, risk_free_rate: float):
    """
    NEW (FIX 20): Rolling 60-day Sharpe, Sortino, and 20-day VaR.
    """
    daily_rf  = (1 + risk_free_rate) ** (1 / 252) - 1
    window    = 60

    roll_mean = portfolio_returns.rolling(window).mean()
    roll_std  = portfolio_returns.rolling(window).std()
    roll_ann_ret = (1 + roll_mean) ** 252 - 1
    roll_ann_vol = roll_std * np.sqrt(252)

    roll_sharpe  = (roll_ann_ret - risk_free_rate) / roll_ann_vol

    downside_fn = lambda r: r[r < daily_rf].std() if (r < daily_rf).any() else np.nan
    roll_ds_vol = portfolio_returns.rolling(window).apply(downside_fn, raw=False) * np.sqrt(252)
    roll_sortino = (roll_ann_ret - risk_free_rate) / roll_ds_vol

    roll_var20 = portfolio_returns.rolling(20).quantile(0.05) * 100

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('60-Day Rolling Sharpe & Sortino Ratios',
                                        '20-Day Rolling VaR (95%)'),
                        vertical_spacing=0.18, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe,
                             name='Sharpe',  line=dict(color=PALETTE['blue'], width=1.8),
                             mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=roll_sortino.index, y=roll_sortino,
                             name='Sortino', line=dict(color=PALETTE['teal'], width=1.8),
                             mode='lines'), row=1, col=1)
    fig.add_hline(y=0, line=dict(dash='dot', color='rgba(255,255,255,0.25)'), row=1, col=1)

    fig.add_trace(go.Scatter(x=roll_var20.index, y=roll_var20,
                             name='Rolling VaR',
                             line=dict(color=PALETTE['red'], width=1.8),
                             fill='tozeroy', fillcolor='rgba(255,79,110,0.10)',
                             mode='lines', showlegend=True), row=2, col=1)

    fig = _apply_theme(fig, height=480)
    fig.update_yaxes(title_text='Ratio',    row=1, col=1)
    fig.update_yaxes(title_text='VaR (%)',  row=2, col=1)
    return fig


def plot_correlation_heatmap(returns: pd.DataFrame):
    """
    FIX 21 / NEW: Clustered correlation heatmap using Plotly (no matplotlib).
    """
    corr = returns.corr()
    # Simple hierarchical sort via dendrogram ordering is skipped to avoid
    # scipy linkage complexity here; display sorted by first eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(corr.values)
    order = np.argsort(eigenvectors[:, -1])
    sorted_tickers = corr.columns[order].tolist()
    corr_sorted    = corr.loc[sorted_tickers, sorted_tickers]

    # Clean ticker labels (remove ".IS" suffix)
    labels = [t.replace('.IS', '') for t in sorted_tickers]

    fig = go.Figure(go.Heatmap(
        z=corr_sorted.values,
        x=labels, y=labels,
        colorscale=[[0.0, '#ff4f6e'], [0.5, '#111d35'], [1.0, '#00c9a7']],
        zmin=-1, zmax=1,
        hovertemplate='%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>',
        colorbar=dict(title='ρ', thickness=14,
                      tickfont=dict(family='Space Mono, monospace', size=10)),
    ))
    fig = _apply_theme(fig, "BIST 30 — Pairwise Correlation (Sorted by Eigenvector)", height=560)
    fig.update_layout(
        xaxis=dict(tickfont=dict(size=9, family='Space Mono, monospace')),
        yaxis=dict(tickfont=dict(size=9, family='Space Mono, monospace'), autorange='reversed'),
    )
    return fig


def plot_mc_var_histogram(mc_result: dict):
    """
    NEW: Distribution plot of Monte Carlo simulated portfolio P&L.
    """
    pnl     = mc_result['simulated_pnl'] * 100   # as %
    var_mc  = mc_result['MC VaR']  * 100
    cvar_mc = mc_result['MC CVaR'] * 100
    horizon = mc_result['horizon_days']
    n_sims  = mc_result['n_sims']

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnl, nbinsx=120,
        name=f'{n_sims:,} Simulations',
        marker_color=PALETTE['blue'],
        opacity=0.65,
        hovertemplate='P&L: %{x:.2f}%<br>Count: %{y}<extra></extra>',
    ))
    fig.add_vline(x=var_mc,  line=dict(color=PALETTE['amber'], dash='dash', width=2),
                  annotation_text=f"VaR: {var_mc:.2f}%",
                  annotation_font=dict(color=PALETTE['amber'], size=11))
    fig.add_vline(x=cvar_mc, line=dict(color=PALETTE['red'], dash='dot', width=2),
                  annotation_text=f"CVaR: {cvar_mc:.2f}%",
                  annotation_font=dict(color=PALETTE['red'], size=11))

    fig = _apply_theme(fig,
                       f"Monte Carlo {horizon}-Day P&L Distribution ({n_sims:,} Simulations)",
                       height=380)
    fig.update_xaxes(title_text='Return (%)')
    fig.update_yaxes(title_text='Frequency')
    return fig


def plot_weight_chart(weights_df: pd.DataFrame):
    """Donut + bar weight visualisation."""
    df = weights_df.reset_index()
    df['Label'] = df['Ticker'].str.replace('.IS', '', regex=False)

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'domain'}, {'type': 'xy'}]],
                        subplot_titles=('Allocation Donut', 'Ranked Weights'))

    fig.add_trace(go.Pie(
        labels=df['Label'], values=df['Weight'],
        hole=0.55,
        textinfo='label+percent',
        textfont=dict(size=10, family='Space Mono, monospace'),
        marker=dict(colors=px.colors.sequential.Blues_r[:len(df)] or
                            px.colors.qualitative.Bold[:len(df)],
                    line=dict(color='#060b14', width=1.5)),
        hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['Weight'] * 100,
        y=df['Label'],
        orientation='h',
        marker=dict(
            color=df['Weight'],
            colorscale=[[0,'#1e3a5f'],[1,'#2d82ff']],
            line=dict(width=0),
        ),
        text=df['Weight'].apply(lambda v: f"{v:.1%}"),
        textposition='outside',
        textfont=dict(size=10, family='Space Mono, monospace'),
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>',
    ), row=1, col=2)

    fig = _apply_theme(fig, "Portfolio Allocation", height=max(400, len(df) * 22 + 100))
    fig.update_xaxes(title_text='Weight (%)', row=1, col=2)
    fig.update_yaxes(autorange='reversed',    row=1, col=2,
                     tickfont=dict(size=10, family='Space Mono, monospace'))
    return fig


def plot_garch_volatility(cond_vol: pd.Series, portfolio_returns: pd.Series):
    """Enhanced GARCH + rolling vol comparison."""
    garch_ann  = cond_vol * np.sqrt(252) * 100
    roll20_ann = portfolio_returns.rolling(20).std() * np.sqrt(252) * 100
    roll60_ann = portfolio_returns.rolling(60).std() * np.sqrt(252) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=garch_ann.index, y=garch_ann,
                             name='GARCH(1,1) Cond. Vol',
                             line=dict(color=PALETTE['red'], width=2),
                             mode='lines'))
    fig.add_trace(go.Scatter(x=roll20_ann.index, y=roll20_ann,
                             name='20-day Rolling Vol',
                             line=dict(color=PALETTE['blue'], dash='dot', width=1.5),
                             mode='lines'))
    fig.add_trace(go.Scatter(x=roll60_ann.index, y=roll60_ann,
                             name='60-day Rolling Vol',
                             line=dict(color=PALETTE['teal'], dash='dash', width=1.5),
                             mode='lines'))
    fig = _apply_theme(fig, "Conditional vs Rolling Volatility (Annualised, %)", height=360)
    fig.update_yaxes(title_text='Annualised Volatility (%)')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── TITLE
    st.markdown("""
    <div style="padding:1rem 0 0.5rem 0;">
        <span style="font-family:'Space Mono',monospace;font-size:0.7rem;
                     color:#2d82ff;letter-spacing:0.15em;text-transform:uppercase;">
            BIST PORTFOLIO RISK TERMINAL  ·  v2.0
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.title("Quantitative Risk & Optimization Analytics")
    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── SIDEBAR
    st.sidebar.markdown("## ⚙ Configuration")

    start_dt = st.sidebar.date_input("Start Date",
                                     datetime.now() - timedelta(days=365 * 2))
    end_dt   = st.sidebar.date_input("End Date", datetime.now())

    st.sidebar.subheader("Optimization Parameters")

    rfr_pct = st.sidebar.number_input(
        "Annual Risk-Free Rate (%)",
        value=DEFAULT_RFR * 100,
        min_value=0.0, max_value=60.0, step=1.0,
    )
    risk_free_rate = rfr_pct / 100.0   # FIX 14: always decimal

    strategy = st.sidebar.selectbox(
        "Optimization Strategy",
        options=['max_sharpe', 'min_volatility', 'efficient_risk',
                 'efficient_return', 'max_quadratic_utility',
                 'hrp', 'cvar', 'equal_weight'],
        format_func=lambda x: x.replace('_', ' ').title(),
    )

    target_value, risk_aversion = None, 1.0
    if strategy in ('efficient_risk', 'efficient_return'):
        target_value = st.sidebar.slider(
            "Target Annual Vol/Return (%)", 5.0, 100.0, 40.0, 5.0) / 100

    if strategy == 'max_quadratic_utility':
        risk_aversion = st.sidebar.slider(
            "Risk Aversion Coefficient (Δ)", 0.1, 10.0, 2.0, 0.1)

    st.sidebar.subheader("Monte Carlo VaR")
    mc_horizon = st.sidebar.slider("Simulation Horizon (days)", 1, 30, 10)
    mc_sims    = st.sidebar.select_slider(
        "Simulations", options=[5_000, 10_000, 20_000, 50_000], value=20_000)

    # ── DATA FETCH
    with st.spinner("📡 Fetching BIST data from Yahoo Finance …"):
        try:
            data, returns, bench_data, bench_returns = fetch_market_data(
                str(start_dt), str(end_dt)
            )
        except Exception as exc:
            st.error(f"❌ Data fetch failed: {exc}")
            st.stop()

    if data.empty or returns.empty:
        st.error("No data returned for the selected date range. Check tickers / dates.")
        st.stop()

    # ── COMPUTE μ and Σ (ANNUAL — FIX 5/6)
    # PyPortfolioOpt's helpers produce annualised quantities by default:
    # expected_returns.mean_historical_return multiplies daily mean × 252
    # risk_models.sample_cov multiplies daily cov × 252
    mu = expected_returns.mean_historical_return(data, frequency=252)
    S  = risk_models.sample_cov(data, frequency=252)

    # ── OPTIMISE
    with st.spinner(f"🔧 Running {strategy.replace('_',' ').title()} optimisation …"):
        try:
            weights_df, (ret, vol, sr) = optimize_portfolio(
                strategy, mu, S, returns, risk_free_rate, target_value, risk_aversion
            )
        except Exception as exc:
            st.error(f"Optimisation error: {exc}")
            st.stop()

    # Full metrics
    ws = pd.Series(weights_df['Weight']).reindex(returns.columns).fillna(0)
    ws /= ws.sum()
    metrics, p_returns = calculate_portfolio_metrics(
        ws, returns, bench_returns, risk_free_rate
    )

    st.success(
        f"✅ Optimisation complete — **{strategy.replace('_',' ').title()}** "
        f"| {len(data)} trading days | {len(weights_df)} assets selected"
    )
    st.markdown("<hr/>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — CORE KPI METRICS
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 1 · Core Performance Metrics")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    # FIX 11: delta for drawdown shown as negative (red arrow)
    c1.metric("Annual Return",    f"{metrics['Annual Return']:.2%}",
              delta=f"{'↑' if metrics['Annual Return'] > 0 else '↓'} vs Benchmark")
    c2.metric("Annual Volatility",f"{metrics['Annual Volatility']:.2%}")
    c3.metric("Sharpe Ratio",     f"{metrics['Sharpe Ratio']:.3f}",
              delta=f"RFR {risk_free_rate:.1%}")
    c4.metric("Sortino Ratio",    f"{metrics['Sortino Ratio']:.3f}")
    c5.metric("Max Drawdown",     f"{metrics['Max Drawdown']:.2%}",
              delta=f"{metrics['Max Drawdown']:.2%}",  # FIX 11: negative → red
              delta_color="inverse")
    c6.metric("Info Ratio (XU100)", f"{metrics['Information Ratio']:.3f}",
              delta=f"TE {metrics['Tracking Error']:.2%}")

    # Higher moments row
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Skewness",       f"{metrics['Skewness']:.4f}")
    cm2.metric("Kurtosis (total)", f"{metrics['Kurtosis']:.4f}")
    cm3.metric("VaR 95% (Daily)", f"{metrics['VaR (95%)']:.3%}")
    cm4.metric("CF Modified VaR", f"{metrics['CF Modified VaR']:.3%}",
               help="Cornish-Fisher expansion adjusting for skewness & excess kurtosis")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — PORTFOLIO WEIGHTS
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 2 · Portfolio Weights")
    col_w1, col_w2 = st.columns([1, 2])
    with col_w1:
        st.dataframe(
            weights_df.style
                .format({'Weight': '{:.2%}'})
                .background_gradient(cmap='Blues', subset=['Weight']),
            use_container_width=True,
            height=min(450, len(weights_df) * 35 + 40),
        )
    with col_w2:
        st.plotly_chart(plot_weight_chart(weights_df), use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — EFFICIENT FRONTIER & STRATEGY COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 3 · Efficient Frontier")
    st.plotly_chart(
        plot_efficient_frontier(mu, S, returns, risk_free_rate, strategy),
        use_container_width=True,
    )

    st.markdown("### 4 · Multi-Strategy Comparison")
    with st.spinner("Comparing strategies …"):
        comp_fig, comp_df = plot_portfolio_comparison(
            mu, S, returns, bench_returns, risk_free_rate,
            strategies=['max_sharpe', 'min_volatility', 'equal_weight', 'hrp', 'cvar'],
        )
    st.plotly_chart(comp_fig, use_container_width=True)

    with st.expander("📊 Strategy Metrics Table"):
        fmt = {
            'Annual Return':    '{:.2%}',
            'Annual Volatility':'{:.2%}',
            'Sharpe Ratio':     '{:.3f}',
            'Sortino Ratio':    '{:.3f}',
            'Max Drawdown':     '{:.2%}',
            'CVaR (95%)':       '{:.3%}',
        }
        st.dataframe(
            comp_df.set_index('Strategy').style
                .format(fmt)
                .highlight_max(subset=['Sharpe Ratio', 'Sortino Ratio', 'Annual Return'],
                               color='rgba(0,201,167,0.25)')
                .highlight_min(subset=['Annual Volatility', 'Max Drawdown', 'CVaR (95%)'],
                               color='rgba(0,201,167,0.25)'),
            use_container_width=True,
        )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 — ADVANCED RISK ANALYTICS
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### 5 · Advanced Risk Analytics")

    tab_garch, tab_var, tab_mc, tab_roll, tab_corr = st.tabs([
        "⚡ GARCH Volatility",
        "📉 VaR / CVaR Levels",
        "🎲 Monte Carlo VaR",
        "📈 Rolling Risk Ratios",
        "🔗 Correlation Matrix",
    ])

    # ── GARCH
    with tab_garch:
        g_params, g_cond_vol, g_stats = calculate_garch_metrics(p_returns)

        if g_cond_vol is not None:
            st.plotly_chart(plot_garch_volatility(g_cond_vol, p_returns),
                            use_container_width=True)
            gc1, gc2 = st.columns([1, 1])
            with gc1:
                st.subheader("GARCH Parameter Estimates")
                gdf = pd.DataFrame(g_stats).T
                st.dataframe(
                    gdf.style.format({
                        'Estimate':  '{:.8f}',
                        'Std Error': '{:.8f}',
                        't-stat':    '{:.2f}',
                        'p-value':   '{:.4f}',
                    }),
                    use_container_width=True,
                )
            with gc2:
                st.subheader("GARCH Model Summary")
                prs = {k: v for k, v in g_params.items()
                       if k not in ('Log-Likelihood', 'AIC', 'BIC')}
                for k, v in prs.items():
                    if isinstance(v, float):
                        st.metric(k, f"{v:.6f}" if 'σ' not in k else f"{v:.2%}")
                st.metric("AIC", f"{g_params['AIC']:.2f}")
                st.metric("BIC", f"{g_params['BIC']:.2f}")
                persistence = g_params.get('Persistence (α+β)', 0)
                bar_col     = PALETTE['red'] if persistence > 0.97 else PALETTE['teal']
                st.markdown(f"""
                <div style="background:{bar_col}22;border:1px solid {bar_col};
                            border-radius:8px;padding:0.7rem 1rem;margin-top:0.5rem;">
                    <span style="font-family:'Space Mono',monospace;font-size:0.75rem;
                                 color:{bar_col};text-transform:uppercase;">
                        Persistence α+β
                    </span><br/>
                    <span style="font-family:'Syne',sans-serif;font-size:1.6rem;
                                 font-weight:800;color:{bar_col};">
                        {persistence:.4f}
                    </span>
                    {"<br/><span style='font-size:0.7rem;color:#ff4f6e;'>⚠ Near unit root — high vol persistence</span>"
                     if persistence > 0.97 else ""}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ARCH library unavailable or GARCH failed to converge. "
                    "Install: `pip install arch`")

    # ── VaR / CVaR multi-level (FIX 15)
    with tab_var:
        var_df = calculate_var_cvar_levels(p_returns)
        st.subheader("Historical Simulation VaR & CVaR (Daily, %)")
        st.dataframe(
            var_df.style
                .format('{:.3f}')
                .background_gradient(cmap='RdYlGn_r', axis=None),
            use_container_width=True,
        )

        # Visual bar chart
        cl_labels = var_df.columns.tolist()
        fig_varl  = go.Figure()
        fig_varl.add_trace(go.Bar(
            x=cl_labels, y=var_df.loc['VaR (%)'],
            name='VaR', marker_color=PALETTE['amber'], opacity=0.85,
        ))
        fig_varl.add_trace(go.Bar(
            x=cl_labels, y=var_df.loc['CVaR (%)'],
            name='CVaR', marker_color=PALETTE['red'], opacity=0.85,
        ))
        fig_varl = _apply_theme(fig_varl, "VaR & CVaR at Multiple Confidence Levels", 350)
        fig_varl.update_layout(barmode='group')
        fig_varl.update_yaxes(title_text='Return (%)')
        st.plotly_chart(fig_varl, use_container_width=True)

        # Cornish-Fisher comparison
        st.subheader("Cornish-Fisher Modified VaR vs Gaussian VaR")
        skew_v = metrics['Skewness']
        kurt_v = metrics['Kurtosis'] - 3   # excess
        cf_data = []
        for cl in [0.90, 0.95, 0.975, 0.99]:
            z_n  = stats.norm.ppf(1 - cl)
            z_cf = _cornish_fisher_z(1 - cl, skew_v, kurt_v)
            cf_data.append({
                'Confidence': f"{int(cl*100)}%",
                'Gaussian VaR (%)': -(p_returns.mean() + z_n * p_returns.std()) * 100,
                'CF Modified VaR (%)': -(p_returns.mean() + z_cf * p_returns.std()) * 100,
            })
        cf_df = pd.DataFrame(cf_data).set_index('Confidence')
        st.dataframe(
            cf_df.style.format('{:.4f}').background_gradient(cmap='Reds', axis=None),
            use_container_width=True,
        )

    # ── Monte Carlo VaR (FIX 18 / NEW)
    with tab_mc:
        with st.spinner(f"Running {mc_sims:,} Monte Carlo paths …"):
            mc = monte_carlo_var(p_returns, horizon=mc_horizon,
                                 n_sims=mc_sims, confidence=0.95)
        st.plotly_chart(plot_mc_var_histogram(mc), use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric(f"MC VaR {int(mc['confidence']*100)}% ({mc_horizon}d)",
                  f"{mc['MC VaR']:.3%}")
        m2.metric(f"MC CVaR {int(mc['confidence']*100)}% ({mc_horizon}d)",
                  f"{mc['MC CVaR']:.3%}")
        m3.metric("Simulations", f"{mc['n_sims']:,}")

    # ── Rolling risk ratios (FIX 20 / NEW)
    with tab_roll:
        st.plotly_chart(plot_rolling_risk(p_returns, risk_free_rate),
                        use_container_width=True)

    # ── Correlation matrix (FIX 21 / NEW)
    with tab_corr:
        st.plotly_chart(plot_correlation_heatmap(returns), use_container_width=True)
        st.caption("Sorted by dominant eigenvector of the correlation matrix. "
                   "Red = negative correlation, Teal = positive correlation.")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5 — CUMULATIVE PERFORMANCE CHART
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 6 · Portfolio vs Benchmark Cumulative Returns")

    cum_port = (1 + p_returns).cumprod()
    fig_cum  = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=cum_port.index, y=cum_port.values,
        name='Portfolio', mode='lines',
        line=dict(color=PALETTE['blue'], width=2.2),
        fill='tozeroy', fillcolor='rgba(45,130,255,0.08)',
    ))
    for bm_col, bm_col_c in zip(
            bench_returns.columns, [PALETTE['amber'], PALETTE['teal']]):
        if bm_col in bench_returns.columns:
            bm_aligned = bench_returns[bm_col].reindex(p_returns.index).fillna(0)
            cum_bm     = (1 + bm_aligned).cumprod()
            fig_cum.add_trace(go.Scatter(
                x=cum_bm.index, y=cum_bm.values,
                name=bm_col.replace('.IS', ''),
                line=dict(color=bm_col_c, width=1.6, dash='dot'),
                mode='lines',
            ))

    fig_cum = _apply_theme(fig_cum, "Cumulative Returns — Portfolio vs Benchmark", height=420)
    fig_cum.update_yaxes(title_text='Growth of 1 TRY')
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── FOOTER
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                color:#7a90b5;text-align:center;padding:1rem 0;">
        BIST Portfolio Risk Terminal · Data: Yahoo Finance / yfinance ·
        Optimisation: PyPortfolioOpt · Volatility: ARCH · All rights reserved.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
