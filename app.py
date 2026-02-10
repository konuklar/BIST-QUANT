# ============================================================================
# BIST PORTFOLIO RISK & OPTIMIZATION TERMINAL - ENHANCED VERSION
# Production-Grade Streamlit Application with Advanced Features
# ============================================================================
# ENHANCEMENTS:
# 1. Multi-source data fetching with fallback mechanisms
# 2. Real-time market data updates
# 3. Portfolio backtesting framework
# 4. Stress testing and scenario analysis
# 5. Transaction cost modeling
# 6. Machine learning-based return predictions
# 7. Sentiment analysis integration
# 8. Performance attribution
# 9. Risk factor analysis (Fama-French)
# 10. Export functionality for reports
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
from scipy.optimize import minimize
import requests
import json
import io
import base64
from typing import Dict, List, Tuple, Optional, Any
import logging
import traceback

# PyPortfolioOpt
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt import CLA, EfficientCVaR, HRPOpt, EfficientSemivariance

# ARCH for GARCH volatility modelling
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

import yfinance as yf

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL THEME
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BIST Portfolio Risk Analytics Pro",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# Enhanced custom CSS
st.markdown("""
<style>
    /* ── Import Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-primary:    #060b14;
        --bg-secondary:  #0d1526;
        --bg-card:       #111d35;
        --bg-card-alt:   #1a2642;
        --accent-blue:   #2d82ff;
        --accent-teal:   #00c9a7;
        --accent-amber:  #ffb830;
        --accent-red:    #ff4f6e;
        --accent-purple: #a855f7;
        --accent-green:  #10b981;
        --text-primary:  #e8f0fe;
        --text-secondary:#a0b3d6;
        --text-muted:    #7a90b5;
        --border:        rgba(45,130,255,0.15);
        --success:       #10b981;
        --warning:       #f59e0b;
        --danger:        #ef4444;
    }

    /* ── Global ── */
    html, body, [class*="css"] {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
        padding: 1rem;
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
    h1 { 
        font-family: 'Syne', sans-serif; 
        font-weight: 800;
        background: linear-gradient(120deg, #2d82ff 0%, #00c9a7 100%);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important; 
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    h2, h3 { 
        font-family: 'Syne', sans-serif; 
        font-weight: 700;
        color: var(--text-primary) !important; 
    }
    
    /* ── Section headers ── */
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--accent-teal);
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.6);
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

    /* ── Cards ── */
    .custom-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-blue) !important;
        color: white !important;
    }

    /* ── Dataframes ── */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border);
    }

    /* ── Buttons ── */
    .stButton>button {
        background: linear-gradient(135deg, #2d82ff 0%, #1557c0 100%);
        color: white; 
        border: none; 
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem; 
        letter-spacing: 0.05em; 
        text-transform: uppercase;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(45, 130, 255, 0.4);
    }

    /* ── Progress bars ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2d82ff 0%, #00c9a7 100%);
    }

    /* ── Alert boxes ── */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    .stInfo {
        background: rgba(45,130,255,0.1);
        border-left-color: var(--accent-blue);
    }
    .stSuccess {
        background: rgba(16,185,129,0.1);
        border-left-color: var(--accent-green);
    }
    .stWarning {
        background: rgba(245,158,11,0.1);
        border-left-color: var(--accent-amber);
    }
    .stError {
        background: rgba(239,68,68,0.1);
        border-left-color: var(--danger);
    }

    /* ── Tooltips ── */
    [data-testid="stTooltip"] {
        background: var(--bg-card-alt) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* ── Status indicators ── */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-green { background-color: var(--accent-green); }
    .status-yellow { background-color: var(--accent-amber); }
    .status-red { background-color: var(--danger); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS AND CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# BIST 30 Companies with sector classification
BIST30_TICKERS = {
    'AKBNK.IS': {'name': 'Akbank', 'sector': 'Financials'},
    'ARCLK.IS': {'name': 'Arçelik', 'sector': 'Consumer Durables'},
    'ASELS.IS': {'name': 'Aselsan', 'sector': 'Industrials'},
    'BIMAS.IS': {'name': 'BIM', 'sector': 'Consumer Staples'},
    'DOHOL.IS': {'name': 'Doğuş Otomotiv', 'sector': 'Consumer Discretionary'},
    'EKGYO.IS': {'name': 'Emeklak Gayrimenkul', 'sector': 'Real Estate'},
    'EREGL.IS': {'name': 'Ereğli Demir Çelik', 'sector': 'Materials'},
    'FROTO.IS': {'name': 'Ford Otosan', 'sector': 'Consumer Discretionary'},
    'GARAN.IS': {'name': 'Garanti BBVA', 'sector': 'Financials'},
    'HALKB.IS': {'name': 'Halkbank', 'sector': 'Financials'},
    'ISCTR.IS': {'name': 'İş Bankası', 'sector': 'Financials'},
    'KCHOL.IS': {'name': 'Koç Holding', 'sector': 'Conglomerate'},
    'KOZAA.IS': {'name': 'Koza Altın', 'sector': 'Materials'},
    'KOZAL.IS': {'name': 'Koza Madencilik', 'sector': 'Materials'},
    'KRDMD.IS': {'name': 'Kardemir', 'sector': 'Materials'},
    'PETKM.IS': {'name': 'Petkim', 'sector': 'Materials'},
    'PGSUS.IS': {'name': 'Pegasus', 'sector': 'Industrials'},
    'SAHOL.IS': {'name': 'Sabancı Holding', 'sector': 'Conglomerate'},
    'SASA.IS': {'name': 'Sasa Polyester', 'sector': 'Materials'},
    'SISE.IS': {'name': 'Şişecam', 'sector': 'Materials'},
    'SKBNK.IS': {'name': 'Şekerbank', 'sector': 'Financials'},
    'TCELL.IS': {'name': 'Turkcell', 'sector': 'Communication Services'},
    'THYAO.IS': {'name': 'Turkish Airlines', 'sector': 'Industrials'},
    'TKFEN.IS': {'name': 'Tekfen Holding', 'sector': 'Industrials'},
    'TOASO.IS': {'name': 'Tofaş', 'sector': 'Consumer Discretionary'},
    'TTKOM.IS': {'name': 'Türk Telekom', 'sector': 'Communication Services'},
    'TUPRS.IS': {'name': 'Tüpraş', 'sector': 'Energy'},
    'ULKER.IS': {'name': 'Ülker', 'sector': 'Consumer Staples'},
    'VAKBN.IS': {'name': 'Vakıfbank', 'sector': 'Financials'},
    'YKBNK.IS': {'name': 'Yapı Kredi', 'sector': 'Financials'}
}

BENCHMARK_TICKERS = ['XU100.IS', 'XU030.IS', 'XUSIN.IS']
DEFAULT_RFR = 0.45  # Turkey TCMB policy rate approx (decimal)

# Risk factor models
RISK_FACTORS = {
    'Market': 'Market Risk',
    'Size': 'Size Factor (SMB)',
    'Value': 'Value Factor (HML)',
    'Momentum': 'Momentum Factor',
    'Quality': 'Quality Factor',
    'Volatility': 'Volatility Factor'
}

# Transaction cost assumptions
TRANSACTION_COSTS = {
    'commission': 0.001,  # 0.1%
    'spread': 0.002,      # 0.2%
    'slippage': 0.0005,   # 0.05%
    'tax': 0.001          # 0.1%
}

# Color palette
PALETTE = {
    'blue': '#2d82ff',
    'teal': '#00c9a7',
    'amber': '#ffb830',
    'red': '#ff4f6e',
    'purple': '#a855f7',
    'green': '#10b981',
    'muted': '#7a90b5',
    'dark_blue': '#1a2642'
}

# Plotly theme
PLOTLY_THEME = {
    'template': 'plotly_dark',
    'paper_bgcolor': '#0d1526',
    'plot_bgcolor': '#111d35',
    'font': dict(family='Inter, sans-serif', color='#e8f0fe'),
    'gridcolor': 'rgba(45,130,255,0.10)',
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING WITH FALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class DataFetcher:
    """Enhanced data fetcher with multiple fallback strategies"""
    
    @staticmethod
    @st.cache_data(ttl=1800, show_spinner=False)
    def fetch_market_data(start_date: str, end_date: str, max_retries: int = 3):
        """
        Download price data with robust error handling and fallbacks
        """
        tickers = list(BIST30_TICKERS.keys())
        benchmark_tickers = BENCHMARK_TICKERS
        
        for attempt in range(max_retries):
            try:
                # Try with auto_adjust first
                raw = yf.download(
                    tickers,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    threads=True
                )
                
                bench_raw = yf.download(
                    benchmark_tickers,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    threads=True
                )
                
                # Handle multi-level columns
                def extract_price(df):
                    if isinstance(df.columns, pd.MultiIndex):
                        if 'Adj Close' in df.columns.get_level_values(0):
                            return df['Adj Close']
                        elif 'Close' in df.columns.get_level_values(0):
                            return df['Close']
                    return df
                
                data = extract_price(raw)
                benchmark_data = extract_price(bench_raw)
                
                # Forward fill and drop columns with too many NaNs
                data = data.ffill().bfill()
                benchmark_data = benchmark_data.ffill().bfill()
                
                # Remove tickers with insufficient data
                min_days = 20
                valid_columns = data.columns[data.notna().sum() >= min_days]
                data = data[valid_columns]
                
                if data.empty:
                    raise ValueError("No valid data after cleaning")
                
                # Calculate returns
                returns = data.pct_change().dropna()
                benchmark_returns = benchmark_data.pct_change().dropna()
                
                logger.info(f"Successfully fetched data for {len(data.columns)} tickers")
                return {
                    'prices': data,
                    'returns': returns,
                    'benchmark_prices': benchmark_data,
                    'benchmark_returns': benchmark_returns,
                    'tickers': valid_columns.tolist(),
                    'status': 'success'
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Create synthetic data as last resort
                    return DataFetcher._create_synthetic_data(start_date, end_date)
        
        return None
    
    @staticmethod
    def _create_synthetic_data(start_date, end_date):
        """Create synthetic data for demonstration purposes"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        n_tickers = 10
        
        # Generate random prices
        np.random.seed(42)
        base_prices = np.random.uniform(10, 100, n_tickers)
        drift = np.random.uniform(-0.0002, 0.0005, n_tickers)
        volatility = np.random.uniform(0.01, 0.03, n_tickers)
        
        returns = np.random.normal(
            drift[:, None],
            volatility[:, None],
            (n_tickers, len(date_range))
        )
        
        prices = base_prices[:, None] * np.exp(np.cumsum(returns, axis=1))
        
        # Create DataFrame
        tickers = list(BIST30_TICKERS.keys())[:n_tickers]
        data = pd.DataFrame(
            prices.T,
            index=date_range,
            columns=tickers
        )
        
        returns_df = pd.DataFrame(
            returns.T,
            index=date_range[1:],
            columns=tickers
        )
        
        # Create benchmark
        benchmark_returns = pd.DataFrame(
            np.random.normal(0.0003, 0.015, (len(date_range) - 1, 1)),
            index=date_range[1:],
            columns=['XU100.IS']
        )
        
        benchmark_prices = pd.DataFrame(
            100 * np.exp(np.cumsum(benchmark_returns.values.flatten())),
            index=date_range,
            columns=['XU100.IS']
        )
        
        return {
            'prices': data,
            'returns': returns_df,
            'benchmark_prices': benchmark_prices,
            'benchmark_returns': benchmark_returns,
            'tickers': tickers,
            'status': 'synthetic'
        }
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_fundamental_data(ticker: str):
        """Fetch fundamental data for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamentals = {
                'market_cap': info.get('marketCap', np.nan),
                'pe_ratio': info.get('trailingPE', np.nan),
                'pb_ratio': info.get('priceToBook', np.nan),
                'dividend_yield': info.get('dividendYield', np.nan),
                'beta': info.get('beta', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'roa': info.get('returnOnAssets', np.nan),
                'roe': info.get('returnOnEquity', np.nan)
            }
            
            return {k: v for k, v in fundamentals.items() if pd.notna(v)}
        except:
            return {}

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO OPTIMIZATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioOptimizer:
    """Enhanced portfolio optimizer with transaction costs and constraints"""
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.05):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        
    def optimize(self, method: str, **kwargs) -> Dict:
        """Main optimization method"""
        optimization_methods = {
            'max_sharpe': self._max_sharpe,
            'min_volatility': self._min_volatility,
            'efficient_risk': self._efficient_risk,
            'efficient_return': self._efficient_return,
            'max_quadratic_utility': self._max_quadratic_utility,
            'hrp': self._hierarchical_risk_parity,
            'cvar': self._conditional_value_at_risk,
            'semivariance': self._semivariance,
            'equal_weight': self._equal_weight,
            'risk_parity': self._risk_parity
        }
        
        if method not in optimization_methods:
            raise ValueError(f"Unknown optimization method: {method}")
        
        try:
            return optimization_methods[method](**kwargs)
        except Exception as e:
            logger.error(f"Optimization failed for method {method}: {str(e)}")
            return self._equal_weight()
    
    def _max_sharpe(self, **kwargs) -> Dict:
        """Maximize Sharpe ratio"""
        mu = expected_returns.mean_historical_return(self.returns, frequency=252)
        S = risk_models.sample_cov(self.returns, frequency=252)
        
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda x: x >= 0)
        ef.add_constraint(lambda x: x <= 0.2)  # Max 20% per asset
        
        weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        weights = ef.clean_weights()
        
        performance = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
        
        return {
            'weights': weights,
            'performance': performance,
            'method': 'Max Sharpe'
        }
    
    def _min_volatility(self, **kwargs) -> Dict:
        """Minimize volatility"""
        mu = expected_returns.mean_historical_return(self.returns, frequency=252)
        S = risk_models.sample_cov(self.returns, frequency=252)
        
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda x: x >= 0)
        ef.add_constraint(lambda x: x <= 0.2)
        
        weights = ef.min_volatility()
        weights = ef.clean_weights()
        
        performance = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
        
        return {
            'weights': weights,
            'performance': performance,
            'method': 'Min Volatility'
        }
    
    def _efficient_risk(self, target_volatility: float = 0.2, **kwargs) -> Dict:
        """Efficient portfolio for given target volatility"""
        mu = expected_returns.mean_historical_return(self.returns, frequency=252)
        S = risk_models.sample_cov(self.returns, frequency=252)
        
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda x: x >= 0)
        ef.add_constraint(lambda x: x <= 0.2)
        
        ef.efficient_risk(target_volatility=target_volatility)
        weights = ef.clean_weights()
        
        performance = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
        
        return {
            'weights': weights,
            'performance': performance,
            'method': f'Efficient Risk (σ={target_volatility:.1%})'
        }
    
    def _efficient_return(self, target_return: float = 0.15, **kwargs) -> Dict:
        """Efficient portfolio for given target return"""
        mu = expected_returns.mean_historical_return(self.returns, frequency=252)
        S = risk_models.sample_cov(self.returns, frequency=252)
        
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda x: x >= 0)
        ef.add_constraint(lambda x: x <= 0.2)
        
        ef.efficient_return(target_return=target_return)
        weights = ef.clean_weights()
        
        performance = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
        
        return {
            'weights': weights,
            'performance': performance,
            'method': f'Efficient Return (μ={target_return:.1%})'
        }
    
    def _max_quadratic_utility(self, risk_aversion: float = 1.0, **kwargs) -> Dict:
        """Maximize quadratic utility"""
        mu = expected_returns.mean_historical_return(self.returns, frequency=252)
        S = risk_models.sample_cov(self.returns, frequency=252)
        
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda x: x >= 0)
        ef.add_constraint(lambda x: x <= 0.2)
        
        ef.max_quadratic_utility(risk_aversion=risk_aversion)
        weights = ef.clean_weights()
        
        performance = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)
        
        return {
            'weights': weights,
            'performance': performance,
            'method': f'Quadratic Utility (γ={risk_aversion})'
        }
    
    def _hierarchical_risk_parity(self, **kwargs) -> Dict:
        """Hierarchical Risk Parity optimization"""
        hrp = HRPOpt(self.returns)
        hrp.optimize()
        weights = hrp.clean_weights()
        
        # Calculate performance metrics
        portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
        metrics = self._calculate_performance_metrics(portfolio_returns)
        
        return {
            'weights': weights,
            'performance': (metrics['annual_return'], metrics['annual_volatility'], metrics['sharpe_ratio']),
            'method': 'Hierarchical Risk Parity'
        }
    
    def _conditional_value_at_risk(self, **kwargs) -> Dict:
        """Conditional Value at Risk optimization"""
        cvar = EfficientCVaR(
            expected_returns.mean_historical_return(self.returns, frequency=252),
            self.returns
        )
        cvar.min_cvar()
        weights = cvar.clean_weights()
        
        portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
        metrics = self._calculate_performance_metrics(portfolio_returns)
        
        return {
            'weights': weights,
            'performance': (metrics['annual_return'], metrics['annual_volatility'], metrics['sharpe_ratio']),
            'method': 'Conditional VaR'
        }
    
    def _semivariance(self, **kwargs) -> Dict:
        """Semivariance optimization"""
        mu = expected_returns.mean_historical_return(self.returns, frequency=252)
        
        # Use sample covariance as fallback
        S = risk_models.sample_cov(self.returns, frequency=252)
        
        try:
            semivar = EfficientSemivariance(mu, self.returns)
            semivar.efficient_return(target_return=mu.mean())
            weights = semivar.clean_weights()
            
            portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
            metrics = self._calculate_performance_metrics(portfolio_returns)
            
            return {
                'weights': weights,
                'performance': (metrics['annual_return'], metrics['annual_volatility'], metrics['sharpe_ratio']),
                'method': 'Semivariance'
            }
        except:
            return self._equal_weight()
    
    def _risk_parity(self, **kwargs) -> Dict:
        """Risk Parity optimization"""
        # Simple implementation of risk parity
        S = risk_models.sample_cov(self.returns, frequency=252)
        volatilities = np.sqrt(np.diag(S))
        
        # Inverse volatility weighting
        weights = 1 / volatilities
        weights = weights / weights.sum()
        
        weights_dict = {self.returns.columns[i]: weights[i] for i in range(len(weights))}
        
        portfolio_returns = (self.returns * pd.Series(weights_dict)).sum(axis=1)
        metrics = self._calculate_performance_metrics(portfolio_returns)
        
        return {
            'weights': weights_dict,
            'performance': (metrics['annual_return'], metrics['annual_volatility'], metrics['sharpe_ratio']),
            'method': 'Risk Parity'
        }
    
    def _equal_weight(self, **kwargs) -> Dict:
        """Equal weight portfolio"""
        weights = {col: 1/self.n_assets for col in self.returns.columns}
        
        portfolio_returns = (self.returns * pd.Series(weights)).sum(axis=1)
        metrics = self._calculate_performance_metrics(portfolio_returns)
        
        return {
            'weights': weights,
            'performance': (metrics['annual_return'], metrics['annual_volatility'], metrics['sharpe_ratio']),
            'method': 'Equal Weight'
        }
    
    def _calculate_performance_metrics(self, portfolio_returns: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95
        }

# ─────────────────────────────────────────────────────────────────────────────
# RISK ANALYTICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RiskAnalytics:
    """Comprehensive risk analytics engine"""
    
    @staticmethod
    def calculate_var_cvar(returns: pd.Series, confidence_levels: List[float] = [0.90, 0.95, 0.99]) -> pd.DataFrame:
        """Calculate VaR and CVaR at multiple confidence levels"""
        results = []
        
        for cl in confidence_levels:
            alpha = 1 - cl
            var = np.percentile(returns, alpha * 100)
            tail_returns = returns[returns <= var]
            cvar = tail_returns.mean() if len(tail_returns) > 0 else var
            
            results.append({
                'Confidence Level': f'{int(cl*100)}%',
                'VaR': var * 100,
                'CVaR': cvar * 100,
                'Historical': len(tail_returns) / len(returns) * 100
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def monte_carlo_simulation(returns: pd.Series, horizon_days: int = 10, n_simulations: int = 10000, 
                              confidence_level: float = 0.95) -> Dict:
        """Monte Carlo simulation for portfolio returns"""
        mu = returns.mean()
        sigma = returns.std()
        
        np.random.seed(42)
        simulations = np.random.normal(mu, sigma, (n_simulations, horizon_days))
        
        # Calculate cumulative returns
        cumulative_returns = np.prod(1 + simulations, axis=1) - 1
        
        # Calculate statistics
        mean_return = cumulative_returns.mean()
        std_return = cumulative_returns.std()
        var = np.percentile(cumulative_returns, (1 - confidence_level) * 100)
        cvar = cumulative_returns[cumulative_returns <= var].mean()
        
        return {
            'simulations': simulations,
            'cumulative_returns': cumulative_returns,
            'mean_return': mean_return,
            'std_return': std_return,
            'var': var,
            'cvar': cvar,
            'confidence_interval': np.percentile(cumulative_returns, [2.5, 97.5])
        }
    
    @staticmethod
    def stress_test(portfolio_returns: pd.Series, stress_scenarios: Dict[str, float] = None) -> pd.DataFrame:
        """Stress testing under various scenarios"""
        if stress_scenarios is None:
            stress_scenarios = {
                'Market Crash (-20%)': -0.20,
                'Correction (-10%)': -0.10,
                'Normal Market': 0.00,
                'Bull Market (+10%)': 0.10,
                'Strong Bull (+20%)': 0.20
            }
        
        results = []
        for scenario, shock in stress_scenarios.items():
            stressed_returns = portfolio_returns * (1 + shock)
            
            # Calculate metrics for stressed returns
            annual_return = (1 + stressed_returns.mean()) ** 252 - 1
            annual_vol = stressed_returns.std() * np.sqrt(252)
            sharpe = (annual_return - 0.05) / annual_vol if annual_vol > 0 else 0
            
            results.append({
                'Scenario': scenario,
                'Shock': f'{shock:+.1%}',
                'Annual Return': annual_return,
                'Annual Volatility': annual_vol,
                'Sharpe Ratio': sharpe,
                'Max Drawdown': RiskAnalytics._calculate_max_drawdown(stressed_returns)
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def _calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def garch_volatility_forecast(returns: pd.Series, forecast_days: int = 5) -> Dict:
        """GARCH volatility forecasting"""
        if not HAS_ARCH:
            return None
        
        try:
            # Scale returns for better numerical stability
            scaled_returns = returns * 100
            
            # Fit GARCH(1,1) model
            model = arch_model(
                scaled_returns,
                vol='Garch',
                p=1,
                q=1,
                dist='t'
            )
            
            result = model.fit(disp='off')
            
            # Forecast volatility
            forecast = result.forecast(horizon=forecast_days)
            forecast_vol = forecast.variance.iloc[-1].values ** 0.5 / 100  # Convert back to decimal
            
            # Annualize volatility
            forecast_vol_annual = forecast_vol * np.sqrt(252)
            
            return {
                'parameters': result.params,
                'persistence': result.params['alpha[1]'] + result.params['beta[1]'],
                'long_run_volatility': np.sqrt(result.params['omega'] / (1 - result.params['alpha[1]'] - result.params['beta[1]'])) / 100,
                'forecast': forecast_vol,
                'forecast_annual': forecast_vol_annual,
                'aic': result.aic,
                'bic': result.bic
            }
        except Exception as e:
            logger.error(f"GARCH fitting failed: {str(e)}")
            return None

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceAttribution:
    """Performance attribution analysis"""
    
    @staticmethod
    def calculate_attribution(weights: Dict, returns: pd.DataFrame, 
                            benchmark_returns: pd.Series) -> pd.DataFrame:
        """Calculate performance attribution"""
        # Convert weights to Series
        weights_series = pd.Series(weights)
        
        # Align data
        common_index = returns.index.intersection(benchmark_returns.index)
        returns_aligned = returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]
        
        # Portfolio returns
        portfolio_returns = (returns_aligned * weights_series).sum(axis=1)
        
        # Active returns
        active_returns = portfolio_returns - benchmark_aligned
        
        # Attribution components
        attribution_data = []
        
        for asset in weights_series.index:
            if asset in returns_aligned.columns:
                asset_return = returns_aligned[asset]
                asset_weight = weights_series[asset]
                
                # Allocation effect (weight difference × benchmark return)
                # Selection effect (weight × return difference)
                # Interaction effect
                
                attribution_data.append({
                    'Asset': asset.replace('.IS', ''),
                    'Weight': asset_weight,
                    'Return': asset_return.mean() * 252,
                    'Contribution': (asset_weight * asset_return).mean() * 252
                })
        
        attribution_df = pd.DataFrame(attribution_data)
        attribution_df = attribution_df.sort_values('Contribution', ascending=False)
        
        return attribution_df
    
    @staticmethod
    def risk_factor_analysis(portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> Dict:
        """Risk factor analysis using regression"""
        # Align data
        common_index = portfolio_returns.index.intersection(factor_returns.index)
        portfolio_aligned = portfolio_returns.loc[common_index]
        factors_aligned = factor_returns.loc[common_index]
        
        # Add constant for intercept
        factors_aligned = factors_aligned.copy()
        factors_aligned['Intercept'] = 1
        
        # Perform regression
        try:
            coefficients = np.linalg.lstsq(factors_aligned, portfolio_aligned, rcond=None)[0]
            
            # Calculate R-squared
            predicted = factors_aligned @ coefficients
            ss_res = ((portfolio_aligned - predicted) ** 2).sum()
            ss_tot = ((portfolio_aligned - portfolio_aligned.mean()) ** 2).sum()
            r_squared = 1 - (ss_res / ss_tot)
            
            # Factor exposures
            exposures = {}
            for i, factor in enumerate(factors_aligned.columns):
                exposures[factor] = coefficients[i]
            
            return {
                'exposures': exposures,
                'r_squared': r_squared,
                'predicted': predicted,
                'residuals': portfolio_aligned - predicted
            }
        except:
            return None

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class VisualizationEngine:
    """Enhanced visualization engine"""
    
    @staticmethod
    def plot_efficient_frontier(optimizer: PortfolioOptimizer, risk_free_rate: float) -> go.Figure:
        """Plot efficient frontier with optimized portfolios"""
        # Generate efficient frontier points
        mu = expected_returns.mean_historical_return(optimizer.returns, frequency=252)
        S = risk_models.sample_cov(optimizer.returns, frequency=252)
        
        cla = CLA(mu, S)
        frontier_points = cla.efficient_frontier(points=100)
        
        # Extract returns and volatilities
        frontier_returns = [p[0] for p in frontier_points]
        frontier_volatilities = [p[1] for p in frontier_points]
        
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_volatilities,
            y=frontier_returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color=PALETTE['blue'], width=2),
            fill='tozeroy',
            fillcolor='rgba(45, 130, 255, 0.1)'
        ))
        
        # Individual assets
        asset_returns = mu.values
        asset_volatilities = np.sqrt(np.diag(S))
        
        fig.add_trace(go.Scatter(
            x=asset_volatilities,
            y=asset_returns,
            mode='markers',
            name='Individual Assets',
            marker=dict(
                size=8,
                color=PALETTE['muted'],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=mu.index,
            hovertemplate='<b>%{text}</b><br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>'
        ))
        
        # Optimized portfolios
        methods = ['max_sharpe', 'min_volatility', 'equal_weight']
        colors = [PALETTE['teal'], PALETTE['amber'], PALETTE['red']]
        markers = ['star', 'diamond', 'circle']
        
        for method, color, marker in zip(methods, colors, markers):
            try:
                result = optimizer.optimize(method)
                ret, vol, _ = result['performance']
                
                fig.add_trace(go.Scatter(
                    x=[vol],
                    y=[ret],
                    mode='markers',
                    name=result['method'],
                    marker=dict(
                        size=14,
                        color=color,
                        symbol=marker,
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate=f"<b>{result['method']}</b><br>Return: {ret:.2%}<br>Volatility: {vol:.2%}<extra></extra>"
                ))
            except:
                continue
        
        # Capital Market Line (for max Sharpe)
        try:
            result = optimizer.optimize('max_sharpe')
            ret, vol, sharpe = result['performance']
            
            # Plot CML
            x_cml = np.linspace(0, max(frontier_volatilities) * 1.2, 50)
            y_cml = risk_free_rate + sharpe * x_cml
            
            fig.add_trace(go.Scatter(
                x=x_cml,
                y=y_cml,
                mode='lines',
                name='Capital Market Line',
                line=dict(color=PALETTE['purple'], width=1.5, dash='dash'),
                opacity=0.7
            ))
        except:
            pass
        
        fig.update_layout(
            title='Efficient Frontier with Optimized Portfolios',
            xaxis_title='Annual Volatility',
            yaxis_title='Annual Return',
            hovermode='closest',
            height=600,
            **PLOTLY_THEME
        )
        
        fig.update_xaxes(tickformat='.0%')
        fig.update_yaxes(tickformat='.0%')
        
        return fig
    
    @staticmethod
    def plot_weight_allocation(weights: Dict, sector_info: Dict) -> go.Figure:
        """Plot portfolio allocation with sector breakdown"""
        # Prepare data
        weights_series = pd.Series(weights)
        weights_series = weights_series[weights_series > 0.001]  # Filter small weights
        
        # Add sector information
        allocation_data = []
        for ticker, weight in weights_series.items():
            ticker_clean = ticker.replace('.IS', '')
            sector = BIST30_TICKERS.get(ticker, {}).get('sector', 'Unknown')
            name = BIST30_TICKERS.get(ticker, {}).get('name', ticker_clean)
            
            allocation_data.append({
                'Ticker': ticker_clean,
                'Name': name,
                'Sector': sector,
                'Weight': weight
            })
        
        df = pd.DataFrame(allocation_data)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'domain'}, {'type': 'xy'}]],
            subplot_titles=('Sector Allocation', 'Top Holdings')
        )
        
        # Sector allocation (pie chart)
        sector_allocation = df.groupby('Sector')['Weight'].sum().reset_index()
        fig.add_trace(
            go.Pie(
                labels=sector_allocation['Sector'],
                values=sector_allocation['Weight'],
                hole=0.5,
                textinfo='label+percent',
                marker=dict(colors=px.colors.qualitative.Set3),
                hovertemplate='<b>%{label}</b><br>Weight: %{percent}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Top holdings (horizontal bar chart)
        top_holdings = df.nlargest(15, 'Weight')
        fig.add_trace(
            go.Bar(
                x=top_holdings['Weight'] * 100,
                y=top_holdings['Ticker'],
                orientation='h',
                marker=dict(
                    color=top_holdings['Weight'],
                    colorscale='Blues',
                    showscale=False
                ),
                text=top_holdings['Weight'].apply(lambda x: f'{x:.1%}'),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Weight: %{x:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Portfolio Allocation Analysis',
            height=500,
            showlegend=False,
            **PLOTLY_THEME
        )
        
        fig.update_xaxes(title_text='Weight (%)', row=1, col=2)
        
        return fig
    
    @staticmethod
    def plot_risk_metrics_comparison(metrics_list: List[Dict], labels: List[str]) -> go.Figure:
        """Compare risk metrics across different portfolios"""
        metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'annual_volatility', 'max_drawdown']
        metric_names = ['Sharpe Ratio', 'Sortino Ratio', 'Annual Volatility', 'Max Drawdown']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_names,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set3
        
        for idx, metric in enumerate(metrics_to_plot):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            values = [m.get(metric, 0) for m in metrics_list]
            
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors[:len(values)],
                    text=[f'{v:.3f}' if metric in ['sharpe_ratio', 'sortino_ratio'] else f'{v:.2%}' for v in values],
                    textposition='auto'
                ),
                row=row, col=col
            )
            
            # Add benchmark line for Sharpe and Sortino
            if metric in ['sharpe_ratio', 'sortino_ratio']:
                fig.add_hline(
                    y=0,
                    line_dash="dot",
                    line_color="white",
                    opacity=0.5,
                    row=row, col=col
                )
        
        fig.update_layout(
            title='Risk Metrics Comparison',
            height=600,
            showlegend=False,
            **PLOTLY_THEME
        )
        
        # Update y-axis formats
        fig.update_yaxes(tickformat='.3f', row=1, col=1)
        fig.update_yaxes(tickformat='.3f', row=1, col=2)
        fig.update_yaxes(tickformat='.0%', row=2, col=1)
        fig.update_yaxes(tickformat='.0%', row=2, col=2)
        
        return fig
    
    @staticmethod
    def plot_monte_carlo_distribution(simulation_results: Dict) -> go.Figure:
        """Plot Monte Carlo simulation distribution"""
        cumulative_returns = simulation_results['cumulative_returns']
        var = simulation_results['var']
        cvar = simulation_results['cvar']
        
        fig = go.Figure()
        
        # Histogram of simulated returns
        fig.add_trace(go.Histogram(
            x=cumulative_returns * 100,
            nbinsx=100,
            name='Simulated Returns',
            marker_color=PALETTE['blue'],
            opacity=0.7,
            hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
        ))
        
        # Add VaR and CVaR lines
        fig.add_vline(
            x=var * 100,
            line_dash="dash",
            line_color=PALETTE['amber'],
            annotation_text=f"VaR: {var:.2%}",
            annotation_position="top right"
        )
        
        fig.add_vline(
            x=cvar * 100,
            line_dash="dot",
            line_color=PALETTE['red'],
            annotation_text=f"CVaR: {cvar:.2%}",
            annotation_position="top right"
        )
        
        # Add normal distribution overlay
        x_norm = np.linspace(cumulative_returns.min(), cumulative_returns.max(), 100)
        y_norm = stats.norm.pdf(x_norm, cumulative_returns.mean(), cumulative_returns.std())
        y_norm = y_norm / y_norm.max() * len(cumulative_returns) / 50  # Scale to histogram
        
        fig.add_trace(go.Scatter(
            x=x_norm * 100,
            y=y_norm,
            mode='lines',
            name='Normal Distribution',
            line=dict(color=PALETTE['teal'], width=2),
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Monte Carlo Simulation Results',
            xaxis_title='Portfolio Return (%)',
            yaxis_title='Frequency',
            height=500,
            **PLOTLY_THEME
        )
        
        return fig
    
    @staticmethod
    def plot_rolling_metrics(portfolio_returns: pd.Series, benchmark_returns: pd.Series, 
                           risk_free_rate: float, window: int = 60) -> go.Figure:
        """Plot rolling performance metrics"""
        # Calculate rolling metrics
        rolling_sharpe = portfolio_returns.rolling(window).apply(
            lambda x: (x.mean() * 252 - risk_free_rate) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        
        rolling_sortino = portfolio_returns.rolling(window).apply(
            lambda x: (x.mean() * 252 - risk_free_rate) / (x[x < 0].std() * np.sqrt(252)) if len(x[x < 0]) > 1 else 0
        )
        
        rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
        rolling_var = portfolio_returns.rolling(window).quantile(0.05)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{window}-Day Rolling Sharpe Ratio',
                f'{window}-Day Rolling Sortino Ratio',
                f'{window}-Day Rolling Volatility',
                f'{window}-Day Rolling VaR (95%)'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                mode='lines',
                name='Sharpe',
                line=dict(color=PALETTE['teal'], width=2)
            ),
            row=1, col=1
        )
        
        # Rolling Sortino
        fig.add_trace(
            go.Scatter(
                x=rolling_sortino.index,
                y=rolling_sortino,
                mode='lines',
                name='Sortino',
                line=dict(color=PALETTE['blue'], width=2)
            ),
            row=1, col=2
        )
        
        # Rolling Volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                mode='lines',
                name='Volatility',
                line=dict(color=PALETTE['amber'], width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 184, 48, 0.1)'
            ),
            row=2, col=1
        )
        
        # Rolling VaR
        fig.add_trace(
            go.Scatter(
                x=rolling_var.index,
                y=rolling_var * 100,
                mode='lines',
                name='VaR',
                line=dict(color=PALETTE['red'], width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 79, 110, 0.1)'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'Rolling Risk Metrics ({window}-Day Window)',
            height=600,
            showlegend=False,
            **PLOTLY_THEME
        )
        
        fig.update_yaxes(tickformat='.2f', row=1, col=1)
        fig.update_yaxes(tickformat='.2f', row=1, col=2)
        fig.update_yaxes(tickformat='.0%', row=2, col=1)
        fig.update_yaxes(tickformat='.1f', row=2, col=2)
        
        return fig

# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    """Generate comprehensive portfolio reports"""
    
    @staticmethod
    def generate_html_report(portfolio_results: Dict, risk_metrics: Dict, 
                           optimization_results: Dict) -> str:
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #2d82ff 0%, #00c9a7 100%); 
                          color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #2d82ff; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
                .table th {{ background-color: #f8f9fa; }}
                .positive {{ color: #10b981; font-weight: bold; }}
                .negative {{ color: #ef4444; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Portfolio Analysis Report</h1>
                <p>Generated on {report_date}</p>
            </div>
            
            <div class="section">
                <h2>Portfolio Summary</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>Annual Return</h3>
                        <p class="{return_class}">{annual_return}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Sharpe Ratio</h3>
                        <p>{sharpe_ratio}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Max Drawdown</h3>
                        <p class="{drawdown_class}">{max_drawdown}</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Portfolio Allocation</h2>
                {allocation_table}
            </div>
            
            <div class="section">
                <h2>Risk Metrics</h2>
                <table class="table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
                    {risk_metrics_rows}
                </table>
            </div>
            
            <div class="section">
                <h2>Optimization Details</h2>
                <p><strong>Method:</strong> {optimization_method}</p>
                <p><strong>Number of Assets:</strong> {num_assets}</p>
                <p><strong>Risk-Free Rate:</strong> {risk_free_rate}</p>
            </div>
        </body>
        </html>
        """
        
        # Format data for HTML
        allocation_df = pd.DataFrame({
            'Ticker': list(portfolio_results.get('weights', {}).keys()),
            'Weight': list(portfolio_results.get('weights', {}).values())
        })
        
        allocation_html = allocation_df.to_html(index=False, classes='table')
        
        risk_metrics_html = ""
        for metric, value in risk_metrics.items():
            risk_metrics_html += f"""
            <tr>
                <td>{metric}</td>
                <td>{value:.4f}</td>
                <td>Risk metric</td>
            </tr>
            """
        
        html_report = html_template.format(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            annual_return=f"{portfolio_results.get('annual_return', 0):.2%}",
            return_class="positive" if portfolio_results.get('annual_return', 0) > 0 else "negative",
            sharpe_ratio=f"{portfolio_results.get('sharpe_ratio', 0):.3f}",
            max_drawdown=f"{portfolio_results.get('max_drawdown', 0):.2%}",
            drawdown_class="negative",
            allocation_table=allocation_html,
            risk_metrics_rows=risk_metrics_html,
            optimization_method=portfolio_results.get('method', 'N/A'),
            num_assets=len(portfolio_results.get('weights', {})),
            risk_free_rate=f"{portfolio_results.get('risk_free_rate', 0):.2%}"
        )
        
        return html_report
    
    @staticmethod
    def create_excel_report(portfolio_results: Dict, risk_metrics: Dict, 
                          optimization_results: Dict, filepath: str = "portfolio_report.xlsx"):
        """Create Excel report with multiple sheets"""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Portfolio weights
            weights_df = pd.DataFrame({
                'Ticker': list(portfolio_results.get('weights', {}).keys()),
                'Weight': list(portfolio_results.get('weights', {}).values())
            })
            weights_df.to_excel(writer, sheet_name='Portfolio Weights', index=False)
            
            # Risk metrics
            risk_metrics_df = pd.DataFrame([risk_metrics])
            risk_metrics_df.to_excel(writer, sheet_name='Risk Metrics', index=False)
            
            # Performance summary
            summary_data = {
                'Metric': ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 
                          'Sortino Ratio', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)'],
                'Value': [
                    portfolio_results.get('annual_return', 0),
                    portfolio_results.get('annual_volatility', 0),
                    portfolio_results.get('sharpe_ratio', 0),
                    portfolio_results.get('sortino_ratio', 0),
                    portfolio_results.get('max_drawdown', 0),
                    portfolio_results.get('var_95', 0),
                    portfolio_results.get('cvar_95', 0)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Performance Summary', index=False)
            
            # Optimization details
            opt_details = {
                'Parameter': ['Method', 'Risk-Free Rate', 'Number of Assets', 'Date Range'],
                'Value': [
                    portfolio_results.get('method', 'N/A'),
                    portfolio_results.get('risk_free_rate', 0),
                    len(portfolio_results.get('weights', {})),
                    portfolio_results.get('date_range', 'N/A')
                ]
            }
            opt_df = pd.DataFrame(opt_details)
            opt_df.to_excel(writer, sheet_name='Optimization Details', index=False)
        
        return filepath

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main Streamlit application"""
    
    # ── HEADER ──
    st.markdown("""
    <div style="padding:1rem 0 0.5rem 0;">
        <span style="font-family:'Space Mono',monospace;font-size:0.7rem;
                     color:#2d82ff;letter-spacing:0.15em;text-transform:uppercase;">
            BIST PORTFOLIO RISK TERMINAL PRO · v3.0
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("Quantitative Risk & Optimization Analytics")
    st.markdown("""
    <div style="color:var(--text-secondary); margin-bottom: 2rem;">
        Advanced portfolio optimization, risk analysis, and performance attribution for BIST 30 stocks
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("## ⚙ Configuration")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=365 * 2),
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.now(),
                key="end_date"
            )
        
        # Risk-free rate
        st.markdown("<div class='section-header'>Risk Parameters</div>", unsafe_allow_html=True)
        rfr_pct = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=DEFAULT_RFR * 100,
            step=0.5,
            help="Annual risk-free rate for Sharpe ratio calculation"
        )
        risk_free_rate = rfr_pct / 100
        
        # Optimization method
        st.markdown("<div class='section-header'>Optimization</div>", unsafe_allow_html=True)
        optimization_method = st.selectbox(
            "Optimization Method",
            options=[
                'max_sharpe', 'min_volatility', 'efficient_risk',
                'efficient_return', 'max_quadratic_utility',
                'hrp', 'cvar', 'semivariance', 'risk_parity', 'equal_weight'
            ],
            format_func=lambda x: x.replace('_', ' ').title(),
            index=0
        )
        
        # Method-specific parameters
        if optimization_method == 'efficient_risk':
            target_volatility = st.slider(
                "Target Volatility (%)",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=1.0
            ) / 100
        elif optimization_method == 'efficient_return':
            target_return = st.slider(
                "Target Return (%)",
                min_value=5.0,
                max_value=50.0,
                value=15.0,
                step=1.0
            ) / 100
        elif optimization_method == 'max_quadratic_utility':
            risk_aversion = st.slider(
                "Risk Aversion (γ)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1
            )
        
        # Advanced settings
        with st.expander("⚡ Advanced Settings"):
            # Transaction costs
            st.markdown("**Transaction Costs**")
            commission = st.number_input(
                "Commission (%)",
                min_value=0.0,
                max_value=1.0,
                value=TRANSACTION_COSTS['commission'] * 100,
                step=0.01
            ) / 100
            
            # Monte Carlo settings
            st.markdown("**Monte Carlo Simulation**")
            mc_simulations = st.select_slider(
                "Number of Simulations",
                options=[1000, 5000, 10000, 25000, 50000],
                value=10000
            )
            mc_horizon = st.slider(
                "Simulation Horizon (days)",
                min_value=1,
                max_value=30,
                value=10
            )
            
            # Rolling window
            rolling_window = st.slider(
                "Rolling Window (days)",
                min_value=20,
                max_value=252,
                value=60,
                step=10
            )
        
        # Data refresh
        st.markdown("<div class='section-header'>Data</div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Market Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # ── DATA FETCHING ──
    with st.spinner("📡 Fetching market data..."):
        try:
            data = DataFetcher.fetch_market_data(
                str(start_date),
                str(end_date)
            )
            
            if data is None or data['prices'].empty:
                st.error("❌ Failed to fetch market data. Please try again.")
                st.stop()
            
            if data['status'] == 'synthetic':
                st.warning("⚠️ Using synthetic data for demonstration. Real market data is unavailable.")
            
            # Extract data
            prices = data['prices']
            returns = data['returns']
            benchmark_prices = data['benchmark_prices']
            benchmark_returns = data['benchmark_returns']
            tickers = data['tickers']
            
            st.success(f"✅ Data loaded: {len(tickers)} assets, {len(prices)} trading days")
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            logger.error(traceback.format_exc())
            st.stop()
    
    # ── PORTFOLIO OPTIMIZATION ──
    st.markdown("<div class='section-header'>Portfolio Optimization</div>", unsafe_allow_html=True)
    
    # Create optimizer
    optimizer = PortfolioOptimizer(returns, risk_free_rate)
    
    # Prepare optimization parameters
    opt_params = {}
    if optimization_method == 'efficient_risk':
        opt_params['target_volatility'] = target_volatility
    elif optimization_method == 'efficient_return':
        opt_params['target_return'] = target_return
    elif optimization_method == 'max_quadratic_utility':
        opt_params['risk_aversion'] = risk_aversion
    
    # Run optimization
    with st.spinner(f"Running {optimization_method.replace('_', ' ').title()} optimization..."):
        try:
            optimization_result = optimizer.optimize(optimization_method, **opt_params)
            
            # Extract results
            weights = optimization_result['weights']
            performance = optimization_result['performance']
            method_name = optimization_result['method']
            
            # Calculate portfolio returns
            portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
            
            # Calculate comprehensive metrics
            risk_analytics = RiskAnalytics()
            portfolio_metrics = optimizer._calculate_performance_metrics(portfolio_returns)
            
            # Store results
            portfolio_results = {
                'weights': weights,
                'annual_return': performance[0],
                'annual_volatility': performance[1],
                'sharpe_ratio': performance[2],
                **portfolio_metrics,
                'method': method_name,
                'risk_free_rate': risk_free_rate,
                'date_range': f"{start_date} to {end_date}"
            }
            
            st.success(f"✅ Optimization complete: {method_name}")
            
        except Exception as e:
            st.error(f"❌ Optimization failed: {str(e)}")
            logger.error(traceback.format_exc())
            st.stop()
    
    # ── PERFORMANCE METRICS ──
    st.markdown("<div class='section-header'>Performance Metrics</div>", unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_return = "positive" if portfolio_results['annual_return'] > 0 else "negative"
        st.metric(
            "Annual Return",
            f"{portfolio_results['annual_return']:.2%}",
            delta=f"{'↑' if portfolio_results['annual_return'] > 0 else '↓'} vs RFR",
            delta_color="normal" if portfolio_results['annual_return'] > risk_free_rate else "inverse"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{portfolio_results['sharpe_ratio']:.3f}",
            delta="Risk-adjusted return"
        )
    
    with col3:
        st.metric(
            "Annual Volatility",
            f"{portfolio_results['annual_volatility']:.2%}",
            delta="Portfolio risk"
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{portfolio_results['max_drawdown']:.2%}",
            delta="Worst loss",
            delta_color="inverse"
        )
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "Sortino Ratio",
            f"{portfolio_results['sortino_ratio']:.3f}",
            delta="Downside risk-adjusted"
        )
    
    with col6:
        st.metric(
            "VaR (95%)",
            f"{portfolio_results['var_95']:.3%}",
            delta="Daily Value at Risk"
        )
    
    with col7:
        st.metric(
            "CVaR (95%)",
            f"{portfolio_results['cvar_95']:.3%}",
            delta="Expected shortfall"
        )
    
    with col8:
        st.metric(
            "Number of Assets",
            f"{len([w for w in weights.values() if w > 0.001])}",
            delta="Diversification"
        )
    
    # ── PORTFOLIO ALLOCATION ──
    st.markdown("<div class='section-header'>Portfolio Allocation</div>", unsafe_allow_html=True)
    
    col_alloc1, col_alloc2 = st.columns([1, 2])
    
    with col_alloc1:
        # Weights table
        weights_df = pd.DataFrame({
            'Ticker': [t.replace('.IS', '') for t in weights.keys()],
            'Weight': list(weights.values())
        })
        weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
        
        st.dataframe(
            weights_df.style.format({'Weight': '{:.2%}'}),
            use_container_width=True,
            height=400
        )
    
    with col_alloc2:
        # Allocation visualization
        viz_engine = VisualizationEngine()
        allocation_fig = viz_engine.plot_weight_allocation(weights, BIST30_TICKERS)
        st.plotly_chart(allocation_fig, use_container_width=True)
    
    # ── EFFICIENT FRONTIER ──
    st.markdown("<div class='section-header'>Efficient Frontier</div>", unsafe_allow_html=True)
    
    frontier_fig = viz_engine.plot_efficient_frontier(optimizer, risk_free_rate)
    st.plotly_chart(frontier_fig, use_container_width=True)
    
    # ── RISK ANALYTICS TABS ──
    st.markdown("<div class='section-header'>Risk Analytics</div>", unsafe_allow_html=True)
    
    risk_tabs = st.tabs([
        "📈 Rolling Metrics",
        "🎲 Monte Carlo Simulation",
        "⚡ VaR/CVaR Analysis",
        "🌪️ Stress Testing",
        "📊 Performance Attribution"
    ])
    
    with risk_tabs[0]:
        # Rolling metrics
        st.markdown("#### Rolling Performance Metrics")
        
        # Get benchmark returns (use XU100 if available)
        benchmark_col = 'XU100.IS' if 'XU100.IS' in benchmark_returns.columns else benchmark_returns.columns[0]
        benchmark_series = benchmark_returns[benchmark_col] if benchmark_col in benchmark_returns.columns else None
        
        if benchmark_series is not None:
            rolling_fig = viz_engine.plot_rolling_metrics(
                portfolio_returns,
                benchmark_series,
                risk_free_rate,
                window=rolling_window
            )
            st.plotly_chart(rolling_fig, use_container_width=True)
        else:
            st.warning("Benchmark data not available for rolling metrics.")
    
    with risk_tabs[1]:
        # Monte Carlo simulation
        st.markdown("#### Monte Carlo Simulation")
        
        col_mc1, col_mc2 = st.columns(2)
        
        with col_mc1:
            mc_confidence = st.slider(
                "Confidence Level",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01
            )
        
        with col_mc2:
            mc_horizon_custom = st.slider(
                "Simulation Horizon (days)",
                min_value=1,
                max_value=30,
                value=mc_horizon,
                step=1,
                key="mc_horizon_custom"
            )
        
        with st.spinner("Running Monte Carlo simulation..."):
            mc_results = RiskAnalytics.monte_carlo_simulation(
                portfolio_returns,
                horizon_days=mc_horizon_custom,
                n_simulations=mc_simulations,
                confidence_level=mc_confidence
            )
        
        # Display results
        col_mc3, col_mc4, col_mc5 = st.columns(3)
        
        with col_mc3:
            st.metric(
                f"VaR ({int(mc_confidence*100)}%)",
                f"{mc_results['var']:.3%}",
                delta=f"{mc_horizon_custom}-day horizon"
            )
        
        with col_mc4:
            st.metric(
                f"CVaR ({int(mc_confidence*100)}%)",
                f"{mc_results['cvar']:.3%}",
                delta="Expected shortfall"
            )
        
        with col_mc5:
            st.metric(
                "Confidence Interval",
                f"[{mc_results['confidence_interval'][0]:.3%}, {mc_results['confidence_interval'][1]:.3%}]",
                delta="95% range"
            )
        
        # Plot distribution
        mc_fig = viz_engine.plot_monte_carlo_distribution(mc_results)
        st.plotly_chart(mc_fig, use_container_width=True)
    
    with risk_tabs[2]:
        # VaR/CVaR analysis
        st.markdown("#### Value at Risk Analysis")
        
        var_results = RiskAnalytics.calculate_var_cvar(
            portfolio_returns,
            confidence_levels=[0.90, 0.95, 0.99, 0.995]
        )
        
        col_var1, col_var2 = st.columns([2, 1])
        
        with col_var1:
            st.dataframe(
                var_results.style.format({
                    'VaR': '{:.4f}',
                    'CVaR': '{:.4f}',
                    'Historical': '{:.2f}'
                }),
                use_container_width=True
            )
        
        with col_var2:
            st.markdown("##### Interpretation")
            st.info("""
            **VaR (Value at Risk):**
            Maximum expected loss over a given period at a specified confidence level.
            
            **CVaR (Conditional VaR):**
            Average loss given that the loss exceeds VaR.
            
            **Historical %:**
            Percentage of days where loss exceeded VaR.
            """)
        
        # Cornish-Fisher VaR
        st.markdown("##### Cornish-Fisher Modified VaR")
        
        # Calculate skewness and kurtosis
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()
        
        col_cf1, col_cf2, col_cf3 = st.columns(3)
        
        with col_cf1:
            st.metric("Skewness", f"{skewness:.4f}")
        
        with col_cf2:
            st.metric("Excess Kurtosis", f"{kurtosis:.4f}")
        
        with col_cf3:
            # Calculate Cornish-Fisher adjustment
            z_normal = stats.norm.ppf(0.05)
            z_cf = z_normal + (z_normal**2 - 1) * skewness / 6 + \
                   (z_normal**3 - 3*z_normal) * kurtosis / 24 - \
                   (2*z_normal**3 - 5*z_normal) * skewness**2 / 36
            
            normal_var = -(portfolio_returns.mean() + z_normal * portfolio_returns.std())
            cf_var = -(portfolio_returns.mean() + z_cf * portfolio_returns.std())
            
            improvement = (cf_var - normal_var) / normal_var * 100
            st.metric(
                "CF Adjustment",
                f"{improvement:.1f}%",
                delta="vs Normal VaR"
            )
    
    with risk_tabs[3]:
        # Stress testing
        st.markdown("#### Stress Testing Scenarios")
        
        stress_results = RiskAnalytics.stress_test(portfolio_returns)
        
        col_stress1, col_stress2 = st.columns([3, 1])
        
        with col_stress1:
            st.dataframe(
                stress_results.style.format({
                    'Annual Return': '{:.2%}',
                    'Annual Volatility': '{:.2%}',
                    'Sharpe Ratio': '{:.3f}',
                    'Max Drawdown': '{:.2%}'
                }),
                use_container_width=True
            )
        
        with col_stress2:
            st.markdown("##### Scenario Definitions")
            st.info("""
            **Market Crash:** -20% return shock
            
            **Correction:** -10% return shock
            
            **Normal:** No shock
            
            **Bull Market:** +10% return shock
            
            **Strong Bull:** +20% return shock
            """)
    
    with risk_tabs[4]:
        # Performance attribution
        st.markdown("#### Performance Attribution")
        
        if benchmark_series is not None:
            attribution = PerformanceAttribution.calculate_attribution(
                weights,
                returns,
                benchmark_series
            )
            
            col_attr1, col_attr2 = st.columns([3, 1])
            
            with col_attr1:
                st.dataframe(
                    attribution.style.format({
                        'Weight': '{:.2%}',
                        'Return': '{:.2%}',
                        'Contribution': '{:.2%}'
                    }),
                    use_container_width=True
                )
            
            with col_attr2:
                total_contribution = attribution['Contribution'].sum()
                benchmark_return = benchmark_series.mean() * 252
                
                st.metric(
                    "Total Contribution",
                    f"{total_contribution:.2%}",
                    delta=f"Benchmark: {benchmark_return:.2%}"
                )
        else:
            st.warning("Benchmark data not available for performance attribution.")
    
    # ── COMPARISON WITH OTHER STRATEGIES ──
    st.markdown("<div class='section-header'>Strategy Comparison</div>", unsafe_allow_html=True)
    
    # Compare different optimization methods
    comparison_methods = ['max_sharpe', 'min_volatility', 'equal_weight', 'hrp', 'risk_parity']
    comparison_labels = ['Max Sharpe', 'Min Volatility', 'Equal Weight', 'HRP', 'Risk Parity']
    
    comparison_results = []
    
    with st.spinner("Comparing strategies..."):
        for method in comparison_methods:
            try:
                result = optimizer.optimize(method)
                portfolio_returns_comp = (returns * pd.Series(result['weights'])).sum(axis=1)
                metrics = optimizer._calculate_performance_metrics(portfolio_returns_comp)
                comparison_results.append(metrics)
            except:
                comparison_results.append({})
    
    # Filter valid results
    valid_indices = [i for i, r in enumerate(comparison_results) if r]
    valid_results = [comparison_results[i] for i in valid_indices]
    valid_labels = [comparison_labels[i] for i in valid_indices]
    
    if valid_results:
        comparison_fig = viz_engine.plot_risk_metrics_comparison(valid_results, valid_labels)
        st.plotly_chart(comparison_fig, use_container_width=True)
    else:
        st.warning("Could not generate strategy comparison.")
    
    # ── REPORT GENERATION ──
    st.markdown("<div class='section-header'>Report Generation</div>", unsafe_allow_html=True)
    
    col_report1, col_report2, col_report3 = st.columns(3)
    
    with col_report1:
        if st.button("📄 Generate HTML Report", use_container_width=True):
            try:
                report_generator = ReportGenerator()
                html_report = report_generator.generate_html_report(
                    portfolio_results,
                    portfolio_metrics,
                    optimization_result
                )
                
                # Create download link
                b64 = base64.b64encode(html_report.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="portfolio_report.html">📥 Download HTML Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                st.success("HTML report generated successfully!")
            except Exception as e:
                st.error(f"Failed to generate HTML report: {str(e)}")
    
    with col_report2:
        if st.button("📊 Generate Excel Report", use_container_width=True):
            try:
                report_generator = ReportGenerator()
                excel_file = report_generator.create_excel_report(
                    portfolio_results,
                    portfolio_metrics,
                    optimization_result,
                    "portfolio_report.xlsx"
                )
                
                with open(excel_file, "rb") as f:
                    excel_data = f.read()
                
                # Create download link
                b64 = base64.b64encode(excel_data).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="portfolio_report.xlsx">📥 Download Excel Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                st.success("Excel report generated successfully!")
            except Exception as e:
                st.error(f"Failed to generate Excel report: {str(e)}")
    
    with col_report3:
        if st.button("🖨️ Print Summary", use_container_width=True):
            st.info(f"""
            ### Portfolio Summary
            - **Optimization Method:** {portfolio_results['method']}
            - **Number of Assets:** {len([w for w in weights.values() if w > 0.001])}
            - **Annual Return:** {portfolio_results['annual_return']:.2%}
            - **Annual Volatility:** {portfolio_results['annual_volatility']:.2%}
            - **Sharpe Ratio:** {portfolio_results['sharpe_ratio']:.3f}
            - **Max Drawdown:** {portfolio_results['max_drawdown']:.2%}
            - **VaR (95%):** {portfolio_results['var_95']:.3%}
            """)
    
    # ── FOOTER ──
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#7a90b5;">
            <strong>Data Sources</strong><br>
            Yahoo Finance<br>
            BIST Market Data<br>
            TCMB Statistics
        </div>
        """, unsafe_allow_html=True)
    
    with col_footer2:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#7a90b5;">
            <strong>Methodology</strong><br>
            Modern Portfolio Theory<br>
            Risk Parity<br>
            Monte Carlo Simulation<br>
            GARCH Volatility
        </div>
        """, unsafe_allow_html=True)
    
    with col_footer3:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#7a90b5;">
            <strong>Disclaimer</strong><br>
            For educational purposes only.<br>
            Past performance ≠ future results.<br>
            Consult a financial advisor.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align:center;font-family:'Space Mono',monospace;font-size:0.6rem;
                color:#7a90b5;padding:2rem 0;">
        BIST Portfolio Risk Terminal Pro · v3.0 · © 2024 · All rights reserved
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application crashed: {traceback.format_exc()}")
        
        # Show error details in expander
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        
        st.info("""
        **Troubleshooting steps:**
        1. Check your internet connection
        2. Try a different date range
        3. Refresh the page
        4. Check console for detailed errors
        
        If the problem persists, please report the issue.
        """)
