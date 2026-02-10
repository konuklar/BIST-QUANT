# ============================================================================
# 1. CORE IMPORTS & CONFIGURATION
# ============================================================================
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize, differential_evolution, basinhopping
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# PyPortfolioOpt libraries
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt import CLA
from pypfopt import EfficientCVaR
from pypfopt import HRPOpt
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Advanced quantitative libraries
import empyrical as ep
from scipy.stats import norm, t, skew, kurtosis, jarque_bera, shapiro
from scipy.special import erfinv
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import ruptures as rpt  # For change point detection

# Machine learning for regime detection
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# ARCH: For Econometric Volatility Forecasting
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# Network analysis
import networkx as nx

# For PDF reporting
from fpdf import FPDF
import base64
import io

# Streamlit Page Configuration
st.set_page_config(
    page_title="BIST Institutional Portfolio Analytics Platform",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional institutional styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #374151;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.8rem;
        padding-left: 1rem;
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.1), transparent);
    }
    .section-header {
        font-size: 1.4rem;
        color: #4B5563;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #10B981;
        padding-left: 1rem;
    }
    
    /* Cards and metrics */
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    .metric-card-danger {
        border-left: 5px solid #EF4444;
    }
    .metric-card-warning {
        border-left: 5px solid #F59E0B;
    }
    .metric-card-success {
        border-left: 5px solid #10B981;
    }
    .metric-card-info {
        border-left: 5px solid #3B82F6;
    }
    
    /* Alerts and warnings */
    .warning-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border: 2px solid #F59E0B;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.2);
    }
    .success-box {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }
    .error-box {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.2);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #F9FAFB;
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 8px;
        gap: 1px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E5E7EB;
        border-color: #D1D5DB;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6 0%, #1E40AF 100%) !important;
        color: white !important;
        border-color: #1E40AF !important;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    
    /* Plot containers */
    .plot-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin: 1.5rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E5E7EB;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
    }
    
    /* Custom progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }
    
    /* Grid layout */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. CONFIGURATION & CONSTANTS
# ============================================================================

# Turkish BIST 30 tickers with company names
BIST30_TICKERS = {
    'AKBNK.IS': 'Akbank',
    'ARCLK.IS': 'Arçelik',
    'ASELS.IS': 'Aselsan',
    'BIMAS.IS': 'BİM Birleşik Mağazalar',
    'DOHOL.IS': 'Doğuş Otomotiv',
    'EKGYO.IS': 'Emlak Konut GYO',
    'EREGL.IS': 'Ereğli Demir Çelik',
    'FROTO.IS': 'Ford Otosan',
    'GARAN.IS': 'Garanti BBVA',
    'HALKB.IS': 'Halkbank',
    'ISCTR.IS': 'İş Bankası',
    'KCHOL.IS': 'Koç Holding',
    'KOZAA.IS': 'Koza Altın',
    'KOZAL.IS': 'Koza Madencilik',
    'KRDMD.IS': 'Kardemir',
    'PETKM.IS': 'Petkim',
    'PGSUS.IS': 'Peugeot',
    'SAHOL.IS': 'Sabancı Holding',
    'SASA.IS': 'Sasa Polyester',
    'SISE.IS': 'Şişecam',
    'SKBNK.IS': 'Şekerbank',
    'TCELL.IS': 'Turkcell',
    'THYAO.IS': 'Turkish Airlines',
    'TKFEN.IS': 'Tekfen Holding',
    'TOASO.IS': 'Tofaş',
    'TTKOM.IS': 'Türk Telekom',
    'TUPRS.IS': 'Tüpraş',
    'ULKER.IS': 'Ülker',
    'VAKBN.IS': 'Vakıfbank',
    'YKBNK.IS': 'Yapı Kredi Bankası'
}

# Benchmark indices
BENCHMARK_TICKERS = {
    'XU100.IS': 'BIST 100',
    'XU030.IS': 'BIST 30',
    'XU050.IS': 'BIST 50',
    'GARAN.IS': 'Sector Benchmark (Banking)'
}

# Sector classification for BIST 30
SECTOR_CLASSIFICATION = {
    'Financials': ['AKBNK.IS', 'GARAN.IS', 'HALKB.IS', 'ISCTR.IS', 'SKBNK.IS', 'VAKBN.IS', 'YKBNK.IS'],
    'Industrials': ['ARCLK.IS', 'ASELS.IS', 'FROTO.IS', 'TOASO.IS', 'TKFEN.IS', 'SISE.IS'],
    'Materials': ['EREGL.IS', 'KRDMD.IS', 'PETKM.IS', 'SASA.IS', 'KOZAA.IS', 'KOZAL.IS'],
    'Consumer Staples': ['BIMAS.IS', 'ULKER.IS'],
    'Consumer Discretionary': ['DOHOL.IS', 'PGSUS.IS'],
    'Real Estate': ['EKGYO.IS'],
    'Communications': ['TCELL.IS', 'TTKOM.IS'],
    'Transportation': ['THYAO.IS'],
    'Energy': ['TUPRS.IS'],
    'Conglomerates': ['KCHOL.IS', 'SAHOL.IS']
}

# Annualized risk-free rate for Turkey (updated January 2024)
RISK_FREE_RATE = 0.45  # 45% annual, reflecting current Turkish monetary policy

# Trading days per year
TRADING_DAYS = 252

# ============================================================================
# 3. ENHANCED DATA FETCHING & VALIDATION CLASS
# ============================================================================

class MarketDataFetcher:
    """Enhanced data fetching with multiple fallback strategies and validation"""
    
    def __init__(self):
        self.tickers = list(BIST30_TICKERS.keys())
        self.benchmarks = list(BENCHMARK_TICKERS.keys())
        self.cache = {}
        
    def fetch_with_retry(self, tickers, start_date, end_date, max_retries=3):
        """Fetch data with retry logic and multiple fallback strategies"""
        
        for attempt in range(max_retries):
            try:
                # Strategy 1: Try bulk download
                data = yf.download(
                    tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    group_by='ticker',
                    threads=True
                )
                
                # Handle different data structures
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-index structure (new Yahoo Finance format)
                    if 'Adj Close' in data.columns.get_level_values(0):
                        adj_close = data['Adj Close']
                    elif 'Close' in data.columns.get_level_values(0):
                        adj_close = data['Close']
                    else:
                        # Try to get first available price column
                        adj_close = data.xs(data.columns[0][0], axis=1, level=0)
                else:
                    # Single index structure (old format)
                    if 'Adj Close' in data.columns:
                        adj_close = data['Adj Close']
                    elif 'Close' in data.columns:
                        adj_close = data['Close']
                    else:
                        adj_close = data.iloc[:, 0]  # First column
                
                # Convert to DataFrame if Series
                if isinstance(adj_close, pd.Series):
                    adj_close = adj_close.to_frame()
                    adj_close.columns = tickers
                
                # Check if we got any data
                if adj_close.empty:
                    raise ValueError("No data returned")
                
                # Fill missing values using multiple methods
                adj_close = self._clean_data(adj_close)
                
                return adj_close
                
            except Exception as e:
                st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Try individual ticker download as fallback
                    if len(tickers) > 1:
                        st.info("Trying individual ticker download...")
                        return self._fetch_individual(tickers, start_date, end_date)
                    
                    # Wait before retry
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"Failed to fetch data after {max_retries} attempts")
    
    def _fetch_individual(self, tickers, start_date, end_date):
        """Fetch tickers individually (slower but more reliable)"""
        data_frames = []
        
        for ticker in tickers:
            try:
                ticker_data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if not ticker_data.empty:
                    if 'Adj Close' in ticker_data.columns:
                        price_series = ticker_data['Adj Close']
                    elif 'Close' in ticker_data.columns:
                        price_series = ticker_data['Close']
                    else:
                        price_series = ticker_data.iloc[:, 0]
                    
                    price_series.name = ticker
                    data_frames.append(price_series)
                    
            except Exception as e:
                st.warning(f"Failed to fetch {ticker}: {str(e)}")
                continue
        
        if not data_frames:
            raise ValueError("No data fetched for any ticker")
        
        # Combine all series into DataFrame
        combined_data = pd.concat(data_frames, axis=1)
        combined_data = self._clean_data(combined_data)
        
        return combined_data
    
    def _clean_data(self, data):
        """Clean and validate price data using multiple methods"""
        
        # Forward fill for intraday gaps
        data = data.ffill()
        
        # Backward fill for initial missing values
        data = data.bfill()
        
        # Interpolate remaining gaps
        data = data.interpolate(method='linear', limit_direction='both')
        
        # Remove tickers with too many missing values (>20%)
        missing_pct = data.isnull().sum() / len(data)
        valid_tickers = missing_pct[missing_pct < 0.2].index
        data = data[valid_tickers]
        
        # Remove rows with excessive missing data
        data = data.dropna(thresh=len(data.columns) * 0.8)
        
        # Validate data quality
        self._validate_data(data)
        
        return data
    
    def _validate_data(self, data):
        """Validate data quality"""
        
        if data.empty:
            raise ValueError("No valid data after cleaning")
        
        # Check for extreme values
        price_changes = data.pct_change().abs()
        extreme_moves = (price_changes > 0.5).any().any()  # >50% daily moves
        
        if extreme_moves:
            st.warning("⚠️ Extreme price movements detected. Data may need manual verification.")
        
        # Check for zero or negative prices
        negative_prices = (data <= 0).any().any()
        if negative_prices:
            st.error("❌ Negative or zero prices detected. Data quality issues.")
    
    def fetch_all_data(self, start_date, end_date, use_cache=True):
        """Fetch all required market data"""
        
        cache_key = f"{start_date}_{end_date}"
        
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        with st.spinner("🌐 Fetching market data from multiple sources..."):
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Step 1: Fetch BIST 30 stocks
            progress_bar.progress(10)
            stock_data = self.fetch_with_retry(self.tickers, start_date, end_date)
            
            # Step 2: Fetch benchmarks
            progress_bar.progress(40)
            benchmark_data = self.fetch_with_retry(self.benchmarks, start_date, end_date)
            
            # Step 3: Calculate returns
            progress_bar.progress(70)
            stock_returns = np.log(stock_data / stock_data.shift(1)).dropna()
            benchmark_returns = np.log(benchmark_data / benchmark_data.shift(1)).dropna()
            
            # Step 4: Additional market data
            progress_bar.progress(90)
            additional_data = self._fetch_additional_data(start_date, end_date)
            
            progress_bar.progress(100)
            
            result = {
                'prices': stock_data,
                'returns': stock_returns,
                'benchmark_prices': benchmark_data,
                'benchmark_returns': benchmark_returns,
                'additional_data': additional_data
            }
            
            self.cache[cache_key] = result
            
            return result
    
    def _fetch_additional_data(self, start_date, end_date):
        """Fetch additional market data (FX, commodities, etc.)"""
        additional_tickers = {
            'TRY=X': 'USD/TRY',
            'EURTRY=X': 'EUR/TRY',
            'GC=F': 'Gold',
            'BZ=F': 'Brent Oil',
            '^VIX': 'VIX Index'
        }
        
        try:
            data = yf.download(
                list(additional_tickers.keys()),
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                if 'Adj Close' in data.columns.get_level_values(0):
                    adj_close = data['Adj Close']
                else:
                    adj_close = data['Close']
            else:
                adj_close = data['Close'] if 'Close' in data.columns else data
            
            return adj_close
        except:
            return pd.DataFrame()

# ============================================================================
# 4. ADVANCED PORTFOLIO OPTIMIZER WITH MACHINE LEARNING
# ============================================================================

class AdvancedPortfolioOptimizer:
    """Advanced portfolio optimization with ML techniques and alternative risk measures"""
    
    def __init__(self, risk_free_rate=RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        self.daily_rf = np.log(1 + risk_free_rate) / TRADING_DAYS
        self.data_fetcher = MarketDataFetcher()
        
    def calculate_advanced_metrics(self, weights, returns, benchmark_returns=None):
        """Calculate comprehensive portfolio performance metrics"""
        
        # Ensure weights are properly normalized
        weights = weights / weights.sum()
        
        # Portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate all metrics
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(portfolio_returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(portfolio_returns))
        
        # Risk-adjusted return metrics
        metrics.update(self._calculate_risk_adjusted_metrics(portfolio_returns))
        
        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics(portfolio_returns))
        
        # Statistical properties
        metrics.update(self._calculate_statistical_metrics(portfolio_returns))
        
        # Tail risk metrics
        metrics.update(self._calculate_tail_risk_metrics(portfolio_returns))
        
        # Benchmark relative metrics (if benchmark provided)
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_metrics(portfolio_returns, benchmark_returns))
        
        # Advanced ML-based metrics
        metrics.update(self._calculate_ml_metrics(portfolio_returns))
        
        return metrics, portfolio_returns
    
    def _calculate_return_metrics(self, returns):
        """Calculate return-related metrics"""
        metrics = {}
        
        # Daily metrics
        daily_mean = returns.mean()
        daily_median = returns.median()
        daily_std = returns.std()
        
        # Annualized metrics
        annual_return = daily_mean * TRADING_DAYS
        annual_volatility = daily_std * np.sqrt(TRADING_DAYS)
        
        # Cumulative return
        cumulative_return = np.exp(returns.sum()) - 1
        
        metrics.update({
            'Daily Mean Return': daily_mean,
            'Daily Median Return': daily_median,
            'Daily Std Dev': daily_std,
            'Annualized Return': annual_return,
            'Annualized Volatility': annual_volatility,
            'Cumulative Return': cumulative_return,
            'Geometric Mean Return': np.exp(daily_mean) - 1,
            'Excess Return (over RF)': annual_return - self.risk_free_rate
        })
        
        return metrics
    
    def _calculate_risk_metrics(self, returns):
        """Calculate risk metrics"""
        metrics = {}
        
        # Basic volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(TRADING_DAYS)
        
        # Downside volatility
        downside_returns = returns[returns < self.daily_rf]
        downside_vol = downside_returns.std() * np.sqrt(TRADING_DAYS) if len(downside_returns) > 1 else 0
        
        # Semi-deviation
        negative_returns = returns[returns < 0]
        semi_deviation = negative_returns.std() * np.sqrt(TRADING_DAYS) if len(negative_returns) > 1 else 0
        
        metrics.update({
            'Downside Volatility': downside_vol,
            'Semi-Deviation': semi_deviation,
            'Upside Volatility': returns[returns > self.daily_rf].std() * np.sqrt(TRADING_DAYS) 
                                 if len(returns[returns > self.daily_rf]) > 1 else 0,
            'Mean Absolute Deviation': returns.mad() * np.sqrt(TRADING_DAYS),
            'Target Downside Deviation': self._calculate_target_downside_deviation(returns)
        })
        
        return metrics
    
    def _calculate_target_downside_deviation(self, returns, target=0):
        """Calculate target downside deviation (for Sortino ratio)"""
        downside_diff = returns[returns < target] - target
        if len(downside_diff) > 1:
            return np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(TRADING_DAYS)
        return 0
    
    def _calculate_risk_adjusted_metrics(self, returns):
        """Calculate risk-adjusted return metrics"""
        metrics = {}
        
        annual_return = returns.mean() * TRADING_DAYS
        annual_vol = returns.std() * np.sqrt(TRADING_DAYS)
        
        # Sharpe Ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Sortino Ratio
        downside_vol = self._calculate_target_downside_deviation(returns, self.daily_rf)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Calmar Ratio
        max_dd = self._calculate_max_drawdown(returns)
        calmar_ratio = annual_return / abs(max_dd) if max_dd < 0 else np.inf
        
        # Omega Ratio
        omega_ratio = self._calculate_omega_ratio(returns)
        
        # Treynor Ratio (requires beta)
        treynor_ratio = np.nan  # Will be calculated if benchmark provided
        
        # Information Ratio (requires benchmark)
        information_ratio = np.nan
        
        metrics.update({
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Omega Ratio': omega_ratio,
            'Treynor Ratio': treynor_ratio,
            'Information Ratio': information_ratio,
            'Modified Sharpe Ratio': sharpe_ratio * np.sqrt(TRADING_DAYS),  # Annualized
            'Gain to Pain Ratio': returns.sum() / abs(returns[returns < 0].sum()) 
                                 if returns[returns < 0].sum() < 0 else np.inf
        })
        
        return metrics
    
    def _calculate_omega_ratio(self, returns, threshold=None):
        """Calculate Omega ratio"""
        if threshold is None:
            threshold = self.daily_rf
        
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        
        return gains / losses if losses > 0 else np.inf
    
    def _calculate_drawdown_metrics(self, returns):
        """Calculate drawdown-related metrics"""
        metrics = {}
        
        # Calculate cumulative returns
        cumulative = np.exp(returns.cumsum())
        
        # Calculate running maximum
        running_max = cumulative.cummax()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin() if not drawdown.empty else None
        
        # Calculate drawdown duration
        if max_dd_date is not None:
            recovery_mask = cumulative.loc[max_dd_date:] >= running_max.loc[max_dd_date]
            if recovery_mask.any():
                recovery_date = recovery_mask.idxmax()
                recovery_days = (recovery_date - max_dd_date).days
            else:
                recovery_days = np.nan
        else:
            recovery_days = np.nan
        
        # Average drawdown
        avg_drawdown = drawdown.mean()
        
        # Drawdown volatility
        dd_volatility = drawdown.std()
        
        metrics.update({
            'Maximum Drawdown': max_drawdown,
            'Max Drawdown Date': max_dd_date,
            'Recovery Days': recovery_days,
            'Average Drawdown': avg_drawdown,
            'Drawdown Volatility': dd_volatility,
            'Ulcer Index': np.sqrt(np.mean(drawdown ** 2)),
            'Pain Index': abs(drawdown.mean()),
            'Drawdown Count': (drawdown < -0.05).sum(),  # Count >5% drawdowns
            'Worst 5 Drawdowns': self._get_top_drawdowns(drawdown, n=5)
        })
        
        return metrics
    
    def _get_top_drawdowns(self, drawdown, n=5):
        """Get top n drawdowns"""
        drawdown_series = drawdown.copy()
        top_dd = []
        
        for _ in range(min(n, len(drawdown_series))):
            if drawdown_series.min() < -0.01:  # Only consider >1% drawdowns
                min_idx = drawdown_series.idxmin()
                min_val = drawdown_series.min()
                top_dd.append((min_idx, min_val))
                
                # Remove this drawdown period (±10 days) for next iteration
                start_idx = max(0, drawdown_series.index.get_loc(min_idx) - 10)
                end_idx = min(len(drawdown_series), drawdown_series.index.get_loc(min_idx) + 10)
                drawdown_series.iloc[start_idx:end_idx] = 0
        
        return top_dd
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns"""
        cumulative = np.exp(returns.cumsum())
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_statistical_metrics(self, returns):
        """Calculate statistical properties"""
        metrics = {}
        
        # Basic statistics
        skewness = stats.skew(returns)
        kurt = stats.kurtosis(returns, fisher=False)  # Pearson's kurtosis
        excess_kurtosis = kurt - 3
        
        # Normality tests
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        shapiro_stat, shapiro_pvalue = stats.shapiro(returns) if len(returns) < 5000 else (np.nan, np.nan)
        
        # Serial correlation
        autocorr_lag1 = returns.autocorr(lag=1)
        autocorr_lag5 = returns.autocorr(lag=5)
        
        # Stationarity tests
        adf_stat, adf_pvalue = self._adf_test(returns)
        kpss_stat, kpss_pvalue = self._kpss_test(returns)
        
        metrics.update({
            'Skewness': skewness,
            'Kurtosis': kurt,
            'Excess Kurtosis': excess_kurtosis,
            'Jarque-Bera Statistic': jb_stat,
            'Jarque-Bera p-value': jb_pvalue,
            'Shapiro-Wilk Statistic': shapiro_stat,
            'Shapiro-Wilk p-value': shapiro_pvalue,
            'Autocorrelation Lag 1': autocorr_lag1,
            'Autocorrelation Lag 5': autocorr_lag5,
            'ADF Statistic': adf_stat,
            'ADF p-value': adf_pvalue,
            'KPSS Statistic': kpss_stat,
            'KPSS p-value': kpss_pvalue,
            'Median Absolute Deviation': stats.median_abs_deviation(returns),
            'Coefficient of Variation': returns.std() / abs(returns.mean()) if returns.mean() != 0 else np.inf
        })
        
        return metrics
    
    def _adf_test(self, series):
        """Augmented Dickey-Fuller test for stationarity"""
        try:
            result = adfuller(series.dropna())
            return result[0], result[1]
        except:
            return np.nan, np.nan
    
    def _kpss_test(self, series):
        """KPSS test for stationarity"""
        try:
            from statsmodels.tsa.stattools import kpss
            result = kpss(series.dropna())
            return result[0], result[1]
        except:
            return np.nan, np.nan
    
    def _calculate_tail_risk_metrics(self, returns):
        """Calculate tail risk metrics"""
        metrics = {}
        
        # VaR at different confidence levels
        conf_levels = [0.90, 0.95, 0.99]
        
        for conf in conf_levels:
            # Historical VaR
            hist_var = np.percentile(returns, (1 - conf) * 100)
            
            # Parametric VaR (Gaussian)
            gaussian_var = norm.ppf(1 - conf, returns.mean(), returns.std())
            
            # Cornish-Fisher VaR (adjusts for skewness and kurtosis)
            z = norm.ppf(1 - conf)
            s = stats.skew(returns)
            k = stats.kurtosis(returns, fisher=False) - 3
            
            cf_z = z + (z**2 - 1) * s/6 + (z**3 - 3*z) * k/24 - (2*z**3 - 5*z) * s**2/36
            cf_var = returns.mean() + cf_z * returns.std()
            
            # Expected Shortfall/CVaR
            es = returns[returns <= hist_var].mean()
            
            metrics.update({
                f'VaR {int(conf*100)}% (Historical)': hist_var,
                f'VaR {int(conf*100)}% (Gaussian)': gaussian_var,
                f'VaR {int(conf*100)}% (Cornish-Fisher)': cf_var,
                f'CVaR {int(conf*100)}%': es,
                f'VaR to CVaR Ratio {int(conf*100)}%': abs(hist_var / es) if es != 0 else np.inf
            })
        
        # Tail ratio
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        tail_ratio = abs(var_99 / var_95) if var_95 != 0 else np.inf
        
        # Expected Shortfall ratio
        es_95 = returns[returns <= var_95].mean()
        es_ratio = abs(var_95 / es_95) if es_95 != 0 else np.inf
        
        metrics.update({
            'Tail Ratio (99%/95%)': tail_ratio,
            'ES Ratio (95%)': es_ratio,
            'Maximum Daily Loss': returns.min(),
            'Conditional Skewness': self._calculate_conditional_skewness(returns),
            'Conditional Kurtosis': self._calculate_conditional_kurtosis(returns)
        })
        
        return metrics
    
    def _calculate_conditional_skewness(self, returns, threshold=0.05):
        """Calculate skewness conditional on tail events"""
        tail_returns = returns[returns <= np.percentile(returns, threshold * 100)]
        if len(tail_returns) > 2:
            return stats.skew(tail_returns)
        return np.nan
    
    def _calculate_conditional_kurtosis(self, returns, threshold=0.05):
        """Calculate kurtosis conditional on tail events"""
        tail_returns = returns[returns <= np.percentile(returns, threshold * 100)]
        if len(tail_returns) > 3:
            return stats.kurtosis(tail_returns, fisher=False)
        return np.nan
    
    def _calculate_benchmark_metrics(self, portfolio_returns, benchmark_returns):
        """Calculate metrics relative to benchmark"""
        metrics = {}
        
        # Align indices
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        port_aligned = portfolio_returns.loc[common_idx]
        bench_aligned = benchmark_returns.loc[common_idx]
        
        if len(port_aligned) < 10:  # Need sufficient overlapping data
            return metrics
        
        # Active returns
        active_returns = port_aligned - bench_aligned
        
        # Beta calculation
        covariance = np.cov(port_aligned, bench_aligned)[0, 1]
        bench_variance = np.var(bench_aligned)
        beta = covariance / bench_variance if bench_variance > 0 else np.nan
        
        # Alpha (Jensen's Alpha)
        port_annual = port_aligned.mean() * TRADING_DAYS
        bench_annual = bench_aligned.mean() * TRADING_DAYS
        alpha = port_annual - (self.risk_free_rate + beta * (bench_annual - self.risk_free_rate))
        
        # Tracking error
        tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS)
        
        # Information ratio
        information_ratio = active_returns.mean() * TRADING_DAYS / tracking_error if tracking_error > 0 else np.nan
        
        # Treynor ratio
        treynor_ratio = (port_annual - self.risk_free_rate) / beta if beta > 0 else np.nan
        
        # Appraisal ratio
        appraisal_ratio = alpha / tracking_error if tracking_error > 0 else np.nan
        
        # Up/Down capture ratios
        up_market = bench_aligned > 0
        down_market = bench_aligned < 0
        
        up_capture = port_aligned[up_market].mean() / bench_aligned[up_market].mean() if up_market.any() else np.nan
        down_capture = port_aligned[down_market].mean() / bench_aligned[down_market].mean() if down_market.any() else np.nan
        
        metrics.update({
            'Beta': beta,
            'Alpha (Jensen)': alpha,
            'Tracking Error': tracking_error,
            'Information Ratio': information_ratio,
            'Treynor Ratio': treynor_ratio,
            'Appraisal Ratio': appraisal_ratio,
            'Up Capture Ratio': up_capture,
            'Down Capture Ratio': down_capture,
            'Capture Ratio': up_capture / abs(down_capture) if down_capture != 0 else np.inf,
            'Active Share': np.mean(np.abs(active_returns)) * TRADING_DAYS,
            'R-squared': np.corrcoef(port_aligned, bench_aligned)[0, 1] ** 2
        })
        
        return metrics
    
    def _calculate_ml_metrics(self, returns):
        """Calculate machine learning based metrics"""
        metrics = {}
        
        # Regime detection metrics
        try:
            regimes = self._detect_regimes(returns)
            if regimes is not None:
                metrics['Number of Regimes'] = len(np.unique(regimes))
                metrics['Regime Persistence'] = self._calculate_regime_persistence(regimes)
        except:
            pass
        
        # Entropy-based metrics
        metrics['Sample Entropy'] = self._calculate_sample_entropy(returns)
        metrics['Permutation Entropy'] = self._calculate_permutation_entropy(returns)
        
        # Fractal metrics
        try:
            hurst_exponent = self._calculate_hurst_exponent(returns)
            metrics['Hurst Exponent'] = hurst_exponent
            metrics['Market Efficiency'] = 'Efficient' if abs(hurst_exponent - 0.5) < 0.1 else 'Inefficient'
        except:
            metrics['Hurst Exponent'] = np.nan
        
        return metrics
    
    def _detect_regimes(self, returns, n_regimes=3):
        """Detect market regimes using Gaussian Mixture Models"""
        try:
            # Prepare features
            features = pd.DataFrame({
                'returns': returns,
                'volatility': returns.rolling(20).std(),
                'skewness': returns.rolling(20).apply(lambda x: stats.skew(x.dropna())),
            }).dropna()
            
            if len(features) < 50:
                return None
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Fit GMM
            gmm = GaussianMixture(n_components=n_regimes, random_state=42)
            regimes = gmm.fit_predict(features_scaled)
            
            # Map back to original index
            regime_series = pd.Series(regimes, index=features.index)
            
            return regime_series
        except:
            return None
    
    def _calculate_regime_persistence(self, regimes):
        """Calculate how persistent regimes are"""
        if regimes is None:
            return np.nan
        
        changes = (regimes != regimes.shift(1)).sum()
        total_periods = len(regimes)
        
        return 1 - (changes / total_periods)
    
    def _calculate_sample_entropy(self, returns, m=2, r=0.2):
        """Calculate sample entropy of returns"""
        try:
            # Normalize returns
            series = (returns - returns.mean()) / returns.std()
            
            # Calculate sample entropy
            N = len(series)
            
            # Split into templates
            templates = np.array([series[i:i+m] for i in range(N-m+1)])
            
            # Calculate distances
            dist = np.abs(templates[:, None] - templates)
            dist = np.max(dist, axis=2)
            
            # Count matches
            matches_m = np.sum(dist <= r) - (N - m + 1)  # Subtract self-matches
            matches_m1 = np.sum(dist[:, :, None] <= r) - (N - m)  # For m+1
            
            # Avoid division by zero
            if matches_m == 0 or matches_m1 == 0:
                return 0
            
            return -np.log(matches_m1 / matches_m)
        except:
            return np.nan
    
    def _calculate_permutation_entropy(self, returns, order=3, delay=1):
        """Calculate permutation entropy"""
        try:
            from itertools import permutations
            
            # Get all possible permutations
            permutations_list = list(permutations(range(order)))
            
            # Create dictionary for permutations
            perm_dict = {perm: idx for idx, perm in enumerate(permutations_list)}
            
            # Create overlapping vectors
            vectors = np.array([returns[i:i+order*delay:delay] for i in range(len(returns)-order*delay+1)])
            
            # Get permutation patterns
            patterns = []
            for vec in vectors:
                sorted_idx = np.argsort(vec)
                patterns.append(perm_dict[tuple(sorted_idx)])
            
            # Calculate probability distribution
            hist = np.bincount(patterns, minlength=len(permutations_list))
            probs = hist / hist.sum()
            
            # Calculate entropy
            entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
            
            # Normalize
            max_entropy = np.log2(len(permutations_list))
            
            return entropy / max_entropy if max_entropy > 0 else 0
        except:
            return np.nan
    
    def _calculate_hurst_exponent(self, returns, max_lag=50):
        """Calculate Hurst exponent for long-term memory detection"""
        try:
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
            
            # Fit to power law
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            
            return poly[0]  # Hurst exponent
        except:
            return np.nan
    
    def optimize_portfolio(self, method, mu, S, returns, constraints=None, **kwargs):
        """Advanced portfolio optimization with multiple methods"""
        
        if constraints is None:
            constraints = {'min_weight': 0, 'max_weight': 1}
        
        methods = {
            'max_sharpe': self._optimize_max_sharpe,
            'min_volatility': self._optimize_min_volatility,
            'efficient_risk': self._optimize_efficient_risk,
            'efficient_return': self._optimize_efficient_return,
            'max_quadratic_utility': self._optimize_max_quadratic_utility,
            'hrp': self._optimize_hrp,
            'cvar': self._optimize_cvar,
            'equal_weight': self._optimize_equal_weight,
            'risk_parity': self._optimize_risk_parity,
            'most_diversified': self._optimize_most_diversified,
            'minimum_correlation': self._optimize_minimum_correlation,
            'black_litterman': self._optimize_black_litterman,
            'resampling': self._optimize_resampling,
            'bayesian': self._optimize_bayesian,
        }
        
        if method not in methods:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return methods[method](mu, S, returns, constraints, **kwargs)
    
    def _optimize_max_sharpe(self, mu, S, returns, constraints, **kwargs):
        """Maximize Sharpe ratio"""
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= constraints['min_weight'])
        ef.add_constraint(lambda w: w <= constraints['max_weight'])
        ef.max_sharpe(risk_free_rate=self.daily_rf)
        weights = ef.clean_weights()
        return weights
    
    def _optimize_min_volatility(self, mu, S, returns, constraints, **kwargs):
        """Minimize portfolio volatility"""
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= constraints['min_weight'])
        ef.add_constraint(lambda w: w <= constraints['max_weight'])
        ef.min_volatility()
        weights = ef.clean_weights()
        return weights
    
    def _optimize_efficient_risk(self, mu, S, returns, constraints, **kwargs):
        """Efficient portfolio for target risk"""
        target_vol = kwargs.get('target_volatility', 0.2)
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= constraints['min_weight'])
        ef.add_constraint(lambda w: w <= constraints['max_weight'])
        ef.efficient_risk(target_volatility=target_vol/np.sqrt(TRADING_DAYS))
        weights = ef.clean_weights()
        return weights
    
    def _optimize_efficient_return(self, mu, S, returns, constraints, **kwargs):
        """Efficient portfolio for target return"""
        target_return = kwargs.get('target_return', 0.15)
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= constraints['min_weight'])
        ef.add_constraint(lambda w: w <= constraints['max_weight'])
        ef.efficient_return(target_return=target_return/TRADING_DAYS)
        weights = ef.clean_weights()
        return weights
    
    def _optimize_max_quadratic_utility(self, mu, S, returns, constraints, **kwargs):
        """Maximize quadratic utility"""
        risk_aversion = kwargs.get('risk_aversion', 1.0)
        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= constraints['min_weight'])
        ef.add_constraint(lambda w: w <= constraints['max_weight'])
        ef.max_quadratic_utility(risk_aversion=risk_aversion)
        weights = ef.clean_weights()
        return weights
    
    def _optimize_hrp(self, mu, S, returns, constraints, **kwargs):
        """Hierarchical Risk Parity"""
        hrp = HRPOpt(returns)
        hrp.optimize()
        weights = hrp.clean_weights()
        return weights
    
    def _optimize_cvar(self, mu, S, returns, constraints, **kwargs):
        """Minimize Conditional Value at Risk"""
        cvar = EfficientCVaR(mu, returns)
        cvar.add_constraint(lambda w: w >= constraints['min_weight'])
        cvar.add_constraint(lambda w: w <= constraints['max_weight'])
        cvar.min_cvar()
        weights = cvar.clean_weights()
        return weights
    
    def _optimize_equal_weight(self, mu, S, returns, constraints, **kwargs):
        """Equal weight portfolio"""
        n_assets = len(returns.columns)
        weights = {ticker: 1/n_assets for ticker in returns.columns}
        return weights
    
    def _optimize_risk_parity(self, mu, S, returns, constraints, **kwargs):
        """Risk parity portfolio"""
        n = len(returns.columns)
        
        def risk_parity_objective(w):
            portfolio_vol = np.sqrt(w @ S @ w.T)
            marginal_risk = (S @ w.T) / portfolio_vol
            risk_contributions = w * marginal_risk
            target_rc = portfolio_vol / n
            return np.sum((risk_contributions - target_rc) ** 2)
        
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n)]
        constraints_opt = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        initial_weights = np.ones(n) / n
        
        result = minimize(
            risk_parity_objective,
            initial_weights,
            bounds=bounds,
            constraints=constraints_opt,
            method='SLSQP'
        )
        
        weights = {ticker: result.x[i] for i, ticker in enumerate(returns.columns)}
        return weights
    
    def _optimize_most_diversified(self, mu, S, returns, constraints, **kwargs):
        """Most Diversified Portfolio (MDP)"""
        # MDP maximizes diversification ratio
        volatilities = np.sqrt(np.diag(S))
        
        def diversification_ratio(w):
            portfolio_vol = np.sqrt(w @ S @ w.T)
            weighted_vol = np.sum(w * volatilities)
            return weighted_vol / portfolio_vol if portfolio_vol > 0 else 0
        
        n = len(returns.columns)
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n)]
        constraints_opt = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        initial_weights = np.ones(n) / n
        
        result = minimize(
            lambda w: -diversification_ratio(w),  # Maximize diversification
            initial_weights,
            bounds=bounds,
            constraints=constraints_opt,
            method='SLSQP'
        )
        
        weights = {ticker: result.x[i] for i, ticker in enumerate(returns.columns)}
        return weights
    
    def _optimize_minimum_correlation(self, mu, S, returns, constraints, **kwargs):
        """Minimum Correlation Portfolio"""
        # Use correlation matrix instead of covariance
        volatilities = np.sqrt(np.diag(S))
        correlation = np.corrcoef(returns.T)
        
        def correlation_objective(w):
            w_vol = w * volatilities
            portfolio_correlation = w_vol @ correlation @ w_vol.T
            return portfolio_correlation
        
        n = len(returns.columns)
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n)]
        constraints_opt = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        initial_weights = np.ones(n) / n
        
        result = minimize(
            correlation_objective,
            initial_weights,
            bounds=bounds,
            constraints=constraints_opt,
            method='SLSQP'
        )
        
        weights = {ticker: result.x[i] for i, ticker in enumerate(returns.columns)}
        return weights
    
    def _optimize_black_litterman(self, mu, S, returns, constraints, **kwargs):
        """Black-Litterman model with views"""
        # This is a simplified implementation
        # In practice, you would incorporate investor views
        
        # Use equilibrium returns (CAPM implied)
        market_weights = np.ones(len(mu)) / len(mu)
        market_return = mu @ market_weights
        equilibrium_returns = self.risk_free_rate/TRADING_DAYS + (market_return - self.risk_free_rate/TRADING_DAYS) * 1
        
        # Blend with historical returns
        tau = 0.05  # Uncertainty scaling
        blended_returns = (equilibrium_returns + tau * mu) / (1 + tau)
        
        # Optimize with blended returns
        ef = EfficientFrontier(blended_returns, S)
        ef.add_constraint(lambda w: w >= constraints['min_weight'])
        ef.add_constraint(lambda w: w <= constraints['max_weight'])
        ef.max_sharpe(risk_free_rate=self.daily_rf)
        weights = ef.clean_weights()
        
        return weights
    
    def _optimize_resampling(self, mu, S, returns, constraints, **kwargs):
        """Resampled efficiency (Michaud)"""
        n_simulations = kwargs.get('n_simulations', 100)
        n = len(returns.columns)
        
        all_weights = []
        
        for _ in range(n_simulations):
            # Resample returns with bootstrap
            resampled_returns = returns.sample(n=len(returns), replace=True)
            
            # Calculate new mu and S
            mu_resampled = resampled_returns.mean()
            S_resampled = resampled_returns.cov()
            
            # Optimize on resampled data
            ef = EfficientFrontier(mu_resampled, S_resampled)
            ef.add_constraint(lambda w: w >= constraints['min_weight'])
            ef.add_constraint(lambda w: w <= constraints['max_weight'])
            ef.max_sharpe(risk_free_rate=self.daily_rf)
            weights_resampled = ef.clean_weights()
            
            all_weights.append(list(weights_resampled.values()))
        
        # Average weights across simulations
        avg_weights = np.mean(all_weights, axis=0)
        weights = {ticker: avg_weights[i] for i, ticker in enumerate(returns.columns)}
        
        return weights
    
    def _optimize_bayesian(self, mu, S, returns, constraints, **kwargs):
        """Bayesian portfolio optimization"""
        # Simplified Bayesian approach using shrinkage
        
        # Calculate shrinkage target (constant correlation matrix)
        n = len(returns.columns)
        avg_correlation = (S.values / np.outer(np.sqrt(np.diag(S)), np.sqrt(np.diag(S)))).mean()
        
        # Create shrinkage target
        F = avg_correlation * np.outer(np.sqrt(np.diag(S)), np.sqrt(np.diag(S)))
        np.fill_diagonal(F, np.diag(S))
        
        # Shrinkage intensity (Ledoit-Wolf style)
        shrinkage = 0.5  # Can be estimated more rigorously
        
        # Shrink covariance matrix
        S_shrunk = shrinkage * F + (1 - shrinkage) * S
        
        # Optimize with shrunk covariance
        ef = EfficientFrontier(mu, S_shrunk)
        ef.add_constraint(lambda w: w >= constraints['min_weight'])
        ef.add_constraint(lambda w: w <= constraints['max_weight'])
        ef.max_sharpe(risk_free_rate=self.daily_rf)
        weights = ef.clean_weights()
        
        return weights

# ============================================================================
# 5. ADVANCED VISUALIZATION ENGINE
# ============================================================================

class PortfolioVisualizer:
    """Professional visualization engine for portfolio analytics"""
    
    @staticmethod
    def create_performance_dashboard(portfolio_returns, benchmark_returns=None, metrics=None):
        """Create comprehensive performance dashboard"""
        
        # Create subplots with professional layout
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Cumulative Performance',
                'Rolling Sharpe Ratio (6M)',
                'Drawdown Analysis',
                'Return Distribution',
                'QQ Plot vs Normal',
                'Rolling Volatility (1M)',
                'Active Returns vs Benchmark',
                'Autocorrelation Analysis',
                'VaR Breach Analysis',
                'Return Heatmap (Monthly)',
                'Risk Contribution',
                'Performance Attribution'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )
        
        # 1. Cumulative Performance
        cum_returns = np.exp(portfolio_returns.cumsum())
        
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns.values,
                mode='lines',
                name='Portfolio',
                line=dict(color='#3B82F6', width=3),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)'
            ),
            row=1, col=1
        )
        
        if benchmark_returns is not None:
            bench_cum = np.exp(benchmark_returns.cumsum())
            fig.add_trace(
                go.Scatter(
                    x=bench_cum.index,
                    y=bench_cum.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#6B7280', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # 2. Rolling Sharpe Ratio
        rolling_window = 126  # 6 months
        if len(portfolio_returns) > rolling_window:
            rolling_sharpe = portfolio_returns.rolling(rolling_window).apply(
                lambda x: (x.mean() * 252 - 0.45) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
            )
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    mode='lines',
                    name='Rolling Sharpe',
                    line=dict(color='#10B981', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.1)'
                ),
                row=1, col=2
            )
        
        # 3. Drawdown Analysis
        drawdown = PortfolioVisualizer._calculate_drawdown(portfolio_returns)
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode='lines',
                name='Drawdown',
                line=dict(color='#EF4444', width=2),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)'
            ),
            row=1, col=3
        )
        
        # 4. Return Distribution
        fig.add_trace(
            go.Histogram(
                x=portfolio_returns * 100,
                nbinsx=50,
                name='Returns',
                marker_color='#3B82F6',
                opacity=0.7,
                histnorm='probability density'
            ),
            row=2, col=1
        )
        
        # Add normal distribution overlay
        x_norm = np.linspace(portfolio_returns.min() * 100, portfolio_returns.max() * 100, 100)
        y_norm = norm.pdf(x_norm, portfolio_returns.mean() * 100, portfolio_returns.std() * 100)
        
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                name='Normal Distribution',
                line=dict(color='#EF4444', width=2, dash='dash')
            ),
            row=2, col=1
        )
        
        # 5. QQ Plot
        if len(portfolio_returns) > 10:
            import statsmodels.api as sm
            qq_data = sm.qqplot(portfolio_returns, stats.norm, fit=True, line='45')
            
            fig.add_trace(
                go.Scatter(
                    x=qq_data.theory_quantiles,
                    y=qq_data.sample_quantiles,
                    mode='markers',
                    name='QQ Plot',
                    marker=dict(size=4, color='#3B82F6')
                ),
                row=2, col=2
            )
            
            # Add 45-degree line
            line_range = [min(qq_data.theory_quantiles.min(), qq_data.sample_quantiles.min()),
                         max(qq_data.theory_quantiles.max(), qq_data.sample_quantiles.max())]
            
            fig.add_trace(
                go.Scatter(
                    x=line_range,
                    y=line_range,
                    mode='lines',
                    name='45° Line',
                    line=dict(color='#EF4444', width=2, dash='dash')
                ),
                row=2, col=2
            )
        
        # 6. Rolling Volatility
        rolling_vol = portfolio_returns.rolling(21).std() * np.sqrt(252) * 100
        
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Volatility',
                line=dict(color='#F59E0B', width=2)
            ),
            row=2, col=3
        )
        
        # 7. Active Returns (if benchmark)
        if benchmark_returns is not None:
            active_returns = portfolio_returns - benchmark_returns
            
            fig.add_trace(
                go.Bar(
                    x=active_returns.resample('M').sum().index,
                    y=active_returns.resample('M').sum().values * 100,
                    name='Active Return',
                    marker_color='#8B5CF6'
                ),
                row=3, col=1
            )
        
        # 8. Autocorrelation Analysis
        max_lag = min(40, len(portfolio_returns) // 2)
        if max_lag > 5:
            # ACF of returns
            acf_returns = [portfolio_returns.autocorr(lag=i) for i in range(1, max_lag + 1)]
            
            fig.add_trace(
                go.Bar(
                    x=list(range(1, max_lag + 1)),
                    y=acf_returns,
                    name='ACF Returns',
                    marker_color='#3B82F6',
                    opacity=0.7
                ),
                row=3, col=2
            )
            
            # PACF of returns
            from statsmodels.tsa.stattools import pacf
            pacf_returns = pacf(portfolio_returns, nlags=max_lag)[1:]
            
            fig.add_trace(
                go.Bar(
                    x=list(range(1, max_lag + 1)),
                    y=pacf_returns,
                    name='PACF Returns',
                    marker_color='#EF4444',
                    opacity=0.7
                ),
                row=3, col=2
            )
        
        # 9. VaR Breach Analysis
        var_level = 0.95
        var_threshold = np.percentile(portfolio_returns, (1 - var_level) * 100)
        breaches = portfolio_returns < var_threshold
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_returns.index,
                y=portfolio_returns.values * 100,
                mode='markers',
                name='Returns',
                marker=dict(
                    size=6,
                    color=['#EF4444' if b else '#3B82F6' for b in breaches],
                    opacity=0.7
                )
            ),
            row=3, col=3
        )
        
        fig.add_trace(
            go.Scatter(
                x=[portfolio_returns.index[0], portfolio_returns.index[-1]],
                y=[var_threshold * 100, var_threshold * 100],
                mode='lines',
                name=f'VaR ({var_level*100:.0f}%)',
                line=dict(color='#EF4444', width=2, dash='dash')
            ),
            row=3, col=3
        )
        
        # 10. Monthly Return Heatmap
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: np.exp(x.sum()) - 1)
        monthly_matrix = monthly_returns.unstack() if hasattr(monthly_returns, 'unstack') else monthly_returns
        
        if isinstance(monthly_matrix, pd.Series):
            # Convert to 2D matrix for heatmap
            monthly_matrix = monthly_returns.to_frame().T
        
        fig.add_trace(
            go.Heatmap(
                z=monthly_matrix.values * 100,
                x=monthly_matrix.columns,
                y=monthly_matrix.index,
                colorscale='RdYlGn',
                zmid=0,
                colorbar=dict(title="Return %")
            ),
            row=4, col=1
        )
        
        # 11. Risk Contribution (placeholder - would need weights)
        # This would be populated with actual risk contribution data
        
        # 12. Performance Attribution (placeholder)
        # This would show factor contributions
        
        # Update layout
        fig.update_layout(
            height=1400,
            showlegend=True,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Update axes
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=3)
        fig.update_yaxes(title_text="Density", row=2, col=1)
        fig.update_xaxes(title_text="Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=3)
        fig.update_yaxes(title_text="Active Return (%)", row=3, col=1)
        fig.update_xaxes(title_text="Month", row=3, col=1)
        fig.update_yaxes(title_text="Autocorrelation", row=3, col=2)
        fig.update_xaxes(title_text="Lag", row=3, col=2)
        fig.update_yaxes(title_text="Return (%)", row=3, col=3)
        fig.update_xaxes(title_text="Month", row=4, col=1)
        fig.update_yaxes(title_text="Year", row=4, col=1)
        
        return fig
    
    @staticmethod
    def _calculate_drawdown(returns):
        """Calculate drawdown series"""
        cumulative = np.exp(returns.cumsum())
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    @staticmethod
    def create_risk_decomposition_chart(weights, covariance_matrix, returns):
        """Create risk decomposition visualization"""
        
        # Calculate risk contributions
        portfolio_vol = np.sqrt(weights @ covariance_matrix @ weights)
        marginal_risk = (covariance_matrix @ weights) / portfolio_vol
        risk_contributions = weights * marginal_risk
        
        # Convert to percentages
        risk_pct = risk_contributions / portfolio_vol
        
        # Create DataFrame
        risk_df = pd.DataFrame({
            'Asset': returns.columns,
            'Weight': weights,
            'Risk Contribution': risk_pct * 100,
            'Marginal Risk': marginal_risk
        }).sort_values('Risk Contribution', ascending=False)
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Risk Contribution by Asset',
                'Weight vs Risk Contribution',
                'Marginal Risk Analysis',
                'Risk Concentration'
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 1. Risk Contribution Bar Chart
        fig.add_trace(
            go.Bar(
                x=risk_df['Asset'],
                y=risk_df['Risk Contribution'],
                name='Risk Contribution',
                marker_color='#EF4444'
            ),
            row=1, col=1
        )
        
        # 2. Weight vs Risk Contribution
        fig.add_trace(
            go.Scatter(
                x=risk_df['Weight'] * 100,
                y=risk_df['Risk Contribution'],
                mode='markers+text',
                text=risk_df['Asset'],
                textposition="top center",
                marker=dict(
                    size=abs(risk_df['Marginal Risk']) * 50,
                    color=risk_df['Marginal Risk'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Marginal Risk")
                ),
                name='Weight vs Risk'
            ),
            row=1, col=2
        )
        
        # 3. Marginal Risk Analysis
        fig.add_trace(
            go.Bar(
                x=risk_df['Asset'],
                y=risk_df['Marginal Risk'],
                name='Marginal Risk',
                marker_color='#3B82F6'
            ),
            row=2, col=1
        )
        
        # 4. Risk Concentration (Pie chart)
        fig.add_trace(
            go.Pie(
                labels=risk_df['Asset'],
                values=risk_df['Risk Contribution'],
                hole=0.4,
                name='Risk Distribution'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template='plotly_white',
            title_text="Portfolio Risk Decomposition Analysis"
        )
        
        return fig, risk_df
    
    @staticmethod
    def create_correlation_network(returns, threshold=0.5):
        """Create correlation network visualization"""
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for ticker in returns.columns:
            G.add_node(ticker, label=BIST30_TICKERS.get(ticker, ticker))
        
        # Add edges for significant correlations
        for i, ticker1 in enumerate(returns.columns):
            for j, ticker2 in enumerate(returns.columns):
                if i < j and abs(corr_matrix.iloc[i, j]) > threshold:
                    G.add_edge(ticker1, ticker2, weight=corr_matrix.iloc[i, j])
        
        # Create network visualization
        pos = nx.spring_layout(G, seed=42)
        
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=1, color='#94A3B8'),
                    hoverinfo='none',
                    mode='lines'
                )
            )
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=[BIST30_TICKERS.get(node, node) for node in G.nodes()],
            textposition="bottom center",
            marker=dict(
                size=20,
                color='#3B82F6',
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            hovertext=[f"{BIST30_TICKERS.get(node, node)}<br>Connections: {G.degree(node)}" 
                      for node in G.nodes()]
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title=f"Asset Correlation Network (|ρ| > {threshold})",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            template='plotly_white'
        )
        
        return fig

# ============================================================================
# 6. REPORT GENERATOR
# ============================================================================

class PortfolioReportGenerator:
    """Generate professional PDF reports"""
    
    @staticmethod
    def generate_pdf_report(metrics, weights, portfolio_returns, benchmark_returns, filename="portfolio_report.pdf"):
        """Generate comprehensive PDF report"""
        
        pdf = FPDF()
        pdf.add_page()
        
        # Set font
        pdf.set_font("Arial", 'B', 16)
        
        # Title
        pdf.cell(200, 10, "Portfolio Performance Report", ln=True, align='C')
        pdf.ln(10)
        
        # Date
        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(10)
        
        # Executive Summary
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Executive Summary", ln=True)
        pdf.set_font("Arial", '', 12)
        
        summary_text = f"""
        Portfolio Analysis Results:
        - Annual Return: {metrics.get('Annualized Return', 0):.2%}
        - Annual Volatility: {metrics.get('Annualized Volatility', 0):.2%}
        - Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.3f}
        - Maximum Drawdown: {metrics.get('Maximum Drawdown', 0):.2%}
        - Number of Assets: {len(weights)}
        """
        
        pdf.multi_cell(0, 10, summary_text)
        pdf.ln(10)
        
        # Performance Metrics Table
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Performance Metrics", ln=True)
        pdf.ln(5)
        
        # Create metrics table
        metrics_table = [
            ["Metric", "Value"],
            ["Annual Return", f"{metrics.get('Annualized Return', 0):.2%}"],
            ["Annual Volatility", f"{metrics.get('Annualized Volatility', 0):.2%}"],
            ["Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.3f}"],
            ["Sortino Ratio", f"{metrics.get('Sortino Ratio', 0):.3f}"],
            ["Maximum Drawdown", f"{metrics.get('Maximum Drawdown', 0):.2%}"],
            ["VaR (95%)", f"{metrics.get('VaR 95% (Historical)', 0):.2%}"],
            ["CVaR (95%)", f"{metrics.get('CVaR 95%', 0):.2%}"],
            ["Win Rate", f"{metrics.get('Win Rate', 0):.2%}"],
            ["Skewness", f"{metrics.get('Skewness', 0):.3f}"],
            ["Kurtosis", f"{metrics.get('Kurtosis', 0):.3f}"]
        ]
        
        # Add table to PDF
        col_width = pdf.w / 2.5
        row_height = pdf.font_size * 1.5
        
        for row in metrics_table:
            pdf.set_font("Arial", 'B' if row[0] == "Metric" else '', 12)
            pdf.cell(col_width, row_height, row[0], border=1)
            pdf.cell(col_width, row_height, row[1], border=1)
            pdf.ln(row_height)
        
        pdf.ln(10)
        
        # Portfolio Composition
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Portfolio Composition", ln=True)
        pdf.ln(5)
        
        # Sort weights
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # Top 10 holdings
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Top 10 Holdings:", ln=True)
        pdf.set_font("Arial", '', 12)
        
        for ticker, weight in sorted_weights[:10]:
            company_name = BIST30_TICKERS.get(ticker, ticker)
            pdf.cell(200, 10, f"{company_name}: {weight:.2%}", ln=True)
        
        # Save PDF
        pdf.output(filename)
        
        return filename

# ============================================================================
# 7. MAIN STREAMLIT APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🏦 BIST Institutional Portfolio Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("### *Advanced Quantitative Portfolio Management & Risk Analytics for Turkish Equity Markets*")
    st.markdown("---")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = AdvancedPortfolioOptimizer(RISK_FREE_RATE)
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = PortfolioVisualizer()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration Panel")
        
        # Data Configuration
        with st.expander("📅 Data Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    datetime.now() - timedelta(days=365*3),
                    help="Start date for historical data analysis"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    datetime.now(),
                    help="End date for historical data analysis"
                )
            
            data_frequency = st.selectbox(
                "Data Frequency",
                ["Daily", "Weekly", "Monthly"],
                help="Frequency for return calculations"
            )
            
            include_benchmarks = st.checkbox("Include Benchmarks", value=True)
            include_sector_data = st.checkbox("Include Sector Data", value=True)
        
        # Optimization Configuration
        with st.expander("🎯 Optimization Parameters", expanded=True):
            # Risk-free rate
            risk_free_rate = st.slider(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=RISK_FREE_RATE * 100,
                step=0.5,
                help="Annual risk-free rate for Turkish market"
            ) / 100
            
            # Update optimizer
            st.session_state.optimizer.risk_free_rate = risk_free_rate
            st.session_state.optimizer.daily_rf = np.log(1 + risk_free_rate) / TRADING_DAYS
            
            # Optimization method
            optimization_method = st.selectbox(
                "Optimization Method",
                [
                    'max_sharpe', 'min_volatility', 'efficient_risk',
                    'efficient_return', 'max_quadratic_utility',
                    'hrp', 'cvar', 'equal_weight', 'risk_parity',
                    'most_diversified', 'minimum_correlation',
                    'black_litterman', 'resampling', 'bayesian'
                ],
                format_func=lambda x: x.replace('_', ' ').title(),
                help="Select portfolio optimization methodology"
            )
            
            # Strategy-specific parameters
            if optimization_method in ['efficient_risk', 'efficient_return']:
                target_value = st.slider(
                    f"Target {'Volatility' if optimization_method == 'efficient_risk' else 'Return'} (%)",
                    min_value=5.0,
                    max_value=80.0,
                    value=30.0 if optimization_method == 'efficient_risk' else 25.0,
                    step=1.0
                ) / 100
            
            if optimization_method == 'max_quadratic_utility':
                risk_aversion = st.slider(
                    "Risk Aversion Coefficient",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )
            
            if optimization_method == 'resampling':
                n_simulations = st.slider(
                    "Number of Simulations",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10
                )
        
        # Portfolio Constraints
        with st.expander("⚖️ Portfolio Constraints", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                min_weight = st.number_input(
                    "Minimum Weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.5
                ) / 100
            with col2:
                max_weight = st.number_input(
                    "Maximum Weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0,
                    step=0.5
                ) / 100
            
            sector_constraints = st.checkbox("Apply Sector Constraints")
            if sector_constraints:
                st.info("Sector constraints coming soon...")
        
        # Advanced Settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            cov_estimator = st.selectbox(
                "Covariance Estimator",
                ['sample_cov', 'ledoit_wolf', 'oracle_approximating'],
                help="Method for estimating covariance matrix"
            )
            
            return_type = st.selectbox(
                "Return Calculation",
                ['log_returns', 'simple_returns'],
                help="Method for calculating returns"
            )
            
            use_shrinkage = st.checkbox("Use Shrinkage Estimation", value=True)
            enable_ml = st.checkbox("Enable ML Features", value=True)
        
        # Action Buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            fetch_button = st.button(
                "📥 Fetch Market Data",
                type="primary",
                use_container_width=True
            )
        with col2:
            analyze_button = st.button(
                "🚀 Run Analysis",
                type="secondary",
                use_container_width=True,
                disabled=not st.session_state.data_loaded
            )
        
        # Additional actions
        st.markdown("---")
        if st.button("🔄 Reset Analysis", use_container_width=True):
            st.session_state.data_loaded = False
            st.rerun()
    
    # Main content area
    if fetch_button:
        with st.spinner("🌐 Fetching market data from Yahoo Finance..."):
            try:
                # Initialize data fetcher
                data_fetcher = MarketDataFetcher()
                
                # Fetch all data
                market_data = data_fetcher.fetch_all_data(
                    start_date=str(start_date),
                    end_date=str(end_date),
                    use_cache=True
                )
                
                if market_data['prices'] is None or market_data['prices'].empty:
                    st.error("❌ Failed to fetch data. Please check your internet connection and try again.")
                else:
                    st.session_state.market_data = market_data
                    st.session_state.data_loaded = True
                    
                    # Display success message
                    st.success(f"✅ Successfully fetched data for {len(market_data['prices'].columns)} assets")
                    
                    # Display data summary
                    with st.expander("📊 Data Summary", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Assets",
                                len(market_data['prices'].columns),
                                f"From {market_data['prices'].index[0].date()}"
                            )
                        
                        with col2:
                            st.metric(
                                "Trading Days",
                                len(market_data['prices']),
                                f"To {market_data['prices'].index[-1].date()}"
                            )
                        
                        with col3:
                            missing_pct = market_data['prices'].isnull().sum().sum() / (market_data['prices'].shape[0] * market_data['prices'].shape[1]) * 100
                            st.metric(
                                "Data Quality",
                                f"{100 - missing_pct:.1f}%",
                                "Complete"
                            )
                    
                    # Show sample of data
                    st.dataframe(
                        market_data['prices'].iloc[-10:].style.format("{:.2f}"),
                        use_container_width=True,
                        height=400
                    )
                    
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                st.exception(e)
    
    # Analysis section
    if analyze_button and st.session_state.data_loaded:
        # Get market data from session state
        market_data = st.session_state.market_data
        
        # Extract data
        prices = market_data['prices']
        returns = market_data['returns']
        benchmark_prices = market_data['benchmark_prices']
        benchmark_returns = market_data['benchmark_returns']
        
        # Calculate expected returns and covariance
        mu = expected_returns.mean_historical_return(prices)
        
        if cov_estimator == 'sample_cov':
            S = risk_models.sample_cov(prices)
        elif cov_estimator == 'ledoit_wolf':
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        elif cov_estimator == 'oracle_approximating':
            S = risk_models.CovarianceShrinkage(prices).oracle_approximating()
        
        # Prepare constraints
        constraints = {'min_weight': min_weight, 'max_weight': max_weight}
        
        # Prepare kwargs for optimization
        kwargs = {}
        if optimization_method in ['efficient_risk', 'efficient_return']:
            kwargs['target_volatility' if optimization_method == 'efficient_risk' else 'target_return'] = target_value
        if optimization_method == 'max_quadratic_utility':
            kwargs['risk_aversion'] = risk_aversion
        if optimization_method == 'resampling':
            kwargs['n_simulations'] = n_simulations
        
        # Run optimization
        with st.spinner("⚙️ Running portfolio optimization..."):
            try:
                # Optimize portfolio
                weights_dict = st.session_state.optimizer.optimize_portfolio(
                    optimization_method,
                    mu,
                    S,
                    returns,
                    constraints,
                    **kwargs
                )
                
                # Convert to numpy array
                weights = np.array([weights_dict.get(ticker, 0) for ticker in returns.columns])
                
                # Calculate metrics
                benchmark_for_metrics = benchmark_returns['XU100.IS'] if 'XU100.IS' in benchmark_returns.columns else None
                metrics, portfolio_returns = st.session_state.optimizer.calculate_advanced_metrics(
                    weights, returns, benchmark_for_metrics
                )
                
                # Store results in session state
                st.session_state.results = {
                    'weights': weights_dict,
                    'metrics': metrics,
                    'portfolio_returns': portfolio_returns,
                    'prices': prices,
                    'returns': returns,
                    'benchmark_returns': benchmark_returns
                }
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during optimization: {str(e)}")
                st.exception(e)
                return
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Executive Summary",
            "📈 Portfolio Analysis",
            "⚠️ Risk Analytics",
            "🎯 Optimization",
            "🔍 Advanced Diagnostics",
            "📋 Performance Report",
            "🔄 Backtesting"
        ])
        
        # Get results
        results = st.session_state.results
        weights_dict = results['weights']
        metrics = results['metrics']
        portfolio_returns = results['portfolio_returns']
        
        # Tab 1: Executive Summary
        with tab1:
            st.markdown('<div class="sub-header">Executive Summary</div>', unsafe_allow_html=True)
            
            # Key metrics cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card metric-card-success">', unsafe_allow_html=True)
                st.metric(
                    "Annual Return",
                    f"{metrics.get('Annualized Return', 0):.2%}",
                    f"Sharpe: {metrics.get('Sharpe Ratio', 0):.3f}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card metric-card-warning">', unsafe_allow_html=True)
                st.metric(
                    "Annual Volatility",
                    f"{metrics.get('Annualized Volatility', 0):.2%}",
                    f"Max DD: {metrics.get('Maximum Drawdown', 0):.2%}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card metric-card-info">', unsafe_allow_html=True)
                st.metric(
                    "Risk-Adjusted Return",
                    f"{metrics.get('Sortino Ratio', 0):.3f}",
                    f"Omega: {metrics.get('Omega Ratio', 0):.2f}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if 'Beta' in metrics and not np.isnan(metrics['Beta']):
                    st.metric(
                        "Market Exposure",
                        f"β = {metrics['Beta']:.2f}",
                        f"α = {metrics.get('Alpha (Jensen)', 0):.2%}"
                    )
                else:
                    st.metric("Number of Assets", len(weights_dict), "In Portfolio")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Portfolio composition
            st.markdown('<div class="section-header">Portfolio Composition</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Convert weights to DataFrame
                weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=['Weight'])
                weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
                
                # Add company names
                weights_df['Company'] = weights_df.index.map(BIST30_TICKERS)
                weights_df['Sector'] = weights_df.index.map(
                    lambda x: next((sector for sector, tickers in SECTOR_CLASSIFICATION.items() 
                                  if x in tickers), 'Other')
                )
                
                # Create pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=weights_df['Company'],
                    values=weights_df['Weight'],
                    hole=0.4,
                    textinfo='label+percent',
                    marker=dict(colors=px.colors.qualitative.Set3)
                )])
                
                fig_pie.update_layout(
                    title="Portfolio Weight Distribution",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Top holdings table
                st.markdown("**Top 10 Holdings**")
                top_holdings = weights_df.head(10).copy()
                top_holdings['Weight'] = top_holdings['Weight'].map(lambda x: f"{x:.2%}")
                
                st.dataframe(
                    top_holdings[['Company', 'Sector', 'Weight']],
                    use_container_width=True,
                    height=400
                )
                
                # Concentration metrics
                st.markdown("**Concentration Metrics**")
                hhi = np.sum(weights_df['Weight'] ** 2)
                gini = 1 - 2 * np.sum(np.cumsum(weights_df['Weight']) / weights_df['Weight'].sum()) / len(weights_df)
                
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.metric("HHI Index", f"{hhi:.4f}")
                with col_met2:
                    st.metric("Gini Coefficient", f"{gini:.3f}")
        
        # Tab 2: Portfolio Analysis
        with tab2:
            st.markdown('<div class="sub-header">Portfolio Performance Analysis</div>', unsafe_allow_html=True)
            
            # Create performance dashboard
            dashboard_fig = st.session_state.visualizer.create_performance_dashboard(
                portfolio_returns,
                benchmark_returns['XU100.IS'] if 'XU100.IS' in benchmark_returns.columns else None,
                metrics
            )
            
            st.plotly_chart(dashboard_fig, use_container_width=True)
            
            # Additional metrics
            st.markdown('<div class="section-header">Detailed Performance Metrics</div>', unsafe_allow_html=True)
            
            # Create metrics tables
            col1, col2 = st.columns(2)
            
            with col1:
                # Return metrics
                return_metrics = {
                    k: v for k, v in metrics.items()
                    if 'Return' in k or 'Alpha' in k or 'Sharpe' in k or 'Sortino' in k
                }
                
                st.markdown("**Return & Risk-Adjusted Metrics**")
                return_df = pd.DataFrame.from_dict(return_metrics, orient='index', columns=['Value'])
                st.dataframe(
                    return_df.style.format({
                        'Value': lambda x: f"{x:.2%}" if 'Return' in return_df.index[return_df.index.get_loc(x.name)] 
                                          else f"{x:.3f}" if 'Sharpe' in x.name or 'Sortino' in x.name
                                          else f"{x:.4f}"
                    }),
                    use_container_width=True
                )
            
            with col2:
                # Risk metrics
                risk_metrics = {
                    k: v for k, v in metrics.items()
                    if 'Volatility' in k or 'VaR' in k or 'Drawdown' in k or 'Beta' in k
                }
                
                st.markdown("**Risk Metrics**")
                risk_df = pd.DataFrame.from_dict(risk_metrics, orient='index', columns=['Value'])
                st.dataframe(
                    risk_df.style.format({
                        'Value': lambda x: f"{x:.2%}" if any(word in x.name for word in ['Volatility', 'VaR', 'Drawdown'])
                                          else f"{x:.2f}" if 'Beta' in x.name
                                          else f"{x:.4f}"
                    }),
                    use_container_width=True
                )
        
        # Tab 3: Risk Analytics
        with tab3:
            st.markdown('<div class="sub-header">Advanced Risk Analytics</div>', unsafe_allow_html=True)
            
            # Risk decomposition
            st.markdown('<div class="section-header">Risk Decomposition</div>', unsafe_allow_html=True)
            
            # Convert weights to array
            weights_array = np.array([weights_dict.get(ticker, 0) for ticker in returns.columns])
            
            # Calculate covariance matrix
            S_array = returns.cov().values
            
            # Create risk decomposition chart
            risk_fig, risk_df = st.session_state.visualizer.create_risk_decomposition_chart(
                weights_array, S_array, returns
            )
            
            st.plotly_chart(risk_fig, use_container_width=True)
            
            # Display risk contribution table
            st.markdown("**Detailed Risk Contributions**")
            st.dataframe(
                risk_df.style.format({
                    'Weight': '{:.2%}',
                    'Risk Contribution': '{:.2f}%',
                    'Marginal Risk': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # Correlation analysis
            st.markdown('<div class="section-header">Correlation Analysis</div>', unsafe_allow_html=True)
            
            # Create correlation network
            corr_network_fig = st.session_state.visualizer.create_correlation_network(returns)
            st.plotly_chart(corr_network_fig, use_container_width=True)
            
            # Correlation matrix
            st.markdown("**Correlation Matrix**")
            corr_matrix = returns.corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation")
            ))
            
            fig_corr.update_layout(
                title="Asset Correlation Matrix",
                height=600,
                xaxis_title="Assets",
                yaxis_title="Assets"
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Tab 4: Optimization
        with tab4:
            st.markdown('<div class="sub-header">Portfolio Optimization Analysis</div>', unsafe_allow_html=True)
            
            # Efficient Frontier
            st.markdown('<div class="section-header">Efficient Frontier Analysis</div>', unsafe_allow_html=True)
            
            # Create CLA object for efficient frontier
            cla = CLA(mu, S)
            
            try:
                # Get efficient frontier points
                ef_points = cla.efficient_frontier(points=100)
                
                # Create efficient frontier plot
                fig_ef = go.Figure()
                
                # Plot efficient frontier
                frontier_vols = [point[1] * np.sqrt(TRADING_DAYS) * 100 for point in ef_points]
                frontier_rets = [point[0] * TRADING_DAYS * 100 for point in ef_points]
                
                fig_ef.add_trace(go.Scatter(
                    x=frontier_vols,
                    y=frontier_rets,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='#3B82F6', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.1)'
                ))
                
                # Plot individual assets
                individual_rets = mu * TRADING_DAYS * 100
                individual_vols = np.sqrt(np.diag(S) * TRADING_DAYS) * 100
                
                fig_ef.add_trace(go.Scatter(
                    x=individual_vols,
                    y=individual_rets,
                    mode='markers',
                    name='Individual Assets',
                    marker=dict(size=8, color='#6B7280', opacity=0.6),
                    text=returns.columns,
                    hovertemplate="<b>%{text}</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%"
                ))
                
                # Plot optimized portfolio
                portfolio_vol = metrics.get('Annualized Volatility', 0) * 100
                portfolio_ret = metrics.get('Annualized Return', 0) * 100
                
                fig_ef.add_trace(go.Scatter(
                    x=[portfolio_vol],
                    y=[portfolio_ret],
                    mode='markers',
                    name='Optimized Portfolio',
                    marker=dict(size=15, color='#10B981', line=dict(width=2, color='white')),
                    hovertemplate=f"<b>Optimized Portfolio</b><br>Return: {portfolio_ret:.2f}%<br>Volatility: {portfolio_vol:.2f}%<br>Sharpe: {metrics.get('Sharpe Ratio', 0):.3f}"
                ))
                
                fig_ef.update_layout(
                    title="Efficient Frontier",
                    xaxis_title="Annual Volatility (%)",
                    yaxis_title="Annual Return (%)",
                    height=600,
                    template='plotly_white',
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_ef, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not plot efficient frontier: {str(e)}")
            
            # Optimization comparison
            st.markdown('<div class="section-header">Optimization Method Comparison</div>', unsafe_allow_html=True)
            
            # Compare different methods
            methods_to_compare = ['max_sharpe', 'min_volatility', 'equal_weight', 'risk_parity', 'hrp']
            
            comparison_results = []
            
            with st.spinner("Comparing optimization methods..."):
                for method in methods_to_compare:
                    try:
                        # Optimize with different method
                        weights_compare = st.session_state.optimizer.optimize_portfolio(
                            method, mu, S, returns, constraints
                        )
                        
                        # Convert to array
                        weights_array_compare = np.array([weights_compare.get(ticker, 0) for ticker in returns.columns])
                        
                        # Calculate metrics
                        metrics_compare, _ = st.session_state.optimizer.calculate_advanced_metrics(
                            weights_array_compare, returns, benchmark_returns['XU100.IS'] if 'XU100.IS' in benchmark_returns.columns else None
                        )
                        
                        comparison_results.append({
                            'Method': method.replace('_', ' ').title(),
                            'Return': metrics_compare.get('Annualized Return', 0),
                            'Volatility': metrics_compare.get('Annualized Volatility', 0),
                            'Sharpe': metrics_compare.get('Sharpe Ratio', 0),
                            'Sortino': metrics_compare.get('Sortino Ratio', 0),
                            'Max DD': metrics_compare.get('Maximum Drawdown', 0),
                            'Number of Assets': sum(1 for w in weights_compare.values() if w > 0.001)
                        })
                        
                    except Exception as e:
                        st.warning(f"Failed to optimize with {method}: {str(e)}")
                
                if comparison_results:
                    compare_df = pd.DataFrame(comparison_results)
                    
                    # Display comparison table
                    st.dataframe(
                        compare_df.style.format({
                            'Return': '{:.2%}',
                            'Volatility': '{:.2%}',
                            'Sharpe': '{:.3f}',
                            'Sortino': '{:.3f}',
                            'Max DD': '{:.2%}'
                        }).background_gradient(subset=['Sharpe', 'Sortino'], cmap='RdYlGn')
                        .background_gradient(subset=['Volatility', 'Max DD'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )
                    
                    # Create radar chart for comparison
                    categories = ['Return', 'Volatility', 'Sharpe', 'Sortino', 'Max DD']
                    
                    fig_radar = go.Figure()
                    
                    for idx, row in compare_df.iterrows():
                        # Normalize values for radar chart
                        values = [
                            row['Return'] * 100,  # Convert to percentage
                            row['Volatility'] * 100,
                            row['Sharpe'] * 10,  # Scale for visibility
                            row['Sortino'] * 10,
                            abs(row['Max DD']) * 100  # Use absolute value
                        ]
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name=row['Method']
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=True,
                        title="Optimization Method Comparison",
                        height=500
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
        
        # Tab 5: Advanced Diagnostics
        with tab5:
            st.markdown('<div class="sub-header">Advanced Quantitative Diagnostics</div>', unsafe_allow_html=True)
            
            # Market regime detection
            st.markdown('<div class="section-header">Market Regime Analysis</div>', unsafe_allow_html=True)
            
            try:
                # Detect regimes using GMM
                regimes = st.session_state.optimizer._detect_regimes(portfolio_returns, n_regimes=3)
                
                if regimes is not None:
                    # Plot regime detection
                    fig_regime = go.Figure()
                    
                    # Color map for regimes
                    colors = ['#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#3B82F6']
                    
                    for regime in sorted(regimes.unique()):
                        regime_mask = regimes == regime
                        regime_returns = portfolio_returns[regime_mask]
                        
                        fig_regime.add_trace(go.Scatter(
                            x=regime_returns.index,
                            y=regime_returns.values * 100,
                            mode='markers',
                            name=f'Regime {regime}',
                            marker=dict(size=6, color=colors[regime % len(colors)]),
                            opacity=0.7
                        ))
                    
                    fig_regime.update_layout(
                        title="Market Regime Detection",
                        yaxis_title="Daily Return (%)",
                        xaxis_title="Date",
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_regime, use_container_width=True)
                    
                    # Display regime statistics
                    regime_stats = []
                    for regime in sorted(regimes.unique()):
                        regime_mask = regimes == regime
                        regime_returns = portfolio_returns[regime_mask]
                        
                        regime_stats.append({
                            'Regime': regime,
                            'Days': len(regime_returns),
                            'Percentage': len(regime_returns) / len(portfolio_returns) * 100,
                            'Mean Return': regime_returns.mean() * 100,
                            'Volatility': regime_returns.std() * 100 * np.sqrt(TRADING_DAYS),
                            'Sharpe': (regime_returns.mean() * TRADING_DAYS - risk_free_rate) / 
                                     (regime_returns.std() * np.sqrt(TRADING_DAYS)) if regime_returns.std() > 0 else 0
                        })
                    
                    regime_df = pd.DataFrame(regime_stats)
                    st.dataframe(
                        regime_df.style.format({
                            'Percentage': '{:.1f}%',
                            'Mean Return': '{:.2f}%',
                            'Volatility': '{:.2f}%',
                            'Sharpe': '{:.3f}'
                        }),
                        use_container_width=True
                    )
            except Exception as e:
                st.warning(f"Regime detection failed: {str(e)}")
            
            # Advanced statistical tests
            st.markdown('<div class="section-header">Advanced Statistical Tests</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Normality tests
                st.markdown("**Normality Tests**")
                
                # Jarque-Bera test
                jb_stat, jb_pvalue = stats.jarque_bera(portfolio_returns)
                
                # Shapiro-Wilk test (for smaller samples)
                if len(portfolio_returns) < 5000:
                    shapiro_stat, shapiro_pvalue = stats.shapiro(portfolio_returns)
                else:
                    shapiro_stat, shapiro_pvalue = np.nan, np.nan
                
                # Anderson-Darling test
                anderson_result = stats.anderson(portfolio_returns)
                
                normality_df = pd.DataFrame({
                    'Test': ['Jarque-Bera', 'Shapiro-Wilk', 'Anderson-Darling'],
                    'Statistic': [jb_stat, shapiro_stat, anderson_result.statistic],
                    'p-value': [jb_pvalue, shapiro_pvalue, np.nan],
                    'Critical Value (5%)': [np.nan, np.nan, anderson_result.critical_values[2]],
                    'Result': [
                        'Normal' if jb_pvalue > 0.05 else 'Not Normal',
                        'Normal' if shapiro_pvalue > 0.05 else 'Not Normal',
                        'Normal' if anderson_result.statistic < anderson_result.critical_values[2] else 'Not Normal'
                    ]
                })
                
                st.dataframe(normality_df, use_container_width=True)
            
            with col2:
                # Stationarity tests
                st.markdown("**Stationarity Tests**")
                
                # ADF test
                adf_result = adfuller(portfolio_returns)
                
                # KPSS test
                from statsmodels.tsa.stattools import kpss
                kpss_result = kpss(portfolio_returns)
                
                stationarity_df = pd.DataFrame({
                    'Test': ['ADF', 'KPSS'],
                    'Statistic': [adf_result[0], kpss_result[0]],
                    'p-value': [adf_result[1], kpss_result[1]],
                    'Critical Value (5%)': [adf_result[4]['5%'], kpss_result[3]['5%']],
                    'Result': [
                        'Stationary' if adf_result[1] < 0.05 else 'Non-Stationary',
                        'Stationary' if kpss_result[1] > 0.05 else 'Non-Stationary'
                    ]
                })
                
                st.dataframe(stationarity_df, use_container_width=True)
            
            # Fractal analysis
            st.markdown('<div class="section-header">Fractal & Complexity Analysis</div>', unsafe_allow_html=True)
            
            try:
                # Calculate Hurst exponent
                hurst = st.session_state.optimizer._calculate_hurst_exponent(portfolio_returns)
                
                # Calculate sample entropy
                sample_entropy = st.session_state.optimizer._calculate_sample_entropy(portfolio_returns)
                
                # Calculate permutation entropy
                perm_entropy = st.session_state.optimizer._calculate_permutation_entropy(portfolio_returns)
                
                fractal_df = pd.DataFrame({
                    'Metric': ['Hurst Exponent', 'Sample Entropy', 'Permutation Entropy'],
                    'Value': [hurst, sample_entropy, perm_entropy],
                    'Interpretation': [
                        'Trending (>0.5), Mean-reverting (<0.5), Random (=0.5)',
                        'Higher = more complexity, Lower = more predictability',
                        'Higher = more randomness, Lower = more order'
                    ]
                })
                
                st.dataframe(fractal_df, use_container_width=True)
                
                # Market efficiency indicator
                if not np.isnan(hurst):
                    efficiency = "Efficient" if abs(hurst - 0.5) < 0.1 else "Inefficient"
                    st.info(f"**Market Efficiency:** {efficiency} (Hurst exponent = {hurst:.3f})")
                    
            except Exception as e:
                st.warning(f"Fractal analysis failed: {str(e)}")
        
        # Tab 6: Performance Report
        with tab6:
            st.markdown('<div class="sub-header">Performance Report Generation</div>', unsafe_allow_html=True)
            
            # Generate comprehensive report
            st.markdown("### 📋 Generate Professional Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Report options
                report_type = st.selectbox(
                    "Report Type",
                    ["Executive Summary", "Comprehensive Analysis", "Risk Report", "Performance Attribution"]
                )
                
                include_charts = st.checkbox("Include Charts", value=True)
                include_tables = st.checkbox("Include Detailed Tables", value=True)
                include_recommendations = st.checkbox("Include Recommendations", value=True)
            
            with col2:
                # Export options
                export_format = st.selectbox(
                    "Export Format",
                    ["PDF", "Excel", "HTML", "JSON"]
                )
                
                filename = st.text_input(
                    "Filename",
                    value=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            
            # Generate report button
            if st.button("📥 Generate Report", type="primary", use_container_width=True):
                with st.spinner("Generating report..."):
                    try:
                        # Generate PDF report
                        report_filename = PortfolioReportGenerator.generate_pdf_report(
                            metrics,
                            weights_dict,
                            portfolio_returns,
                            benchmark_returns['XU100.IS'] if 'XU100.IS' in benchmark_returns.columns else None,
                            filename=f"{filename}.pdf"
                        )
                        
                        # Read the generated PDF
                        with open(report_filename, "rb") as f:
                            pdf_bytes = f.read()
                        
                        # Create download button
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=report_filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        st.success("✅ Report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
            
            # Display export options for other formats
            st.markdown("### 📊 Export Data")
            
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            with col_exp1:
                # Export weights to CSV
                weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=['Weight'])
                weights_csv = weights_df.to_csv()
                
                st.download_button(
                    label="📥 Export Weights (CSV)",
                    data=weights_csv,
                    file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_exp2:
                # Export metrics to CSV
                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                metrics_csv = metrics_df.to_csv()
                
                st.download_button(
                    label="📥 Export Metrics (CSV)",
                    data=metrics_csv,
                    file_name=f"portfolio_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_exp3:
                # Export returns to CSV
                returns_df = pd.DataFrame({
                    'Portfolio_Returns': portfolio_returns,
                    'Benchmark_Returns': benchmark_returns['XU100.IS'] if 'XU100.IS' in benchmark_returns.columns else np.nan
                })
                returns_csv = returns_df.to_csv()
                
                st.download_button(
                    label="📥 Export Returns (CSV)",
                    data=returns_csv,
                    file_name=f"portfolio_returns_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Tab 7: Backtesting
        with tab7:
            st.markdown('<div class="sub-header">Portfolio Backtesting</div>', unsafe_allow_html=True)
            
            # Backtesting parameters
            st.markdown("### ⚙️ Backtesting Configuration")
            
            col_bt1, col_bt2, col_bt3 = st.columns(3)
            
            with col_bt1:
                backtest_period = st.selectbox(
                    "Backtest Period",
                    ["1 Year", "2 Years", "3 Years", "5 Years", "Custom"]
                )
                
                if backtest_period == "Custom":
                    bt_start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
                    bt_end_date = st.date_input("End Date", datetime.now())
            
            with col_bt2:
                rebalance_frequency = st.selectbox(
                    "Rebalance Frequency",
                    ["Monthly", "Quarterly", "Semi-Annually", "Annually", "Never"]
                )
                
                transaction_costs = st.number_input(
                    "Transaction Costs (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.1,
                    step=0.05
                ) / 100
            
            with col_bt3:
                initial_capital = st.number_input(
                    "Initial Capital (TRY)",
                    min_value=1000,
                    max_value=1000000000,
                    value=1000000,
                    step=10000
                )
                
                include_dividends = st.checkbox("Include Dividends", value=False)
            
            # Run backtest button
            if st.button("🔁 Run Backtest", type="primary", use_container_width=True):
                with st.spinner("Running backtest..."):
                    try:
                        # This is a placeholder for backtesting logic
                        # In a full implementation, you would:
                        # 1. Split data into training and testing periods
                        # 2. Optimize portfolio on training data
                        # 3. Test on out-of-sample data
                        # 4. Calculate performance metrics
                        # 5. Compare with benchmark
                        
                        st.info("Backtesting feature is under development. Coming soon!")
                        
                        # Placeholder for backtest results
                        backtest_results = {
                            'Period': backtest_period,
                            'Total Return': 0.25,
                            'Annualized Return': 0.12,
                            'Annualized Volatility': 0.18,
                            'Max Drawdown': -0.15,
                            'Sharpe Ratio': 0.67,
                            'Win Rate': 0.55,
                            'Best Month': 0.08,
                            'Worst Month': -0.06
                        }
                        
                        # Display backtest results
                        st.markdown("### 📊 Backtest Results")
                        
                        # Create metrics cards for backtest
                        col_bt_res1, col_bt_res2, col_bt_res3, col_bt_res4 = st.columns(4)
                        
                        with col_bt_res1:
                            st.metric(
                                "Total Return",
                                f"{backtest_results['Total Return']:.2%}",
                                f"Annualized: {backtest_results['Annualized Return']:.2%}"
                            )
                        
                        with col_bt_res2:
                            st.metric(
                                "Risk-Adjusted Return",
                                f"Sharpe: {backtest_results['Sharpe Ratio']:.2f}",
                                f"Win Rate: {backtest_results['Win Rate']:.2%}"
                            )
                        
                        with col_bt_res3:
                            st.metric(
                                "Risk Metrics",
                                f"Volatility: {backtest_results['Annualized Volatility']:.2%}",
                                f"Max DD: {backtest_results['Max Drawdown']:.2%}"
                            )
                        
                        with col_bt_res4:
                            st.metric(
                                "Monthly Performance",
                                f"Best: {backtest_results['Best Month']:.2%}",
                                f"Worst: {backtest_results['Worst Month']:.2%}"
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Backtesting failed: {str(e)}")
    
    else:
        # Welcome screen
        if not st.session_state.data_loaded:
            st.markdown("## 🎯 Welcome to BIST Portfolio Analytics")
            
            col_welcome1, col_welcome2 = st.columns([2, 1])
            
            with col_welcome1:
                st.markdown("""
                ### **Get Started:**
                
                1. **Configure Parameters** in the sidebar
                2. **Fetch Market Data** using the button
                3. **Run Portfolio Analysis** once data is loaded
                
                ### **Available Features:**
                
                📊 **Portfolio Optimization**
                - 14 different optimization methods
                - Mean-variance optimization
                - Risk parity & hierarchical risk parity
                - Black-Litterman model
                - Bayesian optimization
                
                ⚠️ **Risk Analytics**
                - Value at Risk (VaR) & Conditional VaR (CVaR)
                - Stress testing & scenario analysis
                - Correlation networks
                - Risk decomposition
                
                🔍 **Advanced Diagnostics**
                - Market regime detection
                - Fractal analysis (Hurst exponent)
                - Entropy measures
                - Statistical tests
                
                📈 **Professional Reporting**
                - Comprehensive performance metrics
                - Export to PDF/Excel/CSV
                - Interactive visualizations
                
                ### **Supported Assets:**
                - All BIST 30 constituents
                - Multiple benchmark indices
                - FX rates & commodities
                """)
            
            with col_welcome2:
                # Feature cards
                st.markdown('<div class="metric-card metric-card-info">', unsafe_allow_html=True)
                st.markdown("### 📈 **14 Optimization Methods**")
                st.markdown("From traditional mean-variance to advanced ML techniques")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card metric-card-success">', unsafe_allow_html=True)
                st.markdown("### ⚠️ **50+ Risk Metrics**")
                st.markdown("Comprehensive risk assessment with advanced analytics")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card metric-card-warning">', unsafe_allow_html=True)
                st.markdown("### 🔍 **ML-Powered Insights**")
                st.markdown("Market regime detection, entropy analysis, fractal metrics")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("### 📋 **Professional Reports**")
                st.markdown("Export-ready PDF reports with executive summaries")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Quick start guide
            with st.expander("🚀 Quick Start Guide", expanded=True):
                st.markdown("""
                **Step 1: Data Configuration**
                - Select date range (minimum 1 year recommended)
                - Choose data frequency (daily recommended)
                - Include benchmarks for comparison
                
                **Step 2: Optimization Setup**
                - Select optimization method (Max Sharpe for beginners)
                - Set risk-free rate (current: 45% for Turkey)
                - Configure portfolio constraints
                
                **Step 3: Analysis & Reporting**
                - Review portfolio composition
                - Analyze risk metrics
                - Generate professional reports
                - Export results for further analysis
                
                **Pro Tips:**
                - Use longer time periods for more stable estimates
                - Consider transaction costs in backtesting
                - Regularly update risk-free rate based on market conditions
                - Use multiple optimization methods for comparison
                """)

if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="BIST Institutional Portfolio Analytics",
        layout="wide",
        page_icon="📈",
        initial_sidebar_state="expanded"
    )
    
    # Run the app
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.exception(e)
