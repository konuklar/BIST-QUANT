# ============================================================================
# BIST ENTERPRISE QUANT PORTFOLIO OPTIMIZATION SUITE PRO MAX
# Version: 8.0 | Features: Full PyPortfolioOpt + QuantStats Integration
# Institutional-Grade Analytics & Visualizations
# ============================================================================

import warnings
import sys
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
import requests
import base64
import logging
import traceback
import time
import os
import json
import io
from typing import Dict, List, Optional, Tuple, Any

# ─────────────────────────────────────────────────────────────────────────────
# CORE IMPORTS WITH ROBUST ERROR HANDLING
# ─────────────────────────────────────────────────────────────────────────────

# Force disable some warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing yfinance with fallback
try:
    import yfinance as yf
    yf.pdr_override()  # Override pandas-datareader if present
    HAS_YFINANCE = True
except ImportError as e:
    st.error(f"yfinance import error: {e}")
    HAS_YFINANCE = False

# Try importing PyPortfolioOpt
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    HAS_PYPFOPT = True
except ImportError as e:
    st.error(f"PyPortfolioOpt import error: {e}")
    HAS_PYPFOPT = False

# Try importing QuantStats
try:
    import quantstats as qs
    # Extend pandas for quantstats functionality
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False

# Try importing scikit-learn components
try:
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BIST Quant Portfolio Lab Pro MAX",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/streamlit/streamlit',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# BIST Portfolio Optimization Suite v8.0"
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL CSS THEME WITH GRADIENTS & ANIMATIONS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&family=Source+Sans+Pro:wght@400;600&display=swap');
    
    :root {
        --primary-dark: #0a1929;
        --secondary-dark: #1a2536;
        --accent-blue: #0066cc;
        --accent-green: #00cc88;
        --accent-red: #ff4d4d;
        --accent-purple: #9d4edd;
        --accent-orange: #ff6b35;
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
        --border-color: #2d3748;
        --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 95%;
    }
    
    /* Professional Metrics with Glow Effect */
    .metric-card {
        background: linear-gradient(135deg, var(--secondary-dark), var(--primary-dark));
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 102, 204, 0.3);
        border-color: var(--accent-blue);
    }
    
    /* Custom DataFrames with Scroll */
    .stDataFrame {
        border-radius: 10px;
        border: 1px solid var(--border-color) !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: var(--secondary-dark);
        padding: 0.75rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        background-color: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(0, 102, 204, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-1) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: var(--gradient-3);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
    }
    
    /* Primary Action Button */
    .primary-button > button {
        background: var(--gradient-1) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Headers with Gradient Text */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-green), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    /* Sidebar Enhancement */
    section[data-testid="stSidebar"] {
        background-color: var(--primary-dark);
        border-right: 1px solid var(--border-color);
        box-shadow: 5px 0 25px rgba(0, 0, 0, 0.2);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--primary-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-blue);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-green);
    }
    
    /* Notification Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .badge-success {
        background: linear-gradient(135deg, var(--accent-green), #00b374);
        color: white;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, var(--accent-orange), #ff8c42);
        color: white;
    }
    
    .badge-danger {
        background: linear-gradient(135deg, var(--accent-red), #ff3333);
        color: white;
    }
    
    /* Card Headers */
    .card-header {
        background: linear-gradient(90deg, rgba(0, 102, 204, 0.1), rgba(0, 204, 136, 0.1));
        padding: 1rem;
        border-radius: 8px 8px 0 0;
        border-bottom: 1px solid var(--border-color);
        margin: -1rem -1rem 1rem -1rem;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background: var(--gradient-2);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED DATA STRUCTURES WITH FALLBACK MECHANISMS
# ─────────────────────────────────────────────────────────────────────────────

BIST30_TICKERS = [
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS', 'EKGYO.IS',
    'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'GUBRF.IS', 'HALKB.IS', 'HEKTS.IS',
    'ISCTR.IS', 'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 'ODAS.IS',
    'PETKM.IS', 'PGSUS.IS', 'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TAVHL.IS',
    'TCELL.IS', 'THYAO.IS', 'TKFEN.IS', 'TOASO.IS', 'TSKB.IS', 'TTKOM.IS',
    'TUPRS.IS', 'VAKBN.IS', 'VESTL.IS', 'YKBNK.IS'
]

SECTOR_MAPPING = {
    'Banking': ['AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'HALKB.IS', 'YKBNK.IS', 'TSKB.IS', 'VAKBN.IS'],
    'Industry': ['ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS', 'EREGL.IS', 'GUBRF.IS'],
    'Automotive': ['FROTO.IS', 'TOASO.IS', 'KCHOL.IS'],
    'Technology': ['THYAO.IS', 'TCELL.IS', 'TTKOM.IS'],
    'Energy': ['PETKM.IS', 'TUPRS.IS'],
    'Holding': ['SAHOL.IS', 'KRDMD.IS'],
    'Construction': ['EKGYO.IS', 'ODAS.IS'],
    'Textile': ['SASA.IS'],
    'Glass': ['SISE.IS'],
    'Tourism': ['TAVHL.IS'],
    'Healthcare': ['HEKTS.IS'],
    'Food': ['PGSUS.IS']
}

BENCHMARKS = {
    'BIST 100': 'XU100.IS',
    'BIST 30': 'XU030.IS', 
    'USD/TRY': 'TRY=X',
    'EUR/TRY': 'EURTRY=X',
    'Gold': 'GC=F',
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'BTC-USD': 'BTC-USD'
}

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED DATA SOURCE WITH MULTIPLE FALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedDataSource:
    def __init__(self):
        self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @st.cache_data(ttl=3600, show_spinner="📊 Fetching market data...")
    def fetch_market_data(_self, tickers: List[str], start_date: datetime, 
                         end_date: datetime, interval: str = '1d') -> Optional[Dict]:
        """Robust data fetching with multiple fallback strategies"""
        
        if not HAS_YFINANCE:
            st.error("❌ yfinance is not installed. Please install with: pip install yfinance")
            return None
        
        try:
            # Ensure tickers is a list
            if isinstance(tickers, str):
                tickers = [tickers]
            
            # Remove any duplicates
            tickers = list(set(tickers))
            
            # Log the data fetch attempt
            logger.info(f"Fetching data for {len(tickers)} tickers: {tickers}")
            
            # Download with comprehensive settings
            data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date + timedelta(days=1),  # Add one day to include end_date
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                prepost=False,
                threads=True,
                proxy=None,
                progress=False,
                timeout=30
            )
            
            # Check if data is empty
            if data.empty:
                st.warning(f"⚠️ No data returned for tickers: {tickers}")
                logger.warning(f"No data for tickers: {tickers}")
                return None
            
            # Handle single ticker case
            if len(tickers) == 1:
                ticker = tickers[0]
                processed_data = {
                    'close': data['Close'].rename(ticker).to_frame(),
                    'open': data['Open'].rename(ticker).to_frame(),
                    'high': data['High'].rename(ticker).to_frame(),
                    'low': data['Low'].rename(ticker).to_frame(),
                    'volume': data['Volume'].rename(ticker).to_frame()
                }
            else:
                # Multi-ticker case
                processed_data = {
                    'close': pd.DataFrame(),
                    'open': pd.DataFrame(),
                    'high': pd.DataFrame(),
                    'low': pd.DataFrame(),
                    'volume': pd.DataFrame()
                }
                
                for ticker in tickers:
                    # Check if ticker exists in downloaded data
                    if ticker in data.columns.get_level_values(0):
                        processed_data['close'][ticker] = data[ticker]['Close']
                        processed_data['open'][ticker] = data[ticker]['Open']
                        processed_data['high'][ticker] = data[ticker]['High']
                        processed_data['low'][ticker] = data[ticker]['Low']
                        processed_data['volume'][ticker] = data[ticker]['Volume']
                    else:
                        st.warning(f"⚠️ No data for ticker: {ticker}")
                        logger.warning(f"No data for ticker: {ticker}")
            
            # Clean and forward fill data
            for key in processed_data:
                if not processed_data[key].empty:
                    # Forward fill then backward fill
                    processed_data[key] = processed_data[key].ffill().bfill()
                    # Drop columns with all NaN
                    processed_data[key] = processed_data[key].dropna(axis=1, how='all')
            
            # Calculate returns
            if not processed_data['close'].empty:
                processed_data['returns'] = processed_data['close'].pct_change().dropna()
            
            # Log success
            logger.info(f"Successfully fetched data. Shape: {processed_data['close'].shape}")
            
            return processed_data
            
        except Exception as e:
            st.error(f"❌ Data fetch error: {str(e)}")
            logger.error(f"Data fetch error: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def fetch_fundamental_data(self, ticker: str) -> Optional[Dict]:
        """Fetch fundamental data with error handling"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamental_data = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'volume': info.get('volume'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country')
            }
            
            # Clean None values
            fundamental_data = {k: v for k, v in fundamental_data.items() if v is not None}
            
            return fundamental_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch fundamental data for {ticker}: {str(e)}")
            return None

# ─────────────────────────────────────────────────────────────────────────────
# QUANTITATIVE PORTFOLIO OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class QuantitativePortfolioOptimizer:
    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame):
        self.prices = prices
        self.returns = returns
        self.tickers = prices.columns.tolist()
        
        # Validate data
        self._validate_data()
        
        # Initialize optimization models
        self._initialize_models()
    
    def _validate_data(self):
        """Validate input data"""
        if self.prices.empty or self.returns.empty:
            raise ValueError("Prices or returns data is empty")
        
        if len(self.tickers) < 2:
            raise ValueError("At least 2 assets required for optimization")
        
        # Check for NaN values
        if self.prices.isnull().any().any():
            self.prices = self.prices.ffill().bfill()
        
        if self.returns.isnull().any().any():
            self.returns = self.returns.ffill().bfill()
    
    def _initialize_models(self):
        """Initialize expected returns and risk models"""
        if not HAS_PYPFOPT:
            raise ImportError("PyPortfolioOpt is required for optimization")
        
        # Expected return models
        self.mu_models = {
            'mean_historical': expected_returns.mean_historical_return(self.prices),
            'ema_historical': expected_returns.ema_historical_return(self.prices, span=500),
            'capm_return': expected_returns.capm_return(self.prices),
        }
        
        # Risk models
        self.risk_models = {
            'sample_cov': risk_models.sample_cov(self.returns),
            'semicovariance': risk_models.semicovariance(self.returns),
            'exp_cov': risk_models.exp_cov(self.returns),
            'ledoit_wolf': risk_models.CovarianceShrinkage(self.prices).ledoit_wolf(),
            'oracle_approximating': risk_models.CovarianceShrinkage(self.prices).oracle_approximating(),
        }
    
    def optimize(self, method: str = 'max_sharpe', risk_model: str = 'ledoit_wolf',
                return_model: str = 'mean_historical', **kwargs) -> Tuple[Dict, Tuple]:
        """Main optimization method with comprehensive error handling"""
        
        try:
            # Get selected models
            mu = self.mu_models.get(return_model, self.mu_models['mean_historical'])
            S = self.risk_models.get(risk_model, self.risk_models['ledoit_wolf'])
            
            # Risk-free rate
            risk_free_rate = kwargs.get('risk_free_rate', 0.0)
            
            # Perform optimization based on method
            if method == 'max_sharpe':
                ef = EfficientFrontier(mu, S)
                ef.max_sharpe(risk_free_rate=risk_free_rate)
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                
            elif method == 'min_volatility':
                ef = EfficientFrontier(mu, S)
                ef.min_volatility()
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                
            elif method == 'max_quadratic_utility':
                ef = EfficientFrontier(mu, S)
                ef.max_quadratic_utility()
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                
            elif method == 'efficient_risk':
                ef = EfficientFrontier(mu, S)
                target_vol = kwargs.get('target_volatility', 0.15)
                ef.efficient_risk(target_volatility=target_vol)
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                
            elif method == 'efficient_return':
                ef = EfficientFrontier(mu, S)
                target_return = kwargs.get('target_return', 0.20)
                ef.efficient_return(target_return=target_return)
                weights = ef.clean_weights()
                performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
                
            elif method == 'hrp' and HAS_PYPFOPT:
                hrp = HRPOpt(self.returns)
                hrp.optimize()
                weights = hrp.clean_weights()
                
                # Calculate performance metrics for HRP
                port_returns = (self.returns * pd.Series(weights)).sum(axis=1)
                annual_return = port_returns.mean() * 252
                annual_vol = port_returns.std() * np.sqrt(252)
                sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
                performance = (annual_return, annual_vol, sharpe)
                
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            return weights, performance
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            raise
    
    def generate_efficient_frontier(self, points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate efficient frontier"""
        mu = self.mu_models['mean_historical']
        S = self.risk_models['ledoit_wolf']
        
        ef = EfficientFrontier(mu, S)
        mus, sigmas, weights = ef.efficient_frontier(points=points)
        
        return mus, sigmas, weights
    
    def calculate_discrete_allocation(self, weights: Dict, 
                                     total_portfolio_value: float = 1000000) -> Tuple[Dict, float]:
        """Calculate discrete share allocation"""
        latest_prices = get_latest_prices(self.prices)
        da = DiscreteAllocation(
            weights, 
            latest_prices, 
            total_portfolio_value=total_portfolio_value
        )
        
        # Try both allocation methods
        try:
            allocation, leftover = da.lp_portfolio()
        except:
            allocation, leftover = da.greedy_portfolio()
        
        return allocation, leftover

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED ANALYTICS ENGINE WITH QUANTSTATS
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedAnalyticsEngine:
    def __init__(self, portfolio_returns: pd.Series, 
                 benchmark_returns: pd.Series = None,
                 risk_free_rate: float = 0.0):
        
        # Ensure returns are pandas Series
        if isinstance(portfolio_returns, pd.DataFrame):
            self.portfolio_returns = portfolio_returns.iloc[:, 0]
        else:
            self.portfolio_returns = portfolio_returns
        
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        # Initialize QuantStats if available
        if HAS_QUANTSTATS:
            qs.extend_pandas()
    
    def generate_comprehensive_report(self) -> Optional[str]:
        """Generate full QuantStats HTML report"""
        if not HAS_QUANTSTATS:
            return None
        
        try:
            # Create BytesIO buffer for HTML output
            buffer = io.StringIO()
            
            if self.benchmark_returns is not None:
                qs.reports.html(
                    self.portfolio_returns,
                    self.benchmark_returns,
                    rf=self.risk_free_rate,
                    title='BIST Portfolio Analytics Report',
                    output=buffer,
                    download_filename='portfolio_report.html'
                )
            else:
                qs.reports.html(
                    self.portfolio_returns,
                    rf=self.risk_free_rate,
                    title='Portfolio Analytics Report',
                    output=buffer,
                    download_filename='portfolio_report.html'
                )
            
            html_content = buffer.getvalue()
            buffer.close()
            
            return html_content
            
        except Exception as e:
            logger.error(f"QuantStats report generation error: {str(e)}")
            return None
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['Total Return'] = (1 + self.portfolio_returns).prod() - 1
        metrics['CAGR'] = qs.stats.cagr(self.portfolio_returns) if HAS_QUANTSTATS else 0
        metrics['Annual Volatility'] = self.portfolio_returns.std() * np.sqrt(252)
        metrics['Max Drawdown'] = qs.stats.max_drawdown(self.portfolio_returns) if HAS_QUANTSTATS else 0
        
        # Risk-adjusted metrics
        if HAS_QUANTSTATS:
            metrics['Sharpe Ratio'] = qs.stats.sharpe(self.portfolio_returns, rf=self.risk_free_rate)
            metrics['Sortino Ratio'] = qs.stats.sortino(self.portfolio_returns, rf=self.risk_free_rate)
            metrics['Calmar Ratio'] = qs.stats.calmar(self.portfolio_returns)
            metrics['Omega Ratio'] = qs.stats.omega(self.portfolio_returns, rf=self.risk_free_rate)
        
        # Benchmark comparison metrics
        if self.benchmark_returns is not None:
            excess_returns = self.portfolio_returns - self.benchmark_returns
            metrics['Alpha'] = excess_returns.mean() * 252
            metrics['Beta'] = np.cov(self.portfolio_returns, self.benchmark_returns)[0, 1] / np.var(self.benchmark_returns)
            metrics['Tracking Error'] = excess_returns.std() * np.sqrt(252)
            metrics['Information Ratio'] = metrics['Alpha'] / metrics['Tracking Error'] if metrics['Tracking Error'] > 0 else 0
            
            if HAS_QUANTSTATS:
                metrics['Up Capture'] = qs.stats.up_capture(self.portfolio_returns, self.benchmark_returns)
                metrics['Down Capture'] = qs.stats.down_capture(self.portfolio_returns, self.benchmark_returns)
        
        # Risk metrics
        var_95 = np.percentile(self.portfolio_returns, 5)
        cvar_95 = self.portfolio_returns[self.portfolio_returns <= var_95].mean()
        
        metrics['VaR (95%)'] = var_95
        metrics['CVaR (95%)'] = cvar_95
        metrics['Skewness'] = stats.skew(self.portfolio_returns)
        metrics['Kurtosis'] = stats.kurtosis(self.portfolio_returns)
        
        return metrics
    
    def create_interactive_tearsheet(self) -> go.Figure:
        """Create comprehensive interactive tearsheet"""
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Cumulative Returns', 'Daily Returns',
                'Rolling Sharpe Ratio (6M)', 'Drawdown Periods',
                'Monthly Returns Heatmap', 'Return Distribution',
                'Rolling Volatility (6M)', 'Underwater Plot'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Cumulative Returns
        cum_returns = (1 + self.portfolio_returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns.values,
                mode='lines',
                name='Portfolio',
                line=dict(color='#00cc88', width=3),
                hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        if self.benchmark_returns is not None:
            bench_cum = (1 + self.benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=bench_cum.index,
                    y=bench_cum.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#0066cc', width=2, dash='dash'),
                    hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Daily Returns
        fig.add_trace(
            go.Scatter(
                x=self.portfolio_returns.index,
                y=self.portfolio_returns.values,
                mode='markers',
                marker=dict(
                    size=4,
                    color=self.portfolio_returns.values,
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(title="Return"),
                    cmin=self.portfolio_returns.quantile(0.01),
                    cmax=self.portfolio_returns.quantile(0.99)
                ),
                name='Daily Returns',
                hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Rolling Sharpe Ratio
        rolling_window = 126  # 6 months
        rolling_sharpe = self.portfolio_returns.rolling(rolling_window).apply(
            lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color='#9d4edd', width=2),
                fill='tozeroy',
                fillcolor='rgba(157, 78, 221, 0.2)',
                hovertemplate='%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Drawdown
        if HAS_QUANTSTATS:
            drawdown = qs.stats.to_drawdown_series(self.portfolio_returns)
        else:
            # Manual drawdown calculation
            cum_max = (1 + self.portfolio_returns).cumprod().cummax()
            drawdown = ((1 + self.portfolio_returns).cumprod() / cum_max) - 1
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                line=dict(color='#ff4d4d', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 77, 77, 0.3)',
                hovertemplate='%{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 5. Monthly Returns Heatmap
        monthly_returns = self.portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.strftime('%b'),
            'Return': monthly_returns.values
        })
        
        # Create pivot for heatmap
        monthly_pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pivot = monthly_pivot.reindex(columns=month_order)
        
        fig.add_trace(
            go.Heatmap(
                z=monthly_pivot.values,
                x=monthly_pivot.columns,
                y=monthly_pivot.index,
                colorscale='RdBu_r',
                zmid=0,
                colorbar=dict(title="Return"),
                hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Return Distribution
        fig.add_trace(
            go.Histogram(
                x=self.portfolio_returns.values,
                nbinsx=50,
                name='Return Distribution',
                marker_color='#0066cc',
                opacity=0.7,
                hovertemplate='Return: %{x:.2%}<br>Count: %{y}<extra></extra>'
            ),
            row=3, col=2
        )
        
        # Add normal distribution overlay
        x_range = np.linspace(self.portfolio_returns.min(), self.portfolio_returns.max(), 100)
        pdf = stats.norm.pdf(x_range, self.portfolio_returns.mean(), self.portfolio_returns.std())
        pdf = pdf * len(self.portfolio_returns) * (x_range[1] - x_range[0])
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=pdf,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='#ff6b35', width=2, dash='dash'),
                hovertemplate='Return: %{x:.2%}<br>Density: %{y:.2f}<extra></extra>'
            ),
            row=3, col=2
        )
        
        # 7. Rolling Volatility
        rolling_vol = self.portfolio_returns.rolling(rolling_window).std() * np.sqrt(252)
        
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color='#ff6b35', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 107, 53, 0.2)',
                hovertemplate='%{x|%Y-%m-%d}<br>Volatility: %{y:.2%}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=1400,
            showlegend=True,
            template='plotly_dark',
            title_text="Portfolio Performance Tearsheet",
            title_font_size=24,
            hovermode='x unified'
        )
        
        # Update axes
        for i in range(1, 5):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Date", row=i, col=j)
                if i == 3 and j == 2:
                    fig.update_yaxes(title_text="Density", row=i, col=j)
                elif i == 2 and j == 2:
                    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=i, col=j)
                else:
                    fig.update_yaxes(title_text="Return", tickformat=".0%", row=i, col=j)
        
        return fig

# ─────────────────────────────────────────────────────────────────────────────
# RISK ANALYTICS MODULE
# ─────────────────────────────────────────────────────────────────────────────

class RiskAnalyticsModule:
    def __init__(self, returns: pd.Series, benchmark_returns: pd.Series = None):
        self.returns = returns
        self.benchmark_returns = benchmark_returns
    
    def calculate_var_metrics(self, confidence_levels: List[float] = [0.90, 0.95, 0.99]) -> Dict:
        """Calculate Value at Risk metrics"""
        results = {}
        
        for cl in confidence_levels:
            # Historical VaR
            var_hist = np.percentile(self.returns, (1 - cl) * 100)
            
            # Conditional VaR (Expected Shortfall)
            cvar = self.returns[self.returns <= var_hist].mean()
            
            # Parametric VaR (Normal)
            var_param = self.returns.mean() + stats.norm.ppf(1 - cl) * self.returns.std()
            
            # Cornish-Fisher VaR (accounts for skewness and kurtosis)
            z = stats.norm.ppf(1 - cl)
            s = stats.skew(self.returns)
            k = stats.kurtosis(self.returns)
            z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
            var_cf = self.returns.mean() + z_cf * self.returns.std()
            
            results[f'VaR_{int(cl*100)}'] = {
                'Historical': var_hist,
                'Parametric_Normal': var_param,
                'Cornish_Fisher': var_cf,
                'CVaR': cvar
            }
        
        return results
    
    def calculate_risk_decomposition(self, weights: np.ndarray, covariance: np.ndarray) -> Dict:
        """Decompose portfolio risk into contributions"""
        portfolio_variance = weights.T @ covariance @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Marginal contribution to risk
        marginal_contrib = (covariance @ weights) / portfolio_volatility
        
        # Percent contribution
        percent_contrib = (weights * marginal_contrib) / portfolio_volatility
        
        # Component VaR
        component_var = percent_contrib * portfolio_volatility * stats.norm.ppf(0.95)
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'marginal_contributions': marginal_contrib,
            'percent_contributions': percent_contrib,
            'component_var': component_var
        }
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate comprehensive risk metrics"""
        metrics = {}
        
        # Basic risk metrics
        metrics['Annual Volatility'] = self.returns.std() * np.sqrt(252)
        metrics['Downside Deviation'] = self.returns[self.returns < 0].std() * np.sqrt(252) if len(self.returns[self.returns < 0]) > 0 else 0
        metrics['VaR (95%)'] = np.percentile(self.returns, 5)
        metrics['CVaR (95%)'] = self.returns[self.returns <= metrics['VaR (95%)']].mean()
        
        # Higher moments
        metrics['Skewness'] = stats.skew(self.returns)
        metrics['Excess Kurtosis'] = stats.kurtosis(self.returns)
        
        # Extreme risk metrics
        metrics['Worst Daily Return'] = self.returns.min()
        metrics['Best Daily Return'] = self.returns.max()
        metrics['Avg Loss'] = self.returns[self.returns < 0].mean()
        metrics['Avg Gain'] = self.returns[self.returns > 0].mean()
        
        # Stability metrics
        metrics['Stability'] = len(self.returns[self.returns > 0]) / len(self.returns)
        
        return metrics

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.optimized = False
        st.session_state.portfolio_data = None
        st.session_state.benchmark_data = None
    
    # Custom sidebar header
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">⚡ BIST Quant Pro</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;">v8.0 - Institutional Edition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")
        
        # Date Selection with default 3-year history
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=365 * 3),
                help="Select start date for analysis"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.now(),
                help="Select end date for analysis"
            )
        
        # Asset Selection with search and filtering
        st.markdown("### 📊 Asset Selection")
        
        selected_sector = st.selectbox(
            "Filter by Sector",
            ["All Sectors"] + list(SECTOR_MAPPING.keys()),
            help="Filter assets by sector"
        )
        
        if selected_sector == "All Sectors":
            available_tickers = BIST30_TICKERS
        else:
            available_tickers = SECTOR_MAPPING[selected_sector]
        
        # Search box for tickers
        search_ticker = st.text_input(
            "🔍 Search Ticker",
            placeholder="Type to search...",
            help="Search for specific tickers"
        )
        
        if search_ticker:
            filtered_tickers = [t for t in available_tickers if search_ticker.upper() in t]
        else:
            filtered_tickers = available_tickers
        
        # Multi-select with select all option
        col_select, col_all = st.columns([3, 1])
        with col_all:
            if st.button("Select All", use_container_width=True):
                st.session_state.selected_assets = filtered_tickers
        
        assets = st.multiselect(
            "Select Assets for Portfolio",
            filtered_tickers,
            default=['THYAO.IS', 'GARAN.IS', 'ASELS.IS'],
            key="selected_assets",
            help="Select at least 2 assets for portfolio optimization"
        )
        
        # Benchmark Selection
        st.markdown("### 📈 Benchmark Selection")
        benchmark_symbol = st.selectbox(
            "Benchmark Index",
            list(BENCHMARKS.keys()),
            index=0,
            help="Select benchmark for comparison"
        )
        
        # Optimization Parameters
        st.markdown("### ⚡ Optimization Parameters")
        
        optimization_method = st.selectbox(
            "Optimization Method",
            ['max_sharpe', 'min_volatility', 'efficient_risk', 
             'efficient_return', 'hrp'],
            help="Select portfolio optimization method"
        )
        
        col_risk, col_return = st.columns(2)
        with col_risk:
            risk_model = st.selectbox(
                "Risk Model",
                ['ledoit_wolf', 'sample_cov', 'semicovariance'],
                help="Select covariance estimation method"
            )
        
        with col_return:
            return_model = st.selectbox(
                "Return Model",
                ['mean_historical', 'ema_historical', 'capm_return'],
                help="Select expected return estimation method"
            )
        
        # Advanced Parameters
        with st.expander("🔧 Advanced Parameters", expanded=False):
            risk_free_rate = st.slider(
                "Risk-Free Rate (%)",
                0.0, 50.0, 30.0, 0.1,
                help="Annual risk-free rate in percentage"
            ) / 100
            
            if optimization_method == 'efficient_risk':
                target_volatility = st.slider(
                    "Target Volatility",
                    0.05, 0.50, 0.15, 0.01,
                    help="Target annual volatility for efficient risk optimization"
                )
            else:
                target_volatility = 0.15
            
            if optimization_method == 'efficient_return':
                target_return = st.slider(
                    "Target Return",
                    0.05, 1.0, 0.20, 0.01,
                    help="Target annual return for efficient return optimization"
                )
            else:
                target_return = 0.20
        
        # Reporting Options
        st.markdown("### 📊 Reporting Options")
        generate_full_report = st.checkbox(
            "Generate Full QuantStats Report",
            True,
            help="Generate comprehensive HTML report"
        )
        
        show_tearsheet = st.checkbox(
            "Show Interactive Tearsheet",
            True,
            help="Display interactive performance visualization"
        )
        
        calculate_discrete = st.checkbox(
            "Calculate Discrete Allocation",
            False,
            help="Calculate actual share allocations"
        )
        
        if calculate_discrete:
            portfolio_value = st.number_input(
                "Portfolio Value (TRY)",
                10000, 10000000, 1000000, 10000,
                help="Total portfolio value for discrete allocation"
            )
        
        # Data Refresh Button
        st.markdown("---")
        if st.button("🔄 Refresh All Data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.session_state.data_loaded = False
            st.rerun()
    
    # Main Dashboard Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(10,25,41,0.9) 0%, rgba(26,37,54,0.9) 100%); border-radius: 15px; margin-bottom: 2rem; border: 1px solid #2d3748;">
        <h1 style="margin: 0; font-size: 3rem;">📊 BIST Enterprise Portfolio Analytics Suite</h1>
        <p style="font-size: 1.2rem; color: #b0b0b0; margin-top: 0.5rem;">Professional Portfolio Optimization & Risk Analytics Platform</p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
            <span class="badge badge-success">Real-time Data</span>
            <span class="badge badge-warning">Advanced Analytics</span>
            <span class="badge badge-danger">Risk Management</span>
            <span class="badge" style="background: linear-gradient(135deg, #9d4edd, #5a189a); color: white;">AI-Powered</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check requirements
    if not HAS_YFINANCE:
        st.error("""
        ❌ **yfinance is not installed!**
        
        Please install required packages:
        ```bash
        pip install yfinance pypfopt quantstats streamlit plotly
        ```
        
        If you're using Streamlit Cloud, add these to your `requirements.txt` file.
        """)
        return
    
    if not HAS_PYPFOPT:
        st.warning("⚠️ **PyPortfolioOpt is not fully installed.** Some optimization features may be limited.")
    
    # Validate asset selection
    if len(assets) < 2:
        st.warning("""
        ⚠️ **Please select at least 2 assets for portfolio optimization.**
        
        - Use the asset selection panel in the sidebar
        - You can filter by sector or search for specific tickers
        - Select at least 2 different assets
        """)
        return
    
    # Data Loading Section
    with st.spinner("🔄 Loading market data and performing analysis..."):
        # Initialize data source
        data_source = AdvancedDataSource()
        
        # Fetch portfolio data
        portfolio_data = data_source.fetch_market_data(
            assets,
            start_date,
            end_date
        )
        
        # Fetch benchmark data
        benchmark_data = data_source.fetch_market_data(
            [BENCHMARKS[benchmark_symbol]],
            start_date,
            end_date
        )
        
        # Check if data was fetched successfully
        if portfolio_data is None:
            st.error("""
            ❌ **Failed to load portfolio data!**
            
            Possible reasons:
            1. No internet connection
            2. Yahoo Finance API is down
            3. Selected tickers are invalid
            4. Date range is too large
            
            **Try:**
            - Check your internet connection
            - Reduce the date range
            - Try different tickers
            - Click "Refresh All Data" button
            """)
            return
        
        if benchmark_data is None:
            st.warning(f"⚠️ Could not load benchmark data for {benchmark_symbol}")
            benchmark_returns = None
        else:
            benchmark_returns = benchmark_data['returns'].iloc[:, 0]
        
        # Store data in session state
        st.session_state.data_loaded = True
        st.session_state.portfolio_data = portfolio_data
        st.session_state.benchmark_data = benchmark_data
        
        # Initialize optimizer
        try:
            optimizer = QuantitativePortfolioOptimizer(
                portfolio_data['close'],
                portfolio_data['returns']
            )
            
            # Optimize portfolio
            weights, performance = optimizer.optimize(
                method=optimization_method,
                risk_model=risk_model,
                return_model=return_model,
                target_volatility=target_volatility,
                target_return=target_return,
                risk_free_rate=risk_free_rate
            )
            
            # Calculate portfolio returns
            portfolio_returns = (portfolio_data['returns'] * pd.Series(weights)).sum(axis=1)
            st.session_state.optimized = True
            
        except Exception as e:
            st.error(f"❌ Optimization failed: {str(e)}")
            logger.error(f"Optimization error: {str(e)}")
            return
    
    # Performance Dashboard
    st.markdown("### 📈 Performance Dashboard")
    
    # Top Metrics in a grid layout
    metric_cols = st.columns(5)
    
    with metric_cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Expected Return",
            f"{performance[0]:.2%}",
            delta=None,
            delta_color="normal",
            help="Annualized expected return"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Expected Volatility",
            f"{performance[1]:.2%}",
            delta=None,
            help="Annualized portfolio volatility"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Sharpe Ratio",
            f"{performance[2]:.2f}",
            delta=None,
            help="Risk-adjusted return (Sharpe Ratio)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[3]:
        var_95 = np.percentile(portfolio_returns, 5)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Daily VaR (95%)",
            f"{var_95:.2%}",
            delta=None,
            help="Value at Risk at 95% confidence"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[4]:
        max_dd = qs.stats.max_drawdown(portfolio_returns) if HAS_QUANTSTATS else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Max Drawdown",
            f"{max_dd:.2%}",
            delta=None,
            help="Maximum historical drawdown"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Portfolio Overview",
        "📊 Optimization Analysis",
        "⚠️ Risk Analytics",
        "📈 Performance Analytics",
        "📑 Reports & Export"
    ])
    
    with tab1:
        # Portfolio Overview
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown("### Optimal Allocation")
            
            # Convert weights to DataFrame
            weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
            weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
            
            # Create pie chart
            fig_pie = px.pie(
                weights_df,
                values='Weight',
                names=weights_df.index,
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis,
                labels={'Weight': 'Allocation %'},
                title="Portfolio Allocation"
            )
            fig_pie.update_layout(
                template="plotly_dark",
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="right",
                    x=1.3
                )
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Weights table
            st.markdown("#### Allocation Details")
            st.dataframe(
                weights_df.style.format('{:.2%}')
                .background_gradient(cmap='Blues', subset=['Weight'])
                .set_properties(**{'text-align': 'center'}),
                use_container_width=True,
                height=300
            )
            
            # Discrete allocation if requested
            if calculate_discrete:
                st.markdown("#### 📦 Discrete Allocation")
                allocation, leftover = optimizer.calculate_discrete_allocation(weights, portfolio_value)
                
                if allocation:
                    alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
                    alloc_df['Value (TRY)'] = alloc_df['Shares'] * get_latest_prices(portfolio_data['close'])[alloc_df.index]
                    
                    st.dataframe(
                        alloc_df.style.format({
                            'Shares': '{:,.0f}',
                            'Value (TRY)': '₺{:,.2f}'
                        }),
                        use_container_width=True
                    )
                    st.info(f"💰 **Remaining cash:** ₺{leftover:,.2f}")
        
        with col_right:
            st.markdown("### Cumulative Performance")
            
            # Calculate cumulative returns
            cum_port = (1 + portfolio_returns).cumprod()
            
            fig_cum = go.Figure()
            
            fig_cum.add_trace(go.Scatter(
                x=cum_port.index,
                y=cum_port.values,
                name='Optimized Portfolio',
                line=dict(color='#00cc88', width=3),
                hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>'
            ))
            
            if benchmark_returns is not None:
                cum_bench = (1 + benchmark_returns).cumprod()
                fig_cum.add_trace(go.Scatter(
                    x=cum_bench.index,
                    y=cum_bench.values,
                    name=benchmark_symbol,
                    line=dict(color='#0066cc', width=2, dash='dash'),
                    hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2%}<extra></extra>'
                ))
            
            fig_cum.update_layout(
                template="plotly_dark",
                height=500,
                hovermode='x unified',
                yaxis_title="Cumulative Return",
                xaxis_title="Date",
                yaxis_tickformat=".0%",
                title="Portfolio vs Benchmark Performance",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig_cum, use_container_width=True)
            
            # Rolling metrics
            st.markdown("### 📊 Rolling Metrics (6-Month Window)")
            
            rolling_window = 126  # 6 months
            
            # Calculate rolling metrics
            rolling_sharpe = portfolio_returns.rolling(rolling_window).apply(
                lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
            )
            
            rolling_vol = portfolio_returns.rolling(rolling_window).std() * np.sqrt(252)
            
            fig_rolling = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility'),
                vertical_spacing=0.15
            )
            
            fig_rolling.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    name='Sharpe Ratio',
                    line=dict(color='#9d4edd', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(157, 78, 221, 0.2)',
                    hovertemplate='%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig_rolling.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol.values,
                    name='Volatility',
                    line=dict(color='#ff6b35', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 53, 0.2)',
                    hovertemplate='%{x|%Y-%m-%d}<br>Volatility: %{y:.2%}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig_rolling.update_layout(
                height=400,
                template="plotly_dark",
                showlegend=False,
                hovermode='x unified'
            )
            
            fig_rolling.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
            fig_rolling.update_yaxes(title_text="Volatility", tickformat=".0%", row=2, col=1)
            
            st.plotly_chart(fig_rolling, use_container_width=True)
    
    with tab2:
        # Optimization Analysis
        st.markdown("## Efficient Frontier Analysis")
        
        try:
            # Generate efficient frontier
            mus, sigmas, frontier_weights = optimizer.generate_efficient_frontier()
            
            fig_frontier = go.Figure()
            
            # Plot efficient frontier
            fig_frontier.add_trace(go.Scatter(
                x=sigmas,
                y=mus,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='white', width=3),
                fill='tonexty',
                fillcolor='rgba(255, 255, 255, 0.1)',
                hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            ))
            
            # Plot optimal portfolio
            fig_frontier.add_trace(go.Scatter(
                x=[performance[1]],
                y=[performance[0]],
                mode='markers',
                marker=dict(
                    color='red',
                    size=20,
                    symbol='star',
                    line=dict(color='white', width=2)
                ),
                name='Optimal Portfolio',
                hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{text:.2f}<extra></extra>',
                text=[performance[2]]
            ))
            
            # Plot individual assets
            individual_returns = []
            individual_risks = []
            
            for asset in portfolio_data['close'].columns:
                asset_returns = portfolio_data['returns'][asset]
                asset_return = asset_returns.mean() * 252
                asset_risk = asset_returns.std() * np.sqrt(252)
                
                individual_returns.append(asset_return)
                individual_risks.append(asset_risk)
                
                fig_frontier.add_trace(go.Scatter(
                    x=[asset_risk],
                    y=[asset_return],
                    mode='markers+text',
                    marker=dict(size=12, color='lightblue'),
                    text=[asset],
                    textposition="top center",
                    name=asset,
                    showlegend=False,
                    hovertemplate='%{text}<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                ))
            
            fig_frontier.update_layout(
                template="plotly_dark",
                height=600,
                xaxis_title="Annualized Volatility (Risk)",
                yaxis_title="Annualized Return",
                title="Efficient Frontier with Individual Assets",
                hovermode='closest',
                xaxis_tickformat=".0%",
                yaxis_tickformat=".0%"
            )
            
            st.plotly_chart(fig_frontier, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating efficient frontier: {str(e)}")
    
    with tab3:
        # Risk Analytics
        st.markdown("## ⚠️ Comprehensive Risk Analysis")
        
        # Initialize risk analytics
        risk_analytics = RiskAnalyticsModule(portfolio_returns, benchmark_returns)
        
        col_risk1, col_risk2 = st.columns(2)
        
        with col_risk1:
            st.markdown("### Value at Risk Analysis")
            
            # Calculate VaR metrics
            var_results = risk_analytics.calculate_var_metrics([0.90, 0.95, 0.99])
            
            # Create VaR table
            var_data = []
            for cl, metrics in var_results.items():
                var_data.append({
                    'Confidence Level': cl.replace('VaR_', '') + '%',
                    'Historical VaR': metrics['Historical'],
                    'Parametric VaR': metrics['Parametric_Normal'],
                    'CVaR': metrics['CVaR']
                })
            
            var_df = pd.DataFrame(var_data)
            
            st.dataframe(
                var_df.style.format({
                    'Historical VaR': '{:.4f}',
                    'Parametric VaR': '{:.4f}',
                    'CVaR': '{:.4f}'
                }).background_gradient(cmap='Reds_r', subset=['Historical VaR', 'CVaR']),
                use_container_width=True,
                height=200
            )
            
            # Risk decomposition
            st.markdown("### Risk Decomposition")
            
            try:
                # Get covariance matrix
                if risk_model == 'ledoit_wolf':
                    S = risk_models.CovarianceShrinkage(portfolio_data['close']).ledoit_wolf()
                else:
                    S = risk_models.sample_cov(portfolio_data['returns'])
                
                # Convert weights to array
                w_array = np.array([weights.get(asset, 0) for asset in portfolio_data['close'].columns])
                
                # Calculate risk decomposition
                risk_decomp = risk_analytics.calculate_risk_decomposition(w_array, S.values)
                
                # Create decomposition dataframe
                decomp_df = pd.DataFrame({
                    'Asset': portfolio_data['close'].columns,
                    'Weight': w_array,
                    'Marginal Contribution': risk_decomp['marginal_contributions'],
                    'Percent Contribution': risk_decomp['percent_contributions']
                }).sort_values('Percent Contribution', ascending=False)
                
                st.dataframe(
                    decomp_df.style.format({
                        'Weight': '{:.2%}',
                        'Marginal Contribution': '{:.6f}',
                        'Percent Contribution': '{:.2%}'
                    }).background_gradient(cmap='YlOrRd', subset=['Percent Contribution']),
                    use_container_width=True,
                    height=300
                )
                
            except Exception as e:
                st.warning(f"Could not calculate risk decomposition: {str(e)}")
        
        with col_risk2:
            st.markdown("### Drawdown Analysis")
            
            # Calculate drawdown
            if HAS_QUANTSTATS:
                drawdown_series = qs.stats.to_drawdown_series(portfolio_returns)
            else:
                # Manual drawdown calculation
                cum_returns = (1 + portfolio_returns).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown_series = (cum_returns / running_max) - 1
            
            fig_dd = go.Figure()
            
            fig_dd.add_trace(go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values,
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red', width=2),
                name='Drawdown',
                hovertemplate='%{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>'
            ))
            
            # Find and mark max drawdown
            max_dd_idx = drawdown_series.idxmin()
            max_dd_val = drawdown_series.min()
            
            fig_dd.add_vline(
                x=max_dd_idx,
                line_dash="dash",
                line_color="yellow",
                annotation_text=f"Max DD: {max_dd_val:.2%}",
                annotation_position="top left"
            )
            
            fig_dd.update_layout(
                template="plotly_dark",
                height=400,
                title="Portfolio Drawdown",
                yaxis_title="Drawdown",
                yaxis_tickformat=".0%",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
            
            # Drawdown statistics
            dd_stats = {
                'Max Drawdown': max_dd_val,
                'Average Drawdown': drawdown_series.mean(),
                'Recovery Time (days)': 0,  # Placeholder
                'Drawdown Frequency': len(drawdown_series[drawdown_series < -0.05]) / len(drawdown_series)
            }
            
            dd_stats_df = pd.DataFrame.from_dict(dd_stats, orient='index', columns=['Value'])
            st.dataframe(
                dd_stats_df.style.format('{:.4f}'),
                use_container_width=True
            )
    
    with tab4:
        # Performance Analytics
        st.markdown("## 📈 Advanced Performance Analytics")
        
        # Initialize analytics engine
        analytics_engine = AdvancedAnalyticsEngine(
            portfolio_returns,
            benchmark_returns,
            risk_free_rate
        )
        
        # Calculate advanced metrics
        advanced_metrics = analytics_engine.calculate_performance_metrics()
        
        # Display metrics in a grid
        st.markdown("### Performance Metrics")
        
        # Create 4 columns for metrics
        metric_cols = st.columns(4)
        metric_items = list(advanced_metrics.items())
        
        # Group metrics by category
        return_metrics = {k: v for k, v in advanced_metrics.items() if 'Return' in k or 'CAGR' in k}
        risk_metrics = {k: v for k, v in advanced_metrics.items() if 'Vol' in k or 'VaR' in k or 'Drawdown' in k}
        ratio_metrics = {k: v for k, v in advanced_metrics.items() if 'Ratio' in k or 'Alpha' in k or 'Beta' in k}
        stat_metrics = {k: v for k, v in advanced_metrics.items() if 'Skew' in k or 'Kurt' in k or 'Stability' in k}
        
        # Display metrics in expanders
        with st.expander("📊 Return Metrics", expanded=True):
            ret_cols = st.columns(3)
            for idx, (key, value) in enumerate(return_metrics.items()):
                with ret_cols[idx % 3]:
                    if isinstance(value, float):
                        st.metric(
                            key,
                            f"{value:.2%}" if 'Return' in key or 'Drawdown' in key else f"{value:.2f}"
                        )
        
        with st.expander("⚠️ Risk Metrics"):
            risk_cols = st.columns(3)
            for idx, (key, value) in enumerate(risk_metrics.items()):
                with risk_cols[idx % 3]:
                    if isinstance(value, float):
                        st.metric(
                            key,
                            f"{value:.2%}" if 'Vol' in key or 'VaR' in key or 'Drawdown' in key else f"{value:.2f}"
                        )
        
        with st.expander("📐 Ratio & Statistical Metrics"):
            ratio_cols = st.columns(3)
            all_ratio_metrics = {**ratio_metrics, **stat_metrics}
            for idx, (key, value) in enumerate(all_ratio_metrics.items()):
                with ratio_cols[idx % 3]:
                    if isinstance(value, float):
                        st.metric(key, f"{value:.2f}")
        
        # Generate tearsheet if requested
        if show_tearsheet:
            st.markdown("### Interactive Tearsheet")
            with st.spinner("Generating tearsheet..."):
                tearsheet_fig = analytics_engine.create_interactive_tearsheet()
                st.plotly_chart(tearsheet_fig, use_container_width=True)
        
        # Monthly Returns Heatmap
        st.markdown("### Monthly Returns Heatmap")
        
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.strftime('%b'),
            'Return': monthly_returns.values
        })
        
        # Create pivot table for heatmap
        monthly_pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pivot = monthly_pivot.reindex(columns=month_order)
        
        # Create heatmap
        fig_heatmap = px.imshow(
            monthly_pivot,
            text_auto='.2%',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Monthly Returns Heatmap',
            labels=dict(x="Month", y="Year", color="Return")
        )
        
        fig_heatmap.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Month",
            yaxis_title="Year"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab5:
        # Reports & Export
        st.markdown("## 📑 Professional Reporting")
        
        # QuantStats Full Report
        if generate_full_report and HAS_QUANTSTATS:
            st.markdown("### QuantStats Full Report")
            
            col_report1, col_report2 = st.columns([3, 1])
            
            with col_report1:
                st.info("Generate comprehensive HTML report with all performance metrics and visualizations.")
            
            with col_report2:
                if st.button("📊 Generate Report", use_container_width=True, type="primary"):
                    with st.spinner("Generating comprehensive report..."):
                        # Initialize analytics engine
                        analytics_engine = AdvancedAnalyticsEngine(
                            portfolio_returns,
                            benchmark_returns,
                            risk_free_rate
                        )
                        
                        html_report = analytics_engine.generate_comprehensive_report()
                        
                        if html_report:
                            # Display in expander
                            with st.expander("📋 View HTML Report", expanded=True):
                                st.components.v1.html(html_report, height=800, scrolling=True)
                            
                            # Download button
                            b64 = base64.b64encode(html_report.encode()).decode()
                            href = f'''
                            <a href="data:text/html;base64,{b64}" 
                               download="bist_portfolio_report.html"
                               class="stButton">
                               <button style="
                                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                   color: white;
                                   padding: 0.5rem 1rem;
                                   border: none;
                                   border-radius: 5px;
                                   cursor: pointer;
                                   font-weight: bold;
                                   width: 100%;
                               ">
                               📥 Download HTML Report
                               </button>
                            </a>
                            '''
                            st.markdown(href, unsafe_allow_html=True)
                        else:
                            st.error("Failed to generate report.")
        
        # Data Export Section
        st.markdown("### 📤 Data Export")
        
        export_cols = st.columns(4)
        
        with export_cols[0]:
            if st.button("Export Weights CSV", use_container_width=True):
                weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                csv = weights_df.to_csv()
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'''
                <a href="data:file/csv;base64,{b64}" download="portfolio_weights.csv">
                <button style="
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white;
                    padding: 0.5rem;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-weight: bold;
                    width: 100%;
                ">
                📥 Download Weights
                </button>
                </a>
                '''
                st.markdown(href, unsafe_allow_html=True)
        
        with export_cols[1]:
            if st.button("Export Returns CSV", use_container_width=True):
                returns_df = pd.DataFrame({
                    'Portfolio': portfolio_returns,
                    'Benchmark': benchmark_returns if benchmark_returns is not None else np.nan
                })
                csv = returns_df.to_csv()
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'''
                <a href="data:file/csv;base64,{b64}" download="returns_data.csv">
                <button style="
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white;
                    padding: 0.5rem;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-weight: bold;
                    width: 100%;
                ">
                📥 Download Returns
                </button>
                </a>
                '''
                st.markdown(href, unsafe_allow_html=True)
        
        with export_cols[2]:
            if st.button("Export Metrics CSV", use_container_width=True):
                metrics_df = pd.DataFrame.from_dict(advanced_metrics, orient='index', columns=['Value'])
                csv = metrics_df.to_csv()
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'''
                <a href="data:file/csv;base64,{b64}" download="performance_metrics.csv">
                <button style="
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white;
                    padding: 0.5rem;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-weight: bold;
                    width: 100%;
                ">
                📥 Download Metrics
                </button>
                </a>
                '''
                st.markdown(href, unsafe_allow_html=True)
        
        with export_cols[3]:
            if st.button("Export Configuration", use_container_width=True):
                config = {
                    'timestamp': datetime.now().isoformat(),
                    'assets': assets,
                    'benchmark': benchmark_symbol,
                    'optimization_method': optimization_method,
                    'risk_model': risk_model,
                    'return_model': return_model,
                    'risk_free_rate': risk_free_rate,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'performance': {
                        'return': performance[0],
                        'volatility': performance[1],
                        'sharpe': performance[2]
                    }
                }
                
                json_str = json.dumps(config, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'''
                <a href="data:application/json;base64,{b64}" download="configuration.json">
                <button style="
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white;
                    padding: 0.5rem;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-weight: bold;
                    width: 100%;
                ">
                📥 Download Config
                </button>
                </a>
                '''
                st.markdown(href, unsafe_allow_html=True)
        
        # Configuration Log
        with st.expander("🔧 View Configuration Log", expanded=False):
            config_log = {
                'Date Range': f"{start_date} to {end_date}",
                'Assets Selected': assets,
                'Benchmark': benchmark_symbol,
                'Optimization Method': optimization_method,
                'Risk Model': risk_model,
                'Return Model': return_model,
                'Risk Free Rate': f"{risk_free_rate:.2%}",
                'Target Volatility': f"{target_volatility:.2%}" if optimization_method == 'efficient_risk' else 'N/A',
                'Target Return': f"{target_return:.2%}" if optimization_method == 'efficient_return' else 'N/A',
                'Performance': {
                    'Return': f"{performance[0]:.2%}",
                    'Volatility': f"{performance[1]:.2%}",
                    'Sharpe Ratio': f"{performance[2]:.2f}"
                }
            }
            
            st.json(config_log)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #b0b0b0; font-size: 0.9rem; padding: 1rem;">
        <p>BIST Enterprise Portfolio Analytics Suite v8.0 | Powered by Streamlit, PyPortfolioOpt & QuantStats</p>
        <p>📊 Institutional-Grade Portfolio Optimization & Risk Management</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION ENTRY POINT WITH ERROR HANDLING
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        # Check for required packages
        if not HAS_YFINANCE:
            st.error("""
            ## ❌ Missing Required Packages
            
            Please install the following packages:
            ```bash
            pip install yfinance pypfopt quantstats streamlit plotly pandas numpy scipy
            ```
            
            For Streamlit Cloud, add these to your `requirements.txt` file:
            ```
            yfinance>=0.2.28
            pypfopt>=1.5.5
            quantstats>=0.0.62
            streamlit>=1.28.0
            plotly>=5.17.0
            pandas>=2.0.0
            numpy>=1.24.0
            scipy>=1.11.0
            ```
            """)
        else:
            main()
            
    except Exception as e:
        st.error(f"## 🚨 Application Error")
        st.error(f"**Error Details:** {str(e)}")
        
        with st.expander("🔍 View Technical Details"):
            st.code(traceback.format_exc())
        
        st.info("""
        ## 🔧 Troubleshooting Steps:
        
        1. **Check your internet connection** - The app needs to download data from Yahoo Finance
        2. **Verify package installation** - Make sure all required packages are installed
        3. **Try a smaller date range** - Very large date ranges might cause timeouts
        4. **Select different tickers** - Some tickers might not have data available
        5. **Restart the application** - Click the refresh button in your browser
        
        If the problem persists, please check the Streamlit logs for more details.
        """)
        
        # Refresh button
        if st.button("🔄 Restart Application", type="primary"):
            st.cache_data.clear()
            st.rerun()
