# ============================================================================
# BIST ENTERPRISE QUANT PORTFOLIO OPTIMIZATION SUITE
# Version: 6.0 | Features: Full PyPortfolioOpt + QuantStats Integration
# Institutional-Grade Analytics & Visualizations
# ============================================================================

import warnings
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
import yfinance as yf
from io import BytesIO
import json

# ─────────────────────────────────────────────────────────────────────────────
# QUANTITATIVE LIBRARIES - ENHANCED IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

# PyPortfolioOpt - Enhanced imports with all optimization methods
try:
    from pypfopt import (
        expected_returns, 
        risk_models, 
        EfficientFrontier, 
        HRPOpt, 
        EfficientCVaR,
        EfficientSemivariance,
        CLA,
        black_litterman,
        BlackLittermanModel,
        objective_functions,
        plotting
    )
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    HAS_PYOPTOPT = True
except ImportError as e:
    st.error(f"PyPortfolioOpt import error: {e}")
    HAS_PYOPTOPT = False

# QuantStats - Enhanced with full reporting
try:
    import quantstats as qs
    qs.extend_pandas()
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False

# Machine Learning & Econometrics
try:
    from sklearn.covariance import LedoitWolf, GraphicalLasso
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# RiskMetrics
try:
    import riskmetrics as rm
    HAS_RISKMETRICS = True
except ImportError:
    HAS_RISKMETRICS = False

# Visualization
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Streamlit Configuration
st.set_page_config(
    page_title="BIST Quant Portfolio Lab Pro",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# INSTITUTIONAL CSS THEME
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
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
        --border-color: #2d3748;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Professional Metrics */
    .metric-card {
        background: linear-gradient(135deg, var(--secondary-dark), var(--primary-dark));
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 102, 204, 0.2);
        border-color: var(--accent-blue);
    }
    
    /* Custom DataFrames */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: var(--secondary-dark);
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
        background-color: transparent;
        border-radius: 4px;
        color: var(--text-secondary);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-blue) !important;
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(0, 102, 204, 0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--primary-dark);
        border-right: 1px solid var(--border-color);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED DATA STRUCTURES
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
    'S&P 500': '^GSPC'
}

class EnhancedDataSource:
    def __init__(self):
        self.cache = {}
        
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_enhanced_data(_self, tickers, start_date, end_date, interval='1d'):
        """Enhanced data fetching with multiple price fields"""
        try:
            # Download OHLCV data
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                interval=interval,
                progress=False,
                group_by='ticker',
                auto_adjust=True
            )
            
            # Process data based on number of tickers
            if len(tickers) > 1:
                # Multi-ticker case
                close_prices = pd.DataFrame()
                open_prices = pd.DataFrame()
                high_prices = pd.DataFrame()
                low_prices = pd.DataFrame()
                volumes = pd.DataFrame()
                
                for ticker in tickers:
                    if (ticker, 'Close') in data.columns:
                        close_prices[ticker] = data[(ticker, 'Close')]
                        open_prices[ticker] = data[(ticker, 'Open')]
                        high_prices[ticker] = data[(ticker, 'High')]
                        low_prices[ticker] = data[(ticker, 'Low')]
                        volumes[ticker] = data[(ticker, 'Volume')]
            else:
                # Single ticker case
                ticker = tickers[0]
                close_prices = data['Close'].to_frame(ticker)
                open_prices = data['Open'].to_frame(ticker)
                high_prices = data['High'].to_frame(ticker)
                low_prices = data['Low'].to_frame(ticker)
                volumes = data['Volume'].to_frame(ticker)
            
            # Fill missing values
            close_prices.ffill(inplace=True)
            close_prices.bfill(inplace=True)
            
            return {
                'close': close_prices,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'volume': volumes,
                'returns': close_prices.pct_change().dropna()
            }
            
        except Exception as e:
            st.error(f"Data fetch error: {str(e)}")
            return None
            
    def fetch_fundamental_data(self, ticker):
        """Fetch fundamental data for a single ticker"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'market_cap': info.get('marketCap', None),
                'pe_ratio': info.get('trailingPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None)
            }
        except:
            return None

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED PORTFOLIO OPTIMIZATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedPortfolioOptimizer:
    def __init__(self, prices, returns):
        self.prices = prices
        self.returns = returns
        
        # Multiple expected return models
        self.mu_models = {
            'mean_historical': expected_returns.mean_historical_return(prices),
            'ema_historical': expected_returns.ema_historical_return(prices),
            'capm_return': expected_returns.capm_return(prices),
        }
        
        # Multiple risk models
        self.risk_models = {
            'sample_cov': risk_models.sample_cov(returns),
            'semicovariance': risk_models.semicovariance(returns),
            'exp_cov': risk_models.exp_cov(returns),
            'ledoit_wolf': risk_models.CovarianceShrinkage(prices).ledoit_wolf(),
            'oracle_approximating': risk_models.CovarianceShrinkage(prices).oracle_approximating(),
        }
        
    def optimize_portfolio(self, method='max_sharpe', risk_model='ledoit_wolf', 
                          return_model='mean_historical', **kwargs):
        """Comprehensive portfolio optimization with multiple methods"""
        
        mu = self.mu_models.get(return_model, self.mu_models['mean_historical'])
        S = self.risk_models.get(risk_model, self.risk_models['ledoit_wolf'])
        
        if method == 'max_sharpe':
            ef = EfficientFrontier(mu, S)
            ef.max_sharpe()
            weights = ef.clean_weights()
            
        elif method == 'min_volatility':
            ef = EfficientFrontier(mu, S)
            ef.min_volatility()
            weights = ef.clean_weights()
            
        elif method == 'max_quadratic_utility':
            ef = EfficientFrontier(mu, S)
            ef.max_quadratic_utility()
            weights = ef.clean_weights()
            
        elif method == 'efficient_risk':
            ef = EfficientFrontier(mu, S)
            target_vol = kwargs.get('target_volatility', 0.15)
            ef.efficient_risk(target_volatility=target_vol)
            weights = ef.clean_weights()
            
        elif method == 'efficient_return':
            ef = EfficientFrontier(mu, S)
            target_return = kwargs.get('target_return', 0.20)
            ef.efficient_return(target_return=target_return)
            weights = ef.clean_weights()
            
        elif method == 'hrp':
            hrp = HRPOpt(self.returns)
            hrp.optimize()
            weights = hrp.clean_weights()
            
        elif method == 'cvar':
            ec = EfficientCVaR(mu, self.returns)
            ec.min_cvar()
            weights = ec.clean_weights()
            
        elif method == 'semivariance':
            es = EfficientSemivariance(mu, self.returns)
            es.efficient_return(target_return=kwargs.get('target_return', 0.15))
            weights = es.clean_weights()
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Calculate performance metrics
        if method in ['hrp', 'cvar', 'semivariance']:
            # For special optimizers, calculate manually
            port_returns = (self.returns * pd.Series(weights)).sum(axis=1)
            annual_return = (1 + port_returns.mean()) ** 252 - 1
            annual_vol = port_returns.std() * np.sqrt(252)
            sharpe = (annual_return - kwargs.get('risk_free_rate', 0.0)) / annual_vol if annual_vol > 0 else 0
            performance = (annual_return, annual_vol, sharpe)
        else:
            performance = ef.portfolio_performance(verbose=False)
        
        return weights, performance
    
    def generate_efficient_frontier(self, points=100):
        """Generate efficient frontier points"""
        mu = self.mu_models['mean_historical']
        S = self.risk_models['ledoit_wolf']
        
        ef = EfficientFrontier(mu, S)
        
        # Generate frontier
        mus, sigmas, weights = ef.efficient_frontier(points=points)
        
        return mus, sigmas, weights
    
    def calculate_discrete_allocation(self, weights, total_portfolio_value=100000):
        """Calculate discrete allocation with latest prices"""
        latest_prices = get_latest_prices(self.prices)
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
        allocation, leftover = da.lp_portfolio()
        return allocation, leftover

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED ANALYTICS WITH QUANTSTATS
# ─────────────────────────────────────────────────────────────────────────────

class QuantStatsAnalytics:
    def __init__(self, portfolio_returns, benchmark_returns=None, risk_free_rate=0.0):
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        # Ensure series format for QuantStats
        if isinstance(self.portfolio_returns, pd.DataFrame):
            self.portfolio_returns = self.portfolio_returns.iloc[:, 0]
    
    def generate_full_report(self, mode='basic'):
        """Generate comprehensive QuantStats report"""
        if not HAS_QUANTSTATS:
            return None
        
        try:
            # Create HTML report
            if self.benchmark_returns is not None:
                # Full comparative report
                if mode == 'full':
                    html = qs.reports.html(
                        self.portfolio_returns,
                        self.benchmark_returns,
                        rf=self.risk_free_rate,
                        title='BIST Portfolio Analysis',
                        output=None
                    )
                else:
                    # Basic metrics report
                    html = qs.reports.metrics(
                        self.portfolio_returns,
                        self.benchmark_returns,
                        rf=self.risk_free_rate,
                        display=False,
                        mode='full'
                    )
            else:
                html = qs.reports.html(
                    self.portfolio_returns,
                    rf=self.risk_free_rate,
                    title='Portfolio Analysis',
                    output=None
                )
            
            return html
            
        except Exception as e:
            st.error(f"QuantStats report error: {str(e)}")
            return None
    
    def calculate_advanced_metrics(self):
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        if HAS_QUANTSTATS:
            # Use QuantStats for advanced metrics
            metrics.update(qs.reports.metrics(
                self.portfolio_returns,
                self.benchmark_returns if self.benchmark_returns is not None else self.portfolio_returns,
                rf=self.risk_free_rate,
                display=False
            ))
        
        # Additional custom metrics
        if self.benchmark_returns is not None:
            # Calculate tracking error
            tracking_error = (self.portfolio_returns - self.benchmark_returns).std() * np.sqrt(252)
            metrics['Tracking Error'] = tracking_error
            
            # Information Ratio
            excess_return = (self.portfolio_returns - self.benchmark_returns).mean() * 252
            metrics['Information Ratio'] = excess_return / tracking_error if tracking_error > 0 else 0
            
            # Beta calculation
            covariance = np.cov(self.portfolio_returns, self.benchmark_returns)[0, 1]
            benchmark_variance = np.var(self.benchmark_returns)
            metrics['Beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        return metrics
    
    def generate_tearsheet(self):
        """Generate professional tearsheet"""
        if not HAS_QUANTSTATS:
            return None
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Cumulative Returns', 'Daily Returns',
                          'Rolling Sharpe (6M)', 'Drawdown',
                          'Monthly Returns Heatmap', 'Return Distribution'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Cumulative Returns
        cum_returns = (1 + self.portfolio_returns).cumprod()
        fig.add_trace(
            go.Scatter(x=cum_returns.index, y=cum_returns.values, 
                      name='Portfolio', line=dict(color='blue')),
            row=1, col=1
        )
        
        if self.benchmark_returns is not None:
            bench_cum = (1 + self.benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(x=bench_cum.index, y=bench_cum.values,
                          name='Benchmark', line=dict(color='red', dash='dash')),
                row=1, col=1
            )
        
        # 2. Daily Returns
        fig.add_trace(
            go.Scatter(x=self.portfolio_returns.index, y=self.portfolio_returns.values,
                      mode='markers', marker=dict(size=3, color=self.portfolio_returns.values,
                                                 colorscale='RdBu', showscale=False),
                      name='Daily Returns'),
            row=1, col=2
        )
        
        # 3. Rolling Sharpe (6M)
        rolling_sharpe = self.portfolio_returns.rolling(126).apply(
            lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                      name='Rolling Sharpe', line=dict(color='green')),
            row=2, col=1
        )
        
        # 4. Drawdown
        drawdown = qs.stats.to_drawdown_series(self.portfolio_returns)
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      fill='tozeroy', fillcolor='rgba(255,0,0,0.3)',
                      line=dict(color='red'), name='Drawdown'),
            row=2, col=2
        )
        
        # 5. Monthly Returns Heatmap
        monthly_returns = self.portfolio_returns.resample('M').apply(lambda x: (1+x).prod()-1)
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month_name(),
            'Return': monthly_returns.values
        })
        
        # 6. Return Distribution
        fig.add_trace(
            go.Histogram(x=self.portfolio_returns.values, nbinsx=50,
                        name='Return Distribution', marker_color='blue'),
            row=3, col=2
        )
        
        fig.update_layout(height=1000, showlegend=True, template='plotly_dark')
        return fig

# ─────────────────────────────────────────────────────────────────────────────
# RISK ANALYTICS MODULE
# ─────────────────────────────────────────────────────────────────────────────

class RiskAnalytics:
    def __init__(self, returns, benchmark_returns=None):
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        
    def calculate_var_cvar(self, confidence_levels=[0.90, 0.95, 0.99]):
        """Calculate Value at Risk and Conditional VaR at multiple confidence levels"""
        results = {}
        
        for cl in confidence_levels:
            # Historical VaR/CVaR
            var_hist = np.percentile(self.returns, (1-cl)*100)
            cvar_hist = self.returns[self.returns <= var_hist].mean()
            
            # Parametric VaR (Normal)
            var_param = self.returns.mean() + stats.norm.ppf(1-cl) * self.returns.std()
            
            # Modified VaR (Cornish-Fisher)
            z = stats.norm.ppf(1-cl)
            s = stats.skew(self.returns)
            k = stats.kurtosis(self.returns)
            z_cf = z + (z**2 - 1)*s/6 + (z**3 - 3*z)*k/24 - (2*z**3 - 5*z)*s**2/36
            var_cf = self.returns.mean() + z_cf * self.returns.std()
            
            results[f'VaR_{int(cl*100)}'] = {
                'Historical': var_hist,
                'Parametric': var_param,
                'Cornish_Fisher': var_cf,
                'CVaR_Historical': cvar_hist
            }
        
        return results
    
    def calculate_risk_decomposition(self, weights, covariance_matrix):
        """Decompose portfolio risk into marginal contributions"""
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        
        # Marginal contribution to risk
        marginal_contrib = np.dot(covariance_matrix, weights) / np.sqrt(portfolio_variance)
        
        # Percent contribution
        percent_contrib = (weights * marginal_contrib) / np.sqrt(portfolio_variance)
        
        return {
            'marginal_contribution': marginal_contrib,
            'percent_contribution': percent_contrib,
            'portfolio_volatility': np.sqrt(portfolio_variance)
        }
    
    def calculate_garch_volatility(self, p=1, q=1):
        """Calculate GARCH volatility forecasts"""
        if not HAS_ARCH:
            return None
        
        try:
            am = arch_model(self.returns * 100, vol='Garch', p=p, q=q)
            res = am.fit(disp='off')
            forecast = res.forecast(horizon=5)
            return forecast.variance.iloc[-1].values
        except:
            return None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Sidebar Configuration
    with st.sidebar:
        st.title("⚙️ Configuration Panel")
        
        # Date Selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                     datetime.now() - timedelta(days=365*3))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Asset Selection with Sectors
        st.subheader("Asset Selection")
        selected_sector = st.selectbox("Filter by Sector", 
                                      ["All"] + list(SECTOR_MAPPING.keys()))
        
        if selected_sector == "All":
            available_tickers = BIST30_TICKERS
        else:
            available_tickers = SECTOR_MAPPING[selected_sector]
        
        assets = st.multiselect("Select Assets", 
                               available_tickers,
                               default=['THYAO.IS', 'GARAN.IS', 'ASELS.IS'])
        
        # Benchmark Selection
        benchmark_symbol = st.selectbox("Benchmark", list(BENCHMARKS.keys()))
        
        # Optimization Parameters
        st.subheader("Optimization Parameters")
        optimization_method = st.selectbox(
            "Optimization Method",
            ['max_sharpe', 'min_volatility', 'efficient_risk', 
             'efficient_return', 'hrp', 'cvar', 'semivariance']
        )
        
        risk_model = st.selectbox(
            "Risk Model",
            ['ledoit_wolf', 'sample_cov', 'semicovariance', 'exp_cov', 'oracle_approximating']
        )
        
        return_model = st.selectbox(
            "Return Model",
            ['mean_historical', 'ema_historical', 'capm_return']
        )
        
        # Advanced Parameters
        with st.expander("Advanced Parameters"):
            risk_free_rate = st.number_input("Risk Free Rate (%)", 0.0, 50.0, 30.0) / 100
            target_volatility = st.slider("Target Volatility", 0.05, 0.50, 0.15, 0.01)
            target_return = st.slider("Target Return", 0.05, 1.0, 0.20, 0.01)
        
        # Reporting Options
        st.subheader("Reporting")
        generate_full_report = st.checkbox("Generate Full QuantStats Report", True)
        show_tearsheet = st.checkbox("Show Interactive Tearsheet", True)
        calculate_discrete = st.checkbox("Calculate Discrete Allocation", False)
        
        if calculate_discrete:
            portfolio_value = st.number_input("Portfolio Value (TRY)", 
                                            10000, 10000000, 1000000, 10000)
    
    # Main Dashboard
    st.title("📊 BIST Enterprise Portfolio Analytics Suite")
    st.caption("Professional Portfolio Optimization & Risk Analytics Platform")
    
    if len(assets) < 2:
        st.warning("⚠️ Please select at least 2 assets for portfolio optimization.")
        return
    
    # Data Loading
    with st.spinner("🔄 Loading market data..."):
        data_source = EnhancedDataSource()
        data = data_source.fetch_enhanced_data(assets, start_date, end_date)
        benchmark_data = data_source.fetch_enhanced_data(
            [BENCHMARKS[benchmark_symbol]], start_date, end_date
        )
        
        if data is None or benchmark_data is None:
            st.error("❌ Failed to load data. Please check your connection and try again.")
            return
        
        prices = data['close']
        returns = data['returns']
        benchmark_returns = benchmark_data['returns'].iloc[:, 0]
    
    # Portfolio Optimization
    with st.spinner("⚡ Optimizing portfolio..."):
        optimizer = EnhancedPortfolioOptimizer(prices, returns)
        
        weights, performance = optimizer.optimize_portfolio(
            method=optimization_method,
            risk_model=risk_model,
            return_model=return_model,
            target_volatility=target_volatility,
            target_return=target_return,
            risk_free_rate=risk_free_rate
        )
        
        # Calculate portfolio returns
        portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
    
    # Performance Metrics Dashboard
    st.header("📈 Performance Dashboard")
    
    # Top Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Expected Return", f"{performance[0]:.2%}")
    with col2:
        st.metric("Expected Volatility", f"{performance[1]:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{performance[2]:.2f}")
    with col4:
        var_95 = np.percentile(portfolio_returns, 5)
        st.metric("VaR (95%)", f"{var_95:.2%}")
    with col5:
        max_dd = qs.stats.max_drawdown(portfolio_returns) if HAS_QUANTSTATS else 0
        st.metric("Max Drawdown", f"{max_dd:.2%}")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Portfolio Overview", 
        "📊 Optimization Analysis",
        "⚠️ Risk Analytics", 
        "📈 Performance Analytics",
        "📑 Reports"
    ])
    
    with tab1:
        # Portfolio Overview
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.subheader("Optimal Allocation")
            
            # Convert weights to DataFrame
            weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
            weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
            
            # Pie chart
            fig_pie = px.pie(
                weights_df, 
                values='Weight', 
                names=weights_df.index,
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig_pie.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Weights table
            st.dataframe(
                weights_df.style.format("{:.2%}").background_gradient(cmap='Blues'),
                use_container_width=True
            )
            
            # Discrete allocation if requested
            if calculate_discrete:
                st.subheader("Discrete Allocation")
                allocation, leftover = optimizer.calculate_discrete_allocation(
                    weights, portfolio_value
                )
                
                alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Shares'])
                st.dataframe(alloc_df, use_container_width=True)
                st.info(f"Remaining cash: ₺{leftover:,.2f}")
        
        with col_right:
            st.subheader("Cumulative Performance")
            
            # Calculate cumulative returns
            cum_port = (1 + portfolio_returns).cumprod()
            cum_bench = (1 + benchmark_returns).cumprod()
            
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=cum_port.index, y=cum_port.values,
                name='Optimized Portfolio',
                line=dict(color='#00cc88', width=3)
            ))
            fig_cum.add_trace(go.Scatter(
                x=cum_bench.index, y=cum_bench.values,
                name=benchmark_symbol,
                line=dict(color='#0066cc', width=2, dash='dash')
            ))
            
            fig_cum.update_layout(
                template="plotly_dark",
                height=500,
                hovermode='x unified',
                yaxis_title="Cumulative Return",
                xaxis_title="Date"
            )
            st.plotly_chart(fig_cum, use_container_width=True)
            
            # Rolling metrics
            st.subheader("Rolling Metrics (6M Window)")
            
            # Calculate rolling Sharpe and volatility
            rolling_window = 126  # 6 months
            
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
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                          name='Sharpe', line=dict(color='green')),
                row=1, col=1
            )
            fig_rolling.add_trace(
                go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                          name='Volatility', line=dict(color='red')),
                row=2, col=1
            )
            
            fig_rolling.update_layout(height=400, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_rolling, use_container_width=True)
    
    with tab2:
        # Optimization Analysis
        st.subheader("Efficient Frontier Analysis")
        
        # Generate efficient frontier
        mus, sigmas, frontier_weights = optimizer.generate_efficient_frontier()
        
        fig_frontier = go.Figure()
        
        # Plot frontier
        fig_frontier.add_trace(go.Scatter(
            x=sigmas, y=mus,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='white', width=2)
        ))
        
        # Plot optimal point
        fig_frontier.add_trace(go.Scatter(
            x=[performance[1]], y=[performance[0]],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name='Optimal Portfolio'
        ))
        
        # Plot individual assets
        for i, asset in enumerate(prices.columns):
            mu_i = expected_returns.mean_historical_return(prices[asset])
            sigma_i = np.sqrt(risk_models.sample_cov(returns[asset].to_frame()).iloc[0,0]) * np.sqrt(252)
            
            fig_frontier.add_trace(go.Scatter(
                x=[sigma_i], y=[mu_i],
                mode='markers+text',
                marker=dict(size=10),
                text=[asset],
                textposition="top center",
                name=asset,
                showlegend=False
            ))
        
        fig_frontier.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            title="Efficient Frontier with Individual Assets"
        )
        
        st.plotly_chart(fig_frontier, use_container_width=True)
        
        # Weight sensitivity analysis
        st.subheader("Weight Sensitivity Analysis")
        
        # Create sensitivity analysis
        sensitivity_results = []
        base_weights = pd.Series(weights)
        
        for asset in base_weights.index:
            if base_weights[asset] > 0:
                # Vary weight and calculate impact
                weight_variations = np.linspace(
                    max(0, base_weights[asset] - 0.1),
                    min(1, base_weights[asset] + 0.1),
                    20
                )
                
                for w in weight_variations:
                    adjusted_weights = base_weights.copy()
                    adjusted_weights[asset] = w
                    # Renormalize
                    adjusted_weights = adjusted_weights / adjusted_weights.sum()
                    
                    # Calculate portfolio metrics
                    port_ret = (returns * adjusted_weights).sum(axis=1)
                    ann_ret = (1 + port_ret.mean()) ** 252 - 1
                    ann_vol = port_ret.std() * np.sqrt(252)
                    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0
                    
                    sensitivity_results.append({
                        'Asset': asset,
                        'Weight': w,
                        'Return': ann_ret,
                        'Volatility': ann_vol,
                        'Sharpe': sharpe
                    })
        
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # Plot sensitivity
        if not sensitivity_df.empty:
            fig_sensitivity = px.scatter(
                sensitivity_df,
                x='Volatility',
                y='Return',
                color='Asset',
                size='Weight',
                hover_data=['Sharpe'],
                title='Weight Sensitivity Analysis'
            )
            fig_sensitivity.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig_sensitivity, use_container_width=True)
    
    with tab3:
        # Risk Analytics
        st.header("⚠️ Comprehensive Risk Analysis")
        
        risk_analytics = RiskAnalytics(portfolio_returns, benchmark_returns)
        
        col_risk1, col_risk2 = st.columns(2)
        
        with col_risk1:
            st.subheader("Value at Risk Analysis")
            
            # Calculate VaR/CVaR at different confidence levels
            var_results = risk_analytics.calculate_var_cvar([0.90, 0.95, 0.99])
            
            var_df = pd.DataFrame({
                'Confidence Level': ['90%', '95%', '99%'],
                'Historical VaR': [var_results['VaR_90']['Historical'], 
                                  var_results['VaR_95']['Historical'], 
                                  var_results['VaR_99']['Historical']],
                'Parametric VaR': [var_results['VaR_90']['Parametric'], 
                                  var_results['VaR_95']['Parametric'], 
                                  var_results['VaR_99']['Parametric']],
                'CVaR': [var_results['VaR_90']['CVaR_Historical'], 
                        var_results['VaR_95']['CVaR_Historical'], 
                        var_results['VaR_99']['CVaR_Historical']]
            })
            
            st.dataframe(
                var_df.style.format("{:.4f}").background_gradient(cmap='Reds_r', subset=['Historical VaR', 'CVaR']),
                use_container_width=True
            )
            
            # Risk decomposition
            st.subheader("Risk Decomposition")
            
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf().values
            w_array = np.array([weights.get(asset, 0) for asset in prices.columns])
            
            risk_decomp = risk_analytics.calculate_risk_decomposition(w_array, S)
            
            decomp_df = pd.DataFrame({
                'Asset': prices.columns,
                'Weight': w_array,
                'Marginal Contribution': risk_decomp['marginal_contribution'],
                'Percent Contribution': risk_decomp['percent_contribution']
            }).sort_values('Percent Contribution', ascending=False)
            
            st.dataframe(
                decomp_df.style.format({
                    'Weight': '{:.2%}',
                    'Marginal Contribution': '{:.6f}',
                    'Percent Contribution': '{:.2%}'
                }).background_gradient(cmap='YlOrRd', subset=['Percent Contribution']),
                use_container_width=True
            )
        
        with col_risk2:
            st.subheader("Drawdown Analysis")
            
            # Calculate drawdown series
            drawdown_series = qs.stats.to_drawdown_series(portfolio_returns)
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values,
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red'),
                name='Drawdown'
            ))
            
            # Add max drawdown line
            max_dd_point = drawdown_series.idxmin()
            max_dd_value = drawdown_series.min()
            
            fig_dd.add_vline(
                x=max_dd_point,
                line_dash="dash",
                line_color="yellow",
                annotation_text=f"Max DD: {max_dd_value:.2%}"
            )
            
            fig_dd.update_layout(
                template="plotly_dark",
                height=400,
                title="Portfolio Drawdown",
                yaxis_title="Drawdown",
                yaxis_tickformat=".2%"
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
            
            # Drawdown statistics
            dd_stats = {
                'Max Drawdown': max_dd_value,
                'Avg Drawdown': drawdown_series.mean(),
                'Drawdown Duration (days)': qs.stats.max_duration(portfolio_returns),
                'Recovery Time (days)': qs.stats.time_to_recovery(portfolio_returns)
            }
            
            dd_stats_df = pd.DataFrame.from_dict(dd_stats, orient='index', columns=['Value'])
            st.dataframe(dd_stats_df.style.format("{:.4f}"), use_container_width=True)
            
            # Tail Risk Analysis
            st.subheader("Tail Risk Analysis")
            
            # Calculate exceedances
            var_95 = np.percentile(portfolio_returns, 5)
            exceedances = portfolio_returns[portfolio_returns <= var_95]
            
            tail_stats = {
                'Tail Losses Count': len(exceedances),
                'Average Tail Loss': exceedances.mean(),
                'Worst Tail Loss': exceedances.min(),
                'Tail to Volatility Ratio': abs(exceedances.mean()) / portfolio_returns.std()
            }
            
            tail_df = pd.DataFrame.from_dict(tail_stats, orient='index', columns=['Value'])
            st.dataframe(tail_df.style.format("{:.4f}"), use_container_width=True)
    
    with tab4:
        # Performance Analytics with QuantStats
        st.header("📈 Advanced Performance Analytics")
        
        # Initialize QuantStats analytics
        qs_analytics = QuantStatsAnalytics(
            portfolio_returns,
            benchmark_returns,
            risk_free_rate
        )
        
        # Calculate advanced metrics
        advanced_metrics = qs_analytics.calculate_advanced_metrics()
        
        # Display metrics in columns
        metrics_cols = st.columns(4)
        metric_items = list(advanced_metrics.items())
        
        for idx, (key, value) in enumerate(metric_items[:16]):  # Show first 16 metrics
            with metrics_cols[idx % 4]:
                try:
                    if isinstance(value, (int, float)):
                        if 'Ratio' in key or 'Sharpe' in key or 'Beta' in key:
                            display_val = f"{value:.2f}"
                        elif 'Return' in key or 'Drawdown' in key:
                            display_val = f"{value:.2%}"
                        else:
                            display_val = f"{value:.4f}"
                        st.metric(key, display_val)
                except:
                    pass
        
        # Generate tearsheet if requested
        if show_tearsheet and HAS_QUANTSTATS:
            st.subheader("Interactive Tearsheet")
            
            tearsheet_fig = qs_analytics.generate_tearsheet()
            if tearsheet_fig:
                st.plotly_chart(tearsheet_fig, use_container_width=True)
        
        # Monthly Returns Heatmap
        st.subheader("Monthly Returns Heatmap")
        
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1+x).prod()-1)
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.strftime('%b'),
            'Return': monthly_returns.values
        })
        
        # Create pivot table for heatmap
        monthly_pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
        
        # Ensure correct month order
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pivot = monthly_pivot.reindex(columns=month_order)
        
        # Create heatmap
        fig_heatmap = px.imshow(
            monthly_pivot,
            text_auto='.2%',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Monthly Returns Heatmap'
        )
        fig_heatmap.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Return Distribution with VaR
        st.subheader("Return Distribution Analysis")
        
        fig_dist = ff.create_distplot(
            [portfolio_returns.values, benchmark_returns.values],
            ['Portfolio', 'Benchmark'],
            bin_size=0.001,
            show_rug=False,
            colors=['#00cc88', '#0066cc']
        )
        
        # Add VaR lines
        var_95_port = np.percentile(portfolio_returns, 5)
        var_95_bench = np.percentile(benchmark_returns, 5)
        
        fig_dist.add_vline(
            x=var_95_port,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Port VaR 95%: {var_95_port:.2%}"
        )
        
        fig_dist.add_vline(
            x=var_95_bench,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Bench VaR 95%: {var_95_bench:.2%}"
        )
        
        fig_dist.update_layout(
            template="plotly_dark",
            height=500,
            title="Return Distribution with VaR Levels",
            xaxis_title="Daily Returns",
            yaxis_title="Density"
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab5:
        # Reporting Section
        st.header("📑 Professional Reporting")
        
        # QuantStats Full Report
        if generate_full_report and HAS_QUANTSTATS:
            st.subheader("QuantStats Full Report")
            
            if st.button("Generate QuantStats HTML Report", type="primary"):
                with st.spinner("Generating comprehensive report..."):
                    qs_report = qs_analytics.generate_full_report('full')
                    
                    if qs_report:
                        # Display in expander
                        with st.expander("View HTML Report", expanded=True):
                            st.components.v1.html(qs_report, height=1000, scrolling=True)
                        
                        # Download button
                        b64 = base64.b64encode(qs_report.encode()).decode()
                        href = f'''
                        <a href="data:text/html;base64,{b64}" 
                           download="quantstats_report.html" 
                           class="stButton">
                           📥 Download HTML Report
                        </a>
                        '''
                        st.markdown(href, unsafe_allow_html=True)
        
        # Performance Summary Card
        st.subheader("Performance Summary")
        
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            st.info("**Return Metrics**")
            st.metric("CAGR", f"{performance[0]:.2%}")
            st.metric("Total Return", 
                     f"{((1 + portfolio_returns).prod() - 1):.2%}")
        
        with summary_cols[1]:
            st.info("**Risk Metrics**")
            st.metric("Annual Vol", f"{performance[1]:.2%}")
            st.metric("Sharpe Ratio", f"{performance[2]:.2f}")
            st.metric("Sortino Ratio", 
                     f"{qs.stats.sortino(portfolio_returns, risk_free_rate):.2f}" 
                     if HAS_QUANTSTATS else "N/A")
        
        with summary_cols[2]:
            st.info("**Risk-Adjusted Metrics**")
            st.metric("Calmar Ratio", 
                     f"{qs.stats.calmar(portfolio_returns):.2f}" 
                     if HAS_QUANTSTATS else "N/A")
            st.metric("Omega Ratio", 
                     f"{qs.stats.omega(portfolio_returns, risk_free_rate):.2f}" 
                     if HAS_QUANTSTATS else "N/A")
            st.metric("Information Ratio", 
                     f"{qs.stats.information_ratio(portfolio_returns, benchmark_returns):.2f}" 
                     if HAS_QUANTSTATS and benchmark_returns is not None else "N/A")
        
        # Export Data
        st.subheader("Data Export")
        
        export_cols = st.columns(4)
        
        with export_cols[0]:
            if st.button("Export Weights CSV"):
                weights_df = pd.DataFrame.from_dict(weights, orient='index', 
                                                  columns=['Weight'])
                csv = weights_df.to_csv()
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="portfolio_weights.csv">Download Weights</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with export_cols[1]:
            if st.button("Export Returns CSV"):
                returns_df = pd.DataFrame({
                    'Portfolio': portfolio_returns,
                    'Benchmark': benchmark_returns
                })
                csv = returns_df.to_csv()
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="returns_data.csv">Download Returns</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with export_cols[2]:
            if st.button("Export Performance Metrics"):
                metrics_df = pd.DataFrame.from_dict(advanced_metrics, 
                                                  orient='index', 
                                                  columns=['Value'])
                csv = metrics_df.to_csv()
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="performance_metrics.csv">Download Metrics</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        # Configuration Log
        with st.expander("View Configuration Log"):
            config_log = {
                'Date Range': f"{start_date} to {end_date}",
                'Assets Selected': assets,
                'Benchmark': benchmark_symbol,
                'Optimization Method': optimization_method,
                'Risk Model': risk_model,
                'Return Model': return_model,
                'Risk Free Rate': f"{risk_free_rate:.2%}",
                'Target Volatility': f"{target_volatility:.2%}" if optimization_method == 'efficient_risk' else 'N/A',
                'Target Return': f"{target_return:.2%}" if optimization_method == 'efficient_return' else 'N/A'
            }
            
            st.json(config_log)

if __name__ == "__main__":
    main()
