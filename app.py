# ============================================================================
# BIST ENTERPRISE QUANT PORTFOLIO OPTIMIZATION SUITE
# Version: 7.0 | Professional-Grade Portfolio Analytics
# Features: Advanced PyPortfolioOpt Strategies + Full QuantStats Integration
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
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# QUANTITATIVE LIBRARIES - ENHANCED IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

try:
    # PyPortfolioOpt - All optimization methods
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
        plotting,
        discrete_allocation
    )
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    HAS_PYOPTOPT = True
except ImportError as e:
    st.error(f"PyPortfolioOpt import error: {e}")
    HAS_PYOPTOPT = False

# QuantStats - Full portfolio analytics
try:
    import quantstats as qs
    qs.extend_pandas()
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False

# Machine Learning & Advanced Analytics
try:
    from sklearn.covariance import LedoitWolf, GraphicalLasso, EmpiricalCovariance
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="BIST Quant Portfolio Lab Pro",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL CSS THEME
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

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED DATA SOURCE
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedDataSource:
    def __init__(self):
        self.cache = {}
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_enhanced_data(_self, tickers, start_date, end_date, interval='1d'):
        """Enhanced data fetching with multiple price fields"""
        try:
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                interval=interval,
                progress=False,
                group_by='ticker',
                auto_adjust=True
            )
            
            if len(tickers) > 1:
                close_prices = pd.DataFrame()
                for ticker in tickers:
                    if (ticker, 'Close') in data.columns:
                        close_prices[ticker] = data[(ticker, 'Close')]
            else:
                close_prices = data['Close'].to_frame(tickers[0])
            
            close_prices.ffill(inplace=True)
            close_prices.bfill(inplace=True)
            
            return {
                'close': close_prices,
                'returns': close_prices.pct_change().dropna()
            }
            
        except Exception as e:
            st.error(f"Data fetch error: {str(e)}")
            return None

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED PORTFOLIO OPTIMIZATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedPortfolioOptimizer:
    """Enhanced optimizer with multiple PyPortfolioOpt strategies"""
    
    def __init__(self, prices, returns):
        self.prices = prices
        self.returns = returns
        self.n_assets = len(prices.columns)
        
        # Initialize multiple models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all optimization models"""
        # Expected returns models
        self.mu_models = {
            'mean_historical': expected_returns.mean_historical_return(self.prices),
            'ema_historical': expected_returns.ema_historical_return(self.prices),
            'capm_return': expected_returns.capm_return(self.prices),
            'log_return': expected_returns.mean_historical_return(self.prices, log_returns=True)
        }
        
        # Risk models
        self.risk_models = {
            'sample_cov': risk_models.sample_cov(self.returns),
            'semicovariance': risk_models.semicovariance(self.returns),
            'exp_cov': risk_models.exp_cov(self.returns, span=180),
            'ledoit_wolf': risk_models.CovarianceShrinkage(self.prices).ledoit_wolf(),
            'oracle_approximating': risk_models.CovarianceShrinkage(self.prices).oracle_approximating(),
            'constant_correlation': risk_models.CovarianceShrinkage(self.prices).constant_correlation()
        }
    
    def optimize(self, strategy: str, **kwargs) -> Tuple[Dict, Tuple]:
        """Execute portfolio optimization with selected strategy"""
        
        # Get base parameters
        mu_model = kwargs.get('mu_model', 'mean_historical')
        risk_model = kwargs.get('risk_model', 'ledoit_wolf')
        risk_free_rate = kwargs.get('risk_free_rate', 0.0)
        
        mu = self.mu_models.get(mu_model, self.mu_models['mean_historical'])
        S = self.risk_models.get(risk_model, self.risk_models['ledoit_wolf'])
        
        # Execute strategy
        if strategy == 'max_sharpe':
            return self._max_sharpe(mu, S, risk_free_rate)
        elif strategy == 'min_volatility':
            return self._min_volatility(mu, S)
        elif strategy == 'max_quadratic_utility':
            return self._max_quadratic_utility(mu, S, kwargs.get('risk_aversion', 1.0))
        elif strategy == 'efficient_risk':
            return self._efficient_risk(mu, S, kwargs.get('target_volatility', 0.15))
        elif strategy == 'efficient_return':
            return self._efficient_return(mu, S, kwargs.get('target_return', 0.20))
        elif strategy == 'hrp':
            return self._hrp_optimization()
        elif strategy == 'cvar':
            return self._cvar_optimization(mu, kwargs.get('confidence_level', 0.95))
        elif strategy == 'semivariance':
            return self._semivariance_optimization(mu, kwargs.get('target_return', 0.15))
        elif strategy == 'black_litterman':
            return self._black_litterman_optimization(kwargs.get('views', None))
        elif strategy == 'risk_parity':
            return self._risk_parity_optimization(S)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _max_sharpe(self, mu, S, risk_free_rate: float = 0.0) -> Tuple[Dict, Tuple]:
        """Maximize Sharpe Ratio"""
        ef = EfficientFrontier(mu, S)
        
        # Add constraints if provided
        if hasattr(self, 'constraints'):
            for constraint in self.constraints:
                ef.add_constraint(constraint)
        
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        return weights, performance
    
    def _min_volatility(self, mu, S) -> Tuple[Dict, Tuple]:
        """Minimize Portfolio Volatility"""
        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        return weights, performance
    
    def _max_quadratic_utility(self, mu, S, risk_aversion: float = 1.0) -> Tuple[Dict, Tuple]:
        """Maximize Quadratic Utility"""
        ef = EfficientFrontier(mu, S)
        ef.max_quadratic_utility(risk_aversion=risk_aversion)
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        return weights, performance
    
    def _efficient_risk(self, mu, S, target_volatility: float = 0.15) -> Tuple[Dict, Tuple]:
        """Efficient portfolio for target volatility"""
        ef = EfficientFrontier(mu, S)
        ef.efficient_risk(target_volatility=target_volatility)
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        return weights, performance
    
    def _efficient_return(self, mu, S, target_return: float = 0.20) -> Tuple[Dict, Tuple]:
        """Efficient portfolio for target return"""
        ef = EfficientFrontier(mu, S)
        ef.efficient_return(target_return=target_return)
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        return weights, performance
    
    def _hrp_optimization(self) -> Tuple[Dict, Tuple]:
        """Hierarchical Risk Parity Optimization"""
        hrp = HRPOpt(self.returns)
        hrp.optimize()
        weights = hrp.clean_weights()
        
        # Calculate performance
        port_returns = (self.returns * pd.Series(weights)).sum(axis=1)
        ann_return = (1 + port_returns.mean()) ** 252 - 1
        ann_vol = port_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        return weights, (ann_return, ann_vol, sharpe)
    
    def _cvar_optimization(self, mu, confidence_level: float = 0.95) -> Tuple[Dict, Tuple]:
        """Conditional Value at Risk Optimization"""
        ec = EfficientCVaR(mu, self.returns)
        ec.min_cvar()
        weights = ec.clean_weights()
        performance = ec.portfolio_performance(verbose=False)
        return weights, performance
    
    def _semivariance_optimization(self, mu, target_return: float = 0.15) -> Tuple[Dict, Tuple]:
        """Semi-variance Optimization"""
        es = EfficientSemivariance(mu, self.returns)
        es.efficient_return(target_return=target_return)
        weights = es.clean_weights()
        performance = es.portfolio_performance(verbose=False)
        return weights, performance
    
    def _black_litterman_optimization(self, views: Optional[Dict] = None) -> Tuple[Dict, Tuple]:
        """Black-Litterman Model Optimization"""
        if views is None:
            # Generate default views
            views = {
                'AKBNK.IS': 0.10,  # 10% expected return
                'GARAN.IS': 0.12,
                'THYAO.IS': 0.15
            }
        
        bl = BlackLittermanModel(self.risk_models['sample_cov'], absolute_views=views)
        ret_bl = bl.bl_returns()
        
        ef = EfficientFrontier(ret_bl, self.risk_models['sample_cov'])
        ef.max_sharpe()
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        
        return weights, performance
    
    def _risk_parity_optimization(self, S) -> Tuple[Dict, Tuple]:
        """Risk Parity Optimization"""
        # Simple inverse volatility weighting
        volatilities = np.sqrt(np.diag(S))
        weights = 1 / volatilities
        weights = weights / weights.sum()
        
        weights_dict = {asset: weight for asset, weight in zip(self.prices.columns, weights)}
        
        # Calculate performance
        port_returns = (self.returns * weights).sum(axis=1)
        ann_return = (1 + port_returns.mean()) ** 252 - 1
        ann_vol = port_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        return weights_dict, (ann_return, ann_vol, sharpe)
    
    def generate_efficient_frontier(self, points: int = 100) -> Tuple:
        """Generate efficient frontier points"""
        mu = self.mu_models['mean_historical']
        S = self.risk_models['ledoit_wolf']
        
        ef = EfficientFrontier(mu, S)
        mus, sigmas, weights = ef.efficient_frontier(points=points)
        
        return mus, sigmas, weights

# ─────────────────────────────────────────────────────────────────────────────
# QUANTSTATS ANALYTICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class QuantStatsAnalytics:
    """Comprehensive portfolio analytics using QuantStats"""
    
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series = None,
                 risk_free_rate: float = 0.0):
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        # Ensure series format
        if isinstance(self.portfolio_returns, pd.DataFrame):
            self.portfolio_returns = self.portfolio_returns.iloc[:, 0]
        
        if benchmark_returns is not None and isinstance(benchmark_returns, pd.DataFrame):
            self.benchmark_returns = benchmark_returns.iloc[:, 0]
    
    def calculate_all_metrics(self) -> Dict:
        """Calculate comprehensive performance and risk metrics"""
        metrics = {}
        
        if not HAS_QUANTSTATS:
            return self._calculate_basic_metrics()
        
        try:
            # Performance Metrics
            metrics.update(self._calculate_performance_metrics())
            
            # Risk Metrics
            metrics.update(self._calculate_risk_metrics())
            
            # Risk-Adjusted Return Metrics
            metrics.update(self._calculate_risk_adjusted_metrics())
            
            # Drawdown Metrics
            metrics.update(self._calculate_drawdown_metrics())
            
            # Statistical Metrics
            metrics.update(self._calculate_statistical_metrics())
            
        except Exception as e:
            st.error(f"QuantStats metrics calculation error: {str(e)}")
            metrics.update(self._calculate_basic_metrics())
        
        return metrics
    
    def _calculate_basic_metrics(self) -> Dict:
        """Calculate basic metrics without QuantStats"""
        metrics = {}
        
        # Basic performance
        metrics['Total Return'] = (1 + self.portfolio_returns).prod() - 1
        metrics['CAGR'] = (1 + self.portfolio_returns.mean()) ** 252 - 1
        metrics['Annual Volatility'] = self.portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        if metrics['Annual Volatility'] > 0:
            metrics['Sharpe Ratio'] = (metrics['CAGR'] - self.risk_free_rate) / metrics['Annual Volatility']
        else:
            metrics['Sharpe Ratio'] = 0
        
        # Max Drawdown
        cum_returns = (1 + self.portfolio_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        metrics['Max Drawdown'] = drawdown.min()
        
        return metrics
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics using QuantStats"""
        metrics = {}
        
        # Time-weighted returns
        metrics['Total Return'] = qs.stats.comp(self.portfolio_returns)
        metrics['CAGR'] = qs.stats.cagr(self.portfolio_returns)
        metrics['Expected Return (Annual)'] = qs.stats.expected_return(self.portfolio_returns, aggregate='year')
        
        # Best/Worst periods
        metrics['Best Day'] = qs.stats.best(self.portfolio_returns)
        metrics['Worst Day'] = qs.stats.worst(self.portfolio_returns)
        metrics['Best Month'] = qs.stats.best(self.portfolio_returns, aggregate='month')
        metrics['Worst Month'] = qs.stats.worst(self.portfolio_returns, aggregate='month')
        metrics['Best Year'] = qs.stats.best(self.portfolio_returns, aggregate='year')
        metrics['Worst Year'] = qs.stats.worst(self.portfolio_returns, aggregate='year')
        
        return metrics
    
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk metrics using QuantStats"""
        metrics = {}
        
        # Volatility metrics
        metrics['Annual Volatility'] = qs.stats.volatility(self.portfolio_returns)
        metrics['Monthly Volatility'] = qs.stats.volatility(self.portfolio_returns, aggregate='month')
        metrics['Downside Deviation'] = qs.stats.downside_risk(self.portfolio_returns)
        
        # Value at Risk
        metrics['VaR (95%)'] = qs.stats.value_at_risk(self.portfolio_returns)
        metrics['CVaR (95%)'] = qs.stats.conditional_value_at_risk(self.portfolio_returns)
        
        # Tail risk
        metrics['Skewness'] = qs.stats.skew(self.portfolio_returns)
        metrics['Kurtosis'] = qs.stats.kurtosis(self.portfolio_returns)
        metrics['Tail Ratio'] = qs.stats.tail_ratio(self.portfolio_returns)
        
        return metrics
    
    def _calculate_risk_adjusted_metrics(self) -> Dict:
        """Calculate risk-adjusted return metrics"""
        metrics = {}
        
        # Sharpe family
        metrics['Sharpe Ratio'] = qs.stats.sharpe(self.portfolio_returns, risk_free=self.risk_free_rate)
        metrics['Sortino Ratio'] = qs.stats.sortino(self.portfolio_returns, risk_free=self.risk_free_rate)
        metrics['Modified Sharpe'] = qs.stats.modified_sharpe(self.portfolio_returns, risk_free=self.risk_free_rate)
        
        # Other risk-adjusted ratios
        metrics['Calmar Ratio'] = qs.stats.calmar(self.portfolio_returns)
        metrics['Omega Ratio'] = qs.stats.omega(self.portfolio_returns, risk_free=self.risk_free_rate)
        metrics['Gain to Pain Ratio'] = qs.stats.gain_to_pain_ratio(self.portfolio_returns)
        
        # Information ratio if benchmark exists
        if self.benchmark_returns is not None:
            metrics['Information Ratio'] = qs.stats.information_ratio(self.portfolio_returns, self.benchmark_returns)
            metrics['Tracking Error'] = qs.stats.tracking_error(self.portfolio_returns, self.benchmark_returns)
            metrics['Beta'] = qs.stats.beta(self.portfolio_returns, self.benchmark_returns)
            metrics['Alpha (Annual)'] = qs.stats.alpha(self.portfolio_returns, self.benchmark_returns, risk_free=self.risk_free_rate)
        
        return metrics
    
    def _calculate_drawdown_metrics(self) -> Dict:
        """Calculate drawdown-related metrics"""
        metrics = {}
        
        # Drawdown metrics
        metrics['Max Drawdown'] = qs.stats.max_drawdown(self.portfolio_returns)
        metrics['Avg Drawdown'] = qs.stats.avg_drawdown(self.portfolio_returns)
        metrics['Avg Drawdown Days'] = qs.stats.avg_drawdown(self.portfolio_returns, prepare_returns=False)
        
        # Recovery metrics
        metrics['Recovery Factor'] = qs.stats.recovery_factor(self.portfolio_returns)
        metrics['Ulcer Index'] = qs.stats.ulcer_index(self.portfolio_returns)
        metrics['Serenity Index'] = qs.stats.serenity_index(self.portfolio_returns, risk_free=self.risk_free_rate)
        
        return metrics
    
    def _calculate_statistical_metrics(self) -> Dict:
        """Calculate statistical metrics"""
        metrics = {}
        
        # Win rates
        metrics['Win Rate'] = qs.stats.win_rate(self.portfolio_returns)
        metrics['Win Rate (Monthly)'] = qs.stats.win_rate(self.portfolio_returns, aggregate='month')
        metrics['Win Rate (Yearly)'] = qs.stats.win_rate(self.portfolio_returns, aggregate='year')
        
        # Profit factor and expectancy
        metrics['Profit Factor'] = qs.stats.profit_factor(self.portfolio_returns)
        metrics['Expectancy'] = qs.stats.expectancy(self.portfolio_returns)
        
        # Common ratio
        metrics['Common Sense Ratio'] = qs.stats.common_sense_ratio(self.portfolio_returns)
        
        return metrics
    
    def generate_tearsheet(self) -> go.Figure:
        """Generate professional tearsheet visualization"""
        if not HAS_QUANTSTATS:
            return None
        
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Cumulative Returns', 'Daily Returns', 'Rolling Sharpe (6M)',
                'Drawdown', 'Rolling Volatility (6M)', 'Monthly Returns Heatmap',
                'Return Distribution', 'QQ Plot', 'Underwater Plot'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "histogram"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"colspan": 3, "type": "scatter"}, None, None]
            ]
        )
        
        # 1. Cumulative Returns
        cum_returns = (1 + self.portfolio_returns).cumprod()
        fig.add_trace(
            go.Scatter(x=cum_returns.index, y=cum_returns.values,
                      name='Portfolio', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        if self.benchmark_returns is not None:
            bench_cum = (1 + self.benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(x=bench_cum.index, y=bench_cum.values,
                          name='Benchmark', line=dict(color='red', width=2, dash='dash')),
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
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        
        # 3. Rolling Sharpe (6M)
        rolling_sharpe = self.portfolio_returns.rolling(126).apply(
            lambda x: qs.stats.sharpe(x, risk_free=self.risk_free_rate)
        )
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                      name='Rolling Sharpe', line=dict(color='green', width=2)),
            row=1, col=3
        )
        
        # 4. Drawdown
        drawdown = qs.stats.to_drawdown_series(self.portfolio_returns)
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      fill='tozeroy', fillcolor='rgba(255,0,0,0.3)',
                      line=dict(color='red', width=1), name='Drawdown'),
            row=2, col=1
        )
        
        # 5. Rolling Volatility (6M)
        rolling_vol = self.portfolio_returns.rolling(126).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                      name='Rolling Volatility', line=dict(color='purple', width=2)),
            row=2, col=2
        )
        
        # 6. Monthly Returns Heatmap
        monthly_returns = self.portfolio_returns.resample('M').apply(lambda x: (1+x).prod()-1)
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        monthly_pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
        
        # Create heatmap trace
        heatmap_trace = go.Heatmap(
            z=monthly_pivot.values,
            x=monthly_pivot.columns,
            y=monthly_pivot.index,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title="Return")
        )
        fig.add_trace(heatmap_trace, row=2, col=3)
        
        # 7. Return Distribution
        fig.add_trace(
            go.Histogram(x=self.portfolio_returns.values, nbinsx=50,
                        name='Return Distribution', marker_color='blue',
                        opacity=0.7),
            row=3, col=1
        )
        
        # Add normal distribution overlay
        x = np.linspace(self.portfolio_returns.min(), self.portfolio_returns.max(), 100)
        y = stats.norm.pdf(x, self.portfolio_returns.mean(), self.portfolio_returns.std())
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', name='Normal Dist',
                      line=dict(color='red', dash='dash')),
            row=3, col=1
        )
        
        # 8. QQ Plot
        if len(self.portfolio_returns) > 0:
            qq_data = stats.probplot(self.portfolio_returns.values, dist="norm")
            x_theoretical = qq_data[0][0]
            y_sample = qq_data[0][1]
            
            fig.add_trace(
                go.Scatter(x=x_theoretical, y=y_sample, mode='markers',
                          marker=dict(size=5, color='blue'), name='QQ Plot'),
                row=3, col=2
            )
            
            # Add 45-degree line
            min_val = min(x_theoretical.min(), y_sample.min())
            max_val = max(x_theoretical.max(), y_sample.max())
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                          mode='lines', line=dict(color='red', dash='dash'),
                          name='Normal Line'),
                row=3, col=2
            )
        
        # 9. Underwater Plot
        underwater = self.portfolio_returns.copy()
        underwater[underwater > 0] = 0
        underwater_cum = underwater.cumsum()
        
        fig.add_trace(
            go.Scatter(x=underwater_cum.index, y=underwater_cum.values,
                      fill='tozeroy', fillcolor='rgba(0,0,255,0.3)',
                      line=dict(color='blue', width=1), name='Underwater'),
            row=3, col=3
        )
        
        # 10. Rolling Beta (if benchmark exists)
        if self.benchmark_returns is not None:
            rolling_beta = self.portfolio_returns.rolling(126).apply(
                lambda x: qs.stats.beta(x, self.benchmark_returns.loc[x.index])
            )
            fig.add_trace(
                go.Scatter(x=rolling_beta.index, y=rolling_beta.values,
                          name='Rolling Beta', line=dict(color='orange', width=2)),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            template='plotly_dark',
            title_text="Portfolio Tearsheet",
            title_x=0.5
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Beta", row=4, col=1)
        
        return fig

# ─────────────────────────────────────────────────────────────────────────────
# RISK ANALYTICS MODULE
# ─────────────────────────────────────────────────────────────────────────────

class RiskAnalytics:
    """Advanced risk analytics module"""
    
    def __init__(self, returns: pd.Series, benchmark_returns: pd.Series = None):
        self.returns = returns
        self.benchmark_returns = benchmark_returns
    
    def calculate_var_metrics(self, confidence_levels: List[float] = [0.90, 0.95, 0.99]) -> Dict:
        """Calculate Value at Risk metrics at multiple confidence levels"""
        results = {}
        
        for cl in confidence_levels:
            # Historical VaR/CVaR
            var_hist = np.percentile(self.returns, (1 - cl) * 100)
            cvar_hist = self.returns[self.returns <= var_hist].mean()
            
            # Parametric VaR (Normal)
            var_param = self.returns.mean() + stats.norm.ppf(1 - cl) * self.returns.std()
            
            # Modified VaR (Cornish-Fisher)
            z = stats.norm.ppf(1 - cl)
            s = stats.skew(self.returns)
            k = stats.kurtosis(self.returns)
            z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
            var_cf = self.returns.mean() + z_cf * self.returns.std()
            
            results[f'CL_{int(cl*100)}'] = {
                'Historical_VaR': var_hist,
                'Parametric_VaR': var_param,
                'Cornish_Fisher_VaR': var_cf,
                'Historical_CVaR': cvar_hist
            }
        
        return results
    
    def calculate_risk_contribution(self, weights: np.ndarray, covariance: np.ndarray) -> Dict:
        """Calculate risk contribution analysis"""
        portfolio_variance = np.dot(weights.T, np.dot(covariance, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Marginal contribution to risk
        marginal_contrib = np.dot(covariance, weights) / portfolio_volatility
        
        # Percent contribution
        percent_contrib = (weights * marginal_contrib) / portfolio_volatility
        
        # Component VaR
        component_var = percent_contrib * portfolio_volatility * stats.norm.ppf(0.95)
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'marginal_contribution': marginal_contrib,
            'percent_contribution': percent_contrib,
            'component_var': component_var,
            'diversification_ratio': np.sum(weights * np.sqrt(np.diag(covariance))) / portfolio_volatility
        }
    
    def calculate_stress_test(self, stress_scenarios: Dict = None) -> Dict:
        """Calculate stress test results for different scenarios"""
        if stress_scenarios is None:
            stress_scenarios = {
                'Market Crash': -0.10,  # 10% market drop
                'Volatility Spike': 0.05,  # 5% increase in volatility
                'Interest Rate Hike': -0.03,  # 3% drop due to rate hikes
            }
        
        results = {}
        for scenario, shock in stress_scenarios.items():
            stressed_returns = self.returns * (1 + shock)
            results[scenario] = {
                'Mean Return': stressed_returns.mean() * 252,
                'Volatility': stressed_returns.std() * np.sqrt(252),
                'VaR_95': np.percentile(stressed_returns, 5),
                'Max_Drawdown': qs.stats.max_drawdown(stressed_returns) if HAS_QUANTSTATS else None
            }
        
        return results

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
        
        # Asset Selection
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
        
        # Optimization Strategy
        st.subheader("Optimization Strategy")
        strategy_options = [
            'max_sharpe',
            'min_volatility',
            'max_quadratic_utility',
            'efficient_risk',
            'efficient_return',
            'hrp',
            'cvar',
            'semivariance',
            'risk_parity'
        ]
        
        optimization_strategy = st.selectbox("Strategy", strategy_options)
        
        # Advanced Parameters
        with st.expander("Advanced Parameters"):
            risk_free_rate = st.number_input("Risk Free Rate (%)", 0.0, 50.0, 30.0) / 100
            
            if optimization_strategy == 'efficient_risk':
                target_volatility = st.slider("Target Volatility", 0.05, 0.50, 0.15, 0.01)
            else:
                target_volatility = 0.15
            
            if optimization_strategy == 'efficient_return':
                target_return = st.slider("Target Return", 0.05, 1.0, 0.20, 0.01)
            else:
                target_return = 0.20
            
            risk_model = st.selectbox(
                "Risk Model",
                ['ledoit_wolf', 'sample_cov', 'semicovariance', 'exp_cov', 'oracle_approximating', 'constant_correlation']
            )
            
            return_model = st.selectbox(
                "Return Model",
                ['mean_historical', 'ema_historical', 'capm_return', 'log_return']
            )
        
        # Reporting Options
        st.subheader("Reporting")
        generate_tearsheet = st.checkbox("Generate Tearsheet", True)
        show_metrics = st.checkbox("Show All Metrics", True)
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
        optimizer = AdvancedPortfolioOptimizer(prices, returns)
        
        # Prepare optimization parameters
        opt_params = {
            'risk_free_rate': risk_free_rate,
            'risk_model': risk_model,
            'mu_model': return_model,
        }
        
        if optimization_strategy == 'efficient_risk':
            opt_params['target_volatility'] = target_volatility
        elif optimization_strategy == 'efficient_return':
            opt_params['target_return'] = target_return
        
        weights, performance = optimizer.optimize(optimization_strategy, **opt_params)
        
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
        "📑 Reports & Export"
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
        
        fig_frontier.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            title="Efficient Frontier"
        )
        
        st.plotly_chart(fig_frontier, use_container_width=True)
    
    with tab3:
        # Risk Analytics
        st.header("⚠️ Comprehensive Risk Analysis")
        
        risk_analytics = RiskAnalytics(portfolio_returns, benchmark_returns)
        
        col_risk1, col_risk2 = st.columns(2)
        
        with col_risk1:
            st.subheader("Value at Risk Analysis")
            
            # Calculate VaR/CVaR at different confidence levels
            var_results = risk_analytics.calculate_var_metrics([0.90, 0.95, 0.99])
            
            # Create DataFrame for display
            var_data = []
            for cl, metrics in var_results.items():
                var_data.append({
                    'Confidence Level': cl.replace('CL_', '') + '%',
                    'Historical VaR': metrics['Historical_VaR'],
                    'Parametric VaR': metrics['Parametric_VaR'],
                    'CVaR': metrics['Historical_CVaR']
                })
            
            var_df = pd.DataFrame(var_data)
            st.dataframe(
                var_df.style.format("{:.4f}"),
                use_container_width=True
            )
        
        with col_risk2:
            st.subheader("Drawdown Analysis")
            
            # Calculate drawdown series
            if HAS_QUANTSTATS:
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
                
                fig_dd.update_layout(
                    template="plotly_dark",
                    height=400,
                    title="Portfolio Drawdown",
                    yaxis_title="Drawdown",
                    yaxis_tickformat=".2%"
                )
                
                st.plotly_chart(fig_dd, use_container_width=True)
    
    with tab4:
        # Performance Analytics
        st.header("📈 Advanced Performance Analytics")
        
        # Initialize QuantStats analytics
        qs_analytics = QuantStatsAnalytics(
            portfolio_returns,
            benchmark_returns,
            risk_free_rate
        )
        
        # Calculate all metrics
        if show_metrics:
            with st.spinner("Calculating advanced metrics..."):
                advanced_metrics = qs_analytics.calculate_all_metrics()
                
                # Display metrics in expandable sections
                with st.expander("Performance Metrics", expanded=True):
                    perf_metrics = {k: v for k, v in advanced_metrics.items() 
                                  if any(keyword in k.lower() for keyword in ['return', 'cagr', 'total'])}
                    st.json(perf_metrics)
                
                with st.expander("Risk Metrics"):
                    risk_metrics = {k: v for k, v in advanced_metrics.items() 
                                  if any(keyword in k.lower() for keyword in ['volatility', 'var', 'cvar', 'drawdown'])}
                    st.json(risk_metrics)
                
                with st.expander("Risk-Adjusted Metrics"):
                    ra_metrics = {k: v for k, v in advanced_metrics.items() 
                                if any(keyword in k.lower() for keyword in ['ratio', 'sharpe', 'sortino', 'calmar', 'omega'])}
                    st.json(ra_metrics)
        
        # Generate tearsheet
        if generate_tearsheet and HAS_QUANTSTATS:
            st.subheader("Professional Tearsheet")
            tearsheet_fig = qs_analytics.generate_tearsheet()
            if tearsheet_fig:
                st.plotly_chart(tearsheet_fig, use_container_width=True)
    
    with tab5:
        # Reporting Section
        st.header("📑 Professional Reporting & Export")
        
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
        
        # Configuration Summary
        with st.expander("Configuration Summary"):
            config_summary = {
                'Date Range': f"{start_date} to {end_date}",
                'Assets': assets,
                'Benchmark': benchmark_symbol,
                'Optimization Strategy': optimization_strategy,
                'Risk Model': risk_model,
                'Return Model': return_model,
                'Risk Free Rate': f"{risk_free_rate:.2%}",
                'Performance': {
                    'Expected Return': f"{performance[0]:.2%}",
                    'Expected Volatility': f"{performance[1]:.2%}",
                    'Sharpe Ratio': f"{performance[2]:.2f}"
                }
            }
            st.json(config_summary)

if __name__ == "__main__":
    main()
