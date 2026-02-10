# ============================================================================
# 1. CORE IMPORTS & CONFIGURATION (STREAMLIT ADAPTED)
# ============================================================================
import warnings
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as stats 
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

# Additional quantitative libraries
from scipy.optimize import minimize
from scipy.stats import norm, t, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# ARCH: For Econometric Volatility Forecasting (GARCH)
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# Streamlit Page Configuration
st.set_page_config(
    page_title="BIST Portfolio Risk Analytics | Institutional Platform",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F9FAFB;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        border: 1px solid #10B981;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Turkish BIST 30 tickers
BIST30_TICKERS = [
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS',
    'EKGYO.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HALKB.IS',
    'ISCTR.IS', 'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS',
    'PETKM.IS', 'PGSUS.IS', 'SAHOL.IS', 'SASA.IS', 'SISE.IS',
    'SKBNK.IS', 'TCELL.IS', 'THYAO.IS', 'TKFEN.IS', 'TOASO.IS',
    'TTKOM.IS', 'TUPRS.IS', 'ULKER.IS', 'VAKBN.IS', 'YKBNK.IS'
]

# Annualized risk-free rate for Turkey (updated for 2024)
RISK_FREE_RATE = 0.42  # Updated to reflect current Turkish rates ~42%

# ============================================================================
# 2. ENHANCED TURKISH PORTFOLIO OPTIMIZER CLASS
# ============================================================================

class TurkishPortfolioOptimizer:
    def __init__(self):
        self.tickers = BIST30_TICKERS
        self.benchmark_tickers = ['XU100.IS', 'XU030.IS']
        self.risk_free_rate = RISK_FREE_RATE
        self.data = None
        self.returns = None
        self.benchmark_data = None
        self.benchmark_returns = None
        
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data(_self, start_date='2022-01-01', end_date=None):
        """Fetch data from Yahoo Finance with enhanced error handling"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        try:
            # Download BIST 30 stocks with progress
            with st.spinner(f"Downloading data from {start_date} to {end_date}..."):
                data = yf.download(_self.tickers, start=start_date, end=end_date, 
                                 progress=False, group_by='ticker')
                
                # Handle different data structure formats
                if isinstance(data.columns, pd.MultiIndex):
                    data = data['Adj Close']
                else:
                    # Try to get adjusted close if single column structure
                    data = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                
                # Download benchmark indices
                benchmark_data = yf.download(_self.benchmark_tickers, start=start_date, 
                                           end=end_date, progress=False)
                
                if isinstance(benchmark_data.columns, pd.MultiIndex):
                    benchmark_data = benchmark_data['Adj Close']
                else:
                    benchmark_data = benchmark_data['Adj Close'] if 'Adj Close' in benchmark_data.columns else benchmark_data['Close']
            
            # Forward fill missing values (business days only)
            data = data.ffill().bfill()
            benchmark_data = benchmark_data.ffill().bfill()
            
            # Filter stocks with sufficient data (minimum 80% non-NA)
            min_days_required = int(len(data) * 0.8)
            valid_tickers = data.columns[data.notna().sum() > min_days_required]
            data = data[valid_tickers]
            
            if data.empty:
                st.error("No valid tickers found with sufficient data. Please adjust date range.")
                return None, None, None, None
            
            # Calculate daily returns with log returns for better statistical properties
            returns = np.log(data / data.shift(1)).dropna()
            benchmark_returns = np.log(benchmark_data / benchmark_data.shift(1)).dropna()
            
            return data, returns, benchmark_data, benchmark_returns
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None, None, None, None
    
    def calculate_enhanced_metrics(self, weights_series: pd.Series, returns: pd.DataFrame, 
                                  benchmark_returns: pd.DataFrame, risk_free_rate: float):
        """Calculate comprehensive portfolio performance metrics"""
        
        # Convert annual risk-free rate to daily
        daily_rf = np.log(1 + risk_free_rate) / 252
        
        # Ensure weights are aligned and normalized
        aligned_weights = weights_series.reindex(returns.columns).fillna(0)
        aligned_weights = aligned_weights / aligned_weights.sum()  # Ensure sum to 1
        
        # Portfolio return series
        portfolio_returns = (returns * aligned_weights).sum(axis=1)
        
        # Basic statistics
        mean_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        
        # Annualized metrics (252 trading days)
        annual_return = mean_return * 252
        annual_volatility = volatility * np.sqrt(252)
        
        # Sharpe Ratio (annualized)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum Drawdown with dates
        cum_returns = np.exp(portfolio_returns.cumsum())
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin() if not drawdown.empty else None
        recovery_date = None
        if max_dd_date:
            recovery_idx = (cum_returns.loc[max_dd_date:] >= running_max.loc[max_dd_date]).idxmax() if any(cum_returns.loc[max_dd_date:] >= running_max.loc[max_dd_date]) else None
            recovery_date = recovery_idx if recovery_idx != max_dd_date else None
        
        # Value at Risk (parametric and historical)
        var_95_hist = np.percentile(portfolio_returns, 5)
        var_95_param = norm.ppf(0.05, mean_return, volatility)
        
        # Expected Shortfall/CVaR
        cvar_95 = portfolio_returns[portfolio_returns <= var_95_hist].mean()
        
        # Sortino Ratio
        downside_returns = portfolio_returns[portfolio_returns < daily_rf]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Omega Ratio
        threshold_return = daily_rf
        gains = portfolio_returns[portfolio_returns > threshold_return].sum()
        losses = abs(portfolio_returns[portfolio_returns <= threshold_return].sum())
        omega_ratio = gains / losses if losses > 0 else np.inf
        
        # Information Ratio and Tracking Error
        information_ratio = 0
        tracking_error = 0
        active_return = 0
        benchmark_annual_return = 0
        
        if 'XU100.IS' in benchmark_returns.columns:
            bench_returns = benchmark_returns['XU100.IS'].reindex(portfolio_returns.index).fillna(0)
            if not bench_returns.empty:
                benchmark_annual_return = bench_returns.mean() * 252
                active_return = annual_return - benchmark_annual_return
                active_returns = portfolio_returns - bench_returns
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        # Higher Moments
        skewness = skew(portfolio_returns)
        excess_kurtosis = kurtosis(portfolio_returns, fisher=False)  # Fisher=False gives Pearson's definition
        
        # Beta calculation
        beta = 0
        if 'XU100.IS' in benchmark_returns.columns:
            bench_returns_aligned = benchmark_returns['XU100.IS'].reindex(portfolio_returns.index).dropna()
            port_returns_aligned = portfolio_returns.reindex(bench_returns_aligned.index).dropna()
            if len(port_returns_aligned) > 1:
                covariance = np.cov(port_returns_aligned, bench_returns_aligned)[0, 1]
                bench_variance = np.var(bench_returns_aligned)
                beta = covariance / bench_variance if bench_variance > 0 else 0
        
        # Treynor Ratio
        treynor_ratio = (annual_return - risk_free_rate) / beta if beta > 0 else 0
        
        # Appraisal Ratio (Jensen's Alpha divided by idiosyncratic risk)
        alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate)) if beta > 0 else 0
        idiosyncratic_risk = tracking_error  # Approximation
        appraisal_ratio = alpha / idiosyncratic_risk if idiosyncratic_risk > 0 else 0
        
        # Gain/Loss Ratio (Bernardo-Ledoit)
        positive_returns = portfolio_returns[portfolio_returns > 0]
        negative_returns = portfolio_returns[portfolio_returns < 0]
        gain_loss_ratio = abs(positive_returns.mean() / negative_returns.mean()) if len(negative_returns) > 0 and negative_returns.mean() < 0 else np.inf
        
        # Tail Ratio (95% VaR)
        var_99 = np.percentile(portfolio_returns, 1)
        tail_ratio = abs(var_99 / var_95_hist) if var_95_hist != 0 else 0
        
        # Recovery Time (in days)
        recovery_days = None
        if recovery_date and max_dd_date:
            recovery_days = (recovery_date - max_dd_date).days
        
        metrics = {
            # Return Metrics
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Cumulative Return': float(cum_returns.iloc[-1] - 1) if not cum_returns.empty else 0,
            'Mean Daily Return': mean_return,
            
            # Risk-Adjusted Return Metrics
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Omega Ratio': omega_ratio,
            'Treynor Ratio': treynor_ratio,
            
            # Risk Metrics
            'Max Drawdown': max_drawdown,
            'Max DD Date': max_dd_date,
            'Recovery Days': recovery_days,
            'VaR (95% Historical)': var_95_hist,
            'VaR (95% Parametric)': var_95_param,
            'CVaR (95%)': cvar_95,
            'Tail Ratio (99%/95%)': tail_ratio,
            'Gain/Loss Ratio': gain_loss_ratio,
            
            # Relative Metrics
            'Information Ratio': information_ratio,
            'Tracking Error': tracking_error,
            'Active Return': active_return,
            'Beta': beta,
            'Alpha': alpha,
            'Appraisal Ratio': appraisal_ratio,
            
            # Statistical Metrics
            'Skewness': skewness,
            'Kurtosis': excess_kurtosis,
            'Jarque-Bera Stat': stats.jarque_bera(portfolio_returns)[0] if len(portfolio_returns) > 0 else 0,
            'Jarque-Bera p-value': stats.jarque_bera(portfolio_returns)[1] if len(portfolio_returns) > 0 else 1,
            
            # Additional
            'Downside Volatility': downside_volatility,
            'Upside Volatility': portfolio_returns[portfolio_returns > daily_rf].std() * np.sqrt(252) if len(portfolio_returns[portfolio_returns > daily_rf]) > 1 else 0,
            'Win Rate': len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0,
            'Avg Win / Avg Loss': abs(positive_returns.mean() / negative_returns.mean()) if len(negative_returns) > 0 and negative_returns.mean() < 0 else np.inf,
        }
        
        return metrics, portfolio_returns, cum_returns, drawdown
    
    def optimize_portfolio(self, method, mu, S, returns, risk_free_rate, target_return=None, risk_aversion=1, constraints=None):
        """Portfolio optimization with enhanced methods and constraints"""
        
        daily_rf = risk_free_rate / 252
        
        # Default constraints if none provided
        if constraints is None:
            constraints = {'min_weight': 0, 'max_weight': 1}
        
        try:
            if method == 'max_sharpe':
                ef = EfficientFrontier(mu, S)
                ef.add_constraint(lambda w: w >= constraints.get('min_weight', 0))
                ef.add_constraint(lambda w: w <= constraints.get('max_weight', 1))
                ef.max_sharpe(risk_free_rate=daily_rf)
                weights = ef.clean_weights()
                
            elif method == 'min_volatility':
                ef = EfficientFrontier(mu, S)
                ef.add_constraint(lambda w: w >= constraints.get('min_weight', 0))
                ef.add_constraint(lambda w: w <= constraints.get('max_weight', 1))
                ef.min_volatility()
                weights = ef.clean_weights()
                
            elif method == 'efficient_risk':
                ef = EfficientFrontier(mu, S)
                target_vol = target_return if target_return is not None else mu.std().mean() * np.sqrt(252) * 0.8
                ef.add_constraint(lambda w: w >= constraints.get('min_weight', 0))
                ef.add_constraint(lambda w: w <= constraints.get('max_weight', 1))
                ef.efficient_risk(target_volatility=target_vol/np.sqrt(252))
                weights = ef.clean_weights()
                
            elif method == 'efficient_return':
                ef = EfficientFrontier(mu, S)
                target_ret = target_return if target_return is not None else mu.mean().mean() * 252 * 0.8
                daily_target = target_ret / 252
                ef.add_constraint(lambda w: w >= constraints.get('min_weight', 0))
                ef.add_constraint(lambda w: w <= constraints.get('max_weight', 1))
                ef.efficient_return(target_return=daily_target)
                weights = ef.clean_weights()
                
            elif method == 'max_quadratic_utility':
                ef = EfficientFrontier(mu, S)
                ef.add_constraint(lambda w: w >= constraints.get('min_weight', 0))
                ef.add_constraint(lambda w: w <= constraints.get('max_weight', 1))
                ef.max_quadratic_utility(risk_aversion=risk_aversion)
                weights = ef.clean_weights()
                
            elif method == 'hrp':
                hrp = HRPOpt(returns)
                weights = hrp.optimize()
                weights = hrp.clean_weights()
                
            elif method == 'cvar':
                cvar = EfficientCVaR(mu, returns)
                cvar.add_constraint(lambda w: w >= constraints.get('min_weight', 0))
                cvar.add_constraint(lambda w: w <= constraints.get('max_weight', 1))
                cvar.min_cvar()
                weights = cvar.clean_weights()
                
            elif method == 'equal_weight':
                n_assets = len(returns.columns)
                weights = {ticker: 1/n_assets for ticker in returns.columns}
                
            elif method == 'risk_parity':
                # Risk Parity portfolio
                n = len(returns.columns)
                initial_weights = np.ones(n) / n
                
                def risk_parity_objective(w):
                    portfolio_vol = np.sqrt(w @ S @ w.T)
                    marginal_risk = (S @ w.T) / portfolio_vol
                    risk_contributions = w * marginal_risk
                    target_rc = portfolio_vol / n
                    return np.sum((risk_contributions - target_rc) ** 2)
                
                bounds = [(constraints.get('min_weight', 0), constraints.get('max_weight', 1)) for _ in range(n)]
                constraints_opt = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
                
                result = minimize(risk_parity_objective, initial_weights, 
                                bounds=bounds, constraints=constraints_opt)
                weights = {ticker: result.x[i] for i, ticker in enumerate(returns.columns)}
                
            else:
                raise ValueError(f"Unknown optimization method: {method}")
        
        except Exception as e:
            st.warning(f"Optimization failed for {method}: {str(e)}. Using Equal Weight as fallback.")
            n_assets = len(returns.columns)
            weights = {ticker: 1/n_assets for ticker in returns.columns}
        
        # Convert weights to DataFrame
        weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
        weights_df.index.name = 'Ticker'
        weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
        
        # Calculate performance for the optimized portfolio
        weights_series = pd.Series(weights).reindex(returns.columns).fillna(0)
        metrics, portfolio_returns, cum_returns, drawdown = self.calculate_enhanced_metrics(
            weights_series, returns, None, risk_free_rate
        )
        
        performance = (metrics['Annual Return'], metrics['Annual Volatility'], metrics['Sharpe Ratio'])
        
        return weights_df, performance, metrics

    def plot_enhanced_efficient_frontier(self, mu, S, returns, method='max_sharpe'):
        """Enhanced efficient frontier plot with multiple optimization points"""
        
        try:
            cla = CLA(mu, S)
            ef_points = cla.efficient_frontier(points=100)
            
            # Calculate individual assets
            individual_returns = mu * 252  # Annualized
            individual_vols = np.sqrt(np.diag(S) * 252)
            
            # Create subplots: EF, Risk Contributions, and Asset Cloud
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Efficient Frontier & Optimized Portfolios',
                              'Risk Contribution Analysis',
                              'Asset Risk-Return Characteristics',
                              'Sharpe Ratio Across Frontier'),
                specs=[[{"colspan": 2}, None], [{}, {}]],
                vertical_spacing=0.12,
                horizontal_spacing=0.15
            )
            
            # 1. Efficient Frontier (main plot)
            frontier_vols = [point[1] * np.sqrt(252) for point in ef_points]
            frontier_rets = [point[0] * 252 for point in ef_points]
            
            # Calculate Sharpe Ratios along frontier
            frontier_sharpe = [(ret - self.risk_free_rate) / vol if vol > 0 else 0 
                             for ret, vol in zip(frontier_rets, frontier_vols)]
            
            fig.add_trace(
                go.Scatter(
                    x=frontier_vols,
                    y=frontier_rets,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='#3B82F6', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.1)',
                    hovertemplate="Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{customdata:.3f}",
                    customdata=frontier_sharpe
                ),
                row=1, col=1
            )
            
            # Optimization strategies to plot
            strategies = [
                ('Max Sharpe', 'max_sharpe'),
                ('Min Volatility', 'min_volatility'),
                ('Equal Weight', 'equal_weight'),
                ('Risk Parity', 'risk_parity')
            ]
            
            colors = ['#10B981', '#EF4444', '#F59E0B', '#8B5CF6']
            
            for i, (label, method_name) in enumerate(strategies):
                try:
                    weights_df, performance, metrics = self.optimize_portfolio(
                        method_name, mu, S, returns, self.risk_free_rate
                    )
                    ret, vol, sharpe = performance
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[vol],
                            y=[ret],
                            mode='markers+text',
                            name=label,
                            marker=dict(
                                size=15,
                                color=colors[i],
                                line=dict(width=2, color='white')
                            ),
                            text=[label],
                            textposition="top center",
                            hovertemplate=(
                                f"<b>{label}</b><br>"
                                f"Return: {ret:.2%}<br>"
                                f"Volatility: {vol:.2%}<br>"
                                f"Sharpe: {sharpe:.3f}<br>"
                                f"Max DD: {metrics.get('Max Drawdown', 0):.2%}"
                            )
                        ),
                        row=1, col=1
                    )
                except:
                    continue
            
            # Individual stocks
            fig.add_trace(
                go.Scatter(
                    x=individual_vols,
                    y=individual_returns,
                    mode='markers',
                    name='Individual Assets',
                    marker=dict(
                        size=8,
                        color='#6B7280',
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    text=returns.columns,
                    hovertemplate="<b>%{text}</b><br>Return: %{y:.2%}<br>Volatility: %{x:.2%}"
                ),
                row=1, col=1
            )
            
            # 2. Risk Contribution Analysis (top right)
            try:
                # Get Max Sharpe portfolio for risk decomposition
                weights_df, _, _ = self.optimize_portfolio('max_sharpe', mu, S, returns, self.risk_free_rate)
                weights = weights_df['Weight'].values
                
                # Calculate risk contributions
                portfolio_vol = np.sqrt(weights @ S @ weights.T)
                marginal_risk = (S @ weights.T) / portfolio_vol
                risk_contributions = weights * marginal_risk
                perc_contributions = risk_contributions / portfolio_vol
                
                # Top 10 contributors
                top_indices = np.argsort(perc_contributions)[-10:][::-1]
                top_tickers = returns.columns[top_indices]
                top_contributions = perc_contributions[top_indices]
                
                fig.add_trace(
                    go.Bar(
                        x=top_contributions,
                        y=top_tickers,
                        orientation='h',
                        name='Risk Contribution',
                        marker_color='#F59E0B',
                        hovertemplate="%{y}: %{x:.1%} of total risk"
                    ),
                row=1, col=2  # Note: This will be moved after fixing colspan
                )
            except:
                pass
            
            # 3. Asset Risk-Return (bottom left)
            # Add convex hull of assets
            from scipy.spatial import ConvexHull
            points = np.column_stack([individual_vols, individual_returns])
            if len(points) > 3:
                try:
                    hull = ConvexHull(points)
                    hull_vols = points[hull.vertices, 0]
                    hull_rets = points[hull.vertices, 1]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=hull_vols,
                            y=hull_rets,
                            mode='lines',
                            fill='toself',
                            fillcolor='rgba(168, 85, 247, 0.1)',
                            line=dict(color='#A855F7', dash='dash'),
                            name='Asset Convex Hull'
                        ),
                        row=2, col=1
                    )
                except:
                    pass
            
            # 4. Sharpe Ratio across frontier (bottom right)
            fig.add_trace(
                go.Scatter(
                    x=frontier_vols,
                    y=frontier_sharpe,
                    mode='lines',
                    name='Sharpe Ratio',
                    line=dict(color='#10B981', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.1)'
                ),
                row=2, col=2
            )
            
            # Find max Sharpe point
            max_sharpe_idx = np.argmax(frontier_sharpe)
            fig.add_trace(
                go.Scatter(
                    x=[frontier_vols[max_sharpe_idx]],
                    y=[frontier_sharpe[max_sharpe_idx]],
                    mode='markers',
                    name='Max Sharpe',
                    marker=dict(size=12, color='#10B981', line=dict(width=2, color='white')),
                    hovertemplate="Max Sharpe: %{y:.3f}<br>at Vol: %{x:.2%}"
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=900,
                showlegend=True,
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='gray',
                    borderwidth=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Annual Volatility", row=1, col=1)
            fig.update_yaxes(title_text="Annual Return", row=1, col=1)
            fig.update_xaxes(title_text="Risk Contribution", row=1, col=2)
            fig.update_yaxes(title_text="Asset", row=1, col=2)
            fig.update_xaxes(title_text="Annual Volatility", row=2, col=1)
            fig.update_yaxes(title_text="Annual Return", row=2, col=1)
            fig.update_xaxes(title_text="Annual Volatility", row=2, col=2)
            fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
            
            return fig
            
        except Exception as e:
            st.error(f"Error plotting efficient frontier: {str(e)}")
            return None

    def create_risk_report_dashboard(self, metrics, portfolio_returns, benchmark_returns):
        """Create comprehensive risk report dashboard"""
        
        # Create subplots for risk dashboard
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Cumulative Returns vs Benchmark',
                'Rolling Sharpe Ratio (6M Window)',
                'Drawdown Analysis',
                'Return Distribution',
                'QQ Plot (vs Normal)',
                'Rolling Volatility (1M Window)',
                'Autocorrelation of Returns',
                'Autocorrelation of Squared Returns',
                'VaR Breaches Timeline'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Cumulative Returns
        portfolio_cum = np.exp(portfolio_returns.cumsum())
        if 'XU100.IS' in benchmark_returns.columns:
            bench_cum = np.exp(benchmark_returns['XU100.IS'].reindex(portfolio_returns.index).fillna(0).cumsum())
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio_cum.index,
                    y=portfolio_cum.values,
                    name='Portfolio',
                    line=dict(color='#3B82F6', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=bench_cum.index,
                    y=bench_cum.values,
                    name='XU100',
                    line=dict(color='#6B7280', width=2)
                ),
                row=1, col=1
            )
        
        # 2. Rolling Sharpe Ratio
        rolling_window = 126  # 6 months
        if len(portfolio_returns) > rolling_window:
            rolling_sharpe = portfolio_returns.rolling(rolling_window).apply(
                lambda x: (x.mean() * 252 - self.risk_free_rate) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
            )
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    name='Rolling Sharpe',
                    line=dict(color='#10B981', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.1)'
                ),
                row=1, col=2
            )
        
        # 3. Drawdown Analysis
        fig.add_trace(
            go.Scatter(
                x=metrics.get('drawdown_series', pd.Series()).index,
                y=metrics.get('drawdown_series', pd.Series()).values * 100,
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
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add normal distribution overlay
        x_norm = np.linspace(portfolio_returns.min() * 100, portfolio_returns.max() * 100, 100)
        y_norm = stats.norm.pdf(x_norm, portfolio_returns.mean() * 100, portfolio_returns.std() * 100)
        
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                name='Normal Dist',
                line=dict(color='#EF4444', dash='dash')
            ),
            row=2, col=1
        )
        
        # 5. QQ Plot
        if len(portfolio_returns) > 10:
            import statsmodels.api as sm
            qq_data = sm.qqplot(portfolio_returns, stats.norm, fit=True, line='45')
            qq_theory = qq_data.theory_quantiles
            qq_sample = qq_data.sample_quantiles
            
            fig.add_trace(
                go.Scatter(
                    x=qq_theory,
                    y=qq_sample,
                    mode='markers',
                    name='QQ Points',
                    marker=dict(size=4, color='#3B82F6')
                ),
                row=2, col=2
            )
            
            # Add 45-degree line
            line_range = [min(qq_theory.min(), qq_sample.min()), 
                         max(qq_theory.max(), qq_sample.max())]
            fig.add_trace(
                go.Scatter(
                    x=line_range,
                    y=line_range,
                    mode='lines',
                    name='45° Line',
                    line=dict(color='#EF4444', dash='dash')
                ),
                row=2, col=2
            )
        
        # 6. Rolling Volatility
        rolling_vol = portfolio_returns.rolling(21).std() * np.sqrt(252) * 100
        
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name='Volatility',
                line=dict(color='#F59E0B', width=2)
            ),
            row=2, col=3
        )
        
        # 7. Autocorrelation of Returns
        max_lag = min(40, len(portfolio_returns) // 2)
        if max_lag > 5:
            acf = [portfolio_returns.autocorr(lag=i) for i in range(1, max_lag + 1)]
            
            fig.add_trace(
                go.Bar(
                    x=list(range(1, max_lag + 1)),
                    y=acf,
                    name='ACF Returns',
                    marker_color='#8B5CF6'
                ),
                row=3, col=1
            )
        
        # 8. Autocorrelation of Squared Returns (volatility clustering)
        if max_lag > 5:
            acf_sq = [(portfolio_returns**2).autocorr(lag=i) for i in range(1, max_lag + 1)]
            
            fig.add_trace(
                go.Bar(
                    x=list(range(1, max_lag + 1)),
                    y=acf_sq,
                    name='ACF Squared Returns',
                    marker_color='#EC4899'
                ),
                row=3, col=2
            )
        
        # 9. VaR Breaches
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
        
        # Add VaR line
        fig.add_trace(
            go.Scatter(
                x=[portfolio_returns.index[0], portfolio_returns.index[-1]],
                y=[var_threshold * 100, var_threshold * 100],
                mode='lines',
                name=f'VaR ({var_level*100:.0f}%)',
                line=dict(color='#EF4444', dash='dash')
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes titles
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=3)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=3)
        fig.update_yaxes(title_text="Autocorrelation", row=3, col=1)
        fig.update_yaxes(title_text="Autocorrelation", row=3, col=2)
        fig.update_yaxes(title_text="Return (%)", row=3, col=3)
        
        return fig

# ============================================================================
# 3. ENHANCED GARCH AND RISK FUNCTIONS
# ============================================================================

def calculate_enhanced_garch(returns_series, p=1, q=1):
    """Fit GARCH(p,q) model with enhanced diagnostics"""
    if not HAS_ARCH or len(returns_series) < 100:
        return None, None, None
    
    try:
        # Scale returns for numerical stability
        returns_scaled = returns_series * 100
        
        # Fit GARCH model
        model = arch_model(returns_scaled.dropna(), vol='Garch', p=p, q=q, dist='skewt')
        res = model.fit(disp='off', show_warning=False)
        
        # Conditional volatility (unscaled)
        conditional_volatility = res.conditional_volatility / 100
        
        # Extract parameters with confidence intervals
        params = res.params
        std_err = res.std_err
        tvalues = res.tvalues
        pvalues = res.pvalues
        
        # Calculate persistence and half-life
        persistence = sum([params.get(f'alpha[{i}]', 0) for i in range(1, p+1)]) + \
                     sum([params.get(f'beta[{i}]', 0) for i in range(1, q+1)])
        
        half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf
        
        # Forecast next periods
        forecast_horizons = [1, 5, 20]  # 1 day, 1 week, 1 month
        forecasts = {}
        
        for horizon in forecast_horizons:
            forecast_var = res.forecast(horizon=horizon)
            forecasts[horizon] = {
                'variance': float(forecast_var.variance.iloc[-1, -1]) / 10000,  # Unscale
                'volatility': np.sqrt(float(forecast_var.variance.iloc[-1, -1])) / 100,
                'annualized_vol': np.sqrt(float(forecast_var.variance.iloc[-1, -1]) * 252) / 100
            }
        
        # Model diagnostics
        residuals = res.resid / res.conditional_volatility  # Standardized residuals
        
        garch_params = {
            'Log Likelihood': res.loglikelihood,
            'AIC': res.aic,
            'BIC': res.bic,
            'Persistence': persistence,
            'Half-Life (days)': half_life,
            'Long Run Variance': params.get('omega', 0) / (1 - persistence) if persistence < 1 else np.inf,
            'Skewness Parameter': params.get('lambda', None),  # For skewed t-distribution
            'Degrees of Freedom': params.get('nu', None),  # For t-distribution
        }
        
        # Add forecast information
        for horizon, forecast in forecasts.items():
            garch_params[f'{horizon}-Day Forecast Vol'] = forecast['volatility']
            garch_params[f'{horizon}-Day Annualized Vol'] = forecast['annualized_vol']
        
        # Parameter statistics
        garch_stats = {}
        for param in params.index:
            garch_stats[param] = {
                'Estimate': params[param],
                'Std. Error': std_err.get(param, 0),
                't-Statistic': tvalues.get(param, 0),
                'p-Value': pvalues.get(param, 1)
            }
        
        return garch_params, conditional_volatility, garch_stats, residuals
        
    except Exception as e:
        st.warning(f"GARCH model fitting failed: {str(e)}")
        return None, None, None, None

def calculate_copula_var(portfolio_returns, benchmark_returns, confidence_levels=[0.95, 0.99]):
    """Calculate VaR using copula methods for dependency modeling"""
    
    var_results = {}
    
    for confidence in confidence_levels:
        # Historical VaR
        hist_var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        
        # Gaussian VaR (parametric)
        gaussian_var = norm.ppf(1 - confidence, portfolio_returns.mean(), portfolio_returns.std())
        
        # Student-t VaR (fat tails)
        df, loc, scale = stats.t.fit(portfolio_returns)
        t_var = stats.t.ppf(1 - confidence, df, loc, scale)
        
        # Cornish-Fisher VaR (adjusts for skewness and kurtosis)
        z = norm.ppf(1 - confidence)
        s = skew(portfolio_returns)
        k = kurtosis(portfolio_returns)
        cf_z = z + (z**2 - 1) * s/6 + (z**3 - 3*z) * k/24 - (2*z**3 - 5*z) * s**2/36
        cf_var = portfolio_returns.mean() + cf_z * portfolio_returns.std()
        
        var_results[confidence] = {
            'Historical': hist_var,
            'Gaussian': gaussian_var,
            'Student-t': t_var,
            'Cornish-Fisher': cf_var,
            'Expected Shortfall': portfolio_returns[portfolio_returns <= hist_var].mean()
        }
    
    return var_results

def perform_stress_test(portfolio_returns, scenarios):
    """Perform stress testing under various market scenarios"""
    
    results = {}
    
    # Historical stress periods (example)
    stress_periods = {
        'COVID-19 Crash': ('2020-02-20', '2020-03-23'),
        'Inflation Spike': ('2022-01-01', '2022-12-31'),
        'Recent Volatility': (datetime.now() - timedelta(days=90), datetime.now())
    }
    
    for scenario, (start, end) in stress_periods.items():
        if start in portfolio_returns.index and end in portfolio_returns.index:
            mask = (portfolio_returns.index >= start) & (portfolio_returns.index <= end)
            stress_returns = portfolio_returns[mask]
            
            if len(stress_returns) > 0:
                results[scenario] = {
                    'Start': start,
                    'End': end,
                    'Days': len(stress_returns),
                    'Total Return': np.exp(stress_returns.sum()) - 1,
                    'Worst Day': stress_returns.min(),
                    'Best Day': stress_returns.max(),
                    'Volatility': stress_returns.std() * np.sqrt(252),
                    'Max Drawdown': calculate_max_drawdown(stress_returns)
                }
    
    # Add hypothetical scenarios
    hypothetical = {
        '10% Market Drop': -0.10,
        '5% Market Drop': -0.05,
        'High Volatility (50% annual)': 0.50 / np.sqrt(252),
        'Interest Rate Shock (+5%)': -0.15  # Estimated impact
    }
    
    for scenario, impact in hypothetical.items():
        results[scenario] = {
            'Hypothetical Impact': impact,
            'Portfolio Impact': impact * 0.7  # Assuming 0.7 beta
        }
    
    return results

def calculate_max_drawdown(returns_series):
    """Calculate maximum drawdown from return series"""
    cum_returns = np.exp(returns_series.cumsum())
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

# ============================================================================
# 4. ENHANCED STREAMLIT APPLICATION WITH TAB LAYOUT
# ============================================================================

def main_streamlit_app():
    st.markdown('<div class="main-header">🇹🇷 BIST Portfolio Risk & Optimization Terminal | Institutional Platform</div>', unsafe_allow_html=True)
    st.markdown("*Advanced Portfolio Analytics for Turkish Equity Markets*")
    
    # Initialize optimizer
    optimizer = TurkishPortfolioOptimizer()
    
    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")
        
        # Date selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=365*2),
                help="Start date for historical data"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.now(),
                help="End date for historical data"
            )
        
        st.markdown("---")
        st.markdown("### 📊 Optimization Parameters")
        
        # Risk-free rate
        risk_free_rate = st.slider(
            "Annual Risk-Free Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=RISK_FREE_RATE * 100,
            step=0.5,
            help="Turkish risk-free rate (approximated by government bond yields)"
        ) / 100
        
        # Optimization strategy
        strategy = st.selectbox(
            "Optimization Strategy",
            ['max_sharpe', 'min_volatility', 'efficient_risk', 
             'efficient_return', 'max_quadratic_utility', 
             'hrp', 'cvar', 'equal_weight', 'risk_parity'],
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Select portfolio optimization methodology"
        )
        
        # Strategy-specific parameters
        if strategy in ['efficient_risk', 'efficient_return']:
            target_value = st.slider(
                f"Target {'Risk' if strategy == 'efficient_risk' else 'Return'} (%)",
                min_value=5.0,
                max_value=80.0,
                value=30.0 if strategy == 'efficient_risk' else 25.0,
                step=1.0
            ) / 100
        
        if strategy == 'max_quadratic_utility':
            risk_aversion = st.slider(
                "Risk Aversion Coefficient",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Higher values indicate greater risk aversion"
            )
        
        # Weight constraints
        st.markdown("---")
        st.markdown("### ⚖️ Portfolio Constraints")
        
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
        
        constraints = {'min_weight': min_weight, 'max_weight': max_weight}
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            use_log_returns = st.checkbox("Use Log Returns", value=True)
            cov_estimator = st.selectbox(
                "Covariance Estimator",
                ['sample_cov', 'semicovariance', 'exp_cov'],
                help="Method for estimating covariance matrix"
            )
            
            if cov_estimator == 'exp_cov':
                span = st.slider("Exponential Decay Span", min_value=30, max_value=500, value=180)
        
        st.markdown("---")
        
        # Action button
        analyze_button = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if not analyze_button:
        st.info("👈 Configure parameters in the sidebar and click 'Run Analysis' to begin.")
        
        # Display informational cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("BIST 30 Stocks", "30", "")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Optimization Methods", "9", "")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            with st.container():
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Risk Metrics", "25+", "")
                st.markdown('</div>', unsafe_allow_html=True)
        
        return
    
    # --- DATA FETCHING & PROCESSING ---
    with st.spinner("📥 Fetching market data..."):
        try:
            data, returns, benchmark_data, benchmark_returns = optimizer.fetch_data(
                start_date=str(start_date),
                end_date=str(end_date)
            )
            
            if data is None or returns is None:
                st.error("Failed to fetch data. Please check your internet connection and try again.")
                return
            
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(data)
            
            if cov_estimator == 'sample_cov':
                S = risk_models.sample_cov(data)
            elif cov_estimator == 'semicovariance':
                S = risk_models.semicovariance(data)
            elif cov_estimator == 'exp_cov':
                S = risk_models.exp_cov(data, span=span)
            
            # Perform optimization
            target_return = target_value if 'target_value' in locals() else None
            risk_aversion = risk_aversion if 'risk_aversion' in locals() else 1.0
            
            weights_df, performance, metrics = optimizer.optimize_portfolio(
                strategy, mu, S, returns, risk_free_rate, 
                target_return, risk_aversion, constraints
            )
            
            # Recalculate full metrics
            weights_series = pd.Series(weights_df['Weight']).reindex(returns.columns).fillna(0)
            metrics, portfolio_returns, cum_returns, drawdown = optimizer.calculate_enhanced_metrics(
                weights_series, returns, benchmark_returns, risk_free_rate
            )
            
            # Store drawdown series in metrics
            metrics['drawdown_series'] = drawdown
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.exception(e)
            return
    
    st.success("✅ Analysis completed successfully!")
    
    # --- TABBED INTERFACE ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio Overview",
        "📈 Optimization Analysis",
        "⚠️ Risk Analytics",
        "📋 Performance Report",
        "🔍 Advanced Diagnostics"
    ])
    
    # TAB 1: Portfolio Overview
    with tab1:
        st.markdown('<div class="sub-header">Portfolio Overview & Composition</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Optimized Portfolio",
                f"{metrics['Annual Return']:.2%}",
                f"Sharpe: {metrics['Sharpe Ratio']:.2f}"
            )
        
        with col2:
            st.metric(
                "Portfolio Volatility",
                f"{metrics['Annual Volatility']:.2%}",
                f"Max DD: {metrics['Max Drawdown']:.2%}"
            )
        
        with col3:
            st.metric(
                "Risk-Adjusted Return",
                f"{metrics['Sortino Ratio']:.2f}",
                f"Omega: {metrics['Omega Ratio']:.2f}"
            )
        
        with col4:
            st.metric(
                "Benchmark (XU100)",
                f"{benchmark_returns['XU100.IS'].mean() * 252:.2%}" if 'XU100.IS' in benchmark_returns.columns else "N/A",
                f"Beta: {metrics['Beta']:.2f}"
            )
        
        # Portfolio composition
        st.markdown("### Portfolio Composition")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Weight distribution pie chart
            fig_weights = go.Figure(data=[
                go.Pie(
                    labels=weights_df.index,
                    values=weights_df['Weight'],
                    hole=0.4,
                    textinfo='label+percent',
                    marker=dict(colors=px.colors.qualitative.Set3)
                )
            ])
            
            fig_weights.update_layout(
                title="Portfolio Weight Distribution",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_weights, use_container_width=True)
        
        with col2:
            # Top holdings table
            st.dataframe(
                weights_df.head(10).style.format({'Weight': '{:.2%}'})
                .background_gradient(subset=['Weight'], cmap='Blues'),
                use_container_width=True,
                height=400
            )
        
        # Performance chart
        st.markdown("### Performance Timeline")
        
        fig_perf = go.Figure()
        
        # Portfolio cumulative returns
        fig_perf.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='#3B82F6', width=3)
        ))
        
        # Benchmark if available
        if 'XU100.IS' in benchmark_returns.columns:
            bench_cum = np.exp(benchmark_returns['XU100.IS'].reindex(cum_returns.index).fillna(0).cumsum())
            fig_perf.add_trace(go.Scatter(
                x=bench_cum.index,
                y=bench_cum.values,
                mode='lines',
                name='XU100 Index',
                line=dict(color='#6B7280', width=2, dash='dash')
            ))
        
        fig_perf.update_layout(
            title="Cumulative Performance vs Benchmark",
            yaxis_title="Cumulative Return",
            xaxis_title="Date",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
    
    # TAB 2: Optimization Analysis
    with tab2:
        st.markdown('<div class="sub-header">Portfolio Optimization Analysis</div>', unsafe_allow_html=True)
        
        # Efficient Frontier
        st.markdown("### Efficient Frontier Analysis")
        
        ef_fig = optimizer.plot_enhanced_efficient_frontier(mu, S, returns, strategy)
        if ef_fig:
            st.plotly_chart(ef_fig, use_container_width=True)
        
        # Optimization comparison
        st.markdown("### Optimization Method Comparison")
        
        strategies_to_compare = ['max_sharpe', 'min_volatility', 'equal_weight', 'risk_parity', 'hrp']
        comparison_results = []
        
        for strat in strategies_to_compare:
            try:
                w_df, perf, met = optimizer.optimize_portfolio(strat, mu, S, returns, risk_free_rate)
                comparison_results.append({
                    'Strategy': strat.replace('_', ' ').title(),
                    'Return': perf[0],
                    'Volatility': perf[1],
                    'Sharpe': perf[2],
                    'Max DD': met['Max Drawdown'],
                    'Sortino': met['Sortino Ratio'],
                    'Number of Assets': len(w_df)
                })
            except:
                continue
        
        if comparison_results:
            comp_df = pd.DataFrame(comparison_results)
            
            # Display comparison table
            st.dataframe(
                comp_df.style.format({
                    'Return': '{:.2%}',
                    'Volatility': '{:.2%}',
                    'Sharpe': '{:.2f}',
                    'Max DD': '{:.2%}',
                    'Sortino': '{:.2f}'
                }).background_gradient(subset=['Sharpe', 'Sortino'], cmap='RdYlGn')
                .background_gradient(subset=['Volatility', 'Max DD'], cmap='RdYlGn_r'),
                use_container_width=True
            )
            
            # Radar chart for strategy comparison
            categories = ['Return', 'Volatility', 'Sharpe', 'Sortino']
            
            fig_radar = go.Figure()
            
            for idx, row in comp_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['Return']*100, row['Volatility']*100, row['Sharpe']*2, row['Sortino']*2],
                    theta=categories,
                    fill='toself',
                    name=row['Strategy']
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(comp_df['Return'].max()*150, comp_df['Volatility'].max()*150)]
                    )),
                showlegend=True,
                title="Strategy Comparison Radar Chart",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    
    # TAB 3: Risk Analytics
    with tab3:
        st.markdown('<div class="sub-header">Comprehensive Risk Analytics</div>', unsafe_allow_html=True)
        
        # Risk metrics dashboard
        st.markdown("### Risk Metrics Dashboard")
        
        # Create columns for risk metrics
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
        
        with risk_col1:
            st.metric("Value at Risk (95%)", f"{metrics['VaR (95% Historical)']:.2%}")
            st.metric("Conditional VaR", f"{metrics['CVaR (95%)']:.2%}")
        
        with risk_col2:
            st.metric("Tail Ratio", f"{metrics['Tail Ratio (99%/95%)']:.2f}")
            st.metric("Gain/Loss Ratio", f"{metrics['Gain/Loss Ratio']:.2f}")
        
        with risk_col3:
            st.metric("Skewness", f"{metrics['Skewness']:.2f}")
            st.metric("Kurtosis", f"{metrics['Kurtosis']:.2f}")
        
        with risk_col4:
            st.metric("Jarque-Bera Stat", f"{metrics['Jarque-Bera Stat']:.2f}")
            pval = metrics['Jarque-Bera p-value']
            sig = "🚨" if pval < 0.05 else "✅"
            st.metric("JB p-value", f"{pval:.4f}", sig)
        
        # Risk report dashboard
        st.markdown("### Comprehensive Risk Report")
        
        risk_fig = optimizer.create_risk_report_dashboard(metrics, portfolio_returns, benchmark_returns)
        if risk_fig:
            st.plotly_chart(risk_fig, use_container_width=True)
        
        # GARCH Analysis
        st.markdown("### Volatility Forecasting (GARCH)")
        
        garch_params, conditional_vol, garch_stats, residuals = calculate_enhanced_garch(portfolio_returns)
        
        if garch_params:
            col1, col2 = st.columns(2)
            
            with col1:
                # GARCH parameters table
                st.markdown("**GARCH Model Parameters**")
                garch_df = pd.DataFrame.from_dict(garch_stats, orient='index')
                st.dataframe(
                    garch_df.style.format({
                        'Estimate': '{:.6f}',
                        'Std. Error': '{:.6f}',
                        't-Statistic': '{:.2f}',
                        'p-Value': '{:.4f}'
                    }),
                    use_container_width=True
                )
            
            with col2:
                # GARCH diagnostics
                st.markdown("**Model Diagnostics**")
                diag_data = {
                    'Metric': ['Log Likelihood', 'AIC', 'BIC', 'Persistence', 'Half-Life (days)'],
                    'Value': [
                        garch_params['Log Likelihood'],
                        garch_params['AIC'],
                        garch_params['BIC'],
                        garch_params['Persistence'],
                        garch_params['Half-Life (days)']
                    ]
                }
                diag_df = pd.DataFrame(diag_data)
                st.dataframe(diag_df, use_container_width=True)
            
            # Conditional volatility plot
            st.markdown("**Conditional Volatility Forecast**")
            
            fig_garch = go.Figure()
            
            # Historical volatility
            fig_garch.add_trace(go.Scatter(
                x=conditional_vol.index,
                y=conditional_vol.values * np.sqrt(252) * 100,
                mode='lines',
                name='GARCH Volatility',
                line=dict(color='#EF4444', width=2)
            ))
            
            # Rolling volatility
            rolling_vol = portfolio_returns.rolling(21).std() * np.sqrt(252) * 100
            fig_garch.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='21-Day Rolling Vol',
                line=dict(color='#3B82F6', width=1, dash='dash')
            ))
            
            fig_garch.update_layout(
                title="Conditional Volatility (Annualized %)",
                yaxis_title="Volatility (%)",
                xaxis_title="Date",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_garch, use_container_width=True)
        
        # Stress Testing
        st.markdown("### Stress Testing Scenarios")
        
        stress_results = perform_stress_test(portfolio_returns, {})
        
        if stress_results:
            stress_df = pd.DataFrame.from_dict(stress_results, orient='index')
            st.dataframe(
                stress_df.style.format({
                    'Total Return': '{:.2%}',
                    'Worst Day': '{:.2%}',
                    'Best Day': '{:.2%}',
                    'Volatility': '{:.2%}',
                    'Max Drawdown': '{:.2%}'
                }).background_gradient(subset=['Total Return', 'Max Drawdown'], cmap='RdYlGn_r'),
                use_container_width=True
            )
    
    # TAB 4: Performance Report
    with tab4:
        st.markdown('<div class="sub-header">Comprehensive Performance Report</div>', unsafe_allow_html=True)
        
        # Generate detailed performance report
        st.markdown("### Performance Metrics Summary")
        
        # Categorize metrics
        return_metrics = {
            'Annual Return': metrics['Annual Return'],
            'Cumulative Return': metrics['Cumulative Return'],
            'Active Return': metrics['Active Return'],
            'Alpha': metrics['Alpha']
        }
        
        risk_metrics = {
            'Annual Volatility': metrics['Annual Volatility'],
            'Max Drawdown': metrics['Max Drawdown'],
            'VaR (95% Historical)': metrics['VaR (95% Historical)'],
            'CVaR (95%)': metrics['CVaR (95%)'],
            'Tracking Error': metrics['Tracking Error']
        }
        
        risk_adjusted_metrics = {
            'Sharpe Ratio': metrics['Sharpe Ratio'],
            'Sortino Ratio': metrics['Sortino Ratio'],
            'Information Ratio': metrics['Information Ratio'],
            'Calmar Ratio': metrics['Calmar Ratio'],
            'Treynor Ratio': metrics['Treynor Ratio'],
            'Omega Ratio': metrics['Omega Ratio']
        }
        
        statistical_metrics = {
            'Skewness': metrics['Skewness'],
            'Kurtosis': metrics['Kurtosis'],
            'Win Rate': metrics['Win Rate'],
            'Avg Win / Avg Loss': metrics['Avg Win / Avg Loss']
        }
        
        # Display metrics in expandable sections
        with st.expander("📈 Return Metrics", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            metrics_cols = [col1, col2, col3, col4]
            
            for idx, (metric, value) in enumerate(return_metrics.items()):
                with metrics_cols[idx % 4]:
                    st.metric(
                        metric,
                        f"{value:.2%}" if 'Return' in metric or metric == 'Alpha' else f"{value:.4f}"
                    )
        
        with st.expander("⚠️ Risk Metrics"):
            col1, col2, col3, col4, col5 = st.columns(5)
            risk_cols = [col1, col2, col3, col4, col5]
            
            for idx, (metric, value) in enumerate(risk_metrics.items()):
                with risk_cols[idx % 5]:
                    st.metric(metric, f"{value:.2%}")
        
        with st.expander("📊 Risk-Adjusted Return Metrics"):
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            adj_cols = [col1, col2, col3, col4, col5, col6]
            
            for idx, (metric, value) in enumerate(risk_adjusted_metrics.items()):
                with adj_cols[idx % 6]:
                    st.metric(metric, f"{value:.3f}")
        
        with st.expander("📐 Statistical Properties"):
            col1, col2, col3, col4 = st.columns(4)
            stat_cols = [col1, col2, col3, col4]
            
            for idx, (metric, value) in enumerate(statistical_metrics.items()):
                with stat_cols[idx % 4]:
                    st.metric(metric, f"{value:.3f}")
        
        # Performance attribution
        st.markdown("### Performance Attribution")
        
        if 'XU100.IS' in benchmark_returns.columns:
            # Calculate active returns decomposition
            active_returns = portfolio_returns - benchmark_returns['XU100.IS'].reindex(portfolio_returns.index).fillna(0)
            
            fig_attribution = go.Figure()
            
            fig_attribution.add_trace(go.Bar(
                x=active_returns.resample('M').sum().index.strftime('%Y-%m'),
                y=active_returns.resample('M').sum().values * 100,
                name='Monthly Active Return',
                marker_color='#3B82F6'
            ))
            
            fig_attribution.update_layout(
                title="Monthly Active Return vs Benchmark",
                yaxis_title="Active Return (%)",
                xaxis_title="Month",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_attribution, use_container_width=True)
        
        # Download report
        st.markdown("### Export Report")
        
        # Create downloadable DataFrame
        report_data = {
            **return_metrics,
            **risk_metrics,
            **risk_adjusted_metrics,
            **statistical_metrics
        }
        
        report_df = pd.DataFrame.from_dict(report_data, orient='index', columns=['Value'])
        
        # Convert to CSV
        csv = report_df.to_csv()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Performance Report (CSV)",
                data=csv,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="📥 Download Portfolio Weights (CSV)",
                data=weights_df.to_csv(),
                file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # TAB 5: Advanced Diagnostics
    with tab5:
        st.markdown('<div class="sub-header">Advanced Quantitative Diagnostics</div>', unsafe_allow_html=True)
        
        # Market regime detection
        st.markdown("### Market Regime Analysis")
        
        # Calculate volatility regimes
        volatility_regimes = pd.qcut(portfolio_returns.rolling(21).std().dropna(), q=3, labels=['Low', 'Medium', 'High'])
        
        fig_regime = go.Figure()
        
        # Color points by regime
        colors = {'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'}
        
        for regime in ['Low', 'Medium', 'High']:
            mask = volatility_regimes == regime
            if mask.any():
                fig_regime.add_trace(go.Scatter(
                    x=portfolio_returns[mask].index,
                    y=portfolio_returns[mask].values * 100,
                    mode='markers',
                    name=f'{regime} Volatility Regime',
                    marker=dict(size=6, color=colors[regime]),
                    opacity=0.6
                ))
        
        fig_regime.update_layout(
            title="Market Regime Detection (by Volatility)",
            yaxis_title="Daily Return (%)",
            xaxis_title="Date",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_regime, use_container_width=True)
        
        # Correlation analysis
        st.markdown("### Correlation Structure Analysis")
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation")
        ))
        
        fig_corr.update_layout(
            title="Asset Correlation Matrix",
            height=600,
            xaxis_title="Assets",
            yaxis_title="Assets"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Portfolio sensitivity analysis
        st.markdown("### Portfolio Sensitivity Analysis")
        
        # Vary risk-free rate and see impact
        rf_range = np.linspace(risk_free_rate * 0.5, risk_free_rate * 1.5, 20)
        sharpe_values = []
        
        for rf in rf_range:
            # Recalculate Sharpe with different RF
            sharpe = (metrics['Annual Return'] - rf) / metrics['Annual Volatility'] if metrics['Annual Volatility'] > 0 else 0
            sharpe_values.append(sharpe)
        
        fig_sensitivity = go.Figure()
        
        fig_sensitivity.add_trace(go.Scatter(
            x=rf_range * 100,
            y=sharpe_values,
            mode='lines+markers',
            name='Sharpe Ratio',
            line=dict(color='#3B82F6', width=3)
        ))
        
        fig_sensitivity.add_vline(
            x=risk_free_rate * 100,
            line_dash="dash",
            line_color="red",
            annotation_text="Current RF",
            annotation_position="top right"
        )
        
        fig_sensitivity.update_layout(
            title="Sensitivity to Risk-Free Rate",
            xaxis_title="Risk-Free Rate (%)",
            yaxis_title="Sharpe Ratio",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        # Monte Carlo simulation placeholder
        st.markdown("### Monte Carlo Simulation")
        
        if st.button("Run Monte Carlo Simulation", type="secondary"):
            with st.spinner("Running 10,000 Monte Carlo simulations..."):
                # Simple Monte Carlo simulation
                n_simulations = 10000
                n_days = 252  # 1 year
                
                # Parameters
                mu_daily = portfolio_returns.mean()
                sigma_daily = portfolio_returns.std()
                
                # Generate random walks
                simulations = np.zeros((n_simulations, n_days))
                
                for i in range(n_simulations):
                    random_returns = np.random.normal(mu_daily, sigma_daily, n_days)
                    simulations[i] = np.exp(random_returns.cumsum())
                
                # Calculate statistics
                final_values = simulations[:, -1]
                
                fig_mc = go.Figure()
                
                # Histogram of final values
                fig_mc.add_trace(go.Histogram(
                    x=final_values,
                    nbinsx=50,
                    name='Final Values',
                    marker_color='#3B82F6',
                    opacity=0.7
                ))
                
                # Add vertical lines for statistics
                fig_mc.add_vline(
                    x=np.percentile(final_values, 5),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="5% VaR",
                    annotation_position="top right"
                )
                
                fig_mc.add_vline(
                    x=np.percentile(final_values, 95),
                    line_dash="dash",
                    line_color="green",
                    annotation_text="95% VaR",
                    annotation_position="top right"
                )
                
                fig_mc.update_layout(
                    title="Monte Carlo Simulation Results (1-Year Horizon)",
                    xaxis_title="Portfolio Value",
                    yaxis_title="Frequency",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # Display statistics
                mc_stats = {
                    'Mean Final Value': np.mean(final_values),
                    'Median Final Value': np.median(final_values),
                    'Std Final Value': np.std(final_values),
                    '5% Percentile': np.percentile(final_values, 5),
                    '95% Percentile': np.percentile(final_values, 95),
                    'Probability of Loss': np.mean(final_values < 1)
                }
                
                mc_df = pd.DataFrame.from_dict(mc_stats, orient='index', columns=['Value'])
                st.dataframe(mc_df.style.format('{:.4f}'), use_container_width=True)

# ============================================================================
# 5. APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main_streamlit_app()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.exception(e)
