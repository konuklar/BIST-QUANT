            3. **Industrial:** EREGL.IS, FROTO.IS, TOASO.IS, KCHOL.IS
            4. **Technology:** THYAO.IS, TCELL.IS, TTKOM.IS
            
            **Tips:**
            - Start with 5-10 liquid stocks
            - Select from different sectors for diversification
            - Use recent data (2-3 years) for stable estimates
            """)
        return
    
    # Data Loading Section
    if not st.session_state.data_loaded or run_optimization:
        with st.spinner("🔄 Loading market data..."):
            data_source = EnhancedDataSource()
            
            # Load asset data
            st.info(f"📥 Fetching data for {len(assets)} assets...")
            asset_data = data_source.fetch_enhanced_data(
                assets, 
                start_date, 
                end_date
            )
            
            if asset_data is None or asset_data['close'].empty:
                st.error("Failed to load data. Please check your internet connection and try again.")
                return
            
            # Load benchmark data
            benchmark_ticker = BENCHMARKS[benchmark_symbol]
            st.info(f"📊 Loading benchmark data: {benchmark_ticker}")
            benchmark_data = data_source.fetch_enhanced_data(
                [benchmark_ticker], 
                start_date, 
                end_date
            )
            
            if benchmark_data is None:
                st.warning(f"Could not load benchmark data for {benchmark_ticker}")
                benchmark_returns = None
            else:
                benchmark_returns = benchmark_data['returns'].iloc[:, 0]
            
            # Align dates
            if benchmark_returns is not None:
                common_idx = asset_data['returns'].index.intersection(benchmark_returns.index)
                asset_data['returns'] = asset_data['returns'].loc[common_idx]
                benchmark_returns = benchmark_returns.loc[common_idx]
            
            st.session_state.asset_data = asset_data
            st.session_state.benchmark_returns = benchmark_returns
            st.session_state.data_loaded = True
            
            st.success(f"✅ Data loaded successfully! ({len(asset_data['returns'])} trading days)")
    
    # Main Analysis Tabs
    if st.session_state.data_loaded:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Dashboard", "⚖️ Optimization", "📊 Risk Analytics", 
            "📋 Portfolio Report", "🔄 Rebalancing", "🛠️ Tools"
        ])
        
        with tab1:
            display_dashboard(st.session_state.asset_data)
        
        with tab2:
            run_portfolio_optimization(
                st.session_state.asset_data,
                benchmark_symbol,
                optimization_method,
                risk_model,
                return_model,
                target_volatility if 'target_volatility' in locals() else None,
                target_return if 'target_return' in locals() else None
            )
        
        if st.session_state.optimization_complete:
            with tab3:
                display_risk_analytics()
            
            with tab4:
                display_portfolio_report()
            
            with tab5:
                display_rebalancing_tools()
            
            with tab6:
                display_additional_tools()

def display_dashboard(asset_data):
    """Display comprehensive dashboard"""
    st.header("📈 Market Dashboard")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        returns = asset_data['returns']
        total_returns = (1 + returns).prod() - 1
        avg_return = total_returns.mean() * 100
        st.metric("Avg. Return", f"{avg_return:.2f}%")
    
    with col2:
        volatility = returns.std() * np.sqrt(252) * 100
        avg_vol = volatility.mean()
        st.metric("Avg. Volatility", f"{avg_vol:.2f}%")
    
    with col3:
        sharpe_ratios = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        avg_sharpe = sharpe_ratios.mean()
        st.metric("Avg. Sharpe", f"{avg_sharpe:.2f}")
    
    with col4:
        correlation_matrix = returns.corr()
        avg_corr = correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)].mean()
        st.metric("Avg. Correlation", f"{avg_corr:.2f}")
    
    # Charts Section
    st.subheader("📊 Price Evolution")
    
    fig = go.Figure()
    prices_normalized = asset_data['close'] / asset_data['close'].iloc[0]
    
    for ticker in prices_normalized.columns:
        fig.add_trace(go.Scatter(
            x=prices_normalized.index,
            y=prices_normalized[ticker],
            name=ticker,
            line=dict(width=1.5)
        ))
    
    fig.update_layout(
        height=400,
        template='plotly_dark',
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Normalized Price (Base=100)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("🔥 Correlation Matrix")
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig_corr.update_layout(
        height=500,
        title="Asset Correlation Matrix",
        xaxis_title="",
        yaxis_title="",
        template='plotly_dark'
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Recent Returns Table
    st.subheader("📋 Recent Performance")
    
    recent_stats = pd.DataFrame({
        '1M Return': ((1 + asset_data['returns'].tail(21)).prod() - 1) * 100,
        '3M Return': ((1 + asset_data['returns'].tail(63)).prod() - 1) * 100,
        '6M Return': ((1 + asset_data['returns'].tail(126)).prod() - 1) * 100,
        'YTD Return': ((1 + asset_data['returns']).prod() - 1) * 100,
        'Annual Vol': asset_data['returns'].std() * np.sqrt(252) * 100
    })
    
    st.dataframe(
        recent_stats.style.format("{:.2f}%").background_gradient(cmap='RdYlGn', axis=0),
        use_container_width=True
    )

def run_portfolio_optimization(asset_data, benchmark_symbol, method, 
                              risk_model, return_model, target_vol=None, target_return=None):
    """Run portfolio optimization and display results"""
    st.header("⚖️ Portfolio Optimization")
    
    # Create optimizer
    optimizer = EnhancedPortfolioOptimizer(
        asset_data['close'], 
        asset_data['returns']
    )
    
    # Run optimization
    with st.spinner("🔧 Running optimization..."):
        try:
            weights, performance = optimizer.optimize_portfolio(
                method=method,
                risk_model=risk_model,
                return_model=return_model,
                target_volatility=target_vol,
                target_return=target_return,
                risk_free_rate=st.session_state.risk_free_rate
            )
            
            annual_return, annual_vol, sharpe_ratio = performance
            
            # Store results
            st.session_state.current_weights = weights
            st.session_state.performance = performance
            st.session_state.optimization_complete = True
            
        except Exception as e:
            st.error(f"Optimization failed: {str(e)[:200]}")
            logger.error(f"Optimization error: {e}", exc_info=True)
            return
    
    # Display results
    st.success("✅ Optimization completed!")
    
    # Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <small>Expected Annual Return</small><br>
            <h3 style="color: {'#00cc88' if annual_return >= 0 else '#ff4d4d'}">
                {annual_return:+.2%}
            </h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <small>Expected Annual Volatility</small><br>
            <h3>{annual_vol:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <small>Sharpe Ratio</small><br>
            <h3 style="color: {'#00cc88' if sharpe_ratio >= 0 else '#ff4d4d'}">
                {sharpe_ratio:.2f}
            </h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if hasattr(st.session_state, 'previous_weights') and st.session_state.previous_weights:
            turnover = sum(abs(weights.get(ticker, 0) - 
                             st.session_state.previous_weights.get(ticker, 0)) 
                         for ticker in set(weights.keys()) | 
                         set(st.session_state.previous_weights.keys()))
            st.markdown(f"""
            <div class="metric-card">
                <small>Turnover vs Previous</small><br>
                <h3>{turnover:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <small>Portfolio Diversity</small><br>
                <h3>{len([w for w in weights.values() if w > 0.001])} Assets</h3>
            </div>
            """, unsafe_allow_html=True)
    
    # Portfolio Weights
    st.subheader("📊 Optimal Portfolio Allocation")
    
    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
    weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
    weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
    
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        # Pie chart for allocation
        fig_pie = go.Figure(data=[go.Pie(
            labels=weights_df.index,
            values=[float(w.strip('%'))/100 for w in weights_df['Weight']],
            hole=0.4,
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig_pie.update_layout(
            height=400,
            showlegend=False,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_table:
        st.dataframe(
            weights_df,
            use_container_width=True
        )
    
    # Efficient Frontier
    st.subheader("📈 Efficient Frontier")
    
    try:
        mus, sigmas, frontier_weights = optimizer.generate_efficient_frontier(points=50)
        
        fig_frontier = go.Figure()
        
        # Plot frontier
        if mus and sigmas:
            fig_frontier.add_trace(go.Scatter(
                x=sigmas,
                y=mus,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='#00cc88', width=3)
            ))
        
        # Plot current portfolio
        fig_frontier.add_trace(go.Scatter(
            x=[annual_vol],
            y=[annual_return],
            mode='markers+text',
            name='Optimal Portfolio',
            marker=dict(size=15, color='#0066cc'),
            text=[f"Sharpe: {sharpe_ratio:.2f}"],
            textposition="top right"
        ))
        
        fig_frontier.update_layout(
            height=500,
            xaxis_title="Annual Volatility",
            yaxis_title="Annual Return",
            hovermode='closest',
            template='plotly_dark',
            showlegend=True
        )
        
        st.plotly_chart(fig_frontier, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not generate efficient frontier: {str(e)[:100]}")
    
    # Transaction Costs Analysis
    if 'include_transaction_costs' in st.session_state and st.session_state.include_transaction_costs:
        st.subheader("💸 Transaction Cost Analysis")
        
        if st.session_state.previous_weights:
            cost_analysis = TransactionCostModel.calculate_portfolio_turnover_costs(
                st.session_state.previous_weights,
                weights,
                st.session_state.portfolio_value
            )
            
            col_cost1, col_cost2, col_cost3 = st.columns(3)
            
            with col_cost1:
                st.metric("Turnover Rate", f"{cost_analysis['turnover_rate']:.2%}")
            
            with col_cost2:
                st.metric("Transaction Costs", f"₺{cost_analysis['transaction_costs']:,.2f}")
            
            with col_cost3:
                st.metric("Cost % of Portfolio", f"{cost_analysis['costs_as_percent']:.2%}")
    
    # Discrete Allocation
    if 'calculate_discrete' in st.session_state and st.session_state.calculate_discrete:
        st.subheader("📦 Discrete Allocation")
        
        allocation, leftover = optimizer.calculate_discrete_allocation(
            weights,
            st.session_state.portfolio_value
        )
        
        if allocation:
            allocation_df = pd.DataFrame.from_dict(allocation, orient='index', 
                                                  columns=['Shares'])
            allocation_df['Price'] = asset_data['close'].iloc[-1][allocation_df.index]
            allocation_df['Value'] = allocation_df['Shares'] * allocation_df['Price']
            allocation_df['Weight %'] = allocation_df['Value'] / st.session_state.portfolio_value
            
            st.dataframe(
                allocation_df.style.format({
                    'Price': '₺{:.2f}',
                    'Value': '₺{:,.2f}',
                    'Weight %': '{:.2%}'
                }),
                use_container_width=True
            )
            
            st.info(f"💰 Leftover Cash: ₺{leftover:,.2f} ({leftover/st.session_state.portfolio_value:.2%})")
        else:
            st.warning("Could not calculate discrete allocation")

def display_risk_analytics():
    """Display comprehensive risk analytics"""
    st.header("📊 Risk Analytics")
    
    if not st.session_state.optimization_complete:
        st.warning("Please run optimization first")
        return
    
    # Get portfolio returns
    portfolio_returns = (st.session_state.asset_data['returns'] * 
                        pd.Series(st.session_state.current_weights)).sum(axis=1)
    
    # Initialize risk analyzer
    risk_analyzer = RiskAnalytics(
        portfolio_returns,
        st.session_state.benchmark_returns
    )
    
    # VaR Analysis
    st.subheader("🎯 Value at Risk (VaR) Analysis")
    
    var_results = risk_analyzer.calculate_var_cvar([0.90, 0.95, 0.99])
    
    var_df = pd.DataFrame({
        'Confidence Level': ['90%', '95%', '99%'],
        'Historical VaR': [var_results['VaR_90']['Historical'],
                          var_results['VaR_95']['Historical'],
                          var_results['VaR_99']['Historical']],
        'Parametric VaR': [var_results['VaR_90']['Parametric'],
                          var_results['VaR_95']['Parametric'],
                          var_results['VaR_99']['Parametric']],
        'Conditional VaR': [var_results['VaR_90']['CVaR_Historical'],
                           var_results['VaR_95']['CVaR_Historical'],
                           var_results['VaR_99']['CVaR_Historical']]
    })
    
    col_metrics, col_chart = st.columns(2)
    
    with col_metrics:
        st.dataframe(
            var_df.style.format("{:.2%}"),
            use_container_width=True,
            hide_index=True
        )
    
    with col_chart:
        # Plot VaR comparison
        fig_var = go.Figure()
        
        conf_levels = [0.90, 0.95, 0.99]
        colors = ['#00cc88', '#ffcc00', '#ff4d4d']
        
        for i, cl in enumerate(conf_levels):
            key = f'VaR_{int(cl*100)}'
            fig_var.add_trace(go.Bar(
                name=f'{int(cl*100)}%',
                x=['Historical', 'Parametric', 'Conditional'],
                y=[abs(var_results[key]['Historical']),
                   abs(var_results[key]['Parametric']),
                   abs(var_results[key]['CVaR_Historical'])],
                marker_color=colors[i],
                text=[f"{abs(var_results[key]['Historical']):.2%}",
                      f"{abs(var_results[key]['Parametric']):.2%}",
                      f"{abs(var_results[key]['CVaR_Historical']):.2%}"],
                textposition='auto'
            ))
        
        fig_var.update_layout(
            barmode='group',
            title="VaR at Different Confidence Levels",
            yaxis_title="Value at Risk",
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_var, use_container_width=True)
    
    # Risk Decomposition
    st.subheader("🔍 Risk Decomposition")
    
    try:
        # Calculate covariance matrix
        returns = st.session_state.asset_data['returns']
        weights = st.session_state.current_weights
        covariance_matrix = returns.cov() * 252
        
        decomposition = risk_analyzer.calculate_risk_decomposition(
            weights, 
            covariance_matrix
        )
        
        if not decomposition['marginal_contribution'].empty:
            risk_decomp_df = pd.DataFrame({
                'Asset': decomposition['marginal_contribution'].index,
                'Weight': [weights.get(a, 0) for a in decomposition['marginal_contribution'].index],
                'Marginal Contribution': decomposition['marginal_contribution'].values,
                '% Contribution': decomposition['percent_contribution'].values * 100
            })
            
            risk_decomp_df = risk_decomp_df.sort_values('% Contribution', 
                                                       ascending=False)
            
            fig_risk_decomp = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Marginal Contribution to Risk', 
                               'Percentage Contribution'),
                specs=[[{'type': 'bar'}, {'type': 'pie'}]]
            )
            
            fig_risk_decomp.add_trace(
                go.Bar(
                    x=risk_decomp_df['Asset'],
                    y=risk_decomp_df['Marginal Contribution'],
                    name='Marginal Contribution',
                    marker_color='#0066cc'
                ),
                row=1, col=1
            )
            
            fig_risk_decomp.add_trace(
                go.Pie(
                    labels=risk_decomp_df['Asset'],
                    values=risk_decomp_df['% Contribution'],
                    name='% Contribution',
                    hole=0.4
                ),
                row=1, col=2
            )
            
            fig_risk_decomp.update_layout(
                height=500,
                showlegend=False,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_risk_decomp, use_container_width=True)
            
            st.info(f"💰 Portfolio Volatility: {decomposition['portfolio_volatility']:.2%}")
    except Exception as e:
        st.warning(f"Risk decomposition not available: {str(e)[:100]}")
    
    # Tail Risk Metrics
    st.subheader("⚠️ Tail Risk Metrics")
    
    tail_metrics = risk_analyzer.calculate_tail_risk_metrics()
    
    if tail_metrics:
        col_tail1, col_tail2, col_tail3, col_tail4 = st.columns(4)
        
        with col_tail1:
            st.metric("Skewness", f"{tail_metrics['Skewness']:.3f}")
        
        with col_tail2:
            st.metric("Excess Kurtosis", f"{tail_metrics['Excess_Kurtosis']:.3f}")
        
        with col_tail3:
            st.metric("Tail Ratio", f"{tail_metrics['Tail_Ratio']:.2f}")
        
        with col_tail4:
            st.metric("VaR 99%", f"{tail_metrics['VaR_99']:.2%}")
        
        # Distribution plot
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=portfolio_returns,
            nbinsx=50,
            name='Return Distribution',
            marker_color='#0066cc',
            opacity=0.7
        ))
        
        # Add VaR lines
        for cl, color in zip([0.99, 0.95, 0.90], ['#ff4d4d', '#ff9900', '#ffcc00']):
            var_value = np.percentile(portfolio_returns, (1-cl)*100)
            fig_dist.add_vline(
                x=var_value,
                line_dash="dash",
                line_color=color,
                annotation_text=f"VaR {int(cl*100)}%",
                annotation_position="top left"
            )
        
        fig_dist.update_layout(
            title="Return Distribution with VaR Levels",
            xaxis_title="Daily Return",
            yaxis_title="Frequency",
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Scenario Analysis
    st.subheader("🌪️ Scenario Analysis")
    
    try:
        scenario_analyzer = ScenarioAnalyzer(
            st.session_state.asset_data['returns'],
            st.session_state.current_weights
        )
        
        scenarios = scenario_analyzer.generate_correlation_scenarios()
        scenario_results = scenario_analyzer.stress_test_portfolio(scenarios)
        
        scenario_df = pd.DataFrame.from_dict(scenario_results, orient='index')
        
        # Display scenario results
        st.dataframe(
            scenario_df.style.format({
                'annual_return': '{:.2%}',
                'annual_volatility': '{:.2%}',
                'max_drawdown': '{:.2%}',
                'var_95': '{:.2%}'
            }),
            use_container_width=True
        )
        
    except Exception as e:
        st.warning(f"Scenario analysis not available: {str(e)[:100]}")

def display_portfolio_report():
    """Display comprehensive portfolio report"""
    st.header("📋 Portfolio Performance Report")
    
    if not st.session_state.optimization_complete:
        st.warning("Please run optimization first")
        return
    
    # Get portfolio returns
    portfolio_returns = (st.session_state.asset_data['returns'] * 
                        pd.Series(st.session_state.current_weights)).sum(axis=1)
    
    # Initialize analytics
    analytics = QuantStatsAnalytics(
        portfolio_returns,
        st.session_state.benchmark_returns,
        st.session_state.risk_free_rate
    )
    
    # Display metrics
    metrics = analytics.calculate_advanced_metrics()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    key_metrics = {
        'Annual Return': metrics.get('Annual Return', 0),
        'Annual Volatility': metrics.get('Annual Volatility', 0),
        'Sharpe Ratio': metrics.get('Sharpe Ratio', 0),
        'Max Drawdown': metrics.get('Max Drawdown', 0)
    }
    
    for (name, value), col in zip(key_metrics.items(), [col1, col2, col3, col4]):
        with col:
            if 'Return' in name or 'Drawdown' in name:
                display_val = f"{value:.2%}"
            else:
                display_val = f"{value:.2f}"
            
            st.markdown(f"""
            <div class="metric-card">
                <small>{name}</small><br>
                <h3 style="color: {'#00cc88' if value >= 0 else '#ff4d4d'}">
                    {display_val}
                </h3>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional Metrics
    col5, col6, col7, col8 = st.columns(4)
    additional_metrics = {
        'Sortino Ratio': metrics.get('Sortino Ratio', 0),
        'Calmar Ratio': metrics.get('Calmar Ratio', 0),
        'Omega Ratio': metrics.get('Omega Ratio', 0),
        'Alpha': metrics.get('Alpha', 0)
    }
    
    for (name, value), col in zip(additional_metrics.items(), [col5, col6, col7, col8]):
        with col:
            display_val = f"{value:.2f}" if value != float('inf') else "∞"
            st.metric(name, display_val)
    
    # Tearsheet
    if 'show_tearsheet' in st.session_state and st.session_state.show_tearsheet:
        st.subheader("📈 Interactive Tearsheet")
        
        tearsheet = analytics.generate_tearsheet()
        if tearsheet:
            st.plotly_chart(tearsheet, use_container_width=True)
        else:
            st.warning("Could not generate tearsheet")
    
    # Full QuantStats Report
    if 'generate_full_report' in st.session_state and st.session_state.generate_full_report:
        st.subheader("📊 Full Analytics Report")
        
        with st.spinner("Generating comprehensive report..."):
            html_report = analytics.generate_full_report()
            
            # Display report
            st.components.v1.html(html_report, height=800, scrolling=True)
            
            # Download button
            st.download_button(
                label="📥 Download Report as HTML",
                data=html_report,
                file_name="portfolio_report.html",
                mime="text/html"
            )

def display_rebalancing_tools():
    """Display portfolio rebalancing tools"""
    st.header("🔄 Portfolio Rebalancing")
    
    if not st.session_state.optimization_complete:
        st.warning("Please run optimization first")
        return
    
    col_rebalance, col_tracking = st.columns(2)
    
    with col_rebalance:
        st.subheader("Manual Rebalancing")
        
        # Display current vs target weights
        if st.session_state.previous_weights:
            comparison_df = pd.DataFrame({
                'Asset': list(set(st.session_state.current_weights.keys()) | 
                            set(st.session_state.previous_weights.keys())),
                'Current': [st.session_state.current_weights.get(a, 0) 
                          for a in list(set(st.session_state.current_weights.keys()) | 
                                       set(st.session_state.previous_weights.keys()))],
                'Previous': [st.session_state.previous_weights.get(a, 0) 
                           for a in list(set(st.session_state.current_weights.keys()) | 
                                        set(st.session_state.previous_weights.keys()))]
            })
            
            comparison_df['Change'] = comparison_df['Current'] - comparison_df['Previous']
            comparison_df = comparison_df.sort_values('Change', ascending=False)
            
            st.dataframe(
                comparison_df.style.format({
                    'Current': '{:.2%}',
                    'Previous': '{:.2%}',
                    'Change': '{:+.2%}'
                }).applymap(
                    lambda x: 'color: #00cc88' if isinstance(x, str) and '+' in x 
                    else 'color: #ff4d4d' if isinstance(x, str) and '-' in x 
                    else '', 
                    subset=['Change']
                ),
                use_container_width=True
            )
        
        # Manual adjustment
        st.subheader("Adjust Weights")
        
        adjusted_weights = {}
        for ticker, weight in st.session_state.current_weights.items():
            if weight > 0.001:
                new_weight = st.slider(
                    ticker,
                    0.0, 1.0, float(weight), 0.01,
                    key=f"adj_{ticker}"
                )
                adjusted_weights[ticker] = new_weight
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        if st.button("Apply Adjusted Weights"):
            st.session_state.previous_weights = st.session_state.current_weights.copy()
            st.session_state.current_weights = adjusted_weights
            st.success("Weights updated successfully!")
            st.rerun()
    
    with col_tracking:
        st.subheader("Performance Tracking")
        
        # Calculate tracking error
        if st.session_state.benchmark_returns is not None:
            portfolio_returns = (st.session_state.asset_data['returns'] * 
                               pd.Series(st.session_state.current_weights)).sum(axis=1)
            
            tracking_error = (portfolio_returns - 
                            st.session_state.benchmark_returns).std() * np.sqrt(252)
            
            col_te1, col_te2 = st.columns(2)
            with col_te1:
                st.metric("Tracking Error", f"{tracking_error:.2%}")
            
            with col_te2:
                excess_return = (portfolio_returns.mean() - 
                               st.session_state.benchmark_returns.mean()) * 252
                information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
                st.metric("Information Ratio", f"{information_ratio:.2f}")
            
            # Plot cumulative performance
            cum_portfolio = (1 + portfolio_returns).cumprod()
            cum_benchmark = (1 + st.session_state.benchmark_returns).cumprod()
            
            fig_tracking = go.Figure()
            fig_tracking.add_trace(go.Scatter(
                x=cum_portfolio.index,
                y=cum_portfolio.values,
                name='Portfolio',
                line=dict(color='#00cc88', width=2)
            ))
            fig_tracking.add_trace(go.Scatter(
                x=cum_benchmark.index,
                y=cum_benchmark.values,
                name='Benchmark',
                line=dict(color='#0066cc', width=2, dash='dash')
            ))
            
            fig_tracking.update_layout(
                title="Portfolio vs Benchmark",
                yaxis_title="Cumulative Return",
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig_tracking, use_container_width=True)
        
        # Rebalancing Calendar
        st.subheader("📅 Rebalancing Calendar")
        
        rebalance_freq = st.selectbox(
            "Rebalancing Frequency",
            ["Monthly", "Quarterly", "Semi-Annually", "Annually", "Never"]
        )
        
        last_rebalance = st.date_input(
            "Last Rebalancing Date",
            datetime.now() - timedelta(days=30)
        )
        
        if rebalance_freq != "Never":
            if rebalance_freq == "Monthly":
                next_rebalance = last_rebalance + timedelta(days=30)
            elif rebalance_freq == "Quarterly":
                next_rebalance = last_rebalance + timedelta(days=90)
            elif rebalance_freq == "Semi-Annually":
                next_rebalance = last_rebalance + timedelta(days=180)
            else:  # Annually
                next_rebalance = last_rebalance + timedelta(days=365)
            
            days_until = (next_rebalance - datetime.now().date()).days
            
            st.info(f"⏰ Next rebalancing in **{days_until} days** ({next_rebalance.strftime('%Y-%m-%d')})")
            
            if days_until <= 7:
                st.warning("⚠️ Rebalancing due soon!")
                if st.button("Rebalance Now"):
                    st.info("Rebalancing functionality would trigger optimization here")
        else:
            st.info("Automatic rebalancing is disabled")

def display_additional_tools():
    """Display additional analytical tools"""
    st.header("🛠️ Analytical Tools")
    
    # Monte Carlo Simulation
    with st.expander("🎲 Monte Carlo Simulation", expanded=False):
        st.subheader("Portfolio Monte Carlo Simulation")
        
        n_simulations = st.slider("Number of Simulations", 100, 10000, 1000, 100)
        n_years = st.slider("Time Horizon (Years)", 1, 30, 10, 1)
        
        if st.button("Run Monte Carlo Simulation"):
            with st.spinner("Running simulations..."):
                try:
                    # Get portfolio statistics
                    portfolio_returns = (st.session_state.asset_data['returns'] * 
                                       pd.Series(st.session_state.current_weights)).sum(axis=1)
                    
                    mean_return = portfolio_returns.mean() * 252
                    std_return = portfolio_returns.std() * np.sqrt(252)
                    
                    # Run simulations
                    results = []
                    for _ in range(n_simulations):
                        # Generate random returns
                        random_returns = np.random.normal(
                            mean_return/n_years,
                            std_return/np.sqrt(n_years),
                            n_years
                        )
                        # Calculate terminal value
                        terminal_value = np.prod(1 + random_returns)
                        results.append(terminal_value)
                    
                    # Create histogram
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Histogram(
                        x=results,
                        nbinsx=50,
                        name='Terminal Values',
                        marker_color='#0066cc'
                    ))
                    
                    # Add statistics
                    mean_val = np.mean(results)
                    median_val = np.median(results)
                    percentile_5 = np.percentile(results, 5)
                    percentile_95 = np.percentile(results, 95)
                    
                    fig_mc.add_vline(
                        x=mean_val,
                        line_dash="dash",
                        line_color="#00cc88",
                        annotation_text=f"Mean: {mean_val:.2f}"
                    )
                    
                    fig_mc.add_vline(
                        x=percentile_5,
                        line_dash="dash",
                        line_color="#ff4d4d",
                        annotation_text=f"5%: {percentile_5:.2f}"
                    )
                    
                    fig_mc.add_vline(
                        x=percentile_95,
                        line_dash="dash",
                        line_color="#ffcc00",
                        annotation_text=f"95%: {percentile_95:.2f}"
                    )
                    
                    fig_mc.update_layout(
                        title=f"Monte Carlo Simulation Results ({n_simulations:,} simulations)",
                        xaxis_title="Terminal Portfolio Value",
                        yaxis_title="Frequency",
                        template='plotly_dark',
                        height=500
                    )
                    
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # Display statistics
                    col_mc1, col_mc2, col_mc3, col_mc4 = st.columns(4)
                    
                    with col_mc1:
                        st.metric("Mean", f"{mean_val:.2f}x")
                    
                    with col_mc2:
                        st.metric("Median", f"{median_val:.2f}x")
                    
                    with col_mc3:
                        st.metric("5th Percentile", f"{percentile_5:.2f}x")
                    
                    with col_mc4:
                        st.metric("95th Percentile", f"{percentile_95:.2f}x")
                        
                except Exception as e:
                    st.error(f"Simulation failed: {str(e)[:100]}")
    
    # Portfolio Analytics
    with st.expander("📊 Advanced Portfolio Analytics", expanded=False):
        st.subheader("Portfolio Analytics")
        
        # Rolling statistics
        window = st.slider("Rolling Window (Days)", 21, 252, 126, 21)
        
        portfolio_returns = (st.session_state.asset_data['returns'] * 
                           pd.Series(st.session_state.current_weights)).sum(axis=1)
        
        rolling_sharpe = portfolio_returns.rolling(window).apply(
            lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        
        rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
        
        fig_rolling = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'Rolling Sharpe ({window} days)', 
                          f'Rolling Volatility ({window} days)'),
            vertical_spacing=0.15
        )
        
        fig_rolling.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                name='Rolling Sharpe',
                line=dict(color='#00cc88', width=2)
            ),
            row=1, col=1
        )
        
        fig_rolling.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name='Rolling Volatility',
                line=dict(color='#0066cc', width=2)
            ),
            row=2, col=1
        )
        
        fig_rolling.update_layout(
            height=600,
            showlegend=True,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_rolling, use_container_width=True)
    
    # Data Export
    with st.expander("💾 Export Data", expanded=False):
        st.subheader("Export Portfolio Data")
        
        if st.session_state.optimization_complete:
            # Prepare data for export
            export_data = {
                'weights': st.session_state.current_weights,
                'performance': st.session_state.performance,
                'assets': st.session_state.selected_assets,
                'optimization_date': datetime.now().isoformat(),
                'parameters': {
                    'risk_free_rate': st.session_state.risk_free_rate,
                    'optimization_method': st.session_state.get('opt_method', 'max_sharpe')
                }
            }
            
            # JSON export
            json_data = json.dumps(export_data, indent=2)
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name="portfolio_config.json",
                    mime="application/json"
                )
            
            with col_export2:
                # Export to Excel
                weights_df = pd.DataFrame.from_dict(
                    st.session_state.current_weights, 
                    orient='index', 
                    columns=['Weight']
                )
                
                # Convert to Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    weights_df.to_excel(writer, sheet_name='Portfolio Weights')
                    
                    # Add performance metrics
                    perf_df = pd.DataFrame({
                        'Metric': ['Annual Return', 'Annual Volatility', 'Sharpe Ratio'],
                        'Value': st.session_state.performance
                    })
                    perf_df.to_excel(writer, sheet_name='Performance', index=False)
                
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Download as Excel",
                    data=excel_data,
                    file_name="portfolio_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)[:200]}")
        logger.error(f"Application crashed: {e}", exc_info=True)
        
        # Display traceback in debug mode
        if st.secrets.get("DEBUG_MODE", False):
            st.code(traceback.format_exc())
        
        # Show recovery options
        st.info("""
        **Troubleshooting Tips:**
        1. Try refreshing the page
        2. Reduce the number of selected assets
        3. Use a shorter time period
        4. Check your internet connection
        
        If the problem persists, please contact support.
        """)
