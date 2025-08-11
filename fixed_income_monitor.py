import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import traceback
import requests
import json
warnings.filterwarnings('ignore')

# Page config for mobile-friendly layout
st.set_page_config(
    page_title="Fixed Income Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile responsiveness
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    .stPlotlyChart {
        background-color: black;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: black;
        padding: 0.5rem;
        border-radius: 0.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def fred_get(series_id, start, end):
    """Safe fetch from FRED with caching and detailed error handling."""
    try:
        df = web.DataReader(series_id, 'fred', start, end)
        if df is not None and not df.empty:
            st.success(f"✅ Loaded {series_id}: {len(df)} records")
            return df
        else:
            st.warning(f"⚠️ {series_id}: No data returned")
            return None
    except Exception as e:
        st.error(f"❌ Failed to fetch {series_id}: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_vix_data(start, end):
    """Fetch VIX data with caching and better error handling."""
    try:
        vix_df = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
        if vix_df is not None and not vix_df.empty:
            vix_series = vix_df['Close'].rename('VIX')
            st.success(f"✅ Loaded VIX: {len(vix_series)} records")
            return vix_series
        else:
            st.warning("⚠️ VIX: No data returned")
            return None
    except Exception as e:
        st.error(f"❌ Failed to fetch VIX: {str(e)}")
        return None

@st.cache_data(ttl=21600, show_spinner=False)  # Cache for 6 hours (less frequent updates)
def get_cme_fed_probabilities():
    """Fetch Fed funds rate probabilities from CME (simplified version)."""
    try:
        # Note: This is a placeholder since CME data requires specific API access
        # For now, we'll create sample data structure
        st.info("📊 CME Fed Rate Probabilities: Using sample data (API access required for live data)")
        
        # Sample probabilities structure
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        prob_25bp_hike = [0.2, 0.3, 0.4, 0.5, 0.3, 0.2, 0.1, 0.15, 0.25, 0.35, 0.4, 0.3]
        prob_no_change = [0.6, 0.5, 0.4, 0.3, 0.5, 0.6, 0.7, 0.65, 0.55, 0.45, 0.4, 0.5]
        prob_25bp_cut = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        
        df = pd.DataFrame({
            'Prob_Hike_25bp': prob_25bp_hike,
            'Prob_No_Change': prob_no_change,
            'Prob_Cut_25bp': prob_25bp_cut
        }, index=dates)
        
        return df
    except Exception as e:
        st.error(f"❌ Failed to fetch CME Fed probabilities: {str(e)}")
        return None

def create_plotly_chart(df, title, ylabel='Value'):
    """Create mobile-friendly Plotly chart with better error handling."""
    if df is None:
        st.warning(f"No data available for {title}")
        return None
        
    if isinstance(df, pd.DataFrame) and df.empty:
        st.warning(f"Empty dataset for {title}")
        return None
    
    try:
        fig = go.Figure()
        
        # Handle Series vs DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, col in enumerate(df.columns):
            # Skip NaN values
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                fig.add_trace(go.Scatter(
                    x=clean_data.index,
                    y=clean_data.values,
                    mode='lines',
                    name=str(col),
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'<b>{col}</b><br>Date: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
                ))
                
                # Add latest value annotation
                if len(clean_data) > 0:
                    latest_val = clean_data.iloc[-1]
                    latest_date = clean_data.index[-1]
                    fig.add_annotation(
                        x=latest_date,
                        y=latest_val,
                        text=f'{latest_val:.2f}',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor=colors[i % len(colors)],
                        font=dict(size=10, color=colors[i % len(colors)])
                    )
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Date",
            yaxis_title=ylabel,
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
            plot_bgcolor='black',
            paper_bgcolor='black'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart for {title}: {str(e)}")
        return None

def create_dual_axis_chart(left_data, right_data, title, left_ylabel, right_ylabel):
    """Create a dual-axis chart for comparing different scales."""
    if left_data is None and right_data is None:
        return None
    
    try:
        from plotly.subplots import make_subplots
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        colors_left = ['#1f77b4', '#ff7f0e']
        colors_right = ['#2ca02c', '#d62728']
        
        # Add left axis data (Credit Spreads)
        if left_data is not None:
            if isinstance(left_data, pd.Series):
                left_data = left_data.to_frame()
            
            for i, col in enumerate(left_data.columns):
                clean_data = left_data[col].dropna()
                if len(clean_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=clean_data.index,
                            y=clean_data.values,
                            mode='lines',
                            name=str(col),
                            line=dict(color=colors_left[i % len(colors_left)], width=2),
                            yaxis='y',
                        ),
                        secondary_y=False
                    )
        
        # Add right axis data (EM Performance)
        if right_data is not None:
            if isinstance(right_data, pd.Series):
                right_data = right_data.to_frame()
            
            for i, col in enumerate(right_data.columns):
                clean_data = right_data[col].dropna()
                if len(clean_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=clean_data.index,
                            y=clean_data.values,
                            mode='lines',
                            name=str(col),
                            line=dict(color=colors_right[i % len(colors_right)], width=2),
                            yaxis='y2',
                        ),
                        secondary_y=True
                    )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Date",
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
            plot_bgcolor='black',
            paper_bgcolor='black'
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text=left_ylabel, secondary_y=False)
        fig.update_yaxes(title_text=right_ylabel, secondary_y=True)
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', secondary_y=False)
        
        return fig
    except Exception as e:
        st.error(f"Error creating dual-axis chart: {str(e)}")
        return None
    """Create stacked area chart for Fed rate probabilities."""
    if df is None or df.empty:
        return None
    
    try:
        fig = go.Figure()
        
        # Colors for different probability categories
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        for i, col in enumerate(df.columns):
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                fig.add_trace(go.Scatter(
                    x=clean_data.index,
                    y=clean_data.values,
                    mode='lines',
                    name=col.replace('_', ' ').replace('Prob ', ''),
                    fill='tonexty' if i > 0 else 'tozeroy',
                    line=dict(color=colors[i % len(colors)], width=1),
                    hovertemplate=f'<b>{col}</b><br>Date: %{{x}}<br>Probability: %{{y:.1%}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Date",
            yaxis_title="Probability",
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
            plot_bgcolor='black',
            paper_bgcolor='black'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', tickformat='.0%')
        
        return fig
    except Exception as e:
        st.error(f"Error creating probability chart: {str(e)}")
        return None

def display_key_metrics(data_dict):
    """Display key metrics in a mobile-friendly format."""
    if not data_dict:
        st.warning("No key metrics available")
        return
        
    st.subheader("📊 Key Metrics (Latest Values)")
    
    # Create more columns for more metrics
    cols = st.columns(3)
    col_idx = 0
    
    for name, df in data_dict.items():
        if df is not None and not df.empty:
            try:
                with cols[col_idx % 3]:
                    if isinstance(df, pd.Series):
                        clean_data = df.dropna()
                        if len(clean_data) > 0:
                            latest_val = clean_data.iloc[-1]
                            latest_date = clean_data.index[-1]
                        else:
                            latest_val = 0
                            latest_date = 'N/A'
                    elif isinstance(df, pd.DataFrame):
                        # Take first column for multi-column DataFrames
                        col_name = df.columns[0]
                        clean_data = df[col_name].dropna()
                        if len(clean_data) > 0:
                            latest_val = clean_data.iloc[-1]
                            latest_date = clean_data.index[-1]
                        else:
                            latest_val = 0
                            latest_date = 'N/A'
                    
                    # Format date
                    if latest_date != 'N/A':
                        date_str = latest_date.strftime('%Y-%m-%d')
                    else:
                        date_str = 'N/A'
                    
                    # Format value based on metric type
                    if 'Spread' in name or 'Rate' in name or 'Yield' in name:
                        value_str = f"{latest_val:.2f}%"
                    elif 'CPI' in name and 'YoY' in name:
                        value_str = f"{latest_val:.1f}%"
                    elif 'Index' in name:
                        value_str = f"{latest_val:.0f}"
                    else:
                        value_str = f"{latest_val:.2f}"
                    
                    st.metric(
                        label=name,
                        value=value_str,
                        help=f"Latest: {date_str}"
                    )
                    col_idx += 1
            except Exception as e:
                st.error(f"Error displaying metric {name}: {str(e)}")

# Main app
def main():
    st.title("📈 Comprehensive Fixed Income Monitor")
    st.markdown("*Mobile-optimized market monitoring dashboard*")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Date range selector
        default_start = datetime(2022, 1, 1)
        start_date = st.date_input("Start Date", default_start, max_value=datetime.now())
        end_date = st.date_input("End Date", datetime.now())
        
        # Convert to datetime
        start = datetime.combine(start_date, datetime.min.time())
        end = datetime.combine(end_date, datetime.min.time())
        
        # Chart selection
        st.header("📊 Charts to Display")
        show_policy = st.checkbox("Economic Policy Uncertainty", True)
        show_treasury = st.checkbox("Treasury Yield Curve", True)
        show_spreads = st.checkbox("Treasury Spreads", True) 
        show_credit = st.checkbox("Credit Spreads", True)
        show_rates = st.checkbox("Short Rates (SOFR/EFFR)", True)
        show_fed_prob = st.checkbox("Fed Rate Probabilities", True)
        show_repo = st.checkbox("Overnight Repo Rate", True)
        show_embi = st.checkbox("US Credit vs EM Stocks & Bonds", True)
        show_cpi = st.checkbox("CPI Inflation (YoY & MoM)", True)
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Data loading with progress
    with st.spinner('🔄 Loading comprehensive market data...'):
        
        # 1) Economic Policy Uncertainty
        epu = None
        if show_policy:
            st.write("📈 Fetching Economic Policy Uncertainty...")
            epu = fred_get('USEPUINDXD', start, end)
        
        # 2) Treasury yields - Extended curve
        treasury_df = pd.DataFrame()
        if show_treasury:
            st.write("📊 Fetching Treasury Yield Curve...")
            dgs_ids = {
                'DGS1MO': '1M',
                'DGS3MO': '3M', 
                'DGS6MO': '6M',
                'DGS1': '1Y',
                'DGS2': '2Y',
                'DGS5': '5Y',
                'DGS10': '10Y',
                'DGS30': '30Y'
            }
            treasury_parts = {}
            for sid, label in dgs_ids.items():
                df = fred_get(sid, start, end)
                if df is not None and not df.empty:
                    df = df.rename(columns={sid: label})
                    treasury_parts[label] = df[label]
            
            if treasury_parts:
                treasury_df = pd.concat(treasury_parts, axis=1).sort_index()
        
        # 3) Treasury spreads - Fixed labels
        spreads_data = {}
        if show_spreads and not treasury_df.empty:
            st.write("📏 Computing Treasury Spreads...")
            try:
                if '10Y' in treasury_df.columns and '2Y' in treasury_df.columns:
                    spread = (treasury_df['10Y'] - treasury_df['2Y']).to_frame(name='10Y-2Y')
                    spreads_data['10Y-2Y Spread'] = spread
                if '10Y' in treasury_df.columns and '3M' in treasury_df.columns:
                    spread = (treasury_df['10Y'] - treasury_df['3M']).to_frame(name='10Y-3M')
                    spreads_data['10Y-3M Spread'] = spread
                if '30Y' in treasury_df.columns and '5Y' in treasury_df.columns:
                    spread = (treasury_df['30Y'] - treasury_df['5Y']).to_frame(name='30Y-5Y')
                    spreads_data['30Y-5Y Spread'] = spread
            except Exception as e:
                st.error(f"❌ Error computing spreads: {str(e)}")
        
        # 4) Credit spreads - AAA, BBB, CCC vs Treasury
        credit_df = pd.DataFrame()
        if show_credit:
            st.write("🏦 Fetching Credit Spreads...")
            credit_series = {}
            credit_ids = {
                'BAMLC0A1CAAA': 'AAA',     # AAA Option-Adjusted Spread
                'BAMLC0A4CBBB': 'BBB',     # BBB Option-Adjusted Spread  
                'BAMLH0A3HYC': 'CCC'       # CCC High Yield Option-Adjusted Spread
            }
            for sid, label in credit_ids.items():
                df = fred_get(sid, start, end)
                if df is not None and not df.empty:
                    credit_series[label] = df.iloc[:, 0]
            if credit_series:
                credit_df = pd.concat(credit_series, axis=1)
        
        # 5) Short rates - SOFR and Effective Fed Funds
        rates_data = {}
        if show_rates:
            st.write("💰 Fetching Short-term Rates...")
            sofr = fred_get('SOFR', start, end)
            effr = fred_get('EFFR', start, end)
            if sofr is not None:
                rates_data['SOFR'] = sofr
            if effr is not None:
                rates_data['EFFR'] = effr
        
        # 6) Fed Rate Probabilities
        fed_prob_data = None
        if show_fed_prob:
            st.write("🎯 Fetching Fed Rate Probabilities...")
            fed_prob_data = get_cme_fed_probabilities()
        
        # 7) Overnight Repo Rate
        repo_data = None
        if show_repo:
            st.write("🔄 Fetching Overnight Repo Rate...")
            repo_data = fred_get('RRPONTSYD', start, end)  # Overnight Reverse Repo Rate
        
        # 8) US Credit Spread vs EM Markets
        em_credit_data = {}
        if show_embi:
            st.write("🌍 Fetching US Credit vs EM Market Data...")
            # US High Yield Credit Spread
            us_hy_spread = fred_get('BAMLH0A0HYM2', start, end)  # US High Yield Master II OAS
            # EM Stock Index (using MSCI EM as proxy via Yahoo Finance)
            try:
                em_stocks = yf.download("EEM", start=start, end=end, progress=False, auto_adjust=False)
                if em_stocks is not None and not em_stocks.empty:
                    em_stocks_close = em_stocks['Close'].rename('EM Stocks (EEM)')
                    # Normalize to percentage change from start for comparison
                    em_stocks_norm = ((em_stocks_close / em_stocks_close.iloc[0]) - 1) * 100
                    em_credit_data['EM Stocks % Change'] = em_stocks_norm
                    st.success("✅ Loaded EM Stocks (EEM)")
            except Exception as e:
                st.warning(f"⚠️ Failed to fetch EM stocks: {str(e)}")
            
            # EM Bonds (using EMB ETF as proxy)
            try:
                em_bonds = yf.download("EMB", start=start, end=end, progress=False, auto_adjust=False)
                if em_bonds is not None and not em_bonds.empty:
                    em_bonds_close = em_bonds['Close'].rename('EM Bonds (EMB)')
                    # Normalize to percentage change from start
                    em_bonds_norm = ((em_bonds_close / em_bonds_close.iloc[0]) - 1) * 100
                    em_credit_data['EM Bonds % Change'] = em_bonds_norm
                    st.success("✅ Loaded EM Bonds (EMB)")
            except Exception as e:
                st.warning(f"⚠️ Failed to fetch EM bonds: {str(e)}")
            
            if us_hy_spread is not None:
                em_credit_data['US HY Credit Spread'] = us_hy_spread
        
        # 9) CPI Data - YoY and MoM
        cpi_data = {}
        if show_cpi:
            st.write("📊 Fetching CPI Data...")
            # CPI All Urban Consumers
            cpi_all = fred_get('CPIAUCSL', start, end)
            # Core CPI (less food and energy)
            cpi_core = fred_get('CPILFESL', start, end)
            
            if cpi_all is not None and not cpi_all.empty:
                # Calculate YoY and MoM for CPI
                cpi_yoy = cpi_all.pct_change(periods=12) * 100
                cpi_mom = cpi_all.pct_change(periods=1) * 100
                cpi_data['CPI YoY'] = cpi_yoy.rename(columns={'CPIAUCSL': 'CPI YoY'})
                cpi_data['CPI MoM'] = cpi_mom.rename(columns={'CPIAUCSL': 'CPI MoM'})
            
            if cpi_core is not None and not cpi_core.empty:
                # Calculate YoY and MoM for Core CPI
                core_yoy = cpi_core.pct_change(periods=12) * 100
                core_mom = cpi_core.pct_change(periods=1) * 100
                cpi_data['Core CPI YoY'] = core_yoy.rename(columns={'CPILFESL': 'Core CPI YoY'})
                cpi_data['Core CPI MoM'] = core_mom.rename(columns={'CPILFESL': 'Core CPI MoM'})
    
    st.success("🎉 Data loading completed!")
    
    # Display key metrics
    key_metrics = {}
    
    # Add various metrics to the dashboard
    if epu is not None and not epu.empty:
        key_metrics['Policy Uncertainty'] = epu.iloc[:, 0]
    if not treasury_df.empty and '10Y' in treasury_df.columns:
        key_metrics['10Y Treasury'] = treasury_df['10Y']
    if spreads_data and '10Y-2Y Spread' in spreads_data:
        key_metrics['10Y-2Y Spread'] = spreads_data['10Y-2Y Spread'].iloc[:, 0]
    if not credit_df.empty and 'BBB' in credit_df.columns:
        key_metrics['BBB Credit Spread'] = credit_df['BBB']
    if rates_data and 'SOFR' in rates_data:
        key_metrics['SOFR'] = rates_data['SOFR'].iloc[:, 0]
    if cpi_data and 'CPI YoY' in cpi_data:
        key_metrics['CPI YoY'] = cpi_data['CPI YoY'].iloc[:, 0]
    
    if key_metrics:
        display_key_metrics(key_metrics)
    
    # Display charts
    st.header("📊 Comprehensive Market Charts")
    
    # 1. Economic Policy Uncertainty
    if show_policy and epu is not None:
        st.subheader("📈 Economic Policy Uncertainty Index")
        fig = create_plotly_chart(epu, "Economic Policy Uncertainty Index", "Index Level")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # 2. Treasury Yield Curve
    if show_treasury and not treasury_df.empty:
        st.subheader("📊 U.S. Treasury Yield Curve")
        fig = create_plotly_chart(treasury_df, "U.S. Treasury Yields (1M to 30Y)", "Yield (%)")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # 3. Treasury Spreads
    if show_spreads and spreads_data:
        st.subheader("📏 Treasury Yield Spreads")
        for name, spread_df in spreads_data.items():
            fig = create_plotly_chart(spread_df, f"U.S. {name}", "Spread (bps)")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # 4. Credit Spreads
    if show_credit and not credit_df.empty:
        st.subheader("🏦 Credit Spreads vs Treasury")
        fig = create_plotly_chart(credit_df, "Credit Spreads (AAA, BBB, CCC)", "OAS Spread (bps)")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # 5. Short-term Rates
    if show_rates and rates_data:
        st.subheader("💰 Short-term Interest Rates")
        for name, rate_df in rates_data.items():
            fig = create_plotly_chart(rate_df, f"{name} Rate", "Rate (%)")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # 6. Fed Rate Probabilities
    if show_fed_prob and fed_prob_data is not None:
        st.subheader("🎯 Fed Funds Rate Probability (Next Meeting)")
        fig = create_stacked_probability_chart(fed_prob_data, "Fed Rate Change Probabilities")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # 7. Overnight Repo
    if show_repo and repo_data is not None:
        st.subheader("🔄 Overnight Reverse Repo Operations")
        fig = create_plotly_chart(repo_data, "ON RRP Operations Volume", "Amount ($ Billions)")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # 8. US Credit vs EM Markets
    if show_embi and em_credit_data:
        st.subheader("🌍 US Credit Spread vs Emerging Market Performance")
        
        # Separate credit spreads from EM performance data
        credit_spread_data = None
        em_performance_data = {}
        
        if 'US HY Credit Spread' in em_credit_data:
            credit_spread_data = em_credit_data['US HY Credit Spread']
        
        if 'EM Stocks % Change' in em_credit_data:
            em_performance_data['EM Stocks % Change'] = em_credit_data['EM Stocks % Change']
        if 'EM Bonds % Change' in em_credit_data:
            em_performance_data['EM Bonds % Change'] = em_credit_data['EM Bonds % Change']
        
        if em_performance_data:
            em_perf_df = pd.concat(em_performance_data, axis=1)
            fig = create_dual_axis_chart(
                credit_spread_data, 
                em_perf_df, 
                "US Credit Risk vs Emerging Market Performance",
                "Credit Spread (bps)",
                "EM Performance (% Change)"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **Chart Logic**: When US credit spreads widen (higher risk), EM assets often underperform (negative correlation)")
        else:
            # Fallback to single chart if EM data unavailable
            if credit_spread_data is not None:
                fig = create_plotly_chart(credit_spread_data, "US High Yield Credit Spread", "Spread (bps)")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    # 9. CPI Data
    if show_cpi and cpi_data:
        st.subheader("📊 Consumer Price Index (CPI)")
        
        # Combine YoY data
        if 'CPI YoY' in cpi_data and 'Core CPI YoY' in cpi_data:
            yoy_combined = pd.concat([
                cpi_data['CPI YoY'].iloc[:, 0],
                cpi_data['Core CPI YoY'].iloc[:, 0]
            ], axis=1)
            yoy_combined.columns = ['CPI YoY', 'Core CPI YoY']
            fig = create_plotly_chart(yoy_combined, "CPI Year-over-Year", "YoY Change (%)")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Combine MoM data
        if 'CPI MoM' in cpi_data and 'Core CPI MoM' in cpi_data:
            mom_combined = pd.concat([
                cpi_data['CPI MoM'].iloc[:, 0],
                cpi_data['Core CPI MoM'].iloc[:, 0]
            ], axis=1)
            mom_combined.columns = ['CPI MoM', 'Core CPI MoM']
            fig = create_plotly_chart(mom_combined, "CPI Month-over-Month", "MoM Change (%)")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**📊 Data Sources:**")
    st.markdown("*• FRED (Federal Reserve Economic Data) • Yahoo Finance (EEM, EMB ETFs) • CME Group • Policy Uncertainty*")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC*")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("🚨 Critical Error in Main App")
        st.write("Error details:", str(e))
        st.write("Full traceback:")
        st.code(traceback.format_exc())


