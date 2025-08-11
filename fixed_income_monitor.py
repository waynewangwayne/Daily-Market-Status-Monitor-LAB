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
        background-color: white;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: white;
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
        # st.write(f"Debug info: {traceback.format_exc()}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_vix_data(start, end):
    """Fetch VIX data with caching and better error handling."""
    try:
        # Fix the yfinance warning by explicitly setting auto_adjust
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
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart for {title}: {str(e)}")
        return None

def display_key_metrics(data_dict):
    """Display key metrics in a mobile-friendly format."""
    if not data_dict:
        st.warning("No key metrics available")
        return
        
    st.subheader("📊 Key Metrics (Latest Values)")
    
    cols = st.columns(2)
    col_idx = 0
    
    for name, df in data_dict.items():
        if df is not None and not df.empty:
            try:
                with cols[col_idx % 2]:
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
                    
                    st.metric(
                        label=name,
                        value=f"{latest_val:.3f}",
                        help=f"Latest: {date_str}"
                    )
                    col_idx += 1
            except Exception as e:
                st.error(f"Error displaying metric {name}: {str(e)}")

# Main app
def main():
    st.title("📈 Fixed Income Monitor")
    st.markdown("*Mobile-optimized market monitoring dashboard*")
    
    # Add debug info
    st.sidebar.header("🔧 Debug Info")
    st.sidebar.write(f"App loaded at: {datetime.now()}")
    st.sidebar.write("Status: Running")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Date range selector
        default_start = datetime(2023, 1, 1)  # More recent start date
        start_date = st.date_input("Start Date", default_start, max_value=datetime.now())
        end_date = st.date_input("End Date", datetime.now())
        
        # Convert to datetime
        start = datetime.combine(start_date, datetime.min.time())
        end = datetime.combine(end_date, datetime.min.time())
        
        # Chart selection
        st.header("Charts to Display")
        show_treasury = st.checkbox("Treasury Yields", True)
        show_spreads = st.checkbox("Treasury Spreads", True) 
        show_vix = st.checkbox("VIX Volatility", True)
        show_credit = st.checkbox("Credit Spreads", False)  # Default off for testing
        show_policy = st.checkbox("Economic Policy Uncertainty", False)  # Default off
        show_rates = st.checkbox("Short Rates (SOFR/EFFR)", False)  # Default off
        show_inflation = st.checkbox("Inflation Metrics", False)  # Default off
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Test section - always show something
    st.header("🧪 System Status")
    
    try:
        # Create a simple test chart
        test_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        test_data = pd.Series([1 + 0.1 * i for i in range(len(test_dates))], index=test_dates, name='Test Series')
        
        fig = create_plotly_chart(test_data, "System Test Chart", "Test Value")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.success("✅ Chart rendering system working!")
        else:
            st.error("❌ Chart rendering system failed")
    except Exception as e:
        st.error(f"System test failed: {str(e)}")
        st.write("Full error:", traceback.format_exc())
    
    # Data loading with progress
    data_loaded = False
    
    if show_treasury or show_spreads or show_vix or show_credit or show_policy or show_rates or show_inflation:
        st.header("📊 Market Data")
        
        with st.spinner('Loading market data...'):
            # Start with the most reliable data source - Treasury yields
            treasury_df = pd.DataFrame()
            if show_treasury:
                st.write("Fetching Treasury data...")
                dgs_ids = {
                    'DGS10': '10Y',  # Start with just 10Y for testing
                    'DGS2': '2Y',
                    'DGS5': '5Y',
                }
                treasury_parts = {}
                for sid, label in dgs_ids.items():
                    df = fred_get(sid, start, end)
                    if df is not None and not df.empty:
                        df = df.rename(columns={sid: label})
                        treasury_parts[label] = df[label]
                
                if treasury_parts:
                    treasury_df = pd.concat(treasury_parts, axis=1).sort_index()
                    data_loaded = True
                    st.success(f"✅ Loaded Treasury data: {treasury_df.shape}")
                else:
                    st.error("❌ Failed to load Treasury data")
            
            # Treasury spreads (only if treasury data loaded)
            spreads_data = {}
            if show_spreads and not treasury_df.empty:
                st.write("Computing Treasury spreads...")
                try:
                    if '10Y' in treasury_df.columns and '2Y' in treasury_df.columns:
                        spread = (treasury_df['10Y'] - treasury_df['2Y']).to_frame(name='2Y-10Y')
                        spreads_data['2Y-10Y Spread'] = spread
                        st.success("✅ Computed 2Y-10Y spread")
                except Exception as e:
                    st.error(f"❌ Error computing spreads: {str(e)}")
            
            # VIX data
            vix_series = None
            if show_vix:
                st.write("Fetching VIX data...")
                vix_series = get_vix_data(start, end)
                if vix_series is not None:
                    data_loaded = True
    
    # Display results
    if data_loaded:
        st.success("🎉 Data loading completed!")
        
        # Key metrics
        key_metrics = {}
        if not treasury_df.empty and '10Y' in treasury_df.columns:
            key_metrics['10Y Treasury'] = treasury_df['10Y']
        if spreads_data and '2Y-10Y Spread' in spreads_data:
            key_metrics['2Y-10Y Spread'] = spreads_data['2Y-10Y Spread'].iloc[:, 0]
        if vix_series is not None:
            key_metrics['VIX'] = vix_series
        
        if key_metrics:
            display_key_metrics(key_metrics)
        
        # Charts
        st.header("📈 Charts")
        
        # Treasury Yields
        if show_treasury and not treasury_df.empty:
            st.subheader("Treasury Yield Curve")
            fig = create_plotly_chart(treasury_df, "U.S. Treasury Yields", "Yield (%)")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Treasury Spreads  
        if show_spreads and spreads_data:
            st.subheader("Treasury Spreads")
            for name, spread_df in spreads_data.items():
                fig = create_plotly_chart(spread_df, f"U.S. {name}", "Spread (bp)")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # VIX
        if show_vix and vix_series is not None:
            st.subheader("Market Volatility")
            fig = create_plotly_chart(vix_series, "VIX - Implied Volatility", "VIX Level")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ No market data loaded. Try selecting different options or check your internet connection.")
        st.write("**Troubleshooting tips:**")
        st.write("- Check the sidebar settings")
        st.write("- Try a smaller date range")
        st.write("- Enable just one chart type first")
        st.write("- Use the refresh button")
    
    # Footer
    st.markdown("---")
    st.markdown("*Data sources: FRED (Federal Reserve Economic Data), Yahoo Finance*")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Debug section (expandable)
    with st.expander("🔍 Debug Information"):
        st.write("**Date Range:**", f"{start.date()} to {end.date()}")
        st.write("**Selected Options:**")
        st.write(f"- Treasury: {show_treasury}")
        st.write(f"- Spreads: {show_spreads}")
        st.write(f"- VIX: {show_vix}")
        st.write("**System Info:**")
        st.write(f"- Streamlit version: {st.__version__}")
        st.write(f"- Python path accessible: {True}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("🚨 Critical Error in Main App")
        st.write("Error details:", str(e))
        st.write("Full traceback:")
        st.code(traceback.format_exc())
