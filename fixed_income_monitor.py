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

# --- Page config ---
st.set_page_config(
    page_title="Fixed Income Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Dark mode CSS styling (Streamlit class names may change; this is best-effort) ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .block-container {
        background-color: #0e1117;
        color: #ffffff;
    }
    .css-1d391kg, .css-1offfwp { /* sidebar common classes */
        background-color: #262730 !important;
        color: #ffffff !important;
    }
    .stPlotlyChart {
        background-color: #0e1117;
    }
    .stMetric, .streamlit-expanderHeader, .stButton {
        color: #ffffff;
    }
    h1, h2, h3, p, div, label {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Utilities ---
@st.cache_data(ttl=86400, show_spinner=False)
def fred_get(series_id, start, end):
    """Fetch series from FRED and return a cleaned pandas Series indexed by datetime.
    Cached for 24 hours to satisfy the "update daily" requirement.
    """
    try:
        df = web.DataReader(series_id, 'fred', start, end)
        if df is None or df.empty:
            return None
        # Ensure single Series output
        s = df.iloc[:, 0].rename(series_id)
        # Convert index to DatetimeIndex and sort
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        return s
    except Exception as e:
        st.error(f"Failed to fetch FRED {series_id}: {e}")
        return None

@st.cache_data(ttl=86400, show_spinner=False)
def get_yahoo_data(ticker, start, end, name=None):
    """Download data from Yahoo Finance and return Close price as pandas Series.
    Cached for 24 hours.
    """
    try:
        # Use yfinance directly
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data is None or data.empty:
            return None
        if 'Close' not in data.columns:
            return None
        s = data['Close'].rename(name or ticker)
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        return s
    except Exception as e:
        st.error(f"Failed to fetch {ticker} from Yahoo: {e}")
        return None


def align_and_fill(series, start, end, freq='D'):
    """Reindex series to a fixed frequency between start and end and forward-fill.
    This ensures all charts use the exact same date range (2019-01-01 to latest).
    """
    if series is None:
        return None
    try:
        idx = pd.date_range(start=start, end=end, freq=freq)
        s = series.reindex(idx)
        # Forward fill then backward fill for leading NaNs
        s = s.fillna(method='ffill').fillna(method='bfill')
        return s
    except Exception as e:
        st.error(f"align_and_fill error: {e}")
        return series


def create_plotly_chart(df, title, ylabel='Value'):
    """Create a Plotly chart; accept Series or DataFrame. Uses dark theme colors.
    """
    if df is None:
        st.warning(f"No data for {title}")
        return None
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if df.empty:
        st.warning(f"Empty dataset for {title}")
        return None

    try:
        fig = go.Figure()
        colors = ['#00d4ff', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3']

        for i, col in enumerate(df.columns):
            series = df[col].dropna()
            if series.empty:
                continue
            fig.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                mode='lines',
                name=str(col),
                line=dict(color=colors[i % len(colors)], width=2.5),
                hovertemplate=f'<b>{col}</b><br>Date: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>'
            ))
            # latest annotation
            latest_val = series.iloc[-1]
            latest_date = series.index[-1]
            fig.add_annotation(x=latest_date, y=latest_val, text=f"{latest_val:.2f}", showarrow=True,
                               arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=colors[i % len(colors)],
                               font=dict(size=11, color=colors[i % len(colors)]), bgcolor='rgba(0,0,0,0.7)')

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color='white')),
            xaxis_title='Date', yaxis_title=ylabel,
            hovermode='x unified', showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='white')),
            margin=dict(l=50, r=50, t=70, b=50), height=450,
            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font=dict(color='white')
        )
        fig.update_xaxes(gridcolor='#2d2d2d', color='white')
        fig.update_yaxes(gridcolor='#2d2d2d', color='white')
        return fig
    except Exception as e:
        st.error(f"Error creating plotly chart {title}: {e}")
        return None


def create_dual_axis_chart(left_data, right_data, title, left_ylabel, right_ylabel):
    if left_data is None and right_data is None:
        return None
    try:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        colors_left = ['#00d4ff', '#ff6b6b']
        colors_right = ['#4ecdc4', '#feca57']

        if left_data is not None:
            if isinstance(left_data, pd.Series):
                left_data = left_data.to_frame()
            for i, col in enumerate(left_data.columns):
                s = left_data[col].dropna()
                if s.empty:
                    continue
                fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines', name=str(col),
                                         line=dict(color=colors_left[i % len(colors_left)], width=2.5)), secondary_y=False)
        if right_data is not None:
            if isinstance(right_data, pd.Series):
                right_data = right_data.to_frame()
            for i, col in enumerate(right_data.columns):
                s = right_data[col].dropna()
                if s.empty:
                    continue
                fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines', name=str(col),
                                         line=dict(color=colors_right[i % len(colors_right)], width=2.5)), secondary_y=True)

        fig.update_layout(title=dict(text=title, font=dict(size=18, color='white')),
                          xaxis_title='Date', hovermode='x unified', showlegend=True,
                          margin=dict(l=50, r=50, t=70, b=50), height=450,
                          plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font=dict(color='white'))
        fig.update_yaxes(title_text=left_ylabel, secondary_y=False, gridcolor='#2d2d2d', color='white')
        fig.update_yaxes(title_text=right_ylabel, secondary_y=True, color='white')
        fig.update_xaxes(gridcolor='#2d2d2d', color='white')
        return fig
    except Exception as e:
        st.error(f"Error creating dual-axis chart: {e}")
        return None


def display_key_metrics(data_dict):
    if not data_dict:
        st.warning('No key metrics available')
        return
    st.subheader('📊 Key Metrics (Latest Values)')
    cols = st.columns(3)
    i = 0
    for name, series in data_dict.items():
        if series is None or series.empty:
            continue
        try:
            s = series.dropna()
            if s.empty:
                continue
            latest_val = s.iloc[-1]
            latest_date = s.index[-1]
            date_str = latest_date.strftime('%Y-%m-%d')
            # Formatting heuristics
            if 'Spread' in name or 'Yield' in name or 'Rate' in name:
                value_str = f"{latest_val:.2f}%"
            elif 'Index' in name or 'EPU' in name or 'Policy' in name:
                value_str = f"{latest_val:.0f}"
            else:
                value_str = f"{latest_val:.2f}"
            with cols[i % 3]:
                st.metric(label=name, value=value_str, help=f"Latest: {date_str}")
            i += 1
        except Exception as e:
            st.error(f"Metric error {name}: {e}")


# --- Main app ---

def main():
    st.title("📈 Comprehensive Fixed Income Monitor")
    st.markdown("*Mobile-optimized dark theme market monitoring dashboard — data from 2019-01-01 to latest (auto-updates daily)*")

    # Fixed date range
    start = datetime(2019, 1, 1)
    end = datetime.now()

    st.info(f"📅 Data Range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} (Auto-updated daily)")

    # Sidebar
    with st.sidebar:
        st.header('⚙️ Settings')
        st.write(f"Start: {start.strftime('%Y-%m-%d')}")
        st.write(f"End: {end.strftime('%Y-%m-%d')}")
        st.write('Auto-refresh: Daily (cached for 24h)')
        show_policy = st.checkbox('Economic Policy Uncertainty', True)
        show_treasury = st.checkbox('Treasury Yield Curve', True)
        show_spreads = st.checkbox('Treasury Spreads', True)
        show_credit = st.checkbox('Credit Spreads', True)
        show_rates = st.checkbox('Short Rates (SOFR/EFFR)', True)
        show_repo = st.checkbox('Overnight Repo Rate', True)
        show_embi = st.checkbox('US Credit vs EM Stocks & Bonds', True)
        show_cpi = st.checkbox('CPI Inflation (YoY & MoM)', True)

        if st.button('🔄 Force Refresh All Data'):
            # Attempt to clear caches used above
            try:
                st.cache_data.clear()
            except Exception:
                try:
                    st.experimental_memo_clear()
                except Exception:
                    pass
            st.experimental_rerun()

    # --- Data loading ---
    with st.spinner('🔄 Loading data...'):
        # 1) Economic Policy Uncertainty
        epu = fred_get('USEPUINDXD', start, end) if show_policy else None
        if epu is not None:
            epu = align_and_fill(epu, start, end)

        # 2) Treasury yields
        treasury_df = pd.DataFrame()
        if show_treasury:
            dgs_ids = {'DGS1MO': '1M','DGS3MO': '3M','DGS6MO': '6M','DGS1': '1Y','DGS2': '2Y','DGS5': '5Y','DGS10': '10Y','DGS30': '30Y'}
            parts = {}
            for sid, label in dgs_ids.items():
                s = fred_get(sid, start, end)
                if s is not None:
                    s = align_and_fill(s, start, end)
                    s.name = label
                    parts[label] = s
            if parts:
                treasury_df = pd.concat(parts.values(), axis=1)
                treasury_df.columns = list(parts.keys())

        # 3) Treasury spreads
        spreads_data = {}
        if show_spreads and not treasury_df.empty:
            try:
                if '10Y' in treasury_df.columns and '2Y' in treasury_df.columns:
                    spreads_data['10Y-2Y Spread'] = (treasury_df['10Y'] - treasury_df['2Y'])
                if '10Y' in treasury_df.columns and '3M' in treasury_df.columns:
                    spreads_data['10Y-3M Spread'] = (treasury_df['10Y'] - treasury_df['3M'])
                if '30Y' in treasury_df.columns and '5Y' in treasury_df.columns:
                    spreads_data['30Y-5Y Spread'] = (treasury_df['30Y'] - treasury_df['5Y'])
            except Exception as e:
                st.error(f"Error computing spreads: {e}")

        # 4) Credit spreads
        credit_df = pd.DataFrame()
        if show_credit:
            credit_ids = {'BAMLC0A1CAAA': 'AAA','BAMLC0A4CBBB': 'BBB','BAMLH0A3HYC': 'CCC'}
            cs = {}
            for sid, label in credit_ids.items():
                s = fred_get(sid, start, end)
                if s is not None:
                    s = align_and_fill(s, start, end)
                    s.name = label
                    cs[label] = s
            if cs:
                credit_df = pd.concat(cs.values(), axis=1)
                credit_df.columns = list(cs.keys())

        # 5) Short rates
        rates_data = {}
        if show_rates:
            sofr = fred_get('SOFR', start, end)
            effr = fred_get('EFFR', start, end)
            if sofr is not None:
                rates_data['SOFR'] = align_and_fill(sofr, start, end)
            if effr is not None:
                rates_data['EFFR'] = align_and_fill(effr, start, end)

        # 6) Overnight repo
        repo_data = fred_get('RRPONTSYD', start, end) if show_repo else None
        if repo_data is not None:
            repo_data = align_and_fill(repo_data, start, end)

        # 7) US Credit vs EM Markets
        em_credit_data = {}
        if show_embi:
            us_hy_spread = fred_get('BAMLH0A0HYM2', start, end)
            if us_hy_spread is not None:
                em_credit_data['US HY Credit Spread'] = align_and_fill(us_hy_spread, start, end)
            # EM Stocks and Bonds
            em_stocks = get_yahoo_data('EEM', start, end, 'EEM')
            em_bonds = get_yahoo_data('EMB', start, end, 'EMB')
            # Defensive checks to avoid 'str' object is not callable or other type issues
            if em_stocks is not None and isinstance(em_stocks, pd.Series):
                em_stocks = align_and_fill(em_stocks, start, end)
                try:
                    em_stocks_norm = (em_stocks / em_stocks.iloc[0] - 1) * 100
                    em_credit_data['EM Stocks % Change'] = em_stocks_norm
                except Exception as e:
                    st.error(f"Error normalizing EM Stocks: {e}")
            if em_bonds is not None and isinstance(em_bonds, pd.Series):
                em_bonds = align_and_fill(em_bonds, start, end)
                try:
                    em_bonds_norm = (em_bonds / em_bonds.iloc[0] - 1) * 100
                    em_credit_data['EM Bonds % Change'] = em_bonds_norm
                except Exception as e:
                    st.error(f"Error normalizing EM Bonds: {e}")

        # 8) CPI
        cpi_data = {}
        if show_cpi:
            cpi_all = fred_get('CPIAUCSL', start, end)
            cpi_core = fred_get('CPILFESL', start, end)
            if cpi_all is not None:
                cpi_all = align_and_fill(cpi_all, start, end)
                cpi_yoy = cpi_all.pct_change(12) * 100
                cpi_mom = cpi_all.pct_change(1) * 100
                cpi_data['CPI YoY'] = cpi_yoy.rename('CPI YoY')
                cpi_data['CPI MoM'] = cpi_mom.rename('CPI MoM')
            if cpi_core is not None:
                cpi_core = align_and_fill(cpi_core, start, end)
                core_yoy = cpi_core.pct_change(12) * 100
                core_mom = cpi_core.pct_change(1) * 100
                cpi_data['Core CPI YoY'] = core_yoy.rename('Core CPI YoY')
                cpi_data['Core CPI MoM'] = core_mom.rename('Core CPI MoM')

    st.success('🎉 Data loading completed!')

    # --- Key metrics ---
    key_metrics = {}
    if epu is not None:
        key_metrics['Policy Uncertainty'] = epu
    if not treasury_df.empty and '10Y' in treasury_df.columns:
        key_metrics['10Y Treasury'] = treasury_df['10Y']
    if spreads_data and '10Y-2Y Spread' in spreads_data:
        key_metrics['10Y-2Y Spread'] = spreads_data['10Y-2Y Spread']
    if not credit_df.empty and 'BBB' in credit_df.columns:
        key_metrics['BBB Credit Spread'] = credit_df['BBB']
    if 'SOFR' in rates_data:
        key_metrics['SOFR'] = rates_data['SOFR']
    if 'CPI YoY' in cpi_data:
        key_metrics['CPI YoY'] = cpi_data['CPI YoY']

    if key_metrics:
        display_key_metrics(key_metrics)

    # --- Charts ---
    st.header('📊 Comprehensive Market Charts')

    if show_policy and epu is not None:
        st.subheader('📈 Economic Policy Uncertainty Index')
        fig = create_plotly_chart(epu.to_frame('EPU'), 'Economic Policy Uncertainty Index', 'Index Level')
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    if show_treasury and not treasury_df.empty:
        st.subheader('📊 U.S. Treasury Yield Curve')
        fig = create_plotly_chart(treasury_df, 'U.S. Treasury Yields (1M to 30Y)', 'Yield (%)')
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    if show_spreads and spreads_data:
        st.subheader('📏 Treasury Yield Spreads')
        for name, s in spreads_data.items():
            fig = create_plotly_chart(s.to_frame(name), f'U.S. {name}', 'Spread (bps)')
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    if show_credit and not credit_df.empty:
        st.subheader('🏦 Credit Spreads vs Treasury')
        fig = create_plotly_chart(credit_df, 'Credit Spreads (AAA, BBB, CCC)', 'OAS Spread (bps)')
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    if show_rates and rates_data:
        st.subheader('💰 Short-term Interest Rates')
        for name, s in rates_data.items():
            fig = create_plotly_chart(s.to_frame(name), f'{name} Rate', 'Rate (%)')
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    if show_repo and repo_data is not None:
        st.subheader('🔄 Overnight Reverse Repo Operations')
        fig = create_plotly_chart(repo_data.to_frame('ON RRP'), 'ON RRP Operations Volume', 'Amount')
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    if show_embi and em_credit_data:
        st.subheader('🌍 US Credit Spread vs Emerging Market Performance')
        credit_spread = em_credit_data.get('US HY Credit Spread')
        em_perf_items = {k: v for k, v in em_credit_data.items() if 'EM' in k}
        if em_perf_items:
            em_perf_df = pd.concat(list(em_perf_items.values()), axis=1)
            em_perf_df.columns = list(em_perf_items.keys())
            fig = create_dual_axis_chart(credit_spread, em_perf_df, 'US Credit Risk vs Emerging Market Performance', 'Credit Spread (bps)', 'EM Performance (% Change)')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.info('💡 Chart Logic: When US credit spreads widen, EM assets often underperform (negative correlation)')
        else:
            if credit_spread is not None:
                fig = create_plotly_chart(credit_spread.to_frame('US HY Credit Spread'), 'US High Yield Credit Spread', 'Spread (bps)')
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    if show_cpi and cpi_data:
        st.subheader('📊 Consumer Price Index (CPI)')
        if 'CPI YoY' in cpi_data and 'Core CPI YoY' in cpi_data:
            yoy_df = pd.concat([cpi_data['CPI YoY'], cpi_data['Core CPI YoY']], axis=1)
            yoy_df.columns = ['CPI YoY', 'Core CPI YoY']
            fig = create_plotly_chart(yoy_df, 'CPI Year-over-Year', 'YoY Change (%)')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        if 'CPI MoM' in cpi_data and 'Core CPI MoM' in cpi_data:
            mom_df = pd.concat([cpi_data['CPI MoM'], cpi_data['Core CPI MoM']], axis=1)
            mom_df.columns = ['CPI MoM', 'Core CPI MoM']
            fig = create_plotly_chart(mom_df, 'CPI Month-over-Month', 'MoM Change (%)')
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown('---')
    st.markdown('**📊 Data Sources:** FRED • Yahoo Finance (EEM, EMB) • Policy Uncertainty')
    st.markdown(f"*🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")


if __name__ == '__main__':
    main()
