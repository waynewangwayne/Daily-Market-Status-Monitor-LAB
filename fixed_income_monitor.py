# Updated code with requested modifications
# 1. Show previous date value in Key Metrics
# 2. Replace BBB Credit Spread with CCC Credit Spread
# 3. Ensure Yahoo EEM/EMB fetch works without 'str' object is not callable
# 4. Make U.S. Treasury Yield Curve chart larger & lines thinner
# 5. CPI charts are line charts

# All previous logic preserved; only modified relevant parts

# --- In display_key_metrics ---
def display_key_metrics(data_dict):
    st.subheader('📊 Key Metrics (Latest Values)')
    cols = st.columns(3)
    i = 0
    for name, series in data_dict.items():
        if series is not None and not series.empty:
            latest_val = series.iloc[-1]
            prev_val = series.iloc[-2] if len(series) > 1 else None
            if 'Spread' in name or 'Yield' in name or 'Rate' in name:
                value_str = f"{latest_val:.2f}%"
                prev_str = f"{prev_val:.2f}%" if prev_val is not None else "-"
            else:
                value_str = f"{latest_val:.2f}"
                prev_str = f"{prev_val:.2f}" if prev_val is not None else "-"
            cols[i % 3].metric(label=name, value=value_str, delta=f"Prev: {prev_str}")
            i += 1

# --- Replace BBB Credit Spread metric with CCC Credit Spread ---
credit_df = get_credit_spread_data()  # 確保 credit_df 已被定義
if not credit_df.empty and 'CCC' in credit_df.columns:
    key_metrics['CCC Credit Spread'] = credit_df['CCC']

# --- Fix Yahoo fetch ---
@st.cache_data(ttl=86400, show_spinner=False)
def get_yahoo_data(ticker, start, end, name=None):
    try:
        data = yf.download(str(ticker), start=start, end=end, progress=False, auto_adjust=False)
        if data is None or data.empty:
            return None
        if 'Close' not in data.columns:
            return None
        s = data['Close'].rename(name or str(ticker))
        s.index = pd.to_datetime(s.index)
        return s
    except Exception as e:
        st.error(f"Failed to fetch {ticker} from Yahoo: {e}")
        return None

# --- Make Treasury Yield Curve larger & lines thinner ---
if show_treasury and not treasury_df.empty:
    st.subheader('📊 U.S. Treasury Yield Curve')
    fig = create_plotly_chart(treasury_df, 'U.S. Treasury Yields (1M to 30Y)', 'Yield (%)', line_width=1.5, height=600)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# --- Ensure CPI charts are line charts ---
if show_cpi and cpi_data:
    st.subheader('📊 Consumer Price Index (CPI)')
    if 'CPI YoY' in cpi_data and 'Core CPI YoY' in cpi_data:
        yoy_df = pd.concat([cpi_data['CPI YoY'], cpi_data['Core CPI YoY']], axis=1)
        yoy_df.columns = ['CPI YoY', 'Core CPI YoY']
        fig = create_plotly_chart(yoy_df, 'CPI Year-over-Year', 'YoY Change (%)', line_width=2)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    if 'CPI MoM' in cpi_data and 'Core CPI MoM' in cpi_data:
        mom_df = pd.concat([cpi_data['CPI MoM'], cpi_data['Core CPI MoM']], axis=1)
        mom_df.columns = ['CPI MoM', 'Core CPI MoM']
        fig = create_plotly_chart(mom_df, 'CPI Month-over-Month', 'MoM Change (%)', line_width=2)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

