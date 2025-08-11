# Fixed Income Full Monitor (2019-01-01 to latest)
# Requirements: pip install pandas pandas_datareader yfinance matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import yfinance as yf
from datetime import datetime

# ---------------- Settings ----------------
start = datetime(2019, 1, 1)
end = datetime.today()  # latest available
pd.options.plotting.backend = "matplotlib"

# ---------------- Helper ----------------
def fred_get(series_id, start, end):
    """Safe fetch from FRED. Returns DataFrame or None."""
    try:
        df = web.DataReader(series_id, 'fred', start, end)
        print(f"[OK] FRED: {series_id}  rows={len(df)}")
        return df
    except Exception as e:
        print(f"[WARN] Failed to fetch {series_id} from FRED: {e}")
        return None

def plot_single(df, title, ylabel='Value', date_format=True):
    """Plot a single-series or multi-column dataframe/series with English labels and show latest value label."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        print(f"[SKIP] No data for: {title}")
        return
    plt.figure(figsize=(12, 4))
    # If series, convert to DataFrame for uniform handling
    if isinstance(df, pd.Series):
        df = df.to_frame()

    for col in df.columns:
        plt.plot(df.index, df[col], linewidth=0.8, label=str(col))
        # 標示最新日期的數值
        latest_date = df.index[-1]
        latest_val = df[col].iloc[-1]
        plt.text(latest_date + pd.Timedelta(days=3), latest_val,
                 f"{latest_val:.3f}", fontsize=8, color='black',
                 verticalalignment='center')

    plt.title(f"Spread Monitor - {title}", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

# ---------------- 1) Policy / Sentiment ----------------
# Economic Policy Uncertainty (daily)
epu = fred_get('USEPUINDXD', start, end)  # policyuncertainty.com series (daily)

# ---------------- 2) Treasury yields (various tenors) ----------------
dgs_ids = {
    'DGS1MO':'1M', 'DGS3MO':'3M', 'DGS6MO':'6M',
    'DGS1':'1Y', 'DGS2':'2Y', 'DGS5':'5Y',
    'DGS10':'10Y', 'DGS30':'30Y'
}
treasury_parts = {}
for sid, label in dgs_ids.items():
    df = fred_get(sid, start, end)
    if df is not None:
        # ensure column name is label
        df = df.rename(columns={sid: label})
        treasury_parts[label] = df[label]
if treasury_parts:
    treasury_df = pd.concat(treasury_parts, axis=1).sort_index()
else:
    treasury_df = pd.DataFrame()

# ---------------- 3) Treasury spreads (computed) ----------------
# compute common spreads if underlying yields available
ust_2y_10y = None
if '10Y' in treasury_df.columns and '2Y' in treasury_df.columns:
    ust_2y_10y = (treasury_df['10Y'] - treasury_df['2Y']).to_frame(name='2Y-10Y Spread')
# other spreads
ust_3m_10y = None
if '10Y' in treasury_df.columns and '3M' in treasury_df.columns:
    ust_3m_10y = (treasury_df['10Y'] - treasury_df['3M']).to_frame(name='3M-10Y Spread')
ust_5y_30y = None
if '30Y' in treasury_df.columns and '5Y' in treasury_df.columns:
    ust_5y_30y = (treasury_df['30Y'] - treasury_df['5Y']).to_frame(name='5Y-30Y Spread')

# ---------------- 4) Credit spreads (ICE BofA series on FRED) ----------------
aaa = fred_get('BAMLC0A1CAAA', start, end)   # AAA OAS
bbb = fred_get('BAMLC0A4CBBB', start, end)   # BBB OAS
ccc = fred_get('BAMLH0A3HYC', start, end)    # CCC OAS (high-yield)
# Concatenate available credit series
credit_list = []
if aaa is not None:
    credit_list.append(aaa.rename(columns={'BAMLC0A1CAAA':'AAA'}))
if bbb is not None:
    credit_list.append(bbb.rename(columns={'BAMLC0A4CBBB':'BBB'}))
if ccc is not None:
    credit_list.append(ccc.rename(columns={'BAMLH0A3HYC':'CCC'}))
if credit_list:
    credit_df = pd.concat(credit_list, axis=1)
else:
    credit_df = pd.DataFrame()

# ---------------- 5) Funding / short-rate / liquidity ----------------
sofr = fred_get('SOFR', start, end)   # SOFR
effr = fred_get('EFFR', start, end)   # Effective Federal Funds Rate
rrp = fred_get('RRPONTSYD', start, end)

# ---------------- 6) Volatility / sentiment ----------------
try:
    vix_df = yf.download("^VIX", start=start, end=end, progress=False)
    vix_series = vix_df['Close'].rename('VIX')
    print(f"[OK] Yahoo: VIX rows={len(vix_series)}")
except Exception as e:
    print(f"[WARN] Failed to fetch VIX from Yahoo: {e}")
    vix_series = None

# ---------------- 7) Additional useful series (CPI & breakevens) ----------------
cpi = fred_get('CPIAUCSL', start, end)      # CPI (monthly)
core_cpi = fred_get('CPILFESL', start, end) # Core CPI (monthly)
t5y_be = fred_get('T5YIE', start, end)      # 5-yr breakeven
t10y_be = fred_get('T10YIE', start, end)    # 10-yr breakeven

# ---------------- Plot all charts (each indicator one chart) ----------------
plot_single(epu['USEPUINDXD'] if epu is not None else None, "Economic Policy Uncertainty Index", "Index")
plot_single(treasury_df if not treasury_df.empty else None, "Treasury Yields (1M,3M,6M,1Y,2Y,5Y,10Y,30Y)", "Yield (%)")
plot_single(ust_2y_10y, "U.S. 2Yr/10Yr Spread", "Yield spread")
plot_single(ust_3m_10y, "U.S. 3M/10Yr Spread", "Yield spread")
plot_single(ust_5y_30y, "U.S. 5Y/30Y Spread", "Yield spread")
plot_single(credit_df if not credit_df.empty else None, "Credit Spreads (AAA, BBB, CCC)", "OAS (bp or pct)")
plot_single(sofr['SOFR'] if sofr is not None else None, "SOFR (Secured Overnight Financing Rate)", "Rate")
plot_single(effr['EFFR'] if effr is not None else None, "Effective Federal Funds Rate", "Rate")
plot_single(rrp['RRPONTSYD'] if rrp is not None else None, "ON RRP (usage)", "Amount")
plot_single(vix_series, "VIX Index (implied volatility)", "VIX")
plot_single(cpi['CPIAUCSL'] if cpi is not None else None, "CPI (Headline)", "Index")
plot_single(core_cpi['CPILFESL'] if core_cpi is not None else None, "Core CPI", "Index")
plot_single(t5y_be['T5YIE'] if t5y_be is not None else None, "5Y Breakeven Inflation", "Rate")
plot_single(t10y_be['T10YIE'] if t10y_be is not None else None, "10Y Breakeven Inflation", "Rate")

print("Done. Notes: Some swap curve / OIS series are not freely available via FRED; for full OIS/IRS tenors use vendor APIs (Bloomberg/Refinitiv/Xignite).")
