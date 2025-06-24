import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
from io import StringIO

sns.set_palette("husl")
plt.style.use('seaborn-v0_8')

@st.cache_data
def load_merged_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['Timestamp IST','Date'], dayfirst=True)
    else:
        df = pd.read_csv('output/merged_trader_sentiment_data.csv', parse_dates=['Timestamp IST','Date'], dayfirst=True)
    if 'Timestamp IST' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp IST'], errors='coerce')
    else:
        time_cols = [c for c in df.columns if 'time' in c.lower()]
        if time_cols:
            df['Timestamp'] = pd.to_datetime(df[time_cols[0]], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Hour'] = df['Timestamp'].dt.hour
    return df

@st.cache_data
def load_trader_metrics():
    try:
        dfm = pd.read_csv('output/trader_performance_metrics.csv')
        return dfm
    except:
        return None

def filter_dataframe(df, date_range, sentiments, accounts):
    df2 = df.copy()
    if date_range is not None:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df2 = df2[(df2['Date'] >= start) & (df2['Date'] <= end)]
    if sentiments:
        df2 = df2[df2['sentiment'].isin(sentiments)]
    if accounts:
        df2 = df2[df2['Account'].isin(accounts)]
    return df2

# Streamlit UI
st.set_page_config(page_title="Trader Sentiment Dashboard", layout="wide")
st.title("Trader Behavior & Market Sentiment Dashboard")

# Sidebar
st.sidebar.header("Data Upload & Filters")
uploaded = st.sidebar.file_uploader("Upload merged CSV", type="csv")
try:
    df = load_merged_data(uploaded)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)
if len(date_range) != 2:
    st.sidebar.error("Select start and end dates")
sent_list = df['sentiment'].dropna().unique().tolist()
sel_sent = st.sidebar.multiselect("Select Sentiments", sent_list, default=sent_list)
if st.sidebar.checkbox("Filter by Account", False):
    acct_list = df['Account'].unique().tolist()
    sel_acct = st.sidebar.multiselect("Select Accounts (few)", acct_list)
else:
    sel_acct = []
df_filt = filter_dataframe(df, date_range, sel_sent, sel_acct)

# Main: Summary metrics
st.header("Summary Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Trades", f"{len(df_filt):,}")
with col2:
    avg_pnl = df_filt['Closed PnL'].mean() if 'Closed PnL' in df_filt else np.nan
    st.metric("Avg PnL per Trade", f"{avg_pnl:.2f}")
with col3:
    avg_vol = df_filt['Size USD'].mean() if 'Size USD' in df_filt else np.nan
    st.metric("Avg Trade Size", f"${avg_vol:,.2f}")

# Distribution of sentiments
st.subheader("Market Sentiment Distribution")
sent_counts = df_filt['sentiment'].value_counts()
st.bar_chart(sent_counts)

# Avg PnL by sentiment\st.subheader("Average PnL by Sentiment")
pnl_by_sent = df_filt.groupby('sentiment')['Closed PnL'].mean()
st.bar_chart(pnl_by_sent)

# Volume by sentiment
st.subheader("Total Volume by Sentiment")
vol_by_sent = df_filt.groupby('sentiment')['Size USD'].sum()
st.bar_chart(vol_by_sent)

# Fear & Greed index over time\st.subheader("Fear & Greed Index Over Time")
daily_fg = df_filt.groupby('Date')['fear_greed_value'].first().reset_index()
chart_data = daily_fg.set_index('Date')
st.line_chart(chart_data['fear_greed_value'])

# Trading activity heatmap
st.subheader("Trading Activity Heatmap: Hour vs Sentiment")
heat = df_filt.groupby(['Hour','sentiment']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(heat.T, annot=True, fmt='d', cbar_kws={'label':'Trades'}, ax=ax)
ax.set_xlabel('Hour of Day'); ax.set_ylabel('Sentiment')
st.pyplot(fig)

# Cumulative PnL by sentiment over time
st.subheader("Cumulative PnL by Sentiment Over Time")
fig2, ax2 = plt.subplots(figsize=(8,4))
for sent in df_filt['sentiment'].dropna().unique():
    sub = df_filt[df_filt['sentiment']==sent].sort_values('Timestamp')
    sub['Cumulative_PnL'] = sub['Closed PnL'].cumsum()
    ax2.plot(sub['Timestamp'], sub['Cumulative_PnL'], label=sent)
ax2.legend(); ax2.set_xlabel('Time'); ax2.set_ylabel('Cumulative PnL')
st.pyplot(fig2)

# Trade size distribution
st.subheader("Trade Size Distribution by Sentiment")
fig3, ax3 = plt.subplots(figsize=(8,4))
for sent in df_filt['sentiment'].dropna().unique():
    sub = df_filt[df_filt['sentiment']==sent]
    ax3.hist(sub['Size USD'].dropna(), bins=50, alpha=0.5, density=True, label=sent)
ax3.set_yscale('log'); ax3.set_xlabel('Size USD'); ax3.set_ylabel('Density'); ax3.legend()
st.pyplot(fig3)

# Win rate by sentiment
st.subheader("Win Rate by Sentiment")
win_rates = {}
for sent in df_filt['sentiment'].dropna().unique():
    sub = df_filt[df_filt['sentiment']==sent]
    total = len(sub); wins = (sub['Closed PnL']>0).sum()
    rate = wins/total*100 if total>0 else 0
    win_rates[sent] = rate
st.bar_chart(pd.Series(win_rates))

# Trader clustering (on precomputed metrics or compute on fly)
st.header("Trader Clustering")
tm = load_trader_metrics()
if tm is not None:
    st.write("Sample Trader Metrics:")
    st.dataframe(tm.head())
    # Select features
    feat_cols = [c for c in ['Win_Rate','Avg_Trade_Size','Total_PnL','Avg_Fear_Greed_Value'] if c in tm.columns]
    if feat_cols:
        dfm = tm.dropna(subset=feat_cols)
        X = StandardScaler().fit_transform(dfm[feat_cols])
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)
        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)
        dfm['Cluster'] = labels
        st.write(f"Clustering into {n_clusters} groups")
        st.dataframe(dfm.head())
        # PCA plot
        pca = PCA(n_components=2, random_state=42).fit_transform(X)
        fig4, ax4 = plt.subplots(figsize=(6,4))
        scatter = ax4.scatter(pca[:,0], pca[:,1], c=labels, cmap='tab10', s=20)
        ax4.set_title('PCA of Trader Metrics'); ax4.set_xlabel('PC1'); ax4.set_ylabel('PC2')
        st.pyplot(fig4)
else:
    st.info("Trader metrics not available; run initial analysis first.")

# Time-lag correlation
st.header("Time-lag Correlation between Sentiment and Avg Daily PnL")
daily = df.groupby('Date').agg(avg_pnl=('Closed PnL','mean'), avg_fg=('fear_greed_value','mean')).dropna()
max_lag = st.sidebar.slider("Max lag days", 1, 30, 10)
lags = []
for lag in range(-max_lag, max_lag+1):
    if lag<0: corr = daily['avg_fg'].shift(-lag).corr(daily['avg_pnl'])
    else: corr = daily['avg_fg'].shift(lag).corr(daily['avg_pnl'])
    lags.append(corr)
fig5, ax5 = plt.subplots(figsize=(6,3))
ax5.bar(range(-max_lag, max_lag+1), lags)
ax5.set_title('Cross-correlation'); ax5.set_xlabel('Lag (days)'); ax5.set_ylabel('Correlation')
st.pyplot(fig5)

# Event study around extremes
st.header("Event Study: Avg PnL around Extreme Sentiment Days")
window = st.sidebar.slider("Window days", 1, 30, 5)
ext_fear = daily[daily['avg_fg']<=25]
ext_greed = daily[daily['avg_fg']>=75]
def compute_event(df_ext):
    rec=[]
    for d in df_ext.index:
        for off in range(-window, window+1):
            day = d + pd.Timedelta(days=off)
            if day in daily.index:
                rec.append({'offset':off, 'avg_pnl':daily.at[day,'avg_pnl']})
    return pd.DataFrame(rec).groupby('offset')['avg_pnl'].mean().reset_index()
res_f = compute_event(ext_fear)
res_g = compute_event(ext_greed)
fig6, ax6 = plt.subplots(figsize=(6,3))
ax6.plot(res_f['offset'], res_f['avg_pnl'], label='Extreme Fear')
ax6.plot(res_g['offset'], res_g['avg_pnl'], label='Extreme Greed')
ax6.axvline(0, linestyle='--'); ax6.set_xlabel('Days relative'); ax6.set_ylabel('Avg PnL'); ax6.legend(); ax6.set_title('Event Study')
st.pyplot(fig6)


st.header("Predictive Model for Trade Profitability")
if st.sidebar.checkbox("Run predictive model", False):
    dfp = df.dropna(subset=['Closed PnL','Size USD','Execution Price','fear_greed_value','Timestamp'])
    dfp['target'] = (dfp['Closed PnL']>0).astype(int)
    dfp['Hour'] = dfp['Timestamp'].dt.hour
    features = ['Size USD','Execution Price','fear_greed_value','Hour']
    dfm = dfp[features+['target']].dropna()
    X = dfm[features]; y = dfm['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix")
    st.write(cm)
else:
    st.info("Enable predictive model in sidebar to run")


st.header("Forecasting Avg Daily PnL (SARIMAX)")
if st.sidebar.checkbox("Run forecasting", False):
    try:
        df_daily = daily
        model = SARIMAX(df_daily['avg_pnl'], exog=df_daily[['avg_fg']], order=(1,1,1)).fit(disp=False)
       
        last_exog = df_daily[['avg_fg']].iloc[-10:]
        forecast = model.forecast(steps=10, exog=last_exog)
        st.write("Forecast for next 10 periods:")
        st.line_chart(forecast)
    except Exception as e:
        st.error(f"Forecast error: {e}")
else:
    st.info("Enable forecasting in sidebar to run")


st.header("Anomaly Detection")
if st.sidebar.checkbox("Run anomaly detection", False):
    df_an = df.dropna(subset=['Size USD','Closed PnL','Execution Price'])
    iso = IsolationForest(contamination=0.02, random_state=42)
    df_an['anomaly'] = iso.fit_predict(df_an[['Size USD','Closed PnL','Execution Price']])
    n_anom = (df_an['anomaly']==-1).sum()
    st.write(f"Detected {n_anom} anomalies")
    if st.checkbox("Show anomalies", False):
        st.dataframe(df_an[df_an['anomaly']==-1].head(50))
else:
    st.info("Enable anomaly detection in sidebar to run")


st.header("Sentiment Transitions Impact on PnL")
if st.sidebar.checkbox("Show sentiment transitions", False):
    df_st = df.sort_values('Timestamp')
    df_st['prev_sent'] = df_st['sentiment'].shift(1)
    df_st['change'] = df_st['sentiment'] != df_st['prev_sent']
    trans = df_st[df_st['change']].dropna(subset=['prev_sent','sentiment'])
    res = trans.groupby(['prev_sent','sentiment'])['Closed PnL'].mean().reset_index()
    res.columns = ['From','To','Avg_PnL']
    st.dataframe(res)
else:
    st.info("Enable sentiment transitions in sidebar to view")

st.write("---")
st.write("Dashboard powered by Trader Sentiment Analysis pipeline.")
