"""
============================================================
ARMENIAN POLITICAL BUSINESS CYCLE — TIME SERIES PROJECT
Masters in Applied Statistics | Group Project
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
import logging
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — edit here, not inside functions
# ============================================================
CONFIG = {
    'paths': {
        'cpi':  'EF-cpi-mi1_20260521-185701.xlsx',
        'm2':   '4_Aggregates_eng.xlsx',
        'rate': '10_bank_reference_rate_eng.xlsx',
        'gov':  'EF-gf-3_20260521-185737.xlsx',
    },
    'date_range':        ('2008-01-01', '2025-12-31'),
    'train_test_split':  '2022-01-01',
    'lag_range':         6,
    'forecast_horizon':  8,
    'election_dates': [
        '2008-04-01',
        '2012-04-01',
        '2017-04-01',
        '2021-04-01',
        '2026-04-01',
    ],
    'structural_break_years': [2018, 2020],
}

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('project.log', mode='w'),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ============================================================
# STEP 1 — DATA LOADING & CLEANING
# ============================================================

def load_cpi(path=None):
    path = path or CONFIG['paths']['cpi']
    df = pd.read_excel(path, sheet_name=0, header=1, index_col=0)
    year_vals = df.iloc[0]
    df.columns = [str(int(float(v))) if pd.notna(v) and str(v).replace('.0', '').isdigit()
                  else str(v) for v in year_vals]
    df = df.iloc[1:]
    df = df.loc[['January','February','March','April','May','June',
                 'July','August','September','October','November','December']]
    year_cols = [c for c in df.columns if c.isdigit() and 2003 <= int(c) <= 2025]
    df = df[year_cols]

    records = []
    month_map = {'January':1,'February':2,'March':3,'April':4,
                 'May':5,'June':6,'July':7,'August':8,
                 'September':9,'October':10,'November':11,'December':12}
    for year in year_cols:
        for month_name, month_num in month_map.items():
            try:
                val = float(df.loc[month_name, year])
                records.append({'date': pd.Timestamp(year=int(year), month=month_num, day=1),
                                'CPI_mom': val})
            except Exception:
                pass

    cpi = pd.DataFrame(records).sort_values('date').set_index('date')
    cpi['CPI_mom'] = pd.to_numeric(cpi['CPI_mom'], errors='coerce')
    cpi.loc[cpi['CPI_mom'] == 0, 'CPI_mom'] = np.nan
    cpi['CPI_index'] = cpi['CPI_mom'].div(100).cumprod() * 100
    log.info(f"CPI loaded: {len(cpi)} monthly obs "
             f"({cpi.index.min().date()} to {cpi.index.max().date()})")
    return cpi[['CPI_index']]


def load_m2(path=None):
    import datetime as _dt
    path = path or CONFIG['paths']['m2']
    df = pd.read_excel(path, sheet_name=0, header=3)
    df.columns = ['date_serial','Currency','Demand_AMD','M1',
                  'Time_AMD','M2','Deposits_FX','M2X']
    df = df[df['date_serial'].apply(lambda x: isinstance(x, _dt.datetime))]
    df['date'] = pd.to_datetime(df['date_serial'])
    df = df.set_index('date')[['M2']].sort_index()
    df['M2'] = pd.to_numeric(df['M2'], errors='coerce')
    df = df.dropna()
    log.info(f"M2 loaded: {len(df)} monthly obs "
             f"({df.index.min().date()} to {df.index.max().date()})")
    return df


def load_rate(path=None):
    import datetime as _dt
    path = path or CONFIG['paths']['rate']
    df = pd.read_excel(path, sheet_name=0, header=2)
    df.columns = ['date_serial', 'rate']
    df = df[df['date_serial'].apply(lambda x: isinstance(x, _dt.datetime))]
    df['date'] = pd.to_datetime(df['date_serial'])
    df = df.set_index('date')[['rate']].sort_index()
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    df = df.dropna()
    log.info(f"Rate loaded: {len(df)} monthly obs "
             f"({df.index.min().date()} to {df.index.max().date()})")
    return df


def load_gov(path=None):
    path = path or CONFIG['paths']['gov']
    df = pd.read_excel(path, sheet_name=0, header=0, index_col=0)
    year_row = df.iloc[1]
    df.columns = [str(int(float(v))) if pd.notna(v) and str(v).replace('.0', '').isdigit()
                  else str(v) for v in year_row]
    df = df.iloc[2:]
    # FIX 1: safely coerce index to string before filtering
    df.index = df.index.map(lambda x: str(x) if pd.notna(x) else '')
    row = df[df.index.str.contains('Total expenditures', na=False)].iloc[0]
    gov = pd.DataFrame({'GOV_EXP': row})
    gov.index = pd.to_datetime(gov.index.astype(str) + '-01-01')
    gov['GOV_EXP'] = pd.to_numeric(gov['GOV_EXP'], errors='coerce')
    gov = gov.dropna()
    log.info(f"Gov expenditure loaded: {len(gov)} annual obs "
             f"({gov.index.min().year} to {gov.index.max().year})")
    return gov


def make_election_dummy(start=None, end=None):
    start = start or CONFIG['date_range'][0]
    end   = end   or CONFIG['date_range'][1]
    idx   = pd.date_range(start=start, end=end, freq='QS')
    dummy = pd.Series(0, index=idx, name='ELECTION')
    for ed in CONFIG['election_dates']:
        eq   = pd.Timestamp(ed)
        prev = eq - pd.DateOffset(months=3)
        dummy[dummy.index == eq]   = 1
        dummy[dummy.index == prev] = 1
    return dummy.to_frame()


# ============================================================
# STEP 2 — MERGE INTO QUARTERLY DATASET
# ============================================================

def build_quarterly_dataset():
    cpi  = load_cpi()
    m2   = load_m2()
    rate = load_rate()
    gov  = load_gov()

    cpi_q  = cpi.resample('QS').mean()
    m2_q   = m2.resample('QS').mean()
    rate_q = rate.resample('QS').mean()
    gov_q  = gov.resample('QS').asfreq().interpolate(method='linear')
    elec   = make_election_dummy()

    data = (cpi_q
            .join(m2_q,   how='outer')
            .join(rate_q, how='outer')
            .join(gov_q,  how='outer')
            .join(elec,   how='left'))

    start, end = CONFIG['date_range']
    data = data.loc[start:end]
    data['ELECTION'] = data['ELECTION'].fillna(0)

    log.info(f"Final dataset: {data.shape[0]} quarters x {data.shape[1]} variables")
    log.info(f"\n{data.tail()}")
    data.to_csv('armenia_quarterly.csv')
    log.info("Saved: armenia_quarterly.csv")
    return data


# ============================================================
# STEP 2b — SEASONAL DECOMPOSITION  (item 6)
# ============================================================

def plot_seasonal_decomposition(data):
    from statsmodels.tsa.seasonal import seasonal_decompose
    series = data['CPI_index'].dropna()
    result = seasonal_decompose(series, model='multiplicative', period=4)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('CPI Seasonal Decomposition (Multiplicative, period=4)',
                 fontsize=14, fontweight='bold')
    components = [
        (series,          'Observed',  '#2563eb'),
        (result.trend,    'Trend',     '#16a34a'),
        (result.seasonal, 'Seasonal',  '#9333ea'),
        (result.resid,    'Residual',  '#dc2626'),
    ]
    for ax, (comp, label, color) in zip(axes, components):
        ax.plot(comp.index, comp, color=color, linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig('00_decomposition.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Saved: 00_decomposition.png")
    return result


# ============================================================
# STEP 3 — EDA & OUTLIER DETECTION
# ============================================================

def plot_series_with_elections(data):
    election_quarters = data[data['ELECTION'] == 1].index
    vars_to_plot = ['CPI_index', 'M2', 'GOV_EXP', 'rate']
    titles = ['CPI Index (2003=100)', 'M2 Money Supply (mln AMD)',
              'Government Expenditure (mln AMD)', 'CBA Reference Rate (%)']
    colors = ['#2563eb', '#16a34a', '#dc2626', '#9333ea']

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle('Armenian Macroeconomic Variables with Election Periods',
                 fontsize=16, fontweight='bold', y=0.98)
    for ax, var, title, color in zip(axes, vars_to_plot, titles, colors):
        ax.plot(data.index, data[var], color=color, linewidth=2)
        for eq in election_quarters:
            ax.axvline(eq, color='red', alpha=0.3, linewidth=1.5, linestyle='--')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[0].axvline(pd.Timestamp('2008-01-01'), color='red', alpha=0.3,
                    linewidth=1.5, linestyle='--', label='Election period')
    axes[0].legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('01_time_series_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Saved: 01_time_series_plot.png")


def detect_outliers(data):
    log.info("=== OUTLIER DETECTION (Z-score > 3) ===")
    for var in ['CPI_index', 'M2', 'GOV_EXP', 'rate']:
        s = data[var].dropna()
        z = np.abs((s - s.mean()) / s.std())
        out = s[z > 3]
        if len(out):
            log.info(f"{var}: {len(out)} outlier(s) at {out.index.tolist()}")
        else:
            log.info(f"{var}: no outliers")


def plot_scatter_matrix(data):
    vars_to_plot = ['CPI_index', 'M2', 'GOV_EXP', 'rate']
    plot_data = data[vars_to_plot].dropna()
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle('Scatter Plot Matrix — Armenian Macro Variables',
                 fontsize=14, fontweight='bold')
    election_mask = data['ELECTION'] == 1
    for i, var1 in enumerate(vars_to_plot):
        for j, var2 in enumerate(vars_to_plot):
            ax = axes[i][j]
            if i == j:
                ax.hist(plot_data[var1], bins=20, color='#2563eb', alpha=0.7)
                ax.set_title(var1, fontsize=9)
            else:
                ax.scatter(plot_data[var2], plot_data[var1],
                           alpha=0.5, s=20, color='#6b7280')
                elec_data = plot_data[election_mask.reindex(plot_data.index, fill_value=False)]
                ax.scatter(elec_data[var2], elec_data[var1],
                           alpha=0.8, s=40, color='red')
            if i == 3: ax.set_xlabel(var2, fontsize=8)
            if j == 0: ax.set_ylabel(var1, fontsize=8)
    plt.tight_layout()
    plt.savefig('02_scatter_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Saved: 02_scatter_matrix.png")


def plot_correlation_heatmap(data):
    corr = data[['CPI_index', 'M2', 'GOV_EXP', 'rate', 'ELECTION']].dropna().corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, linewidths=0.5, annot_kws={'size': 12})
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Saved: 03_correlation_heatmap.png")


def cluster_election_vs_nonelection(data):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    vars_cl  = ['CPI_index', 'M2', 'GOV_EXP', 'rate']
    cl_data  = data[vars_cl].dropna()
    X        = StandardScaler().fit_transform(cl_data)
    clusters = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X)
    pca      = PCA(n_components=2)
    X_pca    = pca.fit_transform(X)
    elec_lbl = data['ELECTION'].reindex(cl_data.index).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Clustering of Quarterly Observations', fontsize=14, fontweight='bold')
    for c, color in enumerate(['#2563eb', '#dc2626']):
        mask = clusters == c
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        color=color, alpha=0.7, s=50, label=f'Cluster {c+1}')
    axes[0].set_title('K-means Clusters (k=2)')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    for e, label, color in [(0,'Non-election','#6b7280'),(1,'Election','#dc2626')]:
        mask = elec_lbl == e
        axes[1].scatter(X_pca[mask.values, 0], X_pca[mask.values, 1],
                        color=color, alpha=0.7, s=50, label=label)
    axes[1].set_title('True Election Labels')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('04_clustering.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Saved: 04_clustering.png")


# ============================================================
# STEP 4 — STATIONARITY TESTS
# ============================================================

from statsmodels.tsa.stattools import adfuller, kpss

def test_stationarity(series, name):
    s = series.dropna()
    _, adf_p, *_ = adfuller(s, autolag='AIC')
    try:
        _, kpss_p, *_ = kpss(s, regression='c', nlags='auto')
    except Exception:
        kpss_p = np.nan

    adf_ok  = adf_p  < 0.05
    kpss_ok = kpss_p > 0.05
    if adf_ok and kpss_ok:
        conclusion = 'STATIONARY I(0)'
    elif not adf_ok and not kpss_ok:
        conclusion = 'NON-STATIONARY I(1) — difference'
    else:
        conclusion = 'INCONCLUSIVE'
    log.info(f"{name:30s} ADF p={adf_p:.4f}  KPSS p={kpss_p:.4f}  → {conclusion}")
    return {'Variable': name, 'ADF_p': round(adf_p, 4),
            'KPSS_p': round(kpss_p, 4), 'Conclusion': conclusion}


def run_stationarity_tests(data):
    log.info("="*55 + "\nSTATIONARITY TESTS — LEVELS\n" + "="*55)
    records = []
    for var in ['CPI_index', 'M2', 'GOV_EXP', 'rate']:
        records.append(test_stationarity(data[var].dropna(), var))

    log.info("="*55 + "\nSTATIONARITY TESTS — FIRST DIFFERENCES\n" + "="*55)
    data_model = data.copy()
    for var in ['CPI_index', 'M2', 'GOV_EXP']:
        data_model[f'log_{var}']  = np.log(data_model[var])
        data_model[f'dlog_{var}'] = data_model[f'log_{var}'].diff()
        records.append(test_stationarity(data_model[f'dlog_{var}'].dropna(), f'dlog_{var}'))
    data_model['d_rate'] = data_model['rate'].diff()
    records.append(test_stationarity(data_model['d_rate'].dropna(), 'd_rate'))

    stat_df = pd.DataFrame(records)
    return data_model, stat_df


def plot_acf_pacf(data_model):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    vars_diff = ['dlog_CPI_index', 'dlog_M2', 'dlog_GOV_EXP', 'd_rate']
    fig, axes = plt.subplots(len(vars_diff), 2, figsize=(14, 16))
    fig.suptitle('ACF and PACF — Differenced Series', fontsize=14, fontweight='bold')
    for i, var in enumerate(vars_diff):
        s = data_model[var].dropna()
        plot_acf(s,  ax=axes[i][0], lags=12, title=f'ACF: {var}')
        plot_pacf(s, ax=axes[i][1], lags=12, title=f'PACF: {var}')
    plt.tight_layout()
    plt.savefig('05_acf_pacf.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Saved: 05_acf_pacf.png")


# ============================================================
# STEP 4b — STRUCTURAL BREAK TEST  (item 7)
# ============================================================

def run_structural_break_tests(data_model):
    from statsmodels.tsa.stattools import zivot_andrews
    log.info("="*55 + "\nSTRUCTURAL BREAK — ZIVOT-ANDREWS TEST\n" + "="*55)

    series = data_model['dlog_CPI_index'].dropna()
    za_stat, za_pval, za_cvdict, _, za_breakidx = zivot_andrews(
        series, trim=0.15, regression='c')
    break_date = series.index[za_breakidx]

    log.info(f"ZA statistic : {za_stat:.4f}")
    log.info(f"p-value      : {za_pval:.4f}")
    log.info(f"Break date   : {break_date.date()}")
    log.info(f"Critical vals: {za_cvdict}")
    if za_stat < za_cvdict['5%']:
        log.info("REJECT null → structural break detected")
    else:
        log.info("FAIL TO REJECT null → no significant structural break")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series.values, color='#2563eb', linewidth=1.5, label='dlog(CPI)')
    ax.axvline(break_date, color='red', linewidth=2, linestyle='--',
               label=f'ZA break: {break_date.date()}')
    for yr in CONFIG['structural_break_years']:
        ax.axvline(pd.Timestamp(f'{yr}-01-01'), color='orange',
                   linewidth=1.5, linestyle=':', alpha=0.8, label=str(yr))
    ax.set_title('Zivot-Andrews Structural Break — dlog(CPI)', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('05b_structural_break.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Saved: 05b_structural_break.png")
    return {'za_stat': za_stat, 'za_pval': za_pval, 'break_date': break_date}


# ============================================================
# STEP 5 — VAR MODEL
# ============================================================

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests


def build_var_model(data_model):
    var_vars = ['dlog_CPI_index', 'dlog_M2', 'dlog_GOV_EXP', 'd_rate']
    var_data = data_model[var_vars].dropna()
    exog     = data_model['ELECTION'].reindex(var_data.index).fillna(0)

    log.info("="*55 + "\nVAR LAG SELECTION\n" + "="*55)
    model       = VAR(var_data, exog=exog)
    lag_results = model.select_order(maxlags=CONFIG['lag_range'])
    log.info(str(lag_results.summary()))

    optimal_lag = max(int(lag_results.aic), 1)
    log.info(f"Optimal lag by AIC: {optimal_lag}")
    fitted = model.fit(maxlags=optimal_lag, ic='aic', trend='c')
    log.info(str(fitted.summary()))

    lag_df = pd.DataFrame({
        'AIC':  lag_results.ics['aic'],
        'BIC':  lag_results.ics['bic'],
        'HQIC': lag_results.ics['hqic'],
    })
    return fitted, var_data, exog, lag_df


def check_stability(fitted):
    log.info("="*55 + "\nSTABILITY CHECK\n" + "="*55)
    roots      = fitted.roots
    all_stable = all(np.abs(roots) > 1)
    log.info(f"All characteristic roots outside unit circle: "
             f"{'YES — STABLE' if all_stable else 'NO — UNSTABLE'}")
    log.info(f"Min root modulus: {np.min(np.abs(roots)):.4f}")

    inv_roots = 1.0 / roots
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='black', linewidth=1.5))
    ax.scatter(inv_roots.real, inv_roots.imag,
               color='#dc2626' if not all_stable else '#2563eb', s=60, zorder=5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal')
    ax.set_title('VAR Stability — Inverse Roots (must be inside circle)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('06_stability.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Saved: 06_stability.png")
    return all_stable


def check_residual_autocorrelation(fitted):
    log.info("="*55 + "\nPORTMANTEAU TEST\n" + "="*55)
    pt = fitted.test_whiteness(nlags=10)
    log.info(str(pt.summary()))
    if pt.pvalue > 0.05:
        log.info("p > 0.05 → residuals are white noise, fit adequate.")
    else:
        log.info("p < 0.05 → residual autocorrelation detected, consider more lags.")


def plot_impulse_response(fitted):
    irf = fitted.irf(periods=CONFIG['forecast_horizon'])
    irf.plot(orth=True, impulse='dlog_GOV_EXP', response='dlog_CPI_index', figsize=(10, 4))
    plt.suptitle('IRF: Shock to Gov. Spending → CPI Response', fontweight='bold')
    plt.tight_layout()
    plt.savefig('07_irf_gov_to_cpi.png', dpi=150, bbox_inches='tight')
    plt.show()
    irf.plot(orth=True, figsize=(14, 10))
    plt.suptitle('Impulse Response Functions — All Variables', fontweight='bold')
    plt.tight_layout()
    plt.savefig('07_irf_all.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Saved: 07_irf_gov_to_cpi.png, 07_irf_all.png")
    return irf


def plot_fevd(fitted):
    fitted.fevd(periods=CONFIG['forecast_horizon']).plot(figsize=(14, 8))
    plt.suptitle('Forecast Error Variance Decomposition', fontweight='bold')
    plt.tight_layout()
    plt.savefig('08_fevd.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Saved: 08_fevd.png")


# ============================================================
# STEP 6 — HYPOTHESIS TESTING  (item 2 completes Johansen)
# ============================================================

def run_hypothesis_tests(fitted, var_data, data_model):
    log.info("="*55 + "\nHYPOTHESIS TESTING\n" + "="*55)

    # Granger causality
    log.info("--- GRANGER CAUSALITY ---")
    granger_records = []
    for cause in ['dlog_M2', 'dlog_GOV_EXP', 'd_rate']:
        td  = var_data[['dlog_CPI_index', cause]].dropna()
        res = grangercausalitytests(td, maxlag=4, verbose=False)
        p   = res[1][0]['ssr_ftest'][1]
        conc = 'Granger-causes CPI' if p < 0.05 else 'Does not Granger-cause CPI'
        log.info(f"  {cause} → CPI: p={p:.4f}  {conc}")
        granger_records.append({'Cause': cause, 'p_value': round(p, 4), 'Conclusion': conc})
    granger_df = pd.DataFrame(granger_records)

    # Normality
    log.info("--- NORMALITY (Jarque-Bera) ---")
    log.info(str(fitted.test_normality().summary()))

    # Heteroskedasticity
    log.info("--- HETEROSKEDASTICITY ---")
    try:
        log.info(str(fitted.test_het_arch().summary()))
    except Exception as e:
        log.info(f"Skipped: {e}")

    # Johansen cointegration  (item 2 — fully implemented)
    log.info("--- JOHANSEN COINTEGRATION TEST ---")
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    log_vars = ['log_CPI_index', 'log_M2', 'log_GOV_EXP']
    log_data = data_model[log_vars].dropna()
    joh      = coint_johansen(log_data, det_order=0, k_ar_diff=2)
    trace    = joh.lr1
    crit95   = joh.cvt[:, 1]

    log.info(f"{'r<=':<5} {'Trace Stat':>12} {'Crit 95%':>10} {'Decision'}")
    log.info("-" * 50)
    coint_rank = 0
    for r in range(len(trace)):
        reject = trace[r] > crit95[r]
        if reject:
            coint_rank += 1
        log.info(f"{r:<5} {trace[r]:>12.4f} {crit95[r]:>10.4f} "
                 f"  {'Reject H0 — cointegrated' if reject else 'Fail to reject H0'}")
    log.info(f"Cointegration rank: {coint_rank}")
    if coint_rank >= 1:
        log.info("Cointegration found → VECM appropriate on log-levels.")
    else:
        log.info("No cointegration → VAR in differences is appropriate.")

    return granger_df, coint_rank


# ============================================================
# STEP 6b — VECM  (item 5)
# ============================================================

def build_vecm_model(data_model, fitted_var, coint_rank):
    if coint_rank < 1:
        log.info("No cointegration — skipping VECM.")
        return None

    from statsmodels.tsa.vector_ar.vecm import VECM
    log.info("="*55 + "\nVECM MODEL\n" + "="*55)

    log_vars = ['log_CPI_index', 'log_M2', 'log_GOV_EXP']
    log_data = data_model[log_vars].dropna()
    vecm_fit = VECM(log_data, k_ar_diff=2, coint_rank=coint_rank,
                   deterministic='co').fit()
    log.info(str(vecm_fit.summary()))

    ec_cols  = [f'EC{i+1}' for i in range(coint_rank)]
    log.info("Error correction terms (alpha):")
    log.info(str(pd.DataFrame(vecm_fit.alpha, index=log_vars, columns=ec_cols)))
    log.info("Cointegrating vectors (beta):")
    log.info(str(pd.DataFrame(vecm_fit.beta, index=log_vars, columns=ec_cols)))
    log.info(f"VAR AIC : {fitted_var.aic:.4f}")
    log.info(f"VECM AIC: {vecm_fit.aic:.4f}")
    log.info(f"Better by AIC: {'VECM' if vecm_fit.aic < fitted_var.aic else 'VAR'}")
    return vecm_fit


# ============================================================
# STEP 6c — GARCH ON VAR RESIDUALS  (item 8)
# ============================================================

def run_garch_on_residuals(fitted, data):
    try:
        from arch import arch_model
    except ImportError:
        log.warning("arch library not installed — pip install arch  (skipping GARCH)")
        return None

    log.info("="*55 + "\nGARCH(1,1) ON CPI VAR RESIDUALS\n" + "="*55)
    resid    = fitted.resid['dlog_CPI_index'].dropna() * 100
    gm       = arch_model(resid, vol='Garch', p=1, q=1, dist='normal', rescale=False)
    garch_fit = gm.fit(disp='off')
    log.info(str(garch_fit.summary()))

    cond_vol = garch_fit.conditional_volatility
    elec_q   = data[data['ELECTION'] == 1].index

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cond_vol.index, cond_vol.values, color='#2563eb',
            linewidth=1.5, label='Conditional volatility')
    for eq in elec_q:
        ax.axvline(eq, color='red', alpha=0.25, linewidth=1.5, linestyle='--')
    ax.axvline(elec_q[0], color='red', alpha=0.25, linewidth=1.5,
               linestyle='--', label='Election period')
    ax.set_title('GARCH(1,1) Conditional Volatility — CPI Residuals', fontweight='bold')
    ax.set_ylabel('Conditional Std Dev (%)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('10_garch_volatility.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Saved: 10_garch_volatility.png")
    return garch_fit


# ============================================================
# STEP 7 — FORECASTING & ML COMPARISON
# ============================================================

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler


def forecast_var(fitted, var_data, exog):
    """FIX (item 3): forecast() + manual ±1.96*resid_std confidence bands."""
    split      = CONFIG['train_test_split']
    train      = var_data[var_data.index < split]
    test       = var_data[var_data.index >= split]
    exog_train = exog[exog.index < split]
    exog_test  = exog[exog.index >= split]

    log.info(f"Train: {train.index.min().date()} → {train.index.max().date()} ({len(train)} obs)")
    log.info(f"Test : {test.index.min().date()} → {test.index.max().date()}  ({len(test)} obs)")

    fitted_train = VAR(train, exog=exog_train).fit(maxlags=2, trend='c')
    lag_order    = fitted_train.k_ar
    fc_vals      = fitted_train.forecast(y=train.values[-lag_order:], steps=len(test))
    fc_df        = pd.DataFrame(fc_vals, index=test.index, columns=var_data.columns)

    resid_std = fitted_train.resid.std()
    fc_lower  = fc_df.sub(1.96 * resid_std)
    fc_upper  = fc_df.add(1.96 * resid_std)

    rmse = np.sqrt(mean_squared_error(test['dlog_CPI_index'], fc_df['dlog_CPI_index']))
    mape = mean_absolute_percentage_error(test['dlog_CPI_index'].abs() + 1e-6,
                                          fc_df['dlog_CPI_index'].abs() + 1e-6)
    log.info(f"VAR RMSE: {rmse:.6f}  MAPE: {mape:.4f}")
    return test, fc_df, fc_lower, fc_upper, rmse, mape


def forecast_xgboost(var_data, exog):
    import xgboost as xgb
    split  = CONFIG['train_test_split']
    target = 'dlog_CPI_index'
    df     = var_data.copy()
    df['ELECTION'] = exog.reindex(df.index).fillna(0)
    for lag in range(1, 5):
        for col in ['dlog_CPI_index', 'dlog_M2', 'dlog_GOV_EXP', 'd_rate']:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df = df.dropna()
    feat_cols = [c for c in df.columns if c != target]
    X_tr = df[df.index < split][feat_cols]; y_tr = df[df.index < split][target]
    X_te = df[df.index >= split][feat_cols]; y_te = df[df.index >= split][target]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                              learning_rate=0.1, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    mape  = mean_absolute_percentage_error(y_te.abs() + 1e-6, np.abs(preds) + 1e-6)
    log.info(f"XGBoost RMSE: {rmse:.6f}  MAPE: {mape:.4f}")
    return y_te, preds, rmse, mape


def forecast_lstm(var_data, exog):
    """FIX (item 4): correct inverse transform using per-feature min/max."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    split  = CONFIG['train_test_split']
    target = 'dlog_CPI_index'
    df     = var_data.copy()
    df['ELECTION'] = exog.reindex(df.index).fillna(0)
    df = df.dropna()

    scaler     = MinMaxScaler()
    scaled     = scaler.fit_transform(df)
    target_idx = df.columns.tolist().index(target)
    seq_len    = 4
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i, target_idx])
    X, y      = np.array(X), np.array(y)
    split_idx = df.index.searchsorted(split) - seq_len
    X_tr, X_te = X[:split_idx], X[split_idx:]
    y_tr, y_te = y[:split_idx], y[split_idx:]

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(seq_len, df.shape[1])),
        Dropout(0.2), LSTM(16), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_tr, y_tr, epochs=50, batch_size=8, verbose=0)
    preds_sc = model.predict(X_te).flatten()

    # FIX: use per-feature scale so only target column matters
    t_min    = scaler.data_min_[target_idx]
    t_max    = scaler.data_max_[target_idx]
    preds_inv = preds_sc * (t_max - t_min) + t_min
    y_inv     = y_te    * (t_max - t_min) + t_min

    rmse = np.sqrt(mean_squared_error(y_inv, preds_inv))
    mape = mean_absolute_percentage_error(np.abs(y_inv) + 1e-6, np.abs(preds_inv) + 1e-6)
    log.info(f"LSTM RMSE: {rmse:.6f}  MAPE: {mape:.4f}")
    return y_inv, preds_inv, rmse, mape


def plot_model_comparison(test, fc_var, fc_lower, fc_upper,
                           y_xgb, preds_xgb,
                           rmse_var, rmse_xgb, rmse_lstm,
                           mape_var, mape_xgb, mape_lstm):
    """FIX (item 9): shaded 95% CI band around VAR forecast."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Model Comparison: VAR vs XGBoost vs LSTM — CPI Forecast',
                 fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(test.index, test['dlog_CPI_index'], 'k-',  linewidth=2,   label='Actual')
    ax.plot(test.index, fc_var['dlog_CPI_index'], 'b--', linewidth=1.5, label='VAR')
    ax.fill_between(test.index,
                    fc_lower['dlog_CPI_index'],
                    fc_upper['dlog_CPI_index'],
                    alpha=0.2, color='blue', label='VAR 95% CI')
    ax.plot(test.index[-len(preds_xgb):], preds_xgb, 'g-.', linewidth=1.5, label='XGBoost')
    ax.set_title('Out-of-sample Forecasts (2022–2025)')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylabel('dlog(CPI)')

    ax2    = axes[1]
    models = ['VAR', 'XGBoost', 'LSTM']
    rmses  = [rmse_var, rmse_xgb, rmse_lstm]
    bars   = ax2.bar(models, rmses, color=['#2563eb','#16a34a','#dc2626'],
                     alpha=0.8, edgecolor='white', linewidth=1.5)
    ax2.set_title('RMSE Comparison (lower is better)'); ax2.set_ylabel('RMSE')
    for bar, val in zip(bars, rmses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                 f'{val:.5f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('09_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    best = models[int(np.argmin(rmses))]
    comparison_df = pd.DataFrame({
        'Model':  models,
        'RMSE':   [round(r, 6) for r in rmses],
        'MAPE':   [round(m, 4) for m in [mape_var, mape_xgb, mape_lstm]],
        'Interpretable':       ['Yes (IRF, Granger)', 'Partial (SHAP)', 'No'],
        'Structural meaning':  ['Yes', 'No', 'No'],
    })
    log.info(f"\n{comparison_df.to_string(index=False)}")
    log.info(f"Best model by RMSE: {best}")
    log.info("Saved: 09_model_comparison.png")
    return comparison_df


# ============================================================
# STEP 7b — ROLLING WINDOW GRANGER CAUSALITY  (item 10)
# ============================================================

def rolling_granger_causality(var_data, exog):
    log.info("="*55 + "\nROLLING WINDOW GRANGER CAUSALITY\n" + "="*55)
    min_obs = 20
    records = []
    for end_idx in range(min_obs, len(var_data) + 1):
        td = var_data.iloc[:end_idx][['dlog_CPI_index', 'dlog_GOV_EXP']].dropna()
        if len(td) < min_obs:
            continue
        try:
            res = grangercausalitytests(td, maxlag=2, verbose=False)
            p   = res[2][0]['ssr_ftest'][1]
            records.append({'date': var_data.index[end_idx - 1], 'p_gov_cpi': p})
        except Exception:
            continue

    if not records:
        log.warning("Rolling Granger: no results.")
        return None

    roll_df = pd.DataFrame(records).set_index('date')
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(roll_df.index, roll_df['p_gov_cpi'], color='#2563eb',
            linewidth=2, label='p-value: GOV_EXP → CPI')
    ax.axhline(0.05, color='red', linestyle='--', linewidth=1.5, label='p=0.05')
    ax.axvline(pd.Timestamp('2018-01-01'), color='orange', linestyle=':',
               linewidth=1.5, label='Velvet Revolution 2018')
    ax.fill_between(roll_df.index, roll_df['p_gov_cpi'], 0.05,
                    where=roll_df['p_gov_cpi'] < 0.05,
                    alpha=0.2, color='green', label='Significant')
    ax.set_title('Rolling Granger Causality: GOV_EXP → CPI (expanding window)',
                 fontweight='bold')
    ax.set_ylabel('p-value'); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('11_rolling_granger.png', dpi=150, bbox_inches='tight')
    plt.show()
    log.info("Saved: 11_rolling_granger.png")
    return roll_df


# ============================================================
# STEP 8 — EXPORT RESULTS TO EXCEL  (item 15)
# ============================================================

def generate_report_tables(stat_df, lag_df, granger_df, comparison_df):
    path = 'results_tables.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        stat_df.to_excel(writer,       sheet_name='Stationarity',     index=False)
        lag_df.to_excel(writer,        sheet_name='Lag Selection')
        granger_df.to_excel(writer,    sheet_name='Granger Causality', index=False)
        comparison_df.to_excel(writer, sheet_name='Model Comparison',  index=False)
    log.info(f"Saved: {path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    log.info("="*55 + "\nARMENIAN POLITICAL BUSINESS CYCLE PROJECT\n" + "="*55)

    # Steps 1–2
    data = build_quarterly_dataset()

    # Step 2b
    plot_seasonal_decomposition(data)

    # Step 3
    plot_series_with_elections(data)
    detect_outliers(data)
    plot_scatter_matrix(data)
    plot_correlation_heatmap(data)
    cluster_election_vs_nonelection(data)

    # Step 4
    data_model, stat_df = run_stationarity_tests(data)
    run_structural_break_tests(data_model)
    plot_acf_pacf(data_model)

    # Step 5
    fitted, var_data, exog, lag_df = build_var_model(data_model)
    check_stability(fitted)
    check_residual_autocorrelation(fitted)
    plot_impulse_response(fitted)
    plot_fevd(fitted)

    # Step 6
    granger_df, coint_rank = run_hypothesis_tests(fitted, var_data, data_model)
    build_vecm_model(data_model, fitted, coint_rank)
    run_garch_on_residuals(fitted, data)

    # Step 7
    test, fc_var, fc_lower, fc_upper, rmse_var, mape_var = forecast_var(fitted, var_data, exog)
    y_xgb, preds_xgb, rmse_xgb, mape_xgb = forecast_xgboost(var_data, exog)
    y_lstm, preds_lstm, rmse_lstm, mape_lstm = forecast_lstm(var_data, exog)
    comparison_df = plot_model_comparison(
        test, fc_var, fc_lower, fc_upper,
        y_xgb, preds_xgb,
        rmse_var, rmse_xgb, rmse_lstm,
        mape_var, mape_xgb, mape_lstm,
    )
    rolling_granger_causality(var_data, exog)

    # Step 8
    generate_report_tables(stat_df, lag_df, granger_df, comparison_df)

    log.info("ALL STEPS COMPLETE — see PNG files + results_tables.xlsx")
