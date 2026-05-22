"""
Armenian Political Business Cycle — Interactive Dashboard
Run:  python dashboard.py
Open: http://127.0.0.1:8050

"""

import pandas as pd
import numpy as np


import dash
from dash import dcc, html, dash_table, Input, Output, ctx
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings('ignore')


# ── palette ──────────────────────────────────────────────────
NAVY  = '#0f172a'
GOLD  = '#f59e0b'
BLUE  = '#2563eb'
GREEN = '#059669'
RED   = '#dc2626'
LGRAY = '#f8fafc'
DGRAY = '#64748b'
WHITE = '#ffffff'

CARD = {'background': WHITE, 'borderRadius': '12px',
        'boxShadow': '0 2px 12px rgba(0,0,0,0.08)',
        'padding': '24px', 'marginBottom': '20px'}
INFO = {'background': '#eff6ff', 'border': '1px solid #bfdbfe',
        'borderRadius': '8px', 'padding': '14px 18px',
        'marginBottom': '18px', 'color': '#1e3a8a',
        'fontSize': '14px', 'lineHeight': '1.7'}

ELECTIONS = {
    2008: {'label': '2008', 'color': '#7c3aed'},
    2012: {'label': '2012', 'color': '#2563eb'},
    2017: {'label': '2017', 'color': '#0891b2'},
    2018: {'label': '2018', 'color': '#ea580c'},
    2021: {'label': '2021', 'color': '#059669'},
}
ELEC_STARTS = {2008: '2008-04-01', 2012: '2012-04-01',
               2017: '2017-04-01', 2018: '2018-12-09',
               2021: '2021-06-20'}

VAR_LABELS = {
    'dlog_CPI_index': 'Δlog(CPI)',
    'dlog_M2':        'Δlog(M2)',
    'dlog_GOV_EXP':   'Δlog(Gov Spending)',
    'd_rate':         'Δ Interest Rate',
}

# ── translations ──────────────────────────────────────────────
T = {
    'en': {
        'subtitle':      'Do Elections Move the Economy? A Time Series Investigation.',
        'span_elec':     '4 Elections',
        'span_models':   'VAR · XGBoost · LSTM',
        'tab_home':      '🏠  About',
        'tab_data':      '📈  Data Explorer',
        'tab_elec':      '🗳️  Election Analysis',
        'tab_var':       '🔬  VAR Model',
        'tab_fc':        '🔮  Forecasting',
        'about_h':       'What is the Political Business Cycle?',
        'about_p1':      ('The Political Business Cycle (PBC) theory argues that governments '
                          'manipulate economic policy around elections to boost their chances of '
                          'winning. Before an election, politicians may raise spending, expand '
                          'money supply, or cut rates — creating a short-term economic boost. '
                          'After the election, corrections follow: higher inflation, tightening, cuts.'),
        'about_p2':      ('This project asks: does Armenia show evidence of a PBC? We use quarterly '
                          'data from 2008–2025, covering four elections, and apply VAR, XGBoost, '
                          'and LSTM to test whether elections predict changes in inflation, '
                          'money supply, and government spending.'),
        'stat_q':        'Quarters of data',      'stat_q_s': '2008 Q1 – 2025 Q4',
        'stat_e':        'Elections analysed',    'stat_e_s': '2008, 2012, 2017, 2018, 2021',
        'stat_v':        'Macro variables',       'stat_v_s': 'CPI · M2 · Gov · Rate · Election',
        'stat_m':        'Models compared',       'stat_m_s': 'VAR · XGBoost · LSTM',
        'data_h':        'The Data — What Are We Looking At?',
        'data_sub':      'All variables measured quarterly (every 3 months)',
        'col_var':       'Variable', 'col_mean': 'What it means',
        'col_freq':      'Frequency', 'col_src': 'Source',
        'annual_note':   '* Annual gov. expenditure is linearly interpolated to quarterly.',
        'clean_h':       'How Was the Data Cleaned?',
        'clean_sub':     'Raw government data is messy — here is what we did',
        'c1h': 'Excel structure fix',
        'c1p': 'Armstat exports had year values buried in a data row, not the column header. We extracted them automatically.',
        'c2h': 'Date parsing',
        'c2p': 'Central Bank files stored dates as Python datetime objects — footnote rows were filtered, dates parsed directly.',
        'c3h': 'Quarterly aggregation',
        'c3p': 'Monthly CPI and M2 were averaged to quarterly. Annual gov spending was linearly interpolated. All series merged on a common quarterly index.',
        'elec_cov':      'Elections Covered',
        'data_select':   'Select variable',
        'about_var':     'About this variable: ',
        'elec_h':        'Election-Period Analysis',
        'elec_sub':      'Compare macro behaviour 4 quarters before, during, and after each election',
        'elec_info':     ("Election window = election quarter + quarter before. "
                          "'Before' = 4 preceding quarters. 'After' = 4 following quarters. "
                          "Look for spending surges and M2 expansion before elections, "
                          "and inflation corrections afterward."),
        'elec_yr':       'Select election year',
        'elec_vr':       'Select variable',
        'elec_comp_h':   'All Elections Side-by-Side — CPI',
        'elec_comp_sub': 'CPI indexed to 100 at election quarter. Do all elections look similar?',
        'period_b':      '4 quarters BEFORE',
        'period_d':      'ELECTION window',
        'period_a':      '4 quarters AFTER',
        'col_period':    'Period', 'col_ch': 'Change vs previous',
        'var_h':         'VAR Model Results',
        'var_sub':       'Vector Autoregression — the core econometric model',
        'var_what_h':    'What is a VAR model?',
        'var_what_p':    ('A VAR model treats several time series simultaneously. Instead of '
                          'modelling CPI alone, VAR models CPI, M2, Gov Spending, and Interest Rate '
                          'all at once — each variable depends on past values of all the others. '
                          'This lets us ask: does a shock to government spending cause inflation to rise?'),
        'box1h': 'Stationarity', 'box1p': 'VAR requires stationary series. We take Δlog of CPI, M2, Gov Spending and first-difference of the rate.',
        'box2h': 'Lag selection', 'box2p': 'We test lags 1–6 and pick the lag count minimising AIC — balancing fit vs complexity.',
        'box3h': 'Election dummy', 'box3p': 'Binary variable (1 = election quarter or quarter before) included as exogenous regressor.',
        'irf_h':         'Impulse Response Function (IRF)',
        'irf_sub':       'How does a one-time shock ripple through the economy over 8 quarters?',
        'irf_info':      ('Bars above zero = positive response. A positive CPI response to Gov Spending '
                          'means spending increases cause inflation — a key PBC signature.'),
        'irf_imp':       'Impulse (shock)', 'irf_resp': 'Response',
        'granger_h':     'Granger Causality',
        'granger_sub':   'Does knowing past X help predict CPI better than CPI alone?',
        'granger_info':  ('Granger causality ≠ physical causation. It means X contains predictive '
                          'information about CPI beyond CPI\'s own history. p < 0.05 supports the PBC '
                          'hypothesis when it is gov spending or M2 driving inflation.'),
        'fc_h':          'Forecasting Models Comparison',
        'fc_sub':        'Train on 2008–2021, forecast 2022–2025, compare accuracy',
        'fc_what_h':     'What are we forecasting?',
        'fc_what_p':     ('We forecast Δlog(CPI) — the quarterly log change in price level, '
                          'interpretable as quarterly inflation. All models train on 2008–2021 '
                          'and forecast 2022–2025 without seeing that data.'),
        'fc_line_h':     'Forecast vs Actual',
        'fc_line_sub':   'Shaded blue = 95% CI for VAR  (±1.96 × residual std dev)',
        'fc_bar_h':      'Model Accuracy',
        'fc_bar_sub':    'RMSE = Root Mean Squared Error. Lower is better.',
        'fc_verdict_p':  ('VAR is preferred for policy analysis regardless of raw accuracy — '
                          'it explains WHY prices move via Granger causality and IRFs. '
                          'XGBoost wins on pure prediction.'),
        'wins':          'wins on RMSE',
        'best_rmse':     'Best RMSE: ',
        'actual':        'Actual',
        'lower_better':  'RMSE (lower is better)',
        'badge_econ':    'Econometric',
        'badge_ml':      'Machine Learning',
        'badge_dl':      'Deep Learning',
        'var_d':  'Uses all four macro variables. Interpretable coefficients and confidence intervals.',
        'xgb_d':  'Gradient-boosted trees with 4 lags per variable. Captures non-linear patterns.',
        'lstm_d': 'Neural network over 4-quarter sequences. May underperform with ~55 training obs.',
    },
    'hy': {
        'subtitle':      'Արդյո՞ք ընտրությունները շարժում են տնտեսությունը։ Ժամային շարքերի հետազոտություն։',
        'span_elec':     '4 ընտրություն',
        'span_models':   'VAR · XGBoost · LSTM',
        'tab_home':      '🏠  Մասին',
        'tab_data':      '📈  Տվյալներ',
        'tab_elec':      '🗳️  Ընտրություններ',
        'tab_var':       '🔬  VAR Մոդել',
        'tab_fc':        '🔮  Կանխատեսում',
        'about_h':       'Ի՞նչ է Քաղաքական Բիզնես Ցիկլը',
        'about_p1':      ('Քաղաքական Բիզնես Ցիկլի (ՔԲՑ) տեսությունն ասում է, որ կառավարությունները '
                          'ընտրություններից առաջ շահարկում են տնտեսությունը հաղթելու համար։ '
                          'Ավելացնում են ծախսերը, ընդլայնում դրամական մատակարարումը կամ '
                          'իջեցնում տոկոսադրույքները՝ ստեղծելով կարճաժամկետ աճ։ '
                          'Ընտրություններից հետո հետևում են ուղղումներ՝ գնաճ, խստացում, կրճատումներ։'),
        'about_p2':      ('Այս նախագիծն ուսումնասիրում է՝ արդյո՞ք Հայաստանում ՔԲՑ-ի ապացույցներ կան։ '
                          'Օգտագործել ենք 2008–2025թթ. եռամսյակային տվյալներ, ընդգրկելով 4 ընտրություն, '
                          'և կիրառում ենք VAR, XGBoost, LSTM մոդելներ։'),
        'stat_q':        'Եռամսյակ',             'stat_q_s': '2008 Ե1 – 2025 Ե4',
        'stat_e':        'Ուսումնասիրված ընտրություն', 'stat_e_s': '2008, 2012, 2017, 2018, 2021',
        'stat_v':        'Մակրո փոփոխական',      'stat_v_s': 'ՍԳԻ · M2 · Պետ.Ծախս · Դրույք · Ընտրություն',
        'stat_m':        'Համեմատված մոդել',      'stat_m_s': 'VAR · XGBoost · LSTM',
        'data_h':        'Տվյալները — ի՞նչ ենք ուսումնասիրում',
        'data_sub':      'Բոլոր փոփոխականները չափվում են եռամսյակային (3 ամիսը մեկ)',
        'col_var':       'Փոփոխական', 'col_mean': 'Ի՞նչ է նշանակում',
        'col_freq':      'Հաճախություն', 'col_src': 'Աղբյուր',
        'annual_note':   '* Տարեկան պետական ծախսերը ինտերպոլացվեցին եռամսյակային հաճախությամբ։',
        'clean_h':       'Ինչպե՞ս մաքրվեցին տվյալները',
        'clean_sub':     'Կառավարության հումք տվյալները անկանոն են',
        'c1h': 'Excel կառուցվածքի ուղղում',
        'c1p': 'Armstat-ի ֆայլերում տարեթվերը թաղված էին տվյալների տողում։ Ավտոմատ արդյունահանեցինք։',
        'c2h': 'Ամսաթվերի ճշգրտում',
        'c2p': 'ԿԲ ֆայլերն ամսաթվերը պահում էին datetime-ի տեսքով։ Ծանոթագրությունների տողերը զտեցինք։',
        'c3h': 'Եռամսյակային ամփոփում',
        'c3p': 'Ամսեկան ՍԳԻ-ն ու M2-ը միջինացվեցին։ Տարեկան ծախսերը ինտերպոլացվեցին։ Բոլոր շարքերը ձուլվեցին։',
        'elec_cov':      'Ընդգրկված ընտրություններ',
        'data_select':   'Ընտրեք փոփոխական',
        'about_var':     'Այս փոփոխականի մասին՝ ',
        'elec_h':        'Ընտրաշրջանի Վերլուծություն',
        'elec_sub':      'Համեմատեք վարքը ընտրությունից 4 եռամսյակ առաջ, ընթացքում և հետո',
        'elec_info':     ('Ընտրության պատուհան = ընտրության եռամսյակ + նախորդ եռամսյակ։ '
                          '«Առաջ» = 4 նախորդ եռամսյակ։ «Հետո» = 4 հաջորդ եռամսյակ։ '
                          'Փնտրեք ծախսերի աճ և M2-ի ընդլայնում ընտրություններից առաջ։'),
        'elec_yr':       'Ընտրեք ընտրության տարեթիվ',
        'elec_vr':       'Ընտրեք փոփոխական',
        'elec_comp_h':   'Բոլոր ընտրությունները — ՍԳԻ',
        'elec_comp_sub': 'ՍԳԻ ինդեքսացված 100-ի վրա ընտրության եռամսյակում։',
        'period_b':      '4 եռամսյակ ԱՌԱՋ',
        'period_d':      'ԸՆՏՐՈՒԹՅԱՆ պատուհան',
        'period_a':      '4 եռամսյակ ՀԵՏՈ',
        'col_period':    'Ժամանակաշրջան', 'col_ch': 'Փոփոխություն',
        'var_h':         'VAR Մոդելի Արդյունքներ',
        'var_sub':       'Վեկտոր ավտոռեգրեսիա — հիմնական էկոնոմետրիկ մոդել',
        'var_what_h':    'Ի՞նչ է VAR մոդելը',
        'var_what_p':    ('VAR մոդելը միաժամանակ ուսումնասիրում է մի քանի ժամային շարքեր։ '
                          'Փոխարենը ՍԳԻ-ն առանձին մոդելավորելու, VAR-ն ուսումնասիրում է ՍԳԻ, '
                          'M2, Պետ. ծախսեր և Տոկոսադրույք — յուրաքանչյուրը կախված բոլոր '
                          'մյուսների անցյալ արժեքներից։'),
        'box1h': 'Ստացիոնարություն', 'box1p': 'VAR-ն պահանջում է ստացիոնար շարքեր։ Վերցնում ենք Δlog(ՍԳԻ, M2, Ծախս) և Δ(Դրույք)։',
        'box2h': 'Լագի ընտրություն', 'box2p': 'Ստուգում ենք 1–6 լագ, ընտրում ենք AIC-ն նվազագույնի հասցնողը։',
        'box3h': 'Ընտրությունների dummy', 'box3p': 'Երկուական փոփոխական (1 = ընտրության կամ նախորդ եռամսյակ)։',
        'irf_h':         'Ազդակի Արձագանք (IRF)',
        'irf_sub':       'Ինչպե՞ս է ցնցումն ալիք առաջացնում 8 եռամսյակի ընթացքում',
        'irf_info':      ('Զրոյից վեր ձողիկ = դրական արձագանք։ ՍԳԻ-ի դրական արձագանքը '
                          'Պետ. ծախսերի ցնցմանը = ծախսերն առաջացնում են գնաճ — ՔԲՑ-ի հիմնական ազդանշան։'),
        'irf_imp':       'Ազդակ (ցնցում)', 'irf_resp': 'Արձագանք',
        'granger_h':     'Գրեյնջերի պատճառականություն',
        'granger_sub':   'Արդյո՞ք X-ի անցյալն ավելի լավ կանխատեսում է ՍԳԻ-ն, քան ՍԳԻ-ի սեփական անցյալը',
        'granger_info':  ('Գրեյնջերի պատճառականությունը ֆիզիկական պատճառ չի նշանակում։ '
                          'Նշանակում է X-ն ՍԳԻ-ի կանխատեսիչ տեղեկատվություն է պարունակում։ '
                          'p < 0.05 աջակցում է ՔԲՑ-ի վարկածին, եթե պատճառը Պետ. ծախսն է կամ M2-ը։'),
        'fc_h':          'Կանխատեսման Մոդելների Համեմատություն',
        'fc_sub':        'Ուսուցում 2008–2021, կանխատեսում 2022–2025',
        'fc_what_h':     'Ի՞նչ ենք կանխատեսում',
        'fc_what_p':     ('Կանխատեսում ենք Δlog(ՍԳԻ) — եռամսյակային գնաճի տեմպը։ '
                          'Բոլոր մոդելները վարժեցվում են 2008–2021 թ. վրա, '
                          'կանխատեսում են 2022–2025 թ.՝ առանց այդ տվյալները տեսնելու։'),
        'fc_line_h':     'Կանխատեսում ընդդեմ փաստացիի',
        'fc_line_sub':   'Կապույտ ստվեր = VAR 95% վստահության միջակայք',
        'fc_bar_h':      'Մոդելների ճշգրտություն',
        'fc_bar_sub':    'RMSE = արմատային միջին քառ. սխալ։ Ցածր = ավելի լավ։',
        'fc_verdict_p':  ('VAR-ն ընտրելի է քաղաքականության վերլուծության համար — '
                          'բացատրում է ԻՆՉՈՒ են գները փոխվում՝ Գրեյնջերի և IRF-ի միջոցով։'),
        'wins':          'հաղթում է RMSE-ով',
        'best_rmse':     'Լավագույն RMSE՝ ',
        'actual':        'Փաստացի',
        'lower_better':  'RMSE (ցածր = ավելի լավ)',
        'badge_econ':    'Էկոնոմետրիկ',
        'badge_ml':      'Մ. Ուսուցում',
        'badge_dl':      'Խ. Ուսուցում',
        'var_d':  'Օգտ. չորս մակրո փոփոխականի կապերը։ Մեկնաբանելի գործակիցներ ու վստ. միջակայքներ։',
        'xgb_d':  'Գրադիենտ-ուժ. ծառ՝ 4 լաgrade փոփ.-ից։ Ոչ-գծային ձևեր։',
        'lstm_d': 'Նեյրոնային ցանց՝ 4-եռամսյ. հաջ.վ.։ Կարող է ավելի թույլ կատ. ~55 դիտ.–ով։',
    },
}

VAR_DESC = {
    'CPI_index': ("Consumer Price Index (Base 2003=100)",
                  "Tracks the cost of everyday goods. Rising CPI = inflation. Computed as a "
                  "cumulative index from monthly growth rates, rebased so Jan 2003 = 100."),
    'M2':        ("M2 Money Supply (million AMD)",
                  "All cash plus bank deposits. Governments can expand M2 by lowering rates. "
                  "Pre-election M2 spikes are a classic political business cycle signal."),
    'GOV_EXP':   ("Government Expenditure (million AMD)",
                  "Total spending from Armenia's consolidated budget. The most direct tool "
                  "politicians control — spending increases before elections are a textbook PBC signal."),
    'rate':      ("CBA Reference Interest Rate (%)",
                  "The Central Bank's policy rate. Lower rates stimulate borrowing and growth "
                  "but risk inflation. Watch for rate cuts before elections."),
}

# ── precompute ────────────────────────────────────────────────

def _load():
    return pd.read_csv('armenia_quarterly.csv', index_col=0, parse_dates=True)


def _fit_var(data):

    dm = data.copy()
    for v in ['CPI_index', 'M2', 'GOV_EXP']:
        dm[f'log_{v}']  = np.log(dm[v])
        dm[f'dlog_{v}'] = dm[f'log_{v}'].diff()
    dm['d_rate'] = dm['rate'].diff()
    var_cols = ['dlog_CPI_index', 'dlog_M2', 'dlog_GOV_EXP', 'd_rate']
    var_data = dm[var_cols].dropna()
    exog     = dm['ELECTION'].reindex(var_data.index).fillna(0)
    fitted   = VAR(var_data, exog=exog).fit(maxlags=4, ic='aic', trend='c')
    irf      = fitted.irf(periods=8)
    rows = []
    for cause in ['dlog_M2', 'dlog_GOV_EXP', 'd_rate']:
        td  = var_data[['dlog_CPI_index', cause]].dropna()
        res = grangercausalitytests(td, maxlag=4, verbose=False)
        p   = res[1][0]['ssr_ftest'][1]
        rows.append({'Variable': VAR_LABELS.get(cause, cause),
                     'p-value': round(p, 4),
                     'Significant (p<0.05)': '✅ Yes' if p < 0.05 else '❌ No',
                     'Interpretation': 'Predicts CPI' if p < 0.05 else 'No predictive power'})
    return fitted, var_data, exog, irf, pd.DataFrame(rows)


def _var_forecast(var_data, exog):
    split     = '2022-01-01'
    train     = var_data[var_data.index < split]
    test      = var_data[var_data.index >= split]
    exog_tr   = exog[exog.index < split]
    exog_te   = exog[exog.index >= split]
    fitted_tr = VAR(train, exog=exog_tr).fit(maxlags=2, trend='c')
    lag       = fitted_tr.k_ar
    fc_vals   = fitted_tr.forecast(y=train.values[-lag:], steps=len(test),
                                   exog_future=exog_te.values.reshape(-1, 1))
    fc_df     = pd.DataFrame(fc_vals, index=test.index, columns=var_data.columns)
    return test, fc_df, fitted_tr.resid.std()


def _xgb_forecast(var_data, exog):

    split, target = '2022-01-01', 'dlog_CPI_index'
    df = var_data.copy()
    df['ELECTION'] = exog.reindex(df.index).fillna(0)
    for lag in range(1, 5):
        for col in var_data.columns:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df   = df.dropna()
    feat = [c for c in df.columns if c != target]
    Xtr, ytr = df[df.index < split][feat], df[df.index < split][target]
    Xte, yte = df[df.index >= split][feat], df[df.index >= split][target]
    m = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    m.fit(Xtr, ytr)
    preds = m.predict(Xte)
    return yte, preds, float(np.sqrt(mean_squared_error(yte, preds)))


print("Loading data and fitting VAR (~15 s)…")
DATA                                     = _load()
FITTED, VAR_DATA, EXOG, IRF, GRANGER_DF = _fit_var(DATA)
TEST, FC_VAR, RESID_STD                  = _var_forecast(VAR_DATA, EXOG)
Y_XGB, PREDS_XGB, RMSE_XGB              = _xgb_forecast(VAR_DATA, EXOG)
VAR_COLS  = VAR_DATA.columns.tolist()
NUMERIC   = ['CPI_index', 'M2', 'GOV_EXP', 'rate']
ELEC_Q    = DATA[DATA['ELECTION'] == 1].index

def _rmse(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[m] - b[m])**2)))

RMSE_VAR = _rmse(TEST['dlog_CPI_index'].values, FC_VAR['dlog_CPI_index'].values)
print("Ready — open http://127.0.0.1:8050\n")

# ── helpers ───────────────────────────────────────────────────

def _card(*children, extra=None):
    return html.Div(children, style={**CARD, **(extra or {})})

def _info(text):
    return html.Div([html.Span("ℹ️  "), text], style=INFO)

def _section_title(text, sub=None):
    return html.Div([
        html.H3(text, style={'color': NAVY, 'marginBottom': '4px', 'fontWeight': '700'}),
        html.P(sub, style={'color': DGRAY, 'fontSize': '13px', 'marginTop': '0'}) if sub else None,
    ], style={'marginBottom': '12px'})

def _dd(id_, options, value, label=None):
    return html.Div([
        html.Label(label, style={'fontSize': '12px', 'color': DGRAY,
                                 'fontWeight': '600', 'marginBottom': '4px'}) if label else None,
        dcc.Dropdown(id=id_, options=options, value=value, clearable=False,
                     style={'fontSize': '13px'}),
    ], style={'width': '240px'})

def _badge(text, color=BLUE):
    return html.Span(text, style={'background': color, 'color': WHITE, 'borderRadius': '99px',
                                  'padding': '2px 10px', 'fontSize': '11px', 'fontWeight': '700',
                                  'marginLeft': '8px', 'verticalAlign': 'middle'})

def _bands(fig):
    for eq in ELEC_Q:
        fig.add_vrect(x0=eq, x1=eq + pd.DateOffset(months=3),
                      fillcolor=RED, opacity=0.07, line_width=0)
    return fig

# ── app ───────────────────────────────────────────────────────

app    = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Armenia Macro Pulse"

TS = {'fontFamily': 'Inter,Arial,sans-serif', 'fontSize': '13px', 'fontWeight': '600',
      'color': DGRAY, 'padding': '10px 20px', 'borderBottom': '2px solid transparent'}
TSS = {**TS, 'color': BLUE, 'borderBottom': f'2px solid {BLUE}', 'background': WHITE}

BTN_ON  = {'border': 'none', 'padding': '6px 14px', 'fontWeight': '800',
           'fontSize': '13px', 'cursor': 'pointer', 'background': WHITE, 'color': NAVY}
BTN_OFF = {'border': 'none', 'padding': '6px 14px', 'fontWeight': '700',
           'fontSize': '13px', 'cursor': 'pointer',
           'background': 'rgba(255,255,255,0.15)', 'color': WHITE}

app.layout = html.Div([
    dcc.Store(id='lang', data='en'),
    dcc.Download(id='download-data'),

    # header
    html.Div([
        html.Div([
            html.Div("🇦🇲", style={'fontSize': '36px', 'marginRight': '14px'}),
            html.Div([
                html.H1("Armenia Macro Pulse",
                        style={'margin': '0', 'fontSize': '26px', 'fontWeight': '800',
                               'color': WHITE, 'letterSpacing': '-0.5px'}),
                html.P(id='hdr-sub',
                       style={'margin': '2px 0 0', 'fontSize': '13px',
                              'color': '#94a3b8', 'fontWeight': '400'}),
            ]),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            html.Span("2008–2025", style={'color': GOLD, 'fontWeight': '700',
                                          'fontSize': '13px', 'marginRight': '16px'}),
            html.Span(id='hdr-elec',   style={'color': '#94a3b8', 'fontSize': '13px', 'marginRight': '16px'}),
            html.Span(id='hdr-models', style={'color': '#94a3b8', 'fontSize': '13px', 'marginRight': '24px'}),
            html.Div([
                html.Button('EN', id='btn-en', n_clicks=0,
                            style={**BTN_ON,  'borderRadius': '6px 0 0 6px'}),
                html.Button('ՀՅ', id='btn-hy', n_clicks=0,
                            style={**BTN_OFF, 'borderRadius': '0 6px 6px 0'}),
            ], style={'display': 'flex'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
    ], style={'background': f'linear-gradient(135deg,{NAVY} 0%,#1e3a5f 100%)',
              'padding': '18px 32px', 'display': 'flex',
              'justifyContent': 'space-between', 'alignItems': 'center'}),

    # tabs
    dcc.Tabs(id='tabs', value='home',
             style={'background': '#f1f5f9', 'borderBottom': '1px solid #e2e8f0'},
             children=[
                 dcc.Tab(id='t-home', label='🏠  About',             value='home',     style=TS, selected_style=TSS),
                 dcc.Tab(id='t-data', label='📈  Data Explorer',     value='data',     style=TS, selected_style=TSS),
                 dcc.Tab(id='t-elec', label='🗳️  Election Analysis', value='election', style=TS, selected_style=TSS),
                 dcc.Tab(id='t-var',  label='🔬  VAR Model',         value='var',      style=TS, selected_style=TSS),
                 dcc.Tab(id='t-fc',   label='🔮  Forecasting',       value='forecast', style=TS, selected_style=TSS),
             ]),

    html.Div(id='content',
             style={'maxWidth': '1200px', 'margin': '0 auto',
                    'padding': '28px 32px', 'fontFamily': 'Inter,Arial,sans-serif'}),
], style={'background': LGRAY, 'minHeight': '100vh'})

# ── language toggle ───────────────────────────────────────────

@app.callback(
    Output('lang',       'data'),
    Output('btn-en',     'style'),
    Output('btn-hy',     'style'),
    Output('hdr-sub',    'children'),
    Output('hdr-elec',   'children'),
    Output('hdr-models', 'children'),
    Output('t-home',     'label'),
    Output('t-data',     'label'),
    Output('t-elec',     'label'),
    Output('t-var',      'label'),
    Output('t-fc',       'label'),
    Input('btn-en',      'n_clicks'),
    Input('btn-hy',      'n_clicks'),
)
def toggle_lang(n_en, n_hy):
    lang = 'hy' if ctx.triggered_id == 'btn-hy' else 'en'
    t    = T[lang]
    on_en = {**BTN_ON,  'borderRadius': '6px 0 0 6px'}
    off_en= {**BTN_OFF, 'borderRadius': '6px 0 0 6px'}
    on_hy = {**BTN_ON,  'borderRadius': '0 6px 6px 0'}
    off_hy= {**BTN_OFF, 'borderRadius': '0 6px 6px 0'}
    return (lang,
            on_en  if lang == 'en' else off_en,
            on_hy  if lang == 'hy' else off_hy,
            t['subtitle'], t['span_elec'], t['span_models'],
            t['tab_home'], t['tab_data'], t['tab_elec'], t['tab_var'], t['tab_fc'])

# ── tab router ────────────────────────────────────────────────

@app.callback(Output('content', 'children'),
              Input('tabs', 'value'),
              Input('lang', 'data'))
def render(tab, lang):
    lang = lang or 'en'
    if tab == 'home':     return _tab_home(lang)
    if tab == 'data':     return _tab_data(lang)
    if tab == 'election': return _tab_elec(lang)
    if tab == 'var':      return _tab_var(lang)
    if tab == 'forecast': return _tab_fc(lang)
    return html.Div()

# ══════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════

def _tab_home(lang):
    t = T[lang]
    def stat(val, label, sub):
        return html.Div([
            html.Div(val,   style={'fontSize': '36px', 'fontWeight': '800', 'color': BLUE}),
            html.Div(label, style={'fontSize': '14px', 'fontWeight': '700', 'color': NAVY, 'marginTop': '2px'}),
            html.Div(sub,   style={'fontSize': '11px', 'color': DGRAY, 'marginTop': '2px'}),
        ], style={**CARD, 'textAlign': 'center', 'flex': '1', 'margin': '8px'})

    var_rows = [
        ("CPI Index",          "Tracks everyday goods cost. Rising CPI = inflation. Base Jan 2003=100.", "Monthly→Quarterly", "Armstat"),
        ("M2 Money Supply",    "All cash + bank deposits. Expanding M2 before elections = classic PBC signal.", "Monthly→Quarterly", "CBA"),
        ("Gov. Expenditure",   "Total government spending. Politicians often raise this before elections.", "Annual→Quarterly*", "Armstat"),
        ("Interest Rate",      "CBA policy rate. Lower rate = cheaper loans = more spending = inflation risk.", "Monthly→Quarterly", "CBA"),
        ("Election Dummy",     "1 in election quarter AND quarter before; 0 otherwise. Our key variable.", "Constructed", "Official records"),
    ]

    return html.Div([
        _card(
            html.H2(t['about_h'], style={'color': NAVY, 'marginTop': '0', 'fontWeight': '800'}),
            html.P(t['about_p1'], style={'color': '#334155', 'lineHeight': '1.8', 'fontSize': '15px'}),
            html.P(t['about_p2'], style={'color': '#334155', 'lineHeight': '1.8', 'fontSize': '15px'}),
        ),
        html.Div([stat("72", t['stat_q'], t['stat_q_s']),
                  stat("5",  t['stat_e'], t['stat_e_s']),
                  stat("5",  t['stat_v'], t['stat_v_s']),
                  stat("3",  t['stat_m'], t['stat_m_s'])],
                 style={'display': 'flex', 'gap': '8px', 'marginBottom': '8px'}),
        _card(
            _section_title(t['data_h'], t['data_sub']),
            dash_table.DataTable(
                data=[{t['col_var']: r[0], t['col_mean']: r[1],
                       t['col_freq']: r[2], t['col_src']: r[3]} for r in var_rows],
                columns=[{'name': c, 'id': c}
                         for c in [t['col_var'], t['col_mean'], t['col_freq'], t['col_src']]],
                style_cell={'textAlign': 'left', 'fontSize': '13px', 'padding': '10px 14px',
                            'whiteSpace': 'normal', 'fontFamily': 'Inter,Arial'},
                style_header={'backgroundColor': NAVY, 'color': WHITE, 'fontWeight': '700'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': LGRAY},
                    {'if': {'column_id': t['col_var']}, 'fontWeight': '700', 'color': BLUE},
                ],
                style_table={'borderRadius': '8px', 'overflow': 'hidden'},
            ),
            html.P(t['annual_note'], style={'color': DGRAY, 'fontSize': '12px', 'marginTop': '10px'}),
        ),
        _card(
            _section_title(t['clean_h'], t['clean_sub']),
            html.Div([
                html.Div([
                    html.Div(n, style={'background': c, 'color': WHITE, 'borderRadius': '50%',
                                       'width': '28px', 'height': '28px', 'display': 'flex',
                                       'alignItems': 'center', 'justifyContent': 'center',
                                       'fontWeight': '700', 'fontSize': '13px',
                                       'marginRight': '12px', 'flexShrink': '0'}),
                    html.Div([html.Strong(h),
                              html.P(p, style={'margin': '2px 0 0', 'color': DGRAY, 'fontSize': '13px'})]),
                ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginBottom': '16px'})
                for n, c, h, p in [('1', BLUE, t['c1h'], t['c1p']),
                                    ('2', GREEN, t['c2h'], t['c2p']),
                                    ('3', GOLD,  t['c3h'], t['c3p'])]
            ]),
        ),
        _card(
            _section_title(t['elec_cov']),
            html.Div([
                html.Div([
                    html.Div(str(yr), style={'fontSize': '28px', 'fontWeight': '800', 'color': info['color']}),
                    html.Div("Parliamentary", style={'fontSize': '12px', 'color': '#94a3b8', 'marginTop': '4px'}),
                ], style={'textAlign': 'center', 'flex': '1',
                          'borderLeft': f'3px solid {info["color"]}',
                          'paddingLeft': '12px', 'margin': '4px'})
                for yr, info in ELECTIONS.items()
            ], style={'display': 'flex', 'gap': '12px'}),
        ),
    ])

# ══════════════════════════════════════════════════════════════
# DATA EXPLORER
# ══════════════════════════════════════════════════════════════

def _all_vars_fig():
    fig = go.Figure()
    for var, color in zip(NUMERIC, [BLUE, GREEN, GOLD, '#7c3aed']):
        s = DATA[var].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=VAR_DESC[var][0],
                                 line={'color': color, 'width': 2},
                                 hovertemplate=f'<b>{var}</b><br>%{{x|%Y-%m}}: %{{y:,.1f}}<extra></extra>'))
    for eq in ELEC_Q:
        fig.add_vrect(x0=eq, x1=eq + pd.DateOffset(months=3), fillcolor=RED, opacity=0.07, line_width=0)
    fig.update_layout(height=400, template='plotly_white', hovermode='x unified',
                      legend={'orientation': 'h', 'y': -0.18}, margin={'t': 10},
                      xaxis={'rangeslider': {'visible': True},
                             'rangeselector': {'buttons': [
                                 {'count': 2, 'label': '2Y', 'step': 'year', 'stepmode': 'backward'},
                                 {'count': 5, 'label': '5Y', 'step': 'year', 'stepmode': 'backward'},
                                 {'step': 'all', 'label': 'All'}]}})
    return fig

def _tab_data(lang):
    t = T[lang]
    return html.Div([
        _section_title("Data Explorer", "Select a variable to explore its history"),
        html.Div([
            html.Button("⬇️  Download Cleaned Data (CSV)", id='btn-download',
                        style={'background': BLUE, 'color': WHITE, 'border': 'none',
                               'borderRadius': '8px', 'padding': '10px 20px',
                               'fontSize': '14px', 'fontWeight': '600', 'cursor': 'pointer',
                               'marginBottom': '20px'}),
            html.Span(" armenia_quarterly.csv — 72 quarters, 5 variables, ready for analysis",
                      style={'color': DGRAY, 'fontSize': '13px', 'marginLeft': '12px'}),
        ]),
        _card(
            _dd('data-var', [{'label': f'{v} — {VAR_DESC[v][0]}', 'value': v} for v in NUMERIC],
                'CPI_index', label=t['data_select']),
            html.Div(id='var-desc', style=INFO),
            dcc.Graph(id='data-line', config={'displayModeBar': True}),
            html.Div(id='data-stats'),
        ),
        _card(
            _section_title("All Variables — Overview",
                           "Shaded red = election quarters. Range slider below to zoom."),
            dcc.Graph(id='data-all', figure=_all_vars_fig()),
        ),
    ])

@app.callback(
    Output('download-data', 'data'),
    Input('btn-download', 'n_clicks'),
    prevent_initial_call=True,
)
def download_csv(_n):
    return dcc.send_data_frame(DATA.to_csv, 'armenia_quarterly.csv')


@app.callback(
    Output('data-line',  'figure'),
    Output('var-desc',   'children'),
    Output('data-stats', 'children'),
    Input('data-var',    'value'),
    Input('lang',        'data'),
)
def upd_data(var, lang):
    t     = T[lang or 'en']
    s     = DATA[var].dropna()
    title, desc = VAR_DESC[var]
    fig = go.Figure()
    _bands(fig)
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines', name=title,
                             line={'color': BLUE, 'width': 2.5},
                             fill='tozeroy', fillcolor='rgba(37,99,235,0.06)',
                             hovertemplate=f'<b>%{{x|%Y-%m}}</b><br>{title}: %{{y:,.2f}}<extra></extra>'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Election quarter',
                             line={'color': 'rgba(220,38,38,0.45)', 'width': 10}))
    fig.update_layout(title={'text': title, 'font': {'size': 16, 'color': NAVY}},
                      template='plotly_white', height=360,
                      legend={'orientation': 'h', 'y': -0.15}, margin={'t': 50, 'b': 60},
                      xaxis={'rangeselector': {'buttons': [
                                 {'count': 2, 'label': '2Y', 'step': 'year', 'stepmode': 'backward'},
                                 {'count': 5, 'label': '5Y', 'step': 'year', 'stepmode': 'backward'},
                                 {'step': 'all', 'label': 'All'}]}})
    stats = s.describe().round(3).reset_index()
    stats.columns = ['Statistic', 'Value']
    tbl = dash_table.DataTable(
        data=stats.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in stats.columns],
        style_cell={'textAlign': 'left', 'fontSize': '13px', 'padding': '8px 14px'},
        style_header={'backgroundColor': NAVY, 'color': WHITE, 'fontWeight': '700'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': LGRAY}],
        style_table={'width': '380px', 'borderRadius': '8px', 'overflow': 'hidden'},
    )
    return fig, [html.Strong(t['about_var']), desc], tbl

# ══════════════════════════════════════════════════════════════
# ELECTION ANALYSIS
# ══════════════════════════════════════════════════════════════

def _elec_compare_fig():
    fig = go.Figure()
    for yr, info in ELECTIONS.items():
        elec_d = pd.Timestamp(ELEC_STARTS[yr])
        s      = DATA['CPI_index'].dropna()
        idx    = int(s.index.get_indexer([elec_d], method='nearest')[0])
        if idx < 0:
            continue
        seg  = s.iloc[max(0, idx-4): min(len(s), idx+6)]
        base = seg.iloc[min(4, len(seg)-1)]
        if not base or pd.isna(base):
            continue
        x = list(range(-min(4, idx), len(seg) - min(4, idx)))
        y = (seg.values / base * 100).tolist()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=str(yr),
                                 line={'color': info['color'], 'width': 2.5}, marker={'size': 6},
                                 hovertemplate=f'<b>{yr}</b> Q%{{x:+d}}: %{{y:.1f}}<extra></extra>'))
    fig.add_vline(x=0, line_dash='dash', line_color=RED, line_width=1.5,
                  annotation_text='Election', annotation_position='top right')
    fig.add_hline(y=100, line_dash='dot', line_color='gray', line_width=1)
    fig.update_layout(xaxis_title='Quarters relative to election (0 = election)',
                      yaxis_title='CPI indexed to 100', template='plotly_white', height=380,
                      hovermode='x unified', legend={'title': 'Year'})
    return fig

def _tab_elec(lang):
    t = T[lang]
    return html.Div([
        _section_title(t['elec_h'], t['elec_sub']),
        _info(t['elec_info']),
        _card(
            html.Div([
                _dd('elec-yr', [{'label': str(yr), 'value': yr}
                                for yr in ELECTIONS], 2008, label=t['elec_yr']),
                html.Div(style={'width': '24px'}),
                _dd('elec-var', [{'label': VAR_DESC[v][0], 'value': v} for v in NUMERIC],
                    'CPI_index', label=t['elec_vr']),
            ], style={'display': 'flex', 'alignItems': 'flex-end', 'marginBottom': '20px'}),
            dcc.Graph(id='elec-zoom'),
            html.Div(id='elec-tbl'),
        ),
        _card(
            _section_title(t['elec_comp_h'], t['elec_comp_sub']),
            dcc.Graph(id='elec-comp', figure=_elec_compare_fig()),
        ),
        _card(
            _section_title("Election Comparison — Before / During / After",
                           "Grouped bars show the average of the selected variable across all elections. "
                           "Spot which elections had the biggest pre-election surges."),
            dcc.Graph(id='elec-bars'),
        ),
    ])

@app.callback(
    Output('elec-bars', 'figure'),
    Input('elec-var',   'value'),
)
def upd_elec_bars(var):
    periods   = ['Before', 'During', 'After']
    colors_p  = [BLUE, GOLD, GREEN]
    bar_data  = {p: [] for p in periods}
    yr_labels = []
    for yr in ELECTIONS:
        elec_d = pd.Timestamp(ELEC_STARTS[yr])
        s      = DATA[var].dropna()
        idx    = int(s.index.get_indexer([elec_d], method='nearest')[0])
        if idx < 0:
            continue
        seg = s.iloc[max(0, idx-4): min(len(s), idx+5)]
        bm  = seg.index < elec_d - pd.DateOffset(months=3)
        dm  = (seg.index >= elec_d - pd.DateOffset(months=3)) & (seg.index <= elec_d)
        am  = seg.index > elec_d
        bar_data['Before'].append(round(seg[bm].mean(), 2))
        bar_data['During'].append(round(seg[dm].mean(), 2))
        bar_data['After'].append(round(seg[am].mean(), 2))
        yr_labels.append(str(yr))

    fig = go.Figure()
    for period, color in zip(periods, colors_p):
        fig.add_trace(go.Bar(name=period, x=yr_labels, y=bar_data[period],
                             marker_color=color,
                             hovertemplate=f'<b>{period}</b> %{{x}}: %{{y:,.2f}}<extra></extra>'))
    fig.update_layout(barmode='group', template='plotly_white', height=340,
                      xaxis_title='Election year', yaxis_title=VAR_DESC[var][0],
                      legend={'title': 'Period'}, margin={'t': 10})
    return fig


@app.callback(
    Output('elec-zoom', 'figure'),
    Output('elec-tbl',  'children'),
    Input('elec-yr',    'value'),
    Input('elec-var',   'value'),
    Input('lang',       'data'),
)
def upd_elec(yr, var, lang):
    t      = T[lang or 'en']
    yr     = int(yr)
    color  = ELECTIONS[yr]['color']
    elec_d = pd.Timestamp(ELEC_STARTS[yr])
    s      = DATA[var].dropna()
    idx = int(s.index.get_indexer([elec_d], method='nearest')[0])
    if idx < 0:
        idx = 0
    seg = s.iloc[max(0, idx-6): min(len(s), idx+7)]

    fig = go.Figure()
    fig.add_vrect(x0=elec_d - pd.DateOffset(months=3), x1=elec_d + pd.DateOffset(months=3),
                  fillcolor=color, opacity=0.12, line_width=0,
                  annotation_text='Election window', annotation_position='top left')
    fig.add_trace(go.Scatter(x=seg.index, y=seg.values, mode='lines+markers',
                             line={'color': color, 'width': 2.5}, marker={'size': 7, 'color': color},
                             name=VAR_DESC[var][0],
                             hovertemplate='<b>%{x|%Y-%m}</b>: %{y:,.2f}<extra></extra>'))
    fig.update_layout(title=f'{VAR_DESC[var][0]} — {yr} election',
                      template='plotly_white', height=340, margin={'t': 50})

    bm = seg.index < elec_d - pd.DateOffset(months=3)
    dm = (seg.index >= elec_d - pd.DateOffset(months=3)) & (seg.index <= elec_d)
    am = seg.index > elec_d
    bv, dv, av = seg[bm].mean(), seg[dm].mean(), seg[am].mean()
    rows = [
        {t['col_period']: t['period_b'], 'Mean': round(bv, 2), t['col_ch']: '—'},
        {t['col_period']: t['period_d'], 'Mean': round(dv, 2),
         t['col_ch']: f'{((dv/bv-1)*100):+.1f}%' if bv else '—'},
        {t['col_period']: t['period_a'], 'Mean': round(av, 2),
         t['col_ch']: f'{((av/dv-1)*100):+.1f}%' if dv else '—'},
    ]
    tbl = dash_table.DataTable(
        data=rows,
        columns=[{'name': c, 'id': c} for c in [t['col_period'], 'Mean', t['col_ch']]],
        style_cell={'textAlign': 'left', 'fontSize': '13px', 'padding': '10px 14px'},
        style_header={'backgroundColor': color, 'color': WHITE, 'fontWeight': '700'},
        style_data_conditional=[{'if': {'row_index': 1},
                                  'backgroundColor': '#fef3c7', 'fontWeight': '600'}],
        style_table={'width': '560px', 'borderRadius': '8px', 'overflow': 'hidden', 'marginTop': '16px'},
    )

    # smart interpretation
    bd_chg = ((dv / bv - 1) * 100) if bv and not pd.isna(bv) and not pd.isna(dv) else 0
    da_chg = ((av / dv - 1) * 100) if dv and not pd.isna(dv) and not pd.isna(av) else 0
    var_name = VAR_DESC[var][0]
    if abs(bd_chg) < 0.5:
        trend = f"virtually no change ({bd_chg:+.1f}%) in {var_name} during the election window"
        pbc   = "This provides weak evidence for the PBC hypothesis for this election."
    elif bd_chg > 0:
        trend = f"a {bd_chg:+.1f}% increase in {var_name} during the election window"
        pbc   = ("This is consistent with the Political Business Cycle hypothesis — "
                 "politicians appear to have expanded this variable before the vote.")
    else:
        trend = f"a {bd_chg:+.1f}% decrease in {var_name} during the election window"
        pbc   = "This runs counter to the typical PBC pattern for this variable."
    after_note = (f" After the election, {var_name} {'rose' if da_chg > 0 else 'fell'} "
                  f"by {abs(da_chg):.1f}% — "
                  f"{'a typical post-election correction.' if (bd_chg > 0 and da_chg < 0) else 'no strong reversal observed.'}")

    interp = html.Div([
        html.Strong(f"📊 What does this tell us about {yr}?  "),
        html.Span(f"There was {trend}. {pbc}{after_note}"),
    ], style={**INFO, 'marginTop': '14px'})

    return fig, html.Div([tbl, interp])

# ══════════════════════════════════════════════════════════════
# VAR MODEL
# ══════════════════════════════════════════════════════════════

def _hypothesis_conclusion():
    gov_row = GRANGER_DF[GRANGER_DF['Variable'] == 'Δlog(Gov Spending)']
    m2_row  = GRANGER_DF[GRANGER_DF['Variable'] == 'Δlog(M2)']
    gov_sig = not gov_row.empty and gov_row.iloc[0]['p-value'] < 0.05
    m2_sig  = not m2_row.empty and m2_row.iloc[0]['p-value'] < 0.05
    gov_p   = gov_row.iloc[0]['p-value'] if not gov_row.empty else float('nan')
    m2_p    = m2_row.iloc[0]['p-value']  if not m2_row.empty  else float('nan')

    if gov_sig or m2_sig:
        verdict      = "✅ We REJECT the null hypothesis"
        verdict_sub  = "Evidence supports the Political Business Cycle in Armenia."
        verdict_col  = '#166534'
        verdict_bg   = '#f0fdf4'
        verdict_bdr  = '#86efac'
    else:
        verdict      = "❌ We FAIL TO REJECT the null hypothesis"
        verdict_sub  = "Insufficient evidence for a Political Business Cycle in Armenia."
        verdict_col  = '#991b1b'
        verdict_bg   = '#fff1f2'
        verdict_bdr  = '#fca5a5'

    rows = [
        {'Hypothesis': 'H₀: Gov Spending does NOT Granger-cause CPI',
         'p-value': gov_p, 'Decision': '✅ Reject H₀' if gov_sig else '❌ Fail to reject'},
        {'Hypothesis': 'H₀: M2 does NOT Granger-cause CPI',
         'p-value': m2_p,  'Decision': '✅ Reject H₀' if m2_sig  else '❌ Fail to reject'},
    ]
    return html.Div([
        _section_title("Hypothesis Test Conclusion",
                       "H₀ = elections/spending/M2 do not predict CPI changes  |  α = 0.05"),
        html.Div([
            html.Div(verdict,     style={'fontSize': '18px', 'fontWeight': '800', 'color': verdict_col}),
            html.Div(verdict_sub, style={'fontSize': '14px', 'color': verdict_col, 'marginTop': '4px'}),
        ], style={'background': verdict_bg, 'border': f'1px solid {verdict_bdr}',
                  'borderRadius': '8px', 'padding': '16px 20px', 'marginBottom': '16px'}),
        dash_table.DataTable(
            data=rows,
            columns=[{'name': c, 'id': c} for c in ['Hypothesis', 'p-value', 'Decision']],
            style_cell={'textAlign': 'left', 'fontSize': '13px', 'padding': '10px 14px',
                        'whiteSpace': 'normal', 'fontFamily': 'Inter,Arial'},
            style_header={'backgroundColor': NAVY, 'color': WHITE, 'fontWeight': '700'},
            style_data_conditional=[
                {'if': {'filter_query': '{Decision} contains "Reject H"', 'column_id': 'Decision'},
                 'color': '#166534', 'fontWeight': '700'},
                {'if': {'filter_query': '{Decision} contains "Fail"', 'column_id': 'Decision'},
                 'color': '#991b1b', 'fontWeight': '700'},
                {'if': {'row_index': 'odd'}, 'backgroundColor': LGRAY},
            ],
            style_table={'borderRadius': '8px', 'overflow': 'hidden'},
        ),
        html.P("Note: Granger causality tests whether past values of X improve CPI forecasts. "
               "α = 0.05 significance level used throughout.",
               style={'color': DGRAY, 'fontSize': '12px', 'marginTop': '10px'}),
    ])


def _tab_var(lang):
    t = T[lang]
    return html.Div([
        _section_title(t['var_h'], t['var_sub']),
        _card(
            html.H4(t['var_what_h'], style={'color': NAVY, 'marginTop': 0}),
            html.P(t['var_what_p'], style={'color': '#334155', 'lineHeight': '1.8', 'fontSize': '14px'}),
            html.Div([
                html.Div([html.Strong(t['box1h']), html.Br(),
                          html.Span(t['box1p'], style={'fontSize': '13px', 'color': DGRAY})],
                         style={'flex': '1', 'padding': '12px', 'background': LGRAY,
                                'borderRadius': '8px', 'marginRight': '10px'}),
                html.Div([html.Strong(t['box2h']), html.Br(),
                          html.Span(t['box2p'], style={'fontSize': '13px', 'color': DGRAY})],
                         style={'flex': '1', 'padding': '12px', 'background': LGRAY,
                                'borderRadius': '8px', 'marginRight': '10px'}),
                html.Div([html.Strong(t['box3h']), html.Br(),
                          html.Span(t['box3p'], style={'fontSize': '13px', 'color': DGRAY})],
                         style={'flex': '1', 'padding': '12px', 'background': LGRAY, 'borderRadius': '8px'}),
            ], style={'display': 'flex', 'marginTop': '16px'}),
        ),
        _card(
            _section_title(t['irf_h'], t['irf_sub']),
            _info(t['irf_info']),
            html.Div([
                _dd('irf-imp',  [{'label': VAR_LABELS[v], 'value': v} for v in VAR_COLS],
                    'dlog_GOV_EXP', label=t['irf_imp']),
                html.Div(style={'width': '20px'}),
                _dd('irf-resp', [{'label': VAR_LABELS[v], 'value': v} for v in VAR_COLS],
                    'dlog_CPI_index', label=t['irf_resp']),
            ], style={'display': 'flex', 'marginBottom': '16px'}),
            dcc.Graph(id='irf-fig'),
            html.Div(id='irf-explain', style={**INFO, 'marginTop': '12px'}),
        ),
        _card(
            _section_title(t['granger_h'], t['granger_sub']),
            _info(t['granger_info']),
            html.Div(id='granger-tbl'),
        ),
        _card(_hypothesis_conclusion()),
    ])

@app.callback(
    Output('irf-fig',     'figure'),
    Output('irf-explain', 'children'),
    Output('granger-tbl', 'children'),
    Input('irf-imp',      'value'),
    Input('irf-resp',     'value'),
    Input('lang',         'data'),
)
def upd_var(impulse, response, lang):
    imp_i = VAR_COLS.index(impulse)
    res_i = VAR_COLS.index(response)
    vals  = IRF.irfs[:, res_i, imp_i]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(vals))), y=vals,
                         marker_color=[GREEN if v >= 0 else RED for v in vals],
                         hovertemplate='Q%{x}: %{y:.5f}<extra></extra>'))
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    fig.update_layout(xaxis_title='Quarters after shock',
                      yaxis_title=f'Response of {VAR_LABELS.get(response, response)}',
                      template='plotly_white', height=340, margin={'t': 10},
                      xaxis={'tickvals': list(range(len(vals))),
                             'ticktext': [f'Q{i}' for i in range(len(vals))]})
    peak_q    = int(np.argmax(np.abs(vals)))
    direction = "increases" if vals[peak_q] > 0 else "decreases"
    imp_lbl   = VAR_LABELS.get(impulse,  impulse)
    resp_lbl  = VAR_LABELS.get(response, response)
    pbc_note  = (" This is the key PBC signature: government spending shocks feed into inflation."
                 if impulse == 'dlog_GOV_EXP' and response == 'dlog_CPI_index' and vals[peak_q] > 0
                 else "")
    persists  = sum(1 for v in vals[peak_q:] if (v > 0) == (vals[peak_q] > 0))
    explain   = (f"📈 A one-standard-deviation shock to {imp_lbl} causes {resp_lbl} to "
                 f"{direction} most strongly at quarter {peak_q} after the shock "
                 f"(peak magnitude: {vals[peak_q]:.5f}). "
                 f"The effect persists in the same direction for {persists} quarter(s).{pbc_note}")
    tbl = dash_table.DataTable(
        data=GRANGER_DF.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in GRANGER_DF.columns],
        style_cell={'textAlign': 'left', 'fontSize': '13px', 'padding': '10px 16px', 'whiteSpace': 'normal'},
        style_header={'backgroundColor': NAVY, 'color': WHITE, 'fontWeight': '700'},
        style_data_conditional=[
            {'if': {'filter_query': '{Significant (p<0.05)} = "✅ Yes"'}, 'backgroundColor': '#f0fdf4'},
            {'if': {'filter_query': '{Significant (p<0.05)} = "❌ No"'},  'backgroundColor': '#fff1f2'},
            {'if': {'column_id': 'p-value'}, 'fontWeight': '700'},
        ],
        style_table={'borderRadius': '8px', 'overflow': 'hidden'},
    )
    return fig, explain, tbl

# ══════════════════════════════════════════════════════════════
# FORECASTING
# ══════════════════════════════════════════════════════════════

def _fc_line_fig(actual_label='Actual'):
    actual = TEST['dlog_CPI_index']
    var_fc = FC_VAR['dlog_CPI_index']
    var_lo = var_fc - 1.96 * RESID_STD['dlog_CPI_index']
    var_hi = var_fc + 1.96 * RESID_STD['dlog_CPI_index']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(var_hi.index) + list(var_lo.index[::-1]),
        y=list(var_hi.values) + list(var_lo.values[::-1]),
        fill='toself', fillcolor='rgba(37,99,235,0.12)',
        line={'color': 'rgba(0,0,0,0)'}, name='VAR 95% CI', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode='lines+markers',
                             name=actual_label, line={'color': 'black', 'width': 2.5}, marker={'size': 5},
                             hovertemplate=f'<b>{actual_label}</b> %{{x|%Y-%m}}: %{{y:.5f}}<extra></extra>'))
    fig.add_trace(go.Scatter(x=var_fc.index, y=var_fc.values, mode='lines+markers',
                             name=f'VAR (RMSE={RMSE_VAR:.5f})',
                             line={'color': BLUE, 'dash': 'dash'}, marker={'size': 4},
                             hovertemplate='<b>VAR</b> %{x|%Y-%m}: %{y:.5f}<extra></extra>'))
    if PREDS_XGB is not None:
        xidx = TEST.index[-len(PREDS_XGB):]
        fig.add_trace(go.Scatter(x=xidx, y=PREDS_XGB, mode='lines+markers',
                                 name=f'XGBoost (RMSE={RMSE_XGB:.5f})',
                                 line={'color': GREEN, 'dash': 'dot'}, marker={'size': 4},
                                 hovertemplate='<b>XGBoost</b> %{x|%Y-%m}: %{y:.5f}<extra></extra>'))
    fig.update_layout(xaxis_title='Quarter', yaxis_title='Δlog(CPI)',
                      template='plotly_white', height=380,
                      legend={'orientation': 'h', 'y': -0.2}, hovermode='x unified', margin={'t': 10})
    return fig

def _fc_bar_fig(y_label='RMSE (lower is better)'):
    labels = ['VAR', 'XGBoost']
    rmses  = [RMSE_VAR, RMSE_XGB]
    best_i = int(np.argmin(rmses))
    fig = go.Figure(go.Bar(
        x=labels, y=rmses, marker_color=[BLUE, GREEN],
        marker_line_color=[GOLD if i == best_i else WHITE for i in range(len(labels))],
        marker_line_width=[3 if i == best_i else 1 for i in range(len(labels))],
        text=[f'{r:.5f}' for r in rmses], textposition='outside',
        hovertemplate='<b>%{x}</b> RMSE: %{y:.5f}<extra></extra>',
    ))
    fig.add_annotation(x=labels[best_i], y=rmses[best_i], text="🏆",
                       showarrow=True, arrowhead=2, arrowcolor=GOLD, ay=-40)
    fig.update_layout(yaxis_title=y_label, template='plotly_white', height=300, margin={'t': 30})
    return fig

def _tab_fc(lang):
    t      = T[lang]
    rmses  = [RMSE_VAR, RMSE_XGB]
    best   = ['VAR', 'XGBoost'][int(np.argmin(rmses))]
    return html.Div([
        _section_title(t['fc_h'], t['fc_sub']),
        _card(
            html.H4(t['fc_what_h'], style={'color': NAVY, 'marginTop': 0}),
            html.P(t['fc_what_p'], style={'color': '#334155', 'lineHeight': '1.8', 'fontSize': '14px'}),
            html.Div([
                html.Div([html.Strong("VAR", style={'color': BLUE}), _badge(t['badge_econ']),
                          html.P(t['var_d'], style={'fontSize': '13px', 'color': DGRAY, 'marginTop': '6px'})],
                         style={'flex': '1', 'padding': '14px', 'background': '#eff6ff',
                                'borderRadius': '8px', 'marginRight': '10px', 'borderTop': f'3px solid {BLUE}'}),
                html.Div([html.Strong("XGBoost", style={'color': GREEN}), _badge(t['badge_ml'], GREEN),
                          html.P(t['xgb_d'], style={'fontSize': '13px', 'color': DGRAY, 'marginTop': '6px'})],
                         style={'flex': '1', 'padding': '14px', 'background': '#f0fdf4',
                                'borderRadius': '8px', 'marginRight': '10px', 'borderTop': f'3px solid {GREEN}'}),
                html.Div([html.Strong("LSTM", style={'color': '#7c3aed'}), _badge(t['badge_dl'], '#7c3aed'),
                          html.P(t['lstm_d'], style={'fontSize': '13px', 'color': DGRAY, 'marginTop': '6px'})],
                         style={'flex': '1', 'padding': '14px', 'background': '#faf5ff',
                                'borderRadius': '8px', 'borderTop': '3px solid #7c3aed'}),
            ], style={'display': 'flex', 'marginTop': '16px'}),
        ),
        _card(_section_title(t['fc_line_h'], t['fc_line_sub']),
              dcc.Graph(figure=_fc_line_fig(t['actual']))),
        _card(
            _section_title(t['fc_bar_h'], t['fc_bar_sub']),
            html.Div([
                dcc.Graph(figure=_fc_bar_fig(t['lower_better']), style={'flex': '1'}),
                html.Div([
                    html.Div("🏆", style={'fontSize': '36px', 'marginBottom': '8px'}),
                    html.H4(f"{best} {t['wins']}",
                            style={'color': NAVY, 'marginTop': 0, 'marginBottom': '8px'}),
                    html.P(f"{t['best_rmse']}{min(rmses):.5f}",
                           style={'color': DGRAY, 'fontSize': '13px'}),
                    html.Hr(style={'borderColor': '#e2e8f0'}),
                    html.P(t['fc_verdict_p'],
                           style={'color': '#334155', 'fontSize': '13px', 'lineHeight': '1.7'}),
                ], style={**CARD, 'flex': '1', 'margin': '0 0 0 20px'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ),
    ])

# ── run ───────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=False, port=8050)
