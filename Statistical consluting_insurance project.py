#Importing modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Logistic regression
df_freq= pd.read_csv(r"C:\Users\User\Desktop\KU Leuven. Msc Statistics. Year 1\Statistical Consulting\frequency.csv")
print(df_freq.head())

categorical_columns = ['gender', 'carType', 'carCat', 'job', 'uwYear']
numeric_columns = ['age', 'nYears', 'carVal', 'cover', 'density']

encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df_freq[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

X = pd.concat([df_freq[numeric_columns].reset_index(drop=True), one_hot_df], axis=1).values
y = (df_freq['claimNumbMD'] > 0).astype(int).values


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100]}
freq_model= Pipeline([
    ('scaler', StandardScaler()),
    ('logisticregression', LogisticRegression(solver='liblinear', max_iter=1000))
])

grid_search = GridSearchCV(
    freq_model,
    param_grid,
    cv=5,
    scoring='jaccard',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best C value: {grid_search.best_params_}")
print(f"Best cross-val Jaccard: {grid_search.best_score_:.3f}")

LR_best = grid_search.best_estimator_

yhat      = LR_best.predict(X_test)
yhat_prob = LR_best.predict_proba(X_test)

print(f"\nTest Jaccard Score:  {jaccard_score(y_test, yhat):.3f}")
print(f"\nClassification Report:\n{classification_report(y_test, yhat, target_names=['No Claim', 'Claim'])}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, yhat)}")

#Gamma regression
df_sev= pd.read_csv(r"C:\Users\User\Desktop\KU Leuven. Msc Statistics. Year 1\Statistical Consulting\severity.csv")
print(df_sev.head())

cat_cols = ['uwYear', 'gender', 'carType', 'carCat', 'job', 'cover']
for col in cat_cols:
    df_sev[col] = df_sev[col].astype('category')

df_sev_clean = df_sev[df_sev['claimSizeMD'] > 0].copy()
formula = "claimSizeMD ~ gender + carType + carCat + job + age + nYears + carVal + cover + density"
sev_model = smf.glm(formula=formula, data=df_sev_clean,
                family=sm.families.Gamma(link=sm.families.links.log())).fit()

print(sev_model.summary())

#Graphs
fitted = sev_model.fittedvalues
actual = df_sev_clean['claimSizeMD']
pearson_resid = sev_model.resid_pearson
deviance_resid = sev_model.resid_deviance

BG     = '#f7f5f0'
PANEL  = '#ffffff'
GRID   = '#e0dbd0'
TEXT   = '#2b2b2b'
MUTED  = '#888077'
BLUE   = '#2d6a9f'
RED    = '#c0392b'
GOLD   = '#d4820a'
GREEN  = '#2a7d4f'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': PANEL,
    'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT,
    'xtick.color': TEXT,
    'ytick.color': TEXT,
    'grid.color': GRID,
    'text.color': TEXT,
    'font.family': 'monospace',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

labels = {
    'gender[T.Male]': 'Gender: Male',
    'carType[T.B]': 'Car Type: B',
    'carType[T.C]': 'Car Type: C',
    'carType[T.D]': 'Car Type: D',
    'carType[T.E]': 'Car Type: E',
    'job[T.Housewife]': 'Job: Housewife',
    'job[T.Retired]': 'Job: Retired',
    'job[T.Self-employed]': 'Job: Self-employed',
    'job[T.Unemployed]': 'Job: Unemployed',
    'cover[T.1]': 'Cover: 1',
    'age': 'Age',
    'nYears': 'Years Insured',
    'density': 'Density'
}

coef = sev_model.params.drop('Intercept')
ci   = sev_model.conf_int().drop('Intercept')
pvals = sev_model.pvalues.drop('Intercept')
coef.index  = [labels.get(i, i) for i in coef.index]
ci.index    = coef.index
pvals.index = coef.index
significant = pvals < 0.05
# Graph 1: Coefficient Plot
coef_sig = coef[significant]
ci_sig   = ci[significant]
sorted_idx = coef_sig.abs().argsort()
coef_s = coef_sig.iloc[sorted_idx]
ci_s   = ci_sig.iloc[sorted_idx]
y_pos  = np.arange(len(coef_s))

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(y_pos, coef_s.values,
        xerr=[coef_s.values - ci_s.iloc[:, 0].values,
              ci_s.iloc[:, 1].values - coef_s.values],
        color=BLUE, alpha=0.85, height=0.6,
        error_kw=dict(ecolor='#aaa', capsize=3, lw=1.2))
ax.axvline(0, color=RED, lw=1.5, linestyle='--', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(coef_s.index, fontsize=9)
ax.set_xlabel('Coefficient (log scale)', fontsize=9)
ax.set_title(f'Significant Predictors (p < 0.05)  ·  n={len(coef_s)}  ·  95% CI',
             fontsize=11, fontweight='bold', pad=12)
ax.grid(axis='x', lw=0.7, alpha=0.6)

for i, val in enumerate(coef_s.values):
    pct = (np.exp(val) - 1) * 100
    if abs(pct) > 12:
        ax.text(val + (0.012 if val > 0 else -0.012), i,
                f'{pct:+.0f}%', va='center',
                ha='left' if val > 0 else 'right',
                color=GOLD, fontsize=8)

plt.tight_layout()
plt.savefig('plot1_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()

# Graph 2: Actual vs Fitted
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(np.log(actual), np.log(fitted),
           alpha=0.2, s=8, color=BLUE, rasterized=True)
lims = [min(np.log(actual).min(), np.log(fitted).min()),
        max(np.log(actual).max(), np.log(fitted).max())]
ax.plot(lims, lims, color=RED, lw=1.8, linestyle='--', label='Perfect fit')
ax.set_xlabel('log(Actual Claim Size)', fontsize=10)
ax.set_ylabel('log(Fitted Claim Size)', fontsize=10)
ax.set_title('Actual vs Fitted Values', fontsize=11, fontweight='bold', pad=12)
ax.legend(fontsize=9)
ax.grid(lw=0.7, alpha=0.6)
plt.tight_layout()
plt.savefig('plot2_actual_vs_fitted.png', dpi=150, bbox_inches='tight')
plt.show()

# Graph 3: Residuals vs Fitted
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(np.log(fitted), pearson_resid,
           alpha=0.2, s=8, color=BLUE, rasterized=True)
ax.axhline(0,  color=RED,  lw=1.8, linestyle='--')
ax.axhline(2,  color=MUTED, lw=1,  linestyle=':', alpha=0.7)
ax.axhline(-2, color=MUTED, lw=1,  linestyle=':', alpha=0.7)
ax.text(np.log(fitted).max(), 2.1,  '±2', color=MUTED, fontsize=8, ha='right')
ax.text(np.log(fitted).max(), -2.3, '±2', color=MUTED, fontsize=8, ha='right')
ax.set_xlabel('log(Fitted Values)', fontsize=10)
ax.set_ylabel('Pearson Residuals', fontsize=10)
ax.set_title('Residuals vs Fitted', fontsize=11, fontweight='bold', pad=12)
ax.grid(lw=0.7, alpha=0.6)
plt.tight_layout()
plt.savefig('plot3_residuals.png', dpi=150, bbox_inches='tight')
plt.show()

# Graph 4: Median Claim by Job
job_stats = df_sev_clean.groupby('job')['claimSizeMD'].median().sort_values()
bar_colors = [RED if j in ['Retired', 'Unemployed'] else BLUE
              for j in job_stats.index]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.barh(job_stats.index, job_stats.values,
               color=bar_colors, alpha=0.85, height=0.55)
ax.set_xlabel('Median Claim Size (€)', fontsize=10)
ax.set_title('Median Claim Size by Job Category', fontsize=11, fontweight='bold', pad=12)
ax.grid(axis='x', lw=0.7, alpha=0.6)
for i, v in enumerate(job_stats.values):
    ax.text(v + 5, i, f'€{v:.0f}', va='center', color=GOLD, fontsize=9)
plt.tight_layout()
plt.savefig('plot5_job_medians.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plot1–plot5")

#Combining both models
p_claim = grid_search.predict_proba(X)[:, 1]
e_severity = sev_model.predict(df_freq)
df_freq['p_claim']       = p_claim
df_freq['e_severity']    = e_severity
df_freq['expected_cost'] = p_claim * e_severity
print(df_freq[['p_claim', 'e_severity', 'expected_cost']].describe())

# Assume 15% most expensive clients are unprofitable
threshold = df_freq['expected_cost'].quantile(0.85)
df_freq['profitable'] = np.where(
    df_freq['expected_cost'] > threshold,
    'Unprofitable', 'Profitable'
)
print(df_freq.groupby('profitable')[['age','density','carVal','expected_cost']].mean())

# Profile unprofitable customers
print(df_freq.groupby('profitable')[
    ['age', 'density', 'carVal', 'expected_cost']
].mean().round(2))
print(df_freq.groupby(['profitable', 'job']).size().unstack())
print(df_freq.groupby(['profitable', 'carType']).size().unstack())

# Table of persons from frequency dataset with expected cost and label as profitable/unprofitable
output = df_freq[['gender', 'job', 'carType', 'age', 'density', 'cover',
                   'p_claim', 'e_severity', 'expected_cost']].copy()
output['p_claim']       = output['p_claim'].round(3)
output['e_severity']    = output['e_severity'].round(0).astype(int)
output['expected_cost'] = output['expected_cost'].round(0).astype(int)
output['profitable']    = np.where(output['expected_cost'] > threshold,
                                   'Unprofitable', 'Profitable')
print(output.head(20).to_string(index=True))
