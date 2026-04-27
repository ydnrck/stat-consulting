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

import matplotlib.pyplot as plt
import numpy as np

BLUE  = '#2d6a9f'
RED   = '#c0392b'
MUTED = '#888077'
TEXT  = '#2b2b2b'

# Significant predictors only, as % effect
names   = ['Density', 'Age', 'Years Insured', 'Cover: Yes',
           'Job: Housewife', 'Gender: Male', 'Job: Self-employed',
           'Car Type: C', 'Job: Unemployed', 'Car Type: D',
           'Car Type: E', 'Job: Retired']
effects = [0.16, -1.24, -0.50, -14.1,
           -10.0, 13.9, 8.4,
           -10.9, 27.2, -18.4,
           -20.3, 96.4]

# Sort by effect size
sorted_pairs = sorted(zip(effects, names))
effects_s, names_s = zip(*sorted_pairs)
colors = [RED if e > 0 else BLUE for e in effects_s]

fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_edgecolor('#cccccc')
ax.spines['bottom'].set_edgecolor('#cccccc')

y_pos = np.arange(len(names_s))
ax.barh(y_pos, effects_s, color=colors, alpha=0.88, height=0.6)
ax.axvline(0, color=TEXT, lw=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(names_s, fontsize=8.5, fontfamily='serif')
ax.set_xlabel('Effect on Claim Size (%)', fontsize=9,
              fontfamily='serif', color=TEXT)
ax.set_title('Figure X. Significant Predictors of Claim Severity\n'
             '(percentage change relative to reference category)',
             fontsize=9.5, fontweight='bold', fontfamily='serif',
             color=TEXT, pad=10)
ax.tick_params(colors=TEXT, labelsize=8.5)
ax.grid(axis='x', lw=0.5, alpha=0.4, linestyle='--')
ax.set_axisbelow(True)

# Value labels
for i, v in enumerate(effects_s):
    offset = 1.5 if v > 0 else -1.5
    ha     = 'left' if v > 0 else 'right'
    label  = f'+{v:.1f}%' if v > 0 else f'{v:.1f}%'
    ax.text(v + offset, i, label, va='center', ha=ha,
            fontsize=7.5, color=TEXT, fontfamily='serif')

ax.set_xlim(min(effects_s) - 15, max(effects_s) + 20)

plt.tight_layout(pad=1.5)
plt.savefig('figure_severity_coefficients.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: figure_severity_coefficients.png")

#Unprofitable Rate by Job 
fig1, ax1 = plt.subplots(figsize=(8, 5))
fig1.patch.set_facecolor('white')
serious_style(ax1)
bar_colors = [RED if p > 15 else BLUE for p in unprof_pct]
ax1.barh(job_categories, unprof_pct, color=bar_colors, alpha=0.9, height=0.5)
ax1.axvline(15, color=MUTED, lw=1.5, linestyle='--', label='Portfolio threshold (15%)')
ax1.set_xlabel('Share Flagged as Unprofitable (%)', fontsize=10,
               fontfamily='serif', color=TEXT)
ax1.set_title('Figure 3. Unprofitable Rate by Occupation',
              fontsize=11, fontweight='bold', fontfamily='serif', pad=12, color=TEXT)
ax1.legend(fontsize=9, framealpha=0.4, prop={'family': 'serif'})
ax1.grid(axis='x', lw=0.6, alpha=0.4, linestyle='--')
ax1.set_xlim(0, max(unprof_pct) + 10)
for label in ax1.get_yticklabels():
    label.set_fontfamily('serif')
    label.set_fontsize(10)
for i, v in enumerate(unprof_pct):
    ax1.text(v + 0.5, i, f'{v:.1f}%', va='center',
             fontsize=9, color=TEXT, fontfamily='serif')
plt.tight_layout(pad=2.0)
plt.savefig('figure3_unprofitable_rate.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: figure3_unprofitable_rate.png")
