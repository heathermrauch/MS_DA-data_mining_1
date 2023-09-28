import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error as MSE

# Create output directory
os.makedirs('./output2', exist_ok=True)

# Load the data
medical_clean = pd.read_csv('./medical_clean.csv')

# View the data
print(f'\n{medical_clean.head().to_string()}\n')

# Evaluate the data structures
print(f'\ncontinuous data:\n{medical_clean.describe().T.to_string()}')
cat_data = medical_clean[
    [col for col in medical_clean.columns
     if medical_clean[col].dtype not in ['float64', 'int64']]
]
print(f'\ncategorical data:\n{cat_data.describe().T.to_string()}\n')

# Evaluate data types and check for nulls
medical_clean.info()

# Check for duplicates
print(f'\nduplicates:\n{medical_clean.duplicated().sum()}')

# Check for outliers
num_cols = [col for col in medical_clean.columns
            if medical_clean[col].dtype in ['float64', 'int64']]
df_num = medical_clean[num_cols]
df_zscores = df_num.apply(stats.zscore)
df_outliers = df_zscores.apply(lambda x: (x > 3) | (x < -3))
print(f'\noutliers:\n{df_outliers.sum().to_string()}')

# Look for correlation between features
corr = df_num.corr()
print(f'\ncorrelation:\n{corr.to_string()}')

# Encode categorical features
excluded = ['CaseOrder', 'Customer_id', 'Interaction', 'UID',
            'TimeZone', 'City', 'State', 'County', 'Job']
y_n_cols = [col for col in medical_clean.columns
            if medical_clean[col].nunique() == 2
            and col not in excluded]
num_cols = [col for col in medical_clean.columns
            if medical_clean[col].dtype in ['int64', 'float64']
            and col not in y_n_cols
            and col not in excluded]
cat_cols = [col for col in medical_clean.columns
            if col not in num_cols
            and col not in y_n_cols
            and col not in excluded]
df_encoded = medical_clean[[col for col in medical_clean.columns
                            if col not in excluded]]
df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)
df_encoded.replace({'Yes': 1, 'No': 0}, inplace=True)

# Fix column names
df_encoded.columns = [col.replace(' ', '_') for col in df_encoded.columns]
print(f'\nencoded data:\n{df_encoded.head().to_string()}')

# Scale numeric features
df_scaled = df_encoded.copy()
scaler = StandardScaler()
df_scaled.loc[:, num_cols] = \
    scaler.fit_transform(df_scaled.loc[:, num_cols])
print(f'\nscaled data:\n{df_scaled.head().to_string()}')

# Select features
selector = SelectFwe(f_regression, alpha=0.05)
X = df_scaled[[col for col in df_scaled.columns if col != 'TotalCharge']]
y = df_scaled['TotalCharge']
X = pd.DataFrame(selector.fit_transform(X, y),
                 columns=selector.get_feature_names_out(selector.feature_names_in_))
feature_data = pd.DataFrame({'feature': selector.feature_names_in_,
                             'score': selector.scores_,
                             'pvalue': selector.pvalues_})
top_features = feature_data.iloc[selector.get_support(indices=True)].\
    sort_values(by='score', ascending=False)
print(f'\nfeature scores:\n{top_features.to_string()}')

# Save cleaned data set
df = pd.concat([y, X], axis=1)
print(f'\nfinal data:\n{df.head().to_string()}')
df.to_csv('./output2/cleaned_data_set.csv')

# Split into train and test data
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=42)
X_train.to_csv('./output2/X_train.csv')
X_test.to_csv('./output2/X_test.csv')
y_train.to_csv('./output2/y_train.csv')
y_test.to_csv('./output2/y_test.csv')

# Hyperparameter tuning
model = RandomForestRegressor(n_jobs=-1, random_state=42)
param_grid = {'n_estimators': np.arange(10, 110, 10),
              'max_depth': np.arange(1, 12, 2),
              'min_samples_leaf': np.arange(0.001, 0.01, 0.005),
              'max_samples': np.arange(0.1, 1, 0.2)}
search = RandomizedSearchCV(model, param_grid, cv=10, n_jobs=-1,
                            random_state=42)
search.fit(X_train, y_train)
print(f'\nhyperparameter tuning:'
      f'\n\tbest parameters:')
for k, v in search.best_params_.items():
    print(f'\t\t{k}: {v}')
print(f'\tbest score: {search.best_score_}')

# Make predictions
y_pred = search.predict(X_test)

# Metrics
features = pd.DataFrame({
    'feature': search.feature_names_in_,
    'importance': search.best_estimator_.feature_importances_
})
features_sorted = features.sort_values(by='importance', ascending=False)
print(f'\nfeature importance:\n{features_sorted.to_string()}')

fig, ax = plt.subplots()
features_sorted.sort_values(by='importance', inplace=True)
features_sorted.plot(x='feature', y='importance', kind='barh', ax=ax,
                     logx=True, rot=20, legend=False)
ax.set_xlabel('importance (log scale)')
fig.suptitle('Feature Importance')
fig.set_tight_layout(True)
plt.savefig('./output2/feature_importance.png')
plt.show()

r_squared = r2_score(y_test, y_pred)
mse = MSE(y_test, y_pred)
rmse = mse ** (1/2)
print(f'\nr-squared: {r_squared}'
      f'\nmean squared error: {mse}'
      f'\nroot mean squared error: {rmse}')
