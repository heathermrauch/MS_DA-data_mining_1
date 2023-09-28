import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, RocCurveDisplay

# Create output directory
os.makedirs('./output1', exist_ok=True)

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
excluded = ['CaseOrder', 'Customer_id', 'Interaction',
            'UID', 'City', 'State', 'County', 'Job']
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
df_encoded = pd.get_dummies(df_encoded, columns=cat_cols)
df_encoded.replace({'Yes': 1, 'No': 0}, inplace=True)
# Fix column names
df_encoded.columns = [col.replace(' ', '_') for col in df_encoded.columns]
print(f'\nencoded data:\n{df_encoded.head().to_string()}')

# Scale numeric variables
df_scaled = df_encoded.copy()
scaler = StandardScaler()
df_scaled.loc[:, num_cols] = \
    scaler.fit_transform(df_scaled.loc[:, num_cols])
print(f'\nscaled data:\n{df_scaled.head().to_string()}')

# Select features
selector = SelectKBest()
X = df_scaled[[col for col in df_scaled.columns if col != 'ReAdmis']]
y = df_scaled['ReAdmis']
selector.fit_transform(X, y)
feature_data = pd.DataFrame({'feature': selector.feature_names_in_,
                             'score': selector.scores_,
                             'pvalue': selector.pvalues_})
top_features = feature_data[feature_data.pvalue < 0.05].\
    sort_values(by='score', ascending=False)
print(f'\nfeature scores:\n{top_features.to_string()}')

columns = ['ReAdmis', 'Initial_days', 'Services_CT_Scan', 'Children',
           'Marital_Divorced', 'Services_Intravenous', 'Population',
           'Initial_admin_Emergency_Admission']
df = df_scaled[columns]

# Save cleaned data set
df_encoded.to_csv('./output1/cleaned_data_set.csv')

# Split into train and test data
X = df_encoded.loc[:, df_encoded.columns[1:]]
y = df_encoded.ReAdmis
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train.to_csv('./output1/X_train.csv')
X_test.to_csv('./output1/X_test.csv')
y_train.to_csv('./output1/y_train.csv')
y_test.to_csv('./output1/y_test.csv')

# Instantiate and fit naive bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f'\nAccuracy score: {accuracy}\nAUC: {auc}')

# Plot ROC curve
RocCurveDisplay.from_predictions(y_test, y_pred, color='darkorange')
plt.axline((0, 0), (1, 1), linestyle='dashed', color='gray')
plt.title('ReAdmis Naive Bayes Classifier ROC Curve')
plt.savefig('./output1/ROC_curve.png')
plt.show()
