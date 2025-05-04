


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
columns = [
    'Status', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount', 'Savings', 'EmploymentSince',
    'InstallmentRate', 'PersonalStatusSex', 'OtherDebtors', 'ResidenceSince', 'Property', 'Age',
    'OtherInstallmentPlans', 'Housing', 'ExistingCredits', 'Job', 'NumPeopleLiable', 'Telephone',
    'ForeignWorker', 'Target'
]
df = pd.read_csv(url, sep=' ', header=None, names=columns)

# Encode the target variable: 1 for good credit, 0 for bad credit
df['Target'] = df['Target'].map({1: 1, 2: 0})

# Select continuous variables
continuous_vars = ['Duration', 'CreditAmount', 'Age']

# Encode categorical variables
categorical_vars = ['Status', 'CreditHistory', 'Purpose', 'Savings', 'EmploymentSince',
                    'PersonalStatusSex', 'OtherDebtors', 'Property', 'OtherInstallmentPlans',
                    'Housing', 'Job', 'Telephone', 'ForeignWorker']
df_encoded = df.copy()
for col in categorical_vars:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])

# Prepare features and target
X = df_encoded[continuous_vars + categorical_vars]
y = df_encoded['Target']

# Fit logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Predict probabilities
probabilities = model.predict_proba(X)[:, 1]
log_odds = np.log(probabilities / (1 - probabilities))

# Plotting the relationship between each continuous variable and log-odds
for var in continuous_vars:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[var], y=log_odds)
    plt.title(f'Log-Odds vs {var}')
    plt.xlabel(var)
    plt.ylabel('Log-Odds')
    plt.show()

# Interpretation: These plots help assess whether the assumption of linearity between continuous predictors and the log-odds of the outcome is reasonable.​


# 2. Independence of Observations
# Assumption: Observations should be independent; that is, the outcome for one observation should not influence another.​

# Since the German Credit dataset does not include time-series or grouped data,
# we assume independence of observations. However, if there were time-related data,
# we could check for autocorrelation as follows:

from statsmodels.graphics.tsaplots import plot_acf

# Example: Check autocorrelation in the 'Duration' variable
plot_acf(df['Duration'])
plt.title('Autocorrelation of Duration')
plt.show()

# Interpretation: In the absence of time-series or grouped data, the independence assumption is generally satisfied.​


# 3. No Multicollinearity Among Independent Variables
# Assumption: Independent variables should not be highly correlated with each other.​

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

print(vif_data)

# Interpretation: VIF values greater than 5 (or 10) indicate multicollinearity. If detected, consider removing or combining correlated variables.​

# 4. Sufficient Sample Size
# Assumption: A large enough sample size is needed to ensure reliable estimates. A common rule is at least 10 events per predictor variable.​

# Count the number of events (good credit) and non-events (bad credit)
event_count = df['Target'].sum()
non_event_count = len(df) - event_count
num_predictors = X.shape[1]

print(f'Events: {event_count}, Non-events: {non_event_count}, Predictors: {num_predictors}')

# Interpretation: Ensure that both event and non-event counts are sufficient relative to the number of predictors.​


# 5. No Influential Outliers
# Assumption: Outliers can disproportionately affect model estimates.​

import statsmodels.api as sm

# Add constant term for intercept
X_const = sm.add_constant(X)

# Fit logistic regression model
logit_model = sm.Logit(y, X_const)
result = logit_model.fit(disp=0)

# Calculate influence measures
influence = result.get_influence()
cooks_d = influence.cooks_distance[0]

# Plot Cook's distance
plt.figure(figsize=(8, 4))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.title("Cook's Distance for Influential Points")
plt.xlabel('Observation Index')
plt.ylabel("Cook's Distance")
plt.show()

# Interpretation: Observations with high Cook's distance may be influential and warrant further investigation.​

# 6. Binary Dependent Variable
#Assumption: The dependent variable should be binary for standard logistic regression.​

# Check the unique values in the target variable
print(df['Target'].value_counts())

# Interpretation: The target variable has two classes: 1 (good credit) and 0 (bad credit), satisfying the binary requirement.​
