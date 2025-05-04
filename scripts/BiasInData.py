# Step 1: Load and Prepare Data

import pandas as pd
import numpy as np

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
columns = ['Status', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount', 'Savings', 'EmploymentSince',
           'InstallmentRate', 'PersonalStatusSex', 'OtherDebtors', 'ResidenceSince', 'Property', 'Age',
           'OtherInstallmentPlans', 'Housing', 'ExistingCredits', 'Job', 'NumPeopleLiable', 'Telephone',
           'ForeignWorker', 'Target']
df = pd.read_csv(url, sep=' ', header=None, names=columns)

# Target: 1 = good credit, 2 = bad credit -> Map to binary
df['Target'] = df['Target'].map({1: 1, 2: 0})


# Step 2: Check for Gender Bias
# The 'PersonalStatusSex' field encodes both sex and marital status.

# Group by gender status
df['Gender'] = df['PersonalStatusSex'].apply(lambda x: 'male' if 'male' in x else 'female')

# Approval rates by gender
gender_rates = df.groupby('Gender')['Target'].mean()
print("Approval Rates by Gender:")
print(gender_rates)

# Count of approved applications by gender
print("\nApproved Counts:")
print(df[df['Target'] == 1]['Gender'].value_counts())

# Step 3: Check for Age Bias
# We check approval rates below and above a threshold (e.g., 25 or 50).

# Create age groups
df['AgeGroup'] = df['Age'].apply(lambda x: '<=25' if x <= 25 else '>25')

# Approval rate by age group
age_rates = df.groupby('AgeGroup')['Target'].mean()
print("Approval Rates by Age Group:")
print(age_rates)

# Counts
print("\nApproved Counts by Age Group:")
print(df[df['Target'] == 1]['AgeGroup'].value_counts())

# Step 4: Compute Statistical Parity Difference
#Statistical Parity Difference (SPD):

#SPD = P(Y^=1 ∣ unprivileged) − P(Y^=1 ∣ privileged)
#SPD=P( Y^ =1∣unprivileged)−P(Y^=1∣privileged)
#We’ll assume:
#Privileged gender: male
#Privileged age: >25

# Gender-based SPD
p_male = df[df['Gender'] == 'male']['Target'].mean()
p_female = df[df['Gender'] == 'female']['Target'].mean()
spd_gender = p_female - p_male
print(f"Statistical Parity Difference (female - male): {spd_gender:.3f}")

# Age-based SPD
p_younger = df[df['AgeGroup'] == '<=25']['Target'].mean()
p_older = df[df['AgeGroup'] == '>25']['Target'].mean()
spd_age = p_younger - p_older
print(f"Statistical Parity Difference (<=25 - >25): {spd_age:.3f}")

# Step 5: Visualize Approval Distributions

import seaborn as sns
import matplotlib.pyplot as plt

# Gender
sns.barplot(x='Gender', y='Target', data=df)
plt.title("Approval Rate by Gender")
plt.ylabel("Approval Rate")
plt.show()

# Age
sns.barplot(x='AgeGroup', y='Target', data=df)
plt.title("Approval Rate by Age Group")
plt.ylabel("Approval Rate")
plt.show()

