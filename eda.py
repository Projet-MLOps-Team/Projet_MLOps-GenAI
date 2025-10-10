""" EDA pour MLOPS """

# Pour VSCODE seulement
# python -m venv .venv
# source .venv/bin/activate

from pathlib import Path
import pandas as pd

# 1) Load the dataset from the GitHub URL
url = "https://raw.githubusercontent.com/jiwon-yi/Projet_MLOps/main/Loan_Data.csv"
df = pd.read_csv(url)

print(df.shape)
print(df.columns.tolist()[:8])

display(df)

display(df)
print(df.describe())

nan_counts = df.isna().sum()
print("NaN :\n", nan_counts)

dup_total = df.duplicated().sum()
print(f"Nombre de lignes dupliquées : {dup_total}")

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

default_counts = df['default'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(default_counts, labels=default_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Default')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Select numerical columns (excluding 'customer_id' and 'default' as they are not features for distribution analysis)
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols.remove('customer_id')
numerical_cols.remove('default')

# Create histograms for each numerical column
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Create the 'debt_ratio' feature
df['debt_ratio'] = df['total_debt_outstanding'] / df['income']

# Display the first few rows with the new feature
display(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot of debt_ratio vs default
sns.scatterplot(data=df, x='debt_ratio', y='default', hue='default', alpha=0.6, ax=axes[1])
axes[1].set_title('Relationship between Debt Ratio and Default')
axes[1].set_xlabel('Debt Ratio')
axes[1].set_ylabel('Default (0: No, 1: Yes)')
axes[1].set_yticks([0, 1])
axes[1].grid(axis='y', linestyle='--')

# Scatter plot of fico_score vs default
sns.scatterplot(data=df, x='fico_score', y='default', hue='default', alpha=0.6, ax=axes[0])
axes[0].set_title('Relationship between FICO Score and Default')
axes[0].set_xlabel('FICO Score')
axes[0].set_ylabel('Default (0: No, 1: Yes)')
axes[0].set_yticks([0, 1])
axes[0].grid(axis='y', linestyle='--')

plt.tight_layout()
plt.show()

datasetfinal = df
display(datasetfinal)

"""## Dataset Description

This dataset contains information about customer loan applications and their default status. It includes the following columns:

*   **customer\_id**: Unique identifier for each customer.
*   **credit\_lines\_outstanding**: The number of credit lines outstanding for the customer.
*   **loan\_amt\_outstanding**: The amount of the loan outstanding.
*   **total\_debt\_outstanding**: The total amount of debt outstanding for the customer.
*   **income**: The customer's income.
*   **years\_employed**: The number of years the customer has been employed.
*   **fico\_score**: The customer's FICO score, a creditworthiness indicator.
*   **default**: The target variable, indicating whether the customer defaulted on the loan (1 for default, 0 for no default).
*   **debt\_ratio**: The ratio of total debt outstanding to income (calculated feature).
"""