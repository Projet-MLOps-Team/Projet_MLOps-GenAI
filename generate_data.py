import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'customer_id': range(1, n_samples + 1),
    'credit_lines_outstanding': np.random.randint(0, 20, n_samples),
    'loan_amt_outstanding': np.random.randint(5000, 50000, n_samples),
    'total_debt_outstanding': np.random.randint(10000, 100000, n_samples),
    'income': np.random.randint(30000, 150000, n_samples),
    'years_employed': np.random.randint(0, 40, n_samples),
    'fico_score': np.random.randint(300, 850, n_samples),
    'default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
})

df.to_csv('Loan_Data.csv', index=False)
print('Loan_Data.csv créé avec succès!')
