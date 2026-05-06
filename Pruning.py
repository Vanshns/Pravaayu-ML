# pruning.py
import pandas as pd

# 1. Load your dataset
df = pd.read_csv('flattened_patients.csv')

# 2. Record initial size
initial_cols = len(df.columns)

# 3. Remove "Constant" columns
# These are columns where every single patient has the same value (e.g., all 0s)
df_pruned = df.loc[:, (df != df.iloc[0]).any()]

# 4. Remove "Ultra-Sparse" columns
# If a symptom only appears in 1 or 2 patients out of 144, 
# it's usually too rare to find a statistical correlation.
# Here we keep columns that have at least 5 non-zero entries.
threshold = 5 
df_pruned = df_pruned.loc[:, (df_pruned != 0).sum(axis=0) >= threshold]

# 5. Review the results
print(f"Started with {initial_cols} columns.")
print(f"Ended with {len(df_pruned.columns)} columns.")
print("---")
print("These are your high-signal symptoms for the model:")
print(df_pruned.columns.tolist())

# Save this cleaned version for the next step
df_pruned.to_csv('pruned_clinical_data.csv', index=False)