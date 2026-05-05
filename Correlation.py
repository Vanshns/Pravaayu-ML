import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load your pruned data
df = pd.read_csv('pruned_clinical_data.csv')

# 2. Calculate the correlation matrix
# We focus on how symptoms correlate with each other
corr_matrix = df.corr()

# 3. Filter for "Strong" Relationships
# We don't care about weak correlations (near 0) or a symptom correlating with itself (1.0)
# This snippet unstacks the matrix and filters for values between 0.5 and 0.99
sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                  .stack()
                  .sort_values(ascending=False))

# 4. Display the Top 20 Strongest Links
print("Top 20 Symptom Correlations (The 'Hooks'):")
print(sol.head(20))

# 5. Optional: Visualize the 'Hot Zones'
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title("Clinical Symptom Correlation Map")
plt.show()