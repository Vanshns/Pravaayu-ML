Order to run:
python .\Pravaayu_TransformerScript.py
python .\Pruning.py
python .\Apriori.py

Support:
How frequently a rule appears in the dataset
Support(A → B) = P(A ∩ B)

Confidence:
Probability of B occurring given A
Confidence(A → B) = P(B | A)

Lift:
How much more likely A and B occur together compared to chance
Lift(A → B) = P(A ∩ B) / (P(A) \* P(B))


Support chosen >=0.12
Why: without it you get rules like something occurs in 2% of data but has 100% confidence. That’s overfitting.
This just means that this just appears in more than 12% of the patients, which is not too rare but still allows discovery.
If you go for lower support then you might get rare patterns but the data will have noise too.
If you go for highrer support then you might find only common patterns but it is more stable.

Confidence chosen is 0.75
Means we can say that if A happens then there is a 75% chance that B happens.
This is strong enought to trust but not overly strict.
Lower confidence would mean more rules but they would be weak signals.
Higher confidence would mean fewer rules but stronger signals

Note: We have also removed rules with confidence = 1. This is because this can mean that the subset is 

Lift is chosen to be more than 1.5 but less than 5
It just means that A and B occur together more than chance  also we dont go higher than 5 because instead of insight we would just be describing some patient which removes overfit rules and keeps generalizable patterns.

we set a limit on the number of conditions because too many conditions would mean that for only this exact combination of features we get this result.


# Pravaayu ML Project Documentation

## Project Overview

This project is an end-to-end clinical data processing and analytics pipeline built for Pravaayu.
The goal of the project is to:

1. Fetch patient clinical records from the Pravaayu API.
2. Transform nested CRF (Clinical Record Form) JSON data into structured ML-ready tabular data.
3. Clean and prune noisy/unusable features.
4. Discover symptom relationships using Association Rule Mining (Apriori).
5. Generate patient scoring metrics (currently knee scoring).
6. Analyze symptom correlations.

This document is intended for future interns or developers who will continue maintaining or improving the project.

---

# 1. Full Project Flow

The pipeline works in the following order:

```text
API Fetch
   ↓
patients.json
   ↓
Pravaayu_TransformerScript.py
   ↓
flattened_patients.csv
   ↓
Pruning.py
   ↓
pruned_clinical_data.csv
   ↓
Apriori.py
   ↓
clinical_logic_rules.csv
   ↓
scoring.py
   ↓
scored_patients.csv
```

Additional analysis:

```text
Correlation.py
```

This generates a heatmap and strong feature correlations.

---

# 2. Folder Structure

```text
Pravaayu-ML-main/
│
├── Apriori.py
├── Correlation.py
├── NewdDataFetch.py
├── Pravaayu_TransformerScript.py
├── Pruning.py
├── run_pipeline.py
├── scoring.py
│
├── patients.json
├── flattened_patients.csv
├── pruned_clinical_data.csv
├── scored_patients.csv
├── final_actionable_insights_clean.csv
│
├── requirements.txt
└── readme.md
```

---

# 3. Environment Setup

## Python Version

Recommended:

```bash
Python 3.10+
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Current dependencies:

```text
pandas
numpy
mlxtend
networkx
matplotlib
```

Additional packages sometimes needed:

```bash
pip install seaborn requests
```

---

# 4. Running the Full Pipeline

## Easiest Method

Run:

```bash
python run_pipeline.py
```

This automatically runs:

```python
pipeline = [
    "NewdDataFetch.py",
    "Pravaayu_TransformerScript.py",
    "Pruning.py",
    "Apriori.py",
    "scoring.py"
]
```

---

# 5. Detailed Explanation of Each File

---

# 5.1 NewdDataFetch.py

## Purpose

Fetches patient CRF data from the Pravaayu API and stores it in:

```text
patients.json
```
---

## Key Functions

### load_existing_data()

Loads previously saved patients from:

```text
patients.json
```

This prevents losing previous fetches.

---

### save_data(data_list)

Writes updated patient data into:

```text
patients.json
```

---

### fetch_patient_details(patient_id, file_no)

Fetches detailed patient CRF data.

Logic:

```python
if file_no and file_no != "-":
    use file number
else:
    use patient id
```

This exists because some records may not contain proper file numbers.

---

## Expected Output

```text
patients.json
```

Contains raw nested patient records.

---

## Common Problems

### API returns 401 / unauthorized

Possible reasons:

* Expired auth/session
* API access changed
* Missing headers/cookies

Solution:

* Check browser network requests
* Copy latest headers/cookies
* Update requests logic

---

### JSON corruption

If interrupted during writing.

Fix:

Delete:

```text
patients.json
```

and rerun.

---

# 5.2 Pravaayu_TransformerScript.py

## Purpose

MOST IMPORTANT FILE IN THE PROJECT.

Converts deeply nested clinical JSON into a structured flat dataframe for ML and analytics.

Input:

```text
patients.json
```

Output:

```text
flattened_patients.csv
```

---

## Why This File Exists

Machine learning models and analytics tools require:

```text
rows = patients
columns = features
```

But the API returns nested JSON.

This script converts:

```json
{
  "knee": {
    "left": {
      "pain": "Yes"
    }
  }
}
```

into:

```text
knee_l_pain = 1
```

---

## Important Helper Functions

### normalize(val)

Standardizes strings:

```python
" Yes " -> "yes"
```

Avoids inconsistent values.

---

### yn(val)

Converts:

```text
Yes -> 1
No -> 0
```

Used throughout the pipeline.

---

### extract_duration(durations, key)

Extracts duration values safely.

Handles multiple cases:

```python
{"val": 3, "unit": "days"}
```

OR

```python
"3"
```

Returns:

```python
{
    "duration": int,
    "duration_type": str
}
```

---

## Important Design Decision

This file uses:

```text
fixed-width feature extraction
```

Meaning:

Every patient gets the SAME columns.

This is critical for:

* ML training
* Scoring
* Statistical analysis
* Correlation analysis
* Rule mining

---

## Example Feature Names

```text
knee_l_pain_walking
knee_r_exam_creps_right
lumbar_stiffness
cervical_duration_val
```

Naming pattern:

```text
bodypart_side_category_feature
```

---

## If You Add New Features

You MUST:

1. Add extraction logic.
2. Ensure missing values default safely.
3. Keep naming consistent.
4. Regenerate:

```text
flattened_patients.csv
```

5. Rerun pruning + scoring.

---

## Common Problems

### KeyError

Happens when API structure changes.

Fix:

Use:

```python
.get()
```

instead of direct indexing.

---

### Missing columns

Possible reasons:

* Feature not extracted
* Feature pruned later
* Typo in naming

---

# 5.3 Pruning.py

## Purpose

Removes useless/noisy columns.

Input:

```text
flattened_patients.csv
```

Output:

```text
pruned_clinical_data.csv
```

---

## What It Removes

### 1. Constant Columns

Columns where ALL patients have same value.

Example:

```text
all zeros
```

These provide no information.

---

### 2. Ultra-Sparse Columns

Features appearing in very few patients.

Threshold:

```python
threshold = 5
```

Meaning:

Feature must appear in at least 5 patients.

---

## Why Pruning Matters

Without pruning:

* Huge dimensionality
* Noise
* Overfitting
* Poor Apriori rules
* Slow execution

---

## Important Logic

```python
(df_pruned != 0).sum(axis=0) >= threshold
```

Counts non-zero occurrences.

---

## Output

```text
pruned_clinical_data.csv
```

This is the main analytics dataset.

---

# 5.4 Apriori.py

## Purpose

Discovers clinical relationships between symptoms.

Example:

```text
If symptom A exists,
symptom B is also likely.
```

Uses:

```python
mlxtend.frequent_patterns
```

---

## Main Concepts

### Support

How often rule appears.

Formula:

```text
Support(A → B) = P(A ∩ B)
```

---

### Confidence

Probability of B given A.

Formula:

```text
Confidence(A → B) = P(B | A)
```

---

### Lift

How much more likely A and B occur together compared to random chance.

Formula:

```text
Lift(A → B) = P(A ∩ B) / (P(A) × P(B))
```

---

## Current Threshold Choices

### Support >= 0.12

Reason:

Avoids extremely rare patterns.

Too low:

* noisy
* overfitting

Too high:

* misses useful insights

---

### Confidence >= 0.75

Means:

If A happens,
there is at least 75% chance B happens.

---

### Lift between 1.5 and 5

Reason:

* Above random correlation
* Avoids overfit ultra-specific rules

---

## Key Steps

### Binary conversion

```python
binary_df = (binary_df > 0).astype(bool)
```

Apriori requires boolean data.

---

### Generate frequent itemsets

```python
frequent_itemsets = apriori(...)
```

---

### Generate rules

```python
association_rules(...)
```

---

### Filter rules

```python
confidence > 0.8
```

---

## Output

```text
clinical_logic_rules.csv
```

Contains:

| Column       | Meaning       |
| ------------ | ------------- |
| if_symptom   | antecedent    |
| then_suggest | consequent    |
| support      | frequency     |
| confidence   | rule strength |
| lift         | usefulness    |

---

## Visualization Section

There is partially implemented network visualization code.

Goal:

Generate symptom relationship graphs.

Uses:

```python
networkx
matplotlib
```

This can be expanded in future.

---

# 5.5 scoring.py

## Purpose

Generates patient severity/clinical scores.

Currently implemented:

```text
Knee Score
```

---

## Input

```text
flattened_patients.csv
```

or pruned dataset.

---

## Output

```text
scored_patients.csv
```

---

## Key Logic

### Functional Symptoms

Weighted more heavily:

```python
2 * knee_l_pain_walking
2 * knee_l_pain_stairclimbing
```

Reason:

Functional disability is clinically important.

---

### Structural Findings

Example:

```python
crepitus
```

Adds smaller weight.

---

### Duration Logic

Longer symptoms increase severity.

---

## Unique Filename Logic

Function:

```python
get_unique_filename()
```

Purpose:

Prevent overwriting previous outputs.

Creates:

```text
scored_patients.csv
scored_patients_1.csv
scored_patients_2.csv
```

---

## Extending Scores

Future interns can add:

* Lumbar score
* Cervical score
* Full body inflammatory index
* Ayurvedic severity scoring

Recommended approach:

```python
add_lumbar_score()
add_cervical_score()
```

---

# 5.6 Correlation.py

## Purpose

Analyzes relationships between symptoms using correlation matrices.

---

## What It Does

### Load dataset

```python
pd.read_csv('pruned_clinical_data.csv')
```

---

### Generate correlation matrix

```python
corr_matrix = df.corr()
```

---

### Filter strong correlations

Keeps only:

```text
0.5 to 0.99
```

---

### Visualize heatmap

Uses:

```python
seaborn
matplotlib
```

---

## Why This Matters

Useful for:

* identifying symptom clusters
* discovering hidden relationships
* feature engineering
* rule validation

---

# 5.7 run_pipeline.py

## Purpose

Master automation script.

Runs the entire pipeline sequentially.

---

## Important Function

### run_step(script_name)

Uses:

```python
subprocess.run()
```

Captures:

* stdout
* stderr
* failures

---

## Failure Handling

If any script fails:

```python
sys.exit(1)
```

Pipeline stops immediately.

---

# 6. Main Data Files

---

## patients.json

Raw API response.

Large nested JSON.

Source of truth.

---

## flattened_patients.csv

Structured patient dataset.

Used for:

* scoring
* ML
* pruning
* analytics

---

## pruned_clinical_data.csv

Cleaned high-signal dataset.

Used for:

* Apriori
* correlations
* ML experimentation

---

## scored_patients.csv

Final scored patients.

Contains:

* original features
* calculated severity scores

---

# 7. Important Commands

## Run Full Pipeline

```bash
python run_pipeline.py
```

---

## Run Individual Scripts

### Fetch Data

```bash
python NewdDataFetch.py
```

### Transform JSON

```bash
python Pravaayu_TransformerScript.py
```

### Prune Features

```bash
python Pruning.py
```

### Generate Rules

```bash
python Apriori.py
```

### Generate Scores

```bash
python scoring.py
```

### Correlation Analysis

```bash
python Correlation.py
```

---

# 8. Common Development Tasks

---

## Adding New Symptoms

Steps:

1. Update transformer extraction logic.
2. Add feature naming.
3. Run transformer.
4. Verify columns.
5. Rerun pruning.
6. Update scoring if needed.

---

## Adding New Disease Scores

Example:

```python
add_lumbar_score()
```

Then:

* define weighted features
* calculate severity
* save to dataframe

---

## Changing Apriori Thresholds

Modify:

```python
min_support
min_threshold
confidence filters
lift filters
```

Recommended:

Change gradually and validate clinically.

---

# 9. Debugging Guide

---

## Problem: Empty Apriori Results

Possible reasons:

* support too high
* pruning removed useful features
* dataset too small

Fix:

Lower:

```python
min_support
```

---

## Problem: Transformer Crashes

Usually caused by:

* API schema changes
* missing keys
* unexpected nulls

Fix:

Use:

```python
.get()
```

and safe casting.

---

## Problem: Correlation Heatmap Too Large

Too many features.

Fix:

* stronger pruning
* feature grouping
* visualize subsets

---

## Problem: Scoring Columns Missing

Likely causes:

* renamed columns
* pruned columns
* transformer mismatch

Always verify feature names.

---

# 11. Important Design Decisions

---

## Why CSV Instead of Database?

During experimentation:

* faster iteration
* easier debugging
* easier manual inspection

---

## Why Binary Features?

Most clinical symptoms are:

```text
present / absent
```

Binary encoding simplifies:

* Apriori
* correlation
* scoring

---

## Why Pruning Before Apriori?

Without pruning:

* combinatorial explosion
* noisy rules
* extremely slow execution

---

# 13. Important Notes

* DO NOT manually edit generated CSVs.
* Always regenerate outputs through scripts.
* Keep feature naming consistent.
* Validate every new rule clinically.
* Be careful when changing pruning thresholds.
* API structure may change over time.

---

# 14. Quick Start Checklist

## Setup

```bash
pip install -r requirements.txt
```

---

## Run Full Pipeline

```bash
python run_pipeline.py
```

---

## Main Outputs

```text
patients.json
flattened_patients.csv
pruned_clinical_data.csv
clinical_logic_rules.csv
scored_patients.csv
```

---

# 15. Final Advice for Whoever Continues This Project

The most critical part of the system is:

```text
Pravaayu_TransformerScript.py
```

If the transformer breaks:

everything downstream breaks.

Always:

1. Validate raw JSON.
2. Verify extracted columns.
3. Confirm feature consistency.
4. Test scoring after changes.
5. Regenerate datasets after modifications.

The project has strong potential for:

* AI-driven clinical analytics
* Ayurvedic decision support
* symptom intelligence systems
* severity prediction
* automated clinical recommendations




