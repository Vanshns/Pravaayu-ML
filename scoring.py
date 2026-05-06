import pandas as pd
import os

def get_unique_filename(base_path):
    """
    If file exists, create a new version:
    scored_patients.csv → scored_patients_1.csv → scored_patients_2.csv
    """
    if not os.path.exists(base_path):
        return base_path

    name, ext = os.path.splitext(base_path)
    i = 1

    while True:
        new_path = f"{name}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1


def add_knee_score(df, save_path="scored_patients.csv", only_knee=True):
    print(">>> add_knee_score FUNCTION CALLED")

    def calculate(row):
        if row.get("knee_present", 0) == 0:
            return None

        score = 0

        # Functional
        score += 2 * row.get("knee_l_pain_walking", 0)
        score += 2 * row.get("knee_l_pain_stairclimbing", 0)
        score += 2 * row.get("knee_l_pain_lowsitting", 0)

        score += 2 * row.get("knee_r_pain_walking", 0)
        score += 2 * row.get("knee_r_pain_stairclimbing", 0)
        score += 2 * row.get("knee_r_pain_lowsitting", 0)

        # Structural
        score += row.get("knee_l_exam_creps_left", 0)
        score += row.get("knee_r_exam_creps_right", 0)

        # Severity
        score += 2 * row.get("knee_l_exam_tenderness_knee", 0)
        score += 2 * row.get("knee_r_exam_tenderness_knee", 0)

        # Duration
        score += 2 * (1 if row.get("knee_l_duration_val", 0) > 0 else 0)
        score += 2 * (1 if row.get("knee_r_duration_val", 0) > 0 else 0)

        return score

    # =========================
    # 1. CALCULATE SCORE
    # =========================
    df["knee_score"] = df.apply(calculate, axis=1)

    # =========================
    # 2. CLASSIFY
    # =========================
    def classify(score):
        if pd.isna(score):
            return None
        if score <= 2:
            return "Low"
        elif score <= 5:
            return "Moderate"
        elif score <= 8:
            return "High"
        else:
            return "Severe"

    df["knee_risk"] = df["knee_score"].apply(classify)

    # =========================
    # 3. OPTIONAL FILTER
    # =========================
    if only_knee:
        df = df[df["knee_present"] == 1]

    # =========================
    # 4. CLEAN OUTPUT
    # =========================
    output_cols = [
        col for col in ["patient_name", "patient_index", "knee_score", "knee_risk"]
        if col in df.columns
    ]

    output_df = df[output_cols]

    # =========================
    # 5. SAFE SAVE
    # =========================
    final_path = get_unique_filename(save_path)
    output_df.to_csv(final_path, index=False)

    print(f"Saved: {final_path}")

    return output_df

if __name__ == "__main__":
    print(">>> Running scoring pipeline")

    import pandas as pd

    # Load your flattened data
    df = pd.read_csv("flattened_patients.csv")

    # Run scoring
    result = add_knee_score(df)

    print(">>> Scoring completed")
    print(result.head())