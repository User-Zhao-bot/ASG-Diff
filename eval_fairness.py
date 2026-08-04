
import pandas as pd
import numpy as np
import argparse
from scipy.stats import wasserstein_distance

def load_data(path, names=None):
    if str(path).endswith('.xls') or str(path).endswith('.xlsx'):
        df = pd.read_excel(path, sheet_name='Data', header=1)
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
    else:
        if names:
            df = pd.read_csv(path, header=None, names=names, skipinitialspace=True)
        else:
            df = pd.read_csv(path, skipinitialspace=True)
    
    # Strip whitespace from object columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return df

def calculate_metrics(df, sensitive_col, target_col, dataset_name="Data"):
    print(f"\n--- {dataset_name} Fairness Metrics ---")
    
    # Check if columns exist
    if sensitive_col not in df.columns:
        print(f"Error: Sensitive column '{sensitive_col}' not found.")
        return
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return

    # Basic stats
    print(f"Total samples: {len(df)}")
    
    # 1. DPD (Demographic Parity Difference)
    # P(Y=1 | S=Male) - P(Y=1 | S=Female)
    
    # Identify positive class for target
    # For Adult: '>50K' is positive. 
    # Check unique values
    unique_targets = df[target_col].unique()
    # Assume '>50K' or similar is positive. Or 1 if numeric.
    pos_label = None
    for val in unique_targets:
        if isinstance(val, str) and (">50K" in val):
            pos_label = val
            break
    
    if pos_label is None:
        if pd.api.types.is_numeric_dtype(df[target_col]):
             pos_label = sorted(df[target_col].dropna().unique())[-1]
        else:
             pos_label = [x for x in unique_targets if ">" in str(x)]
             if pos_label:
                 pos_label = pos_label[0]
             else:
                 pos_label = sorted(unique_targets)[-1] # Fallback
    
    print(f"Target Positive Label detected: '{pos_label}'")
    
    # Binarize Target
    y = (df[target_col] == pos_label).astype(int)
    
    # Identify groups
    # For Sex: 'Male', 'Female'
    groups = df[sensitive_col].unique()
    if len(groups) != 2:
        print(f"Warning: Sensitive attribute has {len(groups)} groups: {groups}. Calculating Pairwise DPD.")
    
    # We focus on Male vs Female for Adult
    group_stats = {}
    for g in groups:
        mask = (df[sensitive_col] == g)
        prob = y[mask].mean()
        group_stats[g] = prob
        print(f"P(Y=1 | {sensitive_col}={g}) = {prob:.4f}")
        
    # Calculate Max DPD
    probs = list(group_stats.values())
    dpd = max(probs) - min(probs)
    print(f"DPD (Max Diff): {dpd:.4f}")
    
    # 2. Wasserstein Distance
    # Distribution of Y for S=0 vs S=1
    # Since Y is binary, the distribution is just [1-p, p].
    # Wasserstein between two Bernoulli(p1) and Bernoulli(p2) is |p1 - p2|.
    # So for binary target, Wasserstein = DPD.
    # However, if user interprets "Income Distribution" as specific logits or something, we don't have that here (only CSV).
    # If we treat Y as 0/1 samples:
    
    # Let's try to find Male and Female specifically
    g1, g2 = groups[0], groups[1] # Take first two
    d1 = y[df[sensitive_col] == g1]
    d2 = y[df[sensitive_col] == g2]
    wd = wasserstein_distance(d1, d2)
    print(f"Wasserstein Distance ({g1} vs {g2}): {wd:.4f}")
    
    return dpd, wd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_data_path", type=str, default="data/adult/adult.data")
    parser.add_argument("--fake_data_path", type=str, required=True)
    parser.add_argument("--sensitive_col", type=str, default="sex")
    parser.add_argument("--target_col", type=str, default="income")
    args = parser.parse_args()
    
    col_names = None
    path_lower = args.real_data_path.lower()
    if "adult" in path_lower:
        col_names = [
            "age", "workclass", "fnlwgt", "education", "education.num", "marital.status", 
            "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", 
            "hours.per.week", "native.country", "income"
        ]
    elif "german" in path_lower:
        col_names = [
            "Status", "Duration", "Credit_history", "Purpose", "Credit_amount", 
            "Savings_account_bonds", "Present_employment", "Installment_rate", "Personal_status_sex",
            "Other_debtors_guarantors", "Present_residence", "Property", "Age", "Other_installment_plans",
            "Housing", "Number_of_existing_credits", "Job", "Number_of_people_liable", "Telephone", "Foreign_worker", "Target"
        ]
    
    print("Loading Real Data...")
    real_df = load_data(args.real_data_path, names=col_names)
    
    print("Loading Fake Data...")
    fake_df = load_data(args.fake_data_path) # Fake data usually has headers
    
    # Ensure columns have same names
    # If fake data uses dots instead of spaces?
    # samples.csv typically uses whatever standard training used.
    # Let's check overlap
    print(f"Real Cols: {real_df.columns[:5]}")
    print(f"Fake Cols: {fake_df.columns[:5]}")
    
    # Rename fake columns if needed (e.g. they might be 0, 1, 2...)
    # But ASGDiff samples.csv usually includes headers matching training.
    
    calculate_metrics(real_df, args.sensitive_col, args.target_col, "REAL DATA")
    calculate_metrics(fake_df, args.sensitive_col, args.target_col, "SYNTHETIC DATA")

if __name__ == "__main__":
    main()
