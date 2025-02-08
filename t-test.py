import pandas as pd
import scipy.stats as stats

t1_path = "Data/t1_user_active_min.csv"
t2_path = "Data/t2_user_variant.csv"

t1_data = pd.read_csv(t1_path)
t2_data = pd.read_csv(t2_path)

# Merge the datasets on 'uid' to include the experiment group
merged_data = t1_data.merge(t2_data[['uid', 'variant_number']], on='uid', how='left')

# Separate the two groups
control_group = merged_data[merged_data["variant_number"] == 0]["active_mins"].dropna()
treatment_group = merged_data[merged_data["variant_number"] == 1]["active_mins"].dropna()

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(control_group, treatment_group, equal_var=False)

# Compute mean and median for both groups
group_stats = merged_data.groupby("variant_number")["active_mins"].agg(["mean", "median"])

# Print results
print("Group Statistics (Mean and Median):")
print(group_stats)
print("\nIndependent t-Test Results:")
print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_value:.4f}")

# Interpretation
alpha = 0.05  # Significance level
if p_value <= alpha:
    print("\nResult: There is a statistically significant difference between the two groups.")
else:
    print("\nResult: There is no statistically significant difference between the two groups.")