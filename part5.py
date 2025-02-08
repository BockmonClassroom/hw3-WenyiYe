import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load datasets
t1_path = "Data/t1_user_active_min.csv"  # Post-experiment active minutes
t2_path = "Data/t2_user_variant.csv"  # Treatment assignment
t3_path = "Data/t3_user_active_min_pre.csv"  # Pre-experiment active minutes

t1_data = pd.read_csv(t1_path)
t2_data = pd.read_csv(t2_path)
t3_data = pd.read_csv(t3_path)

# Merge post-experiment data with treatment assignment
merged_data = t1_data.merge(t2_data[['uid', 'variant_number']], on='uid', how='left')

# Define a reasonable threshold for maximum active minutes (e.g., 1440 minutes per day)
threshold = 1440  # Maximum possible minutes in a single day

# Filter out extreme outliers
cleaned_data = merged_data[merged_data["active_mins"] <= threshold]

# Merge pre-experiment data (t3) with cleaned post-experiment data
merged_with_t3 = cleaned_data.merge(
    t3_data.rename(columns={"active_mins": "active_mins_pre"}), 
    on=["uid", "dt"], how="left"
)

# Fill missing pre-experiment active minutes with 0 (assuming no prior activity)
merged_with_t3["active_mins_pre"].fillna(0, inplace=True)

# Compute the difference in active minutes before and after the experiment for each user
merged_with_t3["active_mins_change"] = merged_with_t3["active_mins"] - merged_with_t3["active_mins_pre"]

# Compute mean and median for the change in active minutes by group
group_stats_t3 = merged_with_t3.groupby("variant_number")["active_mins_change"].agg(["mean", "median"])

# Print cleaned statistics
print("Mean and Median Change in Active Minutes:")
print(group_stats_t3)

# Perform an independent t-test on change in active minutes
control_group_change = merged_with_t3[merged_with_t3["variant_number"] == 0]["active_mins_change"].dropna()
treatment_group_change = merged_with_t3[merged_with_t3["variant_number"] == 1]["active_mins_change"].dropna()

t_stat_t3, p_value_t3 = ttest_ind(control_group_change, treatment_group_change, equal_var=False)

# Print new t-test results
print("\nT-Test Results on Change in Active Minutes:")
print(f"T-Statistic: {t_stat_t3}")
print(f"P-Value: {p_value_t3}")

# Interpretation
alpha = 0.05
if p_value_t3 <= alpha:
    print("\nConclusion: There is a statistically significant difference in the change in active minutes between the two groups.")
else:
    print("\nConclusion: There is no statistically significant difference in the change in active minutes between the two groups.")

# Visualize the change in active minutes using a boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x=merged_with_t3["variant_number"], y=merged_with_t3["active_mins_change"])
plt.xticks(ticks=[0, 1], labels=["Control Group", "Treatment Group"])
plt.title("Box Plot of Change in Active Minutes for Control and Treatment Groups")
plt.xlabel("Group")
plt.ylabel("Change in Active Minutes")
plt.show()
