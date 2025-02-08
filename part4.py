import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, ttest_ind

# Load datasets
t1_path = "Data/t1_user_active_min.csv"
t2_path = "Data/t2_user_variant.csv"

t1_data = pd.read_csv(t1_path)
t2_data = pd.read_csv(t2_path)

# Merge the datasets on 'uid' to include the experiment group
merged_data = t1_data.merge(t2_data[['uid', 'variant_number']], on='uid', how='left')

# Separate control and treatment groups before outlier removal
control_group = merged_data[merged_data["variant_number"] == 0]["active_mins"].dropna()
treatment_group = merged_data[merged_data["variant_number"] == 1]["active_mins"].dropna()

# Check normality using Shapiro-Wilk test
shapiro_control = shapiro(control_group.sample(min(5000, len(control_group)))) if len(control_group) > 3 else (None, None)
shapiro_treatment = shapiro(treatment_group.sample(min(5000, len(treatment_group)))) if len(treatment_group) > 3 else (None, None)

# Plot histograms to visualize the distribution of active minutes for both groups
plt.figure(figsize=(12, 5))
sns.histplot(control_group, bins=50, kde=True, label="Control Group", color="blue")
sns.histplot(treatment_group, bins=50, kde=True, label="Treatment Group", color="red")
plt.legend()
plt.title("Distribution of Active Minutes for Control and Treatment Groups")
plt.xlabel("Active Minutes")
plt.ylabel("Frequency")
plt.show()

# Box plot of active minutes
plt.figure(figsize=(8, 6))
sns.boxplot(x=merged_data["variant_number"], y=merged_data["active_mins"])
plt.xticks(ticks=[0, 1], labels=["Control Group", "Treatment Group"])
plt.title("Box Plot of Active Minutes for Control and Treatment Groups")
plt.xlabel("Group")
plt.ylabel("Active Minutes")
plt.show()

# Identify outliers
max_active_minutes = merged_data["active_mins"].max()
summary_stats = merged_data["active_mins"].describe()

# Define reasonable threshold
threshold = 1440  # Maximum possible minutes in a single day

# Filter out extreme outliers
cleaned_data = merged_data[merged_data["active_mins"] <= threshold]

# Redo statistical analysis with cleaned data
control_group_cleaned = cleaned_data[cleaned_data["variant_number"] == 0]["active_mins"].dropna()
treatment_group_cleaned = cleaned_data[cleaned_data["variant_number"] == 1]["active_mins"].dropna()

# Perform an independent t-test again with cleaned data
t_stat_cleaned, p_value_cleaned = ttest_ind(control_group_cleaned, treatment_group_cleaned, equal_var=False)

# Compute mean and median for both groups after cleaning
group_stats_cleaned = cleaned_data.groupby("variant_number")["active_mins"].agg(["mean", "median"])

# Save cleaned group statistics to a CSV file
group_stats_cleaned.to_csv("Data/cleaned_group_statistics.csv")

# Print test results
print("Shapiro-Wilk Test Results (Before Cleaning):")
print(f"Control Group: {shapiro_control}")
print(f"Treatment Group: {shapiro_treatment}\n")

print("Maximum Active Minutes in Dataset:", max_active_minutes)
print("\nSummary Statistics (Before Cleaning):")
print(summary_stats)

print("\nT-Test Results After Removing Outliers:")
print(f"T-Statistic: {t_stat_cleaned}")
print(f"P-Value: {p_value_cleaned}")

# Interpretation of the t-test result
alpha = 0.05  # Significance level
if p_value_cleaned <= alpha:
    print("\nConclusion: There is a statistically significant difference between the two groups.")
else:
    print("\nConclusion: There is no statistically significant difference between the two groups.")
