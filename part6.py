import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load datasets
t1_path = "Data/t1_user_active_min.csv" 
t2_path = "Data/t2_user_variant.csv"  
t3_path = "Data/t3_user_active_min_pre.csv"  
t4_path = "Data/t4_user_attributes.csv" 

t1_data = pd.read_csv(t1_path)
t2_data = pd.read_csv(t2_path)
t3_data = pd.read_csv(t3_path)
t4_data = pd.read_csv(t4_path)

# Merge post-experiment data with treatment assignment
merged_data = t1_data.merge(t2_data[['uid', 'variant_number']], on='uid', how='left')

# Define a reasonable threshold for maximum active minutes
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

# Merge user attributes (t4) with the dataset
merged_with_t4 = merged_with_t3.merge(t4_data, on="uid", how="left")

# Compute mean and median change in active minutes by user type and gender
group_stats_t4 = merged_with_t4.groupby(["user_type", "gender"])["active_mins_change"].agg(["mean", "median"])

# Print the computed statistics
print("\nMean and Median Change in Active Minutes by User Attributes:")
print(group_stats_t4)

# Reset index for plotting
group_stats_t4_reset = group_stats_t4.reset_index()

# 📊 Plot Mean Change in Active Minutes by User Type and Gender
plt.figure(figsize=(12, 6))
sns.barplot(data=group_stats_t4_reset, x="user_type", y="mean", hue="gender")
plt.title("Mean Change in Active Minutes by User Type and Gender")
plt.xlabel("User Type")
plt.ylabel("Mean Change in Active Minutes")
plt.legend(title="Gender")
plt.xticks(rotation=45)
plt.show()

# 📊 Plot Median Change in Active Minutes by User Type and Gender
plt.figure(figsize=(12, 6))
sns.barplot(data=group_stats_t4_reset, x="user_type", y="median", hue="gender")
plt.title("Median Change in Active Minutes by User Type and Gender")
plt.xlabel("User Type")
plt.ylabel("Median Change in Active Minutes")
plt.legend(title="Gender")
plt.xticks(rotation=45)
plt.show()

group_stats_t4_exp = merged_with_t4.groupby(["user_type", "variant_number"])["active_mins_change"].agg(["mean", "median"])

# Print computed statistics
print("\nMean and Median Change in Active Minutes by User Type and Experiment Group:")
print(group_stats_t4_exp)

# Reset index for plotting
group_stats_t4_exp_reset = group_stats_t4_exp.reset_index()

# 📊 Plot Mean Change in Active Minutes by User Type and Experiment Group
plt.figure(figsize=(12, 6))
sns.barplot(data=group_stats_t4_exp_reset, x="user_type", y="mean", hue="variant_number", palette=["blue", "red"])
plt.title("Mean Change in Active Minutes by User Type and Experiment Group")
plt.xlabel("User Type")
plt.ylabel("Mean Change in Active Minutes")
plt.legend(title="Experiment Group", labels=["Control", "Treatment"])
plt.xticks(rotation=45)
plt.show()

# 📊 Plot Median Change in Active Minutes by User Type and Experiment Group
plt.figure(figsize=(12, 6))
sns.barplot(data=group_stats_t4_exp_reset, x="user_type", y="median", hue="variant_number", palette=["blue", "red"])
plt.title("Median Change in Active Minutes by User Type and Experiment Group")
plt.xlabel("User Type")
plt.ylabel("Median Change in Active Minutes")
plt.legend(title="Experiment Group", labels=["Control", "Treatment"])
plt.xticks(rotation=45)
plt.show()