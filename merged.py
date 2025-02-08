import pandas as pd

# Load the datasets
t1_path = "Data/t1_user_active_min.csv"
t2_path = "Data/t2_user_variant.csv"

t1_data = pd.read_csv(t1_path)
t2_data = pd.read_csv(t2_path)

# Merge the datasets on 'uid' to include the experiment group
merged_data = t1_data.merge(t2_data[['uid', 'variant_number']], on='uid', how='left')

# Save the merged data to a new CSV file
merged_data.to_csv("Data/merged_user_activity.csv", index=False)

print(merged_data.head())