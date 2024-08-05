import pandas as pd
import json

# Load the CSV file
csv_path = './data/diffusiondb/imgs.csv'
df = pd.read_csv(csv_path)

# Transform the DataFrame to match the required JSON structure
data = [{"idx": row['image_filename'], "sentence": row['prompt'], "label":"{}"} for _, row in df.iterrows()]

# Save the transformed data to a new JSON file
output_path = 'train.json'
with open(output_path, 'w') as json_file:
    json.dump(data, json_file, indent=2)

print(f"Transformed data saved to {output_path}")
