import random
import json

# Assume we have the JSON data saved in a file named 'data.json'
file_path = 'generate_ner/result/self_consistent_annotate/tb/diffusiondb/self_annotation/train/zs_consist_0.7_5_TSMV/TIME_STAMP_train_diffusiondb_0_response.json'

# Read the JSON data from the file
with open(file_path, 'r') as file:
    data = json.load(file)

# Randomly sample 100 items from the JSON data
sample_data = random.sample(data, 100)

# Save the sampled data to a new file for further use or inspection
sample_file_path = './result_100.json'
with open(sample_file_path, 'w') as sample_file:
    json.dump(sample_data, sample_file, indent=4)

sample_file_path