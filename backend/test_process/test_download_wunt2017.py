#import os
#import base64

#from PIL import Image
#import io

# 配置缓存路径
#os.environ["HF_HOME"] = "/media/user/新增磁碟區/wunt2017_cache"
#os.environ["HF_DATASETS_CACHE"] = "/media/user/新增磁碟區/wunt2017_cache"

# 打印缓存路径以确认
#print(f"HF_DATASETS_CACHE 路径: {os.environ['HF_DATASETS_CACHE']}")
#print(f"HF_HOME 路径: {os.environ['HF_HOME']}")

# 设置目标保存目录
#save_dir = "/media/user/新增磁碟區/wunt2017"
#os.makedirs(save_dir, exist_ok=True)

#from datasets import load_dataset
# 加載 WNUT2017 數據集
#dataset = load_dataset("tner/wnut2017")

#print(dataset)
#print(dataset['train'][0])
#print(dataset['train'].features)
#label_map = dataset['train'].features['tags'].feature
#print(label_map)  # 查看所有標籤名稱
# 如果需要保存为 CSV 格式
# for split in dataset.keys():
#     split_path = os.path.join(save_dir, f"{split}.csv")
#     dataset[split].to_csv(split_path)
#     print(f"{split} 数据保存到: {split_path}")


import pandas as pd

# 原始文件路徑
original_file = "/home/user/PromptSDVis/backend/generate_ner/data/wunt2017/wunt2017/train.csv"

# 新文件保存路徑
new_file = "/home/user/PromptSDVis/backend/generate_ner/data/wunt2017/wunt2017/train_sampled_900.csv"

# 讀取原始文件
df = pd.read_csv(original_file)

# 隨機選取 100 筆資料
sampled_df = df.sample(n=900, random_state=123)

# 保存選取的資料到新文件
sampled_df.to_csv(new_file, index=False, encoding='utf-8')

# 查看選取的資料
# print(sampled_df.head())  # 打印前 5 筆資料


import pandas as pd
import json
import re

# Load the CSV file
file_path = '/home/user/PromptSDVis/backend/generate_ner/data/wunt2017/wunt2017/train_sampled_900.csv'
data = pd.read_csv(file_path, header=0)

# Define the label mapping
label_mapping = {
    "B-corporation": 0, "B-creative-work": 1, "B-group": 2, "B-location": 3, "B-person": 4, "B-product": 5,
    "I-corporation": 6, "I-creative-work": 7, "I-group": 8, "I-location": 9, "I-person": 10, "I-product": 11,
    "O": 12
}

# Initialize a list for the JSON data
json_data = []

# Function to parse the 'tags' string
def parse_tags(tags_str):
    # Remove the brackets
    tags_str = tags_str.strip('[]')
    # Split by any non-digit characters
    tags_list = re.findall(r'\d+', tags_str)
    tags = [int(tag) for tag in tags_list]
    return tags

# Function to parse the 'tokens' string
def parse_tokens(tokens_str):
    # Remove the brackets
    tokens_str = tokens_str.strip('[]')
    # Remove extra quotes
    tokens_str = tokens_str.replace('""', '"').replace("''", "'")
    # Split tokens
    # Handle tokens enclosed in quotes or without quotes
    pattern = r"""
        (?<!\S)'([^']+)'(?!\S)   |  # Tokens enclosed in single quotes
        (?<!\S)"([^"]+)"(?!\S)   |  # Tokens enclosed in double quotes
        (?<!\S)(\S+)(?!\S)          # Tokens without quotes
    """
    matches = re.findall(pattern, tokens_str, re.VERBOSE)
    tokens = []
    for match in matches:
        # Each match is a tuple with one non-empty element
        token = next(filter(None, match))
        tokens.append(token)
    # If tokens list is empty, attempt to split the entire string
    if not tokens:
        # Split on word boundaries
        tokens = re.findall(r'\b\w+\b', tokens_str)
    return tokens

# Process the data
for idx, row in data.iterrows():
    tokens_str = row["tokens"]
    tags_str = row["tags"]

    tokens = parse_tokens(tokens_str)
    tags = parse_tags(tags_str)

    # Check if tokens and tags lengths match
    if len(tokens) != len(tags):
        print(f"Warning: Token and tag lengths do not match at index {idx}. Attempting to align them.")

        # Additional handling: Try to split concatenated tokens further
        concatenated_tokens = re.findall(r'[A-Za-z]+|\d+|[^\s\w]', tokens_str)
        if len(concatenated_tokens) == len(tags):
            tokens = concatenated_tokens
        else:
            print(f"Could not align tokens and tags at index {idx}. Skipping this entry.")
            continue

    # Initialize variables to keep track of entities
    entities = []
    current_entity = []
    current_tag = None

    # Iterate over tokens and tags
    for token, tag in zip(tokens, tags):
        # Get the tag label from the value
        tag_label = None
        for key, value in label_mapping.items():
            if tag == value:
                tag_label = key
                break

        if tag_label is None:
            continue  # Skip if tag not found

        if tag_label.startswith('B-'):
            if current_entity:
                # Save the previous entity
                entity_text = ' '.join(current_entity)
                entities.append((entity_text, current_tag))
                current_entity = []
            current_entity = [token]
            current_tag = tag_label[2:]
        elif tag_label.startswith('I-') and current_entity:
            current_entity.append(token)
        else:
            if current_entity:
                # Save the previous entity
                entity_text = ' '.join(current_entity)
                entities.append((entity_text, current_tag))
                current_entity = []
                current_tag = None

    # Capture any remaining entity
    if current_entity:
        entity_text = ' '.join(current_entity)
        entities.append((entity_text, current_tag))

    # Create a label dictionary
    label_dict = {entity: entity_type for entity, entity_type in entities}

    # Create a JSON entry
    json_entry = {
        "idx": str(idx),
        "sentence": " ".join(tokens),
        "label": str(label_dict)
    }
    json_data.append(json_entry)

# Save the JSON data to a file
output_path = '/home/user/PromptSDVis/backend/generate_ner/data/wunt2017/wunt2017/train_sampled_900.json'
with open(output_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=2)

print(f"JSON data has been saved to {output_path}")
