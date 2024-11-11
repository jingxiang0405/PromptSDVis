import pandas as pd
import json

# Load the CSV file
#csv_path = '/home/user/PromptSDVis/data/diffusiondb_data/random_5k/diffusiondb_random_5k/imgs_rand5k.csv'
#df = pd.read_csv(csv_path)

# Transform the DataFrame to match the required JSON structure
#data = [{"idx": row['image_name'], "sentence": row['prompt'], "label":"{}"} for _, row in df.iterrows()]

# Save the transformed data to a new JSON file
#output_path = 'train.json'
#with open(output_path, 'w') as json_file:
#    json.dump(data, json_file, indent=2)

#print(f"Transformed data saved to {output_path}")



# 請將 'your_file.json' 換成你的 JSON 檔案路徑
#with open('/home/user/PromptSDVis/backend/generate_ner/prompts/self_consistent_annotate/diffusiondb/self_annotation/train/zs/train_prompts_diffusiondb_0.json', 'r', encoding='utf-8') as file:
#    load_data = json.load(file)

# 印出前10個內容
#for item in load_data[:1]:  # 假設 JSON 檔案內容是一個清單
#    print(item)


import pandas as pd

# 讀取 JSON 檔案
file_path = '/home/user/PromptSDVis/backend/generate_ner/data/diffusiondb/train.json'
df = pd.read_json(file_path)

# 找出非字符串的行
non_string_rows = df[~df["sentence"].apply(lambda x: isinstance(x, str))]

# 列出非字符串值
if not non_string_rows.empty:
    print("非字符串的 'sentence' 值如下：")
    print(non_string_rows[["idx", "sentence"]])
else:
    print("所有 'sentence' 值都是字符串。")
