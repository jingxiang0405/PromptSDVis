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


#import pandas as pd

# 讀取 JSON 檔案
#file_path = '/home/user/PromptSDVis/backend/generate_ner/data/diffusiondb/train.json'
#df = pd.read_json(file_path)

# 找出非字符串的行
#non_string_rows = df[~df["sentence"].apply(lambda x: isinstance(x, str))]

# 列出非字符串值
#if not non_string_rows.empty:
#    print("非字符串的 'sentence' 值如下：")
#    print(non_string_rows[["idx", "sentence"]])
#else:
#    print("所有 'sentence' 值都是字符串。")

import re
def clean_and_parse(input_file, output_file):
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace problematic single quotes with double quotes
        content = content.replace("'", '"')
        
        # Fix trailing commas or other common issues
        content = re.sub(r',\s*}', '}', content)  # Remove trailing commas before closing braces
        content = re.sub(r',\s*]', ']', content)  # Remove trailing commas before closing brackets
        
        # Attempt to parse the cleaned content
        parsed_data = json.loads(content)
        
        # Save the cleaned JSON data to an output file
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(parsed_data, json_file, indent=4, ensure_ascii=False)
        
        print(f"Successfully cleaned and saved JSON to {output_file}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Example usage
input_txt_file = '/home/user/PromptSDVis/backend/generate_ner/result/self_consistent_annotate/gptmini/diffusiondb/self_annotation/train/zs_consist_0.7_5_TSMV/TIME_STAMP_train_diffusiondb_0_response.txt'  # Replace with the actual file path
output_json_file = '/home/user/PromptSDVis/backend/generate_ner/result/self_consistent_annotate/gptmini/diffusiondb/self_annotation/train/zs_consist_0.7_5_TSMV/TIME_STAMP_train_diffusiondb_0_response.json'  # Replace with the desired output path

# Convert the TXT file to JSON
clean_and_parse(input_txt_file, output_json_file)