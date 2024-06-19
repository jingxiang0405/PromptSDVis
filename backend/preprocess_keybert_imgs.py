import csv
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# 读取数据
diffusiondb_data = []
with open('../data/diffusiondb_data/imgs.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    diffusiondb_data = [row for row in csv_reader]

# 提取关键字
def preprocess_diffusiondb(prompts):
    kw_model = KeyBERT()
    for idx, prompt in enumerate(prompts):
        keywords = kw_model.extract_keywords(prompt, keyphrase_ngram_range=(1, 1), stop_words='english', use_mmr=True, diversity=0.7)
        diffusiondb_data[idx]['keywords'] = ', '.join([kw[0] for kw in keywords])
    return diffusiondb_data

# 提取并更新数据
prompts = [datas['prompt'] for datas in diffusiondb_data]
preprocess_diffusiondb(prompts)

# 写回 CSV 文件
fieldnames = list(diffusiondb_data[0].keys())
with open('../data/diffusiondb_data/imgs_with_keywords.csv', 'w', newline='') as file:
    csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(diffusiondb_data)

print("Keywords added and file updated successfully.")
