import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # Use 0-index GPU for server pipeline
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import io
from flask import Flask, render_template, request, make_response
from flask_cors import *
import numpy as np
import torch
import math
import json
import clip
import base64
import datasets
from io import BytesIO
from util import *
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch
import spacy
import itertools
import spacy
import itertools
from config import *
from PIL import Image
import random
import matplotlib.pyplot as plt
import csv
from sentence_transformers import SentenceTransformer, util
from itertools import product
import concurrent.futures

from sentence_transformers import SentenceTransformer, util
import torch
import heapq


app = Flask(__name__)
CORS(app, supports_credentials=True)
print(f'torch.cuda.is_available():{torch.cuda.is_available()}')
device = "cuda" if torch.cuda.is_available() else "cpu"
print('[Server] device', device)
model, preprocess_img = clip.load("ViT-B/32", device=device, download_root='../.cache/clip')

diffusiondb_data = []
with open('../data/diffusiondb_data/imgs_with_keywords.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    diffusiondb_data = [row for row in csv_reader]
diffusiondb_data = []
with open('../data/diffusiondb_data/imgs_with_keywords.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    diffusiondb_data = [row for row in csv_reader]

# Global Data
keywords = [datas['keywords'].split(', ') for datas in diffusiondb_data]
# Global Data
keywords = [datas['keywords'].split(', ') for datas in diffusiondb_data]
"""

ajax iput 後續將圖片 + 文字做 CLIP Encoder (文加圖 或圖片)

在用TSNE 顯示 然後再看該位置如何 放圖片上去

然後結合D3 js 去顯示圖片

mouseover 顯示圖片的詳細資料

"""
# Global Value

# 前端輸入Prompt 進入的第一個API
all_process_img_list = []
# 前端輸入Prompt 進入的第一個API
@app.route('/image_overview')
def get_image_overview():
    ###################     Request Data   ######################
    request_data = requestParse(request) #print(request_data)
    ###################     Request Data   ######################
    request_data = requestParse(request) #print(request_data)
    prompt = request_data['prompt_val']
    negative_prompt = request_data['negative_prompt_val']
    guidance_scale = request_data['range_slider_val'].split(',')
    generation_val = int(request_data['total_generation_val'])
    scale_left, scale_right = [float(i) for i in guidance_scale]
    n_sd = math.ceil(generation_val / n_images_per_prompt)

    ###################     spaCy   ######################
    
    print(f'prompt = {prompt}')
    pos_tags = ['NOUN', "PROPN"]  # 指定要提取的詞性，ex 名詞和形容詞
    words = get_target_words(prompt, pos_tags) # words:['person', 'cute', 'hot', 'dog', 'beautiful', 'cat']
    print(f'words:{words}')
    similarity_keywords = find_similarity_keywords(words, keywords) # {'person':[{'church':80}, {'porche':60}], 'person':[{'church':80}, {'porche':60}]}
    print(f'similarity_keywords:{similarity_keywords}')
    permutations = all_permutations(prompt, similarity_keywords) # [['hello, human cute, hot puppy, beautiful dog'], ['hello, human cute, hot puppy, beautiful puppy']] 
    print(f'permutations:{permutations}')
    # print(f'permutations length:{len(permutations)}')
    
    ###################     Stable Diffison   ######################
    
    results = {}
    
    # 使用ProcessPoolExecutor併行
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 使用列表推导和executor.submit提交所有任务
        futures = [executor.submit(sd_infer, combo, negative_prompt, scale_left, scale_right, n_sd) for combo in permutations]
        # 收集所有任务的结果
        results['result'] = [future.result()[0] for future in concurrent.futures.as_completed(futures)]
        # print(f'ProcessPoolExecutor results:{results}')
    # print(f'results:{results}')# {'result': [{'guidance_scale': '17.5', 'height': '256', 'negative_prompt': '', 'prompt': 'terrier', 'seed': '1478360753', 'width': '256'}]}
    
    ###################     TSNE   ######################
    
    sd_origin_image_list = [] 
    all_process_img_list = []
    
    for i in range(len(results['result'])):
        sd_origin_image_list.append(base64_to_image(results['result'][i]['img']))
        all_process_img_list = [preprocess_img(img) for img in sd_origin_image_list]
    
    image_features = encode_image(all_process_img_list, model)
    
    embed_position = embed_feature(image_features.cpu())
    # print(f'embed_position:{embed_position}')
    results['embed_position'] = embed_position
    # print(f'results:{results}')
    return json.dumps(results)

"""
    # 暫時先呈現結果
    plt.figure(figsize=(8, 8))
    print(embed_position[:, 0])
    plt.plot(embed_position[:, 0], embed_position[:, 1], 'o')  # 'o' 是一个圆形标记
    plt.title('2D projection with TSNE')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    result_json = {"images": sd_data, "plt_result": plt_data}

    plt.close()

    ###################     response data   ######################
    image_result_list = []
    for i in range(numberOfGeneration + n_search):
        data_item = {}
        data_item['id'] = str(i)
        data_item['x'] = str(embed_position[i][0])
        data_item['y'] = str(embed_position[i][1])

        compress_image = compress_PIL_image(all_origin_img_list[i])
        data_item['img'] = getImgStr(compress_image)
        img_type = 'generate' if i < numberOfGeneration else 'search'
        data_item['type'] = img_type

        image_result_list.append(data_item)
"""

# request POST to sd_server.py
def sd_infer(prompt: str, negative_prompt: str, scale_left: float, scale_right: float, n_epo=int):
# request POST to sd_server.py
def sd_infer(prompt: str, negative_prompt: str, scale_left: float, scale_right: float, n_epo=int):
    port = 5008
    url = f"{sd_ip}{port}/sd"
    data = {
    url = f"{sd_ip}{port}/sd"
    data = {
        'epo': n_epo,
        'scale_left': scale_left,
        'scale_right': scale_right,
        'prompt': prompt,
        'negative_prompt': negative_prompt,
    }
    # print(f"url: {url}")
    # print(f"data: {data}")
    response = requests.post(url, json=data)
    # print(f"response: {response}")
    return response.json()
    # print(f"url: {url}")
    # print(f"data: {data}")
    response = requests.post(url, json=data)
    # print(f"response: {response}")
    return response.json()

# Encode the image 
def encode_image(image_list, encode_model):
	image_tensor = torch.tensor(np.stack(image_list)).to(device)

	with torch.no_grad():
		image_features = encode_model.encode_image(image_tensor)
		
	return image_features


# Embed the feature into 2D space
def embed_feature(encode_feature):
	# Use TSNE for projection
	embed_position = TSNE(n_components=2, init='random', perplexity=5, metric='cosine').fit_transform(encode_feature)
	# Use UMAP for projection
	# embed_position = umap.UMAP(n_neighbors=5, min_dist=0.001, metric='cosine').fit_transform(encode_feature)
	return embed_position.tolist()

# Get POS Tag Words
def get_target_words(prompt: str, pos_tags: list):
    nlp = spacy.load('en_core_web_sm') # Load en_core_web_sm model
    process_prompt = nlp(prompt) # Process Prompt
    print(f'process_prompt:{process_prompt}')
    for token in process_prompt:
        print(f'token.text:{token.text}')
        print(f'token.pos_:{token.pos_}')
    words = [token.text for token in process_prompt if token.pos_ in pos_tags] # Extract words based on specified POS tags
    return words

def encode_keywords(keywords, model):
    # Flatten the list of keywords and ensure uniqueness
    unique_keywords = list(set([item for sublist in keywords for item in sublist]))
    # Encode and convert to tensor directly
    embeddings = model.encode(unique_keywords, convert_to_tensor=True)
    # Create a dictionary to easily map keywords to their embeddings
    return {keyword: embedding for keyword, embedding in zip(unique_keywords, embeddings)}

def calculate_similarity_matrix(word_embeddings, keyword_embeddings, words, keywords):
    # Flatten keyword list while maintaining uniqueness
    keyword_set = set()
    keyword_map = {}
    for sublist in keywords:
        for kw in sublist:
            if kw not in keyword_set:
                keyword_set.add(kw)
                keyword_map[kw] = sublist  # Map keyword to its sublist
    # Ensure embeddings are tensors
    word_tensor = torch.stack([word_embeddings[word] for word in words if word in word_embeddings])
    keyword_list = list(keyword_set)  # Convert set back to list for indexing
    keyword_tensor = torch.stack([keyword_embeddings[kw] for kw in keyword_list])
    # Calculate cosine similarities
    similarity_matrix = util.pytorch_cos_sim(word_tensor, keyword_tensor) * 100
    similarity_dict = {}
    for i, word in enumerate(words):
        word_similarities = similarity_matrix[i]
        # Filter out the input word and sort based on similarity score
        keyword_similarity_pairs = [(score.item(), keyword) for score, keyword in zip(word_similarities, keyword_list) if keyword != word]
        top_keywords = sorted(keyword_similarity_pairs, reverse=True, key=lambda x: x[0])[:3]
        similarity_dict[word] = [{kw: round(score, 2)} for score, kw in top_keywords]
    return similarity_dict

# Find Similarity By all-mpnet-base-v2
# words:['person', 'cute', 'hot', 'dog', 'beautiful', 'cat']
# keywords :[['omnious', 'church', 'creepy', 'abandoned', 'atmosphere']]
# return format
# {'person':[{'church':80}, {'porche':60}], 'person':[{'church':80}, {'porche':60}]}
def find_similarity_keywords(words, keywords):
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    keyword_embeddings = encode_keywords(keywords, embedding_model)
    word_embeddings = encode_keywords([words], embedding_model)  # Wrap words in a list to use the same function

    return calculate_similarity_matrix(word_embeddings, keyword_embeddings, words, keywords)

# input_prompt = "cat,dog hello"
# data = {
#    'cat': [{'dog': 60.81}, {'puppy': 56.23}, {'catgirls': 54.93}], 
#    'dog': [{'puppy': 77.83}, {'terrier': 68.31}, {'cat': 60.81}]
# }
# return [['cat', 'dog hello'], ['cat', 'puppy hello']]
def all_permutations(input_prompt:str, data:dict):
    items = input_prompt.split(",")# 解析输入提示，根据逗号分隔
    # ['cat'] ['dog hello'] ['hw dog'] 
    print(f'items: {items}')

    # Prepare the data for replacements by flattening the dictionary to direct mappings
    replacement_map = {}
    for key, replacements in data.items():
        # Flatten to a single replacement list
        replacement_map[key] = [list(rep.keys())[0] for rep in replacements]

    replace_data = []
    for item in items:
        replace_item = [item]  # Initialize with the original item
        for key in replacement_map:
            if key in item:
                # Directly replace the key with each of its replacements
                replace_item.extend(item.replace(key, rep) for rep in replacement_map[key])
        replace_data.append(replace_item)

    # Generate all combinations using itertools.product
    product_result = list(itertools.product(*replace_data))

    # Convert tuples to comma-separated strings
    comma_separated_strings = [', '.join(combination) for combination in product_result]

    return comma_separated_strings

# Get POS Tag Words
def get_target_words(prompt: str, pos_tags: list):
    nlp = spacy.load('en_core_web_sm') # Load en_core_web_sm model
    process_prompt = nlp(prompt) # Process Prompt
    words = [token.text for token in process_prompt if token.pos_ in pos_tags] # Extract words based on specified POS tags
    return words

# Find Similarity By all-mpnet-base-v2
# words:['person', 'cute', 'hot', 'dog', 'beautiful', 'cat']
# keywords :[['omnious', 'church', 'creepy', 'abandoned', 'atmosphere']]
# return format
# {'person':[{'church':80}, {'porche':60}], 'person':[{'church':80}, {'porche':60}]}
def find_similarity_keywords(words, keywords):
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    # 向量化
    words_embedding = embedding_model.encode(words, convert_to_tensor=True)
    keywords_embedding = [[embedding_model.encode(subword, convert_to_tensor=True) for subword in keyword_list] for keyword_list in keywords]
    similarity_dict = {}
    # 遍歷每個單詞
    for i, word in enumerate(words):
        word_embedding = words_embedding[i].unsqueeze(0)  # 單個單詞的向量
        similarity_list = [] # 存儲該單詞的相似度結果
        # 遍歷每個關鍵詞列表
        for keyword_list, keyword_embeddings in zip(keywords, keywords_embedding):
            keyword_similarity = {}
            # 計算單詞向量與每個子詞向量之間的餘弦相似度
            for j, subword_embedding in enumerate(keyword_embeddings):
                cosine_similarities = util.pytorch_cos_sim(word_embedding, subword_embedding)
                similarity = cosine_similarities.item() * 100  # 轉換為百分比
                keyword_similarity[keyword_list[j]] = round(similarity, 2)
            similarity_list.append(keyword_similarity)
        #print(f'similarity_list:{similarity_list}')
        # 排序並選擇相似度最高的五個關鍵詞
        flat_similarity_list = [item for sublist in similarity_list for item in sublist.items()]
        flat_similarity_list.sort(key=lambda x: x[1], reverse=True)
        top_k = 5
        unique_similarity_list = []
        seen_keywords = set()
        for keyword, similarity in flat_similarity_list:
            if keyword != word and keyword not in seen_keywords:
                unique_similarity_list.append({keyword: similarity})
                seen_keywords.add(keyword)
                if len(unique_similarity_list) == top_k:
                    break
        similarity_dict[word] = unique_similarity_list
    return similarity_dict



def all_combinations(prompt:str, similarity_keywords:dict):
    # Creating mapping from the data
    mapping = {
        'person': [list(item.keys())[0] for item in similarity_keywords['person']],
        'dog': [list(item.keys())[0] for item in similarity_keywords['dog']],
        'cat': [list(item.keys())[0] for item in similarity_keywords['cat']]
    }
    # 生成所有关键词替换的组合
    all_combinations = list(product(*mapping.values()))

    # 生成新的input prompts
    new_prompts = []
    for combination in all_combinations:
        new_prompt = prompt
        for keyword, replacement in zip(mapping.keys(), combination):
            new_prompt = new_prompt.replace(keyword, replacement)
        new_prompts.append(new_prompt)
    new_prompts.append(prompt)
    return new_prompts

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=5002)
