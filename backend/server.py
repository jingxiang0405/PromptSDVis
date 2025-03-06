import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # Use 0-index GPU for server pipeline
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
from datetime import datetime
from flask import Flask, render_template, request, make_response, jsonify
from flask_cors import *
import numpy as np
import torch
import math
import json
import clip
import base64
import datasets
import uuid
from io import BytesIO
from util import *
from sklearn.manifold import TSNE
import umap
import scipy.cluster.hierarchy as sch
import spacy
import itertools
from config import *
from PIL import Image
import random
import matplotlib.pyplot as plt
import csv
from sentence_transformers import SentenceTransformer, util
from itertools import product
from itertools import cycle
import concurrent.futures
import ast
from collections import defaultdict

from sentence_transformers import SentenceTransformer, util
import torch
import heapq

from generate_ner.code.self_consistent_annotation.GeneratePrompts import generate_prompts_with_parameters
from generate_ner.code.standard.GenerateEmbsGPT import run_generate_embeddings
from generate_ner.code.self_consistent_annotation.AskGPT import ask_gpt_function


app = Flask(__name__)
CORS(app, supports_credentials=True)
print(f'torch.cuda.is_available():{torch.cuda.is_available()}')
device = "cuda" if torch.cuda.is_available() else "cpu"
print('[Server] device', device)
clip_model, preprocess_img = clip.load("ViT-B/32", device=device, download_root='../.cache/clip')

diffusiondb_data = []
with open('../data/diffusiondb_data/random_100k/diffusiondb_random_100k/imgs_rand100k_with_keywords.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    diffusiondb_data = [row for row in csv_reader]

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
@app.route('/image_overview')
def get_image_overview():
    ###################     Request Data   ######################
    request_data = requestParse(request) #print(request_data)
    prompt = request_data['prompt_val']
    negative_prompt = request_data['negative_prompt_val']
    guidance_scale = request_data['range_slider_val']
    generation_val = int(request_data['total_generation_val'])
    random_seed_val = int(request_data['random_seed_val'])
    print(f'guidance_scale->{guidance_scale}')
    n_sd = math.ceil(generation_val / n_images_per_prompt)

    ###################     spaCy   ######################
    
    print(f'prompt = {prompt}')
    #"PROPN"
    pos_tags = ['NOUN']  # 指定要提取的詞性，ex 名詞和形容詞
    words = get_target_words(prompt, pos_tags) # words:['person', 'cute', 'hot', 'dog', 'beautiful', 'cat']
    print(f'words:{words}')
    # keywords 有5個
    similarity_keywords = find_similarity_keywords(words, keywords) # {'person':[{'church':80}, {'porche':60}], 'person':[{'church':80}, {'porche':60}]}
    print(f'similarity_keywords:{similarity_keywords}')
    permutations = all_permutations1(prompt, similarity_keywords, 200)
    #permutations = all_permutations(prompt, similarity_keywords, 200) # [['hello, human cute, hot puppy, beautiful dog'], ['hello, human cute, hot puppy, beautiful puppy']] 
    # print(f'permutations:{permutations}')
    # print(f'permutations length:{len(permutations)}')
    
    ###################     Stable Diffison   ######################
    # 调整端口分配比重为 3:1
    ports = cycle([
        "http://140.119.164.166:9868/sd",  # 快速端口，重复 3 次
        "http://140.119.164.166:9868/sd",
        "http://140.119.164.166:9868/sd",
        "http://140.119.164.19:6887/sd"    # 慢速端口，重复 1 次
    ])

    results = {}
    
    # 使用ProcessPoolExecutor併行
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(sd_infer, combo, negative_prompt, guidance_scale, n_sd, random_seed_val, port)
            for combo, port in zip(permutations, ports)
        ]# 收集所有任务的结果
    results['result'] = [future.result()[0] for future in concurrent.futures.as_completed(futures)]
    # print(f'ProcessPoolExecutor results:{results}')
    # print(f'results:{results}')# {'result': [{'guidance_scale': '17.5', 'height': '256', 'negative_prompt': '', 'prompt': 'terrier', 'seed': '1478360753', 'width': '256'}]}
    # 存圖片 -> 將圖片存取來
    ###################     Save Images and Metadata   ######################
    
    # Create a directory to save images if it doesn't exist
    current_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    
    image_dir = '/media/user/新增磁碟區/sd-generate-images/' + current_time_str
    os.makedirs(image_dir, exist_ok=True)
    
    resized_image_dir = '/media/user/新增磁碟區/sd-generate-resizedimages/' + current_time_str
    os.makedirs(resized_image_dir, exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []

    # Process the result and save images + metadata
    for i, result in enumerate(results['result']):
        image_base64 = result['img']
        
        # Convert base64 to image
        img_data = base64.b64decode(image_base64)
        img = Image.open(BytesIO(img_data))
        
        # Save the image
        image_name = f"{i+1}.png"
        image_filename = f"{image_dir}/{image_name}"
        img.save(image_filename)

        # Save the resized image (1/4 of the original size)
        resized_img = img.resize((img.width // 4, img.height // 4), Image.ANTIALIAS)
        resized_image_name = f"{i+1}.png"
        resized_image_filename = f"{resized_image_dir}/{resized_image_name}"
        resized_img.save(resized_image_filename)

        # Convert resized image to Base64
        buffered = BytesIO()
        resized_img.save(buffered, format="PNG")
        resized_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Add the resized image Base64 to the result
        results['result'][i]['resized_img'] = resized_image_base64


        # Prepare metadata for CSV
        metadata = {
            'prompt': result['prompt'],
            'guidance_scale': result['guidance_scale'],
            'height': result['height'],
            'width': result['width'],
            'negative_prompt': result['negative_prompt'],
            'seed': result['seed'],
            'image_name': image_name
        }
        results['result'][i]['image_name'] = image_name
        csv_data.append(metadata)

    # Write CSV file
    csv_filename = f'{image_dir}/generated_images_metadata.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['prompt', 'guidance_scale', 'height', 'width', 'negative_prompt', 'seed', 'image_name'])
        writer.writeheader()
        writer.writerows(csv_data)
    ###################     TSNE   ######################
    
    sd_origin_image_list = [] 
    all_process_img_list = []
    
    for i in range(len(results['result'])):
        sd_origin_image_list.append(base64_to_image(results['result'][i]['img']))
        all_process_img_list = [preprocess_img(img) for img in sd_origin_image_list]
    
    image_features = encode_image(all_process_img_list)
    print(image_features.shape)
    # 定义 UMAP 参数范围

    embed_position = embed_feature(image_features.cpu())
    # print(f'embed_position:{embed_position}')
    results['embed_position'] = embed_position
    # print(f'results:{results}')
    
    return json.dumps(results)

def sd_infer(prompt: str, negative_prompt: str, guidance_scale: float, n_epo=int, n_seed=int, url=str):
    print(f"Send -> {url}")
    data = {
        'epo': n_epo,
        'guidance_scale': guidance_scale,
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'random_seed': n_seed
    }
    response = requests.post(url, json=data)
    return response.json()

# Encode the image 
def encode_image(image_list):
	image_tensor = torch.tensor(np.stack(image_list)).to(device)

	with torch.no_grad():
		image_features = clip_model.encode_image(image_tensor)
		
	return image_features

# Embed the feature into 2D space
def embed_feature(encode_feature):
	# Use TSNE for projection
	# embed_position = TSNE(n_components=2, init='random', perplexity=3, metric='cosine').fit_transform(encode_feature)
	# embed_position = TSNE(n_components=2, init='random', perplexity=30, metric='cosine').fit_transform(encode_feature)
    # Use UMAP for projection
	embed_position = umap.UMAP(n_neighbors=4, min_dist=0.01, metric='cosine').fit_transform(encode_feature)
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
    """
    如果 words 為空，就直接回傳空字典 (或其他你想要的結構)，
    避免在後面 torch.stack 等操作報錯。
    """
    if not words:
        # 直接回傳空字典，或依你的需求自訂回傳內容
        return {}

    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    keyword_embeddings = encode_keywords(keywords, embedding_model)
    word_embeddings = encode_keywords([words], embedding_model)  # Wrap words in a list to use the same function

    return calculate_similarity_matrix(word_embeddings, keyword_embeddings, words, keywords)


# 递归处理每个括号内和括号外的组合
def generate_combinations(text):

    def split_brackets(text):
        # 提取括号内和括号外的部分
        match = re.match(r'(.*?)\((.*?)\)(.*)', text)
        if match:
            # 提取括号前、括号内、括号后的部分
            before_brackets = match.group(1).strip()
            inside_brackets = match.group(2).split(', ')
            after_brackets = match.group(3).strip()
            return before_brackets, inside_brackets, after_brackets
        else:
            return text, None, None

    # 初始化组合结果列表
    combinations = []

    # 提取括号前、括号内和括号后的内容
    before, in_brackets, after = split_brackets(text)
    #print(f'split_brackets(text)-> {split_brackets(text)}')
    if in_brackets:
        # 如果存在括号内的内容，生成组合
        sub_combinations = []
        for element in in_brackets:
            # 递归处理括号后的部分
            if after:
                # 处理括号内的每个元素与之后的部分
                sub_comb = generate_combinations(f'{element} {after}'.strip())
                sub_combinations.extend(sub_comb)
            else:
                sub_combinations.append(element)

        # 将前后的部分组合起来
        if before:
            combinations.extend([f'{before} {comb}'.strip() for comb in sub_combinations])
        else:
            combinations.extend(sub_combinations)
    else:
        # 没有括号的情况，直接返回文本
        combinations.append(before.strip())
    print(f'combinations->{combinations}')
    return combinations





def all_permutations1(input_prompt: str, data: dict, max_combinations):
    """
    1. 先單純計算「所有括號展開」後的總組合數 (不含任何同義詞 / 相似詞替換)，
        如果已經超過 max_combinations，就直接保留這些超量組合，後續 (括號外) 替換不再執行。
    2. 如果括號展開後尚未超量，才進行替換外部內容 + 再做排列組合。
    """

    # Step 1: 根據逗號(但忽略括號中的逗號)來分割字串
    items = re.split(r',\s*(?![^(]*\))', input_prompt)

    # 如果 data 是空，則無替換規則
    if not data:
        print("No replacement data. Skipping external replacement step.")

    # Step 2: 準備「外部替換」的對照表 (同義詞或相似詞)
    replacement_map = {}
    if data:  # 如果 data 不為空，才建立 replacement_map
        for key, replacements in data.items():
            replacement_map[key] = [list(rep.keys())[0] for rep in replacements]

    print(f'replacement_map: {replacement_map}')

    # ─────────────────────────────────────────────────────────
    # (A) 計算「純括號」的所有展開
    # ─────────────────────────────────────────────────────────
    bracket_expanded_data = []
    total_bracket_combos = 1

    for item in items:
        # 只使用 generate_combinations，展開括號內的所有排列
        bracket_only = generate_combinations(item)
        bracket_expanded_data.append(bracket_only)
        total_bracket_combos *= len(bracket_only)

    print(f"Bracket-only combinations count = {total_bracket_combos}")

    # 如果「純括號展開」已經超過 max_combinations -> 直接保留
    if total_bracket_combos > max_combinations:
        print(f"Bracket expansions exceed max_combinations -> {total_bracket_combos} > {max_combinations}")
        bracket_product = list(itertools.product(*bracket_expanded_data))
        bracket_product = bracket_product[:max_combinations]
        comma_separated_strings = [', '.join(combo) for combo in bracket_product]
        print(f"Len datas (bracket only) = {len(comma_separated_strings)}")
        return comma_separated_strings

    # ─────────────────────────────────────────────────────────
    # (B) 若未超量，才做「外部替換 + 再展開括號」
    # ─────────────────────────────────────────────────────────
    replace_data = []
    for item in items:
        # 先把「原 item」放進去 (不替換)
        replace_item = [item]

        # 如果 replacement_map 不為空，才進行替換
        if replacement_map:
            for key in replacement_map:
                if key in item:
                    # 替換括號外內容
                    replace_item.extend(replace_outside_brackets(item, key, replacement_map[key]))

        # 再對「每個版本」做括號展開
        replace_item_comb = []
        for sub_item in replace_item:
            replace_item_comb.extend(generate_combinations(sub_item))

        replace_data.append(replace_item_comb)

    # 最後做笛卡兒積，並限制 max_combinations
    product_result = list(itertools.product(*replace_data))[:max_combinations]
    print(f'Len datas (with synonyms) = {len(product_result)}')

    comma_separated_strings = [', '.join(combination) for combination in product_result]
    return comma_separated_strings




# 替换括号外的内容，不改变括号内的内容
def replace_outside_brackets(text, key, replacements):
    parts = re.split(r'(\(.*?\))', text)  # 分离括号内外部分
    r_items = []  # 存放替换后的结果

    for rep in replacements:
        new_parts = []
        for part in parts:
            if part.startswith('(') and part.endswith(')'):
                # 不替换括号内的部分
                new_parts.append(part)
            else:
                # 替换括号外的部分
                new_parts.append(part.replace(key, rep))
        replaced_text = ''.join(new_parts)
        if replaced_text != text:
            r_items.append(replaced_text)
    return r_items

# input_prompt = "cat,dog hello"
# data = {
#    'cat': [{'dog': 60.81}, {'puppy': 56.23}, {'catgirls': 54.93}], 
#    'dog': [{'puppy': 77.83}, {'terrier': 68.31}, {'cat': 60.81}]
# }
# return [['cat', 'dog hello'], ['cat', 'puppy hello']]
    '''
    items
    ['(cat, dog) in the park', 'good (dd, yy)', 'nice', '(aa, cc)', 'bb']
    '''
    '''
    replacement_map
    {'cat': ['dog', 'puppy', 'catgirls'], 'dog': ['puppy', 'terrier', 'cat'], 'park': ['tt', 'ff', 'gg']}
    '''
    '''
    replace_item: ['(cat, dog) in the park']
    replace_item: ['good (dd, yy)']
    replace_item: ['nice']
    replace_item: ['(aa, cc)']
    replace_item: ['bb']
    '''
    '''
    replace_data = [
        ['dog', 'puppy'],
        ['cat in the park', 'kitten in the park', 'feline in the park']
    ]
    '''
    '''
    product_result
    [
        ('dog', 'cat in the park'),
        ('dog', 'kitten in the park'),
        ('dog', 'feline in the park'),
        ('puppy', 'cat in the park'),
        ('puppy', 'kitten in the park'),
        ('puppy', 'feline in the park')
    ]
    '''
    '''
    comma_separated_strings
    ['dd, dog in the park', 'dd, puppy in the park', 'dd, terrier in the park', 'dd, cat in the park', 'dd, dog in the tt', 'dd, dog in the ff', 'dd, dog in the gg']
    '''
# 主函数，处理输入提示并生成所有排列组合
def all_permutations(input_prompt: str, data: dict, max_combinations: int):
    import re
    print(f'input_prompt:{input_prompt}')
    # 使用正则表达式查找所有符合条件的部分
    # 這裡依舊示範拆分逗號，但你可以改為適合你需求的拆分方式
    items = re.split(r',\s*(?![^(]*\))', input_prompt)

    # 準備替換映射
    replacement_map = {}
    for key, replacements in data.items():
        # 假設 data = { "dog": [ {"pet": "xxx"}, {"animal": "yyy"} ], ... }
        # 這裡的 replacements 可能是 list，每個元素都是 dict
        # 我們只取 key(字串) 來做替換示意
        replacement_map[key] = [list(rep.keys())[0] for rep in replacements]

    # 用來儲存每個分段（item）對應的「所有替代結果列表」
    replace_data = []

    total_combinations = 1  # 初始化總組合數

    for i, item in enumerate(items):
        # 先把「原文」放進一個容器
        replace_item = [item]

        # 針對該 item，檢查有哪些 key 可以替換
        for key in replacement_map:
            if key in item:
                # 如果有匹配到 key，就將替換後的版本放到 replace_item
                replace_item.extend(replace_outside_brackets(item, key, replacement_map[key]))

        # 產生進一步的組合 (視你的邏輯而定)
        replace_item_comb = []
        for sub_item in replace_item:
            replace_item_comb.extend(generate_combinations(sub_item))
            print(f"replace_item_comb: {replace_item_comb}")
        # 嘗試更新「假設加入這些組合後」的總組合數
        projected = total_combinations * len(replace_item_comb)
        if projected > max_combinations:
            # 如果超出限制，只保留當前 item 的「原文」(不做額外替換)
            print(f"Combination limit reached: {max_combinations}")
            
            # 只留原文 (item 本身)
            replace_item_comb = [item]

            # 先把這個單一選擇 append 進 replace_data
            replace_data.append(replace_item_comb)
            total_combinations = total_combinations * 1  # 只乘 1

            # 後面的 items 通通只留原文
            for k in range(i + 1, len(items)):
                replace_data.append([items[k]])
            # 跳出整個迴圈，不再處理後續
            break
        else:
            # 如果還沒超標，正常加入
            replace_data.append(replace_item_comb)
            # 更新 total_combinations
            total_combinations = projected

    else:
        # 注意：for-else
        # 如果沒有透過 break 中斷，表示所有 items 都處理完，也可在這裡補齊邏輯
        pass

    # 如果前面的處理在中途 break，表示已把後續直接附加「原文」了
    # 如果沒 break，則表示 replace_data 可能剛好不超標

    # 產生排列組合
    # 這裡 product(...) 若 total_combinations 很大，可能很耗時
    product_result = list(itertools.product(*replace_data))

    # 截取前 max_combinations 筆
    product_result = product_result[:max_combinations]
    print(f'product_result->{product_result}')
    # 將 tuple 轉成逗號分隔字串
    comma_separated_strings = [', '.join(combination) for combination in product_result]
    print(f'comma_separated_strings->{comma_separated_strings}')
    return comma_separated_strings


# 初始化 Sentence-BERT 模型


# 處理找出對應類別的data
def process_data_1(file_path_test, file_path_train):
    smodel = SentenceTransformer('all-MiniLM-L6-v2')
    with open(file_path_test, 'r', encoding='utf-8') as f:
        test_data = [eval(line.strip()) for line in f]
    with open(file_path_train, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    # 提取 test 和 train 的句子列表
    test_sentences = [entry['sentence'] for entry in test_data]
    train_sentences = [entry['sentence'] for entry in train_data]

    # 使用 Sentence-BERT 模型進行編碼
    test_embeddings = smodel.encode(test_sentences, convert_to_tensor=True)
    train_embeddings = smodel.encode(train_sentences, convert_to_tensor=True)

    # 計算相似度並找出前3相似的結果
    results = []
    
    for i, test_sentence in enumerate(test_data):
        # 計算與所有 train 句子的相似度
        cosine_scores = util.pytorch_cos_sim(test_embeddings[i], train_embeddings)[0]
        # 找出相似度最高的前5個句子索引
        top3_indices = cosine_scores.topk(3).indices.tolist()

        # 取得相應的 train 句子的 idx 和 prediction
        top3_matches = [
            {'train_idx': train_data[idx]['idx'], 
            'prediction': train_data[idx].get('prediction', {}),
            'sentence': train_data[idx]['sentence'],
            'similarity_score': round(cosine_scores[idx].item(), 4)}
            for idx in top3_indices
        ]

        # 添加到結果中
        results.append({
            'test_idx': test_sentence['idx'],
            'test_sentence': test_sentence['sentence'],
            'test_prediction': test_sentence['prediction'],
            'top3_similar_train': top3_matches
        })

    results_ = {"test_results": results}
    print(results_)
    return jsonify(results_)


def l2_normalize(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def process_data_with_filtered_image_features(
    file_path_test, file_image_test, 
    file_path_train, file_image_train
):
    # 1. 加载 test 和 train 数据
    with open(file_path_test, 'r', encoding='utf-8') as f:
        test_data = [eval(line.strip()) for line in f]
    with open(file_path_train, 'r', encoding='utf-8') as f:
        train_data = json.load(f)


    # 预先编码训练数据的图像和文本特征
    train_image_features = []
    #train_text_features = []
    for entry in train_data:
        train_image_path = os.path.join(file_image_train, entry["idx"])
        train_image = preprocess_img(Image.open(train_image_path)).unsqueeze(0).to(device)
        #train_sentence = preprocess_text(entry['sentence'])
        #train_tokenized = clip.tokenize([train_sentence]).to(device)

        with torch.no_grad():
            image_feature = clip_model.encode_image(train_image).cpu()
            #text_feature = clip_model.encode_text(train_tokenized).cpu()

        train_image_features.append(image_feature)
        #train_text_features.append(text_feature)

    # 堆叠训练特征
    train_image_features_tensor = torch.vstack(train_image_features)  # 形状：(N, 512)
    #train_text_features_tensor = torch.vstack(train_text_features)  # 形状：(N, 512)

    # 对训练图像特征进行 L2 归一化
    train_image_features_tensor = l2_normalize(train_image_features_tensor, dim=1)
    #train_text_features_tensor = l2_normalize(train_text_features_tensor, dim=1)


    # 转置并转换为 float32
    train_image_features_t = train_image_features_tensor.T.float()    # 形状：(512, N)

    # 对测试数据进行处理
    results = []
    for test_entry in test_data:
        #test_image_path = os.path.join(file_image_test, test_entry["idx"])
        #test_image = preprocess_img(Image.open(test_image_path)).unsqueeze(0).to(device)
        test_sentence = preprocess_text(test_entry['sentence'])
        test_tokenized = clip.tokenize([test_sentence]).to(device)
        print(f'test_tokenized:{test_tokenized}')
        with torch.no_grad():
            #test_image_feature = clip_model.encode_image(test_image).cpu()
            test_text_feature = clip_model.encode_text(test_tokenized).cpu()

        # 对测试图像特征进行 L2 归一化
        #test_image_feature = l2_normalize(test_image_feature, dim=1)
        test_text_feature = l2_normalize(test_text_feature, dim=1)

        #test_image_feature_f = test_image_feature.float() 
        test_text_feature_f = test_text_feature.float() 
        # 计算相似度 (cosine similarity)
        similarity_scores = torch.matmul(test_text_feature_f, train_image_features_t)  # 形状：(1, N)

        # 获取相似度最高的 N 个索引
        topk_similarities, topk_indices = similarity_scores.topk(k=10, dim=1)
        topk_indices = topk_indices[0].tolist()
        topk_similarities = topk_similarities[0].tolist()

        # 收集结果
        top_matches = []
        for idx, sim_score in zip(topk_indices, topk_similarities):
            train_entry = train_data[idx]
            top_matches.append({
                "train_idx": train_entry["idx"],
                "prompt": train_entry["sentence"],
                "prediction": train_entry.get("prediction", {}),
                "similarity_score": sim_score
            })

        results.append({
            "test_idx": test_entry["idx"],
            "test_prompt": test_entry["sentence"],
            "test_prediction": test_entry.get("prediction", {}),
            "diffusiondb_sim_search_results": top_matches,
        })
    results_ = {"test_results": results}
    return results_


# 在匡選後 request 後分析Prompt與NER做sentencebert 
@app.route('/prompt_analyze',  methods=['POST'])
def process_sentence_analyze_1():

    print(f'prompt_analyze in')
    # 使用 request.get_json() 接收 JSON 陣列
    data = request.get_json()
    # 指定要操作的文件路徑
    #print(f'Received JSON array: {data}')
    # 建立資料夾
    save_dir = "../data/user-select-image"
    # 取得当前时间并格式化为 年月日_时分秒 的字符串，比如：20231225_153045
    current_time_str = datetime.now().strftime("%Y%m%d%H%M%S")

    # 使用UUID加上时间戳作为文件夹名称
    folder_name = f"{current_time_str}-{uuid.uuid4()}"
    folder_path = os.path.join(save_dir, folder_name)
    # 确保路径存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 创建路径

    # 保存每张图片到文件夹
    new_data = []
    for item in data:
        base64_data = item["image_src"].split(",")[-1]  # 提取 base64 数据部分
        file_name = f"{item['id']}"  # 文件命名
        file_path = os.path.join(folder_path, file_name)
        # 将 base64 解码并保存为图片文件
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(base64_data))
        # 构造 new_data 项
        new_data.append({
            "idx": f"{item['id']}",
            "sentence": item["title"],
            "label": "{}"
        })
    """"""
    file_path = "./generate_ner/data/diffusiondb/"
    file_testjson = "test.json"
    file_testemd = "test_GPTEmb.npy"
    # 以寫入模式 ("w") 打開文件，這會清空文件中的現有內容
    with open(file_path + file_testjson, "w", encoding="utf-8") as f:   
        # 將新的數據寫入文件
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    print(f"Data successfully written to {file_path}")
    # 調用生成嵌入的函數
    
    # Generate Test Prompt Enbeddings 
    """"""
    run_generate_embeddings(
        query_data_path=file_path + file_testjson,
        query_embs_path=file_path + file_testemd,
        dataname="diffusiondb",
        datamode="test",
        emb_model="text-embedding-ada-002",
        emb_encoding="cl100k_base",
        api_key=""
    )
    print("Generated embeddings success")

    file_abb2labelnamejson = "abb2labelname.json"
    file_trainprompt = "self_consistent_annotate/gptmini/train/demo_pool/train_demo_pool_std_c5_4996.json"
    file_trainemb = "self_consistent_annotate/gptmini/train/demo_pool/train_demo_pool_std_c5_4996_GPTEmb.npy"
    file_generatepromptjson = "./generate_ner/prompts/self_consistent_annotate/gptmini/self_supervision/train/fs_pool_std_c5_4996_GPTEmbDvrsKNN_100_Sc/prompts.json"


    
    # Generate Test Prompt 
    """"""
    generate_prompts_with_parameters(
        abb2labelname_path=file_path + file_abb2labelnamejson,
        query_data_path=file_path + file_testjson,
        query_embs_path=file_path + file_testemd,
        demo_data_path=file_path + file_trainprompt,
        demo_embs_path=file_path + file_trainemb,
        save_prompt_path=file_generatepromptjson,
        dataname="diffusiondb",
        datamode="test",
        model="gpt-4o-mini",
        few_shot_setting="pool",
        demo_size=4996,
        demo_datamode="train",
        demo_select_method="std_c5",
        demo_retrieval_method="GPTEmbDvrsKNN",
        diverseKNN_number=30,
        diverseKNN_sampling="Sc",
        few_shot_number=5,
        self_annotate_tag="std_c5"
    )
    
    # AskGPT
    """  
    prompt_path: prompts/self_consistent_annotate/diffusiondb/self_supervision/train/fs_pool_std_c5_21_GPTEmbDvrsKNN_10_Sc/st_std_c5_test_prompts__8.json
    response_path: ./generate_ner/result/self_consistent_annotate/tb/diffusiondb/self_supervision/train/fs_pool_std_c5_21_GPTEmbDvrsKNN_10_Sc/TIME_STAMP_st_std_c5_test__8_response.txt
    log_path: ./generate_ner/log/self_consistent_annotate/tb/diffusiondb/self_supervision/train/fs_pool_std_c5_21_GPTEmbDvrsKNN_10_Sc/TIME_STAMP_st_std_c5_test__8_AskGPT.log
    """
    
    
    response_path = "./generate_ner/result/self_consistent_annotate/tb/diffusiondb/self_supervision/train/fs_pool_std_c5_2348_GPTEmbDvrsKNN_100_Sc/Ask_Test_response_1.txt"
    
    log_path = "./generate_ner/log/self_consistent_annotate/tb/diffusiondb/self_supervision/train/fs_pool_std_c5_2348_GPTEmbDvrsKNN_100_Sc/Ask_Test.log"
    
    ask_gpt_function(
        abb2labelname_path=file_path + file_abb2labelnamejson,
        prompt_path= file_generatepromptjson,
        response_path=response_path,
        log_path=log_path,
        dataname="diffusiondb",
        datamode="test",
        model="gpt-4o-mini",
        few_shot_setting="pool",
        demo_size=4996,
        demo_datamode="train",
        demo_select_method="std_c5",
        demo_retrieval_method="GPTEmbDvrsKNN",
        diverseKNN_number=100,
        diverseKNN_sampling="Sc",
        few_shot_number=10,
        self_annotate_tag="std_c5"
    )
    """"""
    
    # 讀取
    result_train_all_data_path = "./generate_ner/result/self_consistent_annotate/gptmini/diffusiondb/self_annotation/train/zs_consist_0.7_5_TSMV/TIME_STAMP_train_diffusiondb_0_response.json"

    file_image_test = folder_path
    file_image_train = "/home/user/PromptSDVis/data/diffusiondb_data/random_5k/diffusiondb_random_5k/images"
    #result = process_data_1(response_path, result_train_all_data_path)
    result = process_data_with_filtered_image_features(response_path, file_image_test, result_train_all_data_path, file_image_train)
    #print(f'results->{result}')
    
    return result
        
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=5002)
