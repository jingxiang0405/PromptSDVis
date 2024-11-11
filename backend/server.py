import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # Use 0-index GPU for server pipeline
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import io
from flask import Flask, render_template, request, make_response, jsonify
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
from config import *
from PIL import Image
import random
import matplotlib.pyplot as plt
import csv
from sentence_transformers import SentenceTransformer, util
from itertools import product
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
model, preprocess_img = clip.load("ViT-B/32", device=device, download_root='../.cache/clip')

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
    pos_tags = ['NOUN', "PROPN"]  # 指定要提取的詞性，ex 名詞和形容詞
    words = get_target_words(prompt, pos_tags) # words:['person', 'cute', 'hot', 'dog', 'beautiful', 'cat']
    print(f'words:{words}')
    # keywords 有5個
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
        futures = [executor.submit(sd_infer, combo, negative_prompt, guidance_scale, n_sd, random_seed_val) for combo in permutations]
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


# request POST to sd_server.py
def sd_infer(prompt: str, negative_prompt: str, guidance_scale: float, n_epo=int, n_seed=int):
    port = 5008
    url = f"{sd_ip}{port}/sd"
    data = {
        'epo': n_epo,
        'guidance_scale': guidance_scale,
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'random_seed':n_seed
    }
    # print(f"url: {url}")
    # print(f"data: {data}")
    response = requests.post(url, json=data)
    # print(f"response: {response.json()}")
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
	# embed_position = TSNE(n_components=2, init='random', perplexity=2, metric='cosine').fit_transform(encode_feature)
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
        top_keywords = sorted(keyword_similarity_pairs, reverse=True, key=lambda x: x[0])[:5]
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

    return combinations

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
def all_permutations(input_prompt: str, data: dict):
    # 使用正则表达式查找所有符合条件的部分
    items = re.split(r',\s*(?![^(]*\))', input_prompt)
    print(f'Items: {items}')

    # 准备替换映射
    replacement_map = {}
    for key, replacements in data.items():
        replacement_map[key] = [list(rep.keys())[0] for rep in replacements]
    print(f'replacement_map: {replacement_map}')

    # 存储替换后的组合
    replace_data = []
    for item in items:
        replace_item = [item]  # 初始化
        print(f'Original item: {replace_item}')

        # 替换外部内容
        for key in replacement_map:
            if key in item:
                replace_item.extend(replace_outside_brackets(item, key, replacement_map[key]))
                print(f'After replace_item: {replace_item}')

        # 生成排列组合
        replace_item_comb = []
        for sub_item in replace_item:
            replace_item_comb.extend(generate_combinations(sub_item))
        print(f'replace_item_comb: {replace_item_comb}')
        replace_data.append(replace_item_comb)

    # 生成所有排列组合
    product_result = list(itertools.product(*replace_data))
    print(f'product_result: {product_result}')

    # 转换为逗号分隔的字符串
    comma_separated_strings = [', '.join(combination) for combination in product_result]
    return comma_separated_strings

# 處理找出對應類別的data
def process_data(file_path_test, file_path_train):
    # 初始化 SentenceBERT 模型
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 读取txt文件并转化为字典
    def read_test_file(file_path):
        test_data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    data_dict = eval(line.strip())
                    test_data.append(data_dict)
        return test_data
    
    # 读取 JSON 文件并修正 prediction 字段
    def read_train_json(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
            train_data = json.loads(content)
            for item in train_data:
                if isinstance(item['prediction'], str):
                    item['prediction'] = ast.literal_eval(item['prediction'])  # 转换为字典
        return train_data
    
    # 根据类别分类 train 的 prediction
    def extract_category_sentences(train_data, category):
        category_sentences = []
        for item in train_data:
            for key, value in item['prediction'].items():
                if value == category:
                    category_sentences.append((key, item['sentence'], item['idx']))  # 存储该类别的句子和 idx
        return category_sentences
    
    # 计算相似度并聚合相同预测值，取出前10个相似的
    def get_top_n_similar(test_value, category_sentences, n=10):
        if len(category_sentences) == 0:
            return []
        
        # 生成测试值的句子嵌入
        test_embedding = model.encode(test_value, convert_to_tensor=True)
        
        # 存储相似度结果
        aggregated_results = defaultdict(list)
        
        # 计算每个训练句子的句子嵌入并与测试值计算相似度
        for train_value, train_sentence, train_idx in category_sentences:
            train_embedding = model.encode(train_value, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(test_embedding, train_embedding).item()  # 计算余弦相似度
            aggregated_results[train_value].append((train_idx, similarity))
        
        # 按相似度排序，每个聚合的相似度只取最高的一个
        sorted_aggregated_results = []
        for train_value, idx_similarity_pairs in aggregated_results.items():
            best_similarity = max(idx_similarity_pairs, key=lambda x: x[1])[1]  # 取最高相似度
            train_indices = [pair[0] for pair in idx_similarity_pairs]
            sorted_aggregated_results.append((train_value, train_indices, best_similarity))
        
        # 排序并取前 n 个
        sorted_aggregated_results = sorted(sorted_aggregated_results, key=lambda x: x[2], reverse=True)[:n]
        return sorted_aggregated_results
    
    # 生成回传数据
    def generate_results(test_data, train_data):
        results = {"test_results": []}
        
        for test_item in test_data:
            test_idx = test_item['idx']  # 提取 test 数据的 idx
            test_predictions = test_item['prediction']
            result_item = {"test_idx": test_idx, "predictions": []}
            
            for test_key, test_category in test_predictions.items():
                category_sentences = extract_category_sentences(train_data, test_category)
                top_similar = get_top_n_similar(test_key, category_sentences, n=10)
                
                top_similar_list = []
                for train_value, train_indices, sim_score in top_similar:
                    train_indices_str = train_indices  # 列出所有相关联的图片索引
                    top_similar_list.append({
                        "train_value": train_value,
                        "train_indices": train_indices_str,
                        "similarity": sim_score
                    })
                
                result_item["predictions"].append({
                    "test_prediction": test_key,
                    "category": test_category,
                    "top_similar": top_similar_list
                })
            
            results["test_results"].append(result_item)
        #print('results-1:' + results)
        return results
    
    # 读取文件
    train_data = read_train_json(file_path_train)
    test_data = read_test_file(file_path_test)
    
    # 生成并返回结果
    results = generate_results(test_data, train_data)
    return results

# 在匡選後 request 後分析Prompt與NER做sentencebert 
@app.route('/sentence_analyze',  methods=['POST'])
def process_sentence_analyze():
    try:
        print(f'sentence_analyze in')
        # 使用 request.get_json() 接收 JSON 陣列
        data = request.get_json()
        # 指定要操作的文件路徑
        print(f'Received JSON array: {data}')

        new_data = [
            {
                "idx": f"{item['id']}",  # 將 "id" 映射為 "idx"，轉換為字串
                "sentence": item["title"],  # 將 "title" 映射為 "sentence"
                "label": "{}"  # 新增 "label" 欄位，固定為 "{}"
            }
            for item in data
        ]
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
        file_trainprompt = "self_consistent_annotate/tb/train/demo_pool/train_demo_pool_std_c5_2348.json"
        file_trainemb = "self_consistent_annotate/tb/train/demo_pool/train_demo_pool_std_c5_2348_GPTEmb.npy"
        file_generatepromptjson = "./generate_ner/prompts/self_consistent_annotate/diffusiondb/self_supervision/train/fs_pool_std_c5_2348_GPTEmbDvrsKNN_100_Sc/st_std_c5_test_prompts__10.json"


        
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
            model="gpt-3.5-turbo",
            few_shot_setting="pool",
            demo_size=2348,
            demo_datamode="train",
            demo_select_method="std_c5",
            demo_retrieval_method="GPTEmbDvrsKNN",
            diverseKNN_number=100,
            diverseKNN_sampling="Sc",
            few_shot_number=10,
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
            model="gpt-3.5-turbo",
            few_shot_setting="pool",
            demo_size=2348,
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
        result_train_all_data_path = "./generate_ner/result/self_consistent_annotate/tb/diffusiondb/self_annotation/train/zs_consist_0.7_5_TSMV/TIME_STAMP_train_diffusiondb_0_response.json"

        result = process_data(response_path, result_train_all_data_path)
        print(f'results->{result}')
        
        return jsonify(result)
        
    except Exception as e:
        # 记录错误日志
        print(f"Error in sentence_analyze: {e}")
        # 返回错误响应
        return jsonify({"error": str(e)}), 500
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=5002)
