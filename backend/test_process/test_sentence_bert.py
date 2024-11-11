import json
import ast
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

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
    
    return results

# 假设文件路径
file_path_test = '/home/user/PromptSDVis/backend/generate_ner/result/self_consistent_annotate/tb/diffusiondb/self_supervision/train/fs_pool_std_c5_21_GPTEmbDvrsKNN_10_Sc/Ask_Test_response_1.txt'
file_path_train = '/home/user/PromptSDVis/backend/generate_ner/result/self_consistent_annotate/tb/diffusiondb/self_annotation/train/zs_consist_0.7_5_TSMV/back_0902_21_1513/TIME_STAMP_train_diffusiondb_0_response.json'

# 读取文件
train_data = read_train_json(file_path_train)
test_data = read_test_file(file_path_test)

# 生成并返回结果
results = generate_results(test_data, train_data)

# 输出为 json
output_json = json.dumps(results, indent=4)
print(output_json)
