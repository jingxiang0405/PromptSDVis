import json
from sentence_transformers import SentenceTransformer, util

# 初始化 Sentence-BERT 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 讀取 test.txt 和 train.json 檔案
with open('/home/user/PromptSDVis/backend/generate_ner/result/self_consistent_annotate/tb/diffusiondb/self_supervision/train/fs_pool_std_c5_2348_GPTEmbDvrsKNN_100_Sc/Ask_Test_response_1.txt', 'r') as test_file:
    test_data = [eval(line.strip()) for line in test_file]

with open('/home/user/PromptSDVis/backend/generate_ner/result/self_consistent_annotate/tb/diffusiondb/self_annotation/train/zs_consist_0.7_5_TSMV/TIME_STAMP_train_diffusiondb_0_response.json', 'r') as train_file:
    train_data = json.load(train_file)

# 提取 test 和 train 的句子列表
test_sentences = [entry['sentence'] for entry in test_data]
train_sentences = [entry['sentence'] for entry in train_data]

# 使用 Sentence-BERT 模型進行編碼
test_embeddings = model.encode(test_sentences, convert_to_tensor=True)
train_embeddings = model.encode(train_sentences, convert_to_tensor=True)

# 計算相似度並找出前5相似的結果
results = []
for i, test_sentence in enumerate(test_data):
    # 計算與所有 train 句子的相似度
    cosine_scores = util.pytorch_cos_sim(test_embeddings[i], train_embeddings)[0]
    # 找出相似度最高的前5個句子索引
    top3_indices = cosine_scores.topk(3).indices.tolist()

    # 取得相應的 train 句子的 idx 和 prediction
    top5_matches = [
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
        'top5_similar_train': top5_matches
    })

# 打印或儲存結果
print(results)
