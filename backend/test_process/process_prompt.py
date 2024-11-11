import spacy
import itertools
import csv
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

diffusiondb_data = []
with open('../data/diffusiondb_data/imgs_with_keywords.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    diffusiondb_data = [row for row in csv_reader]

# Global Data
keywords = [datas['keywords'].split(', ') for datas in diffusiondb_data]
prompt = 'hello, person cute, hot dog, beautiful cat'
pos_tags = ['NOUN']  # 指定要提取的詞性，ex 名詞和形容詞
# Get POS Tag Words
def get_target_words(prompt: str, pos_tags: list):
    # Load en_core_web_sm model
    nlp = spacy.load('en_core_web_sm')
    # Process Prompt
    process_prompt = nlp(prompt)
    # Extract words based on specified POS tags
    words = [token.text for token in process_prompt if token.pos_ in pos_tags]
    return words
# Find Similarity By all-mpnet-base-v2
# words:['person', 'cute', 'hot', 'dog', 'beautiful', 'cat']
# keywords :[['omnious, church, creepy, abandoned, atmosphere'], ['render, porche, outline, black, vector'], ['paul, grave, rain, stands, anamorphic'], ['lemon, mugshot'], ['render, porche, outline, black, vector'], ['painting, winnie, eyes, creepily, atmosphere']]
# return format
# {'person':[{'church':80}, {'porche':60}]}
def find_similarity_keyword(words, keywords):
    # all-mpnet-base-v2 模型
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    # 向量化
    words_embedding = embedding_model.encode(words, convert_to_tensor=True)
    keywords_embedding = [[embedding_model.encode(subword, convert_to_tensor=True) for subword in keyword_list] for keyword_list in keywords]
    # 存儲結果的字典
    similarity_dict = {}
    # 遍歷每個單詞
    for i, word in enumerate(words):
        word_embedding = words_embedding[i].unsqueeze(0)  # 單個單詞的向量
        # 存儲該單詞的相似度結果
        similarity_list = []
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

words = get_target_words(prompt, pos_tags)
print(f'words:{words}')
result = find_similarity_keyword(words, keywords)
print(result)
