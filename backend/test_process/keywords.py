from keybert import KeyBERT

# 初始化 KeyBERT 模型
kw_model = KeyBERT()

# 從文本提取關鍵字
keywords = kw_model.extract_keywords(
    "stone library by Wally Wood, oil painting",
    keyphrase_ngram_range=(1, 1),  # 只提取單個單詞
    stop_words='english',  # 過濾常見的英文停用詞
    use_mmr=True,  # 使用最大邊際相關性提高關鍵字多樣性
    diversity=0.7  # 設置關鍵字多樣性的權重
)
