import re

def expand_brackets_fully(text):
    """
    對 text 中的所有括號進行「完整展開」並返回所有組合。
    無論最終產生多少筆，都不會被限制截斷。
    
    回傳: list of str
    """
    # 尋找第一組括號
    match = re.search(r'\(([^()]*)\)', text)
    if not match:
        # 沒有括號，直接返回 [text] 表示只有一種組合
        return [text]
    
    # 取得括號前後 + 括號內
    before = text[:match.start()].strip()
    inside = match.group(1).split(',')
    inside = [x.strip() for x in inside]
    after = text[match.end():].strip()
    
    # 先展開這個括號
    expanded_results = []
    for elem in inside:
        combined_text = (before + ' ' + elem).strip() if before else elem
        # 針對 after 再進一步遞歸
        after_expanded = expand_brackets_fully(after)
        for ae in after_expanded:
            new_str = (combined_text + ' ' + ae).strip()
            expanded_results.append(new_str)
    
    return expanded_results

def apply_similarity_keywords(text_list, similarity_dict, limit=200):
    """
    對已展開的 text_list，每一句再做「相似字替換 / 排列」。
    若發現結果數量超過 limit，則後續句子都不再做替換動作。
    
    假設 similarity_dict 的結構:
        {
          'person': [{'church':80}, {'porche':60}],
          'cat':    [{'lion':50}, {'tiger':40}],
          ...
        }
    此函式只是一個「示意用」的替換流程。
    """
    results = []
    total_count = 0

    for txt in text_list:
        # 如果已達到 / 超過限制，就直接把這個 txt 原封加入，不再做相似字替換
        if total_count >= limit:
            results.append(txt)
            continue
        
        # 逐個詞去比對是否在 similarity_dict 中
        # 這邊示例：直接把出現在 similarity_dict 的 key 字詞，替換成它所有相似詞的「排列組合」
        # 也可以依照分詞或更精細的需求來做
        replaced_variants = [txt]  # 先放一個原文做基底

        for key_word, sim_list in similarity_dict.items():
            if key_word in txt.split():
                # 例如 txt = "He is a person" ， key_word = "person"
                new_temp = []
                for rv in replaced_variants: 
                    # 把 rv 裡的 key_word 替換成 sim_list 裡的所有可能
                    for sim_obj in sim_list:
                        sim_word = list(sim_obj.keys())[0]  # 假設 {'church':80}
                        replaced_str = rv.replace(key_word, sim_word)
                        new_temp.append(replaced_str)
                replaced_variants = new_temp
        
        # 把這些變體加入 results
        for variant in replaced_variants:
            if total_count < limit:
                results.append(variant)
                total_count += 1
            else:
                # 一旦超過 limit，就停止添加，但注意不砍既有內容
                break
    
    return results

# -------------------------
# 主程式示例
if __name__ == "__main__":
    prompt = "cat"
    print(f"原始 prompt = {prompt}")
    
    # 第一階段：不管超過限制與否，「所有括號」一律完整展開
    bracket_expanded_list = expand_brackets_fully(prompt)
    print(f"\n[括號完全展開] 產生 {len(bracket_expanded_list)} 筆:")
    # 這裡列示前5筆
    for i, item in enumerate(bracket_expanded_list[:5], 1):
        print(f"{i:02d}. {item}")
    
    # 假設我們從 NLP 得到某些相似詞，用於非括號字詞替換
    # (此處僅範例)
    similarity_keywords={'cat': [{'kitty': 85.21}, {'feline': 82.1}, {'catdog': 80.52}]}

    limit = 200
    # 第二階段：對於 bracket_expanded_list 的每句，嘗試用 similarity_keywords 替換。
    # 若導致超過 limit，則之後就直接不再替換，保持原文。
    final_results = apply_similarity_keywords(bracket_expanded_list, similarity_keywords, limit=limit)

    print(f"\n[相似字替換後] 最終產生 {len(final_results)} 筆 (limit={limit}):")
    for i, res in enumerate(final_results[:400], 1):  # 只顯示前20筆
        print(f"{i:03d}. {res}")
