import re

def compare_sentences(sentences):
    # 将每个句子按逗号分隔为多个部分，再分别处理每个部分的差异
    split_sentences = [re.split(r'\s*,\s*', sentence) for sentence in sentences]

    # 找到最长句子的长度
    max_len = max(len(parts) for parts in split_sentences)

    # 补齐较短的句子，避免 zip 时忽略较短句子的尾部
    for parts in split_sentences:
        while len(parts) < max_len:
            parts.append('')  # 用空字符串补齐较短的句子

    # 初始化结果列表
    result_parts = []

    # 对每个逗号分隔的部分逐一比较
    for part_tuple in zip(*split_sentences):
        # 对逗号分隔的每个部分，按空格再拆分单词进行比较
        word_lists = [part.split() for part in part_tuple]
        max_words_len = max(len(words) for words in word_lists)
        
        # 补齐较短的部分
        for words in word_lists:
            while len(words) < max_words_len:
                words.append('')

        # 比较每个单词位置的差异
        result = []
        for word_tuple in zip(*word_lists):
            unique_words = set(word for word in word_tuple if word)  # 去除空字符串
            if len(unique_words) > 1:
                result.append(f"({', '.join(unique_words)})")  # 不同的单词用括号和逗号隔开
            else:
                result.append(unique_words.pop())  # 如果单词相同，直接使用该单词

        # 将每个部分处理的结果合并为字符串
        result_parts.append(' '.join(result))

    # 最终结果为逗号分隔的各部分组合
    return ', '.join(result_parts)

# 示例句子 1
sentences1 = [
    'dog, in the park, catgirls',
    'cat, in the tt, puppy',
    'dog, in the tt, cat'
]

# 示例句子 2
sentences2 = [
    'Golden terrier playing in the stadium on a sunny day',
    'Golden retriever playing in the lake on a sunny day',
    'Golden terrier playing in the lake on a sunny day',
]

sentences3 = [
    'Golden, terrier playing in the, stadium on a sunny day',
    'Golden, retriever playing in the, lake on a sunny day',
    'Golden, terrier playing in the, lake on a sunny day',
]

sentences4 = [
    '(Golden), terrier playing in the, stadium on a sunny day',
    '(Golden), retriever gaming in the, lake on a sunny day',
    '(Golden), terrier good in the, lake on a sunny day',
]
# 调用函数并输出结果
output1 = compare_sentences(sentences1)
output2 = compare_sentences(sentences2)

print("Result 1:", output1)
print("Result 2:", output2)

output3 = compare_sentences(sentences3)
print("Result 3:", output3)

output4 = compare_sentences(sentences4)
print("Result 4:", output4)