"""
from flask import Flask, render_template
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # 生成一个matplotlib图表
    x = [1, 2, 3, 4, 5]
    y = [6, 7, 8, 9, 10]
    plt.plot(x, y)
    
    # 把图表保存到一个内存缓冲区
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # 将图表转换为base64编码的字符串
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # 渲染网页模板，并将图表数据传递给模板中的变量
    return render_template('index.html', plot_data=plot_data)

if __name__ == '__main__':
    app.run()

"""

"""
import itertools
# 定义初始数据和输入
input_prompt = "cat,dog hello"
data = {
    'cat': [{'dog': 60.81}, {'puppy': 56.23}, {'catgirls': 54.93}], 
    'dog': [{'puppy': 77.83}, {'terrier': 68.31}, {'cat': 60.81}]
}
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

print(comma_separated_strings)
"""

import re
import itertools

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


'''
input_prompt = "(cat, dog) in the park, good, nice, (aa, cc), bb"
(cat, dog) in the park -> 
cat in the park, dog in the park
good, dd
nice, vv
(aa, cc) -> 
aa, cc
bb, xx

good -> dd

ans: 
cat in the park, good, nice, aa, bb
cat in the park, good, nice, cc, bb
dog in the park, good, nice, aa, bb
dog in the park, good, nice, cc, bb
cat in the park, dd, nice, aa, bb
cat in the park, dd, nice, cc, bb
dog in the park, dd, nice, aa, bb
dog in the park, dd, nice, cc, bb
'''

input_prompt1 = "(terrier, playing) (laying, game)"
input_prompt2 = "cat (terrier, playing), (laying, game)"
input_prompt3 = "(terrier, playing) cat, (laying, game)"
input_prompt4 = "terrier, playing (laying, game)"

#input_prompt = "dd, dog in the park"
data = {'terrier': [{'dog': 60.81}, {'puppy': 56.23}, {'catgirls': 54.93}], 
        'cat': [{'puppy': 77.83}, {'terrier': 68.31}, {'cat': 60.81}],
        'game': [{'fine': 36.72}, {'friendly': 36.55}, {'meeting': 36.04}]
        }

'''
['cat in the park, cat',
'cat in the park, dog', 
'cat in the park, puppy', 
'cat in the park, catgirls',

'dog in the park, cat', 
'dog in the park, dog', 
'dog in the park, puppy', 
'dog in the park, catgirls', 

'cat in the tt, cat', 
'cat in the tt, dog', 
'cat in the tt, puppy', 
'cat in the tt, catgirls', 

'dog in the tt, cat', 
'dog in the tt, dog', 
'dog in the tt, puppy', 
'dog in the tt, catgirls', 

'cat in the ff, cat', 
'cat in the ff, dog', 
'cat in the ff, puppy', 
'cat in the ff, catgirls', 

'dog in the ff, cat', 
'dog in the ff, dog', 
'dog in the ff, puppy', 
'dog in the ff, catgirls', 

'cat in the gg, cat', 
'cat in the gg, dog', 
'cat in the gg, puppy', 
'cat in the gg, catgirls', 

'dog in the gg, cat', 
'dog in the gg, dog', 
'dog in the gg, puppy', 
'dog in the gg, catgirls']

'''
# 生成排列组合
permutations1 = all_permutations(input_prompt1, data)
print(f'permutations1->{permutations1}')
permutations2 = all_permutations(input_prompt2, data)
print(f'permutations2->{permutations2}')
permutations3 = all_permutations(input_prompt3, data)
print(f'permutations3->{permutations3}')
permutations4 = all_permutations(input_prompt4, data)
print(f'permutations4->{permutations4}')

'''
['Golden retriever playing in the park on a sunny day', 
'Golden terrier playing in the park on a sunny day', 
'Golden dog playing in the park on a sunny day', 
'Golden puppy playing in the park on a sunny day',
'Golden retriever playing in the stadium on a sunny day', 
'Golden retriever playing in the lake on a sunny day', 
'Golden retriever playing in the beach on a sunny day', 
'Golden retriever playing in the park on a sunny night', 
'Golden retriever playing in the park on a sunny sunset', 
'Golden retriever playing in the park on a sunny date']
'''

'''
['Golden retriever playing in the park on a sunny day', 
'Golden terrier playing in the park on a sunny day', 
'Golden dog playing in the park on a sunny day', 
'Golden puppy playing in the park on a sunny day', 
'Golden retriever playing in the stadium on a sunny day', 
'Golden retriever playing in the lake on a sunny day', 
'Golden retriever playing in the beach on a sunny day', 
'Golden retriever playing in the park on a sunny night', 
'Golden retriever playing in the park on a sunny sunset', 
'Golden retriever playing in the park on a sunny date']
'''

'''
['Golden retriever playing in the park on a sunny day', 
'Golden terrier playing in the park on a sunny day', 
'Golden retriever playing in the stadium on a sunny day', 
'Golden terrier playing in the stadium on a sunny day', 
'Golden retriever playing in the lake on a sunny day', 
'Golden terrier playing in the lake on a sunny day', 
'Golden retriever playing in the beach on a sunny day', 
'Golden terrier playing in the beach on a sunny day', 
'Golden retriever playing in the park on a sunny night', 
'Golden terrier playing in the park on a sunny night', 
'Golden retriever playing in the park on a sunny sunset', 
'Golden terrier playing in the park on a sunny sunset', 
'Golden retriever playing in the park on a sunny date', 
'Golden terrier playing in the park on a sunny date']
'''


