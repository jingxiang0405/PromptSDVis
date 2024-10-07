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

def all_permutations(input_prompt: str):
    # 将输入字符串分割为多个部分
    items = re.split(r',\s*(?![^(]*\))', input_prompt)
    print(f'Items: {items}')

    replace_data = []
    for item in items:
        # 生成排列组合
        replace_item_comb = generate_combinations(item)
        print(f'replace_item_comb: {replace_item_comb}')
        replace_data.append(replace_item_comb)

    # 使用 itertools.product 生成所有排列组合
    product_result = list(itertools.product(*replace_data))
    comma_separated_strings = [', '.join(combination) for combination in product_result]
    return comma_separated_strings

# 测试示例
input_prompt1 = "(terrier, playing) (laying, game)"
result1 = all_permutations(input_prompt1)
# 输出结果
print(f'result1-> {result1}')
    
input_prompt2 = "cat (terrier, playing), (laying, game)"
result2 = all_permutations(input_prompt2)
# 输出结果
print(f'result2-> {result2}')
input_prompt3 = "(terrier, playing) cat, (laying, game)"
result3 = all_permutations(input_prompt3)
# 输出结果
print(f'result3-> {result3}')
input_prompt4 = "terrier, playing (laying, game)"
result4 = all_permutations(input_prompt4)
# 输出结果
print(f'result4-> {result4}')
