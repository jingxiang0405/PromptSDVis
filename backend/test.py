
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

"""