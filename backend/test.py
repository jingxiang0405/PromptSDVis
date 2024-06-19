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
from itertools import product

# Input data
input_prompt = "hello, person cute, hot dog, beautiful cat"
data = {
    'person': [{'human': 66.94}, {'people': 64.57}, {'guy': 58.13}, {'pedestrian': 55.19}, {'woman': 52.89}],
    'dog': [{'puppy': 77.83}, {'terrier': 68.31}, {'cat': 60.81}, {'bark': 60.27}, {'paw': 55.37}],
    'cat': [{'dog': 60.81}, {'puppy': 56.23}, {'catgirls': 54.93}, {'lion': 52.92}, {'tiger': 51.33}]
}



# Creating mapping from the data
mapping = {
    'person': [list(item.keys())[0] for item in data['person']],
    'dog': [list(item.keys())[0] for item in data['dog']],
    'cat': [list(item.keys())[0] for item in data['cat']]
}
# 生成所有关键词替换的组合
all_combinations = list(product(*mapping.values()))

# 生成新的input prompts
new_prompts = []
for combination in all_combinations:
    new_prompt = input_prompt
    for keyword, replacement in zip(mapping.keys(), combination):
        new_prompt = new_prompt.replace(keyword, replacement)
    new_prompts.append(new_prompt)
new_prompts.append(input_prompt)
print(type(new_prompts))
# 打印所有新生成的字符串
print(new_prompts)

