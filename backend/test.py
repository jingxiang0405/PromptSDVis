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

