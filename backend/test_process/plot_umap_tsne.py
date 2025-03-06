import umap
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from io import BytesIO

def generate_umap_tsne_visualization_matrix(encode_feature, output_filename='results_matrix_labeled.png'):
    """
    生成带参数标注的 UMAP 和 t-SNE 可视化图片矩阵。

    参数:
        encode_feature (torch.Tensor): 高维特征张量，形状为 (样本数, 特征数)。
        output_filename (str): 保存图片矩阵的文件名。

    返回:
        None
    """
    # 初始化保存所有结果的容器
    umap_results_matrix = []
    tsne_results_matrix = []

    # 定义 UMAP 嵌入函数
    def embed_umap(encode_feature):
        embed_position = umap.UMAP(n_neighbors=4, min_dist=0.01, metric='cosine').fit_transform(encode_feature)
        return embed_position.tolist()

    # 定义 t-SNE 嵌入函数
    def embed_tsne(encode_feature):
        embed_position = TSNE(n_components=2, init='random', perplexity=5, metric='cosine').fit_transform(encode_feature)
        return embed_position.tolist()

    # 生成 UMAP 可视化
    embed_positions = embed_umap(encode_feature.cpu())

    # 可视化 UMAP 降维结果
    plt.figure(figsize=(6, 5))
    plt.scatter(
        [pos[0] for pos in embed_positions],
        [pos[1] for pos in embed_positions],
        s=10
    )
    plt.title(f"UMAP\nn_neighbors=4, min_dist=0.01")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).copy()
    umap_results_matrix.append(img)
    buf.close()
    plt.close()

    # 生成 t-SNE 可视化
    embed_positions = embed_tsne(encode_feature.cpu())

    # 可视化 t-SNE 降维结果
    plt.figure(figsize=(6, 5))
    plt.scatter(
        [pos[0] for pos in embed_positions],
        [pos[1] for pos in embed_positions],
        s=10
    )
    plt.title(f"t-SNE\nperplexity=3")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).copy()
    tsne_results_matrix.append(img)
    buf.close()
    plt.close()

    # 创建图片矩阵
    cell_width = 256
    cell_height = 256

    # 创建画布
    canvas_width = cell_width
    canvas_height = cell_height * 2  # 两部分结果叠加

    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # 填充 UMAP 结果
    resized_img = umap_results_matrix[0].resize((cell_width, cell_height), Image.ANTIALIAS)
    canvas.paste(resized_img, (0, 0))

    # 填充 t-SNE 结果
    resized_img = tsne_results_matrix[0].resize((cell_width, cell_height), Image.ANTIALIAS)
    canvas.paste(resized_img, (0, cell_height))

    # 保存画布
    canvas.save(output_filename)
    print(f"生成图片矩阵已保存为 {output_filename}")
