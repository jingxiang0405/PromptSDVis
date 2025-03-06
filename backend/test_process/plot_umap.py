from flask import Flask, jsonify
from flask_cors import CORS  # 引入 Flask-CORS
import os
from PIL import Image
import numpy as np
import umap
import torch
import clip

# 初始化 Flask 应用
app = Flask(__name__)

# 啟用 CORS
CORS(app)

# 初始化 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess_img = clip.load("ViT-B/32", device=device, download_root='../.cache/clip')


def process_images_with_umap(image_folder, clip_model, preprocess_img):
    """
    提取圖片特徵向量
    """
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, f)) and f.lower().endswith(('png', 'jpg', 'jpeg'))
    ]

    if not image_files:
        raise ValueError(f"No valid image files found in folder: {image_folder}")

    features = []
    processed_files = []

    for image_file in image_files:
        try:
            img = Image.open(image_file).convert('RGB')
            img_tensor = preprocess_img(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = clip_model.encode_image(img_tensor).cpu().numpy()

            features.append(feature.squeeze())
            processed_files.append(image_file)
        except Exception as e:
            print(f"Error processing file {image_file}: {e}")

    features = np.array(features)
    return features, processed_files


@app.route('/van_gogh', methods=['GET'])
def get_combined_umap_results():
    """
    使用 UMAP 將特徵降維，並返回 JSON 格式的結果
    """
    print("In!")
    # 定义图片文件夹路径
    image_folders = [
        "/home/user/PromptSDVis/data/sd-generate-images/sd-generate-images/20250204212414",
        "/home/user/PromptSDVis/data/sd-generate-images/sd-generate-images/20250205070559",
        "/home/user/PromptSDVis/data/sd-generate-images/sd-generate-images/20250205070840",
    ]

    labels = ["iter1", "iter2", "iter3"]
    colors = ["red", "blue", "green"]

    all_features = []
    all_labels = []
    all_colors = []
    all_files = []

    # 提取特徵並收集圖片資訊
    for folder, label, color in zip(image_folders, labels, colors):
        features, file_names = process_images_with_umap(folder, clip_model, preprocess_img)
        all_features.append(features)
        all_labels.extend([label] * len(features))
        all_colors.extend([color] * len(features))
        all_files.extend(file_names)

    # 合并所有特徵
    all_features = np.vstack(all_features)

    # 使用 UMAP 进行降维
    umap_model = umap.UMAP(n_neighbors=4, min_dist=0.01, metric='cosine')
    reduced_features = umap_model.fit_transform(all_features)

    # 構造返回的 JSON 資料
    response_data = []
    for i, (x, y) in enumerate(reduced_features):
        # 生成相對於 /backend/test_process 的路徑
        relative_path = os.path.relpath(all_files[i], "/home/user/PromptSDVis/backend")
        response_data.append({
            "src": f"/backend/{relative_path.replace(os.sep, '/')}",  # 使用正確的相對路徑
            "x": float(x),  # UMAP 的 X 座標
            "y": float(y),  # UMAP 的 Y 座標
            "label": all_labels[i],  # 圖片的分類標籤
            "color": all_colors[i],  # 分類對應的顏色
        })
    print(response_data)
    return jsonify(response_data)

import torch
from torchvision import models, transforms
# 初始化 VGG19 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
vgg19_model = models.vgg19(pretrained=True).to(device)
vgg19_model.eval()

# 定義 VGG19 的圖像預處理
preprocess_img_vgg19 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_images_with_vgg19(image_folder, vgg19_model, preprocess_img_vgg19):
    """
    使用 VGG19 提取圖像特徵向量
    """
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, f)) and f.lower().endswith(('png', 'jpg', 'jpeg'))
    ]

    if not image_files:
        raise ValueError(f"No valid image files found in folder: {image_folder}")

    features = []
    processed_files = []

    for image_file in image_files:
        try:
            img = Image.open(image_file).convert('RGB')
            img_tensor = preprocess_img_vgg19(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = vgg19_model.features(img_tensor)  # 提取卷積特徵
                feature = torch.flatten(feature, 1).cpu().numpy()  # 展平特徵

            features.append(feature.squeeze())
            processed_files.append(image_file)
        except Exception as e:
            print(f"Error processing file {image_file}: {e}")

    features = np.array(features)
    return features, processed_files

@app.route('/vgg19', methods=['GET'])
def get_combined_vgg19_results():
    """
    使用 VGG19 提取特徵並返回 JSON 格式的結果
    """
    print("VGG19 Endpoint Called!")
    # 定义图片文件夹路径
    image_folders = [
        "/home/user/PromptSDVis/backend/test_process/van_gogh/1886",
        "/home/user/PromptSDVis/backend/test_process/van_gogh/1887",
        "/home/user/PromptSDVis/backend/test_process/van_gogh/1888",
        "/home/user/PromptSDVis/backend/test_process/van_gogh/1889",
        "/home/user/PromptSDVis/backend/test_process/van_gogh/Villege",
        "/home/user/PromptSDVis/backend/test_process/van_gogh/Watercolors",
    ]

    labels = ["1886", "1887", "1888", "1889", "Villege", "Watercolors"]
    colors = ["red", "blue", "green", "orange", "purple", "yellow"]

    all_features = []
    all_labels = []
    all_colors = []
    all_files = []

    # 提取特徵並收集圖片資訊
    for folder, label, color in zip(image_folders, labels, colors):
        features, file_names = process_images_with_vgg19(folder, vgg19_model, preprocess_img_vgg19)
        all_features.append(features)
        all_labels.extend([label] * len(features))
        all_colors.extend([color] * len(features))
        all_files.extend(file_names)

    # 合并所有特徵
    all_features = np.vstack(all_features)

    # 使用 UMAP 进行降维
    umap_model = umap.UMAP(n_neighbors=8, min_dist=0.1, metric='cosine')
    reduced_features = umap_model.fit_transform(all_features)

    # 構造返回的 JSON 資料
    response_data = []
    for i, (x, y) in enumerate(reduced_features):
        # 生成相對於 /backend/test_process 的路徑
        relative_path = os.path.relpath(all_files[i], "/home/user/PromptSDVis/backend")
        response_data.append({
            "src": f"/backend/{relative_path.replace(os.sep, '/')}",  # 使用正確的相對路徑
            "x": float(x),  # UMAP 的 X 座標
            "y": float(y),  # UMAP 的 Y 座標
            "label": all_labels[i],  # 圖片的分類標籤
            "color": all_colors[i],  # 分類對應的顏色
        })
    print(response_data)
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
