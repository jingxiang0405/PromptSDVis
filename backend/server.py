import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # Use 0-index GPU for server pipeline
import io
from flask import Flask, render_template, request, make_response
from flask_cors import *
import numpy as np
import torch
import math
import json
import clip
import base64
import datasets
from io import BytesIO
from util import *
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch

from config import *
from PIL import Image
import random
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app, supports_credentials=True)





device = "cuda" if torch.cuda.is_available() else "cpu"
print('[Server] device', device)
model, preprocess_img = clip.load("ViT-B/32", device=device, download_root='../.cache/clip')


"""

ajax iput 後續將圖片 + 文字做 CLIP Encoder (文加圖 或圖片)

在用TSNE 顯示 然後再看該位置如何 放圖片上去

然後結合D3 js 去顯示圖片

mouseover 顯示圖片的詳細資料

"""
# Global Value
all_process_img_list = []

@app.route('/image_overview')
def get_image_overview():
    request_data = requestParse(request)
    print(request_data)

    prompt = request_data['prompt_val']
    negative_prompt = request_data['negative_prompt_val']
    guidance_scale = request_data['range_slider_val'].split(',')
    generation_val = int(request_data['total_generation_val'])

    scale_left, scale_right = [float(i) for i in guidance_scale]
    n_sd = math.ceil(generation_val / n_images_per_prompt)

    ###################     stable diffison   ######################
    # Stable diffusion produce data
    sd_data = sd_infer(prompt, negative_prompt, scale_left, scale_right, n_sd)

    #print(sd_data)
    ###################     TSNE   ######################
    sd_origin_image_list = []

    for i in range(len(sd_data)):
        sd_origin_image_list.append(base64_to_image(sd_data[i]['img']))
        all_process_img_list = [preprocess_img(img) for img in sd_origin_image_list]

    image_features = encode_image(all_process_img_list, model)

    embed_position = embed_feature(image_features.cpu())


    # 暫時先呈現結果
    plt.figure(figsize=(8, 8))
    print(embed_position[:, 0])
    plt.plot(embed_position[:, 0], embed_position[:, 1], 'o')  # 'o' 是一个圆形标记
    plt.title('2D projection with TSNE')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    result_json = {"images": sd_data, "plt_result": plt_data}

    plt.close()
    """
    ###################     response data   ######################
    image_result_list = []
    for i in range(numberOfGeneration + n_search):
        data_item = {}
        data_item['id'] = str(i)
        data_item['x'] = str(embed_position[i][0])
        data_item['y'] = str(embed_position[i][1])

        compress_image = compress_PIL_image(all_origin_img_list[i])
        data_item['img'] = getImgStr(compress_image)
        img_type = 'generate' if i < numberOfGeneration else 'search'
        data_item['type'] = img_type

        image_result_list.append(data_item)
    """
    #print(result_json)
    return json.dumps(result_json)


def sd_infer(prompt: str,negative_prompt: str,scale_left: float, scale_right: float ,  n_epo=1):
    port = 5008
    params = {
        'epo': n_epo,
        'scale_left': scale_left,
        'scale_right': scale_right,
        'prompt': prompt,
        'negative_prompt': negative_prompt,
    }
    param_encode = urlencode(params)
    url = sd_ip + str(port) + '/sd' + '?' + param_encode
    print("url: {0}".format(url))
    response = requests.get(url=url)

    data = response.json()
    return data

# Encode the image 
def encode_image(image_list, encode_model):
	image_tensor = torch.tensor(np.stack(image_list)).to(device)

	with torch.no_grad():
		image_features = encode_model.encode_image(image_tensor)
		
	return image_features
# Embed the feature into 2D space
def embed_feature(encode_feature):
	# Use TSNE for projection
	embed_position = TSNE(n_components=2, init='random', perplexity=5, metric='cosine').fit_transform(encode_feature)
	# Use UMAP for projection
	# embed_position = umap.UMAP(n_neighbors=5, min_dist=0.001, metric='cosine').fit_transform(encode_feature)
	return embed_position
	

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=5002)
