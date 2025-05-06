from flask import Flask, render_template, request, make_response, jsonify
#from flask_cors import *
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # True 表示 CUDA 可用
print(torch.backends.cudnn.is_available())  # True 表示 cuDNN 可用
# from workflow import *

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


import sys

import time
from config import *
import random
import base64
from io import BytesIO
from threading import Lock
lock = Lock()

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# CORS(app, supports_credentials=True)

device_id = sys.argv[1]
device = "cuda:" + device_id
print(device)

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
print("Begin to load model")
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, force_download=True)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

print("Set device")
pipe = pipe.to(device)
print("Start inference")

def get_img_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='png')
    byte_data = output_buffer.getvalue()
    image_str = base64.b64encode(byte_data).decode('utf-8')
    return image_str
# 沒有設定method get or post 都可以
@app.route('/sd', methods=['POST'])
def sd():
    #. POST 如何取得參數
    args = request.get_json()
    print(f'args: {args}')
    epo = int(args.get('epo'))
    guidance_scale = float(args.get('guidance_scale'))
    prompt = args.get('prompt')
    negative_prompt = args.get('negative_prompt')
    random_seed = args.get('random_seed')
    print('Guidance scale :', guidance_scale)
    print('Random seed :', random_seed)

    if (random_seed < 0):
        random_seed = random.randrange(2**32 - 1)
    


    result_dict = []
    for i in range(int(epo)):
        st = time.time()
        scale = guidance_scale
        # set seed
        generators = []
        seed_list = []
        for i in range(n_images_per_prompt):
            generator = torch.Generator(device='cuda')
            generator = generator.manual_seed(random_seed)
            generators.append(generator)
            seed_list.append(random_seed)
        # 使用锁
        with lock:
            if len(negative_prompt) == 0:
                print('negative_prompt is None')
                images = pipe(prompt = prompt, height = sd_height, width = sd_width, num_inference_steps = n_inference_steps,
                    guidance_scale = float(scale), num_images_per_prompt = n_images_per_prompt, generator  = generators)
            else:
                print('negative_prompt is', negative_prompt)
                images = pipe(prompt = prompt, height = sd_height, width = sd_width, num_inference_steps = n_inference_steps,
                    guidance_scale = float(scale), negative_prompt = negative_prompt, num_images_per_prompt = n_images_per_prompt, generator  = generators)
        print("[Infer time: {0}]".format(time.time() - st))

        for j in range(0, n_images_per_prompt):
            inner = {}
            image = images.images[j]
            inner['img'] = get_img_base64(image)
            inner['prompt'] = prompt
            inner['negative_prompt'] = negative_prompt
            inner['width'] = str(sd_width)
            inner['height'] = str(sd_height)
            inner['guidance_scale'] = str(scale)
            inner['seed'] = str(seed_list[j])
            result_dict.append(inner)

    headers = {
        'content-type':'application/json'
    }
    response = make_response(jsonify(result_dict), 200)
    response.headers = headers
    return response
    # return json.dumps(result_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=True, port = (5008 + int(device_id)))









