from flask import Flask, render_template, request, make_response, jsonify
#from flask_cors import *
import torch

from util import get_img_base64
# from workflow import *

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


import sys

import time
from config import *
import random
from threading import Lock
lock = Lock()


app = Flask(__name__)
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

# 沒有設定method get or post 都可以
@app.route('/sd', methods=['GET', 'POST'])
def sd():
    #. POST 如何取得參數
    args = request.get_json()
    print(f'args: {args}')
    epo = int(args.get('epo'))
    scale_left = float(args.get('scale_left'))
    scale_right = float(args.get('scale_right'))
    prompt = args.get('prompt')
    negative_prompt = args.get('negative_prompt')

    # set scale list
    w_list = []
    while (scale_left <= scale_right):
        w_list.append(scale_left)
        scale_left += 0.5
    w_len = len(w_list)
    print('Guidance scale sample list:', w_list)
    result_dict = []
    for i in range(int(epo)):
        st = time.time()
        scale = w_list[random.randrange(0, w_len)]
        # set seed
        generators = []
        seed_list = []
        for i in range(n_images_per_prompt):
            seed = random.randrange(2**32 - 1)
            generator = torch.Generator(device='cuda')
            generator = generator.manual_seed(seed)
            generators.append(generator)
            seed_list.append(seed)
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









