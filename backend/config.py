# diffusiondb
n_imgs = 10000
batch_size = 1000

# common
n_search = 400
n_compress = 0.2
n_test = 5

# stable diffusion
model_id = "runwayml/stable-diffusion-v1-5"
sd_height = 512
sd_width = 512
n_inference_steps = 50
n_images_per_prompt = 1
# n_epo = 1 # one epo cost 20s
n_device = 1 # number of GPU
sd_ip = 'http://140.119.164.166:'
#rand_left = 212463292853
#rand_right = 958446223563
