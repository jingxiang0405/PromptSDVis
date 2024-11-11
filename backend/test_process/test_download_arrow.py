from datasets import load_from_disk
import csv
from PIL import Image
import io
# 指定 .arrow 檔案的目錄路徑
dataset = load_from_disk('/media/user/新增磁碟區/diffusiondb_dataset_random5k/train')

# 查看數據集的結構
print(dataset)

# 設定要處理的數量
num_images = len(dataset)  # 數據集的行數

csv_file_path = '/media/user/新增磁碟區/diffusiondb_dataset_random5k/imgs_rand5k.csv'


image_output_dir = '/media/user/新增磁碟區/diffusiondb_dataset_random5k/images'
# 打開 CSV 檔案以寫入模式
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # 寫入 CSV 檔案的標頭
    csv_writer.writerow(['image_name', 'prompt', 'seed', 'step', 'cfg', 'sampler', 'width', 'height', 'user_name', 'timestamp', 'image_nsfw', 'prompt_nsfw'])
    
    # 遍歷數據集並保存圖片和其他欄位數據
    for idx in range(num_images):
        example = dataset[idx]  # 注意這裡直接使用 dataset[idx]
        
        # 提取圖片的二進制數據
        img = example['image'] 
        
        # 提取其他欄位的數據
        prompt = example['prompt']
        seed = example['seed']
        step = example['step']
        cfg = example['cfg']
        sampler = example['sampler']
        width = example['width']
        height = example['height']
        user_name = example['user_name']
        timestamp = example['timestamp']
        image_nsfw = example['image_nsfw']
        prompt_nsfw = example['prompt_nsfw']
        
        # 儲存圖片，使用 index 命名
        image_name = f'image_{idx}.png'
        img.save(f'{image_output_dir}/{image_name}')  # 不需要再轉換為 BytesIO
        
        
        # 將其他欄位數據寫入 CSV 檔案
        csv_writer.writerow([image_name, prompt, seed, step, cfg, sampler, width, height, user_name, timestamp, image_nsfw, prompt_nsfw])
        
        # 進度提示
        if idx % 1000 == 0:
            print(f'已保存 {idx} 張圖片及其對應數據')

print('所有圖片及數據已保存完成')