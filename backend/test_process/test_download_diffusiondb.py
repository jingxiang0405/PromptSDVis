import os
import shutil
import base64
from PIL import Image
import io

# 配置缓存路径
os.environ["HF_HOME"] = "/media/user/新增磁碟區/diffusiondb_dataset_random5k"
os.environ["HF_DATASETS_CACHE"] = "/media/user/新增磁碟區/diffusiondb_dataset_random5k"
import numpy as np
from datasets import load_dataset

# 设置目标保存目录
save_dir = "/media/user/新增磁碟區/diffusiondb_dataset_random5k"

# 清空目標目錄中的文件
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)


# 重新創建目錄
os.makedirs(save_dir, exist_ok=True)

# 加载数据集
dataset = load_dataset('poloclub/diffusiondb', '2m_random_5k', trust_remote_code=True)

# 保存数据集到磁盘
dataset.save_to_disk(save_dir)



