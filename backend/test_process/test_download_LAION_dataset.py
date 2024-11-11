import os
import base64

from PIL import Image
import io

# 配置缓存路径
os.environ["HF_HOME"] = "/media/user/新增磁碟區/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "/media/user/新增磁碟區/huggingface_cache"
from datasets import load_dataset
# 打印缓存路径以确认
print(f"HF_DATASETS_CACHE 路径: {os.environ['HF_DATASETS_CACHE']}")
print(f"HF_HOME 路径: {os.environ['HF_HOME']}")

# 设置目标保存目录
save_dir = "/media/user/新增磁碟區/laion_images"
os.makedirs(save_dir, exist_ok=True)

# 加载数据集的前10个样本
dataset = load_dataset("bhargavsdesai/laion_improved_aesthetics_6.5plus_with_images", split="train")

# 遍历并保存图片和文本
for idx, item in enumerate(dataset):
    # 解码 Base64 图片数据
    try:
        image = Image.open(io.BytesIO(item["image"]['bytes']))
        if image.format == 'JPEG':  # 检查格式
            image.save(f"{save_dir}/image_{idx}.jpg")
        else:
            print(f"图像 {idx} 不是 JPEG 格式，跳过保存。")
        
        # 保存文本到指定路径
        with open(f"{save_dir}/text_{idx}.txt", "w") as text_file:
            text_file.write(item["text"])
    
    except Exception as e:
        print(f"处理图像 {idx} 时出错: {e}")

# 清理缓存dataset.cleanup_cache_files()


print(f"已下载到 {save_dir}")
