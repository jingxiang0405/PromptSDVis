import os
from PIL import Image

# 定义原始的目录和转换后的目标目录
input_folder = "/home/user/PromptSDVis/data/diffusiondb_data/images"  # 替换为您的 .jpg 文件所在的目录
output_folder = "/home/user/PromptSDVis/data/diffusiondb_data/images2/"  # 替换为您想保存 .png 文件的目录

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):  # 检查文件是否为 .jpg 或 .jpeg
        # 获取文件的完整路径
        jpg_file_path = os.path.join(input_folder, filename)
        
        # 打开 .jpg 图像
        with Image.open(jpg_file_path) as img:
            # 去掉文件的扩展名，并创建新的 .png 文件名
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_file_path = os.path.join(output_folder, png_filename)
            
            # 将图片转换成 png 并保存
            img.save(png_file_path, "PNG")
            print(f"转换 {filename} 为 {png_filename}")

print("所有文件已转换完成！")