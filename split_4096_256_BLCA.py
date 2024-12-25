from PIL import Image
import os
import numpy as np

# 设置源目录和目标目录
source_dir = './TCGA-BLCA/BLCA_Region'
destination_dir = './TCGA-BLCA/BLCA_Patch'

# 确保目标目录存在
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# 遍历源目录下的所有文件夹
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    
    # 创建相应的目标文件夹
    dest_folder_path = os.path.join(destination_dir, folder_name)
    if not os.path.exists(dest_folder_path):
        os.makedirs(dest_folder_path)
    
    # 遍历文件夹中的所有图像文件
    for image_name in os.listdir(folder_path):
        if image_name.endswith('.jpeg'):
            image_path = os.path.join(folder_path, image_name)
            with Image.open(image_path) as img:
                # 计算切割后的小图像数量
                cols, rows = img.size[0] // 256, img.size[1] // 256
                
                # 切割图像
                for i in range(rows):
                    for j in range(cols):
                        # 计算小图像的坐标
                        left, top, right, bottom = j * 256, i * 256, (j + 1) * 256, (i + 1) * 256
                        # 切割小图像
                        patch = img.crop((left, top, right, bottom))

                        blank_mask = np.all(np.array(patch) > 230, axis=-1)
                        # 计算这些"空白"像素的总数
                        blank_pixels = np.sum(blank_mask)
                        total_pixels = 256 * 256
                        blank_ratio = blank_pixels / total_pixels

                        # 构建小图像的名称和路径
                        patch_name = f"{image_name[:-5]}_{j}_{i}.jpeg"
                        patch_path = os.path.join(dest_folder_path, patch_name)
                        # 保存小图像
                        if blank_ratio < 0.85:
                            patch.save(patch_path)
                        
            print(f"已处理图像：{image_path}")

        # nohup python split_4096_256_BLCA.py > split_4096_256_BLCA.txt 2>&1 &