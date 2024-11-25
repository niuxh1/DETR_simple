import os
import requests
from tqdm import tqdm  # 用于显示进度条

# COCO 官方下载链接
COCO_URLS = {
    'train': 'http://images.cocodataset.org/zips/train2017.zip',
    'val': 'http://images.cocodataset.org/zips/val2017.zip',

    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}


def download_file(url, save_path):
    """
    从指定 URL 下载文件并保存到本地。
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('Content-Length', 0))

    with open(save_path, 'wb') as file, tqdm(
            desc=save_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            ncols=100
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))


def download_coco_data(download_dir):
    """
    下载 COCO 数据集中的训练集、验证集、测试集和标注文件。
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for name, url in COCO_URLS.items():
        print(f"开始下载 {name} 数据...")
        save_path = os.path.join(download_dir, f'{name}.zip')
        if not os.path.exists(save_path):
            download_file(url, save_path)
        else:
            print(f"{name} 数据已存在，跳过下载。")

        # 解压缩文件
        print(f"解压 {name} 数据...")
        os.system(f"unzip -o {save_path} -d {download_dir}")
        print(f"{name} 数据下载和解压完成！")


if __name__ == "__main__":
    # 设置下载路径
    download_dir = 'coco_data'
    download_coco_data(download_dir)
