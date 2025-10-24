#####     Get Bird Photos (Multi-Thread)     #####
import csv
import requests
from pathlib import Path
from retrying import retry
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

rootPath = "./"
species = set()  # 物种集合
valFrac = 5  # 测试集比例：1/5
success = 0  # 成功计数器
success_lock = threading.Lock()  # 成功计数器锁
species_lock = threading.Lock()  # 物种集合锁

@retry(stop_max_attempt_number=3)
def download(url: str, figPath: Path):
    """Download a photo and save, and retry when failed"""
    response = requests.get(url, timeout=10,
        headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    figPath.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    with open(figPath, "+wb") as fig:
        fig.write(response.content)
    response.close()
    return True

stop = False  # 提前停止 (False: 不停止)
def process_row(row_data):
    global success, species
    name, url, id, train = row_data
    
    with species_lock: species.add(name)
    
    aaa = name.split()
    shortName = aaa[0][0] + aaa[1][0]
    folder = rootPath + "birdData/train/" if train\
        else rootPath + "birdData/val/"
    figPath = Path(folder + f"{name}/{shortName}_{id}.jpg")
    
    if figPath.exists():  # 文件已存在
        return "SKIP"
    
    try:
        download(url, figPath)
        with success_lock:
            current_success = success + 1
            success = current_success
            if stop and (current_success >= stop):  # 达到上限时停止
                return "STOP"
        return True
    except Exception as ex:
        return (url, str(ex))

### ---------- Download Photos ---------- ###
if __name__ == "__main__":
    ### 预处理所有数据
    preprocessed_data = []
    with open(rootPath + "species1.csv", "r", encoding="utf-8") as csvFile:
        reader = csv.DictReader(csvFile)
        train_counter = 0
        for _, row in enumerate(reader):
            id = row["id"]
            name = row["scientific_name"]
            url = row["image_url"]
            train_counter = (train_counter+1) % valFrac
            preprocessed_data.append((name, url, id, train_counter!=0))
    
    total = len(preprocessed_data)
    stop_flag = False
    
    ### 下载进程
    print("Downloading...")
    with tqdm(total=total, desc="下载进度") as pbar:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            # 提交任务
            for data in preprocessed_data:
                if stop_flag: break
                futures.append(executor.submit(process_row, data))
            
            # 处理结果
            for future in as_completed(futures):
                result = future.result()
                if result == "STOP":  # 提前停止
                    for f in futures:
                        f.cancel()
                    stop_flag = True
                    break
                elif result == "SKIP":  # 跳过
                    pbar.update(1)
                elif isinstance(result, tuple):  # 失败
                    print(f"\nFail: {result[0]}\n{result[1]}")
                    pbar.update(1)
                else:
                    pbar.update(1)

    print()
    print(f"Downloading completed:  {total}")
    print(f"Successful img:  {success}")
    print(f"Total Species:  {len(species)}")

# Downloading completed:  222759
# Successful img:  222758
# Total Species:  373

# 143180235	Limosa limosa	黑尾塍鹬	https://static.inaturalist.org/photos/245529083/medium.jpg
# ('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))