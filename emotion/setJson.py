import os
import json

# 获取当前目录下的所有文件和目录
emotion_lists = os.listdir('.')

# 初始化一个空字典来存储情感文件路径
emotion_dict = {}

# 遍历每个项
for emotion_list in emotion_lists:
    # 忽略隐藏文件夹（以 . 开头）
    if emotion_list.startswith('.'):
        continue
    
    path = os.path.join('.', emotion_list)
    
    # 只处理目录
    if not os.path.isdir(path):
        continue
    
    emotions = os.listdir(path)
    emotion_dict[emotion_list] = {
        "type": "image",
        "container": []
    }
    
    for emotion in emotions:
        # 构建图标 URL
        icon_url = f"https://fastly.jsdelivr.net/gh/zhaozw-szu/emotion/{emotion_list}/{emotion}"
        icon_html = f"<img src=\"{icon_url}\">"
        
        # 添加到容器
        emotion_dict[emotion_list]["container"].append({
            "icon": icon_html,
            "text": emotion
        })

# 将 emotion_dict 保存到 emotion.json 文件中
with open('emotion.json', 'w', encoding='utf-8') as f:
    json.dump(emotion_dict, f, ensure_ascii=False, indent=4)
