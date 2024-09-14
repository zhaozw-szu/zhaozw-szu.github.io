import os
import json

# 检查本地是否存在 emotion.json 文件
if os.path.exists('emotion.json'):
    with open('emotion.json', 'r', encoding='utf-8') as f:
        emotion_dict = json.load(f)
else:
    # 如果不存在 emotion.json 文件，则构建 emotion_dict
    emotion_lists = [name for name in os.listdir('.') if os.path.isdir(name) and not name.startswith('.')]
    emotion_dict = {}

    for emotion_list in emotion_lists:
        emotions = os.listdir(emotion_list)
        emotion_dict[emotion_list] = {
            "type": "image",
            "container": []
        }
        
        for emotion in emotions:
            icon_path = os.path.join(emotion_list, emotion)
            icon_html = f"![]({icon_path})"
            emotion_dict[emotion_list]["container"].append({
                "icon": icon_html,
                "text": emotion
            })

# 创建 README.md 文件
with open('README.md', 'w', encoding='utf-8') as f:
    for emotion_list, data in emotion_dict.items():
        f.write(f"### {emotion_list}\n\n")
        f.write("#### 示例:\n\n")
        examples = data["container"][:5]
        for example in examples:
            f.write(f"- {example['text']}: {example['icon']}\n")
        f.write("\n")

print("README.md 文件已生成。")