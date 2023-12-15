import torch,sys
import json, os
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_store = "openai/clip-vit-large-patch14"

model = CLIPModel.from_pretrained(clip_store).to(device)
processor = CLIPProcessor.from_pretrained(clip_store)
tokenizer = CLIPTokenizer.from_pretrained(clip_store)


def get_df(path):
    if os.path.exists(os.path.join(path, 'datalist.csv')):
        return pd.read_csv(os.path.join(path, 'datalist.csv'))

    data = []
    for root, dirs, files in os.walk(os.path.join(path, 'dataset')):
        for file in files:
            # 检查文件后缀是否为.jpeg
            if file.endswith(".jpeg"):
                # 获取完整的文件路径
                full_path = os.path.join(root, file)
                # 将文件名和路径添加到数据列表中
                data.append([file.replace(".jpeg",""), full_path, None])
    df = pd.DataFrame(data, columns=["ids", "path", "plots"])

    for i in tqdm(df.index.tolist()):
        df.iloc[i]['plots'] = " ".join(json.load(open(os.path.join(path, 'dataset',df.iloc[i]['ids'] +'.json'), 'r'))['plot']).replace(",","#")
    
    df.to_csv(os.path.join(path, 'datalist.csv'), index=False)
    
    return df


def get_clip_feature(image, text):
        inputs = processor(text=[" "], images=image, return_tensors="pt", padding=True).to(device)
        text_encoding = tokenizer([text], truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt").to(device)
        inputs["input_ids"] = text_encoding["input_ids"]
        inputs["attention_mask"] = text_encoding["attention_mask"]
        with torch.no_grad():
            outputs = model(**inputs)
        return (outputs.image_embeds.cpu().detach().numpy(), 
                outputs.text_embeds.cpu().detach().numpy())

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_topn_similar(clipfeats,n,save_path):
    results = {}
    for d in tqdm(clipfeats):
        image_id = d['guid']
        image_feature = d['image_feature'].squeeze()
        similarities = []
        for other_d in clipfeats:
            if other_d['guid'] != image_id:
                text_feature = other_d['text_feature'].squeeze()
                similarity = cosine_similarity(image_feature, text_feature)
                similarities.append((other_d['guid'], similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        topn_similar = '-'.join([str(s[0]) for s in similarities[:n]])
        results[image_id] = topn_similar
    
    with open(os.join(save_path,"image_retrieval_text.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    results = {}
    for d in tqdm(clipfeats):
        image_id = d['guid']
        text_feature = d['text_feature'].squeeze()
        similarities = []
        for other_d in clipfeats:
            if other_d['guid'] != image_id:
                image_feature = other_d['image_feature'].squeeze()
                similarity = cosine_similarity(text_feature,image_feature)
                similarities.append((other_d['guid'], similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        topn_similar = '-'.join([str(s[0]) for s in similarities[:n]])
        results[image_id] = topn_similar
    
    with open(os.join(save_path,"text_retrieval_image.pkl"), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    path = '../../datasets/mmimdb/'

    df = get_df(path)
    res = []
    for i in tqdm.tqdm(df.index.tolist()):
        guid = df.loc[i, 'guid']
        path_image = df.loc[i, 'path']
        text = df.loc[i, 'text']
        
        try:
            img = Image.open(os.path.join(path,path_image))
        except:
             continue

        (text_feature, clip_img) = get_clip_feature(img, text)
        res.append({'guid': guid, 'imagefile': path_image, 'image_feature':clip_img,'text_feature':text_feature})

    find_topn_similar(res, 1, save_path=path)
