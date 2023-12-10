import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
# from glob import glob
# from collections import defaultdict, Counter
# from .glossary import normalize_word


import pickle
import pandas as pd

imagejiansuotext = pickle.load(open('./datasets/Food101/imagejiansuotext.pkl', 'rb'))
textjiansuoimage = pickle.load(open('./datasets/Food101/textjiansuoimage.pkl', 'rb'))
index = pd.read_csv('./datasets/Food101/output_final.csv')
path2guid = {index['path'][i].split("/")[-1]:index['guid'][i] for i in range(len(index))}

def get_jiansuo(path):
    guid = path2guid[path.split("/")[-1]]
    text_id = int(imagejiansuotext[guid].split("-")[0])
    image_id = int(textjiansuoimage[guid].split("-")[0])
    image_path = "./datasets/Food101/"+ index['path'][image_id]
    text = index['text'][text_id]
    return image_path, text


def make_arrow(root, dataset_root, single_plot=False, missing_type=None):
    image_root = os.path.join(root, 'image')
    
    with open(f"{root}/class_idx.json", "r") as fp:
        FOOD_CLASS_DICT = json.load(fp)
        
    with open(f"{root}/text.json", "r") as fp:
        text_dir = json.load(fp)
        
    with open(f"{root}/split.json", "r") as fp:
        split_sets = json.load(fp)
        
    
    for split, samples in split_sets.items():
        split_type = 'train' if split != 'test' else 'test'
        data_list = []
        for sample in tqdm(samples):
            if sample not in text_dir:
                print("ignore no text data: ", sample)
                continue
            cls = sample[:sample.rindex('_')]
            label = FOOD_CLASS_DICT[cls]
            image_path = os.path.join(image_root, split_type, cls, sample)
            try:
                image_jiansuo_path, text_jiansuo = get_jiansuo(image_path)
            except:
                continue
            image_gen_path = os.path.join(root,'food101_gen', sample)
            text_gen_path = os.path.join(root,'food101_gen', sample.replace('.jpg', '.txt'))

            try:
                with open(image_path, "rb") as fp:
                    binary = fp.read()

                with open(image_jiansuo_path, "rb") as fp:
                    binary_jiansuo = fp.read()
                
                with open(image_gen_path, "rb") as fp:
                    binary_gen = fp.read()
                    
                text = [text_dir[sample]]

                with open(text_gen_path, "r") as fp:
                    text_gen = fp.read()
            except:
                continue
            

            data = (binary, text, label, sample, split, binary_jiansuo, text_jiansuo, binary_gen, text_gen)
            data_list.append(data)
            

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "image",
                "text",
                "label",
                "image_id",
                "split",
                "image_jiansuo",
                "text_jiansuo",
                "image_gen",
                "text_gen",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/food101_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)        