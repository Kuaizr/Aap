import json
import pandas as pd
import pyarrow as pa
import random
import os
import pickle

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from .glossary import normalize_word


imagejiansuotext = pickle.load(open('./datasets/mmimdb/imagejiansuotext.pkl', 'rb'))
textjiansuoimage = pickle.load(open('./datasets/mmimdb/textjiansuoimage.pkl', 'rb'))
index = pd.read_csv('./datasets/mmimdb/output_final.csv')
path2guid = {index['path'][i].split("/")[-1]:index['guid'][i] for i in range(len(index))}


def get_jiansuo(path):
    guid = path2guid[path.split("/")[-1]]
    text_id = imagejiansuotext[guid].split("-")[0]
    image_id = textjiansuoimage[guid].split("-")[0]
    text_id = text_id.zfill(7)
    image_id = image_id.zfill(7)
    image_path = "./datasets/mmimdb/"+ "dataset/" + str(image_id) + ".jpeg"
    text = " ".join(json.load(open('./datasets/mmimdb/dataset/'+ str(text_id) +'.json', 'r'))['plot'])
    return image_path, text

def make_arrow(root, dataset_root, single_plot=False, missing_type=None):
    GENRE_CLASS = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure', 'Horror'
     , 'Documentary', 'Mystery', 'Sci-Fi', 'Fantasy', 'Family', 'Biography', 'War', 'History', 'Music',
     'Animation', 'Musical', 'Western', 'Sport', 'Short', 'Film-Noir']
    GENRE_CLASS_DICT = {}
    for idx, genre in enumerate(GENRE_CLASS):
        GENRE_CLASS_DICT[genre] = idx    

    image_root = os.path.join(root, 'dataset')
    label_root = os.path.join(root, 'dataset')
    
    with open(f"{root}/split.json", "r") as fp:
        split_sets = json.load(fp)
        
    
    total_genres = []
    for split, samples in split_sets.items():
        data_list = []
        for sample in tqdm(samples):
            image_path = os.path.join(image_root, sample+'.jpeg')
            label_path = os.path.join(label_root, sample+'.json')
            # with open(image_path, "rb") as fp:
            #     binary = fp.read()
            # with open(label_path, "r") as fp:
            #     labels = json.load(fp)    
            
            # There could be more than one plot for a movie,
            # if single plot, only the first plots are used
            # if single_plot:
            #     plots = [labels['plot'][0]]
            # else:
            #     plots = labels['plot']
            
            try:
                image_jiansuo_path, text_jiansuo = get_jiansuo(image_path)
            except:
                continue
            image_gen_path = os.path.join(root,'mmimdb_gen', sample+'.png')
            text_gen_path = os.path.join(root,'mmimdb_gen', sample+'.txt')
                

            try:
                with open(image_path, "rb") as fp:
                    binary = fp.read()

                with open(image_jiansuo_path, "rb") as fp:
                    binary_jiansuo = fp.read()
                
                with open(image_gen_path, "rb") as fp:
                    binary_gen = fp.read()

                with open(text_gen_path, "r") as fp:
                    text_gen = fp.read()
                
                with open(label_path, "r") as fp:
                    labels = json.load(fp)

                if single_plot:
                    plots = [labels['plot'][0]]
                else:
                    plots = labels['plot']

                genres = labels['genres']
                label = [1 if g in genres else 0 for g in GENRE_CLASS_DICT]
            except:
                continue


            data = (binary, plots, label, genres, sample, split, binary_jiansuo, text_jiansuo, binary_gen, text_gen)
            data_list.append(data)

        dataframe = pd.DataFrame(
            data_list,
            columns=[
                "image",
                "plots",
                "label",
                "genres",
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
        with pa.OSFile(f"{dataset_root}/mmimdb_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)        