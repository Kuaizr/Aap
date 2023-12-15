
# Data directory structure:

After you use both data augmentation methods, please organize the datasets as follows, otherwise you may need to revise the `write_*.py` files to meet your dataset path and files.

## MM-IMDb
[MM-IMDb](https://github.com/johnarevalo/gmu-mmimdb) [(archive.org mirror)](https://archive.org/download/mmimdb)

    mmimdb
    ├── mmimdb_gen
    │   ├── 00000005.jpeg 
    │   ├── 00000008.txt   
    │   └── ...     
    ├── dataset            
    │   ├── images            
    |   |   ├── 00000005.jpeg
    |   |   ├── 00000008.jpeg
    |   |   └── ...
    │   ├── labels
    |   |   ├── 00000005.json
    |   |   ├── 00000008.json
    |   |   └── ...
    ├── split.json
    ├── datalist.csv
    ├── text_retrieval_image.pkl
    └──image_retrieval_text.pkl


## Food101
[UPMC Food-101](https://visiir.isir.upmc.fr/explore) [(Kaggle)](https://www.kaggle.com/datasets/gianmarco96/upmcfood101?select=texts)

    root
    ├── food101_gen
    │   ├── apple_pie_0.jpg
    │   ├── apple_pie_0.txt
    │   └── ...
    ├── images            
    │   ├── train                
    │   │   ├── apple_pie
    │   │   │   ├── apple_pie_0.jpg        
    │   │   │   └── ...         
    │   │   ├── baby_back_ribs  
    │   │   │   ├── baby_back_ribs_0.jpg        
    │   │   │   └── ...    
    │   │   └── ...
    │   ├── test                
    │   │   ├── apple_pie
    │   │   │   ├── apple_pie_0.jpg        
    │   │   │   └── ...         
    │   │   ├── baby_back_ribs  
    │   │   │   ├── baby_back_ribs_0.jpg        
    │   │   │   └── ...    
    │   │   └── ...
    ├── texts          
    │   ├── train_titles.csv            
    │   └── test_titles.csv         
    ├── class_idx.json         
    ├── text.json         
    └── split.json
    ├── datalist.csv
    ├── text_retrieval_image.pkl
    └──image_retrieval_text.pkl