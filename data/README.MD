## Prerequisites
- Python 3.11.6
- GPU Memory >= 4G
- Numpy
- Pytorch 2.2.0 (http://pytorch.org/)
- Torchvision 0.17.0

## Data
The download link of MARKET1501 dataset is:
Data: [Market-1501](http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)

A quick command line to download Market-1501 is: 
```
pip install gdown
gdown https://drive.google.com/uc\?id\=0B8-rUzbwVRk0c054eEozWG9COHM
```

## Prepare Data Folder (`python prepare.py`)
You may notice that the downloaded folder is organized as:
```
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* Files for multiple query testing 
│   ├── gt_query/                   /* We do not use it 
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
```

Open and edit the script `prepare.py` in the editor. Change the fifth line in `prepare.py` to your download path, such as `D::\\DeepSORT\\data\\Market`. Run this script in the terminal.
```bash
python prepare.py
```

We create a subfolder called `Market` under the download folder. 
```
├── Market-1501-v15.09.15/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* Files for multiple query testing 
│   ├── gt_query/                   /* We do not use it
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
├── Market/
│       ├── gallery/                 /* gallery files  
│       ├── multi-query/    
│       ├── query/                   /* query files  
│       ├── train/                   /* train 
│           ├── 0002
|           ├── 0007
|           ...
│       ├── train_all/               /* train+val      
│       ├── val/                     /* val
```

In every subdir, such as `train/0002`, images with the same ID are arranged in the folder.
Now we have successfully prepared the data for `torchvision` to read the data. 
