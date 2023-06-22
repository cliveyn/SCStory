# SCStory 
Presented at WWW'23 [[Paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583507)]

## Used libraries
- torch 1.12
- sentence-transformers 2.0.0
- pandas 1.2.4
- sklearn 0.24.2
- numpy 1.19.5
- tqdm 4.62.3
### External libraries (also included in the "External_libraries" folder)
- spherical_kmeans ([source](https://github.com/rfayat/spherecluster/blob/scikit_update/spherecluster/spherical_kmeans.py))
- b3 ([source](https://github.com/m-wiesner/BCUBED/blob/master/B3score/b3.py))

## Data sets (in the "Datasets" folder)
### Raw data sets
- External link for large data sets: [link](https://www.dropbox.com/sh/fu4i5lghdq18cfs/AABZvrPRXs2qal9rlpnFicBDa?dl=0)
  - Newsfeed14 ([original source](https://github.com/Priberam/news-clustering/blob/master/download_data.sh))
  - WCEP18, WCEP19 ([original source](https://github.com/complementizer/wcep-mds-dataset))
  - CaseStudy 

### Preprocessing
Run Dataset_preprocessing.ipynb with each of the raw data sets.

## Usage
### Input parameters (with default values)
#### GPU settings
- GPU_NUM = 1 # GPU Number
#### Data sets settings
- dataset = 'News14'
- begin_date = '2014-01-02' # the last date of the first window
- window_size = 7
- slide_size = 1
- min_articles = 8 #the number of articels to initiate the first story. 8 for News14 and 18 for WCEP18/19 (the real avg number of articles in a story in a day)
- max_sens = 50
- true_story = True #indicate if the true story labels are available (for evaluation)
#### Algorithm settings
- thred = 0.5 #to decide to initiate a new story or assign to the most confident story
- sample_thred = 0.5 #the minimum confidence score to be sampled (the lower bound is thred)
- temp = 0.2
- batch = 128
- aug_batch = 128
- epoch= 1
- lr = 1e-5
- head = 4
- dropout = 0

### Running examples
```
python SCStory.py
(or python SCStory.py --param_name param_value)

Parameters parsed: Namespace(aug_batch=128, batch=128, begin_date='2014-01-02', dataset='News14', dropou, thred=0.5, true_story=True, window_size=7)
Current cuda device - 1 NVIDIA RTX A6000
Loading datasets....
Datasets loaded
Begin initializing with the first window
100%|███████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 10.37it/s]
Begin evaluating sliding windows
100%|█████████████████████████████████████████████████████████████████| 346/346 [01:51<00:00,  3.10it/s]
Total 583 valid stories are found. The output is saved to output.json
Dataset begin_date B3-P B3-R B3-F1 AMI ARI all_time eval_time train_time
News14 2014-01-02 : 0.8928 0.8719 0.8772 0.8836 0.8172 0.3387 0.0392 0.2995
```

### Citation
```
@inproceedings{yoon2023scstory,
  title={SCStory: Self-supervised and Continual Online Story Discovery},
  author={Yoon, Susik and Meng, Yu and Lee, Dongha and Han, Jiawei},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={1853--1864},
  year={2023}
}
```

