## GLEN: General-Purpose Event Detection
<p align="left">
  <a href='https://arxiv.org/abs/2303.09093'>
    <img src='https://img.shields.io/badge/Arxiv-2308.16905-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/pdf/2303.09093.pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
  <!-- <a href='TBD'>
  <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a> -->
  <!-- <a href='TBD'>
  <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'></a> -->
  <!-- <a href='TBD'>
    <img src='https://img.shields.io/badge/Bilibili-Video-4EABE6?style=flat&logo=Bilibili&logoColor=4EABE6'></a> -->
  <!-- <a href='TBD'>
    <img src='https://img.shields.io/badge/Zhihu-Doc-2F6BE0?style=flat&logo=Zhihu&logoColor=2F6BE0'></a>  -->
  <a href='https://github.com/ZQS1943/GLEN.git'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=ZQS1943.GLEN&left_color=gray&right_color=orange">
  </a>
</p>

## What's New

* **[Next]** We are currently working on the Python package for our model, CEDAR.
* **[10/27/2023]** The code has been released.

## Data Format
Each data file in `./data/data_split` is in JSON format and contains a list of data instances. Below is an example of a training instance:
```yaml
{ 
    "id": "propbank_15251", # A unique identifier for the sentence
    "document": "propbank_15251", # Source document of the sentence
    "s_id": 0, # Order of the sentence in the source document
    "domain": "propbank", # Source domain
    "sentence": "he had worked with mr. mcdonough on an earlier project and recruited him as architect for the trade center .", # The original sentence text
    "events": [ # List of events in the sentence
        {
            "trigger": ["worked"], # Words in the sentence that trigger the event
            "offset": [2], # Offset positions of the trigger words
            "pb_roleset": "work.01" # The associated PropBank roleset for this event
        }, 
        {
            "trigger": ["recruited"], 
            "offset": [11], 
            "pb_roleset": "recruit.01"
        }
    ],
    "merged_from": "propbank_15251&ontonotes/nw/wsj/14/wsj_1455_58" # Optional attribute indicating merger of instances from different sources
}
```
And here is an example of an annotated test instance:
```yaml
{
    "id": "NapierDianne_183", 
    "document": "./propbank-release/data/oanc/masc/spoken/00/NapierDianne.gold_conll", 
    "s_id": 183, 
    "domain": "./propbank-release/data/oanc/masc", 
    "sentence": "and we would order one pizza and all share it /", 
    "events": [
        {
            "trigger": ["order"], 
            "offset": [3], 
            "pb_roleset": "order.02",
            "candidate_nodes": ['DWD_Q1779371', 'DWD_Q2556572', 'DWD_Q566889'], # List of event nodes mapping to the ProbBank roleset 
            "annotation": [ # If multiple candidate nodes exist, annotators select the most suitable event type
                {
                    'label': 'order: stated intention to engage in a commercial transaction for specific products or services', 
                    'worker': 'A3PQ5TK771LX04'
                }, 
                {
                    'label': 'order: stated intention to engage in a commercial transaction for specific products or services', 
                    'worker': 'AMRYNBWDDIVDE'
                }
            ],  
            "xpo_label": "DWD_Q566889" # Annotation result
        }, 
        {
            "trigger": ["share"], 
            "offset": [8], 
            "pb_roleset": "share.01", 
            "candidate_nodes": ["DWD_Q459221"], 
            "xpo_label": "DWD_Q459221"
        }
    ]
}
```

## Reproduction
### Setup
```sh
    git clone https://github.com/ZQS1943/GLEN.git
    cd GLEN
    pip install -r requirements.txt
```

### Predict

To predict on your own data, download our [checkpoits](https://drive.google.com/file/d/1UU1UVPpYypRh5dPUhQ8TreAJd-uoLEh7/view?usp=sharing), place it under `your_path_to/GLEN/`, and execute the example commands provided:
```sh
unzip ckpts.zip
bash scripts/predict_sentence.sh
```

### Data Preparation
Download [AMR](https://catalog.ldc.upenn.edu/LDC2020T02) and [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19) and place them under `your_path_to/GLEN/data/source_data/`. Ensure the following directory structure:
```sh
source_data
    -LDC2019E81_Abstract_Meaning_Representation_AMR_Annotation_Release_3.0
        -data
        -docs
    -ontonotes-release-5.0
        -data
        -docs
        -tools
```
Then run the following code to map the sentence and preprocess and data:
```sh
export PYTHONPATH=./
python3 data/data_preparation/map_data.py
python3 data/data_preparation/data_preprocessing.py
```



### Model Training

Our model consists of three components: Trigger Identification, Type Ranking, and Type Classification. To train these models, use the provided scripts:
  
![Overview of the framework](asset/model.png)

To train the Trigger Identification model:
```sh
bash scripts/train_trigger_identification.sh
```
To train the Type Ranking model:
```sh
bash scripts/train_type_ranking.sh
```
Before we train Type Classification model, we need to get the top k event types for each sentence in the train set predicted by the trained Type Ranking model. To get this, use the following command:
```sh
bash scripts/predict_type_ranking.sh train_set
# param1: the predicting data
```
For training the Type Classification model, we adopt an incremental self-labeling procedure to handle the partial labels. Please refer to Section 3.3 of the paper for more details. To train a base classifier, use the following command:
```sh
bash scripts/train_type_classifier.sh 0 ./exp/type_ranking/epoch_4/train_data_for_TC.json
# param1: the model number 
# param2: the path to the training data
```
After the base classifier is trained, we use this model to self-label the train set with the following command:
```sh
bash scripts/predict_type_classifier.sh 0 train_set ./exp/type_ranking/epoch_4/train_data_for_TC.json
# param1: the model number
# param2: the predicting data
# param3: the path to the training data
```
Then, we use the self-labeled train set to retrain the Type Classification model using the command below:
```sh
bash scripts/train_type_classifier.sh 1 ./exp/type_classifier_0/epoch_1/train_data_for_TC.json
# param1: the model number 
# param2: the path to the training data
```

### Evaluate

To evaluate the entire piprline, use these commands:
```sh
bash scripts/predict_trigger_identification.sh
bash scripts/predict_type_ranking.sh test_set
# param1: the predicting data
bash scripts/predict_type_classifier.sh 1 test_set
# param1: the model number
# param2: the predicting data
```

## Docker
You can also use Docker to run the GLEN server:
```sh
docker pull qiusi/glen_sentence # pull the image
docker run --gpus all -p 5000:5000 qiusi/glen_sentence # Start the GLEN server
```
<!-- TODO: docker for all cuda version -->