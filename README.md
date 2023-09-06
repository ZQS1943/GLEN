# GLEN: General-Purpose Event Detection
- [GLEN: General-Purpose Event Detection](#glen-general-purpose-event-detection)
  - [Overview](#overview)
  - [Quick Start: Docker](#quick-start-docker)
  - [Data Format](#data-format)
  - [Reproduction](#reproduction)
    - [Setup](#setup)
    - [Model Training](#model-training)
    - [Predict](#predict)

## Overview
This repository contains the code of the paper titled ["GLEN: General-Purpose Event Detection for Thousands of Types"](https:#arxiv.org/pdf/2303.09093.pdf).
***

## Quick Start: Docker
Docker link

## Data Format
Each data file in ./data/data_split is in json format, which contain a list of data instances. 
- The following example shows a training instance.
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
- The following example shows an annotated test instance.
```yaml
{
    "id": "ontonotes/nw/wsj/10/wsj_1057_3", 
    "document": "./propbank-release/data/ontonotes/nw/wsj/10/wsj_1057.gold_conll", 
    "s_id": 3, 
    "domain": "./propbank-release/data/ontonotes/nw", 
    "sentence": "At that price , CBS was the only player at the table when negotiations with the International Olympic Committee started in Toronto Aug. 23 .", 
    "events": [
        {
            "trigger": ["negotiations"], 
            "offset": [13], 
            "pb_roleset": "negotiate.01",
            "candidate_nodes": ["DWD_Q3400581", "DWD_Q202875"], # List of event nodes mapping to the ProbBank roleset 
            "annotation": [ # If multiple candidate nodes exist, annotators select the most suitable event type
                {
                    "label": "negotiation: dialogue between two or more people or parties intended to reach a beneficial outcome", # Selected label by the worker 
                    "worker": "AVYIK8IPJ4865" # Worker's ID
                }, 
                {
                    "label": "negotiation: dialogue between two or more people or parties intended to reach a beneficial outcome", 
                    "worker": "AMRYNBWDDIVDE"
                }
            ],  
            "xpo_label": "DWD_Q202875" # Annotation result
        }, 
        {
            "trigger": ["started"], 
            "offset": [19], 
            "pb_roleset": "start.01", 
            "candidate_nodes": ["DWD_Q28530236"], 
            "xpo_label": "DWD_Q28530236"
        }
    ]
}
```

***
## Reproduction
### Setup
```sh
    git clone https://github.com/ZQS1943/GLEN.git
    cd GLEN
    pip install -r requirements.txt
    python3 ./data/data_preprocessing.py
```

### Model Training

Our model comprises three components:
- **Trigger Identification**: Identify potential triggers in the sentence
- **Type Ranking**: rank the top k possible event types for the sentence
- **Type Classification**: determine the best matching event type for each potential trigger
  
![Overview of the framework](asset/model.png)

To train the Trigger Identification model, use
```sh
bash scripts/train_trigger_identification.sh
```
To train the Type Ranking model, use
```sh
bash scripts/train_type_ranking.sh
```
Before we train Type Classification, we need to get the top k event types for each sentence in the train set predicted by the trained Type Ranking model. To get this, use
```sh
bash scripts/predict_type_ranking.sh train_set
# param1: the predicting data
```
For training the Type Classification model, we adopt an incremental self-labeling procedure to handle the partial labels. Please refer to Section 3.3 of the paper for more details. To train a base classifier, use
```sh
bash scripts/train_type_classifier.sh 0 ./exp/type_ranking/epoch_4/type_ranking_results_of_train_set_with_top_20_events.json
# param1: the model number 
# param2: the path to the training data
```
After the base classifier is trained, we use this model to self-label the train set.
```sh
bash scripts/predict_type_classifier.sh 0 train_set
# param1: the model number
# param2: the predicting data
```

### Predict
