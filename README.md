# GLEN: General-Purpose Event Detection
- [GLEN: General-Purpose Event Detection](#glen-general-purpose-event-detection)
  - [Overview](#overview)
  - [Data Format](#data-format)
  - [Experiments](#experiments)
    - [Setup](#setup)
    - [Model Training](#model-training)
    - [Predict](#predict)
***
## Overview
This repository contains the code of the paper titled ["GLEN: General-Purpose Event Detection for Thousands of Types"](https:#arxiv.org/pdf/2303.09093.pdf).
***

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
## Experiments
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

To train our model

To train the different components of GLEN model, use 
```sh
bash scripts/run_trigger_detector.sh train ./data/tokenized_final_no_other 128 64 16 False wo_other_new_ontology
bash scripts/run_type_ranking.sh train 64 16 5 1e-5 se_id_new_loss_new_ontology -1 new_loss
bash scripts/run_event_trigger_matching_bts.sh train 32 64 5 1e-5 all_data_new_ontology -1
```

### Predict
To get the predicted results, use
```sh
bash scripts/run_trigger_detector.sh predict ./data/tokenized_final_no_other 128 32 64 False wo_other_new_ontology 4
bash scripts/run_type_ranking.sh predict 8 16 5 1e-5 se_id_new_loss_new_ontology -1 new_loss 4
bash scripts/run_event_trigger_matching_bts.sh predict 64 64 5 1e-5 all_data_new_ontology no_file -1 1
```