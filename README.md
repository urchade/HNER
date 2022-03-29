# seq_lab

## Installation
clone the repository

```bash
git clone https://github.com/urchade/HNER.git
cd HNER
pip install -r requirements.txt
pip install .
```


## Training a model

```python
from seqlab.train_model import create_config, train_model

name = "keyphrase_extractor"
dirpath= "."

data_path = "data/dataset.pk"
# the dataset file (pickle) should be a dictionnary containing 'train' and 'dev' splits and a 'tag_to_id' dict
# for instance:

# data['train'] = [
#     {'tokens': ['NER', 'is', 'an'], 'tags': ['B-ENT', 'O', 'O']}
#     , ....]
# data['dev'] = [
#     {'tokens': ['NER', 'is', 'an'], 'tags': ['B-ENT', 'O', 'O']},
#      ....]
#  data['tag_to_id'] = {'O' 0, 'B-ENT': 1, 'I-ENT': 2}
# create config
config = create_config(name, dirpath=dirpath, data_path=data_path, max_epoch=5, model_name='allenai/scibert_scivocab_uncased')

# train model
train_model(config)
```


## Loading a model

```python
from seqlab.inference import load_model

checkpoint = "keyphrase_extractor.ckpt"

# load model
model = load_model(checkpoint)

# prediction
tokens = "Recent years have seen the paradigm shift of Named Entity Recognition systems from sequence labeling to span prediction .".split()

prediction = model.extract_entities([tokens])

print(prediction)
```
