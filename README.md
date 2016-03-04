# semantics_project

## Instructions

The CNN dataset can be downloaded from [here](http://cs.nyu.edu/~kcho/DMQA/).

Global configurations are in `config.py`.

To run naive baselines including word-distance and max-frequency (inclusive/exclusive):
```
python NaiveBaselines.py
```

To train a (very preliminary) two-layered unidirectional LSTM:
```
python DeepLSTMReader.py
```

Generate the vocabulary list if `vocab.txt` does not exist:
```
python gen_vocab.py
```

## Todo

0. check whether is model architecture is correct
(the current model doesn't seem to converge to a meaningful local minimum,
suggesting there might be bugs)
1. optimize speed and memory efficiency
2. implement some other baselines (e.g. SVM)
3. improve the batch loader to handle non-uniform document/query length

