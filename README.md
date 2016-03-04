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

0. optimize the speed and memory efficiency (each minibatch is of size `BATCH_SIZE x (MAX_DOC_LEN + MAX_QRY_LEN) x VOCAB_SIZE`, which is pretty large).
1. improve/complete `DeepLSTMReader.py`.
2. improve the batch loader to handle non-uniform document/query length
...

