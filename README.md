# semantics_project

## Instructions

The CNN dataset can be downloaded from [here](http://cs.nyu.edu/~kcho/DMQA/).

Global configurations are in `config.py`.

To run naive baselines including word-distance and max-frequency (inclusive/exclusive):
```
python model/NaiveBaselines.py
```

To train a two-layered unidirectional LSTM reader,
first unzip `word2vec_embed.tar.gz` to get word2vec embeddings for initializing the embedding layer.
Then run:
```
python train.py
```

`BidirectionalLSTMReader.py` and `UniformLSTMReader` in `model/` are under development.

## Todo

0. add functionalities such as logging and model saving/loading
1. optimize time/memory efficiency
2. implement other baselines (attentive reader, memory nets, etc.)
3. parameter tuning
