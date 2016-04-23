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
python train.py [save_path]
```
Model after each epoch will be saved to `save_path`.

`BidirectionalLSTMReader.py` and `UniformLSTMReader` in `model/` are under development.

To make predictions
```
python predict.py [lstm_model_path] [output_path]
```

By default, `predict.py` uses bidirectional LSTM. 
Each line of the prediction file at `output_path` is formatted as
```
question_id predicted_ans_1 predicted_ans_2 ... predicted_ans_K true_ans
```

## Todo

1. optimize time/memory efficiency
2. implement other baselines (attentive reader, memory nets, etc.)
