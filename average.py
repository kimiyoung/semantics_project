import os
import numpy as np

dataset = 'test'
ensemble_path = 'ensemble/'
models = next(os.walk(ensemble_path))[1]

avg_probs = np.load(ensemble_path+models[0]+'/'+dataset+'.preds.probs.npy')
gt = np.load(ensemble_path+models[0]+'/'+dataset+'.preds.gt.npy')
n = 1
for m in models[1:]:
    avg_probs += np.load(ensemble_path+m+'/'+dataset+'.preds.probs.npy')
    n += 1

avg_probs = avg_probs/n
preds = np.argmax(avg_probs, axis=1)
acc = float((preds==gt).sum())/len(preds)
print acc
