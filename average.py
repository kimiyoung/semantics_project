import os
import numpy as np

dataset = 'validation'
ensemble_path = 'ensemble/'
models = next(os.walk(ensemble_path))[1]

remove_set = ['mul_2l_h256_d.5','mul_2l_h384_d.5']

avg_probs = np.load(ensemble_path+models[0]+'/'+dataset+'.preds.probs.npy')
gt = np.load(ensemble_path+models[0]+'/'+dataset+'.preds.gt.npy')
cpreds = np.argmax(avg_probs, axis=1)
cacc = float((cpreds==gt).sum())/len(cpreds)
print '%s acc = %.3f' % (models[0], cacc)
n = 1
for m in models[1:]:
    if m in remove_set:
        continue
    cprobs = np.load(ensemble_path+m+'/'+dataset+'.preds.probs.npy')
    cpreds = np.argmax(cprobs, axis=1)
    cacc = float((cpreds==gt).sum())/len(cpreds)
    print '%s acc = %.3f' % (m, cacc)
    avg_probs += cprobs
    n += 1

avg_probs = avg_probs/n
preds = np.argmax(avg_probs, axis=1)
acc = float((preds==gt).sum())/len(preds)
print 'ensemble acc = %.3f' % acc
