import numpy as np
from config import *
from model import SequenceLabeler

m = SequenceLabeler.Model(5,3)
d = np.expand_dims(np.asarray([[1,0,1,3],[2,2,1,4]]).astype('int32'), axis=2)
q = np.expand_dims(np.asarray([[2,1],[3,1]]).astype('int32'), axis=2)
dm = np.asarray([[1,1,1,0],[1,1,1,0]]).astype('int32')
qm = np.asarray([[1,1],[1,0]]).astype('int32')
a = np.asarray([0,1]).astype('int32')
for i in range(100):
    print 'loss ', m.train(d,q,a,dm,qm,[])[0]
    print 'acc ', m.validate(d,q,a,dm,qm,[])[1]
