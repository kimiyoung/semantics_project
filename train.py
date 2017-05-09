import numpy as np
import time
import os
import shutil
import cPickle as pkl

from config import *
#from model import GAReaderSelect, GAReaderSelectTied, GAReaderpp, GAKnowledge, GAReaderCoref, GAReaderSelectCoref, GAReaderppAblation, GAReaderSelectAblation, GAReaderppCoref
from model import BiGRU, GA, GAMage
from utils import Helpers, DataPreprocessor, MiniBatchLoader

def main(save_path, params, mode='train'):

    word2vec = params['word2vec']
    dataset = params['dataset']
    base_model = params['model']
    train_cut = params['train_cut']

    # save settings
    shutil.copyfile('config.py','%s/config.py'%save_path)

    use_chars = params['char_dim']>0
    dp = DataPreprocessor.DataPreprocessor()
    data = dp.preprocess(dataset, params['rfile'], max_chains=params['max_chains'], 
            no_training_set=False, use_chars=use_chars)
    if data.validation[0][6]==-1:
        # no clozes
        cloze=False
    else:
        cloze=True

    print("building minibatch loaders ...")
    batch_loader_train = MiniBatchLoader.MiniBatchLoader(data.training, BATCH_SIZE, 
            data.max_num_cand, params['max_chains'], sample=train_cut)
    batch_loader_val = MiniBatchLoader.MiniBatchLoader(data.validation, BATCH_SIZE, 
            data.max_num_cand, params['max_chains'])
    batch_loader_test = MiniBatchLoader.MiniBatchLoader(data.test, BATCH_SIZE, 
            data.max_num_cand, params['max_chains'])
    num_candidates = data.max_num_cand

    print("building network ...")
    W_init, embed_dim, = Helpers.load_word2vec_embeddings(data.dictionary[0], word2vec)
    m = eval(base_model).Model(params, data.vocab_size, data.num_chars, W_init, embed_dim,
            num_candidates, cloze=cloze)

    print("training ...")
    num_iter = 0
    max_acc = 0.
    max_acc_c = 0.
    min_loss = 1e5
    deltas = []

    logger = open(save_path+'/log','a',0)

    if params['reload_']:
        print('loading previously saved model')
        m.load_model('%s/best_model.p'%save_path)

    # train
    if mode=='train':
        tafter = 0.
        for epoch in xrange(NUM_EPOCHS):
            estart = time.time()
            new_max = False
            stop_flag = False

            for (dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, 
                    cl, crd, crq, fnames) in batch_loader_train:
                tc = time.time()-tafter
                loss, tr_acc, probs = m.train(dw, dt, qw, qt, c, a, m_dw, 
                        m_qw, tt, tm, m_c, cl, crd, crq)
                tafter = time.time()

                message = "Epoch %d TRAIN loss=%.4e acc=%.4f elapsed=%.1f (%.1f outside)" % (
                        epoch, loss, tr_acc, time.time()-estart, tc)
                print message
                logger.write(message+'\n')

                num_iter += 1
                if num_iter % VALIDATION_FREQ == 0:
                    total_loss, total_acc, n, n_cand = 0., 0., 0., 0.

                    for (dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, 
                            cl, crd, crq, fnames) in batch_loader_val:
                        outs = m.validate(dw, dt, qw, qt, c, a, 
                                m_dw, m_qw, tt, tm, m_c, cl, crd, crq)
                        loss, acc, probs = outs[:3]

                        bsize = dw.shape[0]
                        total_loss += bsize*loss
                        total_acc += bsize*acc
                        n += bsize

                    val_acc = total_acc/n
                    if val_acc > max_acc:
                        max_acc = val_acc
                        m.save_model('%s/best_model.p'%save_path)
                        new_max = True
                    message = "Epoch %d VAL loss=%.4e acc=%.4f max_acc=%.4f" % (
                        epoch, total_loss/n, val_acc, max_acc)
                    print message
                    logger.write(message+'\n')
                    # stopping
                    val_loss = total_loss/n
                    if val_loss<min_loss: min_loss = val_loss
                    if STOPPING and (val_loss-min_loss)/min_loss>0.1:
                        stop_flag = True
                        break

            #m.save_model('%s/model_%d.p'%(save_path,epoch))
            message = "After Epoch %d: Train acc=%.4f, Val acc=%.4f" % (epoch, tr_acc, max_acc)
            print message
            logger.write(message+'\n')
            
            # learning schedule
            if epoch >=2 and epoch%ANNEAL==0:
                m.anneal()
            # stopping criterion
            #if (STOPPING and not new_max) or val_acc>0.99:
            #    break
            if stop_flag: break

    # test
    mode = 'test'
    m.load_model('%s/best_model.p'%save_path)

    print("testing ...")
    pr = np.zeros((len(batch_loader_test.questions),
        batch_loader_test.max_num_cand)).astype('float32')
    d_pr = np.zeros((len(batch_loader_test.questions),
        batch_loader_test.max_doc_len)).astype('float32')
    fids, attns = [], []
    dreps, qreps = [], []
    all_aggs = []
    total_loss, total_acc, n = 0., 0., 0
    for dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, crd, crq, fnames in batch_loader_test:
        outs = m.validate(dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl, crd, crq)
        loss, acc, probs, drep, qrep, doc_probs = outs[:6]
        aggs = outs[6:6+params['nlayers']]
        dreps.append(drep)
        qreps.append(qrep)
        all_aggs.append(aggs)

        bsize = dw.shape[0]
        total_loss += bsize*loss
        total_acc += bsize*acc

        pr[n:n+bsize,:] = probs
        d_pr[n:n+bsize,:doc_probs.shape[1]] = doc_probs
        fids += fnames
        n += bsize

    message = '%s Loss %.4e acc=%.4f' % (mode.upper(), total_loss/n, total_acc/n)
    print message
    logger.write(message+'\n')

    np.save('%s/%s.probs' % (save_path,mode),np.asarray(pr))
    if callable(getattr(m, "get_output_weights", None)):
        np.save('%s/out_emb.npy' % save_path, m.get_output_weights())
    pkl.dump(attns, open('%s/%s.attns' % (save_path,mode),'w'))
    #pkl.dump([dreps, qreps], open('%s/%s.reps' % (save_path,mode),'w'))
    pkl.dump(all_aggs, open('%s/%s.aggs' % (save_path,mode),'w'))
    f = open('%s/%s.ids' % (save_path,mode),'w')
    for item in fids: f.write(str(item)+'\n')
    f.close()
    logger.close()
