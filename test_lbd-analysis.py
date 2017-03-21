import sys
import numpy as np
import cPickle as pkl
import shutil

from config import *
from model import GAReader, GAReaderpp_prior, StanfordAR, GAReaderpp, GAReaderppp, GAKnowledge
from model import GAReaderpropnc
from model import DeepASReader, DeepAoAReader
from utils import Helpers, DataPreprocessor, MiniBatchLoader

def main(load_path, params, mode='analysis'):

    word2vec = params['word2vec']
    dataset = params['dataset']
    base_model = params['model']

    use_chars = params['char_dim']>0
    dp = DataPreprocessor.DataPreprocessor()
    data = dp.preprocess_analysis(dataset)
    inv_vocab = data.inv_dictionary
    cloze=True

    print("building minibatch loaders ...")
    batch_loader_test = MiniBatchLoader.MiniBatchLoader(data.test, 87, shuffle=False)
    num_candidates = batch_loader_test.max_num_cand

    print("building network ...")
    W_init, embed_dim = Helpers.load_word2vec_embeddings(data.dictionary[0], word2vec)
    m = eval(base_model).Model(params, data.vocab_size, data.num_chars, W_init, 
            embed_dim, num_candidates, cloze=cloze, save_attn=True)
    m.load_model('%s/best_model.p'%load_path)

    print("testing ...")
    pr = np.zeros((len(batch_loader_test.questions),
        batch_loader_test.max_num_cand)).astype('float32')
    d_pr = np.zeros((len(batch_loader_test.questions),
        batch_loader_test.max_doc_len)).astype('float32')
    correct = np.zeros((len(batch_loader_test.questions),))
    fids, attns, answers = [], [], []
    total_loss, total_acc, n = 0., 0., 0
    for dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, crd, crq, fnames, cands in batch_loader_test:
        outs = m.validate(dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl, crd, crq)
        loss, acc, probs, doc_probs = outs[:4]

        bsize = dw.shape[0]
        total_loss += bsize*loss
        total_acc += bsize*acc
        answers += [inv_vocab[cc[np.argmax(probs[ii,:])][0]] for ii,cc in enumerate(cands)]

        pr[n:n+bsize,:] = probs
        d_pr[n:n+bsize,:doc_probs.shape[1]] = doc_probs
        correct[n:n+bsize] = (np.argmax(probs, axis=1)==a)
        fids += fnames
        n += bsize

    logger = open(load_path+'/log','a',0)
    message = '%s Loss %.4e acc=%.4f' % (mode.upper(), total_loss/n, total_acc/n)
    print message
    logger.write(message+'\n')
    logger.close()

    np.save('%s/%s.correct' % (load_path,mode), correct)
    np.save('%s/%s.probs' % (load_path,mode),np.asarray(pr))
    pkl.dump(attns, open('%s/%s.attns' % (load_path,mode),'w'))
    f = open('%s/%s.ids' % (load_path,mode),'w')
    for item in fids: f.write(str(item)+'\n')
    f.close()
    f = open('%s/%s.answers' % (load_path,mode),'w')
    for item in answers: f.write(str(item)+'\n')
    f.close()

    annots = open('../lambada/lambada-analysis/test_coref.annot').read().splitlines()
    perf = {}
    for ii,fi in enumerate(fids):
        co = correct[ii]
        ann = annots[int(fi)].split('\t')[-1].split(', ')
        for aa in ann:
            if aa not in perf: perf[aa] = [0,0]
            perf[aa][1] += 1
            if co: perf[aa][0] += 1

    for k,v in perf.iteritems():
        print k, '\t\t\t', float(v[0])/v[1], '\t', v[1]

if __name__=='__main__':
    params = {}
    params['nhidden'] = 192
    params['char_dim'] = 0
    params['dropout'] = 0.2
    params['word2vec'] = 'word2vec/word2vec_glove.txt'
    params['train_emb'] = 1
    params['use_feat'] = 0
    params['num_coref'] = 14
    params['coref_dim'] = 64
    params['model'] = 'GAReaderpp'
    params['regularizer'] = 'l2'
    params['lambda'] = 0
    params['nlayers'] = 3
    params['dataset'] = 'lambada'
    params['subsample'] = -1
    params['seed'] = 1
    params['train_cut'] = 1.0
    params['gating_fn'] = 'T.mul'
    w2v_filename = params['word2vec'].split('/')[-1].split('.')[0] if params['word2vec'] else 'None'
    save_path = ('crfreader_experiments_babi-v2/'+params['model']+'/'+params['dataset'].split('/')[0]+
            '/m'+
            '_lr%.4f'%LEARNING_RATE+
            '_bsize%d'%BATCH_SIZE+
            '_anneal%d'%ANNEAL+
            '_stop%d'%int(STOPPING)+
            #'reg%s'%params['regularizer']+
            #'%.3f'%params['lambda']+
            '_nhid%d'%params['nhidden']+'_nlayers%d'%params['nlayers']+
            '_dropout%.1f'%params['dropout']+'_%s'%w2v_filename+'_chardim%d'%params['char_dim']+
            '_train%d'%params['train_emb']+
            #'_subsample%d'%params['subsample']+
            '_seed%d'%params['seed']+'_use-feat%d'%params['use_feat']+
            #'_traincut%.1f'%params['train_cut']+'_gf%s'%params['gating_fn']+
            '_corefdim%d'%params['coref_dim']+'/')
    main(save_path, params)
