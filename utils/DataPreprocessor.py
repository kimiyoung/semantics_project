import numpy as np
import glob
import os
import sys
import cPickle as pkl

from itertools import izip
from config import MAX_WORD_LEN

MAX_COREF=100

SYMB_BEGIN = "@begin"
SYMB_END = "@end"

class Data:

    def __init__(self, dictionary, num_entities, training, validation, test, word_counts):
        self.dictionary = dictionary
        self.training = training
        self.validation = validation
        self.test = test
        self.vocab_size = len(dictionary[0])
        self.num_chars = len(dictionary[1])
        self.num_entities = num_entities
        self.inv_dictionary = {v:k for k,v in dictionary[0].items()}
        self.word_counts = word_counts
        self.max_num_cand = max(map(lambda x:len(x[3]), training+validation+test))

class DataPreprocessor:

    def preprocess_analysis(self, question_dir):
        """ for loading lambada analysis file """
        vocab_f = os.path.join(question_dir,"vocab_coref.txt")
        wc_f = os.path.join(question_dir,"wc_coref.npy")
        word_dictionary, char_dictionary, num_entities = \
                self.make_dictionary(question_dir, vocab_file=vocab_f)
        word_counts = self.word_count(question_dir, word_dictionary, wc_f)
        dictionary = (word_dictionary, char_dictionary)
        print "preparing test data ..."
        test = self.parse_all_files('../lambada/lambada-analysis/'+ "/test_coref", 
                dictionary, False)

        data = Data(dictionary, num_entities, [], [], test, word_counts)
        return data

    def preprocess(self, question_dir, relationfile, max_chains=MAX_COREF, 
            no_training_set=False, use_chars=True):
        """
        preprocess all data into a standalone Data object.
        the training set will be left out (to save debugging time) when no_training_set
        is True.
        """
        vocab_f = os.path.join(question_dir,"vocab_coref.txt")
        wc_f = os.path.join(question_dir,"wc_coref.npy")
        word_dictionary, char_dictionary, num_entities = \
                self.make_dictionary(question_dir, vocab_file=vocab_f)
        word_counts = self.word_count(question_dir, word_dictionary, wc_f)
        dictionary = (word_dictionary, char_dictionary)
        if no_training_set:
            training = None
        else:
            print "preparing training data ..."
            training = self.parse_all_files(question_dir + "/training_coref", 
                    relationfile, dictionary, use_chars, max_chains)
        print "preparing validation data ..."
        validation = self.parse_all_files(question_dir + "/validation_coref", 
                relationfile, dictionary, use_chars, max_chains)
        print "preparing test data ..."
        test = self.parse_all_files(question_dir + "/test_coref", 
                relationfile, dictionary, use_chars, max_chains)

        data = Data(dictionary, num_entities, training, validation, test, word_counts)
        return data

    def word_count(self, question_dir, word_dictionary, wc_file):

        if os.path.exists(wc_file):
            print "loading word counts from " + wc_file + " ..."
            word_counts = np.load(wc_file)
        else:
            print "no " + wc_file + " found, constructing the word counts ..."
            word_counts = np.zeros((len(word_dictionary),))

            if os.path.isfile(question_dir+'/training_coref.data'):
                f = open(question_dir+'/training_coref.data')
                for line in f:
                    for w in line.rstrip().split():
                        word_counts[word_dictionary[w]] += 1
                f.close()
                f = open(question_dir+'/validation_coref.data')
                for line in f:
                    for w in line.rstrip().split():
                        word_counts[word_dictionary[w]] += 1
                f.close()
                f = open(question_dir+'/test_coref.data')
                for line in f:
                    for w in line.rstrip().split():
                        word_counts[word_dictionary[w]] += 1
                f.close()
            else:
                fnames = []
                fnames += glob.glob(question_dir + "/test_coref" + "/*.question")
                fnames += glob.glob(question_dir + "/validation_coref" + "/*.question")
                fnames += glob.glob(question_dir + "/training_coref" + "/*.question")

                n = 0.
                for fname in fnames:
                    
                    fp = open(fname)
                    fp.readline()
                    fp.readline()
                    document = fp.readline().split()
                    fp.readline()
                    query = fp.readline().split()
                    fp.readline()
                    answer = fp.readline().split()
                    fp.close()

                    for w in document+query+answer:
                        word_counts[word_dictionary[w]] += 1
            np.save(wc_file, word_counts)

        return word_counts

    def make_dictionary(self, question_dir, vocab_file):

        if os.path.exists(vocab_file):
            print "loading vocabularies from " + vocab_file + " ..."
            vocabularies = map(lambda x:x.strip(), open(vocab_file).readlines())
        else:
            print "no " + vocab_file + " found, constructing the vocabulary list ..."

            vocab_set = set()

            if os.path.isfile(question_dir+'/training_coref.data'):
                print "found data file"
                f = open(question_dir+'/training_coref.data')
                for line in f:
                    vocab_set |= set(line.rstrip().split())
                f.close()
                f = open(question_dir+'/validation_coref.data')
                for line in f:
                    vocab_set |= set(line.rstrip().split())
                f.close()
                f = open(question_dir+'/test_coref.data')
                for line in f:
                    vocab_set |= set(line.rstrip().split())
                f.close()
            else:
                fnames = []
                fnames += glob.glob(question_dir + "/test_coref" + "/*.question")
                fnames += glob.glob(question_dir + "/validation_coref" + "/*.question")
                fnames += glob.glob(question_dir + "/training_coref" + "/*.question")

                n = 0.
                for fname in fnames:
                    
                    fp = open(fname)
                    fp.readline()
                    fp.readline()
                    document = fp.readline().split()
                    fp.readline()
                    query = fp.readline().split()
                    fp.readline()
                    answer = fp.readline().split()
                    fp.close()

                    vocab_set |= set(document) | set(query) | set(answer)

                    # show progress
                    n += 1
                    if n % 10000 == 0:
                        print '%3d%%' % int(100*n/len(fnames))

            entities = set(e for e in vocab_set if e.startswith('@entity'))

            # @placehoder, @begin and @end are included in the vocabulary list
            tokens = vocab_set.difference(entities)
            tokens.add(SYMB_BEGIN)
            tokens.add(SYMB_END)

            vocabularies = list(entities)+list(tokens)

            print "writing vocabularies to " + vocab_file + " ..."
            vocab_fp = open(vocab_file, "w")
            vocab_fp.write('\n'.join(vocabularies))
            vocab_fp.close()

        vocab_size = len(vocabularies)
        word_dictionary = dict(zip(vocabularies, range(vocab_size)))
        char_set = set([c for w in vocabularies for c in list(w)])
        char_set.add(' ')
        char_dictionary = dict(zip(list(char_set), range(len(char_set))))
        num_entities = len([v for v in vocabularies if v.startswith('@entity')])
        print "vocab_size = %d" % vocab_size
        print "num characters = %d" % len(char_set)
        print "%d anonymoused entities" % num_entities
        print "%d other tokens (including @placeholder, %s and %s)" % (
                vocab_size-num_entities, SYMB_BEGIN, SYMB_END)

        return word_dictionary, char_dictionary, num_entities

    @staticmethod
    def process_question(doc_raw, qry_raw, ans_raw, cand_raw, w_dict, 
            c_dict, use_chars, fname):
        # wrap the query with special symbols
        qry_raw.insert(0, SYMB_BEGIN)
        qry_raw.append(SYMB_END)
        try:
            cloze = qry_raw.index('@placeholder')
        except ValueError:
            try:
                at = qry_raw.index('@')
                print '@placeholder not found in ', fname, '. Fixing...'
                qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at+2])] + qry_raw[at+2:]
                cloze = qry_raw.index('@placeholder')
            except ValueError:
                cloze = -1

        # tokens/entities --> indexes
        doc_words = map(lambda w:w_dict[w], doc_raw)
        qry_words = map(lambda w:w_dict[w], qry_raw)
        if use_chars:
            doc_chars = map(lambda w:map(lambda c:c_dict.get(c,c_dict[' ']), 
                list(w)[:MAX_WORD_LEN]), doc_raw)
            qry_chars = map(lambda w:map(lambda c:c_dict.get(c,c_dict[' ']), 
                list(w)[:MAX_WORD_LEN]), qry_raw)
        else:
            doc_chars, qry_chars = [], []
        ans = map(lambda w:w_dict.get(w,0), ans_raw.split())
        cand = [map(lambda w:w_dict.get(w,0), c) for c in cand_raw]

        return doc_words, qry_words, ans, cand, doc_chars, qry_chars, cloze

    def parse_one_file(self, fname, dictionary, use_chars, max_chains):
        """
        parse a *.question file into tuple(document, query, answer, filename)
        """
        w_dict, c_dict = dictionary[0], dictionary[1]
        raw = open(fname).readlines()
        try:
            doc_raw = raw[2].split() # document
            qry_raw = raw[4].split() # query
            ans_raw = raw[6].strip().split(':')[0] # answer

            # candidates and corefs
            def _parse_coref(line):
                all_tokens = filter(lambda x:x, [mi for m in line.split() 
                    for mi in m.split('|')])
                return set([int(t.rsplit(':',1)[1])-1 for t in all_tokens])
            try:
                # corefs
                split = raw[8:].index('\n')
                cand_raw = map(lambda x:x.strip().split(':')[0].split(), 
                        raw[8:8+split]) # candidate answers
                coref = [_parse_coref(line.rstrip()) for line in raw[9+split:9+split+(max_chains-1)]]
            except ValueError:
                # no corefs
                cand_raw = map(lambda x:x.strip().split(':')[0].split(), 
                        raw[8:]) # candidate answers
                coref = []
            #if not any(aa in doc_raw for aa in ans_raw.split()):
            #    print "answer not in doc %s" % fname
            #    return None
        except IndexError:
            print "something wrong in ", fname
            return None

        return self.process_question(doc_raw, qry_raw, ans_raw, cand_raw, w_dict, c_dict,
                use_chars, fname) + (coref,)

    def parse_data_file(self, fdata, fcoref, dictionary, use_chars, stops, max_chains):
        """
        parse a *.data file into list of tuple(document, query, answer, filename)
        """
        w_dict, c_dict = dictionary[0], dictionary[1]
        questions = []
        with open(fdata) as data, open(fcoref) as coreffile:
            all_chains = pkl.load(coreffile)
            for ii, raw in enumerate(data):
                sents = raw.rstrip().rsplit(' . ', 1) # doc and query
                doc_raw = sents[0].split()+['.'] # document
                qry_tok = sents[1].rstrip().split()
                qry_raw, ans_raw =  qry_tok[:-1], qry_tok[-1] # query and answer
                cand_raw = filter(lambda x:x not in stops, set(doc_raw))
                if ans_raw not in cand_raw: continue
                cand_raw = [[cd] for cd in cand_raw]

                # candidates and corefs
                #def _parse_coref(line):
                #    all_tokens = filter(lambda x:x, [mi for m in line.split() 
                #            for mi in m.split('|')])
                #    return set([int(t.rsplit(':',1)[1])-1 for t in all_tokens])
                #all_coref = corefs.rstrip()
                #if all_coref: 
                #    coref = [_parse_coref(line) 
                #            for line in all_coref.split('\t')[:max_chains-1]]
                #else: coref = []
                coref = all_chains[ii][:max_chains-1]
                if not any(aa in doc_raw for aa in ans_raw.split()):
                    print "answer not in doc %s" % ii
                    continue

                questions.append(self.process_question(doc_raw, qry_raw, ans_raw, 
                    cand_raw, w_dict, c_dict, use_chars, ii) + (coref,ii))

        return questions

    def parse_all_files(self, directory, rfile, dictionary, use_chars, max_chains):
        """
        parse all files under the given directory into a list of questions,
        where each element is in the form of (document, query, answer, filename)
        """
        if os.path.isfile(directory+'.data'):
            print "found data file!"
            basedir = directory.rsplit('/',1)[0]
            stops = open(basedir+'/shortlist-stopwords.txt').read().splitlines()
            questions = self.parse_data_file(directory+'.data', 
                    directory+'.'+rfile,
                    dictionary, use_chars, stops, max_chains)
        else:
            all_files = glob.glob(directory + '/*.question')
            questions = []
            for f in all_files:
                qn = self.parse_one_file(f, dictionary, use_chars, max_chains)
                if qn is not None: questions.append(qn+(f,))
        return questions

    def gen_text_for_word2vec(self, question_dir, text_file):

            fnames = []
            fnames += glob.glob(question_dir + "/training/*.question")

            out = open(text_file, "w")

            for fname in fnames:
                
                fp = open(fname)
                fp.readline()
                fp.readline()
                document = fp.readline()
                fp.readline()
                query = fp.readline()
                fp.close()
                
                out.write(document.strip())
                out.write(" ")
                out.write(query.strip())

            out.close()

if __name__ == '__main__':
    dp = DataPreprocessor()
    dp.gen_text_for_word2vec("cnn/questions", "/tmp/cnn_questions.txt")

