import glob, time

MAX_INT = 9999999

# maximum per word penalty in word-dist model
MAX_PENALTY = 8

def predict_answer(fname):
    """ Predict the answer of a given document-query pair
    Args:
        fname: the *.question file
    Returns:
        0/1 indicators of the correctness of predictions made by dist/freq/freq_excl models
    """

    # Load document, query and answer from file
    fp = open(fname)
    fp.readline()
    fp.readline()
    document = fp.readline().split() # document d
    fp.readline()
    query = fp.readline().split() # query q
    fp.readline()
    ans = fp.readline().strip()
    fp.close()

    # Get the set of candidate answers
    entities = [w for w in document if w.startswith('@entity')]
    candidates = set(entities)

    # word_offsets[w] is a list of w's locations in d
    word_offsets = {w: [] for w in document}
    for i, w in enumerate(document):
        word_offsets[w].append(i)

    # Location of @placeholder in q
    placeholder_offset = query.index('@placeholder')

    # Compute the alignment score for each candidate answer
    best_score = MAX_INT
    best_c = -1
    for c in candidates:
        score = MAX_INT
        # Enumerate over all possible anchors to place @placeholder in d
        for x in word_offsets[c]:
            # Score of c when @placeholder is aligned with location x in d
            score_given_x = 0
            for i, w in enumerate(query):
                if w in word_offsets:
                    # Distance of each word w in q to its nearest neighbor in d
                    min_dist = MAX_INT
                    for y in word_offsets[w]:
                        min_dist = min(min_dist, abs(x + i - placeholder_offset - y))
                    score_given_x += min(min_dist, MAX_PENALTY)
            # As each candidate can appear in multiple locations in d,
            # we only pick the location that leads to minimum distance
            score = min(score, score_given_x)

        if score < best_score:
            best_score = score
            best_c = c

    # Prediction by word-distance model
    ans_dist = best_c

    # Prediction by max-frequency
    ans_freq = max(entities, key=entities.count)

    # Prediction by max-frequency (excluding entities appeared in q)
    entities_excl = [w for w in entities if w not in query]
    ans_freq_excl = max(entities_excl, key=entities_excl.count)

    return map(lambda x:x == ans, (ans_dist, ans_freq, ans_freq_excl))

if __name__ == '__main__':

    start = time.time()

    fnames = glob.glob("cnn/questions/validation/*.question")

    n, hits_dist, hits_freq, hits_freq_excl = 0, 0., 0., 0.
    for fname in fnames:
        indicator = predict_answer(fname)
        hits_dist += indicator[0]
        hits_freq += indicator[1]
        hits_freq_excl += indicator[2]
        n += 1
        # print out the accurarcy of the three methods
        print "%5d dist=%.4f freq=%.4f freq_excl=%.4f" % (
                n, hits_dist/n, hits_freq/n, hits_freq_excl/n)
        
    print "elapse %f secs" % (time.time() - start)

