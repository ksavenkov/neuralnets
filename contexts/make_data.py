from optparse import OptionParser
import nltk
import numpy
import scipy.io
import codecs

verbose = 1    

def status(s):
    print s

def main():
    parser = OptionParser()

    parser.add_option("-f", "--file", dest="filename",
                  help="write report to FILE", metavar="FILE")
    parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

    (options, args) = parser.parse_args()
    filename = options.filename
    verbose = options.verbose

    print 'Processing %s ' % options.filename

    file = codecs.open(options.filename, encoding='iso-8859-1')
    raw = file.read()
    status('\tread %s symbols' % len(raw))
    tokens = nltk.wordpunct_tokenize(raw)
    status('\tgot %s tokens' % len(tokens))
    text = nltk.Text(tokens)

    words = [w.lower() for w in text]
    vocab = sorted(set(words))
    vocab_indexes = dict([(vocab[n],n) for n in range(len(vocab))])
    status('\tvocabulary size %s' % len(vocab))
    status('\tbuilding offsets...')
    text_offsets = [vocab_indexes[w] for w in words]
    status('\tdone')

    ngram_size = 4
    ngrams_number = 1 + len(text_offsets) - ngram_size;
    assert(ngrams_number > 0)
    test_size = int(ngrams_number * 0.1)
    validation_size = int(ngrams_number * 0.1)
    train_size = ngrams_number - test_size - validation_size
    status('Training set size: %s' % train_size)
    status('Test set size: %s' % test_size)
    status('Validation set size: %s' % validation_size)

    assert(test_size > 0 and validation_size > 0 and train_size > 0)
    assert(test_size + train_size + validation_size == ngrams_number)

    status('\tbuilding datasets...')
    
    train_data = [text_offsets[n:n+ngram_size] for n in range(train_size - ngram_size)]
    test_data = [text_offsets[n:n+ngram_size] for n in range(train_size, train_size + test_size - ngram_size)]
    validation_data = [text_offsets[n:n+ngram_size] for n in range(train_size + test_size, 
                                                       train_size + test_size + validation_size - ngram_size)]
    status('\tdone')

    datafile = 'sample.mat'
    print('Saving data to %s' % datafile)

    scipy.io.savemat(datafile, 
        mdict = {'vocab': vocab, 
                 'trainData': numpy.asmatrix(train_data), 
                 'testData': numpy.asmatrix(test_data), 
                 'validData': numpy.asmatrix(validation_data)})
main()
