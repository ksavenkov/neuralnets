import scipy.io

def load_data(filename, num_batches):
    print 'Loading data from %s, splitting in %s mini-batches' % (filename,num_batches)
    data = {}
    scipy.io.loadmat(filename, mdict=data)
    numdims = data['trainData'].shape[1]
    D = numdims - 1
    M = int(data['trainData'].shape[0] / num_batches)
    train_input = data['trainData'][0: num_batches * M, 0:D].reshape(num_batches,M,D)
    train_target = data['trainData'][0: num_batches * M, D].reshape(num_batches,M,1)
    valid_input = data['validData'][:,0:D]
    valid_target = data['validData'][:,D]
    test_input = data['validData'][:,0:D]
    test_target = data['validData'][:,D]
    vocab = data['vocab']
    return (train_input, train_target, valid_input, valid_target, test_input, test_target, vocab)

def 
