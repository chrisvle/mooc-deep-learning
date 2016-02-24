import logging
import six.moves.cPickle as pickle

_logger = logging.getLogger("theano.gof.callcache")


class CallCache(object):
    def __init__(self, filename=None):
        self.filename = filename
        try:
            if filename is None:
                raise IOError('bad filename')  # just goes to except
            f = open(filename, 'r')
            self.cache = pickle.load(f)
            f.close()
        except IOError:
            self.cache = {}

    def persist(self, filename=None):
        if filename is None:
            filename = self.filename
        f = open(filename, 'w')
        pickle.dump(self.cache, f)
        f.close()

    def call(self, fn, args=(), key=None):
        if key is None:
            key = (fn, tuple(args))
        if key not in self.cache:
            _logger.debug('cache miss %i', len(self.cache))
            self.cache[key] = fn(*args)
        else:
            _logger.debug('cache hit %i', len(self.cache))
        return self.cache[key]

    def __del__(self):
        try:
            if self.filename:
                self.persist()
        except Exception as e:
            _logger.error('persist failed %s %s', self.filename, e)