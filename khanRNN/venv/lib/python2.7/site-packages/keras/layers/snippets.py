from __future__ import print_function
# class VectorDot(Layer):
#     '''
#         Take dot production between two vectors.
#         Ingest input of shape (nb_samples, 2, dims)
#         and return input of shape (nb_samples, 1)
#     '''
#     def __init__(self, input_dim):
#         self.input_dim = input_dim
#         self.input = T.tensor3()
#         self.params = []

#     def output(self, train):
#         X = self.get_input(train)
#         return T.dot(X[:, 0, :], X[:, 1, :])


test_relationships = [
    [("man", "woman"), ["king", "developer"]], 
    [("paris", "france"), ["berlin", "london", "tokyo"]], 
    [("microsoft", "ballmer"), ["tesla", "apple", "yc"]], 
]
for (w1, w2), ws in test_relationships:
    w1_e = embed_word(w1)
    w2_e = embed_word(w2)

    relationship = w2_e - w1_e

    print(w2, "is to", w1, "as:")
    for w in ws:
        w_e = embed_word(w)
        target = w_e + relationship
        closest = closest_to_point(target)
        print("...", closest[0], "is to", w)
    print('-')


words = ["hardware", "mobile", "javascript", "haskell", "apple", "bitcoin", "android", "ycombinator", "uber", "china",
"currency", "failure", "ai", "btc", "stockmarket", "hacker", "hack", "functional", "mongodb"
]



class CovarianceDense(Layer):
    def __init__(self, input_dim, output_dim, init='glorot_uniform', activation='linear', weights=None):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.matrix()
        self.W = self.init((self.input_dim ** 2, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.b]

        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train):
        X = self.get_input(train)
        #C = T.outer(X, X).flatten()
        #C = T.tensordot(X, X, axes=([], [])).flatten()
        #print(C.shape)

        def outer(x):
            return T.outer(x, x).flatten()

        C, _ = theano.scan(fn = outer,
                                sequences = X,
                                outputs_info = None)

        output = self.activation(T.dot(C, self.W) + self.b)
        return output