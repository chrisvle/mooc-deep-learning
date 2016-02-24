from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
import numpy as np
import pandas as pd
import random
import sys
import re
import enchant
'''
   THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -i writing_generation.py

'''

data = pd.read_csv('training_set_rel3.tsv', sep='\t')

essay_set_1 = data[data.essay_set == 1]
good_essay_set_1 = essay_set_1[essay_set_1.domain1_score >= 10]
essays = good_essay_set_1.essay

SPELL_CHECK = False
if SPELL_CHECK:
  sc = enchant.Dict("en_US")


#ending_punctuation = [',', '.', ':', ';']
#splitting_punctuation = ['/', '\\']
kaggle_anonymization_terms = ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'MONTH', 'EMAIL', 'NUM', 'CAPS', 'DR', 'CITY', 'STATE'] 

def convert_essays(essays):
  """
  Splits essays as strings into lists of individual words. Takes apart punctuation, and re-words kaggle anonymization terms.
  Takes in sequence of essays as strings and 
  returns a list of lists whose elements are split words
  """
  converted_essays = []
  for essay in essays:
    print("finished an essay")
    final_essay = re.findall(r"[\w'@]+|[/.,!?;\"]", essay)
    for i in range(len(final_essay)):
      current_word = final_essay[i]
      if current_word[0] == "@":
        for term in kaggle_anonymization_terms:
          if term in current_word:
            current_word = "kaggleanon"+ current_word[1:]
            current_word = ''.join(i for i in current_word if not i.isdigit())
            final_essay[i] = current_word
      elif SPELL_CHECK:
        if current_word not in ['/','.',',','!','?',';','\"']: 
        #print("spell checking")
          if not sc.check(current_word):
            suggest = sc.suggest(current_word)
            if suggest:
              squashed = ''.join(suggest[0].split()) #Don't want to insert for now, just squashing
              #print("correcting " + current_word + " to " + squashed)
              final_essay[i] = squashed #replaces with most likely suggestion
        
    final_essay = [word.lower() for word in final_essay]
    final_essay.append('endofessay')
    converted_essays.append(final_essay)
  return converted_essays

converted_essays = convert_essays(essays)

words = set()
for essay in converted_essays:
  for word in essay:
    words.add(word)

word_indices = dict((w, i) for i, w in enumerate(words))
indices_word = dict((i, w) for i, w in enumerate(words))

window_len = 15
sequences = []
outputs = []
for essay in converted_essays:
  for i in range(0, len(essay) - window_len - 1):
    sequences.append(essay[i: i + window_len])
    outputs.append(essay[i+1: i + window_len + 1])

X = np.zeros((len(sequences), window_len))
y = np.zeros((len(outputs), window_len, len(words)), dtype=np.bool)
for i, sequence in enumerate(sequences):
  for t, word in enumerate(sequence):
    X[i, t] = word_indices[word]
    y[i, t, word_indices[outputs[i][t]]] = 1

#X_Train = sequence.pad_sequences(sequences, maxlen=maxlen)

HIDDEN_SIZE = 512

print('Building model')
model = Sequential()
model.add(Embedding(len(words), 128))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributedDense(len(words)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#model.fit(X,y,batch_size=256,nb_epoch=2)

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 40):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=256, nb_epoch=1)

    start_index = random.randint(0, len(X))

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = []
        sentence = X[start_index]
        word_sentence = [indices_word[wd] for wd in sentence]
        generated.extend(word_sentence)
        new_str = ' '.join(generated)
        print('----- Generating with seed: ' + new_str)
        #sys.stdout.write(word_sentence)

        for iteration in range(15):
            x = np.zeros((1, window_len))
            for t, char in enumerate(sentence):
                x[0, t] = sentence[t]
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds[window_len-1], diversity)
            next_word = indices_word[next_index]

            generated.append(next_word)
            sentence = np.append(sentence[1:], [next_index])

            #sys.stdout.write(next_word)
            #sys.stdout.flush()
        word_sentence = [indices_word[wd] for wd in sentence]
        print(' '.join(generated))
        print()
