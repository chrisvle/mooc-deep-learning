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
import csv

'''
   THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -i writing_generation_las.py

'''

data = pd.read_csv('training_set_rel3.tsv', sep='\t')

essay_set_1 = data[data.essay_set == 1]
good_essay_set_1 = essay_set_1[essay_set_1.domain1_score >= 10]
essays = good_essay_set_1.essay

SPELL_CHECK = True
if SPELL_CHECK:
  sc = enchant.Dict("en_US")

ms = {} #stands for memoized spelling

spell_counts = {}

#can put hard-coded spelling fixes here. otherwise, uses enchant's auto correct
#need to redo spell checking to later be able to insert multiple new words
ms['dr'] = 'dr'
ms['myspace'] = 'myspace'
ms['facebook'] = 'facebook'
ms['online'] = 'online'
ms['youtube'] = 'youtube'
ms['anyting'] = 'anything'
ms['ipod'] = 'ipod'
ms['lifes'] = 'lives'
ms['cyber'] = 'cyber'
ms['excersize'] = 'exercise'
ms['excersising'] = 'exercising'
ms['exersize'] = 'exercise'
ms['excersise'] = 'exercise'
ms['excersing'] = 'exercising'
ms['excersizing'] = 'exercising'
ms['excersicing'] = 'exercising'
ms['exersizing'] = 'exercising'
ms['excersice'] = 'exercise'
ms['excirsize'] = 'exercise'
ms['whatch'] = 'watch'
ms['aquire'] = 'acquire'
ms['wouldn'] = "wouldn't"
ms['lastest'] = 'last'
ms['asigned'] = 'assigned'
ms['learing'] = 'learning'
ms["isn'@caps3"] = "isn't"
ms["don'@caps1"] = "don't"
ms['alow'] = 'allow'
ms['reson'] = 'reason'
ms['sociatey'] = 'society'
ms['ther'] = 'there'
ms['insted'] = 'instead'
ms['belive'] = 'believe'
ms['skype'] = 'skype'
ms['dicide'] = 'decide'
ms['onlin'] = 'online'
ms['thier'] = 'their'
ms['daly'] = 'daily'
ms['cordnation'] = 'coordination'
ms['cordination'] = 'coordination'
ms['manualy'] = 'manually'
ms['mabe'] = 'maybe'
ms['gamming'] = 'gaming'
ms['hiting'] = 'hitting'
ms['teilor'] = 'tailor'
ms['evry'] = 'every'
ms['physcology'] = 'psychology'
ms['onemorelevel'] = 'one more level'
ms['dispite'] = 'despite'
ms['cafe'] = 'cafe'
ms['plaing'] = 'playing'
ms['dispite'] = 'despite'
ms['triathlete'] = 'triathlete'
ms['adiction'] = 'addiction'
ms['internet'] = 'internet'
ms['definatly'] = 'definitely'
ms['defenetly'] = 'definitely'
ms['definetley'] = 'definitely'
ms['finaly'] = 'finally'
ms['moniter'] = 'monitor'
ms['compters'] = 'computers'
ms['compter'] = 'computer'
ms['comitt'] = 'commit'
ms['slideshows'] = 'slide shows'
ms['wher'] = 'where'
ms['cousing'] = 'causing'
ms['bam'] = 'bam'
ms['geting'] = 'getting'
ms['relly'] = 'really'
ms["can'@caps1"] = "can't"
ms['ebay'] = 'ebay'
ms['adress'] = 'address'
ms['aquiring'] = 'acquiring'
ms['te'] = 'the'
ms['becaus'] = 'because'
ms['beggining'] = 'beginning'
ms['playes'] = 'plays'
ms['beter'] = 'better'
ms['msn'] = 'msn'
ms['chatrooms'] = 'chat rooms'
ms['chatroom'] = 'chat room'
ms['buissness'] = 'business'
ms['acess'] = 'access'
ms["youre"] = "you're"
ms['chating'] = 'chatting'
ms['convient'] = 'convenient'
ms['convinent'] = 'convenient'
ms['negitive'] = 'negative'
ms['personaly'] = 'personally'
ms['alows'] = 'allows'
ms['didn'] = "didn't"
ms['peole'] = 'people'
ms['oovoo'] = 'oovoo'
ms["don'@CAPS6"] = "don't"
ms["'"] = "'"
ms['happend'] = 'happened'
ms["im"] = "i'm"
ms['cyberbulling'] = 'cyber bullying'
ms['cyberbullying'] = 'cyber bullying'
ms["dont"] = "don't"
ms['apps'] = 'apps'
ms['ecofriendly'] = 'ecofriendly'
ms['wikipedia'] = 'wikipedia'
ms['wether'] = 'whether'
ms['orginiced'] = 'organized'
ms['recived'] = 'received'
ms['internent'] = 'internet'
ms['fatest'] = 'fattest'
ms['convinence'] = 'convenience'
ms['compture'] = 'computer'
ms['infact'] = 'infact'
ms['usualy'] = 'usually'
ms['simplier'] = 'simpler'
ms['itunes'] = 'itunes'
ms['cleary'] = 'clearly'
ms['afordable'] = 'affordable'
ms['foren'] = 'foreign'
ms['deffinately'] = 'definitely'
ms['writen'] = 'written'
ms['lik'] = 'like'
ms['thru'] = 'through'
ms['org'] = 'org'
ms['som'] = 'some'
ms['app'] = 'app'
ms['vaction'] = 'vacation'
ms['myspce'] = 'myspace'
ms['acount'] = 'account'
ms['goin'] = 'going'
ms['sexting'] = 'sexting'
ms['iphone'] = 'iphone'
ms["don'@caps4"] = "don't"
ms['tak'] = 'talk'
ms['forien'] = 'foreign'
ms["can'@caps3"] = "can't"
ms['easey'] = 'easy'
ms['likly'] = 'likely'
ms['acount'] = 'account'
ms['canbe'] = 'can be'
ms['powerpoint'] = 'powerpoint'
ms['texting'] = 'texting'
ms['catalouges'] = 'catalog'
ms['webkinz'] = 'webkinz'

#ending_punctuation = [',', '.', ':', ';']
#splitting_punctuation = ['/', '\\']
kaggle_anonymization_terms = ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'MONTH', 'EMAIL', 'NUM', 'CAPS', 'DR', 'CITY', 'STATE'] 

def convert_essays(essays):
  """
  """
  converted_essays = []
  for essay in essays:
    #print("finished an essay")
    final_essay = re.findall(r"[\w'@]+|[/.,!?;\"]", essay)
    true_final_essay = ['startofessay']
    for i in range(len(final_essay)):
      current_word = final_essay[i].lower()
      if current_word[0] == "@":
        for term in kaggle_anonymization_terms:
          if term.lower() in current_word:
            current_word = "kaggleanon"+ current_word[1:]
            current_word = ''.join(i for i in current_word if not i.isdigit())
            true_final_essay.append(current_word)
      elif SPELL_CHECK:
        if current_word not in ['/','.',',','!','?',';','\"']: 
        #print("spell checking")
          if current_word in ms:
            replacement = ms[current_word]
            split = replacement.split()
            for word in split:
              true_final_essay.append(word)
            if (current_word, ms[current_word]) in spell_counts:
              spell_counts[(current_word, ms[current_word])] += 1 
          elif not sc.check(current_word):
            suggest = sc.suggest(current_word)
            if suggest:
              #squashed = ''.join(suggest[0].split()) #Don't want to insert for now, just squashing
              replacement = suggest[0]
              #print("correcting " + current_word + " to " + squashed)
              split = replacement.split()
              for word in split:
                true_final_essay.append(word)
              if current_word != replacement.lower():
                ms[current_word] = replacement
                spell_counts[(current_word, ms[current_word])] = 1 #spell counts is only counting those spelling checks from enchant
          else:
            true_final_essay.append(current_word)
        else:
          true_final_essay.append(current_word)
        
    true_final_essay = [word.lower() for word in true_final_essay]
    true_final_essay.append('endofessay')
    converted_essays.append(true_final_essay)
  return converted_essays

#converted_essays = convert_essays(essays)
#with open('converted_essays_10_28.csv', 'wb') as fp:
#  a = csv.writer(fp, delimiter=',')
#  a.writerows(converted_essays)
converted_essays = []
with open('converted_essays_10_28.csv', 'rb') as csvfile:
  reader = csv.reader(csvfile)
  for row in reader:
    converted_essays.append(row)

reversed = {}
for key, value in spell_counts.items():
  if value in reversed:
    reversed[value].append(key)
  else:
    reversed[value] = [key]

words = set()
for essay in converted_essays:
  for word in essay:
    words.add(word)

word_indices = dict((w, i) for i, w in enumerate(words, 1)) #0 is reserved for padding
indices_word = dict((i, w) for i, w in enumerate(words, 1))

longest_essay_len = max(len(essay) for essay in converted_essays)

window_len = 40
step_size = 5
iteration_count = 100
output_iteration = 10
output_iteration_flag = True
output_dir = "modelsaves/"
generated_examples = {}
sequences = []
outputs = []
sequences_hold_out = []
outputs_hold_out = []

essay_count = 0
hold_out = True
for essay in converted_essays:
  #print("reading in essay")
  essay_len = len(essay)
  current_window = np.zeros(window_len)
  current_output = np.zeros(window_len)
  #Do one pass, in case window len >>> essay_len to begin with
#  i = 0
#  while i < essay_len and i < window_len:
#    current_window[i] = word_indices[essay[i]]
#    if i != essay_len - 1:
#      current_output[i] = word_indices[essay[i+1]]
#    else:
#      current_output[i] = word_indices['endofessay']
#    i+=1
#  sequences.append(current_window)
#  outputs.append(current_output)
  #print("finished first pass")
    
  #now that first pass is done, try to fill up any window that has > 50% of window filled
  for i in range(0, essay_len - window_len // 2, step_size):
    current_window = np.zeros(window_len)
    current_output = np.zeros(window_len)
    essay_i = i
    window_i = 0
    while essay_i < essay_len and window_i < window_len:
      current_window[window_i] = word_indices[essay[essay_i]]
      if essay_i != essay_len - 1:
        current_output[window_i] = word_indices[essay[essay_i+1]]
      else:
        current_output[window_i] = word_indices['endofessay']
      essay_i += 1
      window_i += 1
    if hold_out and essay_count % 20 == 0:
      sequences_hold_out.append(current_window)
      outputs_hold_out.append(current_output)
    else:
      sequences.append(current_window)
      outputs.append(current_output)
  essay_count += 1  
  #print("finished " + str(i) + " pass")

X = np.zeros((len(sequences), window_len))
y = np.zeros((len(outputs), window_len, len(words)), dtype=np.bool)
for i, sequence in enumerate(sequences):
  for t, word_index in enumerate(sequence):
    X[i, t] = word_index
    if outputs[i][t] > 0:
      y[i, t, outputs[i][t] - 1] = 1

X_test = np.zeros((len(sequences_hold_out), window_len))
y_test = np.zeros((len(outputs_hold_out), window_len, len(words)), dtype=np.bool)
for i, sequence in enumerate(sequences_hold_out):
  for t, word_index in enumerate(sequence):
    X_test[i, t] = word_index
    if outputs_hold_out[i][t] > 0:
      y_test[i, t, outputs_hold_out[i][t] - 1] = 1

#X_Train = sequence.pad_sequences(sequences, maxlen=maxlen)

HIDDEN_SIZE = 512

print('Building model')
model = Sequential()
model.add(Embedding(len(words)+1, 128, mask_zero = True))
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
total_iterations = 0
saved_generations = {}

# use execfile instead
execfile('testing_loop.py')

"""
for iteration in range(1, 20):
    total_iterations += 1
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
        print('----- Generating with seed: ' + new_str + ' and with window size: ' + str(window_len))
        #sys.stdout.write(word_sentence)

        for iteration in range(20):
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
"""
