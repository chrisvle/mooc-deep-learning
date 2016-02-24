# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import six.moves.cPickle
import gzip
import json
import os, sys
import re
from keras.preprocessing import text

'''
 Dataset of 5,845,908 HN comments
'''

html_tags = re.compile(r'<.*?>')
to_replace = [('&#x27;', "'")]
hex_tags = re.compile(r'&.*?;')

def clean_comment(comment):
    c = str(comment.encode("utf-8"))
    c = html_tags.sub(' ', c)
    for tag, char in to_replace:
        c = c.replace(tag, char)
    c = hex_tags.sub(' ', c)
    return c

def text_generator(path=os.path.expanduser("~/")+"HNCommentsAll.1perline.json"):
    f = open(path)
    for i, l in enumerate(f):
        comment_data = json.loads(l)
        comment_text = comment_data["comment_text"]
        comment_text = clean_comment(comment_text)
        if i % 10000 == 0:
            print(i)
        yield comment_text
    f.close()

def make_hn_dataset():
    '''
        Turn the raw text to sequence data. 
        Makes training somewhat faster, but
        takes quite a lot of RAM.
    '''
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(text_generator())
    sequences = tokenizer.texts_to_sequences(text_generator())
    word_index = tokenizer.word_index
    print('*'*50)
    print(len(word_index))
    six.moves.cPickle.dump(sequences, open('hn_comments_sequences.pkl', 'w'))
    six.moves.cPickle.dump(word_index, open('hn_comments_word_index.pkl', 'w'))


if __name__ == "__main__":
    make_hn_dataset()
