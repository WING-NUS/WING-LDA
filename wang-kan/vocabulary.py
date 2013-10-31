#!/usr/bin/python

'''
Created on 10 Oct, 2013

@author: Aobo Wang
'''


class Vocabulary:
    '''
	Attrbutes:
		docsList: A list of docs where each item is a list of words from a doc
		wordList: The full list of unique words. The indexs of words are kept in word_idDict
		word_idDict: The dictionary holds <word:id> entries
	'''

    def __init__(self):
        self.docs = list()
        self.vocabulary = list()
        self.word_id_dict = dict()

    # end of init ()

    def loadfile(self, filepath):
        f = open(filepath, 'r')
        doc = ""
        for line in f:
            if line.__eq__("\n"):                            # where a doc ends
                self.docs.append(doc.strip().split())
                doc = ""
            else:
                doc += line
        # end for
        f.close()
        return self.docs

    # end of loadfile


    def word_to_id(self, word):
        if word not in self.word_id_dict:
            word_id = len(self.vocabulary)
            self.word_id_dict[word] = word_id
            self.vocabulary.append(word)
        else:
            word_id = self.word_id_dict[word]

        return word_id

    # end of word_to_id


    def encode_doc_with_id(self, doc):
        id_list = list()
        for word in doc:
            wordID = self.word_to_id(word)
            id_list.append(wordID)
        # end of for
        return id_list



    def process_docs(self):
        docs=[self.encode_doc_with_id(doc) for doc in self.docs]
        return docs

    def __getitem__(self, index):
        return self.vocabulary[index]

    def get_vocab_size(self):
        return len(self.vocabulary)

# end of Vocabulary class