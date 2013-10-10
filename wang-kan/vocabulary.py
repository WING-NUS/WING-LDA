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
        self.docsList = list()
        self.wordList = list()
        self.word_id_Dict= dict()
    # end of init ()

    def loadfile (self,filepath):
        f = open(filepath,'r')
        for line in f:
            docStr=""
            if not line:                            # where a doc ends
                self.docsList.append(docStr.strip().split())
            else:
                docStr+=line
        # end for
        f.close()
        return self.docsList
    # end of loadfile


    def word_to_id(self,word):
        if word not in self.word_id_Dict:
            wordID=len(self.wordList)
            self.word_id_Dict[word]=wordID
            self.wordList.append(word);
        else:
            wordID= self.word_id_Dict[word]

        return wordID
    # end of word_to_id


    def encode_doc_with_wordID(self,doc):

        idList = list()
        for word in doc:
            wordID = self.word_to_id(self,word)
            idList.append(wordID)
        # end of for
        return idList

    def process_file(self,filepath):
        self.loadfile(self,filepath)

        for doc in self.docsList:
            idList = self.encode_doc_with_wordID(doc)
            doc














# end of Vocabulary class