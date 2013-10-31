'''
Created on Oct 9, 2013

@author: taochen, yiping
'''

class Dictionary:
    '''
    
    Attributes:
        docs: A list of docs where each item is a list of words.
        id_word: dict<id: word>
        word_id: dict<word: id>
        
    '''

    def __init__(self):
        self.id_word = dict()
        self.word_id = dict()
    
    def load_file(self, file_path):
        """Load the documents into a list of docs.
        Args:
            file_path: The file path of documents.
        Returns:
            self.docs.
        """
        f = open(file_path, "r")
        sentences = f.readlines()
 
        word_count = 0

        for sentence in sentence:  
            for word in sentence.strip().split(" "):
                if not (word in self.word_id):  #word not in dictionary
                    word_id[word] = word_count
                    word_count += 1

        #self.doc = [[self.word_id[word] for word in sentence.strip().split(" ")] for sentence in sentences]

    '''
    def process_word(self, word):
        """Stem word and filter out non-meaningful word (e.g., stop words).
        Consider use: https://pypi.python.org/pypi/stemming/1.0 or
                      http://text-processing.com/demo/stem/
     
        Args:
            word: the word to be processed
        Returns:
            A bool value indicate whether keep this word or not, and the processed word.
        """
        
    def cut_low_freq(self, docs, min_freq=1):
        """Filter out rare words that occur less than min_freq."""
    
    def word_to_id(self, word):
        """Return the id of the query word."""
    
    def id_to_word(self, word_id):
        """Return the word of the query id."""
        
    def doc_to_ids(self, doc):
    '''    
#end of class dictionary           
