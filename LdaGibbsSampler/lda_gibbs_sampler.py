#!/usr/bin/env python

class LdaGibbsSampler:
    """Performs Gibbs sampling for approximate inference on LDA

    Attributes:
        documents: int[][] entry represent word id in m(th) document
                   and n(th) word
        V: int, the size of the vocabulary
        K: int, # of topics
        iteration: int; maximum number of iterations of Gibbs sampling
        alpha: float. document--topic associations
        beta: float. topic--term associations
        int topic_assign[M][N]: topic assignment for each word
        int doc_topic[M][K]: number of words in document i assigned to topic j.
        int word_topic[V][K]: number of instances of word i (term?) assigned to topic j.
        int word_sum_topic[K]: total number of words assigned to topic j.
        int word_sum_doc[M]: total number of words in document i.
        
    """    

    
 
    def __init__(self, documents, V):
        """Initialize the sampler with input data

        Args:
            documents: int[][] entry represent word id in m(th) document
                       and n(th) word
            V: int, the size of the vocabulary
        """
        self.documents = documents
        self.V = V
    #end of __init__ 

    def configure(self, iteration):
        """Set parameters for tuning

        Args:
            iteration: int; maximum number of iterations of Gibbs sampling
            TODO: more parameters
        """
        self.iteration = iteration
    #end of configure

    def gibbs_sampling(self, K, alpha, beta):
        """Performs Gibbs Sampling 
           
           Select initial state ? Repeat a large number of times: 
               1. Select an element 
               2. Update conditional on other elements. 
        
        Args:
            K: int. # of topics
            alpha: float. symmetric prior parameter on document--topic associations
            beta: symmetric prior parameter on topic--term associations
        """
        self.K = K
        self.alpha = alpha
        self.beta = beta

        #initialize the state of the Markov chain
        initialize_state()

        for i in range(iteration):
            for m in range(len(z)):  #for each document
                for n in range(len(z[m])):  #for each word
                    #perform full conditional inference
                    topic = sample_full_conditional(m,n)
                    z[m][n] = topic

    #end of gibbs_sampling

    def initialize_state(self):
        """ Initialisation: assignment topics to words, increment counts
           
        """
        print "initialize_state"
    #end of initialize_state

    

#end of LdaGibbsSampler class
