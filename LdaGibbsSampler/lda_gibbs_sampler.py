#!/usr/bin/env python
import numpy

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
        int count_doc_topic[M][K]: number of words in document i assigned to topic j.
        int count_term_topic[V][K]: number of instances of word i (term?) assigned to topic j.
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

    def inference_gibbs_sampling(self, K, alpha, beta):
        """Performs Gibbs Sampling 
           
           Select initial state ? Repeat a large number of times: 
               1. Select an element 
               2. Update conditional on other elements. 
        
        elf.documents)rgs:
            K: int. # of topics
            alpha: float. symmetric prior parameter on document--topic associations
            beta: symmetric prior parameter on topic--term associations

        Refer to pseudo code in figure 9
        """
        #self.alpha = [alpha] * K
        #self.beta = [beta] * V
        #lines above disabled for idiot's version
        self.alpha = alpha
        self.beta = beta
        self.K = K
        #initialize the state of the Markov chain
        self.initialize_state()

        for i in range(self.iteration):
            for m in range(len(self.topic_assign)):  #for each document
                for n in range(len(self.topic_assign[m])):  #for each word
                    #decrement counts and sums
                    old_topic = self.topic_assign[m][n]
                    self.count_doc_topic[m][old_topic] -= 1
                    self.word_sum_topic[old_topic] -= 1
                    self.count_term_topic[self.documents[m][n]][old_topic] -= 1

                    #perform full conditional inference
                    new_topic = self.sample_full_conditional(m,n)
                    self.topic_assign[m][n] = new_topic

                    self.count_doc_topic[m][new_topic] += 1
                    self.word_sum_topic[new_topic] += 1
                    self.count_term_topic[self.documents[m][n]][new_topic] += 1
            
            #TODO: if converged and L sampling iterations since last read out then read out parameters        
    #end of gibbs_sampling

    def initialize_state(self):
        """ Initialisation: assignment topics to words, increment counts
           
        Args:
            K: number of topics
        """

        self.topic_assign = []
        #initialize all to zeros
        self.count_doc_topic = numpy.zeros((len(self.documents), self.K))
        self.word_sum_topic = numpy.zeros(self.K)
        self.count_term_topic = numpy.zeros((self.V, self.K))

        for m in range(len(self.documents)):  #for each document
            self.topic_assign.append([])
            for n in range(len(self.documents[m])):  #for each word
                # sample topic for the current word
                topic = numpy.random.multinomial(100,[1/self.K]*self.K).argmax()
                self.topic_assign[m].append(topic)
                #increment counts
                self.count_doc_topic[m][topic] += 1
	        self.word_sum_topic[topic] += 1
                try:
                    self.count_term_topic[self.documents[m][n]][topic] += 1
                except:
                    print self.documents[m][n]	

    #end of initialize_state
 
    def sample_full_conditional(self, m, n):
        """calculate the full conditional distribution for a word token with index i=(m,n)

        Args:
            m: m(th) document
            n: n(th) word

        Refer to formula (78) in paper
        """
        #pseudo_prob_topic_dist = (count_term_topic[self.documents[m][n]] + [self.beta]*self.K) / (word_sum_topic + [self.beta]*self.K) * (count_doc_topic[:, m] + self.alpha)
     
        #idoit's version
        pseudo_prob_topic_dist = []
        for k in range(self.K):
            prob = (self.count_term_topic[self.documents[m][n]][k] + self.beta) * (self.count_doc_topic[m][k] 
              + self.alpha) / (self.word_sum_topic[k] + self.V*self.beta)
            pseudo_prob_topic_dist.append(prob)
        #end for
        
        #multinomial sample a new topic for word[m][n]
        new_topic = numpy.random.multinomial(1, pseudo_prob_topic_dist).argmax()
        return new_topic

    #end of sample_full_conditional

    def get_theta(self):   
        """calculated theta based on document_topic and alpha
        refer to formula (82)
        """
        theta = numpy.zeros((len(self.documents), self.K))
        for i in range(len(self.count_doc_topic)):
            topic_sum = sum(self.count_doc_topic[i])
            theta[i] = [float(count)/topic_sum for count in self.count_doc_topic[i]]

        return theta
    #end of get_theta

    def get_phi(self):
        """calculated phi based on word_topic and beta
        refer to formula (81)
        """ 
        phi = numpy.zeros((self.V, self.K))
        for i in range(len(self.count_term_topic)):
            topic_sum = sum(self.count_term_topic[i])
            phi[i] = [float(count)/topic_sum for count in self.count_term_topic[i]]
       
        return phi
    #end of get_phi


#end of LdaGibbsSampler class
 
def main():
    documents = [[1,2,3,4,5,0,1],[5,4,5,6,5],[1,4,2,1,1]]  #pseudo document should call vocab instead
    lda = LdaGibbsSampler(documents, 7)
    lda.configure(500)  #500 iterations
    lda.inference_gibbs_sampling(10, 0.05, 100)
   
    theta = lda.get_theta()
    print "theta: "
    print theta
    phi = lda.get_phi()
    print "phi: "
    print phi

#end of main 

if __name__ == '__main__':
    main()
