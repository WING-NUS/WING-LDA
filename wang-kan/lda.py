"""
Copyright 2013 by National University of Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy
import vocabulary

class LDA:
    """Latent Dirichlet Allocation Topic Modelling library

    Parameters
    ----------
            docs: vocabulary data structure
            	encodes the documents and their tokens
            V: integer
            	total unique vocabulary size
            alpha: float [0,1]
            	alpha hyperparameter
            beta: float [0,1]
            	beta hyperparameter
            K: integer
            	number of topics

    Returns
    -------
            Topic associations \z; multinominal parameters \Phi and \Theta; hyperparameter estimates \alpha, \beta

    3) Calculate perplexity, call LDA.perplexity
    Optional - v2
    """
    def __init__(self, docs, V, alpha = 0.01, beta = 0.01, K = 2):
        self.K = K             # number of topics
        self.alpha = alpha   # topic prior (used to influence \Theta) 
        self.beta = beta        # word prior (used to influence \Phi)
        self.docs = docs

        # topic assignments for words in documents
        self.z_m_n = list()     # to be populated by inner loop
        
        # zero all count variables: sum_m_k, sum_m, sum_k_t, sum_k
        self.sum_m_z = numpy.zeros((len(docs),self.K)) # document-topic count sum_m_k (dim MxK)
        # self.sum_m   = numpy.zeros(len(docs)) # document-topic sum sum_m (dim M), not actually used
        self.sum_z_t = numpy.zeros((self.K,V)) # topic-term count sum_z_t (dim KxV)
        self.sum_z   = numpy.zeros(self.K) # topic-term sum sum_z (dim K)
        self.word_count = 0
        # TODO: use the hyperparameters

        self.sum_m_z += self.alpha # add \alpha to each element of document-topic matrix to smooth counts
        # self.sum_m += self.alpha * self.K # add \alpha to document-topic sum, not used
        self.sum_z_t += self.beta # add \beta to each element of topic-term matrix to smooth counts
        self.sum_z += self.beta * V # add \alpha to document-topic sum, not used

        for m, doc in enumerate(docs): 
            # for all documents m \in [1,M] do
            print "Doc", m
            z_n = numpy.zeros(len(doc), dtype=numpy.int8) # (empty) topic assignments for doc m (use integers, not floats)
            for n, word in enumerate(doc):
                # for all words n \in [1,n_m] in doc_m do
                # TODO: Smart Init: sample topic index z_m_n = K from Multinomial(1/K)
                
                p_z = self.sum_z_t[:, word] * self.sum_m_z[m] / self.sum_z
                print "P_z:", p_z
                z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
#                z = numpy.random.randint(0, self.K) # choose a topic for each word
                z_n[n] = z

                print "  Word", n, ":", word, "; assigned to topic=", z
                
                self.sum_m_z[m,z] += 1 # increment document-topic count sum_m_z += 1
                # self.sum_m[m] += 1 # increment document-topic sum sum_m += 1, not used
                self.sum_z_t[z,word] += 1 # increment topic-term count sum_z_t += 1
                self.sum_z[z] += 1 # increment topic-term sum sum_z += 1
            # end for all words n
            self.word_count += len(doc)
            self.z_m_n.append(z_n)

        # end for all documents m

        # TODO:to be deleted later
        # print "Corpus has length ", self.word_count
        # for i in range(self.K):
        #    print "Topic", i, self.sum_z[i]
        # for i in range(len(self.docs)):
        #    print "Doc", i, self.sum_m[i], self.sum_m_z[i]
        # end of method __init 

    def inference(self):
        """
        Gibbs Sampling over burn-in period and sampling period

        This procedure should be run multiple times until reaching a
        fixed number of iterations or other terminating conditions
        (e.g., perplexity)
        """

#        print ("Inference:")
        # for all documents m \in [1,M] do
        for m in range(len(self.docs)):
            # for all words n \in [1,N_m] in document m do
            for n, word in enumerate(self.docs[m]):
                # // for the current assignment of k to a term t for word w_m_n
                z = self.z_m_n[m][n] # current z
#                print "Topic for word %s (%s in doc %s) is z = %s " % (word, n, m, z)

                # decrement counts and sums: 
                self.sum_m_z[m][z] -= 1 # document-topic count
                # self.sum_m[m] -= 1      # document-topic sum, not needed
                self.sum_z_t[z][word] -= 1 # topic-term count
                self.sum_z[z] -= 1         # topic-term sum
                ## BUG: can result in some 0 counts where the normalization of P_z later is NaN (undefined)

                ## // multinomial sampling acc. to Eq. 78 (decrements from previous step):

                # sample topic index k from p (z_i|z_not_i, w) Eqs. 74-8  in Heinrich TR
                # print (self.sum_z_t[:, word]) # 2nd d slice of orig 2D array [KxV] = 1D [K]
                # print (self.sum_m_z[m]) # row vector of orig 2D [MxK] = 1D [K]
                # print (self.sum_z) # 1D size [K]
                p_z = self.sum_z_t[:, word] * self.sum_m_z[m] / self.sum_z
                normalized_p_z = p_z / p_z.sum()
                draw = numpy.random.multinomial(1, normalized_p_z)
                new_z = draw.argmax()
                # // for the new assignment of z_m_n to the term t for word w_m_n:
                self.z_m_n[m][n] = new_z
#                print "P_z (Normalized):", p_z, "(", normalized_p_z, ") // Draw: ", draw, " // z : ", z, "=>", new_z

                # increment counts and sums:
                self.sum_m_z[m][new_z] += 1 # document-topic count
                # self.sum_m[m] += 1      # document topic sum, not needed
                self.sum_z_t[new_z,word] += 1 # topic-term count
                self.sum_z[new_z] += 1        # topic-term sum
            # end for all words 
        # end for all documents
        # end of method inference

    def output_word_topic_dist(self, voca):
        phi = self.sum_z_t / self.sum_z[:,numpy.newaxis] # normalize counts to probabilities
        for z in range(self.K):
            print "\n-- topic: %d" % z
            for t in numpy.argsort(-1 * phi[z]): # -1 for reverse sort
                print " %s: %f" % (voca[t], phi[z,t])
        # End of output_word_topic_dist

    def perplexity(self):
        # Eq. 96 in Heinrich (pg 29)
        # log perplexity (w_m|model) = \Sum_over_all_words log \Sum \phi_k_t x \theta_m_k
        phi = self.sum_z_t / self.sum_z[:,numpy.newaxis] # normalize counts to probabilities, dim KxV
        log_perplexity = 0.0
        
        Kalpha = self.K * self.alpha

        # loop over all words and accumulate
        for m, doc in enumerate(self.docs): 
            # for all documents m \in [1,M] do
            # print "Doc", m
            # Eq. 86 in Heinrich (pg 26)
            theta = self.sum_m_z[m] / (len(self.docs[m]) + Kalpha) # dim MxK 
            for n, word in enumerate(doc):
                log_perplexity += numpy.log(numpy.inner(phi[:,word],theta)) # scalar

        log_perplexity /= self.word_count
        print log_perplexity
        log_perplexity = numpy.exp(-log_perplexity);
        print log_perplexity
        return log_perplexity
        # end of method perplexity

# end of class LDA
