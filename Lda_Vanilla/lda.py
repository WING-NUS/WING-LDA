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


class LDA:
    """Latent Dirichlet Allocation Topic Modelling library

    Args:
            Word vectors \w;
        TODO - hyperparameters \alpha, \beta; topic number

    Returns:
            Topic associations \z; multinominal parameters \Phi and \Theta;
            hyperparameter estimates \alpha, \beta

    1) Create a LDA, call LDA.init
    Necessary!

    2) Do inference per round, call LDA.inference
    Necessary!

    3) Calculate perplexity, call LDA.perplexity
    Optional - v2
    """

    def __init__(self, docs, V):
        self.K = 2             # number of topics
        self.alpha = 0.01    # should have been a vector
        self.beta = 0.01        # should have been a vector
        self.docs = docs
        self.M = len(self.docs)

        # Topics of words of documents.
        self.z_m_n = list()
        # Word count of each document and topic.
        self.n_m_k = numpy.zeros(self.M, self.K)
        # Word count of each document
        self.n_m = numpy.zeros(self.M)
        # Word count of each topic and vocabulary.
        self.n_k_t = numpy.zeros((self.K, V))
        # Word count of each topic.
        self.n_k = numpy.zeros(self.K)

        for m, doc in enumerate(docs):
            # Word count for the current document
            self.n_m[m] = len(doc)
            # Sampling a Multinomial Distribution for the current document
            doc_topic_dist = numpy.random.Dirichlet([self.alpha] * self.K)
            z_n = numpy.zeros(len(doc))
            for n, word in enumerate(doc):
                # Sampling a topic from the sampled Multinomial Distribution
                z = numpy.random.multinomial(1, doc_topic_dist).argmax()
                z_n[n] = z
                # increment document-topic count sum_m_z += 1
                self.n_m_k[m, z] += 1
                # increment topic-term count sum_z_t += 1
                self.n_k_t[z, word] += 1
                # increment topic-term sum sum_z += 1
                self.n_k[z] += 1
            self.z_m_n.append(z_n)

    def inference(self):
        """
        Gibbs Sampling over burn-in period and sampling period

        This procedure should be run multiple times until reaching a
        fixed number of iterations or other terminating conditions
        (e.g., perplexity)
        """

        for m, doc in enumerate(self.docs):
            for n, t in enumerate(doc):
                # Topic assigned to the current word
                zmn = self.z_m_n[m, n]
                # Decrement count from the document-topic matrix
                self.n_m_k[m, zmn] -= 1
                # Decrement count for the topic vocab matrix
                self.n_k_t[zmn, t] -= 1
                # Decrement count for topic vector
                self.n_k[zmn] -= 1
                #self.n_m[m] -= 1
                # Calculating the probability ratios without the current word
                p_z = (self.n_k_t[:, t] + self.beta) * (self.n_m_k[m] +
                                                        self.alpha) / \
                      (self.n_k + self.beta)
                # Feeding these probabilities to the multinomial and sampling a
                # new topic (the prob. values are normalized as well)
                new_z = numpy.random.multinominal(1, p_z / p_z.sum()).argmax()
                # Updating all the variables
                self.z_m_n[m, n] = new_z
                self.n_m_k[m, new_z] += 1
                self.n_k_t[new_z, t] += 1
                self.n_k[new_z] += 1

    def perplexity(self):
        1                       # placeholder
        # end of method perplexity

# end of class LDA
