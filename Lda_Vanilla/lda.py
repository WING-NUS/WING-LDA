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
        self.seed = 1
        self.docs = docs
        self.M = len(self.docs)

        # topic assignments for words in documents
        self.z_m_n = list()
        self.n_m_k = numpy.zeros(self.M, self.K)
        self.n_m = numpy.zeros(self.M)
        self.n_k_t = numpy.zeros((self.K, V))
        self.n_k = numpy.zeros(self.K)

        numpy.random.seed(self.seed)

        for m, doc in enumerate(docs):
            z_n = numpy.zeros(len(doc))
            for n, word in enumerate(doc):
                # Smart Init sample topic index z_m_n = K from Multinomial(1/K)
                z = numpy.random.randint(0, self.K)
                z_n[n] = z
                # increment document-topic count sum_m_z += 1
                self.n_m_k[m, z] += 1
                # increment document-topic sum sum_m += 1
                self.n_m[m] += 1
                # increment topic-term count sum_z_t += 1
                self.n_k_t[z, n] += 1
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

        for m in range(10):
            for n in range(3):
                zmn = self.z_m_n[m, n]
                self.n_m_k[m, zmn] -= 1
                self.n_m[m] -= 1
                self.n_k[zmn] -= 1
            # end for all words
        # end for all documents
        # end of method inference

    def perplexity(self):
        1                       # placeholder
        # end of method perplexity

# end of class LDA
