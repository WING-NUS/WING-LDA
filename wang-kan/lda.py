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
            Topic associations \z; multinominal parameters \Phi and \Theta; hyperparameter estimates \alpha, \beta

    1) Create a LDA, call LDA.init
    Necessary!

    2) Do inference per round, call LDA.inference 
    Necessary!

    3) Calculate perplexity, call LDA.perplexity
    Optional - v2
    """
    
    def __init__(self, docs, V):
        """
        Initialization
        """
        self.K = 2             # number of topics
        self.alpha = 0.01    # topic prior (used to influence \Theta) 
        self.beta = 0.01        # word prior (used to influence \Phi)

        # zero all count variables, n_m_k n_m, n_k_t, n_k
        # for all documents m \in [1,M] do
        for m in range(10):
            # for all words n \in [1,n_m] in doc_m do
            for n in range(3):
                1
                # sample topic index z_m_n = K from Multinomial(1/K)
                # increment document-topic count n_m_k += 1
                # increment document-topic sum n_m += 1
                # increment topic-term count n_k_t += 1
                # increment topic-term count n_k += 1
            # end for all words n
        # end for all documents m
        # end of method __init 

    def inference():
        """
        Gibbs Sampling over burn-in period and sampling period

        This procedure should be run multiple times until reaching a
        fixed number of iterations or other terminating conditions
        (e.g., perplexity)
        """

        # for all documents m \in [1,M] do
        for m in range(10):
            # for all words n \in [1,N_m] in document m do
            for n in range(3):
                # // for the current assignment of k to a term t for word w_m_n
                # decrement counts and sums:
                n_m_k -= 1
                n_m -= 1
                n_k_t -= 1
                n_k -= 1
                # // multinomial sampling acc. to Eq. 78 (decrements from previous step):
                # sample topic index k from p (z_i|z_not_i, w)
                # // for the new assignment of z_m_n to the term t for word w_m_n:
                # increment counts and sums:
                n_m_k += 1
                n_m += 1
                n_k_t += 1
                n_k += 1
            # end for all words 
        # end for all documents
        # end of method inference

    def perplexity():
        1                       # placeholder
        # end of method perplexity

# end of class LDA
