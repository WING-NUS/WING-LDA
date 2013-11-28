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
import sys

def main():
    """Main function for command line invocation
    """
    import lda

    vocab = vocabulary.Vocabulary()
    docs = vocab.loadfile(sys.argv[1])
    docs = vocab.process_docs()

    # set the random seed for replicability
    numpy.random.seed(3)

    iterations = 100

    lda = lda.LDA(docs, vocab.get_vocab_size())
    for i in range(iterations):
        lda.inference()
        lda.output_word_topic_dist(vocab)
        lda.perplexity()
        print "Iteration %s:" % i
# end main()

# execute class if called from command-line
if __name__ == "__main__":
    main()
