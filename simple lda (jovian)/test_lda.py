#!/usr/bin/python

import numpy

class FileOutput:
	def __init__(self, file):
		import datetime
		self.file = file + datetime.datetime.now().strftime('_%m%d_%H%M%S.txt')
	def out(self, st):
		with open(self.file, 'a') as f:
			print >>f,  st
# End of FileOutput class.

def lda_learning(f, LDA, smartinit, K, alpha, beta, docs, voca, iteration, plimit=1):
	import time
	t0 = time.time()
	lda = LDA(K, alpha, beta, docs, voca.size(), smartinit)
	pre_perp = lda.perplexity()
	f.out("alg=%s smart_init=%s initial perplexity=%f" % (LDA.__name__, smartinit, pre_perp))
	pc = 0
	for i in range(iteration):
		if i % 10==0:
			output_word_topic_dist(f, lda, voca)
		lda.inference()
		perp = lda.perplexity()
		f.out("-%d p=%f" % (i + 1, perp))
		if pre_perp is not None:
			if pre_perp < perp:
				pc += 1
				if pc >= plimit:
					output_word_topic_dist(f, lda, voca)
					pre_perp = None
			else:
				pc = 0
				pre_perp = perp
	output_word_topic_dist(f, lda, voca)
	t1 = time.time()
	f.out("time = %f\n" % (t1 - t0))
# End of lda_learning(...).

def output_word_topic_dist(f, lda, voca):
	phi = lda.worddist()
	for k in range(lda.K):
		f.out("\n-- topic: %d" % k)
		for w in numpy.argsort(-phi[k])[:20]:
			f.out("%s: %f" % (voca[w], phi[k,w]))
# End of output_word_topic_dist(...).

def main():
	import lda
	import vocabulary

	FILE_LOCATION	= 'test.txt'
	voca			= vocabulary.Vocabulary()
	corpus			= voca.load_file(FILE_LOCATION)
	docs			= voca.process_corpus(threshold=2)

	K				= 2
	alpha			= 0.01
	beta			= 0.01
	V				= voca.size()
	iteration		= 500

	f = FileOutput("lda_test")
	f.out("corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(docs), V, K, alpha, beta))

	lda_learning(f, lda.LDA, False, K, alpha, beta, docs, voca, iteration, 2)
	lda_learning(f, lda.LDA, True,  K, alpha, beta, docs, voca, iteration, 2)
# End of main().

if __name__ == "__main__":
	main()