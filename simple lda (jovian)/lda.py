#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

class LDA:
	
	def __init__(self, K, alpha, beta, docs, V, smartinit=True):
		self.K		= K		# The number of topics.
		self.alpha	= alpha	# Parameter of topics prior.
		self.beta	= beta	# Parameter of words prior.
		self.docs	= docs	# List of docs, in which each doc is a list of word IDs.
		self.V		= V		# The total number of unique words in all the docs.
		
		self.z_m_n	= list()									# Topics of words of documents.
		self.n_m_z	= numpy.zeros((len(self.docs), K)) + alpha	# Word count of each document and topic.
		self.n_z_t	= numpy.zeros((K, V)) + beta				# Word count of each topic and vocabulary.
		self.n_z	= numpy.zeros(K) + V * beta					# Word count of each topic.
		self.N		= 0											# Total number of (not unique) words in all documents.
		
		for m, doc in enumerate(docs):
			self.N += len(doc)
			z_n = list()
			for t in doc:
				if smartinit:
					p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z
					z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
				else:
					z = numpy.random.randint(0, K)
				# End of if/else statement.
				z_n.append(z)
				self.n_m_z[m, z] += 1
				self.n_z_t[z, t] += 1
				self.n_z[z]      += 1
			# End of for loop.
			self.z_m_n.append(numpy.array(z_n))
		# End of for loop.
	# End of __init__(...).

	def inference(self):
		for m, doc in enumerate(self.docs):
			z_n = self.z_m_n[m]
			n_m_z = self.n_m_z[m]
			for n, t in enumerate(doc):
				#----------------------------------------#
				# Discount for n-th word t with topic z. #
				#----------------------------------------#
				z = z_n[n]
				n_m_z[z]         -= 1
				self.n_z_t[z, t] -= 1
				self.n_z[z]      -= 1
				#-----------------------------#
				# Sampling topic new_z for t. #
				#-----------------------------#
				p_z = self.n_z_t[:, t] * n_m_z / self.n_z
				new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
				#------------------------------------------------#
				# Set z as the new topic and increment counters. #
				#------------------------------------------------#
				z_n[n] = new_z
				n_m_z[new_z]         += 1
				self.n_z_t[new_z, t] += 1
				self.n_z[new_z]      += 1
			# End of for loop.
		# End of for loop.
	# End of inference(...).

	def worddist(self):
		#------------------------------#
		# Get topic-word distribution. #
		#------------------------------#
		return self.n_z_t / self.n_z[:, numpy.newaxis]
	# End of worddist(...)

	def perplexity(self, docs=None):
		if docs is None:
			docs = self.docs
		# end of if statement.
		phi = self.worddist()
		log_per = 0
		N = 0
		Kalpha = self.K * self.alpha
		for m, doc in enumerate(docs):
			theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
			for w in doc: # Here, "w" represents an ID of a word in the doc.
				log_per -= numpy.log(numpy.inner(phi[:,w], theta))
			# End of for loop.
			N += len(doc)
		# End of for loop.
		return numpy.exp(log_per / N)
	# End of perplexity(...).

# End of LDA class.