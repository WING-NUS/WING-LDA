#!/usr/bin/env python
import numpy
from Constants import Constants
from Vocabulary import Vocabulary


class LDA:

	def __init__(self, K, alpha, beta, docs, V, smartinit=True):		
		self.docs	= docs	# List of docs, in which each doc is a list of word IDs.
		self.V		= V		# The total number of unique words in all the docs.
		
		self.z_m_n	= list()									# Topics of words of documents.
		self.n_m_z	= numpy.zeros((len(self.docs), K)) + alpha	# Word count of each document and topic.
		self.n_z_t	= numpy.zeros((K, V)) + beta				# Word count of each topic and vocabulary.
		self.n_z	= numpy.zeros(K) + V * beta					# Word count of each topic.
		self.N		= 0											# Total number of (not unique) words in all documents.

	def main():
	       voc			= Vocabulary()
	       corpus			= voc.load_file(Constants["FILE_LOCATION"])