#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Constants import Constants
import os

class Vocabulary:
	def __init__(self):
		self.corpus	= list()	# List of docs whereby each doc is a list of words (*not* word IDs).
		self.vocas	= list()	# ID to word (list).
		self.vocas_id	= dict()	# Word to ID (dictionary).
		self.docfreq	= list()	# ID to document frequency (list).
		print ("Hello I am here")
	# End of __init__(...).

	def load_file(self, file_location):
		print ("Hello I am here")
                os.chdir(file_location)
		for i in range(1):
			f = open(i+".txt", 'r')
			for line in f:
				doc = line.strip().split()	# We expect the line to be whitespace-separated.
				doc = filter(None, doc)		# Remove any empty String from a list of Strings.
				doc = map(str.strip, doc)	# Strip each word in the document.
				if len(doc) > 0:
					self.corpus.append(doc)
				# End of if statement.
			# End of for loop.
			f.close()
		print ("Hello I am here")
		print (self.corpus)
		return self.corpus
	# End of load_file(...).