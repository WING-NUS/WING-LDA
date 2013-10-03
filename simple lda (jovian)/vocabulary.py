#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

class Vocabulary:

	def __init__(self):
		self.corpus		= list()	# List of docs whereby each doc is a list of words (*not* word IDs).
		self.vocas		= list()	# ID to word (list).
		self.vocas_id	= dict()	# Word to ID (dictionary).
		self.docfreq	= list()	# ID to document frequency (list).
	# End of __init__(...).

	def load_file(self, file_location):
		f = open(file_location, 'r')
		for line in f:
			doc = line.strip().split()	# We expect the line to be whitespace-separated.
			doc = filter(None, doc)		# Remove any empty String from a list of Strings.
			doc = map(str.strip, doc)	# Strip each word in the document.
			if len(doc) > 0:
				self.corpus.append(doc)
			# End of if statement.
		# End of for loop.
		f.close()
		return self.corpus
	# End of load_file(...).

	def process_corpus(self, threshold=1):
		docs = [self.doc_to_ids(doc) for doc in self.corpus]
		docs = self.cut_low_freq(docs, threshold)
		return docs
	# End of process_corpus(...).

	def term_to_id(self, term):
		if term not in self.vocas_id:
			voca_id = len(self.vocas)
			self.vocas_id[term] = voca_id
			self.vocas.append(term)
			self.docfreq.append(0)
		else:
			voca_id = self.vocas_id[term]
		# End of if/else statement.
		return voca_id
	# End of term_to_id(...).

	def doc_to_ids(self, doc):
		list_of_ids	= list()
		words		= dict()
		for term in doc:
			id = self.term_to_id(term)
			list_of_ids.append(id)
			if not words.has_key(id):
				words[id] = 1
				self.docfreq[id] += 1
			# End of if statement.
		# End of for loop.
		if "close" in dir(doc):
			doc.close()
		# End of if statement.
		return list_of_ids
	# End of doc_to_ids(...).

	def cut_low_freq(self, docs, threshold=1):
		new_vocas		= list()	# ID to word (list).
		new_docfreq		= list()	# ID to document frequency (list).
		self.vocas_id	= dict()	# Word to ID (dictionary).
		conv_map		= dict()
		for id, term in enumerate(self.vocas):
			freq = self.docfreq[id]
			if freq > threshold:
				new_id = len(new_vocas)
				self.vocas_id[term] = new_id
				new_vocas.append(term)
				new_docfreq.append(freq)
				conv_map[id] = new_id
			# End of if statement.
		# End of for loop.
		self.vocas = new_vocas
		self.docfreq = new_docfreq
		def conv(doc):
			new_doc = list()
			for id in doc:
				if id in conv_map:
					new_doc.append(conv_map[id])
				# End of if statement.
			# End of for loop.
			return new_doc
		# End of conv(...).
		return [conv(doc) for doc in docs]
	# End of cut_low_freq(...).

	def __getitem__(self, v):
		# http://stackoverflow.com/questions/926574/why-does-defining-getitem-on-a-class-make-it-iterable-in-python
		return self.vocas[v]
	# End of __getitem__(...).

	def size(self):
		return len(self.vocas)
	# End of size(...).

# End of Vocabulary class.