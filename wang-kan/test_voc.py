def main():
        import vocabulary

        FILE_LOCATION = "test.txt"
        vocab = vocabulary.Vocabulary()
        docs = vocab.loadfile(FILE_LOCATION)
        docs = vocab.process_docs()

        print(docs)
        print(vocab.vocabulary)
        print(vocab.word_id_dict)
        print(vocab.get_vocab_size())


# End of main().

if __name__ == "__main__":
        main()
