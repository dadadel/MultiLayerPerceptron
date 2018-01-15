import os
from pnmimage import PnmImage

class LettersData(object):
    def __init__(self, data_folder="data/", filename_expected_data="list_expected_data.txt", car_filename_fmt="ext_ln{}_car{}.pgm"):
        self.filename_expected_data = filename_expected_data
        self.car_filename_fmt = car_filename_fmt
        self.data_folder = data_folder
        self.expected_data_list = []
        self.expected_letters = []
        self.letter_to_int = {}
        self.letters_classes = {}
        self.extract_vocab = []
        self.vocab = {}
        self._images = []
        self._expected_values = []
        self.nb_vocab = 0
        self.input_image_size = 0
        self.selection_vocab = {}
        self.selection_batch = []
        self.letter_to_vector = {}
        self.vector_to_letter = {}

    def _gen_ext_filenames_list(self):
        for ln in range(100):
            for car in range(100):
                n = self.data_folder + self.car_filename_fmt.format(ln, car)
                if os.path.exists(n) and n.replace(data_folder, '') in self.filenames_list:
                    yield(n)

    def _get_expected_letters_list(self):
        """Get expected letters list from expected data file.

        :return: list of tuples with the letter and the corresponding filename

        """
        data_list = []
        name = self.filename_expected_data
        with open(name) as f:
            lst = [(line.split(" ")[0], line.split(" ", 1)[1].replace('\n', '')) for line in f.readlines() if line.strip() != '']
            # list of tuples with (letter, filename)
            data_list = [(e[1], e[0]) for e in lst]
        return data_list

    def _extract_data(self):
        """Extract data.
        Create list of expected letters/filenames, base letters with info, classes.
        """
        self.expected_data_list = self._get_expected_letters_list()
        """list of tuples with letter and filename"""
        self.extract_vocab = sorted(set([l[0] for l in self.expected_data_list]))
        """list of data vocab letters sorted"""
        self.nb_vocab = len(self.extract_vocab)

        self.vocab = {}
        letters_list = [l[0] for l in self.expected_data_list]
        for i, c in enumerate(self.extract_vocab):
            vec = [0] * self.nb_vocab
            vec[i] = 1
            self.vocab[c] = {
                'index': i,
                'count': letters_list.count(c),
                'vector': vec
                }
        self.letter_to_int = {c: i for i, c in enumerate(self.extract_vocab)}
        """dict of letters with index of each"""
        self.letters_classes = {}
        """dict of letters with vector representation of each letter"""

        for letter, idx in self.letter_to_int.items():
            cls = [0] * self.nb_vocab
            cls[idx] = 1
            self.letters_classes[letter] = cls

    def get_vocab_with_min_count(self, min_count):
        """Get the vocab. A dictionary of letters with count, index and vector.
        The index is re-computed and also the vector to match the number of subelements
        """
        subvocab = {}
        i = 0
        for c, info in self.vocab.items():
            if info['count'] >= min_count:
                subvocab[c] = {
                    'count': info['count'],
                    'index': i,
                    'vector': None
                    }
                i += 1
        nb_vocab = len(subvocab)
        self.letter_to_vector = {}
        self.vector_to_letter = {}
        for c in subvocab:
            vec = [0] * nb_vocab
            vec[subvocab[c]['index']] = 1
            subvocab[c]['vector'] = vec
            self.letter_to_vector[c] = vec
        return subvocab

    def get_letter_of_vector(self, vector):
        """Get the letter corresponding to a given vector.

        :return: the found letter else None
        """
        ret = None
        for letter, vec in self.letter_to_vector.items():
            if vec == vector:
                ret = letter
                break
        return ret

    def get_batches(self, min_count=0, mini_batch_size=None):
        """Get the selection data based on min count of letters in the dataset

        :param min_count: minimal count of same letters to be added (default 0 for the whole dataset)
        :param mini_batch_size: size of a mini batch, if None the whole dataset size (default None)
            if whole size is not factor of the mini batch size then the last mini batch
            has a size < mini batch size
        :return: a list of mini batches (at least list of one)
        """
        if mini_batch_size is None:
            mini_batch_size = len(self.expected_data_list)
        self.selection_vocab = self.get_vocab_with_min_count(min_count)
        self.selection_batch = []
        X = []
        Y = []
        bsize = 0
        for letter, name in self.expected_data_list:
            path = '{}{}'.format(self.data_folder, name)
            if letter in self.selection_vocab:
                img = PnmImage()
                if img.load(path) == False:
                    print("ERROR: failed to open '{}'".format(path))
                image_size = img.get_size()
                bsize += 1
                X.append(img.get_data_bin())
                Y.append(self.selection_vocab[letter]['vector'])
                if bsize >= mini_batch_size:
                    self.selection_batch.append((X, Y))
                    X = []
                    Y = []
                    bsize = 0
        if bsize > 0:
            self.selection_batch.append((X, Y))
        return self.selection_batch

    def process(self):
        self._extract_data()

    def get_data_with_min_count(self, min_count):
        vocab = self.get_vocab_with_min_count(min_count)
        #todo
        return images, expected_values
 
    def get_vocab(self):
        """Get the vocab. A dictionary of letters with count, index and vector"""
        return self.vocab

    def get_classes(self):
        return self.letters_classes

    def get_class_element_size(self):
        return self.nb_vocab

    def get_input_image_size(self):
        return self.input_image_size