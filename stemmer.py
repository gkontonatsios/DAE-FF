from nltk.stem.porter import *
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from console_progressbar import ProgressBar


class StemTokenizer(object):

    counter = 0

    def __init__(self, num_docs=None):
        self.num_docs = num_docs
        self.pb = ProgressBar(total=self.num_docs, suffix='Pre-processing documents', decimals=0, length=50, fill='█', zfill='-')

    def __call__(self, doc):

        """
        Takes as an input a document and returns a list of stems corresponding to the
        constituent words of that document

        Filters out:
            1) Words whose pos_tag is contained in the stop_pos_tags list
            2) Words that are contained in a stop word list, i.e. stopwords.words('english')
            3) Words whose stem is contained in a stop word list, i.e. stopwords.words('english')
        """

        # pos_tags to exclude
        stop_pos_tags = ['CD', 'RB', 'CC', 'DT']

        stemmer = PorterStemmer()
        stemmed_words = []
        # Tokenise document
        tokenised_text = word_tokenize(doc)
        # Pos tag document
        tagged_text = pos_tag(tokenised_text)

        for tag in tagged_text:
            word = tag[0]
            p_tag = tag[1]
            stemmed_word = stemmer.stem(word)

            '''
                Check whether:
                    1) length of word is greater than 1 and
                    2) and pos tag of word, i.e. p_tag, is not contained in the stop_pos_tags list and
                    3) word is not contained in the  stopwords.words('english')
                    4) stemmed_word is not contained in the  stopwords.words('english')
            '''
            if len(word) > 1 and p_tag not in stop_pos_tags \
                and word not in stopwords.words('english') \
                and stemmed_word not in stopwords.words('english'):
                stemmed_words.append(stemmed_word)

        StemTokenizer.counter += 1
        # print('Done processing pre-processing doc', StemTokenizer.counter)
        self.pb.print_progress_bar(StemTokenizer.counter)
        return stemmed_words


