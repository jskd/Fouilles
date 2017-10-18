import sklearn
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def read_dataset(filename):
    """ Reads the file at the given path that should contain one
    type and one text separated by a tab on each line, and returns
    pairs of type/text.
    Args:
        filename: a file path.
    Returns:
        a list of (type, text) tuples. Each type is either 0 or 1.
    """
    x = []
    y = []
    with open(filename, 'r') as f:
        for raw in f.readline():
            row= raw.strip('\n')
            tokens= row.split('\t')
            cleanup= "".join([l for l in tokens[1] if l not in string.punctuation])
            x.append(cleanup)
            y.append(0 if tokens[0] == "ham" else 1)
    return x,y
x,y= read_dataset("SMSSpamCollection")
print(x )
print(y)

def spams_count(pairs):
    """ Returns the number of spams from a list of (type, text) tuples.
    Args:
        pairs: a list of (type, text) tuples.
    Returns:
        an integer representing the number of spams.
    """

def transform_text(pairs):
    """ Transforms the pair data into a matrix X containing tf-idf values
    for the messages and a vector y containing 0s and 1s (for hams and
    spams respectively).
    Row i in X corresponds to the i-th element of y.
    Args:
        pairs: a list of (type, message) tuples.
    Returns:
        X: a sparse TF-IDF matrix where each row represents a message and
        each column represents a word.
        Y: a vector whose i-th element is 0 if the i-th message is a ham,
        else 1.
    """



