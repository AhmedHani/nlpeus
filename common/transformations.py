import functools
from utils.text_utils import Preprocessor


class TextTransformations(object):

    def __init__(self, *args):
        pass 
    
    def __new__(cls, *args):
        return [trasformation for trasformation in args]

    class CharPad(object):
        
        def __init__(self, size):
            pass
        
        def __new__(cls, size):
            return functools.partial(Preprocessor.char_based_pad, size=size)
    
    class CharTruncate(object):

        def __init__(self, size):
            pass

        def __new__(cls, size):
            return functools.partial(Preprocessor.chat_based_truncate, size=size)
    
    class WordPad(object):
        
        def __init__(self, size):
            pass

        def __new__(cls, size):
            return functools.partial(Preprocessor.word_based_pad, size)
    
    class WordTruncate(object):

        def __init__(self, size):
            pass

        def __new__(cls, size):
            return functools.partial(Preprocessor.word_based_truncate, size)
