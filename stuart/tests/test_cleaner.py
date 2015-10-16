import unittest

from cleaner import *

class TestCleaner(unittest.TestCase):

    def test_clean_text(self):
        stemmer=SnowballStemmer('english')

        text="in a running    Stuart  breakthrough.  nope ,      "
        result=clean_text(text,stemmer=stemmer,remove_periods=False)
        expected="run Stuart breakthrough _period nope"
        self.assertEqual(result,expected)

    def test_tokenize(self):
        self.maxDiff=None
        text="bob Bob joe...  and joe, but, joe? Jimbo' gazoo - billius __EOS__     bill  "
        result=tokenize_text(text)
        expected="bob Bob joe _ellipses and joe _comma but _comma joe _question Jimbo gazoo _dash billius _eos bill".split(" ")
        self.assertEqual(result,expected)

if __name__=="__main__":
    unittest.main()
