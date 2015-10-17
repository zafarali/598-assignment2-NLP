import unittest

from utilities import *
from cleaner import *

class TestUtilities(unittest.TestCase):

    def test_caps_count(self):
        previous_word="_period"
        word="Apples"
        self.assertEqual(is_caps_meaningful(previous_word,word),False)

        previous_word="then"
        word="Alex"
        self.assertEqual(is_caps_meaningful(previous_word,word),True)

        previous_word="_period"
        word="PGP"
        self.assertEqual(is_caps_meaningful(previous_word,word),True)

        previous_word="then"
        word="PGP"
        self.assertEqual(is_caps_meaningful(previous_word,word),True)

    def test_count(self):
        ngram=(("bob","_comma","_period"))
        self.assertEqual(2,count_tokens(ngram))

    def test_ngrams(self):
        words=("a","b","c","d")
        expected=[("a","b"),("b","c"),("c","d")]
        result=get_ngrams(words,2)
        self.assertEqual(result,expected)

        expected=[("a","b","c"),("b","c","d")]
        result=get_ngrams(words,3)
        self.assertEqual(result,expected)

    def test_cumulative(self):
        words=("a","b","c")
        expected=[("a","b"),("b","c"),("a",),("b",),("c",)]
        result=get_cumulative_ngrams(words,2)
        self.assertEqual(result,expected)

if __name__=="__main__":
    unittest.main()
