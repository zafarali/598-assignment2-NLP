import unittest

from gimli import Gimli
from utilities import *

class TestGimli(unittest.TestCase):

    def test_process_data(self):
        gimli=Gimli("tests/test1.csv",silent=1,filter_threshold=0,validation_ratio=1)
        ngram=("b","c")
        result=gimli.ngram_count.get(ngram,0)
        expected=[2,1,0,0]
        self.assertEqual(expected,result)

        result=gimli.ngram_scores.get(ngram,0)
        expected=[2/3,1/3,0,0]
        self.assertEqual(expected,result)

    def test_prediction(self):
        gimli=Gimli("tests/test1.csv",filter_threshold=0,silent=1,validation_ratio=1)
        text="a b c a b c"
        result=gimli.get_prediction(text)
        expected=0
        self.assertEqual(expected,result)

        text="... bob joe jimbo c b a b"
        result=gimli.get_prediction(text)
        expected=1
        self.assertEqual(expected,result)

if __name__=="__main__":
    unittest.main()
