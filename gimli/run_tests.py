#this searches the tests directory for all unit tests and runs them

import glob, os, unittest

test_filenames = glob.glob('tests/*.py')
modules = [filename[:-3] if filename.endswith(".py") else filename for filename in test_filenames]
modules=[m.replace(os.sep,".") for m in modules]
suites = [unittest.defaultTestLoader.loadTestsFromName(m) for m in modules]
testSuite = unittest.TestSuite(suites)
text_runner = unittest.TextTestRunner().run(testSuite)
