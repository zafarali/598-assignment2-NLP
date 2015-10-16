CSV Cleaner:
	python3 cleaner.py ml_dataset_train.csv
	python3 cleaner.py ml_dataset_test_out.csv

This produces 16 files named ml_dataset_train-XXXX.csv, then the other 16 test files. Instead of "ml_dataset_train.csv" you can type a full or relative path to any file. Just typing "ml_dataset_train.csv" works if that file is in the same folder as the Python script. The XXXX part describes how the data was cleaned. For example: train-0111.csv

0 remove_stops. Set to false, do not remove stops words like "at" and "the".
1 stem. Set to true, stem words like "running" into "run".
1 remove_tokens. Set to true - remove all tokens (comma, dash, question mark). Does not remove periods though.
1 remove_periods. Set to true - remove all periods.

It also does other nice things like replace "," with "_comma", removes crazy characters, fixes caps.

Gimli:
	python3 main.py train-1000.csv test-1000.csv --validate
	python3 main.py -h             (to see help)

Tests:
	python3 run_tests.py

Runs all my unit tests.
