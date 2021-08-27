HOW TO RUN PROGRAM

Method1:
	[CBOW word embedding training]
	1. place reviews_data.txt.gz in the same directory
	2. type command line: python3 main.py

	[Sentiment Analysis]
	3. place twitter-sentiment.csv, twitter-sentiment-testset.csv in the same directory
	4. type command line: python3 logistic_regression.py −c −mv=10000 −lr=0.1 −fn=myTest

Method2:
	[CBOW word embedding training]
	1. visit my colab: https://colab.research.google.com/drive/1TOhTDbQo_KZ1jfuH-XKxFYE7SQdaE8u3?usp=sharing
	2. upload reviews_data.txt.gz 
	3. run all code with gpu 
	4. download the word_emb_mat.txt file and word2vec.csv file, place them in the same directory with my code

	[Sentiment Analysis]
	5. go to the code directory, type command line: python3 logistic_regression.py −c −mv=10000 −lr=0.1 −fn=myTest


