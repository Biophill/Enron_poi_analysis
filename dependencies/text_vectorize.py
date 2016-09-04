#!/usr/bin/python

import os
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split


def main():
	email_dict = {"LAY KENNETH L":{"email":
					["kenneth_lay@enron.net",    
					"kenneth_lay@enron.com",
					"klay.enron@enron.com",
					"kenneth.lay@enron.com", 
					"klay@enron.com",
					"layk@enron.com",
					"chairman.ken@enron.com"],
					"poi": True}}
	email_list, poi_list = email_open(email_dict)
	words_list = email_process(email_list)
	features_train, features_test, labels_train, labels_test = email_vectorizer(
															words_list, poi_list)

'''
After testing, it was observed that the email received by poi are less 
informative. Persons of interest being in position of power, it makes 
sense that they are CCed to many important emails. This has the adverse
effect of reducing the specific patterns associated with those emails 
since they are send by many different people. Therefore it was decided
to only account for the sent email (code above) and the received emails
code was removed.

The counter also restrict the total number of email analysed for each
individuals. This counter restrict memory use and make it runable on a
single computer. Also, it has the effect of not over reprensenting the
email of some individuals. Indeed, it is possible to see that some
people sent more then 10 times the number of emails compared to their
colleagues. This could bias the classifier toward only accounting for
the writting patterns of these individuals, but this restriction minimise
this potential effect.
'''
															
def email_open(email_dict):
	complete_from_list = []
	complete_poi_list = []
	for key in email_dict:
		email_list = email_dict[key]["email"]
		counter = 0
		for email in email_list:
			from_mail =  "emails_by_address\\from_" + email + ".txt"
			path = os.path.dirname(os.path.abspath(__file__))
			from_path_file = os.path.join(path, from_mail)
			try:
				open_from = open(from_path_file, "r")
				for from_path in open_from:
					#Restriction to make it runnable on a single computer.
					counter += 1
					if counter < 200:
						complete_from_list.append(from_path.rstrip("\.\n"))
						complete_poi_list.append(email_dict[key]["poi"])
				open_from.close()
			except:
				None
	return complete_from_list, complete_poi_list
	
def email_process(email_list):
	words = []
	poi = []
	for email in email_list:
		path = "/".join(email.split("/")[1:])
		path = os.path.join('..', path)
		email = open(path, "r")
		### use parseOutText to extract the text from the opened email
		parsed_text = str(parseOutText(email))
		### The script was ran 10 times with random seeds and every signature 
		### words identified with a feature importance over 0.05 were put in
		### this signature list to be removed. Other signatures may be remaining,
		### but they are less important and are potentially diluted in the
		### large dataset.
		signatures = ["kevin", "steve", "hannonenron", "communicationsenron", "skeannsf",
					  "wes", "colwellhouectect", "colwel", "lavorato", "jforney", "ddelainnsf",
					  "jmf", "delaineyhouect", "forney", "delainey", "david", "tim"]
		for signature in signatures:
			parsed_text = parsed_text.replace(signature, "")
		words.append(parsed_text)
	print "Emails processed\n"
	return words

'''
Vectorize the words from the Email dataset and classify them using poi and 
non-poi labels. This uses a splitting method to create training and testing
set which will be the same length. The vectorized features and the labels
are returned.
'''
	
def email_vectorizer(word_list, indiv_list):
	features_train, features_test, labels_train, labels_test = train_test_split(
											word_list, indiv_list, test_size=0.5)
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
								stop_words='english')
	features_train = vectorizer.fit_transform(features_train).toarray()
	features_test  = vectorizer.transform(features_test).toarray()
	print "Emails Vectorized\n"
	##This block was used to identify signature words
	# from sklearn import tree
	# clf = tree.DecisionTreeClassifier()
	# clf.fit(features_train, labels_train)
	# print clf.score(features_test,labels_test)
	# for i, j in enumerate(clf.feature_importances_):
		# if j > 0.05:
			# print vectorizer.get_feature_names()[i], j
	return features_train, features_test, labels_train, labels_test

if __name__ == "__main__":
	main()

