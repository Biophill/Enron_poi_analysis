#!/usr/bin/python

import sys
import os
import pickle
sys.path.append("../tools/")
sys.path.append("../dependencies/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from outlier_detection import outlierFinder, outlierToTable
from poi_email_addresses import poiEmails
from text_vectorize import email_open, email_process, email_vectorizer
from tester import test_classifier
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_selection import SelectPercentile, SelectFpr
from sklearn import grid_search


'''
The main function was built to control the investigation. Every step of the 
investigation can be imported or exported using the different parameters of
the function.

	-The investigation_phase parameter print a list of important features of
	the dataset including size, numbers of NAs and total numbers of high
	observations for each individuals.
	
	-The text_classify parameter activate a full fitting of the Emails of the
	enron dataset.
	
	-The main_classify parameter activate an investigation of multiple
	classifiers and	multiple parameters using the financial data of the enron
	dataset.
	
	-The classifier_dump parameter take the results of the analysis and dump
	the top features classifier to pickle files that will be read by a testing
	program.
'''
def main(investigation_phase = False, text_classify = False,
				main_classify = False, classifier_dump = True):
### Load the dictionary containing the dataset
	with open("final_project_dataset.pkl", "r") as data_file:
		data_dict = pickle.load(data_file)
	if investigation_phase:
		print "Total potential data points (individuals): "	+ str(
				len(data_dict.keys()))
		print "Total potential features by data points: " + str(
				len(data_dict["METTS MARK"])-2)
		outlier_handling(data_dict)
	data_dict = outlier_removal(data_dict)
	data_dict = feature_creation(data_dict)
	pca, features, labels = feature_select(data_dict)
	classifiers = ["NB", "DecisionTree", "SupportVectorMachine", "Kmeans"]
	if text_classify:
		text_classifier(data_dict)
	if main_classify:	
		for classifier in classifiers:
			print "Classifying using PCA"
			classifier_tester(pca, labels, classifier)
			print "Classifying using selected features"
			classifier_tester(features, labels, classifier)
	if classifier_dump:
		features_list = ['poi', 'bonus', 'exercised_stock_options',
						'total_stock_value']
		clf = GaussianNB()
		dump_classifier_and_data(clf, data_dict, features_list)
	

'''
Function that identified the number of time a name is in the top 10%
of the data and its total number of NA. It also identifies the number
of NA in every features and separates it by poi and non_poi.

It prints two files with the outlier data.
  -outlier_array.out contains every time an individual had a value in
   the top 10% of the feature and a count of all NAs
  -feature_NA_array.out count the total number of NAs and there ratio
   between affected and non-affected for all the features.
'''
def outlier_handling(data_dict):
	outlier_dict, feature_dict = outlierFinder(data_dict)
	outlier_array = outlierToTable(outlier_dict)
	#These arrays are written to two files for manual inspection of outliers
	with open("outlier_array.out", "w") as f:
		for data_slice in outlier_array:
			f.write("\t".join(data_slice) + "\n")	
	with open("feature_NA_array.out", "w") as f:
		f.write("Features\tNA count\tNA ratio in non poi\tNA ratio in poi\n")
		for feature in feature_dict.keys():
			f.write(feature + "\t" + str(feature_dict[feature]["count"]) +
			"\t" + str(feature_dict[feature]["na_non_poi"]) +
			"\t" + str(feature_dict[feature]["na_poi"]) + "\n")
			
'''
This successfully identifies the TOTAL as an outlier. It can be found as it
is systematically the highest number in the financial data and has NA in
all email data.	LOCKHART EUGENE E also need to be removed since it only
contains NA's. While many other individuals have a high amount of NAs, they
can still be informative for some key features and will be kept. The
"Travel agency in the park", which is mostly NAs and have a name that
seem to point toward it	not being an individual will also be removed.

On the side of features, multiple things comes out. Only 4 people had
loan_advances, which could be a problem. Sometime, features with high 
NAs may overfit our classifier, due to a bias in the distribution of
the training (which could contain all the non NA values) and the test set
(which could contain only NA values for this feature). Therefore, it is 
important to follow the distribution of such features between set, or use
cross validations. We will use the second option and keep tabs on such 
features. This analysis, also raise another problem, there is a strong 
difference in the number of NA between poi and non poi. This could be an
interesting feature, however it is most likely simply a bias in data
acquisition, which should not be used to classify poi. We will keep these
features in the analysis, but final features should be assessed for 
this potential problem.
'''
def outlier_removal(data_dict):
	outlier = ["TOTAL", "LOCKHART EUGENE E", "THE TRAVEL AGENCY IN THE PARK"]
	try:
		for out in outlier:
			del data_dict[out]
	except:
		print "Outlier was not specified"
	return data_dict

	
'''
New features were created for testing. First a ratio feature that account for 
the proportion of emails sent and received by poi was created. Second a ratio
of the exercised vs restricted stock was also created. The rational for the
first is that some people simply have to send more emails and a direct use of
the number of Emails interacting with poi may be biased. The second variable
created was done to account for a potential change in the term of contract for
individuals that were aware that the company was going down.
'''
def feature_creation(data_dict):
	for key in data_dict:
		person = data_dict[key]
		
		is_na_from_to = person["from_poi_to_this_person"] == "NaN"
		is_na_to_message = person["to_messages"] == "NaN"
		if is_na_from_to or is_na_to_message:
			data_dict[key]["ratio_from_to_messages"] = "NaN"
		else:
			num_from_to = float(person["from_poi_to_this_person"])
			num_to_message = float(person["to_messages"])
			data_dict[key]["ratio_from_to_messages"] = num_from_to / num_to_message
			
		is_na_to_from = person["from_this_person_to_poi"] == "NaN"
		is_na_from_message = person["from_messages"] == "NaN"
		if is_na_to_from or is_na_from_message:
			data_dict[key]["ratio_to_from_messages"] = "NaN"
		else:
			num_to_from = float(person["from_this_person_to_poi"])
			num_from_message = float(person["from_messages"])
			data_dict[key]["ratio_to_from_messages"] = num_to_from / num_from_message
			
		is_na_exe_stock = person["exercised_stock_options"] == "NaN"
		is_na_rest_stock = person["restricted_stock"] == "NaN"
		if is_na_exe_stock or is_na_rest_stock:
			data_dict[key]["ratio_stock"] = "NaN"
		else:
			num_exe_stock = float(person["exercised_stock_options"])
			num_rest_stock = float(person["restricted_stock"])
			data_dict[key]["ratio_stock"] = num_exe_stock / num_rest_stock
	return data_dict
	
'''
This features created the PCA and select the top 10 percentile features
'''
def feature_select(data_dict):
	features_list = ['poi','salary', 'deferral_payments', 'total_payments',
					'loan_advances', 'bonus', 'restricted_stock_deferred',
					'deferred_income', 'total_stock_value', 'expenses',
					'exercised_stock_options', 'other', 'long_term_incentive',
					'restricted_stock', 'director_fees', 'to_messages',
					'from_poi_to_this_person', 'from_messages',
					'from_this_person_to_poi', 'shared_receipt_with_poi',
					'ratio_from_to_messages', 'ratio_to_from_messages',
					'ratio_stock']

	data = featureFormat(data_dict, features_list, sort_keys = True)
	pca = PCA(n_components = 2)
	pca.fit(data[:,1:])
	pca_features = pca.transform(data[:,1:])

	labels, features = targetFeatureSplit(data)
	selector = SelectPercentile(percentile=10)
	selector.fit(features, labels)
	selected_features = selector.transform(features)
	
	return pca_features, selected_features, labels

'''
Extract words from poi and vectorize them for word based classification.
'''
def text_classifier(data_dict):
	email_dict = poiEmails()
	for key in email_dict:
		email_dict[key] ={"email": email_dict[key],
						  "poi": True}
	for key in data_dict:
		if key in email_dict:
			continue
		else:
			email_dict[key] ={"email": [data_dict[key]["email_address"]],
							  "poi": data_dict[key]["poi"]}
						   
	for key in email_dict:
		try:
			for i in email_dict[key]["email"]:
				None
		except:
			print key 
	email_list, poi_list = email_open(email_dict)
	word_list = email_process(email_list)
	features_train, features_test, labels_train, labels_test = email_vectorizer(word_list, poi_list)
	
	clf = DecisionTreeClassifier()
	clf.fit(features_train, labels_train)
	predict = list(clf.predict(features_test))

	print "Number of POI emails in test set: " + str(predict.count(1.))
	print "Number of emails in test set: " + str(len(features_test))
	print "Number of real POI emails in test set: " + str(labels_test.count(1))
	print "Precision is: " + str(metrics.precision_score(labels_test, predict))
	print "Recall is: " + str(metrics.recall_score(labels_test,predict))
	
'''
Trying multiple classifiers and multiple parameters for classifying
poi based on financial and email data.
'''	
def classifier_handler(features, labels, classifier, new = True, scaling = False):
	if scaling:
		min_max_scaler = MinMaxScaler()
		features = min_max_scaler.fit_transform(features)
	if new:
		if classifier == "DecisionTree":
			parameters = {'min_samples_split': [2,5,10]}
			clf = grid_search.GridSearchCV(DecisionTreeClassifier(),
													parameters, cv = 10)
			clf.fit(features,labels)
			clf = clf.best_estimator_
		elif classifier == "SupportVectorMachine":
			parameters = {'C': [1,10,100,1000,10000], 
							'kernel': ('sigmoid', 'rbf')}
			clf = grid_search.GridSearchCV(SVC(), parameters, cv = 10)
			clf.fit(features,labels)
			clf = clf.best_estimator_
		elif classifier == "NB":
			clf = GaussianNB()
			clf.fit(features,labels)
		elif classifier == "Kmeans":
			clf = KMeans(n_clusters = 2)
			clf.fit(features,labels)
	else:
		clf = classifier
		clf.fit(features,labels)
	return clf
	
'''	 
Testing the classifier using a ShuffleSplit method with 20 permutations.
'''
def classifier_tester(features, labels, classifier):
	ss = ShuffleSplit(len(features), n_iter=20,
						test_size=0.10, random_state = 0)
	precision = []
	recall = []
	labels = np.array(labels)
	new = True
	print "Fitting %s classifier" % classifier
	for train_index, test_index in ss:
		feature_train = features[train_index]
		labels_train = labels[train_index]
		feature_test = features[test_index]
		labels_test = labels[test_index]
		is_SVM = classifier == "SupportVectorMachine"
		is_Kmeans = classifier == "Kmeans"
		if new and is_SVM or is_Kmeans:
			clf = classifier_handler(feature_train, labels_train, classifier, scaling = True)
			new = False
		elif new:
			clf = classifier_handler(feature_train, labels_train, classifier)
			new = False
		elif not new and is_SVM or is_Kmeans:
			clf = classifier_handler(feature_train, labels_train, clf, new = False, scaling = True)
		else:
			clf = classifier_handler(feature_train, labels_train, clf, new = False)
		predict = list(clf.predict(feature_test))
		precision.append(metrics.precision_score(labels_test, predict))
		recall.append(metrics.recall_score(labels_test,predict))
	print "\tPrecision of the classifier " + str(sum(precision) / len(precision))
	print "\tRecall of the classifier " + str(sum(recall) / len(recall)) + "\n"

if __name__ == '__main__':
	main()
