#!/usr/bin/python

import pickle
import numpy as np
import math
from feature_format import featureFormat
import pickle

def main():
	with open("../final_project/final_project_dataset.pkl", "r") as data_file:
		data_dict = pickle.load(data_file)
	outlier_dict, feature_dict = outlierFinder(data_dict)
	outlier_array = outlierToTable(outlier_dict)
	with open("outlier_array.out", 'w') as f:
		for data_slice in outlier_array:
			f.write("\t".join(data_slice) + "\n")
	
'''
Will identify the top 10% of data for each features
and will print a list of the number of times that each
names are part of this category and which categories
have they been outliers in.
'''
	
def outlierFinder(data_dict):
	names = np.array([name for name in data_dict])
	outlier_dict = dict((name,{"count": 0,
							"categories": [],
							"nacount": 0,
							"nacategories": []}) for name in names)
	
	features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances',
				'bonus', 'restricted_stock_deferred', 'deferred_income',
				'total_stock_value', 'expenses', 'exercised_stock_options',
				'other', 'long_term_incentive', 'restricted_stock', 
				'director_fees', 'to_messages', 'from_poi_to_this_person',
				'from_messages', 'from_this_person_to_poi',
				'shared_receipt_with_poi']
				
	feature_dict = dict((feature,{"count": 0,
							"na_poi": 0.,
							"na_non_poi": 0.}) for feature in features)
							
	data = featureFormat(data_dict, features, remove_all_zeroes=False)
	
	#For the feature dictionnary, we will need the total number of poi and 
	#non poi individuals
	count_poi = 0.0
	count_non_poi = 0.0
	for key in data_dict:
		if data_dict[key]["poi"]:
			count_poi += 1.0
		else:
			count_non_poi += 1.0
	
	for i in range(0,len(data[0])):
		#First I extract one specific feature
		col = data[:,i]	
		#Second I identify the position of all NAs and increase count for the 
		#persons affected.
		NAposition = np.where(col == 0)[0]
		feature_dict[features[i]]["count"] = len(NAposition)
		NAnames = names[NAposition]
		na_poi_counter = feature_dict[features[i]]["na_poi"]
		na_non_poi_counter = feature_dict[features[i]]["na_non_poi"]
		for name in NAnames:
			outlier_dict[name]["nacount"] += 1
			outlier_dict[name]["nacategories"].append(features[i])
			if data_dict[name]["poi"]:
				na_poi_counter += 1.0
			else:
				na_non_poi_counter += 1.0
		
		#Finally the feature count ratio are made
		feature_dict[features[i]]["na_poi"] = na_poi_counter/count_poi
		feature_dict[features[i]]["na_non_poi"] = na_non_poi_counter/count_non_poi
		
		#Third I identify the top and bottom 10% of the feature's data
		non_na_length = float(len(np.delete(col, NAposition)))
		top_percent = int(math.ceil(non_na_length / 10))
		top_index = col.argsort()[-top_percent:]
		top_names = names[top_index]
		for name in top_names:
			outlier_dict[name]["count"] += 1
			outlier_dict[name]["categories"].append(features[i])
			
	return outlier_dict, feature_dict

'''
Transforms the dictionary of outliers to an array.
'''	
def outlierToTable(outlier_dict):
	outlier_array = np.chararray([len(outlier_dict)+1, 5], itemsize = 500)
	counter = 0
	outlier_array[counter] = ["NAME", "COUNT", "CATEGORY", "NACOUNT", "NACATEGORY"]
	for name in outlier_dict:
		counter += 1
		count = str(outlier_dict[name]["count"])
		category = " ".join((outlier_dict[name]["categories"]))
		nacount = str(outlier_dict[name]["nacount"])
		nacategory = " ".join((outlier_dict[name]["nacategories"]))
		outlier_array[counter] = [name, count, category, nacount, nacategory]
			
	return outlier_array
		
	
if __name__ == '__main__':
    main()
