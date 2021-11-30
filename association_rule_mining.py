#-------------------------------------------------------------------------
# AUTHOR: Josephine Nguyen
# FILENAME: association_rule_mining.py
# SPECIFICATION: This program reads "retail_dataset.csv" to use association rule mining
# FOR: CS 4200- Assignment #5
# TIME SPENT: ~45 min
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)

encoded_vals = []
for index, row in df.iterrows():

    labels = {}
    #Use itemset to create keynames, set default to 0 (false):
    for item in itemset:
        labels[str(item)] = 0

    #Populate labels, ensure entry is NOT nan:
    for entry in row.values:
        if str(entry).lower() != 'nan':
            labels[str(entry)] = 1

    encoded_vals.append(labels)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)


#iterate the rules data frame and print the apriori algorithm results by using the following format:

#Sample Ouput:
#Meat, Cheese -> Eggs
#   Support: 0.21587301587301588
#   Confidence: 0.6666666666666666
#   Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825

for i, rule in rules.iterrows():
    #Print all antecedents:
    for antecendent in list(rule.antecedents):
        print(antecendent + ", ", end='')
    print(" -> ", end='')

    #print all consequents:
    for consequent in list(rule.consequents):
        print(consequent)

    print("\tSupport: " + str(rule.support))
    print("\tConfidence: " + str(rule.confidence) + "\n")


#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
prior = rules["consequent support"][i]
#supportCount = rules["consequent support"]
#prior = supportCount/len(encoded_vals)
print("Gain in Confidence: " + str(100*(rule.confidence - prior)/prior))


#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()