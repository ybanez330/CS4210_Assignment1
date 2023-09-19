#-------------------------------------------------------------------------
# AUTHOR: Julia Ybanez
# FILENAME: decision_tree.py
# SPECIFICATION: The program will read 'contact_lens.csv' and output an ID3 algorithm decision tree.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 1 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: 
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
featureDict = {'Young': 1, 'Presbyopic': 2, 'Prepresbyopic': 3, 'Myope': 1, 'Hypermetrope': 2, 'Yes': 1, 'No': 2, 'Normal': 1, 'Reduced': 2}
num_attributes = 4
for row in range(len(db)):
    temp = []
    for feature in range(0, num_attributes):
        temp.append(int(featureDict[db[row][feature]]))
    X.append(temp)

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
for row in range(len(db)):
    Y.append(int(featureDict[db[row][4]]))

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()