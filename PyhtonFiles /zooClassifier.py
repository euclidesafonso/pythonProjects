from sklearn import tree
from sklearn import linear_model
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import sys, os, time
import numpy as np
import pydot
import pydotplus
import pandas as pd
import math






'''
 Relevant Information:
   -- A simple database containing 17 Boolean-valued attributes.  The "type"
      attribute appears to be the class attribute.  Here is a breakdown of
      which animals are in which type: (I find it unusual that there are
      2 instances of "frog" and one of "girl"!)

           1 (30) fruitbat, giraffe, girl, goat, gorilla, hamster,
                  hare, leopard, lion, lynx, mink, mole, mongoose,
                  opossum, oryx, platypus, polecat, pony,
                  porpoise, puma, pussycat, raccoon, reindeer,
                  seal, sealion, squirrel, vampire, vole, wallaby,wolf
           2 (20) chicken, crow, dove, duck, flamingo, gull, hawk,
                  kiwi, lark, ostrich, parakeet, penguin, pheasant,
                  rhea, skimmer, skua, sparrow, swan, vulture, wren
           3 (5)  pitviper, seasnake, slowworm, tortoise, tuatara
           4 (13) bass, carp, catfish, chub, dogfish, haddock,
                  herring, pike, piranha, seahorse, sole, stingray, tuna
           5 (4)  frog, frog, newt, toad
           6 (8)  flea, gnat, honeybee, housefly, ladybird, moth, termite, wasp
           7 (10) clam, crab, crayfish, lobster, octopus,
                  scorpion, seawasp, slug, starfish, worm

    Attribute Information: (name of attribute and type of value domain)
    ==== ==============================================================
   1. animal name:      Unique for each instance
   2. hair		Boolean
   3. feathers		Boolean
   4. eggs		Boolean
   5. milk		Boolean
   6. airborne		Boolean
   7. aquatic		Boolean
   8. predator		Boolean
   9. toothed		Boolean
  10. backbone		Boolean
  11. breathes		Boolean
  12. venomous		Boolean
  13. fins		Boolean
  14. legs		Numeric (set of values: {0,2,4,5,6,8})
  15. tail		Boolean
  16. domestic		Boolean
  17. catsize		Boolean
  18. type		Numeric (integer values in range [1,7])


Deciding the Best split

To decide the best split we calculate first the expected error rate of the system.
First: Calculate the Probability of each item.

     (Number records of particular class)
 P = ____________________________________

           (total number of records)



Type 1: 23/50 ---------------> Best Split
Type 2: 9/50
Type 3: 0/50 ----------------> Prunned
Type 4: 7/50
Type 5: 2/50
Type 6: 5/50
Type 7: 4/50

Second: Calculate the gini for the system as a whole, multiplying the Probability of each item.
gini = (23/50) * (9/50) *(0/50)* (7/50) * (2/50) * (5/50) * (4/50)
     = 0.00000370944
To decide the Best split we calculate the gini for each category (18)
0. animal name:    Was remove because it is unique for each instance (ordinal value)

01. hair
02. feathers
03. eggs
04. milk
05. airborne
06. aquatic
07. predator
08. toothed
09. backbone
10. breathes
11. venomous
12. fins
13. legs
14. tail
15. domestic
16. catsize




'''
(23/50) * (9/50) *(0/50)* (7/50) * (2/50) * (5/50) * (4/50)

zoofeatures_name = ['hair','feathers','eggs','milk', 'airborne',
                    'aquatic', 'predator', 'toothed', 'backbone' ,
                    'breathes' , 'venomous','fins', 'legs', 'tail' ,
                    'domestic', 'catsize', 'type']


total =[24,	9,	27,	23,	13,	16,	27,	32,	41,	39,	2,	8,	154,	34,	9,	20,	137]
for i in range(17):

    print("{} {}".format(zoofeatures_name[i],total[i]))

zootarget_names = ['1','2','3','4','5','6','7']





animalsData = pd.read_csv("/Users/euclidesafonso/zoo.csv")


features = np.array(animalsData.iloc[:50,:16])
labels= np.array(animalsData.iloc[:50,16:])



labelTesting= (animalsData.iloc[50:,16:])
##labels = label.reshape(1,50)

trainingSet = np.array(animalsData.iloc[50:,:16])



####--------- Decision Tree Classifier  ----------------



clfzoo = tree.DecisionTreeClassifier(min_samples_split=3, random_state=16)
clfzoo.fit(features, labels)

dot_data = StringIO()
graph = tree.export_graphviz(clfzoo,
                     out_file = dot_data,
                     feature_names=zoofeatures_name,
                     class_names = zootarget_names,
                     filled = True, rounded = True,
                     impurity = False,
                     dot_data =tree.export_graphviz(clfzoo,feature_names = zoofeatures_name,
                     out_file = dot_data))

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("zoo_DecisionTree5.pdf")



######--------- Prediction Algorithm --------------------

prediction = np.array(clfzoo.predict(trainingSet))
print ("The predicted values are: {}".format(prediction))
y_true = labelTesting
y_pred = prediction.reshape(50,1)
accuracy = accuracy_score(y_true, y_pred, normalize=True)*100
print("The Accuracy of your trainned model is : {}". format(accuracy))

(n) = len(features)
correct = (accuracy*n)/100

print("From the training labels, {}  out of {} are correct!".format(correct,n))








####-----------------------------------------------------

'''
data = features
target_attr =labelTesting):

    """
    Calculates the entropy of the given data set for the target attribute.
    """
val_freq     = {}
data_entropy = 0.0


    # Calculate the frequency of each of the values in the target attr
for record in data:
if (val_freq.has_key(record[target_attr])):
    val_freq[record[target_attr]] += 1.0
else:
    val_freq[record[target_attr]]  = 1.0

    # Calculate the entropy of the data for the target attribute
for freq in val_freq.values():
    data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2)

print(data_entropy)
'''
##-------------- Naive Bayes Classifier -----------------

##----------------- Hunt’s Algorithm --------------------



















##--------------- KNeighbors Classifier -----------------







##
##from sklearn.naive_bayes import GaussianNB
##gnb = GaussianNB()
##y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
##print("Number of mislabeled points out of a total %d points : %d"
##      % (iris.data.shape[0],(iris.target != y_pred).sum()))
##Number of mislabeled points out of a total 150 points : 6


#features X
#labels y
#X_train, y_train = features, labels in the training set
#y_train, y_test = features and labels in the testing set




##featuresTrain_X = np.array(animalsData.iloc[:50,:16])
##labelsTrain = np.array(animalsData.iloc[:50,16:])
##labelsTrain_X = labelsTrain
##
##
##
##featuresTest = np.array(animalsData.iloc[50:,:16])
##featuresTest_Y=featuresTest.transpose()
##labelsTest = np.array(animalsData.iloc[50:,16:])
##labelsTest_Y = labelsTest.transpose()
