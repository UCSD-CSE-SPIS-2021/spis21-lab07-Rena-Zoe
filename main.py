
#Part 2
#step 2:
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.classify.util import accuracy

# "Stop words" that you might want to use in your project/an extension
stop_words = set(stopwords.words('english'))

def format_sentence(sent):
    ''' format the text setence as a bag of words for use in nltk'''
    tokens = nltk.word_tokenize(sent)
    return({word: True for word in tokens})

def get_reviews(data, rating):
    ''' Return the reviews from the rows in the data set with the
        given rating '''
    rows = data['Rating']==rating
    return list(data.loc[rows, 'Review'])

#Trying to split up the data in a list based off of a number 0-1
#So 0.25 will split around 1/4, 0.6 will split it around 1/2
def split_train_test(data, train_prop):
    ''' input: A list of data, train_prop is a number between 0 and 1
              specifying the proportion of data in the training set.
        output: A tuple of two lists, (training, testing)
    '''
    #The casting function int() and list slicing can make this function just one line:
    #data[0:int(len(data)*train_prop)]
    #Cast is changing the data type of the operation -> string of 1 in the int function, makes it an integer 1 - can multiply integers, but not strings
    #data.slice(0, -1, int(train_prop))
      #Train_prop is the 0-1 number
    #Use the index of the list to correspond to the train proportion

    # TODO: You will write this function, and change the return value
    return (data[0:int(len(data)*train_prop)], data[int(len(data)*train_prop):len(data)])

print(split_train_test(["A", "B", "C", "D"], 0.78))
print(split_train_test(["A", "B", "C", "D"], 0.20))
print(split_train_test(["A", "B", "C", "D"], 0.1))
print(split_train_test(["A", "B", "C", "D"], 0.3))
print(split_train_test(["A", "B", "C", "D"], 0.55))
#Convert place you want to split the list into an integer
#Passes through - you want to find corresponding index in the data, where the 0/25 cut off would be


