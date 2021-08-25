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


def split_train_test(data, train_prop):
    #The casting function int() and list slicing can make this function just one line:
    #data[0:int(len(data)*train_prop)]
    #Cast is changing the data type of the operation -> string of 1 in the int function, makes it an integer 1 - can multiply integers, but not strings
    #Float cannot be a list index
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

def format_for_classifier(data_list, label):
  #Colon separates key in dictionary from value

  #if word in tokens

  return [[format_sentence(document),label] for document in data_list]
  #Akshat helped us >:D
  #In the brackets is what you are returning
    #Run a for loop through data list, then return the format sentence to the document
    #This is a list: [format_sentence(document),label] -> to get the list for each element need to use the for loop
print(format_for_classifier(["A good one", "The best!"], "pos"))

#print(format_for_classifier(["Today is a good day!", "Pygame is fun"], "pos"))

def classify_reviews():
    ''' Perform sentiment classification on movie reviews ''' 
    # Read the data from the file
    data = pd.read_csv("data/movie_reviews.csv")

    # get the text of the positive and negative reviews only.
    # positive and negative will be lists of strings
    # For now we use only very positive and very negative reviews.
    positive = get_reviews(data, 4)
    negative = get_reviews(data, 0)

    # Split each data set into training and testing sets.
    # You have to write the function split_train_test
    (pos_train_text, pos_test_text) = split_train_test(positive, 0.8)
    (neg_train_text, neg_test_text) = split_train_test(negative, 0.8)

    # Format the data to be passed to the classifier.
    # You have to write the format_for_classifier function
    pos_train = format_for_classifier(pos_train_text, 'pos')
    neg_train = format_for_classifier(neg_train_text, 'neg')

    # Create the training set by appending the pos and neg training examples
    training = pos_train + neg_train

    # Format the testing data for use with the classifier
    pos_test = format_for_classifier(pos_test_text, 'pos')
    neg_test = format_for_classifier(neg_test_text, 'neg')
    # Create the test set
    test = pos_test + neg_test


    # Train a Naive Bayes Classifier
    # Uncomment the next line once the code above is working
    classifier = NaiveBayesClassifier.train(training)

    # Uncomment the next two lines once everything above is working
    print("Accuracy of the classifier is: " + str(accuracy(classifier, test)))
    classifier.show_most_informative_features()

    # TODO: Calculate and print the accuracy on the positive and negative
    # documents separately
    # You will want to use the function classifier.classify, which takes
    # a document formatted for the classifier and returns the classification
    # of that document ("pos" or "neg").  For example:
    # classifier.classify(format_sentence("I love this movie. It was great!"))
    # will (hopefully!) return "pos"

    # TODO: Print the misclassified examples


if __name__ == "__main__":
    classify_reviews()
