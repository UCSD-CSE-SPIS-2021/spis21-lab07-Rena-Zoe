
import random

#Step 2

#Niema's code from lecture
#counts = dict()
#for word in words:
  #if word not in counts:
   # counts[word] = 1
 # else:
   # counts[word] += 1
#print(counts)

s = "Yeah baby I like it like that You gotta believe me when I tell you I said I like it like that"

#s = "Rena is cool Nikki is cool Diego plays the guitar"

def train(s):
  #We want the function to spit out what word happens after the previous one 
    #So ex: after I this will be spit out: 'I': ['like', 'tell', 'said', 'like']

  #Splitting the string stuff
  #s.split()
  words_list = s.split()
   #This is a string which is the parameter
    #Will automatically split the commas

  #Made a dictionary for the words
  next_words_dict = dict()

  for index in range(len(words_list)-1):
      #For the length of the list
    #Goal is to have a list: there is a key and a list
    word = words_list[index]
    if word not in next_words_dict:
      next_words_dict[word] = []
    next_words_dict[word].append(words_list[index + 1])
    #We want to add the word to the dictionary
      #Adding the next words to the dictionary
  return next_words_dict
#print(train(s))
  #Can change s to be whatever string we want

#Step 3:

def generate(model, first_word, num_words):
#we want to use random generated strings from the nonzero probability list to create a sentence
  #model – a dictionary representing the trained model as output from the train method
  #first_word – the word to use as the first word in the generated text
  #num_words – the number of words in the returned generated string
  model = train(s)
  while num_words != 0:
    print(first_word)
    next_word = random.choice(model[first_word])
    num_words = num_words - 1
    first_word = next_word
      #Will print the next word instead of the first word because it's being updated and moving over + 1 

generate(s, 'I', 5)
#Michael helped us with this code ^