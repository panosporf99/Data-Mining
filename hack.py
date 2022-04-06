import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential 
from keras.layers import Dense,Flatten,Embedding
import matplotlib
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score



dataset = pd.read_csv(r"C:\Users\User\OneDrive - The American College of Greece\Documents\Εξόρυξη Δεδομένων\spam_or_not_spam.csv")


print(dataset.label.value_counts())
dataset = dataset.dropna()



def tokenizer_sequences(num_words, X):
    
    # when calling the texts_to_sequences method, only the top num_words are considered while
    # the Tokenizer stores everything in the word_index during fit_on_texts
    tokenizer = Tokenizer(num_words=num_words)
    
    # From doc: By default, all punctuation is removed, turning the texts into space-separated sequences of words
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    #print(tokenizer.word_counts)#Ektiponei oles tis lexeis kai poses fores emfanizontai
    #print(tokenizer.document_count)#Ektiponei to sinoliko plithos ton grammon poy xrisimopoiithikan
    #print(tokenizer.word_index)#Kodikopoei tis lexeis me toyw monadiko tropo 
    #print(tokenizer.word_docs)#poses fores emfanizetai i kathe lexi se kathe grammi (email) se dictionary
    
    return tokenizer, sequences

#The model takes account only the 1000 most common words
max_words=10000
#Each email will have a 300 word sequence of integers
maxlength= 300 

tokenizer, sequences = tokenizer_sequences(max_words, dataset.email.copy())

word_index = tokenizer.word_index#Find the unique coded sequences
print('Found %s unique tokens.' % len(word_index))

# # We will pad all input sequences to have the length of 300. Each email will be the same length of sequence.
X = pad_sequences(sequences, maxlen=maxlength) #The function transforms a list of sequences into a 2D Numpy Array

y = dataset.label.copy()

print('Shape of data tensor:', X.shape)
print('Shape of label tensor:', y.shape)

#Offset Encoding. O is for no data and 1 is for the word indexes
max_words = len(tokenizer.word_index) + 1 #max_words is the maximum size of the dictionary
#print(max_words)
embedding_dim=100 #The dimension of the word vector

model = Sequential()
#The Embedding model turns positive integers into dense vectors of fixed size
model.add(Embedding(max_words,embedding_dim,input_length=maxlength))#For the embedded dictionary there are 22120*100 parameters 
#So a 2d Vector will be the input and a 3D output vector will be returned 

#Flattens the input 
model.add(Flatten())

model.add(Dense(1,activation='sigmoid'))#Output layer only has one output because its a binary classification problem

model.summary()

#Splits the data set to Test and Train (Y is the label column)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,stratify=y)

print(X_train.shape)
print(y_train.shape)



#Compiles the model with the ADAM optimizer
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['Precision'])


history=model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)


#print(y_test.shape)
#print(y_train.shape)

y_pred = model.predict(X_test, batch_size=32, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
#print(classification_report(y_test, y_pred))
#print(history)
#print(y_test)
print(y_pred)


print(f1_score(y_test, y_pred_bool , average="micro"))

#For the graph
# precision = history.history['precision']
# val_precision = history.history['val_precision']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(precision) + 1)

# plt.plot(epochs, precision, 'g', color='red', label='Training acc')
# plt.plot(epochs, val_precision, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend();