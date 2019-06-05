import pandas as pd
path = "/home/raja/Desktop/cnn-text-classification/sentiment labelled sentences/sentiment labelled sentences"
filepath_dict = {'yelp': '/home/raja/Desktop/cnn-text-classification/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt' ,
'amazon': '/home/raja/Desktop/cnn-text-classification/sentiment labelled sentences/sentiment labelled sentences/amazon_cells_labelled.txt','imdb': 
                  '/home/raja/Desktop/cnn-text-classification/sentiment labelled sentences/sentiment labelled sentences/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
   df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
   # Add another column filled with the source name
   df['source'] = source 
   df_list.append(df)
   df = pd.concat(df_list)

print(df.head())


from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer                    
from keras.preprocessing.sequence import pad_sequences
df_yelp = df[df['source'] == 'yelp']

sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

sentences_train,sentences_test,y_train,y_test = train_test_split(
                                                sentences, y,  
                                                test_size=0.25,  
                                                random_state=1000,)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
# Adding 1 because of  reserved 0 index
vocab_size = len(tokenizer.word_index) + 1                          

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



import numpy as np

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  
    # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


embedding_dim = 50
embedding_matrix = create_embedding_matrix('/home/raja/Desktop/studen_proj/chatbot-gateway/pretrained_models/glove.6B.50d.txt' ,
                                            tokenizer.word_index,  
                                            embedding_dim)



from keras.models import Sequential
from keras import layers
embedding_dim = 100

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),batch_size=10)









