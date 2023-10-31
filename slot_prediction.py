from domain_prediction_functions import load_json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, TimeDistributed
from tensorflow.keras.metrics import Precision, Recall

json_file_path = 'dialogues_002.json'
data = load_json(json_file_path)
utterance_lists=[]
utterances=[]

for dialogue in data:
    dialogue_id = dialogue['dialogue_id']
    for turn in dialogue['turns']:
        if turn['speaker'] == 'USER':
            utterance = turn['utterance']
            utterance_list = []
            slot_values = turn['frames'][0]['state']['slot_values']  # Assuming only one frame per turn
            words = utterance.split()
            for word in words:
                found_slot = False
                for slot, values in slot_values.items():
                    if any(word.lower() in value.lower() for value in values):
                        utterance_list.append(slot)
                        found_slot = True
                        break
                if not found_slot:
                    utterance_list.append('0')
            utterance_lists.append(utterance_list)
            utterances.append(utterance)
            #print(f"Dialog ID: {dialogue_id}, Utterance: {utterance}, Utterance List: {utterance_list}")

df = pd.DataFrame({'sentences': utterances, 'slot_labels': utterance_lists})

# Display the DataFrame
print(df)

X_train, X_test, y_train, y_test = train_test_split(df['sentences'], df['slot_labels'], 
                                                    test_size = 0.2, random_state = 1)

NUM_WORDS = 7501
OOV_TOKEN = "<UNK>"

#Initialize Tokenizers
tokenizer = Tokenizer(num_words = NUM_WORDS, filters = '', lower = False, split = ' ', oov_token = OOV_TOKEN)
tokenizer.fit_on_texts(list(X_train))

y_tokenizer = Tokenizer(filters = '', lower = False, split = ' ')
y_tokenizer.fit_on_texts(list(y_train))

#Convert text to sequences
X_seq = tokenizer.texts_to_sequences(list(X_train))
X_test_seq = tokenizer.texts_to_sequences(list(X_test))

y_seq = y_tokenizer.texts_to_sequences(list(y_train))
y_test_seq = y_tokenizer.texts_to_sequences(list(y_test))

MAX_SEQ_LEN = 35

#Pad the sequences
X_train_padded = pad_sequences(X_seq, maxlen = MAX_SEQ_LEN, padding = 'post')
X_test_padded = pad_sequences(X_test_seq, maxlen = MAX_SEQ_LEN, padding = 'post')

y_train_padded = pad_sequences(y_seq, maxlen = MAX_SEQ_LEN, padding = 'post')
y_test_padded = pad_sequences(y_test_seq, maxlen = MAX_SEQ_LEN, padding = 'post')

#Convert labels to one-hot vectors
y_train_encoded = utils.to_categorical(y_train_padded)
y_test_encoded = utils.to_categorical(y_test_padded)
print(y_train_encoded.shape, y_test_encoded.shape)

#Reshape the input for Bi-LSTM
X_train_padded = np.reshape(X_train_padded, (X_train_padded.shape[0], X_train_padded.shape[1], 1))
X_test_padded = np.reshape(X_test_padded, (X_test_padded.shape[0], X_test_padded.shape[1], 1))
print(X_test_padded.shape, X_test_padded.shape)

VAL_SPLIT = 0.1
BATCH_SIZE = 32
EPOCHS = 7
EMBEDDING_DIM = 64
NUM_UNITS = 32
VOCAB_SIZE = NUM_WORDS
Y_VOCAB_SIZE = len(y_tokenizer.word_index) + 1


#Define a Bi-LSTM model
bilstm_model = Sequential()
bilstm_model.add(Embedding(input_dim = VOCAB_SIZE, output_dim = EMBEDDING_DIM, input_length = MAX_SEQ_LEN))
bilstm_model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu', return_sequences=True)))
bilstm_model.add(TimeDistributed(Dense(Y_VOCAB_SIZE, activation='softmax')))

#Compile the model
bilstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Precision(), Recall(), 'accuracy'])

bilstm_model.summary()

#Fit the model on training data
bilstm_history = bilstm_model.fit(X_train_padded, y_train_encoded, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = VAL_SPLIT)

bilstm_score = bilstm_model.evaluate(X_test_padded, y_test_encoded, batch_size = BATCH_SIZE)


label_list = list(y_tokenizer.word_index.keys())
index_list = list(y_tokenizer.word_index.values())

#Input sentence
sentence = "I want to book a table for 4 people at a mexican place."
input_seq = tokenizer.texts_to_sequences([sentence])
input_features = pad_sequences(input_seq, maxlen = MAX_SEQ_LEN, padding = 'post')

#Predict the slots
prediction = bilstm_model.predict(input_features)
slots = [label_list[index_list.index(j)] for j in [np.argmax(x) for x in prediction[0][:]] if j in index_list]
print(sentence)
print(slots)