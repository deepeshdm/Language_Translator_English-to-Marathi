
import numpy as np
from tensorflow import keras
from keras.optimizers import *

#------------------------------------------------------------------------

num_samples = 10000
data_path = r"C:\Users\dipesh\Desktop\mar-eng\mar.txt"

with open(data_path, "r", encoding="utf-8") as f:
    # \n - new line
    # List containing each line in file as a value
    lines = f.read().split("\n")

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

# len(lines) = 42703
for line in lines[: min(num_samples, len(lines) - 1)]:

    # "\t" - tabs
    input_text, target_text, _ = line.split("\t")

    # Adding start & end tokens
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)

    # Creating character vocabulary
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)


input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# No.of samples: len(input_texts)
# No.of unique input tokens: num_encoder_tokens
# No.of unique output tokens: num_decoder_tokens
# Maximum seq length for inputs: max_encoder_seq_length
# Maximum seq length for outputs: max_decoder_seq_length


input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])


# Create a 3D array to store all one-hot encoded sentences.
# This 3D array contains "n" 2D arrays where n=no.of samples.Each sentence is represented as 2D array
# Each 2D array contains "n" 1D arrays where n=max_seq_length.Each word is represented by 1D array
# Each 1D array contains "n" elements where n=no.of unique characters in data.Each char in word is represented by 1/0
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0


#------------------------------------------------------------------------

# Building the encoder & decoder model for Training Phase.

latent_dim = 256

# encoder
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True) #lstm_1

encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

#decoder
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

#------------------------------------------------------------------------

batch_size = 64
epochs = 30

# we have 2 inputs (1 for encoder & 1 for decoder since we are following "teacher-forcing")
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001),
              loss='categorical_crossentropy', metrics=["accuracy"])

model.fit([encoder_input_data, decoder_input_data],decoder_target_data,
          batch_size=batch_size,epochs=epochs,validation_split=0.2)

# Saving the Model
model.save("Eng_Marathi_NMT")

#------------------------------------------------------------------------

# Building the encoder & decoder model again for Testing Phase since we are using
# "Teacher-Forcing" decoder works differently during training & testing.

# encoder
encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

# decoder
decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4_")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


# Reverse-lookup token indexes to decode the sequence to make it readable
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

#------------------------------------------------------------------------

# After generating the empty sequence of length 1, the model should know when
# to start and stop reading the text.To read the model will check out for \t in this case.
# Keep two conditions, either when the max length of sentence is hit or find stop character \n.
# Keep on updating the target sequence by one and update the states.

def decode_sequence(input_seq):

    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]
    return decoded_sentence

#------------------------------------------------------------------------

# Picks a random sentence from dataset & translates it

i = np.random.choice(len(input_texts))
input_seq = encoder_input_data[i:i+1]
translation = decode_sequence(input_seq)

print('English :', input_texts[i])
print('Marathi Translation:', translation)

''' 
NOTE :- 
Since we are using a one-hot encoding with dictionary lookup,
this model can translate only those words which are present in its given dataset.
It may not work well (may not work at all!) if any other words are given.
Solution to this is "MORE DATA" (by more I mean ALOTT !)
'''
