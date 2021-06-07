# Language_Translator_English-to-Marathi
Neural Machine Translation (Translating English to Marathi) using Encoder-Decoder LSTM Neural Network Architecture

 Main Stages in creating a Neural Machine Translation System : 

1] Encoding or Vectorizing the dataset (during training)

2] Building the Model architecture

3] Decoding the predicted outputs to strings (during testing)

Dataset link : http://www.manythings.org/anki/ (Download "mar-eng.zip" file)

![image](https://user-images.githubusercontent.com/63066870/121022246-c99b2100-c7bf-11eb-9770-7806365f89b8.png)


NOTE : This model uses one-hot encoding & dictionary lookup to vertorize string sentences.Due to this the model can only translate the words present in its lookup dictionary or vocabulary,that why during testing we randomly choose samples from our already encoded dataset.To make it a more generalised, it needs to be trained on a lot of data (By alot I mean ALOT !) , or better if used an word-embedding like word2vec.

reference : https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
