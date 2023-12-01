import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

data="In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."
corpus = data.lower().split("\n")

# print(corpus, "\n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index)+1
# print(total_words)

#take the corpus and turn it into training data
input_sequences=[]

for line in corpus: # iterate from all corpus
    token_list = tokenizer.texts_to_sequences([line])[0] #tokenize the current line
    for i in range(1, len(token_list)): #iterate over the current line
        n_gram_sequence = token_list[:i+1] #generate the subpharses (e.g. [4,2], [4,2,8], [4,2,8,16]...)
        input_sequences.append(n_gram_sequence) #append the subphrases into list

max_length = max([len(x) for x in input_sequences]) # find the longest sequence length by iterate through the input_sequences
# print(max_length)

#using pre because we want to take the last character as the label therefore we pad 0 before the character
input_pad = pad_sequences(input_sequences, maxlen= max_length, padding='pre')
input_sequences = np.array(input_pad)

x = input_sequences[:,:-1] #x is all character except last character
y = input_sequences[:,-1] #y is the last character
y = tf.keras.utils.to_categorical(y, num_classes = total_words) # one hot encode to convert a list into categorical

model = tf.keras.Sequential([
#64 = number of dimensions or embedding dimensions, input_length "-1" because we cropped off the last word of each sequence to get the label
    tf.keras.layers.Embedding(total_words, 64, input_length = max_length-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics= ['accuracy'])

model.fit(x, y,
          epochs = 500,
          )


#predicting next word
def predict(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
        probabilities = model.predict(token_list)
        predicted = np.argmax(probabilities, axis= -1)[0]

        if predicted !=0:
            output_word = tokenizer.index_word[predicted]
            seed_text += " "+output_word
        else:
            print("predicted = 0")

    return seed_text

seed_text = "In the town of Athy one Jeremy"
next_words = 10

print(predict(seed_text, next_words))