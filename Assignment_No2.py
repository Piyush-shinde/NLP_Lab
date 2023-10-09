###  Assignment No 2 ###
#Name : Piyush Chandrakant Shinde
#Batch : B3
#Roll No : 55
#Assignment Title : Assignment based on bag of word,TF-IDF and word2Vec 
import gensim
from gensim import corpora
import numpy as np
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from multiprocessing import cpu_count
import gensim.downloader as api
import gzip  
# Import the gzip module

# Set the cache directory
api.BASE_DIR = r"C:\Users\Lenovo\Desktop\python\Lib\site-packages\.gensim"  # Replace with your desired cache directory path

text1 = ["""Three cats chased three playful mice around the garden.
            Five colorful balloons floated high above the carnival.
            Seven students eagerly awaited their turn to present.
            Two tall trees swayed gently in the evening breeze."""]

tokens1 = [[item for item in line.split()] for line in text1]
g_dict1 = corpora.Dictionary(tokens1)

print("The dictionary has: " + str(len(g_dict1)) + " tokens\n")
print(g_dict1.token2id)

print("\n")

"""----------------Creating Bag of Words---------------"""

g_bow = [g_dict1.doc2bow(token, allow_update=True) for token in tokens1]
print("Bag of Words : ", g_bow)

print("\n")

"""-----------------------TF-IDF------------------------"""

g_dict = corpora.Dictionary([simple_preprocess(line) for line in text1])
g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in text1]

print("Dictionary : ")
for item in g_bow:
    print([[g_dict[id], freq] for id, freq in item])

g_tfidf = TfidfModel(g_bow, smartirs='ntc')

print("TF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])

# ----------------------------------------- Word2Vec ------------------------------------------

# Load a smaller subset of the "text8" dataset
dataset = api.load("text8", return_path=True)

# Decompress the dataset using gzip and read it
with gzip.open(dataset, 'rb') as file:
    text8_data = file.read().decode('utf-8').split()

data1 = text8_data[:100000]  # Load a smaller subset (adjust as needed)
w2v_model = Word2Vec([data1], min_count=0, workers=cpu_count(), sg=1)  # Use skip-gram (sg=1) for better results

# You can now use the w2v_model for Word2Vec operations

# Example: Finding similar words to a given word
similar_words = w2v_model.wv.most_similar("cat", topn=5)
print("Words similar to 'cat':", similar_words)

# Example: Getting the vector representation of a word
vector = w2v_model.wv["cat"]
print("Vector representation of 'cat':", vector)

# Example: Calculating similarity between two words
similarity = w2v_model.wv.similarity("cat", "dog")
print("Similarity between 'cat' and 'dog':", similarity)

# ... (Perform other Word2Vec operations as needed)

# Save the Word2Vec model for future use
w2v_model.save("word2vec.model")

# Load the Word2Vec model in the future
# loaded_model = Word2Vec.load("word2vec.model")

# The dictionary has: 32 tokens

# {'Five': 0, 'Seven': 1, 'Three': 2, 'Two': 3, 'above': 4, 'around': 5, 'awaited': 6, 'balloons': 7, 'breeze.': 8, 'carnival.': 9, 'cats': 10, 'chased': 11, 'colorful': 12, 'eagerly': 13, 'evening': 14, 'floated': 15, 'garden.': 16, 'gently': 17, 'high': 18, 'in': 19, 'mice': 20, 'playful': 21, 'present.': 22, 'students': 23, 'swayed': 24, 'tall': 25, 'the': 26, 'their': 27, 'three': 28, 'to': 29, 'trees': 30, 'turn': 31}


# Bag of Words :  [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 3), (27, 1), (28, 1), (29, 1), (30, 1), (31, 1)]]


# Dictionary :
# [['above', 1], ['around', 1], ['awaited', 1], ['balloons', 1], ['breeze', 1], ['carnival', 1], ['cats', 1], ['chased', 1], ['colorful', 1], ['eagerly', 1], ['evening', 1], ['five', 1], ['floated', 1], ['garden', 1], ['gently', 1], ['high', 1], ['in', 1], ['mice', 1], ['playful', 1], ['present', 1], ['seven', 1], ['students', 1], ['swayed', 1], ['tall', 1], ['the', 3], ['their', 1], ['three', 2], ['to', 1], ['trees', 1], ['turn', 1], ['two', 1]]
# TF-IDF Vector:
# [['above', 0.15], ['around', 0.15], ['awaited', 0.15], ['balloons', 0.15], ['breeze', 0.15], ['carnival', 0.15], ['cats', 0.15], ['chased', 0.15], ['colorful', 0.15], ['eagerly', 0.15], ['evening', 0.15], ['five', 0.15], ['floated', 0.15], ['garden', 0.15], ['gently', 0.15], ['high', 0.15], ['in', 0.15], ['mice', 0.15], ['playful', 0.15], ['present', 0.15], ['seven', 0.15], ['students', 0.15], ['swayed', 0.15], ['tall', 0.15], ['the', 0.46], ['their', 0.15], ['three', 0.31], ['to', 0.15], ['trees', 0.15], ['turn', 0.15], ['two', 0.15]]
# Words similar to 'cat': [('surplus', 0.364635705947876), ('athenian', 0.352490097284317), ('needy', 0.3470882773399353), ('quicksilver', 0.33637046813964844), ('controlling', 0.3165172040462494)]
# Vector representation of 'cat': [-0.00421606 -0.00882593  0.00642969  0.0065887   0.00156688 -0.00562923
#   0.00511397  0.00097949 -0.00308182  0.0011393   0.00662805 -0.00751488
#  -0.00191927 -0.00135542  0.00689249  0.00781162  0.00160174  0.00690459
#  -0.00132369 -0.00199113  0.00533631  0.006224    0.00524139  0.00730691
#  -0.00910861  0.00629609  0.00808119  0.00763798  0.00545977  0.00872943
#  -0.00303607 -0.00266312  0.00682694 -0.00174904 -0.00655542 -0.00991271
#   0.00868801  0.00021952  0.00435293  0.00197958  0.00908333  0.00988311
#  -0.00553019 -0.00010288  0.00391529 -0.0093305   0.00190426 -0.00730526
#  -0.0061933   0.00013496  0.00882402 -0.00160715  0.00836566 -0.00095052
#  -0.00094374 -0.00011189  0.0063664   0.00271382  0.00101603 -0.00694345
#   0.00703154  0.00860466 -0.00590454 -0.00979341 -0.0037588   0.00818667
#   0.00761754 -0.00410352  0.00895276 -0.00029889 -0.00970295 -0.00518836
#  -0.00223871  0.0025583  -0.00658627  0.00988786 -0.00176687  0.00787927
#  -0.00989669  0.00937788 -0.00154297 -0.00654141  0.00779983 -0.0004494
#  -0.00213458 -0.00094042  0.00420846 -0.00502848 -0.00493851 -0.00898551
#  -0.00313735  0.00615826  0.00935429  0.00854484  0.0037217  -0.0003978
#   0.00026676  0.00514039  0.00196898 -0.0023705 ]
# Similarity between 'cat' and 'dog': 0.12184672