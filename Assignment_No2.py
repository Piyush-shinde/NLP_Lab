import gensim
from gensim import corpora

text1 = ["""Three cats chased three playful mice around the garden.
            Five colorful balloons floated high above the carnival.
            Seven students eagerly awaited their turn to present.
            Two tall trees swayed gently in the evening breeze."""]

tokens1 = [[item for item in line.split()] for line in text1]
g_dict1 = corpora.Dictionary(tokens1)

print("The dictionary has: " +str(len(g_dict1)) + " tokens\n")
print(g_dict1.token2id)

print()

"""----------------Creating Bag of Words---------------""" 

g_bow =[g_dict1.doc2bow(token, allow_update = True) for token in tokens1]
print("Bag of Words : ", g_bow)