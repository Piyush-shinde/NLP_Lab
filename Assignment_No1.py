###  Assignment No 1 ###
#Name : Piyush Chandrakant Shinde
#Batch : B3
#Roll No : 55
#Assignment Title : Text pre-processing using NLP operation : perform Tokenization
#Stop word removal, Punctuation removal,using Spacy or NLTK Library 

#import library
 
import spacy
nlp=spacy.load("en_core_web_sm")
from collections import Counter

print("\n##########   Sentence Detection    ###############\n")
#Sentence Detection

text1=("My name is Piyush "
       "I am the student of Information-technology"
       "My college name is Sanjivani College of Engineering, Kopargaon")

about_doc=nlp(text1)
sentences=list(about_doc.sents)
len(sentences)

for sentence in sentences:
    print(f"{sentence[:5]}...")
print("\n#############       Tokenization         ###############\n")

#tokenization

for token in about_doc:
    print(token,token.idx)
print("\n###############           Stop Words Removal       ###############\n")

#stop words removal

spacy_stopwords=spacy.lang.en.stop_words.STOP_WORDS
len(spacy_stopwords)
print()
for stop_word in list(spacy_stopwords)[:15]:
    print(stop_word)

print([token for token in about_doc if not token.is_stop])

print("\n###############          Lemmatization            ###############\n")

for token in about_doc:
    if str(token)!= str(token.lemma_):
        print(f"{str(token):>20}:{str(token.lemma_)}")

print("\n###############           Word Frequency       ###############\n")

words=[
    token.text
    for token in about_doc
    if not token.is_stop and not token.is_punct
]
print(Counter(words).most_common(5))

print("\n###############           Part_of_speech Tagging       ###############\n")

for token in about_doc:
    print(
        f"""
        TOKEN:{str(token)}
        =====
        TAG: {str(token.tag_):10} POS: {token.pos_}
        EXPLANATION: {spacy.explain(token.tag_)}"""
    )
"""
##########   Sentence Detection    ###############

My name is Piyush I...

#############       Tokenization         ###############

My 0
name 3
is 8
Piyush 11
I 18
am 20
the 23
student 27
of 35
Information 38
- 49
technologyMy 50
college 63
name 71
is 76
Sanjivani 79
College 89
of 97
Engineering 100
, 111
Kopargaon 113

###############           Stop Words Removal       ###############


this
much
sixty
beforehand
might
over
’re
have
be
become
did
nowhere
she
what
whereby
[Piyush, student, Information, -, technologyMy, college, Sanjivani, College, Engineering, ,, Kopargaon]

###############          Lemmatization            ###############

                  My:my
                  is:be
                  am:be
                  is:be

###############           Word Frequency       ###############

[('Piyush', 1), ('student', 1), ('Information', 1), ('technologyMy', 1), ('college', 1)]

###############           Part_of_speech Tagging       ###############


        TOKEN:My
        =====
        TAG: PRP$       POS: PRON
        EXPLANATION: pronoun, possessive

        TOKEN:name
        =====
        TAG: NN         POS: NOUN
        EXPLANATION: noun, singular or mass

        TOKEN:is
        =====
        TAG: VBZ        POS: AUX
        EXPLANATION: verb, 3rd person singular present

        TOKEN:Piyush
        =====
        TAG: NNP        POS: PROPN
        EXPLANATION: noun, proper singular

        TOKEN:I
        =====
        TAG: PRP        POS: PRON
        EXPLANATION: pronoun, personal

        TOKEN:am
        =====
        TAG: VBP        POS: AUX
        EXPLANATION: verb, non-3rd person singular present

        TOKEN:the
        =====
        TAG: DT         POS: DET
        EXPLANATION: determiner

        TOKEN:student
        =====
        TAG: NN         POS: NOUN
        EXPLANATION: noun, singular or mass

        TOKEN:of
        =====
        TAG: IN         POS: ADP
        EXPLANATION: conjunction, subordinating or preposition

        TOKEN:Information
        =====
        TAG: NNP        POS: PROPN
        EXPLANATION: noun, proper singular

        TOKEN:-
        =====
        TAG: HYPH       POS: PUNCT
        EXPLANATION: punctuation mark, hyphen

        TOKEN:technologyMy
        =====
        TAG: NNP        POS: PROPN
        EXPLANATION: noun, proper singular

        TOKEN:college
        =====
        TAG: NN         POS: NOUN
        EXPLANATION: noun, singular or mass

        TOKEN:name
        =====
        TAG: NN         POS: NOUN
        EXPLANATION: noun, singular or mass

        TOKEN:is
        =====
        TAG: VBZ        POS: AUX
        EXPLANATION: verb, 3rd person singular present

        TOKEN:Sanjivani
        =====
        TAG: NNP        POS: PROPN
        EXPLANATION: noun, proper singular

        TOKEN:College
        =====
        TAG: NNP        POS: PROPN
        EXPLANATION: noun, proper singular

        TOKEN:of
        =====
        TAG: IN         POS: ADP
        EXPLANATION: conjunction, subordinating or preposition

        TOKEN:Engineering
        =====
        TAG: NNP        POS: PROPN
        EXPLANATION: noun, proper singular

        TOKEN:,
        =====
        TAG: ,          POS: PUNCT
        EXPLANATION: punctuation mark, comma

        TOKEN:Kopargaon
        =====
        TAG: NNP        POS: PROPN
        EXPLANATION: noun, proper singular
        """