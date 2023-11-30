####  Assignment No 6 ###
#Name : Piyush Chandrakant Shinde
#Batch : B3
#Roll No : 55
#Assignment Title : Dependancy parsing
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
text1 = '''
Kartik is Playing Cricket.
He score a 100 Runs as a opener batsman.
they won the match.
'''
doc1 = nlp(text1)
for token in doc1:
    print(
        f"""
TOKEN: {token.text}
=====
{token.tag_ = }
{token.head.text = }
{token.dep_ = }"""
    )

displacy.serve(doc1, style="dep")
"""
OUTPUT

TOKEN: 

=====
token.tag_ = '_SP'
token.head.text = 'Kartik'
token.dep_ = 'dep'

TOKEN: Kartik
=====
token.tag_ = 'NNP'
token.head.text = 'Playing'
token.dep_ = 'nsubj'

TOKEN: is
=====
token.tag_ = 'VBZ'
token.head.text = 'Playing'
token.dep_ = 'aux'

TOKEN: Playing
=====
token.tag_ = 'NNP'
token.head.text = 'Playing'
token.dep_ = 'ROOT'

TOKEN: Cricket
=====
token.tag_ = 'NNP'
token.head.text = 'Playing'
token.dep_ = 'dobj'

TOKEN: .
=====
token.tag_ = '.'
token.head.text = 'Playing'
token.dep_ = 'punct'

TOKEN:

=====
token.tag_ = '_SP'
token.head.text = '.'
token.dep_ = 'dep'

TOKEN: He
=====
token.tag_ = 'PRP'
token.head.text = 'score'
token.dep_ = 'nsubj'

TOKEN: score
=====
token.tag_ = 'VBP'
token.head.text = 'score'
token.dep_ = 'ROOT'

TOKEN: a
=====
token.tag_ = 'DT'
token.head.text = 'Runs'
token.dep_ = 'det'

TOKEN: 100
=====
token.tag_ = 'CD'
token.head.text = 'Runs'
token.dep_ = 'nummod'

TOKEN: Runs
=====
token.tag_ = 'NNS'
token.head.text = 'score'
token.dep_ = 'dobj'

TOKEN: as
=====
token.tag_ = 'IN'
token.head.text = 'score'
token.dep_ = 'prep'

TOKEN: a
=====
token.tag_ = 'DT'
token.head.text = 'batsman'
token.dep_ = 'det'

TOKEN: opener
=====
token.tag_ = 'JJR'
token.head.text = 'batsman'
token.dep_ = 'amod'

TOKEN: batsman
=====
token.tag_ = 'NN'
token.head.text = 'as'
token.dep_ = 'pobj'

TOKEN: .
=====
token.tag_ = '.'
token.head.text = 'score'
token.dep_ = 'punct'

TOKEN:

=====
token.tag_ = '_SP'
token.head.text = '.'
token.dep_ = 'dep'

TOKEN: they
=====
token.tag_ = 'PRP'
token.head.text = 'won'
token.dep_ = 'nsubj'

TOKEN: won
=====
token.tag_ = 'VBD'
token.head.text = 'won'
token.dep_ = 'ROOT'

TOKEN: the
=====
token.tag_ = 'DT'
token.head.text = 'match'
token.dep_ = 'det'

TOKEN: match
=====
token.tag_ = 'NN'
token.head.text = 'won'
token.dep_ = 'dobj'

TOKEN: .
=====
token.tag_ = '.'
token.head.text = 'won'
token.dep_ = 'punct'

TOKEN:

=====
token.tag_ = '_SP'
token.head.text = '.'
token.dep_ = 'dep'
"""