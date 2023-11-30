####  Assignment No 5 ###
#Name : Piyush Chandrakant Shinde
#Batch : B3
#Roll No : 55
#Assignment Title : regular Expression
import re

def find_entities(text):

    result = {
        'HTML tags': re.findall(r'<\/?([a-zA-Z][^\s>]*)\/?>', text),
        'Email Address': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
        'Pin code': re.findall(r'[0-9]{6}', text),
    }
    return result

# Example usage:
sample_text = """
<p>This is a paragraph.</p>
<div class="container">
    <h1>Title</h1>
    <p>Another paragraph.</p>
</div>
<span>This is a span.</span>

if Any query contact piyushshinde123@gmail.com

My Address is Kopargaon, 121212
"""

result = find_entities(sample_text)

for entity_type, entities in result.items():
    print(f"{entity_type}: {entities}")

    """OUTPUT:
    HTML tags: ['p', 'p', 'h1', 'h1', 'p', 'p', 'div', 'span', 'span']
    Email Address: ['piyushshinde123@gmail.com']
    Pin code: ['121212']
    """