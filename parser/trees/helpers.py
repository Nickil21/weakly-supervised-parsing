import nltk

def extract_sentence(sentence):
    t = nltk.Tree.fromstring(sentence)
    return " ".join(item[0] for item in t.pos())
