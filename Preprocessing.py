import spacy
import re

def alphabet_validate(token):
    for i in token:
        if not ( i >= u'\u0061' and i <= u'\u007A'):
            return False
    return True

def Preprocessing():
    file = open("./Trec_microblog11.txt", 'r')
    sp = spacy.load('en_core_web_sm')
    token_array = []
    line = file.readline()
    while line:
        temp = []
        sentence = sp(line)
        for token in sentence:
            if not token.is_stop and not token.is_punct and not token.is_space and token.text[:7]!='http://':
                if not alphabet_validate(token.lower_):
                    result = re.findall(r'[a-z]+', token.lower_)
                    for i in result:
                        if len(i) > 2:
                            temp.append(i.lower())
                    continue
                if len(token.text) > 2:
                    temp.append(token.lemma_.lower())
        temp.insert(0, line.split()[0])
        token_array.append(temp)
        line = file.readline()
    file.close()
    print(token_array)

Preprocessing()
