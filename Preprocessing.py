import spacy
import re

def only_alphabet(token):
    for i in token:
        if ( i >= u'\u0041' and i <= u'\u005A' ) or ( i >= u'\u0061' and i <= u'\u007A'):
            continue
        else:
            return False
    return True

def Preprocessing():
    file = open("./Trec_microblog11.txt", mode='r', encoding='UTF-8-sig')
    sp = spacy.load('en_core_web_sm')
    token_array = []
    line = file.readline()

    while line:
        id = line.split()[0]
        temp = []
        sentence = sp(line)
        for token in sentence:
            if not token.is_stop and not token.is_punct and not token.is_space and token.text[:7]!='http://':
                if not only_alphabet(token.text):
                    result = re.findall(r'[a-zA-Z]+', token.text)
                    for i in result:
                        if len(i) > 2:
                            temp.append(i)
                    continue
                if len(token.text) > 2:
                    temp.append(token.lemma_)
        temp.insert(0, id)
        token_array.append(temp)
        line = file.readline()

    file.close()
    print(token_array)

Preprocessing()
