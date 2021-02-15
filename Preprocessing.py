import spacy
import re
import time

def queryToTokens(queryString,sp=spacy.load('en_core_web_sm')):
    tokens = []
    sentence = sp(queryString)
    for token in sentence:
        if not token.is_stop and not token.is_punct and not token.is_space and not token.like_url:
            if (not token.is_alpha or not token.is_ascii):
                result = re.findall(r'[a-z]+', token.lemma_)
                for i in result:
                    if len(i) > 2:
                        tokens.append(i.lower())
                continue
            if len(token.text) > 2:
                tokens.append(token.lemma_.lower())
    return(tokens)

def Preprocessing():
    start = time.time()
    file = open("./Trec_microblog11.txt", mode='r', encoding='UTF-8-sig')
    token_array = []
    sp = spacy.load('en_core_web_sm')
    line = file.readline()
    count = 0
    while line:
        count+=1
        if (count%1000 == 0):
            print("Processing line {}".format(str(count)),end="\r")
        tokens = queryToTokens(line,sp)
        tokens.insert(0, line.split()[0])
        # print(tokens)
        token_array.append(tokens)
        line = file.readline()
    file.close()
    print("\nFinished preprocessing!")
    print("Time:%ss"%(time.time()-start))
    return(token_array)

Preprocessing()
