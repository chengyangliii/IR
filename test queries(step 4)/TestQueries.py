path = "Text/topics_MB1-49.txt"


import random
import time


def fakeStep3(query):
    FakeIds=[]
    Scores=[]
    for i in range(10):
        FakeIds.append(query+"Fake id "+str(i))
        Scores.append(random.random())
    return dict(zip(FakeIds,Scores))


def testQueries(path):
    title = []
    num = []
    with open(path, "r") as f:  
        for line in f:        
            if line[:7] == "<title>":
                title.append(line[8:-10])
            elif line[:5] == "<num>": 
                num.append(line[15:-8])
    return dict(zip(num,title))

pair = testQueries(path)

tests = input()

if tests == "1": # test read title and numID in test files
    for key, value in pair.items():
        print(key, value)
if tests == "2": # test fake function
    for i in range(10):
        print(fakeStep3(""))
if tests == "3": # combine test case1&2
    outPath = 'output/{}.txt'.format(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    with open(outPath, 'w+') as f:
        f.write('topic_id Q0 docno rank score tag\n')
        for num, query in pair.items():
            result = fakeStep3(query)
            sortedResult=dict(sorted(result.items(),key=lambda result:result[1], reverse=True))
            #print(sortedResult)
            
            rank = 1
            for docNum,score in sortedResult.items():
                f.write("{} {} {} {} {} {}".format(num[2:].strip('0'), 'Q0', docNum, rank, score,'myRun\n'))
                rank+=1
