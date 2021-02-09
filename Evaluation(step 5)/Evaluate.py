import random
import time

expectedPath = "Text/Trec_microblog11-qrels.txt"
queryPath = "Text/topics_MB1-49.txt"

def fakeStep3(query:"Query in String")->"Dict in the format tweetID:Score":
    FakeIds=[]
    Scores=[]
    init = 33823403328671744
    for i in range(15):
        FakeIds.append(str(29509222337085430+i))
        Scores.append((10-i)/10)
    return dict(zip(FakeIds,Scores))

def testQueries(path:"String of test query path")->"Dict in the format {queryNum('B001'):queryString('BBC World Service staff cuts')}":
    title = []
    num = []
    with open(path, "r") as f:  
        for line in f:        
            if line[:7] == "<title>":
                title.append(line[8:-10])
            elif line[:5] == "<num>": 
                num.append(line[15:-8])
    return dict(zip(num,title))

def loadResult(queriesPair:"dict(QueryNum:QueryString)")->"Dict{queryNum:Dict_DesendingScoreOrder{tweetid:score}}":
    returnValue = {}
    for num, query in queriesPair.items():
        result = fakeStep3(query)
        sortedResult=dict(sorted(result.items(),key=lambda result:result[1], reverse=True))       
        returnValue[(int)(num[2:].lstrip('0'))] = sortedResult
    return returnValue
    

def loadExpectedResult(expectedPath:"path of Trec_microblog11-qrels.txt")->"Dict{QueryNum:Set{all relevant ids}}":
    title = []
    num = []
    lastQueryNum = 0
    returnValue = {}
    with open(expectedPath, "r") as ef:  
        for line in ef: 
            lst = line[:-1].split("\t") #[queryNum, 0, id, relevant]
            currentQueryNum = int(lst[0])
            if (currentQueryNum != lastQueryNum):
                returnValue[currentQueryNum] = set() # add an empty set in the dict
            lastQueryNum = currentQueryNum
            if lst[3] == '1': # if relevant          
                returnValue[currentQueryNum].add(lst[2]) # add id to the set
                
    return returnValue

def computeMAP(result:"Dict{queryNum:Dict_DesendingScoreOrder{tweetid:score}}",expect:"Dict{QueryNum:Set{all relevant ids}}",
               first10:"set True if compute AP only for first 10"=False) -> "MAP or P@10":
    
    APs=[]
    for queryNum,resultMap in result.items():

        expectForTheQuery = expect[queryNum]
        ids = list(resultMap.keys())
        relevant = 1
        relevantDocument = []
        
        for i in range(1, min(11, len(ids)+1) if first10 else len(ids)+1):
            if ids[i-1] in expectForTheQuery:
                relevantDocument.append(relevant/i)
                relevant+=1
        if len(relevantDocument) != 0:
            AP = sum(relevantDocument) / len(relevantDocument)
        else:
            AP = 0
        APs.append(round(AP, 4))
        
    MAP = sum(APs) / len(APs)
    return round(MAP, 4)


if __name__ == '__main__':
    queriesPair = testQueries(queryPath)
    result = loadResult(queriesPair)
    expect = loadExpectedResult(expectedPath)
    print(computeMAP(result,expect))
    print(computeMAP(result,expect,first10=True))