path = "Text/topics_MB1-49.txt"

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
for key, value in pair.items():
	print(key, value)
