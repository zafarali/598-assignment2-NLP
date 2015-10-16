import sys, os.path

filename=sys.argv[1]
column=int(sys.argv[2])
if not os.path.isfile(filename):
    print("Not a file: '%s'"%filename)

with open(filename,"r") as f:
    lines=f.readlines()

counter={}
for line in lines:
    try:
        item=line.split(",")[column].strip()
    except:
        pass
    if item not in counter:
        counter[item]=0
    counter[item]+=1

keys=sorted(list(counter.keys()))
for key in keys:
    print("%s = %s"%(key,counter[key]))
