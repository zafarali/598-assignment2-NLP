from utilities import ConfusionMatrix

a=[0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
p=[0,1,1,1,1,2,2,2,2,2,3,1,1,3,3,3]

cm=ConfusionMatrix(a,p)

print("ACTUAL")
print(a)
print("PREDICTED")
print(p)

for i in range(4):
    print("\ncategory %s:"%i)
    print("TP: ",end="")
    print(cm.TP(label=i))
    print("FP: ",end="")
    print(cm.FP(label=i))
    print("TN: ",end="")
    print(cm.TN(label=i))
    print("FN: ",end="")
    print(cm.FN(label=i))

print("avg accuracy: ",end="")
print(cm.average_accuracy())
print("accuracy: ",end="")
print(cm.accuracy())
print("precision: ",end="")
print(cm.precision())
print("recall: ",end="")
print(cm.recall())
