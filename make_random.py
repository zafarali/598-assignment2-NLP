import random

with open("random-predictions.csv","w") as f:
    f.write("Id,Prediction")
    for i in range(5917):
        p=random.randint(0,3)
        f.write("\n%s,%s"%(i,p))
