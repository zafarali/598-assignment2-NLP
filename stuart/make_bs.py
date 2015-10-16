with open("validate-all.bs","w") as f:
    for i in range(16):
        b=bin(i)[2:].zfill(4)
        f.write("\npython3 main.py clean/train-%s.csv clean/test-%s.csv --validate --ngram-max=3"%(b,b))
    f.write("\n")
