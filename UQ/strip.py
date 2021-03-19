import os

path  = os.path.dirname(os.path.abspath(__file__))   
        
f = open(os.path.join(path,"UQpy.py"))
for line in f:
        line = line.rstrip()
        if line:
                f2 = open(os.path.join(path,"UQpy2.py"), "a+")
                f2.write(line + "\n")
                f2.close()
                print(line, end = '')