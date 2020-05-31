import os
from subprocess import Popen, PIPE
import numpy as np
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Process, Lock, Queue

def func(q,land_val):
        try:
            os.chdir("sugar demand val: "+str(land_val))
        except OSError:
            os.mkdir("sugar demand val: "+str(land_val))
            os.chdir("sugar demand val: "+str(land_val))
        p=Popen(["python","/media/satyandra/5dae37fd-fa2a-4125-bcc9-48f89f1a9ae6/git/simapi/sugarcane_test.py",str(land_val)],stdout=PIPE,stderr=PIPE)
        out,err=p.communicate()
        print "here"+str(out)
        count1,count2,count3,t1avg_temp,t2avg_temp,t3avg_temp,t1avg_temp_rel,t2avg_temp_rel,t3avg_temp_rel=out.split(",")
        print out
        x=[land_val,t1avg_temp,t2avg_temp,t3avg_temp,t1avg_temp_rel,t2avg_temp_rel,t3avg_temp_rel,count1,count2,count3]
        q.put(x)
        
        os.chdir("..")





try:
    os.chdir("test1_new_values10-6-18")
except OSError:
    os.mkdir("test1_new_values10-6-18")
    os.chdir("test1_new_values10-6-18")
    
q = Queue()
try:
    os.chdir("changing sugar demand value")
except OSError:
    os.mkdir("changing sugar demand value")
    os.chdir("changing sugar demand value")
#msp_st=np.array([1000,1000,1000])
land_val=270
x_axis=[]
t1avg=[]
t2avg=[]
t3avg=[]
t1avg_rel=[]
t2avg_rel=[]
t3avg_rel=[]
count1=[]
count2=[]
count3=[]
p=[]
while land_val<=380:
    if len(p)<12:
        p.append(Process(target=func,args=(q,land_val)))
        p[-1].start()
        land_val+=10
        
    else:
        while True:
            if(len(p)>0 and all(pq.is_alive() for pq in p)):
                time.sleep(10)
            else:
                temp=[pq.is_alive() for pq in p]
                j=0
                while j<len(p):
                    if not temp[j]:
                        p.pop(j)
                        temp.pop(j)
                    else:
                        j+=1
                break
            
while(any(pq.is_alive() for pq in p)):
    time.sleep(5)
    
while(not q.empty()):
    temp=q.get()
    x_axis.append(temp[0])
    t1avg.append(float(temp[1]))
    t2avg.append(float(temp[2]))
    t3avg.append(float(temp[3]))
    t1avg_rel.append(float(temp[4]))
    t2avg_rel.append(float(temp[5]))
    t3avg_rel.append(float(temp[6]))
    count1.append(float(temp[7]))
    count2.append(float(temp[8]))
    count3.append(float(temp[9]))

order=np.argsort(x_axis)
x_axis=[x_axis[i] for i in order]
t1avg=[t1avg[i] for i in order]
t2avg=[t2avg[i] for i in order]
t3avg=[t3avg[i] for i in order]
t1avg_rel=[t1avg_rel[i] for i in order]
t2avg_rel=[t2avg_rel[i] for i in order]
t3avg_rel=[t3avg_rel[i] for i in order]


print x_axis
plt.figure()
plt.plot(x_axis,t1avg)
plt.title("Type 1 savings")
plt.savefig("Type 1 savings")
plt.figure()
plt.plot(x_axis,t2avg)
plt.title("Type 2 savings ")
plt.savefig("Type 2 savings ")
plt.figure()
plt.plot(x_axis,t3avg)
plt.title("Type 3 savings")
plt.savefig("Type 3 savings")
plt.figure()
plt.plot(x_axis,t1avg_rel)
plt.title("Type 1 savings relative to land")
plt.savefig("Type 1 savings relative to land")
plt.figure()
plt.plot(x_axis,t2avg_rel)
plt.title("Type 2 savings relative to land")
plt.savefig("Type 2 savings relative to land")
plt.figure()
plt.plot(x_axis,t3avg_rel)
plt.title("Type 3 savings relative to land")
plt.savefig("Type 3 savings relative to land")
plt.ylim(ymin=0)
plt.figure()
plt.plot(x_axis,count1)
plt.title("ratio of type 1 farmers moving out ")
plt.ylim(ymin=0)
plt.savefig("ratio of type 1 farmers moving out")
plt.figure()
plt.plot(x_axis,count2)
plt.title("ratio of type 2 farmers moving out")
plt.ylim(ymin=0)
plt.savefig("ratio of type 2 farmers moving out")
plt.figure()
plt.plot(x_axis,count3)
plt.title("ratio of type 3 farmers moving out")
plt.ylim(ymin=0)
plt.savefig("ratio of type 3 farmers moving out")
plt.close("all")

        
    

