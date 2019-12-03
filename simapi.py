import simpy
import re
import numpy as np


class stock(object):
    def __init__(self,env,name,initval,generator_function,dependency,interrupt_events,dependency_equations=None,equations=None):
        self.name=name
        self.env=env
        self.return_vals=[]
        self.dependency=dependency
        self.dependency_equations=["" for i in self.dependency]
        self.view_equation=""
        self.equation=""
        self.take_user_input=False
        self.generator_function=generator_function
        if (type(self.generator_function)==str):
            self.generator_function=self.convert_equation(self.generator_function)
        self.interrupt_events=interrupt_events
        self.action=env.process(self.run(initval,equations,dependency_equations))
        self.interrupt=False
        
    def convert_equation(self,string):
        l=re.split("(\$[0-9]+)",string)
        for i in range(len(l)):
            if "new_val" in l[i]:
                if self.generator_function is None:
                    self.take_user_input=True
            if len(l[i])>1:
                if l[i][0]=='$':
                    if (int(l[i][1:])==0):
                        l[i]="self.return_vals[-1]"
                    else:
                        tempstr=str(int(l[i][1:])-1)
                        l[i]="self.dependency["+tempstr+"].return_vals[-1]"
        string1=""
        for i in l:
            string1=string1+i
        return string1
    
    
    def take_equation(self,equations=None):
        if equations is None:        
            print "previous value of "+self.name+" as $0\n"
            print "new value from user as new_val\n"
            for i in range(len(self.dependency)):
                print self.dependency[i].name+" as $"+str(i+1)+"\n"
            self.equation=raw_input("enter the equation using the above dictionary\n")
        else:
            self.equation=equations
        l=re.split("(\$[0-9]+)",self.equation)
        view_l=re.split("(\$[0-9]+)",self.equation)
        for i in range(len(l)):
            if "new_val" in l[i]:
                if self.generator_function is None:
                    self.take_user_input=True
                else:
                    self.take_user_input=False
            if len(l[i])>1:
                if l[i][0]=='$':
                    if (int(l[i][1:])==0):
                        l[i]="self.return_vals[-1]"
                        view_l[i]=self.name
                    else:
                        tempstr=str(int(l[i][1:])-1)
                        l[i]="self.dependency["+tempstr+"].return_vals[-1]"
                        view_l[i]=self.dependency[int(tempstr)].name
        self.equation=""
        self.view_equation=""
        for i in l:
            self.equation=self.equation+i
        for i in view_l:
            self.view_equation=self.view_equation+i
    
    def change_dependencies_and_equations(self):
        print "following are the current dependencies:"
        for i in range(len(self.dependency)):
            print str(i)+" "+self.dependency[i].name
        ret2=raw_input("Enter the number corresponding to the dependency that you want to remove seperated by a comma")
        ret2=ret2.split(',')
        l=[]
        for i in range(len(ret2)):
            if(ret2[i]!=''):
                l.append(int(ret2[i]))
        temp1=[]
        temp2=[]
        l.sort()
        for i in range(len(self.dependency)):
            if(len(l)>0 and i==l[0]):
                l.pop(0)
            else:
                temp1.append(self.dependency[i])
                temp2.append(self.dependency_equations[i])
        self.dependency = temp1
        self.dependency_equations = temp2
        self.equation=""
        self.view_equation=""
        print "equation has been reset please enter the new equation\n\n"
        self.take_equation()
        print "following is the equation\n"+self.view_equation+"\n"
    
    def take_dependence_equations(self,dependency_equations=None):
        if dependency_equations is None:
            if(len(self.dependency)>0):
                print "Please enter how the values of the dependencies change. If they dont then just enter a blank character\n\n"
                print "previous value of "+self.name+" as $0\n"
                print "new value from user as new_val\n"
                for i in range(len(self.dependency)):
                    print self.dependency[i].name+" as $"+str(i+1)+"\n"
                for i in range(len(self.dependency)):
                    temp=raw_input("enter the equation for "+self.dependency[i].name+"\n")
                    l=re.split("(\$[0-9]+)",temp)
                    view_l=re.split("(\$[0-9]+)",temp)
                    for j in range(len(l)):
                        if len(l[j])>1:
                            if l[j][0]=='$':
                                if (int(l[j][1:])==0):
                                    l[j]="self.return_vals[-1]"
                                    view_l[j]=self.name
                                else:
                                    tempstr=str(int(l[j][1:])-1)
                                    l[j]="self.dependency["+tempstr+"].return_vals[-1]"
                                    view_l[j]=self.dependency[int(tempstr)].name
                    temp1=""
                    temp2=""
                    for j in l:
                        temp1=temp1+j
                    for j in view_l:
                        temp2=temp2+j
                    self.dependency_equations[i]=temp1
                    #print "\n\nThe equation for "+self.dependency[i].name+" is "+temp2
            else:
                    print "There  are no dependencies"
        else:
            for i in range(len(self.dependency)):
                    temp=dependency_equations[i]
                    l=re.split("(\$[0-9]+)",temp)
                    view_l=re.split("(\$[0-9]+)",temp)
                    for j in range(len(l)):
                        if len(l[j])>1:
                            if l[j][0]=='$':
                                if (int(l[j][1:])==0):
                                    l[j]="self.return_vals[-1]"
                                    view_l[j]=self.name
                                else:
                                    tempstr=str(int(l[j][1:])-1)
                                    l[j]="self.dependency["+tempstr+"].return_vals[-1]"
                                    view_l[j]=self.dependency[int(tempstr)].name
                    temp1=""
                    temp2=""
                    for j in l:
                        temp1=temp1+j
                    for j in view_l:
                        temp2=temp2+j
                    self.dependency_equations[i]=temp1
                    #print "\n\nThe equation for "+self.dependency[i].name+" is "+temp2
    
    def value(self,i=None):
        if i:
            return self.return_vals[i]
        else:
            return self.return_vals

    def run(self,initval,equation,dependency_equations):
        take_user_input=False
        #print "\n\n\n\n     "+self.name+"\n\n\n"
        self.take_equation(equation)
        #print "following is the equation\n"+self.view_equation+"\n"
        self.take_dependence_equations(dependency_equations)
        
        while True:
            #print "\n\n     "+self.name+"\n\n"
            if(len(self.return_vals)==0):
                self.return_vals.append(initval)
                yield self.env.timeout(1)
                continue
            new_val=0
            if self.take_user_input:
                for i in range(len(self.dependency)):
                    print "\n\n     "+self.name+"\n\n"
                    print "\nvalue of "+self.dependency[i].name+" is "+str(self.dependency[i].return_vals[-1])
                temp=raw_input("Enter value for new_val of "+self.name)
                try:
                    new_val=int(temp)
                except ValueError:
                    new_val=float(temp)
            else:
                if self.generator_function is not None:
                    if(type(self.generator_function)==str):
                        new_val=eval(self.generator_function)
                    elif(type(self.generator_function)==int or type(self.generator_function)==float or type(self.generator_function)==np.ndarray):
                        new_val=self.generator_function
                    elif(callable(self.generator_function)):
                        new_val=self.generator_function()
            
            if (len(self.interrupt_events)>0 or self.interrupt):
                l=[]
                interrupt_now=False
                if self.interrupt:
                    interrupt_now=True
                    self.interrupt=False
                else:
                    for i in range(len(self.interrupt_events)):
                        if(type(self.interrupt_events[i])==str):
                            if(eval(self.interrupt_events[i])):
                                interrupt_now=True
                        else:
                            if(self.interrupt_events[i]()):
                                interrupt_now=True
                if(interrupt_now):
                    ret=raw_input("\n\nDo you want to make changes int the current equations or dependencies?[y/n]")
                    if ret=='y':
                        self.change_dependencies_and_equations()
                    ret=raw_input("\n\nDo you wish to change how the dependency values change?[y/n]")
                    if ret=='y':
                        self.take_dependence_equations()
                    ret=raw_input("\n\nDo you wish to enter manual input?[y/n]")
                    if ret=='y':
                        temp=raw_input("Enter value for new_val of "+self.name)
                        try:
                            new_val=int(temp)
                        except ValueError:
                            new_val=float(temp)
            
            
            temp=eval(self.equation)
            for i in range(len(self.dependency_equations)):
                if (type(self.dependency_equations[i])==str) and self.dependency_equations[i]!="":
                    self.dependency[i].return_vals[-1]=eval(self.dependency_equations[i])
                elif callable(self.dependency_equations[i]):
                    self.dependency[i].return_vals[-1]=self.dependency_equations[i]()
            #print self.name+" now: "+str(self.env.now)+" "+str(temp)
            self.return_vals.append(temp)
            '''try:
                if((self.env.now==0) and(len(self.dependency)>0)):
                    while True:
                        ret=raw_input("\n\nDo u want to make changes in the dependencies of current dependencies? [y/n]")
                        if ret=='n':
                            break
                        for i in range(len(self.dependency)):
                            print self.dependency[i].name+" as "+str(i)+"\n"
                        try:
                            ret=int(raw_input("\n\nEnter your choice"))
                        except ValueError:
                            print "Invalid choice"
                            continue
                        self.dependency[ret].interrupt=True
                        print("\n\n"+self.dependency[ret].name+" has been interrupted changes to the equation can be made later.\n")
                        if self in self.dependency[ret].dependency:
                            print ("\n\n"+self.name+" is already in the dependency of "+ self.dependency[ret].name+" you can make changes when the process is invoked")
                        else:
                            ret2=raw_input("\n\nDo you want to add "+self.name+" as a dependency for "+self.dependency[ret].name+" [y/n]")
                            if ret2=='y':
                                self.dependency[ret].dependency.append(self)
                                self.dependency[ret].dependency_equations.append("")
                yield self.env.timeout(1)
            except simpy.Interrupt:
                ret=raw_input ("\n\n"+self.name+" was interrupted do you want to change the dependencies? [y/n]")
                if ret=='y':
                    self.change_dependencies_and_equations()
                ret=raw_input("\n\nDo you wish to change how the dependency values change?[y/n]")
                if ret=='y':
                    self.take_dependence_equations()
                yield self.env.timeout(1)'''
            yield self.env.timeout(1)
