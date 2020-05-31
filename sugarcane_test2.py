import simpy
import simapi as simapi
import random
import os
import math
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process
from sklearn import datasets, linear_model
import time
'''
Parameters for stock are as follows
env,name,initial value,generator_function,dependency,interrupt_events,dependency_equations=None,equations=None
'''




'''
no dependency between type 1 and 3
'''

class type_1_farmer():
    
    def __init__(self,env,name,savings,land,msp,crop,labour,water_details,family_size):
        self.name=name
        self.env=env
        self.family_size=family_size
        self.per_person_charge=10000*pow((1.001),self.env.now)
        self.interest=0.005
        self.initial_land=land
        self.water_details=water_details
        self.loan_duration=240
        self.higher_interest=0.010
        self.higher_loan_duration=240
        self.type_of_crop=-1
        self.installments=np.zeros(2,dtype=float)
        self.safety_buffer=land*100
        self.land=land
        self.collateral=0
        self.land_value=1000
        self.land_allocated=0
        self.when_planted=-1
        self.number_of_defaults_allowed=4
        self.water_lender=None
        self.water=10
        self.credit_rating=1000
        self.credit_penalty=300
        self.rural_credit=True
        self.credit_rating_raise=1
        self.upper_limit_on_land=np.array([0.9*self.land,0.9*self.land,0.9*self.land])
        self.estimated_produce=[]
        self.water_required=[]
        self.crop_type=[]
        self.quality=1
        self.wages=[10]
        self.labour_need=[0]
        self.water_price=10
        self.water_received=0
        self.land_allocated=0
        self.allocated=0
        self.crop=crop
        self.type_of_crop_list=[]
        self.storage_money=0
        self.labour_details=labour
        self.income_expectation=[0,0,0]
        self.income_expectation[:]=msp
        for i in range(len(self.income_expectation)):
            self.income_expectation[i]=max(self.income_expectation[i],self.crop.price()[i])
        self.people_quality=1
        self.adjusting_land_allocated=simapi.stock(env,name+" adjusting land allocation",0,self.adjust_land,[],[],[],"$0")
        self.income_exp=simapi.stock(env,name+" produce",0,self.income_expectaion_change,[],[],[],"new_val")
        self.total_savings=simapi.stock(env,name+" total savings",savings,self.total_savings_gen,[],[],[],"$0-new_val")
        #self.plantation_cost=simapi.stock(env,name+"planting cost",0,self.choose_crop,[self.total_savings],[],["$1-new_val"],"new_val")
        self.loan=simapi.stock(env,name+" loan",np.zeros(2,dtype=float),self.apply_loan,[self.total_savings],[],["$1+sum(new_val)"],"$0+new_val")
        self.water_pay=simapi.stock(env,name+" water_pay",0,self.water_payment,[self.total_savings],[],["$1-new_val"],"new_val")
        self.labour=simapi.stock(env,name+" labour",0,self.labour_gen,[self.total_savings],[],["$1-new_val"],"new_val")
        self.premium=simapi.stock(env,name+" premium",0,self.pay_installments,[self.total_savings,self.loan],[],["$1-new_val[0]","$2-new_val[1]"],"new_val[0]")
        self.stock_to_be_sold=simapi.stock(env,name+" stock to be sold",np.array([0,0,0],dtype=float),self.returns_gen,[],[],[],"new_val")
        self.produce=simapi.stock(env,name+" produce",np.array([0,0,0],dtype=float),self.returns_gen,[],[],[],"new_val")
        self.sold=simapi.stock(env,name+" sold",np.array([0,0,0],dtype=float),np.array([0,0,0],dtype=float),[],[],[],"new_val")
        self.temp_sold=simapi.stock(env,name+" temp sold",np.array([0,0,0],dtype=float),np.array([0,0,0],dtype=float),[],[],[],"new_val")
        self.end_crop=simapi.stock(env,name+" end_crop",0,self.end_cycle,[],[],[],"$0")
        self.storage=[[],[],[]]
        self.income=simapi.stock(env,name+" end_crop",0,0,[],[],[],"new_val")
        self.calculate_water_need_per_crop=simapi.stock(env,name+" calculate_water_need_per_crop",0,self.calc_water_req,[],[],[],"new_val")
    
    
    def income_expectaion_change(self):
        if self.type_of_crop!=-1:
            if (self.env.now-self.when_planted)%self.crop.harvest_cycle()[self.type_of_crop]==1 and self.env.now>=self.when_planted+self.crop.harvest_cycle()[self.type_of_crop]:
                self.income_expectation[self.type_of_crop]=sum(self.income.value()[-1*self.crop.harvest_cycle()[self.type_of_crop]:])/sum(i[self.type_of_crop]+0.00001 for i in self.produce.value()[-1*self.crop.harvest_cycle()[self.type_of_crop]:])
    
    
    def total_savings_gen(self):
        self.per_person_charge*=1.001
        if self.family_size*self.per_person_charge>self.total_savings.value(-1):
            self.people_quality*=self.total_savings.value(-1)/(self.family_size*self.per_person_charge)
        else:
            self.people_quality=1
            
        return min(self.family_size*self.per_person_charge,self.total_savings.value(-1))-0.0001*self.total_savings.value(-1)
    
    def end_cycle(self):
        if self.type_of_crop!=-1:
            if(((self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])==0 and self.env.now>self.when_planted):
                #x=np.array([i for i in range(self.crop.cycle()[self.type_of_crop])])
                #x=x.reshape(-1,1)
                #y=np.array([self.total_savings.value()[-1*self.crop.cycle()[self.type_of_crop]:]])
                #y=y.reshape(-1,1)
                #regr=linear_model.LinearRegression()
                #regr.fit(x,y)
                #if(regr.coef_[0][0]>0):                        
                    #self.upper_limit_on_land[self.type_of_crop]=max(self.land_allocated[self.type_of_crop]*max(1+(math.atan(regr.coef_[0][0])*2/math.pi),1.9),self.upper_limit_on_land[self.type_of_crop])
                    #self.upper_limit_on_land[self.type_of_crop]=min(self.upper_limit_on_land[self.type_of_crop],self.land)
                    
                #else:
                    #self.upper_limit_on_land[self.type_of_crop]=max(1+(math.atan(regr.coef_[0][0])*2/math.pi),0.1)*self.land_allocated[self.type_of_crop]
                    #self.upper_limit_on_land[self.type_of_crop]=max(self.upper_limit_on_land[self.type_of_crop],0.1*self.land) 
                #self.land_allocated[self.type_of_crop]=0
                #if(self.water_received>0 and self.water_lender is not None):
                    #self.water_lender.excess_water+=self.water_received
                    #self.water_received=0
                    #self.water_lender=None
                self.land_allocated[self.type_of_crop]=0
                if(self.water_received>0 and self.water_lender is not None):
                    self.water_lender.excess_water+=self.water_received
                    self.water_received=0
                    self.water_lender=None
                self.type_of_crop=-1
                self.when_planted=-1
        return 0

    
    def water_payment(self):
        self.water_price*=1.00001
        if self.when_planted==-1:
            return 0
        
        if (self.type_of_crop==-1):
            return 0
        else:
            if((self.total_savings.value(-1)-self.per_person_charge*self.family_size)<max(self.crop.water()[self.type_of_crop]-self.water,0)*self.water_price*self.land_allocated[self.type_of_crop]):
                self.quality*=max(min(((self.total_savings.value(-1)-self.family_size*self.per_person_charge)/(max(self.crop.water()[self.type_of_crop]-self.water,1)*self.water_price*self.land_allocated[self.type_of_crop]))/1.2,1),0)
            return min(max(self.crop.water()[self.type_of_crop]-self.water,0)*self.water_price*self.land_allocated[self.type_of_crop],(self.total_savings.value(-1)-self.family_size*self.per_person_charge))
                       
                       
                       
    def adjust_land(self):
        self.type_of_crop_list.append(str(self.type_of_crop)+" "+str(self.land_allocated)+" "+str(self.land))
        if self.type_of_crop!=-1:
            if len(self.sold.value())>=self.crop.cycle()[self.type_of_crop] and self.type_of_crop!=-1 and self.env.now==self.when_planted+self.crop.cycle()[self.type_of_crop]:
                x=np.array([i for i in range(self.crop.cycle()[self.type_of_crop])])
                x=x.reshape(-1,1)
                y=np.array([self.total_savings.value()[-1*self.crop.cycle()[self.type_of_crop]:]])
                y=y.reshape(-1,1)
                regr=linear_model.LinearRegression()
                regr.fit(x,y)
                if(regr.coef_[0][0]>0):                        
                    self.upper_limit_on_land[self.type_of_crop]=max(self.land_allocated[self.type_of_crop]*max(1+(math.atan(regr.coef_[0][0])*2/math.pi),1.9),self.upper_limit_on_land[self.type_of_crop])
                    self.upper_limit_on_land[self.type_of_crop]=min(self.upper_limit_on_land[self.type_of_crop],self.land)
                    
                else:
                    self.upper_limit_on_land[self.type_of_crop]=max(1+(math.atan(regr.coef_[0][0])*2/math.pi),0.1)*self.land_allocated[self.type_of_crop]
                    self.upper_limit_on_land[self.type_of_crop]=max(self.upper_limit_on_land[self.type_of_crop],0.1*self.land) 
                    

    def calc_water_req(self):
        if(self.type_of_crop==-1 or ((self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])==0):
            self.land_allocated=[0 for i in range(len(self.crop.types()))]
            self.estimated_produce=[]
            self.water_required=[]
            self.crop_type=[]
            profit=[]
            for i in range(len(self.crop.types())):
                
                self.land_allocated[i]+=max(min(0.7*(self.total_savings.value(-1)-self.safety_buffer-self.per_person_charge*self.family_size*self.crop.harvest_cycle()[i])/(self.crop.initial_cost()[i]+self.crop.labour_need()[i]*self.crop.cycle()[i]*self.labour_details.wages()[i]+max(self.crop.water()[i]-self.water,0)*self.water_price),self.upper_limit_on_land[i]),0)
                if(self.loan.value(-1)[0]==0):
                    self.land_allocated[i]+=max(min(0.7*0.8*(self.total_savings.value(-1)-self.safety_buffer-self.per_person_charge*self.family_size*self.crop.harvest_cycle()[i])/(self.crop.initial_cost()[i]+self.crop.labour_need()[i]*self.crop.cycle()[i]*self.labour_details.wages()[i]+max(self.crop.water()[i]-self.water,0)*self.water_price),self.upper_limit_on_land[i]),0)
                if(self.loan.value(-1)[1]==0):
                    self.land_allocated[i]+=min(0.7*self.land*self.land_value/(self.crop.initial_cost()[i]+self.crop.labour_need()[i]*self.crop.cycle()[i]*self.labour_details.wages()[i]+max(self.crop.water()[i]-self.water,0)*self.water_price),self.upper_limit_on_land[i]-self.land_allocated[i])
                self.land_allocated[i]=min(self.upper_limit_on_land[i],self.land_allocated[i])
                self.water_required.append(max(self.crop.water()[i]-self.water,0)*self.land_allocated[i])
                self.estimated_produce.append(self.land_allocated[i]*self.crop.returns()[i]*math.floor(self.crop.cycle()[i]/self.crop.harvest_cycle()[i]))
                self.crop_type.append(i)
                profit.append((self.crop.cycle()[i]/self.crop.harvest_cycle()[i])*self.crop.returns()[i]*self.land_allocated[i]*self.income_expectation[i]-(self.land_allocated[i]*(self.crop.initial_cost()[i]+(max(self.crop.water()[i]-self.water,0)*self.water_price)*self.crop.cycle()[i]+self.crop.labour_need()[i]*self.crop.cycle()[i]*self.labour_details.wages()[i])))
            self.estimated_produce=[self.estimated_produce[j] for j in np.flip(np.argsort(profit),0)]
            self.crop_type=[self.crop_type[j] for j in np.flip(np.argsort(profit),0)]
            self.water_required=[self.water_required[j] for j in np.flip(np.argsort(profit),0)]
            return np.zeros(2,dtype=float)
        else:
            return np.zeros(2,dtype=float)
                
            
    def plantation(self):
        if self.type_of_crop!=-1:
            self.when_planted=self.env.now
            self.quality=1
            prod=self.crop.initial_cost()[self.type_of_crop]*self.land_allocated[self.type_of_crop]
            if prod>(self.total_savings.value(-1)-self.per_person_charge*self.family_size*self.crop.harvest_cycle()[self.type_of_crop]):
                ans=np.zeros(2,dtype=float)
                if self.loan.value(-1)[0]==0 and (self.total_savings.value(-1)-1.2*self.per_person_charge*self.family_size*self.crop.harvest_cycle()[self.type_of_crop])<((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop])):
                    ans[0]=min((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop])+self.safety_buffer,0.8*self.total_savings.value(-1))
                if self.loan.value(-1)[1]==0 and (ans[0]+(self.total_savings.value(-1)-self.per_person_charge*self.family_size*self.crop.harvest_cycle()[self.type_of_crop]))<((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop])):
                    ans[1]=(self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop])+self.safety_buffer-ans[0]
                    self.number_of_defaults_allowed=4
                    #print self.name+" ans1 value: "+str(ans[1])
                    self.collateral=ans[1]/self.land_value
                self.installments=ans*(self.interest*pow(1+self.interest,self.loan_duration)/(pow(1+self.interest,self.loan_duration)-1))
                self.total_savings.return_vals[-1]+=sum(ans)
                self.loan.return_vals[-1]+=ans
                self.total_savings.return_vals[-1]+=sum(ans)
            self.total_savings.return_vals[-1]-=prod
            return self.crop.initial_cost()[self.type_of_crop]*self.land_allocated[self.type_of_crop]
        else:
            return 0        
        
    def labour_gen(self):
        if self.when_planted==-1:
            return 0
        
        if (self.type_of_crop==-1):
            return 0
        else:
            if((self.total_savings.value(-1)-self.per_person_charge*self.family_size)<self.crop.labour_need()[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]*self.land_allocated[self.type_of_crop]):
                self.quality*=max(min(((self.total_savings.value(-1)-self.family_size*self.per_person_charge)/self.crop.labour_need()[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]*self.land_allocated[self.type_of_crop])/1.2,1),0)
            return min(self.crop.labour_need()[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]*self.land_allocated[self.type_of_crop],(self.total_savings.value(-1)-self.family_size*self.per_person_charge))
            #return self.allocated*self.wages[self.type_of_crop]
    
    def apply_loan(self):
        if(self.type_of_crop!=-1):
            if self.when_planted>0:
                ans=np.zeros(2,dtype=float)
                if self.loan.value(-1)[0]==0 and (self.total_savings.value(-1)-self.family_size*self.per_person_charge*self.crop.harvest_cycle()[self.type_of_crop])<((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]+max(self.crop.water()[self.type_of_crop]-self.water,0)*self.water_price)):
                    ans[0]=min((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]+max(self.crop.water()[self.type_of_crop]-self.water,0)*self.water_price)+self.safety_buffer,0.8*self.total_savings.value(-1))
                if self.loan.value(-1)[1]==0 and (ans[0]+(self.total_savings.value(-1)-self.family_size*self.per_person_charge*self.crop.harvest_cycle()[self.type_of_crop]))<((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]+max(self.crop.water()[self.type_of_crop]-self.water,0)*self.water_price)):
                    ans[1]=(self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]+max(self.crop.water()[self.type_of_crop]-self.water,0)*self.water_price)+self.safety_buffer-ans[0]
                    self.number_of_defaults_allowed=4
                    #print self.name+" ans1 value: "+str(ans[1])
                    self.collateral=ans[1]/self.land_value
                    self.installments=ans*(self.interest*pow(1+self.interest,self.loan_duration)/(pow(1+self.interest,self.loan_duration)-1))
                    return ans
                else:
                    return np.zeros(2,dtype=float)
            else:
                return np.zeros(2,dtype=float)
        else:
            ans=np.zeros(2,dtype=float)
            if self.loan.value(-1)[0]==0 and (self.total_savings.value(-1)-self.family_size*self.per_person_charge):
                    ans[0]=0.8*self.total_savings.value(-1)
            if self.loan.value(-1)[1]==0 and (ans[0]+(self.total_savings.value(-1)-self.family_size*self.per_person_charge)):
                ans[1]=self.land_value*self.land/2
                self.number_of_defaults_allowed=4
                #print self.name+" ans1 value: "+str(ans[1])
                self.collateral=self.land/2
                self.installments=ans*(self.interest*pow(1+self.interest,self.loan_duration)/(pow(1+self.interest,self.loan_duration)-1))
            return ans
        
    def pay_installments(self):
        new_val=[0,np.zeros(2,dtype=float)]
        
        if((self.total_savings.value(-1)-self.family_size*self.per_person_charge)>self.installments[1]+self.loan.value(-1)[1]):
            new_val[0]+=self.installments[1]+self.loan.value(-1)[1]
            new_val[1][1]+=self.loan.value(-1)[1]
            self.installments[1]=0
            if (self.total_savings.value(-1)-self.family_size*self.per_person_charge)-new_val[0]>(self.installments[0]+self.loan.value(-1)[0]):
                new_val[0]+=self.installments[0]+self.loan.value(-1)[0]
                new_val[1][0]+=self.loan.value(-1)[0]
            return new_val
        else:
            
            if (self.total_savings.value(-1)-self.family_size*self.per_person_charge)>=self.installments[1]:
                new_val[0]+=self.installments[1]
                new_val[1][1]+=self.installments[1]-self.loan.value(-1)[1]*self.interest
                if (self.total_savings.value(-1)-self.family_size*self.per_person_charge)-new_val[0]>=(self.installments[0]):
                    new_val[0]+=self.installments[0]
                    new_val[1][0]+=self.installments[0]-self.loan.value(-1)[0]*self.interest
                return new_val
            else:
                #print self.name + " "+str(self.number_of_defaults_allowed)+" "+str(self.collateral)
                if self.number_of_defaults_allowed==0:
                    self.land=max(0,self.land-self.collateral)
                    self.land_allocated=np.minimum(self.land_allocated,self.land)
                    self.upper_limit_on_land=np.minimum(self.upper_limit_on_land,self.land)
                    self.total_savings.return_vals[-1]+=max(self.collateral*self.land_value-self.loan.return_vals[-1][1],0)
                    self.collateral=0
                    new_val[0]=0
                    new_val[1][1]=0
                    self.loan.return_vals[-1][1]=0
                    self.installments[1]=0
                self.number_of_defaults_allowed-=1
                if (self.total_savings.value(-1)-self.family_size*self.per_person_charge)-new_val[0]>(self.installments[0]+self.loan.value(-1)[0]):
                    new_val[0]+=self.installments[0]+self.loan.value(-1)[0]
                    new_val[1][0]+=self.loan.value(-1)[0]
                elif (self.total_savings.value(-1)-self.family_size*self.per_person_charge)-new_val[0]>=(self.installments[0]):
                    new_val[0]+=self.installments[0]
                    new_val[1][0]+=self.installments[0]-self.loan.value(-1)[0]*self.interest
                else:
                    new_val[0]=0
                    new_val[1][0]=-self.loan.value(-1)[0]*self.interest
                return new_val
            
    def returns_gen(self):
        ans=[]
        self.quality=min(self.quality,1)
        for i in range(len(self.crop.types())):
            if(self.when_planted>0and i==self.type_of_crop and (self.env.now-self.when_planted)%self.crop.harvest_cycle()[i]==0):
                ans.append(self.crop.returns()[self.type_of_crop]*self.land_allocated[self.type_of_crop]*self.quality)
            else:
                ans.append(0)
        self.storage_money=0.2*self.total_savings.value(-1)
        return np.array(ans,dtype=float)

        
    def plot(self,k):
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),self.total_savings.value()))
        plt.title("type1 farmer total savings")
        plt.savefig(self.name+"totsav.png")
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),self.loan.value()))
        plt.title(self.name+" loan")
        plt.savefig(self.name+" loan"+".png")
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),self.premium.value()))
        plt.title(self.name+" premium")
        plt.savefig(self.name+" premium"+".png")
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),[z[0] for z in self.produce.value()]))
        plt.title(self.name+" stock to be sold type 0")
        plt.savefig(self.name+" stock to be sold type 0"+".png")
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),self.income.value()))
        plt.title(self.name+" income")
        plt.savefig(self.name+" income"+".png")
        plt.close()
        
        

class type_2_farmer():
    
    def __init__(self,env,name,savings,land,msp,crop,labour,stored,family_size):
        self.name=name
        self.env=env
        self.initial_land=land
        self.family_size=family_size
        self.per_person_charge=50000*pow((1.001),env.now)
        self.interest=0.005
        self.loan_duration=240
        self.stored=stored
        self.type_of_crop=-1
        self.installments=np.zeros(2,dtype=float)
        self.safety_buffer=1000
        self.collateral=0
        self.number_of_defaults_allowed=4
        self.land=land
        self.land_value=1000
        self.storage_price=[10]
        self.allocated=0
        self.land_allocated=0
        self.upper_limit_on_land=[0.9*self.land,0.9*self.land,0.9*self.land]
        self.wages=[10]
        self.labour_need=[0]
        self.storage_money=0
        self.crop=crop
        self.storedinfo=False
        self.quality=1
        self.when_planted=0
        self.labour_details=labour
        self.income_expectation=[0,0,0]
        self.income_expectation[:]=msp
        self.people_quality=1
        self.type_of_crop_list=[]
        self.adjusting_land_allocated=simapi.stock(env,name+" adjusting land allocation",0,self.adjust_land,[],[],[],"$0")
        self.income_exp=simapi.stock(env,name+" produce",0,self.income_expectaion_change,[],[],[],"new_val")
        self.total_savings=simapi.stock(env,name+" total savings",savings,self.total_savings_gen,[],[],[],"$0-new_val")
        self.plantation_cost=simapi.stock(env,name+"planting cost",0,self.choose_crop,[self.total_savings],[],["$1-new_val"],"new_val")
        self.loan=simapi.stock(env,name+" loan",np.zeros(2,dtype=float),self.apply_loan,[self.total_savings],[],["$1+sum(new_val)"],"$0+new_val")
        self.labour=simapi.stock(env,name+" labour",0,self.labour_gen,[self.total_savings],[],["$1-new_val"],"new_val")
        self.premium=simapi.stock(env,name+" premium",0,self.pay_installments,[self.total_savings,self.loan],[],["$1-new_val[0]","$2-new_val[1]"],"new_val[0]")
        self.stock_to_be_sold=simapi.stock(env,name+" stock to be sold",np.array([0,0,0],dtype=float),self.returns_gen,[self.plantation_cost],[],[""],"new_val")
        self.produce=simapi.stock(env,name+" produce",np.array([0,0,0],dtype=float),self.returns_gen,[],[],[],"new_val")
        self.storage=[[],[],[]]
        self.income=simapi.stock(env,name+" end_crop",0,0,[],[],[],"new_val")
        self.sold=simapi.stock(env,name+" sold",np.array([0],dtype=float),np.array([0,0,0],dtype=float),[],[],[],"new_val")
        self.temp_sold=simapi.stock(env,name+" temp sold",np.array([0,0,0],dtype=float),np.array([0,0,0],dtype=float),[],[],[],"new_val")
        
        
        
        
    def income_expectaion_change(self):
        if self.type_of_crop!=-1:
            if (self.env.now-self.when_planted)%self.crop.harvest_cycle()[self.type_of_crop]==1 and self.env.now>=self.when_planted+self.crop.harvest_cycle()[self.type_of_crop]:
                self.income_expectation[self.type_of_crop]=sum(self.income.value()[-1*self.crop.harvest_cycle()[self.type_of_crop]:])/sum(i[self.type_of_crop] for i in self.produce.value()[-1*self.crop.harvest_cycle()[self.type_of_crop]:])
        
    def total_savings_gen(self):
        self.per_person_charge*=1.001
        if self.family_size*self.per_person_charge>self.total_savings.value(-1):
            self.people_quality*=self.total_savings.value(-1)/(self.family_size*self.per_person_charge)
        else:
            self.people_quality=1
            
        return min(self.family_size*self.per_person_charge,self.total_savings.value(-1))-0.0001*self.total_savings.value(-1)
    
    
    
    def adjust_land(self):
        self.type_of_crop_list.append(str(self.type_of_crop)+" "+str(self.land_allocated)+" "+str(self.land))
        if self.type_of_crop!=-1:
            if len(self.sold.value())>=self.crop.cycle()[self.type_of_crop] and self.type_of_crop!=-1 and self.env.now==self.when_planted+self.crop.cycle()[self.type_of_crop]:
                x=np.array([i for i in range(self.crop.cycle()[self.type_of_crop])])
                x=x.reshape(-1,1)
                y=np.array([self.total_savings.value()[-1*self.crop.cycle()[self.type_of_crop]:]])
                y=y.reshape(-1,1)
                regr=linear_model.LinearRegression()
                regr.fit(x,y)
                if(regr.coef_[0][0]>0):                        
                    self.upper_limit_on_land[self.type_of_crop]=max(self.land_allocated*max(1+(math.atan(regr.coef_[0][0])*2/math.pi),1.9),self.upper_limit_on_land[self.type_of_crop])
                    self.upper_limit_on_land[self.type_of_crop]=min(self.upper_limit_on_land[self.type_of_crop],self.land)
                    
                else:
                    self.upper_limit_on_land[self.type_of_crop]=max((1+(math.atan(regr.coef_[0][0])*2/math.pi))*self.land_allocated,10)
                    self.upper_limit_on_land[self.type_of_crop]=min(self.upper_limit_on_land[self.type_of_crop],10) 

        
    def choose_crop(self):
        ans =-1
        if(self.type_of_crop==-1 or ((self.plantation_cost.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])==0):
            self.quality=1
            self.land_allocated=0
            land_allocated=[0,0,0]
            self.type_of_crop=ans
            supply=np.ones(len(self.crop.types()),dtype=float)
            if self.storedinfo:
                for i in range(len(self.crop.types())):
                    for j in self.stored[i]:
                        supply[i]+=j[0]
                        
                        
            for i in range(len(self.crop.types())):
                land_allocated[i]+=max(min(0.9*self.total_savings.value(-1)/(self.crop.initial_cost()[i]+self.crop.labour_need()[i]*self.crop.cycle()[i]*self.labour_details.wages()[i]),self.upper_limit_on_land[i]),0)
                if self.loan.value(-1)[0]==0:
                    land_allocated[i]+=max(0.9*0.8*self.total_savings.value(-1)/(self.crop.initial_cost()[i]+self.crop.labour_need()[i]*self.crop.cycle()[i]*self.labour_details.wages()[i]),0)
                if(self.loan.value(-1)[1]==0):
                    land_allocated[i]+=max(min(0.7*self.land*self.land_value/(self.crop.initial_cost()[i]+self.crop.labour_need()[i]*self.crop.cycle()[i]*self.labour_details.wages()[i]),self.upper_limit_on_land[i]-land_allocated[i]),0)
                    self.number_of_defaults_allowed=4
                if self.storedinfo:
                    land_allocated[i]=max(min((self.crop.export_quantity()[i]+self.need.value(-1)[i]+self.storage_details.capacity()[i]-supply[i])/self.crop.returns()[i],land_allocated[i]),0)
                else:
                    land_allocated[i]=max(min(0.7*(self.crop.export_quantity()[i]+self.need.value(-1)[i])/self.crop.returns()[i],land_allocated[i]),0)
                land_allocated[i]=min(self.upper_limit_on_land[i],land_allocated[i])
            
            if self.storedinfo:
                priority=np.array(self.crop.price())/supply
                ans=np.argmax(priority)
            else:
                priority=(np.array(self.income_expectation)*land_allocated*self.crop.returns()*np.array(self.crop.cycle())/np.array(self.crop.harvest_cycle()))-((np.array(self.crop.initial_cost())+np.array(self.crop.labour_need())*np.array(self.crop.cycle())*np.array(self.labour_details.wages()))*land_allocated)
                temp=(np.array(self.income_expectation)*land_allocated*self.crop.returns()*np.array(self.crop.cycle())/np.array(self.crop.harvest_cycle()))-((np.array(self.crop.initial_cost())+np.array(self.crop.labour_need())*np.array(self.crop.cycle())*np.array(self.labour_details.wages()))*land_allocated)
                #print self.crop.price()
                ans=np.argmax(priority)
                if temp[ans]<1.5*((np.array(self.crop.initial_cost())[ans]+np.array(self.crop.labour_need())[ans]*np.array(self.crop.cycle())[ans]*np.array(self.labour_details.wages())[ans])*land_allocated[ans]) or temp[ans]<0:
                    ans=-1
            
            if ans!=-1:
                self.when_planted=self.plantation_cost.env.now
                self.land_allocated=land_allocated[ans]
                self.type_of_crop=ans
                ans1=ans
                ans=np.zeros(2,dtype=float)
                if self.loan.value(-1)[0]==0 and (self.total_savings.value(-1)-self.family_size*self.per_person_charge*self.crop.harvest_cycle()[self.type_of_crop])<((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])):
                    ans[0]=min((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])+self.safety_buffer,0.8*self.total_savings.value(-1))
                if self.loan.value(-1)[1]==0 and (ans[0]+(self.total_savings.value(-1)-self.family_size*self.per_person_charge*self.crop.harvest_cycle()[self.type_of_crop]))<((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])):
                    ans[1]=(self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])+self.safety_buffer-ans[0]
                    self.number_of_defaults_allowed=4
                    #print self.name+" ans1 value: "+str(ans[1])
                    self.collateral=ans[1]/self.land_value
                self.installments=ans*(self.interest*pow(1+self.interest,self.loan_duration)/(pow(1+self.interest,self.loan_duration)-1))
                self.total_savings.return_vals[-1]+=sum(ans)
                self.loan.return_vals[-1]+=ans
                self.total_savings.return_vals[-1]+=sum(ans)
                return self.crop.initial_cost()[ans1]*land_allocated[ans1]
            else:
                return 0
            '''self.quality=1
            self.type_of_crop=1
            return self.crop.initial_cost()[1]*self.land'''
        else:
            return 0
        
        
    def labour_gen(self):
        
        if (self.type_of_crop==-1):
            return 0
        else:
            if(self.total_savings.value(-1)<self.crop.labour_need()[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]*self.land_allocated):
                self.quality*=(self.total_savings.value(-1)/self.crop.labour_need()[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]*self.land_allocated)
            return min(self.crop.labour_need()[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]*self.land_allocated,self.total_savings.value(-1))
            '''print self.allocated
            return self.allocated*self.wages[self.type_of_crop]'''
    
    def apply_loan(self):
        if(self.type_of_crop!=-1):
            if self.when_planted>0:
                ans=np.zeros(2,dtype=float)
                if self.loan.value(-1)[0]==0 and (self.total_savings.value(-1)-self.family_size*self.per_person_charge*self.crop.harvest_cycle()[self.type_of_crop])<((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])):
                    ans[0]=min((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])+self.safety_buffer,0.8*self.total_savings.value(-1))
                if self.loan.value(-1)[1]==0 and (ans[0]+(self.total_savings.value(-1)-self.family_size*self.per_person_charge*self.crop.harvest_cycle()[self.type_of_crop]))<((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])):
                    ans[1]=(self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])+self.safety_buffer-ans[0]
                    self.number_of_defaults_allowed=4
                    #print self.name+" ans1 value: "+str(ans[1])
                    self.collateral=ans[1]/self.land_value
                    self.installments=ans*(self.interest*pow(1+self.interest,self.loan_duration)/(pow(1+self.interest,self.loan_duration)-1))
                    return ans
                else:
                    return np.zeros(2,dtype=float)
            else:
                return np.zeros(2,dtype=float)
        else:
            ans=np.zeros(2,dtype=float)
            if self.loan.value(-1)[0]==0 and (self.total_savings.value(-1)-self.family_size*self.per_person_charge):
                    ans[0]=0.8*self.total_savings.value(-1)
            if self.loan.value(-1)[1]==0 and (ans[0]+(self.total_savings.value(-1)-self.family_size*self.per_person_charge)):
                ans[1]=self.land_value*self.land/2
                self.number_of_defaults_allowed=4
                #print self.name+" ans1 value: "+str(ans[1])
                self.collateral=self.land/2
                self.installments=ans*(self.interest*pow(1+self.interest,self.loan_duration)/(pow(1+self.interest,self.loan_duration)-1))
            return ans
        
    def pay_installments(self):
        new_val=[0,np.zeros(2,dtype=float)]
        
        if((self.total_savings.value(-1)-self.family_size*self.per_person_charge)>self.installments[1]+self.loan.value(-1)[1]):
            new_val[0]+=self.installments[1]+self.loan.value(-1)[1]
            new_val[1][1]+=self.loan.value(-1)[1]
            self.installments[1]=0
            if (self.total_savings.value(-1)-self.family_size*self.per_person_charge)-new_val[0]>(self.installments[0]+self.loan.value(-1)[0]):
                new_val[0]+=self.installments[0]+self.loan.value(-1)[0]
                new_val[1][0]+=self.loan.value(-1)[0]
            return new_val
        else:
            
            if (self.total_savings.value(-1)-self.family_size*self.per_person_charge)>=self.installments[1]:
                new_val[0]+=self.installments[1]
                new_val[1][1]+=self.installments[1]-self.loan.value(-1)[1]*self.interest
                if (self.total_savings.value(-1)-self.family_size*self.per_person_charge)-new_val[0]>=(self.installments[0]):
                    new_val[0]+=self.installments[0]
                    new_val[1][0]+=self.installments[0]-self.loan.value(-1)[0]*self.interest
                return new_val
            else:
                #print self.name + " "+str(self.number_of_defaults_allowed)+" "+str(self.collateral)
                if self.number_of_defaults_allowed==0:
                    self.land=max(0,self.land-self.collateral)
                    self.land_allocated=np.minimum(self.land_allocated,self.land)
                    self.upper_limit_on_land=np.minimum(self.upper_limit_on_land,self.land)
                    self.total_savings.return_vals[-1]+=max(self.collateral*self.land_value-self.loan.return_vals[-1][1],0)
                    self.collateral=0
                    new_val[0]=0
                    new_val[1][1]=0
                    self.loan.return_vals[-1][1]=0
                    self.installments[1]=0
                self.number_of_defaults_allowed-=1
                if (self.total_savings.value(-1)-self.family_size*self.per_person_charge)-new_val[0]>(self.installments[0]+self.loan.value(-1)[0]):
                    new_val[0]+=self.installments[0]+self.loan.value(-1)[0]
                    new_val[1][0]+=self.loan.value(-1)[0]
                elif (self.total_savings.value(-1)-self.family_size*self.per_person_charge)-new_val[0]>=(self.installments[0]):
                    new_val[0]+=self.installments[0]
                    new_val[1][0]+=self.installments[0]-self.loan.value(-1)[0]*self.interest
                else:
                    new_val[0]=0
                    new_val[1][0]=-self.loan.value(-1)[0]*self.interest
                return new_val
            
    def returns_gen(self):
        ans=[]
        self.quality=min(self.quality,1)
        for i in range(len(self.crop.types())):
            if(i==self.type_of_crop and (self.plantation_cost.env.now-self.when_planted)%self.crop.harvest_cycle()[i]==0 and self.plantation_cost.value(-1)==0):
                ans.append(self.crop.returns()[self.type_of_crop]*self.land_allocated*self.quality)
            else:
                ans.append(0)
        self.storage_money=0.2*self.total_savings.value(-1)
        return np.array(ans,dtype=float)
        
    def plot(self,k):
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),self.total_savings.value()))
        plt.title("type2 farmer total savings")
        plt.savefig(self.name+"totsav.png")
        '''plt.plot(self.water_req.value())
        plt.title(self.name+" water price")
        plt.figure()'''
        '''plt.plot(self.labour.value())
        plt.title(self.name+" labour")
        plt.figure()
        plt.plot(self.plantation_cost.value())
        plt.title(self.name+" plantation cost")
        plt.figure()'''
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),self.loan.value()))
        plt.title(self.name+" loan")
        plt.savefig(self.name+" loan"+".png")
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),self.premium.value()))
        plt.title(self.name+" premium")
        plt.savefig(self.name+" premium"+".png")
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),[z[0] for z in self.produce.value()]))
        plt.title(self.name+" stock to be sold type 0")
        plt.savefig(self.name+" stock to be sold type 0"+".png")
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),self.income.value()))
        plt.title(self.name+" income")
        plt.savefig(self.name+" income"+".png")
        plt.close()


class type_3_farmer():
    
    def __init__(self,env,name,savings,land,msp,crop,labour,no_of_t3,stored,storage_details,family_size):
        self.name=name
        self.per_person_charge=80000*pow((1.001),env.now)
        self.env=env
        self.initial_land=land
        self.family_size=family_size
        self.interest=0.005
        self.loan_duration=240
        self.type_of_crop=-1
        self.installments=np.zeros(2,dtype=float)
        self.safety_buffer=1000
        self.collateral=0
        self.land=land
        self.land_value=1000
        self.crop=crop
        self.stock_payment=np.zeros(1,dtype=float)
        self.crop_type_recommended=[0]
        self.labour_details=labour
        self.allocated=0
        self.excess_water=10000000
        self.land_allocated=0
        self.number_of_defaults_allowed=4
        self.when_planted=0
        self.no_of_t3=no_of_t3
        self.storage_money=0
        self.upper_limit_on_land=[0.9*self.land,0.9*self.land,0.9*self.land]
        self.quality=1
        self.income_expectation=[0,0,0]
        self.income_expectation[:]=msp
        self.type_of_crop_list=[]
        self.income_exp=simapi.stock(env,name+" produce",0,self.income_expectaion_change,[],[],[],"new_val")
        self.total_savings=simapi.stock(env,name+" total savings",savings,"0.0001*$0",[],[],[],"$0+new_val")
        self.adjusting_land_allocated=simapi.stock(env,name+" adjusting land allocation",0,self.adjust_land,[],[],[],"$0")
        self.plantation_cost=simapi.stock(env,name+"planting cost",0,self.choose_crop,[self.total_savings],[],["$1-new_val"],"new_val")
        self.loan=simapi.stock(env,name+" loan",np.zeros(2,dtype=float),self.apply_loan,[self.total_savings],[],["$1+sum(new_val)"],"$0+new_val")
        self.labour=simapi.stock(env,name+" labour",0,self.labour_gen,[self.total_savings],[],["$1-new_val"],"new_val")
        self.premium=simapi.stock(env,name+" premium",0,self.pay_installments,[self.total_savings,self.loan],[],["$1-new_val[0]","$2-new_val[1]"],"new_val[0]")
        self.stock_to_be_sold=simapi.stock(env,name+" stock to be sold",np.array([0,0,0],dtype=float),self.returns_gen,[self.plantation_cost],[],[""],"new_val")
        self.produce=simapi.stock(env,name+" produce",np.array([0,0,0],dtype=float),self.returns_gen,[],[],[],"new_val")
        self.storage=[[],[],[]]
        self.income=simapi.stock(env,name+" end_crop",0,0,[],[],[],"new_val")
        self.sold=simapi.stock(env,name+" sold",np.array([0,0,0],dtype=float),np.array([0,0,0],dtype=float),[],[],[],"new_val")
        self.temp_sold=simapi.stock(env,name+" temp sold",np.array([0,0,0],dtype=float),np.array([0,0,0],dtype=float),[],[],[],"new_val")
        self.adjusting_land_allocated=simapi.stock(env,name+" adjusting land allocation",0,self.adjust_land,[],[],[],"$0")
        self.stored=stored
        self.storedinfo=False
        self.storage_details=storage_details
        env.process(self.resetting_stock_payments(env))

        
    
    def income_expectaion_change(self):
        if self.type_of_crop!=-1:
            if (self.env.now-self.when_planted)%self.crop.harvest_cycle()[self.type_of_crop]==1 and self.env.now>=self.when_planted+self.crop.harvest_cycle()[self.type_of_crop]:
                self.income_expectation[self.type_of_crop]=sum(self.income.value()[-1*self.crop.harvest_cycle()[self.type_of_crop]:])/sum(i[self.type_of_crop] for i in self.produce.value()[-1*self.crop.harvest_cycle()[self.type_of_crop]:])
    
    def adjust_land(self):
        self.type_of_crop_list.append(str(self.type_of_crop)+" "+str(self.land_allocated)+" "+str(self.land))
        if self.type_of_crop!=-1:
            if len(self.sold.value())>=self.crop.cycle()[self.type_of_crop] and self.type_of_crop!=-1 and self.env.now==self.when_planted+self.crop.cycle()[self.type_of_crop]:
                x=np.array([i for i in range(self.crop.cycle()[self.type_of_crop])])
                x=x.reshape(-1,1)
                y=np.array([self.total_savings.value()[-1*self.crop.cycle()[self.type_of_crop]:]])
                y=y.reshape(-1,1)
                regr=linear_model.LinearRegression()
                regr.fit(x,y)
                if(regr.coef_[0][0]>0):                        
                    self.upper_limit_on_land[self.type_of_crop]=max(self.land_allocated*max(1+(math.atan(regr.coef_[0][0])*2/math.pi),1.9),self.upper_limit_on_land[self.type_of_crop])
                    self.upper_limit_on_land[self.type_of_crop]=min(self.upper_limit_on_land[self.type_of_crop],self.land)
                    
                else:
                    self.upper_limit_on_land[self.type_of_crop]=max((1+(math.atan(regr.coef_[0][0])*2/math.pi))*self.land_allocated,10)
                    self.upper_limit_on_land[self.type_of_crop]=min(self.upper_limit_on_land[self.type_of_crop],10)
                    
                    
                    
    def choose_crop(self):
        ans =-1
        if(self.type_of_crop==-1 or ((self.plantation_cost.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])==0):
            self.quality=1
            self.land_allocated=0
            land_allocated=[0,0,0]
            self.type_of_crop=ans
            supply=np.ones(len(self.crop.types()),dtype=float)
            if self.storedinfo:
                for i in range(len(self.crop.types())):
                    for j in self.stored[i]:
                        supply[i]+=j[0]
                        
                        
            for i in range(len(self.crop.types())):
                land_allocated[i]+=max(min(0.9*self.total_savings.value(-1)/(self.crop.initial_cost()[i]+self.crop.labour_need()[i]*self.crop.cycle()[i]*self.labour_details.wages()[i]),self.upper_limit_on_land[i]),0)
                if self.loan.value(-1)[0]==0:
                    land_allocated[i]+=max(0.9*0.8*self.total_savings.value(-1)/(self.crop.initial_cost()[i]+self.crop.labour_need()[i]*self.crop.cycle()[i]*self.labour_details.wages()[i]),0)
                if(self.loan.value(-1)[1]==0):
                    land_allocated[i]+=max(min(0.7*self.land*self.land_value/(self.crop.initial_cost()[i]+self.crop.labour_need()[i]*self.crop.cycle()[i]*self.labour_details.wages()[i]),self.upper_limit_on_land[i]-land_allocated[i]),0)
                    self.number_of_defaults_allowed=4
                if self.storedinfo:
                    land_allocated[i]=max(min((self.crop.export_quantity()[i]+self.need.value(-1)[i]+self.storage_details.capacity()[i]-supply[i])/self.crop.returns()[i],land_allocated[i]),0)
                else:
                    land_allocated[i]=max(min(0.7*(self.crop.export_quantity()[i]+self.need.value(-1)[i])/self.crop.returns()[i],land_allocated[i]),0)
                land_allocated[i]=min(self.upper_limit_on_land[i],land_allocated[i])
            
            if self.storedinfo:
                priority=np.array(self.crop.price())/supply
                ans=np.argmax(priority)
            else:
                priority=(np.array(self.income_expectation)*land_allocated*self.crop.returns()*np.array(self.crop.cycle())/np.array(self.crop.harvest_cycle()))-((np.array(self.crop.initial_cost())+np.array(self.crop.labour_need())*np.array(self.crop.cycle())*np.array(self.labour_details.wages()))*land_allocated)
                temp=(np.array(self.income_expectation)*land_allocated*self.crop.returns()*np.array(self.crop.cycle())/np.array(self.crop.harvest_cycle()))-((np.array(self.crop.initial_cost())+np.array(self.crop.labour_need())*np.array(self.crop.cycle())*np.array(self.labour_details.wages()))*land_allocated)
                #print self.crop.price()
                ans=np.argmax(priority)
                if temp[ans]<1.5*((np.array(self.crop.initial_cost())[ans]+np.array(self.crop.labour_need())[ans]*np.array(self.crop.cycle())[ans]*np.array(self.labour_details.wages())[ans])*land_allocated[ans]) or temp[ans]<0:
                    ans=-1
            
            if ans!=-1:
                self.when_planted=self.plantation_cost.env.now
                self.land_allocated=land_allocated[ans]
                self.type_of_crop=ans
                ans1=ans
                ans=np.zeros(2,dtype=float)
                if self.loan.value(-1)[0]==0 and (self.total_savings.value(-1))<((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])):
                    ans[0]=min((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])+self.safety_buffer,0.8*self.total_savings.value(-1))
                if self.loan.value(-1)[1]==0 and (ans[0]+(self.total_savings.value(-1)))<((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])):
                    ans[1]=(self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])+self.safety_buffer-ans[0]
                    self.number_of_defaults_allowed=4
                    #print self.name+" ans1 value: "+str(ans[1])
                    self.collateral=ans[1]/self.land_value
                self.installments=ans*(self.interest*pow(1+self.interest,self.loan_duration)/(pow(1+self.interest,self.loan_duration)-1))
                self.total_savings.return_vals[-1]+=sum(ans)
                self.loan.return_vals[-1]+=ans
                self.total_savings.return_vals[-1]+=sum(ans)
                return self.crop.initial_cost()[ans1]*land_allocated[ans1]
            else:
                return 0
            '''self.quality=1
            self.type_of_crop=1
            return self.crop.initial_cost()[1]*self.land'''
        else:
            return 0
        
        
    def labour_gen(self):
        
        if (self.type_of_crop==-1):
            return 0
        else:
            if(self.total_savings.value(-1)<self.crop.labour_need()[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]*self.land_allocated):
                self.quality*=(self.total_savings.value(-1)/self.crop.labour_need()[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]*self.land_allocated)
            return min(self.crop.labour_need()[self.type_of_crop]*self.labour_details.wages()[self.type_of_crop]*self.land_allocated,self.total_savings.value(-1))
            '''print self.allocated
            return self.allocated*self.wages[self.type_of_crop]'''
    
    def apply_loan(self):
        if(self.type_of_crop!=-1):
            if self.when_planted>0:
                ans=np.zeros(2,dtype=float)
                if self.loan.value(-1)[0]==0 and (self.total_savings.value(-1))<((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])):
                    ans[0]=min((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])+self.safety_buffer,0.8*self.total_savings.value(-1))
                if self.loan.value(-1)[1]==0 and (ans[0]+(self.total_savings.value(-1)))<((self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])):
                    ans[1]=(self.crop.cycle()[self.type_of_crop]-(self.env.now-self.when_planted)%self.crop.cycle()[self.type_of_crop])*(self.crop.labour_need()[self.type_of_crop]*self.land_allocated*self.labour_details.wages()[self.type_of_crop])+self.safety_buffer-ans[0]
                    self.number_of_defaults_allowed=4
                    #print self.name+" ans1 value: "+str(ans[1])
                    self.collateral=ans[1]/self.land_value
                    self.installments=ans*(self.interest*pow(1+self.interest,self.loan_duration)/(pow(1+self.interest,self.loan_duration)-1))
                    return ans
                else:
                    return np.zeros(2,dtype=float)
            else:
                return np.zeros(2,dtype=float)
        else:
            ans=np.zeros(2,dtype=float)
            if self.loan.value(-1)[0]==0 and (self.total_savings.value(-1)):
                    ans[0]=0.8*self.total_savings.value(-1)
            if self.loan.value(-1)[1]==0 and (ans[0]+(self.total_savings.value(-1))):
                ans[1]=self.land_value*self.land/2
                self.number_of_defaults_allowed=4
                #print self.name+" ans1 value: "+str(ans[1])
                self.collateral=self.land/2
                self.installments=ans*(self.interest*pow(1+self.interest,self.loan_duration)/(pow(1+self.interest,self.loan_duration)-1))
            return ans
        
    def pay_installments(self):
        new_val=[0,np.zeros(2,dtype=float)]
        
        if((self.total_savings.value(-1))>self.installments[1]+self.loan.value(-1)[1]):
            new_val[0]+=self.installments[1]+self.loan.value(-1)[1]
            new_val[1][1]+=self.loan.value(-1)[1]
            self.installments[1]=0
            if (self.total_savings.value(-1))-new_val[0]>(self.installments[0]+self.loan.value(-1)[0]):
                new_val[0]+=self.installments[0]+self.loan.value(-1)[0]
                new_val[1][0]+=self.loan.value(-1)[0]
            return new_val
        else:
            
            if (self.total_savings.value(-1))>=self.installments[1]:
                new_val[0]+=self.installments[1]
                new_val[1][1]+=self.installments[1]-self.loan.value(-1)[1]*self.interest
                if (self.total_savings.value(-1))-new_val[0]>=(self.installments[0]):
                    new_val[0]+=self.installments[0]
                    new_val[1][0]+=self.installments[0]-self.loan.value(-1)[0]*self.interest
                return new_val
            else:
                #print self.name + " "+str(self.number_of_defaults_allowed)+" "+str(self.collateral)
                if self.number_of_defaults_allowed==0:
                    self.land=max(0,self.land-self.collateral)
                    self.land_allocated=np.minimum(self.land_allocated,self.land)
                    self.upper_limit_on_land=np.minimum(self.upper_limit_on_land,self.land)
                    self.total_savings.return_vals[-1]+=max(self.collateral*self.land_value-self.loan.return_vals[-1][1],0)
                    self.collateral=0
                    new_val[0]=0
                    new_val[1][1]=0
                    self.loan.return_vals[-1][1]=0
                    self.installments[1]=0
                self.number_of_defaults_allowed-=1
                if (self.total_savings.value(-1))-new_val[0]>(self.installments[0]+self.loan.value(-1)[0]):
                    new_val[0]+=self.installments[0]+self.loan.value(-1)[0]
                    new_val[1][0]+=self.loan.value(-1)[0]
                elif (self.total_savings.value(-1))-new_val[0]>=(self.installments[0]):
                    new_val[0]+=self.installments[0]
                    new_val[1][0]+=self.installments[0]-self.loan.value(-1)[0]*self.interest
                else:
                    new_val[0]=0
                    new_val[1][0]=-self.loan.value(-1)[0]*self.interest
                return new_val
            
    def returns_gen(self):
        ans=[]
        self.quality=min(self.quality,1)
        for i in range(len(self.crop.types())):
            if(i==self.type_of_crop and (self.plantation_cost.env.now-self.when_planted)%self.crop.harvest_cycle()[i]==0 and self.plantation_cost.value(-1)==0):
                ans.append(self.crop.returns()[self.type_of_crop]*self.land_allocated*self.quality)
            else:
                ans.append(0)
        ans+=self.stock_payment
        self.storage_money=0.2*self.total_savings.value(-1)
        return np.array(ans,dtype=float)
    
    def resetting_stock_payments(self,env):
        while True:
            self.stock_payment[:]=np.zeros(len(self.stock_payment),dtype=float)
            yield env.timeout(1)
        
    def plot(self,k):
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),self.total_savings.value()))
        plt.title("type3 farmer total savings")
        plt.savefig(self.name+"totsav.png")
        '''plt.plot(self.water_req.value())
        plt.title(self.name+" water price")
        plt.figure()'''
        '''plt.plot(self.labour.value())
        plt.title(self.name+" labour")
        plt.figure()
        plt.plot(self.plantation_cost.value())
        plt.title(self.name+" plantation cost")
        plt.figure()'''
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),self.loan.value()))
        plt.title(self.name+" loan")
        plt.savefig(self.name+" loan"+".png")
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),self.premium.value()))
        plt.title(self.name+" premium")
        plt.savefig(self.name+" premium"+".png")
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),[z[0] for z in self.produce.value()]))
        plt.title(self.name+" stock to be sold type 0")
        plt.savefig(self.name+" stock to be sold type 0"+".png")
        plt.figure()
        plt.plot(np.append(np.zeros(k-len(self.total_savings.value()),dtype=float),self.income.value()))
        plt.title(self.name+" income")
        plt.savefig(self.name+" income"+".png")
        plt.close()



class crop():
    
    def __init__(self):
        self.values=[100,1,1]
        self.mult_fact=np.array([100,0.00001,0.00001],dtype=float)
        self.min_exports_val=np.array([1000,100000,100000])
        self.exports_val=np.array([10000,10000000,10000000])
        self.min_import_vals=np.array([8000,80000,80000])
        self.max_import_vals=np.array([10000,10000000,10000000])
        self.export_price_vals=np.array([1550,1,1])
        self.returns_vals=[1000,1,1]
    
    def types(self):
        return ["type1","type2","type3"]
    
    def cycle(self):
        return [2,2,2]
    
    def harvest_cycle(self):
        return [1,1,1]
    
    
    def export_price(self):
        return self.export_price_vals
    
    def export_quantity(self):
        return self.exports_val
    
    def min_export_quantity(self):
        return self.min_exports_val
    
    def minimum_import(self):
        return self.min_import_vals
    def max_import(self):
        return self.max_import_vals
    
    def initial_cost(self):
        return [10000,10000,80000]
    
    def fertilizer_pesticide_cost(self):
        return [17805,8000,1200]
    
    def water(self):
        return [50,30,20]
    
    def minimum_produce(self):
        return [0,0,0]
        
    def returns(self):
        return self.returns_vals
        
    def labour_need(self):
        return [150,100,100]
    
    def price(self):
        return self.values
    
class labour():
    def __init__(self):
        self.wages_vals=np.array([200,200,200],dtype=float)
    
    
    def number(self):
        return 30
    
    def preference(self):
        return [10,10,10]
    
    def wages(self):
        return self.wages_vals


class storage_facilities():
    def __init__(self,total_capacity=None):
        if total_capacity is None:
            self.total_capacity=np.array([1,1000000,1000000],dtype=float)
        else:
            self.total_capacity=total_capacity
        self.total_capacity_ref=np.array(self.total_capacity)
        self.feeval=np.array([0.01,0.01,0.01],dtype=float)
    
    def number(self):
        1
    
    def capacity(self):
        return self.total_capacity
    
    def loss_rate(self):
        return np.array([0.9,0.14,0.25],dtype=float)
    
    def expiration_time(self):
        return np.array([1,12,6],dtype=float)
    
    def fee(self):
        return self.feeval
need_func_val=5000
need_func_change=[2.5,1.5,1.5]
def need_func():
    global need_func_val
    return [need_func_val,1,1]

def maint(env,crop,price,t1,t2,t3,stored,fsval,labour_details,msp,msp_parameter,mill_dues,mill_agent,water_details,need,time_to_end):
    f1=[]
    f2=[]
    f3=[]
    count1=0.0
    count2=0.0
    count3=0.0
    overall_increase_factor=1
    while True:
        '''if env.now<11 and env.now>0:
            x=env.now
            crop.values=(crop.values*env.now+price.value(-1))/(env.now+1)
        elif env.now>0:
            crop.values=crop.values+(price.value(-1)-price.value(-10))/10'''
        if(len(price.value())<5):
            crop.values=np.mean(price.value(),axis=0)
        else:
            crop.values=np.mean(price.value()[-5:],axis=0)
        mill_dues.write("\n"+str(mill_agent.dues))
        temp=[0,0,0]
        reclaimed_land=0
        for i in t1:
            if len(i.total_savings.value())>=0 and all(np.array(i.total_savings.value()[-4:])<i.per_person_charge*i.family_size*4):
                reclaimed_land+=i.initial_land
                i.land=0
        for i in t2:
            if len(i.total_savings.value())>=0 and all(np.array(i.total_savings.value()[-4:])<i.per_person_charge*i.family_size*4):
                reclaimed_land+=i.initial_land
                i.land=0
        for i in t3:
            if len(i.total_savings.value())>=4 and all(np.array(i.total_savings.value()[-4:])<i.family_size*i.per_person_charge*4):
                reclaimed_land+=i.initial_land
                i.land=0
        '''        
        while reclaimed_land>0:
            if reclaimed_land>=12:
                ans=np.random.multinomial(1,[0.2,0.3,0.5])
                if ans[2]==1:
                    if reclaimed_land<12:
                        t3.append(type_3_farmer(env,"type 3 "+str(len(t3)),(1000000+random.uniform(0,100000))*overall_increase_factor,reclaimed_land,msp,crops,labour_details,no_of_t3,stored,storage_details,3))
                        t3[-1].need=need
                        reclaimed_land=0
                    else:
                        assign_land=random.uniform(8,min(reclaimed_land,12))
                        reclaimed_land-=assign_land
                        t3.append(type_3_farmer(env,"type 3 "+str(len(t3)),(1000000+random.uniform(0,100000))*overall_increase_factor,assign_land,msp,crops,labour_details,no_of_t3,stored,storage_details,3))
                        t3[-1].need=need
                elif ans[1]==1:
                    if reclaimed_land<8:
                        t2.append(type_2_farmer(env,"type 2 "+str(len(t2)),(100000+random.uniform(0,10000))*overall_increase_factor,reclaimed_land,msp,crops,labour_details,stored,random.uniform(2,4)))
                        t2[-1].need=need
                        reclaimed_land=0
                    else:
                        assign_land=random.uniform(8,min(reclaimed_land,10))
                        reclaimed_land-=assign_land
                        t2.append(type_2_farmer(env,"type 2 "+str(len(t2)),(100000+random.uniform(0,10000))*overall_increase_factor,assign_land,msp,crops,labour_details,stored,random.uniform(2,4)))
                        t2[-1].need=need
                else:
                    if reclaimed_land<5:
                        t1.append(type_1_farmer(env,"type 1 "+str(len(t1)),(10000+random.uniform(0,10000))*overall_increase_factor,reclaimed_land,msp,crops,labour_details,water_details,random.uniform(3,5)))
                        reclaimed_land=0
                    else:
                        assign_land=random.uniform(5,min(reclaimed_land,8))
                        reclaimed_land-=assign_land
                        t1.append(type_1_farmer(env,"type 1 "+str(len(t1)),(10000+random.uniform(0,10000))*overall_increase_factor,assign_land,msp,crops,labour_details,water_details,random.uniform(3,5)))
                        
            elif reclaimed_land>=10:
                ans=np.random.binomial(1,[0.4])
                if ans==1:
                    if reclaimed_land<8:
                        t2.append(type_2_farmer(env,"type 2 "+str(len(t2)),(100000+random.uniform(0,10000))*overall_increase_factor,reclaimed_land,msp,crops,labour_details,stored,random.uniform(2,4)))
                        t2[-1].need=need
                        reclaimed_land=0
                    else:
                        assign_land=random.uniform(8,min(reclaimed_land,10))
                        reclaimed_land-=assign_land
                        t2.append(type_2_farmer(env,"type 2 "+str(len(t2)),(100000+random.uniform(0,10000))*overall_increase_factor,assign_land,msp,crops,labour_details,stored,random.uniform(2,4)))
                        t2[-1].need=need
                else:
                    if reclaimed_land<5:
                        t1.append(type_1_farmer(env,"type 1 "+str(len(t1)),(10000+random.uniform(0,10000))*overall_increase_factor,reclaimed_land,msp,crops,labour_details,water_details,random.uniform(3,5)))
                        reclaimed_land=0
                    else:
                        assign_land=random.uniform(5,min(reclaimed_land,8))
                        reclaimed_land-=assign_land
                        t1.append(type_1_farmer(env,"type 1 "+str(len(t1)),(10000+random.uniform(0,10000))*overall_increase_factor,assign_land,msp,crops,labour_details,water_details,random.uniform(3,5)))
            else:
                if reclaimed_land<6:
                    t1.append(type_1_farmer(env,"type 1 "+str(len(t1)),(10000+random.uniform(0,10000))*overall_increase_factor,reclaimed_land,msp,crops,labour_details,water_details,random.uniform(3,5)))
                    reclaimed_land=0
                else:
                    assign_land=random.uniform(5,min(reclaimed_land,6))
                    reclaimed_land-=assign_land
                    t1.append(type_1_farmer(env,"type 1 "+str(len(t1)),(10000+random.uniform(0,10000))*overall_increase_factor,assign_land,msp,crops,labour_details,water_details,random.uniform(3,5)))'''
        
        overall_increase_factor*=1.001
        for i in range(len(stored)):
            for j in stored[i]:
                temp[i]+=j[0]
        fsval.write("\n"+str(temp))
        labour_details.wages_vals[:]*=1.0001
        msp[:]*=(1+msp_parameter)
        crop.mult_fact[:]*=1.00001
        if env.now==time_to_end-1:
            count=0.0
            for i in t1:
                if i.land==0:
                    count+=1
            count1=count/len(t1)
            count=0.0
            for i in t2:
                if i.land==0:
                    count+=1
            count2=count/len(t2)
            count=0.0
            for i in t3:
                if i.land==0:
                    count+=1
            count3=count/len(t3)
            print str(count1)+','+str(count2)+','+str(count3)+',',
        yield env.timeout(1)

class price_func():
    def __init__(self,need,total_new_stock,crop,stored,imported):
        self.need=need
        self.total_new_stock=total_new_stock
        self.total_stock=np.zeros(3,dtype=float)
        self.crop=crop
        self.stored=stored
        self.imported=imported
        
    def price_gen(self):
        self.total_stock=np.zeros(len(self.total_new_stock.value(-1)),dtype=float)
        for i in range(len(self.total_new_stock.value(-1))):
            for j in self.stored[i]:
                self.total_stock[i]+=j[0]
        self.total_stock+=self.total_new_stock.value(-1)
        self.total_new_stock.return_vals[-1]=self.total_stock
        self.total_stock+=1
        #no_bound=2/(1+np.exp(-2*(self.need.value(-1)/self.total_stock-np.ones(len(self.total_stock)))))
        need_len = len(self.need.return_vals)
        last_vals = max(4-need_len,0)
        iter = max(-4, need_len*-1)
        weight = 0.4
        need_term = np.array([0,0,0],dtype=float)
        while weight>0 :
            need_term += np.array(self.need.value(iter), dtype=float)*weight
            weight -= 0.1
            iter = min(-1, iter+1)
        no_bound=pow((np.maximum((2*need_term/self.total_stock)-np.ones(len(self.total_stock)),0.1)),1)
        no_bound=np.maximum(no_bound,0.4)
        no_bound=np.minimum(no_bound,1.1)
        return no_bound*self.crop.mult_fact
 
def storage_fee_calc(env,storage_details,t1,t2,t3,stored):
    fee_mult=0.01
    while True:
        totalsum=np.array([1,1,1],dtype=float)
        for i in t1:
            totalsum=totalsum+i.stock_to_be_sold.value(-1)
            
        for i in t2:
            totalsum=totalsum+i.stock_to_be_sold.value(-1)
        
        for i in t3:
            totalsum=totalsum+i.stock_to_be_sold.value(-1)
        storage_details.feeval[:]= totalsum*fee_mult/(storage_details.total_capacity+1)
        fee_mult*=1.00001
        yield env.timeout(1)


def storing(env,storage_details,t1,t2,t3,stored,crops=None,price=None,total_new_stock=None):
    
    if price is None:
        
        while True:
            
            t1storage=[]
            totalsum=np.array([0,0,0],dtype=float)
            for i in t1:
                t1storage.append(np.minimum(np.array([i.storage_money,i.storage_money,i.storage_money],dtype=float)/storage_details.fee(),i.stock_to_be_sold.value(-1)))
                totalsum=totalsum+t1storage[-1]
                
            
            t2storage=[]
            for i in t2:
                t2storage.append(np.minimum(np.array([i.storage_money,i.storage_money,i.storage_money],dtype=float)/storage_details.fee(),i.stock_to_be_sold.value(-1)))
                totalsum=totalsum+t2storage[-1]
            
            t3storage=[]
            for i in t3:
                t3storage.append(np.minimum(np.array([i.storage_money,i.storage_money,i.storage_money],dtype=float)/storage_details.fee(),i.stock_to_be_sold.value(-1)))
                totalsum=totalsum+t3storage[-1]
                
            numer=np.minimum(totalsum,storage_details.total_capacity)
            for i in range(3):
                for j in range(len(t1storage)):
                    if(totalsum[i]>0 and t1[j].total_savings.value(-1)>(t1storage[j][i]*numer[i]/totalsum[i])*storage_details.fee()[i] and (t1storage[j][i]*numer[i]/totalsum[i])>0):
                        t1[j].total_savings.return_vals[-1]-=(t1storage[j][i]*numer[i]/totalsum[i])*storage_details.fee()[i]
                        stored[i].append([max((t1storage[j][i]*numer[i]/totalsum[i]),0),t1[j],env.now+storage_details.expiration_time()[i]])
                        t1[j].storage[i].append([max((t1storage[j][i]*numer[i]/totalsum[i]),0),env.now+storage_details.expiration_time()[i]])
                        storage_details.total_capacity[i]-=max((t1storage[j][i]*numer[i]/totalsum[i]),0)
                
                for j in range(len(t2storage)):
                    if(totalsum[i]>0 and t2[j].total_savings.value(-1)>(t2storage[j][i]*numer[i]/totalsum[i])*storage_details.fee()[i] and (t2storage[j][i]*numer[i]/totalsum[i])>0):
                        t2[j].total_savings.return_vals[-1]-=(t2storage[j][i]*numer[i]/totalsum[i])*storage_details.fee()[i]
                        stored[i].append([max((t2storage[j][i]*numer[i]/totalsum[i]),0),t2[j],env.now+storage_details.expiration_time()[i]])
                        t2[j].storage[i].append([max((t2storage[j][i]*numer[i]/totalsum[i]),0),env.now+storage_details.expiration_time()[i]])
                        storage_details.total_capacity[i]-=max((t2storage[j][i]*numer[i]/totalsum[i]),0)
                
                for j in range(len(t3storage)):
                    if(totalsum[i]>0 and t3[j].total_savings.value(-1)>(t3storage[j][i]*numer[i]/totalsum[i])*storage_details.fee()[i] and (t3storage[j][i]*numer[i]/totalsum[i])>0):
                        t3[j].total_savings.return_vals[-1]-=(t3storage[j][i]*numer[i]/totalsum[i])*storage_details.fee()[i]
                        stored[i].append([max((t3storage[j][i]*numer[i]/totalsum[i]),0),t3[j],env.now+storage_details.expiration_time()[i]])
                        t3[j].storage[i].append([max((t3storage[j][i]*numer[i]/totalsum[i]),0),env.now+storage_details.expiration_time()[i]])
                        storage_details.total_capacity[i]-=max((t3storage[j][i]*numer[i]/totalsum[i]),0)
                    
            yield env.timeout(1)
    
    else:
        
        while True:
            price_priority=np.argsort(np.array(price.value(-1)))
            for j in price_priority:
                if price.value(-1)[j]>crops.mult_fact[j]:
                    continue
                for i in t3:
                    to_be_stored=min(i.storage_money/storage_details.fee()[j],i.stock_to_be_sold.value(-1)[j]*(1-price.value(-1)[j]/crops.mult_fact[j]))
                    storage_details.total_capacity[j]-=max(min(to_be_stored,storage_details.total_capacity[j]),0)
                    stored[j].append([max(min(to_be_stored,storage_details.total_capacity[j]),0),i,env.now+storage_details.expiration_time()[j]])
                    i.storage[j].append([max(min(to_be_stored,storage_details.total_capacity[j]),0),i,env.now+storage_details.expiration_time()[j]])
                    i.total_savings.return_vals[-1]-=max(min(to_be_stored,storage_details.total_capacity[j]),0)*storage_details.fee()[j]
                    total_new_stock.return_vals[-1][j]-=max(min(to_be_stored,storage_details.total_capacity[j]),0)
                    i.storage_money-=max(min(to_be_stored,storage_details.total_capacity[j]),0)*storage_details.fee()[j]
                
                
                for i in t2:
                    to_be_stored=min(i.storage_money/storage_details.fee()[j],i.stock_to_be_sold.value(-1)[j]*(1-price.value(-1)[j]/crops.mult_fact[j]))
                    storage_details.total_capacity[j]-=max(min(to_be_stored,storage_details.total_capacity[j]),0)
                    stored[j].append([max(min(to_be_stored,storage_details.total_capacity[j]),0),i,env.now+storage_details.expiration_time()[j]])
                    i.storage[j].append([max(min(to_be_stored,storage_details.total_capacity[j]),0),i,env.now+storage_details.expiration_time()[j]])
                    i.total_savings.return_vals[-1]-=max(min(to_be_stored,storage_details.total_capacity[j]),0)*storage_details.fee()[j]
                    total_new_stock.return_vals[-1][j]-=max(min(to_be_stored,storage_details.total_capacity[j]),0)
                    i.storage_money-=max(min(to_be_stored,storage_details.total_capacity[j]),0)*storage_details.fee()[j]
                    
                    
                
                for i in t1:
                    to_be_stored=min(i.storage_money/storage_details.fee()[j],i.stock_to_be_sold.value(-1)[j]*(1-price.value(-1)[j]/crops.mult_fact[j]))
                    storage_details.total_capacity[j]-=max(min(to_be_stored,storage_details.total_capacity[j]),0)
                    stored[j].append([max(min(to_be_stored,storage_details.total_capacity[j]),0),i,env.now+storage_details.expiration_time()[j]])
                    i.storage[j].append([max(min(to_be_stored,storage_details.total_capacity[j]),0),i,env.now+storage_details.expiration_time()[j]])
                    i.total_savings.return_vals[-1]-=max(min(to_be_stored,storage_details.total_capacity[j]),0)*storage_details.fee()[j]
                    total_new_stock.return_vals[-1][j]-=max(min(to_be_stored,storage_details.total_capacity[j]),0)
                    i.storage_money-=max(min(to_be_stored,storage_details.total_capacity[j]),0)*storage_details.fee()[j]
                
            yield env.timeout(1)
        
        
        
class water_management():
    def __init__(self,env,t1,ws,crops,storage_details,stored):
        self.t1=t1
        self.env=env
        self.ws=ws
        self.crops=crops
        self.storage_details=storage_details
        self.stored=stored
        self.started=time.time()
        wm=simapi.stock(env,"water management",0,self.water_management_function,[],[],[],"new_val")
        
    def water_management_function(self):
        x=[]
        
        for i in self.t1:
            if(len(i.water_required)>0 and i.type_of_crop==-1):
                x.append(i)
        minimum_produce=self.crops.minimum_produce()
        number_of_crops=len(self.crops.types())
        self.matching(x,self.ws,number_of_crops)
        return 0
        #now using need and supply(the total stock in storage) u need to assign each t1 the crop, based on the deficit. Now also set the land allocated and water required and accordingly.
        # then go for the higher number of banks. Finish it before tonight. Then u need to generate the results. NO COC OR ANY OTHER DISTRACTIONS TODAY. U HAVE EXACTLY ONE MONTH TO FINISH THE WORK.
    
    
    def matching(self,x,y,number_of_crops):
        water_reqs=[]
        crop_types=[]
        water_received=np.zeros(len(x))
        if(len(x)==0):
            return 0
        #restricted_types=[[]for i in y]
        #produce_for_each_crop=np.array([np.zeros(number_of_crops) for i in y])
        #total_produce_for_each_crop=np.zeros(number_of_crops)
        #x_allocated_to_y=[[] for i in y]
        #x=[x[i] for i in np.random.permutation(len(x))]
        type_of_crop=np.array([-1 for i in x])
        ans=[]
        rerun=False
        water_available=[]
        crop_type_recommended=[]
        for i in x:
            water_reqs.append(i.water_required)
            crop_types.append(i.crop_type)
        #for i in y:
            #crop_type_recommended.append(i.crop_type_recommended)
        
        to_be_allocated=[i for i in range(len(x))]
        
        
        ##for i in range(len(y)):
            ##if y[i].excess_water==0:
                ##continue
            ##while len(to_be_allocated)>0:
                ##ans=0
                ##'''sortedorder=np.argsort(temp_produce)
                ##ans=sortedorder[0]
                ##for j in sortedorder:
                    ##if x[to_be_allocated[j]].estimated_produce[curr_max]<need_and_storage[curr_max]:
                        ##break
                    ##else:
                        ##ans=j'''
                ##type_of_crop[to_be_allocated[ans]]=curr_max
                ##if x[to_be_allocated[ans]].water_required[curr_max]>y[i].excess_water:
                    ##x[to_be_allocated[ans]].land_allocated[curr_max]=x[to_be_allocated[ans]].land_allocated[curr_max]*y[i].excess_water/x[to_be_allocated[ans]].water_required[curr_max]
                    ##x[to_be_allocated[ans]].estimated_produce[curr_max]= x[to_be_allocated[ans]].estimated_produce[curr_max]*y[i].excess_water/x[to_be_allocated[ans]].water_required[curr_max]
                    ##x[to_be_allocated[ans]].water_required[curr_max]=y[i].excess_water
                ##x[to_be_allocated[ans]].water_lender=y[i]
                ##x[to_be_allocated[ans]].type_of_crop=curr_max
                ###print self.env.now
                ###print y[i].excess_water
                ###produce_received[curr_max]+=min(need_and_storage[curr_max],x[to_be_allocated[ans]].estimated_produce[curr_max])
                ##produce_received[curr_max]+=x[to_be_allocated[ans]].estimated_produce[curr_max]
                ##y[i].excess_water-=x[to_be_allocated[ans]].water_required[curr_max]
                ##x[to_be_allocated[ans]].water_received=x[to_be_allocated[ans]].water_required[curr_max]
                ##priority[:]=np.array(self.crops.mult_fact)/(supply+produce_for_each_crop)
                ##to_be_allocated[:]=[j for j in to_be_allocated if j!=to_be_allocated[ans]]
                ###if(all(need_and_storage==0) or y[i].excess_water==0):
                ##if y[i].excess_water==0:
                    ##break
                
                
        #all or nothing in water
        '''        
        permute_array=np.array([i for i in range(len(y))])
        while True:
            rerun=False
            for i in range(len(x)):
                breakflag=False
                #permute_array=np.random.permutation(len(x))
                for k in range(len(crop_types[i])):
                    if breakflag:
                        break
                    if water_received[i]>0:
                        break
                    for j in range(len(y)):
                        if water_reqs[i][k]==0 and x[i].land_allocated[crop_types[i][k]]>0:
                                x[i].water_received=0
                                type_of_crop[i]=crop_types[i][k]
                                x[i].type_of_crop=type_of_crop[i]
                                x[i].water_lender=None
                                breakflag=True
                                
                        elif((crop_type_recommended[permute_array[j]]==[] or np.isin(crop_types[i][k],crop_type_recommended[permute_array[j]])) and(not np.isin(crop_types[i][k], restricted_types[permute_array[j]]))):
                            if(y[permute_array[j]].excess_water>=water_reqs[i][k] and water_reqs[i][k]>0):
                                water_received[i]=water_reqs[i][k]
                                x[i].water_received=water_received[i]
                                type_of_crop[i]=crop_types[i][k]
                                x[i].type_of_crop=type_of_crop[i]
                                y[permute_array[j]].excess_water=y[permute_array[j]].excess_water-water_reqs[i][k]
                                produce_for_each_crop[permute_array[j]][type_of_crop[i]]+=x[i].estimated_produce[k]
                                total_produce_for_each_crop[type_of_crop[i]]+=x[i].estimated_produce[k]
                                x_allocated_to_y[permute_array[j]].append([i,type_of_crop[i]])
                                x[i].water_lender=y[permute_array[j]]
                                break
                        
            if(not rerun):
                break
        for i in range(len(type_of_crop)):
            if(type_of_crop[i]==-1):
                for j in range(len(water_reqs[i])):
                    if(water_reqs[i][j]==0 and x[i].land_allocated[crop_types[i][j]]>0):
                        type_of_crop[i]=crop_types[i][j]
                        x[i].type_of_crop=type_of_crop[i]
                        break
        '''
        #equal water distribution
        water_bound=y.excess_water
        if len(x)>0:
            water_bound=y.excess_water/len(x)
        for i in range(number_of_crops):
            permute_array=np.argsort([j[i] for j in water_reqs])
            for j in permute_array:
                if water_reqs[j][i]==0 and x[j].land_allocated[crop_types[j][i]]>0 and type_of_crop[j]==-1:
                    x[j].water_received=0
                    type_of_crop[j]=crop_types[j][i]
                    x[j].type_of_crop=type_of_crop[j]
                    x[j].water_lender=None
                    if len([k for k in type_of_crop if k==-1])>0:
                        water_bound=y.excess_water/len([k for k in type_of_crop if k==-1])
                    else:
                        water_bound=y.excess_water
                elif water_reqs[j][i]>0 and type_of_crop[j]==-1:
                    water_received[j]=min(water_reqs[j][i],water_bound)
                    x[j].water_received=water_received[j]
                    type_of_crop[j]=crop_types[j][i]
                    if water_received[j]<water_bound:
                        if len([k for k in type_of_crop if k==-1])>0:
                            water_bound=y.excess_water/len([k for k in type_of_crop if k==-1])
                        else:
                            water_bound=y.excess_water
                    else:
                        x[j].land_allocated[crop_types[j][i]]*=water_received[j]/x[j].water_required[i]
                        x[j].water_required[i]=water_received[j]
                    x[j].type_of_crop=type_of_crop[j]
                    y.excess_water=y.excess_water-water_received[j]
                    x[j].water_lender=y
            if all(type_of_crop!=-1):
                break
                    
        
                    
        for i in range(len(type_of_crop)):
            x[i].estimated_produce=[]
            x[i].water_required=[]
            x[i].crop_type=[]
            if(type_of_crop[i]!=-1):
                x[i].plantation()
                #x[i].water_payment()
        ended=time.time()
        self.started=ended
    
    
class storage_stock_management():
    def __init__(self,env,stored,storage_details,imported,msp_storage):
        self.stored=stored
        self.env=env
        self.imported=imported
        self.storage_details=storage_details
        self.msp_storage=msp_storage
        
    def removal(self):
        for i in range(len(self.stored)):
            for j in self.stored[i]:
                if j[1] is None:
                    if j[0]>self.imported[i]:
                        j[0]-=self.imported[i]
                        self.imported[i]=0
                    else:
                        self.imported[i]-=j[0]
                        j[0]=0
                
                elif j[0]>j[1].temp_sold.value(-1)[i]:
                    j[0]-=j[1].temp_sold.value(-1)[i]
                    self.storage_details.total_capacity[i]+=j[1].temp_sold.value(-1)[i]
                    j[1].temp_sold.value(-1)[i]=0
                else:
                    j[1].temp_sold.value(-1)[i]-=j[0]
                    self.storage_details.total_capacity[i]+=j[0]
                    j[0]=0
                if j[2]==self.env:
                    self.storage_details.total_capacity[i]+=j[0]
                    j[0]=0
            self.stored[i][:]=[it for it in self.stored[i] if it[0]>0]
        for i in range(len(self.msp_storage)):
            msp_storage[i][:]=[it for it in msp_storage[i] if it[0]>0 and it[1]>self.env.now]
            
        for i in range(len(self.imported)):
            self.imported[i]=0
    
    
    
class sale():
    def __init__(self,mill_agent,need,price,total_new_stock,imported,stored,msp,msp_storage,msp_maxstock,env,storage_details,t1,t2,t3):
        self.mill_agent=mill_agent
        self.need=need
        self.price=price
        self.imported=imported
        self.total_new_stock=total_new_stock
        self.stored=stored
        self.msp=msp
        self.t1=t1
        self.t2=t2
        self.t3=t3
        self.msp_maxstock=msp_maxstock
        self.msp_storage=msp_storage
        self.env=env
        self.percentage_bpl=np.array([0.0001,0.6,0.6])
        self.storage_details=storage_details
        
    def sale_func(self):
        self.imported[:]=np.array([min(self.imported[z],self.need.return_vals[-1][z]*np.divide(self.imported,self.total_new_stock.return_vals[-1],out=np.zeros(len(self.total_new_stock.return_vals[-1])),where=self.total_new_stock.return_vals[-1]!=0)[z]) for z in range(len(self.total_new_stock.return_vals[-1]))],dtype=float)
        
        
        for i in range(len(self.msp_storage)):
            if i==0:
                continue
            if self.price.value(-1)[i]>self.msp[i]*1.05:
                loc_sum=0
                for j in self.msp_storage[i]:
                    loc_sum+=j[0]
                sold_from_msp_storages=min(self.need.return_vals[-1][i]*self.percentage_bpl[i],loc_sum)
                self.need.return_vals[-1][i]-=sold_from_msp_storages
                for i in self.msp_storage[i]:
                    if sold_from_msp_storages<i[0]:
                        i[0]-=sold_from_msp_storages
                        sold_from_msp_storages=0
                    else:
                        sold_from_msp_storages-=i[0]
                        i[0]=0
        
        for i in range(len(self.stored)):
            for j in self.stored[i]:
                if j[1] is None:
                    self.imported[i]+=j[0]
                else:
                    j[1].stock_to_be_sold.return_vals[-1][i]+=j[0]
                    
        for i in self.t1:
            #print "in sale"
            #print i.sold.value()
            for j in range(len(i.stock_to_be_sold.value(-1))):
                if self.price.value(-1)[j]<self.msp[j]:
                    total_new_stock.return_vals[-1][j]-=min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j])
                    i.total_savings.return_vals[-1]+=min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j])*self.msp[j]
                    i.income.return_vals[-1]+=min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j])*self.msp[j]
                    msp_storage[j].append([min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j]),self.env.now+self.storage_details.expiration_time()[j]])
                    i.stock_to_be_sold.return_vals[-1][j]-=min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j])
                    
                    
            temp=np.array([min(i.stock_to_be_sold.return_vals[-1][z],self.need.return_vals[-1][z]*np.divide(i.stock_to_be_sold.return_vals[-1],self.total_new_stock.return_vals[-1],out=np.zeros(len(self.total_new_stock.return_vals[-1])),where=self.total_new_stock.return_vals[-1]!=0)[z]) for z in range(len(self.total_new_stock.return_vals[-1]))],dtype=float)
            
            i.stock_to_be_sold.return_vals[-1]-=temp
            
            i.sold.return_vals[-1]+=temp
            i.temp_sold.return_vals[-1]+=temp
            
            
            i.income.return_vals[-1]+=sum(temp*self.price.return_vals[-1])
            
            i.total_savings.return_vals[-1]=i.total_savings.return_vals[-1]+sum(temp*self.price.return_vals[-1])
            #print "after sale"
            #print i.sold.value()
            
            
        for i in self.t2:
            for j in range(len(i.stock_to_be_sold.value(-1))):
                if self.price.value(-1)[j]<self.msp[j]:
                    total_new_stock.return_vals[-1][j]-=min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j])
                    i.total_savings.return_vals[-1]+=min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j])*self.msp[j]
                    i.income.return_vals[-1]+=min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j])*self.msp[j]
                    msp_storage[j].append([min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j]),self.env.now+self.storage_details.expiration_time()[j]])
                    i.stock_to_be_sold.return_vals[-1][j]-=min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j])
                    
                    
            temp=np.array([min(i.stock_to_be_sold.return_vals[-1][z],self.need.return_vals[-1][z]*np.divide(i.stock_to_be_sold.return_vals[-1],self.total_new_stock.return_vals[-1],out=np.zeros(len(self.total_new_stock.return_vals[-1])),where=self.total_new_stock.return_vals[-1]!=0)[z]) for z in range(len(self.total_new_stock.return_vals[-1]))],dtype=float)
            
            i.stock_to_be_sold.return_vals[-1]-=temp
            
            i.sold.return_vals[-1]+=temp
            
            i.temp_sold.return_vals[-1]+=temp
            
            i.income.return_vals[-1]+=sum(temp*self.price.return_vals[-1])
            
            i.total_savings.return_vals[-1]=i.total_savings.return_vals[-1]+sum(temp*self.price.return_vals[-1])
            
            
        for i in self.t3:
            for j in range(len(i.stock_to_be_sold.value(-1))):
                if self.price.value(-1)[j]<self.msp[j]:
                    total_new_stock.return_vals[-1][j]-=min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j])
                    i.total_savings.return_vals[-1]+=min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j])*self.msp[j]
                    i.income.return_vals[-1]+=min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j])*self.msp[j]
                    msp_storage[j].append([min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j]),self.env.now+self.storage_details.expiration_time()[j]])
                    i.stock_to_be_sold.return_vals[-1][j]-=min(self.msp_maxstock[j],i.stock_to_be_sold.return_vals[-1][j])
                    
                    
            temp=np.array([min(i.stock_to_be_sold.return_vals[-1][z],self.need.return_vals[-1][z]*np.divide(i.stock_to_be_sold.return_vals[-1],self.total_new_stock.return_vals[-1],out=np.zeros(len(self.total_new_stock.return_vals[-1])),where=self.total_new_stock.return_vals[-1]!=0)[z]) for z in range(len(self.total_new_stock.return_vals[-1]))],dtype=float)
            
            i.stock_to_be_sold.return_vals[-1]-=temp
            
            i.sold.return_vals[-1]+=temp
            
            i.temp_sold.return_vals[-1]+=temp
            
            i.income.return_vals[-1]+=sum(temp*self.price.return_vals[-1])
            
            i.total_savings.return_vals[-1]=i.total_savings.return_vals[-1]+sum(temp*self.price.return_vals[-1])
        
        temp=np.zeros(len(self.total_new_stock.return_vals[-1]),dtype=float)
        for k in range(len(temp)):
            temp[k]=min(self.mill_agent.refining.return_vals[-1][k],(self.need.return_vals[-1][k]*self.mill_agent.refining.return_vals[-1][k]/self.total_new_stock.return_vals[-1][k]))
            
        self.mill_agent.refining.return_vals[-1]-=temp
        
        self.mill_agent.sold.return_vals[-1]+=temp
        
        
        self.mill_agent.income.return_vals[-1]+=sum(temp*self.price.return_vals[-1])
        
        self.mill_agent.total_savings.return_vals[-1]=self.mill_agent.total_savings.return_vals[-1]+sum(temp*self.price.return_vals[-1])
        #print "after sale"
        #print i.sold.value()
            
        return np.array([min(self.need.return_vals[-1][z],self.total_new_stock.return_vals[-1][z]) for z in range(len(self.total_new_stock.return_vals[-1]))],dtype=float)
    
    
    

class import_export_handler():
    def __init__(self,env,mill_agent,stored,price,crops,need,storage_details,fimp,fexp,tax,t1,t2,t3):
        self.env=env
        self.mill_agent=mill_agent
        self.stored=stored
        self.crops=crops
        self.need=need
        self.t1=t1
        self.t2=t2
        self.t3=t3
        self.storage_details=storage_details
        self.import_export=simapi.stock(env,"import and export",np.array([0,0,0],dtype=float),self.import_export_func,[],[],[],"new_val")
        self.fimp=fimp
        self.fexp=fexp
        self.tax=tax
        self.price=price
        
    def import_export_func(self):
        total_stock=np.ones(len(self.crops.types()))
        return_vals=np.zeros(3)
        total_stock+=self.mill_agent.refining.value(-1)
        for i in self.t1:
            total_stock+=i.stock_to_be_sold.value(-1)
        for i in self.t2:
            total_stock+=i.stock_to_be_sold.value(-1)
        for i in self.t3:
            total_stock+=i.stock_to_be_sold.value(-1)
        #print total_stock
        for i in range(len(total_stock)):
            for j in self.stored[i]:
                total_stock[i]+=j[0]
        return_vals=np.zeros(3)
        no_exports_flag=False
        no_imports_flag=False
        (self.need.value(-1)/total_stock-np.ones(len(total_stock)))
        no_bound=(self.need.value(-1)/total_stock-np.ones(len(total_stock)))
        no_bound=np.maximum(no_bound,0.3)
        no_bound=np.minimum(no_bound,3)
        curr_price=no_bound*self.crops.mult_fact
        for i in range(len(total_stock)):
            if curr_price[i]<self.crops.export_price()[i]*(1-self.tax[i]) and not no_exports_flag:
                max_exports=min(min((self.crops.mult_fact[i]/(curr_price[i]))*self.crops.min_export_quantity()[i],total_stock[i]),self.crops.exports_val[i])
                millsold=min(self.mill_agent.refining.return_vals[-1][i],max_exports*(self.mill_agent.refining.return_vals[-1][i]/total_stock[i]) )
                self.mill_agent.sold.return_vals[-1][i]+=millsold
                return_vals[i]-=millsold
                
                self.mill_agent.refining.return_vals[-1][i]-=millsold
                
                self.mill_agent.income.return_vals[-1]+=millsold*self.crops.export_price()[i]
                
                self.mill_agent.total_savings.return_vals[-1]=self.mill_agent.total_savings.return_vals[-1]+(self.mill_agent.sold.return_vals[-1][i]*self.crops.export_price()[i])
                
                for j in self.t1:
                    #print "in here"
                    #print j.sold.value()
                    j.sold.return_vals[-1][i]=min(j.stock_to_be_sold.return_vals[-1][i],max_exports*(j.stock_to_be_sold.return_vals[-1][i]/total_stock[i]) )
                    
                    j.temp_sold.return_vals[-1][i]=min(j.stock_to_be_sold.return_vals[-1][i],max_exports*(j.stock_to_be_sold.return_vals[-1][i]/total_stock[i]) )
                    
                    return_vals[i]-=j.sold.return_vals[-1][i]
                    
                    j.stock_to_be_sold.return_vals[-1][i]-=j.sold.value(-1)[i]
                    
                    j.income.return_vals[-1]+=j.sold.return_vals[-1][i]*self.crops.export_price()[i]
                    
                    j.total_savings.return_vals[-1]=j.total_savings.return_vals[-1]+(j.sold.return_vals[-1][i]*self.crops.export_price()[i])*(1-self.tax[i])
                    
                    
                for j in self.t2:
                    #print "in here2"
                    #print j.sold.value()
                    j.sold.return_vals[-1][i]=min(j.stock_to_be_sold.return_vals[-1][i],max_exports*(j.stock_to_be_sold.return_vals[-1][i]/total_stock[i]) )
                    
                    j.temp_sold.return_vals[-1][i]=min(j.stock_to_be_sold.return_vals[-1][i],max_exports*(j.stock_to_be_sold.return_vals[-1][i]/total_stock[i]) )
                    
                    return_vals[i]-=j.sold.return_vals[-1][i]
                    
                    j.stock_to_be_sold.return_vals[-1][i]-=j.sold.value(-1)[i]
                    
                    j.income.return_vals[-1]+=j.sold.return_vals[-1][i]*self.crops.export_price()[i]
                    
                    j.total_savings.return_vals[-1]=j.total_savings.return_vals[-1]+(j.sold.return_vals[-1][i]*self.crops.export_price()[i])*(1-self.tax[i])
                    
                for j in self.t3:
                    #print "in here3"
                    #print j.sold.value()
                    
                    j.sold.return_vals[-1][i]=min(j.stock_to_be_sold.return_vals[-1][i],max_exports*(j.stock_to_be_sold.return_vals[-1][i]/total_stock[i]) )
                    
                    j.temp_sold.return_vals[-1][i]=min(j.stock_to_be_sold.return_vals[-1][i],max_exports*(j.stock_to_be_sold.return_vals[-1][i]/total_stock[i]) )
                    
                    return_vals[i]-=j.sold.return_vals[-1][i]
                    
                    j.stock_to_be_sold.return_vals[-1][i]-=j.sold.value(-1)[i]
                    
                    j.income.return_vals[-1]+=j.sold.return_vals[-1][i]*self.crops.export_price()[i]
                    
                    j.total_savings.return_vals[-1]=j.total_savings.return_vals[-1]+(j.sold.return_vals[-1][i]*self.crops.export_price()[i])*(1-self.tax[i])
                    
            elif self.crops.mult_fact[i]<curr_price[i] and not no_imports_flag:
                self.stored[i].append([min(self.crops.minimum_import()[i]*curr_price[i]/self.crops.mult_fact[i],self.crops.max_import()[i]),None,self.env.now+5000000000])
                return_vals[i]=min(self.crops.minimum_import()[i]*curr_price[i]/self.crops.mult_fact[i],self.crops.max_import()[i])
        
        return return_vals
    
    
    
    
   
class water_seller():
    def __init__(self,quantity,price):
        self.excess_water=quantity



class policy_agent():
    def __init__(self,env,crops,t1,t2,t3,price,msp,msp_storage,tax,msp_parameter):
        self.crops=crops
        self.price=price
        self.env=env
        self.msp=msp
        self.tax=tax
        self.msp_parameter=msp_parameter
        env.process(self.policy_agent_action())
        
    def policy_agent_action(self):
        while True:
            for i in range(len(self.price.value(-1))):
                if self.price.value(-1)[i]<self.crops.mult_fact[i]:
                    self.crops.min_exports_val[i]*=1.5
                    self.crops.min_exports_val[i]=min(self.crops.min_exports_val[i],10000)
                    self.crops.exports_val[i]*=1.5
                    self.crops.exports_val[i]=min(self.crops.exports_val[i],10000)
                    self.tax[i]=max(self.tax[i]/2,0.1)
                    self.crops.max_import_vals[i]/=1.5
                    self.crops.max_import_vals[i]=max(self.crops.max_import_vals[i],1000)
                    self.crops.min_import_vals[i]/=1.5
                    self.crops.min_import_vals[i]=max(self.crops.min_import_vals[i],100)
                    '''if len(self.price.value())>5:
                        if self.msp[i]<np.mean([j[i] for j in self.price.value()[-5:]])*0.5:
                            self.msp[i]*=(1+self.msp_parameter)'''
                    
                elif self.price.value(-1)[i]>self.crops.mult_fact[i]*1.5:
                    self.crops.max_import_vals[i]*=1.5
                    self.crops.max_import_vals[i]=min(self.crops.max_import_vals[i],10000)
                    self.crops.min_import_vals[i]*=1.5
                    self.crops.min_import_vals[i]=min(self.crops.min_import_vals[i],10000)
                    self.crops.min_exports_val[i]/=1.5
                    self.crops.min_exports_val[i]=max(self.crops.min_exports_val[i],1000)
                    self.crops.exports_val[i]/=1.5
                    self.crops.exports_val[i]=min(self.crops.exports_val[i],10000)
                    self.tax[i]=min(self.tax[i]*2,1)
            yield self.env.timeout(1)
                
    
        
def recompute_price(env,pr,price):
    while True:
        price.return_vals[-1]=pr.price_gen()
        yield env.timeout(1)
        
def recompute_need(env,need,price,crop):
    global need_func_val
    global need_func_change
    while True:
        need.return_vals[-1]=np.array([need_func_val,need_func_val,need_func_val])*np.maximum(np.minimum(np.array(crop.mult_fact)/price.value(-1),np.array(need_func_change)),1/np.array(need_func_change))
        yield env.timeout(1)


class mill():
    def __init__(self,env,msp,t1,t2,t3,ethanol_price,ethanol_req,crop):
        self.sugarcane_recieved=0
        self.minimum_savings=1000000
        self.env=env
        self.t1=t1
        self.t2=t2
        self.capacity=8000000000
        self.t3=t3
        self.msp=msp
        self.expenditure=2000
        self.cost_conversion_sugar=575
        self.cost_conversion_ethanol=10
        self.processing=1000
        self.ethanol_req=ethanol_req
        self.ethanol_price=ethanol_price
        self.dues=0
        self.duees=[]
        self.stockneed=None
        self.sugar_made=0
        for i in t1:
            self.duees.append([i,0])
        for i in t2:
            self.duees.append([i,0])
        for i in t3:
            self.duees.append([i,0])
        self.total_savings=simapi.stock(env,"savings of mill",300000000,self.savings_gen,[],[],[],"$0-new_val")
        self.acquiring=simapi.stock(env,"savings of mill",0,self.acquiring_gen,[],[],[],"new_val")
        self.income=simapi.stock(env,"income",0,0,[],[],[],"new_val")
        self.sold=simapi.stock(env,"sold",np.array([0,0,0],dtype=float),np.array([20000,0,0],dtype=float),[],[],[],"new_val")
        self.refining=simapi.stock(env,"refining",np.array([0,0,0],dtype=float),self.refining_gen,[],[],[],"$0+new_val")
        self.ethanol_produce=simapi.stock(env,"ethanol produce",0,self.ethanol_gen,[],[],[],"$0+new_val")
        self.ethanol_produce_logged=simapi.stock(env,"ethanol produce logged",0,0,[self.ethanol_produce],[],[""],"$1+new_val")
        self.sale_of_ethanol=simapi.stock(env,"ethanol sale",0,self.ethanol_sale_gen,[self.total_savings],[],["$1+new_val[1]"],"new_val[0]")
        self.crop=crop
        self.clearing_dues=None
        self.price=None

    def acquiring_gen(self):
        count=0
        self.sugarcane_recieved=0
        for i in self.t1:
            if i.stock_to_be_sold.value(-1)[0]>0:
                count+=1
        for i in self.t2:
            if i.stock_to_be_sold.value(-1)[0]>0:
                count+=1
        for i in self.t3:
            if i.stock_to_be_sold.value(-1)[0]>0:
                count+=1
        
        temp=[j[0] for j in self.duees]
        to_be_returned=0
        capacity=self.capacity
        if self.price is None or self.price.value(-1)[0]>self.crop.mult_fact[0]:
            capacity=self.capacity
        elif self.price is not None:
            if self.refining.value(-1)[0]<2*self.sold.value(-1)[0] and self.price.value(-1)[0]<self.ethanol_price*100:
                sugar_req=max(4*max(max((np.mean(self.sold.value(),axis=0)[0],20000))-self.refining.value(-1)[0],self.stockneed.return_vals[-1][0]),0)
                eth_req=min(max(4*self.ethanol_req-self.ethanol_produce.value(-1),0),self.ethanol_req)
                capacity=min((14.15*sugar_req+0.208*eth_req),self.capacity)
            else:
                capacity=min(max(self.capacity*self.price.value(-1)[0]/self.crop.mult_fact[0],10000),self.capacity)
                
        capacity=min(max(0,self.total_savings.value(-1)-self.minimum_savings-self.expenditure)*0.9/(self.cost_conversion_sugar*0.107+self.cost_conversion_ethanol*0.045*20),capacity)
        per_agent_max=capacity
        if count>0:
            per_agent_max=capacity/count
        for i in self.t3:
            i.sold.return_vals[-1][0]=min(i.stock_to_be_sold.value(-1)[0],per_agent_max)
            self.sugarcane_recieved+=i.sold.return_vals[-1][0]
            i.stock_to_be_sold.value(-1)[0]=0
            if i in temp:
                j=temp.index(i)
                self.duees[j][1]+=(self.msp[0])*i.sold.value(-1)[0]
            else:
                self.duees.append([i,(self.msp[0])*i.sold.value(-1)[0]])
        for i in self.t2:
            i.sold.return_vals[-1][0]=min(i.stock_to_be_sold.value(-1)[0],per_agent_max)
            self.sugarcane_recieved+=i.sold.return_vals[-1][0]
            i.stock_to_be_sold.value(-1)[0]=0
            if i in temp:
                j=temp.index(i)
                self.duees[j][1]+=(self.msp[0])*i.sold.value(-1)[0]
            else:
                self.duees.append([i,(self.msp[0])*i.sold.value(-1)[0]])
        for i in self.t1:
            i.sold.return_vals[-1][0]=min(i.stock_to_be_sold.value(-1)[0],per_agent_max)
            self.sugarcane_recieved+=i.sold.return_vals[-1][0]
            i.stock_to_be_sold.value(-1)[0]=0
            if i in temp:
                j=temp.index(i)
                self.duees[j][1]+=(self.msp[0])*i.sold.value(-1)[0]
            else:
                self.duees.append([i,(self.msp[0])*i.sold.value(-1)[0]]) 
        self.dues+=self.msp[0]*self.sugarcane_recieved
        return self.sugarcane_recieved

    
    def savings_gen(self):
        if self.total_savings.value(-1)<=self.minimum_savings:
            return self.total_savings.value(-1)
        return min(self.expenditure-0.0001*self.total_savings.value(-1),self.total_savings.value(-1))

    def refining_gen(self):
        if self.ethanol_price*100*0.55*0.074<=self.price.value(-1)[0]*0.107*0.55:
            temp=self.sugarcane_recieved*0.107*0.55
            self.sugarcane_recieved=self.sugarcane_recieved
            self.minimum_savings*=1.0001
            self.expenditure*=1.0001
            self.total_savings.return_vals[-1]-=self.cost_conversion_sugar*temp
            return np.array([temp,0,0])
        else:
            eth_req=min(max(2*self.ethanol_req-self.ethanol_produce.value(-1),0),self.ethanol_req)
            if eth_req>self.sugarcane_recieved*0.045*20:
                to_ethanol=eth_req-self.sugarcane_recieved*0.045*20
                temp=max((self.sugarcane_recieved-to_ethanol/(0.55*0.074*100))*0.107*0.55,0)
                self.sugar_made=max(self.sugarcane_recieved-to_ethanol/(0.55*0.074*100),0)
                self.minimum_savings*=1.0001
                self.expenditure*=1.0001
                self.total_savings.return_vals[-1]-=self.cost_conversion_sugar*temp      
                return np.array([temp,0,0])
            else:
                temp=self.sugarcane_recieved*0.107*0.55
                self.sugar_made=self.sugarcane_recieved
                self.minimum_savings*=1.0001
                self.expenditure*=1.0001
                self.total_savings.return_vals[-1]-=self.cost_conversion_sugar*temp
                return np.array([temp,0,0])
                
    
    def ethanol_gen(self):
        #ethanol in litres
        temp=self.sugarcane_recieved*0.045*20
        self.total_savings.return_vals[-1]-=self.cost_conversion_ethanol*temp
        eth_req=min(max(2*self.ethanol_req-self.ethanol_produce.value(-1),0),self.ethanol_req)
        if eth_req>temp:
            to_be_used=min(max(self.sugarcane_recieved-self.sugar_made,0),(eth_req-temp)/(0.55*0.074*100))
            temp+=to_be_used*0.55*0.074*100
            
            self.total_savings.return_vals[-1]-=(self.cost_conversion_sugar-self.cost_conversion_ethanol)*to_be_used*0.55*0.074
        self.sugar_made=0
        self.ethanol_price*=1.0001
        self.sugarcane_recieved=0
        return temp
    
    def ethanol_sale_gen(self):
        new_val=[]
        new_val.append(min(self.ethanol_produce.value(-1),self.ethanol_req))
        new_val.append(min(self.ethanol_produce.value(-1),self.ethanol_req)*self.ethanol_price)
        self.ethanol_produce.return_vals[-1]-=new_val[0]
        return new_val
        
    
    
    def cleardues(self):
        sum_val=0
        for i in self.duees:
            sum_val+=i[1]
        if sum_val==0:
            return 0
        max_val=max((self.total_savings.value(-1) - self.minimum_savings),0)*0.6
        for i in self.duees:
            if i[1]<0:
                i[1]=1
            i[0].total_savings.return_vals[-1]+=i[1]*min(max_val/sum_val,1)
            i[0].income.return_vals[-1]+=i[1]*min(max_val/sum_val,1)
            #print str(max_val)+" "+str(i[0].income.return_vals[-1])+" "+str(sum_val)
            self.total_savings.return_vals[-1]-=i[1]*min(max_val/sum_val,1)
            #print i[1]
            i[1]-=i[1]*min(max_val/sum_val,1)
            #print i[1]
        return 0

def plotting(i,time_to_end):
    i.plot(time_to_end)


class total_stock_gen():
    def __init__(self,t1,t2,t3,mill_agent):
        self.t1=t1
        self.t2=t2
        self.t3=t3
        self.mill_agent=mill_agent
        
    def gen(self):
        temp=np.array([1,1,1],dtype=float)
        for i in self.t1:
            temp+=i.stock_to_be_sold.value(-1)
        for i in self.t2:
            temp+=i.stock_to_be_sold.value(-1)
        for i in self.t3:
            temp+=i.stock_to_be_sold.value(-1)
        temp+=self.mill_agent.refining.value(-1)
        return temp



env=simpy.Environment()
time_to_end=50
crops=crop()
labour_details=labour()
msp_maxstock=np.array([1,1,100000000],dtype=float)
additional_land=0
additional_money=0
ethanol_req=1
msp_parameter=0.0005
msp=np.array([475,1,1],dtype=float)
total_capacity=np.array([100000,100000,100000],dtype=float)
water_quantity=1000000000000000000000000000
ethanol_price=51
tax=np.array([0.05,0.2,0.3],dtype=float)
if(len(sys.argv)>1):
    msp[0]=float(sys.argv[1])
else:
    try:
        os.chdir("test")
    except OSError:
        os.mkdir("test")
        os.chdir("test")

storage_details=storage_facilities(total_capacity)
fsval = open("storage_values.txt","w")
fimp = open("imports.txt","w") 
fexp = open("exports.txt","w") 
mill_dues=open("dues.txt","w")
no_of_t3=1
stored=[[],[],[]]
imported=np.zeros(3)
msp_storage=[[],[],[]]
storage_management=storage_stock_management(env,stored,storage_details,imported,msp_storage)
water_details=water_seller(water_quantity,10)
t1=[]
for i in range(100):
    t1.append(type_1_farmer(env,"type 1 "+str(i),random.gauss(500000+additional_money,1000),random.gauss(1.5+additional_land,0.5),msp,crops,labour_details,water_details,random.uniform(3,6)))
t2=[]
for i in range(20):
    t2.append(type_2_farmer(env,"type 2 "+str(i),random.gauss(3000000+additional_money,10000),random.gauss(3,1),msp,crops,labour_details,stored,random.uniform(3,4)))
t3=[]
for i in range(10):
    t3.append(type_3_farmer(env,"type 3 "+str(i),random.gauss(5000000,20000),random.gauss(5,2),msp,crops,labour_details,no_of_t3,stored,storage_details,3))
mill_agent=mill(env,msp,t1,t2,t3,ethanol_price,ethanol_req,crops)
#allocation_of_labour(env,labour_details,t1,t2,t3)
need=simapi.stock(env,"customer need",np.array([500,0,0],dtype=float),need_func,[],[],[],"new_val")
for i in t3:
    i.need=need
for i in t2:
    i.need=need
#msp[1]=100
wm=water_management(env,t1,water_details,crops,storage_details,stored)
#wp=water_loan_payment(t1,t3)
#water_loaned=simapi.stock(env,"taking stock from loaned entites",0,wp.payment_in_stock,[],[],[],"$0")
total_stock_gen_obj=total_stock_gen(t1,t2,t3,mill_agent)
total_new_stock=simapi.stock(env,"total stock",np.array([0.00001,0.00001,0.00001]),total_stock_gen_obj.gen,[],[],[],"new_val")
pr=price_func(need,total_new_stock,crops,stored,imported)
price=simapi.stock(env,"price",np.array(crops.price(),dtype=float),pr.price_gen,[],[],[],"new_val")
mill_agent.price=price
env.process(recompute_need(env,need,price,crops))
#sold=simapi.stock(env,"stock in market",np.zeros(3),np.zeros(3),[t1[0].total_savings,t2[0].total_savings,t3[0].total_savings,need,price,t1[0].stock_to_be_sold,t2[0].stock_to_be_sold,t3[0].stock_to_be_sold,total_stock,t1[1].total_savings,t1[1].stock_to_be_sold, t1[2].total_savings,t1[2].stock_to_be_sold],[],["$1+sum((np.array([min($6[z],$4[z]*np.divide($6,$9,out=np.zeros(len($9)),where=$9!=0)[z]) for z in range(len($9))]))*$5)","$2+sum((np.array([min($7[z],$4[z]*np.divide($7,$9,out=np.zeros(len($9)),where=$9!=0)[z]) for z in range(len($9))]))*$5)","$3+sum((np.array([min($8[z],$4[z]*np.divide($8,$9,out=np.zeros(len($9)),where=$9!=0)[z]) for z in range(len($9))]))*$5)","","","np.array([max($6[z]-$4[z]*np.divide($6,$9,out=np.zeros(len($9)),where=$9!=0)[z],0) for z in range(len($9))])","np.array([max($7[z]-$4[z]*np.divide($7,$9,out=np.zeros(len($9)),where=$9!=0)[z],0) for z in range(len($9))])","np.array([max($8[z]-$4[z]*np.divide($8,$9,out=np.zeros(len($9)),where=$9!=0)[z],0) for z in range(len($9))])","","$10+sum((np.array([min($11[z],$4[z]*np.divide($11,$9,out=np.zeros(len($9)),where=$9!=0)[z]) for z in range(len($9))]))*$5)","np.array([max($11[z]-$4[z]*np.divide($11,$9,out=np.zeros(len($9)),where=$9!=0)[z],0) for z in range(len($9))])","$12+sum((np.array([min($13[z],$4[z]*np.divide($13,$9,out=np.zeros(len($9)),where=$9!=0)[z]) for z in range(len($9))]))*$5)","np.array([max($13[z]-$4[z]*np.divide($13,$9,out=np.zeros(len($9)),where=$9!=0)[z],0) for z in range(len($9))])"],"np.array([min($4[z],$9[z]) for z in range(len($9))])")

#env.process(clearance(storage_details,t1,t2,t3,stored))
env.process(storage_fee_calc(env,storage_details,t1,t2,t3,stored))

sl=sale(mill_agent,need,price,total_new_stock,imported,stored,msp,msp_storage,msp_maxstock,env,storage_details,t1,t2,t3)



env.process(storing(env,storage_details,t1,t2,t3,stored,crops,price,total_new_stock))

ieh=import_export_handler(env,mill_agent,stored,price,crops,need,storage_details,fexp,fimp,tax,t1,t2,t3)
env.process(recompute_price(env,pr,price))


sold=simapi.stock(env,"stock in market",np.zeros(3),sl.sale_func,[], [],[],"new_val")
mill_agent.stockneed=need



#storage=storing(env,storage_details,t1,t2,t3)
#storage_loss=simapi.stock(env,"storage loss",np.zeros(3),0.2,[t1[0].stock_to_be_sold,t2[0].stock_to_be_sold,t3[0].stock_to_be_sold,t1[1].stock_to_be_sold,t1[2].stock_to_be_sold],[],["[0,0,0]","[0,0,0]","[0,0,0]","[0,0,0]","[0,0,0]"],"($1+$2+$3+$4+$5)")
#wastage=


st_manag=simapi.stock(env,"managing storage",np.array([0,0,0],dtype=float),storage_management.removal,[],[],[],"$0")

env.process(storing(env,storage_details,t1,t2,t3,stored))

mill_agent.clearing_dues=simapi.stock(env,"stock in market",0,mill_agent.cleardues,[], [],[],"new_val")

pol_agent=policy_agent(env,crops,t1,t2,t3,price,msp,msp_storage,tax,msp_parameter)
env.process(maint(env,crops,price,t1,t2,t3,stored,fsval,labour_details,msp,msp_parameter,mill_dues,mill_agent,water_details,need,time_to_end))
'''total_savings=simapi.stock(env,"total savings",10000,"0",[],[],[],"$0+100")
labour=simapi.stock(env,"labour",0,"1000",[total_savings],[],["$1-new_val"],"new_val")
planting_cost=simapi.stock(env,"planting cost",0,"0",[total_savings],["self.env.now%8==0"])
loan=simapi.stock(env,"loan",0,None,[total_savings,labour,planting_cost],[])
returns=simapi.stock(env,"returns",0,"0",[],["self.env.now%8>4"])
price=simapi.stock(env,"price",0,None,[],[])
sale=simapi.stock(env,"sale",0,None,[returns,price,total_savings],[])
loss=simapi.stock(env,"loss",0,None,[returns],[])

domestic=simapi.stock(env,"domestic",0,None,[supply])
total_sav=simapi.stock(env,"tot_sav",1000,None,[export,domestic])'''
env.run(until=time_to_end)
fsval.close()
fimp.close()
fexp.close()
mill_dues.close()
fsval.close()
f1=[]
f2=[]
f3=[]
i=0
processes=[]
while i<len(t1):
    f1=(open("typeofcrop1 "+str(i)+".txt","w"))
    f1.write("\n"+str(t1[i].type_of_crop_list))
    f1.close()
    i+=1
i=0
while i<len(t2):
    f1=(open("typeofcrop2 "+str(i)+".txt","w"))
    f1.write("\n"+str(t2[i].type_of_crop_list))
    f1.close()
    i+=1
i=0
while i<len(t3):
    f1=(open("typeofcrop3 "+str(i)+".txt","w"))
    f1.write("\n"+str(t3[i].type_of_crop_list))
    f1.close()
    i+=1

plt.figure()
plt.plot([z[0] for z in price.value()])
plt.title("price type 0")
plt.savefig("price type 0.png")
plt.figure()
plt.figure()
plt.plot([z[1] for z in price.value()])
plt.title("price type 1")
plt.savefig("price type 1.png")
plt.figure()
plt.figure()
plt.plot([z[2] for z in price.value()])
plt.title("price type 2")
plt.savefig("price type 2.png")
plt.figure()
plt.plot([np.floor(z[0]) if z[0]>=0 else 0 for z in ieh.import_export.value()])
plt.title("imports type 0")
plt.savefig("imports type 0.png")
plt.figure()
plt.plot([np.floor(-1*z[0]) if z[0]<=0 else 0 for z in ieh.import_export.value()])
plt.title("EXPORTS type 0")
plt.savefig("exports type 0.png")

plt.figure()
plt.plot([np.floor(z[1]) if z[1]>=0 else 0 for z in ieh.import_export.value()])
plt.title("imports type 1")
plt.savefig("imports type 1.png")
plt.figure()
plt.plot([np.floor(-1*z[1]) if z[1]<=0 else 0 for z in ieh.import_export.value()])
plt.title("EXPORTS type 1")
plt.savefig("exports type 1.png")


plt.figure()
plt.plot([np.floor(z[2]) if z[2]>=0 else 0 for z in ieh.import_export.value()])
plt.title("imports type 2")
plt.savefig("imports type 2.png")
plt.figure()
plt.plot([np.floor(-1*z[2]) if z[2]<=0 else 0 for z in ieh.import_export.value()])
plt.title("EXPORTS type 2")
plt.savefig("exports type 2.png")



plt.figure()
plt.plot(sold.value())
plt.title("sold")
plt.savefig("sold")

plt.figure()
plt.plot(mill_agent.total_savings.value())
plt.title("mill savings")
plt.savefig("mill savings")

plt.figure()
plt.plot(mill_agent.sold.value())
plt.title("mill sold")
plt.savefig("mill sold")


plt.figure()
plt.plot(mill_agent.income.value())
plt.title("mill income")
plt.savefig("mill income")
plt.figure()
plt.plot(mill_agent.ethanol_produce_logged.value())
plt.title("ethanol produce")
plt.savefig("ethanol produce")
plt.figure()
plt.plot(mill_agent.sale_of_ethanol.value())
plt.title("ethanol sale")
plt.savefig("ethanol sale")

plt.figure()
plt.plot(mill_agent.refining.value())
plt.title("refining sale")
plt.savefig("refining sale")

counter=0
for i in t1:
    if counter==10:
        break
    plotting(i,time_to_end)
    plt.close("all")
    counter+=1
counter=0
for i in t2:
    if counter==10:
        break
    plotting(i,time_to_end)
    plt.close("all")
    counter+=1
counter=0
for i in t3:
    if counter==10:
        break
    plotting(i,time_to_end)
    plt.close("all")
    counter+=1
        
'''plt.figure()
plt.plot([z[0] for z in need.value()])
plt.title("need type 0")
plt.figure()
plt.plot([z[1] for z in need.value()])
plt.title("need type 1")
plt.figure()
plt.plot([z[2] for z in need.value()])
plt.title("need type 2")'''

os.chdir("..")
t1avg= np.mean([i.total_savings.value(-1) for i in t1])
t2avg= np.mean([i.total_savings.value(-1) for i in t2])
t3avg= np.mean([i.total_savings.value(-1) for i in t3])

t1_avg_rel=np.mean([np.mean(i.income.value())/i.initial_land for i in t1])
t2_avg_rel=np.mean([np.mean(i.income.value())/i.initial_land for i in t2])
t3_avg_rel=np.mean([np.mean(i.income.value())/i.initial_land for i in t3])


print str(t1avg)+','+str(t2avg)+','+str(t3avg)+','+str(t1_avg_rel)+','+str(t2_avg_rel)+','+str(t3_avg_rel)
'''
plt.figure()
plt.plot(storage_loss.value())
plt.title("total storage loss")'''
#plt.show()
