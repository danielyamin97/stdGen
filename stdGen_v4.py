'''
    This script generates a STD network and writes it to a file.
    As the network needs to be expanded in time, we define
    t_delta = time step in seconds (or duration in of 1 unit)
    t_horizon = time horizon
    t_max = T/t_delta ----------- Note that, it is better to have time units in t_delta     
    Finally, for t>t_horizon the travel time distribution is assumed to be fixed
    
    The discrete travel times are selected based on quantiles
    
    The outputfile characteristics are 
    (i) Nodes Links and T_max
    (ii) List of links represented as tnode and hnode
    (iii) The travel time realizations and probabilities: t_node, h_node, ts at t_node, traveltime<>probability
    
    Version1: Here the travel times are randomly generated as per normal/lognormal distribution
    Version3: The link lengths are random generated between 250m and 1000m 
    Version4: Minor changes in the code (handling exceptions/errors)/DY
    
    Please cite the following work when using this code_
    - Prakash, A. A., & Srinivasan, K. K. (2017). Finding the most reliable strategy on stochastic and time-dependent transportation networks: A hypergraph based formulation. Networks and Spatial Economics, 17(3), 809-840.
    - Yamín, D., Medaglia, A. L., & Prakash, A. A. (2022). Exact bidirectional algorithm for the least expected travel-time path problem on stochastic and time-dependent networks. Computers & Operations Research, 141, 105671.
    
    Feel free to change parameters/seeting depending on your application. 
'''
from networkx import DiGraph
import numpy as np
import random
import datetime as dt
from scipy import stats,inf
from operator import itemgetter
import pdb,math,csv,os,copy
from collections import OrderedDict


#--------GLOBALS-----------------------------------------------------------------------------------


t_delta = 30 # in seconds
t_horizon = 7200 # in sec
t_max = t_horizon/t_delta
offpeak = 30 # in t_delta units
p1,p2,p3 = 60,60,60 # in t_delta units
chi = 1.5

# Network Files
# Travel Time files
#linkmap_fname = os.path.join(inputfolder,'old-new-linkmap.csv')
#linktt_fname = os.path.join(inputfolder,'linkwise_tts_8_11.csv')
#ltypett_fname = os.path.join(inputfolder,'ltype_tts_8_11.csv')

numNodes,numArcs =0,0

#-------The mean and var ranges--------
#seed = 125
#np.random.seed(seed)


mean_lb,mean_ub = 60,120 # off peak mean travel times (secs) /km
#len_lb,len_ub = 0.250, 1 # Kilometers
#std_lb,std_ub = 65,500

#--------------------------------------------------------------------------------------------------

net_filename = "./preprocessed/AnaheimAux.txt" #File from which the network is built


cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory


global numNodes,numArcs
G = DiGraph()
with open(net_filename,'r') as netfile:
    numNodes = netfile.readline().strip()
    numArcs = netfile.readline().strip()
    for arow in netfile:
        arow = arow.strip()
        arow = arow.split()
        tnode,hnode = arow[0],arow[1]
        length,ltype = arow[2],0                 # assuming 1000 m length and a dummy link type 0
        G.add_edge(tnode,hnode,length=length,ltype=ltype)  

#--------------------------------------------------------------------------------------------------

'''
    get_linktt(): Generates time-dependent link travel time distributions
'''
# Creating a standard dictionary
stan_dict = {}
#time_stamps = range(0,t_horizon,t_delta)
time_stamps = range(int(0),int(t_max))
for ts in time_stamps:
    stan_dict[ts] = {'mean':0,'std':0}

assert offpeak*2+p1+p2+p3 == t_max
# creating dictionary for factors to multiply offpeak mean link tt
mult_mean = {}
for ts in time_stamps[0:offpeak]:
    mult_mean[ts] = 1.
for ts in time_stamps[-offpeak:]:
    mult_mean[ts] = 1.        
for ts in time_stamps[offpeak:offpeak+p1]:
    slope = (chi-1.)/p1
    mult_mean[ts] = 1 + (ts-offpeak)*slope
for ts in time_stamps[offpeak+p1:offpeak+p1+p2]:
    mult_mean[ts] = chi        
for ts in time_stamps[offpeak+p1+p2:offpeak+p1+p2+p3]:
    slope = (1.-chi)/p3
    mult_mean[ts] = chi + (ts-(offpeak+p1+p2))*slope

aList = [mult_mean[i] for i in time_stamps]
assert 0. not in aList

meanList = []
linktt = {}
for link in G.edges():
    linktt[link] = copy.deepcopy(stan_dict)
    for ts in time_stamps:
        mean = np.random.uniform(mean_lb,mean_ub) * mult_mean[ts] * float(G.edges[link]['length'])
        std = (60./1.6)*(-0.4736 + 0.9936*mean/60.)*float(G.edges[link]['length'])       # applying Mahmassani's equation
        linktt[link][ts]['mean'] = mean
        linktt[link][ts]['std'] = std
        meanList.append(mean)
        


#--------------------------------------------------------------------------------------------------

edgedist_dict = {}
for edge in G.edges:
    edgedist_dict[edge] = {}
    for ts in range(int(t_max)):
        m,v = linktt[edge][ts]['mean'],linktt[edge][ts]['std']**2
        sig = math.sqrt(math.log(1+v/m**2))
        mu = math.log(m) - 0.5*sig**2            
        edgett_rv = stats.lognorm(scale=math.exp(mu),s=sig)
        tt_list = list(edgett_rv.ppf([0.05,0.25,0.5,0.75,0.95]))
        tt_list = [item for item in tt_list if item>0]
        tt_list = [int(math.ceil(i)) for i in tt_list]
        unique = []
        [unique.append(item) for item in tt_list if item not in unique]
        unique.sort()

        #print edge, ts, unique
        #pdb.set_trace()
        
        # Discretizing the distribution - DONE BASED ON QUNATILES
        arrtime_probs = {}
        low_lim = -inf
        currTime = ts*t_delta
        for i in range(int(len(unique)-1)):
            up_lim = unique[i]
            arrtime_probs[currTime+up_lim] = edgett_rv.cdf(up_lim) - edgett_rv.cdf(low_lim)
            low_lim = up_lim
        up_lim = unique[-1]    
        arrtime_probs[currTime+up_lim] = 1 - edgett_rv.cdf(low_lim)
        #pdb.set_trace()
        edgedist_dict[edge][ts] = arrtime_probs        # Adding the arrival prob dict to edge attribute dictionary
            

#--------------------------------------------------------------------------------------------------- 


graph = G
out_filename = "./STDFiles/AnaheimSTD.txt"   #File in which the STD network is written
with open(out_filename,'w') as outfile:
    outfile.write(str(numNodes)+"\n")
    outfile.write(str(graph.number_of_edges())+"\n")
    outfile.write(str(int(t_horizon/t_delta - 1))+"\n")
    edgeList = graph.edges()
    edgeList = sorted(edgeList,key=lambda x:(int(x[0]),int(x[1])))
    for edge in edgeList:
        tnode,hnode = edge[0],edge[1]
        outfile.write(str(int(tnode)-1)+","+str(int(hnode)-1)+"\n")
    header = "tnode,hnode,ts,traveltime<>probability"
    outfile.write(header + "\n")
    for edge in edgeList:
        tnode,hnode = edge
        for ts in edgedist_dict[edge]:
            a_dict = edgedist_dict[edge][ts]
            a_dict = list(a_dict.items())
            a_dict = sorted(a_dict,key=itemgetter(0))
            b_dict = OrderedDict(a_dict) 
            a_dict = OrderedDict()
            #pdb.set_trace()
            # making dictionary of travel times as per the t_delta resolution
            for key,val in b_dict.items():
                akey = round(key/t_delta - ts)
                if akey == 0:
                    akey = 1
                #--if key is already present, we add to its probability
                if akey in a_dict:
                    a_dict[akey] = a_dict[akey]+val
                else:
                    a_dict[akey] = val
            a_list = [str(k)+'<>'+"%.12f"%v for k,v in a_dict.items()]
            b_list = [int(tnode)-1,int(hnode)-1,ts]
            b_list = ','.join([str(i) for i in b_list])
            a_list = ';'.join([str(i) for i in a_list])
            ab_list = b_list + ',' + a_list
            outfile.write(ab_list+ "\n")

#--------------------------------------------------------------------------------------------------

