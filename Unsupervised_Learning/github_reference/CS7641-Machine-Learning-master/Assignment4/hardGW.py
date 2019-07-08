'''
Created on Apr 9, 2017

@author: Jon
'''
import sys
sys.path.append('./burlap.jar')
import java
from collections import defaultdict
from time import clock
from burlap.behavior.policy import Policy;
from burlap.assignment4 import BasicGridWorld;
from burlap.behavior.singleagent import EpisodeAnalysis;
from burlap.behavior.singleagent.auxiliary import StateReachability;
from burlap.behavior.singleagent.auxiliary.valuefunctionvis import ValueFunctionVisualizerGUI;
from burlap.behavior.singleagent.learning.tdmethods import QLearning;
from burlap.behavior.singleagent.planning.stochastic.policyiteration import PolicyIteration;
from burlap.behavior.singleagent.planning.stochastic.valueiteration import ValueIteration;
from burlap.behavior.valuefunction import ValueFunction;
from burlap.domain.singleagent.gridworld import GridWorldDomain;
from burlap.oomdp.core import Domain;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent import SADomain;
from burlap.oomdp.singleagent.environment import SimulatedEnvironment;
from burlap.oomdp.statehashing import HashableStateFactory;
from burlap.oomdp.statehashing import SimpleHashableStateFactory;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent.explorer import VisualExplorer;
from burlap.oomdp.visualizer import Visualizer;
from burlap.assignment4.util import BasicRewardFunction;
from burlap.assignment4.util import BasicTerminalFunction;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.assignment4.EasyGridWorldLauncher import visualizeInitialGridWorld
from burlap.assignment4.util.AnalysisRunner import calcRewardInEpisode, simpleValueFunctionVis,getAllStates
import csv
from collections import deque

def dumpCSV(iters, times,rewards,steps,convergence,world,method):
    fname = '{} {}.csv'.format(world,method)
    assert len(iters)== len(times)
    assert len(iters)== len(rewards)
    assert len(iters)== len(steps)
    assert len(iters)== len(convergence)
    with open(fname,'wb') as f:
        f.write('iter,time,reward,steps,convergence\n')
        writer = csv.writer(f,delimiter=',')
        writer.writerows(zip(iters,times,rewards,steps,convergence))

def dumpCSVp(iters, times,rewards,steps,convergence,world,method,policy):
    fname = '{} {}.csv'.format(world,method)
    assert len(iters)== len(times)
    assert len(iters)== len(rewards)
    assert len(iters)== len(steps)
    assert len(iters)== len(convergence)
    assert len(iters)== len(policy)
    with open(fname,'wb') as f:
        f.write('iter,time,reward,steps,convergence,policy\n')
        writer = csv.writer(f,delimiter=',')
        writer.writerows(zip(iters,times,rewards,steps,convergence,policy))
    
    
def runEvals(initialState,plan,rewardL,stepL):
    r = []
    s = []
    for trial in range(evalTrials):
        ea = plan.evaluateBehavior(initialState, rf, tf,300);
        r.append(calcRewardInEpisode(ea))
        s.append(ea.numTimeSteps())
    rewardL.append(sum(r)/float(len(r)))
    stepL.append(sum(s)/float(len(s))) 



if __name__ == '__main__':
    if len(sys.argv) == 3:
        epsilon = float(sys.argv[1])
        lr = float(sys.argv[2])
        doQ = True
    else:
        doQ = True
        epsilon = 0.7
        lr = 0.1
    
    world = 'Hard'
    discount=0.90
    MAX_ITERATIONS = 200;
    NUM_INTERVALS = 200;
    evalTrials = 1;
    userMap = [[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
     [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
     [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
     [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1]]


        
    n = len(userMap)
    tmp = java.lang.reflect.Array.newInstance(java.lang.Integer.TYPE,[n,n])
    for i in range(n):
        for j in range(n):
            tmp[i][j]= userMap[i][j]
    userMap = MapPrinter().mapToMatrix(tmp)
    maxX = maxY= n-1
    
    gen = BasicGridWorld(userMap,maxX,maxY)
    domain = gen.generateDomain()
    initialState = gen.getExampleState(domain);

    rf = BasicRewardFunction(maxX,maxY,userMap)
    tf = BasicTerminalFunction(maxX,maxY)
    env = SimulatedEnvironment(domain, rf, tf,initialState);
#    Print the map that is being analyzed
    print "/////Hard Grid World Analysis/////\n"
    MapPrinter().printMap(MapPrinter.matrixToMap(userMap));
    visualizeInitialGridWorld(domain, gen, env)
#     input('c')
    hashingFactory = SimpleHashableStateFactory()
    increment = MAX_ITERATIONS/NUM_INTERVALS
    timing = defaultdict(list)
    rewards = defaultdict(list)
    steps = defaultdict(list)
    convergence = defaultdict(list)
    policy_converged = defaultdict(list)
    last_policy = defaultdict(list)
#     # Value Iteration
    iterations = range(1,MAX_ITERATIONS+1)
  

    print "//Hard Value Iteration Analysis//"
    for nIter in iterations:
        startTime = clock()
        vi = ValueIteration(domain,rf,tf,discount,hashingFactory,-1, nIter); #//Added a very high delta number in order to guarantee that value iteration occurs the max number of iterations for comparison with the other algorithms.
            # run planning from our initial state
        vi.setDebugCode(0)
        p = vi.planFromState(initialState);
        timing['Value'].append((clock()-startTime)*1000)
        convergence['Value'].append(vi.latestDelta)
        # evaluate the policy with evalTrials roll outs
        runEvals(initialState,p,rewards['Value'],steps['Value'])
        if nIter == 1:
            simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory, "Value Iteration {}".format(nIter))
    MapPrinter.printPolicyMap(vi.getAllStates(), p, gen.getMap());
    print "\n\n\n"
    simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory, "Value Iteration {}".format(nIter))
    dumpCSV(iterations, timing['Value'], rewards['Value'], steps['Value'],convergence['Value'], world, 'Value')


    print "//Hard Policy Iteration Analysis//"
    for nIter in iterations:
        startTime = clock()
        pi = PolicyIteration(domain,rf,tf,discount,hashingFactory,-1,1, nIter); #//Added a very high delta number in order to guarantee that value iteration occurs the max number of iterations for comparison with the other algorithms.
        # run planning from our initial state
        pi.setDebugCode(0)
        p = pi.planFromState(initialState);
        timing['Policy'].append((clock()-startTime)*1000)
        convergence['Policy'].append(pi.lastPIDelta)
        # evaluate the policy with one roll out visualize the trajectory
        runEvals(initialState,p,rewards['Policy'],steps['Policy'])
        if (nIter == 1):
            simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, "Policy Iteration{}".format(nIter))
        #ea = p.evaluateBehavior(initialState, rf, tf);
        #rewards['Policy'].append(calcRewardInEpisode(ea));
        #steps['Policy'].append(ea.numTimeSteps());
        policy = pi.getComputedPolicy()
        allStates = pi.getAllStates()
        current_policy = [[(action.ga, action.pSelection) for action in policy.getActionDistributionForState(state)] for state in allStates]
        policy_converged['Policy'].append(current_policy == last_policy)
        last_policy = current_policy
    MapPrinter.printPolicyMap(pi.getAllStates(), p, gen.getMap());
    print "\n\n\n"
    simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, "Policy Iteration {}".format(nIter))
    dumpCSVp(iterations, timing['Policy'], rewards['Policy'], steps['Policy'],convergence['Policy'], world, 'Policy',policy_converged['Policy'])

    MAX_ITERATIONS = 500;
    NUM_INTERVALS = 500;
    increment = MAX_ITERATIONS/NUM_INTERVALS
    iterations = range(1,MAX_ITERATIONS+1)
    last10Rewards= deque([10]*10,maxlen=10)
    Qname = 'Q-Learning L{:0.1f} E{:0.1f}'.format(lr,epsilon)
    agent = QLearning(domain,discount,hashingFactory,1,lr,epsilon)
    agent.setDebugCode(0)
    print "//Hard {} Iteration Analysis//".format(Qname)
    print Qname

    for nIter in iterations:
        startTime = clock()
        if nIter%50==0:
            print nIter
        ea = agent.runLearningEpisode(env)
        env.resetEnvironment()
        agent.initializeForPlanning(rf, tf, 1)
        p = agent.planFromState(initialState)     # run planning from our initial state
        timing[Qname].append((clock()-startTime)*1000)
        last10Rewards.append(agent.maxQChangeInLastEpisode)
        convergence[Qname].append(sum(last10Rewards)/10.)
        # evaluate the policy with one roll out visualize the trajectory
        runEvals(initialState,p,rewards[Qname],steps[Qname])
    MapPrinter.printPolicyMap(getAllStates(domain,rf,tf,initialState), p, gen.getMap());
    print "\n\n"
    simpleValueFunctionVis(agent, p, initialState, domain, hashingFactory, Qname+' {}'.format(nIter))
    #input('d')
    dumpCSV(iterations, timing[Qname], rewards[Qname], steps[Qname],convergence[Qname], world, Qname)
     