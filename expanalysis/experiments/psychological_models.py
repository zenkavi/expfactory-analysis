from math import exp
import numpy

class Two_Stage_Model(object):
    def __init__(self,alpha1,alpha2,lam,B1,B2,W,p):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.lam = lam
        self.B1= B1
        self.B2 = B2
        self.W = W
        self.p = p
        # stage action possibilities
        self.stage_action_list = {0: (0,1), 1: (2,3), 2: (4,5)}
        # transition counts
        self.transition_counts = {(0,1):0, (0,2):0, (1,1):0, (1,2):0}
        # initialize Q values
        self.Q_TD_values = numpy.ones((3,6))*0
        self.Q_MB_values = numpy.ones((3,6))*0
        self.sum_neg_ll = None

    def updateQTD(self,r,s1,a1,s2=None,a2=None,alpha=.05):
        if s2 == None:
            delta = r - self.Q_TD_values[s1,a1]
        else:
            delta = r + self.Q_TD_values[s2,a2] - self.Q_TD_values[s1,a1]
        self.Q_TD_values[s1,a1] += alpha*delta
        return delta
    
    def updateQMB(self,T):
        self.Q_MB_values[1:3,:] = self.Q_TD_values[1:3,:]
        for a in self.stage_action_list[0]:
            self.Q_MB_values[0,a] = T[(a,1)] * numpy.max(self.Q_TD_values[1,2:4]) + \
                                T[(a,2)] * numpy.max(self.Q_TD_values[2,4:6])
        
    def trialUpdate(self,s1,s2,a1,a2,r,alpha1,alpha2,lam):
        # update TD values
        delta1 = self.updateQTD(0,s1, a1, s2, a2, alpha1)
        delta2 = self.updateQTD(r, s2, a2, alpha=alpha2)
        self.Q_TD_values[(s1, a1)] += alpha1*lam*delta2
        # update MB values
        self.transition_counts[(a1,s2)] += 1
        # define T:
        if (self.transition_counts[(0,1)]+self.transition_counts[(1,2)]) > \
            (self.transition_counts[(0,2)]+self.transition_counts[(1,1)]):
            T = {(0,1):.7, (0,2):.3, (1,1):.3, (1,2):.7}
        else: 
            T = {(0,1):.3, (0,2):.7, (1,1):.7, (1,2):.3}
        self.updateQMB(T)
    
    def get_softmax_probs(self,stages,last_choice):
        W = self.W
        if type(stages) != list:
            stages = [stages]
        # stage one and two choices
        P_action = numpy.zeros(2)
        # choice probabilities
        choice_probabilities = []
        for stage in stages:
            for i,a in enumerate(self.stage_action_list[stage]):
                Qnet = (W)*self.Q_MB_values[stage,a] + (1-W)*self.Q_TD_values[stage,a]
                repeat = (self.p*(a==last_choice))
                P_action[i] = exp(self.B1*(Qnet+repeat))
            P_action/=numpy.sum(P_action)
            choice_probabilities.append(P_action.copy())
        return choice_probabilities
    
    def run_trial(self,trial,last_choice):
        s1 = int(trial['stage']); s2 = int(trial['stage_second'])
        a1 = int(trial['stim_selected_first']); a2 = int(trial['stim_selected_second'])
        r = int(trial['feedback'])
        # return probability of all actions
        probs1, probs2 = self.get_softmax_probs([s1,s2],last_choice)
        # get probability of selected actions
        Pa1 = probs1[a1]
        Pa2 = probs2[self.stage_action_list[s2].index(a2)]
        self.trialUpdate(s1,s2,a1,a2,r,self.alpha1,self.alpha2,self.lam)
        return Pa1,Pa2
        
    def run_trials(self, df):
        # run trials
        last_choice = -1
        action_probs = []
        Q_vals = []
        MB_vals = []
        for i,trial in df.iterrows():
            Q_vals.append(self.Q_TD_values.copy())
            MB_vals.append(self.Q_MB_values.copy())
            Pa1, Pa2 = self.run_trial(trial,last_choice)
            action_probs.append((Pa1,Pa2))
            last_choice = trial['stim_selected_first']
        self.sum_neg_ll = numpy.sum(-numpy.log(list(zip(*action_probs))[0])) + numpy.sum(-numpy.log(list(zip(*action_probs))[1]))   
    
    def simulate(self, ntrials=10):
        trials = []
        reward_probs = numpy.random.rand(6)*.5+.25 #rewards for each action
        reward_probs[0:2] = 0
        transition_probs = [.7,.3] #transition to new stages (probability to go to stage 2)
        # initial conditions
        last_choice = -1
        for trial in range(ntrials):
            s1 = 0
            # get first choice without knowing the second choice
            first_action_probs = self.get_softmax_probs(s1,last_choice)[0]
            a1 = numpy.random.choice(self.stage_action_list[s1], p=first_action_probs)
            # get second stage
            s2 = numpy.random.binomial(1,transition_probs[a1])+1
            second_action_probs = self.get_softmax_probs(s2,last_choice)[0]
            a2 = numpy.random.choice(self.stage_action_list[s2], p=second_action_probs)
            feedback = numpy.random.binomial(1,reward_probs[a2]) 
            trials.append({'stage':s1, 'stage_second':s2,
                           'stim_selected_first':a1,'stim_selected_second':a2,
                           'feedback':feedback,
                           'first_action_prob': first_action_probs[a1],
                            'second_action_prob': second_action_probs[self.stage_action_list[s2].index(a2)]})
            self.trialUpdate(s1,s2,a1,a2,feedback,self.alpha1,self.alpha2,self.lam)
            last_choice = a1
            reward_probs[2:]+=numpy.random.randn(4)*.025
            reward_probs[2:] = numpy.maximum(numpy.minimum(reward_probs[2:],.75),.25)
        return trials
    
    def get_neg_ll(self):
        return self.sum_neg_ll

