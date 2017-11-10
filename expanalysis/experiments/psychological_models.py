from math import exp
import numpy
import json
from itertools import product
from scipy.stats.distributions import beta

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


# Functions to define Hierarchical Rule MoE Model (Frank & Badre, 2011)
class Flat_SubExpert():
    def __init__(self, features, data, kappa):
        self.kappa = kappa # kappaerature for softmax
        self.features = features
        self.feature_types = [numpy.unique(data.loc[:, f]) for f in features]
        self.actions = sorted(numpy.unique([a for a in data.key_press if a > 0]))
        # create dictionary of learned beta parameters
        self.reward_probabilities = {}
        for key in product(*self.feature_types, self.actions):
            self.reward_probabilities[key] = {'a':1, 'b': 1}
            
    def update(self, trial):
        correct = trial.correct
        action = trial.key_press
        features = trial[self.features].tolist()
        update_key = 'a' if correct else 'b'
        self.reward_probabilities[tuple(features + [action])][update_key] += 1
        
    def get_action_probs(self, trial):
        def subset(lst1, lst2):
            """test if lst1 is subset of lst2"""
            return set(lst1) <= set(lst2)
        features = trial[self.features].tolist()
        key_subset = [k for k in self.reward_probabilities.keys() if subset(features,k)]
        action_probs = {}
        for key in key_subset:
            raw_prob = beta(**self.reward_probabilities[key]).mean()
            # take softmax
            prob = numpy.e**(raw_prob/self.kappa)
            action_probs[key[-1]] = prob
        # normalize by total
        sum_probs = sum(action_probs.values())
        action_probs = {k: v/sum_probs for k,v in action_probs.items()}
        return action_probs
    
class Hierarchical_SubExpert():
    """ Suboridinate Hierarchical Expert
    
    This class instantatiates an expert that learns about one or more features
    contextualized on another feature. In the context of the model, this class
    is used for simple experts that learn (for example) the reward probabilities
    of (shape|color).
    """
    def __init__(self, features, context, data, kappa):
        self.kappa = kappa # kappaerature for softmax
        self.features = features
        self.feature_types = [numpy.unique(data.loc[:, f]) for f in features]
        self.context = context
        self.context_types = numpy.unique(data.loc[:, context])
        self.actions = sorted(numpy.unique([a for a in data.key_press if a > 0]))
        # create dictionary of learned beta parameters
        self.reward_probabilities = {}
        for c in self.context_types:
            self.reward_probabilities[c] = {}
            for key in product(*self.feature_types, self.actions):
                self.reward_probabilities[c][key] = {'a':1, 'b': 1}
            
    def update(self, trial):
        correct = trial.correct
        action = trial.key_press
        features = trial[self.features].tolist()
        context = trial[self.context]
        update_key = 'a' if correct else 'b'
        self.reward_probabilities[context][tuple(features + [action])][update_key] += 1
        
    def get_action_probs(self, trial):
        def subset(lst1, lst2):
            """test if lst1 is subset of lst2"""
            return set(lst1) <= set(lst2)
        features = trial[self.features].tolist()
        context = trial[self.context]
        key_subset = [k for k in self.reward_probabilities[context].keys() if subset(features,k)]
        action_probs = {}
        for key in key_subset:
            raw_prob = beta(**self.reward_probabilities[context][key]).mean()
            # take softmax
            prob = numpy.e**(raw_prob/self.kappa)
            action_probs[key[-1]] = prob
        # normalize by total
        sum_probs = sum(action_probs.values())
        action_probs = {k: v/sum_probs for k,v in action_probs.items()}
        return action_probs


class Expert():
    def update_confidence(self, trial):
        choice = trial.key_press
        r = trial.correct # reward
        e_action_probs = [e.get_action_probs(trial) for e in self.experts]
        # models are assigned credit for the choice if their action_probs
        # gave a higher value to the actual choice than the others
        credit_assignment = []
        for action_probs in e_action_probs:
            choice_prob = action_probs[choice]
            others = [v for k,v in action_probs.items() if k != choice]
            credited = all(numpy.less(others, choice_prob))
            credit_assignment.append(credited)
        # update 
        # if the expert contributed the the choice, set reward to "r"
        # if the expert did not contribute, set the reward to 1-r.
        updates = [r*c+(1-r)*(1-c) for c in credit_assignment]
        # update alpha and beta based on updates
        for i, update in enumerate(updates):
            update_param = ['b','a'][int(update)]
            self.confidences[i][update_param] += 1
            
    def update_experts(self, trial):
        for e in self.experts:
            # if the subordinate experts are "subordinate" they only have an
            # update method. otherwise they have an update_experts method
            try:
                e.update(trial)
            except AttributeError:
                e.update_confidence(trial)
                e.update_experts(trial)
            
    def get_action_probs(self, trial):
        e_action_probs = [e.get_action_probs(trial) for e in self.experts]
        e_confidences = self.get_expert_confidences(trial)
        action_probs = {}
        for action in self.actions:
            # get action probs across experts
            probs = [e[action] for e in e_action_probs]
            # weight probs by expert attention
            weighted_prob = numpy.dot(probs, e_confidences)
            action_probs[action] = weighted_prob
        return action_probs        
            
class Flat_Expert(Expert):
    """ Model for Hierarchical Rule Learning Task
    
    ref: Frank, M. J., & Badre, D. (2012). Mechanisms of hierarchical... (Part1)
    """
    def __init__(self, data, kappa, zeta, alphaC, alphaO, alphaS,
                 beta2, beta3):
        """ Initialize the model
        
        Args:
            data: dataframe of hierarchical rule task
            kappa: kappaerature for individual experts softmax function
            zeta: softmax parameter to arbitrate between experts
            uni_confidences: initial alpha and beta params
                for unidimensional experts. These should be supplied in
                orient, color, shape order.
            full_confidence: initial alpha and beta perams for fully
                conjunctive expert
            beta2: beta parameter for 2-way conjunctions. Alpha parameter
                is determined by unidimensional experts
        """
        self.zeta = zeta
        self.actions = sorted(numpy.unique([a for a in data.key_press if a > 0]))
        # single feature
        self.orient_e = Flat_SubExpert(['orientation'], data, kappa)
        self.color_e = Flat_SubExpert(['border'], data, kappa)
        self.shape_e = Flat_SubExpert(['stim'], data, kappa)
        # 2 combination 
        self.orient_color_e = Flat_SubExpert(['orientation', 'border'], data, kappa)
        self.orient_shape_e = Flat_SubExpert(['orientation', 'stim'], data, kappa)
        self.shape_color_e = Flat_SubExpert(['stim', 'border'], data, kappa)   
        # all 3
        self.all_e = Flat_SubExpert(['orientation','border','stim'], data,kappa)
        self.experts = [self.orient_e,
                       self.color_e,
                       self.shape_e,
                       self.orient_color_e,
                       self.orient_shape_e,
                       self.shape_color_e,
                       self.all_e]
        # create disctionary of params for beta distributions representing
        # the confidence in each expert
        O_confidence = {'a': 1+alphaO, 'b': 2}
        C_confidence = {'a': 1+alphaC, 'b': 2}
        S_confidence = {'a': 1+alphaS, 'b': 2}
        OC_confidence = {'a': 1+(alphaO+alphaC)/2, 'b': 2+beta2}
        OS_confidence = {'a': 1+(alphaO+alphaS)/2, 'b': 2+beta2}
        SC_confidence = {'a': 1+(alphaS+alphaC)/2, 'b': 2+beta2}
        OSC_confidence = {'a': 1+(alphaO+alphaS+alphaC)/3, 'b': 3+beta3}

        
        self.confidences = [O_confidence,
                            C_confidence,
                            S_confidence,
                            OC_confidence,
                            OS_confidence,
                            SC_confidence,
                            OSC_confidence]
    
    def get_expert_confidences(self, trial):
        # get attention weights (softmax of confidences)
        e_confidences = [numpy.e**(beta(**p).mean()/self.zeta) for p in self.confidences]
        e_confidences = [i/sum(e_confidences) for i in e_confidences]
        return e_confidences
    
class Hierarchical_Expert(Expert):
    """ Hierarchical expert with two subordinate
    
    This expert reflects the combination of two subordinate experts over one
    context. For example, if "color" is the context, the two subordinate would
    be shape|color and orientation|color
    """
    def __init__(self, subfeatures, context, data, kappa, zeta):
        self.actions = sorted(numpy.unique([a for a in data.key_press if a > 0]))
        self.zeta = zeta
        self.context = context
        self.context_types = numpy.unique(data.loc[:, context])
        # define subordinate condition experts
        self.experts = []
        for feature in subfeatures:
            expert = Hierarchical_SubExpert([feature], context, data, kappa)
            self.experts.append(expert)
        # create disctionary of params for beta distributions representing
        # the confidence in each expert
        self.confidences = {}
        for c in self.context_types:
            self.confidences[c] = [{'a': 1, 'b': 1} for _ in range(len(self.experts))]

    def update_confidence(self, trial):
        choice = trial.key_press
        r = trial.correct # reward
        context = trial[self.context]
        e_action_probs = [e.get_action_probs(trial) for e in self.experts]
        # models are assigned credit for the choice if their action_probs
        # gave a higher value to the actual choice than the others
        credit_assignment = []
        for action_probs in e_action_probs:
            choice_prob = action_probs[choice]
            others = [v for k,v in action_probs.items() if k != choice]
            credited = all(numpy.less(others, choice_prob))
            credit_assignment.append(credited)
        # update 
        # if the expert contributed the the choice, set reward to "r"
        # if the expert did not contribute, set the reward to 1-r.
        updates = [r*c+(1-r)*(1-c) for c in credit_assignment]
        # update alpha and beta based on updates
        for i, update in enumerate(updates):
            update_param = ['b','a'][int(update)]
            self.confidences[context][i][update_param] += 1
    
    def get_expert_confidences(self, trial):
        # get attention weights (softmax of confidences)
        c = trial[self.context]
        e_confidences = [numpy.e**(beta(**p).mean()/self.zeta) for p in self.confidences[c]]
        e_confidences = [i/sum(e_confidences) for i in e_confidences]
        return e_confidences
    
class Hierarchical_SuperExpert(Expert):
    """ Instantiates the superordinate hierarchical expert
    
    This class instantiates three hierarchical experts each with a different
    context - either color, orientation, or shape
    """
    def __init__(self, data, kappa, zeta):
        self.actions = sorted(numpy.unique([a for a in data.key_press if a > 0]))
        # define hierarchical experts
        self.color_expert = Hierarchical_Expert(['stim','orientation'], 'border', data, kappa, zeta)
        self.orientation_expert = Hierarchical_Expert(['stim','border'], 'orientation', data, kappa, zeta)
        self.shape_expert = Hierarchical_Expert(['border','orientation'], 'stim', data, kappa, zeta)  
        self.experts = [self.color_expert, 
                        self.orientation_expert, 
                        self.shape_expert]
        # create disctionary of params for beta distributions representing
        # the confidence in each expert
        self.confidences = [{'a': 1, 'b': 1} for _ in range(len(self.experts))]
        

    def get_expert_confidences(self, trial):
        # get attention weights - unclear if softmax
        e_confidences = [beta(**p).mean() for p in self.confidences]
        #e_confidences = [numpy.e**(beta(**p).mean()/self.zeta) for p in self.confidences
        e_confidences = [i/sum(e_confidences) for i in e_confidences]
        return e_confidences
            
class MoE_Model(Expert):
    def __init__(self, data, kappa, zeta, xi, alphaC, alphaO, alphaS,
                 beta2, beta3, beta_hierarchy):
        """
        
        Args:
            data: dataframe for hierarchical rule task
            kappa: softmax parameter for action probabilities, passed to
                    subordinate experts
            zeta: softmax parameter for arbitration between subordinate experts
                    of hierarchical and flat experts
            xi: softmax parameter for arbitration between 
                hierarchical and flat experts
        """
        self.actions = sorted(numpy.unique([a for a in data.key_press if a > 0]))
        self.xi = xi
        # set up experts
        self.hierarchical_expert = Hierarchical_SuperExpert(data, kappa, zeta)
        self.flat_expert = Flat_Expert(data, kappa, zeta, alphaC, alphaO, alphaS,
                 beta2, beta3)
        self.experts = [self.hierarchical_expert, self.flat_expert]
        # create disctionary of params for beta distributions representing
        # the confidence in each expert
        self.confidences = [{'a': 1, 'b': beta_hierarchy}, {'a': 1, 'b': 1}]
        
            
    def get_expert_confidences(self, trial):
        # get attention weights - unclear if softmax
        e_confidences = [numpy.e**(beta(**p).mean()/self.xi) for p in self.confidences]
        e_confidences = [i/sum(e_confidences) for i in e_confidences]
        return e_confidences
    
    def get_all_confidences(self, trial):
        confidences = {}
        confidences['hierarchy'] = self.get_expert_confidences(trial)[0]
        
        hierarchical_expert = self.experts[0]
        color, orientation, shape = hierarchical_expert.get_expert_confidences(trial)
        confidences['hier_color'] = color
        confidences['hier_orientation'] = orientation
        confidences['hier_shape'] = shape
        
        flat_expert = self.experts[1]
        flat_confidences = flat_expert.get_expert_confidences(trial)
        confidences['flat_orientation'] = flat_confidences[0]
        confidences['flat_color'] = flat_confidences[1]
        confidences['flat_shape'] = flat_confidences[2]
        confidences['flat_OC'] = flat_confidences[3]
        confidences['flat_OS'] = flat_confidences[4]
        confidences['flat_CS'] = flat_confidences[5]
        confidences['flat_OSC'] = flat_confidences[6]
        
        return confidences
    
    
from lmfit import Minimizer, Parameters
# Functions to define Shift Task model (Wilson & Niv, 2012)
class fRL_Model():
    def __init__(self, data, decay_weights=False,
                 verbose=False):
        self.data = data
        # scrub data
        self.data = data.query('rt != -1')
        # get features
        stim_features = json.loads(data.stims[0])
        colors =  [i['color'] for i in stim_features]
        patterns =  [i['pattern'] for i in stim_features]
        shapes =  [i['shape'] for i in stim_features]
        all_features = colors+patterns+shapes
        # set up class vars
        self.weights = {f: 0 for f in all_features}
        self.decay = 0
        self.beta=1
        self.eps=0
        self.lr = .01
        self.decay_weights=decay_weights
        self.verbose=verbose
        
    def get_stim_value(self, stim):
        return numpy.sum([self.weights[v] for v in stim.values()]) 
    
    def get_choice_prob(self, trial):
        stims = json.loads(trial.stims)
        stim_values = [self.get_stim_value(stim) for stim in stims]
        # compute softmax decision probs
        f = lambda x: numpy.e**(self.beta*x)
        softmax_values = [f(v) for v in stim_values]
        normalized = [v/numpy.sum(softmax_values) for v in softmax_values]
        # get prob of choice
        choice_prob = normalized[int(trial.choice_position)]
        # incorporate eps
        choice_prob = (1-self.eps)*choice_prob + (self.eps)*(1/3)
        return choice_prob
    
    def get_params(self):
        return {'beta': self.beta,
                'decay': self.decay,
                'lr': self.lr,
                'eps': self.eps}
        
    def update(self, trial):
        choice = eval(trial.choice_stim)
        reward = trial.feedback
        value = self.get_stim_value(choice)
        delta = self.lr*(reward-value)
        for key in choice.values():
            self.weights[key] += delta
        # decay non choice features
        for key in set(self.weights.keys()) - set(choice.values()):
            self.weights[key] *= (1-self.decay)
            
    def run_data(self):
        probs = []
        attention_weights = []
        for i, trial in self.data.iterrows():
            probs.append(self.get_choice_prob(trial))
            self.update(trial)
            attention_weights.append(self.weights.copy())
        return probs, attention_weights
    
    def optimize(self):
        def loss(pars):
            #unpack params
            parvals = pars.valuesdict()
            self.beta = parvals['beta']
            self.decay = parvals['decay']
            self.lr = parvals['lr']
            self.eps = parvals['eps']
            probs, attention_weights = self.run_data()
            neg_log_likelihood = -numpy.sum(numpy.log(probs))
            return neg_log_likelihood
        
        def track_loss(params, iter, resid):
            if iter%100==0:
                print(iter, resid)
            
        params = Parameters()
        if self.decay_weights:
            params.add('decay', value=0, min=0, max=1)
        else:
            params.add('decay', value=0, vary=False)
        params.add('beta', value=1, min=.01, max=100)
        params.add('eps', value=0, min=0, max=1)
        params.add('lr', value=.1, min=.000001, max=1)
        
        if self.verbose==False:
            fitter = Minimizer(loss, params)
        else:
            fitter = Minimizer(loss, params, iter_cb=track_loss)
        fitter.scalar_minimize(method='Nelder-Mead', options={'xatol': 1e-3,
                                                              'maxiter': 200})

        
        
        
        