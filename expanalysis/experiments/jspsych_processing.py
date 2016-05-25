"""
analysis/experiments/jspsych_processing.py: part of expfactory package
functions for automatically cleaning and manipulating jspsych experiments
"""
import re
import pandas
import numpy
import hddm
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import zscore

"""
Generic Functions
"""

def EZ_diffusion(df):
    assert 'correct' in df.columns, 'Could not calculate EZ DDM'
    pc = df['correct'].mean()
    vrt = numpy.var(df.query('correct == True')['rt'])
    mrt = numpy.mean(df.query('correct == True')['rt'])
    drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
    return {'EZ_drift': drift, 'EZ_thresh': thresh, 'EZ_non_decision': non_dec}
 
def multi_worker_decorate(func):
    """Decorator to ensure that dv functions have only one worker
    """
    def multi_worker_wrap(group_df, use_check = True):
        group_dvs = {}
        if len(group_df) == 0:
            return group_dvs, ''
        if 'passed_check' in group_df.columns and use_check:
            group_df = group_df.query('passed_check == True')
        for worker in pandas.unique(group_df['worker_id']):
            df = group_df.query('worker_id == "%s"' %worker)
            group_dvs[worker], description = func(df)
        return group_dvs, description
    return multi_worker_wrap

def calc_common_stats(df):
    dvs = {}
    dvs['avg_rt'] = df['rt'].median()
    dvs['std_rt'] = df['rt'].std()
    if 'correct' in df.columns:
        dvs['accuracy'] = df['correct'].mean()
        if 'possible_responses' in df.columns:
            df = df.query('possible_responses == possible_responses')
            possible_responses = numpy.unique(df['possible_responses'].map(sorted))
            if (len(possible_responses) == 1 and \
                len(possible_responses[0]) == 2 ):
                try:
                    diffusion_params = EZ_diffusion(df)
                    dvs.update(diffusion_params)
                except ValueError:
                    pass
    return dvs

    
"""
Post Processing functions
"""

def adaptive_nback_post(df):
    if df.query('trial_id == "stim"').iloc[0]['possible_responses'] == [37,40]:
        response_dict = {True: 37, False: 40}
    else:
        response_dict = {True: 37, False: -1}
    if 'correct_response' not in df.columns:
        df.loc[:,'correct_response'] = numpy.nan
    nan_index = df.query('target == target and correct_response != correct_response').index
    hits = df.loc[nan_index, 'stim'].str.lower() == df.loc[nan_index,'target'].str.lower()
    df.loc[nan_index,'correct_response'] = hits.map(lambda x: response_dict[x])
    df.loc[nan_index,'correct'] = df.loc[nan_index,'correct_response'] == df.loc[nan_index,'key_press']
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    if 'feedback_duration' in df.columns:    
        df.drop('feedback_duration', axis = 1, inplace = True)
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    return df
    
    
def ANT_post(df):
    correct = df['correct_response'] == df['key_press']
    if 'correct' in df.columns:
        df.loc[:,'correct'] = correct
    else:
        df.loc[:,'correct'] = correct
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    df=df.dropna(subset = ['possible_responses'])
    return df
    
def ART_post(df):
    round_over_list = df.query('trial_id == "round_over"').index
    if 'caught_blue' not in df.columns:
        df.loc[:,'caught_blue'] = numpy.nan
        df['caught_blue'] = df['caught_blue'].astype(object)
    for i in round_over_list:
        if pandas.isnull(df.loc[i]['caught_blue']):
            index = df.index.get_loc(i)
            caught_blue = df.iloc[index-1]['mouse_click'] == 'goFish'
            df.set_value(i,'caught_blue', caught_blue)
    return df
  
def choice_reaction_time_post(df):
    for worker in numpy.unique(df['worker_id']):
        subset = df.query('worker_id == "%s" and exp_stage == "practice"' %worker)
        response_dict = subset.groupby('stim_id')['correct_response'].mean().to_dict()
        test_index = df.query('exp_stage == "test"').index      
        df.loc[test_index, 'correct_response'] = df.loc[test_index,'stim_id'].map(lambda x: response_dict[x] if x == x else numpy.nan)
    correct = df['correct_response'] == df['key_press']
    if 'correct' in df.columns:
        df.loc[:,'correct'] = correct
    else:
        df.loc[:,'correct'] = correct
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    return df
        
def directed_forgetting_post(df):
    if 'probeType' in df.columns:
        df['probe_type'] = df['probe_type'].fillna(df['probeType'])
        df.drop('probeType',axis = 1, inplace = True)
    if 'cue' not in df.columns:
        df.loc[:,'cue'] = numpy.nan
    if 'stim' in df.columns:
        df['cue'] = df['cue'].fillna(df['stim'])
        df.drop('stim',axis = 1, inplace = True)
    df['stim_bottom'] = df['stim_bottom'].fillna(df['stim_bottom'].shift(3))
    df['stim_top'] = df['stim_top'].fillna(df['stim_bottom'].shift(3))
    df['cue'] = df['cue'].fillna(df['cue'].shift(2))
    return df

def DPX_post(df):
    if not 'correct' in df.columns:
        df.loc[:,'correct'] = numpy.nan
        df.loc[:,'correct'] = df['correct'].astype(object)
    subset = df.query('trial_id == "probe" and correct != correct and rt != -1')
    for i,row in subset.iterrows():
        correct = ((row['condition'] == 'AX' and row['key_press'] == 37) or \
            (row['condition'] != 'AX' and row['key_press'] == 40))
        df.set_value(i, 'correct', correct)
    return df
    
def hierarchical_post(df):
    correct =  [float(trial['correct_response'] == trial['key_press']) if not pandas.isnull(trial['correct_response']) else numpy.nan for i, trial in df.iterrows()]
    if 'correct' in df.columns:
        df.loc[:,'correct'] = correct
    else:
        df.loc[:,'correct'] = correct
    return df

def IST_post(df):
    if 'trial_num' not in df.columns:
        df = df.drop('box_id', axis = 1)
        tmp = df['mouse_click'].apply(lambda x: 'choice' if x in ['26','27'] else numpy.nan)
        df.loc[:,'trial_id'] = tmp.fillna(df['trial_id'])
        subset=df[df['trial_id'].apply(lambda x: x in ['choice','stim'])][1:]['trial_id']
        trial_num = 0
        trial_nums = []
        for row in subset:
            trial_nums.append(trial_num)
            if row == "choice":
                trial_num+=1
            if trial_num == 10:
                trial_num = 0
        df.loc[subset.index,'trial_num'] = trial_nums
        
def keep_track_post(df):
    for i,row in df.iterrows():
        if not pandas.isnull(row['responses']) and row['trial_id'] == 'response':
            response = row['responses']
            response = response[response.find('":"')+3:-2]
            response = re.split(r'[,; ]+', response)
            response = [x.lower().strip() for x in response]
            df.set_value(i,'responses', response)
    if 'correct_responses' in df.columns:
        df.loc[:,'possible_score'] = numpy.nan
        df.loc[:,'score'] = numpy.nan
        subset = df[[isinstance(i,dict) for i in df['correct_responses']]]
        for i,row in subset.iterrows():
            targets = row['correct_responses'].values()
            response = row['responses']
            score = sum([word in targets for word in response])
            df.set_value(i, 'score', score)
            df.set_value(i, 'possible_score', len(targets))
    return df

def probabilistic_selection_post(df):
    if (numpy.sum(pandas.isnull(df.query('exp_stage == "test"')['correct']))>0):
        def get_correct_response(stims):
            if stims[0] > stims[1]:
                return 37
            else:
                return 39
        df.replace(to_replace = 'practice', value = 'training', inplace = True)
        subset = df.query('exp_stage == "test"')
        correct_responses = subset['condition'].map(lambda x: get_correct_response(x.split('_')))
        correct = correct_responses == subset['key_press']
        df.loc[correct_responses.index,'correct_response'] = correct_responses
        df.loc[correct.index,'correct'] = correct
    if ('optimal_response' not in df.columns):
        df.loc[:,'optimal_response'] = df['condition'].map(lambda x: [37,39][numpy.diff([int(a) for a in x.split('_')])[0]>0] if x == x else numpy.nan)
    
    # add FB column
    df.loc[:,'feedback'] = df[df['exp_stage'] == "training"]['correct']
    df.loc[:,'feedback'] = df['feedback'].map(lambda x: float(x) if (x==x) else numpy.nan)
    df.loc[:,'correct'] = df['key_press'] == df['optimal_response']
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if (x==x) else numpy.nan)
    df = df.drop('optimal_response', axis = 1)
    # add condition collapsed column
    df.loc[:,'condition_collapsed'] = df['condition'].map(lambda x: '_'.join(sorted(x.split('_'))) if x == x else numpy.nan)
    # add column indicating stim chosen
    choices = [[37,39].index(x) if x in [37,39] else numpy.nan for x in df['key_press']]
    stims = df['condition'].apply(lambda x: x.split('_') if x==x else numpy.nan)
    df.loc[:,'stim_chosen'] = [s[c] if c==c else numpy.nan for s,c in zip(stims,choices)]
    #learning check - ensure during test that worker performed above chance on easiest training pair
    passed_workers = df.query('exp_stage == "test" and condition_collapsed == "20_80"').groupby('worker_id')['correct'].mean()>.5
    if numpy.sum(passed_workers) < len(passed_workers):
        print "Probabilistic Selection: %s failed the manipulation check" % list(passed_workers[passed_workers == False].index)    
    passed_workers = list(passed_workers[passed_workers].index)
    df.loc[:,"passed_check"] = df['worker_id'].map(lambda x: x in passed_workers)
    return df
    
   
def shift_post(df):
    if not 'shift_type' in df.columns:
        df.loc[:,'shift_type'] = numpy.nan
        df['shift_type'] = df['shift_type'].astype(object)
        last_feature = ''
        last_dim = ''
        for i,row in df.iterrows():
            if row['trial_id'] == 'stim':
                if last_feature == '':
                    shift_type = 'stay'
                elif row['rewarded_feature'] == last_feature:
                    shift_type = 'stay'
                elif row['rewarded_dim'] == last_dim:
                    shift_type = 'intra'
                else:
                    shift_type = 'extra'
                last_feature = row['rewarded_feature']
                last_dim = row['rewarded_dim']
                df.set_value(i,'shift_type', shift_type)
            elif row['trial_id'] == 'feedback':
                df.set_value(i,'shift_type', shift_type)
    if 'FB' in df.columns:
        df.loc[:,'feedback'] = df['FB']
        df = df.drop('FB', axis = 1)
    return df
    
def span_post(df):
    df = df[df['rt'].map(lambda x: isinstance(x,int))]
    if 'correct' not in df.columns:
        df.loc[:,'correct'] = numpy.nan
    if 'feedback' in df.columns:
        df['correct'].fillna(df['feedback'])
        df = df.drop('feedback', axis = 1)
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    return df
    
def stop_signal_post(df):
    df.loc[:,'stopped'] = df['key_press'] == -1
    df.loc[:,'correct'] = df['key_press'] == df['correct_response']
    if 'SSD' in df.columns:
        df.drop('SSD',inplace = True, axis = 1)
    if df['experiment_exp_id'].iloc[0] == "motor_selective_stop_signal" and 'condition' in df.columns:
        df.drop('condition', inplace = True, axis = 1)
    return df  

def threebytwo_post(df):
    for worker in numpy.unique(df['worker_id']):
        correct_responses = {}
        subset = df.query('trial_id == "stim" and worker_id == "%s"' % worker)
        if (numpy.sum(pandas.isnull(subset.query('exp_stage == "test"')['correct']))>0):
            correct_responses['color'] = subset.query('task == "color"').groupby('stim_color')['correct_response'].mean().to_dict()
            correct_responses['parity'] = subset.query('task == "parity"').groupby(subset['stim_number']%2 == 1)['correct_response'].mean().to_dict()
            correct_responses['magnitude'] = subset.query('task == "magnitude"').groupby(subset['stim_number']>5)['correct_response'].mean().to_dict()
            color_responses = (subset.query('task == "color"')['stim_color']).map(lambda x: correct_responses['color'][x])
            parity_responses = (subset.query('task == "parity"')['stim_number']%2==1).map(lambda x: correct_responses['parity'][x])
            magnitude_responses = (subset.query('task == "magnitude"')['stim_number']>5).map(lambda x: correct_responses['magnitude'][x])
            df.loc[color_responses.index,'correct_response'] = color_responses
            df.loc[parity_responses.index,'correct_response'] = parity_responses
            df.loc[magnitude_responses.index,'correct_response'] = magnitude_responses
            df.loc[subset.index,'correct'] =df.loc[subset.index,'key_press'] == df.loc[subset.index,'correct_response']
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    return df
        
def TOL_post(df):
    labels = ['practice'] + range(12)
    if 'problem_id' not in df.columns:
        df_index = df.query('(target == target and rt != -1) or trial_id == "feedback"').index
        problem_time = 0
        move_stage = 'to_hand'
        problem_id = 0
        for loc in df_index:
            if df.loc[loc,'trial_id'] != 'feedback':
                df.loc[loc,'trial_id'] = move_stage
                df.loc[loc,'problem_id'] = labels[problem_id%13]
                if move_stage == 'to_board':
                    move_stage = 'to_hand'
                else:
                    move_stage = 'to_board'
                problem_time += df.loc[loc,'rt']
            else:
                df.loc[loc,'problem_time'] = problem_time
                problem_time = 0
                problem_id += 1
    # Change current position type to list if necessary
    index = [not isinstance(x,list) and x==x for x in df['current_position']]
    df.loc[index,'current_position'] = df.loc[index,'current_position'].map(lambda x: [x['0'], x['1'], x['2']])
    if 'correct' not in df:
        df.loc[:,'correct'] = (df['current_position'] == df['target'])
    else:
        subset = df.query('trial_id != "feedback"').index
        df.loc[subset,'correct'] = (df.loc[subset,'current_position'] == df.loc[subset,'target'])
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    return df
    

def two_stage_decision_post(df):
    try:
        group_df = pandas.DataFrame()
        trials = df.groupby('exp_stage')['trial_num'].max()
        for worker_i, worker in enumerate(numpy.unique(df['worker_id'])):
            rows = []
            worker_df = df.query('worker_id == "%s"' % worker)
            for stage in ['practice', 'test']:
                stage_df = worker_df[worker_df['exp_stage'] == stage]
                for i in range(int(trials[stage]+1)):
                    trial = stage_df.loc[df['trial_num'] == i]
                    #set row to first stage
                    row = trial.iloc[0].to_dict()  
                    ss,fb = {}, {}
                    row['trial_id'] = 'incomplete_trial'
                    if len(trial) >= 2:
                        ss = trial.iloc[1]
                        row['time_elapsed'] = ss['time_elapsed']
                    if len(trial) == 3:
                        fb = trial.iloc[2]
                        row['time_elapsed'] = fb['time_elapsed']
                        row['trial_id'] = 'complete_trial'
                    row['rt_first'] = row.pop('rt')
                    row['rt_second'] = ss.get('rt',-1)
                    row['stage_second'] = ss.get('stage',-1)
                    row['stim_selected_first'] = row.pop('stim_selected')
                    row['stim_selected_second'] = ss.get('stim_selected',-1)
                    row['stage_transition'] = ss.get('stage_transition',numpy.nan)
                    row['feedback'] = fb.get('feedback',numpy.nan)
                    row['FB_probs'] = fb.get('FB_probs',numpy.nan)
                    rows.append(row)
            worker_df = pandas.DataFrame(rows)
            trial_index = ["%s_%s_%s" % ('two_stage_decision',worker_i,x) for x in range(len(worker_df))]
            worker_df.index = trial_index
            #manipulation check
            win_stay = 0.0
            for stage in numpy.unique(worker_df.query('exp_stage == "test"')['stage_second']):
                stage_df=worker_df.query('exp_stage == "test"')[worker_df.query('exp_stage == "test"')['stage_second'].map(lambda x: x == stage)][['stage','feedback','stim_selected_second']]
                stage_df['next_choice'] = stage_df['stim_selected_second'].shift(-1)
                stage_df['stay'] = stage_df['stim_selected_second'] == stage_df['next_choice']
                win_stay+= stage_df[stage_df['feedback']==1]['stay'].sum()
            win_stay_proportion = win_stay/worker_df.query('exp_stage == "test"')['feedback'].sum()
            if win_stay_proportion > .5:
                worker_df.loc[:,'passed_check'] = True
            else:
                worker_df.loc[:,'passed_check'] = False
                print 'Two Stage Decision: Worker %s failed manipulation check. Win stay = %s' % (worker, win_stay_proportion)
            group_df = pandas.concat([group_df,worker_df])
        group_df.loc[:,'switch'] = group_df['stim_selected_first'].diff()!=0
        group_df.loc[:,'stage_transition_last'] = group_df['stage_transition'].shift(1)
        group_df.loc[:,'feedback_last'] = group_df['feedback'].shift(1)
        df = group_df
    except:
        print('Could not process two_stage_decision dataframe with workers: %s' % numpy.unique(df['worker_id']))
    return df
    
 
"""
DV functions
"""

@multi_worker_decorate
def calc_adaptive_n_back_DV(df):
    """ Calculate dv for adaptive_n_back task. Maximum load
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice"')
    dvs = {'max_load': df['load'].max()}
    description = 'max load'
    return dvs, description
 
@multi_worker_decorate
def calc_ANT_DV(df):
    """ Calculate dv for attention network task: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1')
    dvs = calc_common_stats(df)
    dvs['missed_percent'] = missed_percent
    description = 'standard'  
    return dvs, description
    
@multi_worker_decorate
def calc_ART_sunny_DV(df):
    """ Calculate dv for choice reaction time: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and key_press != -1')
    dvs = calc_common_stats(df)
    dvs['missed_percent'] = missed_percent
    scores = df.groupby('release').max()['tournament_bank']
    clicks = df.groupby('release').mean()['trial_num']
    dvs['Keep_score'] = scores['Keep']    
    dvs['Release_score'] = scores['Release']  
    dvs['Keep_clicks'] = clicks['Keep']    
    dvs['Release_clicks'] = clicks['Release']  
    description = 'DVs are the total tournament score for each condition and the average number of clicks per condition'  
    return dvs, description

@multi_worker_decorate
def calc_choice_reaction_time_DV(df):
    """ Calculate dv for choice reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1')
    dvs = calc_common_stats(df)
    dvs['missed_percent'] = missed_percent
    description = 'standard'  
    return dvs, description

@multi_worker_decorate
def calc_digit_span_DV(df):
    """ Calculate dv for digit span: forward and reverse span
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice" and rt != -1')
    dvs = calc_common_stats(df)
    span = df.groupby(['condition'])['num_digits'].mean()
    dvs['forward_span'] = span['forward']
    dvs['reverse_span'] = span['reverse']
    description = 'standard'  
    return dvs, description


@multi_worker_decorate
def calc_DPX_DV(df):
    """ Calculate dv for dot pattern expectancy task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1')
    dvs = calc_common_stats(df)
    df.loc[:,'z_rt'] = zscore(df['rt'])
    contrast_df = df.groupby('condition')['rt'].median()
    dvs['AY_diff'] = contrast_df['AY'] - df['rt'].median()
    dvs['BX_diff'] = contrast_df['BX'] - df['rt'].median()
    dvs['missed_percent'] = missed_percent
    description = 'standard'  
    return dvs, description
    
@multi_worker_decorate
def calc_hierarchical_rule_DV(df):
    """ Calculate dv for hierarchical learning task. 
    DVs
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1')
    dvs = calc_common_stats(df)
    dvs['score'] = df['correct'].sum()
    dvs['missed_percent'] = missed_percent
    description = 'average reaction time'  
    return dvs, description

@multi_worker_decorate
def calc_keep_track_DV(df):
    """ Calculate dv for choice reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice" and rt != -1')
    score = df['score'].sum()/df['possible_score'].sum()
    dvs = {}
    dvs['score'] = score
    description = 'percentage of items remembered correctly'  
    return dvs, description

@multi_worker_decorate
def calc_probabilistic_selection_DV(df):
    """ Calculate dv for probabilistic selection task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    def get_value_diff(lst, values):
        return abs(values[lst[0]] - values[lst[1]])
    def get_value_sum(lst,values):
        return values[lst[0]] + values[lst[1]]
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1')
    train = df.query('exp_stage == "training"')
    values = train.groupby('stim_chosen')['feedback'].mean()
    df.loc[:,'value_diff'] = df['condition_collapsed'].apply(lambda x: get_value_diff(x.split('_'), values) if x==x else numpy.nan)
    df.loc[:,'value_sum'] =  df['condition_collapsed'].apply(lambda x: get_value_sum(x.split('_'), values) if x==x else numpy.nan)   
    test = df.query('exp_stage == "test"')
    rs = smf.glm(formula = 'correct ~ value_diff*value_sum', data = test, family = sm.families.Binomial()).fit()
    dvs = calc_common_stats(df)
    dvs['value_sensitivity'] = rs.params['value_diff']
    dvs['positive_learning_bias'] = rs.params['value_diff:value_sum']
    dvs['overall_test_acc'] = test['correct'].mean()
    dvs['missed_percent'] = missed_percent
    description = 'standard'  
    return dvs, description

@multi_worker_decorate
def calc_ravens_DV(df):
    """ Calculate dv for ravens task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('stim_response == stim_response')
    dvs = calc_common_stats(df)
    dvs['score'] = df['score_response'].sum()
    description = 'Score is the number of correct responses out of 18'
    return dvs,description    
    
@multi_worker_decorate
def calc_simple_RT_DV(df):
    """ Calculate dv for simple reaction time. Average Reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1')
    dvs = calc_common_stats(df)
    dvs['avg_rt'] = df['rt'].median()
    dvs['missed_percent'] = missed_percent
    description = 'average reaction time'  
    return dvs, description

@multi_worker_decorate
def calc_spatial_span_DV(df):
    """ Calculate dv for spatial span: forward and reverse mean span
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice" and rt != -1')
    dvs = calc_common_stats(df)
    span = df.groupby(['condition'])['num_spaces'].mean()
    dvs['forward_span'] = span['forward']
    dvs['reverse_span'] = span['reverse']
    description = 'standard'  
    return dvs, description
    
@multi_worker_decorate
def calc_stroop_DV(df):
    """ Calculate dv for stroop task. Incongruent-Congruent, median RT and Percent Correct
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1')
    dvs = calc_common_stats(df)
    df.loc[:,'z_rt'] = zscore(df['rt'])
    contrast_df = df.groupby('condition')[['rt','correct']].agg(['mean','median'])
    contrast = contrast_df.loc['incongruent']-contrast_df.loc['congruent']
    dvs['stroop_rt'] = contrast['rt','median']
    dvs['stroop_correct'] = contrast['correct', 'mean']
    dvs['missed_percent'] = missed_percent
    description = 'stroop effect: incongruent-congruent'
    return dvs, description

@multi_worker_decorate
def calc_stop_signal_DV(df):
    """ Calculate dv for stop signal task. Common states like rt, correct and
    DDM parameters are calculated on go trials only
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice" and SS_trial_type == "go"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and ((SS_trial_type == "stop") or (SS_trial_type == "go" and rt != -1))')
    dvs = calc_common_stats(df.query('SS_trial_type == "go"'))
    dvs = {'go_' + key: dvs[key] for key in dvs.keys()}
    dvs['SSRT'] = df.query('SS_trial_type == "go"')['rt'].median()-df['SS_delay'].median()
    dvs['stop_success'] = df.query('SS_trial_type == "stop"')['stopped'].mean()
    dvs['stop_avg_rt'] = df.query('SS_trial_type == "stop" and rt > 0')['rt'].median()
    dvs['missed_percent'] = missed_percent
    description = """ SSRT calculated as the difference between median go RT
    and median SSD. Missed percent calculated on go trials only.
    """
    return dvs, description

@multi_worker_decorate
def calc_threebytwo_DV(df):
    """ Calculate dv for 3 by 2 task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1')
    df.loc[:,'z_rt'] = zscore(df['rt'])
    dvs = calc_common_stats(df)
    dvs['cue_switch_cost'] = df.query('task_switch == "stay"').groupby('cue_switch')['rt'].median().diff()['switch']
    dvs['task_switch_cost'] = df.groupby(df['task_switch'].map(lambda x: 'switch' in x))['rt'].median().diff()[True]
    dvs['task_inhibition_of_return'] =  df[['switch' in x for x in df['task_switch']]].groupby('task_switch')['rt'].median().diff()['switch_old']
    dvs['missed_percent'] = missed_percent
    description = """ Task switch cost defined as rt difference between task "stay" trials
    and both task "switch_new" and "switch_old" trials. Cue Switch cost is defined only on 
    task stay trials. Inhibition of return is defined as the difference in reaction time between
    task "switch_old" and task "switch_new" trials. Positive values indicate higher RTs (cost) for
    task switches, cue switches and switch_old
    """
    return dvs, description



@multi_worker_decorate
def calc_TOL_DV(df):
    df = df.query('exp_stage == "test" and rt != -1')
    dvs = {}
    # When they got it correct, did they make the minimum number of moves?
    dvs['num_optimal_solutions'] =  numpy.sum(df.query('correct == 1')[['num_moves_made','min_moves']].diff(axis = 1)['min_moves']==0)
    # how long did it take to make the first move?    
    dvs['planning_time'] = df.query('num_moves_made == 1 and trial_id == "to_hand"')['rt'].median()
    # how long did it take on average to take an action    
    dvs['avg_move_time'] = df.query('trial_id in ["to_hand", "to_board"]')['rt'].median()
    # how many moves were made overall
    dvs['total_moves'] = numpy.sum(df.groupby('problem_id')['num_moves_made'].max())
    dvs['num_correct'] = numpy.sum(df['correct']==1)
    description = 'many dependent variables related to tower of london performance'
    return dvs, description
    
    
    
@multi_worker_decorate
def calc_two_stage_decision_DV(df):
    """ Calculate dv for choice reaction time: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['trial_id']=="incomplete_trial").mean()
    df = df.query('exp_stage != "practice" and trial_id == "complete_trial"')
    
        

    rs = smf.glm(formula = 'switch ~ feedback_last * stage_transition_last', data = df, family = sm.families.Binomial()).fit()
    rs.summary()
    dvs = {}
    dvs['avg_rt'] = numpy.mean(df[['rt_first','rt_second']].mean())
    dvs['model_free'] = rs.params['feedback_last']
    dvs['model_based'] = rs.params['feedback_last:stage_transition_last[T.infrequent]']
    dvs['missed_percent'] = missed_percent
    description = 'standard'  
    return dvs, description
    
    
@multi_worker_decorate
def calc_generic_dv(df):
    """ Calculate dv for choice reaction time: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1')
    dvs = calc_common_stats(df)
    dvs['missed_percent'] = missed_percent
    description = 'standard'  
    return dvs, description
    
    