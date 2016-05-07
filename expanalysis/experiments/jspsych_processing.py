"""
analysis/experiments/jspsych_processing.py: part of expfactory package
functions for automatically cleaning and manipulating jspsych experiments
"""
import re
import pandas
import numpy
import hddm

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
    def multi_worker_wrap(group_df):
        group_dvs = {}
        for worker in pandas.unique(group_df['worker_id']):
            df = group_df.query('worker_id == "%s"' %worker)
            group_dvs[worker], description = func(df)
        return group_dvs, description
    return multi_worker_wrap

def calc_common_stats(df):
    dvs = {}
    dvs['avg_rt'] = df['rt'].median()
    if 'correct' in df.columns:
        dvs['accuracy'] = df['correct'].mean()
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
   
def ANT_post(df):
    correct = df['correct_response'] == df['key_press']
    if 'correct' in df.columns:
        df.loc[:,'correct'] = correct
    else:
        df.insert(1,'correct',correct)
    df.loc[:,'correct'] = df['correct'].astype(object)
    return df
    
def ART_post(df):
    round_over_list = df.query('trial_id == "round_over"').index
    if 'caught_blue' not in df.columns:
        df.insert(2,'caught_blue', numpy.nan)
        df['caught_blue'] = df['caught_blue'].astype(object)
    for i in round_over_list:
        if pandas.isnull(df.loc[i]['caught_blue']):
            index = df.index.get_loc(i)
            caught_blue = df.iloc[index-1]['mouse_click'] == 'goFish'
            df.set_value(i,'caught_blue', caught_blue)
    return df
  
def choice_reaction_time_post(df):
    correct = df['correct_response'] == df['key_press']
    if 'correct' in df.columns:
        df.loc[:,'correct'] = correct
    else:
        df.insert(2,'correct',correct)
    df.loc[:,'correct'] = df['correct'].astype(object)
    return df
        
def directed_forgetting_post(df):
    if 'probeType' in df.columns:
        df['probe_type'] = df['probe_type'].fillna(df['probeType'])
        df.drop('probeType',axis = 1, inplace = True)
    if 'cue' not in df.columns:
        df.insert(4,'cue',numpy.nan)
    if 'stim' in df.columns:
        df['cue'] = df['cue'].fillna(df['stim'])
        df.drop('stim',axis = 1, inplace = True)
    df['stim_bottom'] = df['stim_bottom'].fillna(df['stim_bottom'].shift(3))
    df['stim_top'] = df['stim_top'].fillna(df['stim_bottom'].shift(3))
    df['cue'] = df['cue'].fillna(df['cue'].shift(2))
    return df

def DPX_post(df):
    if not 'correct' in df.columns:
        df.insert(3,'correct',numpy.nan)
        df.loc[:,'correct'] = df['correct'].astype(object)
    subset = df.query('trial_id == "probe" and correct != correct and rt != -1')
    for i,row in subset.iterrows():
        correct = ((row['condition'] == 'AX' and row['key_press'] == 37) or \
            (row['condition'] != 'AX' and row['key_press'] == 40))
        df.set_value(i, 'correct', correct)
    return df
    
def hierarchical_post(df):
    correct =  [trial['correct_response'] == trial['key_press'] if not pandas.isnull(trial['correct_response']) else numpy.nan for i, trial in df.iterrows()]
    if 'correct' in df.columns:
        df.loc[:,'correct'] = correct
    else:
        df.insert(3,'correct',correct)
    df.loc[:,'correct'] = df['correct'].astype(object) 
    return df
     
def keep_track_post(df):
    for i,row in df.iterrows():
        if not pandas.isnull(row['responses']):
            response = row['responses']
            response = response[response.find('":"')+3:-2]
            response = re.split(r'[,; ]+', response)
            response = [x.lower().strip() for x in response]
            df.set_value(i,'responses', response)
    if 'correct_responses' in df.columns:
        df.insert(12,'possible_score',numpy.nan)
        df.insert(15,'score',numpy.nan)
        subset = df[[isinstance(i,dict) for i in df['correct_responses']]]
        for i,row in subset.iterrows():
            targets = row['correct_responses'].values()
            score = sum([word in targets for word in response])
            df.set_value(i, 'score', score)
            df.set_value(i, 'possible_score', len(targets))
    return df

def shift_post(df):
    if not 'shift_type' in df.columns:
        df.insert(18,'shift_type',numpy.nan)
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
    return df
                
    
def span_post(df):
    df = df[df['rt'].map(lambda x: isinstance(x,int))]
    return df

def stop_signal_post(df):
    insert_index = df.columns.get_loc('time_elapsed')
    df.insert(insert_index, 'stopped', df['key_press'] == -1)
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
    missed_percent = (df['rt']==-1).mean()
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
    missed_percent = (df['rt']==-1).mean()
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
    missed_percent = (df['rt']==-1).mean()
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
def calc_hierarchical_rule_DV(df):
    """ Calculate dv for hierarchical learning task. 
    DVs
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1')
    dvs = calc_common_stats(df)
    dvs['score'] = df['correct'].sum()
    dvs['missed_percent'] = missed_percent
    description = 'average reaction time'  
    return dvs, description

@multi_worker_decorate
def calc_simple_RT_DV(df):
    """ Calculate dv for simple reaction time. Average Reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df['rt']==-1).mean()
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
    missed_percent = (df['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1')
    dvs = calc_common_stats(df)
    contrast_df = df.groupby('condition')[['rt','correct']].agg(['mean','median'])
    contrast = contrast_df.loc['incongruent']-contrast_df.loc['congruent']
    dvs['stroop_rt'] = contrast['rt','median']
    dvs['stroop_correct'] = contrast['correct', 'mean']
    dvs['missed_percent'] = missed_percent
    description = 'stroop effect: incongruent-congruent'
    return dvs, description
    
@multi_worker_decorate
def calc_generic_dv(df):
    """ Calculate dv for choice reaction time: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1')
    dvs = calc_common_stats(df)
    dvs['missed_percent'] = missed_percent
    description = 'standard'  
    return dvs, description
    
    