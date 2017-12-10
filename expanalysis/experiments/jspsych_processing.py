"""
analysis/experiments/jspsych_processing.py: part of expfactory package
functions for automatically cleaning and manipulating jspsych experiments
"""
from expanalysis.experiments.ddm_utils import EZ_diffusion, get_HDDM_fun
from expanalysis.experiments.psychological_models import (
        fRL_Model, Two_Stage_Model)
from expanalysis.experiments.r_to_py_utils import glmer
import hddm
import json
from math import ceil, factorial, floor
import numpy
import pandas
import re
import requests
from scipy import optimize
from scipy.stats import binom, chi2_contingency, mstats, norm
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys

# ignore pandas error
pandas.options.mode.chained_assignment = None  # default='warn'

"""
Generic Functions
"""



def group_decorate(group_fun_getter = None, group_fun_args = {}):
    """ Group decorate is a wrapper for multi_worker_decorate to pass an optional group level
    DV function
    :group_fun_args: arguments passed to group_fun
    :group_fun: a function to apply to the entire group that returns a dictionary with DVs
    for each subject (i.e. fit_HDDM)
    """
    def multi_worker_decorate(fun):
        """Decorator to ensure that dv functions (i.e. calc_stroop_DV) have only one worker
        :func: function to apply to each worker individuals
        """
        def multi_worker_wrap(group_df, use_check = False, use_group_fun = True, **kwargs):
            exps = group_df.experiment_exp_id.unique()
            group_dvs = {}
            if len(group_df) == 0:
                return group_dvs, ''
            if len(exps) > 1:
                print('Error - More than one experiment found in dataframe. Exps found were: %s' % exps)
                return group_dvs, ''
            # remove practice trials
            if 'exp_stage' in group_df.columns:
                group_df = group_df.query('exp_stage != "practice"')
            # remove workers who haven't passed some check
            if 'passed_check' in group_df.columns and use_check:
                group_df = group_df[group_df['passed_check']]
            # apply group func if it exists
            if group_fun_getter and use_group_fun:
                group_fun_args.update(**kwargs)
                group_fun = group_fun_getter(**group_fun_args)
                group_dvs = group_fun(group_df)
            # apply function on individuals
            for worker in pandas.unique(group_df['worker_id']):
                df = group_df.query('worker_id == "%s"' %worker)
                dvs = group_dvs.get(worker, {})
                try:
                    worker_dvs, description = fun(df, dvs)
                    group_dvs[worker] = worker_dvs
                except Exception as e:
                    print('%s DV calculation failed for worker: %s' % (exps[0], worker))
                    print(e)
            return group_dvs, description
        return multi_worker_wrap
    return multi_worker_decorate

  
def get_post_error_slow(df):
    """df should only be one subject's trials where each row is a different trial. Must have at least 4 suitable trials
    to calculate post-error slowing
    """
    index = [(j-1, j+1) for j in [df.index.get_loc(i) for i in df.query('correct == False and rt != -1').index] if j not in [0,len(df)-1]]
    post_error_delta = []
    for i,j in index:
        pre_rt = df.iloc[i]['rt']
        post_rt = df.iloc[j]['rt']
        if pre_rt != -1 and post_rt != -1 and df.iloc[i]['correct'] and df.iloc[j]['correct']:
            post_error_delta.append(post_rt - pre_rt) 
    if len(post_error_delta) >= 4:
        return numpy.mean(post_error_delta)
    else:
        return numpy.nan

"""
Post Processing functions
"""

def adaptive_nback_post(df):
    df['correct_response'] = (df['target'].str.lower()==df['stim'].str.lower()).map(lambda x: 37 if x else 40)
    df.loc[:,'correct'] = df['correct'].astype(float)
    return df
    
    
def ANT_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    return df
    
def ART_post(df):
    df['caught_blue'] = df['caught_blue'].astype(float)
    num_clicks = []
    click_num = 0
    for i in df.mouse_click:
        if i == "goFish":
            click_num += 1
        num_clicks.append(click_num)
        if i not in ["goFish","Collect"]:
            click_num = 0
    df.loc[:,'clicks_before_end'] =  num_clicks
    round_over_index = [df.index.get_loc(i) for i in df.query('trial_id == "round_over"').index]
    df.ix[round_over_index,'tournament_bank'] = df.iloc[[i-1 for i in round_over_index]]['tournament_bank'].tolist()
    df.ix[round_over_index,'trip_bank'] = df.iloc[[i-1 for i in round_over_index]]['trip_bank'].tolist()
    return df

def bickel_post(df):
    df.loc[:,'later_time_days'] = numpy.where(numpy.isnan(df['later_time_days']), 180, df['later_time_days'])
    df.insert(0, 'implied_k', ((df['larger_amount']/df['smaller_amount'])- 1)/(df['later_time_days']))
    df.insert(0, 'patient1_impatient0', numpy.where(df['choice'] == 'larger_later', 1, numpy.where(df['choice'] == 'smaller_sooner', 0, numpy.nan)).tolist())
    return df

def CCT_fmri_post(df):
    df['clicked_on_loss_card'] = df['clicked_on_loss_card'].astype(float)
    df['action'] = df.key_press.replace({89:'draw_card',71:'end_round'})
    total_cards = df.loc[df.query('trial_id == "ITI"').index-1].num_click_in_round
    df.loc[df.query('trial_id == "ITI"').index-1,'total_cards'] = total_cards
    # add a click to each end round
    df.loc[df.loc[:,'action'] == "end_round", "num_click_in_round"]+=1
    # add additional variables
    df.loc[:,'cards_left'] = df.num_cards-(df.num_click_in_round-1)
    df.loc[:,'loss_probability'] = df.num_loss_cards/df.cards_left
    df.loc[:,'gain_probability'] = 1-df.loss_probability
    # compute expected value
    EV = df.gain_amount*df.gain_probability \
         + df.loss_amount*df.loss_probability
    df.loc[:,'EV'] = EV-EV.mean()
    # compute risk of each action
    risk = (df.gain_probability * (df.gain_amount-df.EV)**2 \
           + df.loss_probability * (df.loss_amount-df.EV)**2)**.5
    df.loc[:,'risk'] = risk-risk.mean() 
    
    return df
    
def CCT_hot_post(df):
    df['clicked_on_loss_card'] = df['clicked_on_loss_card'].astype(float)
    subset = df[df['mouse_click'] == "collectButton"]
    total_cards = subset.num_click_in_round-1
    df.insert(0,'total_cards', total_cards)
    df.loc[:,'clicked_on_loss_card'] = df['clicked_on_loss_card'].astype(float) 
    # Correct bug with incorrect round_type calculation
    bug_fix_date =  "2016-07-29T01:30:00.845212Z"
    bugged_subset = df.query('finishtime < "%s"' % bug_fix_date)
    loss_index = []
    for worker in bugged_subset.worker_id.unique():
        rigged_loss_rounds = []
        worker_subset = bugged_subset[bugged_subset['worker_id'] == worker]
        bug_loss_rounds = worker_subset.query('round_type == "rigged_loss"').which_round.unique()
        rigged_loss_rounds = [i-1 if i > 2 else i for i in bug_loss_rounds if i != 2]
        if len(rigged_loss_rounds) == 3:
            rigged_loss_rounds.append(28)
        loss_index += list(worker_subset.query('which_round in %s' %rigged_loss_rounds).index)
    df.loc[((df.finishtime < bug_fix_date) & (df.round_type == 'rigged_loss')),'round_type'] = 'rigged_win'
    df.loc[loss_index,'round_type'] = 'rigged_loss'
    return df
    
def choice_reaction_time_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    return df
       
def cognitive_reflection_post(df):
    correct_responses = ['3', '15', '4', '29', '20', 'c'] * int(len(df)/6)
    intuitive_responses = ['6', '20', '9', '30', '10', 'b'] * int(len(df)/6)
    df.loc[:,'correct_response'] = correct_responses
    df.loc[:,'intuitive_response'] = intuitive_responses
    df.loc[:,'correct'] = df['correct_response'] == df['response']
    df.loc[:,'correct'] = df['correct'].map(lambda x: float(x) if x==x else numpy.nan)
    df.loc[:,'responded_intuitively'] = (df['intuitive_response'] == df['response']).astype(float)
    return df

def conditional_stop_signal_post(df):
    df.insert(0,'stopped',df['key_press'] == -1)
    df.loc[:,'correct'] = (df['key_press'] == df['correct_response']).astype(float)
    
    # quality check
    stop_counts = df.query('exp_stage == "test" and condition == "stop"').groupby('worker_id').stopped.sum()
    possible_stops = df.query('exp_stage == "test" and condition == "stop"').groupby('worker_id').stopped.count()
    # reject people who stop significantly less or more than 50% of the time
    passed_check = numpy.logical_and(stop_counts <= binom.ppf(.975, n=possible_stops, p=.5), stop_counts >= binom.ppf(.025, n=possible_stops, p=.5))
    passed_check = passed_check[passed_check]
    # reject people who do not stop on ignore trials significantly less than on stop trials
    stop_counts = df.query('condition != "go" and exp_stage == "test"').groupby(['worker_id','condition']).stopped.sum().reset_index()
    stop_counts.loc[:,'goed'] = 60 - stop_counts.stopped 
    for worker in list(passed_check.index):
        stop_counts[stop_counts.worker_id == worker]
        obs = numpy.matrix(stop_counts[stop_counts.worker_id == worker][['stopped','goed']])
        p = chi2_contingency(obs)[1]
        if obs[0,0]>obs[0,1] or p>.05:
            passed_check.loc[worker] = False
    passed_check = passed_check[passed_check]
    df.loc[:, 'passed_check'] = df['worker_id'].map(lambda x: x in passed_check)
    return df
    
def dietary_decision_post(df):
    df['stim_rating'] = df['stim_rating'].apply(lambda x: json.loads(x) if x==x else numpy.nan)
    df['reference_rating'] = df['reference_rating'].apply(lambda x: json.loads(x) if x==x else numpy.nan)
    # subset list to only decision trials where the item was rated on both health and taste
    group_subset = df[df['stim_rating'].apply(lambda lst: all(isinstance(x, int) for x in lst.values()) if lst == lst else False)]
    for finishtime in group_subset['finishtime']:
        subset = group_subset[group_subset['finishtime'] == finishtime]
        reference = list({x['health']:x for x in subset['reference_rating']}.values())
        assert len(reference) == 1, "More than one reference rating found"
        reference = reference[0]
        subset.insert(0,'health_diff',subset['stim_rating'].apply(lambda x: x['health'] - reference['health']))
        subset.insert(0,'taste_diff', subset['stim_rating'].apply(lambda x: x['taste'] - reference['taste']))
        labels = []
        for i,row in subset.iterrows():
            if row['health_diff'] > 0:
                health_label = 'Healthy'
            elif row['health_diff'] < 0:
                health_label = 'Unhealthy'
            else:
                health_label = 'Neutral'
    
            if row['taste_diff'] > 0:
                taste_label = 'Liked'
            elif row['taste_diff'] < 0:
                taste_label = 'Disliked'
            else:
                taste_label = 'Neutral'
            labels.append(taste_label + '-' + health_label)
        subset.insert(0,'decision_label', labels)
        if 'decision_label' not in df.columns:
            df = df.join(subset[['health_diff','taste_diff','decision_label']])
        else:
            df.loc[subset.index, ['health_diff', 'taste_diff', 'decision_label']] = subset[['health_diff','taste_diff','decision_label']]
    df['coded_response'] = df['coded_response'].astype(float)
    return df
    
def directed_forgetting_post(df):
    df['stim_bottom'] = df['stim_bottom'].fillna(df['stim_bottom'].shift(3))
    df['stim_top'] = df['stim_top'].fillna(df['stim_top'].shift(3))
    df['cue'] = df['cue'].fillna(df['cue'].shift(2))
    df.loc[:,'correct'] = df.correct.astype(float)
    return df

def discount_titrate_post(df):
    df.insert(0, 'patient1_impatient0', numpy.where(df['key_press'] == 80, 1, numpy.where(df['key_press'] == 81, 0, numpy.nan)).tolist())
    df.insert(0, 'sooner_days', numpy.where(df['sooner_delay'] == 'today', 0, numpy.where(df['sooner_delay'] == '2 weeks', 14, numpy.nan)).tolist())
    df.insert(0, 'later_days', numpy.where(df['later_delay'] == '2 weeks', 14, numpy.where(df['later_delay'] == '4 weeks', 28, numpy.where(df['later_delay'] == '6 weeks', 42, numpy.nan))).tolist())
    df.insert(0, 'indiff_k', numpy.where(df['exp_stage'] == 'test',(df['larger_amount'].astype(float) - df['smaller_amount'].astype(float))/(df['smaller_amount'].astype(float)*df['later_days'].astype(float) - df['larger_amount'].astype(float)*df['sooner_days'].astype(float)) , numpy.nan).tolist())
    df.insert(0, 'rel_diff', numpy.where(df['exp_stage']=='test', (df['larger_amount'].astype(float)/df['smaller_amount'].astype(float)).map(lambda x: round(x,2)-1), numpy.nan).tolist())
    df.insert(0, 'now1_notnow0', numpy.where(df['sooner_delay'] == 'today', 1, 0).tolist())							
    return df
    
def DPX_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    index = df[(df['trial_id'] == 'fixation') & (df['possible_responses'] != 'none')].index
    if len(index) > 0:
        df.loc[index,'fixation'] = 'none'
    return df
    
def hierarchical_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    return df

def holt_laury_post(df):
    df.insert(0, 'safe1_risky0', numpy.where(df['response'].astype(float) == 1, 1, numpy.where(df['response'].astype(float) == 2, 0, numpy.nan)).tolist())			
    return df
    
def IST_post(df):
    # rename conditions
    df.replace({'Fixed Win': 'Fixed_Win', 'Decreasing Win': 'Decreasing_Win'}, inplace = True)
    # change correct trials
    df.loc[:,'correct'] = df['correct'].astype(float)
    subset = df[(df['trial_id'] == 'choice') & (df['exp_stage'] != 'practice')]
    # Add chosen and total boxes clicked to choice rows and score
    final_choices = subset[['worker_id','exp_stage','color_clicked','trial_num','which_click_in_round']]
    stim_subset = df[(df['trial_id'] == 'stim') & (df['exp_stage'] != 'practice')]
    # get an index of motivation based on speed vs number of boxes clicked
    motivation_subset = df.query('exp_stage != "practice" and trial_id in ["stim","choice"]')
    try:
        box_clicks = stim_subset.groupby(['worker_id','exp_stage','trial_num'])['color_clicked'].value_counts()
        time_to_decision = motivation_subset.groupby(['worker_id','exp_stage','trial_num']).time_elapsed.agg(lambda x: x.max()-x.min())
        counts = []
        latencies = []
        for i,row in final_choices.iterrows():
            try:
                index = row[['worker_id','exp_stage','trial_num']].tolist()
                chosen_count = box_clicks[index[0], index[1], index[2]].get(row['color_clicked'],0)
                counts.append(chosen_count)
                latency = (row['which_click_in_round']-1)/time_to_decision[index[0], index[1], index[2]]
                latencies.append(latency)
            except KeyError:
                counts.append(0)
                latencies.append(numpy.nan)
        df.insert(0,'chosen_boxes_clicked',pandas.Series(index = final_choices.index, data = counts))
        df.insert(0,'clicks_before_choice', pandas.Series(index = final_choices.index, data =  subset['which_click_in_round']-1))    
        df.insert(0,'points', df['reward'].shift(-1))
        df.insert(0,'box_open_latency', pandas.Series(index = final_choices.index, data =  latencies))
        # calculate probability of being correct
        def get_prob(boxes_opened,chosen_boxes_opened):
            if boxes_opened == boxes_opened:
                z = 25-int(boxes_opened)
                a = 13-int(chosen_boxes_opened)
                if a < 0:
                    return 1.0
                else:
                    return numpy.sum([factorial(z)/float(factorial(k)*factorial(z-k)) for k in range(a,z+1)])/2**z
            else:
                return numpy.nan
        probs=numpy.vectorize(get_prob)(df['clicks_before_choice'],df['chosen_boxes_clicked'])
        df.insert(0,'P_correct_at_choice', probs)
    except IndexError:
        print('Workers: %s did not open any boxes ' % df.worker_id.unique())
    return df
        
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

def kirby_post(df):
    df.insert(0, 'patient1_impatient0', numpy.where(df['key_press'] == 80, 1, numpy.where(df['key_press'] == 81, 0, numpy.nan)).tolist())		
    df.insert(0, 'reward_size', numpy.where((df['large_amount'] <36), 'small', numpy.where((df['large_amount']>49) & (df['large_amount']<61), 'medium', 
                                             numpy.where((df['large_amount']>74)&(df['large_amount']<86), 'large', numpy.nan))).tolist())		
    return df
    
def local_global_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    conflict = (df['local_shape']==df['global_shape']).apply(lambda x: 'congruent' if x else 'incongruent')
    neutral = (df['local_shape'].isin(['o']) | df['global_shape'].isin(['o']))
    df.loc[conflict.index, 'conflict_condition'] = conflict
    df.loc[neutral,'conflict_condition'] = 'neutral'
    df.switch.replace({0:'stay', 1:'switch'}, inplace = True)
    return df
    
def probabilistic_selection_post(df):
    # move credit var to end block if necessary
    post_task_credit = df[(df['trial_id'] == "post task questions") & (df['credit_var'].map(lambda x: isinstance(x,bool)))]
    numeric_index = [pandas.Index.get_loc(df.index,i)+1 for i in post_task_credit.index]
    df.ix[numeric_index,'credit_var'] = post_task_credit['credit_var'].tolist()
    df.loc[post_task_credit.index,'credit_var'] = None
    # convert bool to float
    df.loc[:,'correct'] = df['correct'].astype(float)
    df.loc[:,'feedback'] = df['feedback'].astype(float)
    # add condition collapsed column
    df.loc[:,'condition_collapsed'] = df['condition'].map(lambda x: '_'.join(sorted(x.split('_'))) if x == x else numpy.nan)
    #learning check - ensure during test that worker performed above chance on easiest training pair
    passed_workers = df.query('exp_stage == "test" and condition_collapsed == "20_80"').groupby('worker_id')['correct'].agg(lambda x: (numpy.mean(x)>.5) and (len(x) == 6)).astype('bool')
    if numpy.sum(passed_workers) < len(passed_workers):
        print("Probabilistic Selection: %s failed the manipulation check" % list(passed_workers[passed_workers == False].index))
    passed_workers = list(passed_workers[passed_workers].index) 
    df.loc[:,"passed_check"] = df['worker_id'].map(lambda x: x in passed_workers)
    return df
    
def PRP_post(df):
    # separate choice and rt for the two choices
    df.loc[:,'key_presses'] = df['key_presses'].map(lambda x: json.loads(x) if x==x else x)
    df.loc[:,'rt'] = df['rt'].map(lambda x: json.loads(x) if isinstance(x,str) else x)
    subset = df[(df['trial_id'] == "stim") & (~pandas.isnull(df['stim_durations']))]
    # separate rt
    try:
        df.insert(0, 'choice1_rt', pandas.Series(index = subset.index, data = [json.loads(x)[0] for x in subset['rt']]))
        df.insert(0, 'choice2_rt', pandas.Series(index = subset.index, data = [json.loads(x)[1] for x in subset['rt']]) - subset['ISI'])
    except TypeError:
        df.insert(0, 'choice1_rt', pandas.Series(index = subset.index, data = [x[0] for x in subset['rt']]))
        df.insert(0, 'choice2_rt', pandas.Series(index = subset.index, data = [x[1] for x in subset['rt']]) - subset['ISI'])
    df = df.drop('rt', axis = 1)
    # separate key press
    df.insert(0, 'choice1_key_press', pandas.Series(index = subset.index, data = [x[0] for x in subset['key_presses']]))
    df.insert(0, 'choice2_key_press', pandas.Series(index = subset.index, data = [x[1] for x in subset['key_presses']]))
    df = df.drop('key_presses', axis = 1)
    # calculate correct
    choice1_correct = (df['choice1_key_press'] == df['choice1_correct_response']).astype(float)
    choice2_correct = (df['choice2_key_press'] == df['choice2_correct_response']).astype(float)
    df.insert(0,'choice1_correct', pandas.Series(index = subset.index, data = choice1_correct))
    df.insert(0,'choice2_correct', pandas.Series(index = subset.index, data = choice2_correct))
    return df

def ravens_post(df):
    # label trials
    practice_questions = df.stim_question.map(lambda x: 'practice' in x if not pandas.isnull(x) else False) 
    practice_question_index = practice_questions[practice_questions].index
    test_questions = df.stim_question.map(lambda x: 'practice' not in x and 'bottom' in x if not pandas.isnull(x) else False)    
    test_question_index = test_questions[test_questions].index
    df.set_value(practice_question_index,'exp_stage','practice')
    df.set_value(practice_question_index,'trial_id','question')
    df.set_value(test_question_index,'exp_stage','test')
    df.set_value(test_question_index,'trial_id','question')
    # relabel trial nums
    df.loc[test_question_index,'trial_num'] += 2
    df.set_value(df[df.stim_question.map(lambda x: 'practice_bottom_2' in x if  not pandas.isnull(x) else False)].index, 'trial_num', 1)
    # score questions
    correct_responses = {0: 'C', 1: 'F', 2: 'B', 3: 'E', 4: 'G', 5: 'B', 6: 'C', 7: 'B',
                          8: 'E', 9: 'B', 10: 'B', 11: 'E', 12: 'A', 13: 'E',
                           14: 'A', 15: 'C', 16: 'B', 17: 'E', 18: 'F', 19: 'D'}   
    df.insert(0,'correct_response',df.trial_num.replace(correct_responses))
    df.insert(0,'correct', df.stim_response == df.correct_response)
    df.drop(['qnum', 'score_response', 'times_viewed', 'response_range'], axis = 1, inplace = True)
    return df
    
def recent_probes_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    df['stim'] = df['stim'].fillna(df['stim'].shift(2))
    df['stims_1back'] = df['stims_1back'].fillna(df['stims_1back'].shift(2))
    # correct probeTypes
    subset = df.query('trial_id == "probe"')
    pos_neg = [['neg','pos'][probe in stim] for stim,probe 
               in subset.loc[:,['stim','probe_letter']].values]
    rec_xrec = [['xrec','rec'][probe in stim] for stim,probe 
               in subset.loc[:,['stims_1back','probe_letter']].values]
    probeType = pandas.Series([rec+'_'+pos for rec,pos in zip(rec_xrec,pos_neg)],
                          index=subset.index)
    df.probeType.fillna(probeType, inplace=True)
    return df

def shape_matching_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    return df
    
def shift_post(df):
    df.loc[:,'choice_position'] = df.key_press.map(lambda x: {37: 0, 40: 1, 39: 2, -1: -1}.get(x,numpy.nan) if x==x else x)
    chosen_stims = []
    for i,x in df.iterrows():
        try:
            stims = json.loads(x.stims)
            chosen_stims.append(stims[int(x.choice_position)])
        except TypeError:
            chosen_stims.append(numpy.nan)
            
    df.loc[:,'choice_stim'] = chosen_stims
    df.loc[:,'correct'] = df['correct'].astype(float)
    return df

def simon_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    subset = df[df['trial_id']=='stim']
    condition = (subset.stim_side.map(lambda x: 37 if x=='left' else 39) == subset.correct_response).map \
                   (lambda y: 'congruent' if y else 'incongruent')
    df.loc[subset.index,'condition'] =  condition
    return df
    
def span_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    return df
    
def stop_signal_post(df):
    df.insert(0,'stopped',df['key_press'] == -1)
    df.loc[:,'correct'] = (df['key_press'] == df['correct_response']).astype(float)
    
    #reject people who stop significantly less or more than 50% of the time
    stop_counts = df.query('exp_stage == "test" and SS_trial_type == "stop"').groupby('worker_id').stopped.sum()
    possible_stops = df.query('exp_stage == "test" and SS_trial_type == "stop"').groupby('worker_id').stopped.count()
    passed_check = numpy.logical_and(stop_counts <= binom.ppf(.975, n=possible_stops, p=.5), stop_counts >= binom.ppf(.025, n=possible_stops, p=.5))
    passed_check = passed_check[passed_check]
    df.loc[:, 'passed_check'] = df['worker_id'].map(lambda x: x in passed_check)
    return df  

def stroop_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    return df
    
def threebytwo_post(df):
    df.insert(0, 'CTI', pandas.Series(data = df[df['trial_id'] == "cue"].block_duration.tolist(), \
                                        index = df[df['trial_id'] == "stim"].index))
    stay_i = df[(df['task_switch'] != 'stay')].index
    df.loc[stay_i, 'cue_switch'] = numpy.nan
    df.loc[:, 'switch_type'] = df.task_switch.apply(lambda x: 'task_' + str(x))
    # task stay trials
    stay_i = df[(df['task_switch'] == 'stay')].index
    df.loc[stay_i, 'switch_type'] = df.loc[stay_i].cue_switch \
                                    .apply(lambda x: 'cue_' + str(x))
    return df

def twobytwo_post(df):
    df.insert(0, 'CTI', pandas.Series(data = df[df['trial_id'] == "cue"].block_duration.tolist(), \
                                        index = df[df['trial_id'] == "stim"].index))
    switch_i = df[(df['task_switch'] != 'stay')].index
    df.loc[switch_i, 'cue_switch'] = numpy.nan
    df.loc[:, 'switch_type'] = df.task_switch.apply(lambda x: 'task_' + str(x))
    # task stay trials
    stay_i = df[(df['task_switch'] == 'stay')].index
    df.loc[stay_i, 'switch_type'] = df.loc[stay_i].cue_switch \
                                    .apply(lambda x: 'cue_' + str(x))
    return df
    
def TOL_post(df):
    index = df.query('trial_id == "feedback"').index
    i_index = [df.index.get_loc(i)-1 for i in index]
    df.loc[index,'num_moves_made'] = df.iloc[i_index]['num_moves_made'].tolist()
    df.loc[index,'min_moves'] = df.iloc[i_index]['min_moves'].tolist()
    df.loc[:,'correct'] = df['correct'].astype(float)
    return df
    

def two_stage_decision_post(df):
    group_df = pandas.DataFrame()
    trials = df.groupby('exp_stage')['trial_num'].max()
    for worker_i, worker in enumerate(numpy.unique(df['worker_id'])):
        try:
            rows = []
            worker_df = df[df['worker_id'] == worker]
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
                    row['key_press_first'] = row.pop('key_press')
                    row['key_press_second'] = ss.get('key_press',-1)
                    row['rt_first'] = row.pop('rt')
                    row['rt_second'] = ss.get('rt',-1)
                    row['stage_second'] = ss.get('stage',-1)
                    row['stim_order_first'] = row.pop('stim_order')
                    row['stim_order_second'] = ss.get('stim_order_second',-1)
                    row['stim_selected_first'] = row.pop('stim_selected')
                    row['stim_selected_second'] = ss.get('stim_selected',-1)
                    row['stage_transition'] = ss.get('stage_transition',numpy.nan)
                    row['feedback'] = fb.get('feedback',numpy.nan)
                    row['FB_probs'] = fb.get('FB_probs',numpy.nan)
                    if row['rt_second'] != -1:
                        row['trial_id'] = 'complete_trial'
                    rows.append(row)
            rows.append(worker_df.iloc[-1].to_dict())
            worker_df = pandas.DataFrame(rows)
            trial_index = ["%s_%s" % ('two_stage_decision',str(x).zfill(3)) for x in range(len(worker_df))]
            worker_df.index = trial_index
            #manipulation check
            win_stay = 0.0
            subset = worker_df[worker_df['exp_stage']=='test']
            for stage in numpy.unique(subset['stage_second']):
                stage_df=subset[(subset['stage_second']==stage) & (subset['trial_id'] == "complete_trial")][['feedback','stim_selected_second']]
                stage_df.insert(0, 'next_choice', stage_df['stim_selected_second'].shift(-1))
                stage_df.insert(0, 'stay', stage_df['stim_selected_second'] == stage_df['next_choice'])
                win_stay+= stage_df[stage_df['feedback']==1]['stay'].sum()
            win_stay_proportion = win_stay/subset['feedback'].sum()
            if win_stay_proportion > .5:
                worker_df.loc[:,'passed_check'] = True
            else:
                worker_df.loc[:,'passed_check'] = False
                print('Two Stage Decision: Worker %s failed manipulation check. Win stay = %s' % (worker, win_stay_proportion))
            group_df = pandas.concat([group_df,worker_df])
        except:
            print('Could not process two_stage_decision dataframe with worker: %s' % worker)
            df.loc[df.worker_id == worker,'passed_check'] = False
    if (len(group_df)>0):
        group_df.insert(0, 'switch', group_df['stim_selected_first'].diff()!=0)
        group_df.insert(0, 'stage_transition_last', group_df['stage_transition'].shift(1))
        group_df.insert(0, 'feedback_last', group_df['feedback'].shift(1))
        df = group_df
    return df

def WATT_post(df):
    # correct bug where exp stage is incorrectly labeled practice in some trials
    test_index = df.loc[:,'condition'].apply(lambda x: x in  ['PA_with_intermediate', 'PA_without_intermediate'])
    df.loc[test_index,'exp_stage']='test'
    # add problem id to feedback rows
    index = df.query('trial_id == "feedback"').index
    i_index = [df.index.get_loc(i)-1 for i in index]
    df.loc[index,'problem_id'] = df.iloc[i_index]['problem_id'].tolist()
    df.loc[index,'condition'] = df.iloc[i_index]['condition'].tolist()
    return df
 
"""
DV functions
"""

@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'adaptive_n_back'})
def calc_adaptive_n_back_DV(df, dvs = {}):
    """ Calculate dv for adaptive_n_back task. Maximum load
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    control_df = df.query('exp_stage == "control"')
    df = df.query('exp_stage == "adaptive"')
    
    # post error slowing
    post_error_slowing = get_post_error_slow(df)
    
    # subset df
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df.query('load == 2')))  
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    
    dvs['mean_load'] = {'value':  df.groupby('block_num').load.mean().mean(), 'valence': 'Pos'}
    #dvs['proactive_interference'] = {'value':  rs.params['recency'], 'valence': 'Neg'}
    description =  """
    mean load used as a measure of memory performance. DDM parameters only calculated
    when load = 2"""
    return dvs, description

@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'attention_network_task'})
def calc_ANT_DV(df, dvs = {}):
    """ Calculate dv for attention network task: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add columns for congruency sequence effect
    df.insert(0,'flanker_shift', df.flanker_type.shift(1))
    df.insert(0, 'correct_shift', df.correct.shift(1))
    
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True')
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df, condition = 'flanker_type'))
    # get DDM across all trials
    ez_alltrials = EZ_diffusion(df)
    dvs.update(ez_alltrials)
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    # Get three network effects
    cue_rt = df_correct.groupby('cue').rt.median()
    flanker_rt = df_correct.groupby('flanker_type').rt.median()
    cue_acc = df.groupby('cue').correct.mean()
    flanker_acc = df.groupby('flanker_type').correct.mean()
    
    dvs['alerting_rt'] = {'value':  (cue_rt.loc['nocue'] - cue_rt.loc['double']), 'valence': 'Pos'}
    dvs['orienting_rt'] = {'value':  (cue_rt.loc['center'] - cue_rt.loc['spatial']), 'valence': 'Pos'}
    dvs['conflict_rt'] = {'value':  (flanker_rt.loc['incongruent'] - flanker_rt.loc['congruent']), 'valence': 'Neg'}

    dvs['congruent_rt'] = {'value':  flanker_rt.loc['congruent'], 'valence': 'Pos'}
    dvs['incongruent_rt'] = {'value':  flanker_rt.loc['incongruent'], 'valence': 'Pos'}
    dvs['neutral_rt'] = {'value':  flanker_rt.loc['neutral'], 'valence': 'Pos'}

    dvs['alerting_acc'] = {'value':  (cue_acc.loc['nocue'] - cue_acc.loc['double']), 'valence': 'NA'}
    dvs['orienting_acc'] = {'value':  (cue_acc.loc['center'] - cue_acc.loc['spatial']), 'valence': 'NA'}
    dvs['conflict_acc'] = {'value':  (flanker_acc.loc['incongruent'] - flanker_acc.loc['congruent']), 'valence': 'Pos'}
    
    # DDM equivalents
    param_valence = {'drift': 'Pos', 'thresh': 'Pos', 'non_decision': 'NA'}
    if set(['EZ_drift_congruent', 'EZ_drift_incongruent']) <= set(dvs.keys()):
        cue_ddm = EZ_diffusion(df,'cue')
        for param in ['drift','thresh','non_decision']:
            dvs['alerting_EZ_' + param] = {'value':  cue_ddm['EZ_' + param + '_nocue']['value'] - cue_ddm['EZ_' + param + '_double']['value'], 'valence': param_valence[param]}
            dvs['orienting_EZ_' + param] = {'value':  cue_ddm['EZ_' + param + '_center']['value'] - cue_ddm['EZ_' + param + '_spatial']['value'], 'valence': param_valence[param]}
            dvs['conflict_EZ_' + param] = {'value':  dvs['EZ_' + param + '_incongruent']['value'] - dvs['EZ_' + param + '_congruent']['value'], 'valence': param_valence[param]}

    for param in ['drift','thresh','non_decision']:
        if set(['hddm_' + param + '_congruent', 'hddm_' + param + '_incongruent']) <= set(dvs.keys()):
            dvs['conflict_hddm_' + param] = {'value':  dvs['hddm_' + param + '_incongruent']['value'] - dvs['hddm_' + param + '_congruent']['value'], 'valence': param_valence[param]}
        if set(['hddm_' + param + '_center', 'hddm_' + param + '_spatial', 'hddm_' + param + '_nocue', 'hddm_' + param + '_double']) <= set(dvs.keys()):
            dvs['alerting_hddm_' + param] = {'value':  dvs['hddm_' + param + '_nocue']['value'] - dvs['hddm_' + param + '_double']['value'], 'valence': param_valence[param]}
            dvs['orienting_hddm_' + param] = {'value':  dvs['hddm_' + param + '_center']['value'] - dvs['hddm_' + param + '_spatial']['value'], 'valence': param_valence[param]}
    # remove unnecessary cue dvs
    for key in list(dvs.keys()):
        if any(x in key for x in ['center','double', 'spatial', 'nocue']):
            del dvs[key]
            
    #congruency sequence effect
    congruency_seq_rt = df_correct.query('correct_shift == True').groupby(['flanker_shift','flanker_type']).rt.median()
    congruency_seq_acc = df.query('correct_shift == True').groupby(['flanker_shift','flanker_type']).correct.mean()
    
    try:
        seq_rt = (congruency_seq_rt['congruent','incongruent'] - congruency_seq_rt['congruent','congruent']) - \
            (congruency_seq_rt['incongruent','incongruent'] - congruency_seq_rt['incongruent','congruent'])
        seq_acc = (congruency_seq_acc['congruent','incongruent'] - congruency_seq_acc['congruent','congruent']) - \
            (congruency_seq_acc['incongruent','incongruent'] - congruency_seq_acc['incongruent','congruent'])
        dvs['congruency_seq_rt'] = {'value':  seq_rt, 'valence': 'NA'}
        dvs['congruency_seq_acc'] = {'value':  seq_acc, 'valence': 'NA'}
    except KeyError:
        pass
    
    description = """
    DVs for "alerting", "orienting" and "conflict" attention networks are of primary
    interest for the ANT task, all concerning differences in RT. 
    Alerting is defined as nocue - double cue trials. Positive values
    indicate the benefit of an alerting double cue. Orienting is defined as center - spatial cue trials.
    Positive values indicate the benefit of a spatial cue. Conflict is defined as
    incongruent - congruent flanker trials. Positive values indicate the benefit of
    congruent trials (or the cost of incongruent trials). RT measured in ms and median
    RT are used for all comparisons.
    
    DDM comparisons are used for conflict conditions. Drift is expected to be lower for the more difficult
    incongruent condition (incongruent-congruent is negative). Thus higher values are good, meaning there
    is less of a disparity. Thresholds may be higher in conflict conditions. Higher values (incongruent-congruent)
    are also good, as they indicate adapatible changes in the amount of evidence needed based on the 
    difficulty of the trial.
    """
    return dvs, description


@group_decorate()
def calc_ART_sunny_DV(df, dvs = {}):
    """ Calculate dv for choice reaction time: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('trial_id == "round_over"').reset_index(drop = True)
    adjusted_df = df.query('caught_blue == 0')
    scores = adjusted_df.groupby('release').max()['tournament_bank']
    clicks = adjusted_df.groupby('release').mean()['clicks_before_end']
    percent_blue = df.groupby('release').caught_blue.mean()
    dvs['keep_score'] = {'value':  scores['Keep'], 'valence': 'Pos'}    
    dvs['release_score'] = {'value':  scores['Release'], 'valence': 'Pos'}  
    dvs['keep_adjusted_clicks'] = {'value':  clicks['Keep'], 'valence': 'Neg'}    
    dvs['release_adjusted_clicks'] = {'value':  clicks['Release'], 'valence': 'Neg'}
    dvs['keep_loss_percent'] = {'value':  percent_blue['Keep'], 'valence': 'Neg'}
    dvs['release_loss_percent'] = {'value':  percent_blue['Release'], 'valence': 'Neg'}    
    description = """DVs are the total tournament score for each condition, the average number of clicks per condition, 
                    and the percent of time the blue fish is caught"""  
    return dvs, description

@group_decorate()
def calc_bickel_DV(df, dvs = {}):
    """ Calculate dv for bickel task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """         

    df = df.query('exp_stage == "test"')

    df_small = df.query('larger_amount == 10')
    df_medium = df.query('larger_amount == 1000')
    df_large = df.query('larger_amount == 1000000')

    warnings = []

    if df.shape[0] != 90:
        warnings.append('Incorrect number of total trials for worker_id:'+ list(set(df['worker_id']))[0])
    if df_small.shape[0] != 30:
        warnings.append('Incorrect number of trials in small condition for worker_id:'+ list(set(df['worker_id']))[0])
    if df_medium.shape[0] != 30:
            warnings.append('Incorrect number of trials in medium condition for worker_id:'+ list(set(df['worker_id']))[0])
    if df_large.shape[0] != 30:
                warnings.append('Incorrect number of trials in large condition for worker_id:'+ list(set(df['worker_id']))[0])

    def geo_mean(l):
        return mstats.gmean(l, axis=0)

    def get_hyp_decayed_value(data_cut):
        implied_k_for_titrator = numpy.nan
        data_cut = data_cut.sort_values(by = 'implied_k')
        if(len(numpy.unique(numpy.diff(data_cut['patient1_impatient0']))) == 1):
            if(set(data_cut['patient1_impatient0']) == {0.0}):
                implied_k_for_titrator = max(data_cut['implied_k'])
            elif(set(data_cut['patient1_impatient0']) == {1.0}):
                implied_k_for_titrator = min(data_cut['implied_k'])
        else:
            switch_point = numpy.where(abs(numpy.diff(data_cut['patient1_impatient0'])) == 1)[0][0]
            a = list(data_cut['implied_k'])[switch_point]
            b = list( data_cut['implied_k'])[switch_point+1]
            implied_k_for_titrator = mstats.gmean([a,b], axis=0)
        decayed_value = float(list(set(data_cut['larger_amount']))[0])/(1+implied_k_for_titrator*float(list(set(data_cut['later_time_days']))[0]))
        return decayed_value
        
    def get_raw_decayed_value(data_cut):
        if(len(numpy.unique(numpy.diff(data_cut['patient1_impatient0']))) == 1):
            if(set(data_cut['patient1_impatient0']) == {0.0}):
                decayed_value = min(data_cut['smaller_amount'])
            elif(set(data_cut['patient1_impatient0']) == {1.0}):
                decayed_value = max(data_cut['smaller_amount'])
        else:
            switch_point = numpy.where(abs(numpy.diff(data_cut['patient1_impatient0'])) == 1)[0][0]
            a = list(data_cut['smaller_amount'])[switch_point]
            b = list(data_cut['smaller_amount'])[switch_point+1]
            decayed_value = (a+b)/2
        return decayed_value    

    def minimize_rss(x0, decayed_values, larger_amount):
        k = x0
        decayed_values_df = pandas.DataFrame(list(decayed_values.items()), columns=['later_time_days', 'decayed_val'])
        decayed_vals = decayed_values_df['decayed_val']
        later_time_days = decayed_values_df['later_time_days']
        larger_amount = numpy.repeat(larger_amount, len(decayed_vals))
        pred_decayed_vals = [a/(1+(k*d)) for a,d in zip(larger_amount, later_time_days)]
        rss = sum([(a - b)*(a-b) for a, b in zip(decayed_vals, pred_decayed_vals)])
        return float(rss)
            
    def optim_hyp_discount_rate_nm(decayed_values, larger_amount):
        hyp_discount_rate_nm = 0.0
        try:
            x0=0
            xopt = optimize.fmin(minimize_rss,x0,args=(decayed_values,larger_amount),xtol=1e-6,ftol=1e-6, disp=False)
            fopt = minimize_rss(xopt, decayed_values, larger_amount)
            hyp_discount_rate_nm = xopt[0]
        except:
            warnings.append(sys.exc_info()[1])
            hyp_discount_rate_nm = 'NA'
            fopt = 'NA'
        return {"hyp_discount_rate_nm":hyp_discount_rate_nm, "min_rss":fopt} 

    def calculate_discount_rate(data):
        decayed_values = {}
        for given_delay in list(set(data['later_time_days'])):
            data_cut = data[data.later_time_days.isin([given_delay])]
            decayed_values[given_delay] = get_hyp_decayed_value(data_cut)
        discount_rate = optim_hyp_discount_rate_nm(decayed_values, list(set(data['larger_amount'])))['hyp_discount_rate_nm']
        min_rss = optim_hyp_discount_rate_nm(decayed_values, list(set(data['larger_amount'])))['min_rss']
        num_trials = data.shape[0]
        return {"discount_rate":discount_rate, "min_rss":min_rss, "num_trials":num_trials}
        
    def calculate_auc(data):
        decayed_values = {}
        for given_delay in list(set(data['later_time_days'])):
            data_cut = data[data.later_time_days.isin([given_delay])]
            decayed_values[given_delay] = get_raw_decayed_value(data_cut)
        
        large_amt = list(set(data['larger_amount']))
        max_delay = max(decayed_values.keys())
        decayed_values.update((x, y/large_amt) for x, y in decayed_values.items())
        normalized_keys = [x / max_delay for x in list(decayed_values.keys())]
        tr = dict(zip(decayed_values.keys(), normalized_keys))
        normalized_decayed_values = {tr[k]: v for k, v in decayed_values.items()}
                                     
        xs = [x for x,y in normalized_decayed_values.items()]
        xs.extend([0])
        xs = sorted(xs)
        ys = [y for x,y in normalized_decayed_values.items()]
        ys.extend([1])
        ys = sorted(ys, reverse=True) 
                  
        aucs = []
        
        for i in range(len(xs)-1):
            trap = (xs[i+1]-xs[i])*((ys[i]+ys[i+1])/2)
            aucs.extend([trap])
            
        auc = sum(aucs)
        return float(auc)
        

    discount_rates = df.groupby('larger_amount').apply(calculate_discount_rate)
        
    dvs['hyp_discount_rate_small'] = {'value': discount_rates[10]['discount_rate'], 'valence': 'Neg'}
    dvs['hyp_discount_rate_medium'] = {'value': discount_rates[1000]['discount_rate'], 'valence': 'Neg'}
    dvs['hyp_discount_rate_large'] = {'value': discount_rates[1000000]['discount_rate'], 'valence': 'Neg'}
    dvs['min_rss_small'] = {'value': discount_rates[10]['min_rss'], 'valence': 'Neg'}
    dvs['min_rss_medium'] = {'value': discount_rates[1000]['min_rss'], 'valence': 'Neg'}
    dvs['min_rss_large'] = {'value': discount_rates[1000000]['min_rss'], 'valence': 'Neg'}
    dvs['num_trials_small'] = {'value': discount_rates[10]['num_trials'], 'valence': 'Pos'}
    dvs['num_trials_medium'] = {'value': discount_rates[1000]['num_trials'], 'valence': 'Pos'}
    dvs['num_trials_large'] = {'value': discount_rates[1000000]['num_trials'], 'valence': 'Pos'}
    dvs['auc_small'] = {'value': calculate_auc(df_small), 'valence': 'Pos'}
    dvs['auc_medium'] = {'value': calculate_auc(df_medium), 'valence': 'Pos'}
    dvs['auc_large'] = {'value': calculate_auc(df_large), 'valence': 'Pos'}
    dvs['warnings'] = {'value': warnings, 'valence': 'NA'}   
                
    description = """
    Three hyperbolic discount rates and areas under the curve for each subject: 
    One for each reward size ($10, $1000, $1000000)"""
    return dvs, description

@group_decorate()
def calc_CCT_cold_DV(df, dvs = {}):
    """ Calculate dv for ccolumbia card task, cold version
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    rs = smf.ols(formula = 'num_cards_chosen ~ gain_amount + loss_amount + num_loss_cards', data = df).fit()
    dvs['avg_cards_chosen'] = {'value':  df['num_cards_chosen'].mean(), 'valence': 'NA'}
    dvs['gain_sensitivity'] = {'value':  rs.params['gain_amount'], 'valence': 'Pos'}
    dvs['loss_sensitivity'] = {'value':  rs.params['loss_amount'], 'valence': 'Pos'}
    dvs['probability_sensitivity'] = {'value':  rs.params['num_loss_cards'], 'valence': 'Pos'}
    dvs['information_use'] = {'value':  numpy.sum(rs.pvalues[1:]<.05), 'valence': 'Pos'}
    dvs['log_ll'] = {'value':  rs.llf, 'valence': 'NA'}
    dvs['num_trials'] = {'value': df.shape[0], 'valence': 'Pos'}
    description = """
        Avg_cards_chosen is a measure of risk ttaking
        gain sensitivity: beta value for regression predicting number of cards
            chosen based on gain amount on trial
        loss sensitivty: as above for loss amount
        probability sensivitiy: as above for number of loss cards
        information use: ranges from 0-3 indicating how many of the sensivitiy
            parameters significantly affect the participant's 
            choices at p < .05
    """
    return dvs, description


@group_decorate()
def calc_CCT_hot_DV(df, dvs = {}):
    """ Calculate dv for ccolumbia card task, cold version
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('mouse_click == "collectButton" and round_type == "rigged_win"').reset_index(drop = True)
    subset = df[~df['clicked_on_loss_card'].astype(bool)]
    rs = smf.ols(formula = 'total_cards ~ gain_amount + loss_amount + num_loss_cards', data = subset).fit()
    dvs['avg_cards_chosen'] = {'value':  subset['total_cards'].mean(), 'valence': 'NA'}
    dvs['gain_sensitivity'] = {'value':  rs.params['gain_amount'], 'valence': 'Pos'}
    dvs['loss_sensitivity'] = {'value':  rs.params['loss_amount'], 'valence': 'Pos'}
    dvs['probability_sensitivity'] = {'value':  rs.params['num_loss_cards'], 'valence': 'Pos'}
    dvs['information_use'] = {'value':  numpy.sum(rs.pvalues[1:]<.05), 'valence': 'Pos'}
    dvs['log_ll'] = {'value':  rs.llf, 'valence': 'NA'}
    dvs['num_trials'] = {'value': subset.shape[0], 'valence': 'Pos'}
    description = """
        Avg_cards_chosen is a measure of risk ttaking
        gain sensitivity: beta value for regression predicting number of cards
            chosen based on gain amount on trial
        loss sensitivty: as above for loss amount
        probability sensivitiy: as above for number of loss cards
        information use: ranges from 0-3 indicating how many of the sensivitiy
            parameters significantly affect the participant's 
            choices at p < .05
    """
    return dvs, description

@group_decorate()
def calc_CCT_fmri_DV(df, dvs = {}):
    """ Calculate dv for ccolumbia card task, fmri version
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # relable action as categorical variable
    df.loc[:,'action'] = pandas.Categorical(df.action, categories = ['draw_card', 'end_round'])
    rs = smf.glm(formula = 'action ~ risk + EV + num_click_in_round', 
                 data = df, family = sm.families.Binomial()).fit()
    dvs['Intercept'] = {'value':  rs.params['Intercept'], 'valence': 'Pos'}
    dvs['EV_sensitivity'] = {'value':  rs.params['EV'], 'valence': 'Pos'}
    dvs['risk_sensitivity'] = {'value':  rs.params['risk'], 'valence': 'NA'}
    description = """
        Expected value and risk are calculated for each choice. These independent
        variables are then used as predictors of choice: either taking a card
        or stopping
    """
    return dvs, description
    
@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'choice_reaction_time'})
def calc_choice_reaction_time_DV(df, dvs = {}):
    """ Calculate dv for choice reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df))
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    description = 'standard'  
    return dvs, description

@group_decorate()
def calc_cognitive_reflection_DV(df, dvs = {}):
    dvs['correct_proportion'] = {'value':  df.correct.mean(), 'valence': 'Pos'} 
    dvs['intuitive_proportion'] = {'value':  df.responded_intuitively.mean(), 'valence': 'Neg'}

    description = 'how many questions were answered correctly (acc) or were mislead by the obvious lure (intuitive proportion'
    return dvs,description

@group_decorate()
def calc_dietary_decision_DV(df, dvs = {}):
    """ Calculate dv for dietary decision task. Calculate the effect of taste and
    health rating on choice
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df[~ pandas.isnull(df['taste_diff'])].reset_index(drop = True)
    rs = smf.ols(formula = 'coded_response ~ health_diff + taste_diff', data = df).fit()
    dvs['health_sensitivity'] = {'value':  rs.params['health_diff'], 'valence': 'Pos'} 
    dvs['taste_sensitivity'] = {'value':  rs.params['taste_diff'], 'valence': 'Neg'} 
    dvs['log_ll'] = {'value':  rs.llf, 'valence': 'NA'}
    dvs['num_trials'] = {'value': df.shape[0], 'valence': 'Pos'}
    df_cut = df.query('exp_stage == "decision"')
    num_healthy = len(df_cut.query('health_diff>0 & (mouse_click == "Yes" | mouse_click == "Strong_Yes")'))
    denom_healthy = len(df_cut.query('mouse_click != "-1"'))

    dvs['prop_healthy_choice'] = {'value': num_healthy/denom_healthy , 'valence': 'Pos'}

    description = """
        Both taste and health sensitivity are calculated based on the decision phase.
        On each trial the participant indicates whether they would prefer a food option
        over a reference food. Their choice is regressed on the subjective health and
        taste difference between that option and the reference item. Positive values
        indicate that the option's higher health/taste relates to choosing the option
        more often. Also includes proportion of healthy choices.
    """
    return dvs,description
    
@group_decorate()
def calc_digit_span_DV(df, dvs = {}):
    """ Calculate dv for digit span: forward and reverse span
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add trial nums
    base = list(range(14))    
    df.insert(0,'trial_num', base * int(len(df)/14))
    
    # subset
    df = df.query('rt >= 0 -1 and trial_num > 3').reset_index(drop = True)
    
    # calculate DVs
    span = df.groupby(['condition'])['num_digits'].mean()
    dvs['forward_span'] = {'value':  span['forward'], 'valence': 'Pos'} 
    dvs['reverse_span'] = {'value':  span['reverse'], 'valence': 'Pos'} 
    
    description = 'Mean span after dropping the first 4 trials'  
    return dvs, description

@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'directed_forgetting'})
def calc_directed_forgetting_DV(df, dvs = {}):
    """ Calculate dv for directed forgetting
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    post_error_slowing = get_post_error_slow(df)
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df, condition = 'probe_type'))
    # get DDM across all trials
    ez_alltrials = EZ_diffusion(df)
    dvs.update(ez_alltrials)
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    # context effects
    rt_contrast = df_correct.groupby('probe_type').rt.median()
    acc_contrast = df.groupby('probe_type').correct.mean()
    dvs['proactive_interference_rt'] = {'value':  rt_contrast['neg'] - rt_contrast['con'], 'valence': 'Neg'} 
    dvs['proactive_interference_acc'] = {'value':  acc_contrast['neg'] - acc_contrast['con'], 'valence': 'Pos'} 
    # DDM equivalents
    if set(['EZ_drift_neg', 'EZ_drift_con']) <= set(dvs.keys()):
        dvs['proactive_interference_EZ_drift'] = {'value':  dvs['EZ_drift_neg']['value'] - dvs['EZ_drift_con']['value'], 'valence': 'Pos'}
        dvs['proactive_interference_EZ_thresh'] = {'value':  dvs['EZ_thresh_neg']['value'] - dvs['EZ_thresh_con']['value'], 'valence': 'Pos'}
        dvs['proactive_interference_EZ_non_decision'] = {'value':  dvs['EZ_non_decision_neg']['value'] - dvs['EZ_non_decision_con']['value'], 'valence': 'NA'}
    param_valence = {'drift': 'Pos', 'thresh': 'Pos', 'non_decision': 'NA'}
    for param in ['drift','thresh','non_decision']:
        if set(['hddm_' + param + '_neg', 'hddm_' + param + '_con']) <= set(dvs.keys()):
            dvs['proactive_interference_hddm_' + param] = {'value':  dvs['hddm_' + param + '_neg']['value'] - dvs['hddm_' + param + '_con']['value'], 'valence': param_valence[param]}

    description = """
    Each DV contrasts trials where subjects were meant to forget the letter vs.
    trials where they had never seen the letter. On both types of trials the
    subject is meant to respond that the letter was not in the memory set. RT
    contrast is only computed for correct trials. Interference for both is calculated as
    Negative - Control trials, so interference_RT should be higher for worse interference and interference_acc
    should be lower for worse interference.
    
    DDM comparisons are used as well. Drift is expected to be lower for the more difficult
    "negative" condition (negative-control is negative). Thus higher values are good, meaning there
    is less of a disparity. Thresholds may be higher in the negative condition. Higher values (negative-control)
    are also good, as they indicate adapatible changes in the amount of evidence needed based on the 
    difficulty of the trial.
    """ 
    return dvs, description
    
@group_decorate()
def calc_discount_fixed_DV(df, dvs={}):
    dvs['placeholder'] = {'value': 999, 'valence': 'NA'}
    description = """ 
    To be filled
    """
    return dvs, description

@group_decorate()
def calc_discount_titrate_DV(df, dvs = {}):
    """ Calculate dv for discount_titrate task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    
    import random
    
    #initiate warnings array for any errors during estimation
    warnings = []
    
    #filter only the test stage choice data
    df = df.query('exp_stage == "test"')
    if df.shape[0] != 36:
        warnings.append('Incorrect number of trials for worker_id:'+ set(df['worker_id']))
    
    dvs = {}
    #Simples dv: percent of patient choices
    dvs['percent_patient'] = {'value': df['patient1_impatient0'].mean(), 'valence': 'NA'}
    #Second dv: hyperbolic discount rates calculated using glm on implied indifference discount rates by each choice
    #for people who have maxed out use the min/max implied indiff_k
                
    def calculate_hyp_discount_rate_glm(data):
        hyp_discount_rate_glm = 0
        if(set(data['patient1_impatient0']) == {0.0}):
            hyp_discount_rate_glm = max(data['indiff_k'])
        elif(set(data['patient1_impatient0']) == {1.0}):
            hyp_discount_rate_glm = min(data['indiff_k'])
        else:
            try:
                rs = smf.glm(formula = 'patient1_impatient0 ~ indiff_k', data = data, family = sm.families.Binomial()).fit()
                hyp_discount_rate_glm = -rs.params[0]/rs.params[1]
                log_ll = rs.llf
                num_trials = data.shape[0]
            except:                                                                         
                #error behavior if glm fails
                #first save error message
                warnings.append(sys.exc_info()[1])
                #then try estimating on a noisier column
                try:
                    def add_noise(column):
                        noise_rows = random.sample(range(len(column)), 3)
                        noisy_column = []
                        for i in range(len(column)):
                          if i in noise_rows:
                            noisy_column.append((column[i]*-1)+1)
                          else:
                            noisy_column.append(column[i])
                        return noisy_column
                        
                    data['patient1_impatient0_noisy'] = add_noise(data['patient1_impatient0'])
                    rs = smf.glm(formula = 'patient1_impatient0_noisy ~ indiff_k', data = data, family = sm.families.Binomial()).fit()
                    hyp_discount_rate_glm = -rs.params[0]/rs.params[1]
                    log_ll = rs.llf
                    num_trials = data.shape[0]
                except:
                    warnings.append(sys.exc_info()[1])
                    #if that also fails assign NA
                    hyp_discount_rate_glm = 'NA'
                    log_ll = 'NA'
                    num_trials = 'NA'
        return {"hyp_discount_rate_glm":hyp_discount_rate_glm, "log_ll":log_ll, 'num_trials': num_trials}           
    
    discount_rates_glm = calculate_hyp_discount_rate_glm(df)

    dvs['hyp_discount_rate_glm'] = {'value': discount_rates_glm['hyp_discount_rate_glm'], 'valence': 'Neg'}
    dvs['log_ll_glm'] = {'value': discount_rates_glm['log_ll'], 'valence': 'Neg'}
    dvs['num_trials_glm'] = {'value': discount_rates_glm['num_trials'], 'valence': 'Pos'}
    
    #Third dv: hyperbolic discount rates calculated using nelder-mead optimizations
    def calculate_hyp_discount_rate_nm(x0, data):
                
        beta = x0[0]
        k = x0[1]
        
        smaller_amount = list(data['smaller_amount'])
        sooner_days = list(data['sooner_days'])
        larger_amount = list(data['larger_amount'])
        later_days = list(data['later_days'])
        patient1_impatient0 = list(data['patient1_impatient0'])

        u_ss = []
        for i in range(data.shape[0]):
          u_ss.append(float(smaller_amount[i])/float(1+k*(float(sooner_days[i]))))
        u_ll = []
        for i in range(data.shape[0]):
          u_ll.append(float(larger_amount[i])/float(1+k*(float(later_days[i]))))
          
        u_diff = [a - b for a, b in zip(u_ss, u_ll)]
    
        #Calculate choice probs
        #logt: smaller beta (p[1]) larger error
        
        prob = [1/float((1+numpy.exp(beta*x))) for x in u_diff]
    
       #replace 1 and 0 to avoid log(1) and log(0)
        prob = numpy.where(prob == 1, 0.9999, numpy.where(prob == 0, 0.0001, prob)).tolist()                                                            
    
        #get log likelihood
        err = []
        for i in range(data.shape[0]):
            err.append((patient1_impatient0[i] * numpy.log(prob[i])) + ((1 - patient1_impatient0[i])*numpy.log(1-prob[i])))
    
        #sum of negative log likelihood (to be minimized)
        sumerr = -1*sum(err)
    
        return sumerr

    def optim_hyp_discount_rate_nm(data):
        hyp_discount_rate_nm = 0.0
        try:
            x0=[0,0]
            xopt = optimize.fmin(calculate_hyp_discount_rate_nm,x0,args=(data,),xtol=1e-6,ftol=1e-6, disp=False)
            hyp_discount_rate_nm = xopt[1]
            fopt = calculate_hyp_discount_rate_nm(xopt, data)
            num_trials = data.shape[0]
        except:
            warnings.append(sys.exc_info()[1])
            if(set(data['patient1_impatient0']) == {0.0}):
                hyp_discount_rate_nm = max(data['indiff_k'])
                fopt= 'NA'
                num_trials = data.shape[0]
            elif(set(data['patient1_impatient0']) == {1.0}):
                hyp_discount_rate_nm = min(data['indiff_k'])
                fopt= 'NA'
                num_trials = data.shape[0]
            else:
                hyp_discount_rate_nm = 'NA'
                fopt = 'NA'
                num_trials = 'NA'
        return {"hyp_discount_rate_nm":hyp_discount_rate_nm, "neg_log_ll":fopt, "num_trials": num_trials}
            
    discount_rates_nm =  optim_hyp_discount_rate_nm(df)   

    dvs['hyp_discount_rate_nm'] = {'value': discount_rates_nm['hyp_discount_rate_nm'], 'valence': 'Neg'}
    dvs['neg_log_ll_nm'] = {'value': discount_rates_nm['neg_log_ll'], 'valence': 'Neg'}
    dvs['num_trials_nm'] = {'value': discount_rates_nm['num_trials'], 'valence': 'Pos'}
                
    #Fourth dv: discount rate glm for now trials only
    df_now = df.query('now1_notnow0 == 1')
    
    discount_rates_glm_now = calculate_hyp_discount_rate_glm(df_now)
    
    dvs['hyp_discount_rate_glm_now'] = {'value': discount_rates_glm_now['hyp_discount_rate_glm'], 'valence': 'Neg'}
    dvs['log_ll_glm_now'] = {'value': discount_rates_glm_now['log_ll'], 'valence': 'Neg'}
    dvs['num_trials_glm_now'] = {'value': discount_rates_glm_now['num_trials'], 'valence': 'Pos'}
    #Fifth dv: discount rate nm for now trials only
    
    discount_rates_nm_now = optim_hyp_discount_rate_nm(df_now)
    
    dvs['hyp_discount_rate_nm_now'] = {'value': discount_rates_nm_now['hyp_discount_rate_nm'], 'valence': 'Neg'}
    dvs['neg_log_ll_nm_now'] = {'value': discount_rates_nm_now['neg_log_ll'], 'valence': 'Pos'}
    dvs['num_trials_nm_now'] = {'value': discount_rates_nm_now['num_trials'], 'valence': 'Pos'}

    #Sixth dv: discount rate glm for not now trials only
    df_notnow = df.query('now1_notnow0 == 0')
    
    discount_rates_glm_notnow = calculate_hyp_discount_rate_glm(df_notnow)
    
    dvs['hyp_discount_rate_glm_notnow'] = {'value': discount_rates_glm_notnow['hyp_discount_rate_glm'], 'valence': 'Neg'}
    dvs['log_ll_glm_notnow'] = {'value': discount_rates_glm_notnow['log_ll'], 'valence': 'Neg'}
    dvs['num_trials_glm_notnow'] = {'value': discount_rates_glm_notnow['num_trials'], 'valence': 'Pos'}
    #Seventh dv: discount rate nm for not now trials only
    
    discount_rates_nm_notnow = optim_hyp_discount_rate_nm(df_notnow)
    
    dvs['hyp_discount_rate_nm_notnow'] = {'value': discount_rates_nm_notnow['hyp_discount_rate_nm'], 'valence': 'Neg'}
    dvs['neg_log_ll_nm_notnow'] = {'value': discount_rates_nm_notnow['neg_log_ll'], 'valence': 'Neg'}
    dvs['num_trials_nm_notnow'] = {'value': discount_rates_nm_notnow['neg_log_ll'], 'valence': 'Pos'}
    
    #Add any warnings
    dvs['warnings'] = {'value': warnings, 'valence': 'NA'}
    
    description = """
    Calculated percent of patient responses and hyperbolic discount rate. Lower discount rates mean more patient choices. 
    Used two optimization methods: glm and nelder-mead. Also calculates separate discount rates for now and not_now trials.
    """
    return dvs, description
    
@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'dot_pattern_expectancy'})
def calc_DPX_DV(df, dvs = {}):
    """ Calculate dv for dot pattern expectancy task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    N = df.trial_num.max()+1
    # post error slowing
    post_error_slowing = get_post_error_slow(df)
    
    # subset df
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df, condition = 'condition'))
    # get DDM across all trials
    ez_alltrials = EZ_diffusion(df)
    dvs.update(ez_alltrials)
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    # calculate Dprime, adjusting extreme values using the fourth suggestion from 
    # Stanislaw, H., & Todorov, N. (1999). Calculation of signal detection theory measures.
    hit_rate = min(df.query('condition == "AX"').correct.mean(), 1-(.5/N))
    FA_rate = max((1-df.query('condition == "BX"').correct.mean()), .5/N)
    dprime = norm.ppf(hit_rate) - norm.ppf(FA_rate)
    bias = -.5 * (norm.ppf(hit_rate)+norm.ppf(FA_rate))
    dvs['dprime'] = {'value': dprime, 'valence': 'Pos'}
    dvs['bias'] = {'value': bias, 'valence': 'NA'}
    
    # context effects
    rt_contrast_df = df_correct.groupby('condition')['rt'].median()
    acc_contrast_df = df.groupby('condition').correct.mean()
    dvs['AY-BY_rt'] = {'value':  rt_contrast_df['AY'] - rt_contrast_df['BY'], 'valence': 'NA'} 
    try:
        dvs['BX-BY_rt'] = {'value':  rt_contrast_df['BX'] - rt_contrast_df['BY'], 'valence': 'NA'} 
    except KeyError: # in case one condition had no correct values
        pass
    dvs['AY-BY_acc'] = {'value': acc_contrast_df['AY'] - acc_contrast_df['BY'], 'valence': 'NA'} 
    dvs['BX-BY_acc'] = {'value': acc_contrast_df['BX'] - acc_contrast_df['BY'], 'valence': 'NA'} 
    # DDM equivalents
    if set(['EZ_drift_AY', 'EZ_drift_BY']) <= set(dvs.keys()):
        dvs['AY-BY_EZ_drift'] = {'value':  dvs['EZ_drift_AY']['value'] - dvs['EZ_drift_BY']['value'], 'valence': 'NA'}
        dvs['AY-BY_EZ_thresh'] = {'value':  dvs['EZ_thresh_AY']['value'] - dvs['EZ_thresh_BY']['value'], 'valence': 'NA'}
        dvs['AY-BY_EZ_non_decision'] = {'value':  dvs['EZ_non_decision_AY']['value'] - dvs['EZ_non_decision_BY']['value'], 'valence': 'NA'}
    if set(['EZ_drift_BX', 'EZ_drift_BY']) <= set(dvs.keys()):
        dvs['BX-BY_EZ_drift'] = {'value':  dvs['EZ_drift_BX']['value'] - dvs['EZ_drift_BY']['value'], 'valence': 'NA'}
        dvs['BX-BY_EZ_thresh'] = {'value':  dvs['EZ_thresh_BX']['value'] - dvs['EZ_thresh_BY']['value'], 'valence': 'NA'}
        dvs['BX-BY_EZ_non_decision'] = {'value':  dvs['EZ_non_decision_BX']['value'] - dvs['EZ_non_decision_BY']['value'], 'valence': 'NA'}
    for param in ['drift','thresh','non_decision']:
        if set(['hddm_' + param + '_AY', 'hddm_' + param + '_BY']) <= set(dvs.keys()):
            dvs['AY-BY_hddm_' + param] = {'value':  dvs['hddm_' + param + '_AY']['value'] - dvs['hddm_' + param + '_BY']['value'], 'valence': 'NA'}
        if set(['hddm_' + param + '_BX', 'hddm_' + param + '_BY']) <= set(dvs.keys()):
            dvs['BX-BY_hddm_' + param] = {'value':  dvs['hddm_' + param + '_BX']['value'] - dvs['hddm_' + param + '_BY']['value'], 'valence': 'NA'}


    #Lit review DVs
    dvs['AX_errors'] = {'value':  len(df.query('condition == "AX" & correct == 0')), 'valence': 'Neg'}
    dvs['AX_rt'] = {'value':  df.query('condition =="AX"').rt.median(), 'valence': 'NA'}
    dvs['AY_errors'] = {'value':  len(df.query('condition == "AY" & correct == 0')), 'valence': 'Neg'}
    dvs['AY_rt'] = {'value':  df.query('condition =="AY"').rt.median(), 'valence': 'NA'}
    dvs['BX_errors'] = {'value':  len(df.query('condition == "BX" & correct == 0')), 'valence': 'Neg'}
    dvs['BX_rt'] = {'value':  df.query('condition =="BX"').rt.median(), 'valence': 'NA'}
    dvs['BY_errors'] = {'value':  len(df.query('condition == "BY" & correct == 0')), 'valence': 'Neg'}
    dvs['BY_rt'] = {'value':  df.query('condition =="BY"').rt.median(), 'valence': 'NA'}

    description = """
    D' is calculated as hit rate on AX trials - false alarm rate on BX trials (see Henderson et al. 2012).
    Primary contrasts are AY and BX vs the "control" condition, BY. Proactive control should aid BX condition
    but harm AY trials.
    """
    return dvs, description


@group_decorate()
def calc_go_nogo_DV(df, dvs = {}):
    """ Calculate dv for go-nogo task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    dvs['acc'] = {'value': df['correct'].mean(), 'valence': 'Pos'} 
    dvs['go_acc'] = {'value': df[df['condition'] == 'go']['correct'].mean(), 'valence': 'Pos'} 
    dvs['nogo_acc'] = {'value': df[df['condition'] == 'nogo']['correct'].mean(), 'valence': 'Pos'} 
    dvs['go_rt'] ={'value':  df[(df['condition'] == 'go') & (df['rt'] != -1)]['rt'].median(), 'valence': 'Pos'} 
    # calculate Dprime, adjusting extreme values using the fourth suggestion from 
    # Stanislaw, H., & Todorov, N. (1999). Calculation of signal detection theory measures.
    hit_rate = min(df.query('condition == "go"').correct.mean(), 1-(.5/len(df)))
    FA_rate = max((1-df.query('condition == "nogo"').correct.mean()), .5/len(df))
    dprime = norm.ppf(hit_rate) - norm.ppf(FA_rate)
    bias = -.5 * (norm.ppf(hit_rate)+norm.ppf(FA_rate))
    dvs['dprime'] = {'value': dprime, 'valence': 'Pos'}
    dvs['bias'] = {'value': bias, 'valence': 'NA'}

    dvs['commission_errors'] = {'value': 1-df.query('condition == "nogo"').correct.mean(), 'valence':'Neg'}
    dvs['omission_errors'] = {'value': 1-df.query('condition == "go"').correct.mean(), 'valence':'Neg'}

    description = """
        Calculated accuracy for go/stop conditions. 75% of trials are go. D_prime is calculated as the P(response|go) - P(response|nogo)
    """
    return dvs, description
    
@group_decorate()
def calc_hierarchical_rule_DV(df, dvs = {}):
    """ Calculate dv for hierarchical learning task. 
    DVs
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # post error slowing
    post_error_slowing = get_post_error_slow(df)
    
    # subset df
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
        
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    
    #calculate hierarchical success
    dvs['score'] = {'value':  df['correct'].sum(), 'valence': 'Pos'} 
    
    description = 'average reaction time'  
    return dvs, description
	
@group_decorate()
def calc_holt_laury_DV(df, dvs = {}):				
	#total number of safe choices
	#adding total number of risky choices too in case we are aiming for DVs where higher means more impulsive
	#number of switches (should be 1 or for those who max out 0. The originial paper does not exclude subjects with more switches but states that results do not change significantly. For us it could serve as a sanity check)
    """ Calculate dv for holt and laury risk aversion titrator. 
    DVs
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    dvs['number_of_switches'] = {'value': sum(numpy.diff(df['safe1_risky0']) != 0), 'valence': 'NA'} 
    dvs['safe_choices'] = {'value': df['safe1_risky0'].sum(), 'valence': 'NA'} 
    dvs['risky_choices'] = {'value': 10 - df['safe1_risky0'].sum(), 'valence': 'NA'} 

    warnings = []

    def calculate_risk_aversion_nm(x0, data):
                
        risk_aversion = x0[0]
        prob_weighting = x0[1]
        beta = x0[2]

        #limit parameter space brute force
#        if risk_aversion>1:
#            risk_aversion = 1
#        elif risk_aversion<0:
#            risk_aversion = 0
#            
#        if prob_weighting>1:
#            prob_weighting = 1
#        elif prob_weighting<0:
#            prob_weighting = 0

        #These could either be input or parsed from stimuli
        amt1_safe = 100 
        amt2_safe = 80
        amt1_risky = 190
        amt2_risky = 5
#        pred_choices = []
        u_safe = []
        u_risky = []
        safe1_risky0 = list(data['safe1_risky0'])
        
        for prob in numpy.arange(0.1, 1.1, 0.1):
#            u_safe = numpy.exp(-pow(-numpy.log(prob), 1-prob_weighting))*pow(amt1_safe, 1-risk_aversion)+ numpy.exp(-pow(-numpy.log(1-prob), 1-prob_weighting))*pow(amt2_safe, 1-risk_aversion)
#            u_risky = numpy.exp(-pow(-numpy.log(prob), 1-prob_weighting))*pow(amt1_risky, 1-risk_aversion)+ numpy.exp(-pow(-numpy.log(1-prob), 1-prob_weighting))*pow(amt2_risky, 1-risk_aversion)
#            if u_safe>u_risky:
#                pred_choices.append(1)
#            elif u_safe<u_risky:
#                pred_choices.append(0)
#            else:
#                pred_choices.append(numpy.random.binomial(1,0.5))
            u_safe.append(numpy.exp(-pow(-numpy.log(prob), 1-prob_weighting))*pow(amt1_safe, 1-risk_aversion)+ numpy.exp(-pow(-numpy.log(1-prob), 1-prob_weighting))*pow(amt2_safe, 1-risk_aversion))
            u_risky.append(numpy.exp(-pow(-numpy.log(prob), 1-prob_weighting))*pow(amt1_risky, 1-risk_aversion)+ numpy.exp(-pow(-numpy.log(1-prob), 1-prob_weighting))*pow(amt2_risky, 1-risk_aversion))

               
        #get error percent (choice is hardmax not stochastic)
#        real_choices = data['safe1_risky0']
#        match_vector = real_choices == pred_choices
#        error_percent = 1-float(sum(match_vector))/data.shape[0]
#        
#        return error_percent

        u_diff = [a - b for a, b in zip(u_risky, u_safe)]
    
        #Calculate choice probs
        #logt: smaller beta (p[1]) larger error
        
        choice_prob = [1/float((1+numpy.exp(beta*x))) for x in u_diff]
    
       #replace 1 and 0 to avoid log(1) and log(0)
        choice_prob = numpy.where(choice_prob == 1, 0.9999, numpy.where(choice_prob == 0, 0.0001, choice_prob)).tolist()                                                            
    
        #get log likelihood
        err = []
        for i in range(data.shape[0]):
            err.append((safe1_risky0[i] * numpy.log(choice_prob[i])) + ((1 - safe1_risky0[i])*numpy.log(1-choice_prob[i])))
    
        #sum of negative log likelihood (to be minimized)
        sumerr = -1*sum(err)
    
        return sumerr
        
    def optim_risk_aversion_nm(data):
        risk_aversion = 0.0
        prob_weighting = 0.0
        try:
            x0=[0.5,0.5,1]
            xopt = optimize.fmin(calculate_risk_aversion_nm,x0,args=(data,),xtol=1e-6,ftol=1e-6, disp=False)
            risk_aversion = xopt[0]
            prob_weighting = xopt[1]
            beta = xopt[2]
            neg_log_ll = calculate_risk_aversion_nm(xopt, data)
        except:
            warnings.append(sys.exc_info()[1])
            if(set(data['safe1_risky0']) == {0.0}):
                risk_aversion = -0.95
                beta = 'NA'
            elif(set(data['safe1_risky0']) == {1.0}):
                risk_aversion = 1.37
                beta = 'NA'
            else:
                risk_aversion = 'NA'
                prob_weighting = 'NA'
                beta = 'NA'
                neg_log_ll='NA'
        return [risk_aversion, prob_weighting, beta, neg_log_ll]


    dvs['risk_aversion'] = {'value': optim_risk_aversion_nm(df)[0], 'valence': 'Neg'}
    dvs['prob_weighting'] = {'value': optim_risk_aversion_nm(df)[1], 'valence': 'Neg'}
    dvs['beta'] = {'value': optim_risk_aversion_nm(df)[2], 'valence': 'NA'}
    dvs['neg_log_ll'] = {'value': optim_risk_aversion_nm(df)[3], 'valence': 'NA'}
    #Add any warnings
    dvs['warnings'] = {'value': warnings, 'valence': 'NA'}   

    description = 'Number of switches from safe to risky options (or vice versa) as well as number of safe and risky decisions out of 10. Risk aversion is the curvature of the value function and the prob weighting is the curvature of the probability weighting function. Model parameter implementation taken from Toubia et al. (2012) sign independent CPT with slight modification to make larger parameters mean more risk averse and more distorted probability.'  
    return dvs, description
    
@group_decorate()
def calc_IST_DV(df, dvs = {}):
    """ Calculate dv for information sampling task
    DVs
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    points_df = df[df['trial_id'] == "choice"].groupby('exp_stage')['points'].sum()
    contrast_df = df[df['trial_id'] == "choice"].groupby('exp_stage')['correct','P_correct_at_choice','clicks_before_choice'].mean()
    motivation_df = df[df['trial_id'] == "choice"].groupby('exp_stage')['box_open_latency'].mean()
    for condition in ['Decreasing_Win', 'Fixed_Win']:
        dvs[condition + '_motivation'] = {'value':  motivation_df.loc[condition], 'valence': 'NA'} 
        dvs[condition + '_total_points'] = {'value':  points_df.loc[condition], 'valence': 'Pos'} 
        dvs[condition + '_boxes_opened'] = {'value':  contrast_df.loc[condition,'clicks_before_choice'], 'valence': 'NA'} 
        dvs[condition + '_acc'] = {'value':  contrast_df.loc[condition, 'correct'], 'valence': 'Pos'} 
        dvs[condition + '_P_correct'] = {'value':  contrast_df.loc[condition, 'P_correct_at_choice'], 'valence': 'Pos'} 
    description = """ Each dependent variable is calculated for the two conditions:
    DW (Decreasing Win) and FW (Fixed Win). "RT" is the median rt over every choice to open a box,
    "boxes opened" is the mean number of boxes opened before choice, "accuracy" is the percent correct
    over trials and "P_correct" is the P(correct) given the number and distribution of boxes opened on that trial
    """
    return dvs, description

@group_decorate()
def calc_keep_track_DV(df, dvs = {}):
    """ Calculate dv for choice reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('rt != -1').reset_index(drop = True)
    score = df['score'].sum()/df['possible_score'].sum()
    dvs['score'] = {'value': score, 'valence': 'Pos'} 
    description = 'percentage of items remembered correctly'  
    return dvs, description
    
@group_decorate()
def calc_kirby_DV(df, dvs = {}):
    """ Calculate dv for kirby task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """			

    #filter only the test stage choice data
    df = df.query('exp_stage == "test"')
	
    #subset test stage choice data by reward size
    df_small = df.query('reward_size == "small"')
    df_medium = df.query('reward_size == "medium"')
    df_large = df.query('reward_size == "large"')
				
    #create empty list to push all warnings in				
    warnings = []
				
    #check if there are correct number of trials for each conditions
    if df.shape[0] != 27:
        warnings.append('Incorrect number of total trials for worker_id:'+ list(set(df['worker_id']))[0])
    if df_small.shape[0] != 9:
        warnings.append('Incorrect number of trials in small condition for worker_id:'+ list(set(df['worker_id']))[0])
    if df_medium.shape[0] != 9:
        warnings.append('Incorrect number of trials in medium condition for worker_id:'+ list(set(df['worker_id']))[0])
    if df_large.shape[0] != 9:
        warnings.append('Incorrect number of trials in large condition for worker_id:'+ list(set(df['worker_id']))[0])
    

    #Add dv: percent of patient choices
    dvs['percent_patient'] = {'value': df['patient1_impatient0'].mean(), 'valence': 'NA'}

    percent_patients = df.groupby('reward_size').patient1_impatient0.mean()
    dvs['percent_patient_small'] = {'value': percent_patients['small'], 'valence': 'NA'}
    dvs['percent_patient_medium'] = {'value': percent_patients['medium'], 'valence': 'NA'}
    dvs['percent_patient_large'] = {'value': percent_patients['large'], 'valence': 'NA'}

   
    
    #Helper function to get geometric means
    def geo_mean(l):
        return mstats.gmean(l, axis=0)
				
    #Function to calculate discount rates			
    def calculate_discount_rate(data):
					
        def get_match_percent(k, data):
            real_choices = data['patient1_impatient0']
            pred_choices = numpy.where(data['large_amount']/(1+k*data['later_delay']) > data['small_amount'], 1, 0)	
            match_vector = real_choices == pred_choices
            match_percent = float(sum(match_vector))/data.shape[0]
            return match_percent
    
        possible_ks = [0.00016, geo_mean([0.00016, 0.0004]), geo_mean([0.0004, 0.001]), geo_mean([0.001, 0.0025]), geo_mean([0.0025, 0.006]), geo_mean([0.006, 0.016]), geo_mean([0.016, 0.041]), geo_mean([0.041, 0.10]), geo_mean([0.1, 0.25]), 0.25]
        
        match_percentages = {}

        for current_k in possible_ks:
            match_percentages[current_k] = get_match_percent(current_k, data)	
												
        discount_rate = geo_mean([k for k in match_percentages if match_percentages.get(k) == max(match_percentages.values())])
					
        return {"discount_rate": discount_rate, "max_match_pct":max(match_percentages.values())}
        
        
    def calculate_exp_discount_rate(data):
					
        def get_match_percent(exp_k, data):
            real_choices = data['patient1_impatient0']
            pred_choices = numpy.where(data['large_amount']*(exp_k**data['later_delay']) > data['small_amount'], 1, 0)	
            match_vector = real_choices == pred_choices
            match_percent = float(sum(match_vector))/data.shape[0]
            return match_percent
    
        #small_amt = (exp_k**later_del) * large_amt
        #(exp_k**later_del) = small_amt/large_amt
        #exp_k = (small_amt/large_amt)**(1/later_del)    
            
        possible_ks = [0.866, geo_mean([0.866, 0.938]), geo_mean([0.938, 0.970]), geo_mean([0.970, 0.987]), geo_mean([0.987, 0.994]), geo_mean([0.994, 0.997]), geo_mean([0.997, 0.999]), geo_mean([0.999, 0.9996]), geo_mean([0.9996, 0.9998]), 0.9998]
        
        match_percentages = {}

        for current_k in possible_ks:
            match_percentages[current_k] = get_match_percent(current_k, data)	
												
        discount_rate = geo_mean([k for k in match_percentages if match_percentages.get(k) == max(match_percentages.values())])
					
        return {"discount_rate": discount_rate, "max_match_pct":max(match_percentages.values())}
	
    dvs['hyp_discount_rate'] = {'value': calculate_discount_rate(df)['discount_rate'], 'valence': 'Neg'}

    hyp_discount_rates = df.groupby('reward_size').apply(calculate_discount_rate)

    dvs['hyp_discount_rate_small'] = {'value': hyp_discount_rates['small']['discount_rate'], 'valence': 'Neg'}
    dvs['hyp_discount_rate_medium'] = {'value': hyp_discount_rates['medium']['discount_rate'], 'valence': 'Neg'}
    dvs['hyp_discount_rate_large'] = {'value': hyp_discount_rates['large']['discount_rate'], 'valence': 'Neg'}

    dvs['exp_discount_rate'] = {'value': calculate_exp_discount_rate(df)['discount_rate'], 'valence': 'Neg'}
    
    exp_discount_rates = df.groupby('reward_size').apply(calculate_exp_discount_rate)

    dvs['exp_discount_rate_small'] = {'value': exp_discount_rates['small']['discount_rate'], 'valence': 'Neg'}
    dvs['exp_discount_rate_medium'] = {'value': exp_discount_rates['medium']['discount_rate'], 'valence': 'Neg'}
    dvs['exp_discount_rate_large'] = {'value': exp_discount_rates['large']['discount_rate'], 'valence': 'Neg'}

    dvs['hyp_match_pct'] = {'value': calculate_discount_rate(df)['max_match_pct'], 'valence': 'Neg'}
    dvs['hyp_match_pct_small'] = {'value': hyp_discount_rates['small']['max_match_pct'], 'valence': 'Neg'}
    dvs['hyp_match_pct_medium'] = {'value': hyp_discount_rates['medium']['max_match_pct'], 'valence': 'Neg'}
    dvs['hyp_match_pct_large'] = {'value': hyp_discount_rates['large']['max_match_pct'], 'valence': 'Neg'}
    dvs['exp_match_pct'] = {'value': calculate_exp_discount_rate(df)['max_match_pct'], 'valence': 'Neg'}
    dvs['exp_match_pct_small'] = {'value': exp_discount_rates['small']['max_match_pct'], 'valence': 'Neg'}
    dvs['exp_match_pct_medium'] = {'value': exp_discount_rates['medium']['max_match_pct'], 'valence': 'Neg'}
    dvs['exp_match_pct_large'] = {'value': exp_discount_rates['large']['max_match_pct'], 'valence': 'Neg'}
	
    #Add any warnings
    dvs['warnings'] = {'value': warnings, 'valence': 'NA'}   
						
    description = """
    Four hyperbolic and exponential discount rates and number of patient choices for each subject: 
    One for all items, and three depending on the reward size (small, medium, large)"""
    return dvs, description
    
@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'local_global_letter'})
def calc_local_global_DV(df, dvs = {}):
    """ Calculate dv for hierarchical learning task. 
    DVs
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add columns for congruency sequence effect
    df.insert(0,'conflict_condition_shift', df.conflict_condition.shift(1))
    if 'correct_shift' not in df.columns:
        df.insert(0, 'correct_shift', df.correct.shift(1))
    
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df, condition = 'conflict_condition'))
    # get DDM across all trials
    ez_alltrials = EZ_diffusion(df)
    dvs.update(ez_alltrials)
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    # Get congruency effects
    rt_contrast = df_correct.groupby('conflict_condition').rt.median()
    acc_contrast = df.groupby('conflict_condition').correct.mean()

    dvs['congruent_facilitation_rt'] = {'value':  (rt_contrast['neutral'] - rt_contrast['congruent']), 'valence': 'Neg'} 
    dvs['incongruent_harm_rt'] = {'value':  (rt_contrast['incongruent'] - rt_contrast['neutral']), 'valence': 'Neg'} 
    dvs['congruent_facilitation_acc'] = {'value':  (acc_contrast['neutral'] - acc_contrast['congruent']), 'valence': 'Pos'} 
    dvs['incongruent_harm_acc'] = {'value':  (acc_contrast['incongruent'] - acc_contrast['neutral']), 'valence': 'Pos'} 
    dvs['conflict_rt'] = {'value':  dvs['congruent_facilitation_rt']['value'] + dvs['incongruent_harm_rt']['value'], 'valence': 'Neg'} 
    dvs['conflict_acc'] = {'value':  dvs['congruent_facilitation_acc']['value'] + dvs['incongruent_harm_acc']['value'], 'valence': 'Pos'} 
    # DDM equivalents
    if set(['EZ_drift_congruent', 'EZ_drift_incongruent']) <= set(dvs.keys()):
        dvs['conflict_EZ_drift'] = {'value':  dvs['EZ_drift_incongruent']['value'] - dvs['EZ_drift_congruent']['value'], 'valence': 'Pos'}
        dvs['conflict_EZ_thresh'] = {'value':  dvs['EZ_thresh_incongruent']['value'] - dvs['EZ_thresh_congruent']['value'], 'valence': 'Pos'}
        dvs['conflict_EZ_non_decision'] = {'value':  dvs['EZ_non_decision_incongruent']['value'] - dvs['EZ_non_decision_congruent']['value'], 'valence': 'NA'}
    param_valence = {'drift': 'Pos', 'thresh': 'Pos', 'non_decision': 'NA'}
    for param in ['drift','thresh','non_decision']:
        if set(['hddm_' + param + '_congruent', 'hddm_' + param + '_incongruent']) <= set(dvs.keys()):
            dvs['conflict_hddm_' + param] = {'value':  dvs['hddm_' + param + '_incongruent']['value'] - dvs['hddm_' + param + '_congruent']['value'], 'valence': param_valence[param]}

    #congruency sequence effect
    congruency_seq_rt = df_correct.query('correct_shift == True').groupby(['conflict_condition_shift','conflict_condition']).rt.median()
    congruency_seq_acc = df.query('correct_shift == True').groupby(['conflict_condition_shift','conflict_condition']).correct.mean()
    
    try:
        seq_rt = (congruency_seq_rt['congruent','incongruent'] - congruency_seq_rt['congruent','congruent']) - \
            (congruency_seq_rt['incongruent','incongruent'] - congruency_seq_rt['incongruent','congruent'])
        seq_acc = (congruency_seq_acc['congruent','incongruent'] - congruency_seq_acc['congruent','congruent']) - \
            (congruency_seq_acc['incongruent','incongruent'] - congruency_seq_acc['incongruent','congruent'])
        dvs['congruency_seq_rt'] = {'value':  seq_rt, 'valence': 'NA'} 
        dvs['congruency_seq_acc'] = {'value':  seq_acc, 'valence': 'NA'} 
    except KeyError:
        pass
    
    # switch costs
    switch_rt = df_correct.query('correct_shift == 1').groupby('switch').rt.median()
    switch_acc = df.query('correct_shift == 1').groupby('switch').correct.mean()
    dvs['switch_cost_rt'] = {'value':  (switch_rt['switch'] - switch_rt['stay']), 'valence': 'Neg'} 
    dvs['switch_cost_acc'] = {'value':  (switch_acc['switch'] - switch_acc['stay']), 'valence': 'Pos'} 
    # DDM equivalents
    switch_dvs = EZ_diffusion(df.query('correct_shift == 1'), condition = 'switch')
    switch_dvs.update(dvs)
    dvs = switch_dvs
    if set(['EZ_drift_switch', 'EZ_drift_stay']) <= set(dvs.keys()):
        dvs['switch_cost_EZ_drift'] = {'value':  dvs['EZ_drift_switch']['value'] - dvs['EZ_drift_stay']['value'], 'valence': 'Pos'}
        dvs['switch_cost_EZ_thresh'] = {'value':  dvs['EZ_thresh_switch']['value'] - dvs['EZ_thresh_stay']['value'], 'valence': 'Pos'}
        dvs['switch_cost_EZ_non_decision'] = {'value':  dvs['EZ_non_decision_switch']['value'] - dvs['EZ_non_decision_stay']['value'], 'valence': 'NA'}
    param_valence = {'drift': 'Pos', 'thresh': 'Pos', 'non_decision': 'NA'}
    for param in ['drift','thresh','non_decision']:
        if set(['hddm_' + param + '_switch', 'hddm_' + param + '_stay']) <= set(dvs.keys()):
            dvs['switch_cost_hddm_' + param] = {'value':  dvs['hddm_' + param + '_switch']['value'] - dvs['hddm_' + param + '_stay']['value'], 'valence': param_valence[param]}
    
    # Calculate additional statistics for lit review comparison
    dvs['congruent_rt'] = {'value':  df.query('conflict_condition == "congruent"').rt.median(), 'valence': 'Neg'}
    dvs['global_conflict_rt'] = {'value': df.query('conflict_condition == "incongruent" & condition == "global"').rt.median() - df.query('conflict_condition == "congruent" & condition == "global"').rt.median(), 'valence': 'Neg'}
    dvs['global_congruent_errors'] = {'value': 1-df.query('conflict_condition == "congruent" & condition == "global"').correct.mean() , 'valence': 'Neg'}
    dvs['global_congruent_rt'] = {'value': df.query('conflict_condition == "congruent" & condition == "global"').rt.median() , 'valence': 'Neg'}
    dvs['global_error_cost'] = {'value': df.query('condition == "global" & correct == 1').rt.median() - df.query('condition == "global" & correct == 0').rt.median() , 'valence': 'NA'}
    dvs['global_incongruent_errors'] = {'value': 1-df.query('conflict_condition == "incongruent" & condition == "global"').correct.mean() , 'valence': 'Neg'}
    dvs['global_incongruent_rt'] = {'value':  df.query('conflict_condition == "incongruent" & condition == "global"').rt.median(), 'valence': 'Neg'} 
    dvs['incongruent_rt'] = {'value':  df.query('conflict_condition == "incongruent"').rt.median(), 'valence': 'Neg'}
    dvs['local_conflict_rt'] = {'value': df.query('conflict_condition == "incongruent" & condition == "local"').rt.median() - df.query('conflict_condition == "congruent" & condition == "local"').rt.median() , 'valence': 'Neg'}
    dvs['local_congruent_errors'] = {'value': 1-df.query('conflict_condition == "congruent" & condition == "local"').correct.mean() , 'valence': 'Neg'}
    dvs['local_congruent_rt'] = {'value': df.query('conflict_condition == "congruent" & condition == "local"').rt.median() , 'valence': 'Neg'}
    dvs['local_error_cost'] = {'value': df.query('condition == "local" & correct == 1').rt.median() - df.query('condition == "local" & correct == 0').rt.median() , 'valence': 'Neg'}
    dvs['local_incongruent_errors'] = {'value': 1-df.query('conflict_condition == "incongruent" & condition == "local"').correct.mean() , 'valence': 'Neg'}
    dvs['local_incongruent_rt'] = {'value': df.query('conflict_condition == "incongruent" & condition == "local"').rt.median() , 'valence': 'Neg'}    

    description = """
    local-global incongruency effect calculated for accuracy and RT. 
    Facilitation for RT calculated as neutral-congruent. Positive values indicate speeding on congruent trials.
    Harm for RT calculated as incongruent-neutral. Positive values indicate slowing on incongruent trials
    Facilitation for accuracy calculated as congruent-neutral. Positives values indicate higher accuracy for congruent trials
    Harm for accuracy calculated as neutral - incongruent. Positive values indicate lower accuracy for incongruent trials
    Switch costs calculated as switch-stay for rt and stay-switch for accuracy. Thus positive values indicate slowing and higher
    accuracy on switch trials. Expectation is positive rt switch cost, and negative accuracy switch cost
    RT measured in ms and median RT is used for comparison.
        
    DDM comparisons are used for conflict and switch comparisons. Drift is expected to be lower for 
    the more difficultincongruent and switch conditions (incongruent-congruent and switch-stay is negative). 
    Thus higher values are good, meaning there mis less of a disparity. Thresholds may be higher in 
    conflict and switch conditions. Higher values (incongruent-congruent and switch-stay) are also good, as they indicate 
    adapatible changes in the amount of evidence needed based on the difficulty of the trial.
    """
    return dvs, description

@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'motor_selective_stop_signal'})
def calc_motor_selective_stop_signal_DV(df, dvs = {}):
    # subset df to test trials
    df = df.query('exp_stage not in ["practice","NoSS_practice"]').reset_index(drop = True)
    
    # get trials where a response would have to be stopped
    critical_response = df['stop_response'].unique()[0]
    critical_df = df.query('correct_response == %s' % critical_response)
    noncritical_df = df.query('correct_response != %s' % critical_response)
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df.query('condition == "go"')))
    
    # Calculate basic statistics - accuracy, RT and error RT
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['go_acc'] = {'value':  df.query('condition == "go"').correct.mean(), 'valence': 'Pos'} 
    dvs['stop_acc'] = {'value':  df.query('condition == "stop"').correct.mean(), 'valence': 'Pos'} 
    dvs['ignore_acc'] = {'value':  df.query('condition == "ignore"').correct.mean(), 'valence': 'Pos'} 
    
    dvs['go_rt_error'] = {'value':  df.query('correct == False and condition == "go"').rt.median(), 'valence': 'Neg'} 
    dvs['go_rt_std_error'] = {'value':  df.query('correct == False and condition == "go"').rt.std(), 'valence': 'NA'} 
    dvs['go_rt'] = {'value':  df.query('correct == True and condition == "go"').rt.median(), 'valence': 'Neg'} 
    dvs['go_rt_std'] = {'value':  df.query('correct == True and condition == "go"').rt.std(), 'valence': 'NA'} 
    dvs['stop_rt_error'] = {'value':  df.query('stopped == False and condition == "stop"').rt.median(), 'valence': 'Neg'} 
    dvs['stop_rt_error_std'] = {'value':  df.query('stopped == False and condition== "stop"').rt.std(), 'valence': 'NA'} 
    dvs['ignore_rt_error'] = {'value':  df.query('stopped == False and condition == "ignore"').rt.median(), 'valence': 'Neg'} 
    dvs['ignore_rt_error_std'] = {'value':  df.query('stopped == False and condition== "ignore"').rt.std(), 'valence': 'NA'}  
    
    dvs['SS_delay'] = {'value':  df.query('condition == "stop"').SS_delay.mean(), 'valence': 'Pos'}
    
    # calculate SSRT for critical trials
    go_trials = critical_df.query('condition == "go"')
    stop_trials = critical_df.query('condition == "stop"')
    sorted_go = go_trials.query('rt != -1').rt.sort_values(ascending = True)
    prob_stop_failure = (1-stop_trials.stopped.mean())
    corrected = prob_stop_failure/numpy.mean(go_trials.rt!=-1)
    index = corrected*(len(sorted_go)-1)
    index = [floor(index), ceil(index)]
    dvs['SSRT'] = {'value': sorted_go.iloc[index].mean() - stop_trials.SS_delay.mean(), 'valence': 'Neg'}
    
    # Condition metrics
    reactive_control = noncritical_df.query('condition == "ignore" and correct == True').rt.median() - \
                                noncritical_df.query('condition == "go" and correct == True').rt.median()
    proactive_control = critical_df.query('condition == "go"').rt.median() - noncritical_df.query('condition == "go"').rt.median()
    dvs['reactive_control'] = {'value': reactive_control, 'valence': 'Neg'}
    dvs['selective_proactive_control']= {'value': proactive_control, 'valence': 'Pos'}
    
    description = """SSRT is calculated by calculating the percentage of time there are stop failures during
    stop trials. The assumption is that the go process is racing against the stop process and "wins" on the 
    faster proportion of trials. SSRT is thus the go rt at the percentile specified by the failure percentage.
    In the motor selective stop signal only "critical" trials (where the "stopping" hand are used) are used to calculate
    SSRT
    
    Reactive control is the difference between reaction time on noncritical "signal" trials and noncritical non-signal trials.
    Reaction time is expected to be longer on "signal" trials, with greater differences associated with poorer reactive control.
    Selective proactive control is the difference between go trials for the critical and noncritical hand. Proactive control would
    lead to slowing for the critical hand.
    """    
    return dvs, description


# reject if test phase training pair accuracy below .5.

# predict left/right
# choice ~ value_diff + value_diff*value_sum + last_choice

#win-stay/lose-switch: look at first 5 of each pair, proportion of winstay/loseswitch strategy

@group_decorate()
def calc_probabilistic_selection_DV(df, dvs = {}):
    """ Calculate dv for probabilistic selection task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # define helper functions
    def get_value_diff(lst, values):
        lst = [int(i) for i in lst]
        return values[lst[0]] - values[lst[1]]
    def get_value_sum(lst,values):
        lst = [int(i) for i in lst]
        return values[lst[0]] + values[lst[1]]
    
    # convert stim chosen to int
    missed_percent = (df['rt']==-1).mean()
    df = df[df['rt'] != -1].reset_index(drop = True)
    
    #Calculate regression DVs
    train = df.query('exp_stage == "training"')
    values = train.groupby('stim_chosen')['feedback'].mean()
    # fill in values if the subject never selected that stimulus
    for v in [20,30,40,60,70,80]:
        if v not in values.index:
            values.loc[v] = v/100.0
        
    df.loc[:,'value_diff'] = df['condition'].apply(lambda x: get_value_diff(x.split('_'), values) if x==x else numpy.nan)
    df.loc[:,'value_sum'] = df['condition'].apply(lambda x: get_value_sum(x.split('_'), values) if x==x else numpy.nan)  
    df.loc[:, 'choice'] = df['key_press'].map(lambda x: ['right','left'][x == 37]).astype('category', categories = ['left','right'])
    df.loc[:,'choice_lag'] = df['choice'].shift(1)
    test = df.query('exp_stage == "test"')
    rs = smf.glm(formula = 'choice ~ value_diff*value_sum - value_sum + choice_lag', data = test, family = sm.families.Binomial()).fit()
    
    dvs['value_sensitivity'] = {'value':  rs.params['value_diff'], 'valence': 'Pos'} 
    dvs['positive_learning_bias'] = {'value':  rs.params['value_diff:value_sum'], 'valence': 'NA'}
    dvs['log_ll'] = {'value':  rs.llf, 'valence': 'NA'}
    dvs['num_trials'] = {'value':  test.shape[0], 'valence': 'Pos'} 
    dvs['overall_test_acc'] = {'value':  test['correct'].mean(), 'valence': 'Pos'} 
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'} 

    #addiional lit review dv's
    df = df.query('exp_stage == "test"')
    df.loc[:,'low_stim'] = df['condition_collapsed'].apply(lambda x: x.split('_')[0] if x==x else numpy.nan)
    df.loc[:,'high_stim'] = df['condition_collapsed'].apply(lambda x: x.split('_')[1] if x==x else numpy.nan)
    df.loc[:,'approach'] = numpy.where(df['high_stim'] == '80', 1, 0)
    df.loc[:,'avoid'] = numpy.where(df['low_stim'] == '20', 1, 0)
    dvs['approach_trial_acc'] = {'value': df.query('approach == 1').correct.mean(), 'valence': 'Pos'}
    dvs['approach_trial_rt'] = {'value': df.query('approach == 1').rt.median(), 'valence': 'Neg'}
    dvs['avoid_trial_acc'] = {'value': df.query('avoid == 1').correct.mean(), 'valence': 'Pos'}
    dvs['avoid_trial_rt'] = {'value': df.query('avoid == 1').rt.median(), 'valence': 'Neg'}

    description = """
        The primary DV in this task is whether people do better choosing
        positive stimuli or avoiding negative stimuli. Two different measurements
        are calculated. The first is a regression that predicts participant
        accuracy based on the value difference between the two options (defined by
        the participant's actual experience with the two stimuli) and the sum of those
        values. A significant effect of value difference would say that participants
        are more likely to be correct on easier trials. An interaction between the value
        difference and value-sum would say that this effect (the relationship between
        value difference and accuracy) differs based on the sum. A positive learning bias
        would say that the relationship between value difference and accuracy is greater 
        when the overall value is higher.
        
        Another way to calculate a similar metric is to calculate participant accuracy when 
        choosing the two most positive stimuli over other novel stimuli (not the stimulus they 
        were trained on). Negative accuracy can similarly be calculated based on the 
        probability the participant avoided the negative stimuli. Bias is calculated as
        their positive accuracy/negative accuracy. Thus positive values indicate that the
        subject did better choosing positive stimuli then avoiding negative ones. 
        Reference: http://www.sciencedirect.com/science/article/pii/S1053811914010763
    """
    return dvs, description

@group_decorate()
def calc_PRP_two_choices_DV(df, dvs = {}):
    """ Calculate dv for shift task. I
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = ((df['choice1_rt']==-1) | (df['choice2_rt'] <= -1)).mean()
    df = df.query('choice1_rt != -1 and choice2_rt > -1').reset_index(drop = True)
    contrast = df.groupby('ISI').choice2_rt.median()
    dvs['PRP_slowing'] = {'value':  contrast.loc[50] - contrast.loc[800], 'valence': 'NA'} 
    rs = smf.ols(formula = 'choice2_rt ~ ISI', data = df).fit()
    dvs['PRP_slope'] = {'value':  -rs.params['ISI'], 'valence': 'NA'}
    dvs['log_ll'] = {'value':  rs.llf, 'valence': 'NA'}
    dvs['num_trials'] = {'value':  df.shape[0], 'valence': 'Pos'} 
    dvs['task1_acc'] = {'value':  df.choice1_correct.mean(), 'valence': 'Pos'} 
    dvs['task2_acc'] = {'value':  df.choice2_correct.mean(), 'valence': 'Pos'} 
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'} 
    description = """
        The PRP task leads to a slowing of reaction times when two tasks are performed in 
        very quick succession. We define two variables. "PRP slowing" is the difference in median RT
        when the second task follows the first task by 50 ms vs 800 ms. Higher values indicate more slowing.
        "PRP slope" quantifies the same thing, but looks at the slope of the linear regression going through the 
        4 ISI's. Higher values again indicated greater slowing from an ISI of 50 ms to 800ms
        """
    return dvs, description
    
    
@group_decorate()
def calc_ravens_DV(df, dvs = {}):
    """ Calculate dv for ravens task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage == "test" and trial_id == "question"').reset_index(drop = True)
    dvs['score'] = {'value':  df['correct'].sum(), 'valence': 'Pos'} 
    description = 'Score is the number of correct responses out of 18'
    return dvs,description    
    
@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'recent_probes'})
def calc_recent_probes_DV(df, dvs = {}):
    """ Calculate dv for recent_probes
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    post_error_slowing = get_post_error_slow(df)
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df, condition = 'probeType'))
    # get DDM across all trials
    ez_alltrials = EZ_diffusion(df)
    dvs.update(ez_alltrials)
    
    # get single diffusion parameters over all not-recent probes
    xrec_diffusion = EZ_diffusion(df.query('probeType in ["xrec_pos", "xrec_neg"]'))
    dvs['EZ_drift_xrec'] = xrec_diffusion['EZ_drift']
    dvs['EZ_thresh_xrec'] = xrec_diffusion['EZ_thresh']    
    dvs['EZ_non_decision_xrec'] = xrec_diffusion['EZ_non_decision']
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    # calculate contrast dvs
    rt_contrast = df_correct.groupby('probeType').rt.median()
    acc_contrast = df.groupby('probeType').correct.mean()
    dvs['proactive_interference_rt'] = {'value':  rt_contrast['rec_neg'] - rt_contrast['xrec_neg'], 'valence': 'Neg'} 
    dvs['proactive_interference_acc'] = {'value':  acc_contrast['rec_neg'] - acc_contrast['xrec_neg'], 'valence': 'Pos'} 
    # DDM equivalents
    if set(['EZ_drift_rec_neg', 'EZ_drift_xrec_neg']) <= set(dvs.keys()):
        dvs['proactive_interference_EZ_drift'] = {'value':  dvs['EZ_drift_rec_neg']['value'] - dvs['EZ_drift_xrec_neg']['value'], 'valence': 'Pos'}
        dvs['proactive_interference_EZ_thresh'] = {'value':  dvs['EZ_thresh_rec_neg']['value'] - dvs['EZ_thresh_xrec_neg']['value'], 'valence': 'Pos'}
        dvs['proactive_interference_EZ_non_decision'] = {'value':  dvs['EZ_non_decision_rec_neg']['value'] - dvs['EZ_non_decision_xrec_neg']['value'], 'valence': 'NA'}
    param_valence = {'drift': 'Pos', 'thresh': 'Pos', 'non_decision': 'NA'}
    for param in ['drift','thresh','non_decision']:
        if set(['hddm_' + param + '_rec_neg', 'hddm_' + param + '_xrec_neg']) <= set(dvs.keys()):
            dvs['proactive_interference_hddm_' + param] = {'value':  dvs['hddm_' + param + '_rec_neg']['value'] - dvs['hddm_' + param + '_xrec_neg']['value'], 'valence': param_valence[param]}

    description = """
    proactive interference defined as the difference in reaction time and accuracy
    for negative trials (where the probe was not part of the memory set) between
    "recent" trials (where the probe was part of the previous trial's memory set)
    and "non-recent trials" where the probe wasn't.
    
    DDM comparisons are used as well. Drift is expected to be lower for the more difficult
    "recent negative" condition (recent-not recent is negative). Thus higher values are good, meaning there
    is less of a disparity. Thresholds may be higher in the recent negative condition. Higher values 
    (recent negative-not recent) are also good, as they indicate adapatible changes in the amount of evidence 
    needed based on the difficulty of the trial.
    """ 
    return dvs, description
    
@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'shape_matching'})
def calc_shape_matching_DV(df, dvs = {}):
    """ Calculate dv for shape_matching task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df, condition = 'condition'))
    # get DDM across all trials
    ez_alltrials = EZ_diffusion(df)
    dvs.update(ez_alltrials)
        
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    
    contrast = df_correct.groupby('condition').rt.median()
    dvs['stimulus_interference_rt'] = {'value':  contrast['SDD'] - contrast['SNN'], 'valence': 'Neg'} 
    # DDM equivalents
    if set(['EZ_drift_SDD', 'EZ_drift_SNN']) <= set(dvs.keys()):
        dvs['stimulus_interference_EZ_drift'] = {'value':  dvs['EZ_drift_SDD']['value'] - dvs['EZ_drift_SNN']['value'], 'valence': 'Pos'}
        dvs['stimulus_interference_EZ_thresh'] = {'value':  dvs['EZ_thresh_SDD']['value'] - dvs['EZ_thresh_SNN']['value'], 'valence': 'Pos'}
        dvs['stimulus_interference_EZ_non_decision'] = {'value':  dvs['EZ_non_decision_SDD']['value'] - dvs['EZ_non_decision_SNN']['value'], 'valence': 'NA'}
    param_valence = {'drift': 'Pos', 'thresh': 'Pos', 'non_decision': 'NA'}
    for param in ['drift','thresh','non_decision']:
        if set(['hddm_' + param + '_SDD', 'hddm_' + param + '_SNN']) <= set(dvs.keys()):
            dvs['stimulus_interference_hddm_' + param] = {'value':  dvs['hddm_' + param + '_SDD']['value'] - dvs['hddm_' + param + '_SNN']['value'], 'valence': param_valence[param]}
    
    previous_stim = df.loc[:,['distractor_id','target_id','probe_id']].shift(1)
    # negative priming: when last distractor (and not probe or target) is the current target
    df.loc[:, 'negative_prime_trial'] = [df['target_id'][i] == previous_stim.loc[i,'distractor_id'] 
                                            and df['target_id'][i] not in previous_stim.loc[i, ['target_id', 'probe_id']].values for i in df.index]
    # positive priming: when last target or probe (and not distractor) is the current target
    df.loc[:, 'positive_prime_trial'] = [df['target_id'][i] in previous_stim.loc[i,['target_id', 'probe_id']].values 
                                           and df['target_id'][i] != previous_stim.loc[i,'distractor_id'] for i in df.index]
    # control for priming: when current target wasn't in the last trial
    df.loc[:, 'control_prime_trial'] = [df['target_id'][i] not in previous_stim.loc[i].values for i in df.index]
    
    dvs['negative_priming'] = {'value': df.query('negative_prime_trial').rt.median()-df.query('control_prime_trial').rt.median(), 'valence':'NA'}
    dvs['positive_priming'] = {'value': df.query('positive_prime_trial').rt.median()-df.query('control_prime_trial').rt.median(), 'valence':'NA'}

    df.loc[:,'distractor_lag'] = df['distractor_id'].shift(1)
    df.loc[:, 'prime_trial'] = numpy.where(df['distractor_lag'] == df['target_id'],1,0)
    dvs['prime_trials'] = {'value': df.prime_trial.mean(), 'valence': 'NA'}
    dvs['negative_priming'] = {'value': df.query('prime_trial == 1').rt.median()-df.query('prime_trial == 0').rt.median(), 'valence':'NA'}
  
    description = """standard and negative priming effect, which is the difference 
                    between median response times of trials where the target was the distractor 
                    in the previous trial versus trials where the target was not the distractor 
                    in the previous trial. Note, however, we did not design the task to balance 
                    prime trials so the priming effect is calculated based on different number of trials for subjects.""" 
    return dvs, description


@group_decorate()
def calc_shift_DV(df, dvs = {}):
    """ Calculate dv for shift task. I
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # post error slowing
    post_error_slowing = get_post_error_slow(df)
    
    # subset df
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
        
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt'] = {'value':  df.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    # fit model
    model = fRL_Model(df, decay_weights=True)
    model.optimize()
    fit = numpy.sum(numpy.log(model.run_data()[0]))
    params = model.get_params()
    dvs['model_learning_rate'] = {'value':  params['lr']  , 'valence': 'Pos'}
    dvs['model_decay'] = {'value':  params['decay']  , 'valence': 'Neg'}
    dvs['model_beta'] = {'value':  params['beta']  , 'valence': 'NA'}
    dvs['model_fit'] = {'value':  fit  , 'valence': 'NA'}
    
    try:
        rs = smf.glm('correct ~ trials_since_switch+trial_num', data = df, family = sm.families.Binomial()).fit()
        learning_rate = rs.params['trials_since_switch']
        learning_to_learn = rs.params['trial_num']
        log_ll = rs.llf
        num_trials = df.shape[0]
    except ValueError:
        learning_rate = 'NA'
        learning_to_learn = 'NA'
        log_ll = 'NA'
        num_trials = 'NA'
    
    dvs['learning_to_learn'] = {'value': learning_to_learn, 'valence':'Pos'}
    dvs['learning_rate'] = {'value':  learning_rate  , 'valence': 'Pos'}
    dvs['log_ll'] = {'value':  log_ll  , 'valence': 'NA'}
    dvs['num_trials'] = {'value':  num_trials  , 'valence': 'Pos'}

    #conceptual_responses: The CLR score is the total number of consecutive correct responses in a sequence of 3 or more.
    CLR_score = 0
    CLR_thresh_sum= 0 # must be >= 3
    #fail_to_maintain_set: The FTMS score is the number of sequences of 5 correct responses or more, 
    #followed by an error, before attaining the 10 necessary for a set change - for us just counting number of streaks of >5 since 10 isn't necessary for set change
    FTMS_score = 0
    FTMS_thresh_sum = 0
    
    for i, row in df.iterrows():
        if row.correct==True:
            CLR_thresh_sum += 1
            FTMS_thresh_sum += 1
        else:
            if FTMS_thresh_sum >= 5:
                FTMS_score += 1
            CLR_thresh_sum = 0 
            FTMS_thresh_sum = 0
        if CLR_thresh_sum>=3:
            CLR_score+=1
    
    dvs['conceptual_responses'] = {'value': CLR_score, 'valence':'Pos'}
    dvs['fail_to_maintain_set'] = {'value': FTMS_score, 'valence':'Pos'}
    
    
    #add last_rewarded_feature column by switching the variable to the feature in the row right before a switch and assigning to the column until there is another switch
    df['last_rewarded_feature'] = "NaN"
    last_rewarded_feature = "NaN"
    for i, row in df.iterrows():
        if row.shift_type != 'stay':
            last_rewarded_feature = df.rewarded_feature.iloc[i-1]
        df.last_rewarded_feature.iloc[i] = last_rewarded_feature
        
    #perseverative_responses: length of df where the choice_stim includes the last_rewarded_feature
    perseverative_responses = df[df.apply(lambda row: row.last_rewarded_feature in str(row.choice_stim), axis=1)]
    dvs['perseverative_responses'] = {'value': len(perseverative_responses),'valence':'Neg'}
    #perseverative_errors: length of perseverative_responses df that is subsetted by incorrect responses
    dvs['perseverative_errors'] = {'value': len(perseverative_responses.query("correct == 0")),'valence':'Neg'}
    #total_errors
    dvs['total_errors'] = {'value': len(df.query("correct==0")), 'valence':'Neg'}
    #nonperseverative_errors
    dvs['nonperseverative_errors'] = {'value': len(df.query("correct==0")) - dvs['perseverative_errors']['value'], 'valence': 'Neg'}
    
    description = """
        Shift task has a complicated analysis. Right now just using accuracy and 
        slope of learning after switches (which I'm calling "learning rate")
        """
    return dvs, description
    
@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'simon'})
def calc_simon_DV(df, dvs = {}):
    """ Calculate dv for simon task. Incongruent-Congruent, median RT and Percent Correct
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add columns for congruency sequence effect
    df.insert(0,'condition_shift', df.condition.shift(1))
    df.insert(0, 'correct_shift', df.correct.shift(1))
    
    # post error slowing
    post_error_slowing = get_post_error_slow(df)
    
    # subset df
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df, condition = 'condition'))
    # get DDM across all trials
    ez_alltrials = EZ_diffusion(df)
    dvs.update(ez_alltrials)
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    # Get congruency effects
    rt_contrast = df_correct.groupby('condition').rt.median()
    acc_contrast = df.groupby('condition').correct.mean()
    dvs['simon_rt'] = {'value':  rt_contrast['incongruent']-rt_contrast['congruent'], 'valence': 'Neg'} 
    dvs['simon_acc'] = {'value':  acc_contrast['incongruent']-acc_contrast['congruent'], 'valence': 'Pos'} 
    # DDM equivalents
    if set(['EZ_drift_congruent', 'EZ_drift_incongruent']) <= set(dvs.keys()):
        dvs['simon_EZ_drift'] = {'value':  dvs['EZ_drift_incongruent']['value'] - dvs['EZ_drift_congruent']['value'], 'valence': 'Pos'}
        dvs['simon_EZ_thresh'] = {'value':  dvs['EZ_thresh_incongruent']['value'] - dvs['EZ_thresh_congruent']['value'], 'valence': 'Pos'}
        dvs['simon_EZ_non_decision'] = {'value':  dvs['EZ_non_decision_incongruent']['value'] - dvs['EZ_non_decision_congruent']['value'], 'valence': 'NA'}
    param_valence = {'drift': 'Pos', 'thresh': 'Pos', 'non_decision': 'NA'}
    for param in ['drift','thresh','non_decision']:
        if set(['hddm_' + param + '_congruent', 'hddm_' + param + '_incongruent']) <= set(dvs.keys()):
            dvs['simon_hddm_' + param] = {'value':  dvs['hddm_' + param + '_incongruent']['value'] - dvs['hddm_' + param + '_congruent']['value'], 'valence': param_valence[param]}

    #congruency sequence effect
    congruency_seq_rt = df_correct.query('correct_shift == True').groupby(['condition_shift','condition']).rt.median()
    congruency_seq_acc = df.query('correct_shift == True').groupby(['condition_shift','condition']).correct.mean()
    
    try:
        seq_rt = (congruency_seq_rt['congruent','incongruent'] - congruency_seq_rt['congruent','congruent']) - \
            (congruency_seq_rt['incongruent','incongruent'] - congruency_seq_rt['incongruent','congruent'])
        seq_acc = (congruency_seq_acc['congruent','incongruent'] - congruency_seq_acc['congruent','congruent']) - \
            (congruency_seq_acc['incongruent','incongruent'] - congruency_seq_acc['incongruent','congruent'])
        dvs['congruency_seq_rt'] = {'value':  seq_rt, 'valence': 'NA'} 
        dvs['congruency_seq_acc'] = {'value':  seq_acc, 'valence': 'NA'} 
    except KeyError:
        pass

    dvs['congruent_acc'] = {'value': df.query('condition == "congruent"').correct.mean(), 'valence': 'Pos'}
    dvs['incongruent_acc'] = {'value': df.query('condition == "incongruent"').correct.mean(), 'valence': 'Pos'}
    dvs['congruent_avg_rt'] = {'value': df.query('condition == "congruent"').rt.median(), 'valence': 'Neg'}
    dvs['incongruent_avg_rt'] = {'value': df.query('condition == "incongruent"').rt.median(), 'valence': 'Neg'}
    dvs['congruent_sd_rt'] = {'value': df.query('condition == "congruent"').rt.std(), 'valence': 'NA'}
    dvs['incongruent_sd_rt'] = {'value': df.query('condition == "incongruent"').rt.std(), 'valence': 'NA'}
    
    description = """
        simon effect calculated for accuracy and RT: incongruent-congruent.
        RT measured in ms and median RT is used for comparison.
        """
    return dvs, description
    
@group_decorate()
def calc_simple_RT_DV(df, dvs = {}):
    """ Calculate dv for simple reaction time. Average Reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    dvs['avg_rt'] = {'value':  df['rt'].median(), 'valence': 'Pos'} 
    dvs['std_rt'] = {'value':  df['rt'].std(), 'valence': 'NA'} 
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    description = 'average reaction time'  
    return dvs, description
    
@group_decorate()
def calc_spatial_span_DV(df, dvs = {}):
    """ Calculate dv for spatial span: forward and reverse mean span
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add trial nums
    base = list(range(14))    
    df.insert(0,'trial_num', base * int(len(df)/14))
    
    # subset
    df = df.query('rt >= 0 -1 and trial_num > 3').reset_index(drop = True)
    
    # calculate DVs
    span = df.groupby(['condition'])['num_spaces'].mean()
    dvs['forward_span'] = {'value':  span['forward'], 'valence': 'Pos'} 
    dvs['reverse_span'] = {'value':  span['reverse'], 'valence': 'Pos'} 
    
    description = 'Mean span after dropping the first 4 trials'   
    return dvs, description

@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'stim_selective_stop_signal'})
def calc_stim_selective_stop_signal_DV(df, dvs = {}):
    """ Calculate dv for stop signal task. Common states like rt, correct and
    DDM parameters are calculated on go trials only
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """    
    # post error slowing
    #post_error_slowing = get_post_error_slow(df.query('exp_stage == "test" and SS_trial_type == "go"'))
    
    # subset df to test trials
    df = df.query('exp_stage not in ["practice","NoSS_practice"]').reset_index(drop = True)
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df.query('condition == "go"')))
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['go_acc'] = {'value':  df.query('condition == "go"').correct.mean(), 'valence': 'Pos'} 
    dvs['stop_acc'] = {'value':  df.query('condition == "stop"').correct.mean(), 'valence': 'Pos'} 
    dvs['ignore_acc'] = {'value':  df.query('condition == "ignore"').correct.mean(), 'valence': 'Pos'} 
    
    dvs['go_rt_error'] = {'value':  df.query('correct == False and condition == "go"').rt.median(), 'valence': 'Neg'} 
    dvs['go_rt_std_error'] = {'value':  df.query('correct == False and condition == "go"').rt.std(), 'valence': 'NA'} 
    dvs['go_rt'] = {'value':  df.query('correct == True and condition == "go"').rt.median(), 'valence': 'Neg'} 
    dvs['go_rt_std'] = {'value':  df.query('correct == True and condition == "go"').rt.std(), 'valence': 'NA'} 
    dvs['stop_rt_error'] = {'value':  df.query('stopped == False and condition == "stop"').rt.median(), 'valence': 'Neg'} 
    dvs['stop_rt_error_std'] = {'value':  df.query('stopped == False and condition== "stop"').rt.std(), 'valence': 'NA'} 
    dvs['ignore_rt_error'] = {'value':  df.query('stopped == False and condition == "ignore"').rt.median(), 'valence': 'Neg'} 
    dvs['ignore_rt_error_std'] = {'value':  df.query('stopped == False and condition== "ignore"').rt.std(), 'valence': 'NA'} 
    
    dvs['SS_delay'] = {'value':  df.query('condition == "stop"').SS_delay.mean(), 'valence': 'Pos'} 
    #dvs['post_error_slowing'] = {'value':  post_error_slowing
    
    # Calculate SSRT ignoring ignore trials
    #SSRT
    go_trials = df.query('condition == "go"')
    stop_trials = df.query('condition == "stop"')
    sorted_go = go_trials.query('rt != -1').rt.sort_values(ascending = True)
    prob_stop_failure = (1-stop_trials.stopped.mean())
    corrected = prob_stop_failure/numpy.mean(go_trials.rt!=-1)
    index = corrected*(len(sorted_go)-1)
    index = [floor(index), ceil(index)]
    dvs['SSRT'] = {'value': sorted_go.iloc[index].mean() - stop_trials.SS_delay.mean(), 'valence': 'Neg'}
    
    description = """SSRT is calculated by calculating the percentage of time there are stop failures during
    stop trials. The assumption is that the go process is racing against the stop process and "wins" on the 
    faster proportion of trials. SSRT is thus the go rt at the percentile specified by the failure percentage.
    """
    return dvs, description
    
@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'stop_signal'})
def calc_stop_signal_DV(df, dvs = {}):
    """ Calculate dv for stop signal task. Common states like rt, correct and
    DDM parameters are calculated on go trials only
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """    
    # post error slowing
    #post_error_slowing = get_post_error_slow(df.query('exp_stage == "test" and SS_trial_type == "go"'))
    
    # subset df to test trials
    df = df.query('exp_stage not in ["practice","NoSS_practice"]').reset_index(drop = True)
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df.query('SS_trial_type == "go"')))
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['go_acc'] = {'value':  df.query('SS_trial_type == "go"').correct.mean(), 'valence': 'Pos'} 
    dvs['stop_acc'] = {'value':  df.query('SS_trial_type == "stop"').correct.mean(), 'valence': 'Pos'} 
    
    dvs['go_rt_error'] = {'value':  df.query('correct == False and SS_trial_type == "go"').rt.median(), 'valence': 'Neg'} 
    dvs['go_rt_std_error'] = {'value':  df.query('correct == False and SS_trial_type == "go"').rt.std(), 'valence': 'NA'} 
    dvs['go_rt'] = {'value':  df.query('correct == True and SS_trial_type == "go"').rt.median(), 'valence': 'Neg'} 
    dvs['go_rt_std'] = {'value':  df.query('correct == True and SS_trial_type == "go"').rt.std(), 'valence': 'NA'} 
    dvs['stop_rt_error'] = {'value':  df.query('stopped == False and SS_trial_type == "stop"').rt.median(), 'valence': 'Neg'} 
    dvs['stop_rt_error_std'] = {'value':  df.query('stopped == False and SS_trial_type == "stop"').rt.std(), 'valence': 'NA'} 
    
    dvs['SS_delay'] = {'value':  df.query('SS_trial_type == "stop"').SS_delay.mean(), 'valence': 'Pos'} 
    #dvs['post_error_slowing'] = {'value':  post_error_slowing
    
    if 'condition' in df.columns:
        # Calculate SSRT for both conditions
        for c in df.condition.unique():
            c_df = df[df.condition == c]
            
            #SSRT
            go_trials = c_df.query('SS_trial_type == "go"')
            stop_trials = c_df.query('SS_trial_type == "stop"')
            sorted_go = go_trials.query('rt != -1').rt.sort_values(ascending = True)
            prob_stop_failure = (1-stop_trials.stopped.mean())
            corrected = prob_stop_failure/numpy.mean(go_trials.rt!=-1)
            index = corrected*(len(sorted_go)-1)
            index = [floor(index), ceil(index)]
            dvs['SSRT_' + c] = {'value': sorted_go.iloc[index].mean() - stop_trials.SS_delay.mean(), 'valence': 'Neg'}
        #take average of both conditions SSRT
        dvs['SSRT'] = {'value':  numpy.mean([dvs['SSRT_' + c]['value'] for c in df.condition.unique()]), 'valence': 'Neg'} 
        
        # Condition metrics
        dvs['proactive_slowing'] = {'value':  -df.query('SS_trial_type == "go" and correct == True').groupby('condition').rt.mean().diff()['low'], 'valence': 'Pos'} 
        dvs['proactive_SSRT_speeding'] = {'value':  dvs['SSRT_low']['value'] - dvs['SSRT_high']['value'], 'valence': 'Pos'} 
    else:
        #SSRT
        go_trials = df.query('SS_trial_type == "go"')
        stop_trials = df.query('SS_trial_type == "stop"')
        sorted_go = go_trials.query('rt != -1').rt.sort_values(ascending = True)
        prob_stop_failure = (1-stop_trials.stopped.mean())
        corrected = prob_stop_failure/numpy.mean(go_trials.rt!=-1)
        index = corrected*(len(sorted_go)-1)
        index = [floor(index), ceil(index)]
        dvs['SSRT'] = {'value': sorted_go.iloc[index].mean() - stop_trials.SS_delay.mean(), 'valence': 'Neg'}

    #inhibition_slope
    rs = smf.glm('correct ~ SS_delay', data =  df.query('SS_trial_type == "stop"'), family = sm.families.Binomial()).fit()
    dvs['inhibition_slope'] = {'value':  rs.params['SS_delay']  , 'valence': 'Pos'}
    dvs['log_ll'] = {'value':  rs.llf  , 'valence': 'NA'}
    dvs['num_stop_trials'] = {'value':  df.query('SS_trial_type == "stop"').shape[0]  , 'valence': 'NA'}
    dvs['commission_errors'] = {'value': 1-df.query('SS_trial_type == "stop"').correct.mean(), 'valence':'Neg'}
    dvs['omission_errors'] = {'value': 1-df.query('SS_trial_type == "go"').correct.mean(), 'valence':'Neg'}
    dvs['total_errors'] = {'value': 1-df.correct.mean(), 'valence': 'Neg'}


    description = """SSRT is calculated by calculating the percentage of time there are stop failures during
    stop trials. The assumption is that the go process is racing against the stop process and "wins" on the 
    faster proportion of trials. SSRT is thus the go rt at the percentile specified by the failure percentage.
    
    Here we correct the failure percentage by omission rate. There are also two conditions in this task where stop signal
    probability is either 40% (high) or 20% (low). Overall SSRT is the average of the two. Proactive slowing is measured
    by comparing go RT between the two conditions (high-low). Proactive SSRT speeding does the same but low-high. If the
    subject is sensitive to the task statistics both quantities should increase.
    """
    return dvs, description

@group_decorate(group_fun_getter = get_HDDM_fun, group_fun_args={'task': 'stroop'})
def calc_stroop_DV(df, dvs = {}):
    """ Calculate dv for stroop task. Incongruent-Congruent, median RT and Percent Correct
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add columns for congruency sequence effect
    df.insert(0,'condition_shift', df.condition.shift(1))
    df.insert(0, 'correct_shift', df.correct.shift(1))
    
    # post error slowing
    post_error_slowing = get_post_error_slow(df)
    
    # subset df
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    dvs.update(EZ_diffusion(df, condition = 'condition'))
    # get DDM across all trials
    ez_alltrials = EZ_diffusion(df)
    dvs.update(ez_alltrials)
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    # Get congruency effects
    rt_contrast = df_correct.groupby('condition').rt.median()
    acc_contrast = df.groupby('condition').correct.mean()
    dvs['stroop_rt'] = {'value':  rt_contrast['incongruent']-rt_contrast['congruent'], 'valence': 'Neg'} 
    dvs['stroop_acc'] = {'value':  acc_contrast['incongruent']-acc_contrast['congruent'], 'valence': 'Pos'} 
    # DDM equivalents
    if set(['EZ_drift_congruent', 'EZ_drift_incongruent']) <= set(dvs.keys()):
        dvs['stroop_EZ_drift'] = {'value':  dvs['EZ_drift_incongruent']['value'] - dvs['EZ_drift_congruent']['value'], 'valence': 'Pos'}
        dvs['stroop_EZ_thresh'] = {'value':  dvs['EZ_thresh_incongruent']['value'] - dvs['EZ_thresh_congruent']['value'], 'valence': 'Pos'}
        dvs['stroop_EZ_non_decision'] = {'value':  dvs['EZ_non_decision_incongruent']['value'] - dvs['EZ_non_decision_congruent']['value'], 'valence': 'NA'}
    param_valence = {'drift': 'Pos', 'thresh': 'Pos', 'non_decision': 'NA'}
    for param in ['drift','thresh','non_decision']:
        if set(['hddm_' + param + '_congruent', 'hddm_' + param + '_incongruent']) <= set(dvs.keys()):
            dvs['stroop_hddm_' + param] = {'value':  dvs['hddm_' + param + '_incongruent']['value'] - dvs['hddm_' + param + '_congruent']['value'], 'valence': param_valence[param]}
 
    #congruency sequence effect
    congruency_seq_rt = df_correct.query('correct_shift == True').groupby(['condition_shift','condition']).rt.median()
    congruency_seq_acc = df.query('correct_shift == True').groupby(['condition_shift','condition']).correct.mean()
    
    try:
        seq_rt = (congruency_seq_rt['congruent','incongruent'] - congruency_seq_rt['congruent','congruent']) - \
            (congruency_seq_rt['incongruent','incongruent'] - congruency_seq_rt['incongruent','congruent'])
        seq_acc = (congruency_seq_acc['congruent','incongruent'] - congruency_seq_acc['congruent','congruent']) - \
            (congruency_seq_acc['incongruent','incongruent'] - congruency_seq_acc['incongruent','congruent'])
        dvs['congruency_seq_rt'] = {'value':  seq_rt, 'valence': 'NA'} 
        dvs['congruency_seq_acc'] = {'value':  seq_acc, 'valence': 'NA'} 
    except KeyError:
        pass
    
    dvs['errors'] = {'value': len(df.query('correct == 0')), 'valence': 'Neg'}
    dvs['congruent_errors'] = {'value': len(df.query('correct == 0 & condition == "congruent"')), 'valence': 'Neg'}
    dvs['incongruent_errors'] = {'value': len(df.query('correct == 0 & condition == "incongruent"')), 'valence': 'Neg'}
    dvs['congruent_rt'] = {'value': df.query('condition == "congruent"').rt.median(), 'valence': 'Neg'}
    dvs['incongruent_rt'] = {'value': df.query('condition == "incongruent"').rt.median(), 'valence': 'Neg'}
    
    description = """
        stroop effect calculated for accuracy and RT: incongruent-congruent.
        RT measured in ms and median RT is used for comparison.
        """
    return dvs, description

@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'threebytwo'})
def calc_threebytwo_DV(df, dvs = {}):
    """ Calculate dv for 3 by 2 task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add columns for shift effect
    df.insert(0,'correct_shift', df.correct.shift(1))
    df.insert(0, 'task_switch_shift', df.task_switch.shift(1))

    # post error slowing
    post_error_slowing = get_post_error_slow(df)
    
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True and correct_shift == True').reset_index(drop = True)
    # make dataframe for EZ_DDM comparisons
    df_EZ = df.query('correct_shift == 1 and rt > .01')
    # convert reaction time to seconds to match with HDDM
    df_EZ = df_EZ.rename(columns = {'rt':'old_rt'})
    df_EZ['rt'] = df_EZ['old_rt']/1000
    # get DDM across all trials
    ez_alltrials = EZ_diffusion(df)
    dvs.update(ez_alltrials)
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    # calculate task set inhibition (difference between CBC and ABC)
    selection = ['switch' in x.task_switch and 'stay' not in x.task_switch_shift for i,x in df_correct.iterrows()]
    task_inhibition_contrast =  df_correct[selection].groupby(['CTI','task','task_switch']).rt.median().diff()

    #switch costs
    for CTI in df.CTI.unique():
        dvs['task_inhibition_%s' % CTI] = {'value':  task_inhibition_contrast.reset_index().query('task_switch == "switch_old" and CTI == %s' % CTI).mean().rt, 'valence': 'Neg'} 
        CTI_df = df_correct.query('CTI == %s' % CTI)
        CTI_df_EZ = df_EZ.query('CTI == %s' % CTI)
        dvs['cue_switch_cost_rt_%s' % CTI] = {'value':  CTI_df.groupby('cue_switch')['rt'].median().diff()['switch'], 'valence': 'Neg'} 
        task_switch_rt = CTI_df.groupby(CTI_df['task_switch'].map(lambda x: 'switch' in x)).rt.median()[True]
        cue_switch_rt = CTI_df.groupby('cue_switch').rt.median()['switch']
        dvs['task_switch_cost_rt_%s' % CTI] = {'value':  task_switch_rt - cue_switch_rt, 'valence': 'Neg'}
        
        # DDM equivalents
        
        # calculate EZ_diffusion for cue switchin and task switching
        # cue switch
        for c in ['switch', 'stay']:
            try:
                subset = CTI_df_EZ[CTI_df_EZ['cue_switch'] == c]
                pc = subset['correct'].mean()
                # edge case correct
                if pc == 1:
                    pc = 1-(1.0/(2*len(subset)))
                vrt = numpy.var(subset.query('correct == True')['rt'])
                mrt = numpy.mean(subset.query('correct == True')['rt'])
                drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
                dvs['EZ_drift_cue_' + c + '_%s' % CTI] = {'value': drift, 'valence': 'Pos'}
                dvs['EZ_thresh_cue_' + c + '_%s' % CTI] = {'value': thresh, 'valence': 'NA'}
                dvs['EZ_non_decision_cue_' + c + '_%s' % CTI] = {'value': non_dec, 'valence': 'Neg'}
            except ValueError:
                continue
            
        # task switch
        for c in [['switch_old','switch_new']]:
            try:
                subset = CTI_df_EZ[CTI_df_EZ['task_switch'].isin(c)]
                pc = subset['correct'].mean()
                # edge case correct
                if pc == 1:
                    pc = 1-(1.0/(2*len(subset)))
                vrt = numpy.var(subset.query('correct == True')['rt'])
                mrt = numpy.mean(subset.query('correct == True')['rt'])
                drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
                dvs['EZ_drift_task_' + c[0].split('_')[0] + '_%s' % CTI] = {'value': drift, 'valence': 'Pos'}
                dvs['EZ_thresh_task_' + c[0].split('_')[0] + '_%s' % CTI] = {'value': thresh, 'valence': 'NA'}
                dvs['EZ_non_decision_task_' + c[0].split('_')[0] + '_%s' % CTI] = {'value': non_dec, 'valence': 'Neg'}
            except ValueError:
                continue
            
        param_valence = {'drift': 'Pos', 'thresh': 'Pos', 'non_decision': 'NA'}
        for param in ['drift','thresh','non_decision']:
            if set(['EZ_' + param + '_cue_switch'  + '_%s' % CTI, 'EZ_' + param + '_cue_stay' + '_%s' % CTI]) <= set(dvs.keys()):
                dvs['cue_switch_cost_EZ_' + param + '_%s' % CTI] = {'value':  dvs['EZ_' + param + '_cue_switch' + '_%s' % CTI]['value'] - dvs['EZ_' + param + '_cue_stay' + '_%s' % CTI]['value'], 'valence': param_valence[param]}
                if set(['EZ_' + param + '_task_switch' + '_%s' % CTI, 'EZ_' + param + '_cue_switch' + '_%s' % CTI]) <= set(dvs.keys()):
                    dvs['task_switch_cost_EZ_' + param + '_%s' % CTI] = {'value':  dvs['EZ_' + param + '_task_switch' + '_%s' % CTI]['value'] - dvs['EZ_' + param + '_cue_switch' + '_%s' % CTI]['value'], 'valence': param_valence[param]}
        for param in ['drift','thresh','non_decision']:
            if set(['hddm_' + param + '_cue_switch', 'hddm_' + param + '_cue_stay']) <= set(dvs.keys()):
                dvs['cue_switch_cost_hddm_' + param] = {'value':  dvs['hddm_' + param + '_cue_switch']['value'] - dvs['hddm_' + param + '_cue_stay']['value'], 'valence': param_valence[param]}
                if set([ 'hddm_' + param + '_task_switch', 'hddm_' + param + '_cue_switch']) <= set(dvs.keys()):
                    dvs['task_switch_cost_hddm_' + param] = {'value':  dvs['hddm_' + param + '_task_switch']['value'] - dvs['hddm_' + param + '_cue_switch']['value'], 'valence': param_valence[param]}
             
    description = """ Task switch cost defined as rt difference between task "stay" trials
    and both task "switch_new" and "switch_old" trials. Cue Switch cost is defined only on 
    task stay trials. Inhibition of return is defined as the difference in reaction time between
    task "switch_old" and task "switch_new" trials (ABC vs CBC). The value is the mean over the three tasks. 
    Positive values indicate higher RTs (cost) for
    task switches, cue switches and switch_old
    """
    return dvs, description

@group_decorate(group_fun_getter=get_HDDM_fun, group_fun_args={'task': 'twobytwo'})
def calc_twobytwo_DV(df, dvs = {}):
    """ Calculate dv for 2 by 2 task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add columns for shift effect
    df.insert(0,'correct_shift', df.correct.shift(1))
    df.insert(0, 'task_switch_shift', df.task_switch.shift(1))

    # post error slowing
    post_error_slowing = get_post_error_slow(df)
    
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True and correct_shift == True').reset_index(drop = True)
    # make dataframe for EZ_DDM comparisons
    df_EZ = df.query('correct_shift == 1 and rt > .01')
    # convert reaction time to seconds to match with HDDM
    df_EZ = df_EZ.rename(columns = {'rt':'old_rt'})
    df_EZ['rt'] = df_EZ['old_rt']/1000
    # get DDM across all trials
    ez_alltrials = EZ_diffusion(df)
    dvs.update(ez_alltrials)
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    #switch costs
    for CTI in df.CTI.unique():
        CTI_df = df_correct.query('CTI == %s' % CTI)
        CTI_df_EZ = df_EZ.query('CTI == %s' % CTI)
        dvs['cue_switch_cost_rt_%s' % CTI] = {'value':  CTI_df.groupby('cue_switch')['rt'].median().diff()['switch'], 'valence': 'Neg'} 
        task_switch_rt = CTI_df.groupby(CTI_df['task_switch'].map(lambda x: 'switch' in x)).rt.median()[True]
        cue_switch_rt = CTI_df.groupby('cue_switch').rt.median()['switch']
        dvs['task_switch_cost_rt_%s' % CTI] = {'value':  task_switch_rt - cue_switch_rt, 'valence': 'Neg'} 
        
        # DDM equivalents
        
        # calculate EZ_diffusion for cue switchin and task switching
        # cue switch
        for c in ['switch', 'stay']:
            try:
                subset = CTI_df_EZ[CTI_df_EZ['cue_switch'] == c]
                pc = subset['correct'].mean()
                # edge case correct
                if pc == 1:
                    pc = 1-(1.0/(2*len(subset)))
                vrt = numpy.var(subset.query('correct == True')['rt'])
                mrt = numpy.mean(subset.query('correct == True')['rt'])
                drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
                dvs['EZ_drift_cue_' + c + '_%s' % CTI] = {'value': drift, 'valence': 'Pos'}
                dvs['EZ_thresh_cue_' + c + '_%s' % CTI] = {'value': thresh, 'valence': 'NA'}
                dvs['EZ_non_decision_cue_' + c + '_%s' % CTI] = {'value': non_dec, 'valence': 'Neg'}
            except ValueError:
                continue
            
        # task switch
        for c in ['switch']:
            try:
                subset = CTI_df_EZ[CTI_df_EZ['task_switch'] == c]
                pc = subset['correct'].mean()
                # edge case correct
                if pc == 1:
                    pc = 1-(1.0/(2*len(subset)))
                vrt = numpy.var(subset.query('correct == True')['rt'])
                mrt = numpy.mean(subset.query('correct == True')['rt'])
                drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
                dvs['EZ_drift_task_' + c + '_%s' % CTI] = {'value': drift, 'valence': 'Pos'}
                dvs['EZ_thresh_task_' + c + '_%s' % CTI] = {'value': thresh, 'valence': 'NA'}
                dvs['EZ_non_decision_task_' + c + '_%s' % CTI] = {'value': non_dec, 'valence': 'Neg'}
            except ValueError:
                continue
            
        param_valence = {'drift': 'Pos', 'thresh': 'Pos', 'non_decision': 'NA'}
        for param in ['drift','thresh','non_decision']:
            if set(['EZ_' + param + '_cue_switch'  + '_%s' % CTI, 'EZ_' + param + '_cue_stay' + '_%s' % CTI]) <= set(dvs.keys()):
                dvs['cue_switch_cost_EZ_' + param + '_%s' % CTI] = {'value':  dvs['EZ_' + param + '_cue_switch' + '_%s' % CTI]['value'] - dvs['EZ_' + param + '_cue_stay' + '_%s' % CTI]['value'], 'valence': param_valence[param]}
                if set(['EZ_' + param + '_task_switch' + '_%s' % CTI, 'EZ_' + param + '_task_stay' + '_%s' % CTI]) <= set(dvs.keys()):
                    dvs['task_switch_cost_EZ_' + param + '_%s' % CTI] = {'value':  dvs['EZ_' + param + '_task_switch' + '_%s' % CTI]['value'] - dvs['EZ_' + param + '_cue_switch' + '_%s' % CTI]['value'], 'valence': param_valence[param]}
        for param in ['drift','thresh','non_decision']:
            if set(['hddm_' + param + '_cue_switch', 'hddm_' + param + '_cue_stay']) <= set(dvs.keys()):
                dvs['cue_switch_cost_hddm_' + param] = {'value':  dvs['hddm_' + param + '_cue_switch']['value'] - dvs['hddm_' + param + '_cue_stay']['value'], 'valence': param_valence[param]}
                if set([ 'hddm_' + param + '_task_switch', 'hddm_' + param + '_task_stay']) <= set(dvs.keys()):
                    dvs['task_switch_cost_hddm_' + param] = {'value':  dvs['hddm_' + param + '_task_switch']['value'] - dvs['hddm_' + param + '_cue_switch']['value'], 'valence': param_valence[param]}
             
    description = """ Task switch cost defined as rt difference between task "stay" trials
    and both task "switch_new" and "switch_old" trials. Cue Switch cost is defined only on 
    task stay trials. Inhibition of return is defined as the difference in reaction time between
    task "switch_old" and task "switch_new" trials (ABC vs CBC). The value is the mean over the three tasks. 
    Positive values indicate higher RTs (cost) for
    task switches, cue switches and switch_old
    """
    return dvs, description
    
@group_decorate()
def calc_TOL_DV(df, dvs = {}):
    feedback_df = df.query('trial_id == "feedback"')
    analysis_df = pandas.DataFrame(index = range(df.problem_id.max()+1))
    analysis_df.loc[:,'optimal_solution'] = feedback_df.apply(lambda x: x['num_moves_made'] == x['min_moves'] and x['correct'] == True, axis = 1).tolist()
    analysis_df.loc[:,'planning_time'] = df.query('num_moves_made == 1 and trial_id == "to_hand"').groupby('problem_id').rt.mean()
    analysis_df.loc[:,'solution_time'] = df.query('trial_id in ["to_hand", "to_board"]').groupby('problem_id').rt.sum() - analysis_df.loc[:,'planning_time']
    analysis_df.loc[:,'num_moves_made'] = df.groupby('problem_id').num_moves_made.max()  
    analysis_df.loc[:,'extra_moves'] = analysis_df.loc[:,'num_moves_made'] - feedback_df.min_moves.tolist()
    analysis_df.loc[:,'avg_move_time'] = analysis_df['solution_time']/(analysis_df['num_moves_made']-1)
    analysis_df.loc[:,'correct_solution'] = feedback_df.correct.tolist()
    
    dvs['num_correct'] = {'value':  analysis_df['correct_solution'].sum(), 'valence': 'Pos'} 
    # When they got it correct, did they make the minimum number of moves?
    dvs['num_optimal_solutions'] = {'value':   analysis_df['optimal_solution'].sum(), 'valence': 'Pos'}
    # how many extra moves did they take over the entire task?
    dvs['num_extra_moves'] = {'value':   analysis_df['extra_moves'].sum(), 'valence': 'Neg'}
    # how long did it take to make the first move?    
    dvs['planning_time'] = {'value':  analysis_df['planning_time'].mean(), 'valence': 'NA'} 
    # how long did it take on average to take an action on correct trials
    dvs['avg_move_time'] = {'value':  analysis_df.query('correct_solution == True')['avg_move_time'].mean(), 'valence': 'NA'} 
    # how many moves were made on optimally completed solutiongs
    dvs['weighted_performance_score'] = {'value':  analysis_df.query('optimal_solution == True').num_moves_made.sum(), 'valence': 'Pos'} 
    description = 'many dependent variables related to tower of london performance'
    return dvs, description

def get_twostage_glm(**kwargs):
    def two_stage_glm(data):
        data = data.copy()
        data = data.query('trial_id == "complete_trial" and feedback_last != -1').reset_index(drop = True)
        data.loc[:,'stay'] = 1-data.switch.astype(int)
        data.loc[:, 'stage_transition_last'] = pandas.Categorical(data.stage_transition_last, categories = ['infrequent','frequent'])
        data = data.loc[:,['worker_id','stay','stage_transition_last','feedback_last']].dropna()
        formula = "stay ~ feedback_last*stage_transition_last + (feedback_last*stage_transition_last|worker_id)"
        fixed,random = glmer(data,formula)
        # create group dv object
        group_dvs = {}
        for i,worker in enumerate(data.worker_id.unique()):
            dvs = {}
            dvs['perseverance'] = {'value': random.iloc[i,0], 'valence': 'Neg'}
            dvs['model_free'] = {'value': random.iloc[i,1], 'valence': 'NA'}
            dvs['model_based'] = {'value': random.iloc[i,3], 'valence': 'Pos'}
            group_dvs[worker]  = dvs
        return group_dvs
    return two_stage_glm

@group_decorate(group_fun_getter = get_twostage_glm)
def calc_two_stage_decision_DV(df, dvs = {}):
    """ Calculate dv for choice reaction time: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df['trial_id']=="incomplete_trial").mean()
    df = df.query('trial_id == "complete_trial" and feedback_last != -1').reset_index(drop = True)
    dvs['avg_rt'] = {'value':  numpy.mean(df[['rt_first','rt_second']].mean()), 'valence': 'Neg'} 
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'} 
    
    description = 'standard'  
    return dvs, description

@group_decorate()
def calc_WATT_DV(df, dvs = {}):
    #Kaller, C. P., Rahm, B., Spreer, J., Weiller, C., & Unterrainer, J. M. (2011). Dissociable contributions of left and right dorsolateral prefrontal cortex in planning. Cerebral Cortex, 21(2), 307–317. http://doi.org/10.1093/cercor/bhq096
    feedback_df = df.query('trial_id == "feedback"')
    analysis_df = pandas.DataFrame(index = range(int(df.problem_id.max()+1)))
    analysis_df.loc[:,'condition'] = feedback_df.condition.tolist()
    # planning time defined as time to first action
    analysis_df.loc[:,'planning_time'] = df.query('num_moves_made == 1 and \
                                                          trial_id == "to_hand"').groupby('problem_id').rt.mean()
    # movement time defined as the total time to make all actions minus planning time
    analysis_df.loc[:,'movement_time'] = df.query('trial_id in ["to_hand", "to_board"]') \
                                                            .groupby('problem_id').rt.sum() - analysis_df.loc[:,'planning_time']
    # accuracy defined as whether the solution was optimal 
    analysis_df.loc[:,'correct'] = (feedback_df.num_moves_made==feedback_df.min_moves).tolist()
    
    contrast_df = analysis_df.groupby('condition').mean()                                                        
    # how long did it take to make the first move?    
    dvs['PA_with_planning_time'] = {'value':  contrast_df.loc['PA_with_intermediate','planning_time'], 'valence': 'NA'} 
    dvs['PA_without_planning_time'] = {'value':  contrast_df.loc['PA_without_intermediate','planning_time'], 'valence': 'NA'} 
    # how long did it take on average to take an action on correct trials
    dvs['PA_with_movement_time'] = {'value':  contrast_df.loc['PA_with_intermediate','movement_time'], 'valence': 'NA'} 
    dvs['PA_without_movement_time'] = {'value':  contrast_df.loc['PA_without_intermediate','movement_time'], 'valence': 'NA'} 
    # accuracy (optimal number of moves)
    dvs['PA_with_acc'] = {'value':  contrast_df.loc['PA_with_intermediate','correct'], 'valence': 'NA'} 
    dvs['PA_without_acc'] = {'value':  contrast_df.loc['PA_without_intermediate','correct'], 'valence': 'NA'} 
    
    description = '''
                many dependent variables related to watt performance in two 
                conditions: Partially Ambiguous with an intermediate move and
                without an intermediate move'''
    return dvs, description
 
@group_decorate()
def calc_writing_DV(df, dvs = {}):
    """ Calculate dv for writing task using basic text statistics
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('trial_id == "write"')
    txt = df.final_text.iloc[0]
    
    # sentiment analysis
    try:
        res = requests.post('http://text-processing.com/api/sentiment/', data = {'text': txt}).json()   
        dvs['sentiment_label'] = {'value': res['label'], 'valence': 'NA'}
        dvs['positive_probability'] = {'value': res['probability']['pos'], 'valence': 'Pos'}
        dvs['neutral_probability'] = {'value': res['probability']['neutral'], 'valence': 'NA'}
    except requests.ConnectionError:
        print('Could not establish connection to complete NLP analysis')
    # key stroke analysis    
    # to be filled in...
    
    description = 'Sentiment analysis accessed here: http://text-processing.com/docs/sentiment.html'  
    return dvs, description

