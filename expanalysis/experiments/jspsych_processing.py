"""
analysis/experiments/jspsych_processing.py: part of expfactory package
functions for automatically cleaning and manipulating jspsych experiments
"""
import re
import pandas
import numpy
import hddm
from scipy.stats import binom
import statsmodels.formula.api as smf
import statsmodels.api as sm
import json
from math import ceil, factorial, floor

"""
Generic Functions
"""

def EZ_diffusion(df):
    assert 'correct' in df.columns, 'Could not calculate EZ DDM'
    df = df.copy()
    # convert reaction time to seconds to match with HDDM
    df['rt'] = df['rt']/1000
    # ensure there are no missed responses or extremely short responses (to fit with EZ)
    df = df.query('rt > .01')
    
    # calculate EZ params
    pc = df['correct'].mean()
    vrt = numpy.var(df.query('correct == True')['rt'])
    mrt = numpy.mean(df.query('correct == True')['rt'])
    drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
    EZ_dvs = {}
    EZ_dvs['EZ_drift'] = {'value': drift, 'valence': 'Pos'}
    EZ_dvs['EZ_thresh'] = {'value': thresh, 'valence': 'NA'}
    EZ_dvs['EZ_non_decision'] = {'value': non_dec, 'valence': 'Neg'}
    return EZ_dvs
    
def fit_HDDM(df, response_col = 'correct', condition = None):
    # set up data
    data = (df.loc[:,'rt']/1000).astype(float).to_frame()
    data.insert(0, 'response', df[response_col].astype(float))
    if condition:
        data.insert(0, 'condition', df[condition])
    # add subject ids 
    data.insert(0,'subj_idx', df['worker_id'])
    subj_ids = data.subj_idx.unique()
    ids = {subj_ids[i]:int(i) for i in range(len(subj_ids))}
    data.replace(subj_ids, [ids[i] for i in subj_ids],inplace = True)
    
    # remove missed responses and extremely short response
    data = data.query('rt > .01')
    
    # run hddm
    if condition:
        m = hddm.HDDM(data, depends_on={'a': 'condition', 'v': 'condition', 't': 'condition'})
    else:
        m = hddm.HDDM(data)
    # find a good starting point which helps with the convergence.
    m.find_starting_values()
    # start drawing 10000 samples and discarding 1000 as burn-in
    m.sample(2500, burn=500)
    
    # extract dvs
    group_dvs = {}
    if not condition:
        thresh = m.nodes_db.loc[m.nodes_db.index.str.contains('a_subj'),'mean']
        drift = m.nodes_db.loc[m.nodes_db.index.str.contains('v_subj'),'mean']
        non_decision = m.nodes_db.loc[m.nodes_db.index.str.contains('t_subj'),'mean']
        
        # create DV variable
        for i,subj in enumerate(subj_ids):
            group_dvs[subj] = {'hddm_thresh': thresh[i], 'hddm_drift': drift[i], 'hddm_non_decision': non_decision[i]}
            
    if condition:
        conditions = data.condition.unique()
        for c in conditions:
            thresh = m.nodes_db.loc[m.nodes_db.index.str.contains('a_subj\(' + c),'mean']
            drift = m.nodes_db.loc[m.nodes_db.index.str.contains('v_subj\(' + c),'mean']
            non_decision = m.nodes_db.loc[m.nodes_db.index.str.contains('t_subj\(' + c),'mean']
            # create DV variable
            for i,subj in enumerate(subj_ids):
                if subj not in group_dvs.keys():
                    group_dvs[subj] = {'hddm_thresh_' + c: {'value': thresh[i], 'valence': 'NA'}, 
                                        'hddm_drift_' + c: {'value': drift[i], 'valence': 'Pos'},
                                        'hddm_non_decision_' + c: {'value': non_decision[i], 'valence': 'Neg'}}
                else:
                    group_dvs[subj].update({'hddm_thresh_' + c: {'value': thresh[i], 'valence': 'NA'}, 
                                            'hddm_drift_' + c: {'value': drift[i], 'valence': 'Pos'},
                                            'hddm_non_decision_' + c: {'value': non_decision[i]}, 'valence': 'Neg'})
    return group_dvs

def group_decorate(group_fun = None):
    """ Group decorate is a wrapper for multi_worker_decorate to pass an optional group level
    DV function
    :group_fun: a function to apply to the entire group that returns a dictionary with DVs
    for each subject (i.e. fit_HDDM)
    """
    def multi_worker_decorate(fun):
        """Decorator to ensure that dv functions (i.e. calc_stroop_DV) have only one worker
        :func: function to apply to each worker individuals
        """
        def multi_worker_wrap(group_df, use_check = True, use_group_fun = True):
            exp = group_df.experiment_exp_id.unique()
            group_dvs = {}
            if len(group_df) == 0:
                return group_dvs, ''
            if len(exp) > 1:
                print('Error - More than one experiment found in dataframe. Exps found were: %s' % exp)
                return group_dvs, ''
            # remove workers who haven't passed some check
            if 'passed_check' in group_df.columns and use_check:
                group_df = group_df[group_df['passed_check']]
            # apply group func if it exists
            if group_fun and use_group_fun:
                group_dvs = group_fun(group_df)
            # apply function on individuals
            for worker in pandas.unique(group_df['worker_id']):
                df = group_df.query('worker_id == "%s"' %worker)
                try:
                    worker_dvs, description = fun(df)
                    if worker not in group_dvs.keys():
                        group_dvs[worker] = worker_dvs
                    else:
                        group_dvs[worker].update(worker_dvs)
                except:
                    print('%s DV calculation failed for worker: %s' % (exp[0], worker))
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
        pre_rt = df.ix[i,'rt']
        post_rt = df.ix[j,'rt']
        if pre_rt != -1 and post_rt != -1 and df.ix[i,'correct'] and df.ix[j,'correct']:
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
        if not i:
            click_num = 0
    df.loc[:,'clicks_before_end'] =  num_clicks
    round_over_index = [df.index.get_loc(i) for i in df.query('trial_id == "round_over"').index]
    df.ix[round_over_index,'tournament_bank'] = df.iloc[[i-1 for i in round_over_index]]['tournament_bank'].tolist()
    df.ix[round_over_index,'trip_bank'] = df.iloc[[i-1 for i in round_over_index]]['trip_bank'].tolist()
    
    
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
    df['stim_top'] = df['stim_top'].fillna(df['stim_bottom'].shift(3))
    df['cue'] = df['cue'].fillna(df['cue'].shift(2))
    df.loc[:,'correct'] = df.correct.astype(float)
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

def IST_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    subset = df[(df['trial_id'] == 'choice') & (df['exp_stage'] != 'practice')]
    # Add chosen and total boxes clicked to choice rows and score
    final_choices = subset[['worker_id','exp_stage','color_clicked','trial_num']]
    stim_subset = df[(df['trial_id'] == 'stim') & (df['exp_stage'] != 'practice')]
    try:
        box_clicks = stim_subset.groupby(['worker_id','exp_stage','trial_num'])['color_clicked'].value_counts()
        counts = []
        for i,row in final_choices.iterrows():
            try:
                index = row[['worker_id','exp_stage','trial_num']].tolist()
                chosen_count = box_clicks[index[0], index[1], index[2]].get(row['color_clicked'],0)
                counts.append(chosen_count)
            except KeyError:
                counts.append(0)
        df.insert(0,'chosen_boxes_clicked',pandas.Series(index = final_choices.index, data = counts))
        df.insert(0,'clicks_before_choice', pandas.Series(index = final_choices.index, data =  subset['which_click_in_round']-1))    
        df.insert(0,'points', df['reward'].shift(-1))
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

def local_global_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    conflict = (df['local_shape']==df['global_shape']).apply(lambda x: 'congruent' if x else 'incongruent')
    neutral = (df['local_shape'].isin(['o']) | df['global_shape'].isin(['o']))
    df.loc[conflict.index, 'conflict_condition'] = conflict
    df.loc[neutral,'conflict_condition'] = 'neutral'
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
    df.insert(0, 'choice1_rt', pandas.Series(index = subset.index, data = [x[0] for x in subset['rt']]))
    df.insert(0, 'choice2_rt', pandas.Series(index = subset.index, data = [json.loads(x)[1] for x in subset['rt']]) - subset['ISI'])
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

def recent_probes_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    df['stim'] = df['stim'].fillna(df['stim'].shift(2))
    df['stims_1back'] = df['stims_1back'].fillna(df['stims_1back'].shift(2))
    df['stims_2back'] = df['stims_2back'].fillna(df['stims_2back'].shift(2))
    return df

def shape_matching_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    return df
    
def shift_post(df):
    df.loc[:,'choice_stim'] = [json.loads(i) if isinstance(i,str) else numpy.nan for i in df['choice_stim']]
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
    stop_counts = df.query('exp_stage != "practice" and SS_trial_type == "stop"').groupby('worker_id').stopped.sum()
    passed_check = numpy.logical_and(stop_counts <= binom.ppf(.975, n=180, p=.5), stop_counts >= binom.ppf(.025, n=180, p=.5))
    passed_check = passed_check[passed_check]
    df.loc[:, 'passed_check'] = df['worker_id'].map(lambda x: x in passed_check)
    return df  

def stroop_post(df):
    df.loc[:,'correct'] = df['correct'].astype(float)
    return df
    
def threebytwo_post(df):
    df.insert(0, 'CTI', pandas.Series(data = df[df['trial_id'] == "cue"].block_duration.tolist(), \
                                        index = df[df['trial_id'] == "stim"].index))
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
                        row['trial_id'] = 'complete_trial'
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
                    rows.append(row)
            rows.append(worker_df.iloc[-1].to_dict())
            worker_df = pandas.DataFrame(rows)
            trial_index = ["%s_%s_%s" % ('two_stage_decision',worker_i,x) for x in range(len(worker_df))]
            worker_df.index = trial_index
            #manipulation check
            win_stay = 0.0
            subset = worker_df[worker_df['exp_stage']=='test']
            for stage in numpy.unique(subset['stage_second']):
                stage_df=subset[subset['stage_second']==stage][['feedback','stim_selected_second']]
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
    
 
"""
DV functions
"""

@group_decorate(group_fun = fit_HDDM)
def calc_adaptive_n_back_DV(df):
    """ Calculate dv for adaptive_n_back task. Maximum load
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    control_df = df.query('exp_stage == "control"')
    df = df.query('exp_stage == "adaptive"')
    
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df['rt']==-1).mean()
    df = df.query('rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    try:
        dvs = EZ_diffusion(df)
    except ValueError:
        dvs = {}
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    
    block = 0
    count = 0
    recency = []
    start = False
    for y,i in enumerate(df.correct_response):
        if df.iloc[y].block_num != block:
            block = df.iloc[y].block_num
            count = 0
            start = False
        if i==37:
            count = 0
            start = True
        recency.append(count)
        if start:
            count+=1
    df.loc[:,'recency'] = recency
    rs = smf.ols(formula = 'rt ~ recency', data = df.query('recency > 0')).fit()
    
    dvs['mean_load'] = {'value':  df.groupby('block_num').load.mean().mean(), 'valence': 'Pos'}
    dvs['proactive_interference'] = {'value':  rs.params['recency'], 'valence': 'Neg'}
    description = 'max load'
    return dvs, description
 
@group_decorate(group_fun = fit_HDDM)
def calc_ANT_DV(df):
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
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True')
    
    # Get DDM parameters
    try:
        dvs = EZ_diffusion(df)
    except ValueError:
        dvs = {}
    
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
    dvs['alerting_acc'] = {'value':  (cue_acc.loc['nocue'] - cue_acc.loc['double']), 'valence': 'NA'}
    dvs['orienting_acc'] = {'value':  (cue_acc.loc['center'] - cue_acc.loc['spatial']), 'valence': 'NA'}
    dvs['conflict_acc'] = {'value':  (flanker_acc.loc['incongruent'] - flanker_acc.loc['congruent']), 'valence': 'Pos'}
    
    #congruency sequence effect
    congruency_seq_rt = df_correct.query('correct_shift == True').groupby(['flanker_shift','flanker_type']).rt.median()
    congruency_seq_acc = df.query('correct_shift == True').groupby(['flanker_shift','flanker_type']).correct.mean()
    
    seq_rt = (congruency_seq_rt['congruent','incongruent'] - congruency_seq_rt['congruent','congruent']) - \
        (congruency_seq_rt['incongruent','incongruent'] - congruency_seq_rt['incongruent','congruent'])
    seq_acc = (congruency_seq_acc['congruent','incongruent'] - congruency_seq_acc['congruent','congruent']) - \
        (congruency_seq_acc['incongruent','incongruent'] - congruency_seq_acc['incongruent','congruent'])
    dvs['congruency_seq_rt'] = {'value':  seq_rt, 'valence': 'NA'}
    dvs['congruency_seq_acc'] = {'value':  seq_acc, 'valence': 'NA'}
    
    description = """
    DVs for "alerting", "orienting" and "conflict" attention networks are of primary
    interest for the ANT task, all concerning differences in RT. 
    Alerting is defined as nocue - double cue trials. Positive values
    indicate the benefit of an alerting double cue. Orienting is defined as center - spatial cue trials.
    Positive values indicate the benefit of a spatial cue. Conflict is defined as
    incongruent - congruent flanker trials. Positive values indicate the benefit of
    congruent trials (or the cost of incongruent trials). RT measured in ms and median
    RT are used for all comparisons.
    """
    return dvs, description
    
@group_decorate()
def calc_ART_sunny_DV(df):
    """ Calculate dv for choice reaction time: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice" and trial_id == "round_over"').reset_index(drop = True)
    adjusted_df = df.query('caught_blue == 0')
    dvs = {}
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
def calc_CCT_cold_DV(df):
    """ Calculate dv for ccolumbia card task, cold version
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice"').reset_index(drop = True)
    rs = smf.ols(formula = 'num_cards_chosen ~ gain_amount + loss_amount + num_loss_cards', data = df).fit()
    dvs = {}
    dvs['avg_cards_chosen'] = {'value':  df['num_cards_chosen'].mean(), 'valence': 'NA'}
    dvs['gain_sensitivity'] = {'value':  rs.params['gain_amount'], 'valence': 'Pos'}
    dvs['loss_sensitivity'] = {'value':  rs.params['loss_amount'], 'valence': 'Pos'}
    dvs['probability_sensitivity'] = {'value':  rs.params['num_loss_cards'], 'valence': 'Pos'}
    dvs['information_use'] = {'value':  numpy.sum(rs.pvalues[1:]<.05), 'valence': 'Pos'}
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
def calc_CCT_hot_DV(df):
    """ Calculate dv for ccolumbia card task, cold version
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice" and mouse_click == "collectButton" and round_type == "rigged_win"').reset_index(drop = True)
    subset = df[~df['clicked_on_loss_card'].astype(bool)]
    rs = smf.ols(formula = 'total_cards ~ gain_amount + loss_amount + num_loss_cards', data = subset).fit()
    dvs = {}
    dvs['avg_cards_chosen'] = {'value':  subset['total_cards'].mean(), 'valence': 'NA'}
    dvs['gain_sensitivity'] = {'value':  rs.params['gain_amount'], 'valence': 'Pos'}
    dvs['loss_sensitivity'] = {'value':  rs.params['loss_amount'], 'valence': 'Pos'}
    dvs['probability_sensitivity'] = {'value':  rs.params['num_loss_cards'], 'valence': 'Pos'}
    dvs['information_use'] = {'value':  numpy.sum(rs.pvalues[1:]<.05), 'valence': 'Pos'}
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


@group_decorate(group_fun = fit_HDDM)
def calc_choice_reaction_time_DV(df):
    """ Calculate dv for choice reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    try:
        dvs = EZ_diffusion(df)
    except ValueError:
        dvs = {}
    
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
def calc_cognitive_reflection_DV(df):
    dvs = {}
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'} 
    dvs['intuitive_proportion'] = {'value':  df.responded_intuitively.mean(), 'valence': 'Neg'}

    description = 'how many questions were answered correctly (acc) or were mislead by the obvious lure (intuitive proportion'
    return dvs,description

@group_decorate()
def calc_dietary_decision_DV(df):
    """ Calculate dv for dietary decision task. Calculate the effect of taste and
    health rating on choice
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df[~ pandas.isnull(df['taste_diff'])].reset_index(drop = True)
    rs = smf.ols(formula = 'coded_response ~ health_diff + taste_diff', data = df).fit()
    dvs = {}
    dvs['health_sensitivity'] = {'value':  rs.params['health_diff'], 'valence': 'Pos'} 
    dvs['taste_sensitivity'] = {'value':  rs.params['taste_diff'], 'valence': 'NA'} 
    description = """
        Both taste and health sensitivity are calculated based on the decision phase.
        On each trial the participant indicates whether they would prefer a food option
        over a reference food. Their choice is regressed on the subjective health and
        taste difference between that option and the reference item. Positive values
        indicate that the option's higher health/taste relates to choosing the option
        more often
    """
    return dvs,description
    
@group_decorate()
def calc_digit_span_DV(df):
    """ Calculate dv for digit span: forward and reverse span
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add trial nums
    base = list(range(14))    
    df.insert(0,'trial_num', base * int(len(df)/14))
    
    # subset
    df = df.query('exp_stage != "practice" and rt != -1 and trial_num > 3').reset_index(drop = True)
    dvs = {}
    
    # calculate DVs
    span = df.groupby(['condition'])['num_digits'].mean()
    dvs['forward_span'] = {'value':  span['forward'], 'valence': 'Pos'} 
    dvs['reverse_span'] = {'value':  span['reverse'], 'valence': 'Pos'} 
    
    description = 'Mean span after dropping the first 4 trials'  
    return dvs, description

@group_decorate(group_fun = fit_HDDM)
def calc_directed_forgetting_DV(df):
    """ Calculate dv for directed forgetting
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    try:
        dvs = EZ_diffusion(df)
    except ValueError:
        dvs = {}
        
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
    dvs['proactive_inteference_rt'] = {'value':  rt_contrast['neg'] - rt_contrast['con'], 'valence': 'Neg'} 
    dvs['proactive_inteference_acc'] = {'value':  acc_contrast['neg'] - acc_contrast['con'], 'valence': 'Pos'} 
    description = """
    Each DV contrasts trials where subjects were meant to forget the letter vs.
    trials where they had never seen the letter. On both types of trials the
    subject is meant to respond that the letter was not in the memory set. RT
    contrast is only computed for correct trials. Interference for both is calculated as
    Negative - Control trials, so interference_RT should be higher for worse interference and interference_acc
    should be lower for worse interference.
    
    """ 
    return dvs, description

@group_decorate(group_fun = fit_HDDM)
def calc_DPX_DV(df):
    """ Calculate dv for dot pattern expectancy task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    try:
        dvs = EZ_diffusion(df)
    except ValueError:
        dvs = {}
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    dvs["dprime"] = {'value': df.query('condition == "AX"').correct.mean() - (1-df.query('condition == "BX"').correct.mean()), 'valence': 'Pos'}
    
    # context effects
    rt_contrast_df = df_correct.groupby('condition')['rt'].median()
    acc_contrast_df = df.groupby('condition').correct.mean()
    dvs['AY-BY_rt'] = {'value':  rt_contrast_df['AY'] - rt_contrast_df['BY'], 'valence': 'NA'} 
    dvs['BX-BY_rt'] = {'value':  rt_contrast_df['BX'] - rt_contrast_df['BY'], 'valence': 'NA'} 
    dvs['AY-BY_acc'] = {'value': acc_contrast_df['AY'] - acc_contrast_df['BY'], 'valence': 'NA'} 
    dvs['BX-BY_acc'] = {'value': acc_contrast_df['BX'] - acc_contrast_df['BY'], 'valence': 'NA'} 
    
    description = """D' is calculated as hit rate on AX trials - false alarm rate on BX trials (see Henderson et al. 2012).
                    Primary contrasts are AY and BX vs the "control" condition, BY. Proactive control should aid BX condition
                    but harm AY trials.
                """
    return dvs, description


@group_decorate()
def calc_go_nogo_DV(df):
    """ Calculate dv for go-nogo task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice"').reset_index(drop = True)
    dvs = {}
    dvs['overall_acc'] = {'value': df['correct'].mean(), 'valence': 'Pos'} 
    dvs['go_acc'] = {'value': df[df['condition'] == 'go']['correct'].mean(), 'valence': 'Pos'} 
    dvs['nogo_acc'] = {'value': df[df['condition'] == 'nogo']['correct'].mean(), 'valence': 'Pos'} 
    dvs['go_rt'] ={'value':  df[(df['condition'] == 'go') & (df['rt'] != -1)]['rt'].median(), 'valence': 'Pos'} 
    dprime = df.query('condition == "go"').correct.mean() - (1-df.query('condition == "nogo"').correct.mean())
    dvs['dprime'] = {'value': dprime, 'valence': 'Pos'}
    description = """
        Calculated accuracy for go/stop conditions. 75% of trials are go. D_prime is calculated as the P(response|go) - P(response|nogo)
    """
    return dvs, description


@group_decorate()
def calc_hierarchical_rule_DV(df):
    """ Calculate dv for hierarchical learning task. 
    DVs
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # post error slowing
    post_error_slowing = get_post_error_slow(df)
    
    # subset df
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    dvs = {}
    
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
def calc_IST_DV(df):
    """ Calculate dv for information sampling task
    DVs
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice"').reset_index(drop = True)
    dvs = {}
    latency_df = df[df['trial_id'] == "stim"].groupby('exp_stage')['rt'].median()
    points_df = df[df['trial_id'] == "choice"].groupby('exp_stage')['points'].sum()
    contrast_df = df[df['trial_id'] == "choice"].groupby('exp_stage')['correct','P_correct_at_choice','clicks_before_choice'].mean()
    for condition in ['Decreasing Win', 'Fixed Win']:
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
def calc_keep_track_DV(df):
    """ Calculate dv for choice reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    score = df['score'].sum()/df['possible_score'].sum()
    dvs = {}
    dvs['score'] = {'value':  score, 'valence': 'Pos'} 
    description = 'percentage of items remembered correctly'  
    return dvs, description

@group_decorate(group_fun = fit_HDDM)
def calc_local_global_DV(df):
    """ Calculate dv for hierarchical learning task. 
    DVs
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add columns for congruency sequence effect
    df.insert(0,'conflict_condition_shift', df.conflict_condition.shift(1))
    df.insert(0, 'correct_shift', df.correct.shift(1))
    
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    try:
        dvs = EZ_diffusion(df)
    except ValueError:
        dvs = {}
    
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

    dvs['congruent_facilitation_rt'] = {'value':  (rt_contrast['neutral'] - rt_contrast['congruent']), 'valence': 'Pos'} 
    dvs['incongruent_harm_rt'] = {'value':  (rt_contrast['incongruent'] - rt_contrast['neutral']), 'valence': 'Neg'} 
    dvs['congruent_facilitation_acc'] = {'value':  (acc_contrast['congruent'] - acc_contrast['neutral']), 'valence': 'Pos'} 
    dvs['incongruent_harm_acc'] = {'value':  (acc_contrast['neutral'] - acc_contrast['incongruent']), 'valence': 'Neg'} 
    dvs['conflict_rt'] = {'value':  dvs['congruent_facilitation_rt']['value'] + dvs['incongruent_harm_rt']['value'], 'valence': 'Neg'} 
    dvs['conflict_acc'] = {'value':  dvs['congruent_facilitation_acc']['value'] + dvs['incongruent_harm_acc']['value'], 'valence': 'Neg'} 
    
    #congruency sequence effect
    congruency_seq_rt = df_correct.query('correct_shift == True').groupby(['conflict_condition_shift','conflict_condition']).rt.median()
    congruency_seq_acc = df.query('correct_shift == True').groupby(['conflict_condition_shift','conflict_condition']).correct.mean()
    
    seq_rt = (congruency_seq_rt['congruent','incongruent'] - congruency_seq_rt['congruent','congruent']) - \
        (congruency_seq_rt['incongruent','incongruent'] - congruency_seq_rt['incongruent','congruent'])
    seq_acc = (congruency_seq_acc['congruent','incongruent'] - congruency_seq_acc['congruent','congruent']) - \
        (congruency_seq_acc['incongruent','incongruent'] - congruency_seq_acc['incongruent','congruent'])
    dvs['congruency_seq_rt'] = {'value':  seq_rt, 'valence': 'NA'} 
    dvs['congruency_seq_acc'] = {'value':  seq_acc, 'valence': 'NA'} 
    
    # switch costs
    switch_rt = df_correct.query('correct_shift == 1').groupby('switch').rt.median()
    switch_acc = df.query('correct_shift == 1').groupby('switch').correct.mean()
    dvs['switch_cost_rt'] = {'value':  (switch_rt[1] - switch_rt[0]), 'valence': 'Neg'} 
    dvs['switch_cost_acc'] = {'value':  (switch_acc[1] - switch_acc[0]), 'valence': 'Pos'} 

    
    description = """
        local-global incongruency effect calculated for accuracy and RT. 
        Facilitation for RT calculated as neutral-congruent. Positive values indicate speeding on congruent trials.
        Harm for RT calculated as incongruent-neutral. Positive values indicate slowing on incongruent trials
        Facilitation for accuracy calculated as congruent-neutral. Positives values indicate higher accuracy for congruent trials
        Harm for accuracy calculated as neutral - incongruent. Positive values indicate lower accuracy for incongruent trials
        Switch costs calculated as switch-stay for rt and stay-switch for accuracy. Thus positive values indicate slowing and higher
        accuracy on switch trials. Expectation is positive rt switch cost, and negative accuracy switch cost
        RT measured in ms and median RT is used for comparison.
        """
    return dvs, description
    
@group_decorate()
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
    df = df[df['rt'] != -1].reset_index(drop = True)
    
    #Calculate regression DVs
    train = df.query('exp_stage == "training"')
    values = train.groupby('stim_chosen')['feedback'].mean()
    df.loc[:,'value_diff'] = df['condition_collapsed'].apply(lambda x: get_value_diff(x.split('_'), values) if x==x else numpy.nan)
    df.loc[:,'value_sum'] = df['condition_collapsed'].apply(lambda x: get_value_sum(x.split('_'), values) if x==x else numpy.nan)  
    test = df.query('exp_stage == "test"')
    rs = smf.glm(formula = 'correct ~ value_diff*value_sum', data = test, family = sm.families.Binomial()).fit()
    
    #Calculate non-regression, simpler DVs
    pos_subset = test[test['condition_collapsed'].map(lambda x: '20' not in x)]
    neg_subset = test[test['condition_collapsed'].map(lambda x: '80' not in x)]
    chose_A = pos_subset[pos_subset['condition_collapsed'].map(lambda x: '80' in x)]['stim_chosen']=='80'
    chose_C = pos_subset[pos_subset['condition_collapsed'].map(lambda x: '70' in x and '80' not in x and '30' not in x)]['stim_chosen']=='70'
    pos_acc = (numpy.sum(chose_A) + numpy.sum(chose_C))/float((len(chose_A) + len(chose_C)))
    
    avoid_B = neg_subset[neg_subset['condition_collapsed'].map(lambda x: '20' in x)]['stim_chosen']!='20'
    avoid_D = neg_subset[neg_subset['condition_collapsed'].map(lambda x: '30' in x and '20' not in x and '70' not in x)]['stim_chosen']!='30'
    neg_acc = (numpy.sum(avoid_B) + numpy.sum(avoid_D))/float((len(avoid_B) + len(avoid_D)))
    
    #dvs = calc_common_stats(df)
    dvs = {}
    dvs['reg_value_sensitivity'] = {'value':  rs.params['value_diff'], 'valence': 'Pos'} 
    dvs['reg_positive_learning_bias'] = {'value':  rs.params['value_diff:value_sum'], 'valence': 'NA'} 
    dvs['positive_acc'] = {'value':  pos_acc, 'valence': 'Pos'} 
    dvs['negative_acc'] = {'value':  neg_acc, 'valence': 'Pos'} 
    dvs['positive_learning_bias'] = {'value':  pos_acc/neg_acc, 'valence': 'NA'} 
    dvs['overall_test_acc'] = {'value':  test['correct'].mean(), 'valence': 'Pos'} 
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'} 
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
def calc_PRP_two_choices_DV(df):
    """ Calculate dv for shift task. I
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    dvs = {}
    df = df.query('exp_stage != "practice"')
    missed_percent = ((df['choice1_rt']==-1) | (df['choice2_rt'] <= -1)).mean()
    df = df.query('choice1_rt != -1 and choice2_rt > -1').reset_index(drop = True)
    contrast = df.groupby('ISI').choice2_rt.median()
    dvs['PRP_slowing'] = {'value':  contrast.loc[50] - contrast.loc[800], 'valence': 'NA'} 
    rs = smf.ols(formula = 'choice2_rt ~ ISI', data = df).fit()
    dvs['PRP_slope'] = {'value':  -rs.params['ISI'], 'valence': 'NA'} 
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
def calc_ravens_DV(df):
    """ Calculate dv for ravens task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    df = df.query('stim_response == stim_response').reset_index(drop = True)
    dvs = {}
    dvs['score'] = {'value':  df['score_response'].sum(), 'valence': 'Pos'} 
    description = 'Score is the number of correct responses out of 18'
    return dvs,description    

@group_decorate(group_fun = fit_HDDM)
def calc_recent_probes_DV(df):
    """ Calculate dv for recent_probes
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    try:
        dvs = EZ_diffusion(df)
    except ValueError:
        dvs = {}
        
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
    dvs['proactive_inteference_rt'] = {'value':  rt_contrast['rec_neg'] - rt_contrast['xrec_neg'], 'valence': 'Neg'} 
    dvs['proactive_inteference_acc'] = {'value':  acc_contrast['rec_neg'] - acc_contrast['xrec_neg'], 'valence': 'Pos'} 
    description = """
    proactive interference defined as the difference in reaction time and accuracy
    for negative trials (where the probe was not part of the memory set) between
    "recent" trials (where the probe was part of the previous trial's memory set)
    and "non-recent trials" where the probe wasn't.
    """ 
    return dvs, description
    
@group_decorate(group_fun = fit_HDDM)
def calc_shape_matching_DV(df):
    """ Calculate dv for shape_matching task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    try:
        dvs = EZ_diffusion(df)
    except ValueError:
        dvs = {}
        
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    
    contrast = df_correct.groupby('condition').rt.median()
    dvs['stimulus_interference'] = {'value':  contrast['SDD'] - contrast['SNN'], 'valence': 'Neg'} 
    description = 'standard'  
    return dvs, description
    
@group_decorate()
def calc_shift_DV(df):
    """ Calculate dv for shift task. I
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    
    dvs = {}
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt'] = {'value':  df.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    rs = smf.glm('correct ~ trials_since_switch', data = df, family = sm.families.Binomial()).fit()
    dvs['learning_rate'] = {'value':  rs.params['trials_since_switch']  , 'valence': 'Pos'}   
    
    description = """
        Shift task has a complicated analysis. Right now just using accuracy and 
        slope of learning after switches (which I'm calling "learning rate")
        """
    return dvs, description
    
#@group_decorate(group_fun = lambda df: fit_HDDM(df, condition = 'condition'))
@group_decorate(group_fun = fit_HDDM)
def calc_simon_DV(df):
    """ Calculate dv for simon task. Incongruent-Congruent, median RT and Percent Correct
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add columns for congruency sequence effect
    df.insert(0,'condition_shift', df.condition.shift(1))
    df.insert(0, 'correct_shift', df.correct.shift(1))
    
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    try:
        dvs = EZ_diffusion(df)
    except ValueError:
        dvs = {}
    
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
    
    #congruency sequence effect
    congruency_seq_rt = df_correct.query('correct_shift == True').groupby(['condition_shift','condition']).rt.median()
    congruency_seq_acc = df.query('correct_shift == True').groupby(['condition_shift','condition']).correct.mean()
    
    seq_rt = (congruency_seq_rt['congruent','incongruent'] - congruency_seq_rt['congruent','congruent']) - \
        (congruency_seq_rt['incongruent','incongruent'] - congruency_seq_rt['incongruent','congruent'])
    seq_acc = (congruency_seq_acc['congruent','incongruent'] - congruency_seq_acc['congruent','congruent']) - \
        (congruency_seq_acc['incongruent','incongruent'] - congruency_seq_acc['incongruent','congruent'])
    dvs['congruency_seq_rt'] = {'value':  seq_rt, 'valence': 'NA'} 
    dvs['congruency_seq_acc'] = {'value':  seq_acc, 'valence': 'NA'} 
    
    description = """
        simon effect calculated for accuracy and RT: incongruent-congruent.
        RT measured in ms and median RT is used for comparison.
        """
    return dvs, description
    
@group_decorate()
def calc_simple_RT_DV(df):
    """ Calculate dv for simple reaction time. Average Reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    dvs = {}
    dvs['avg_rt'] = {'value':  df['rt'].median(), 'valence': 'Pos'} 
    dvs['std_rt'] = {'value':  df['rt'].std(), 'valence': 'NA'} 
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    description = 'average reaction time'  
    return dvs, description
    
@group_decorate()
def calc_spatial_span_DV(df):
    """ Calculate dv for spatial span: forward and reverse mean span
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add trial nums
    base = list(range(14))    
    df.insert(0,'trial_num', base * int(len(df)/14))
    
    # subset
    df = df.query('exp_stage != "practice" and rt != -1 and trial_num > 3').reset_index(drop = True)
    dvs = {}
    
    # calculate DVs
    span = df.groupby(['condition'])['num_spaces'].mean()
    dvs['forward_span'] = {'value':  span['forward'], 'valence': 'Pos'} 
    dvs['reverse_span'] = {'value':  span['reverse'], 'valence': 'Pos'} 
    
    description = 'Mean span after dropping the first 4 trials'   
    return dvs, description
    
@group_decorate(group_fun = fit_HDDM)
def calc_stroop_DV(df):
    """ Calculate dv for stroop task. Incongruent-Congruent, median RT and Percent Correct
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add columns for congruency sequence effect
    df.insert(0,'condition_shift', df.condition.shift(1))
    df.insert(0, 'correct_shift', df.correct.shift(1))
    
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    # subset df
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True').reset_index(drop = True)
    
    # Get DDM parameters
    try:
        dvs = EZ_diffusion(df)
    except ValueError:
        dvs = {}
    
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
    
    #congruency sequence effect
    congruency_seq_rt = df_correct.query('correct_shift == True').groupby(['condition_shift','condition']).rt.median()
    congruency_seq_acc = df.query('correct_shift == True').groupby(['condition_shift','condition']).correct.mean()
    
    seq_rt = (congruency_seq_rt['congruent','incongruent'] - congruency_seq_rt['congruent','congruent']) - \
        (congruency_seq_rt['incongruent','incongruent'] - congruency_seq_rt['incongruent','congruent'])
    seq_acc = (congruency_seq_acc['congruent','incongruent'] - congruency_seq_acc['congruent','congruent']) - \
        (congruency_seq_acc['incongruent','incongruent'] - congruency_seq_acc['incongruent','congruent'])
    dvs['congruency_seq_rt'] = {'value':  seq_rt, 'valence': 'NA'} 
    dvs['congruency_seq_acc'] = {'value':  seq_acc, 'valence': 'NA'} 
    
    description = """
        stroop effect calculated for accuracy and RT: incongruent-congruent.
        RT measured in ms and median RT is used for comparison.
        """
    return dvs, description

@group_decorate()
def calc_stop_signal_DV(df):
    """ Calculate dv for stop signal task. Common states like rt, correct and
    DDM parameters are calculated on go trials only
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """    
    # post error slowing
    #post_error_slowing = get_post_error_slow(df.query('exp_stage == "test" and SS_trial_type == "go"'))
    
    # subset df
    df = df.query('exp_stage not in ["practice","NoSS_practice"]').reset_index(drop = True)
    
    # Get DDM parameters
    try:
        dvs = EZ_diffusion(df.query('SS_trial_type == "go"'))
    except ValueError:
        dvs = {}
    
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
    
    # Calculate SSRT for both conditions
    for c in df.condition.unique():
        c_df = df[df.condition == c]
        
        #SSRT
        go_trials = c_df.query('SS_trial_type == "go"')
        stop_trials = c_df.query('SS_trial_type == "stop"')
        sorted_go = go_trials.query('rt != -1').rt.sort_values(ascending = False)
        prob_stop_failure = (1-stop_trials.stopped.mean())
        corrected = prob_stop_failure/numpy.mean(go_trials.rt!=-1)
        index = corrected*len(sorted_go)
        index = [floor(index), ceil(index)]
        dvs['SSRT_' + c] = {'value': sorted_go.iloc[index].mean() - stop_trials.SS_delay.mean(), 'valence': 'Neg'}

    dvs['SSRT'] = {'value':  (dvs['SSRT_high']['value'] + dvs['SSRT_low']['value'])/2.0, 'valence': 'Neg'} 
    
    # Condition metrics
    dvs['proactive_slowing'] = {'value':  -df.query('SS_trial_type == "go"').groupby('condition').rt.mean().diff()['low'], 'valence': 'Pos'} 
    dvs['proactive_SSRT_speeding'] = {'value':  dvs['SSRT_low']['value'] - dvs['SSRT_high']['value'], 'valence': 'Pos'} 
    #take average of both conditions SSRT
    

    #motor selective
    description = """SSRT is calculated by calculating the percentage of time there are stop failures during
    stop trials. The assumption is that the go process is racing against the stop process and "wins" on the 
    faster proportion of trials. SSRT is thus the go rt at the percentile specified by the failure percentage.
    
    Here we correct the failure percentage by omission rate. There are also two conditions in this task where stop signal
    probability is either 40% (high) or 20% (low). Overall SSRT is the average of the two. Proactive slowing is measured
    by comparing go RT between the two conditions (high-low). Proactive SSRT speeding does the same but low-high. If the
    subject is sensitive to the task statistics both quantities should increase.
    """
    return dvs, description

@group_decorate(group_fun = fit_HDDM)
def calc_threebytwo_DV(df):
    """ Calculate dv for 3 by 2 task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # add columns for shift effect
    df.insert(0,'correct_shift', df.correct.shift(1))
    
    # post error slowing
    post_error_slowing = get_post_error_slow(df.query('exp_stage == "test"'))
    
    
    missed_percent = (df.query('exp_stage != "practice"')['rt']==-1).mean()
    df = df.query('exp_stage != "practice" and rt != -1').reset_index(drop = True)
    df_correct = df.query('correct == True and correct_shift == True').reset_index(drop = True)
    
    # Get DDM parameters
    try:
        dvs = EZ_diffusion(df)
    except ValueError:
        dvs = {}
    
    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
    dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
    dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
    dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
    dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
    
    #switch costs
    dvs['cue_switch_cost'] = {'value':  df_correct.query('task_switch == "stay"').groupby('cue_switch')['rt'].median().diff()['switch'], 'valence': 'Neg'} 
    task_switch_cost = df_correct.groupby(df_correct['task_switch'].map(lambda x: 'switch' in x)).rt.median().diff()[True]
    dvs['task_switch_cost'] = {'value':  task_switch_cost - dvs['cue_switch_cost']['value'], 'valence': 'Neg'} 
    task_inhibition_contrast =  df_correct[['switch' in x for x in df_correct['task_switch']]].groupby(['task','task_switch']).rt.median().diff()
    dvs['task_inhibition'] = {'value':  task_inhibition_contrast.reset_index().query('task_switch == "switch_old"').mean().rt, 'valence': 'Neg'} 
    
    description = """ Task switch cost defined as rt difference between task "stay" trials
    and both task "switch_new" and "switch_old" trials. Cue Switch cost is defined only on 
    task stay trials. Inhibition of return is defined as the difference in reaction time between
    task "switch_old" and task "switch_new" trials (ABC vs CBC). The value is the mean over the three tasks. 
    Positive values indicate higher RTs (cost) for
    task switches, cue switches and switch_old
    """
    return dvs, description

@group_decorate()
def calc_TOL_DV(df):
    df = df.query('exp_stage == "test"').reset_index(drop = True)
    dvs = {}
    # When they got it correct, did they make the minimum number of moves?
    dvs['num_optimal_solutions'] = {'value':   numpy.sum(df.query('correct == 1')[['num_moves_made','min_moves']].diff(axis = 1)['min_moves']==0), 'valence': 'Pos'} 
    # how long did it take to make the first move?    
    dvs['planning_time'] = {'value':  df.query('num_moves_made == 1 and trial_id == "to_hand"')['rt'].median(), 'valence': 'NA'} 
    # how long did it take on average to take an action    
    dvs['avg_move_time'] = {'value':  df.query('trial_id in ["to_hand", "to_board"]')['rt'].median(), 'valence': 'NA'} 
    # how many moves were made overall
    dvs['total_moves'] = {'value':  numpy.sum(df.groupby('problem_id')['num_moves_made'].max()), 'valence': 'NA'} 
    dvs['num_correct'] = {'value':  numpy.sum(df['correct']==1), 'valence': 'Pos'} 
    description = 'many dependent variables related to tower of london performance'
    return dvs, description
    
@group_decorate()
def calc_two_stage_decision_DV(df):
    """ Calculate dv for choice reaction time: Accuracy and average reaction time
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    missed_percent = (df.query('exp_stage != "practice"')['trial_id']=="incomplete_trial").mean()
    df = df.query('exp_stage != "practice" and trial_id == "complete_trial"').reset_index(drop = True)
    rs = smf.glm(formula = 'switch ~ feedback_last * stage_transition_last', data = df, family = sm.families.Binomial()).fit()
    rs.summary()
    dvs = {}
    dvs['avg_rt'] = {'value':  numpy.mean(df[['rt_first','rt_second']].mean()), 'valence': 'Neg'} 
    dvs['model_free'] = {'value':  rs.params['feedback_last'], 'valence': 'Pos'} 
    dvs['model_based'] = {'value':  rs.params['feedback_last:stage_transition_last[T.infrequent]'], 'valence': 'Pos'} 
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'} 
    description = 'standard'  
    return dvs, description
    