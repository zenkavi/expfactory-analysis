"""
analysis/processing.py: part of expfactory package
functions for automatically cleaning and manipulating experiments by operating
on an expanalysis Result.data dataframe
"""
from copy import deepcopy
from expanalysis.experiments.jspsych_processing import adaptive_nback_post, \
    ANT_post, ART_post, bickel_post, CCT_fmri_post, CCT_hot_post, \
    choice_reaction_time_post, cognitive_reflection_post, \
    conditional_stop_signal_post, dietary_decision_post, \
    directed_forgetting_post, discount_titrate_post, DPX_post, \
    hierarchical_post, holt_laury_post, IST_post, keep_track_post, kirby_post, \
    local_global_post, probabilistic_selection_post, PRP_post, ravens_post, \
    recent_probes_post, shape_matching_post, shift_post, simon_post, \
    span_post,stop_signal_post, stroop_post, TOL_post, threebytwo_post, \
    twobytwo_post, two_stage_decision_post, WATT_post
from expanalysis.experiments.jspsych_processing import calc_adaptive_n_back_DV,\
    calc_ANT_DV, calc_ART_sunny_DV, calc_bickel_DV, calc_CCT_cold_DV, \
    calc_CCT_hot_DV, calc_CCT_fmri_DV, calc_choice_reaction_time_DV, \
    calc_cognitive_reflection_DV, calc_dietary_decision_DV, \
    calc_digit_span_DV, calc_directed_forgetting_DV, calc_discount_fixed_DV, \
    calc_discount_titrate_DV, \
    calc_DPX_DV, calc_go_nogo_DV, calc_hierarchical_rule_DV, \
    calc_holt_laury_DV, calc_IST_DV, calc_keep_track_DV, calc_kirby_DV, \
    calc_local_global_DV, calc_motor_selective_stop_signal_DV, \
    calc_probabilistic_selection_DV, calc_PRP_two_choices_DV, \
    calc_recent_probes_DV, calc_ravens_DV, calc_shape_matching_DV, \
    calc_shift_DV, calc_simon_DV, calc_simple_RT_DV, calc_spatial_span_DV, \
    calc_stop_signal_DV, calc_stim_selective_stop_signal_DV, calc_stroop_DV, \
    calc_threebytwo_DV, calc_twobytwo_DV, calc_TOL_DV, calc_two_stage_decision_DV, \
    calc_WATT_DV, calc_writing_DV
from expanalysis.experiments.survey_processing import \
    calc_survey_DV, calc_bis11_DV, calc_eating_DV, calc_leisure_time_DV, calc_SSS_DV, calc_demographics_DV, \
    self_regulation_survey_post, sensation_seeking_survey_post
from expanalysis.experiments.utils import get_data, lookup_val, select_experiment, drop_null_cols
import pandas
import numpy
import os
import random
import time

#***********************************
# POST PROCESSING
#***********************************
def clean_data(df, exp_id = None, apply_post = True, drop_columns = None, lookup = True):
    '''clean_df returns a pandas dataset after removing a set of default generic
    columns. Optional variable drop_cols allows a different set of columns to be dropped
    :df: a pandas dataframe
    :param experiment: a string identifying the experiment used to automatically drop unnecessary columns. df should not have multiple experiments if this flag is set!
    :param apply_post: bool, if True apply post-processig function retrieved using apply_post
    :param drop_columns: a list of columns to drop. If not specified, a default list will be used from utils.get_dropped_columns()
    :param lookup: bool, default true. If True replaces all values in dataframe using the lookup_val function
    :param return_reject: bool, default false. If true returns a dataframe with rejected experiments
    '''
    if apply_post:
        # apply post processing
        df = post_process_exp(df, exp_id)
    if lookup == True:
        #convert vals based on lookup
        for col in df.columns:
            df.loc[:,col] = df[col].map(lookup_val)
    # Drop unnecessary columns
    if drop_columns == None:
        drop_columns = get_drop_columns()
    df.drop(drop_columns, axis=1, inplace=True, errors='ignore')
    if exp_id != None:
        assert sum(df['experiment_exp_id'] == exp_id) == len(df), \
            "An experiment was specified, but the dataframe has other experiments!"
        drop_rows = get_drop_rows(exp_id)
        # Drop unnecessary rows, all null rows
        for key in drop_rows.keys():
            df = df.query('%s not in  %s' % (key, drop_rows[key]))
    df = df.dropna(how = 'all')
    #drop columns with only null values
    drop_null_cols(df)
    return df



def get_drop_columns():
    return ['view_history', 'trial_index', 'internal_node_id',
           'stim_duration', 'block_duration', 'feedback_duration','timing_post_trial',
           'test_start_block','exp_id']

def get_drop_rows(exp_id):
    '''Function used by clean_df to drop rows from dataframes with one experiment
    :experiment: experiment key used to look up which rows to drop from a dataframe
    '''
    gen_cols = ['welcome', 'text','instruction', 'attention_check','end', 'post task questions', 'fixation', \
                'practice_intro', 'rest', 'rest_block', 'test_intro', 'task_setup', 'test_start_block'] #generic_columns to drop
    lookup = {'adaptive_n_back': {'trial_id': gen_cols + ['update_target', 'update_delay', 'delay_text']},
                'angling_risk_task_always_sunny': {'trial_id': gen_cols + ['test_intro','intro','ask fish','set_fish', 'update_performance_var']},
                'attention_network_task': {'trial_id': gen_cols + ['spatialcue', 'centercue', 'doublecue', 'nocue', 'rest block', 'intro']},
                'bickel_titrator': {'trial_id': gen_cols + ['update_delay', 'update_mag', 'gap']},
                'choice_reaction_time': {'trial_id': gen_cols + ['practice_intro', 'reset trial']},
                'columbia_card_task_cold': {'trial_id': gen_cols + ['calculate reward','reward','end_instructions']},
                'columbia_card_task_hot': {'trial_id': gen_cols + ['calculate reward', 'reward', 'test_intro']},
                'columbia_card_task_fmri': {'trial_id': gen_cols + ['ITI', 'calculate reward', 'reward']},
                'dietary_decision': {'trial_id': gen_cols + ['start_taste', 'start_health']},
                'digit_span': {'trial_id': gen_cols + ['start_reverse', 'stim', 'feedback']},
                'directed_forgetting': {'trial_id': gen_cols + ['ITI_fixation', 'intro_test', 'stim', 'cue', 'instruction_images']},
                'discount_fixed': {'trial_id': gen_cols},
                'dot_pattern_expectancy': {'trial_id': gen_cols + ['instruction_images', 'cue', 'feedback']},
                'go_nogo': {'trial_id': gen_cols + ['reset_trial']},
                'hierarchical_rule': {'trial_id': gen_cols + ['feedback', 'test_intro']},
                'information_sampling_task': {'trial_id': gen_cols + ['DW_intro', 'reset_round']},
                'keep_track': {'trial_id': gen_cols + ['practice_end', 'stim', 'wait', 'prompt']},
                'kirby': {'trial_id': gen_cols + ['prompt', 'wait']},
                'local_global_letter': {'trial_id': gen_cols + []},
                'motor_selective_stop_signal': {'trial_id': gen_cols + ['prompt_fixation', 'feedback']},
                'probabilistic_selection': {'trial_id': gen_cols + ['first_phase_intro', 'second_phase_intro']},
                'psychological_refractory_period_two_choices': {'trial_id': gen_cols + ['feedback']},
                'ravens': {'trial_type': ['poldrack-text', 'poldrack-instructions', 'text']},
                'recent_probes': {'trial_id': gen_cols + ['intro_test', 'ITI_fixation', 'stim']},
                'shift_task': {'trial_id': gen_cols + ['alert', 'feedback', 'reset_trial_count']},
                'simon':{'trial_id': gen_cols + ['reset_trial']},
                'simple_reaction_time': {'trial_id': gen_cols + ['reset_trial', 'gap-message']},
                'shape_matching': {'trial_id': gen_cols + ['mask']},
                'spatial_span': {'trial_id': gen_cols + ['start_reverse_intro', 'stim', 'feedback']},
                'stim_selective_stop_signal': {'trial_id': gen_cols + ['feedback']},
                'stop_signal': {'trial_id': gen_cols + ['reset', 'feedback']},
                'stroop': {'trial_id': gen_cols + []},
                'survey_medley': {'trial_id': gen_cols},
                'threebytwo': {'trial_id': gen_cols + ['cue', 'gap', 'set_stims']},
                'twobytwo': {'trial_id': gen_cols + ['cue', 'gap', 'set_stims']},
                'tower_of_london': {'trial_id': gen_cols + ['advance', 'practice']},
                'two_stage_decision': {'trial_id': ['end']},
                'ward_and_allport': {'trial_id': gen_cols + ['practice_start_block', 'reminder', 'test_start_block']},
                'willingness_to_wait': {'trial_id': gen_cols + []},
                'writing_task': {'trial_id': gen_cols}}
    to_drop = lookup.get(exp_id, {})
    return to_drop

def post_process_exp(df, exp_id):
    '''Function used to post-process a dataframe extracted via extract_row or extract_experiment
    :exp_id: experiment key used to look up appropriate grouping variables
    '''
    lookup = {'adaptive_n_back': adaptive_nback_post,
              'angling_risk_task': ART_post,
              'angling_risk_task_always_sunny': ART_post,
              'attention_network_task': ANT_post,
              'bickel_titrator': bickel_post,
              'choice_reaction_time': choice_reaction_time_post,
              'cognitive_reflection_survey': cognitive_reflection_post,
              'columbia_card_task_fmri': CCT_fmri_post,
              'columbia_card_task_hot': CCT_hot_post,
              'dietary_decision': dietary_decision_post,
              'discount_titrate': discount_titrate_post,
              'digit_span': span_post,
              'directed_forgetting': directed_forgetting_post,
		   'discount_titrate': discount_titrate_post,
              'dot_pattern_expectancy': DPX_post,
              'hierarchical_rule': hierarchical_post,
              'holt_laury_survey': holt_laury_post,
              'information_sampling_task': IST_post,
              'keep_track': keep_track_post,
              'kirby': kirby_post,
              'local_global_letter': local_global_post,
              'motor_selective_stop_signal': conditional_stop_signal_post,
              'probabilistic_selection': probabilistic_selection_post,
              'psychological_refractory_period_two_choices': PRP_post,
              'ravens': ravens_post,
              'recent_probes': recent_probes_post,
              'self_regulation_survey': self_regulation_survey_post,
              'sensation_seeking_survey': sensation_seeking_survey_post,
              'shape_matching': shape_matching_post,
              'shift_task': shift_post,
              'simon': simon_post,
              'spatial_span': span_post,
              'stim_selective_stop_signal': conditional_stop_signal_post,
              'stop_signal': stop_signal_post,
              'stroop': stroop_post,
              'tower_of_london': TOL_post,
              'threebytwo': threebytwo_post,
              'twobytwo': twobytwo_post,
              'two_stage_decision': two_stage_decision_post,
              'ward_and_allport': WATT_post}

    fun = lookup.get(exp_id, lambda df: df)
    return fun(df).sort_index(axis = 1)

def post_process_data(data):
    """ applies post_process_exp to an entire dataset
    """
    time_taken = {}
    post_processed = []
    for row in data.iterrows():
        print(row[0])
        exp_id = row[1]['experiment_exp_id']
        try:
            df = extract_row(row[1], clean = False)
        except TypeError:
            post_processed.append(numpy.nan)
            continue
        tic = time.time()
        df = post_process_exp(df,exp_id)
        toc = time.time() - tic
        time_taken.setdefault(exp_id,[]).append(toc)
        post_processed.append({'trialdata': df.values.tolist(),'columns':df.columns, 'index': df.index})
    for key in time_taken.keys():
        time_taken[key] = numpy.mean(time_taken[key])
    print(time_taken)
    data.loc[:,'data'] = post_processed
    data.loc[:,'process_stage'] = 'post'

#***********************************
# FUNCTIONS TO RETRIEVE DATA
#***********************************
def extract_row(row, clean = True, apply_post = True, drop_columns = None):
    '''Returns a dataframe that has expanded the data of one row of a results object
    :row:  one row of a Results data dataframe
    :param clean: boolean, if true call clean_df on the data
    :param drop_columns: list of columns to pass to clean_df
    :param drop_na: boolean to pass to clean_df
    :return df: dataframe containing the extracted experiment
    '''
    exp_id = row['experiment_exp_id']
    if row.get('process_stage') == 'post':
        zfill_length = len(str(len(row['data']['index'])))
        df = pandas.DataFrame(row['data']['trialdata'])
        df.columns = row['data']['columns']
        df.index = ['_'.join(t)+'_'+n.zfill(zfill_length)
                    for *t,n in [i.split('_') for i in row['data']['index']]]
        df.sort_index(inplace = True)
        if clean == True:
            df = clean_data(df, row['experiment_exp_id'], False, drop_columns)
    else:
        exp_data = get_data(row)
        for trial in exp_data:
            trial['battery_name'] = row['battery_name']
            trial['experiment_exp_id'] = row['experiment_exp_id']
            trial['worker_id'] = row['worker_id']
            trial['finishtime'] = row['finishtime']
        df = pandas.DataFrame(exp_data)
        zfill_length = len(str(len(exp_data)))
        trial_index = ["%s_%s" % (exp_id,str(x).zfill(zfill_length))
                        for x in range(len(exp_data))]
        df.index = trial_index
        if clean == True:
            df = clean_data(df, row['experiment_exp_id'], apply_post, drop_columns)
    return df

def extract_experiment(data, exp_id, clean = True, apply_post = True,
                       drop_columns = None, return_reject = False,
                       clean_fun = clean_data):
    '''Returns a dataframe that has expanded the data column of the results object for the specified experiment.
    Each row of this new dataframe is a data row for the specified experiment.
    :data: the data from an expanalysis Result object
    :experiment: a string identifying one experiment
    :param clean: boolean, if true call clean_df on the data
    :param drop_columns: list of columns to pass to clean_df
    :param return_reject: bool, default false. If true returns a dataframe with rejected experiments
    :param clean_fun: an alternative "clean" function. Must return a dataframe of the cleaned data
    :return df: dataframe containing the extracted experiment
    '''
    trial_index = []
    df = select_experiment(data, exp_id)
    if 'flagged' in df.columns:
        df_reject = df.query('flagged == True')
        df = df.query('flagged == False')
        if len(df) == 0:
            print('All %s datasets were flagged')
            return df,df_reject
    #report if there is only one dataset for each battery/experiment/worker combination
    if sum(df.groupby(['battery_name', 'experiment_exp_id', 'worker_id']).size()>1)!=0:
        print("More than one dataset found for at least one battery/worker/%s combination" %exp_id)
    if numpy.unique(df.get('process_stage'))=='post':
        group_df = pandas.DataFrame()
        for i,row in df.iterrows():
            tmp_df = extract_row(row, clean, False, drop_columns)
            group_df = pandas.concat([group_df, tmp_df ])
            insert_i = tmp_df.index[0].rfind('_')
            trial_index += [x[:insert_i] + '_s%s' % str(i).zfill(3)
                            + x[insert_i:] for x in tmp_df.index]
        df = group_df
        df.index = trial_index
        df.sort_index(inplace = True)
    else:
        trial_list = []
        for i,row in df.iterrows():
            exp_data = get_data(row)
            for trial in exp_data:
                trial['battery_name'] = row['battery_name']
                trial['experiment_exp_id'] = row['experiment_exp_id']
                trial['worker_id'] = row['worker_id']
                trial['finishtime'] = row['finishtime']
            trial_list += exp_data
            trial_index += ["%s_%s_%s" % (exp_id,str(i).zfill(3),str(x).zfill(3)) for x in range(len(exp_data))]
        df = pandas.DataFrame(trial_list)
        df.index = trial_index
        if clean == True:
            df = clean_fun(df, exp_id, apply_post, drop_columns)
    if return_reject:
        return df, df_reject
    else:
        return df

def export_experiment(filey, data, exp_id, clean = True):
    """ Exports data from one experiment to path specified by filey. Must be .csv, .pkl or .json
    :filey: path to export data
    :data: the data from an expanalysis Result object
    :experiment: experiment to export
    :param clean: boolean, default True. If true cleans the experiment df before export
    """
    df = extract_experiment(data, exp_id, clean)
    file_name,ext = os.path.splitext(filey)
    if ext.lower() == ".csv":
        df.to_csv(filey)
    elif ext.lower() == ".pkl":
        df.to_pickle(filey)
    elif ext.lower() == ".json":
        df.to_json(filey)
    else:
        print("File extension not recognized, must be .csv, .pkl, or .json.")




#***********************************
# DEPENDENT VARIABLE FUNCTIONS
#***********************************
def organize_DVs(DVs):
    """
    Convert DVs from a dictionary of values and valences to two separate
    pandas dataframes: one for values and one for valence
    """
    valence = deepcopy(DVs)
    for key,val in valence.items():
        valence[key] = val
        for subj_key in val.keys():
            val[subj_key]=val[subj_key]['valence']
    for key,val in DVs.items():
        for subj_key in val.keys():
            val[subj_key]=val[subj_key]['value']
    DVs = pandas.DataFrame.from_dict(DVs).T
    valence = pandas.DataFrame.from_dict(valence).T
    return DVs, valence

def calc_exp_DVs(df, use_check = True, use_group_fun = True, group_kwargs=None):
    '''Function to calculate dependent variables
    :experiment: experiment key used to look up appropriate grouping variables
    :param use_check: bool, if True exclude dataframes that have "False" in a
    passed_check column, if it exists. Passed_check would be defined by a post_process
    function specific to that experiment
    '''
    lookup = {'adaptive_n_back': calc_adaptive_n_back_DV,
              'angling_risk_task_always_sunny': calc_ART_sunny_DV,
              'attention_network_task': calc_ANT_DV,
              'bickel_titrator': calc_bickel_DV,
              'bis11_survey': calc_bis11_DV,
              'bis_bas_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='bis_bas_survey'),
              'brief_self_control_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='brief_self_control_survey'),
              'choice_reaction_time': calc_choice_reaction_time_DV,
              'columbia_card_task_cold': calc_CCT_cold_DV,
              'columbia_card_task_hot': calc_CCT_hot_DV,
              'columbia_card_task_fmri': calc_CCT_fmri_DV,
              'cognitive_reflection_survey': calc_cognitive_reflection_DV,
              'demographics_survey': calc_demographics_DV,
              'dietary_decision': calc_dietary_decision_DV,
              'dickman_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='dickman_survey'),
              'digit_span': calc_digit_span_DV,
              'directed_forgetting': calc_directed_forgetting_DV,
              'discount_fixed': calc_discount_fixed_DV,
              'discount_titrate': calc_discount_titrate_DV,
              'dospert_eb_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='dospert_eb_survey'),
              'dospert_rp_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='dospert_rp_survey'),
              'dospert_rt_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='dospert_rt_survey'),
              'dot_pattern_expectancy': calc_DPX_DV,
              'eating_survey': calc_eating_DV,
              'erq_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='erq_survey'),
              'five_facet_mindfulness_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='five_facet_mindfulness_survey'),
              'future_time_perspective_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='^future_time_perspective_survey'),
              'go_nogo': calc_go_nogo_DV,
              'grit_scale_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='grit_scale_survey'),
              'hierarchical_rule': calc_hierarchical_rule_DV,
              'holt_laury_survey': calc_holt_laury_DV,
              'impulsive_venture_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='impulsive_venture_survey'),
              'information_sampling_task': calc_IST_DV,
              'keep_track': calc_keep_track_DV,
              'kirby': calc_kirby_DV,
              'leisure_time_activity_survey': calc_leisure_time_DV,
              'local_global_letter': calc_local_global_DV,
              'mindful_attention_awareness_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='mindful_attention_awareness_survey'),
              'motor_selective_stop_signal': calc_motor_selective_stop_signal_DV,
              'mpq_control_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='mpq_control_survey'),
              'probabilistic_selection': calc_probabilistic_selection_DV,
              'psychological_refractory_period_two_choices': calc_PRP_two_choices_DV,
              'ravens': calc_ravens_DV,
              'recent_probes': calc_recent_probes_DV,
              'selection_optimization_compensation_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='selection_optimization_compensation_survey'),
              'self_regulation_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='self_regulation_survey'),
              'sensation_seeking_survey': calc_SSS_DV,
              'simon': calc_simon_DV,
              'simple_reaction_time': calc_simple_RT_DV,
              'shape_matching': calc_shape_matching_DV,
              'shift_task': calc_shift_DV,
              'spatial_span': calc_spatial_span_DV,
              'stim_selective_stop_signal': calc_stim_selective_stop_signal_DV,
              'stop_signal': calc_stop_signal_DV,
              'stroop': calc_stroop_DV,
              'ten_item_personality_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='ten_item_personality_survey'),
              'theories_of_willpower_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='theories_of_willpower_survey'),
              'time_perspective_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='^time_perspective_survey'),
              'threebytwo': calc_threebytwo_DV,
              'twobytwo': calc_twobytwo_DV,
              'tower_of_london': calc_TOL_DV,
              'two_stage_decision': calc_two_stage_decision_DV,
              'upps_impulsivity_survey': lambda df, use_check: calc_survey_DV(df, use_check, survey_name='upps_impulsivity_survey'),
              'ward_and_allport': calc_WATT_DV,
              'writing_task': calc_writing_DV}
    assert (len(df.experiment_exp_id.unique()) == 1), "Dataframe has more than one experiment in it"
    exp_id = df.experiment_exp_id.unique()[0]
    fun = lookup.get(exp_id, None)
    if group_kwargs is None:
        group_kwargs = {}
    if fun:
        try:
            DVs,description = fun(df, use_check=use_check, use_group_fun=use_group_fun, kwargs=group_kwargs)
        except TypeError:
            DVs,description = fun(df, use_check)
        DVs, valence = organize_DVs(DVs)
        return DVs, valence, description
    else:
        return None, None, None


def get_exp_DVs(data, exp_id, use_check = True, use_group_fun = True, group_kwargs=None):
    '''Function used by clean_df to post-process dataframe
    :experiment: experiment key used to look up appropriate grouping variables
    :param use_check: bool, if True exclude dataframes that have "False" in a
    passed_check column, if it exists. Passed_check would be defined by a post_process
    function specific to that experiment
    '''
    if group_kwargs is None:
        group_kwargs = {}
    df = extract_experiment(data,exp_id)
    return calc_exp_DVs(df, use_check, use_group_fun, group_kwargs)

def extract_proptrials(df, proptrials = 1, rand = False):
    #extract practice
    if 'exp_stage' in df.columns:
        df = df.query('exp_stage != "practice"')

    def get_proptrials(df, proptrials, rand):
        nrows = len(df)
        ntrials = round(nrows*proptrials)
        if rand:
            rtrials = random.sample(range(1,nrows),ntrials)
            out_df = df.iloc[rtrials].reset_index(drop=True)
        else:
            out_df = df.head(ntrials).reset_index(drop=True)
        return out_df

    out_df = df.groupby('worker_id').apply(get_proptrials, proptrials, rand)

    return out_df

def get_exp_DVs_proptrials(data, exp_id, proptrials = 1, rand = False, use_check = True, use_group_fun = True, group_kwargs=None):
    '''Function used by clean_df to post-process dataframe
    :experiment: experiment key used to look up appropriate grouping variables
    :param use_check: bool, if True exclude dataframes that have "False" in a
    passed_check column, if it exists. Passed_check would be defined by a post_process
    function specific to that experiment
    '''
    if group_kwargs is None:
        group_kwargs = {}
    #df = extract_experiment(data,exp_id)
    df = extract_proptrials(df, proptrials, rand)
    return calc_exp_DVs(df, use_check, use_group_fun, group_kwargs)

def get_battery_DVs(data, use_check = True, use_group_fun = True):
    '''Calculate DVs for each subject and each experiment. Returns a subject x DV matrix
    '''
    DVs = pandas.DataFrame()
    valence = pandas.DataFrame()
    for exp in numpy.sort(data.experiment_exp_id.unique()):
        print('Calculating DV for %s' % exp)
        exp_DVs,exp_valence,description = get_exp_DVs(data, exp, use_check, use_group_fun)
        if not exp_DVs is None:
            exp_DVs.columns = [exp + '.' + c for c in exp_DVs.columns]
            exp_valence.columns = [exp + '.' + c for c in exp_valence.columns]
            DVs = pandas.concat([DVs,exp_DVs], axis = 1)
            valence = pandas.concat([valence,exp_valence], axis = 1)
    return DVs, valence

def add_DV_columns(data, use_check = True, use_group_fun = True):
    """Calculate DVs for each experiment and stores the results in data
    :data: the data dataframe of a expfactory Result object
    :param use_check: bool, if True exclude dataframes that have "False" in a
    passed_check column, if it exists. Passed_check would be defined by a post_process
    function specific to that experiment
    """
    data.loc[:,'DV'] = numpy.nan
    data.loc[:,'DV'] = data['DV'].astype(object)
    data.loc[:,'DV_description'] = ''
    for exp_id in numpy.sort(numpy.unique(data['experiment_exp_id'])):
        print('Calculating DV for %s' % exp_id)
        tic = time.time()
        subset = data[data['experiment_exp_id'] == exp_id]
        dvs, valence, description = get_exp_DVs(subset,exp_id, use_check, use_group_fun)
        if not dvs is None:
            subset = subset.query('worker_id in %s' % list(dvs.index))
            if len(dvs) == len(subset):
                data.loc[subset.index,'DV'] = [dvs.loc[worker].to_dict() for worker in subset.worker_id]
                data.loc[subset.index,'DV_valence'] = [valence.loc[worker].to_dict() for worker in subset.worker_id]
                data.loc[subset.index,'DV_description'] = description
        toc = time.time() - tic
        print(exp_id + ': ' + str(toc))

def extract_DVs(data, use_check = True, use_group_fun = True):
    """Calculate if necessary and extract DVs into a new dataframe where rows
    are workers and columns are DVs
    :data: the data dataframe of a expfactory Result object
    :param use_check: bool, if True exclude dataframes that have "False" in a
    passed_check column, if it exists. Passed_check would be defined by a post_process
    function specific to that experiment
    """
    if not 'DV' in data.columns:
        add_DV_columns(data, use_check, use_group_fun)
    data = data[data['DV'].isnull()==False]
    DV_list = []
    valence_list = []
    for worker in numpy.unique(data['worker_id']):
        DV_dict = {'worker_id': worker}
        valence_dict = {'worker_id': worker}
        subset = data.query('worker_id == "%s"' % worker)
        for i,row in subset.iterrows():
            DVs = row['DV']
            DV_valence = row['DV_valence']
            exp_id = row['experiment_exp_id']
            for key in DVs:
                DV_dict[exp_id +'.' + key] = DVs[key]
                valence_dict[exp_id +'.' + key] = DV_valence[key]
        DV_list.append(DV_dict)
        valence_list.append(valence_dict)
    DV_df = pandas.DataFrame(DV_list)
    DV_df.set_index('worker_id', inplace = True)
    valence_df = pandas.DataFrame(valence_list)
    valence_df.set_index('worker_id', inplace = True)
    return DV_df, valence_df

#***********************************
# OTHER
#***********************************

def generate_reference(data, file_base):
    """ Takes a results data frame and returns an experiment dictionary with
    the columsn and column types for each experiment (after apply post_processing)
    :data: the data dataframe of a expfactory Result object
    :file_base:
    """
    exp_dic = {}
    for exp_id in numpy.unique(data['experiment_exp_id']):
        exp_dic[exp_id] = {}
        df = extract_experiment(data,exp_id, clean = False)
        col_types = df.dtypes
        exp_dic[exp_id] = col_types
    pandas.to_pickle(exp_dic, file_base + '.pkl')

def flag_data(data, reference_file):
    """ function to flag data for rejections by checking the columns of the
    extracted data against a reference file generated by generate_reference.py
    :data: the data dataframe of a expfactory Result object
    :exp_id: an expfactory exp_id corresponding to the df
    :save_post_process: bool, if True replaces the data in each row with post processed data
    :reference file: a pickle follow created by generate_reference

    """
    flagged = []
    lookup_dic = pandas.read_pickle(reference_file)
    for i,row in data.iterrows():
        exp_id = row['experiment_exp_id']
        df = extract_row(row, clean = False)
        col_types = df.dtypes
        lookup = lookup_dic[exp_id]
        #drop unimportant cols which are sometimes not recorded
        for col in ['trial_num', 'responses', 'credit_var', 'performance_var']:
            if col in lookup:
                lookup.drop(col, inplace = True)
            if col in col_types:
                col_types.drop(col, inplace = True)
        flag = not ((col_types.index.tolist() == lookup.index.tolist()) and \
            (col_types.tolist() == lookup.tolist()))
        flagged.append(flag)
    data['flagged'] = flagged
