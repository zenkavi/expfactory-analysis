'''
expanalysis/experiments/stats.py: part of expfactory package
stats functions
'''

from expanalysis.maths import check_numeric
from expanalysis.experiments.plots import plot_groups
from expanalysis.experiments.processing import extract_experiment,  get_DV
from expanalysis.experiments.utils import result_filter
from matplotlib import pyplot as plt
import pandas
import numpy
import hddm

def results_check(data, exp_id = None, worker = None, columns = ['correct', 'rt'], remove_practice = True, use_groups = True,  plot = False, silent = False):
    """Outputs info for a basic data check on the results object. Uses data_check to group, describe and plot
    dataframes. Function first filters the results object as specified,
    loops through each experiment and worker contained in the results object, performs some basic dataframe manipulation
    and runs data_check
    :data: the data from an expanalysis Result object
    :param experiment: a string or array of strings to select the experiment(s) before calculating basic stats
    :param worker: a string or array of strings to select the worker(s) before calculating basic stats
    :param columns: array of columns to subset summary statistics, if they exist
    :param remove_practice: bool, default True. If True will remove any rows labeled "practice" in the "exp_stage" column, if it exists
    :param use_groups: bool, default True. If True will lookup grouping variables using get_groupby for the experiment
    :param silent: bool, default False. If True will not print output
    :param plot: bool, default False: If True plots data using plot_groups
    :return summary, p: summary data frame and plot object
    """
    assert 'worker_id' in data.columns and 'experiment_exp_id' in data.columns, \
        "Results data must have 'worker_id' and 'experiment_exp_id' in columns"
    stats = {}
    results = result_filter(data, exp_id = exp_id, worker = worker)
    
    print '******************************************************************************'
    print 'Input: Type "exit" to end, "skip" to skip to the next experiment, or hit enter to continue'
    print '******************************************************************************'
    for experiment in numpy.unique(results['experiment_exp_id']):
        stats[experiment] = {}
        if not silent or plot:
            print '******************************************************************************'
            print '    Experiment: ',  experiment
            print '******************************************************************************'
        if use_groups:
            groupby = get_groupby(experiment)
        else:
            groupby = []
        experiment_df = extract_experiment(results, experiment)
        for worker in pandas.unique(experiment_df['worker_id']):
            if not silent or plot:
                print '******************************************************************************'
                print '    Worker: ',  worker
                print '******************************************************************************'
            df = experiment_df.query('worker_id == "%s"' % worker)
            summary, p = data_check(df, columns, remove_practice, groupby, silent, plot)
            #add summary and plot to dictionary of summaries
            stats[experiment]= {worker: {'summary': summary, 'plot': p}}
            if not silent or plot:
                input_text = raw_input("Press Enter to continue...")
                plt.close()
                if input_text in ['skip', 'exit']:
                    break
        if input_text == 'exit': 
            break
    return stats

def data_check(df, columns = [], remove_practice = True, groupby = [], silent = False, plot = False):
    """Outputs info for a basic data check on one experiment. Functionality to group, describe and plot
    dataframes
    :df:
    :param columns: array of columns to subset summary statistics, if they exist
    :param remove_practice: bool, default True. If True will remove any rows labeled "practice" in the "exp_stage" column, if it exists
    :param groupby: list of columns in df to groupby using pandas .groupby function
    :param silent: bool, default False. If True will not print output
    :param plot: bool, default False: If True plots data using plot_groups
    :return summary, p: summary data frame and plot object
    """
    assert len(pandas.unique(df['experiment_exp_id'])) == 1, \
        "More than one experiment found"
    generic_drop_cols = ['correct_response', 'question_num', 'focus_shifts', 'full_screen', 'stim_id', 'trial_id', 'index', 'trial_num', 'responses', 'key_press', 'time_elapsed']
    df.replace(-1, numpy.nan, inplace = True)
    
    if remove_practice and 'exp_stage' in df.columns:
        df = df.query('exp_stage != "practice"')
        
    if not set(columns).issubset(df.columns) or len(columns) == 0:
        print "Columns selected were not found for %s. Printing generic info" % df['experiment'].iloc[0]
        keep_cols = df.columns
    else:
        keep_cols = groupby + columns
    df = df[keep_cols]
    drop_cols = [col for col in generic_drop_cols if col in df.columns]
    df = df.drop(drop_cols, axis = 1)
        
    #group summary if groupby variables exist
    if len(groupby) != 0:
        summary = df.groupby(groupby).describe()
        summary.reset_index(inplace = True)
        #reorder columns
        stats_level = 'level_%s' % len(groupby)
        summary.insert(0, 'Stats', summary[stats_level])
        summary.drop(stats_level, axis = 1, inplace = True)
        summary = summary.query("Stats in ['mean','std','min','max','50%']")
    else:
        summary = df.describe()
        summary.insert(0, 'Stats', summary.index)
    if plot:
        p = plot_groups(df, groupby)
    else:
        p = numpy.nan
    if not silent:
        print(summary)
        print('\n')
    return summary, p
    
def get_groupby(exp_id):
    '''Function used by basic_stats to group data ouptut
    :experiment: experiment key used to look up appropriate grouping variables
    '''
    lookup = {'adaptive_n_back': ['load'],
                'angling_risk_task_always_sunny': [], 
                'attention_network_task': ['flanker_type', 'cue'], 
                'bickel_titrator': [], 
                'choice_reaction_time': [], 
                'columbia_card_task_cold': [], 
                'columbia_card_task_hot': [], 
                'dietary_decision': [], 
                'digit_span': ['condition'],
                'directed_forgetting': [],
                'dot_pattern_expectancy': [],
                'go_nogo': [],
                'hierarchical_rule': [],
                'information_sampling_task': [],
                'keep_track': [],
                'kirby': [],
                'local_global_letter': [],
                'motor_selective_stop_signal': ['SS_trial_type'],
                'probabilistic_selection': [],
                'psychological_refractory_period_two_choices': [],
                'recent_probes': [],
                'shift_task': [],
                'simple_reaction_time': [],
                'spatial_span': ['condition'],
                'stim_selective_stop_signal': ['condition'],
                'stop_signal': ['condition', 'SS_trial_type'],
                'stroop': ['condition'], 
                'simon':['condition'], 
                'threebytwo': ['task_switch', 'cue_switch'],
                'tower_of_london': [],
                'two_stage_decision': [],
                'willingness_to_wait': [],
                'writing_task': []} 
                
    try:
        return lookup[exp_id]
    except KeyError:
        print "Automatic lookup of groups failed: %s not found in lookup table." % exp_id
        return []


def calc_DVs(data):
    """Calculate DVs for each experiment
    :results: results object
    """
    data['DV_val'] = numpy.nan
    data['DV_val'] = data['DV_val'].astype(object)
    data['DV_description'] = ''
    for exp_id in numpy.unique(data['experiment_exp_id']):
        dvs, description = get_DV(data,exp_id)   
        for worker, val in dvs.items():
            i = data.query('worker_id == "%s" and experiment_exp_id == "%s"' %(worker,exp_id)).index[0]
            data.set_value(i,'DV_val', val)
            data.set_value(i,'DV_description', description)
    
def extract_DVs(data):
    if not 'DV_val' in data.columns:
        calc_DVs(data)
    data = data[data['DV_val'].isnull()==False]
    DV_list = []
    for worker in numpy.unique(data['worker_id']):
        DV_dict = {'worker_id': worker}
        subset = data.query('worker_id == "%s"' % worker)
        for i,row in subset.iterrows():
            DVs = row['DV_val'].copy()
            exp_id = row['experiment_exp_id']
            for key in DVs:
                DV_dict[exp_id +'.' + key] = DVs[key]
        DV_list.append(DV_dict)
    return pandas.DataFrame(DV_list)
    
    
def EZ_diffusion(df):
    assert 'correct' in df.columns, 'Could not calculate EZ DDM'
    pc = df['correct'].mean()
    vrt = numpy.var(df.query('correct == True')['rt'])
    mrt = numpy.mean(df.query('correct == True')['rt'])
    drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
    return {'drift': drift, 'thresh': thresh, 'non_decision': non_dec}
    

def run_DDM(results, EZ = True):
    experiment_DDM = {}
    if EZ:
        for experiment in results.get_experiments():
            df = extract_experiment(results, experiment)
            if 'correct' in df.columns:
                experiment_DDM[experiment] = EZ_diffusion(df)
            else:
                print 'Could not calculate EZ DDM for %s. No "correct" column.' % (experiment)
    return experiment_DDM
    
        
