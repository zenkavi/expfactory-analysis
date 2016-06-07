'''
analysis/experiments/jspsych.py: part of expfactory package
jspsych functions
'''
import numpy
from expanalysis.experiments.utils import get_data, lookup_val, select_worker
from expanalysis.experiments.processing import extract_experiment

def calc_time_taken(data):
    '''Selects a worker (or workers) from results object and sorts based on experiment and time of experiment completion
    '''
    instruction_lengths = []
    exp_lengths = []
    for i,row in data.iterrows():
        if row['experiment_template'] == 'jspsych':
            exp_data = get_data(row)
            #ensure there is a time elapsed variable
            assert 'time_elapsed' in exp_data[-1].keys(), \
                '"time_elapsed" not found for at least one dataset in these results'
            #sum time taken on instruction trials
            instruction_length = numpy.sum([trial['time_elapsed'] for trial in exp_data if lookup_val(trial.get('trial_id')) == 'instruction'])        
            #Set the length of the experiment to the time elapsed on the last 
            #jsPsych trial
            experiment_length = exp_data[-1]['time_elapsed']
            instruction_lengths.append(instruction_length/1000.0)
            exp_lengths.append(experiment_length/1000.0)
        else:
            instruction_lengths.append(numpy.nan)
            exp_lengths.append(numpy.nan)
    data.loc[:,'total_time'] = exp_lengths
    data.loc[:,'instruct_time'] = instruction_lengths
    data.loc[:,'ontask_time'] = data['total_time'] - data['instruct_time']
        
def print_time(data, time_col = 'ontask_time'):
    '''Prints time taken for each experiment in minutes
    :param time_col: Dataframe column of time in seconds
    '''
    df = data.copy()    
    assert time_col in df, \
        '"%s" has not been calculated yet. Use calc_time_taken method' % (time_col)
    #drop rows where time can't be calculated
    df = df.dropna(subset = [time_col])
    time = (df.groupby('experiment_exp_id')[time_col].mean()/60.0).round(2)
    print(time)
    return time

def get_average_variable(results, var):
    '''Prints time taken for each experiment in minutes
    '''
    averages = {}
    for exp in results.get_experiments():
        data = extract_experiment(results,exp)
        try:
            average = data[var].mean()
        except TypeError:
            print "Cannot average %s" % (var)
        averages[exp] = average
    return averages
    
    
def get_post_task_responses(results):
    question_responses = {}
    for worker in results.get_workers():
        data = select_worker(results, worker)
        worker_responses = {}
        for i,row in data.iterrows():
            if check_template(row['data']) == 'jspsych':
                if 'responses' in row['data'][-2]['trialdata'].keys():
                    response = row['data'][-2]['trialdata']['responses']
                    worker_responses[row['experiment']] = response
                else:
                    worker_responses[row['experiment']] = ''
        question_responses[worker] = worker_responses
    return question_responses
    


    
    
    