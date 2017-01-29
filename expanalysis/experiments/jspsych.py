'''
analysis/experiments/jspsych.py: part of expfactory package
jspsych functions
'''
import numpy
from expanalysis.experiments.utils import get_data, lookup_val, select_worker
from expanalysis.experiments.processing import extract_experiment, extract_row

def calc_time_taken(data):
    '''Selects a worker (or workers) from results object and sorts based on experiment and time of experiment completion
    '''
    instruction_lengths = []
    exp_lengths = []
    for row in data.iterrows():
        if row[1]['experiment_template'] == 'jspsych':
            try:
                exp_data = extract_row(row[1],clean = False)
            except TypeError:
                exp_lengths.append(numpy.nan)
                instruction_lengths.append(numpy.nan)
                continue
            #ensure there is a time elapsed variable
            assert 'time_elapsed' in exp_data.iloc[-1].keys(), \
                '"time_elapsed" not found for at least one dataset in these results'
            #sum time taken on instruction trials
            instruction_length = exp_data[(exp_data.trial_id == 'instruction') | (exp_data.trial_id == 'instructions')]['rt'].sum()     
            #Set the length of the experiment to the time elapsed on the last 
            #jsPsych trial
            experiment_length = exp_data.iloc[-1]['time_elapsed']
            instruction_lengths.append(instruction_length/1000.0)
            exp_lengths.append(experiment_length/1000.0)
        else:
            instruction_lengths.append(numpy.nan)
            exp_lengths.append(numpy.nan)
    data.loc[:,'total_time'] = exp_lengths
    data.loc[:,'instruct_time'] = instruction_lengths
    data.loc[:,'ontask_time'] = data['total_time'] - data['instruct_time']
    print('Finished calculating time taken')
        

def get_average_variable(results, var):
    '''Prints time taken for each experiment in minutes
    '''
    averages = {}
    for exp in results.get_experiments():
        data = extract_experiment(results,exp)
        try:
            average = data[var].mean()
        except TypeError:
            print("Cannot average %s" % (var))
        averages[exp] = average
    return averages
    
    
def get_post_task_responses(data):
    question_responses = [numpy.nan] * len(data)
    for i,row in zip(range(0, len(data)),data.iterrows()):
        if row[1]['experiment_template'] == 'jspsych':
            try:
                row_data = extract_row(row[1],clean = False)
            except TypeError:
                continue
            if row_data.iloc[-2].get('trial_id') =='post task questions' and \
                'responses' in row_data.iloc[-2].keys():
                question_responses[i]= (row_data.iloc[-2]['responses'])
    data.loc[:,'post_task_responses'] = question_responses
    print('Finished extracting post task responses')

    


    
    
    