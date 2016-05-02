"""
expanalysis/experiments/utils.py: part of expfactory package
functions for working with experiment factory results
"""

import requests
import __init__
import pandas
import json
import os
import unicodedata
import re
 

def get_data(row):
    """Data can be stored in different forms depending on the experiment template.
    This function returns the data in a standard form (a list of trials)
    :row:  one row of a results dataframe
    """
    def get_response_text(question):
        """Returns the response text that corresponds to the value recorded in a survey question
        :question: A dictionary corresponding to a survey question
        """
        val = question['response']
        if 'options' in question.keys():
            options = question['options']
            text = [lookup_val(opt['text']) for opt in options if 'value' in opt.keys() and opt['value'] == val]
            if len(text) == 1: text = text[0]
        else:
            text = pandas.np.nan
        return text
    try:
        data = row['data']
    except:
        print 'No data column found!'
    if row['experiment_template'] == 'jspsych':
        if len(data) == 1:
            return data[0]['trialdata']
        elif len(data) > 1:
           return  [trial['trialdata'] for trial in data]
        else:
            print "No data found"
    elif row['experiment_template'] == 'survey':
        survey =  data.values()
        for i in survey:
            i['question_num'] = int(re.search(r'%s_([0-9]{1,2})*' % row['experiment_exp_id'], i['id']).group(1))
            i['response_text'] = get_response_text(i)
        survey = sorted(survey, key=lambda k: k['question_num'])
        return survey
    elif row['experiment_template'] == 'unknown':
        print "Couldn't determine data template"
        
    
def lookup_val(val):
    """function that modifies a string so that it conforms to expfactory analysis by 
    replacing it with an interpretable synonym
    :val: val to lookup
    """
    if isinstance(val,(str,unicode)):
        #convert unicode to str
        if isinstance(val, unicode):
            val = unicodedata.normalize('NFKD', val).encode('ascii', 'ignore')
        lookup_val = val.strip().lower()
        lookup_val = val.replace(" ", "_")
        #define synonyms
        lookup = {
        'reaction time': 'rt',
        'instructions': 'instruction',
        'correct': 1,
        'incorrect': 0}
        return lookup.get(lookup_val,val)
    else:
        return val
    
def select_battery(df, battery):
    '''Selects a battery (or batteries) from results object and sorts based on worker and time of experiment completion
    :results: a Results object
    :battery: a string or array of strings to select the battery(s)
    :return df: dataframe containing the appropriate result subset
    '''
    Pass = True
    if isinstance(battery, (unicode, str)):
        battery = [battery]
    for b in battery:
        if not b in df['battery_name'].values:
            print "Alert!:  The battery '%s' not found in results. Try resetting the results" % (b)  
            Pass = False
    assert Pass == True, "At least one battery was not found in results"
    df = df.query("battery_name in %s" % battery)
    df = df.sort_values(by = ['battery_name', 'experiment_exp_id', 'worker_id', 'finishtime'])
    df.reset_index(inplace = True, drop = True)
    return df
    
def select_experiment(df, exp_id):
    '''Selects an experiment (or experiments) from results object and sorts based on worker and time of experiment completion
    :results: a Results object
    :param exp_id: a string or array of strings to select the experiment(s)
    :return df: dataframe containing the appropriate result subset
    '''
    Pass = True
    if isinstance(exp_id, (unicode, str)):
        exp_id = [exp_id]
    for e in exp_id:
        if not e in df['experiment_exp_id'].values:
            print "Alert!: The experiment '%s' not found in results. Try resetting the results" % (e)
            Pass = False
    assert Pass == True, "At least one experiment was not found in results"
    df = df.query("experiment_exp_id in %s" % exp_id)
    df = df.sort_values(by = ['experiment_exp_id', 'worker_id', 'battery_name', 'finishtime'])
    df.reset_index(inplace = True, drop = True)
    return df
    
def select_worker(df, worker):
    '''Selects a worker (or workers) from results object and sorts based on experiment and time of experiment completion
    :results: a Results object
    :worker: a string or array of strings to select the worker(s)
    :return df: dataframe containing the appropriate result subset
    '''
    Pass = True
    if isinstance(worker, (unicode, str)):
        worker = [worker]
    for w in worker:
        if not w in df['worker_id'].values:
            print "Alert!: The experiment '%s' not found in results. Try resetting the results" % (w)
            Pass = False
    assert Pass == True, "At least one worker was not found in results"
    df = df.query("worker_id in %s" % worker)
    df = df.sort_values(by = ['worker_id', 'experiment_exp_id', 'battery_name', 'finishtime'])
    df.reset_index(inplace = True, drop = True)
    return df   

def select_template(df, template):
    '''Selects a template (or templates) from results object and sorts based on experiment and time of experiment completion
    :results: a Results object
    :template: a string or array of strings to select the worker(s)
    :return df: dataframe containing the appropriate result subset
    '''
    if isinstance(template, (unicode, str)):
        template = [template]
    template = map(str.lower,template)
    df = df.query("experiment_template in %s" % template)
    assert len(df) != 0, "At least one template was not found in results"
    df = df.sort_values(by = ['worker_id', 'experiment_exp_id', 'battery_name', 'finishtime'])
    df.reset_index(inplace = True, drop = True)
    return df
    
def select_finishtime(df, finishtime, all_data = True):
     '''Get results after a finishtime 
    :results: a Results object
    :finishtime: a date string
    :param all_data: boolean, default True. If true, only select data where the entire dataset was collected afte rthe finishtime
    :return df: dataframe containing the appropriate result subset
    '''
     if all_data:
        passed_df = df.groupby('worker_id')['finishtime'].min() >= finishtime
        workers = list(passed_df[passed_df].index)
        df = select_worker(df, workers)
     else:
        df = df.query('finishtime >= "%s"' % finishtime) 
        df.reset_index(inplace = True, drop = True)
     return df
    


