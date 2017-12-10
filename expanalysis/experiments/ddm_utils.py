import hddm
import itertools
import numpy
import pandas
import pickle

def not_regex(txt):
    return '^((?!%s).)*$' % txt

def EZ_diffusion(df, condition = None):
    assert 'correct' in df.columns, 'Could not calculate EZ DDM'
    df = df.copy()
    # convert reaction time to seconds to match with HDDM
    df['rt'] = df['rt']/1000
    # ensure there are no missed responses or extremely short responses (to fit with EZ)
    df = df.query('rt > .05')
    # convert any perfect accuracies to .95
    
    EZ_dvs = {}
    # calculate EZ params for each condition
    if condition:
        conditions = df[condition].unique()
        conditions = conditions[~pandas.isnull(conditions)]
        for c in conditions:
            subset = df[df[condition] == c]
            pc = subset['correct'].mean()
            # edge case correction using the fourth suggestion from 
            # Stanislaw, H., & Todorov, N. (1999). Calculation of signal detection theory measures.
            if pc == 1:
                pc = 1-(.5/len(subset))
            vrt = numpy.var(subset.query('correct == True')['rt'])
            mrt = numpy.mean(subset.query('correct == True')['rt'])
            try:
                drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
                EZ_dvs['EZ_drift_' + c] = {'value': drift, 'valence': 'Pos'}
                EZ_dvs['EZ_thresh_' + c] = {'value': thresh, 'valence': 'Pos'}
                EZ_dvs['EZ_non_decision_' + c] = {'value': non_dec, 'valence': 'Neg'}
            except ValueError:
                continue
    else:
        # calculate EZ params
        try:
            pc = df['correct'].mean()
            # edge case correct
            if pc == 1:
                pc = 1-(1.0/(2*len(df)))
            vrt = numpy.var(df.query('correct == True')['rt'])
            mrt = numpy.mean(df.query('correct == True')['rt'])
            drift, thresh, non_dec = hddm.utils.EZ(pc, vrt, mrt)
            EZ_dvs['EZ_drift'] = {'value': drift, 'valence': 'Pos'}
            EZ_dvs['EZ_thresh'] = {'value': thresh, 'valence': 'Pos'}
            EZ_dvs['EZ_non_decision'] = {'value': non_dec, 'valence': 'Neg'}
        except ValueError:
            return {}
    return EZ_dvs
    
def fit_HDDM(df, 
             response_col = 'correct', 
             formula_cols = [], 
             formulas = None,
             outfile = None, 
             loadfile = None,
             samples=80000,
             burn=10000,
             thin=2):
    """ wrapper to run hddm analysis
    
    Args:
        df: dataframe to perform hddm analyses on
        respones_col: the columnof correct/incorrect values
        formulas_cols: (optional) single dictionary, orlist of dictionaries, 
            whose key is a hddm param
            The  values of each dictare column names to be used in a regression model
            If none are passed, no regression will be performed. For instance, 
            if formula_cols = [{'v': ['condition1']}] then a regression will be 
            run of the form: "v ~ C(condition1, Sum)"
        formulas: (optional) if given overrides automatic formulas
        outfile: if given, models will be saved to this location
        loadfile: if given, this model will be loaded before running analyses
        samples: number of samples to run HDDM
        burn: burn in time for HDDM
        thin: thin parameter passed to HDDM
    """  
    variable_conversion = {'a': ('thresh', 'Pos'), 'v': ('drift', 'Pos'), 't': ('non_decision', 'NA')}
    db = None
    extra_cols = []
    # set up data
    data = (df.loc[:,'rt']/1000).astype(float).to_frame()
    data.insert(0, 'response', df[response_col].astype(float))
    # if formula_cols is a single dictionary, convert to a list of length 1
    if type(formula_cols) == dict:
        formula_cols = [formula_cols]
    for formula in formula_cols:
        extra_cols += list(formula.values())[0]
    for col in extra_cols:
        data.insert(0, col, df[col])
    # state cols dropped when using deviance coding
    dropped_vals = [sorted(data[col].unique())[-1] for col in extra_cols]
    # add subject ids 
    data.insert(0,'subj_idx', df['worker_id'])
    # remove missed responses and extremely short response
    data = data.query('rt > .05')
    subj_ids = data.subj_idx.unique()
    ids = {subj_ids[i]:int(i) for i in range(len(subj_ids))}
    data.replace(subj_ids, [ids[i] for i in subj_ids],inplace = True)
    if outfile:
        db = outfile + '_traces.db'
    
    if loadfile:
        m = hddm.load(loadfile)
        print('Loaded model from %s' % loadfile)
    else:
        # run if estimating variables for the whole task
        if len(extra_cols) == 0:
            # run hddm
            m = hddm.HDDM(data)
        else:
            # if no explicit formulas have been set, create them
            if formulas is None:
                formulas = []
                # iterate through formula cols
                for fc in formula_cols:
                    (ddm_var, cols), = fc.items() # syntax needed for single key-value pair
                    regressor = 'C(' + ', Sum)+C('.join(cols) + ', Sum)'
                    formula = '%s ~ %s' % (ddm_var, regressor)
                    formulas.append(formula)
            m = hddm.models.HDDMRegressor(data, formulas, 
                                          group_only_regressors=False)
        
        
    # find a good starting point which helps with the convergence.
    m.find_starting_values()
    # run model
    m.sample(samples, burn=burn, thin=thin, dbname=db, db='pickle')
    
    if outfile:
        try:
            pickle.dump(m, open(outfile + '.model', 'wb'))
        except Exception:
            print('Saving model failed')
            
    # get average ddm params
    # regex match to find the correct rows
    dvs = {}
    for var in ['a','v','t']:
        match = '^'+var+'(_subj|_Intercept_subj)'
        dvs[var] = m.nodes_db.filter(regex=match, axis=0)['mean']
    
    # output of regression (no interactions)
    condition_dvs = {}
    for ddm_var in ['a','v','t']:
        var_dvs = {}
        for col, dropped in zip(extra_cols, dropped_vals):
            col_dvs = {}
            included_vals = [i for i in data[col].unique() if i != dropped]
            for val in included_vals:
                # regex match to find correct rows
                match='^'+ddm_var+'.*S.'+str(val)+']_subj' 
                # get the individual diff values and convert to list
                ddm_vals = m.nodes_db.filter(regex=match, axis=0).filter(regex=not_regex(':'), axis=0)['mean'].tolist()
                if len(ddm_vals) > 0:
                    col_dvs[val] = ddm_vals
            if len(col_dvs.keys()) > 0:
                # construct dropped dvs
                dropped_dvs = []
                for vs in zip(*col_dvs.values()):
                    dropped_dvs.append(-1*sum(vs))
                col_dvs[dropped] = dropped_dvs
            var_dvs.update(col_dvs)
        if len(var_dvs)>0:
            condition_dvs[ddm_var] = var_dvs
    # interaction
    interaction_dvs = {}
    all_levels = []
    for col in extra_cols:
        all_levels += list(data.loc[:,col].unique())
    for ddm_var in ['a','v','t']:
        var_dvs = {}
        for x, y in itertools.permutations(all_levels,2):
            # regex match to find correct rows
            match='^'+ddm_var+'.*'+str(x)+'].*:.*'+str(y)+']_subj'
            # get the individual diff values and convert to list
            ddm_vals = m.nodes_db.filter(regex=match, axis=0)['mean'].tolist()
            if len(ddm_vals) > 0:
                var_dvs['%s:%s' % (str(x), str(y))] = ddm_vals
        if len(var_dvs) > 0:
            interaction_dvs[ddm_var] = var_dvs
    
    group_dvs = {}
    # create output ddm dict
    for i,subj in enumerate(subj_ids):
        group_dvs[subj] = {}
        hddm_vals = {}
        for var in ['a','v','t']:
            var_name, var_valence = variable_conversion[var]
            if var in list(dvs.keys()):
                hddm_vals.update({'hddm_' + var_name: {'value': dvs[var][i], 'valence': var_valence}})
            if var in condition_dvs.keys():
                for k,v in condition_dvs[var].items():
                    tmp = {'value': v[i], 'valence': var_valence}
                    hddm_vals.update({'hddm_'+var_name+'_'+k: tmp})
            if var in interaction_dvs.keys():
                for k,v in interaction_dvs[var].items():
                    tmp = {'value': v[i], 'valence': var_valence}
                    hddm_vals.update({'hddm_'+var_name+'_'+k: tmp})
        group_dvs[subj].update(hddm_vals)
            
    return group_dvs

def ANT_HDDM(df, outfile=None, **kwargs):
    group_dvs = fit_HDDM(df, 
                         formula_cols = {'v': ['flanker_type', 'cue']}, 
                         outfile = outfile,
                         **kwargs)
    return group_dvs

def local_global_HDDM(df, outfile=None, **kwargs):
    df = df.copy()
    df.insert(0, 'correct_shift', df.correct.shift(1))
    group_dvs = fit_HDDM(df, 
                         formula_cols = {'v': ['conflict_condition', 'switch']}, 
                         outfile = outfile,
                         **kwargs)
    return group_dvs


def motor_SS_HDDM(df, outfile=None, **kwargs):
    df = df.query('SS_trial_type == "go" and \
                 exp_stage not in ["practice","NoSS_practice"]')
    group_dvs = fit_HDDM(df, 
                         outfile = outfile,
                         **kwargs)
    return group_dvs


def stim_SS_HDDM(df, outfile=None, **kwargs):
    df = df.query('SS_trial_type == "go" and \
                 exp_stage not in ["practice","NoSS_practice"]')
    group_dvs = fit_HDDM(df, 
                         outfile = outfile,
                         **kwargs)
    return group_dvs

def SS_HDDM(df, outfile=None, **kwargs):
    df = df.query('SS_trial_type == "go" \
                 and exp_stage not in ["practice","NoSS_practice"]')
    group_dvs = fit_HDDM(df, 
                         outfile = outfile,
                         **kwargs)
    return group_dvs

def threebytwo_HDDM(df, outfile=None, **kwargs):
    df = df.copy()
    
    df.loc[:,'cue_switch_binary'] = df.cue_switch.map(lambda x: ['cue_stay','cue_switch'][x!='stay'])
    df.loc[:,'task_switch_binary'] = df.task_switch.map(lambda x: ['task_stay','task_switch'][x!='stay'])
        
    formula = "v ~ (C(cue_switch_binary, Sum)+C(task_switch_binary, Sum))*C(CTI,Sum) - C(CTI,Sum)"
    group_dvs = fit_HDDM(df, 
                         formula_cols = {'v': ['cue_switch_binary', 'task_switch_binary', 'CTI']}, 
                         formulas = formula,
                         outfile = outfile,
                         **kwargs)
    return group_dvs

def twobytwo_HDDM(df, outfile=None, **kwargs):
    df = df.copy()
    
    df.loc[:,'cue_switch_binary'] = df.cue_switch.map(lambda x: ['cue_stay','cue_switch'][x!='stay'])
    df.loc[:,'task_switch_binary'] = df.task_switch.map(lambda x: ['task_stay','task_switch'][x!='stay'])
        
    formula = "v ~ (C(cue_switch_binary, Sum)+C(task_switch_binary, Sum))*C(CTI,Sum) - C(CTI,Sum)"
    group_dvs = fit_HDDM(df, 
                         formula_cols = {'v': ['cue_switch_binary', 'task_switch_binary', 'CTI']}, 
                         formulas = formula,
                         outfile = outfile,
                         **kwargs)
    return group_dvs


def get_HDDM_fun(task=None, outfile=None, **kwargs):
    if outfile is None:
        outfile=task
    hddm_fun_dict = \
    {
        'adaptive_n_back': lambda df: fit_HDDM(df.query('load == 2'), 
                                               outfile=outfile,
                                               **kwargs),
        'attention_network_task': lambda df: ANT_HDDM(df, outfile, **kwargs),
        'choice_reaction_time': lambda df: fit_HDDM(df, 
                                                    outfile=outfile,
                                                    **kwargs),
        'directed_forgetting': lambda df: fit_HDDM(df.query('trial_id=="probe"'), 
                                                   formula_cols = {'v': ['probe_type']},
                                                   outfile=outfile,
                                                   **kwargs),
        'dot_pattern_expectancy': lambda df: fit_HDDM(df, 
                                                      formula_cols = {'v': ['condition']}, 
                                                      outfile = outfile,
                                                      **kwargs), 
                                                      
        'local_global_letter': lambda df: local_global_HDDM(df, outfile, **kwargs),
        'motor_selective_stop_signal': lambda df: motor_SS_HDDM(df, outfile, **kwargs),
        'recent_probes': lambda df: fit_HDDM(df, 
                                            formula_cols = {'v': ['probeType']},
                                            outfile = outfile,
                                            **kwargs),
        'shape_matching': lambda df: fit_HDDM(df, 
                                              formula_cols = {'v': ['condition']}, 
                                              outfile = outfile,
                                              **kwargs), 
        'simon': lambda df: fit_HDDM(df, 
                                     formula_cols = {'v': ['condition']}, 
                                     outfile = outfile,
                                     **kwargs), 
        'stim_selective_stop_signal': lambda df: stim_SS_HDDM(df, outfile, **kwargs),
        'stop_signal': lambda df: SS_HDDM(df, outfile, **kwargs),
        'stroop': lambda df: fit_HDDM(df, 
                                      formula_cols = {'v': ['condition']}, 
                                      outfile = outfile,
                                      **kwargs), 
        'threebytwo': lambda df: threebytwo_HDDM(df, outfile, **kwargs),
        'twobytwo': lambda df: twobytwo_HDDM(df, outfile, **kwargs)
    }
    if task is None:
        return hddm_fun_dict
    else:
        return hddm_fun_dict[task]