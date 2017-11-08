"""
analysis/experiments/survey_processing.py: part of expfactory package
functions for automatically cleaning and manipulating surveys
"""
import numpy
import os
import pandas

# reference for calculating subscales
file_loc = os.path.dirname(os.path.realpath(__file__))
reference_scores = pandas.DataFrame.from_csv(os.path.join(file_loc,'survey_subscale_reference.csv'))

"""
Generic Functions
"""

def multi_worker_decorate(func):
    """Decorator to ensure that dv functions have only one worker
    """
    def multi_worker_wrap(group_df, use_check = True):
        group_dvs = {}
        if len(group_df) == 0:
            return group_dvs, ''
        if 'passed_check' in group_df.columns and use_check:
            group_df = group_df[group_df['passed_check']]
        for worker in pandas.unique(group_df['worker_id']):
            df = group_df.query('worker_id == "%s"' %worker)
            try:
                group_dvs[worker], description = func(df)
            except:
                print('DV calculated failed for worker: %s' % worker)
        return group_dvs, description
    return multi_worker_wrap

def get_scores(survey):
    subset = reference_scores.filter(regex = survey, axis = 0)
    subscale_dict = {}
    for name, values in subset.iterrows():
        subscale_name = name.split('.')[1]
        subscale_items = [int(i) for i in values.tolist()[2:] if i == i]
        subscale_valence = values.iloc[1]
        subscale_dict[subscale_name] = [subscale_items, subscale_valence]
    return subscale_dict
    
    
"""
Demographics
"""

def get_response_text(data, qnum):
    text = numpy.nan
    if qnum in data.question_num.tolist():
        text = data[data.question_num == qnum].response_text[0]
        if text:
            text = text.strip()
    return text

def get_response_value(data,qnum, nan_values = []):
    value = numpy.nan
    if qnum in data.question_num.tolist():
        if not isinstance(nan_values,list):
            nan_values = [nan_values]
        try:
            value = int(data[data.question_num == qnum].response)
            if value in nan_values:
                value = numpy.nan
        except ValueError:
            pass
    return value
    
@multi_worker_decorate
def get_demographics_DV_text(df):
    dvs = {}
    dvs['age'] = {'value':  get_response_value(df, 3), 'valence': 'NA'}
    dvs['sex'] = {'value':  get_response_text(df, 2), 'valence': 'NA'}
    dvs['race'] = {'value':  list(df[df.question_num == 4].response), 'valence': 'NA'}
    dvs['hispanic?'] = {'value':  get_response_text(df, 6), 'valence': 'NA'}
    dvs['education'] = {'value':  get_response_text(df, 7), 'valence': 'Pos'}
    dvs['height(inches)'] = {'value':  get_response_value(df, 8), 'valence': 'NA'}
    dvs['weight(pounds)'] = {'value':  get_response_value(df, 9), 'valence': 'NA'}
    # calculate bmi
    weight_kilos = dvs['weight(pounds)']['value']*0.453592
    height_meters = dvs['height(inches)']['value']*.0254
    BMI = weight_kilos/height_meters**2
    if dvs['height(inches)']['value'] > 30 and dvs['weight(pounds)']['value'] > 50:
        dvs['BMI'] = {'value': BMI, 'valence': 'Neg'}
    else:
        dvs['BMI'] = {'value': numpy.nan, 'valence': 'Neg'}
    dvs['relationship_status'] = {'value':  get_response_text(df, 10), 'valence': 'NA'}
    dvs['divoce_count'] = {'value':  get_response_text(df, 11), 'valence': 'NA'}
    dvs['longest_relationship(months)'] = {'value':  get_response_value(df, 12), 'valence': 'NA'}
    dvs['relationship_count'] = {'value':  get_response_text(df, 13), 'valence': 'NA'}
    dvs['children_count'] = {'value':  get_response_text(df, 14), 'valence': 'NA'}
    dvs['household_income(dollars)'] = {'value':  get_response_value(df, 15), 'valence': 'NA'}
    dvs['retirement_account?'] = {'value':  get_response_text(df, 16), 'valence': 'NA'}
    dvs['percent_retirement_in_stock'] = {'value':  get_response_text(df, 17), 'valence': 'NA'}
    dvs['home_status'] = {'value':  get_response_text(df, 18), 'valence': 'NA'}
    dvs['mortage_debt'] = {'value':  get_response_text(df, 19), 'valence': 'NA'}
    dvs['car_debt'] = {'value':  get_response_text(df, 20), 'valence': 'NA'}
    dvs['education_debt'] = {'value':  get_response_text(df, 21), 'valence': 'NA'}
    dvs['credit_card_debt'] = {'value':  get_response_text(df, 22), 'valence': 'NA'}
    dvs['other_sources_of_debt'] = {'value':  get_response_text(df, 24), 'valence': 'NA'}
    #calculate total caffeine intake
    caffeine_intake = \
        get_response_value(df, 25)*100 + \
        get_response_value(df, 26)*40 + \
        get_response_value(df, 27)*30 + \
        get_response_value(df, 28)
    dvs['caffeine_intake'] = {'value':  caffeine_intake, 'valence': 'NA'}
    dvs['gambling_problem?'] = {'value':  get_response_text(df, 29), 'valence': 'Neg'}
    dvs['traffic_ticket_count'] = {'value':  get_response_text(df, 30), 'valence': 'Neg'}
    dvs['traffic_accident_count'] = {'value':  get_response_text(df, 31), 'valence': 'Neg'}
    dvs['arrest_count'] = {'value':  get_response_text(df, 32), 'valence': 'Neg'}
    dvs['mturk_motivation'] = {'value':  list(df[df.question_num == 33].response), 'valence': 'NA'}
    dvs['other_motivation'] = {'value':  get_response_text(df, 34), 'valence': 'NA'}
    description = "Outputs various demographic variables"
    return dvs,description

    
@multi_worker_decorate
def calc_demographics_DV(df):
    dvs = {}
    dvs['age'] = {'value': get_response_value(df,3), 'valence': 'NA'}
    dvs['sex'] = {'value':  get_response_text(df,2), 'valence': 'NA'}
    dvs['race'] = {'value':  list(df[df.question_num == 4].response), 'valence': 'NA'}
    dvs['hispanic?'] = {'value':  get_response_text(df,6), 'valence': 'NA'}
    dvs['education'] = {'value':  get_response_value(df,7), 'valence': 'Pos'}
    dvs['height(inches)'] = {'value':  get_response_value(df,8), 'valence': 'NA'}
    dvs['weight(pounds)'] = {'value':  get_response_value(df,9), 'valence': 'NA'}
    # calculate bmi
    weight_kilos = dvs['weight(pounds)']['value']*0.453592
    height_meters = dvs['height(inches)']['value']*.0254
    BMI = weight_kilos/height_meters**2
    if dvs['height(inches)']['value'] > 30 and dvs['weight(pounds)']['value'] > 50:
        dvs['BMI'] = {'value': BMI, 'valence': 'Neg'}
    else:
        dvs['BMI'] = {'value': numpy.nan, 'valence': 'Neg'}
    dvs['relationship_status'] = {'value':  get_response_text(df,10), 'valence': 'NA'}
    dvs['divoce_count'] = {'value':  get_response_value(df,11), 'valence': 'NA'}
    dvs['longest_relationship(months)'] = {'value':  get_response_value(df,12), 'valence': 'NA'}
    dvs['relationship_count'] = {'value':  get_response_value(df,13), 'valence': 'NA'}
    dvs['children_count'] = {'value':  get_response_value(df,14), 'valence': 'NA'}
    dvs['household_income(dollars)'] = {'value':  get_response_value(df,15), 'valence': 'NA'}
    dvs['retirement_account?'] = {'value':  get_response_text(df,16), 'valence': 'NA'}
    dvs['percent_retirement_in_stock'] = {'value':  get_response_value(df,17), 'valence': 'NA'}
    dvs['home_status'] = {'value':  get_response_text(df,18), 'valence': 'NA'}
    dvs['mortage_debt'] = {'value':  get_response_value(df,19,nan_values = 0), 'valence': 'NA'}
    dvs['car_debt'] = {'value':  get_response_value(df,20,nan_values = 0), 'valence': 'NA'}
    dvs['education_debt'] = {'value':  get_response_value(df,21,nan_values = 0), 'valence': 'NA'}
    dvs['credit_card_debt'] = {'value':  get_response_value(df,22,nan_values = 0), 'valence': 'NA'}
    dvs['other_sources_of_debt'] = {'value':  get_response_value(df,24,nan_values = 0), 'valence': 'NA'}
    #calculate total caffeine intake
    caffeine_intake = \
        get_response_value(df,25)*100 + \
        get_response_value(df,26)*40 + \
        get_response_value(df,27)*30 + \
        get_response_value(df,28)
    dvs['caffeine_intake'] = {'value':  caffeine_intake, 'valence': 'NA'}
    dvs['gambling_problem?'] = {'value':  get_response_text(df,29), 'valence': 'Neg'}
    dvs['traffic_ticket_count'] = {'value':  get_response_value(df,30), 'valence': 'Neg'}
    dvs['traffic_accident_count'] = {'value':  get_response_value(df,31), 'valence': 'Neg'}
    dvs['arrest_count'] = {'value':  get_response_value(df,32), 'valence': 'Neg'}
    dvs['mturk_motivation'] = {'value':  list(df[df.question_num == 33].response), 'valence': 'NA'}
    dvs['other_motivation'] = {'value':  df[df.question_num == 34].response.iloc[0] or numpy.nan, 'valence': 'NA'}
    description = "Outputs various demographic variables"
    return dvs,description

"""
Post Processing functions
"""
def self_regulation_survey_post(df):
    def abs_diff(lst):
        if len(lst)==2:
            return abs(lst[0]-lst[1])
        else:
            return numpy.nan
    df.response = df.response.astype(float)
    avg_response = df.query('question_num in %s' %[24,26]).groupby('worker_id').response.mean().tolist()
    abs_diff = df.query('question_num in %s' %[24,26]).groupby('worker_id').response.agg(abs_diff)
    df.loc[df.question_num == 26,'response'] = avg_response
    df.loc[df.question_num == 26,'repeat_response_diff'] = abs_diff
    df = df.query('question_num != 24')
    return df

"""
DV functions
"""

@multi_worker_decorate
def calc_bis11_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('bis11_survey')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            DVs[score] = {'value': score_subset.mean(), 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
    DVs['Attentional'] = {'value':  DVs['first_order_attention']['value'] + DVs['first_order_cognitive_stability']['value'], 'valence': 'Neg'}
    DVs['Motor'] = {'value':  DVs ['first_order_motor']['value'] + DVs['first_order_perseverance']['value'], 'valence': 'Neg'}
    DVs['Nonplanning'] = {'value':  DVs['first_order_self_control']['value'] + DVs['first_order_cognitive_complexity']['value'], 'valence': 'Neg'}
    description = """
        Score for bis11. Higher values mean
        greater expression of that "impulsive" factor. High values are negative traits. "Attentional", "Motor" and "Nonplanning"
        are second-order factors, while the other 6 are first order factors.
    """
    return DVs,description

@multi_worker_decorate
def calc_bis_bas_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('bis_bas_survey')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            DVs[score] = {'value': score_subset.mean(), 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
    description = """
        Score for bias/bas. Higher values mean
        greater expression of that factor. BAS: "behavioral approach system",
        BIS: "Behavioral Inhibition System"
    """
    return DVs,description

@multi_worker_decorate
def calc_brief_DV(df):
    DVs = {'self_control': {'value': df['response'].astype(float).sum(), 'valence': 'Pos'}}
    description = """
        Grit level. Higher means more gritty
    """
    return DVs,description

@multi_worker_decorate
def calc_dickman_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('dickman')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            DVs[score] = {'value': score_subset.mean(), 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
    description = """
        Score for all dickman impulsivity survey. Higher values mean
        greater expression of that factor. 
    """
    return DVs,description
    
@multi_worker_decorate
def calc_dospert_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('dospert')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            DVs[score] = {'value': score_subset.mean(), 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
    description = """
        Score for all dospert scales. Higher values mean
        greater expression of that factor. 
    """
    return DVs,description
    
@multi_worker_decorate
def calc_eating_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('eating')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            raw_score = df.query('question_num in %s' % subset[0]).numeric_response.sum()
            normalized_score = (raw_score-len(subset[0]))/(len(subset[0])*3)*100
            DVs[score] = {'value': normalized_score, 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
         
    description = """
        Score for three eating components. Higher values mean
        greater expression of that value
    """
    return DVs,description
    
@multi_worker_decorate
def calc_erq_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('erq')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            DVs[score] = {'value': score_subset.mean(), 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
    description = """
        Score for different emotion regulation strategies. Higher values mean
        greater expression of that strategy
    """
    return DVs,description
    
@multi_worker_decorate
def calc_five_facet_mindfulness_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('five_facet')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            DVs[score] = {'value': score_subset.mean(), 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
    description = """
        Score for five factors mindfulness. Higher values mean
        greater expression of that value
    """
    return DVs,description

@multi_worker_decorate
def calc_future_time_perspective_DV(df):
    DVs = {'future_time_perspective': {'value': df['response'].astype(float).sum(), 'valence': 'Pos'}}
    description = """
        Future time perspective (FTP) level. Higher means being more attentive/
        influenced by future states
    """
    return DVs,description
    
@multi_worker_decorate
def calc_grit_DV(df):
    DVs = {'grit': {'value': df['response'].astype(float).sum(), 'valence': 'Pos'}}
    description = """
        Grit level. Higher means more gritty
    """
    return DVs,description
        
@multi_worker_decorate
def calc_i7_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('impulsive_venture')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            DVs[score] = {'value': score_subset.mean(), 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
    description = """
        Score for i7. Higher values mean
        greater expression of that value. One question was removed from the original
        survey for venturesomeness: "Would you like to go pot-holing"
    """
    return DVs,description
    
@multi_worker_decorate
def calc_leisure_time_DV(df):
    DVs = {'activity_level': {'value': float(df.iloc[0]['response']), 'valence': 'Pos'}}
    description = """
        Exercise level. Higher means more exercise
    """
    return DVs,description

@multi_worker_decorate
def calc_maas_DV(df):
    DVs = {'mindfulness': {'value': df['response'].astype(float).mean(), 'valence': 'Pos'}}
    description = """
        mindfulness level. Higher levels means higher levels of "dispositional mindfulness"
    """
    return DVs,description

@multi_worker_decorate
def calc_mpq_control_DV(df):
    DVs = {'control': {'value': df['response'].astype(float).sum(), 'valence': 'Pos'}}
    description = """
        control level. High scorers on this scale describe themselves as:
            Reflective; cautious, careful, plodding; rational, 
            sensible, level-headed; liking to plan activities in detail.
    """
    return DVs,description

@multi_worker_decorate
def calc_SOC_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('selection_optimization')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            DVs[score] = {'value': score_subset.mean(), 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
    description = """
        Score for five different personality measures. Higher values mean
        greater expression of that personality
    """
    return DVs,description
    
@multi_worker_decorate
def calc_SSRQ_DV(df):
    DVs = {'control': {'value': df['response'].astype(float).sum(), 'valence': 'Pos'}}
    description = """
        control level. High scorers means higher level of endorsement
    """
    return DVs,description

@multi_worker_decorate
def calc_SSS_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('sensation_seeking_survey')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            DVs[score] = {'value': score_subset.mean(), 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
    DVs['total'] = {'value': df['numeric_response'].sum(), 'valence': 'Pos'}
    description = """
        Score for SSS-V. Higher values mean
        greater expression of that trait
    """
    return DVs,description
    
@multi_worker_decorate
def calc_ten_item_personality_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('ten_item')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            DVs[score] = {'value': score_subset.mean(), 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
    description = """
        Score for five different personality measures. Higher values mean
        greater expression of that personality
    """
    return DVs,description

@multi_worker_decorate
def calc_theories_of_willpower_DV(df):
    DVs = {'endorse_limited_resource': {'value': df['response'].astype(float).sum(), 'valence': 'Neg'}}
    description = """
        Higher values on this survey indicate a greater endorsement of a 
        "limited resource" theory of willpower
    """
    return DVs,description

@multi_worker_decorate
def calc_time_perspective_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('^time_perspective_survey')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            DVs[score] = {'value': score_subset.mean(), 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
    description = """
        Score for five different time perspective factors. High values indicate 
        higher expression of that value
    """
    return DVs,description
    
@multi_worker_decorate
def calc_upps_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = get_scores('upps')
    DVs = {}
    for score,subset in scores.items():
        score_subset = df.query('question_num in %s' % subset[0]).numeric_response
        if len(subset[0]) == len(score_subset):
            DVs[score] = {'value': score_subset.mean(), 'valence': subset[1]}
        else:
            print("%s score couldn't be calculated for subject %s" % (score, df.worker_id.unique()[0]))
    description = """
        Score for five different upps+p measures. Higher values mean
        greater expression of that factor
    """
    return DVs,description
    