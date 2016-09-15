"""
analysis/experiments/survey_processing.py: part of expfactory package
functions for automatically cleaning and manipulating surveys
"""
import pandas
import numpy


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

"""
Demographics
"""
@multi_worker_decorate
def calc_demographics_DV(df):
    dvs = {}
    dvs['age'] = {'value':  int(df[df.question_num == 3].response), 'valence': 'NA'}
    dvs['sex'] = {'value':  df[df.question_num == 2].response_text[0], 'valence': 'NA'}
    dvs['race'] = {'value':  list(df[df.question_num == 4].response), 'valence': 'NA'}
    dvs['hispanic?'] = {'value':  df[df.question_num == 6].response_text[0], 'valence': 'NA'}
    dvs['education'] = {'value':  df[df.question_num == 7].response_text[0], 'valence': 'Pos'}
    dvs['height(inches)'] = {'value':  int(df[df.question_num == 8].response), 'valence': 'NA'}
    dvs['weight(pounds)'] = {'value':  int(df[df.question_num == 9].response), 'valence': 'NA'}
    # calculate bmi
    weight_kilos = dvs['weight(pounds)']['value']*0.453592
    height_meters = dvs['height(inches)']['value']*.0254
    BMI = weight_kilos/height_meters**2
    if dvs['height(inches)']['value'] > 30 and dvs['weight(pounds)']['value'] > 50:
        dvs['BMI'] = {'value': BMI, 'valence': 'Neg'}
    else:
        dvs['BMI'] = {'value': numpy.nan, 'valence': 'Neg'}
    dvs['relationship_status'] = {'value':  df[df.question_num == 10].response_text[0], 'valence': 'NA'}
    dvs['divoce_count'] = {'value':  df[df.question_num == 11].response_text[0], 'valence': 'NA'}
    dvs['longest_relationship(months)'] = {'value':  int(df[df.question_num == 12].response), 'valence': 'NA'}
    dvs['relationship_count'] = {'value':  df[df.question_num == 13].response_text[0], 'valence': 'NA'}
    dvs['children_count'] = {'value':  df[df.question_num == 14].response_text[0], 'valence': 'NA'}
    dvs['household_income(dollars)'] = {'value':  int(df[df.question_num == 15].response), 'valence': 'NA'}
    dvs['retirement_account?'] = {'value':  df[df.question_num == 16].response_text[0], 'valence': 'NA'}
    dvs['percent_retirement_in_stock'] = {'value':  df[df.question_num == 17].response[0], 'valence': 'NA'}
    dvs['home_status'] = {'value':  df[df.question_num == 18].response_text[0], 'valence': 'NA'}
    dvs['mortage_debt'] = {'value':  df[df.question_num == 19].response_text[0], 'valence': 'NA'}
    dvs['car_debt'] = {'value':  df[df.question_num == 20].response_text[0], 'valence': 'NA'}
    dvs['education_debt'] = {'value':  df[df.question_num == 21].response_text[0], 'valence': 'NA'}
    dvs['credit_card_debt'] = {'value':  df[df.question_num == 22].response_text[0], 'valence': 'NA'}
    dvs['other_sources_of_debt'] = {'value':  df[df.question_num == 23].response_text[0], 'valence': 'NA'}
    #calculate total caffeine intake
    caffeine_intake = \
        int(df[df.question_num == 25].response)*100 + \
        int(df[df.question_num == 26].response)*40 + \
        int(df[df.question_num == 27].response)*30 + \
        int(df[df.question_num == 28].response)
    dvs['caffeine_intake'] = {'value':  caffeine_intake, 'valence': 'NA'}
    dvs['gambling_problem?'] = {'value':  df[df.question_num == 29].response_text[0], 'valence': 'Neg'}
    dvs['traffic_ticket_count'] = {'value':  df[df.question_num == 30].response_text[0], 'valence': 'Neg'}
    dvs['traffic_accident_count'] = {'value':  df[df.question_num == 31].response_text[0], 'valence': 'Neg'}
    dvs['arrest_count'] = {'value':  df[df.question_num == 32].response_text[0], 'valence': 'Neg'}
    dvs['mturk_motivation'] = {'value':  list(df[df.question_num == 33].response), 'valence': 'NA'}
    dvs['other_motivation'] = {'value':  df[df.question_num == 34].response_text[0], 'valence': 'NA'}
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
    scores = {
        'attention': ([6,10,12,21,29], 'Neg'),
        'cognitive_stability': ([7,25,27], 'Neg'),
        'motor': ([3,4,5,18,20,23,26], 'Neg'),
        'perseverance': ([17,22,24,31], 'Neg'),
        'self-control': ([2,8,9,13,14,15], 'Neg'),
        'cognitive_complexity': ([11,16,19,28,30], 'Neg'),
    }
    DVs = {}
    firstorder = {}
    for score,subset in scores.items():
         firstorder[score] = df.query('question_num in %s' % subset[0]).numeric_response.sum()
         DVs['first_order_' + score] = {'value': df.query('question_num in %s' % subset[0]).numeric_response.sum(), 'valence': subset[1]}
    DVs['Attentional'] = {'value':  firstorder['attention'] + firstorder['cognitive_stability'], 'valence': 'Neg'}
    DVs['Motor'] = {'value':  firstorder ['motor'] + firstorder['perseverance'], 'valence': 'Neg'}
    DVs['Nonplanning'] = {'value':  firstorder['self-control'] + firstorder['cognitive_complexity'], 'valence': 'Neg'}
    description = """
        Score for bis11. Higher values mean
        greater expression of that "impulsive" factor. High values are negative traits. "Attentional", "Motor" and "Nonplanning"
        are second-order factors, while the other 6 are first order factors.
    """
    return DVs,description

@multi_worker_decorate
def calc_bis_bas_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'BAS_drive': ([4,10,13,22], 'NA'),
        'BAS_fun_seeking': ([6,11,16,21], 'NA'),
        'BAS_reward_responsiveness': ([5,8,15,19,24], 'NA'),
        'BIS': ([3,9,14,17,20,23,25], 'NA'),

    }
    DVs = {}
    for score,subset in scores.items():
         DVs[score] = {'value': df.query('question_num in %s' % subset[0]).numeric_response.sum(), 'valence': subset[1]}
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
    scores = {
        'dysfunctional': ([2,5,8,10,11,14,15,18,19,22,23,24], 'Neg'),
        'functional': ([3,4,6,7,9,12,13,16,17,20,21], 'Pos')
    }
    DVs = {}
    for score,subset in scores.items():
         DVs[score] = {'value': df.query('question_num in %s' % subset[0]).numeric_response.sum(), 'valence': subset[1]}
    description = """
        Score for all dickman impulsivity survey. Higher values mean
        greater expression of that factor. 
    """
    return DVs,description
    
@multi_worker_decorate
def calc_dospert_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'ethical': ([7,10,11,17,30,31], 'NA'), 
        'financial': ([4,5,9,13,15,19], 'NA'), 
        'health/safety': ([6,16,18,21,24,27], 'NA'), 
        'recreational': ([3,12,14,20,25,26], 'NA'), 
        'social': ([2,8,22,23,28,29], 'NA')
    }
    DVs = {}
    for score,subset in scores.items():
         DVs[score] = {'value': df.query('question_num in %s' % subset[0]).numeric_response.sum(), 'valence': subset[1]}
    description = """
        Score for all dospert scales. Higher values mean
        greater expression of that factor. 
    """
    return DVs,description
    
@multi_worker_decorate
def calc_eating_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'cognitive_restraint': ([3,12,13,16,17,19], 'Pos'), 
        'uncontrolled_eating': ([2,5,6,8,9,10,14,15,18], 'Neg'),
        'emotional_eating': ([4,7,11], 'Neg')
    }
    DVs = {}
    for score,subset in scores.items():
         raw_score = df.query('question_num in %s' % subset[0]).numeric_response.sum()
         normalized_score = (raw_score-len(subset[0]))/(len(subset[0])*3)*100
         DVs[score] = {'value': normalized_score, 'valence': subset[1]}
    description = """
        Score for three eating components. Higher values mean
        greater expression of that value
    """
    return DVs,description
    
@multi_worker_decorate
def calc_erq_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'reappraisal': ([2,4,6,8,9,11], 'NA'),
        'suppression': ([3,5,7,10], 'NA')
    }
    DVs = {}
    for score,subset in scores.items():
        DVs[score] = {'value': df.query('question_num in %s' % subset[0]).numeric_response.mean(), 'valence': subset[1]}
    description = """
        Score for different emotion regulation strategies. Higher values mean
        greater expression of that strategy
    """
    return DVs,description
    
@multi_worker_decorate
def calc_five_facet_mindfulness_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'observe': ([2,7,12,16,21,27,32,37], 'Pos'),
        'describe': ([3,8,13,17,23,28,33,38], 'Pos'),
        'act_with_awareness': ([6,9,14,19,23,29,35,39], 'Pos'),
        'nonjudge': ([4,11,15,18,26,31,36,39], 'Pos'),
        'nonreact': ([5,10,20,22,25,30,34], 'Pos')
    }
    DVs = {}
    for score,subset in scores.items():
         DVs[score] = {'value': df.query('question_num in %s' % subset[0]).numeric_response.sum(), 'valence': subset[1]}
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
    scores = {
        'impulsiveness': ([6,7,8,11,13,15,16,17,18,21,22,24,26,29,30,31], 'Neg'),
        'venturesomeness': ([2,3,4,5,9,10,12,14,19,20,23,25,27,28,32],  'NA')
    }
    DVs = {}
    for score,subset in scores.items():
         DVs[score] = {'value': df.query('question_num in %s' % subset[0]).numeric_response.sum(), 'valence': subset[1]}
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
    scores = {
        'elective_selection': (list(range(2,14)), 'Pos'),
        'loss-based_selection': (list(range(14,26)), 'Pos'),
        'optimization': (list(range(26,38)), 'Pos'),
        'compensation': (list(range(38,50)), 'Pos')
    }
    DVs = {}
    for score,subset in scores.items():
        DVs[score] = {'value': df.query('question_num in %s' % subset[0]).numeric_response.mean(), 'valence': subset[1]}
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
    scores = {
        'boredom_susceptibility': ([3,6,8,9,16,25,28,32,35,40], 'NA'),
        'disinhibition': ([2,13,14,26,30,31,33,34,36,37], 'NA'),
        'experience_seeking': ([5,7,10,11,15,19,20,23,27,38], 'NA'),
        'thrill_adventure_seeking': ([4,12,17,18,21,22,24,29,39,41], 'NA')
    }
    DVs = {}
    for score,subset in scores.items():
        DVs[score] = {'value': df.query('question_num in %s' % subset[0]).numeric_response.mean(), 'valence': subset[1]}
    description = """
        Score for SSS-V. Higher values mean
        greater expression of that trait
    """
    return DVs,description
    
@multi_worker_decorate
def calc_ten_item_personality_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'extraversion': ([3,8], 'NA'),
        'agreeableness': ([4, 9], 'NA'),
        'conscientiousness': ([5,10], 'NA'),
        'emotional_stability': ([6,11], 'NA'),
        'openness': ([7,12], 'NA')
    }
    DVs = {}
    for score,subset in scores.items():
        DVs[score] = {'value': df.query('question_num in %s' % subset[0]).numeric_response.mean(), 'valence': subset[1]}
    description = """
        Score for five different personality measures. Higher values mean
        greater expression of that personality
    """
    return DVs,description

@multi_worker_decorate
def calc_theories_of_willpower_DV(df):
    DVs = {'endorse_limited_resource': {'value': df['response'].astype(float).sum(), 'valence': 'Pos'}}
    description = """
        Higher values on this survey indicate a greater endorsement of a 
        "limited resource" theory of willpower
    """
    return DVs,description

@multi_worker_decorate
def calc_time_perspective_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'past_negative': ([5,6,17,23,28,34,35,37,51,55], 'NA'),
        'present_hedonistic': ([2,9,13,18,20,24,27,29,32,33,43,45,47,49,56], 'NA'),
        'future': ([7,10,11,14,19,22,25,31,41,14,46,52,57], 'NA'),
        'past_positive': ([3,8,11,16,21,26,30,42,50], 'NA'),
        'present_fatalistic': ([4,15,36,38,39,40,48,53,54], 'NA')
    }
    DVs = {}
    for score,subset in scores.items():
        DVs[score] = {'value': df.query('question_num in %s' % subset[0]).numeric_response.mean(), 'valence': subset[1]}
    description = """
        Score for five different time perspective factors. High values indicate 
        higher expression of that value
    """
    return DVs,description
    
@multi_worker_decorate
def calc_upps_DV(df):
    df.insert(0,'numeric_response', df['response'].astype(float))
    scores = {
        'negative_urgency': ([3,8,13,18,23,30,35,40,45,51,54,59], 'NA'),
        'lack_of__premeditation': ([2,7,12,17,22,29,34,39,44,49,56], 'NA'),
        'lack_of_perseverance': ([5,10,15,20,25,28,33,38,43,48], 'NA'),
        'sensation_seeking': ([4,9,14,19,24,27,32,37,42,47,52,57], 'NA'),
        'positive_urgency': ([6,11,16,21,26,31,36,41,46,50,53,55,58,60], 'NA')
    }
    DVs = {}
    for score,subset in scores.items():
        DVs[score] = {'value': df.query('question_num in %s' % subset[0]).numeric_response.sum(), 'valence': subset[1]}
    description = """
        Score for five different upps+p measures. Higher values mean
        greater expression of that factor
    """
    return DVs,description
    