from expanalysis.experiments.utils import remove_duplicates
from expanalysis.results import get_filters, Result

# default filters to clean up the downloaded results object
filters = get_filters()
# expfactory.org/token
access_token = "1111111111" # expfactory.org/token
# url of the battery
# for example, if the battery url is: http://expfactory.org/batteries/999/
# the url below will be: http://www.expfactory.org/new_api/results/999/
url = 'http://www.expfactory.org/new_api/results/999/'

# create a results object
results = Result(access_token, filters = filters, url = url)

# we care about the data
data = results.data
# remove duplicates - there shouldn't be any, just a safety precaution
remove_duplicates(data)

"""
The results.data object has one row per worker/experiment pair. 

The column "data" holds the data as a dictionary. Unofrunately, this
data is in a 1-length list, and the data's "trialdata" is really what you want.

To index the first row's data you would index the results.data object like so:
>>> first_row_data = results.data.iloc[0]['data'][0]['trialdata']

Which can then be converted into a dataframe easily
>>> first_row_data = pandas.DataFrame(first_row_data)
"""

