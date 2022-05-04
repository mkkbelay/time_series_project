import glob
import pandas as pd
import os
files = glob.glob("bing_data/QueriesByState_*.tsv")
dfs = [pd.read_csv(f, sep="\t") for f in files]

bingdata = pd.concat(dfs,ignore_index=True)
bingdata

covid_bing=bingdata[(bingdata['State']=='Texas') | (bingdata['State']=='Georgia')| (bingdata['State']=='South Dakota')| (bingdata['State']=='Virginia')]



def create_dict(news_agency=None):
    '''
    This function takes in a list of string items and outputs a dictionary object, turning each item in the
    inputted list to a key and the number of times each item appears in the list to its corresponding value.
    '''
    news_agency_story = dict()
    for word in news_agency:
        if word in news_agency_story:
            news_agency_story[word][0] += 1
        else:
            news_agency_story[word] = [1]
    return news_agency_story#sorted(news_agency_story, key=news_agency_story.get, reverse=True)[:5]

subset=covid_bing[(covid_bing['State']=='Texas')].filter(['Query']).values.tolist()
flat_list = [item for sublist in subset for item in sublist]
output=create_dict(flat_list)
top_5=dict(list(dict(sorted(output.items(), reverse=True,key=lambda item: item[1])).items())[:5])


import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.barh(list(top_5.keys()), [value for item in list(top_5.values()) for value in item], align='center')
plt.title('Texas: Top 5 Search Queries')


subset=covid_bing[(covid_bing['State']=='Georgia')].filter(['Query']).values.tolist()
flat_list = [item for sublist in subset for item in sublist]
output=create_dict(flat_list)
top_5=dict(list(dict(sorted(output.items(), reverse=True,key=lambda item: item[1])).items())[:5])
plt.barh(list(top_5.keys()), [value for item in list(top_5.values()) for value in item], align='center')
plt.title('Georgia: Top 5 Search Queries')



subset=covid_bing[(covid_bing['State']=='South Dakota')].filter(['Query']).values.tolist()
flat_list = [item for sublist in subset for item in sublist]
output=create_dict(flat_list)
top_5=dict(list(dict(sorted(output.items(), reverse=True,key=lambda item: item[1])).items())[:5])
plt.barh(list(top_5.keys()), [value for item in list(top_5.values()) for value in item], align='center')
plt.title('South Dakota: Top 5 Search Queries')


subset=covid_bing[(covid_bing['State']=='Virginia')].filter(['Query']).values.tolist()
flat_list = [item for sublist in subset for item in sublist]
output=create_dict(flat_list)
top_5=dict(list(dict(sorted(output.items(), reverse=True,key=lambda item: item[1])).items())[:5])
plt.barh(list(top_5.keys()), [value for item in list(top_5.values()) for value in item], align='center')
plt.title('Virginia: Top 5 Search Queries')
