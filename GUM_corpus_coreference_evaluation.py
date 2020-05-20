#!/usr/bin/env python
# coding: utf-8

# # GOLD ANNOTATIONS

# ### Import functions

# In[1]:


import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import re


# ### File names

# In[2]:


file_names = pd.read_csv('gum-master/annis/corpus.annis', delim_whitespace = True, quoting = 3,                    skipfooter = 1, header = None, engine = 'python')[1]
path_coref_begin = 'gum-master/coref/conll/'
path_coref_end = '.conll'
path_text_begin = 'gum-master/paula/GUM/'
path_text_end = '.text.xml'


# ### Coreference annotations

# In[3]:


coref_annotations = {}
for name in file_names:
    coref_annotations[name] = pd.read_csv(path_coref_begin + name + path_coref_end,                                                delim_whitespace = True, header=None,                                                quoting=3, skipfooter=2, skiprows=1, engine='python')


# ### Creating key files (annotated files) for metrics

# In[6]:


#formatting for evaluation metrics

for name in file_names:
    coref_annotations[name][2] = coref_annotations[name][2].apply(lambda x: re.sub("[a-zA-Z\-]", "", x))
    coref_annotations[name][2] = coref_annotations[name][2].apply(lambda x: re.sub("_", "-", x))


# In[7]:


begin_key_file = '# begin document '
end_key_file = '# end document'
for name in file_names:
    f= open("formatted_files/key_files/" + name + ".key","w+")
    f.write(begin_key_file + name + '\n')
    for i in range(coref_annotations[name].shape[0]):
        string = str(coref_annotations[name].iloc[i, 0]) + "\t" +        coref_annotations[name].iloc[i, 1] + "\t" + coref_annotations[name].iloc[i, 2] + "\n"
        f.write(string)
    f.write(end_key_file)
    f.close() 


# # COREFERENCE OUTPUT

# ### Import functions

# In[9]:


import spacy
from spacy.vocab import Vocab
from spacy.tokens import Doc
import neuralcoref


# ### Raw texts

# In[ ]:


texts = {}
for name in file_names:
    tree = ET.parse(path_text_begin + name + '/' + name + path_text_end)
    texts[name] = tree.getroot()[1].text


# ## NEURAL COREF

# ### Setting up model: en_core_web_lg

# In[10]:


# making custom tokenizer from tokens in annotated data
tokens_dict = {}
for name in file_names:
    tokens_dict[texts[name]] = list(coref_annotations[name][1])
    
nlp = spacy.load('en_core_web_lg')

def custom_tokenizer(text):
    if text in tokens_dict:
        return Doc(nlp.vocab, tokens_dict[text])
    else:
        raise ValueError('No tokenization available for input.')

nlp.tokenizer = custom_tokenizer

neuralcoref.add_to_pipe(nlp)


# ### Create doc object for each raw text

# In[11]:


docs = {}
for name in file_names:
    docs[name] = nlp(texts[name])


# ### Create dataframes for coreference outputs

# In[12]:


coref_output = {}
for name in file_names:
    rows = []
    for token in docs[name]:
        if token._.in_coref:
            coref_clusters_raw = [cluster for cluster in token._.coref_clusters]
            coref_clusters_final = coref_clusters_raw + [np.nan] * (4 - len(coref_clusters_raw))
            rows.append([str(token)] + coref_clusters_final)
        else:
            rows.append([str(token), '-'] + [np.nan] * 3)
    coref_output[name] = pd.DataFrame(rows, columns=['token', 'coref1', 'coref2', 'coref3', 'coref4'])


# ### Conversion dictionaries: from coref chain to number

# In[14]:


conversions = {}
for name in file_names:
    conversion_dict = {}
    for i in range(len(docs[name]._.coref_clusters)):
        conversion_dict[docs[name]._.coref_clusters[i]] = str(i + 1)
    conversions[name] = conversion_dict


# In[15]:


for name in file_names:
    for col in coref_output[name].columns[1:]:
        coref_output[name][col] = coref_output[name][col].map(conversions[name]).fillna(coref_output[name][col])


# ### Joining all chains into one column

# In[16]:


for name in file_names:
    coref_output[name]['coref'] = coref_output[name][coref_output[name].columns[1:]].values.tolist()


# ### Getting the final format into the column 'f_coref'

# In[17]:


for name in file_names:
    
    coref_output[name]['f_coref'] = coref_output[name].loc[:, ('token')]
    
    for i in range(coref_output[name].shape[0]):
    
        if '-' in coref_output[name].loc[i, ('coref')]:
            coref_output[name].loc[i, ('f_coref')] = '-' #coref_output[name]['coref'][i][0]
    
        elif i == 0:
            raw_str = ''
            for element in coref_output[name].loc[i, ('coref')]:
                if type(element) == str:
                    if element not in coref_output[name].loc[i+1, ('coref')]:
                        raw_str += '(' + element + ')'
                    elif element in coref_output[name].loc[i+1, ('coref')]:
                        raw_str += '(' + element
            coref_output[name].loc[i, ('f_coref')] = raw_str
                
        elif i == coref_output[name].shape[0]:
            raw_str = ''
            for element in coref_output[name].loc[i, ('coref')]:
                if type(element) == str:
                    if element not in coref_output[name].loc[i-1, ('coref')]:
                        raw_str += '(' + element + ')'
                    elif element in coref_output[name].loc[i-1, ('coref')]:
                        raw_str += element + ')'
            coref_output[name].loc[i, ('f_coref')] = raw_str
               
        else:
            raw_str = ''
            for element in coref_output[name].loc[i, ('coref')]:
                if type(element) == str:
                    if element not in coref_output[name].loc[i-1, ('coref')] and element not in coref_output[name].loc[i+1, ('coref')]:
                        raw_str += '(' + element + ')'
                    elif element not in coref_output[name].loc[i-1, ('coref')] and element in coref_output[name].loc[i+1, ('coref')]:
                        raw_str += '(' + element
                    elif element in coref_output[name].loc[i-1, ('coref')] and element not in coref_output[name].loc[i+1, ('coref')]:
                        raw_str += element + ')'
            if raw_str == '':
                raw_str = '-'
            coref_output[name].loc[i, ('f_coref')] = raw_str


# ### Creating response files

# In[20]:


begin_response_file = '# begin document '
end_response_file = '# end document'
for name in file_names:
    f= open("formatted_files/response_files/en_core_web_lg/" + name + ".response","w+")
    f.write(begin_response_file + name + '\n')
    for i in range(coref_output[name].shape[0]):
        string = str(i) + "\t" + coref_output[name].loc[i, ('token')] +        "\t" + coref_output[name].loc[i, ('f_coref')] + "\n"
        f.write(string)
    f.write(end_response_file)
    f.close() 


# # COREF METRICS

# In[31]:


f= open("formatted_files/file_names.txt","w+")
for name in file_names:
    f.write(name + '\n')
f.close() 


# In[15]:


results_lg = {}
name = ''
metric = ''
with open("results_en_core_web_lg.txt") as f:
    for line in f:
        if re.compile("^GUM[a-zA-Z\_1-9]*$").match(line):
            name = line[:-1]
            results_lg[name] = {}
            search_mentions = True
        if re.compile("^METRIC").match(line):
            metric = line.split()[1][:-1]
            if metric != 'blanc':
                blanc = False
                results_lg[name][metric] = {}
            else:
                blanc = True
        if re.compile("^Coreference:").match(line) and not blanc:
            coref = line.split()
            results_lg[name][metric] = {'recall': float(coref[5][:-1]),
                                         'precision' : float(coref[10][:-1]),
                                        'F1' : float(coref[12][:-1])}
        if re.compile("^Identification of Mentions:").match(line) and search_mentions:
            mentions = line.split()
            results_lg[name]['mentions'] = {'recall': float(mentions[7][:-1]),
                                         'precision' : float(mentions[12][:-1]),
                                        'F1' : float(mentions[14][:-1])}
            search_mentions = False


# In[19]:


results_sm = {}
name = ''
metric = ''
with open("results_en_core_web_sm.txt") as f:
    for line in f:
        if re.compile("^GUM[a-zA-Z\_1-9]*$").match(line):
            name = line[:-1]
            results_sm[name] = {}
            search_mentions = True
        if re.compile("^METRIC").match(line):
            metric = line.split()[1][:-1]
            if metric != 'blanc':
                blanc = False
                results_sm[name][metric] = {}
            else:
                blanc = True
        if re.compile("^Coreference:").match(line) and not blanc:
            coref = line.split()
            results_sm[name][metric] = {'recall': float(coref[5][:-1]),
                                         'precision' : float(coref[10][:-1]),
                                        'F1' : float(coref[12][:-1])}
        if re.compile("^Identification of Mentions:").match(line) and search_mentions:
            mentions = line.split()
            results_sm[name]['mentions'] = {'recall': float(mentions[7][:-1]),
                                         'precision' : float(mentions[12][:-1]),
                                        'F1' : float(mentions[14][:-1])}
            search_mentions = False


# In[79]:


metrics = ['recall', 'precision', 'F1']
coref_metrics = ['mentions', 'muc', 'bcub', 'ceafm', 'ceafe']

rs_sm = {}
for name in results_sm.keys():
    rs_sm[name] = []
    for metric in coref_metrics:
        for m in metrics:
            rs_sm[name].append(results_sm[name][metric][m])
            
rs_lg = {}
for name in results_lg.keys():
    rs_lg[name] = []
    for metric in coref_metrics:
        for m in metrics:
            rs_lg[name].append(results_lg[name][metric][m])


# In[80]:


columns = [tuple([c,m]) for c in coref_metrics for m in metrics]
sm = pd.DataFrame.from_dict(rs_sm, orient = 'index')
lg = pd.DataFrame.from_dict(rs_lg, orient = 'index')
sm.columns=pd.MultiIndex.from_tuples(columns)
lg.columns=pd.MultiIndex.from_tuples(columns)


# In[113]:


sm_c = sm.rename(lambda x:re.search('_[a-z]*_', x).group()[1:-1], axis ="index")
lg_c = lg.rename(lambda x:re.search('_[a-z]*_', x).group()[1:-1], axis ="index")


# In[121]:


sm_mean = sm_c.groupby(by=sm_c.index, axis=0).mean().round(decimals=2)
lg_mean = lg_c.groupby(by=lg_c.index, axis=0).mean().round(decimals=2)


# In[162]:


sm_mean_c = sm_mean.append(sm_mean.mean().rename('MEAN').round(decimals=2))
lg_mean_c = lg_mean.append(lg_mean.mean().rename('MEAN').round(decimals=2))


# In[172]:


columns_summary = [('MEAN', m) for m in metrics]
sm_summary = sm_mean_c.groupby(level=1,axis=1).mean().round(decimals=2)
lg_summary = lg_mean_c.groupby(level=1,axis=1).mean().round(decimals=2)
sm_summary.columns=pd.MultiIndex.from_tuples(columns_summary)
lg_summary.columns=pd.MultiIndex.from_tuples(columns_summary)


# In[179]:


pd.concat([sm_mean_c, sm_summary], axis = 1).to_excel("results_en_core_web_sm.xlsx")
pd.concat([lg_mean_c, lg_summary], axis = 1).to_excel("results_en_core_web_lg.xlsx")


# # CATEGORY ANALYSIS

# ### functions for reading the key and response files and converting them to entity dataframes

# In[527]:


def handle_open(annotation, holder):
    copy_ann = annotation
    if copy_ann.count('(') == 0:
        pass
    elif copy_ann.count('(') == 1:
        entity = int(re.compile('\(\d*').search(copy_ann).group()[1:])
        holder.append(entity)
    else:
        copy_ann = copy_ann[copy_ann.find('('):]
        entity = int(re.compile('\(\d*').search(copy_ann).group()[1:])
        holder.append(entity)
        copy_ann = copy_ann[1:][copy_ann[1:].find('('):]
        handle_open(copy_ann, holder)
        
def handle_close(annotation, holder, ready_to_go):
    copy_ann = annotation
    if copy_ann.count(')') == 0:
        pass
    elif copy_ann.count(')') == 1:
        entity = int(re.compile('\d*\)').search(copy_ann).group()[:-1])
        holder.remove(entity)
        ready_to_go.append(entity)
    else:
        entity = int(re.compile('\d*\)').search(copy_ann).group()[:-1])
        holder.remove(entity)
        ready_to_go.append(entity)        
        copy_ann = copy_ann[re.search('\d*\)', copy_ann).end():]
        handle_close(copy_ann, holder, ready_to_go)
        
def annotation_to_entities(df):

    entities = []
    active_entities = []
    ready_entities = []
    holder = {}

    for i in range(df.shape[0]):
        annotation = df.iloc[i,2]
        token = df.iloc[i,1]
        #open
        handle_open(annotation, active_entities)
        for entity in active_entities:
            if entity not in holder.keys():
                holder[entity] = [token, i]
            else:
                holder[entity][0] += ' ' + token
        #close
        handle_close(annotation, active_entities, ready_entities)
        for entity in ready_entities:
            entities.append([holder[entity][0], entity, holder[entity][1], i])
            del holder[entity]
        ready_entities = []
            
    return pd.DataFrame(entities,columns=['mention','entity','start','end']).sort_values(by=['start','end']) 


# ### Coref annotations (key and response)

# In[541]:


all_coref_paths = {'key' : {'start': 'formatted_files/key_files/', 'end': '.key'},
                  'resp_sm' : {'start': 'formatted_files/response_files/en_core_web_sm/', 'end': '.response'},
                  'resp_lg' : {'start': 'formatted_files/response_files/en_core_web_lg/', 'end': '.response'}}

crf_ann = {}
for path in all_coref_paths:
    crf_ann[path] = {}
    for name in file_names:
        crf_ann[path][name] = pd.read_csv(all_coref_paths[path]['start'] + name + all_coref_paths[path]['end'],
                   sep = '\t',
                   skipfooter = 1,
                   quoting = 3,
                   header = None,
                   skiprows = 1,
                   engine = 'python')


# ### Create entities dataframes (key and response)

# In[544]:


entities = {}
for path in crf_ann:
    entities[path] = {}
    for name in file_names:
        entities[path][name] = annotation_to_entities(crf_ann[path][name])


# # Assigning categories

# ### creating lists of pronouns for categories

# In[639]:


definite = ['the']
demonstrative = ['this', 'that', 'those', 'these', 'here', 'there', 'such', 'none', 'neither']

#following are just meant to be alone
third_pronouns = ['he', 'him', 'his', 'himself',
                  'she', 'her', 'hers', 'herself',
                  'it', 'its', 'itself',
                  'they', 'them', 'their', 'theirs', 'themself', 'themselves']
speech_pronouns = ['i', 'me', 'my', 'mine', 'myself',
                  'you', 'your', 'yours', 'yourself', 'yourself', 'yourselves',
                  'we', 'us', 'our', 'ours', 'ourself', 'ourselves']

proper_names = []


# ### proper names from PENN POS annotations, speech tag = NNP

# In[640]:


path_pos_begin = 'gum-master/paula/GUM/'
path_pos_end = '.tok_penn_pos.xml'


# In[666]:


temporary_nnp = []
for name in file_names:
    hold_nnp = ''
    tree = ET.parse(path_pos_begin + name + '/' + name + path_pos_end)
    root = tree.getroot()
    for i in range(crf_ann['key'][name].shape[0]):
        if root[1][i].attrib['value'] == 'NNP':
            temporary_nnp.append(crf_ann['key'][name].iloc[i,1])
            if hold_nnp == '':
                hold_nnp = crf_ann['key'][name].iloc[i,1]                
            else:
                hold_nnp += ' ' + crf_ann['key'][name].iloc[i,1]
                temporary_nnp.append(hold_nnp)
        else:
            if hold_nnp != '':
                temporary_nnp.append(hold_nnp)
                hold_nnp = ''
proper_names = list(set(temporary_nnp))               


# ### Actual assigning

# In[667]:


for path in entities:
    for name in file_names:
        entities[path][name]['refexpr'] = ['speech' if row.lower() in speech_pronouns
                        else 'third' if row.lower() in third_pronouns 
                        else 'proper' if row in proper_names
                        else 'definite' if re.match(r'^the', row.lower())
                        else 'demonstrative' if re.compile(r"^"+'|'.join(demonstrative)+r".*").match(row.lower())
                        else 'other'
                        for row in entities[path][name]['mention']]


# In[689]:


summary = {}
for path in entities:
    summary[path] = []
    for name in file_names:
        summary_series = entities[path][name]['refexpr'].value_counts()
        summary_series.name = name
        summary[path].append(summary_series)


# In[699]:


summary_df = {}
for path in summary:
    summary_df[path] = pd.concat(summary[path], axis=1, keys=[s.name for s in summary[path]], sort = True)


# In[741]:


summary_sum = {}
for path in summary_df:
    summary_sum[path] = summary_df[path].T.rename(lambda x : re.search('_[a-z]*_', x).group()[1:-1], axis = 'index')
    temporary_copy = summary_sum[path].copy()
    summary_sum[path] = temporary_copy.groupby(by=temporary_copy.index, axis=0).sum().round()
    temporary = summary_sum[path].copy().append(summary_sum[path].sum().rename('SUM'))
    summary_sum[path] = temporary
    summary_sum[path]['SUM'] = temporary.sum(axis=1) 


# In[769]:


sum_percent = {}
summary_cols = ['speech','third','demonstrative','definite', 'proper', 'other']
summary_cols_2 = ['speech','third','demonstrative','definite', 'proper', 'other', 'SUM']

for path in summary_sum:
    multi_col_category = [(path, cat) for cat in summary_cols_2]
    sum_percent[path] = summary_sum[path].copy()[summary_cols_2]
    sum_percent[path][summary_cols] = (sum_percent[path][summary_cols].div(sum_percent[path]['SUM'].values,axis=0)                                       *100).round(decimals=2)
    temp = sum_percent[path].copy()
    temp.columns = pd.MultiIndex.from_tuples(multi_col_category)
    sum_percent[path] = temp


# In[772]:


pd.concat([sum_percent[path] for path in sum_percent], axis=1).to_excel("summary_categories.xlsx")


# ### Calculating error mention TP, FN, FP

# In[874]:


import datacompy
categories = ['speech','third','demonstrative','definite', 'proper', 'other']


# In[883]:


def error_mention(model, name, category, store):
    
    key = entities['key'][name]
    resp = entities[model][name]
    
    key_base=datacompy.Compare(key.loc[key['refexpr'] == category],resp,join_columns='mention')
    FN = key_base.df1_unq_rows.shape[0]
    TP1 = key.loc[key['refexpr'] == category].shape[0] - key_base.df1_unq_rows.shape[0]
    
    resp_base=datacompy.Compare(key,resp.loc[resp['refexpr'] == category],join_columns='mention')
    FP = resp_base.df2_unq_rows.shape[0]
    TP2 = resp.loc[resp['refexpr'] == category].shape[0] - resp_base.df2_unq_rows.shape[0]
    
    if TP2 != TP1:
        raise Exception("TP based on key and TP based on resp don'nt match")
    
    store['TP'] = TP1
    store['FP'] = FP
    store['FN'] = FN
    if (TP1 + FN) == 0:
        store['recall'] = 0
    else:
        store['recall'] = round((TP1 / (TP1 + FN))*100, 2)
    if (TP2 + FP) == 0:
        store['precision'] = 0
    else:
        store['precision'] = round((TP2 / (TP2 + FP))*100, 2)
    if (2*TP1 + FP + FN) == 0:
        store['F1'] = 0
    else:
        store['F1'] = round((2*TP1/(2*TP1 + FP + FN))*100, 2)


# In[884]:


mention_errors = {}
for model in ['resp_sm', 'resp_lg']:
    mention_errors[model] = {}
    for name in file_names:
        mention_errors[model][name] = {}
        for category in categories:
            mention_errors[model][name][category] = {}
            error_mention(model, name, category, mention_errors[model][name][category])


# In[899]:


pd.DataFrame.from_dict(mention_errors['resp_sm']['GUM_academic_art']).T[['precision', 'recall', 'F1']].T


# In[903]:


mention_errors['resp_sm']['GUM_academic_art']['speech']


# In[904]:


errors = {}
for path in mention_errors:
    errors[path] = {}
    for name in file_names:
        errors[path][name] = []
        for category in categories:
            for measure in ['recall', 'precision', 'F1']:
                errors[path][name].append(mention_errors[path][name][category][measure])
        


# In[917]:


multicolssm = [tuple(['sm',c,m]) for c in categories for m in ['r', 'p', 'f1']]
multicolslg = [tuple(['lg',c,m]) for c in categories for m in ['r', 'p', 'f1']]

errors_sm = pd.DataFrame.from_dict(errors['resp_sm']).T
errors_lg = pd.DataFrame.from_dict(errors['resp_lg']).T
errors_sm.columns=pd.MultiIndex.from_tuples(multicolssm)
errors_lg.columns=pd.MultiIndex.from_tuples(multicolslg)


# In[928]:


comp = pd.concat([errors_sm, errors_lg], axis=1)
complete = comp.rename(lambda x:re.search('_[a-z]*_', x).group()[1:-1], axis ="index")
complete_mean = complete.groupby(by=complete.index, axis=0).mean().round(decimals=2)
complete_mean.T.to_excel("mention_errors_by_categories.xlsx")
#sm_mean = sm_c.groupby(by=sm_c.index, axis=0).mean().round(decimals=2)


# In[940]:


complete_mean.T


# In[ ]:




