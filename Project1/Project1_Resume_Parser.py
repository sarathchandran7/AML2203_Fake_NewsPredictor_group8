import pandas as pd
from PyPDF2 import PdfReader
import re
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

reader = PdfReader("Resume_Varun_Sharma_Dataanalyst.pdf")

text = ""

# Using PyPDF package to parse the resume
for i in range(len(reader.pages)):
    page = reader.pages[i]
    text = text + page.extract_text()
text = text.lower()


# Cleaning resume using nltk libraries
def cleanResume(resumeText):

    text = word_tokenize(resumeText)
    stop_words = set(stopwords.words("english"))
    filtered_words = []
    for word in text:
        if word.casefold() not in stop_words:
            filtered_words.append(word)
    wn = nltk.WordNetLemmatizer()
    w = [wn.lemmatize(word) for word in filtered_words]
    return w


x = cleanResume(text)

# create a dictionary with the keywords for each job
job_desc = {'Data_Engineer':['Microsoft','Office','SharePoint','Tableau','JIRA','MIRO','Alteryx',
                              'Qlik', 'Clarity','Planview','Remedy','Computer','Science','Data','Quality','Analysis','Preparation','Modeling','metrics',
                              'Project','Management','Business','Metrics','Dashboards','Change','Incident','Management','quality',
                              'Business','Intelligence','Analytics','Portfolio','Resource','Management','Executive','Management','Presentations','SAFe','Agilist','SA','Practitioner','SP',
                              'Process','Improvement','Scrum Master'],
        'financial_analysts':['Accounting','Principles','Analysis','Balance','Sheet','Budget','Analysis','Business','Analytics','Cost',
                                 'Cost','Management','Data','Analysis','Economics','Finance','Model','Financial','Modeling','Financial','Analysis','Financial','Reporting',
                                 'Forecasting','FP&A (Financial Planning and Analysis)','GAAP (Generally Accepted Accounting Principles)',
                                 'Gross','Margin','IBM','Cognos','Impromptu','Income','Statement','Microsoft','Dynamics','Microsoft','Excel','NetSuite',
                                 'Oracle','Business','Intelligence','Statistics','SAS','Financial','Management'],
        'Supply_chain':['abc','analysis','apics','customer','customs','delivery','distribution','eoq','epq',
                        'fleet','forecast','inventory','logistic','materials','outsourcing','procurement',
                        'reorder point','rout','safety stock','scheduling','shipping','stock','suppliers',
                        'third party logistics','transport','transportation','traffic','supply chain',
                        'vendor','warehouse','wip','work in progress'],
        'Project_management':['administration','agile','budget','cost','direction','feasibility','analysis',
                              'finance','kanban','leader','leadership','management','milestones','planning',
                              'pmi','pmp','problem','project','risk','schedule','scrum','stakeholders'],
        'Data_analytics':['analytics','api','aws','big data','busines','intelligence','clustering','code',
                          'coding','data','database','data','mining','data','science','deep','learning','hadoop',
                          'hypothesis','test','iot','internet','machine','learning','modeling','nosql','nlp',
                          'predictive','programming','python','r','sql','tableau','text mining',
                          'visualuzation']}

# Matching keywords from Job description and Resume
match_dict = {'Data_Engineer': [], 'financial_analysts': [], 'Supply_chain': [],  'Project_management': [], 'Data_analytics': []}

for key, value in job_desc.items():
    for i in value:
        if i in x:
            match_dict[key].append(i)


# counting the matched keywords for the job descriptions
count_dict = {}
for key, value in job_desc.items():
    count_dict[key] = (len(match_dict[key])/len(job_desc[key]))*100

# plotting bar graph and getting the percentage matched for a job_description
df_count = pd.DataFrame.from_dict(count_dict, orient='index')

df_count.plot.bar()
plt.xlabel("Job Title")
plt.ylabel("Resume Matched(in percentage)")
plt.title("Resume Scanning")
plt.xticks(rotation=80)
plt.tight_layout()
plt.show()

















