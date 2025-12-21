#data analysis
import pandas as pd
from factor_analyzer import FactorAnalyzer
import pingouin as pg

#data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

#conceptual model analysis
from pyprocessmacro import Process

#data cleaning and pipeline modules
#from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector


#read in data
data=pd.read_csv("Database_individual_assignment.csv")



#check data
print(data.head())
print(data.dtypes.value_counts())


# convert object string to numeric data typee
objects=data.columns[data.dtypes.eq('object')]

data[objects]=data[objects].apply(pd.to_numeric, errors='coerce')

print(data.dtypes.value_counts())

#check for null values
sum=0
sum_null=0


null_element=data.isnull().sum(axis=0).tolist()
for element in null_element:
    if element ==0:
        sum+=1
    elif element >1:
        sum_null+=1
    

print("Columns without null values:", sum)
print("Columns with null values:",sum_null )

#visualise null values

plt.figure(figsize=(20, 36))
sns.heatmap(data.isnull())
plt.figure('Null values')
plt.show()
#workexperience null value due to no work experience thus replace with 0
data['workexperience'].fillna(0, inplace=True)

#Pipeline to clean data

numeric_transformer=Pipeline(steps=[("imputer", SimpleImputer(
    strategy='mean'))])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category"))])
numerical_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor)])

data_cleaned = numerical_pipeline.fit_transform(data)
df=pd.DataFrame(data_cleaned, columns=data.columns, index=data.index)


#check
plt.figure(figsize=(20, 36))
sns.heatmap(df.isnull())
plt.figure('Null values after cleaning')
plt.show()

#remove fixed control variables: age, idnumber, workexperience, gender, nationality
df1=df.iloc[:, 5:]

#DATA ANALYSIS

#Factor Analysis
fa = FactorAnalyzer(rotation='varimax')
fa.fit(df1)
ev, v = fa.get_eigenvalues()

#sort eigenvalues
ev_sorted=sorted(ev)

#display Screen Plot of Factors and Eigenvalues
plt.scatter(range(1,df1.shape[1]+1),ev)
plt.plot(range(1,df1.shape[1]+1),ev)
plt.title('Screen Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# retain all factors up to and including the point at which the plot levels off (slope flattens)
# Defined Factor Analysis ( Number of factors known)

n=50 # number of factors
fa2=FactorAnalyzer(rotation = 'varimax')
fa2.set_params(n_factors=n,rotation='varimax')
fa2.fit(df1)
loadings = fa2.loadings_

# Create column names from 'Factor1' to 'Factor10'
column_names = [f'Factor{i}' for i in range(1, n+1)]

#Factor loadings represent the correlation coefficients between the observed variables and the latent factors. 
loading_df = pd.DataFrame(loadings, columns= column_names,index=[df1.columns])

#How much variance is explained by factor to determine factor importance
factor_var=fa2.get_factor_variance()
total_var=factor_var[2][-1]
print(str("Total Variance is ")+ str(round(total_var*100,2))+str("%"))

#remove columns that do not have a single loading per factor > = 0.5

#find indices where all values are less than 0.5
indices = loading_df.index[((loading_df < 0.45) & (loading_df > -0.45)).all(axis=1)].tolist()
# Convert tuple indices to string
columns_drop = [index[0] for index in indices]

# Drop columns in df1 corresponding to the indices found in loading_df
df1_simple = df1.drop(columns=columns_drop,axis=1)
update_loading= loading_df.drop(columns_drop)

#create new dataframe to display aggregated variables part 1 
df_new=pd.DataFrame()
#baseline
df_new['neg_t0'] = pd.DataFrame(df1_simple.loc[:, 'negb_t0':'negj_t0'].mean(axis=1))
df_new['perf_t0'] = pd.DataFrame(df1_simple.loc[:, 'perfa_t0':'perfd_t0'].mean(axis=1))
df_new['burn_t0'] = pd.DataFrame(df1_simple.loc[:, 'burna_t0':'burni_t0'].mean(axis=1))

#negative affects week 1 - week 4
df_new['neg_t1'] = pd.DataFrame(df1_simple.loc[:, 'nega_t1':'negj_t1'].mean(axis=1))
df_new['neg_t2'] = pd.DataFrame(df1_simple.loc[:, 'nega_t2':'negj_t2'].mean(axis=1))
df_new['neg_t3'] = pd.DataFrame(df1_simple.loc[:, 'nega_t3':'negg_t3'].mean(axis=1))
df_new['neg_t4'] = pd.DataFrame(df1_simple.loc[:, 'nega_t4':'negj_t4'].mean(axis=1))

#social support week 1 - week 4
df_new['support_t1'] = pd.DataFrame(df1_simple.loc[:, 'supporta_t1':'supportd_t1'].mean(axis=1))
df_new['support_t2'] = pd.DataFrame(df1_simple.loc[:, 'supporta_t2':'supportd_t2'].mean(axis=1))
df_new['support_t3'] = pd.DataFrame(df1_simple.loc[:, 'supporta_t3':'supportc_t3'].mean(axis=1))
df_new['support_t4'] = pd.DataFrame(df1_simple.loc[:, 'supporta_t4':'supportd_t4'].mean(axis=1))

#burnout week 1 - week 4
df_new['burn_t1'] = pd.DataFrame(df1_simple.loc[:, 'burng_t1':'burna_t1'].mean(axis=1))
df_new['burn_t2'] = pd.DataFrame(df1_simple.loc[:, 'burng_t2':'burnh_t2'].mean(axis=1))
df_new['burn_t3'] = pd.DataFrame(df1_simple.loc[:, 'burng_t3':'burnh_t3'].mean(axis=1))
df_new['burn_t4'] = pd.DataFrame(df1_simple.loc[:, 'burng_t4':'burnh_t4'].mean(axis=1))

#performance week 1 - week 4
df_new['perf_t1'] = pd.DataFrame(df1_simple.loc[:, 'perfa_t1':'perfd_t1'].mean(axis=1))
df_new['perf_t2'] = pd.DataFrame(df1_simple.loc[:, 'perfa_t2':'perfd_t2'].mean(axis=1))
df_new['perf_t3'] = pd.DataFrame(df1_simple.loc[:, 'perfa_t3':'perfd_t3'].mean(axis=1))
df_new['perf_t4'] = pd.DataFrame(df1_simple.loc[:, 'perfa_t4':'perfd_t4'].mean(axis=1))


# Personality Traits

df_new['Extraversion'] = pd.DataFrame(df1_simple.loc[:, 'extraa':'extraf'].mean(axis=1))
df_new['Conscientiousness'] = pd.DataFrame(df1_simple.loc[:, 'consb':'conse'].mean(axis=1))
df_new['Neurotism'] = pd.DataFrame(df1_simple.loc[:, 'neurb':'neurd'].mean(axis=1))
df_new['Openness'] = pd.DataFrame(df1_simple.loc[:, 'openb':'openf'].mean(axis=1))


#reliability test using cronbach alpha
df2= df_new.loc[:, 'neg_t1':'neg_t4']
df3= df_new.loc[:, 'support_t1':'support_t4']
df4= df_new.loc[:, 'burn_t1':'burn_t4']
df5= df_new.loc[:, 'perf_t1':'perf_t4'] 

neg_reliability=pg.cronbach_alpha(data=df2)
supp_reliability=pg.cronbach_alpha(data=df3)
burn_reliability=pg.cronbach_alpha(data=df4)
perf_reliability=pg.cronbach_alpha(data=df5) #reliability lower than 0.7 but higher than 0.6 could be questionable

extra_reliability=pg.cronbach_alpha(data=df1_simple.loc[:, 'extraa':'extraf'])
consc_reliability=pg.cronbach_alpha(data=df1_simple.loc[:, 'consb':'conse'])
neuro_reliability=pg.cronbach_alpha(data=df1_simple.loc[:, 'neurb':'neurd']) #unacceptable alpha value < 0.5 do not use in analysis
open_reliability=pg.cronbach_alpha(data=df1_simple.loc[:, 'openb':'openf'])

#aggregate main variables for conceptual model 
#create new dataframe to display aggregated values part 2
df_final=df.iloc[:, :5] # control variables

df_final['neg_avg']= pd.DataFrame(df_new.loc[:, 'neg_t1':'neg_t4'].mean(axis=1))
df_final['support_avg']= pd.DataFrame(df_new.loc[:, 'support_t1':'support_t4'].mean(axis=1))
df_final['burn_avg']= pd.DataFrame(df_new.loc[:, 'burn_t1':'burn_t4'].mean(axis=1))
df_final['perf_avg']= pd.DataFrame(df_new.loc[:, 'perf_t1':'perf_t4'].mean(axis=1))

#personality traits (additional control variables)

df_final['Extraversion'] = pd.DataFrame(df1_simple.loc[:, 'extraa':'extraf'].mean(axis=1))
df_final['Conscientiousness'] = pd.DataFrame(df1_simple.loc[:, 'consb':'conse'].mean(axis=1))
df_final['Openness'] = pd.DataFrame(df1_simple.loc[:, 'openb':'openf'].mean(axis=1))



#box plots to visualise outliers
def boxplot(column):
    sns.boxplot(data=df_final,x=df_final[f"{column}"])
    plt.title(f"Boxplot of {column}")
    plt.show()
    
    
#control variables with possible outliers 
boxplot('Extraversion')
boxplot('Conscientiousness')
boxplot('Openness')
boxplot('age')

boxplot('burn_avg') #IV
boxplot('perf_avg') #DV

describe=df_final.describe()

#removing outliers based on boxplot and creating final dataframe for conceptual model
df11 = df_final[(df_final['age']<26) & 
            (df_final['Extraversion']> 3.25) & 
            (df_final['Conscientiousness']>3) & 
            (df_final['Openness']>2.5) & 
            (df_final['Openness']<4.9) &
            (df_final['burn_avg']<3.75)&
            (df_final['perf_avg']>3.5)].copy()


info_df11=df11.info()
describe_df11=df11.iloc[:,1:].describe()

#correlation analysis

correlation_matrix=df11.corr()
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))  # Adjust figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')



# Analysis using pinguoin module

# Model with the mediator (Negative affect)
#process analysis using model template 4 
p = Process(data=df11, 
            model=4,
            x="burn_avg", 
            y="perf_avg",
            m=["neg_avg"],
            controls=["workexperience","gender","age","nationality","Conscientiousness","Extraversion", "Openness"],
            center=True, boot=10000)
print(p.summary())

#moderator (social support) model between X and Mediator (negative affect) using model template 1
p2 = Process(data=df11, 
             model=1, 
             x="burn_avg", 
             y="neg_avg", 
             m="supp_avg",
             controls=["workexperience","gender","age","nationality","Conscientiousness","Extraversion","Openness"],
             center=True, boot=10000)
print(p2.summary())

# Model with moderator (Social Support) - mediator (Negative Affect)
    
p3 = Process(data=df11, 
             model=7, 
             x="burn_avg", 
             y="perf_avg", 
             w="support_avg", 
             m=["neg_avg"], 
             controls=["workexperience","gender","age","nationality","Conscientiousness","Extraversion","Openness"],
             center=True, boot=10000)
print(p3.summary())

