import pandas as pd
import os

path = '/home/hail09/recsys/book_review'
feature = 'user_network_attr.csv'
data = 'BX-Book-Ratings.csv'
df_f = pd.read_csv(os.path.join(path, feature))
df = pd.read_csv(os.path.join(path, data))

total = pd.merge(df,df_f, how='left',on='User-ID')

total = total.loc[:, ['User-ID', 'ISBN', 'Age', 'age_bin',
       'eigen_rated', 'deg_rated', 'eigen_imp', 'deg_imp', 'Book-Rating']]


new_total = pd.read_csv(os.path.join(path,'total.csv'))
print()
mu = new_total['eigen_rated'].mean()
new_total['eigen_bin']=new_total['eigen_rated'].apply(lambda x: 1 if x>= mu else 0)

age_mu = new_total[new_total.Age<=100]['Age'].mean()


new_total['Age']= new_total['Age'].apply(lambda x: age_mu if (x >= 95)  else x)

new_total['Age'].apply(lambda x: (x/10)-1 ).astype(int)
new_total['age_bin']=new_total['Age'].apply(lambda x: (x/10)-1).astype(int)


new_total = new_total.loc[:, ['User-ID', 'ISBN',  'age_bin', 'eigen_bin','Age',
       'eigen_rated', 'deg_rated', 'eigen_imp', 'deg_imp', 'Book-Rating']]

# new_total.to_csv(os.path.join(path,'total.csv'),index=False)