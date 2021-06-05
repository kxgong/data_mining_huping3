import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from math import ceil
%matplotlib inline
data = pd.read_csv("vgsales.csv",encoding="utf-8")
data.head()

print('-------Data statistics-------')
data.info()

data=data.dropna(axis=0, how='any')
#data=data.reset_index(drop=True)
data.info()

def lineplot(df, ylabel, title='Sales by Year', legendsize = 10, legendloc = 'upper left'):
    year = df.index.values
    na = df.NA_Sales
    eu = df.EU_Sales
    jp = df.JP_Sales
    other = df.Other_Sales
    global_ = df.Global_Sales
    

    region_list = [na, eu, jp, other, global_]
    columns = ['NA', 'EU', 'JP', 'OTHER', 'WORLD WIDE']
    
    color_list = ['#dca9a4', '#936275', '#3fe129', '#aca8ad', '#586278']
    for i, region in enumerate(region_list):
        plt.plot(year, region, label = columns[i], linewidth=2, c=color_list[i])

    plt.ylabel(ylabel)
    plt.xlabel('Year')
    plt.title(title)
    plt.legend(loc=legendloc, prop = {'size':legendsize})
    plt.show()
    plt.clf()

lineplot(total_sales_group, title = 'Sales (Year)', ylabel="Sales (Millions)", legendsize = 14)


platGenre = pd.crosstab(data.Platform,data.Genre)
platGenreTotal = platGenre.sum(axis=1).sort_values(ascending = False)
plt.figure(figsize=(8,6))

sns.barplot(y = platGenreTotal.index, x = platGenreTotal.values, orient='h', palette="Set2")
plt.show()


platGenreTotal = platGenre.sum(axis=0).sort_values(ascending = False)
plt.figure(figsize=(8,6))

sns.barplot(y = platGenreTotal.index, x = platGenreTotal.values, orient='h', palette="Set2")
plt.show()

# rank publishers
Publisher_data = data.groupby(['Publisher']).sum().loc[:,"Global_Sales"].sort_values(ascending = False)
print("Number of Pulishers={}".format(len(Publisher_data)))

# remove outliers
Publisher_data = Publisher_data[Publisher_data.values > 100]

# plot
plt.figure(figsize=(8,6))
sns.barplot(y = Publisher_data.index, x = Publisher_data.values, orient='h', palette="Set2")

plt.show()


area_list=["NA_Sales","EU_Sales","JP_Sales","Global_Sales"]
df_show=pd.DataFrame(columns=['Year' ,'Rank1','Rank2','Rank3'])
temp=0
df_area=data[["Year","Genre","Publisher",area_list[i]]]
for year in range(int(df_area["Year"].min()+1),int(df_area["Year"].max()-3)):
    df_eachyear = df_area[df_area['Year'] == year]
    df_eachyear=df_eachyear[["Publisher",area_list[i]]].groupby(by="Publisher").count().sort_values(by=area_list[i])[::-1]
    df_array=df_eachyear.head(3).index.values

    df_array= np.append(year,df_array)

    df_show.loc[temp]=df_array
    temp+=1


df_show.head(int(df_area["Year"].max()-3)-int(df_area["Year"].min()+1))


# split training set
revenue=df[["Year","Global_Sales"]]
revenue=revenue.groupby(by="Year").sum()
train_x=revenue.head(36).index.values.flatten()
train_y=revenue.head(36).values.flatten()
f1 = np.polyfit(train_x, train_y, 5)
p1 = np.poly1d(f1)
yvals = p1(train_x)
years=[]
for i in range(1980,2016):
    years.append(i)
plot_frame=pd.DataFrame(columns=['Year' ,'predicted_sales','real_sales','difference'])

for i in range(0,36):
    temp_array=[]
    temp_array=np.append(temp_array,years[i])
    temp_array=np.append(temp_array,yvals[i])
    temp_array=np.append(temp_array,train_y[i])
    temp_array=np.append(temp_array,round(np.abs(train_y[i]-yvals[i]),2))
    plot_frame.loc[i]=temp_array
    

plot1 = plt.plot(train_x, train_y, 's',label='True values', color='black')
plot2 = plt.plot(train_x, yvals, 'r',label='Predict values' , linewidth=2, color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4, prop = {'size':14}) 
plt.show()


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=200,min_samples_split=6,random_state=10)
train_x = np.array(train_x)
train_y = np.array(train_y)
train_x = train_x.reshape(-1, 1)
train_y = train_y.reshape(-1, 1)
print(type(train_x))
rf_model.fit(train_x,train_y)
y_pred = rf_model.predict(train_x)
n = len(train_x)
p = train_x.shape[1]

plot1 = plt.plot(train_x, train_y, 's',label='', color='black')
plot2 = plt.plot(train_x, y_pred, 'r',label='min_samples_split=6' , linewidth=2, color='green')


rf_model = RandomForestRegressor(n_estimators=200,min_samples_split=2,random_state=10)
rf_model.fit(train_x,train_y)
y_pred = rf_model.predict(train_x)
plot1 = plt.plot(train_x, train_y, 's',label='', color='black')
plot2 = plt.plot(train_x, y_pred, 'r',label='min_samples_split=2' , linewidth=2, color='red')

plt.legend(loc=4, prop = {'size':14}) 

