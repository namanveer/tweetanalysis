#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tweepy


# In[2]:


pip install textblob


# In[3]:



pip install wordcloud


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from textblob import TextBlob
from wordcloud import WordCloud


# In[5]:


import plotly.graph_objects as go
import plotly.express as px


# In[6]:


Trump=pd.read_csv(r"C:\Users\naman\Desktop\Capstone 2k20\US Election using twitter sentiment\Trumpall2.csv", encoding='latin-1')
Biden=pd.read_csv(r"C:\Users\naman\Desktop\Capstone 2k20\US Election using twitter sentiment\Bidenall2.csv", encoding='latin-1')


# In[8]:


Trump.head()
Biden.head()


# In[9]:



Biden['text'][500]


# In[10]:


Trump['text'][10]


# In[11]:


text_blob_object1 = TextBlob(Trump['text'][10])
print(text_blob_object1.sentiment)


# In[13]:


text_blob_object2 = TextBlob(Biden['text'][500])
print(text_blob_object2.sentiment)


# In[15]:


text_blob_object2 = TextBlob(Biden['text'][100])
print(text_blob_object2.sentiment)


# In[16]:


def find_pol(review):
    return TextBlob(review).sentiment.polarity

Trump['Sentiment_Polarity'] = Trump['text'].apply(find_pol)
Trump.tail()


# In[17]:


def find_pol(review):
    return TextBlob(review).sentiment.polarity

Biden['Sentiment_Polarity'] = Biden['text'].apply(find_pol)
Biden.tail()


# In[18]:


Trump['Expression Label'] = np.where(Trump['Sentiment_Polarity']>0,'positive', 'negative')
Trump['Expression Label'][Trump.Sentiment_Polarity ==0] = "Neutral"
Trump.tail()


# In[19]:


Biden['Expression Label'] = np.where(Biden['Sentiment_Polarity']>0,'positive', 'negative')
Biden['Expression Label'][Biden.Sentiment_Polarity ==0] = "Neutral"
Biden.tail()


# In[20]:


new1 = Trump.groupby('Expression Label').count()
x = list(new1['Sentiment_Polarity'])
y = list(new1.index)
tuple_list = list(zip(x,y))
df = pd.DataFrame(tuple_list, columns=['x','y'])
df['color'] = 'blue'
df['color'][1] = 'red'
df['color'][2] = 'green'
fig = go.Figure(go.Bar(x=df['x'],
                y=df['y'],
                orientation ='h',
                marker={'color': df['color']}))
fig.update_layout(title_text='Trump\'s Reviews Analysis' )
fig.show()


# In[21]:


new2 = Biden.groupby('Expression Label').count()
x = list(new2['Sentiment_Polarity'])
y = list(new2.index)
tuple_list = list(zip(x,y))
df = pd.DataFrame(tuple_list, columns=['x','y'])
df['color'] = 'blue'
df['color'][1] = 'red'
df['color'][2] = 'green'
fig = go.Figure(go.Bar(x=df['x'],
                y=df['y'],
                orientation ='h',
                marker={'color': df['color']}))
fig.update_layout(title_text='Biden\'s Reviews Analysis' )
fig.show()


# In[22]:


reviews1 = Trump[Trump['Sentiment_Polarity'] == 0.0000]
reviews1.shape
cond1 = Trump['Sentiment_Polarity'].isin(reviews1['Sentiment_Polarity'])
Trump.drop(Trump[cond1].index, inplace = True)
Trump.shape 


# In[23]:


reviews2 = Biden[Biden['Sentiment_Polarity'] == 0.0000]
reviews2.shape
cond2 = Biden['Sentiment_Polarity'].isin(reviews1['Sentiment_Polarity'])
Biden.drop(Biden[cond2].index, inplace = True)
Biden.shape


# In[24]:


np.random.seed(10)
remove_n =324
drop_indices = np.random.choice(Trump.index, remove_n, replace=False)
df_subset_trump = Trump.drop(drop_indices)
df_subset_trump.shape


# In[25]:


np.random.seed(10)
remove_n =31
drop_indices = np.random.choice(Biden.index, remove_n, replace=False)
df_subset_biden = Biden.drop(drop_indices)
df_subset_biden.shape


# In[26]:


sns.distplot(df_subset_trump['Sentiment_Polarity'])
sns.boxplot([df_subset_trump.Sentiment_Polarity])
plt.show()


# In[27]:


sns.distplot(df_subset_biden['Sentiment_Polarity'])
sns.boxplot([df_subset_biden.Sentiment_Polarity])
plt.show()


# In[28]:


count_1 = df_subset_trump.groupby('Expression Label').count()
print(count_1)
negative_per1 = (count_1['Sentiment_Polarity'][0]/1000)*100
positive_per1 = (count_1['Sentiment_Polarity'][1]/1000)*100


# In[29]:


count_2 = df_subset_biden.groupby('Expression Label').count()
print(count_2)
negative_per2 = (count_2['Sentiment_Polarity'][0]/1000)*100
positive_per2 = (count_2['Sentiment_Polarity'][1]/1000)*100


# In[30]:


Politicians = ['Donald Trump', 'Joe Biden']
lis_pos = [positive_per1, positive_per2]
lis_neg = [negative_per1, negative_per2]

fig = go.Figure(data=[
    go.Bar(name='Positive', x=Politicians, y=lis_pos),
    go.Bar(name='Negative', x=Politicians, y=lis_neg)
])


# In[31]:


fig.update_layout(barmode='group')
fig.show()


# In[32]:


most_positive1 = df_subset_trump[df_subset_trump.Sentiment_Polarity == 1].text.head()
pos_txt1 = list(most_positive1)
pos1 = df_subset_trump[df_subset_trump.Sentiment_Polarity == 1].Sentiment_Polarity.head()
pos_pol1 = list(pos1)
fig = go.Figure(data=[go.Table(columnorder = [1,2], 
                               columnwidth = [50,400],
                               header=dict(values=['Polarity','Most Positive Replies on Trump\'s Handle'],
                               fill_color='paleturquoise',
                               align='left'),
               cells=dict(values=[pos_pol1, pos_txt1],
                               fill_color='lavender',
                               align='left'))])
 
fig.show()


# In[33]:


most_negative1 = df_subset_trump[df_subset_trump.Sentiment_Polarity == -1].text.head()
neg_txt1 = list(most_negative1)
neg1 = df_subset_trump[df_subset_trump.Sentiment_Polarity == -1].Sentiment_Polarity.head()
neg_pol1 = list(neg1)
fig = go.Figure(data=[go.Table(columnorder = [1,2],
                               columnwidth = [50,400],
                               header=dict(values=['Polarity','Most Negative Replies on Trump\'s handle'],
                               fill_color='paleturquoise',
                               align='left'),
                cells=dict(values=[neg_pol1, neg_txt1],
                           fill_color='lavender',
                           align='left'))])

fig.show()


# In[34]:


most_positive2 = df_subset_biden[df_subset_biden.Sentiment_Polarity == 1].text.tail()
pos_txt2 = list(most_positive2)
pos2 = df_subset_biden[df_subset_biden.Sentiment_Polarity == 1].Sentiment_Polarity.tail()
pos_pol2 = list(pos2)
fig = go.Figure(data=[go.Table(columnorder = [1,2],
                               columnwidth = [50,400],
                               header=dict(values=['Polarity','Most Positive Replies on Biden\'s handle'],
                               fill_color='paleturquoise',
                               align='left'),
                cells=dict(values=[pos_pol2, pos_txt2],
                           fill_color='lavender',
                           align='left'))])

fig.show()


# In[35]:


most_negative2 = df_subset_biden[df_subset_biden.Sentiment_Polarity == -1].text.head()
neg_txt2 = list(most_negative2)
neg2 = df_subset_biden[df_subset_biden.Sentiment_Polarity == -1].Sentiment_Polarity.head()
neg_pol2 = list(neg2)
fig = go.Figure(data=[go.Table(columnorder = [1,2],
                               columnwidth = [50,400],
                               header=dict(values=['Polarity','Most Negative Replies on Biden\'s handle'],
                               fill_color='paleturquoise',
                               align='left'),
                cells=dict(values=[neg_pol2, neg_txt2],
                           fill_color='lavender',
                           align='left'))])

fig.show()


# In[36]:


text = str(df_subset_biden.text)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_font_size=100, max_words=500, scale=10, relative_scaling=.6, background_color="black", colormap = "rainbow").generate(text)
# Display the generated image:
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[38]:


text = str(Biden.text)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_font_size=100, max_words=500,scale=10,relative_scaling=.6,background_color="black", colormap = "rainbow").generate(text)
# Display the generated image:
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[39]:


text = str(df_subset_trump.text)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_font_size=100, max_words=500, scale=10, relative_scaling=.6, background_color="black", colormap = "rainbow").generate(text)
# Display the generated image:
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[40]:


text = str(Trump.text)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_font_size=100, max_words=500,scale=10,relative_scaling=.6,background_color="black", colormap = "rainbow").generate(text)
# Display the generated image:
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[41]:


labels =  ['Negative_Trump', 'Negative_Biden'] 
sizes = lis_neg
explode = (0.1, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle=90)
ax1.set_title('Negative tweets on both the handles')
plt.show()


# In[42]:


labels =  ['Positive_Trump', 'Positive_Biden'] 
sizes = lis_pos
explode = (0.1, 0.1)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle=90)
ax1.set_title('Positive tweets on both the handles')
plt.show()


# In[ ]:




