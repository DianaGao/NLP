#_________________IMPORTING_LIBRARIES_____________________________________________________________________
import plotly
import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import sqlite3
import sqlalchemy
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from alpha_vantage.techindicators import TechIndicators
import json
import requests
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
from datetime import datetime, timedelta
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import base64
#_________________STRUCTURED_DATA:_BOLLINGER_BANDS_____________________________________________________________________

conn = sqlite3.connect('a1.db')
style.use('fivethirtyeight')
API_URL = "https://www.alphavantage.co/query"
#TAKING DATA FROM API AS JSON 
data = { "function": "TIME_SERIES_INTRADAY", 
        "symbol": "AMZN",
        "interval" : "60min",
	"outputsize":"full",       
        "datatype": "json", 
        "apikey": "XXX" } 
response = requests.get(API_URL, data)
data=response.json()
data=data['Time Series (60min)']
#USING PANDA TO CREATE DATA FRAMES
df=pd.DataFrame(columns=['date','open','high','low','close','volume'])
for d,p in data.items():
    date=datetime.strptime(d,'%Y-%m-%d %H:%M:%S')
    data_row=[date,float(p['1. open']),float(p['2. high']),float(p['3. low']),float(p['4. close']),int(p['5. volume'])]
    df.loc[-1,:]=data_row
    df.index=df.index+1
data=df.sort_values('date')
data['close']=data['close'].astype(float)
data['5min']=np.round(data['close'].rolling(window=5).mean(),2)
data[['5min','close']].plot()

for item in data['close']:
        x=data['close'].rolling(window=20).mean()  
        y=data['close'].rolling(window=20).std()      
i=0
Upper_Band=[]
Lower_Band=[]
for item in x:
        Upper_Band.append(item+(y[i]*2))
        Lower_Band.append(item-(y[i]*2))
        i=i+1

data['middle'] = x
data['upper'] = Upper_Band
data['lower'] = Lower_Band
print (data['middle'])
#PASSING DATA FRAMES TO DATABASE
data.to_sql('tab', conn, if_exists='replace', index=True)
pd.read_sql('select * from tab', conn)


#___________________PASSING_TO_SQL_DATABASE____________________________________________________________________________

databasesql = data.values
con1 = sqlite3.connect('Struct.db')
with con1:
    cur1 = con1.cursor()    
    cur1.execute("CREATE TABLE IF NOT EXISTS St1(Date TIMESTAMP, Open REAL, High REAL, Low REAL, Close REAL, Volumn REAL, Fivemin REAL, Middle BB, Upper BB, Lower BB)")
    for row in databasesql:
        cur1.execute("INSERT INTO St1 VALUES(?,?,?,?,?,?,?,?,?,?)", tuple(row))
    cur1.execute("DROP TABLE St1")


#____________________OUR_RECOMMENDATION__________________________________________________________________________

recommend = ""
if (Upper_Band[len(Upper_Band)-1] - data['close'][len(data)-1]) > (data['close'][len(data)-1] - Lower_Band[len(Lower_Band)-1]):
    recommend = "BUY"
else:
    recommend = "SELL"

    
#___________________CURRENT_STOCK_PRICE_TABLE__________________________________________________________________

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[0][col]) for col in dataframe.columns
        ]) ]
    )


#___________UNSTRUCTURED_DATA:_SENTIMENT_ANALYSIS,_BOW___________________________________________________________________________

pos=[]
neg=[]
date2=[]
def text_process(mess):
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in mess.split() if word.lower() not in stopwords.words('english')]
    
    # percentage calculation
def percentage(part, whole):
    try:
        return ((100 * float(part))/float(whole))
    except ZeroDivisionError:
        return 0
      
dataset = ""
r=datetime.today()
# Read news of past 1 days from fortune
for i in range (0,1):
    tdy = r - timedelta(days=i)
    tdy = tdy.strftime('%y-%m-%d')
    url = 'https://newsapi.org/v2/everything?q=amazon&sources=fortune&from='+tdy+'&to='+tdy+'&apiKey=5b24a802f83547559d54c63afd500453'
    response = requests.get(url)
    json_data = json.loads(response.text)
#Normalization
    try:
        for item in json_data['articles']:
            if item['content'] != None:
                a=text_process(item['content'])
#Sentiment Analysis
                sid= SentimentIntensityAnalyzer()
            summary = {"positive":0,"neutral":0,"negative":0}
            for x in a: 
                ss = sid.polarity_scores(x)
                if ss["compound"] == 0.0: 
                    summary["neutral"] +=1
                elif ss["compound"] > 0.0:
                    summary["positive"] +=1
                else:
                    summary["negative"] +=1
            positive_score_percentage = percentage(summary['positive'], summary['positive']+summary['negative'])
            negative_score_percentage = percentage(summary['negative'], summary['positive']+summary['negative'])
            dataset += str(item['content'])
            pos.append(positive_score_percentage)
            neg.append(negative_score_percentage)
            date2.append(tdy)
    except KeyError:
        item=[]

#cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
       
#remove the numbers and punctuations,keeping only the letters
review = re.sub('[^a-z A-Z]', ' ', dataset)
#replace with lowercases
review = review.lower()
#remove non significant words 'or and are prepositions ......'
#convert string into list
review = review.split()
#keepin the root of the word = stemming
ps = PorterStemmer()
#go through set is faster than the list
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#join back to a string, join function
review =' '. join(review)
#append to the corpus
corpus.append(review)

text = " ".join(words for words in corpus)
print ("There are {} words in the combination of all review.".format(len(text)))
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('fig.png')
    
##creating the bag of word model
##create sparse matirx
pos=np.array(pos)
neg=np.array(neg)
date2=np.array(date2)

counter_pos = 0
for item in pos:
    if item > 50:
        counter_pos = counter_pos + 1

counter_neg = len(neg) - counter_pos
percent_pos= 0
percent_neg= 0
percent_pos = 100*counter_pos/(counter_pos+counter_neg)
percent_neg = 100*counter_neg/(counter_pos+counter_neg)


#____________________DASHBOARD_DESIGNING__________________________________________________________________

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

image_filename = 'fig.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(children=[
    html.H1(children='Stock Info for AMAZON',),html.H2(children=['Our Recommendation Based on Bollinger Bands Methodology: ', recommend]),html.H4(children='Real Time Stock Prices'),
    generate_table(data), 
    dcc.Textarea(
    placeholder='Enter a value...',
    value=['Our Recommendation Based on Bollinger Bands Methodology: ', recommend],
    style={'width': '100%'}
    ),
    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),

    html.Div(children='''
        
    '''),

    dcc.Graph(
        id='graph1',
        figure={
            'data': [
                {'x': df.date, 'y': df.close, 'type': 'line'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']

                
                }, 'title': 'Closing Price TimeSeries'
            }
        }
        
    ), 
    html.Div(children='''
        
    '''),

    dcc.Graph(
        id='graph2',
        figure={
            'data': [
                {'x': data.date, 'y': data.close, 'type': 'line' ,'name': 'Closing'},{'x': data.date, 'y': data.upper, 'type': 'scatter','name': 'Upper Bollinger Band'},{'x': data.date, 'y': data.lower, 'type': 'scatter','name': 'Lower Bollinger Band'},{'x': data.date, 'y': data.middle, 'type': 'scatter','name': '20 Day Moving Average'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']

                
                }, 'title': 'Bollinger Bands and Running Average'
            }
        }
    ),
    dcc.Graph(
        id='graph3',
        figure={
            'data': [
                {'x': ['Positive%'], 'y': [percent_pos] , 'type': 'bar' ,'name': 'Percentage of positive sentiments', 'mode': 'markers','marker': {'color': 'green'}},
                {'x': ['Negative%'], 'y': [percent_neg] , 'type': 'bar' ,'name': 'Percentage of negative sentiments', 'mode': 'markers','marker': {'color': 'red'}},
                
                

            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']

                
                }, 
                'title': 'Sentiment Analysis'
            }
        }
    )
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


if __name__ == '__main__':
    app.run_server(debug=True)


#▀▀█▀▀ █░░█ █▀▀█ █▀▀▄ █░█   █░░█ █▀▀█ █░░█         █▀▀▄ █▀▀█ ░   ▀▀█▀▀ █░░█ █▀▀█ █▀▀ █▀▀ █▀▀ █▀▀█ █▀▀▄
#░░█░░ █▀▀█ █▄▄█ █░░█ █▀▄   █▄▄█ █░░█ █░░█         █░░█ █▄▄▀ ▄   ░░█░░ █░░█ █▄▄▀ █▀▀ ▀▀█ ▀▀█ █░░█ █░░█
#░░▀░░ ▀░░▀ ▀░░▀ ▀░░▀ ▀░▀   ▄▄▄█ ▀▀▀▀ ░▀▀▀         ▀▀▀░ ▀░▀▀ █   ░░▀░░ ░▀▀▀ ▀░▀▀ ▀▀▀ ▀▀▀ ▀▀▀ ▀▀▀▀ ▀░░▀
#