from flask import Flask,render_template,request
import  pickle
import numpy as np

app = Flask(__name__)

popular_df1=pickle.load(open('popular1.pkl','rb'))
pt1=pickle.load(open('pt1.pkl','rb'))
new1=pickle.load(open('new1.pkl','rb'))
similarity_score1=pickle.load(open('similarity_score1.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html',
                           city=list(popular_df1['City'].values),
                           place=list(popular_df1['Place'].values),
                           review=list(popular_df1['Review'].values),
                           votes=list(popular_df1['num_ratings'].values),
                           rating=list(popular_df1['avg_ratings'].values)
                           )
@app.route('/recommend')
def recommend_place():
    return render_template('recommend.html')

@app.route('/recommend_places',methods=['POST'])
def recommend():
    user_input=request.form.get('user_input')
    index = np.where(pt1.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_score1[index])), key=lambda x: x[1], reverse=True)[1:6]

    data = []
    for i in similar_items:
        item = []
        temp_df = new1[new1['City'] == pt1.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('City')['Place'].values))
        item.extend(list(temp_df.drop_duplicates('City')['Review'].values))
        item.extend(list(temp_df.drop_duplicates('City')['Rating'].values))
        data.append(item)

    print(data)
    return render_template('recommend.html',data=data)

if __name__ == '__main__':
    app.run(debug=True)









