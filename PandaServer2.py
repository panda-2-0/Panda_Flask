import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from flask import Flask, request,jsonify
from tensorflow import keras
from konlpy.tag import Okt
import numpy as np
import tensorflow
import random
import json
okt = Okt()

################################################################
# 챗봇 분석 전에 가져올  항목들
with open('intents.json', 'rb') as file:
    file_content = file.read().decode('utf-8')
data = json.loads(file_content)
# 챗봇 분석 전에 처리할 항목들
words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = okt.morphs(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [okt.morphs(w.lower())[0] for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)
################################################################

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/hello')
def helloPy():
    return 'hello'

@app.route('/api/sendAssociation',methods=['GET','POST'])
def receiveData():
    recData=request.get_json()

    # 'otherList' 값 가져오기
    myList= recData['myList']
    otherList = recData['otherList']

    # 값들을 '/'로 분리하여 2차원 배열로 만들기(다른사람 것들)
    values_array = []
    for email, value in otherList.items():
        value_list = value.split('/')
        values_array.append(value_list)

    # 값들을 정수로 형변환
    for value_list in values_array:
        for i in range(len(value_list)):
            value_list[i] = int(value_list[i])


    print(values_array)
    print(set(myList))

    te=TransactionEncoder()
    te_ary=te.fit(values_array).transform(values_array)
    df=pd.DataFrame(te_ary,columns=te.columns_)
    #사용자 리스트
    userSet=set(myList)


    frequent_itemsets=apriori(df,min_support=0.1, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.1)
    result = rules[['lift', 'antecedents', 'consequents']].values

    df_result = pd.DataFrame(result, columns=['lift', 'antecedents', 'consequents'])  #향상도, 인과관계 변수2개(antecedents,consequents)만 뽑아냄
    sorted_result = df_result.sort_values(by='lift', ascending=False).values  #향상도 순으로 정렬

    recommendSet=[]

    for items in sorted_result:
        if set(items[1]).issubset(userSet) and items[0]>1:    #향상도가 1이상 이고, 사용자가 구매or조회한 제품 선택
            recommendSet.append(items[2])

    normal_array = [list(item) for item in recommendSet]  #frozenset인거 일반 list타입으로 변경

    list2 = sum(normal_array, [])   #1차원 배열로 만듦

    list3=list(set(list2))  #중복 제거과정
    print(list3)


    data = {
        'list': list3
    }
    return jsonify(data)


@app.route('/api/classification', methods=['GET', 'POST'])
def classification():
    received_data = request.get_json()
    img_data = received_data['img'] # 사용자가 입력한 이미지

    model = keras.models.load_model("best-cnn-model.h5")    # 모델 불러오기
    t = img_data.reshape(-1, 32, 32, 3) # 모델의 입력에 맞도록 reshape
    predict_result = model.predict(t)   # 이미지 데이터로 모델에 예측
    # 예측 결과 중 가장 답인 확률이 높은 것 반환
    data = {
        'predictResult': np.argmax(predict_result)
    }
    return jsonify(data)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = okt.morphs(s)
    s_words = [okt.morphs(word.lower())[0] for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag).reshape(1, -1)

@app.route('/chatbot',methods=['GET','POST'])
def chatbotData():

    inp = request.get_data(as_text=True)
    print('전달된 문자열:', inp)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [okt.morphs(w.lower())[0] for w in doc]
        # print(wrds)
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    tensorflow.compat.v1.reset_default_graph()
    model = keras.models.load_model("model.h5")

    results = model.predict([bag_of_words(inp, words)])

    max_prob = max(results[0])

    if max_prob < 0.45:
        return '이해가 가질 않습니다'
    else:
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        message = random.choice(responses)
        print(message)

        return message

if __name__ == '__main__':
    app.run()
