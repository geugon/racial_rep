import streamlit as st
import os
import numpy as np
import pandas as pd

from train import build_model

max_demo = {"white": 97,
            "black": 65,
            "latino": 80,
            "asian": 56,
            "native": 22,
            "other": 20,}

@st.cache
def load_model(weight_file):
    model = build_model()
    model.load_weights(weight_file)
    return model

@st.cache
def predict_multi(model, x):
    raw = np.array([x['white']/100,
                    x['black']/100,
                    x['latino']/100,
                    x['asian']/100,
                    x['native']/100,
                    x['other']/100,
                  ])
    
    # Vary the ratio of each race while holding other in their current ratio
    # ratio[i,j] is what fraction of pop is race j, ignoring race i (ignore diagnol)
    ratios = raw.reshape(-1,1)/(1-raw.reshape(1,-1))

    ref = 1-(np.arange(100)/100)
    data = []
    for i in range(len(raw)):
        inputs = ref.reshape(-1,1)*ratios[:,i].reshape(1,-1)
        inputs[:,i] = 1-ref
        data.append(inputs)
    data.append([raw])
    data = np.concatenate(data)[:,:-1] #model doesn't need "Other" data

    return model.predict(data)


def main():
    st.title('Predicting Ethnicity of Representation')

    model = load_model('weights.h5')

    demograhics = {}
    demograhics["black"] = st.sidebar.slider('Black %', 0, 100, 13)
    demograhics["latino"] = st.sidebar.slider('Latino %', 0, 100, 12)
    demograhics["asian"] = st.sidebar.slider('Asian %', 0, 100, 4)
    demograhics["native"] = st.sidebar.slider('Native %', 0, 100, 1)
    demograhics["other"] = st.sidebar.slider('Other %', 0, 100, 2)
    demograhics["white"] = 100 - demograhics["black"] - demograhics["latino"] - demograhics["asian"] - demograhics["native"] - demograhics["other"]
    st.sidebar.text("White %:" +str(demograhics["white"]))

    for race in max_demo.keys():
        if demograhics[race] > max_demo[race]:
           st.write("No existing congressional district is over {}% {}, results may not be accurate".format(max_demo[race], race))

    if demograhics["white"]<0:
        st.write("Total Population over 100%!  Adjust Settings")
    else:        
        preds = predict_multi(model, demograhics)
        st.write("(set district demographics in sidebar at left)")
        st.write("Probabilities for Representative's Race:")
        st.write("Black",   round(preds[-1,1]*100,1), "%")
        st.write("Hispanic", round(preds[-1,2]*100,1), "%")
        st.write("Asian",   round(preds[-1,3]*100,1), "%")
        st.write("Native",  round(preds[-1,4]*100,1), "%")
        st.write("White",   round(preds[-1,0]*100,1), "%")

        data = np.concatenate([
        preds[  0:100,0],
        preds[100:200,1],
        preds[200:300,2],
        preds[300:400,3],
        preds[400:500,4],
        ]).reshape(5,100).T
    
        df = pd.DataFrame(data,columns="White Black Hispanic Asian Native".split())
        st.write("Probability of Distriubtion of Representation")
        st.line_chart(df)
        st.write("Precent of District of Ethnicity (other races adjusted proportianally)")
        # st.line_chart(preds[  0:100,0])
        # st.line_chart(preds[100:200,1])
        # st.line_chart(preds[200:300,2])
        # st.line_chart(preds[300:400,3])
        # st.line_chart(preds[400:500,4])
        # st.line_chart(preds[500:600,5])


if __name__ == '__main__':
    main()