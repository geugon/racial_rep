import streamlit as st
import os

from train import build_model

@st.cache
def load_model(weight_file):
    model = build_model()
    model.load_weights(weight_file)
    return model

@st.cache
def predict(model, white, black, latino, asian, native, other):
    return model.predict([[white/100, black/100, latino/100, asian/100, native/100]]) 


def main():
    st.title('Predicting Ethnicity of Representation')

    model = load_model('weights.h5')

    black = st.slider('Black %', 0, 100, 13)
    latino = st.slider('Latino %', 0, 100, 12)
    asian = st.slider('Asian %', 0, 100, 4)
    native = st.slider('Native %', 0, 100, 1)
    other = st.slider('Other %', 0, 100, 2)
    white = 100 - black - latino - asian - native - other
    st.write("White:", white, "%")
    if white<0:
        st.write("Total Population over 100%!  Adjust Settings")
    else:        
        st.write(predict(model, white, black, latino, asian, native, other))

if __name__ == '__main__':
    main()