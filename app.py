import streamlit as st
import pandas as pd
from inference import *
import utils

list_models = {}
list_models_name = []
record_columns = ["Issue Key", "Title", "Description", "Story Points"]

for pname in datasetDict:
    list_models_name.append(pname)
    list_models[pname] = DeepSE(pname, max_len=MAX_LEN)

if __name__ == '__main__':
    st.set_page_config(page_title="Story Point Estimaton System", layout="wide")
    st.image('./static/images/TPS_AI.png')
    st.title('Story Point Estimaton System')
    st.write('This system takes input of title and description of an issue and estimates the number of story points required to complete it.')

    selected_models = st.multiselect('Select models', options=list_models_name)
    title = st.text_input('Title (max length: 100)')
    descr = st.text_area('Description (max length: 100)')

    if st.button('Estimate'):
        prediction = 0
        histories = []

        if title == "" or descr == "":
            st.error('Please fill in all above fields!')
        else:
            with st.spinner(text='Estimating in progress...'):
                for model_name in selected_models:
                    sp, history = list_models[model_name].inference([title], [descr], return_history=True)
                    prediction += sp[0]
                    histories += history[0]

                st.success('Done')

            prediction = utils.nearest_fib(prediction / len(selected_models))
            histories_df = pd.DataFrame(histories, columns=record_columns)

            st.header('Story points: %d' % prediction)
            st.table(histories_df)
