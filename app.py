import streamlit as st
from inference import *

deep_se_model = DeepSE(emb_weight_file, dict_file, model_file, max_len=MAX_LEN)

if __name__ == '__main__':
    st.image('./static/images/TPS_AI.png')
    st.title('Story Point Estimaton System')
    st.write('This system takes input of title and description of an issue and estimates the number of story points required to complete it.')

    title = st.text_input('Title (max length: 100)')
    descr = st.text_area('Description (max length: 100)')

    if st.button('Estimate'):
        prediction = 0

        if title == "" or descr == "":
            st.error('Please fill in all above fields!')
        else:
            with st.spinner(text='Estimating in progress...'):
                prediction = deep_se_model.inference([title], [descr])[0]
                st.success('Done')

            st.header('Story points: %d' % int(prediction))
