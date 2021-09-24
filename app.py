import streamlit as st
from inference import list_models_name, get_result

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
                prediction, histories = get_result(title, descr, selected_models)
                st.success('Done')

            st.header('Story points: %d' % prediction)
            if histories.shape[0] == 0:
                st.warning('No similar ticket is found in the project historical data. Prediction might not be accurate!')
            st.table(histories)
