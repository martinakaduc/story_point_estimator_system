from DeepSE import *
import pandas as pd
import utils

list_models = {}
list_models_name = []
record_columns = ["Issue Key", "Title", "Description", "Story Points"]

for pname in datasetDict:
    list_models_name.append(pname)
    list_models[pname] = DeepSE(pname, max_len=MAX_LEN)

def get_result(title, descr, selected_models):
    story_point = 0
    histories = []

    for model_name in selected_models:
        sp, history = list_models[model_name].inference([title], [descr], return_history=True)
        story_point += sp[0]
        histories += history[0]

    story_point = utils.nearest_fib(story_point / len(selected_models))
    histories_df = pd.DataFrame(histories, columns=record_columns)

    return story_point, histories_df
