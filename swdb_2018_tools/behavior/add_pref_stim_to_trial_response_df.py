
import numpy as np
import pandas as pd


def get_mean_sem_trace(group):
    mean_response = np.mean(group['mean_response'])
    sem_response = np.std(group['mean_response'].values) / np.sqrt(len(group['mean_response'].values))
    mean_trace = np.mean(group['trace'])
    sem_trace = np.std(group['trace'].values) / np.sqrt(len(group['trace'].values))
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response,
                      'mean_trace': mean_trace, 'sem_trace': sem_trace})


def annotate_trial_response_df_with_pref_stim(trial_response_df):
    """
    For each cell, gets the trial averaged response per image, for the first flash after the change_time, and identifies which image evoked the max response.
    Creates a column in the trial_response_df called 'pref_stim' that is set to True for trials where the change_image_name corresponds to the preferred stimulus.
    Averages across go and catch trials when taking the mean response per image.
    """
    rdf = trial_response_df.copy()
    rdf['pref_stim'] = False
    mean_response = rdf.groupby(['cell', 'change_image_name']).apply(get_mean_sem_trace)
    m = mean_response.unstack()
    for cell in m.index:
        image_index = np.where(m.loc[cell]['mean_response'].values == np.max(m.loc[cell]['mean_response'].values))[0][0]
        pref_image = m.loc[cell]['mean_response'].index[image_index]
        trials = rdf[(rdf.cell == cell) & (rdf.change_image_name == pref_image)].index
        for trial in trials:
            rdf.loc[trial, 'pref_stim'] = True
    return rdf


if __name__ == '__main__':
    from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
    from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

    #change this for AWS or hard drive
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'
    experiment_id = 672185644
    dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
    analysis = ResponseAnalysis(dataset)

    trial_response_df = annotate_trial_response_df_with_pref_stim(trial_response_df)

