from sklearn.manifold import TSNE
import os
import plotly.express as px
import numpy as np


def T_sne_train_or_load(Dim_reduction_tech,data,n_components,early_exaggeration):
    
    os.chdir('./saved')

    TSNE_file = 'Tsne_XT_' + Dim_reduction_tech + '_9.npy'
    Tsne_data = 'Tsne_XT_' + Dim_reduction_tech + '_9'

    if os.path.exists(TSNE_file):
        Tsne_data=np.asarray(np.load(TSNE_file,allow_pickle='TRUE').tolist(),dtype=np.float32)
    else:
        Model_ = TSNE(n_components = n_components, early_exaggeration=early_exaggeration, learning_rate='auto', random_state=0)

        Tsne_data= Model_.fit_transform(data)

        np.save(TSNE_file,Tsne_data)
        
    return Tsne_data

def plot_Tsne(Tsne_data,label,title):
    fig = px.scatter(x=Tsne_data[:,0], y=Tsne_data[:,1], color=label)

    fig.update_traces(hoverinfo="all")
    fig.update_layout(
            # update layout with titles
            title={
                "text": title,
                "x": 0.5,
            },
        )

    fig.update_layout(height=900,showlegend=True)
    return fig