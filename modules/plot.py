import plotly.graph_objects as go
import numpy as np

def plot_results(
        model,
        train_x,
        train_y,
        valid_x,
        valid_y,
        save_plot: bool = False,
):

    model.eval()
    model.likelihood.eval()
    MAE_test = np.abs(model(valid_x).mean.detach().cpu().numpy() - valid_y.detach().cpu().numpy())
    MAE_test = np.mean(MAE_test)

    MAE_train = np.abs(model(train_x).mean.detach().cpu().numpy() - train_y.detach().cpu().numpy())
    MAE_train = np.mean(MAE_train)

    train_pred = model(train_x).mean.detach().cpu().numpy()
    valid_pred = model(valid_x).mean.detach().cpu().numpy()

    min = np.min(np.concatenate((train_pred, valid_pred)))
    max = np.max(np.concatenate((train_pred, valid_pred)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = np.linspace(min,max, 10), y = np.linspace(min,max, 10), mode = 'lines', line=dict(color='grey'), showlegend=False))
    fig.add_trace(go.Scatter(x = train_pred, y = train_y.detach().cpu().numpy(), mode = 'markers', name= f'MAE<sub>train</sub> = {MAE_train*1000:.3f} meV', marker = dict(size = 15, color = '#636EFA')))
    fig.add_trace(go.Scatter(x = valid_pred, y = valid_y.detach().cpu().numpy(), mode = 'markers',  name= f'MAE<sub>valid</sub>  = {MAE_test*1000:.3f} meV', marker = dict(size = 15, color = '#FFA15A')))

    fig.update_xaxes(title = 'E<sub>model</sub>, eV',
                     ticklabelstep=2,
                     title_font=dict(size=25),
                     tickfont=dict(size=25),
                     automargin=True,
                     #range=[-2.99, -1.71]
                     )
    fig.update_yaxes(title = 'E<sub>DFT</sub>, eV',
                     ticklabelstep=2,
                     automargin=True,
                     title_font=dict(size=25),
                     tickfont=dict(size=25),
                     #range=[-2.99, -1.71]
                     )
    fig.update_layout(width = 800, height = 600,
                      margin=dict(
            #l=50,
            r=50,
            #b=50,
            t=50,
            pad=4
        ),
                      font=dict(
                      family="Arial",
                      size=23,)
                     )
    fig.update_legends(x = 0.77, y = 0.95)

    if save_plot:
        fig.write_image('gpr_accuracy.pdf')

    fig.show()