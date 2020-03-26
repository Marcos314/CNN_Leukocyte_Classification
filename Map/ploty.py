import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv('/home/marcos/Desktop/LAB1_BF.csv')
df.head()


df['text'] = df['uf'] + '<br>Qtd pagamentos: ' + (df['pgt_realizados']/1e3).astype(str) + 'mil'
#qtd estados em determinada situação
limits = [(0,2),(3,14),(15,21),(21,28)]
colors = ["royalblue","crimson","lightseagreen","orange"]
cities = ['acre','brasilia']


fig = go.Figure()

for i in range(len(limits)):
    lim = limits[i]
    df_sub = df[lim[0]:lim[1]]
    #print(df_sub)    
    fig.add_trace(go.Scattergeo(
        #locations = ["Brazil"],
        locationmode = 'USA-states',
        #locationmode = 'USA-states',
        lon = df_sub['lon'],
        lat = df_sub['lat'],
        text = df_sub['text'],
        marker = dict(
            size = df_sub['pgt_realizados']/27,
            color = colors[i],
            line_color='rgb(40,40,40)',
            line_width=0.5,
            sizemode = 'area'            
        ),
        name = '{0} - {1}'.format(lim[0],lim[1])))

fig.update_layout(
        title_text = 'Brazil',
        showlegend = True,
        geo = dict(
            scope = 'south america',
            landcolor = 'rgb(217, 217, 217)',
        )
    )

fig.show()