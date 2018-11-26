import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np

x = np.linspace(-np.pi, np.pi, 10)
y = np.sin(x)
app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='H1の文章'),

    html.Div(children='''
        divの文章
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': x, 'y': np.sin(x), 'type': 'line', 'name': 'line'},
                {'x': x, 'y': np.cos(x), 'type': 'bar', 'name': 'bar'},
            ],
            'layout': {
                'title': 'グラフのタイトル'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True, host='52.193.163.244', port=8050)
