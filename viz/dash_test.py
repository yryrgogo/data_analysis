import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd

# For Google SpreadSheet 
from oauth2client.service_account import ServiceAccountCredentials
from httplib2 import Http
import gspread

scopes = ['https://www.googleapis.com/auth/spreadsheets']
json_file = '../../../privacy/My Project-c4d8e8d43930.json'#OAuth用クライアントIDの作成でダウンロードしたjsonファイル

credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file, scopes=scopes)
http_auth = credentials.authorize(Http())

# スプレッドシート用クライアントの準備
doc_id = '1oJx3w0Ks5iDhr4gE_QeL7nOjEDndNJ9NetS70zogCWQ'#スプレッドシートのURLのうちhttps://docs.google.com/spreadsheets/d/以下, /editより前の部分
client = gspread.authorize(credentials)
gfile   = client.open_by_key(doc_id)#読み書きするgoogle spreadsheet
sh  = gfile.sheet1

# 指定した行の値を全て取得する
row_list = sh.row_values(1)
# 指定した範囲の値を取得する
#  cell_list = ws.range('A1:B7')

# SpreadSheetからデータを読み込み
# .col_values()で指定した列の値を全て取得する
df = pd.DataFrame(data=np.array([sh.col_values(2)[1:], sh.col_values(4)[1:]]).T, columns=['date', 'category'])
df = df.groupby(['date', 'category']).size().reset_index().rename(columns={0:'cnt'}).set_index(['date', 'category'])
df = df.unstack().fillna(0)['cnt']

# 描画する値
x = np.array(df.index)
columns = np.array(df.columns)


trace_list = []
for col in columns:
    trace_list.append({'x': x, 'y': df[col].values, 'type': 'line', 'name': col})

app = dash.Dash()

app.layout = html.Div(children=[

    html.H1(children='Latest 30days Pomodoro'),

    # Live
    html.Div([
    dcc.Input(
        id='my-id',
        value='initial value',
        type='text'
    ),
    html.Div(id='my-div')
    ]),

    html.Div(children='''
直近30日のポモドーロ状況
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': trace_list,
            'layout': {
                'title': 'Pomodoro Freq',
                #  'barmode': 'relative',
            }
        }
    )
])

@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    return 'You have entered "{}"'.format(input_value)

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
