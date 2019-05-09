from flask import Flask, jsonify, abort, make_response, request

app = Flask(__name__)


"""
@app.routeデコレータの引数にパス・メソッド種別を指定しています。
このget関数ではapplication/json(json形式)でレスポンスを返しています。
レスポンスをJSONにするには、jsonify関数を使って作成したJSON型データを
make_response関数の引数に渡し、それをAPI関数の返り値とします。
make_responseはレスポンスヘッダ及びレスポンスボディの情報を持つオブジェクトを作成します。
このオブジェクトを介して、リクエストヘッダやボディの編集を行うことができます。
"""

# sample 1

@app.route("/get", methods=["GET"])
def get():
    # URLパラメータ
    params = request.args
    response = {}
    if 'param' in params:
        response.setdefault('res', 'params is : ' + params.get('param'))
    return make_response(jsonify(response))


@app.route("/get", methods=['POST'])
def post():
    # ボディ（application/json）パラメータ
    params = request.json
    response = {}
    if 'param' in params:
        response.setdefault('res', 'param is : ' + params.get('param'))
    return make_response(jsonify(response))

# sample 2

# @app.route('/')
# def hello_world():
#     return '<html><body><h1>sample<h1><body><html>'
#
# if __name__=='__main__':
#     # app.run(host="127.0.0.1", port=5000)
#     app.run()
