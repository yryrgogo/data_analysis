from flask import Flask, render_template, request


app = Flask(__name__)


@app.route('/')
def index():
    # use templates is [templates/index.html]
    # the template is used that set the variable as [message] and input "Hello"
    return render_template('index.html', message="Hello")


# FlaskでGetパラメータを扱うには下記の様に書く
@app.route('/hello')
def hello():
    # query parameter is included in request.args
    val = request.args.get("msg", "Not defined")
    return 'Hello World ' + val


# FlaskでPOST通信を受け付けて、POSTパラメータを取得するには以下のように実装します。