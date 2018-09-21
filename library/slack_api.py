
def sendMessage(channel_key='test', sendText='test'):
    import urllib.parse
    import urllib.request
    import requests
    import json
    import sys
    import yaml
    from xml.sax.saxutils import unescape   # エスケープされた記号を戻す

    # 設定ファイルのロード
    config = {}
    with open('config.yaml', "r", encoding="utf-8") as cf:
        config = yaml.load(cf)

    ' Slack token load'
    token = config["slack_token"]

    url = {'dm': "https://slack.com/api/im.history?",
           'ch': "https://slack.com/api/channels.history?"
           }

    ' チャンネル '
    channel = {'dm': "D6FJTP6HE",  # target channel id
               'times_go': "C6G6K2L11",
               'test': "G9Q8V5340"
               }
    '''
    Explain:
        指定されたチャンネルにメッセージを投稿する
    Args:
        channel_id (str)
            送信対象のチャンネルのID文字列
        sendText (str)
            送信するメッセージの内容
    Return:
        None
    '''

    URL='https://slack.com/api/chat.postMessage'
    url_ch = {'dm': "https://slack.com/api/im.history?",
           'ch': "https://slack.com/api/channels.history?"
           }
    ' チャンネル '
    channel = {'dm': "D6FJTP6HE",  # target channel id
               'times_go': "C6G6K2L11",
               'test': "G9Q8V5340"
               }

    channel_id = channel[channel_key]

    params = {
        'token': token,
        'channel': channel_id,
        'text': sendText,
        'username': 'DataRobot_Bot',
        #  "icon_emoji": u":datarobot:",
        "icon_emoji": u":hori:",
        'link_names': 1,  # 名前をリンク化
    }

    requests.post(URL, data=params)

    return


def create_params(channel, count=10):
    params = {
        'token':      token,
        'channel':    channel,
        'count':      count
    }
    return urllib.parse.urlencode(params)


def get_messages():
    res = urllib.request.urlopen(url['ch'] + params)
    body = res.read().decode("utf-8")
    response = json.loads(body)

    for r in response['messages']:
        if r['type'] == 'message':
            print(unescape(r['text'].replace('\n', ' ')))


if __name__ == '__main__':
    #     params = create_params(channel=channel['times'], count=10)
    #     get_messages()

    message = 'hello'
    ' チャンネルにメッセージを流す '
    sendMessage('test', message)
