import requests
import bs4
from urllib.parse import urlparse, parse_qs
import re
import sys
import numpy as np
import pandas as pd
import datetime
from time import sleep


lxml = "lxml"
html = "html.parser"
tb_search = 'http://google.com/search?q=食べログ　'
global cnt, add_time
cnt=0
add_time=0

path = '../input/tabelog_store_data_shitamachi.csv'

# 港区の20駅
station_list = [
#    '志木駅',
#    '川越駅',
#    '朝霞台駅',
#    'ふじみ野駅',
#    '大宮駅',
#    '浦和駅',
#    '所沢駅',
#    '上福岡駅',
#    '川口駅',
#    '熊谷駅',
#    '草加駅',
#    '春日部駅',
#    'さいたま新都心駅'
    '池袋駅',
#    '新宿駅',
    '森下駅',
    '錦糸町駅',
    '浅草橋駅',
#    '秋葉原駅',
    '上野駅',
#    '新橋駅',
#    '飯田橋駅',
#    '水道橋駅',
#    '赤羽橋駅',
#    '麻布十番駅',
#    '白金高輪駅',
#    '三田駅',
#    '芝公園駅',
#    '神谷町駅',
#    '虎ノ門駅',
#    '御成門駅',
#    '広尾駅',
#    '白金台駅',
#    '汐留駅',
#    '浜松町駅',
#    '田町駅',
#    '乃木坂駅',
    '六本木一丁目駅',
    '六本木駅'
#    ,
#    '赤坂見附駅',
#    '表参道駅',
#    '青山一丁目駅'
]

store_columns = [
    'name',
    'station',
    'local',
    'street',
    'longitude',
    'latitude',
    'genre',
    'rate',
    'review',
    'open_date',
    'seat',
    'dinner_budget',
    'lunch_budget',
    'dinner_flg',
    'lunch_flg',
    'nomiho_flg',
    'tabeho_flg',
    'osya_flg',
    'couple_flg',
    'private_flg',
    'relax_flg',
    'night_view_flg',
    'hideout_flg',
    'wine_flg',
    'sake_flg',
    'vegetable_flg',
    'kodawari_flg',
    'sommelier_flg',
    'toll_flg',
    'net_reserve_flg',
    'coupon_flg',
    'pr_comment'
]


def get_soup(url, analysis, wait_time):
    global cnt, add_time
    # 良心的なスクレイピング
    sleep(wait_time + add_time)
    
    try:
        res = requests.get(url)  # + ' '.join(input()))
    except :
        print('Tabelog is angry!')
        sleep(300)
        res = requests.get(url)  # + ' '.join(input()))

    if str(res) == "<Response [400]>":
        print("ERROR")
        return 0
    res.raise_for_status()
    cnt+=1
    print('{} data get'.format(cnt))
    return bs4.BeautifulSoup(res.text, analysis)


def Google_Search(station):

    soup = get_soup(tb_search + station, lxml, 0)
    link = soup.select(".r a")
    regex = re.compile(r'''https://(.*)rstLst''')

    for i in range(len(link)):
        if len(regex.findall(str(link[i].get('href')))) > 0:
            return regex.findall(str(link[i].get('href')))
    print('Page is not found')
    sys.exit()


# search_areaで取得し移動した食べログのエリアページ（該当エリアの飲食店が一覧で表示されるページ）のURLを全て取得する
#（1ページ20店舗しかのっていないので、それ以降のページURLも取得しておく）
def get_tb_page(page):

    page_list = []
    soup = get_soup(page, lxml, 3)
    page_elems = soup.select('.c-pagination__item a')
    second_page = page_elems[0].get('href')
    
    return get_tb_page_all(second_page)


# そのエリアの店舗一覧ページを取得しきるまでループ(最下部の1,2,3,4,5～といったページ番号のURL)
def get_tb_page_all(page):

    limit=0
    regex_page_url = re.compile(r'''(.*?)rstLst/(\d+)/(.*)''')
    url_elem = regex_page_url.findall(page)
    
    #page_list = [url_elem[0][0] + 'rstLst/' + str(j) +  '/' + url_elem[0][2] for j in range(1, 21, 1)]
    page_list = [url_elem[0][0] + 'rstLst/' + str(j) +  '/' for j in range(1, 60, 1)]
    
    return page_list
    
#     for i in [10, 15, 20, 25]:
        
#         url = url_elem[0][0] + 'rstLst/' + str(i) + '/' + url_elem[0][2]
#         print(url)
#         res = get_soup(url, lxml, 3)
#         if res == 0:
#             limit = i
#             break
#         page_list = [url_elem[0][0] + 'rstLst/' + str(j) +  '/' + url_elem[0][2] for j in range(1, i, 1)]
#         [print(page_list[i]) for i in range(len(page_list))]
    
#     if limit==0:return page_list
#     elif limit>0:
#         for i in range(1, 5, 1):
#             url = url_elem[0][0] + 'rstLst/' + str(limit-i) +  '/' + url_elem[0][2]
#             res = get_soup(url)
#             if res != 0:
#                 limit -= i
#                 break
#                 return page_list
            
#     print('PAGE GET ERROR!!')
#     sys.exit()


# page = 店舗一覧ページ
def store_page_get(page):
    soup = get_soup(page, html, 5)
    store_elems = soup.select('.list-rst__rst-name-target')
    return store_elems


# 店舗名とURLを返す
def store_url_get(store_page):
    regex = re.compile(r'''target="_blank">(.*?)</a>''')
    return regex.findall(str(store_page))[0], store_page.get('href')


# エリア範囲を指定したい時に使う
# 下記の例は00mエリアのURLを取得する。
def area_page(area_300):
    res = requests.get(area_300)
    res.raise_for_status()
    soup = bs4.BeautifulSoup(res.text, 'lxml')
    page_elems = soup.select('.icon-b-arrow-orange')
    # 忘れたけど、全てのURLを取るから、300,500～といったエリア順でURLがリストに入ってる？
    return page_elems[0].get('href')
    # for i in range(len(page_elems)):
    # area_page300.append(page_elems[i].get('href'))
    # return area_page300[0]


# 店舗ページから住所を取得
def store_info_get(name, url):

    soup = get_soup(url, lxml, 1)

    #PRコメント
    pr_elem = soup.find("h3", class_="pr-comment-title js-pr-title")
    regex_pr = re.compile('''js-pr-title">(.*?)<''')
    pr = cleansing(str(regex_pr.findall(str(pr_elem))))

    # 緯度・経度を取得
    geo = soup.find("img", class_="js-map-lazyload")
    longitude, latitude = parse_qs(
        urlparse(geo["data-original"]).query)["center"][0].split(",")

    # 有料会員有無、ネット予約有無を取得
    toll_flg = 0
    net_reserve_flg = 0
    coupon_flg = 0
    
    toll_elem = soup.find("meta", attrs={"name": "description"}).attrs['content']
    tel_elem = soup.find("span", class_="rstdtl-booking-tel-modal__tel-num")
    regex_tel = re.compile(r'''\s(\d+)-''')
    tel = cleansing(str(regex_tel.findall(str(tel_elem))))
    if toll_elem.count('ネット予約可'): net_reserve_flg = 1
    if toll_elem.count('クーポンあり'): coupon_flg = 1
    if tel=='050' or net_reserve_flg==1:toll_flg = 1
        
    # ランチ・ディナー情報を取得
    dinner_elem = soup.find("em", class_="gly-b-dinner")
    lunch_elem = soup.find("em", class_="gly-b-lunch")
    regex_dinner = re.compile(r'''">￥(.*?)～''')
    regex_lunch = re.compile(r'''">￥(.*?)～''')
    
    dinner_budget = cleansing(regex_dinner.findall(str(dinner_elem))).replace(',', '')
    lunch_budget  = cleansing(regex_lunch.findall(str(lunch_elem))).replace(',', '')
    dinner_budget = empty_fill(dinner_budget, None)
    lunch_budget  = empty_fill(lunch_budget, None)
    
    if dinner_budget == None: dinner_flg = 0
    else: dinner_flg = 1
    if lunch_budget == None: lunch_flg = 0
    else: lunch_flg = 1

    # サービス・空間・設備情報を取得
    nomiho_flg = 0
    tabeho_flg = 0
    osya_flg = 0
    couple_flg = 0
    private_flg = 0
    relax_flg = 0
    night_view_flg = 0
    hideout_flg = 0
    wine_flg = 0
    sake_flg = 0
    vegetable_flg = 0
    kodawari_flg = 0
    sommelier_flg = 0
    
    tag_p = soup.find_all("p")
    regex_p = re.compile(r"<p>(.*?)<")
    regex_strong = re.compile(r"<p><strong>(.*?)<")
    p_elem = regex_p.findall(str(tag_p))
    strong_elem = regex_strong.findall(str(tag_p))
    
    for st in strong_elem:
        if st.count('飲み放題'):nomiho_flg = 1
        if st.count('食べ放題'):tabeho_flg = 1
    
    for p in p_elem:
        if p.count('オシャレ'): osya_flg = 1
        if p.count('カップルシート'): couple_flg = 1
        if p.count('個室'): private_flg = 1
        if p.count('落ち着いた'): relax_flg = 1
        if p.count('夜景'): night_view_flg = 1
        if p.count('隠れ家 '): hideout_flg = 1
        if p.count('ワイン'): wine_flg = 1
        if p.count('日本酒'): sake_flg = 1
        if p.count('野菜'): vegetable_flg = 1
        if p.count('こだわる'): kodawari_flg = 1
        if p.count('ソムリエ'): sommelier_flg = 1

    # 席数を取得
    seat = None
    regex_seat = re.compile(r"(\d*?)席")
    seat_elem = regex_seat.findall(str(tag_p))
    for i in seat_elem:
        if len(i) > 0:
            seat = i

    # 住所、口コミ数、点数、ジャンル、オープン日を取得
    regex_local = re.compile(r'''"addressLocality":"(.*?)",''')
    regex_street = re.compile(r'''"streetAddress":"(.*?)",''')
    regex_street_num = re.compile(r'''(\D+\d)''')
    regex_genre = re.compile(r'''"servesCuisine":"(.*?)",''')
    regex_rate = re.compile(r'''"ratingValue":"(.*?)"''')
    regex_review = re.compile(r'''"ratingCount":"(.*?)",''')
    regex_open = re.compile(r'''"rstinfo-opened-date">(.*?)</p>''')

    local  = cleansing(str(regex_local.findall(str(soup))))
    street = cleansing(str(regex_street.findall(str(soup))))
    #street = str(regex_street_num.findall(street)[0])
    genre = cleansing(str(regex_genre.findall(str(soup))))
    rate = cleansing(str(regex_rate.findall(str(soup))))
    review = cleansing(str(regex_review.findall(str(soup))))
    open_date = cleansing(str(regex_open.findall(str(soup))))
    
    if len(rate) == 0: rate = 0
    if len(review) == 0: review = 0

    return local, street, longitude, latitude, genre, rate, review , open_date, seat, dinner_budget, lunch_budget, dinner_flg, lunch_flg, nomiho_flg, tabeho_flg, osya_flg, couple_flg, private_flg, relax_flg, night_view_flg, hideout_flg, wine_flg, sake_flg, vegetable_flg, kodawari_flg, sommelier_flg, toll_flg, net_reserve_flg, coupon_flg, pr


def cleansing(x):
    if x.count('u0026nbsp'):
        x = None
    x = str(x).replace(r"['", "")
    x = str(x).replace(r"']", "")
    x = str(x).replace(r"[]", "")
    x = str(x).replace(r"\u3000", "")
    return x


def empty_fill(value, x):
    if len(value) == 0:
        value = x
    return value


def exist_data(path):
    data = pd.read_csv(path)
    return data['name'].values


def main():
 
    today = datetime.date.today().isoformat().replace('-', '')
    first = 1
    exist_flg=0

    # test
#     name="test"
#     url ="https://tabelog.com/tokyo/A1307/A130702/13172739/"
#     store_data = pd.DataFrame()
#     store_data = store_data.append(pd.Series([name] + list(store_info_get(name, url)), index=store_columns),ignore_index=True)
#     print(store_data)
#     sys.exit()
    
    # 取得済みの店舗リスト
#    name_list = exist_data(path)
    name_list=[]
    
    # 食べログのURLへ移動
    #「駅名 食べログ」で検索すればGoogle Searchの一番上に出てくるが、変わったら終わる。その時は要変更
    for station in station_list:
        search_area_url = 'https://' + str(Google_Search(station)[0]) + 'rstLst/?LstRange=SG&svd=' + today + '&svt=2330&svps=2'
        
        page_list = get_tb_page(search_area_url)
        
        print(station)
        print("Get {} pages. Probably {} stores exist.".format(len(page_list), len(page_list)*20))

        for i in range(len(page_list)):
            
            print("Current page number {}/{}".format(i+1, len(page_list)))
            
            # 取得した店舗一覧URLから、各店舗のURLを取得
            store_page_list = store_page_get(page_list[i])
            # 取得したstore_dataを格納していく
            store_data = pd.DataFrame()

            for store_page in store_page_list:
                name, url = store_url_get(store_page)
                
                # 取得済みのデータだったらcontinue
#                for sn in name_list:
#                    if sn.count(name):exist_flg = 1
#                if exist_flg==1:
#                    exist_flg=0
#                    continue
                    
#                 print("New store data:{}".format(name))

                # URLから店舗情報をスクレイピングしてDFへ格納
                store_data = store_data.append(pd.Series([name, station] + list(store_info_get(name, url)), index=store_columns),ignore_index=True)
                
#             print(store_data)
            # ページ一覧毎に書き込み
            if first==1:
                store_data.to_csv(path, header=True, index=False, encoding="utf-8", mode='a')
                first=0
            elif first==0:
                store_data.to_csv(path, header=False, index=False, encoding="utf-8", mode='a')

if __name__ == "__main__":

    main()
