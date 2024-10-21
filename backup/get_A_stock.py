import urllib
import requests
import re
import json
import datetime
def get_html(url):
    with open('html.txt','w',encoding='utf-8') as f:
        f.write(requests.get(url).text)

# 股票信息主页
# http://quote.eastmoney.com/center/gridlist.html#hs_a_board
# 沪深A股列表真实链接
url = "http://11.push2.eastmoney.com/api/qt/clist/get?cb=jQuery112405084455909086425_1610764452571&pn=1&pz=4400&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1610764452635"

def read_file():
    with open('html.txt','r',encoding='utf-8') as f:
        all_contents = f.read()
    
    # 正则表达式，获取股票列表信息json串
    re_cont = re.findall('\\((.*?)\\);',all_contents,re.S)
    js_cont = json.loads(re_cont[0])
    stock_list=[]
    for i in js_cont['data']['diff']:
        #_stock={}
        #_stock['code']=i['f12']
        #_stock['name']=i['f14']
        #_stock['new_price']=i['f2']
        conten=[]  # 存放单只股票信息
        conten.append(i['f12']) # 股票代码
        conten.append(i['f14']) # 股票名称
        conten.append(i['f2']) # 最新价
        conten.append(i['f3']) # 涨跌幅
        conten.append(i['f4']) # 涨跌额
        conten.append(i['f5']) # 成交量（手）
        conten.append(i['f6']) # 成交额
        conten.append(i['f7']) # 振幅
        conten.append(i['f15']) # 最高
        conten.append(i['f16']) # 最低
        conten.append(i['f17']) # 今开
        conten.append(i['f18']) # 昨收
        conten.append(i['f10']) # 量比
        conten.append(i['f8']) # 换手率
        conten.append(i['f9']) # 市盈率（动态）
        conten.append(i['f23']) # 市净率
        stock_list.append(conten)
    
    with open(f'stock_{datetime.date.today()}.csv','w+',encoding='utf-8') as f:
        # 写入表头
        n=0
        single=(str(n)+','+'股票代码'+','+'股票名称'+','+'最新价'+','+'涨跌幅'+','
                +'涨跌额'+','+'成交量（手）'+','+'成交额（亿）'+','+'振幅'+','+'最高'+','
                +'最低'+','+'今开'+','+'昨收'+','+'量比'+','+'换手率'+','
                +'市盈率（动态）'+','+'市净率'+'\n')
        f.write(single)
        # 写入股票数据
        for _ in stock_list:
            n+=1
            single = str(n)
            for val in _ :
                single=single+','+str(val)
            single=single+'\n'
            f.write(single)

get_html(url)
read_file()