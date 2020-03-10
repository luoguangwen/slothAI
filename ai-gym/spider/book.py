# -*- encoding: utf-8 -*-
"""
@File    :   dataset.py

@Contact :   luoguangwen@163.com


@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/2/22 14:16    Kevin        1.0          None
"""


'''
爬虫练习，本文仅做学习练习使用。
网站：https://www.cntofu.com/book/85/index.html
目标：该网站是一份机器学习相关的文档资料，通过网站URL发现每个页面都是一个md文件，本练习目标是将各页面内容抓取并存储成一个md文件

'''

import re
import time
import requests
from bs4 import BeautifulSoup
import os



class HandleBook(object):
    def __init__(self):
        #使用session保存cookies信息
        self.baidu_seddion = requests.session()
        self.header = {
         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
        }
        self.city_list = ""

    #请求首页，取得MD文件列表
    def handle_index(self):
        md_search = re.compile(r'href="/book/85(.*)\.md"\s*title=')
        index_url = "https://www.cntofu.com/book/85/index.html"
        result = self.handle_request(method="GET",url=index_url)

        soup = BeautifulSoup(result, 'lxml')

        # 找出包含md文件内容的div标签
        soupTagUl = soup.find_all("ul", class_="book-menu-bar")
        print(soupTagUl)
        #使用正则表达式获取md文件列表
        self.md_list = (md_search.findall(soupTagUl.__str__()))
        #self.md_list = set(md_search.findall(soupTagUl.__str__()))  #set函数会打乱顺序？？
        self.baidu_seddion.cookies.clear()
        print('md file list:',self.md_list)

        return self.md_list


    # http request请求接口封装
    def handle_request(self,method,url,data=None,info=None):
        while True:
            try:
                if method == "GET":
                    self.baidu_seddion.keep_alive =False

                    response = self.baidu_seddion.get(url=url, headers=self.header)

                elif method == "POST":
                    response = self.baidu_seddion.post(url=url, headers=self.header, data=data)
            except Exception as e :
                # 需要先清除cookies信息
                self.baidu_seddion.cookies.clear()
                print(str(e))
                time.sleep(10)
                continue
            response.encoding = 'utf-8'

            return response.text


# 从html文本中提取md文档内容
def extract_md_from_html(htmlstr):

    soup = BeautifulSoup(htmlstr, 'lxml')

    # 找出包含md文件内容的div标签
    result = soup.find_all("div", class_="md-content-section")

    # 去掉<p>  </p> ,转成md后带p标签不能显示数据公式
    reg = re.compile(r'<[/]?p>')
    res = reg.sub('', result.__str__())

    # 去掉<div>  </div>
    reg = re.compile(r'<[/]?div\s*.*"*\s?>')
    res = reg.sub('', res)

    # HTML标签
    # reg = re.compile('</?\w+[^>]*>')
    # res = reg.sub('', res)  # 去掉HTML 标签

    # </?h[0-9]+[^>]*>   --> <h*     >  </h*>
    reg = re.compile('</?h[0-9]+[^>]*>')
    res = reg.sub('', res)  # 去掉h*标签

    # </?ol[^>]*>        --> 去ol标签
    reg = re.compile('</?ol[^>]*>')
    res = reg.sub('', res)  # 去掉ol标签

    # </?li[^>]*>        --> 去li标签
    reg = re.compile('</?li[^>]*>')
    res = reg.sub('', res)  # 去掉li标签

    # </?ul[^>]*>        --> 去ul标签
    reg = re.compile('</?ul[^>]*>')
    res = reg.sub('', res)  # 去掉ul标签

    # </?hr[^>]*>        --> 去hr标签
    reg = re.compile('</?hr[^>]*>')
    res = reg.sub('', res)  # 去掉hr标签
    return res

if __name__ == '__main__':

    book = HandleBook()

    #取得md文件列表
    md_list = book.handle_index()
    print(len(md_list))

    #循环请求各md文件
    fullname = os.path.join('./', 'book_f.md')
    with open(fullname, 'ab+')as md_file:
        for index in md_list:
            result = book.handle_request(method="GET", url="https://www.cntofu.com/book/85{}.md".format(index))

            md_content = extract_md_from_html(result)

            tempstr = '# ' + '*' *5 + index + '.md' + '*' *5 + '\n\n'
            print(tempstr)
            md_file.write(tempstr.encode(encoding='UTF-8'))
            md_file.write(md_content.encode(encoding='UTF-8'))
