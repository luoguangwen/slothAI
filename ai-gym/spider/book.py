import json
import re
import time
import requests
import multiprocessing
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

    #百度疫情首页
    def handle_index(self):
        md_search = re.compile(r'href="/book/85(.*)\.md" title=')
        index_url = "https://www.cntofu.com/book/85/index.html"
        result = self.handle_request(method="GET",url=index_url)
        print(result)
        #使用正则表达式获取城市列表
        self.md_list = set(md_search.findall(result))
        self.baidu_seddion.cookies.clear()
        return self.md_list


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

# 清洗HTML标签文本
# @param htmlstr HTML字符串.
def filter_tags(htmlstr):
    # 过滤DOCTYPE
    htmlstr = ' '.join(htmlstr.split()) # 去掉多余的空格
    re_doctype = re.compile(r'<!DOCTYPE .*?> ')
    s = re_doctype.sub('',htmlstr)

    # 过滤CDATA
    re_cdata = re.compile('//<!CDATA\[[ >]∗ //\] > ', re.I)
    s = re_cdata.sub('', s)

    # # HTML注释
    re_comment = re.compile('<!--[^>]*-->')
    s = re_comment.sub('', s)

    # Script
    #re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)

    re_script = re.compile('<\s*script(.*)>(.*)\s<\s/script >', re.I)\

    s = re_script.sub('', s)  # 去掉SCRIPT

    # head
    re_script = re.compile('<head>(.*)</head>', re.I)
    s = re_script.sub('', s)  # 去掉head
    # style
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)
    s = re_style.sub('', s)  # 去掉style

    # 处理换行
    # re_br = re.compile('<br\s*?/?>')
    # s = re_br.sub('', s)     # 将br转换为换行
    #
    # # HTML标签
    # re_h = re.compile('</?\w+[^>]*>')
    # s = re_h.sub('', s)  # 去掉HTML 标签
    #


    # # 多余的空行
    # blank_line = re.compile('\n+')
    # s = blank_line.sub('', s)

    # blank_line_l = re.compile('\n')
    # s = blank_line_l.sub('', s)
    #
    # blank_kon = re.compile('\t')
    # s = blank_kon.sub('', s)
    #
    # blank_one = re.compile('\r\n')
    # s = blank_one.sub('', s)
    #
    # blank_two = re.compile('\r')
    # s = blank_two.sub('', s)
    #
    # blank_three = re.compile(' ')
    # s = blank_three.sub('', s)
    #
    # # 剔除超链接
    # http_link = re.compile(r'(http://.+.html)')
    # s = http_link.sub('', s)
    return s

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
    return res
    # print(soup.prettify())
    # print(soup.title)
    # print(soup.title.name)
    # print(soup.title.string)
    # print(soup.title.parent.name)
    # print(soup.p)
    # print(soup.find_all(class_="book-content-section  md-content-section  uk-margin-bottom"))
    # print(soup.find_all(attrs = {"class": "uk-grid"}))
    # print(soup.find_all("div", class_="md-content-section"))
if __name__ == '__main__':

    book = HandleBook()
    #取得md文件列表
    md_list = book.handle_index()
    md_url_list = ["https://www.cntofu.com/book/85{}.md".format(index) for index in md_list]
    print(md_url_list)

    fullname = os.path.join('./data', 'a_book.md')
    with open(fullname, 'ab+')as md_file:
        for index in md_list:
            result = book.handle_request(method="GET", url="https://www.cntofu.com/book/85{}.md".format(index))

            md_content = extract_md_from_html(result)

            md_file.write(md_content.encode(encoding='UTF-8'))

    # file = os.path.join('./data', 'nlp-tf-idf.html')
    # with open('./data/nlp-tf-idf.html', 'r', encoding='UTF-8') as file:
    #     soup = BeautifulSoup(file, 'lxml')
    #     # print(soup.prettify())
    #     # print(soup.title)
    #     # print(soup.title.name)
    #     # print(soup.title.string)
    #     # print(soup.title.parent.name)
    #     # print(soup.p)
    #     # print(soup.find_all(class_="book-content-section  md-content-section  uk-margin-bottom"))
    #     # print(soup.find_all(attrs = {"class": "uk-grid"}))
    #     # print(soup.find_all("div", class_="md-content-section"))
    #     result = soup.find_all("div", class_="md-content-section")  # 找出包含md文件内容的div标签
    #     # print(result.__str__())
    #     reg = re.compile(r'<[/]?p>')  # 去掉<p>  </p>
    #     res = reg.sub('', result.__str__())
    #
    #     reg = re.compile(r'<[/]?div\s*.*"*\s?>')  # 去掉<div>  </div>
    #     res = reg.sub('', res)
    #     print(res)
