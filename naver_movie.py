# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote
from urllib.request import urlopen
import json
import re
import requests

# 네이버 오픈 API 키 값이다.
naver_client_id = "vkHZtqE5hdY3qyvhESs2"
naver_client_secret = "rHVGxoGggK"

url1 = 'http://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code='
url2 = '&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page='


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def searchByTitle(title):
    myurl = 'https://openapi.naver.com/v1/search/movie.json?display=100&query=' + quote(title)
    request = urllib.request.Request(myurl)
    request.add_header("X-Naver-Client-Id", naver_client_id)
    request.add_header("X-Naver-Client-Secret", naver_client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if (rescode == 200):
        response_body = response.read()
        d = json.loads(response_body.decode('utf-8'))
        if (len(d['items']) > 0):
            return d['items']
        else:
            return None

    else:
        print("Error Code:" + rescode)


def findItemByInput(items):  # items로부터 값을 가져온다.
    for index, item in enumerate(items):
        navertitle = cleanhtml(item['title'])
        navertitle = re.sub(':', '', navertitle)
        # 영화 제목에 : 가 있을 경우 뺀다. 왜냐면 텍스트 파일의 제목에 : 가 올 수 없기 때문이다.
        # 예를 들어서 어벤져스 : 인피니티 워라면 어벤져스 인피니티 워로 저장한다.
        naversubtitle = cleanhtml(item['subtitle'])
        naverpubdate = cleanhtml(item['pubDate'])
        naveractor = cleanhtml(item['actor'])
        naverlink = cleanhtml(item['link'])
        naveruserScore = cleanhtml(item['userRating'])

        navertitle1 = navertitle.replace(" ", "")
        navertitle1 = navertitle1.replace("-", ",")
        navertitle1 = navertitle1.replace(":", ",")

        # 기자 평론가 평점
        spScore = getSpecialScore(naverlink)

        # 영화 고유 id로 이 고유 id로 유저 평점을 크롤링 할 수 있다.
        naverid = re.split("code=", naverlink)[1]
        getReviewResult(index, navertitle, naversubtitle, naveruserScore, spScore, naverid)

        # 영화의 타이틀 이미지
        # if (item['image'] != None and "http" in item['image']):
        #    response = requests.get(item['image'])
        #    img = Image.open(BytesIO(response.content))
        #    img.show()

        # 인덱스, 영화 제목, 영어 제목, 네이버 유저 평균 평점, 네이버 전문가 평점을 출력한다.
        # print(index, navertitle, naversubtitle, naveruserScore, spScore)


def getInfoFromNaver(searchTitle):
    items = searchByTitle(searchTitle)
    # print(items)
    # items에는 title : 영화 제목, link : 네이버 영화 링크, subtitle : 영어 제목,
    # pubData : 개봉 년도, director : 감독 이름, actor : 배우 이름들, userRating : 평점이 들어가있다.

    if (items != None):
        findItemByInput(items)
    else:
        print("No result")


def get_soup(url):
    source_code = requests.get(url)
    plain_text = source_code.text
    soup = BeautifulSoup(plain_text, 'lxml')
    return soup


def getReviewResult(index, navertitle, naversubtitle, naveruserScore, spScore, code):
    index = str(index)
    f = open(index + '_' + navertitle + '.txt', 'w')
    # f.write(str(index))
    # f.write(" ")
    # 제목
    f.write(str(navertitle))
    f.write(" ")
    # english 제목
    f.write(str(naversubtitle))
    f.write(" ")
    # 평론가 점수
    f.write(str(naveruserScore))
    f.write(" ")
    # 관객 점수
    f.write(str(spScore))
    f.write("\n")

    # 인덱스_영화제목.txt 파일에다가 크롤링한 데이터를 저장할 예정이다.
    page = int(1)
    # 리뷰 댓글 페이지는 1페이지부터 시작한다.
    count = 1
    # 네이버 리뷰 댓글은 1페이지까지 크롤링한다.
    # 단, 댓글 페이지가 10페이지가 안 될 경우 마지막 페이지가 반복 크롤링 될 수 있다.

    while count:
        URL = url1 + code + url2 + str(page)
        # 네이버 영화 소개 url로부터 네이버 영화 댓글란 url을 완성시킨다.
        open2 = urlopen(URL)
        html = open2.read().decode('utf-8')
        soup = BeautifulSoup(html, "lxml")
        score_result = soup.find('div', class_='score_result')
        # score_result는 평점, 리뷰, 아이디 등이 모두 적혀있는 상태이다.

        if (score_result != None):
            lis = score_result.find_all('li')
        else:
            break

        for li in lis:
            page = int(page)
            reple = li.find('div', class_='score_reple').find('p').get_text()
            score = li.find('div', class_='star_score').find('em').get_text()
            # children=li.find('dt')
            # child=children.findChildren()

            reple = str(reple)
            score = str(score)
            # time=str(child[3])
            # time_s=time[4:17]

            print(reple)
            print(score)
            # print(time_s)

            # f.write(time_s)
            # f.write(" ")
            f.write(score)
            f.write(" ")
            f.write(reple)
            f.write('\n')

        count -= 1

        if not count:
            break

        page += 1
    f.close()


def getSpecialScore(URL):
    soup = get_soup(URL)
    scorearea = soup.find_all('div', "spc_score_area")
    newsoup = BeautifulSoup(str(scorearea), 'lxml')
    score = newsoup.find_all('em')
    if (score and len(score) > 5):
        scoreis = score[1].text + score[2].text + score[3].text + score[4].text
        return float(scoreis)
    else:
        return 0.0


# getInfoFromNaver(u"물괴")  # 영화 제목을 입력한다.