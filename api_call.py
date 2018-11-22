import requests
import json
from datetime import datetime
import naver_moive

#
# Intent model을 거친 intent ==> type Str --> intent_list = ["find_movie", "get_maker","get_grade", "getRunningTime", "getTrailer", "getCast", "findPerson", "Search", "getPoster", "isTrue", "Recommend", "findSimilar"]
# Entity model을 거친 entity ==> type Dict --> {"배우": "","영화명" : "", "국가" : "" , "장르" : "", "감독명" : "", "제작년도" : "", "제작월" : "", "O" : ""}
#

api_key = "e37539e7225dfe02a4952798941eadcb"
code_search_url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieList.json"
movieinfo_search_url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieInfo.json"
actor_search_url  = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/people/searchPeopleList.json"
boxoffice_search_url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"
# movie_name = "반지의 제왕"
# actor_name = "성동일"

def movie_code_search(movie_name):
    params = {"key" : api_key, "movieNm": movie_name}
    response = requests.get(code_search_url, params)
    movie_info = json.loads(response.text)
    try:
        return movie_info["movieListResult"]["movieList"][0]["movieCd"]
    except:
        print("영화 이름을 다시 한번 확인해주세요.")



def movie_info_search(movie_code):
    params = {"key":api_key, "movieCd":movie_code}
    response = requests.get(movieinfo_search_url, params)
    movie_info = json.loads(response.text)
    return movie_info

def movie_info_simplify(movie_info):
    genre_list = []
    actor_list = []
    director_list = []
    movie_dict = {"영화이름":"","상영시간":"","개봉일":"","장르":""}
    movie_dict['영화이름'] = movie_info["movieInfoResult"]["movieInfo"]["movieNm"]
    movie_dict['상영시간'] = movie_info["movieInfoResult"]["movieInfo"]["showTm"]
    movie_dict['개봉일'] = movie_info["movieInfoResult"]["movieInfo"]["openDt"]
    movie_dict['이용가'] = movie_info["movieInfoResult"]["movieInfo"]["audits"][0]['watchGradeNm']
    for i in range(len(movie_info["movieInfoResult"]["movieInfo"]["genres"])):
        genre_list.append(movie_info["movieInfoResult"]["movieInfo"]["genres"][i]["genreNm"])
    movie_dict['장르'] = genre_list
    for i in range(len(movie_info["movieInfoResult"]["movieInfo"]["directors"])):
        director_list.append(movie_info["movieInfoResult"]["movieInfo"]["directors"][i]['peopleNm'])

    for i in range(min([10,len(movie_info["movieInfoResult"]["movieInfo"]["actors"])])):                                ## 배우는 최대 10명까지 출력
        actor_list.append(movie_info["movieInfoResult"]["movieInfo"]["actors"][i]['peopleNm'])
    movie_dict['배우'] = actor_list
    for i in range(len(movie_info["movieInfoResult"]["movieInfo"]["companys"])):
        if movie_info["movieInfoResult"]["movieInfo"]["companys"][i]['companyPartNm'] == '배급사':
            movie_dict['배급사'] = movie_info["movieInfoResult"]["movieInfo"]["companys"][i]['companyNm']

    return movie_dict

def filmo_list_search(actor_name):
    filmo_list = []
    params = {"key" : api_key, "peopleNm" : actor_name }
    response = requests.get(actor_search_url, params)
    movie_info = json.loads(response.text)
    # for i in len(movie_info["peopleListResult"]["peopleList"]):
    #     actor_list.append()
    filmo_list = movie_info["peopleListResult"]["peopleList"][0]['filmoNames'].split("|")
    return filmo_list[:10]

def boxoffice_search():
    boxoffice_list = []
    now = datetime.now()
    params = {"key" : api_key, "targetDt" : str(now.year)+str(now.month)+str(now.day-1)}
    response = requests.get(boxoffice_search_url, params)
    movie_info = json.loads(response.text)
    return len(movie_info["boxOfficeResult"]["dailyBoxOfficeList"])


def chat_model(intent, entity_dict={}):


    if intent == u'find_movie':
        try:
            if entity_dict["배우명"] is not None:
                return filmo_list_search(entity_dict["배우명"])

        except:
            return

    elif intent == u'search':
        try:
            return naver_moive.getInfoFromNaver(entity_dict["영화명"])
        except:
            return "영화명을 다시 한번 확인해주세요"

    elif intent == u"get_maker":
        try:
            movie_code = movie_code_search(entity_dict["영화명"])
            movie_info = movie_info_search(movie_code)
            movie_info = movie_info_simplify(movie_info)
            return movie_info["배급사"]

        except:
            return "영화명을 다시 한번 확인해주세요"

    elif intent == u'get_grade':
        try:
            movie_code = movie_code_search(entity_dict["영화명"])
            return movie_info_search(movie_code)["movieInfoResult"]["movieInfo"]["audits"][0]['watchGradeNm']
        except:
            return "영화명을 다시 한번 확인해주세요"

    elif intent == u'getRunningTime':
        try:
            movie_code = movie_code_search(entity_dict["영화명"])
            return movie_info_search(movie_code)["movieInfoResult"]["movieInfo"]["showTm"]
        except:
            return "영화명을 다시 한번 확인해주세요"


    elif intent == u'getTrailer':
        # 다음 movie API가 서비스 종료
        return "트레일러 영상은 어떻게 보여줬는지 알았는데 지금은 기억나질 않아요"
    elif intent == u'getCast':                      # 캐스팅 정보 /  영화 제목 또는 배우이름 / 안시성에서 조인성이 한 역할은 / 영화 제목 + 배우명 검색

        return 0
    elif intent == u'findPerson':
        return "아직 정보 학습이 부족해서 답변드릴 수 없어요"

    elif intent == u'getPoster':
        return 0
    elif intent == u'isTrue':
        return 0
    elif intent == u'Recommend':
        return "아직 내가 뭘 추천해줄 수 있을정도로 학습하지는 못했어요"
    elif intent == u'findSimilar':
        return 0

    return 0



# movie_code = movie_code_search()
# print(movie_code)
# movie_info = movie_info_search(movie_code)
# movie_info
# print(movie_info_simplify(movie_info))


# print(filmo_list_search())
# print(boxoffice_search())

entity_dict = {"배우": "","영화명" : "반지의 제왕", "국가" : "" , "장르" : "", "감독명" : "", "제작년도" : "", "제작월" : "", "O" : ""}
print(chat_model("get_maker",entity_dict))

