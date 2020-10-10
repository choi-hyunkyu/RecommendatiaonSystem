#-*- coding:utf-8 -*-
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

'''
참고 : https://lsjsj92.tistory.com/m/568?category=853217
디코드에러 참고 :  http://blog.naver.com/PostView.nhn?blogId=ex122388&logNo=221223526752&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
'''
rating_data = pd.read_csv('./data/BasedOnItem/ratings.csv', encoding='CP949')
movie_data = pd.read_csv('./data/BasedOnItem/movies.csv', encoding='CP949')
rating_data.head(3)
movie_data.head(3)


'''
불필요한 column drop 및 movieId 기준으로 병합
'''
user_movie_rating = pd.merge(rating_data, movie_data, on = 'movieId')
user_movie_rating.head(6)

'''
vipot table 생성
data : 영화 평점 rating
index : 영화 title
columns : userId
'''
movie_user_rating = user_movie_rating.pivot_table('rating', index = 'title', columns = 'userId')
movie_user_rating.head()


'''
결측치 치환
'''
movie_user_rating.fillna(0, inplace = True)
movie_user_rating.head()


'''
Cosine similarity with scikit learn
'''
item_based_collabor = cosine_similarity(movie_user_rating)
item_based_collabor

df_item_based_collabor = pd.DataFrame(data = item_based_collabor, index = movie_user_rating.index, columns = movie_user_rating.index)
df_item_based_collabor

'''
get item based collarbor
'''
def get_item_based_collabor(title):
    return df_item_based_collabor[title].sort_values(ascending = False)[:6]

get_item_based_collabor('Toy Story (1995)')