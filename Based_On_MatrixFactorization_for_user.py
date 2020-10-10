'''
참고 : https://github.com/lsjsj92/recommender_system_with_Python/blob/master/004.%20recommender%20system%20basic%20with%20Python%20-%203%20Matrix%20Factorization.ipynb
행렬변환 관련 참고 matrix = user_movie_rating.values : https://www.inflearn.com/questions/30765
'''

from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

'''
read data
'''
rating_data = pd.read_csv('./data/BasedOnMatrixFactorization/ratings.csv', encoding='CP949')
movie_data = pd.read_csv('./data/BasedOnMatrixFactorization/movies.csv', encoding='CP949')

'''
Unnecessary column drop
'''

rating_data.drop('timestamp', axis = 1, inplace = True)
movie_data.drop('genres', axis = 1, inplace = True)
rating_data.head()
movie_data.head()

'''
movieId 기준으로 병합
'''
user_movie_data = pd.merge(rating_data, movie_data, on = 'movieId')
user_movie_data.head()
user_movie_data.shape
type(user_movie_data)

'''
vipot table 생성
rating, index: userId, columns: title
'''
user_movie_rating = user_movie_data.pivot_table('rating', index = 'userId', columns = 'movieId').fillna(0)
user_movie_rating.head()

'''
1) pivot table to matrix
2) through the np.mean(axis = 1), calcaulate mean average with each user
3) chage the value of user-mean data from value of 1) and 2)
'''
matrix = user_movie_rating.values
user_ratings_mean = np.mean(matrix, axis = 1)
matrix_user_mean = matrix - user_ratings_mean.reshape(-1, 1)
matrix

matrix.shape
user_ratings_mean.shape
matrix_user_mean.shape

pd.DataFrame(matrix_user_mean, columns = user_movie_rating.columns).head()

'''
svd 진행
'''
U, sigma, Vt = svds(matrix_user_mean, k = 12)
U.shape, sigma.shape, Vt.shape

'''
1차원 행렬인 sigma를 0이 포함된 2차원 대칭행렬로 변환
'''
sigma = np.diag(sigma)
sigma[0]
sigma[1]
sigma.shape

'''
분해된 행렬을 복구
'''
svd_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
svd_pred = pd.DataFrame(svd_user_predicted_ratings, columns = user_movie_rating.columns)
svd_pred.head()
svd_pred.shape


'''
영화 추천 함수 제작
'''
def recommend_movies(svd_pred, user_id, ori_movies_df, ori_ratings_df, num_recommendations=5):
    
    #현재는 index로 적용이 되어있으므로 user_id - 1을 해야함.
    user_row_number = user_id - 1 
    
    # 최종적으로 만든 pred_df에서 사용자 index에 따라 영화 데이터 정렬 -> 영화 평점이 높은 순으로 정렬 됌
    sorted_user_predictions = svd_pred.iloc[user_row_number].sort_values(ascending=False)
    
    # 원본 평점 데이터에서 user id에 해당하는 데이터를 뽑아낸다. 
    user_data = ori_ratings_df[ori_ratings_df.userId == user_id]
    
    # 위에서 뽑은 user_data와 원본 영화 데이터를 합친다. 
    user_history = user_data.merge(ori_movies_df, on = 'movieId').sort_values(['rating'], ascending=False)
    
    # 원본 영화 데이터에서 사용자가 본 영화 데이터를 제외한 데이터를 추출
    recommendations = ori_movies_df[~ori_movies_df['movieId'].isin(user_history['movieId'])]
    # 사용자의 영화 평점이 높은 순으로 정렬된 데이터와 위 recommendations을 합친다. 
    recommendations = recommendations.merge( pd.DataFrame(sorted_user_predictions).reset_index(), on = 'movieId')
    # 컬럼 이름 바꾸고 정렬해서 return
    recommendations = recommendations.rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :]
                      

    return user_history, recommendations

already_rated, predictions = recommend_movies(svd_pred, 330, movie_data, rating_data, 10)

'''
사용자가 남긴 평점
'''
already_rated.head()

'''
사용자가 남긴 평점에 대해서 추천된 영화
'''
predictions.head()