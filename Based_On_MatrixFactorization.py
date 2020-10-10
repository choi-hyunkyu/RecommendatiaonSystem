from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

import numpy as np
import pandas as pd
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
user_movie_rating = user_movie_data.pivot_table('rating', index = 'userId', columns = 'title').fillna(0)
user_movie_rating


'''
특정 영화와 비슷한 영화 추천

transpose
'''

movie_user_rating = user_movie_rating.values.T
movie_user_rating.shape


'''
Singular Value Decomposition - SVD 진행
'''
SVD = TruncatedSVD(n_components = 12)
matrix = SVD.fit_transform(movie_user_rating)
matrix.shape
matrix[0]

'''
피어슨 상관계수를 통해 상관계수 출력
'''
corr = np.corrcoef(matrix)
corr.shape

corr2 = corr[:200, :200]
corr2.shape

'''
seaborn heatmap
'''
plt.figure(figsize = (16, 10))
sns.heatmap(corr2)

movie_title = user_movie_rating.columns
movie_title_list = list(movie_title)
coffey_hands = movie_title_list.index("Toy Story (1995)")

corr_coffey_hands = corr[coffey_hands]
list(movie_title[(corr_coffey_hands >= 0.9)])[:50]