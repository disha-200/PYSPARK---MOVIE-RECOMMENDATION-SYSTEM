
import pandas as pd
import math
import mlflow
import os
import pyspark
import time

from pyspark.sql.types import *
from pyspark.sql.functions import col, mean, udf, lit, current_timestamp, unix_timestamp, array_contains
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql import SparkSession


def set_configurations(sparkConf):
    sparkConf.setAppName("Movie Recommendation System")
    sparkConf.set("spark.dynamicAllocation.enabled", "true")
    sparkConf.set("spark.executor.cores", 6)
    sparkConf.set("spark.dynamicAllocation.minExecutors","1")
    sparkConf.set("spark.dynamicAllocation.maxExecutors","3")
    sparkConf.set("spark.executor.memory", "2g")
    sparkConf.set("spark.driver.memory", "2g")
    sparkConf.set("spark.ui.port","4050")
    sparkConf.set("spark.memory.fraction", 0.5)
    sparkConf.set("spark.executor.memoryOverhead","512m")
    return sparkConf

def read_data():
    ratings = spark_session.read.csv('ratings.csv', header=True)
    movies = spark_session.read.csv('movies.csv', header=True)
    tags = spark_session.read.csv('tags.csv', header=True)
    links = spark_session.read.csv('links.csv', header=True)
    return ratings, movies, tags, links

# from pyspark.conf import SparkConf
# sparkConf = SparkConf()
# sparkConf.setAppName("My app")
# sparkConf.set("spark.dynamicAllocation.enabled", "true")
# sparkConf.set("spark.executor.cores", 8)
# sparkConf.set("spark.dynamicAllocation.minExecutors","1")
# sparkConf.set("spark.dynamicAllocation.maxExecutors","5000")
# sparkConf.set("spark.executor.memory", "8g")
# sparkConf.set("spark.ui.port","4050")
# sparkConf.set("spark.memory.fraction", 0.9)

from pyspark.conf import SparkConf
sparkConf = SparkConf()
sparkConf = set_configurations(sparkConf)

spark_session = SparkSession.builder.master('local[*]').config(conf=sparkConf).getOrCreate()

tables = ['movies', 'ratings', 'tags', 'links']

dataframeList = {} #defaultdict(None)

ratings, movies, tags, links = read_data()

ratings.printSchema()
movies.printSchema()
tags.printSchema()
links.printSchema()

movies.createOrReplaceTempView("movies_df")
spark_session.sql("SELECT * FROM movies_df limit 20").show(20)

ratings.createOrReplaceTempView("ratings_df")
spark_session.sql("SELECT * FROM ratings_df limit 20").show(20)

links.createOrReplaceTempView("links_df")
spark_session.sql("SELECT * FROM links_df limit 20").show(20)

tags.createOrReplaceTempView("tags_df")
spark_session.sql("SELECT * FROM tags_df limit 20").show(20)

movies.registerTempTable("movies")
ratings.registerTempTable("ratings")
links.registerTempTable("links")
tags.registerTempTable("tags")

min_rating_user_id = ratings.groupBy("userID").count().toPandas()['count'].min()
min_rating_movie_id = ratings.groupBy("movieId").count().toPandas()['count'].min()

print('Minimum number of ratings per user: {}'.format(min_rating_user_id))
print('Minimum number of ratings per movie: {}'.format(min_rating_movie_id))

one_rating_movie_id = sum(ratings.groupBy("movieId").count().toPandas()['count'] == 1)
one_rating_unique_movies = ratings.select('movieId').distinct().count()

print('movies are rated by only one user: {} out of {} '.format(one_rating_movie_id, one_rating_unique_movies))

# number of distinct users
unique_users = spark_session.sql("SELECT count (distinct userID) as num_users FROM ratings")
ratings.select("userId").distinct().count()

# number of movies
unique_movies = spark_session.sql("SELECT count (distinct movieID) as num_movies FROM movies")
print(unique_movies)
print(movies.select('movieID').distinct().count())

rated_movies = ratings.select('movieID').distinct().count()
print('Total Number of movies rated by users:', rated_movies)

# null rated movies
spark_session.sql("SELECT movies.title, movies.genres ,ratings.rating FROM movies left JOIN ratings ON ratings.movieId = movies.movieID WHERE ratings.rating IS null LIMIT 10").show()

# movie genres
spark_session.sql("SELECT DISTINCT(genres) FROM movies LIMIT 10").show()

genres = udf(lambda x: x.split("|"), ArrayType(StringType()))
movies_genres = movies.select("movieId", "title", genres("genres").alias("genres"))

movies_genres.createOrReplaceTempView("movies_genres")

print(spark_session.sql("SELECT * FROM movies_genres limit 5"))

movies_genres.show(5)

# All movie categories
total_genres = list(set(movies_genres.select('genres').rdd.flatMap(tuple).flatMap(tuple).collect()))
total_genres

movies_dataframe = movies.toPandas()
movies_list = list(movies_dataframe['title'])

# Data type convert
all_ratings=ratings.drop('timestamp')

all_ratings = all_ratings.withColumn("userId", all_ratings["userId"].cast(IntegerType()))
all_ratings = all_ratings.withColumn("movieId", all_ratings["movieId"].cast(IntegerType()))
all_ratings = all_ratings.withColumn("rating", all_ratings["rating"].cast(FloatType()))

all_ratings.show(50)
all_ratings.createOrReplaceTempView("all_ratings")
print(spark_session.sql("SELECT * FROM all_ratings limit 10"))

all_ratings_sample = all_ratings.sample(False, 1/500)
all_ratings_sample.show()

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

alternating_least_squares = ALS( userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative = True, implicitPrefs = False, coldStartStrategy="drop")

#(trainData, testData) = movie_ratings.randomSplit([0.1,0.02])

(training_data, testing_data) = all_ratings.randomSplit([0.8,0.2])

training = training_data.sample(fraction=0.0020)
testing = testing_data.sample(fraction=0.0020)

print(training_data.count())
print(testing_data.count())
print(training.count())
print(testing.count())

print(len(training.toPandas()))
print(testing)

parameter_grid = ParamGridBuilder() \
            .addGrid(alternating_least_squares.rank, [10, 50, 100, 150]) \
            .addGrid(alternating_least_squares.regParam, [.01, .05, .1, .15]) \
            .addGrid(alternating_least_squares.maxIter, [15]) \
            .build()

print ("Num models to be tested: ", len(parameter_grid))
starting_time = time.time()

regression_evaluator = RegressionEvaluator( metricName="rmse", labelCol="rating", predictionCol="prediction")

# Build cross validation using CrossValidator
cross_validator = CrossValidator(estimator=alternating_least_squares, estimatorParamMaps=parameter_grid, evaluator=regression_evaluator, numFolds=5)

cross_validator_model = cross_validator.fit(training)

ending_time = time.time()
execution_time = ending_time - starting_time

print(f"Execution time: {execution_time} seconds")

best_model = cross_validator_model.bestModel

prediction = best_model.transform(testing)

root_mean_square_error = regression_evaluator.evaluate(prediction)

print(root_mean_square_error)

# Best Model parameters
print ("Rank: ", best_model._java_obj.parent().getRank())
print ("MaxIter: ", str(best_model._java_obj.parent().getMaxIter()))
print ("RegParam:",  best_model._java_obj.parent().getRegParam())

"""### Model Testing"""

#Generate predictions and evaluate using RMSE
prediction=best_model.transform(testing)
root_mean_square_error = regression_evaluator.evaluate(prediction)
#Print Test RMSE
print(root_mean_square_error)

#Extract best model from the tuning exercise using ParamGridBuilder

alternating_least_squares_best = ALS(maxIter=15, rank=10, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
trained_model = alternating_least_squares_best.fit(training)

prediction.createOrReplaceTempView("predictions")
spark_session.sql("SELECT * FROM predictions limit 10").show()

transformed_data=best_model.transform(all_ratings)
root_mean_square_error = regression_evaluator.evaluate(transformed_data)
print ("Error is ",root_mean_square_error)

transformed_data.registerTempTable("data")

recommendations_top_10 = best_model.recommendForAllUsers(10)
recommendations_top_10.createOrReplaceTempView("Top10Recs")
spark_session.sql("SELECT * FROM Top10Recs limit 10").show()

recommendations_top_10.registerTempTable("Top_10_Recommendations")

# seperate the value of 'recommendations' in user_recs
unwrap_recommendations = spark_session.sql('SELECT userId, explode(recommendations) AS MovieRec FROM Top_10_Recommendations')
unwrap_recommendations.createOrReplaceTempView("unwrapRecs")
spark_session.sql("SELECT * FROM unwrapRecs limit 10").show()
final_recommendations = spark_session.sql("SELECT userId,movieIds_and_ratings.movieId\
                       AS movieId, movieIds_and_ratings.rating AS prediction\
                       FROM Top_10_Recommendations\
                       LATERAL VIEW explode(recommendations) exploded_table AS movieIds_and_ratings")
final_recommendations.createOrReplaceTempView("Recommendations")
spark_session.sql("SELECT * FROM Recommendations limit 10").show()

"""### Prediction of Users who haven't seen the Movie"""

recommendation = final_recommendations.join(all_ratings,['userId','movieId'],'left').filter(all_ratings.rating.isNull())
recommendation.createOrReplaceTempView("final_recommendations")
spark_session.sql("SELECT * FROM final_recommendations limit 20").show()

recommendation.registerTempTable("final_Recommendations")
movies.registerTempTable("movies_df")

"""### Recommending Movies to certain users"""

spark_session.sql("SELECT DISTINCT(userID) FROM final_recommendations limit 200").show(n=200)

spark_session.sql("SELECT userId, title \
            FROM final_Recommendations t1 \
            LEFT JOIN movies_df t2 \
            ON t1.movieId = t2.movieId \
            WHERE t1.userId=37 LIMIT 10").show()

spark_session.sql("SELECT userId, title \
            FROM final_Recommendations t1 \
            LEFT JOIN movies_df t2 \
            ON t1.movieId = t2.movieId \
            WHERE t1.userId=436 \
            LIMIT 10").show()