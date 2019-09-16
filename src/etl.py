import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.functions import from_unixtime
from pyspark.sql.types import *


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_ACCESS']['AWS_SECRET_ACCESS_KEY']
INPUT_BUCKET = config['S3']['INPUT_BUCKET']
OUTPUT_BUCKET = config['S3']['OUTPUT_BUCKET']

def create_spark_session():
    """Create a Spark Session to execute data processing.

    Returns
        spark: Spark Session object, object of Spark that enables commands

    """
    spark = SparkSession \
        .builder \
        .config('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:2.7.0') \
        .getOrCreate()
    return spark


def build_song_schema():
    """Build and return a schema to use for the song data.

    Returns
        schema: StructType object, a representation of schema and defined fields

    """
    schema = StructType(
        [
            StructField('artist_id', StringType(), True),
            StructField('artist_latitude', DecimalType(), True),
            StructField('artist_longitude', DecimalType(), True),
            StructField('artist_location', StringType(), True),
            StructField('artist_name', StringType(), True),
            StructField('duration', DecimalType(), True),
            StructField('num_songs', IntegerType(), True),
            StructField('song_id', StringType(), True),
            StructField('title', StringType(), True),
            StructField('year', IntegerType(), True)
        ]
    )
    return schema


def build_log_schema():
    """Build and return a Spark schema to use for the log data.

    Returns
        schema: StructType object, a representation of schema and defined fields

    """
    schema = StructType(
        [
            StructField('artist', StringType(), True),
            StructField('auth', StringType(), True),
            StructField('firstName', StringType(), True),
            StructField('gender', StringType(), True),
            StructField('itemInSession', IntegerType(), True),
            StructField('lastName', StringType(), True),
            StructField('length', DecimalType(), True),
            StructField('level', StringType(), True),
            StructField('location', StringType(), True),
            StructField('method', StringType(), True),
            StructField('page', StringType(), True),
            StructField('registration', LongType(), True),
            StructField('sessionId', IntegerType(), True),
            StructField('song', StringType(), True),
            StructField('status', IntegerType(), True),
            StructField('ts', LongType(), True),
            StructField('userAgent', StringType(), True),
            StructField('userId', StringType(), True)
        ]
    )
    return schema


def process_song_data(spark, input_bucket, output_bucket):
    """Transform the raw JSON data from S3 to Parquet columnar tables in S3.

    Args
        spark: Spark Session object, object of Spark that enables commands
        input_bucket: string, S3 bucket of raw data
        output_bucket: string, S3 bucket of processed data

    """
    # get filepath to song data files
    input_song_data = os.path.join(input_bucket, 'song_data')

    # read song data file
    song_df = spark.read.json(input_song_data, schema=build_song_schema())

    # extract columns to create songs table
    songs_table = song_df.select('song_id', 'title', 'artist_id', 'year', 'duration')

    # write songs table to parquet files partitioned by year and artist
    songs_table_output = os.path.join(output_bucket, 'songs_table.parquet')
    songs_table.write.parquet(path=songs_table_output, mode='overwrite', partitionBy=['year', 'artist_id'])

    # extract columns to create artists table
    artists_table = song_df.select('artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude')

    # write artists table to parquet files
    artists_table_output = os.path.join(output_bucket, 'artists_table.parquet')
    artists_table.write.parquet(path=artists_table_output, mode='overwrite')


def process_log_data(spark, input_bucket, output_bucket):
    """Transform the raw JSON data from S3 to Parquet columnar tables in S3.

    Args
        spark: Spark Session object, object of Spark that enables commands
        input_bucket: string, S3 bucket of raw data
        output_bucket: string, S3 bucket of processed data

    """
    # get filepath to log data files
    input_log_data = os.path.join(input_bucket, 'log_data')

    # read log data file
    log_df = spark.read.json(input_log_data, schema=build_log_schema())

    # filter by actions for song plays
    log_df = log_df.filter(col('page')=='Next Song')

    # extract columns for users table
    users_table = log_df.select('userId', 'firstName', 'lastName', 'gender', 'level')

    # write users table to parquet files
    users_data_output = os.path.join(output_bucket, 'users_table.parquet')
    users_table.write.parquet(path=users_data_output, mode='overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda ts: ts.divide(1000), TimestampType())
    log_df = log_df.withColumn('start_time', get_timestamp(col('ts')))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda ts: from_unixtime(ts.divide(1000), TimestampType()))
    log_df = log_df.withColumn('datetime', get_datetime(col('ts')))

    # extract columns to create time table
    time_table = log_df.select('start_time',
                               hour('datetime').alias('hour'),
                               dayofmonth('datetime').alias('dayofmonth'),
                               weekofyear('datetime').alias('weekofyear'),
                               month('datetime').alias('month'),
                               year('datetime').alias('year'),
                               date_format('datetime', 'EEEE').alias('weekday'))

    # write time table to parquet files partitioned by year and month
    time_data_output = os.path.join(output_bucket, 'time_table.parquet')
    time_table.write.parquet(path=time_data_output, mode='overwrite', partitionBy=['year', 'month'])

    # read in song data to use for songplays table
    input_song_data = os.path.join(input_bucket, 'song_data')
    song_df = spark.read.json(input_song_data, schema=build_song_schema())

    # extract columns from joined song and log datasets to create songplays table
    log_df.createOrReplaceTempView('log')
    song_df.createOrReplaceTempView('song')
    songplays_table = spark.sql("""SELECT
                                       log.start_time, \
                                       log.userId, log.level, song.song_id, song.artist_id, \
                                       log.sessionId, song.artist_location, log.userAgent,
                                       year(log.datetime) AS year, month(log.datetime) AS month \
                                   FROM log \
                                   JOIN song ON \
                                       log.song = song.title \
                                       AND log.artist = song.artist_name \
                                       AND log.length = song.duration""")

    # write songplays table to parquet files partitioned by year and month
    songplays_data_output = os.path.join(output_bucket, 'songplays_table.parquet')
    songplays_table.write.parquet(path=songplays_data_output, mode='overwrite', partitionBy=['year', 'month'])


def main():
    """Transform raw data from JSON files to parquet files using Spark."""
    # instantiate spark session
    spark = create_spark_session()

    # process data
    process_song_data(spark, INPUT_BUCKET, OUTPUT_BUCKET)
    process_log_data(spark, INPUT_BUCKET, OUTPUT_BUCKET)

if __name__ == '__main__':
    main()
