from pyspark import SparkConf
from pyspark.sql import SparkSession

BUCKET = "dmacademy-course-assets"
file_pre = "vlerick/pre_release.csv"
file_after = "vlerick/after_release.csv"


config = {
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
    "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
}
conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

df_pre = spark.read.csv(f"s3a://{BUCKET}/{file_pre}", header=True)
df_after = spark.read.csv(f"s3a://{BUCKET}/{file_after}", header=True)

df_pre.show()
df_after.show()

