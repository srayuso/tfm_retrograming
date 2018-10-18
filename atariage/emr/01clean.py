SEED = 42
file_in = "s3://eherraiz-bigdatacert/data/atari_source.json"
file_out = "s3://eherraiz-bigdatacert/data/atari_clean"

import pyspark
import pyspark.sql.types as typ
import pyspark.sql.functions as fn
from pyspark.sql.functions import col
spark = pyspark.sql.SparkSession.builder.appName("forum_clean").getOrCreate()

print("cargado en memoria...")
df = spark.read.json(file_in)
df.cache()
df.createOrReplaceTempView('df')


print("cargando queries de limpieza...")
spark.sql("""CREATE OR REPLACE TEMP VIEW clean_temp1 AS
SELECT * FROM df WHERE thread_code_url not in 
	('33233-sorted-table-of-contents/',
	 '47348-high-score-club-the-rules-info-updated-11715/',
	 '72981-contribute-2600-game-descriptions-read-me/',
	 '146202-how-to-use-this-forum-read-this-post/',
	 '213378-use-tags-when-posting/',
	 '156176-please-use-ebay-bbcode-tags-when-posting-ebay-auctions-sellers-stores/',
	 '122324-psn-ids/',
	 '263706-switch-friend-code-list/',
	 '216296-atariage-nintendo-network-id-list-wii-u/',
	 '145504-ea-online-buddy-ids/')""")

spark.sql("""CREATE OR REPLACE TEMP VIEW clean_temp2 AS
SELECT * FROM clean_temp1 WHERE forum_code not in ('70-member-blogs/', '26-announcements/')""")

# spark.sql("""CREATE OR REPLACE TEMP VIEW clean_temp3 AS
# SELECT * FROM clean_temp2 WHERE WHERE subforum_title not like '%High Score Club%'""")

# spark.sql("""CREATE OR REPLACE TEMP VIEW thread_count AS
# SELECT thread_code as thread_code_count, COUNT(thread_code) as count FROM clean_temp3 GROUP BY thread_code""")


spark.sql("""CREATE OR REPLACE TEMP VIEW thread_count AS
SELECT thread_code as thread_code_count, COUNT(thread_code) as count FROM clean_temp2 GROUP BY thread_code""")

spark.sql("""CREATE OR REPLACE TEMP VIEW clean_temp4 AS
SELECT * FROM clean_temp2 AS v1
JOIN thread_count AS v2 ON v1.thread_code = v2.thread_code_count
WHERE v2.count > 1""")

print("aplicando queries de limpieza...")
cleaned = spark.sql("SELECT * FROM clean_temp4")\
  .drop('thread_code_count')\
  .drop('count')\
  .drop('user_link')\
  .drop('user_id')\
  .drop('thread_code')\
  .drop('quote_post')


cleaned.cache()

print("escribiendo a disco...")
cleaned.write.mode("overwrite").save(file_out, format="json")


print("done")