import pyspark
import pyspark.sql.functions as fn
from pyspark.sql.functions import col

SEED = 42

file_in = "s3://eherraiz-bigdatacert/data/atari_topics_ids"
file_out = "s3://eherraiz-bigdatacert/data/atari_topic_names"
summary_out = "s3://eherraiz-bigdatacert/data/atari_topic_summary"

num_topics = 45

spark = pyspark.sql.SparkSession.builder.appName("forum_topic_names").getOrCreate()

print("cargando relacion topic_id-topic_names")
topic_titles = {
    0: "Dev",
    1: "HW",  # mods, repairs
    2: "HW",  # mods
    3: "Market",
    4: "Games",  # releases
    5: "Console Systems",  # console parts, details, photos, marketplace...
    6: "Music",  # dev, soundtracks...
    7: "Games",
    8: "Games",
    9: "Dev",
    10: "HW",  # mods, upgrades
    11: "Games",  # discussion, expo, conventions
    12: "Dev",  # + Game discussion
    13: "Games",  # play, guides, cheats
    14: "Console Systems",  # Commodore o Games, es muy especifico de Commodore
    15: "HW",  # especifico de joysticks
    16: "Social",  # games, console systems, discussion...
    17: "Social",  # discussion, what if, favorites...
    18: "Dev",
    19: "HW",
    20: "Social",  # console systems, mods, devs, help...
    21: "Console Systems",
    22: "Social",  # expos, scans, magazines
    23: "HW",
    24: "Social",  # games, hw
    25: "Social",  # games, recommendations
    26: "HW",  # mods, controllers
    27: "Market",
    28: "HW",  # diy, problems..
    29: "Social",  # pero sobre temas de marketplace
    30: "Social",  # questions, offtopic, recommendations
    31: "Dev",  # DIY
    32: "Dev",  # con mucho HW
    33: "Console Systems",  # Console systems... muy Jaguar specific
    34: "Market",
    35: "Social",  # collections sobre todo, dicussions
    36: "Social",  # nintendo, videos
    37: "Games",  # discussion
    38: "Social",  # discussion
    39: "Social",  # collections, looking for stuff...
    40: "Games",  # expo, releases
    41: "Market",
    42: "HW",  # sobre todo video, + marketplace
    43: "Social",  # discussion about everything
    44: "Console Systems"
 }
lookup = spark.sparkContext.parallelize(topic_titles.items()).toDF(["topic_id", "topic_title"])

print("cargado datos en memoria...")
df = spark.read.json(file_in)
df = df.join(lookup, ["topic_id"], "left").na.fill("topic_title", "--")
df.createOrReplaceTempView('df')

print("renombrando algunos topic titles...")
spark.sql("""CREATE OR REPLACE TEMP VIEW non_english AS
SELECT *, IF(forum_code = '9-international/', 'Non-English', topic_title) AS topic_title_2 FROM df""")

spark.sql("""CREATE OR REPLACE TEMP VIEW hsc AS
SELECT *,  IF(subforum_title like '%High Score Club%', 'HSC', topic_title_2) AS topic_title_3 FROM non_english""")

df = spark.sql("SELECT * FROM hsc")\
        .drop('topic_title').drop('topic_title_2')\
        .withColumnRenamed('topic_title_3', 'topic_title')


print("grabando a disco...")
df.write.mode("overwrite").save(file_out, format="json")
print("listo")

print("agrupando hilos por topic...")
threads = df\
    .groupBy(col('thread_code_url'), col('topic_id'), col('topic_title'))\
    .agg(fn.collect_list('post_text').alias('thread_text'), fn.min('post_date').alias('thread_date'), fn.min('thread_title').alias('thread_title'))\
    .withColumn('thread_text', fn.concat_ws(' ', col('thread_text')))
threads.createOrReplaceTempView('threads')

print("grabando ejemplo de topics a disco...")
summary = spark.sql("""
SELECT
  topic_id,
  topic_title,
  thread_title,
  thread_text
FROM (
  SELECT
    topic_id,
    topic_title,
    thread_text,
    thread_title,
    rank() OVER (PARTITION BY topic_id ORDER BY thread_date DESC) as rank
  FROM threads) tmp
WHERE
  rank <= 20
ORDER BY topic_title, topic_id ASC
""")

summary.repartition(1).write.mode("overwrite").save(summary_out, format="json")

summary = summary.take(100000)
table_rows = ["<tr><td>t{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(r['topic_id'], r['topic_title'].encode("ascii", errors="ignore"), r['thread_title'].encode("ascii", errors="ignore"), r['thread_text'].encode("ascii", errors="ignore")) for r in summary]
html = """<html>
<head></head>
<body>
<table border=1>{}</table>
</body>
</html>""".format("\n".join(table_rows))

f = open("/tmp/summary.html", "w+")
f.write(html)
f.close()
print("listo")
