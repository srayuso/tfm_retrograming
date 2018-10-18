
import pyspark
import json
from pyspark.sql.functions import col
import pyspark.ml.feature as ft
import pyspark.ml.clustering as clus

spark = pyspark.sql.SparkSession.builder.appName("stats_and_sankey").getOrCreate()


SEED = 42
num_topics = 45
file_in = "s3://eherraiz-bigdatacert/data/atari_topic_names"
models_in = "s3://eherraiz-bigdatacert/models"
files_out = '/tmp/'

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
    14: "Console Systems",  # o Games, es muy especifico
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
    33: "Console Systems",  # Console systems... very specific
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

# http://colorbrewer2.org/?type=qualitative&scheme=Set3&n=11

colors = [
    'rgb(141,211,199)',
    'rgb(255,255,179)',
    'rgb(190,186,218)',
    'rgb(251,128,114)',
    'rgb(128,177,211)',
    'rgb(253,180,98)',
    'rgb(179,222,105)',
    'rgb(252,205,229)',
    'rgb(217,217,217)',
    'rgb(188,128,189)',
    'rgb(204,235,197)'
]

topic_list = {t for t in topic_titles.values()}
topic_list.add('Non-English')
topic_list.add('HSC')

topic_color = {}
for i, v in enumerate(topic_list):
    topic_color[v] = colors[i]


df = spark.read.json(file_in)

links_A = df\
    .groupBy(col('forum_title'), col('topic_title'))\
    .count()\
    .sort(col('forum_title'), col('topic_title'))\
    .take(1000)

links = [{'source': e['forum_title'], 'target': e['topic_title'], 'value': e['count'], 'type': e['topic_title'], 'color': topic_color.get(e['topic_title'], 'rgb(0,0,0)')} for e in links_A]

f = open(files_out+"links.json", "w+")
f.write(json.dumps(links))
f.close()


print("links done ##############################################################")

node_set = set()
for n in links:
    node_set.add(n['source'])
    node_set.add(n['target'])

nodes = [{'id': t, 'title': t} for t in node_set]

f = open(files_out+"nodes.json", "w+")
f.write(json.dumps(nodes))
f.close()


print("nodes done ##############################################################")

df_forum_titles = df.select(col('forum_title')).distinct().sort(col('forum_title')).take(1000)
order_forum_titles = [r['forum_title'] for r in df_forum_titles if r['forum_title'] in node_set]

df_topic_titles = df.select(col('topic_title')).distinct().sort(col('topic_title')).take(1000)
order_topic_titles = [r['topic_title'] for r in df_topic_titles if r['topic_title'] in node_set]


order = [
    [
        order_forum_titles
    ],
    [
        order_topic_titles
    ]
]

f = open(files_out+"order.json", "w+")
f.write(json.dumps(order))
f.close()

print("order done ##############################################################")

sankey = {
    "links": links,
    "order": order,
    "nodes": nodes
}


f = open(files_out+"sankey.json", "w+")
f.write(json.dumps(sankey))
f.close()

print("all done ##############################################################")

print("##############################################################")

df.groupBy('topic_title').count().orderBy('count').show(num_topics+2)


print("##############################################################")
print("##############################################################")


print('cargando modelos en memoria...')
print('- vectorizer...')
count_vectorizer_model_load = ft.CountVectorizerModel.load(models_in+'/count_vectorizer_model')
print('- lda...')
lda_model_load = clus.DistributedLDAModel.load(models_in+'/lda_model')


print("calculando resumen de topics...")
topics = lda_model_load.describeTopics(15).take(100)
topic_terms = dict()
topic_index = dict()
j = 0
for topic in topics:
    print("topic{}: {}".format(j, topic))
    j = j + 1
    terms_in_topic = []
    for i in topic['termIndices']:
        if i not in topic_index:
            topic_index[i] = count_vectorizer_model_load.vocabulary[i]
        terms_in_topic.append(topic_index[i])

    topic_terms[topic['topic']] = terms_in_topic
    # topic_terms[topic['topic']] = [count_vectorizer_model_load.vocabulary[i] for i in topic['termIndices']]

f = open(files_out+"topic_terms.json", "w+")
f.write(json.dumps(topic_terms))
f.close()

print("##############################################################")
print("##############################################################")


print("##############################################################")
print("##############################################################")
df.groupBy('topic_title').count().orderBy('count').show()
print("##############################################################")
print("##############################################################")
