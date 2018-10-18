
import pyspark
import json
from pyspark.sql.functions import col
import pyspark.ml.feature as ft
import pyspark.ml.clustering as clus

spark = pyspark.sql.SparkSession.builder.appName("stats_and_sankey").getOrCreate()


SEED = 42
num_topics=45
file_in = "s3://eherraiz-bigdatacert/data/atari_topic_names"
models_in = "s3://eherraiz-bigdatacert/models"


df = spark.read.json(file_in)

# topic_forum_distr = df.groupBy(col('forum_title'), col('topic_title')).count().sort(col('forum_title'), col('topic_title')).take(1000)
links_A = df\
	.where(col('subforum_title') == 'NA')\
	.groupBy(col('forum_title'), col('topic_title'))\
	.count()\
	.sort(col('forum_title'), col('topic_title'))\
	.take(1000)

links = df\
	.where(col('subforum_title') != 'NA')\
	.groupBy(col('forum_title'), col('subforum_title'))\
	.count()\
	.sort(col('forum_title'), col('subforum_title'))\
	.take(1000)

links_C = df\
	.where(col('subforum_title') != 'NA')\
	.groupBy(col('subforum_title'), col('topic_title'))\
	.count()\
	.sort(col('subforum_title'), col('topic_title'))\
	.take(1000)


links = [{'source': e['forum_title'], 'target': e['topic_title'], 'value': e['count'], 'type': e['topic_title']} for e in links_A] + \
		[{'source': e['forum_title'], 'target': e['subforum_title'], 'value': e['count'], 'type': e['forum_title']} for e in links_B] + \
		[{'source': e['subforum_title'], 'target': e['topic_title'], 'value': e['count'], 'type': e['topic_title']} for e in links_C]

print("##############################################################")
print("##############################################################")


print(json.dumps(links))
f = open("/tmp/links.json","w+")
f.write(json.dumps(links))
f.close()


print("##############################################################")
print("##############################################################")

node_set = set()
for n in links:
    node_set.add(n['source'])
    node_set.add(n['target'])

nodes = [{'id': t, 'title': t} for t in node_set]

print(json.dumps(nodes))
f = open("/tmp/nodes.json","w+")
f.write(json.dumps(nodes))
f.close()


print("##############################################################")
print("##############################################################")

df_forum_titles = df.select(col('forum_title')).distinct().sort(col('forum_title')).take(1000)
order_forum_titles = [r['forum_title'] for r in df_forum_titles if r['forum_title'] in nodes]

df_subforum_titles = df.where(col('subforum_title') != 'NA').select(col('forum_title'), col('subforum_title')).distinct().sort(col('forum_title'), col('subforum_title')).take(1000)
order_subforum_titles = [r['subforum_title'] for r in df_subforum_titles if r['subforum_title'] in nodes]

df_topic_titles = df.select(col('topic_title')).distinct().sort(col('topic_title')).take(1000)
order_topic_titles = [r['topic_title'] for r in df_topic_titles if r['topic_title'] in nodes]


order = [
	[
		order_forum_titles
	],
	[
		order_subforum_titles
	],
	[
		order_topic_titles
	]
]

print(json.dumps(order))
f = open("/tmp/order.json","w+")
f.write(json.dumps(order))
f.close()

print("##############################################################")
print("##############################################################")

sankey = {
	"links": links,
	"order": order,
	"nodes": nodes
}

f = open("/tmp/sankey.json","w+")
f.write(json.dumps(sankey))
f.close()




print("##############################################################")
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

f = open("/tmp/topic_terms.json","w+")
f.write(json.dumps(topic_terms))
f.close()

print("##############################################################")
print("##############################################################")


print("##############################################################")
print("##############################################################")
df.groupBy('topic_title').count().orderBy('count').show()
print("##############################################################")
print("##############################################################")

