SEED = 42
models_in = "s3://eherraiz-bigdatacert/models"
file_in = "s3://eherraiz-bigdatacert/data/atari_clean"
file_out = "s3://eherraiz-bigdatacert/data/atari_topics_ids"

num_topics=45

import pyspark
import pyspark.sql.types as typ
import pyspark.sql.functions as fn
from pyspark.sql.functions import col
spark = pyspark.sql.SparkSession.builder.appName("forum_transform").getOrCreate()


import string
import re

from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.types import ArrayType, StringType


class WordNoiseRemover(Transformer, HasInputCol, HasOutputCol):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(WordNoiseRemover, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def _transform(self, dataset):

        def f(s):
            terms = list()
            domains = re.compile("https?:\/\/([a-z0-9.-]+)\/")
            # el tokenizer ya separa las URLs en cachitos, asi que los dominios ya nos llegan sin necesidad de esto
            # eso si, no llega con formato noseque.com, sino solo "noseque", porque el tokenizer parte por los puntos
            for t in s:
                add = True
                term_to_add = ""
                url = re.match(domains, t)
                if url is not None:
                    term_to_add = url.group(0)
                elif "0x" in t:
                    term_to_add = "MEM"
                else:
                    term_to_add = ""
                    try:
                        if type(t) == str:
                            in_ascii = t
                        else:
                            in_ascii = t.encode('ascii', errors='ignore')

                        term_to_add = in_ascii.strip(string.punctuation + " ")
                        add = True
                        for c in "()!#/\\:?!@":
                            if c in term_to_add:
                                add = False
                       
                        if "\xa0" in term_to_add:
                            add = False

                    except:
                        add = False

                if add and len(term_to_add) > 1:
                    terms.append(term_to_add)
            return terms

        t = ArrayType(StringType())
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))

# ----------------

from pyspark.sql.functions import udf

def vector_to_array_udf(vector):
    values = vector.values
    max_index = max(range(len(values)), key=values.__getitem__)
    return max_index

vector_to_array = udf(vector_to_array_udf, typ.IntegerType())



import pyspark.ml.feature as ft
import pyspark.ml.clustering as clus
from pyspark.ml import Pipeline, PipelineModel

# spark.conf.set('spark.sql.autoBroadcastJoinThreshold', -1)

print('cargando modelos en memoria...')
print('- tokenizer...')
regex_tokenizer_load = ft.RegexTokenizer.load(models_in+'/regex_tokenizer_model')
print('- noise remover...')
word_noise_remover_load = WordNoiseRemover(inputCol=regex_tokenizer_load.getOutputCol(), outputCol = "noise_removed")
print('- stopwords...')
stopwords_remover_load = ft.StopWordsRemover.load(models_in+'/stopwords_remover_model')
print('- vectorizer...')
count_vectorizer_model_load = ft.CountVectorizerModel.load(models_in+'/count_vectorizer_model')
print('- lda...')
lda_model_load = clus.DistributedLDAModel.load(models_in+'/lda_model')

pipeline_model = PipelineModel(stages=[
                                 regex_tokenizer_load,
                                 word_noise_remover_load,
                                 stopwords_remover_load,
                                 count_vectorizer_model_load,
                                 lda_model_load])
print("listo")



print("cargando datos en memoria...")
df = spark.read.json(file_in)
df.cache()

print("agregando hilos...")
threads = df.withColumn('post_text', fn.lower(col('post_text')))\
    .groupBy(col('thread_code_url'))\
    .agg(fn.collect_list('post_text').alias('thread_text'))\
    .withColumn('thread_text', fn.concat_ws(' ', col('thread_text')))\
    .cache()
print("listo")


print("aplicando modelo...")
threads_with_topic_prob = pipeline_model.transform(threads)

print("extrayendo topic id por hilos...")
threads_with_topic_id = threads_with_topic_prob.withColumn('topic_id', vector_to_array(col('topicDistribution')))\
  .select('thread_code_url', 'topic_id')\
  .withColumnRenamed('thread_code_url', 'desambiguated_index')

print("extrayendo topic id por post...")
posts_with_topic_id = df\
    .join(threads_with_topic_id, df['thread_code_url'] == threads_with_topic_id['desambiguated_index'], how='inner')\
    .drop('desambiguated_index')

print("grabando topic id a disco")
posts_with_topic_id.write.mode("overwrite").save(file_out, format="json")
print("listo")

print("calculando resumen de topics...")
topics = lda_model_load.describeTopics(15).take(100)
topic_terms = dict()
topic_index = dict()
for topic in topics:
    terms_in_topic = []
    for i, v in enumerate(topic['termIndices']):
        if v not in topic_index:
            topic_index[v] = (count_vectorizer_model_load.vocabulary[v], topic['termWeights'][i])
        terms_in_topic.append(topic_index[v])

    topic_terms[topic['topic']] = terms_in_topic

f = open("/tmp/topic_terms.json","w+")
f.write(json.dumps(topic_terms))
f.close()

lookup_weights = spark.sparkContext.parallelize(topic_terms.items()).toDF(["topic_id", "tuple"])
df_term_weight = posts_with_topic_id.groupBy('topic_id')\
    .count()\
    .join(lookup_weights, ["topic_id"], "left")\
    .na.fill("tuple", "--")\
    .withColumn('tuple', fn.explode(col('tuple')))\
    .select(col("topic_id"), col("tuple._1"), col('tuple._2'), col('count'))\
    .withColumn('weight_per_topic', col('_2') * col('count'))\
    .select(col('_1').alias('term'), col('weight_per_topic'))\
    .groupBy('term')\
    .agg(fn.sum('weight_per_topic').alias('weight'))\
    .take(45*15)

max_font_size = 100
min_font_size = 10

vega_term_weights = [{"text": r['term'], 'value': r['weight']} for r in df_term_weight]
all_weights = [t['value'] for t in vega_term_weights]
factor = (max(all_weights) - min(all_weights)) / (max_font_size - min_font_size)
vega_term_weights = [{"text": r['term'], 'value': r['weight'] / factor + min_font_size} for r in df_term_weight]

f = open("/tmp/vega_term_weights.json","w+")
f.write(json.dumps(vega_term_weights))
f.close()
