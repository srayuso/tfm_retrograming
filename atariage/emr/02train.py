
SEED = 42
file_in = "s3://eherraiz-bigdatacert/data/atari_clean"
models_out = "s3://eherraiz-bigdatacert/models"
train_percent = 1.0

num_topics=45
method='em'
doc_concentration = [1.1]
topic_concentration = 4.0


import pyspark
import pyspark.sql.types as typ
import pyspark.sql.functions as fn
from pyspark.sql.functions import col
spark = pyspark.sql.SparkSession.builder.appName("forum_train").getOrCreate()



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



print("cargado en memoria...")
# df = spark.read.json(file_in)
# for_train = df.where(col('forum_code') != '9-international/')
# for_train.cache()

df = spark.read.json(file_in)
df.createOrReplaceTempView('df')

spark.sql("""CREATE OR REPLACE TEMP VIEW for_train AS
SELECT * FROM df WHERE WHERE subforum_title not like '%High Score Club%' and forum_code != '9-international/' """)

for_train = spark.sql("SELECT * FROM for_train")
for_train.cache()



print("agregando hilos...")
threads = for_train.withColumn('post_text', fn.lower(col('post_text')))\
    .groupBy(col('thread_code_url'))\
    .agg(fn.collect_list('post_text').alias('thread_text'))\
    .withColumn('thread_text', fn.concat_ws(' ', col('thread_text')))\
    .cache()

threads_sample = threads.sample(False, train_percent, SEED)
print("listo para entrenar")



df_input=threads_sample
doc_column='thread_text'
index_column='thread_code_url'


import pyspark.ml.feature as ft
import pyspark.ml.clustering as clus
from pyspark.ml import Pipeline


if method == 'online':
    if doc_concentration is None:
        doc_concentration = [1.0/num_topics] # 1.0/k es el default
    if topic_concentration is None:
        topic_concentration = float(1.0/num_topics) # 1.0/k es el default
elif method == 'em':
    if doc_concentration is None:
        doc_concentration = [1.1]
    if topic_concentration is None:
        topic_concentration = 1.1
if not isinstance(doc_concentration, list):
    doc_concentration = [doc_concentration]    

regex_tokenizer = ft.RegexTokenizer(
       inputCol=doc_column,
       outputCol='tokenized',
       pattern='\s+|[,.\"]')

word_noise_remover = WordNoiseRemover(inputCol=regex_tokenizer.getOutputCol(),
                                      outputCol = "noise_removed")

stopwords_remover = ft.StopWordsRemover(
       inputCol=word_noise_remover.getOutputCol(),
       outputCol='no_stop_words')


stopwords_collected = [
  u'00',
  u'3d',
  u'actually',
  u'ago',
  u'already',
  u'also',
  u'always',
  u'another',
  u'anyone',
  u'anything',
  u'around',
  u'atari',
  u'attach',
  u'back',
  u'better',
  u'bit',
  u'boy',
  u'box',
  u'cart',
  u'carts',
  u'destination',
  u'different',
  u'done',
  u'edited',
  u'even',
  u'find',
  u'first',
  u'forum',
  u'game',
  u'games',
  u'get',
  u'go',
  u'going',
  u'good',
  u'got',
  u'great',
  u'guy',
  u'guys',
  u'hi',
  u'high',
  u'hour',
  u'hours',
  u'jpg',
  u'kb',
  u'know',
  u'last',
  u'like',
  u'liked',
  u'likes',
  u'little',
  u'look',
  u'looking',
  u'looks',
  u'lot',
  u'lots',
  u'made',
  u'make',
  u'making',
  u'many',
  u'may',
  u'maybe',
  u'might',
  u'much',
  u'need',
  u'never',
  u'next',
  u'new',
  u'nice',
  u'old',
  u'one',
  u'original',
  u'people',
  u'play',
  u'please',
  u'pm',
  u'png',
  u'post',
  u'pretty',
  u'probably',
  u'put',
  u'really',
  u'reply',
  u'right',
  u'say',
  u'see',
  u'shirt',
  u'since',
  u'someone',
  u'something',
  u'still',
  u'stuff',
  u'sure',
  u'system',
  u'thanks',
  u'thing',
  u'things',
  u'think',
  u'though',
  u'take',
  u'time',
  u'try',
  u'two',
  u'us',
  u'use',
  u'used',
  u'using',
  u'vb',
  u'version',
  u'virtual',
  u'want',
  u'way',
  u'well',
  u'word',
  u'work',
  u'years']

stopwords_numbers = [str(n) for n in range(100)]
stopwords_punctuation = [c for c in string.punctuation] + [c * 2 for c in string.punctuation] + [c * 3 for c in string.punctuation]
stopwords_default = stopwords_remover.loadDefaultStopWords('english')
stopwords_mix = stopwords_collected + stopwords_numbers + stopwords_punctuation + stopwords_default

stopwords_remover.setStopWords(stopwords_mix)

count_vectorizer = ft.CountVectorizer(
       inputCol=stopwords_remover.getOutputCol(),
       outputCol="vector_counts")

lda = clus.LDA(k=num_topics,
   optimizer=method,
   featuresCol=count_vectorizer.getOutputCol(),
   seed=SEED,
   docConcentration=doc_concentration,
   topicConcentration=topic_concentration)

pipeline = Pipeline(stages=[
           regex_tokenizer,
           word_noise_remover,
           stopwords_remover,
           count_vectorizer,
           lda]
)

print("fiteando modelo...")
pipeline_model = pipeline.fit(df_input)
print("fiteo completado")



print("guardando modelos a disco...")
regex_tokenizer_model = pipeline_model.stages[0]
stopwords_remover_model = pipeline_model.stages[2]
count_vectorizer_model = pipeline_model.stages[3]
lda_model = pipeline_model.stages[4]

print("- tokenizer...")
regex_tokenizer_model.write().overwrite().save(models_out+'/regex_tokenizer_model')
print("- stopwords...")
stopwords_remover_model.write().overwrite().save(models_out+'/stopwords_remover_model')
print("- vectorizer...")
count_vectorizer_model.write().overwrite().save(models_out+'/count_vectorizer_model')
print("- lda...")
lda_model.write().overwrite().save(models_out+'/lda_model')

print("calculando resumen de topics...")
topics = lda_model.describeTopics(7).take(100)
topic_terms = dict()
for topic in topics:
    topic_terms[topic['topic']] = [count_vectorizer_model.vocabulary[i] for i in topic['termIndices']]

for k, v in topic_terms.items():
    print("#{}: {}".format(k, v))

