import pyspark
import pyspark.sql.types as typ
import pyspark.sql.functions as fn
from pyspark.sql.functions import col
from graphframe import *
import datetime

f = open("/home/hadoop/execution_times.log", "w")
f.write("Iniciando ejecucion...\n")

spark = pyspark.sql.SparkSession.builder.appName("graphs").getOrCreate()

edges_in = "s3://eherraiz-bigdatacert/data/atari_graphs/1_edges_ungrouped"
vertices_in = "s3://eherraiz-bigdatacert/data/atari_graphs/1_vertices"

file_out_lpa = "s3://eherraiz-bigdatacert/data/atari_graphs/1_results_lpa"
file_out_pagerank = "s3://eherraiz-bigdatacert/data/atari_graphs/1_results_pagerank"

print("Cargando ficheros...")

edges = spark.read.csv(edges_in).toDF('src', 'dst', 'topic_title', 'relationship')
vertices = spark.read.csv(vertices_in).toDF('id')

msg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" Creando grafo...\n"
f.write(msg)
print(msg)

g = GraphFrame(vertices, edges)

msg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" Calculando LPA...\n"
f.write(msg)
print(msg)

lpa = g.labelPropagation(maxIter=3)

msg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" Calculando PageRank...\n"
f.write(msg)
print(msg)

pagerank = g.pageRank(resetProbability=0.15, tol=0.001)

msg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" PageRank con GraphFrames terminado.\n"
f.write(msg)
print(msg)

f.close()
