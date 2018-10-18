import pyspark
import pyspark.sql.types as typ
import pyspark.sql.functions as fn
from pyspark.sql.functions import col
from graphframe import *
import datetime
import networkx as nx
import pandas as pd
from networkx.algorithms import community

f = open("/home/hadoop/execution_times_networkx.log", "w")
f.write("Iniciando ejecucion...\n")

spark = pyspark.sql.SparkSession.builder.appName("graphs").getOrCreate()

edges_in = "s3://eherraiz-bigdatacert/data/atari_graphs/1_edges_ungrouped"
vertices_in = "s3://eherraiz-bigdatacert/data/atari_graphs/1_vertices"

edges = spark.read.csv(edges_in).toDF('src', 'dst', 'topic_title', 'relationship')
vertices = spark.read.csv(vertices_in).toDF('id')

msg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" Creando grafo...\n"
f.write(msg)
print(msg)

edges_ungrouped_pd = edges.toPandas()

g_nx_ungrp = nx.from_pandas_edgelist(edges_ungrouped_pd, 'src', 'dst', ['topic_title', 'relationship'], create_using=nx.DiGraph())

msg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" Calculando PageRank...\n"
f.write(msg)
print(msg)

pg_nx_ungrp = nx.pagerank(g_nx_ungrp, tol=0.001)

msg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" PageRank con NetworkX terminado.\n"
f.write(msg)
print(msg)


g_nx_ungrp_undir = nx.from_pandas_edgelist(edges_ungrouped_pd, 'src', 'dst', ['topic_title', 'relationship'], create_using=nx.Graph())

msg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" Calculando LPA con NetworkX...\n"
f.write(msg)
print(msg)

lpa = community.label_propagation_communities(g_nx_ungrp_undir)

for c in lpa:
  print(c)


msg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" LPA con NetworkX terminado.\n"
f.write(msg)
print(msg)

f.close()
