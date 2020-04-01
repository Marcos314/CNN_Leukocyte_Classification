import findspark
findspark.init()

from pyspark import SparkContext, SparkConf


sc = pyspark.SparkContext('local[*]')

txt = sc.textFile('/home/marcos/Desktop/TCC_2020/dados/201501.csv')

print(txt.count())

python_lines = txt.filter(lambda line: 'python' in line.lower())
print(python_lines.count())