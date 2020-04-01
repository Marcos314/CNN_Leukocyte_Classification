import pandas as pd
import os


#file = [ f for f in os.listdir('/var/lib/neo4j/import/') if f.endswith('1.csv')]


csv = pd.read_csv('/var/lib/neo4j/import/201501.csv')

print(csv.columns)

csv.columns = csv.columns.str.replace(' ','_')

csv = pd.to_csv('/var/lib/neo4j/import/201501-2.csv')

print(csv.columns)
