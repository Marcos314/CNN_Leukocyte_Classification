import sh
import os

novaLinha = 'MES_REFERENCIA;MES_COMPETENCIA;UF;CODIGO_MUNICIPIO_SIAFI;NOME_MUNICIPIO;NIS_FAVORECIDO;NOME_FAVORECIDO;VALOR_PARCELA'
directory = os.fsencode("/var/lib/neo4j/import/")

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith("1.csv"): 
         sh.sed("-i", "1s/.*/" + novaLinha + "/", filename)