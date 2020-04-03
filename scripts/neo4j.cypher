LOAD CSV WITH HEADERS FROM "file:///201501.csv" as line FIELDTERMINATOR ';' CREATE (:Pagamento { mesReferencia: line.MES_REFERENCIA, mesCompetencia: line.MES_COMPETENCIA, nomeMunicipio: line.NOME_MUNICIPIO, valorParcela: line.VALOR_PARCELA,codigoMunicipio: line.CODIGO_MUNICIPIO_SIAFI, uf: line.UF})

LOAD CSV WITH HEADERS FROM "file:///home/marcos/Desktop/201501.csv" as line FIELDTERMINATOR ';' CREATE (:Pagamento { mesReferencia: line[0], mesCompetencia: line[1], nomeMunicipio: line[4],  valorParcela: line[7],codigoMunicipio: line[3], uf: line[2]})


 LOAD CSV WITH HEADERS FROM "file:///201501.csv" as line  FIELDTERMINATOR ';' CREATE (:Pagamento {mesReferencia: line.MES_REFERENCIA, mesCompetencia: line.MES_COMPETENCIA, nomeMunicipio: line.NOME_MUNICIPIO, valorParcela: line.VALOR_PARCELA,codigoMunicipio: line.CODIGO_MUNICIPIO_SIAFI, uf: line.UF})