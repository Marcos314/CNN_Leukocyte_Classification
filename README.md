# Repositório de códigos do TCC de Marcos Wesley


## Instalação do ambiente com conda

#### Criação do ambiente virtual (executar apenas uma vez)
`conda create -n machine-learning python=3.6`

#### Ativar o ambiente virtual criado (sempre que for usar)

`conda activate machine-learning`


#### Instalação dos pacotes necessários (apenas na primeira vez que entrar no ambiente)

1. Pode ser feito de uma vez usando:
```Bash
conda env create -f environment.yml

```

2. Pode ser feito separadamente, usando:

```Bash
conda install pandas numpy tensorflow keras pillow scikit-learn jupyterlab -y

conda install -c conda-forge matplotlib opencv tqdm -y

conda install -c anaconda seaborn -y

```

#### Sair do ambiente virtual

`conda deactivate`	


## Dataset utilizado

Dataset original: https://data.mendeley.com/datasets/snkd93bnjr/1

O dataset original foi usado como única fonte de imagens, removendo as classes não estudadas. Os tipos de células que foram estudadas são:

- **Basófilos**
- **Leucócitos**
- **Neutrófilos**
- **Eosinófilos**
- **Monócitos**

Este novo dataset será migrado para um plataforma livre após finalizar o TCC com a mesma licença do original (CC BY 4.0).

