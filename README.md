# 📊 A Qualificação da investigação criminal no Rio Grande do Norte: fatores determinantes para a elucidação de homicídios

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repositório oficial com os códigos-fonte, *scripts* estatísticos e a base de dados anonimizada utilizados na Dissertação de Mestrado focada na eficácia da investigação de homicídios dolosos no Estado do Rio Grande do Norte.

## 📌 Sobre o Projeto

Este projeto aplica técnicas avançadas de **Ciência de Dados e Machine Learning** para auditar, mensurar e prever a eficácia dos Inquéritos Policiais de homicídio conduzidos pela Polícia Civil do Rio Grande do Norte (PCRN). 

A pesquisa refuta a intuição analógica e comprova, matematicamente, a transição para um novo paradigma: a superação da impunidade depende da adoção de uma racionalidade investigativa alicerçada na tríade composta por **tecnologia analítica, especialização funcional e proatividade policial**.

## 🗂️ O Banco de Dados

A análise foi conduzida sobre uma amostra robusta de **N = 172 inquéritos policiais** reais (entre casos com autoria elucidada e casos arquivados/infrutíferos). 

* **Anonimização:** Em estrito cumprimento à Lei Geral de Proteção de Dados (LGPD) e às diretrizes éticas de pesquisa, a base de dados disponibilizada neste repositório (`metadados_inqueritos.csv`) foi totalmente anonimizada. Nomes de investigados, vítimas, números de inquérito e dados sensíveis foram removidos, preservando-se apenas os **metadados quantitativos** (diligências executadas, prazos e desfechos) necessários para a reprodutibilidade dos modelos estatísticos.

## 🛠️ Tecnologias e Modelos Utilizados

O *script* principal (`analise_inqueritos.py`) utiliza a linguagem Python e bibliotecas consagradas de análise de dados (`pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`) para rodar os seguintes algoritmos:

1. **Random Forest Classifier:** Para medir o "peso preditivo" (Feature Importance) de cada prova criminal.
2. **Decision Tree Classifier:** Para mapear o fluxograma lógico e as "encruzilhadas" que levam a investigação ao sucesso ou ao fracasso.
3. **Bernoulli Naive Bayes:** Para simular cenários hipotéticos ("What-If") e calcular a probabilidade estrita de indiciamento baseada em combinações de provas (ex: "Inteligência Silenciosa").
4. **Regressão Logística:** Para calcular a "Barreira de Energia" (o volume mínimo de atos necessários para ultrapassar 50% de chance de elucidação).
5. **Regressão Linear Múltipla:** Para aferir o "Custo Temporal" (em dias) que cada diligência adiciona ou subtrai do prazo final do inquérito.
6. **Matriz de Correlação (Pearson):** Para descobrir os "Combos Investigativos" mais fortes da polícia judiciária.

## 📊 Principais Descobertas

* **A Queda da Prova Testemunhal:** O modelo *Random Forest* revelou que a Testemunha Ocular possui um peso preditivo residual (apenas 3,3%), inferior à Microbalística e à Preservação de Local.
* **A Barreira de Energia:** A Regressão Logística comprovou que inquéritos precisam romper a barreira de **~13,8 atos investigativos** para que a probabilidade matemática de indiciamento supere o acaso (50%).
* **O Efeito Combo:** A tecnologia não age sozinha. A Árvore de Decisão provou que a Inteligência Analítica mitiga o fracasso, mas o sucesso definitivo exige a materialização da prova em incursões físicas (Prisão Preventiva + Busca e Apreensão).
* **Paridade Tecnológica:** Surpreendentemente, não há diferença significativa na taxa de adoção tecnológica (proporcional) entre as Delegacias Especializadas (DHPP) e as Territoriais/Generalistas. O gargalo do interior não é tecnológico, mas de asfixia operacional braçal.

## 🚀 Como Executar o Código (Reprodutibilidade)

A maneira mais fácil de interagir com os dados é através do **Google Colab**, que não exige instalação de softwares locais.

1. Faça o *download* do arquivo `analise_inqueritos.py` e da base `metadados_inqueritos.csv` deste repositório.
2. Abra o [Google Colab](https://colab.research.google.com/).
3. Crie um "Novo Notebook" e copie/cole o conteúdo do script Python na primeira célula.
4. Execute a célula (`Shift + Enter`).
5. O script abrirá um botão interativo pedindo para você fazer o *upload* do arquivo CSV. Selecione a base de dados baixada.
6. Os modelos serão treinados em tempo real e todos os gráficos serão gerados na sua tela.

## ✒️ Autor e Citação

**Pesquisador:** Licurgo Nunes Neto
**Instituição:** Universidade Federal de Goiás/PPGIDH

Caso utilize este código ou metodologia em sua própria pesquisa, por favor, cite a dissertação original:

> NUNES NETO, Licurgo. *A Qualificação da investigação criminal no Rio Grande do Norte: fatores determinantes para a elucidação de homicídios*. Dissertação de Mestrado. [Universidade Federal de Goiás], [2026].
