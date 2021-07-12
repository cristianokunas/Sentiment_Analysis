<h1 align="">Welcome to Sentiment_Analysis 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/Version-1.0-blue" />
  <a href="https://github.com/cristianokunas/Sentiment_Analysis/blob/master/LICENSE" target="_blank">
    <img alt="License: GPL-3.0 License" src="https://img.shields.io/badge/Licence-GPL--3.0-important" />
  </a>
  <a href="https://github.com/cristianokunas/Sentiment_Analysis/blob/master/documents/209665_1.pdf">
    <img alt="Resumo Expandido" src="https://img.shields.io/badge/Resumo%20Espandido-WSCAD--WIC-blueviolet?logo=read-the-docs&logoColor=white"/>
  </a>
</p>

> Rede Neural Artificial para análise de sentimentos em sentenças em língua inglesa. A Rede Neural Artificial Recursiva Long Short-Term Memory foi implementada para o treinamento do modelo de análise. Com a aplicação da RNA desenvolvida sobre uma base de dados pública com 50.000 registros de filmes usando GPU foi possível reduzir o tempo de treinamento das RNAs em até 91,8% e aumentar a acurácia para 87,7%.

## Dataset

Você pode baixá-lo [aqui](https://drive.google.com/file/d/1Ul2Fz6wSZUD1aMyP-M716wfjkyBqBNLF/view?usp=sharing). <br />
Depois de baixar o arquivo, descompacte e coloque-o na pasta **dataset/**.

## Dependências

* Python 3.8
* CUDA Toolkit 10.0.130

```
pip3 install -r documents/requirements.txt
python3 -m nltk.downloader stopwords
```

## Execute
```
python3 main.py
```


## Author

👤 **Cristiano Alex Künas**

* Github: [@cristianokunas](https://github.com/cristianokunas)
* LinkedIn: [@cristianokunas](https://linkedin.com/in/cristianokunas)

## 📝 License

Copyright © 2020 [Cristiano Alex Künas](https://github.com/cristianokunas).
