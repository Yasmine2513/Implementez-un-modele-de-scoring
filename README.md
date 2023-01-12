
# Implémentez un modèle de scoring

Ce projet fait partie du parcours de formation en data science sur Openclassrooms. Il a pour but de développer un système de scoring pour identifier les clients à risque de défaut de remboursement de prêts bancaires. Les missions sont : 
- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
- Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.
- Mettre en production le modèle de scoring de prédiction à l’aide d’une API, ainsi que le dashboard interactif qui appelle l’API pour les prédictions.

## Documentation



Ce fichier utilise ce jeu de [données](https://www.kaggle.com/c/home-credit-default-risk/data) pour entraîner et évaluer le modèle de scoring .
## Installation

Prerequisites
Python 3.7 +


```bash
pip install python

```

Fastapi


```bash
 pip install fastapi
 pip install "uvicorn[standard]"

```
Streamlit

```bash
 pip install streamlit
```

Les librairies  suivantes : numpy, pandas, scikit-learn, LightGBM classifier...
## Usage/Examples

Le fichier principal de ce projet est myfastapi.py. Vous pouvez l'exécuter en utilisant la commande suivante :

```bash
uvicorn myfastapi:app

```



## Deploiment

- L'API a été dployé sur Heroku, vous devez d'abord créer un compte Heroku et installer l'outil de ligne de commande Heroku. 
Ensuite, vous pouvez créer une nouvelle application Heroku en utilisant la commande heroku create dans votre terminal. 
Il est important de noter que le fichier requirements.txt doit être mis à jour avec les dépendances nécessaires pour votre application. Vous pouvez ensuite utiliser la commande git push heroku master pour déployer votre application sur Heroku. Il est recommandé de vérifier les logs de l'application pour détecter d'éventuelles erreurs lors du déploiement. Vous pouvez utiliser la commande heroku logs pour accéder aux logs de l'application.

- Le dashboard a été deployé sur Streamlit Share
Vous devez d'abord créer un compte Streamlit et installer l'outil de ligne de commande Streamlit. Ensuite, vous pouvez utiliser la commande streamlit login pour vous connecter à votre compte Streamlit.

Une fois connecté, vous pouvez utiliser la commande streamlit run nom_de_votre_fichier.py pour lancer votre application en local. Si tout fonctionne comme prévu, vous pouvez utiliser la commande streamlit share pour déployer votre application sur Streamlit Share. Il vous sera demandé de choisir un nom d'utilisateur pour votre déploiement et de créer un lien partageable pour votre application.

Il est important de noter que toutes les dépendances doivent être correctement installées sur votre ordinateur local et doivent être spécifiées dans le fichier requirements.txt pour que le déploiement sur Streamlit Share fonctionne correctement. Il est également recommandé de vérifier les logs de l'application pour détecter d'éventuelles erreurs lors du déploiement.
## Auteur

- [@Yasmine U.](https://www.github.com/Yasmine2513)


## Contributing

Contributions are always welcome!




## Feedback

If you have any feedback, please reach out to me at :


## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yasmine-u-0033a9166/)



