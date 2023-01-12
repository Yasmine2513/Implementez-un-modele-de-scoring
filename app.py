"""
Created on Dec 2022
@author: Yasmine UNEAU
"""

#Import necessary libraries
import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import numpy as np 
import requests
import urllib.request
import json
import matplotlib.pyplot as plt
import shap
import plotly.express as px
import plotly.graph_objs as go


# Set page Title
st.set_page_config(page_title='Prêt à dépenser : Application Web de prediction de remboursement de prêt par Yasmine UNEAU',  layout='wide', page_icon="🏦")
#st.set_page_config(layout = 'wide')

#Import Data
data = pd.read_csv(open('x_test1.csv'), encoding ='utf-8')
target = pd.read_csv(open('y_test.csv'), encoding ='utf-8')
dataset = pd.read_csv(open("df.csv"),index_col =0, encoding ='utf-8')
data_mdl = pd.read_csv(open('x_test1.csv'),index_col =0, encoding ='utf-8')
description_df = pd.read_csv(open('HomeCredit_columns_description.csv'),index_col ='Row', encoding ='utf-8')
description_df = description_df['Description']


#Loading the trained model
pickle_in = open("C:/Users/Yasmine/Desktop/FORMATION OCR DATA/P7/LightGBMmodel.pkl","rb")
classifier=pickle.load(pickle_in)

#Create a function to get predict_probability
@st.cache
def load_prediction(data_mdl, id, classifier):
    score = classifier.predict_proba(data_mdl[data_mdl.index == int(id)])[:,1]
    return score


st.warning('👈 Veuillez Saisir un ID client !')

#Define logo
logo = Image.open('logo.jpg')
logo = logo.resize((200,200))
st.title("Évaluation du risque de défaut de remboursement d’un crédit")
st.subheader("L'ambition du scoring 💪!")
st.sidebar.image(logo, use_column_width=True)

#######################################
    # SIDEBAR
#######################################

st.sidebar.subheader("About App")
st.sidebar.info("Cette application Web vous aide à prédire le risque de défaut de paiement d'un prêt bancaire contracté par un client, et donc vous aide dans la prise de décision. Le model utilisé est LightGBM. C'est un outil utilisé en apprentissage automatique permettant de créer des modèles prédictifs à partir de données en utilisant des algorithmes d'arbres de décision et en les combinant de manière à améliorer les prédictions.")
st.sidebar.title ("👤 Données client")
st.sidebar.write("Veuillez selectionner un **id client** et appuyer sur le **bouton predict** pour afficher la prediction.", align="center")


#Define client function
def client():
    
    ## Identite
    id_client = data['SK_ID_CURR'].values
    id = st.sidebar.selectbox("Client ID", id_client)
    client_data = dataset[dataset['SK_ID_CURR'] == id]
    other_data =  dataset[dataset['SK_ID_CURR'] != id]

    ### Display client information in the sidebar ###
    # View selected customer information
    if id:
        st.header("Informations du client")
        st.write("Vous trouverez dans ce tableau toutes les informations du client enregistrées sur la base de données.")
        # Get selected custumer data
        st.write(client_data)
        # Calculate the average Income of other clients
        other_mean = other_data['AMT_INCOME_TOTAL'].mean()
        
        # Display the income value for the selected client
        client_income = client_data['AMT_INCOME_TOTAL'].values[0]
        st.write("Ce barplot montre le revenu moyen du client par rapport à la moyenne des revenus des autres clients.")
        #Create a bar chart using plotly express
        df = pd.DataFrame({
        'Données_client': ['Moyenne Autres clients', 'Client sélectionné'],
        'Income': [other_mean, client_income]
        })
        fig = px.bar(df, x='Données_client', y='Income')
        #Show bar plot 
        st.plotly_chart(fig)
        
        st.write("Sur ce scatterplot, bougez la souris pour localiser le client séléctionné. Cela vous permettra d'avoir son âge et le mantant de ses crédits.")
        #Create scatterplot to show client data and locate selected client
        fig = px.scatter(other_data, x='AMT_CREDIT', y='AGE')
        fig.add_trace(go.Scatter(x=client_data['AMT_CREDIT'], y=client_data['AGE'],mode="markers", name="Client séléctionné"))
        st.plotly_chart(fig)
        
        st.write("Vous pouvez également le visualizer directement ici.")
        #Scatter plot
        fig, ax = plt.subplots(figsize=(7, 4))
        plt.title("Scatterplot du montant des crédits contractés en fonction de l'âge", fontsize = 8)
        ax.scatter(other_data['AMT_CREDIT'],other_data["AGE"], label='Autres clients')
        ax.scatter(client_data['AMT_CREDIT'], client_data["AGE"], label = 'Client séléctionné')
        ax.set_xlabel("Montant des crédits contractés en M$")
        ax.set_ylabel("Age en années")
        ax.grid()
        plt.legend()
        st.pyplot(fig)

        
    #Predict : when the user clicks on button it will fetch the fastapi converting input to a jason format
    # Define API URL of FastAPI
    api_url = "http://127.0.0.1:8000/prediction"
    

    if st.sidebar.button("Predict"):
            
           st.subheader("**Analyse des données client**")
           # Send a GET request to the FastAPI endpoint with the customer id as a parameter
           response = requests.get(api_url, params= {"sk_id_cust":id}) 
           
           # get prediction 
           st.success(f" La prediction depuis le model est : {response.text}")

           #Customer solvability display
           
           prediction_score = load_prediction(data_mdl, id, classifier)
           st.write("**Score de prédiction du risque de paiement par le client :** {:.0f} %".format(round(float(prediction_score)*100, 2)))
           st.write("Pour comprendre ce résultat, les valeurs SHAP donnent des détails sur l'importance des variables ou données du client.")
           
           #Get shap features importance plot
           shap.initjs()
           data_cust = data_mdl[data_mdl.index == int(id)]

           # Calculate the SHAP values for the data
           
           explainer = shap.TreeExplainer(classifier)
           shap_values = explainer.shap_values(data_cust)
           
           # Create a summaryplot of the SHAP values
           fig, ax = plt.subplots(figsize=(10, 10))
           plt.title("Évaluation de l'importance des variables en fonction des valeurs Shap")
           shap.summary_plot(shap_values[1],data_cust)
           st.pyplot(fig)
           plt.clf()
           
           st.write('Ci-dessus les variables qui contribuent le plus à cette prédiction. Les variables en rouge augmentent le risque, en bleu le diminuent.')
           st.write("Dans le graphique ci-dessus, les valeurs des variables ne sont pas affichées. Les valeurs SHAP sont représentées par la longueur de la barre spécifique. Cependant, il n'est pas tout à fait clair. Vous pouvez verifier le graphique ci-dessous pour avoir plus de détails.")
           
           # Create a dataframe with the SHAP values
           shap_table=pd.DataFrame(shap_values[1],columns=data_cust.columns,index=['SHAP Value'] )
           # Create a bar plot of the SHAP values
           fig = px.bar(shap_table, x=shap_table.columns, y=shap_table.index, orientation='h')

           # Display the plot in the Streamlit app
           st.success(f"Voici le détail de l'importance de chaque variable pour ce client :")
           st.plotly_chart(fig)

    # Create checkbox for features description
    if st.checkbox("Avez vous bedoin de mieux comprendre la description de chaque variable ?") :
        list_features = description_df.index.to_list()
        feature = st.selectbox('Liste des variable', list_features)
        st.table(description_df.loc[description_df.index == feature])
        
  
client()

#Infos & feedback note
st.sidebar.info("N'oubliez pas de noter cette app !")
feedback = st.sidebar.slider('Combien évalueriez-vous cette application ?',min_value=0,max_value=5,step=1)

if feedback:
  st.header("Merci d'avoir évalué l'application 😊 !")


st.markdown("***")
# This is app is created by Yasmine UNEAU (PhD, Data Scientist Student) https://www.linkedin.com/in/yasmine-u-0033a9166/
"""
Créé en décembre 2022, 
@author : Yasmine UNEAU
"""
st.markdown( "Merci d'avoir consulté cette application. J'espère que cette mini analyse vous aidera ! J'aimerai avoir un feedback à ce sujet, donc si vous voulez me contacter, vous pouvez me trouver sur [Linkedin](https://www.linkedin.com/in/yasmine-u-0033a9166/).")
