import numpy as np
import pickle
import streamlit as st
st.title("Sentiment Analysis of Students' Mental Health ")   #Interface Utilisateur avec Streamlit

sentiment=st.text_area("How do you feel",height=150)
#st.write("vous avez saisie",sentiment)
encoder=pickle.load(open('encoder.pickle','rb'))
model=pickle.load(open('LoanModel.pickle','rb'))

#Transformation des Variables Catégorielles
sentiment=encoder.fit_transform([sentiment])


#Préparation des Données pour la Prédiction
#X=[[gender,married,dep,self_emp,edu, float(income),float(co_income),float(loan_amt),int(loan_amt_term),cred_hist,prop]]
X=[[sentiment]]
v=np.array(X,dtype=object)  #Création d'un vecteur d'entrée : Les valeurs transformées sont regroupées dans une liste X, qui devient un tableau numpy v.
if st.button("Analysis"):
    pred=model.predict(v)
    if pred[0]==0:
        pred='Yes'
    else:
        pred='No'
    st.success("Advices : {}".format(pred))
    








