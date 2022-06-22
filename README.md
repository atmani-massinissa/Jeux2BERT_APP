# Jeux2BERT_APP

Web application for the modèle Jeux2BERT, which is a Flaubert language model augmented by the lexico-semantic network JeuxDeMots.
Thus, this model tries to capture the distributional and relational properties of words,
but also tries to discriminate the different relational properties between words or syntagms.

The Web application includes three Tasks : Link Prediction (Classification de triplets), Relation Prediction (Prédiction de Relation) and Triple Ranking (Classement de triplets).



## JeuxDeMots

GWAPs (Games With A Purpose), or “games with the purpose of acquiring
resources” is one of the contributory approaches to populating a knowledge base.
meeting. The lexico-semantic network JeuxDeMots was built following the GWAP approach. 
It has also been augmented with counterplays to correct and refine the acquired knowledge. 
To date, JeuxDeMots encompasses more than 377 million relationships between more than 5.3 million terms (words, syntagms, nouns and phrases).


# Installation

pip install requirements.txt

# Run

streamlit run app.py

# Demo Link

https://share.streamlit.io/atmani-massinissa/jeux2bert_app/main/app.py?page=Classement+de+triplets

The task Triple Ranking (Classement de triplets) don't run smoothly on the streamlit server because of the time of inference, so it's better to run it locally instead on the demo's server. 
