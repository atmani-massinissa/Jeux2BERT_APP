import streamlit as st
from interface import Jeux2BERT
from description import description
import pandas as pd


model_path = "Massinissa/Jeux2BERT"
#model_path = "../Flaubert/37"
entities_path = "entities_sm.txt"
app = Jeux2BERT(model_path=model_path)

PAGES = [
        'Classification de triplets',
        'Prédiction de relations',
        'Classement de triplets',
        "Déscription des types de relations sémantiques"
    ]


def run_UI():
    st.set_page_config(
        page_title="'Jeux2BERT",
        page_icon="🏠",
        initial_sidebar_state="expanded",
        menu_items={
            'Report a bug': "https://github.com/arup-group/social-data/issues/new/choose",
            'About': """            
         If you're seeing this, we would love your contribution! If you find bugs, please reach out or create an issue on our 
         [GitHub](https://github.com/arup-group/social-data) repository. If you find that this interface doesn't do what you need it to, you can create an feature request 
         at our repository or better yet, contribute a pull request of your own. You can reach out to the team on LinkedIn or 
         Twitter if you have questions or feedback.
    
        More documentation and contribution details are at our [GitHub Repository](https://github.com/arup-group/social-data).
        
         This app is the result of hard work by our team:
        - [Jared Stock 🐦](https://twitter.com/jaredstock) 
        - [Angela Wilson 🐦](https://twitter.com/AngelaWilson925) (alum)
        - Sam Lustado
        - Lingyi Chen
        - Kevin McGee (alum)
        - Jen Combs
        - Zoe Temco
        - Prashuk Jain (alum)
        - Sanket Shah (alum)
        Special thanks to Julieta Moradei and Kamini Ayer from New Story, Kristin Maun from the city of Tulsa, 
        Emily Walport, Irene Gleeson, and Elizabeth Joyce with Arup's Community Engagment team, and everyone else who has given feedback 
        and helped support this work. Also thanks to the team at Streamlit for their support of this work.
        The analysis and underlying data are provided as-is as an open source project under an [MIT license](https://github.com/arup-group/social-data/blob/master/LICENSE). 
        Made by [Arup](https://www.arup.com/).
        """
        }
    )


    if st.session_state.page:
        page=st.sidebar.radio('Navigation', PAGES, index=st.session_state.page)
    else:
        page=st.sidebar.radio('Navigation', PAGES, index=1)

    st.experimental_set_query_params(page=page)
    

    if(page == 'Déscription des types de relations sémantiques'):
        st.header("Déscription des types de relations sémantiques")
        st.table(pd.DataFrame(
            description,
            columns=(["Type de la relation", "Déscription de la relation"])))

    elif(page == 'Classification de triplets'):
        st.header('Classification de triplets')

        st.sidebar.write("""
            ## À propos
            La tâche consiste à prédire la validité du triplé en question exprimant une relation sémantique entre entre deux entités textuelles. 
        """)
        
        with st.form("lp form"):
            st_s, st_r, st_o = st.columns(3)
            s = st_s.text_input('Entité Sujet', 'petit chaton')
            r = st_r.selectbox('Type de Relation', app.rp_label_list)
            o = st_o.text_input('Entité Objet', 'félin')

            submitted = st.form_submit_button("Ask")
            if submitted:
                with st.spinner("Veuillez attendre..."):
                    label, score  = app.lp(s, r, o)
                    st.write(app.take_decision("lp", label, score))
                    #st.write(label)
                    #st.write(score)


    if(page == 'Prédiction de relations'):
        st.header('Prédiction de relations')

        st.sidebar.write("""
            ## À propos
            La tâche consiste à prédire les types de relations sémantiques qui pourraient lier deux entités textuelles. 
        """)

        with st.form("rp form"):
            st_s, st_o = st.columns(2)
            s = st_s.text_input('Entité Sujet', 'pirate')
            o = st_o.text_input('Entité Objet', 'piller')

            submitted = st.form_submit_button("Ask")
            if submitted:
                with st.spinner("Veuillez attendre..."):
                    labels, scores  = app.rp(s, o)
                    st.write(app.take_decision("rp", labels, scores))
                    #st.write(labels)
                    #st.write(scores)
            
            
    if(page == 'Classement de triplets'):
        st.header('Classement de triplets')
        st.sidebar.write("""
            ## À propos
            La tâche consiste à prédire à partir d'un dictionnaire de mots l'entité manquante 
            (?, r, t) avec la configuration head ou (h, r, ?) avec la configuration tail.
            
            Un dictionnaire de mots spécifique peut être saisi en etrée.  
        """)

        with st.form("ranking form"):
            st_s, st_r, st_o, st_m = st.columns(4)
            st_f, st_e = st.columns(2)
            s = st_s.text_input('Entité Sujet', 'petit chaton')
            r = st_r.selectbox('Type de Relation', app.rp_label_list)
            o = st_o.text_input('Entité Sujet', 'félin')
            mode = st_m.selectbox('Mode', ("head", "tail"))
            f = st_f.file_uploader("Choisissez un fichier")
            encodage = st_e.selectbox('Encodage fichier', ("latin-1", "utf-8"))
            entities = []
            submitted = st.form_submit_button("Ask")
            if submitted:
                with st.spinner("Veuilez attendre..."):
                    if f is not None:
                        #print(f.getvalue().decode("latin-1"))
                        app.entities = app.set_entities(f.getvalue().decode(encodage).strip().split("\n"))
                        entities = app.entities
                    elif app.origin_entities is not None:
                        entities = app.origin_entities
                    else:
                        app.origin_entities = app.set_origin_entities(entities_path)
                        entities = app.origin_entities
                    
                    entities, scores  = app.triple_ranking(s, r, o, mode, entities)
                    st.write(entities)
                    #st.write(scores)


if __name__ == '__main__':

    if st._is_running_with_streamlit:
        url_params = st.experimental_get_query_params()
        if 'loaded' not in st.session_state:
            if len(url_params.keys()) == 0:
                st.experimental_set_query_params(page='Classification de triplets')
                url_params = st.experimental_get_query_params()

            st.session_state.page = PAGES.index(url_params['page'][0])

        run_UI()


