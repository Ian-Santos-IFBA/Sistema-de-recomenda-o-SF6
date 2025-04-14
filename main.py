import streamlit as st
from treinar_modelo import *
import json
from pathlib import Path

def descricao_personagem(name: str):
    with open("docs/personagens_sf6.json", "rb") as arq:
        jsonFile = arq.read()
    
    jsonDict = json.loads(jsonFile)

    for char in jsonDict["chars"]:
        if(char["Nome"]) == name:
            return char["Desc"]
    else:
        return ""
    
def main():

    st.title("Recomendação de personagem de Street Fighter 6")
    st.divider()

    usuario_pergunta = st.text_input("Descreva como você imagina o seu personagem ideal")

    if(usuario_pergunta):

        personagem = resposta(usuario_pergunta)
        
        texto = f"{personagem} é uma boa opção para você"
        desc = descricao_personagem(personagem)

        st.text(personagem)
        st.text(desc)
        st.image(Path(f"imgs/{str(personagem).lower()}.png"))

main()