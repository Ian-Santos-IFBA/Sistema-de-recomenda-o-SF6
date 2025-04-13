import json
from pathlib import Path

class Treino:
    
    treinos = []

    def __init__(self):
        self.file = Path("personagens_sf6.json")
        with open(self.file, "r", encoding="utf-8") as f:
            self.dados = json.load(f)

    def gerar_treinos_para_personagens_gameplay(self, base_ou_dlc:str):
        """
        base: Personagens-Base
        ano 1: ANO1
        ano 2: ANO2
        """
        dataset = []
        for p in self.dados[base_ou_dlc]:
            prompt = f"Quero um personagem do arquétipo {p['Arquétipo'].lower()} com estilo {p['EstiloDeJogo'].lower()} e dificuldade {p['Dificuldade'].lower()}."
            response = f"{p['Nome']}"
            dataset.append({"input": prompt, "label": response})

        for item in dataset:
            self.treinos.append(item)

    def gerar_treinos_para_personagens_aparência(self, base_ou_dlc:str):
        """
        base: Personagens-Base
        ano 1: ANO1
        ano 2: ANO2
        """
        dataset = []
        for p in self.dados[base_ou_dlc]: 
            aparencia = p["Aparência"]
            prompt = f"Quero um personagem do gênero/sexo {p['Gênero'].lower()} com o cabelo {aparencia['Cabelo'].lower()} com o rosto {aparencia['Rosto'].lower()}, com o corpo {aparencia['Corpo'].lower()}, com a altura {aparencia['Altura'].lower()}, com as roupas {aparencia['Traje'].lower()} e também {aparencia['Outros'].lower()}."
            response = f"{p['Nome']}"
            dataset.append({"input": prompt, "label": response})

        for item in dataset:
            self.treinos.append(item)


    def salvar_treino(self):
        # Salvar para arquivo de treino
        with open("treino.jsonl", "w", encoding="utf-8") as f:
            for item in self.treinos:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

treinos = Treino()

tipos = ["Personagens-Base", "ANO1", "ANO2"]
for tipo in tipos:
    treinos.gerar_treinos_para_personagens_gameplay(tipo)
    treinos.gerar_treinos_para_personagens_aparência(tipo)

treinos.salvar_treino()