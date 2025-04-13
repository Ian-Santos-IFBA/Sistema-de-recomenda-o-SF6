# CHAT DE RECOMENDAÇÕES DE PERSONAGEM DE STREET FIGHTER 6	


 O objetivo desse projeto é criar um chat de interação, onde um usuário, possivelmente novato no street fighter 6, ou até em jogos de luta no geral, possa descrever como ele gostaria de jogar, e receber como resposta uma recomendação de personagem para ele jogar, e, adicionalmente, uma lista de motivos do porquê aquele lutador se encaixa para ele.

## Coleta de Dados
Atualmente, street fighter 6 conta com um cast de 25 personagens, com diferentes características, curvas de aprendizado, padrões de ataques, arquétipos, etc… Cada personagem será previamente classificado com suas principais características, dificuldade, etc… 

## Tipo de IA
Aprendizado supervisionado

## Implementação
O sistema foi integrado a uma interface web simples utilizando streamlit

## Exemplos de interação
- "Quero um personagem com poderes de fogo"
- "Quero um personagem que tenha ataques rápidos"
- "Quero um personagem que seja fácil para iniciantes"

## Instalação
### 1. Clone este repositório na sua máquina
```cmd
git clone https://github.com/Ian-Santos-IFBA/Sistema-de-recomenda-o-SF6.git
```
### 2. Crie o ambiente virtual
```cmd
py -m venv venv
```
### 3. Ative o ambiente virtual
```cmd
venv/scripts/activate
```
### 4. Instale as dependências
```cmd
pip install -r requirements.txt
```

## Execução
### 1. Execute o treino do modelo
```cmd
py treino_execucao.py
```
### 2. Execute o programa
```cmd
py streamlit run main.py  
```
