import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

@st.cache_resource
def carregar_modelo_e_codificadores():
    tabela = pd.read_csv("clientes.csv")
    codificador_profissao = LabelEncoder()
    codificador_credito = LabelEncoder()
    codificador_pagamento = LabelEncoder()
    tabela["profissao"] = codificador_profissao.fit_transform(tabela["profissao"])
    tabela["mix_credito"] = codificador_credito.fit_transform(tabela["mix_credito"])
    tabela["comportamento_pagamento"] = codificador_pagamento.fit_transform(tabela["comportamento_pagamento"])
    y = tabela["score_credito"]
    x = tabela.drop(columns=["score_credito", "id_cliente"])
    modelo_arvoredecisao = RandomForestClassifier()
    modelo_arvoredecisao.fit(x, y)
    return x, modelo_arvoredecisao, codificador_profissao, codificador_credito, codificador_pagamento

x, modelo_arvoredecisao, codificador_profissao, codificador_credito, codificador_pagamento = carregar_modelo_e_codificadores()

st.title("Previsão de Score de Crédito do Cliente do Banco y")
st.subheader("Informe os dados do cliente:")
idade = st.number_input("Idade:")
salario_anual = st.number_input("Salário Anual:")
profissoes_texto = codificador_profissao.classes_
mix_credito_texto = codificador_credito.classes_
comportamento_pagamento_texto = codificador_pagamento.classes_

profissao_escolhida = st.selectbox("Profissão:", profissoes_texto)
mix_credito_escolhido = st.selectbox("Mix de Crédito:", mix_credito_texto)
comportamento_pagamento_escolhido = st.selectbox("Comportamento de Pagamento:", comportamento_pagamento_texto)

if st.button("Prever Score de Crédito"):
    colunas = x.columns
    novo_cliente = {col: 0 for col in colunas}
    novo_cliente["idade"] = idade
    novo_cliente["salario_anual"] = salario_anual  # Corrigido aqui!
    novo_cliente["profissao"] = codificador_profissao.transform([profissao_escolhida])[0]
    novo_cliente["mix_credito"] = codificador_credito.transform([mix_credito_escolhido])[0]
    novo_cliente["comportamento_pagamento"] = codificador_pagamento.transform([comportamento_pagamento_escolhido])[0]

    novo_cliente_df = pd.DataFrame([novo_cliente])
    score_credito = modelo_arvoredecisao.predict(novo_cliente_df)
    st.write(f"O score de crédito previsto para o cliente é: {score_credito[0]}")