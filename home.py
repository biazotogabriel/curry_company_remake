import streamlit as st
from PIL import Image

st.set_page_config(page_title='Home',
                   page_icon='✊',
                   layout='wide')

image = Image.open('logo.png')
st.sidebar.image(image, width=120)
st.sidebar.markdown('# Cury Company')
st.sidebar.markdown('## Fastest Delivery in Town')

st.markdown(
    """
    ### Este Dashboard foi construído para acompanhar parâmetros estatísticos de negócio 
    - Empresa:
        - Gerencial: Indices de varianção diário e semanal
        - Geográfica: Manchas de cores da população, quantidade de entregas e potencial de crescimento
    - Entregador:
        - Indicadores gerais da base de entregadores parceiros
    - Restaurantes
        - Indicadores gerais da base de restaurantes parceiros
""")