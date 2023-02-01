import streamlit as st
import data
from streamlit_folium import folium_static

st.set_page_config(page_title='Empresa',
                   page_icon='✊',
                   layout='wide')

# =============================
# Layout
# =============================

period_min_value = data.df_deliveries['ordered_datetime'].min().to_pydatetime()
period_max_value = data.df_deliveries['ordered_datetime'].max().to_pydatetime()
period_data = st.sidebar.slider(
    'Período',
    value=(period_min_value,
           period_max_value),
    min_value=period_min_value,
    max_value=period_max_value,
    format='DD-MM-YYYY'
)
st.sidebar.markdown("""---""")
festival_data = st.sidebar.radio('Incluir dados de festivais?',
                                 ('Sim', 'Não'))
data.apply_filters(*period_data, festival_data)

with st.container():
    st.markdown('# <center>Empresa</center>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(['Gerencial', 'Geográfica'])
with tab1:
    period = st.selectbox(
    'Selecione o periodo de análise',
    ('diariamente', 'semanalmente'))
    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.markdown("#### Quantidade de Entregas")
        st.plotly_chart(data.total_deliveries_evolution(period), use_container_width=True)
        st.markdown("""---""")
        st.markdown("#### Velocidade Média")
        st.plotly_chart(data.mean_speed_evolution(period), use_container_width=True)
        st.markdown("""---""")
    with col2:
        st.markdown("#### Avaliação Média")
        st.plotly_chart(data.mean_rating_evolution(period), use_container_width=True)
        st.markdown("""---""")
        st.markdown("#### Índice Geral")
        st.plotly_chart(data.overal_indice_evolution(period), use_container_width=True)
        st.markdown("""---""")
with tab2:
    st.markdown('#### Mancha de Cores')
    option = st.selectbox(
    'Qual informação você deseja analisar?',
    ('população', 'entregas', 'potencial de expanção'))
    folium_static(data.heatmap_map(option))
    
    
make_map_responsive= """
 <style>
 [title~="st.iframe"] { width: 100%}
 </style>
"""
st.markdown(make_map_responsive, unsafe_allow_html=True)