import streamlit as st
import data
from streamlit_folium import folium_static

st.set_page_config(page_title='Restaurantes',
                   page_icon='✊',
                   layout='wide')

# =============================
# Layout
# =============================
with st.container():
    st.markdown('# <center>Restaurantes</center>', unsafe_allow_html=True)
with st.container():  
    #st.metric('Total de Restaurantes', data.total_restaurants())
    st.markdown('#### Total')
    st.markdown('# %d' % data.total_restaurants())
    st.markdown("""---""")
with st.container():  
    st.markdown('#### Entregas')
    st.metric('Média por Restaurante', '%.1f' % data.deliveries_mean_by_restaurant())
    st.markdown('###### Distribuição da quantidade de entregas')
    st.plotly_chart(data.deliveries_histogram_by_restaurants(), use_container_width=True)
    st.markdown('###### Evolução do Índice de Distribuição das Entregas entre Restaurantes')
    st.plotly_chart(data.restaurant_deliveries_distrubution_evolution(), use_container_width=True)
    st.markdown("""---""")
col1, col2 = st.columns(2, gap='large')
with col1:
    with st.container():
        st.markdown('#### Tempo de Preparo')
        st.metric('Média por Restaurante', '%.1f' % data.prepare_time_mean_by_restaurant())
        st.markdown('###### Distribuição')
        st.plotly_chart(data.prepare_time_histogram_by_restaurants(), use_container_width=True)
with col2:
    with st.container():
        st.markdown('#### Distâncias das Entregas')
        st.metric('Média por Restaurante', '%.1f' % data.deliveries_mean_distance_by_restaurant())
        st.markdown('###### Distribuição')
        st.plotly_chart(data.deliveries_distance_histogram_by_restaurant(), use_container_width=True)
st.markdown("""---""")
with st.container():
    st.markdown('#### Mapa de Distribuição dos Restaurantes')
    folium_static(data.restaurants_distribution_map())

make_map_responsive= """
 <style>
 [title~="st.iframe"] { width: 100%}
 </style>
"""
st.markdown(make_map_responsive, unsafe_allow_html=True)