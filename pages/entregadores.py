import streamlit as st
import data

st.set_page_config(page_title='Entregadores',
                   page_icon='✊',
                   layout='wide')


# =============================
# Layout
# =============================
with st.container():
    st.markdown('# <center>Entregadores</center>', unsafe_allow_html=True)
    st.markdown('#')

with st.container():
    col1, col2, col3, col4 = st.columns(4, gap='large')
    with col1:
        col1.metric('Total', data.total_deliverers())
    with col2:
        col2.metric('Média de Entregas', '%.2f' % data.delivery_mean_by_deliverer())
    with col3:
        col3.metric('Maior Idade', data.oldest_deliverer())
    with col4:
        col4.metric('Menor Idade', data.youngest_deliverer())
    st.markdown("""---""")
with st.container():
    st.markdown("### Distribuição de Entregadores pela Quantidade de Entregas")
    st.plotly_chart(data.deliveries_histogram_by_deliverer(), use_container_width=True)
    st.markdown("""---""")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        col1.markdown("### Top 10 Entregadores mais bem Avaliados")
        col1.dataframe(data.top_rated_deliverers(), use_container_width=True)
    with col2:
        col2.markdown("### Top 10 Entregadores mais Rápidos")
        col2.dataframe(data.top_fastest_deliverers(), use_container_width=True)