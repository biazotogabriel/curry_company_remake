import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap

df_deliveries = pd.read_csv('datasets/data.csv')
df_deliveries['ordered_datetime'] = pd.to_datetime(df_deliveries['ordered_datetime'])
df_restaurants = pd.read_csv('datasets/restaurants.csv')
df_cities = pd.read_csv('datasets/cities.csv')
data = pd.read_csv('datasets/train.csv')
df = df_deliveries

### FILTERS ###
def apply_filters(data_period_min, data_period_max, include_festival):
    global df_deliveries, df
    df = df_deliveries
    df = df.loc[(df['ordered_datetime'] >= data_period_min) & (df['ordered_datetime'] <= data_period_max), :]
    if include_festival == 'Sim':
        df = df
    else:
        df = df[df['festival'] == 'No']
    
### DELIVERERS ###

def total_deliverers():
    return len(df['id_deliverer'].unique())

def delivery_mean_by_deliverer():
    df_aux = df[['id_delivery','id_deliverer']].groupby('id_deliverer').agg(total=('id_delivery','count')).reset_index()
    return df_aux['total'].mean()

def youngest_deliverer():
    return df['deliverer_age'].min()

def oldest_deliverer():
    return df['deliverer_age'].max()

def deliveries_histogram_by_deliverer():
    df_aux = df[['id_delivery','id_deliverer']].groupby('id_deliverer').agg(total=('id_delivery','count')).reset_index()
    return px.histogram(df_aux, 
             x='total',
             nbins=30).update_layout(yaxis_title="Total de Entregadores", 
                                     xaxis_title='Quantidade de Entregas')

def deliverer_deliveries_distrubution_evolution():
    df_aux = (df.loc[:,['ordered_datetime','id_deliverer','id_delivery']]
                .groupby([pd.Grouper(key='ordered_datetime', freq='W-MON'), 'id_deliverer'])
                .count()
                .reset_index()
                .sort_values('ordered_datetime')
                .groupby(pd.Grouper(key='ordered_datetime', freq='W-MON'))
                .agg(desvio=('id_delivery','std'))
                .reset_index()
                .rename(columns={'ordered_datetime':'semana',
                                 'desvio': 'desvio'}))
    return (px.line(df_aux, 'semana', 'desvio')
              .update_layout(yaxis_title='Desvio',
                             xaxis_title='Semana'))

def top_rated_deliverers():
    #Top 10 entregadores mais bem avaliados
    return (df.loc[:,['id_deliverer', 'deliverer_rating', 'deliverer_name']]
              .groupby('id_deliverer')
              .agg(name=('deliverer_name', 'first'),
                   rating_mean=('deliverer_rating', 'mean'))
              .sort_values('rating_mean', ascending=False)
              .reset_index()
              .head(10)
              .loc[:, ['name', 'rating_mean']]
              .rename(columns={'name': 'Nome', 'rating_mean': 'Rating Médio'})
              .style.hide_index())

def top_fastest_deliverers():
    #Top 10 entregadores mais bem avaliados
    return (df.loc[(df['restaurant_latitude'] != 0) & (df['destination_latitude'] != 0),['id_deliverer', 'delivery_mean_speed', 'deliverer_name']]
            .groupby('id_deliverer')
            .agg(name=('deliverer_name', 'first'),
                 speed_mean=('delivery_mean_speed', 'mean'))
            .sort_values('speed_mean', ascending=True)
            .reset_index()
            .head(10)
            .loc[:, ['name', 'speed_mean']]
            .rename(columns={'name': 'Nome', 'speed_mean': 'Velocidade Média [km/min]'})
            .style.hide_index())

### RESTAURANTS ###

def total_restaurants():
    return len(df['id_restaurant'].unique())


def deliveries_mean_by_restaurant():
    df_aux = (df.loc[df['id_restaurant'] != 31, ['id_delivery','id_restaurant']]
                .groupby('id_restaurant')
                .agg(total=('id_delivery','count'))
                .reset_index())
    return df_aux['total'].mean()
    
def deliveries_histogram_by_restaurants():
    df_aux = (df.loc[df['id_restaurant'] != 31, ['id_delivery','id_restaurant']]
                .groupby('id_restaurant')
                .agg(total=('id_delivery','count'))
                .reset_index())
    return px.histogram(df_aux, 
                        x='total',
                        nbins=30).update_layout(yaxis_title="Total de Restaurantes", 
                                                xaxis_title='Quantidade de Entregas')

def restaurant_deliveries_distrubution_evolution():
    df_aux = (df.loc[:,['ordered_datetime','id_restaurant','id_delivery']]
                .groupby([pd.Grouper(key='ordered_datetime', freq='W-MON'), 'id_restaurant'])
                .count()
                .reset_index()
                .sort_values('ordered_datetime')
                .groupby(pd.Grouper(key='ordered_datetime', freq='W-MON'))
                .agg(desvio=('id_delivery','std'))
                .reset_index()
                .rename(columns={'ordered_datetime':'semana',
                                 'desvio': 'desvio'}))
    return (px.line(df_aux, 'semana', 'desvio')
              .update_layout(yaxis_title='Desvio',
                             xaxis_title='Semana'))

def prepare_time_mean_by_restaurant():
    df_aux = (df.loc[df['id_restaurant'] != 31, ['pickup_time','id_restaurant']]
                .groupby('id_restaurant')
                .agg(mean=('pickup_time','mean'))
                .reset_index())
    return df_aux['mean'].mean()

def prepare_time_histogram_by_restaurants():
    df_aux = (df.loc[df['id_restaurant'] != 31, ['pickup_time','id_restaurant']]
                .groupby('id_restaurant')
                .agg(mean=('pickup_time','mean'))
                .reset_index())
    return px.histogram(df_aux, 
                        x='mean',
                        nbins=30).update_layout(yaxis_title="Total de Restaurantes", 
                                                xaxis_title='Tempo Médio de Preparação')

def deliveries_mean_distance_by_restaurant():
    df_aux = (df.loc[df['id_restaurant'] != 31, ['points_distance','id_restaurant']]
                .groupby('id_restaurant')
                .agg(mean=('points_distance','mean'))
                .reset_index())
    return df_aux['mean'].mean()

def deliveries_distance_histogram_by_restaurant():
    df_aux = (df.loc[df['id_restaurant'] != 31, ['points_distance','id_restaurant']]
                .groupby('id_restaurant')
                .agg(mean=('points_distance','mean'))
                .reset_index())
    return px.histogram(df_aux, 
                        x='mean',
                        nbins=30).update_layout(yaxis_title="Total de Restaurantes", 
                                                xaxis_title='Distância Média da Entrega')

def restaurants_distribution_map():
    mp = folium.Map(location=[df_restaurants['latitude'].mean(), df_restaurants['longitude'].mean()], zoom_start=5)
    def mark_restaurants(restaurant):
        if restaurant['latitude'] != 0:
            folium.Marker(location=[restaurant['latitude'], restaurant['longitude']],
                          icon=folium.Icon(icon="cloud"),
                          popup='<strong>%s</strong><br>%d entregas' % (restaurant['name'], restaurant['total_orders']),
    ).add_to(mp)
    df_restaurants.apply(mark_restaurants, axis=1)

    heatmap = HeatMap(df_restaurants[['latitude','longitude']],#,'total_orders'
                      name='Heatmap',
                      min_opacity=0.2,
                      max_val=380,#df_restaurants['total_orders'].max(),
                      radius=50,
                      blur=50,
                      show=True,
                      control=True,
                      overlay=False,
                      max_zoom=1)
    mp.add_child(heatmap)
    return mp

### COMPANY ###
def total_deliveries_evolution(period = 'diariamente'):
    df_aux = (df.loc[:,['ordered_datetime','id_delivery']]
                .groupby(pd.Grouper(key='ordered_datetime', freq='D' if period == 'diariamente' else 'W-MON'))
                .count()
                .reset_index()
                .rename(columns={'ordered_datetime':'dia' if period == 'diariamente' else 'semana',
                                 'id_delivery':'quantidade'}))
    return (px.bar(df_aux, 'dia' if period == 'diariamente' else 'semana', 'quantidade')
              .update_layout(yaxis_title='Quantidade de Entregas',
                             xaxis_title='Dia' if period == 'diariamente' else 'Semana'))

def mean_rating_evolution(period = 'diariamente'):
    df_aux = (df.loc[:,['ordered_datetime','deliverer_rating']]
                .groupby(pd.Grouper(key='ordered_datetime', freq='D' if period == 'diariamente' else 'W-MON'))
                .mean()
                .reset_index()
                .rename(columns={'ordered_datetime':'dia' if period == 'diariamente' else 'semana',
                                 'deliverer_rating':'media'}))
    return (px.line(df_aux, 'dia' if period == 'diariamente' else 'semana', 'media')
              .update_layout(yaxis_title='Avaliação Média',
                             xaxis_title='Dia' if period == 'diariamente' else 'Semana'))

def mean_speed_evolution(period = 'diariamente'):
    df_aux = (df.loc[:,['ordered_datetime','delivery_mean_speed']]
                .groupby(pd.Grouper(key='ordered_datetime', freq='D' if period == 'diariamente' else 'W-MON'))
                .mean()
                .reset_index()
                .rename(columns={'ordered_datetime':'dia' if period == 'diariamente' else 'semana',
                                 'delivery_mean_speed':'media'}))
    return (px.line(df_aux, 'dia' if period == 'diariamente' else 'semana', 'media')
              .update_layout(yaxis_title='Velocidade Média [km/min]',
                             xaxis_title='Dia' if period == 'diariamente' else 'Semana'))

def overal_indice_evolution(period = 'diariamente'):
    df_aux = df[['ordered_datetime']]
    df_aux['indice'] = df['delivery_mean_speed'] * df['deliverer_rating']
    df_aux = (df_aux.loc[:,['ordered_datetime','indice']]
                    .groupby(pd.Grouper(key='ordered_datetime', freq='D' if period == 'diariamente' else 'W-MON'))
                    .sum()
                    .reset_index()
                    .rename(columns={'ordered_datetime':'dia' if period == 'diariamente' else 'semana',
                                     'indice':'indice'}))
    return (px.line(df_aux, 'dia' if period == 'diariamente' else 'semana', 'indice')
              .update_layout(yaxis_title='Indice',
                             xaxis_title='Dia' if period == 'diariamente' else 'Semana'))

def heatmap_map(info = 'população'):
    if info == 'população':
        info_column = 'population'
    elif info == 'entregas':
        info_column = 'entregas'
    else:
        info_column = 'potencial'
    mp = folium.Map(location=[df_cities['latitude'].mean(), df_cities['longitude'].mean()], zoom_start=5)
    heatmap = HeatMap(df_cities[['latitude','longitude',info_column]],
                      name=info_column,
                      min_opacity=0.2,
                      radius=50,
                      blur=50,
                      gradient={0.25:'blue',
                                0.35:'cyan',
                                0.50:'green',
                                0.65:'yellow',
                                0.75:'red'},

                      max_zoom=1)
    mp.add_child(heatmap)
    return mp