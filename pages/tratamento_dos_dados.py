import streamlit as st
import data



st.markdown('Abaixo segue os dados e todo o tratamento realizado com base original')

col1, col2, col3, col4 = st.columns(4)

col1.download_button(label="Entregas", data=data.df.to_csv().encode('utf-8'), file_name='deliveries.csv', mime='text/csv')
col2.download_button(label="Restaurantes", data=data.df_restaurants.to_csv().encode('utf-8'), file_name='restaurants.csv', mime='text/csv')
col3.download_button(label="Cidades", data=data.df_cities.to_csv().encode('utf-8'), file_name='cities.csv', mime='text/csv')
col4.download_button(label="Original", data=data.data.to_csv().encode('utf-8'), file_name='original.csv', mime='text/csv')

st.markdown('# Installs')
body = """
!pip install haversine
!pip install names
!pip install Nominatim
!pip install requests
#!pip install streamlit-folium
"""
st.code(body, language="python")
st.markdown("""---""")

st.markdown('# Imports')
body = """
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from folium.plugins import HeatMap
#from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from haversine import haversine
import names
import requests
import sklearn.cluster as cluster
"""
st.code(body, language="python")
st.markdown("""---""")

st.markdown('# Carregamento, Tratamento e Limpeza')
st.markdown('#### Carregamento')
body = """
data = pd.read_csv('drive/MyDrive/Colab Notebooks/train.csv')
data.info()
"""
st.code(body, language="python")
body = """
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 45593 entries, 0 to 45592
Data columns (total 20 columns):
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   ID                           45593 non-null  object 
 1   Delivery_person_ID           45593 non-null  object 
 2   Delivery_person_Age          45593 non-null  object 
 3   Delivery_person_Ratings      45593 non-null  object 
 4   Restaurant_latitude          45593 non-null  float64
 5   Restaurant_longitude         45593 non-null  float64
 6   Delivery_location_latitude   45593 non-null  float64
 7   Delivery_location_longitude  45593 non-null  float64
 8   Order_Date                   45593 non-null  object 
 9   Time_Orderd                  45593 non-null  object 
 10  Time_Order_picked            45593 non-null  object 
 11  Weatherconditions            45593 non-null  object 
 12  Road_traffic_density         45593 non-null  object 
 13  Vehicle_condition            45593 non-null  int64  
 14  Type_of_order                45593 non-null  object 
 15  Type_of_vehicle              45593 non-null  object 
 16  multiple_deliveries          45593 non-null  object 
 17  Festival                     45593 non-null  object 
 18  City                         45593 non-null  object 
 19  Time_taken(min)              45593 non-null  object 
dtypes: float64(4), int64(1), object(15)
memory usage: 7.0+ MB
"""
st.text(body)

st.markdown('#### Remoção dos espaços externos vazios de todas as celulas')
body = """
for column in data.columns:
  if data[column].dtype == 'object':
    data[column] = data[column].str.strip()
"""
st.code(body, language="python")

st.markdown('#### Remoção de todas as linhas com valores NaN')
body = """
data = data.replace('NaN', np.NaN)
data = data.dropna(axis=0, how='any')
data.shape
"""
st.code(body, language="python")
st.text('(41368, 20)')

st.markdown('#### Vizualizando os dados')
body = """
data.columns
"""
st.code(body, language="python")
body = """
Index(['ID', 'Delivery_person_ID', 'Delivery_person_Age',
       'Delivery_person_Ratings', 'Restaurant_latitude',
       'Restaurant_longitude', 'Delivery_location_latitude',
       'Delivery_location_longitude', 'Order_Date', 'Time_Orderd',
       'Time_Order_picked', 'Weatherconditions', 'Road_traffic_density',
       'Vehicle_condition', 'Type_of_order', 'Type_of_vehicle',
       'multiple_deliveries', 'Festival', 'City', 'Time_taken(min)'],
      dtype='object')
"""
st.text(body)

st.markdown('#### Convertendo os tipos de dados')
body = """
df_deliveries = pd.DataFrame()
df_deliveries['id_delivery'] = data['ID'].astype('object')
df_deliveries['id_deliverer'] = data['Delivery_person_ID'].astype('object')
df_deliveries['deliverer_age'] = data['Delivery_person_Age'].astype('int')
df_deliveries['deliverer_rating'] = data['Delivery_person_Ratings'].astype('float')
df_deliveries['restaurant_latitude'] = data['Restaurant_latitude'].astype('float')
df_deliveries['restaurant_longitude'] = data['Restaurant_longitude'].astype('float')
df_deliveries['destination_latitude'] = data['Delivery_location_latitude'].astype('float')
df_deliveries['destination_longitude'] = data['Delivery_location_longitude'].astype('float')
df_deliveries['weather_condition'] = data['Weatherconditions'].astype('object')
df_deliveries['traffic_density'] = data['Road_traffic_density'].astype('object')
df_deliveries['vehicle_condition'] = data['Vehicle_condition'].astype('int')
df_deliveries['vehicle_type'] = data['Type_of_vehicle'].astype('object')
df_deliveries['order_type'] = data['Type_of_order'].astype('object')
df_deliveries['multiple_deliveries'] = data['multiple_deliveries'].astype('int')
df_deliveries['festival'] = data['Festival'].astype('object')
"""
st.code(body, language="python")

st.markdown('#### Tratando colunas de tempo')
body = """
df_deliveries['Order_Date'] = pd.to_datetime(data['Order_Date'], format='%d-%m-%Y')
df_deliveries['Time_Orderd'] = pd.to_datetime(df_deliveries['Order_Date'].dt.strftime("%d/%m/%Y")+' '+data['Time_Orderd'], format='%d/%m/%Y %H:%M:%S')
df_deliveries['Time_Order_picked'] = pd.to_datetime(df_deliveries['Order_Date'].dt.strftime("%d/%m/%Y")+' '+data['Time_Order_picked'], format='%d/%m/%Y %H:%M:%S')
#Ajustando os Time_Order_picked para datas pegas no dia anterior da entrega
def fix_time_orderd_day(order):
  if (order['Time_Order_picked'] - order['Time_Orderd']).total_seconds() < 0:
    return order['Time_Order_picked'] + pd.Timedelta(days=1)
  return order['Time_Order_picked']
df_deliveries['Time_Order_picked'] = df_deliveries.apply(fix_time_orderd_day, axis=1)
#Removendo o texto da coluna time taken
def clean_time_taken(row):
  return re.search(r'\d+', row['Time_taken(min)']).group(0)
df_deliveries['Time_taken(min)'] = data.apply(clean_time_taken, axis=1).astype('int')
"""
st.code(body, language="python")

st.markdown('##### Descobrindo se "Time_taken(min)" é o mesmo que "Time_Order_picked" - "Time_Orderd"')
body = """
df_aux = df_deliveries[['Time_Order_picked','Time_Orderd','Time_taken(min)']].copy()
df_aux['time_for_pickup']  = (df_aux['Time_Order_picked'] - df_aux['Time_Orderd']).dt.total_seconds()/60
len(df_aux.loc[(df_aux['Time_taken(min)']-df_aux['time_for_pickup']) == 0,:])
"""
st.code(body, language="python")
body = """715"""
st.text(body)

body = """
Apenas 715 de 41368 registros possuem os tempos acima iguais. Logo muito provavelmente os valores se referem a coisas diferentes
"""
st.markdown(body)

st.markdown('##### Descobrindo se "Time_taken(min)" é o tempo da hora da ordem até a entrega do cliente ou da hora de recolha até a entrega do cliente')
body = """
len(df_aux.loc[(df_aux['Time_taken(min)']-df_aux['time_for_pickup']) < 0,:])
"""
st.code(body, language="python")
body = """1054"""
st.text(body)

body = """
1054 de 41368 registros possuem os tempos taken menores que os de pickup. 
Logo muito provavelmente os valores se referem ao tempo levado da recolha na loja até o cliente. 
O tempo de entrega ao cliente ser maior que o tempo de recolha faz total Sentido uma vez que muitos apps de entrega atualmente desegnam áreas para que os entregadores se posicionem onde existem maior número de expedição de Pedidos. Reduzindo assim o tempo levado para fazer o recolimento do mesmo é possivel cruzar estes dados com a satisfação do cliente para verificar se existe a correlação e qual delas é mais clara.
"""
st.markdown(body)
body = """
df_aux['time_total'] = df_aux['time_for_pickup'] + df_aux['Time_taken(min)']
df_aux['rating'] = df_deliveries['deliverer_rating']
df_aux
daa = (df_aux.loc[:,['time_total','rating']]
             .groupby('time_total')
             .mean()
             .reset_index())
print(plt.scatter(daa['time_total'], daa['rating']))
del(daa)
del(df_aux)
"""
st.code(body, language="python")
st.image('images/image1.png')
body = """
tanto a "time_total" quanto o "Time_taken" tem correlação negativa com o rating com comportamento linear até cerca de 30 minutos e então queda em platôs. Após 50 minutos as avaliações voltam a subir.
Duas coisas precisariam ser analisadas aqui... 
1º - porque a queda ocorrem em platôs? Provavelmente tem alguma outra feature que explique
2º - porque a avaliação volta a subir após 50 minutos?
Ter acesso ao time de negócios da empresa ajudaria para solucionar estes pontos.
Como não existe esta possibilidade vou continuar com as definições quanto ao que se refere cada tempo feita nas linhas anteriores.
"""
st.markdown(body)

body = """
##### Com base na analise superior serão criadas as seguintes colunas:
* ordered_datetime
* pickedup_datetime
* delivered_datetime
* pickup_time
* deliver_time
* total_time
##### Serão tambem dropadas as seguintes colunas:
* Time_Orderd
* Time_Order_picked
* Order_Date
"""
st.markdown(body)

body = """
df_deliveries['ordered_datetime'] = df_deliveries['Time_Orderd']
df_deliveries['pickedup_datetime'] = df_deliveries['Time_Order_picked']
def sum_time_taken(order):
  return order['Time_Order_picked'] + pd.Timedelta(minutes=order['Time_taken(min)'])
df_deliveries['delivered_datetime'] = df_deliveries.apply(sum_time_taken, axis=1)
df_deliveries['pickup_time'] = ((df_deliveries['pickedup_datetime'] - df_deliveries['ordered_datetime']).dt.total_seconds() / 60).astype('int')
df_deliveries['delivery_time'] = ((df_deliveries['delivered_datetime'] - df_deliveries['pickedup_datetime']).dt.total_seconds() / 60).astype('int')
df_deliveries['total_time'] = df_deliveries['pickup_time'] + df_deliveries['delivery_time']
df_deliveries = df_deliveries.drop(columns=['Time_Orderd', 'Time_Order_picked', 'Order_Date','Time_taken(min)'])
"""
st.code(body, language="python")

st.markdown('#### Ajustando as latitudes negativas')
body = """
#Ajustando as latitudes com valores negativos que estavam apontando no oceano pacífico
df_deliveries.loc[df_deliveries['restaurant_latitude'] < 0, 'restaurant_latitude'] = df_deliveries['restaurant_latitude'] * -1
"""
st.code(body, language="python")

st.markdown('#### Ajustando as condições de tempo')
body = """
df_deliveries['weather_condition'] = df_deliveries['weather_condition'].str.findall('(?<=conditions )\w+')
df_deliveries['weather_condition'] = df_deliveries['weather_condition'].str[0]
df_deliveries['weather_condition'].unique()
"""
st.code(body, language="python")

st.markdown('#### Calculando distância média radial entre os pontos')
body = """
def distance_between(order):
  return haversine((order['restaurant_latitude'], order['restaurant_longitude']),
                   (order['destination_latitude'], order['destination_longitude']),
                   'km')
df_deliveries['points_distance'] = df_deliveries.apply(distance_between, axis=1)
df_deliveries.head(5)
"""
st.code(body, language="python")

body = """
#### Criando cadastro fictício de entregadores para a visão entregadores.
Poderia ser pego os dados com o time de dados. Não havendo esta possibilidade...
"""
st.markdown(body)
body = """
df_deliverers = pd.DataFrame(df_deliveries['id_deliverer'].unique(), columns=['id_deliverer'])
def create_delivers_name(row):
  return names.get_first_name()
df_deliverers['name'] = df_deliverers.apply(create_delivers_name, axis=1)
df_deliverers
"""
st.code(body, language="python")

st.markdown('#### Calculando a velocidade média por entrega considerando a distancia radial')
body = """
df_deliveries['delivery_mean_speed'] = df_deliveries['points_distance'] / df_deliveries['total_time']
"""
st.code(body, language="python")

body = """
#### Criando cadastro de restaurantes agrupando com base no raio de distância entre entregas
Em um caso real provavelmente seria possível apanhar o dado em algum banco de dados.
O método abaixo dificilmente seria possível de ser usado em um caso real pois:

se usarmos um raio curto:
 - podemos não agrupar um mesmo restaurante onde a ordem foi pega dentro mas o entregador marcou a entrega como pega em outro local.
 - o gps pode ter interferencia.
se usarmos raios grandes:
 - podemos estar agrupandos restaurantes que ficam muito próximos como casos em shoppings.

Mas a título de gerar um informação para prática e também praticar a lógica vou seguir desta forma.

Uma outra possível forma de definir os restaurantes seria utilizar um ML de clusterização usando para isso não apenas a localização mas tambem outras features que talvez estejam relacionadas com o que define cada restaurante. Mas isso seria muito mais complicado oque demandaria tempo que não pretendo desprender aqui
"""
st.markdown(body)
body = """
distance_limit = 15 #meters
restaurants = []
def create_restaurants(order):
  global restaurants
  lat = order['restaurant_latitude']
  lon = order['restaurant_longitude']
  for restaurant, [mean_lat, mean_lon, qt] in enumerate(restaurants):
    if haversine((lat, lon), (mean_lat, mean_lon), 'm') < distance_limit:
      restaurants[restaurant][0] = (mean_lat * qt + lat) / (qt + 1)
      restaurants[restaurant][1] = (mean_lon * qt + lon) / (qt + 1)
      restaurants[restaurant][2] = qt + 1
      return restaurant
  restaurants.append([lat, lon, 1])
  return len(restaurants) - 1
df_deliveries['id_restaurant']  = df_deliveries.apply(create_restaurants, axis=1)
"""
st.code(body, language="python")
body = """
df_restaurants = pd.DataFrame(restaurants)
df_restaurants.columns = ['latitude', 'longitude', 'total_orders']
df_restaurants['id_restaurant'] = df_restaurants.index
df_restaurants
"""
st.code(body, language="python")

body = """
#### Atribuindo nomes de restaurantes dos estados unidos aleatóriamente ao cadastro de restaurantes
Apenas para titulo de melhor vizualização dos dados. Em um caso real basteria apanhar esta informação no banco de dados. 
O dataset com as informação foi baixado do kaggle e pode ser acessado através do link abaixo:
https://www.kaggle.com/datasets/michaelbryantds/us-restaurants?resource=download&select=chainness_point_2021_part1.csv
"""
st.markdown(body)

body = """
us_restaurants = pd.read_csv('drive/MyDrive/Colab Notebooks/us_restaurants.csv')
us_restaurants.info()
"""
st.code(body, language="python")

body = """
df_restaurants['name'] = us_restaurants['RestaurantName'].unique()[0:380]
df_restaurants = df_restaurants[['id_restaurant','name','latitude','longitude','total_orders']]
"""
st.code(body, language="python")

st.markdown('#### Ajustando o restaurante 31 para desconhecido visto que a coordenada é 0,0')
body = """
df_restaurants.loc[31,'name'] = 'Desconhecido'
"""
st.code(body, language="python")

st.markdown('#### Criando cadastro de cidades')
body = """
from datetime import datetime
last_time = datetime.now(tz=None)
def check_time():
  global last_time
  print(datetime.now(tz=None) - last_time)
  last_time = datetime.now(tz=None)
  return None

check_time()
"""
st.code(body, language="python")

body = """
#Apanhando o nome real das cidades, base nas coordenadas dos restaurantes
#Para isso utilizarei inicialmente a biblioteca Nominatim
geolocator = Nominatim(user_agent="Biazoto")
def city_by_location(delivery):
    coord = f"{delivery['restaurant_latitude']}, {delivery['restaurant_longitude']},zoom=10"
    location = geolocator.reverse(coord, exactly_one=True)
    check_time()
    address = location.raw['address']
    state_district = address.get('state_district', '')
    check_time()
    return state_district
    
df_deliveries.apply(city_by_location, axis=1)
# ~0.5 seguntos por linha. Para 42 mil linhas será necessário 5h e 50min
"""
st.code(body, language="python")

body = """
#tentando requisições diretas ao servidor nominatim para verificar se o tempo é menor
def city_by_location(delivery):
  api_url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={delivery['restaurant_latitude']}&lon={delivery['restaurant_longitude']}&zoom=8"
  location = requests.get(api_url)
  check_time()
  state_district = location.json()['address'].get('state_district','')
  return state_district
    
df_deliveries.apply(city_by_location, axis=1)
#mesma situação ~5s por delivery
"""
st.code(body, language="python")

body = """
##### Utilizando k-means para agrupar as entregas por cidade
Assim será possivel requisitar o nome da cidade por grupo ao invés de faze-lo por entrega onde diversas entregas são feitas na mesma cidade.
"""
st.markdown(body)

body = """
##### endendendo como o k-Means funciona
"""
st.markdown(body)

body = """
points = [[2,1],
          [2,2],
          [3,2],
          [2,7],
          [3,6],
          [5,3],
          [5,4],
          [6,4],
          [9,1],
          [9,2],
          [9,3]]
[x, y] = zip(*points)
plt.scatter(x=x,y=y)

model = cluster.KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=10, random_state=0)
model.fit(points)
"""
st.code(body, language="python")

body = """
print(model.cluster_centers_)
print(model.labels_) #a qual cluster cada ponto pertence
print(model.n_features_in_) #features analisadas x e y neste caso
print(model.n_iter_)
print(model.inertia_) #soma das distancia entre os pontos e o cluster elevado ao quadrado
#quanto menor a inertia melhor
"""
st.code(body, language="python")

body = """
[[5.33333333 3.66666667]
 [9.         2.        ]
 [2.5        6.5       ]
 [2.33333333 1.66666667]]
[3 3 3 2 2 0 0 0 1 1 1]
2
2
5.666666666666667
"""
st.text(body)

st.markdown('#### Descobrindo o melhor valor de k (menor inertia)')
body = """
X = df_deliveries[["restaurant_latitude","restaurant_longitude"]]
max_k = 30
## iterations
distortions = []
for i in range(1, max_k+1):
    if len(X) >= i:
       model = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
       model.fit(X)
       distortions.append(model.inertia_)
"""
st.code(body, language="python")

st.markdown('#### Plotando as distorções')
body = """
fig, axs = plt.subplots(2,2, figsize=(10,10))
axs[0,0].plot(distortions)
axs[0,1].plot(distortions)
axs[0,1].set_ylim(0, 2500000)
axs[1,0].plot(distortions)
axs[1,0].set_ylim(0, 10000)
axs[1,1].plot(distortions)
axs[1,1].set_xlim(20, 30)
axs[1,1].set_ylim(0, 200)
axs[1,1].set_xticks(range(20,31))
plt.show()
"""
st.code(body, language="python")
st.image('images/image2.png')

body = """
redução drástica até 22 a partir de então não ha grandes mudanças
por análise a distribuição de pontos dentro do próprio mapa será adotado 23 cidades como sendo o valor para K-means
"""
st.markdown(body)

body = """
model = cluster.KMeans(n_clusters=23, init='k-means++', max_iter=300, n_init=10, random_state=1)
model.fit(X)
"""
st.code(body, language="python")

st.markdown('#### Plotando os dados gerados pelo k-means em um mapa')
body = """
mp = folium.Map(location=[df_restaurants['latitude'].mean(), df_restaurants['longitude'].mean()], zoom_start=5)
def mark_restaurants(restaurant):
  if restaurant['latitude'] != 0:
    folium.Marker(
        location=[restaurant['latitude'], restaurant['longitude']],
        icon=folium.Icon(icon="cloud"),
    ).add_to(mp)
df_restaurants.apply(mark_restaurants, axis=1)
for lat, lon in model.cluster_centers_:
  marker = folium.Marker(
      location=[lat, lon],
      icon=folium.Icon(icon="cloud",color='red')
  ).add_to(mp)
  mp.keep_in_front(marker)
mp
"""
st.code(body, language="python")
st.image('images/image3.png')

body = """
#ajustando nome das colunas city e definindo os nomes reais das cidades de acordo com a localização
model.labels_ 
df_cities = pd.DataFrame(model.cluster_centers_, columns=['latitude', 'longitude'])
df_cities['id_city'] = df_cities.index
df_cities.loc[1,'latitude'] = 0
df_cities.loc[1,'longitude'] = 0
df_cities
"""
st.code(body, language="python")

st.markdown('#### Usando Nominatium para apanhar o nome das 23 cidades')
body = """
geolocator = Nominatim(user_agent="Biazoto")
def city_by_location(city):
    coord = f"{city['latitude']}, {city['longitude']}"
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    city = address.get('city', '')
    state_district = address.get('state_district', '')
    return city if city else state_district
df_cities['name'] = df_cities.apply(city_by_location, axis=1)
df_cities.loc[1, 'name'] = 'Desconhecido'
df_cities
"""
st.code(body, language="python")

st.markdown('#### Setando as cidades no df_deliveries')
body = """
df_deliveries['id_city'] = model.labels_
"""
st.code(body, language="python")

st.markdown('#### Apanhando dados relativos as cidades da Índia da internet')
body = """
#Funciona mas não retorna a população e as coordenadas.
#Se fosse apanhar estas duas informações via api levaria muito tempo visto que cerca de 5k linhas
import requests
import json

config = {
  'method': 'GET',
  'url': 'https://andruxnet-world-cities-v1.p.rapidapi.com/',
  'headers': {
    'X-RapidAPI-Key': '93d0bd1f30mshb042a7251134ac9p1784f4jsnaec15200b295',
    'X-RapidAPI-Host': 'andruxnet-world-cities-v1.p.rapidapi.com'
    },
  'params': {
    'query': 'india', 
    'searchby': 'country'
    } 
}
response = requests.request(**config)
cities = response.json()
del cities
"""
st.code(body, language="python")

body = """
#Cria dataframe com as cidades da india com população superior a 1000 habitantes
def create_india_cities_dataframe():
  url = "https://documentation-resources.opendatasoft.com/api/records/1.0/search/?dataset=geonames-all-cities-with-a-population-1000&q=&rows=4000&refine.country_code=IN"
  request = requests.get(url)
  cities = request.json()['records']
  def transform_data_api(row):
    return {'name': row['fields']['name'],
            'latitude': row['fields']['coordinates'][0],
            'longitude': row['fields']['coordinates'][1],
            'population': row['fields']['population']
            }
  cities = map(transform_data_api, cities)
  return pd.DataFrame(cities)

df_india_cities = create_india_cities_dataframe()
df_india_cities
"""
st.code(body, language="python")

st.markdown('#### Merging dos dados obtidos da internet')
body = """
#Ajustando o nome das cidades em df_cities com base no nome e na localização feita de modo manual
df_india_cities[df_india_cities['name'].str.contains('\w*kulam\w*', regex='True', na=True)]
df_cities.loc[df_cities['name'] == 'Mysuru', 'name'] = 'Mysore'
df_cities.loc[df_cities['name'] == 'Pune City', 'name'] = 'Pune'
df_cities.loc[df_cities['name'] == 'Ludhiana', 'name'] = 'Ludhiāna'
df_cities.loc[df_cities['name'] == 'North Goa', 'name'] = 'Goa Velha'
df_cities.loc[df_cities['name'] == 'Dehradun', 'name'] = 'Dehra Dūn'
df_india_cities.loc[(df_india_cities['name'] == 'Pratāpgarh') & (df_india_cities['longitude'] == 74.78162), 'name'] = 'Pratāpgarh 2'
df_cities.loc[df_cities['name'] == 'Prayagraj', 'name'] = 'Pratāpgarh'
df_cities.loc[df_cities['name'] == 'Bhopal', 'name'] = 'Bhopāl'
df_cities.loc[df_cities['name'] == 'Ernakulam', 'name'] = 'Edakkulam'
"""
st.code(body, language="python")

body = """
df_cities = (df_cities.merge(df_india_cities, on='name', how='outer')
                      .sort_values('id_city')
                      .loc[:,['id_city','name','latitude_y','longitude_y','population']])
"""
st.code(body, language="python")

body = """
df_cities['id_city'] = df_cities.index
df_cities = df_cities.rename(columns={'latitude_y':'latitude',
                                      'longitude_y':'longitude'})
df_cities.loc[1,'latitude'] = 0
df_cities.loc[1,'longitude'] = 0
df_cities.loc[1,'population'] = 0
df_cities

del df_india_cities
"""
st.code(body, language="python")

body = """
df_aux = (df_cities.merge(df_deliveries, on='id_city', how='outer')
                   .loc[:,['id_city', 'name', 'latitude', 'longitude','population','id_delivery']]
                   .groupby('id_city', dropna=False)
                   .agg(name=('name','first'),
                        latitude=('latitude','first'),
                        longitude=('longitude','first'),
                        population=('population','first'),
                        entregas=('id_delivery','count'))
                   .reset_index()
                   .sort_values('id_city').head(30))
"""
st.code(body, language="python")

body = """
#cidade id 28 é um outlier. 
# pode ser que a popuação esta errada ou que hajam entregas nela que não estaõ nos data sets
df_aux.loc[df_aux['id_city'] == 28, 'population'] = 1
df_aux.loc[df_aux['entregas'] == 0, 'entregas'] = 1
df_aux.loc[df_aux['population'] == 0, 'population'] = 1

df_aux['potencial'] = df_aux['population'] / df_aux['entregas']

df_cities = df_aux.loc[:,:]
"""
st.code(body, language="python")