# Problema de negócio
A Cury Company é uma empresa localizada na Índia que possui um aplicativo que conecta restaurantes, entregadores e clientes. Esse aplicativo funciona de forma muito similar aos aplicativos iFood, Uber eats, Doordash, GrubHub comumente conhecidos mais ao ocidente.

De maneira geral, a empresa realiza os negócios conectando 3 partes:
Os restaurantes que desejam fazer entregas de seus pedidos.
Os entregadores, pessoas que tem enteresse na plataforma para faturar fazendo entregas.
Os clientes, pessoas que se utilizam da plataforma para ordenar alimentos.

Atualmente, o CEO da Cury Company tem ciência que o faturamento da empresa esta crescendo. Porém, gostaria de entrar mais a fundo sob o crescimento da mesma a fim de entender oque cresce quando a empresa cresce e de que forma cresce. Assim ele poderia desenvolver maior acuracidade sob quais alavancas acionar para estimular ou não o crescimento da empresa conforme as estratégias futuras.

Sob esta demanda, fica definida a necessidade inicial de se criar uma visão geral sob crescimento e o atual estado de cada um dos pilares do negócio da Cury Company (clientes, restaurantes e entregadores).

Como não existe a possibilidade de criar a visão de clientes tendo em vista a falta de informações, ela será substituida por uma visão geral da empresa principalmente relativo as entregas realizadas dentro da plataforma.

# Premissas assumidas para a análise dos dados
- Os dados utilizados são de 11/02/2022 à 06/04/2022
- Será elaborado 3 principais visões sob o negócio (entregas, restaurantes e entregadores)
- Assume-se que a empresa possui o modelo de negócio do tipo Marketplace
- Será utilizado algorítimos de agrupamento para criar um cadastro fictício de restaurantes a fim de ser possivel a sua análise.

# Estratégia da solução
Será criado 3 painéis contendo algumas visões sob os negócios da empresa

### Visão do crescimento da empresa (entregas)
- Quantidade de entregas diárias e semanais
- Indice geral das entregas calculado através da fórmula:
velocidade média * avaliação da entrega
- Evolução da avaliação média das entregas
- Evolução da velocidade média das entregas em km/min
- Mapa com a mancha de cores das entregas, população da Índia e potencial de crescimento

### Visão do crescimento dos restaurantes
- Quantidade total de restaurantes
- Média de entregas e distribuição de entregas por restaurante bem como a evolução da distribuição das entregas ao longo das semanas.
- Tempo médio levado para o preparo e sua distribuição
- Distância média das entregas e sua distribuição

### Visão do crescimento dos entregadores
- Quantidade total de entregadores
- Média de entregas por entregador
- Menor idade entre os entregadores
- Maior idade entre os entregadores
- Distribuição da quantidade de entregas entre os entregadores
- Evolução semanal da distribuição da quantidade de entrega dos entregadores

# Top insights
- Existem 3 grupos de entregadores muito bem definidos quanto a quantidade de entregas.
- Também existem 2 grupos de restaurantes bem definidos quanto a quantidade de entregas.
- As manchas de cores demonstram que ainda existem regiões da Índia com claro potencial de crescimento quando comparada a outras regiões. Isso não determina que apenas essas regiões possam possuir crescimento, mas somente que estas regiões possuem potencial claro de crescimento.

# O produto final
O painel é um produto online, encontra-se disponível na internet e pode ser acessada através do link abaixo:
https://biazotogabriel-curry-company-remake-home-u84f9u.streamlit.app/entregadores

# Conclusão
O objetivo deste projeto foi a elaboração de um produto que aprofunde o crescimento do faturamento da empresa.
Através do mesmo foi possível verificar que assim como o faturamento, o número de pedidos da empresa cresce. Com isso os parâmetros que definem os três pilares do negócio passam por transformações também. Estas transformações podem ser resultado do crescimento da empresa bem como podem ser causadores do mesmo. Cabe a trabalhos posteriores analisar quais destas transformações poderiam ser utilizadas como alavancas no futuro.

# Proximos passos
- Existe a necessidade de futuramente aprofundar as analises sobre os insights gerados. Principalmente sobre os grupos de etregadores e restaurantes identificados através do número de entregas.
- A criação de mais filtros para a ferramenta poderia enrriquecer muito as analises.
- Analisar quais transformações nos dados poderiam ser utilizadas como alavancas de crescimento.
