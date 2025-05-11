# -*- coding: utf-8 -*-
"""
toolkitGIS
----------
"""

# Funções complementar GIS
# Imports
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import timeit

pd.options.mode.chained_assignment = None  # default='warn'

# Funcions
def SinuosityLineString(geom_linestring):
    '''
    Calcula a sinoosidade de uma linha
    ----------
    Parameters
    ----------
    geom : shapely.LineString
        Line geometry
    Returns
    -------
    sinuosity : float
        Sinuosity value. If the length = 0, the sinuosity = 0
        If the delta distance between the start point and end point is 0, return
        np.inf value.
    '''
    
    # Length of linestring
    length = geom_linestring.length
    
    # Start and end points
    start_pt = geom_linestring.interpolate(0)
    end_pt = geom_linestring.interpolate(1, normalized=True)
    
    # Linear distance
    delta_dist = start_pt.distance(end_pt)
    
    # Sinuosity calculator
    if delta_dist==0:
        if length==0:
            sinuosity = 0
        else:
            sinuosity = np.inf
    else:
        sinuosity = length / delta_dist
    
    return sinuosity

def SortPointsBySpaceTime(
        gdf,
        closest_point=None,
        time_column=None,
        max_gap_time_sec=1,
        max_dist_gap_time_meter=50,
        sort_column="SEQUENCE ORDER",
        distance_to_next_point_column="DISTANCE TO NEXT POINT",
        print_progress=False
        ):
    
    '''
    Ordena um conjunto de ponto linearmente por meio de atributos espaço-temporais
    ----------
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        gpd.GeoDataFrame file, with Coordinate Reference System (CRS) in unit meter
        geometry is a shapely.Point
    closest_point : shapely.Point, optional
        Reference point near the first point. If not specified, uses the point 
        furthest from the centroid formed by the set of geometry points.
    time_column : str, optional
        Refers to the datetime column in the gdf. If not specified, uses Euclidean distance
        to order the points (bad performace). If specified, that user to choice the
        logic points in the trajectory.
    max_gap_time_sec : int, optional (default = 1)
        Refers to the time buffer relative to the time to seek the next point in seconds. 
        Buffer defined as t +/- max_gap_time_sec. If not point in the buffer, uses Euclidean distance
        to order the points. Requires column "time_column".
    max_dist_gap_time_meter : float, optional (default = 50)
        Rrefers to the maximum distance between consecutive points by the time buffer criterion.
        Useful for sequences of points with repeating times. If the distance is greater, uses Euclidean
        distance to order the points. Requires column "time_column".
    sort_column : str, optional (default = "SEQUENCE ORDER")
        Refers to the column with the ordering of the points.
    distance_to_next_point_column : str, optional (default = "DISTANCE TO NEXT POINT")
        Refers to the column with the distance between two sequential points.
    print_progress : bool, , optional (default = False)
        If True, print the progress os operation.
    Returns
    -------
    gdf : gpd.GeoDataFrame
        The same gdf input, sorted by column "sort_column" and new columns
        * distance_to_next_point_column
    '''
    
    # Sistea CRS do arquivo original
    crs_gdf = gdf.crs
    
    # Timer
    if print_progress:
        start_timer = timeit.default_timer()
    
    # Se não tiver "closest_point", estima o ponto mais extremo como inicial
    if closest_point==None:
        # Ponto correspondente ao centroide dos pontos
        centroid_point = gdf.dissolve()['geometry'].geometry.centroid[0]
        # Distância de cada ponto ao centroide
        gdf['distance_to_centroid'] = gdf['geometry'].apply(lambda x:shapely.distance(x,centroid_point))
        # Ponto mais distante
        closest_point = gdf.sort_values('distance_to_centroid',ascending=False)
        closest_point = closest_point['geometry'].iloc[0]
        # Remove a coluna auxiliar "distance_to_centroid"
        gdf = gdf.drop(columns=['distance_to_centroid'])
    
    # Se não for passado uma "time_column", o ordenamento é feito ponto a ponto
    # Estima o próximo ponto pela distância ponto a ponto
    if time_column==None:
        pass
    else:
        gdf['instant'] = (gdf[time_column]-gdf[time_column].min()).dt.total_seconds()
    
    # Cria uma lista para armazenar os GeoDataFramas já processados e reconcatenar no fim
    gdf_sorted_list = []
    
    # Inicia o contador da sequência
    order_count = 1
    # Máximo do contador
    max_count = len(gdf)
    
    # Considera os critérios temporais inicialmente falsos
    temporal_validation_criterion = False
    closest_instant_time_count = 0
    dist_to_next_point = 0
        
    # Enquanto não remover todos os pontos que não foram processados, executar
    while len(gdf)>0:
        # Se tiver coluna "time_column" e não for a primeira execução, já haverá
        # "closest_instant_time_count", que refere-se a quantidade de pontos pré filtrados
        # Para a próxima execussão, depende do buffer temporal
        if time_column!=None and order_count>1 and closest_instant_time_count>0:
            # A partir da segunda execução,
            # Ordena pelo "delta_absolute_instant", calculado na iteração anterior
            # O ponto mais próximo corresponde a esse com a menor distância
            # O próprio ponto usade de referência no cálculo foi removido Na iteração anterior
            gdf = gdf.sort_values('delta_instant')
            
            # Verifica se a distância máxima do ponto selecionado está a uma distância
            # Razoável, se não, ignora e executa o modo "bruto" com "temporal_validation_criterion = False"
            
            dist_to_next_point = shapely.distance(gdf['geometry'].iloc[0],closest_point).values[0]
            if dist_to_next_point<=max_dist_gap_time_meter:
                temporal_validation_criterion = True
                # Toma o ponto próximo
                gdf_closest_point = gdf.iloc[:1]
                # Atualiza a distância espacial
                gdf_closest_point[distance_to_next_point_column] = dist_to_next_point
                # Modo de execução
                mode_process = 'Time'
                
            else:
                temporal_validation_criterion = False
        
        # print(f"Critérios {time_column,order_count,closest_instant_time_count,dist_to_next_point,temporal_validation_criterion}")
        # Método direto pela distância euclidiana
        if time_column==None or order_count==1 or closest_instant_time_count==0 or not temporal_validation_criterion:
            # Calcula a distância dos pontos até o ponto mais próximo
            gdf[distance_to_next_point_column] = gdf['geometry'].apply(lambda x:shapely.distance(x,closest_point))
            # Ordena os valores do menor para o maior e reseta o index
            gdf = gdf.sort_values(distance_to_next_point_column)

            # Toma o ponto próximo (a distância já foi calculada com os demais pontos)
            gdf_closest_point = gdf.iloc[:1]
            
            # Modo de execução
            mode_process = 'Space'
        
        # Atribui a ordem
        gdf_closest_point[sort_column] = order_count
        
        # Adiciona a lista "gdf_sorted"
        gdf_sorted_list.append(gdf_closest_point)
        
        # Atauliza o contador
        order_count = order_count + 1
        # Atualiza o ponto mais próximo
        closest_point = gdf_closest_point['geometry']
        # Se tiver a coluna "time_column", registra o último intervalo de tempo
        if time_column==None:
            pass
        else:
            # Pega o instante do ponto
            closest_instant_time = gdf_closest_point['instant'].values[0]
            # Calcula o delta em relação ao pontos
            gdf['delta_instant'] = abs(gdf['instant']-closest_instant_time)
            # Filtra e verifica a quantidade de pontos pelo critério 
            # -1 para descontar o 0 que vai aparecer do prórprio ponto
            gdf_buffer_time = gdf[gdf['delta_instant'].between(0,max_gap_time_sec)]
            closest_instant_time_count = len(gdf_buffer_time)-1
        
        # Remove o ponto selecionado dos pontos possíveis
        gdf = gdf[1:]
        
        # Fica em loop até o shape terminar
        # Exibir progresso
        if print_progress:
            # Estimativa de tempo
            stop_timer = timeit.default_timer()
            count_timer = stop_timer - start_timer
            eta = (count_timer/(order_count-1))*(max_count-order_count-1)
            print(f"Processando Modo={mode_process}... {order_count-1}/{max_count} (Run: {int(count_timer/60)}min:{int(count_timer%60)}s - ETC:{int(eta/60)}min:{int(eta%60)}s)")
    
    # Concatena a lista de pontos e ordena pela coluna "sort_column"
    gdf_sorted = gpd.GeoDataFrame(pd.concat(gdf_sorted_list,ignore_index=True),geometry='geometry',crs=crs_gdf)
    gdf_sorted = gdf_sorted.sort_values(sort_column)
    
    # Se a coluna "time_column" for diferente de None, remove as colunas auxiliares
    if time_column==None:
        pass
    else:
        gdf_sorted = gdf_sorted.drop(columns=['instant','delta_instant'])
    
    # Retorna o gdf ordenado com as colunas de ordem e distância entre pontos sequenciais
    return gdf_sorted

def SplitLineStringByMaxLengthRandom(
        line,
        max_length=20.0,
        random=False,
        mean=0,
        max_diff=1):
    '''
    Divide uma linha em segmentos de extensão "max_length" (exatas) e permite variações
    normalmente distribuidas com média "mean" e diferença máxima "max_diff"
    ----------
    Parameters
    ----------
    line : shapely.LineString
        Describe
    max_length : float, optional (default = 20.0)
        Describe
    random : bool, optional (default = True)
        Describe
    mean : float, optional (default = 0)
        Describe
    max_diff : float, optional (default = 1)
        Describe
    Returns
    -------
    list_segments : list
        Return a list of segments (shapely.LineString)
    '''
    # Verifica se o comprimento é válido
    if max_length<=0:
        raise ValueError(f"O comprimento {max_length} é inválido.")
    
    # Se a linha já é menor ou igual ao comprimento máximo, retorna ela mesma
    if line.length<=max_length:
        return [line]
    
    # Estima os limites de divisão
    split_lenght = np.arange(0,line.length+max_length,max_length)
    # Acrescenta um fator estocástico a divisão
    if random:
        split_lenght_rand = np.random.normal(mean,max_diff/6,len(split_lenght)-2)
        split_lenght_rand = np.concatenate((np.zeros(2),split_lenght_rand,np.zeros(1)))
        split_lenght = split_lenght + split_lenght_rand[1:] - split_lenght_rand[:-1]

    # Cria substrings com a extensão desejada a partir da string original
    list_segments = [shapely.ops.substring(line,start,end) for start,end in zip(split_lenght[:-1],split_lenght[1:])]
    
    return list_segments

if __name__=="__main__":
    pass
                
