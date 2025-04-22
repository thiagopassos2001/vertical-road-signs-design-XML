import xml.etree.ElementTree as ET
from thefuzz import fuzz,process
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import timeit
import ezdxf
import sys
import os

sys.path.append("toolkitGIS")
from toolkitGIS import *

pd.options.mode.chained_assignment = None  # default='warn'

def ParseElementXML(element,item):
        """
        Pega iterativamente os textos do arquivo ".xml"
        """
        if len(list(element))==0:
            item[element.tag] = element.text
        else:
            for child in list(element):
                ParseElementXML(child,item)

def XMLtoDatFrame(file_path):
    """
    Transforma o arquivo ".xml" em um pd.DataFrame
    Cada boudingbox é uma linha do dataframe
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    main_info = []
    data = []
    merge_data = []

    for child in list(root):
        item = {}
        ParseElementXML(child,item)
        if child.tag=='object':
            data.append(item)
        else:
            main_info.append(item)

    for i in data:
        complete_main_info = {}
        for j in main_info:
            complete_main_info.update(j)
        complete_main_info.update(i)
        merge_data.append(complete_main_info)

    return pd.DataFrame.from_dict(merge_data)

def ImportFromXML(folder_path):
    """
    Recebe como imput uma pasta
    Retorna os arquivos .xml da pasta (e subpastas em um dataframe)
    """
    # Todos os arquivos da pasta
    all_files = [os.path.join(dirpath,f) for (dirpath,dirnames,filenames) in os.walk(folder_path) for f in filenames]
    
    # Cria um dataframe concatenanto cada um dos dataframes com origem nos XMLs
    df = pd.concat([XMLtoDatFrame(f) for f in all_files],ignore_index=True)
    
    # Ajuste do tipos de dados
    df['xmin'] = df['xmin'].astype(int)
    df['xmax'] = df['xmax'].astype(int)
    df['ymin'] = df['ymin'].astype(int)
    df['ymax'] = df['ymax'].astype(int)
    df['width'] = df['width'].astype(int)
    df['height'] = df['height'].astype(int)

    # Outras informações geométricas
    df['xcenter'] = df['xmin'] + (df['xmax']-df['xmin'])*0.5
    df['ycenter'] = df['ymin'] + (df['ymax']-df['ymin'])*0.5

    # Retona o dataframe
    return df

def AdjustCropLimits(df,w_h=None,padding_px=10):
    """
    Recebe um dataframe padrão xml do labelimg
    'w_h' = w/h = width/height refere-se a relação da proporção entre largura e altura
    Retorna o dataframe com as colunas referents ao recorte da imagem
    """
    if w_h!=None:
        df['crop_keep'] = df.apply(lambda x:'h' if (x['ymax']-x['ymin']+2*padding_px)>=(x['xmax']-x['xmin']+2*padding_px) else 'w',axis=1)

        df['crop_xmin'] = df['xcenter'] - df.apply(lambda x:(x['xmax']-x['xmin']+2*padding_px)*0.5 if x['crop_keep']=='w' else (x['ymax']-x['ymin']+2*padding_px)*0.5*w_h,axis=1)
        df['crop_xmax'] = df['xcenter'] + df.apply(lambda x:(x['xmax']-x['xmin']+2*padding_px)*0.5 if x['crop_keep']=='w' else (x['ymax']-x['ymin']+2*padding_px)*0.5*w_h,axis=1)
        df['crop_ymin'] = df['ycenter'] - df.apply(lambda x:(x['ymax']-x['ymin']+2*padding_px)*0.5 if x['crop_keep']=='h' else (x['xmax']-x['xmin']+2*padding_px)*0.5*(1/w_h),axis=1)
        df['crop_ymax'] = df['ycenter'] + df.apply(lambda x:(x['ymax']-x['ymin']+2*padding_px)*0.5 if x['crop_keep']=='h' else (x['xmax']-x['xmin']+2*padding_px)*0.5*(1/w_h),axis=1)
    else:
        df['crop_keep'] = np.nan
        df['crop_xmin'] = df['xmin'] - padding_px
        df['crop_xmax'] = df['xmax'] + padding_px 
        df['crop_ymin'] = df['ymin'] - padding_px 
        df['crop_ymax'] = df['ymax'] + padding_px

    df['crop_height'] = df['crop_ymax'] - df['crop_ymin']
    df['crop_width'] = df['crop_xmax'] - df['crop_xmin']

    # Correção de borda
    # Largura
    df['crop_xmin'] = df['crop_xmin'].apply(lambda x:x if x>0 else 0)
    df['crop_xmax'] = df.apply(lambda x:x['crop_xmax'] if x['crop_xmin']>0 else x['crop_width'],axis=1)
    df['crop_xmax'] = df.apply(lambda x:x['crop_xmax'] if x['crop_xmax']<=x['width'] else x['width'],axis=1)
    df['crop_xmin'] = df.apply(lambda x:x['crop_xmin'] if x['crop_xmax']<=x['width'] else x['width'] - x['crop_width'],axis=1)
    # Altura
    df['crop_ymin'] = df['crop_ymin'].apply(lambda x:x if x>0 else 0)
    df['crop_ymax'] = df.apply(lambda x:x['crop_ymax'] if x['crop_ymin']>0 else x['crop_height'],axis=1)
    df['crop_ymax'] = df.apply(lambda x:x['crop_ymax'] if x['crop_ymax']<=x['height'] else x['height'],axis=1)
    df['crop_ymin'] = df.apply(lambda x:x['crop_ymin'] if x['crop_ymax']<=x['height'] else x['height'] - x['crop_height'],axis=1)

    return df

def CheckPatternLabel(label,valid_label_list,similarity_threshold=80):
    # Melhor correspondência e similaridade
    best_match, similarity = process.extractOne(
        label,
        valid_label_list,
        scorer=fuzz.ratio)

    # Resultado, mantém se similarity for menor que o limite mínimo
    result = best_match if similarity >= similarity_threshold else label
    
    return result

def GetCoordFromXLSX(id,df_xlsx,lat_col="LAT",lon="LON"):
    pass

def GetCoordFromGPKG(df,gdf_gpkg,match_col):
    """
    Recebe o dataframe e um geodataframe como input, 
    Retorna um geodataframe com as coordenadas e o id alfanumérico da imagem
    """
    df = df.merge(gdf_gpkg[[match_col,"ID","geometry"]],on=match_col,how="left").rename(columns={"ID":"ID IMAGE",})
    
    # Conversão para geodataframe
    gdf = gpd.GeoDataFrame(
        df,
        geometry="geometry",
        crs="EPSG:31984").to_crs(31984)
    gdf["ID IMAGE"] = gdf["ID IMAGE"].astype(str)
    gdf["LONGITUDE"] = gdf.to_crs(4326)["geometry"].x.astype("float64")
    gdf["LATITUDE"] = gdf.to_crs(4326)["geometry"].y.astype("float64")
    
    return gdf

def GetCoordFromImage(id,folder_path,lat_col="LAT",lon="LON"):
    pass

def AdjuntIDDuplicated(id_series):
    pass

def CreateAxisFromGPKGImage(
        gdf,
        first_point_name=None,
        max_length=20,
        random=False,
        mean=0,
        max_diff=1,
        time_column="Timestamp",
        return_type="line",
        tolerance=0.5):
    """
    Recebe o .gpkg tratado, gerado de diversas fontes, 
    como a extensão "ImportPhotos" do QGIS

    Retorna um geodataframe de pontos ou linhas
    "line" para a linha completa
    "point" para o ponto final da linha
    """
    valid_return_option = ["line","point"]
    if return_type not in valid_return_option:
        return ValueError(f"'{return_type}' inválido! Escolha entre {valid_return_option}")
    
    # Ponto inicial de início do ordenamento
    closest_point = None
    # Se indicar o ponto inicial, caltera o "closest_point"
    if first_point_name!=None:
        first_point = gdf[gdf["Name"]==first_point_name]
        if not first_point.empty:
            # Geometria
            closest_point = first_point["geometry"].iloc[0]
        else:
            raise Warning(f"O nome {first_point_name} não foi encontrado. Executando com o padrão 'None'.")
            
    # Ordena os pontos para cria um eixo 
    gdf_sorted = SortPointsBySpaceTime(
        gdf,
        closest_point=closest_point,
        time_column=time_column,
        sort_column="ORDEM",
        distance_to_next_point_column="DISTANCIA"
    )

    # Cria um eixo com os pontos de campo e simplifica a geometria um pouco
    axis_line_string = shapely.LineString(gdf_sorted["geometry"].apply(lambda value:list(value.coords)[0]).tolist())
    axis_line_string = axis_line_string.simplify(tolerance=tolerance,preserve_topology=True)
    gdf_line_string = gpd.GeoDataFrame(geometry=[axis_line_string],crs="EPSG:31984")
    gdf_line_string["COMPRIMENTO"] = gdf_line_string.length.astype("float64")

    # Quebra o eixo em segmentos se reta e organiza um geodataframe
    list_segments = SplitLineStringByMaxLengthRandom(
        axis_line_string,
        max_length=max_length,
        random=random,
        mean=mean,
        max_diff=max_diff)
    
    gdf_segment = gpd.GeoDataFrame(geometry=list_segments,crs="EPSG:31984")
    # Calcula o KM acumulado do segmento
    gdf_segment["COMPRIMENTO"] = gdf_segment.length.astype("float64")
    gdf_segment["KM"] = gdf_segment["COMPRIMENTO"].cumsum()/1000
    gdf_segment["KM"] = gdf_segment["KM"].round(3).astype("float64")

    # Se o formato de saída for ponto
    if return_type=="point":
        first_row = gpd.GeoDataFrame(
            {"COMPRIMENTO":[0],
             "KM":[0]},
            geometry=gdf_segment.iloc[0:1]["geometry"].apply(lambda value:shapely.Point(value.coords[0])),
            crs="EPSG:31984")
        gdf_segment["geometry"] = gdf_segment["geometry"].apply(lambda value:shapely.Point(value.coords[-1]))

        gdf_segment = gpd.GeoDataFrame(pd.concat([first_row,gdf_segment],ignore_index=True),geometry="geometry",crs="EPSG:31984")

    return gdf_line_string,gdf_segment

def CreateDesignAxis(
        sre,
        gdf,
        first_point_name=None,
        output_dxf_folder_path="data/dxf",
        output_gpkg_folder_path="data/gpkg",
        time_column="DATAHORA",
        max_length=15,
        export=True):

    gdf = gdf.to_crs(31984)
    gdf = gdf[gdf["SRE"]==sre]

    if gdf.empty:
        return ValueError(f"Não há fotos do SRE {sre}.")
    
    if not gdf[gdf["SENTIDO"]=="CRESCENTE"].empty:
        gdf = gdf[gdf["SENTIDO"]=="CRESCENTE"]
    elif not gdf[gdf["SENTIDO"]=="DECRESCENTE"].empty:
        gdf = gdf[gdf["SENTIDO"]=="DECRESCENTE"]
    
    if gdf.empty:
         return ValueError(f"Não há fotos do SRE {sre} no sentido 'CRESCENTE' ou 'DECRESCENTE'.")
    
    dxf_output_file_path = os.path.join(output_dxf_folder_path,sre+"_eixo_projeto.dxf")
    gpkg_output_file_path = os.path.join(output_gpkg_folder_path,sre+"_eixo_projeto.gpkg")
    
    design_axis,design_axis_segment = CreateAxisFromGPKGImage(
        gdf,
        first_point_name=first_point_name,
        time_column=time_column,
        max_length=max_length,
        return_type="point")
    
    if export:
        ExportGeoDataFrameToDXF(design_axis,dxf_output_file_path)
        design_axis_segment.to_file(gpkg_output_file_path,driver="GPKG",index=False)
    
    return design_axis,design_axis_segment

def CreateDesignAxisMultiSRE(sre_list,gdf,output_dxf_folder_path,time_column="DATAHORA",):

    gdf = gdf.to_crs(31984)
    
    dxf_concat = []
    for sre in sre_list:
        print(f"{sre} Processando...")
        try:
            design_axis,_ = CreateDesignAxis(
                sre,
                gdf,
                first_point_name=None,
                time_column=time_column,
                export=False)
            
            design_axis["layer"] = sre
            dxf_concat.append(design_axis)
        
        except Exception as e:
            print(sre,e)
        finally:
            print(f"{sre} Finalizado!")
    
    dxf_concat = gpd.GeoDataFrame(pd.concat(dxf_concat),geometry="geometry",crs="EPSG:31984")

    ExportGeoDataFrameToDXF(dxf_concat,output_dxf_folder_path,set_layer=True)

def ExportGeoDataFrameToDXF(gdf,output_file_path,set_layer=False):
    # Criar novo DXF
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    # Converter cada geometria para DXF
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        # Converter para coordenadas locais (ajuste conforme necessário)
        if geometry.geom_type == "Point":
            x, y = geometry.x, geometry.y
            if set_layer:
                msp.add_point((x, y),dxfattribs={"layer": row["layer"]})
            else:
                msp.add_point((x, y))
        elif geometry.geom_type == "LineString":
            points = list(geometry.coords)
            if set_layer:
                msp.add_lwpolyline(points,dxfattribs={"layer": row["layer"]})
            else:
                msp.add_lwpolyline(points)
        elif geometry.geom_type == "Polygon":
            exterior = list(geometry.exterior.coords)
            if set_layer:
                msp.add_lwpolyline(exterior, close=True,dxfattribs={"layer": row["layer"]})
            else:
                msp.add_lwpolyline(exterior, close=True)

    # Salvar o arquivo DXF
    doc.saveas(output_file_path)     

def CreatePatternVerticalRoadSigns(xml_folder_path,sinv_pattern_code_file_path,gpkg_file_path,axis_file_path=None):
    
    df_xml = ImportFromXML(xml_folder_path)
    df_sinv_pattern_code = pd.read_csv(sinv_pattern_code_file_path)
    gdf_gpkg = gpd.read_file(gpkg_file_path).to_crs(31984)
    
    
    # Novo dataframe
    df = pd.DataFrame()
    
    df["ID PROJETO"] = df_xml["filename"].str.split(".").str[0].str.lower().astype(str)
    # df["ID INTERNO"] = df.index.astype("int64")

    df["CAMINHO"] = df_xml["path"].astype(str)
    df["SRE"] = df["CAMINHO"].str.split("\\").str[-2].astype(str)
    df["SENTIDO"] = df["CAMINHO"].str.split("\\").str[-3].astype(str)
    df["RODOVIA"] = df["CAMINHO"].str.split("\\").str[-4].astype(str)

    # Ajuste da codificação
    # Lista de códigos válidos
    valid_label_list = df_sinv_pattern_code["CÓDIGO"].tolist()
    # Código oriinal
    df["CÓDIGO"] = df_xml["name"].str.strip().astype(str)
    # Verificação necessária, independente da correção automática
    df["VERIFICAR"] = (-df["CÓDIGO"].isin(valid_label_list)).astype(bool)
    # Substitui o código original pelo corrigido
    df["CÓDIGO"] = df["CÓDIGO"].apply(lambda value:CheckPatternLabel(
        value,
        valid_label_list,
        similarity_threshold=80)).astype(str)
    
    # Cria o id para o match da coordenada com as informações do XML
    gdf_gpkg["ID MATCH COORD"] = gdf_gpkg["Path"].apply(lambda value:"_".join(os.path.normpath(value).split(os.sep)[-4:]))
    df["ID MATCH COORD"] = df["CAMINHO"].str.split("\\").str[-4:].apply("_".join)
    # Transforma o df em gdf (shape)
    gdf = GetCoordFromGPKG(df,gdf_gpkg,match_col="ID MATCH COORD")
    
    # Remove colunas auxiliares
    gdf = gdf.drop(columns=[
        "ID MATCH COORD"
    ])
    
    return gdf

if __name__=="__main__":
    print("Running tests...")