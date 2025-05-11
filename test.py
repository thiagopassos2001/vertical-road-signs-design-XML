import geopandas as gpd
import shapely

if __name__=="__main__":
    CRS = "EPSG:31984"
    datetime_column = "Timestamp"
    file_path = "data/image/gpkg/M01-S01-CE-350-1-BRUTO-FOTO.gpkg"
    
    gdf = gpd.read_file(file_path).to_crs(CRS)
    gdf = gdf.sort_values(by=datetime_column)
    
    line_string = shapely.LineString(list(gdf["geometry"].apply(lambda value:list(value.coords)[0])))
    gdf_line = gpd.GeoDataFrame({"SENTIDO":["CRESCENTE"]},geometry=[line_string],crs=CRS)

    gdf_line.to_file("data/image/gpkg/M01-S01-CE-350-1-BRUTO-FOTO-LINHA.gpkg",driver="GPKG",index=False)
    print("'-'")