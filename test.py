import geopandas as gpd

if __name__=="__main__":
    CRS = "EPSG:31984"
    file_path = r"C:\Users\User\Desktop\Repositórios Locais\vertical-road-signs-design-XML\data\image\gpkg\M01-S01-CE-350-1-BRUTO-FOTO.gpkg"
    
    gdf = gpd.read_file(file_path).to_crs(CRS)

    print(gdf)