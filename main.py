from model import *

if __name__=="__main__":
    start_timer = timeit.default_timer()
    gdf_file_path = "data/LVC source/src points.parquet"
    sre = "232ECE0010S0"

    CreateDesignAxis(sre,gpd.read_parquet(gdf_file_path),)
    
    stop_timer = timeit.default_timer()
    count_timer = stop_timer - start_timer
    print(f"Execução: {int(count_timer//60)}min:{round(count_timer%60,2)}s")