from model import *

if __name__=="__main__":
    start_timer = timeit.default_timer()
    print("Executando...")
    gdf_file_path = "data/shape/src points.parquet"
    sre_list = """
        371ECE0100S0
        371ECE0260S0
        266ECE0120S0
        323ECE0008S0
        371ECE0220S0
        265ECE0125S0
        085ECE0260E0
        085ECE0250E0
        350ECE0008E0
        321ECE0108S0
        123ECE0280S0
        179ECE0150S0
        179ECE0030S0
        025ECE0070S0
        371ECE0225S0
        265ECE0120S0
        348ECE0090S0
        156ECE0030S0
        385ECE0110S0
        253ECE0480S0
        366ECE0030S0
        329ECE0050S0
        179ECE0240S0
        060ECE0620S0
        350ECE0006S0
        166ECE0300S0
        292ECE0140S0
        """.replace(" ","").strip().split("\n")
    CreateDesignAxisMultiSRE(sre_list,gpd.read_parquet(gdf_file_path),"C:/Users/thiagop/Desktop/Repositório (Local)/data/dxf/DXF_test2.dxf")
    # CreateDesignAxis("040ECE0010E0",gpd.read_parquet(gdf_file_path))
    
    stop_timer = timeit.default_timer()
    count_timer = stop_timer - start_timer
    print(f"Execução: {int(count_timer//60)}min:{round(count_timer%60,2)}s")