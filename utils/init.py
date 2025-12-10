from pathlib import Path
import yaml

def create_directories():
    patches_path = "../data/patches/"
    coords_path = "../checkpoints/coords_pickles"
    results_path = "../results"
    overview_path = "../results/overview_wsis"
    
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for pth in [patches_path, coords_path, results_path, overview_path]:
        
        Path(pth).mkdir(exist_ok=True)
        
    for e in config["paths"]:
        pth = config["paths"][e]
        print(e,':',pth)
        Path(pth).mkdir(exist_ok=True)


if __name__ == "__main__":
    create_directories()