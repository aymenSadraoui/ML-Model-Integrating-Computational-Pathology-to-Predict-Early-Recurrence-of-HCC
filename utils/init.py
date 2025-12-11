from pathlib import Path
import yaml


def create_directories():
    results_path = "./results"
    overview_path = "./results/overview_wsis"

    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for pth in [results_path, overview_path]:

        Path(pth).mkdir(exist_ok=True)

    for e in config["paths"]:
        pth = config["paths"][e]
        print(e, ":", pth)
        Path(pth).mkdir(exist_ok=True)


if __name__ == "__main__":
    create_directories()
