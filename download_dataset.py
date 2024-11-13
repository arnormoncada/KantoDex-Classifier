import logging
import shutil
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ignore_files = {
    "Abra": ["10a9f06ec6524c66b779ea80354f8519.jpg", "Abra19.jpg"],
    "Alakazam": ["23bf91dbb0c043db80b76cb6243153b2.jpg"],
    "Bulbasaur": [
        "00000118.jpg",
        "00000113.png",
        "00000130.png",
        "00000148.png",
        "00000159.jpg",
        "00000181.jpg",
        "00000203.png",
        "00000207.jpg",
        "00000215.png",
        "00000222.jpg",
        "00000219.jpg",
        "00000231.jpg",
        "00000234.png",
        "00000239.jpg",
        "00000212.png",
        "00000224.png",
        "00000228.png",
    ],
    "Chansey": ["8d7e7967d00d418a9ed5b8a73496785a.jpg"],
    "Charmander": [
        "1f6dd61f0cbc4ebf88d622d7e9a800b0.jpg",
        "00000093.jpg",
        "00000125.png",
        "00000131.jpg",
        "00000157.jpg",
        "00000162.jpg",
        "00000175.jpg",
        "00000174.png",
        "00000173.jpg",
        "00000170.jpg",
        "00000176.png",
        "00000179.jpg",
        "00000183.jpg",
        "00000189.png",
        "00000188.jpg",
        "00000187.jpg",
        "00000206.jpg",
        "00000222.jpg",
        "00000220.jpg",
        "00000219.png",
        "00000223.png",
        "00000236.jpg",
        "00000232.jpg",
        "00000240.jpg",
        "00000244.jpg",
        "d37312ac10fe49c1bbe98c2b9ef43d53.jpg",
    ],
    "Dewgong": [
        "6288af352a514d5696d6021521d3b6f6.jpg",
        "9770de189bb6474b83d1cf0a069d8cfc.jpg",
        "d3d9e0a1961c4cbfbf53f70763ef91a1.jpg",
        "Dewgong55.jpg",
        "e36993f80c2c4a1b82ff5d7c10899501.jpg",
        "ec16cac8a0f546c09621f507a801ad5e.jpg",
    ],
    "Dratini": ["c039c3696a6a47beb74759366ceb177f.jpg"],
    "Electrode": ["bed127108dcb4803b692a087c09c2d75.jpg", "ffedeae7cbc046ee98968736704cd4de.jpg"],
    "Exeggcute": ["00000019.jpg"],
    "Fearow": ["00000031.png", "32c7d55dddd34598a9482b68ef4a4b6d.jpg"],
    "Geodude": ["9c83b9187db342cabdd84ee0ffbd09e3.jpg"],
    "Golem": ["631e5deb288f443abae5cebd81577789.jpg"],
    "Hitmonchan": ["8e397dd7852b4174a04ce78072de2b0c.jpg"],
    "Hitmonlee": ["13f315b8244b49408c686c4e5fd74d6c.jpg"],
    "Kabuto": [
        "1d8c4ffee6fa43cf9d3bdd715b756538.jpg",
        "39b137abcbd44cbda125494eb3a65367.jpg",
        "78928422c4b447b9a645bb44036c5b32.jpg",
        "c7262d41a0fc4f3a93e733099a8c584a.jpg",
        "ae6a41ca2a6940128b307ce7cb198893.jpg",
    ],
    "Kabutops": [
        "2121c985eeac42c18c442253f1375a1d.jpg",
        "36642dc632664ec0b583dcd83b70f76a.jpg",
        "Kabutops4.jpg",
    ],
    "Krabby": ["4a8e404aa34343dd946589ff88dd96aa.jpg"],
    "Lapras": [
        "2ddd0604de054de7a7a65220a09203fd.jpg",
        "47ed2c521c36487e99117bf0e80e7c57.jpg",
        "066a989fcb794137b0d39492dff5c38f.jpg",
        "e430383487c04a0f96959cde7185718d.jpg",
    ],
    "Machop": [
        "30e3a695403d44a3a7dc31aae9d9102b.jpg",
        "37f97f8d51c4419a9933dce05209fb0c.jpg",
        "60cb23a9539945dd89033df34ab7f5b1.jpg",
    ],
    "Magikarp": [
        "5ae06bf971674697bb663d63dea4cea1.jpg",
        "35d85bd72ace4430ad258bfe98a3710f.jpg",
        "ad4fb0954e204f7282241b70ffb5ba40.jpg",
    ],
    "Marowak": [
        "00000072.jpg",
        "a214d5754b5c48fea4dccdedf59644dd.jpg",
        "e0af71c14a2b4c298d81b0173d8f8c09.jpg",
        "eaa2ecfe794b422f8294d4a9bbce9797.jpg",
        "f2f2769669734a28bb848108a8da09ad.jpg",
    ],
    "Meowth": ["a490ecc81efe4069bd900d07a9257591.jpg", "a200003733ad491ab36414702fe9156e.jpg"],
    "Metapod": ["0ae588aca262462ba57aac0ee0245451.jpg"],
    "Mewtwo": ["00000136.png", "00000172.png", "00000167.jpg", "00000181.png", "00000220.png"],
    "Moltres": ["5762c703ef05447d88388d3c037eac2a.jpg", "7192048a7d214ad397940f42e8390f3f.jpg"],
    "Oddish": ["64cbc63d8bc24738bb0c4edda6e94e52.jpg"],
    "Omanyte": ["00bc5f276d5843aa9af9851a0d5663c2.jpg", "00000112.png"],
    "Onix": [
        "ac44b0aa3090452ca5fb83fec787b215.jpg",
        "ea0964e6378843f6816021304eed3803.jpg",
        "e9b62c9d35cc42898d5ef5a3ecfe6e47.jpg",
        "Onix8.jpg",
    ],
    "Paras": [
        "0ee0ce401e654c6aaeef7d795580751b.jpg",
        "1b13303aa991402bac27bd3397f65b7f.jpg",
        "5c72b416789747068b54f3047ae53604.jpg",
        "7f2d4c1160fa428c8b7faabc8c58dca2.jpg",
        "85aad36b538a47bba78348ef0a70f73d.jpg",
        "c6e26c309f8747b4b23e07f2caebfe19.jpg",
        "724278d940704093a26a2b66999ba6e5.jpg",
    ],
    "Parasect": ["00000109.jpg", "a2c42db1d4434c4da4d63f8c088891da.jpg"],
    "Persian": [
        "284ee5ec765c403f8f0d3f69af7d9dd3.jpg",
        "50ba74897f774d23a6acacf34d331c88.jpg",
        "af887faeb0694e4db2be38625970250e.jpg",
        "74558e7270f549448f99ef750898d4af.jpg",
    ],
    "Pidgeot": [
        "1a6bd2eda25c4514b727a0ef555c02d5.jpg",
        "00000008.png",
        "0062468234d74584ae2603c7b2159830.jpg",
        "d203cb0958144a8ead60b7ce7077cd03.jpg",
    ],
    "Pidgeotto": [
        "1a04d932099d4dad942c4952f5f98d88.jpg",
        "5ff3c9d934a44a8fb9c1488717de1d37.jpg",
        "00000039.jpg",
        "00000094.gif",
        "95b5c0653e3b42eeadf5184b44fffb78.jpg",
        "00000116.jpg",
        "00000140.png",
        "00000158.jpg",
        "00000160.png",
        "6548c0a6f0cc431f8ea9f0a26f7d50da.jpg",
        "25306602dca2438397dbbd464f355702.jpg",
        "70974662f4c74ab8b3d55e36dee4fd77.jpg",
        "91795253f23b46d99980aaadd9502ff7.jpg",
        "bdd8f7f65b6540e192798ea36331fd68.jpg",
        "b65acf4373db4a678a71ecccfac83df9.jpg",
        "d73ccd282f4f4a608635db5cfcb36e3d.jpg",
        "d74e0e1159254b4ba57b833510bb6c69.jpg",
        "f1bc5d2a66b14ef39dd9d7501474d3c4.jpg",
    ],
    "Pidgey": [
        "00000010.jpg",
        "00000044.jpg",
        "00000089.png",
        "00000126.jpg",
        "00000091.jpg",
        "00000127.jpg",
        "00000177.png",
        "00000188.gif",
        "00000205.gif",
        "00000207.jpg",
        "00000210.jpg",
        "00000211.jpg",
        "710fd270480c46cf82c5daf4fc325d99.jpg",
        "241b704930624ff4bb508c77d598f75c.jpg",
        "00000225.png",
        "4019e7fca5db44f8b8bbc67ca1118920.jpg",
        "c0a6916a9fa94688b2712cc12e91df1c.jpg",
        "d5e5287acb6b4cdbaf49eebbb10ce2ec.jpg",
        "e6a4a2a20c37442c8c17e033d5a3eefc.jpg",
        "e7cc09b645214cfab813610942684ffd.jpg",
        "e5998f1157b147b6ba01ff0525843f9f.jpg",
        "eeac38b11fec44caae3208b448011e0e.jpg",
        "f69d6fb9b32943e895fbeceb2898d7fe.jpg",
    ],
    "Pikachu": [
        "053ebc7e14ad4419bfe46f8b27253214.jpg",
        "00000094.jpg",
        "00000130.jpg",
        "00000142.jpg",
        "00000158.png",
        "00000170.jpg",
        "00000177.png",
        "00000182.jpg",
        "00000184.jpg",
        "00000191.jpg",
        "00000189.jpg",
        "5139f5a857fa4f908926080fb1009a80.jpg",
        "746c0c9e6b6c43a4a4b4fd413a182dae.jpg",
    ],
    "Pinsir": ["74cdae96a1f34de9bc5399270490ab25.jpg", "00000091.jpg"],
    "Primeape": [
        "5c06f13f6f2542e6aac224c7bfdb62ff.jpg",
        "4106913f5eb448d7a4351fc4dbf44d97.jpg",
        "acab654981ca4d589be1d56572ca9b8a.jpg",
        "b1887588235d49c9bb7090406cb64eaf.jpeg",
        "ec87e20d376b40faa4284decfbd1af1a.jpg",
        "ef0e8c7a8a8a47c18467bab7da5ebb35.jpg",
    ],
    "Psyduck": ["e21b7c660d9a47d2a15d44062075833e.jpg"],
    "Raichu": ["8cfe18ccc3d64cb3a58340576388fcf6.jpg", "b2518ee585c44042a47fad1211c8f6a9.jpg"],
    "Rattata": ["a95f4d0006844b1095fc6791d8f25478.jpg"],
    "Rhyhorn": [
        "4bc229ab9b2e4849a828decf9bafcd04.jpg",
        "00000023.png",
        "00000052.png",
        "44ac2f8c6e834ed3a08efa99246d7a90.jpg",
        "00000103.jpg",
        "00000086.jpg",
        "9445bdd31f574986a48eb5c1c5cc1174.jpg",
        "7679991dbdeb4fb1937272f31c6cb00d.jpg",
        "a6eb19523dd442649bac262989f23de3.jpg",
        "a6ac27e194ba4dc69d3c21e1d569adc9.jpg",
        "7148fcc7111840ebb31275a55f7f5b77.jpg",
        "3296cfb2ab7446059224fe76a878d8a3.jpg",
        "b7beb309ffc84330be87649abb8320d3.jpg",
        "b497ca45e5894a4d9471e1cece975c5d.jpg",
        "f43e06a30ab34b9f9b307081ac5815e9.jpg",
        "d0f64508493541e7b3b01871afa7bb89.jpg",
        "cfbbe229124a4b32a5c07d5d273aa8f8.jpg",
        "c520c1d78867437495ed6f04b8a6de73.jpg",
    ],
    "Sandshrew": [
        "1d8829feacc34223972589460fb64b0c.jpg",
        "f2e37e41918344ebb1b7661e83fdbecd.jpg",
        "1fc0f99cc95240138620f2b8a1ba86ec.jpg",
        "a6ff19772460426898e76517f9b08edf.jpg",
    ],
    "Seadra": [
        "1b139b8ee1f04cdf83d6bc9eb5317129.jpg",
        "1be20b2918d14cbcb7a24faa3fb447b7.jpg",
        "6c7dfb5120ab4c2fbb9b9c4a9eee5ac3.jpg",
        "e11807551c6143cc8de8ec32d35e67df.jpg",
        "fdf8bd42d6a44d088d99a6fd69b0be82.jpg",
        "db3214b1501148f59a0c83a86aade499.jpg",
        "Seadra28.jpg",
    ],
    "Seel": [
        "2ee695fb27c94e9faedc3c2b3eda757b.jpg",
        "00000055.jpg",
        "ad58986c4d1b4cf48c84152e2b8b97f2.jpg",
        "fbc066c15aff49d098065c4fe2768e75.jpg",
    ],
    "Shellder": [
        "9fc301b3b3c0455baf921553004fca94-1.jpg",
        "9fc301b3b3c0455baf921553004fca94-4.jpg",
        "9fc301b3b3c0455baf921553004fca94-5.jpg",
        "9fc301b3b3c0455baf921553004fca94-6.jpg",
        "9fc301b3b3c0455baf921553004fca94-7.jpg",
        "9fc301b3b3c0455baf921553004fca94-8.jpg",
        "9fc301b3b3c0455baf921553004fca94-6.jpg",
        "9fc301b3b3c0455baf921553004fca94-7.jpg",
        "9fc301b3b3c0455baf921553004fca94-8.jpg",
        "9fc301b3b3c0455baf921553004fca94-26.jpg",
        "9fc301b3b3c0455baf921553004fca94-27.jpg",
        "9fc301b3b3c0455baf921553004fca94-28.jpg",
        "9fc301b3b3c0455baf921553004fca94-29.jpg",
        "9fc301b3b3c0455baf921553004fca94-30.jpg",
        "9fc301b3b3c0455baf921553004fca94-31.jpg",
        "9fc301b3b3c0455baf921553004fca94-37.jpg",
        "9fc301b3b3c0455baf921553004fca94-37.jpg",
        "9fc301b3b3c0455baf921553004fca94-39.jpg",
        "9fc301b3b3c0455baf921553004fca94-41.jpg",
        "9fc301b3b3c0455baf921553004fca94-42.jpg",
        "9fc301b3b3c0455baf921553004fca94-43.jpg",
        "9fc301b3b3c0455baf921553004fca94-44.jpg",
        "9fc301b3b3c0455baf921553004fca94-42.jpg",
        "9fc301b3b3c0455baf921553004fca94-43.jpg",
        "9fc301b3b3c0455baf921553004fca94-44.jpg",
        "de281bae056e4f73b2110b7b2802df21.jpg",
    ],
    "Slowbro": [
        "11a6092cf48d47f89231c63f22765c87.jpg",
        "c5adfc374b424046a1bdfa3d680239f6.jpg",
        "cf8e56d4c1df4d4a8489b9199fa31a58.jpg",
        "0e8a4df27259437b97e471637ca4612b.jpg",
        "6c7090a2b2404c3abe0eff15f2f875c0.jpg",
    ],
    "Slowpoke": [
        "2af482f2b7b84ab79ab7d1b221e65c46.jpg",
        "3787f22487524e55bed8f51db83ac7a7.jpg",
        "00000172.jpg",
        "dd1e7b62575d4906863b03133587f77d.jpg",
        "Slowpoke50.jpg",
    ],
    "Snorlax": [
        "00000024.png",
        "44ffe8bfa335424e9297e63f07d617e4.jpg",
        "00000066.gif",
        "97d6635c2a5f4ed088ae61030651d23f.jpg",
        "a7aaa89a22144137b65cf4bec31e78d2.jpg",
    ],
    "Spearow": [
        "5f6b982003f040a084af1faee84da53e.jpg",
        "15a87e00379545db85b163b75ac69c19.jpg",
        "59ceac7b16ae4ffea7124acafd458ebc.jpg",
        "00000110.jpg",
    ],
    "Squirtle": [
        "00000074.jpg",
        "00000080.jpg",
        "00000117.png",
        "00000129.png",
        "00000134.jpg",
        "00000145.jpg",
        "00000164.jpg",
        "00000179.png",
        "00000174.jpg",
        "00000173.jpg",
        "00000182.jpg",
        "00000190.jpg",
        "00000188.jpg",
        "00000200.jpg",
        "00000201.png",
        "00000203.jpg",
        "00000212.jpg",
        "00000208.png",
        "00000214.jpg",
        "00000217.jpg",
        "00000228.jpg",
        "00000224.jpg",
        "cd77275f4b434312b374d211993bc1dd.jpg",
    ],
    "Staryu": ["3f3fba87a098425bbc63111ced541a94.jpg"],
    "Tentacool": [
        "82a0660589bb47c2bd7cf4ba333148c8.jpg",
        "79b9cedc0c2d46a39c5bbfa99c6afde6.jpg",
        "00000044.jpg",
    ],
    "Venusaur": [
        "00000069.png147437d570714abd81fc60973267cc33.jpg",
        "baf1ff6940a54fde9c1e890c16c58740.jpg",
        "aa48554103e746d5a8d40db24420fdcb.jpg",
        "b6e808182dfb4fdcb149a348f259b21c.jpg",
        "caca9ce6f70d4af3895096b1730ab44c.jpg",
        "baf1ff6940a54fde9c1e890c16c58740.jpg",
    ],
    "Victreebel": [
        "00000061.png",
        "00000031.png",
        "00000039.png",
        "00000060.jpg",
        "00000112.jpg",
        "00000148.png",
        "d15e684aa1ab4d90ab0e8fc300cc47d8.jpg",
        "d386162d816b4fbd87d1019720230c79.jpg",
        "db3f45e13f4e49b18092023f98522af8.jpg",
    ],
    "Voltorb": ["00000066.png", "00000127.png", "00000200.jpg"],
    "Vulpix": ["f3d32f16c7684b2eb3d91f5d943bcbcb.jpg"],
    "Wartortle": ["88fdb8c5a5aa4f30b5709165537c45fc.jpg", "00000163.png", "00000143.jpg"],
    "Wigglytuff": ["6486ed280346401ea1fe95346bb29907.jpg"],
    "Zapdos": ["00000071.jpg"],
    "MrMime": [
        # Add filenames for MrMime here if applicable
    ],
}


def download_kaggle_dataset(dataset_name, download_path):
    from kaggle.api.kaggle_api_extended import KaggleApi

    """
    Download dataset from Kaggle.

    Args:
        dataset_name (str): Kaggle dataset name in 'owner/dataset' format.
        download_path (str): Path to download the dataset.

    """
    api = KaggleApi()
    api.authenticate()
    logging.info(f"Downloading dataset {dataset_name} to {download_path}...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    logging.info("Download completed.")


def organize_dataset(raw_path, processed_path, skip_folders=None):
    """
    Organize the dataset into processed directories.

    Args:
        raw_path (str): Path where raw dataset is downloaded.
        processed_path (str): Path to save processed dataset.
        skip_folders (list[str], optional): List of folders to skip.

    """
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)
    processed_path.mkdir(parents=True, exist_ok=True)

    for folder in raw_path.iterdir():
        if folder.is_dir():
            label = folder.name
            if label == "Mr. Mime":
                label = "MrMime"
            if skip_folders and label in skip_folders:
                continue
            label_dir = processed_path / label
            label_dir.mkdir(parents=True, exist_ok=True)
            for img in folder.glob("*.*"):
                if img.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                    if img.name in ignore_files.get(label, []):
                        print(f"Ignoring {img.name}")
                        continue
                    shutil.move(str(img), label_dir / img.name)
    logging.info("Dataset organized.")


def main(dataset_name=None, raw_path=None, processed_path=None, extra_path=None, skip_folders=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load environment variables
    load_dotenv(".env")

    # Create directories if they don't exist
    Path(raw_path).mkdir(parents=True, exist_ok=True)
    Path(processed_path).mkdir(parents=True, exist_ok=True)

    # Download dataset
    download_kaggle_dataset(dataset_name, raw_path)

    raw_dataset_path = raw_path + "/" + extra_path
    # Organize dataset
    organize_dataset(raw_dataset_path, processed_path, skip_folders)

    logging.info("Dataset downloaded and organized.")


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict[str, Any]: Configuration parameters.

    """
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config("src/config/config.yaml")
    use_both_datasets = config["data"]["use_both_datasets"]
    if use_both_datasets:
        main(
            dataset_name="bhawks/pokemon-generation-one-22k",
            raw_path="data/raw",
            processed_path="data/processed",
            extra_path="PokemonData",
        )
        main(
            dataset_name="thedagger/pokemon-generation-one",
            raw_path="data/raw",
            processed_path="data/processed",
            extra_path="dataset",
            skip_folders=[
                "Nidorina",
                "Nidorino",
            ],  # Skip these folders since nidoran-f and nidoran-m are accidentally in these folders
        )
        main(
            dataset_name="mikoajkolman/pokemon-images-first-generation17000-files",
            raw_path="data/raw",
            processed_path="data/processed",
            extra_path="pokemon",
            skip_folders=[
                "Nidorina",
                "Nidorino",
            ],  # Skip these folders since nidoran-f and nidoran-m are accidentally in these folders
        )
