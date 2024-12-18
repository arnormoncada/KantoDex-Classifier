import os
import shutil

# Specify your top-level directory
top_dir = "KantoDex-Classifier/data/processed/"

# Directory where images will be consolidated
images_dir = "images"

# Create the "images" directory if it doesn't exist
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Gather all subfolders (excluding the 'images' directory itself if it already exists)
folders = [
    f
    for f in os.listdir(top_dir)
    if os.path.isdir(os.path.join(top_dir, f)) and f != "images"
]

# Define a list of possible image extensions
image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

# Sort folders alphabetically
folders.sort()
pokemon_classes = [
    "Abra",
    "Aerodactyl",
    "Alakazam",
    "Alolan Sandslash",
    "Arbok",
    "Arcanine",
    "Articuno",
    "Beedrill",
    "Bellsprout",
    "Blastoise",
    "Bulbasaur",
    "Butterfree",
    "Caterpie",
    "Chansey",
    "Charizard",
    "Charmander",
    "Charmeleon",
    "Clefable",
    "Clefairy",
    "Cloyster",
    "Cubone",
    "Dewgong",
    "Diglett",
    "Ditto",
    "Dodrio",
    "Doduo",
    "Dragonair",
    "Dragonite",
    "Dratini",
    "Drowzee",
    "Dugtrio",
    "Eevee",
    "Ekans",
    "Electabuzz",
    "Electrode",
    "Exeggcute",
    "Exeggutor",
    "Farfetch'd",
    "Fearow",
    "Flareon",
    "Gastly",
    "Gengar",
    "Geodude",
    "Gloom",
    "Golbat",
    "Goldeen",
    "Golduck",
    "Golem",
    "Graveler",
    "Grimer",
    "Growlithe",
    "Gyarados",
    "Haunter",
    "Hitmonchan",
    "Hitmonlee",
    "Horsea",
    "Hypno",
    "Ivysaur",
    "Jigglypuff",
    "Jolteon",
    "Jynx",
    "Kabuto",
    "Kabutops",
    "Kadabra",
    "Kakuna",
    "Kangaskhan",
    "Kingler",
    "Koffing",
    "Krabby",
    "Lapras",
    "Lickitung",
    "Machamp",
    "Machoke",
    "Machop",
    "Magikarp",
    "Magmar",
    "Magnemite",
    "Magneton",
    "Mankey",
    "Marowak",
    "Meowth",
    "Metapod",
    "Mew",
    "Mewtwo",
    "Moltres",
    "Mr. Mime",
    "Muk",
    "Nidoking",
    "Nidoqueen",
    "Nidoran♀",
    "Nidoran♂",
    "Nidorina",
    "Nidorino",
    "Ninetales",
    "Oddish",
    "Omanyte",
    "Omastar",
    "Onix",
    "Paras",
    "Parasect",
    "Persian",
    "Pidgeot",
    "Pidgeotto",
    "Pidgey",
    "Pikachu",
    "Pinsir",
    "Poliwag",
    "Poliwhirl",
    "Poliwrath",
    "Ponyta",
    "Porygon",
    "Primeape",
    "Psyduck",
    "Raichu",
    "Rapidash",
    "Raticate",
    "Rattata",
    "Rhydon",
    "Rhyhorn",
    "Sandshrew",
    "Sandslash",
    "Scyther",
    "Seadra",
    "Seaking",
    "Seel",
    "Shellder",
    "Slowbro",
    "Slowpoke",
    "Snorlax",
    "Spearow",
    "Squirtle",
    "Starmie",
    "Staryu",
    "Tangela",
    "Tauros",
    "Tentacool",
    "Tentacruel",
    "Vaporeon",
    "Venomoth",
    "Venonat",
    "Venusaur",
    "Victreebel",
    "Vileplume",
    "Voltorb",
    "Vulpix",
    "Wartortle",
    "Weedle",
    "Weepinbell",
    "Weezing",
    "Wigglytuff",
    "Zapdos",
    "Zubat",
]

# Sort classes alphabetically
pokemon_classes.sort()

idx = 0
for folder in folders:
    folder_path = os.path.join(top_dir, folder)
    # Find the first image in the folder
    first_image = None
    for file_name in os.listdir(folder_path):
        if any(file_name.lower().endswith(ext) for ext in image_extensions):
            first_image = file_name
            break

    # If an image was found, move and rename it
    if first_image:
        src = os.path.join(folder_path, first_image)
        # Keep the same extension as the original image
        _, ext = os.path.splitext(first_image)
        # Rename the image to the class name
        dst = os.path.join(images_dir, f"{pokemon_classes[idx]}{ext}")
        dst = os.path.join(images_dir, f"{folder}{ext}")
        idx += 1
        # copy the image
        shutil.copy(src, dst)

# Print the list of folders processed
print(folders)
