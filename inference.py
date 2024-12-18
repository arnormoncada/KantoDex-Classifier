import os
import time
from collections import Counter, deque

import cv2
import numpy as np
import pypokedex
import torch

from src.models.model import KantoDexClassifier

pokemon_classes = [
    "Abra",
    "Aerodactyl",
    "Alakazam",
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
    "Nidora-f",
    "Nidoran-m",
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
pokemon_classes.sort()


def load_pokemon_images(image_dir):
    images = {}
    for cls in pokemon_classes:
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = os.path.join(image_dir, cls + ext)
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    images[cls] = img
                    break
    return images


def load_pokeball_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return img


def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """
    Overlay img_overlay onto img at (x, y) with alpha_mask.
    alpha_mask must be same size as img_overlay and values are [0.0, 1.0].
    """
    y1, y2 = y, y + img_overlay.shape[0]
    x1, x2 = x, x + img_overlay.shape[1]

    # Clip if outside bounds
    if x1 >= img.shape[1] or y1 >= img.shape[0]:
        return img
    if x2 <= 0 or y2 <= 0:
        return img

    if x1 < 0:
        img_overlay = img_overlay[:, -x1:]
        alpha_mask = alpha_mask[:, -x1:]
        x1 = 0
    if y1 < 0:
        img_overlay = img_overlay[-y1:, :]
        alpha_mask = alpha_mask[-y1:, :]
        y1 = 0
    if x2 > img.shape[1]:
        img_overlay = img_overlay[:, : img.shape[1] - x1]
        alpha_mask = alpha_mask[:, : img.shape[1] - x1]
        x2 = img.shape[1]
    if y2 > img.shape[0]:
        img_overlay = img_overlay[: img.shape[0] - y1, :]
        alpha_mask = alpha_mask[: img.shape[0] - y1, :]
        y2 = img.shape[0]

    blended = (alpha_mask * img_overlay[..., :3] + (1 - alpha_mask) * img[y1:y2, x1:x2, :3]).astype(
        np.uint8
    )
    img[y1:y2, x1:x2, :3] = blended
    return img


def wrap_text(img, text, org, font, font_scale, color, thickness, max_width=380, line_height=30):
    words = text.split(" ")
    current_line = ""
    lines = []
    for w in words:
        test_line = current_line + w + " "
        (text_width, _) = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        if text_width > max_width:
            lines.append(current_line.strip())
            current_line = w + " "
        else:
            current_line = test_line
    if current_line.strip():
        lines.append(current_line.strip())

    x, y = org
    for line in lines:
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness)
        y += line_height
    return y  # Return the ending y position


def rotate_image(image, angle):
    """
    Rotate an RGBA image around its center. Returns rotated RGBA image.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Separate alpha if present
    if image.shape[2] == 4:
        bgr = image[..., :3]
        alpha = image[..., 3]

        # Rotate BGR
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_bgr = cv2.warpAffine(
            bgr,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        # Rotate alpha
        rotated_alpha = cv2.warpAffine(
            alpha,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        rotated = np.dstack((rotated_bgr, rotated_alpha))
    else:
        # no alpha channel
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    return rotated


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KantoDexClassifier(model_name="custom", num_classes=151, custom_config={})
    checkpoint = torch.load("models/checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    if device.type == "cuda":
        model.half()

    torch.backends.cudnn.benchmark = True

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)

    pokemon_images = load_pokemon_images("images")
    pokeball_img = load_pokeball_image("pokeball.png")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    prev_time = time.time()

    input_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)
    if device.type == "cuda":
        input_tensor = input_tensor.half()

    cv2.namedWindow("Pokémon Classifier", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pokémon Classifier", 1400, 750)

    predictions_deque = deque(maxlen=10)
    stable_prediction = "Unknown"
    last_stable_prediction = None

    fade_in_value = 0.0
    fade_speed = 0.05

    # Rotating pokeball angle
    pokeball_angle = 0.0

    # To store fetched Pokemon data
    current_pokemon_data = None

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

            frame_tensor = torch.from_numpy(frame_resized).to(device)
            frame_tensor = frame_tensor.half() if device.type == "cuda" else frame_tensor.float()

            frame_tensor = frame_tensor.permute(2, 0, 1) / 255.0
            frame_tensor = (frame_tensor - mean[..., None, None]) / std[..., None, None]
            input_tensor.copy_(frame_tensor.unsqueeze(0))

            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_class = torch.max(probs, dim=1)

            predicted_class_name = pokemon_classes[top_class.item()]
            confidence = top_prob.item() * 100.0

            if confidence < 35:
                predicted_class_name = "Unknown"
                confidence = 100.0 - confidence

            # Update stable prediction
            predictions_deque.append(predicted_class_name)
            most_common = Counter(predictions_deque).most_common(1)[0][0]
            if most_common != stable_prediction:
                stable_prediction = most_common
                fade_in_value = 0.0
                current_pokemon_data = None  # reset data so we fetch again

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            display_frame = cv2.resize(frame, (1000, 750))
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (1000, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)

            # HUD info
            text_color = (255, 255, 255)
            cv2.putText(
                display_frame,
                f"Prediction: {stable_prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_DUPLEX,
                1.2,
                text_color,
                2,
            )
            cv2.putText(
                display_frame,
                f"Confidence: {confidence:.2f}%",
                (20, 90),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                text_color,
                2,
            )
            cv2.putText(
                display_frame,
                f"FPS: {fps:.2f}",
                (20, 130),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                text_color,
                2,
            )

            # Confidence bar
            bar_x, bar_y, bar_w, bar_h = 700, 60, 200, 20
            cv2.rectangle(
                display_frame,
                (bar_x, bar_y),
                (bar_x + bar_w, bar_y + bar_h),
                (255, 255, 255),
                2,
            )
            fill_w = int((confidence / 100.0) * (bar_w - 4))
            cv2.rectangle(
                display_frame,
                (bar_x + 2, bar_y + 2),
                (bar_x + 2 + fill_w, bar_y + bar_h - 2),
                (0, 255, 0),
                -1,
            )

            # Fade in Pokémon image if stable
            if stable_prediction == last_stable_prediction and fade_in_value < 1.0:
                fade_in_value = min(fade_in_value + fade_speed, 1.0)
            elif stable_prediction != last_stable_prediction:
                fade_in_value = 0.0

            # Rotate and show Poké Ball if known Pokémon
            if stable_prediction != "Unknown" and pokeball_img is not None:
                pokeball_angle += 5.0  # Increase angle to spin faster or slower
                original_h, original_w = pokeball_img.shape[:2]
                scale_factor = 2 * bar_h / float(original_h)
                new_width = int(original_w * scale_factor)
                new_height = int(original_h * scale_factor)
                resized_pokeball = cv2.resize(
                    pokeball_img,
                    (new_width, new_height),
                    interpolation=cv2.INTER_LINEAR,
                )

                # Rotate the pokeball
                rotated_pokeball = rotate_image(resized_pokeball, pokeball_angle)

                pokeball_x = bar_x + bar_w + 10
                pokeball_y = bar_y + (bar_h // 2) - (new_height // 2)

                if rotated_pokeball.shape[2] == 4:
                    alpha_channel = (rotated_pokeball[..., 3] / 255.0)[..., None]
                    display_frame = overlay_image_alpha(
                        display_frame,
                        rotated_pokeball,
                        pokeball_x,
                        pokeball_y,
                        alpha_channel,
                    )
                else:
                    end_x = min(pokeball_x + new_width, display_frame.shape[1])
                    end_y = min(pokeball_y + new_height, display_frame.shape[0])
                    display_frame[pokeball_y:end_y, pokeball_x:end_x] = rotated_pokeball[
                        : end_y - pokeball_y, : end_x - pokeball_x, :3
                    ]

            # Display Pokémon image if available and not unknown
            if stable_prediction in pokemon_images and stable_prediction != "Unknown":
                poke_img = pokemon_images[stable_prediction]
                poke_img_display = cv2.resize(poke_img, (200, 200), interpolation=cv2.INTER_LINEAR)
                if poke_img_display.shape[2] == 4:
                    alpha_channel = (poke_img_display[..., 3] / 255.0) * fade_in_value
                    alpha_channel = alpha_channel[..., None]
                    display_frame = overlay_image_alpha(
                        display_frame, poke_img_display, 780, 500, alpha_channel
                    )
                else:
                    roi = display_frame[500:700, 780:980]
                    blended = cv2.addWeighted(
                        poke_img_display, fade_in_value, roi, 1 - fade_in_value, 0
                    )
                    display_frame[500:700, 780:980] = blended

            # Fetch Pokémon data if stable and known
            if stable_prediction != "Unknown" and (current_pokemon_data is None):
                try:
                    current_pokemon_data = pypokedex.get(name=stable_prediction.lower())
                except:
                    current_pokemon_data = None

            # Create info panel (black background)
            info_panel = np.zeros((750, 400, 3), dtype=np.uint8)

            if current_pokemon_data is not None:
                info_color = (255, 255, 255)
                line_y = 30
                line_height = 30
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.7
                thickness = 1

                # Name and Dex
                cv2.putText(
                    info_panel,
                    f"Name: {current_pokemon_data.name.capitalize()}",
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                )
                line_y += line_height
                cv2.putText(
                    info_panel,
                    f"Dex: {current_pokemon_data.dex}",
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                )
                line_y += line_height

                # Types
                types_str = ", ".join(t.capitalize() for t in current_pokemon_data.types)
                cv2.putText(
                    info_panel,
                    f"Types: {types_str}",
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                )
                line_y += line_height

                # Abilities
                abilities_str = ", ".join(
                    f"{a.name}{' (Hidden)' if a.is_hidden else ''}"
                    for a in current_pokemon_data.abilities
                )
                cv2.putText(
                    info_panel,
                    "Abilities:",
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                )
                line_y += line_height
                line_y = wrap_text(
                    info_panel,
                    abilities_str,
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                    max_width=380,
                    line_height=line_height,
                )
                line_y += 10  # extra space after abilities

                # Base stats
                stats = current_pokemon_data.base_stats
                cv2.putText(
                    info_panel,
                    f"HP: {stats.hp}",
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                )
                line_y += line_height
                cv2.putText(
                    info_panel,
                    f"Attack: {stats.attack}",
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                )
                line_y += line_height
                cv2.putText(
                    info_panel,
                    f"Defense: {stats.defense}",
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                )
                line_y += line_height
                cv2.putText(
                    info_panel,
                    f"Sp. Atk: {stats.sp_atk}",
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                )
                line_y += line_height
                cv2.putText(
                    info_panel,
                    f"Sp. Def: {stats.sp_def}",
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                )
                line_y += line_height
                cv2.putText(
                    info_panel,
                    f"Speed: {stats.speed}",
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                )
                # Weight, height,
                line_y += line_height
                cv2.putText(
                    info_panel,
                    f"Weight: {current_pokemon_data.weight/10} kg",
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                )
                line_y += line_height
                cv2.putText(
                    info_panel,
                    f"Height: {current_pokemon_data.height/10} m",
                    (10, line_y),
                    font,
                    font_scale,
                    info_color,
                    thickness,
                )

                # final
                line_y += line_height

            last_stable_prediction = stable_prediction

            # Combine display_frame and info_panel into one window
            combined_frame = np.hstack((display_frame, info_panel))
            cv2.rectangle(
                combined_frame,
                (0, 0),
                (combined_frame.shape[1], combined_frame.shape[0]),
                (255, 215, 0),
                4,
            )
            cv2.imshow("Pokémon Classifier", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
