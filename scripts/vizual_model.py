from collections import defaultdict
from PIL import ImageFont, Image, ImageDraw
import visualkeras
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers

DIRECTRORY = 'D:/python/safevision/SafeVision/snoring detection by Egor\images/final_model_architecture/'

def vizual_model():
    color_map = defaultdict(dict)
    color_map[layers.Conv2D]['fill'] = 'orange'
    color_map[layers.Dense]['fill'] = 'green'
    color_map[layers.Flatten]['fill'] = 'teal'

    font = ImageFont.truetype("arial.ttf", 32)

    visualkeras.layered_view(
        model, 
        to_file=DIRECTRORY + 'Arch.png', 
        min_xy=100, 
        min_z=100, 
        scale_xy=100, 
        scale_z=100, 
        one_dim_orientation='x', 
        color_map=color_map,
        font=font
    )

    legend_fig, ax = plt.subplots(figsize=(3, 2))
    ax.axis("off")
    legend_elements = [
        plt.Line2D([0], [0], color='orange', lw=4, label='Conv2D Layer'),
        plt.Line2D([0], [0], color='green', lw=4, label='Dense Layer'),
        plt.Line2D([0], [0], color='teal', lw=4, label='Flatten Layer'),
    ]
    ax.legend(handles=legend_elements, loc="center", fontsize="large")
    plt.savefig(DIRECTRORY + "legend.png", bbox_inches="tight")

    model_image = Image.open(DIRECTRORY + "Arch.png")
    legend_image = Image.open(DIRECTRORY + "legend.png")
    combined_width = model_image.width + legend_image.width + 10
    combined_height = max(model_image.height, legend_image.height)
    combined_image = Image.new("RGB", (combined_width, combined_height), "white")
    combined_image.paste(model_image, (0, 0))
    combined_image.paste(legend_image, (model_image.width + 10, 0))
    combined_image.save(DIRECTRORY + "Arch_with_legend.png")