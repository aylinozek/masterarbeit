import seaborn as sns

# ------------------------
# Monochromatic palettes
# ------------------------
# gender stratified colors
color_palette={'female':'#9ef0f0', 'male':'#8a3ffc'}
color="tab:blue" #single color

# Purple shades (5 colors)
purple_palette = ["#f6f2ff",'#e8daff', "#d4bbff", "#a56eff", "#6929c4", "#31135e",'#1c0f30']

# Teal shades (5 colors)
teal_palette = ["#d9fbfb",'#9ef0f0', "#3ddbd9", "#009d9a", "#005d5d", "#022b30", '#081a1c']

# Blue shades (5 colors)
blue_palette = ["#edf5ff",'#d0e2ff', "#a6c8ff", "#4589ff", "#0043ce", "#001d6c", '#001141']

# ------------------------
# Polychromatic palette
# ------------------------
mixed_palette = [
    "#8a3ffc",  # purple
    "#3ddbd9",  # teal lighter
    "#0f62fe",  # blue
    "#d4bbff",  # purple lighter
    "#007d79",  # teal
    "#a6c8ff",  # blue lighter
    "#012749"   # cyan dark
]

def set_palette(name="mixed"):
    """
    Set global Seaborn palette by name:
    - "purple", "teal", "blue", "mixed"
    """
    if name == "purple":
        sns.set_palette(purple_palette)
    elif name == "teal":
        sns.set_palette(teal_palette)
    elif name == "blue":
        sns.set_palette(blue_palette)
    elif name == "mixed":
        sns.set_palette(mixed_palette)
    else:
        raise ValueError("Palette name not recognized. Use 'purple', 'teal', 'blue', or 'mixed'.")