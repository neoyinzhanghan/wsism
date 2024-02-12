import openslide

slide_path = '/media/hdd2/PBs/H23-84;S12;MSKT - 2023-06-15 09.11.44.ndpi'

# open the slide and print its level 0 dimensions
slide = openslide.OpenSlide(slide_path)
print(slide.level_dimensions[0])