from PIL import Image

from util import list_all_images

def convert(src: str, dest: str):
    im = Image.open(src)
    rgb_im = im.convert('RGB')
    rgb_im.save(dest)

if __name__ == '__main__':
    ls: list[str] = list_all_images(root='real_images/imagenet_64', image_exts=['png'])
    for i, elem in enumerate(ls):
        convert(src=elem, dest=elem.replace('imagenet_64', 'imagenet_64_jpg').replace('png', 'jpg'))