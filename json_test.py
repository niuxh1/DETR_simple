import os

from PIL import Image
from PIL import ImageDraw
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

json_path = 'coco_data/annotations/instances_val2017.json'

coco = COCO(annotation_file=json_path)

ids = list(sorted(coco.imgs.keys()))
print(ids)

coco_classes = dict([(v['id'], v['name']) for k, v in coco.cats.items()])

# print(coco_classes.items())

for img_id in ids[:3]:

    ann_id = coco.getAnnIds(imgIds=img_id)

    targets = coco.loadAnns(ann_id)

    path = coco.loadImgs(img_id)[0]['file_name']

    image = Image.open(os.path.join('coco_data/val2017', path)).convert('RGB')
    draw = ImageDraw.Draw(image)

    for target in targets:
        x, y, w, h = target['bbox']
        x1, y1, x2, y2 = x, y, int(x + w),int( y + h)
        draw.rectangle((x1, y1, x2, y2))
        draw.text((x1, y1), coco_classes[target['category_id']], fill='red')

    plt.imshow(image)
    plt.show()
