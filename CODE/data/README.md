# Dataset phan loai rac

Bo nhan nay phu hop voi bai toan transfer learning bang MobileNetV2. Anh duoc sap xep theo ten lop; ten thu muc phai trung voi cot `label_en` trong `classes.csv`.

## Cau truc du lieu

```text
data/
  classes.csv
  dataset.csv
  images/
    train/
      cardboard/
      glass/
      metal/
      paper/
      plastic/
      trash/
    val/
      cardboard/
      glass/
      metal/
      paper/
      plastic/
      trash/
    test/
      cardboard/
      glass/
      metal/
      paper/
      plastic/
      trash/
```

`dataset.csv` la manifest mau gom duong dan anh, nhan lop, tap du lieu va huong dan xu ly. Cac file anh trong manifest la mau ten duong dan; can them anh JPG/PNG tu TrashNet hoac bo anh tu chup truoc khi huan luyen.

Moi anh nen la JPG/PNG, mot vat the chinh, anh sang tu nhien va khong chen nhan lop vao ten file. Khuyen nghi chia theo nguoi chup hoặc bo canh, khong chia ngau nhien cac frame lien tiep vao ca train va test.

## Nguon anh de bat dau

Co the dung TrashNet lam tap du lieu ban dau. Tap nay co 6 lop: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`. Kiem tra dieu khoan su dung cua nguon anh truoc khi phan phoi lai. Sau do bo sung anh rac trong boi canh Viet Nam de giam sai lech mien du lieu.

Muc tieu ban dau: toi thieu 300 anh/lop cho train, 60 anh/lop cho validation va 60 anh/lop cho test. Uu tien can bang lop, nhieu goc chup, kich thuoc vat the va dieu kien sang.

## Nap du lieu voi TensorFlow

```python
import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/images/train", image_size=(224, 224), batch_size=32,
    label_mode="int", seed=42
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/images/val", image_size=(224, 224), batch_size=32,
    label_mode="int", shuffle=False
)
```

Thu tu nhan cua TensorFlow duoc sap xep theo alphabet va trung voi 6 thu muc o tren. Khi hien thi ket qua, doc `classes.csv` de lay `label_vi` va `disposal_guidance`.

## Quy tac gan nhan

- `cardboard`: bia carton, hop giay day.
- `paper`: giay mong, bao, giay in.
- `glass`: vat dung chu yeu bang thuy tinh.
- `metal`: lon va vat dung chu yeu bang kim loai.
- `plastic`: vat dung chu yeu bang nhua.
- `trash`: rac con lai khong thuoc 5 lop tren.

Vat da lam tu nhieu chat lieu duoc gan theo vat lieu chiem uu the; neu khong chac chan, gan `trash` va ghi lai de xem xet bo sung nhan sau.
