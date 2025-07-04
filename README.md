# deteksi-objek-computer-vision

install dulu : 

```
pip install streamlit ultralytics opencv-python-headless matplotlib pillow
```

Lalu jalankan di streamlit : 
```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Title
st.title("🔍 Deteksi Objek Menggunakan YOLOv8")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar dengan PIL
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Tampilkan gambar asli
    st.image(img_array, caption='Gambar yang Diupload', use_container_width=True)

    # Load model YOLOv8
    with st.spinner("Memuat model dan mendeteksi objek..."):
        model = YOLO("yolov8n.pt")  # Pastikan model ini sudah diunduh

        # Simpan gambar ke sementara untuk deteksi
        temp_path = "temp.jpg"
        image.save(temp_path)

        # Deteksi objek
        results = model(temp_path)

        # Ambil hasil anotasi
        annotated_img = results[0].plot()

        # Tampilkan hasil
        st.image(annotated_img, caption='Hasil Deteksi Objek', use_container_width=True)

        # Print hasil deteksi
        st.subheader("📋 Detail Deteksi:")
        for result in results[0].boxes:
            label = model.names[int(result.cls)]
            confidence = result.conf.item()
            coordinates = result.xyxy.cpu().numpy()
            st.write(f"**Label**: {label}, **Confidence**: {confidence:.2f}")
            st.write(f"Koordinat: {coordinates}")


```
Hasil : 
![objek deteksi](https://github.com/user-attachments/assets/5eaa353b-5f85-4ae5-ba37-dab3949fa63f)
![hasil deteksi](https://github.com/user-attachments/assets/910fc238-3b66-492e-a17e-2a33f48772e7)
