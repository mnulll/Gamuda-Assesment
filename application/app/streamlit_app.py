# Streamlit frontend for the defect segmentation app.
# Uploads an image, sends it to the API, shows the result.

import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Defect Segmentation")
st.write("Upload an image to find cracks, spalling, corrosion, and deformation.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, caption="Input image")

    if st.button("Run inference"):
        # Send the image to the API.
        try:
            resp = requests.post(
                API_URL + "/predict",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            st.error("Could not reach the API: " + str(e))
            st.stop()

        # Show the quality result.
        quality = data["quality"]
        if quality["passed"]:
            st.success("Quality OK")
        else:
            st.warning("Quality warning (inference still ran):")
            for w in quality["warnings"]:
                st.write("- " + w)

        st.write("Brightness:", quality["brightness"])
        st.write("Focus score:", quality["blur_score"])

        # Show the annotated image.
        image_bytes = base64.b64decode(data["annotated_image"])
        result_image = Image.open(io.BytesIO(image_bytes))
        st.image(result_image, caption="Result")

        st.download_button(
            "Download result",
            data=image_bytes,
            file_name="result.png",
            mime="image/png",
        )

        # Show the detections.
        detections = data["detections"]
        st.write("Defects found:", len(detections))
        if detections:
            st.table(detections)
        else:
            st.write("No defects detected.")
