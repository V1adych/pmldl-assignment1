import base64
import io
import os

import requests
import streamlit as st


API_BASE_URL = os.environ.get("API_BASE_URL", "http://api:8000")


def post_image_to_backend(image_bytes: bytes):
    files = {"image": ("upload.png", image_bytes, "image/png")}
    url = f"{API_BASE_URL}/api/v1/bin/detect_spikes"
    resp = requests.post(url, files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    st.set_page_config(page_title="Spike Detector", page_icon="🧵", layout="centered")
    st.title("Spike Detector")
    st.write("Upload an image of a thread to detect spikes.")

    with st.sidebar:
        st.markdown("**Backend**")
        st.code(API_BASE_URL)

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image_bytes = uploaded.read()
        st.image(image_bytes, caption="Input", use_container_width=True)

        if st.button("Analyze", type="primary"):
            with st.spinner("Running detection..."):
                try:
                    result = post_image_to_backend(image_bytes)
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")
                    return

            if result.get("success") != 1:
                st.warning("Detection did not succeed.")
                st.json(result)
                return

            b64_img = result.get("image")
            if b64_img:
                try:
                    img_bytes = base64.b64decode(b64_img)
                    st.image(
                        io.BytesIO(img_bytes),
                        caption="Detections",
                        use_container_width=True,
                    )
                except Exception:
                    st.error("Failed to decode result image.")

            st.subheader("Detections")
            st.json({"spikes": result.get("spikes")})


if __name__ == "__main__":
    main()
