"""Combine TinyNet trainer and Moons explorer in one Streamlit app."""
import streamlit as st
import tinynet
import moons_streamlit

APPS = {
    "TinyNet Trainer": tinynet.main,
    "Moons Explorer": moons_streamlit.main,
}

def main() -> None:
    st.sidebar.title("TinyNetLab")
    app_names = list(APPS.keys())
    choice = st.sidebar.radio("Select App", app_names)
    APPS[choice]()

if __name__ == "__main__":
    main()
