"""Combine TinyNet trainer and Moons explorer in one Streamlit app."""
import streamlit as st
import tinynet
import moons_streamlit
from i18n import t

APPS = {
    "moons": moons_streamlit.main,
    "tinynet": tinynet.main,
}

def main() -> None:
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"

    st.sidebar.title(t("app_title"))
    st.sidebar.selectbox(
        t("language"),
        ["en", "zh"],
        format_func=lambda x: "English" if x == "en" else "中文",
        key="lang",
    )

    choice = st.sidebar.radio(
        t("select_app"),
        list(APPS.keys()),
        format_func=lambda x: t("moons_explorer") if x == "moons" else t("tinynet_trainer"),
    )
    APPS[choice]()

if __name__ == "__main__":
    main()
