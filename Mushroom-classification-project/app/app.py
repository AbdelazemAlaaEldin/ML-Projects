import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from preprocessing import MushroomPreprocessor


MODEL_PATH = PROJECT_ROOT / "models" / "mushroom_categorical_nb_pipeline.pkl"

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="centered"
)


st.title("🍄 Mushroom Classification")

st.write(
    "Classify a mushroom based on its physical characteristics."
)

st.warning(
    "⚠️ Educational project only. "
    "This model must NOT be used to determine whether a real mushroom "
    "is safe to eat."
)

st.subheader("Mushroom Features")


cap_shape = st.selectbox(
    "Cap Shape",
    ["b", "c", "f", "k", "s", "x"]
)

cap_surface = st.selectbox(
    "Cap Surface",
    ["f", "g", "s", "y"]
)

cap_color = st.selectbox(
    "Cap Color",
    ["b", "c", "e", "g", "n", "p", "r", "u", "w", "y"]
)

bruises = st.selectbox(
    "Bruises",
    ["t", "f"]
)

odor = st.selectbox(
    "Odor",
    ["a", "c", "f", "l", "m", "n", "p", "s", "y"]
)

gill_attachment = st.selectbox(
    "Gill Attachment",
    ["a", "d", "f", "n"]
)

gill_spacing = st.selectbox(
    "Gill Spacing",
    ["c", "w"]
)

gill_size = st.selectbox(
    "Gill Size",
    ["b", "n"]
)

gill_color = st.selectbox(
    "Gill Color",
    ["b", "e", "g", "h", "k", "n", "o", "p", "r", "u", "w", "y"]
)

stalk_shape = st.selectbox(
    "Stalk Shape",
    ["e", "t"]
)

stalk_root = st.selectbox(
    "Stalk Root",
    ["b", "c", "e", "r", "Unknown"]
)

stalk_surface_above = st.selectbox(
    "Stalk Surface Above Ring",
    ["f", "k", "s", "y"]
)

stalk_surface_below = st.selectbox(
    "Stalk Surface Below Ring",
    ["f", "k", "s", "y"]
)

stalk_color_above = st.selectbox(
    "Stalk Color Above Ring",
    ["b", "c", "e", "g", "n", "o", "p", "w", "y"]
)

stalk_color_below = st.selectbox(
    "Stalk Color Below Ring",
    ["b", "c", "e", "g", "n", "o", "p", "w", "y"]
)

veil_color = st.selectbox(
    "Veil Color",
    ["n", "o", "w", "y"]
)

ring_number = st.selectbox(
    "Ring Number",
    ["n", "o", "t"]
)

ring_type = st.selectbox(
    "Ring Type",
    ["e", "f", "l", "n", "p"]
)

spore_print_color = st.selectbox(
    "Spore Print Color",
    ["b", "h", "k", "n", "o", "r", "u", "w", "y"]
)

population = st.selectbox(
    "Population",
    ["a", "c", "n", "s", "v", "y"]
)

habitat = st.selectbox(
    "Habitat",
    ["d", "g", "l", "m", "p", "u", "w"]
)


input_data = pd.DataFrame({
    "cap-shape": [cap_shape],
    "cap-surface": [cap_surface],
    "cap-color": [cap_color],
    "bruises": [bruises],
    "odor": [odor],
    "gill-attachment": [gill_attachment],
    "gill-spacing": [gill_spacing],
    "gill-size": [gill_size],
    "gill-color": [gill_color],
    "stalk-shape": [stalk_shape],
    "stalk-root": [stalk_root],
    "stalk-surface-above-ring": [stalk_surface_above],
    "stalk-surface-below-ring": [stalk_surface_below],
    "stalk-color-above-ring": [stalk_color_above],
    "stalk-color-below-ring": [stalk_color_below],
    "veil-type": ["p"],
    "veil-color": [veil_color],
    "ring-number": [ring_number],
    "ring-type": [ring_type],
    "spore-print-color": [spore_print_color],
    "population": [population],
    "habitat": [habitat]
})

if st.button("🔮 Predict"):

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    classes = model.named_steps["model"].classes_

    probability_dict = dict(
        zip(classes, probabilities)
    )

    st.subheader("Prediction Result")

    if prediction == "p":

        st.error(
            "⚠️ Model Prediction: Poisonous"
        )

    else:

        st.success(
            "🍄 Model Prediction: Edible"
        )

    st.metric(
        "Model Confidence",
        f"{probability_dict[prediction] * 100:.2f}%"
    )

    st.write("Class probabilities:")

    st.write({
        "Edible": f"{probability_dict.get('e', 0) * 100:.2f}%",
        "Poisonous": f"{probability_dict.get('p', 0) * 100:.2f}%"
    })