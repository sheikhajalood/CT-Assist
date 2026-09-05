import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Lung Candidate Detection",
    page_icon="🫁",
    layout="wide"
)

# --------------------------------------------------------
# FUTURISTIC AESTHETIC CSS INJECTION
# --------------------------------------------------------
st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

        :root{
            --bg-dark: #071026;
            --accent-a: #00e6ff;
            --accent-b: #7a00ff;
            --muted: #9aa6b2;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: radial-gradient(1200px 600px at 10% 20%, rgba(122,0,255,0.06), transparent 8%),
                                    radial-gradient(900px 500px at 95% 90%, rgba(0,230,255,0.04), transparent 10%),
                                    var(--bg-dark) !important;
            color: #dbeafe;
            font-family: Inter, sans-serif;
        }

        header, .css-1y4p8pa, .css-18e3th9 { background: transparent !important; }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg,var(--accent-a),var(--accent-b)) !important;
            color: #001219 !important;
            border: none !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6), 0 0 20px rgba(0,230,255,0.06);
            border-radius: 10px !important;
            padding: 8px 12px !important;
            font-weight: 700 !important;
        }

        /* Sidebar glass */
        .stSidebar, [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)) !important;
            border-radius: 12px !important;
            padding: 12px !important;
        }

        /* Images and cards */
        .stImage img, .stMetric > div {
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.03) !important;
            box-shadow: 0 12px 40px rgba(2,6,23,0.7) !important;
        }

        /* Headline / title accent */
        .css-1d391kg, .css-10trblm { font-family: 'Orbitron', sans-serif !important; color: var(--accent-a) !important; text-shadow: 0 0 8px rgba(0,230,255,0.12) !important; }

        /* Subtext and captions */
        .stCaption, .stText, .stMarkdown p { color: var(--muted) !important; }

        /* Dataframe/table styling */
        [data-testid="stDataFrame"] table {
            background: rgba(6,10,16,0.65) !important;
            color: #cfeefb !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }

        /* Expander */
        .streamlit-expanderHeader { background: rgba(255,255,255,0.01) !important; border-radius: 8px !important; }

        /* Neon label helper */
        .neon-label { color: var(--accent-a) !important; text-shadow: 0 0 6px rgba(0,230,255,0.28), 0 0 18px rgba(122,0,255,0.04) !important; font-weight:700 !important; }

        /* Scrollbar */
        ::-webkit-scrollbar { height:10px; width:10px; }
        ::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#00e6ff44,#7a00ff44); border-radius:10px; }
        </style>
        """,
        unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("lung_candidate_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load lung_candidate_model.pkl")
    st.code(str(e))
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_image(image):
    """
    Normalize any uploaded image to uint8 grayscale.
    """

    image = np.asarray(image)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = image.astype(np.float32)

    minimum = image.min()
    maximum = image.max()

    if maximum > minimum:
        image = (
            (image - minimum)
            / (maximum - minimum)
            * 255
        )

    return image.astype(np.uint8)


def extract_features(image, contour):
    """
    Extract the same seven features expected by
    the Random Forest model.
    """

    x, y, w, h = cv2.boundingRect(contour)

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    aspect_ratio = (
        w / h
        if h > 0
        else 0
    )

    bounding_area = w * h

    extent = (
        area / bounding_area
        if bounding_area > 0
        else 0
    )

    mask = np.zeros(
        image.shape,
        dtype=np.uint8
    )

    cv2.drawContours(
        mask,
        [contour],
        -1,
        255,
        -1
    )

    pixels = image[
        mask == 255
    ]

    if len(pixels) > 0:

        mean_intensity = float(
            np.mean(pixels)
        )

        intensity_std = float(
            np.std(pixels)
        )

    else:

        mean_intensity = 0.0
        intensity_std = 0.0

    if perimeter > 0:

        circularity = (
            4 * np.pi * area
            / (perimeter ** 2)
        )

    else:

        circularity = 0.0

    return {
        "X": x,
        "Y": y,
        "Width": w,
        "Height": h,
        "Area": area,
        "Perimeter": perimeter,
        "Aspect Ratio": aspect_ratio,
        "Extent": extent,
        "Mean Intensity": mean_intensity,
        "Intensity Std": intensity_std,
        "Circularity": circularity
    }


def remove_duplicate_candidates(candidates):
    """
    Remove heavily overlapping candidate boxes.
    """

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda c: c["Area"],
        reverse=True
    )

    kept = []

    for candidate in candidates:

        x1 = candidate["X"]
        y1 = candidate["Y"]
        x2 = x1 + candidate["Width"]
        y2 = y1 + candidate["Height"]

        duplicate = False

        for existing in kept:

            ex1 = existing["X"]
            ey1 = existing["Y"]
            ex2 = ex1 + existing["Width"]
            ey2 = ey1 + existing["Height"]

            intersection_x1 = max(
                x1,
                ex1
            )

            intersection_y1 = max(
                y1,
                ey1
            )

            intersection_x2 = min(
                x2,
                ex2
            )

            intersection_y2 = min(
                y2,
                ey2
            )

            if (
                intersection_x2 > intersection_x1
                and
                intersection_y2 > intersection_y1
            ):

                intersection_area = (
                    intersection_x2 - intersection_x1
                ) * (
                    intersection_y2 - intersection_y1
                )

                candidate_area = (
                    candidate["Width"]
                    * candidate["Height"]
                )

                if candidate_area > 0:

                    overlap = (
                        intersection_area
                        / candidate_area
                    )

                    if overlap > 0.65:
                        duplicate = True
                        break

        if not duplicate:
            kept.append(candidate)

    return kept


def detect_candidates(image):
    """
    Robust multi-threshold candidate detection.

    Several thresholds are tested so that an image
    with weak contrast does not automatically produce
    zero candidates.
    """

    height, width = image.shape

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # Local contrast
    # --------------------------------------------------------

    local_mean = cv2.GaussianBlur(
        blurred,
        (21, 21),
        0
    )

    contrast = cv2.absdiff(
        blurred,
        local_mean
    )

    # --------------------------------------------------------
    # Approximate central lung region
    #
    # This is intentionally broad because CT images
    # can have different framing/cropping.
    # --------------------------------------------------------

    roi_mask = np.zeros_like(image)

    x1 = int(width * 0.08)
    x2 = int(width * 0.92)

    y1 = int(height * 0.08)
    y2 = int(height * 0.92)

    roi_mask[
        y1:y2,
        x1:x2
    ] = 255

    # --------------------------------------------------------
    # Try multiple thresholds
    # --------------------------------------------------------

    thresholds = [
        40,
        35,
        30,
        25,
        20,
        15
    ]

    all_candidates = []

    best_mask = None
    best_contours = []

    for threshold in thresholds:

        mask = np.where(
            contrast > threshold,
            255,
            0
        ).astype(np.uint8)

        # Morphological cleaning
        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        clean = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # Restrict to broad central region
        clean = cv2.bitwise_and(
            clean,
            roi_mask
        )

        contours, _ = cv2.findContours(
            clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            # Adaptive minimum area
            minimum_area = max(
                15,
                int((width * height) * 0.00003)
            )

            if area < minimum_area:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            # Reject extremely large regions
            if area > (
                width * height * 0.08
            ):
                continue

            # Reject very thin structures
            if w < 4 or h < 4:
                continue

            # Candidate must remain reasonably
            # inside image
            center_x = x + w / 2
            center_y = y + h / 2

            if not (
                x1 <= center_x <= x2
                and
                y1 <= center_y <= y2
            ):
                continue

            features = extract_features(
                image,
                contour
            )

            features["Contour"] = contour
            features["Threshold"] = threshold

            all_candidates.append(
                features
            )

        # Keep the mask from the threshold that
        # generated the most useful candidates.
        if len(contours) > len(best_contours):

            best_contours = contours
            best_mask = clean

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    all_candidates = remove_duplicate_candidates(
        all_candidates
    )

    # --------------------------------------------------------
    # If still too few candidates, use the
    # strongest local-contrast regions.
    #
    # These remain "candidates", NOT diagnoses.
    # --------------------------------------------------------

    if len(all_candidates) < 3:

        # Use threshold 15 for a permissive fallback
        fallback_mask = np.where(
            contrast > 15,
            255,
            0
        ).astype(np.uint8)

        fallback_kernel = np.ones(
            (5, 5),
            np.uint8
        )

        fallback_mask = cv2.morphologyEx(
            fallback_mask,
            cv2.MORPH_OPEN,
            fallback_kernel
        )

        fallback_mask = cv2.bitwise_and(
            fallback_mask,
            roi_mask
        )

        fallback_contours, _ = cv2.findContours(
            fallback_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        fallback_candidates = []

        for contour in fallback_contours:

            area = cv2.contourArea(
                contour
            )

            if area < 10:
                continue

            if area > (
                width * height * 0.05
            ):
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w < 3 or h < 3:
                continue

            center_x = x + w / 2
            center_y = y + h / 2

            if not (
                x1 <= center_x <= x2
                and
                y1 <= center_y <= y2
            ):
                continue

            features = extract_features(
                image,
                contour
            )

            features["Contour"] = contour
            features["Threshold"] = 15

            fallback_candidates.append(
                features
            )

        fallback_candidates = (
            remove_duplicate_candidates(
                fallback_candidates
            )
        )

        # Combine candidates
        all_candidates.extend(
            fallback_candidates
        )

        all_candidates = (
            remove_duplicate_candidates(
                all_candidates
            )
        )

    # --------------------------------------------------------
    # Final safety fallback
    #
    # If no contour is available, create
    # small central analysis windows based on
    # strongest contrast locations.
    # --------------------------------------------------------

    if len(all_candidates) == 0:

        # Find strongest contrast points
        smooth_contrast = cv2.GaussianBlur(
            contrast,
            (15, 15),
            0
        )

        # Mask outside broad central region
        smooth_contrast = cv2.bitwise_and(
            smooth_contrast,
            roi_mask
        )

        flat_indices = np.argsort(
            smooth_contrast.ravel()
        )[::-1]

        used_points = []

        window_size = max(
            12,
            int(min(height, width) * 0.035)
        )

        for index in flat_indices:

            y, x = np.unravel_index(
                index,
                smooth_contrast.shape
            )

            if smooth_contrast[y, x] <= 0:
                break

            # Avoid points too close together
            too_close = False

            for px, py in used_points:

                distance = np.sqrt(
                    (x - px) ** 2
                    +
                    (y - py) ** 2
                )

                if distance < window_size * 2:
                    too_close = True
                    break

            if too_close:
                continue

            half = window_size // 2

            bx = max(
                0,
                x - half
            )

            by = max(
                0,
                y - half
            )

            bw = min(
                window_size,
                width - bx
            )

            bh = min(
                window_size,
                height - by
            )

            roi = np.zeros_like(image)

            cv2.rectangle(
                roi,
                (bx, by),
                (bx + bw, by + bh),
                255,
                -1
            )

            fake_contour = np.array(
                [
                    [
                        [bx, by]
                    ],
                    [
                        [bx + bw, by]
                    ],
                    [
                        [bx + bw, by + bh]
                    ],
                    [
                        [bx, by + bh]
                    ]
                ],
                dtype=np.int32
            )

            features = extract_features(
                image,
                fake_contour
            )

            features["Contour"] = fake_contour
            features["Threshold"] = 15

            all_candidates.append(
                features
            )

            used_points.append(
                (x, y)
            )

            if len(all_candidates) >= 5:
                break

    # Sort by image-processing strength
    all_candidates = sorted(
        all_candidates,
        key=lambda c: (
            c["Intensity Std"]
            * 0.6
            +
            c["Circularity"]
            * 50
            * 0.2
            +
            c["Extent"]
            * 50
            * 0.2
        ),
        reverse=True
    )

    # Limit to useful number
    all_candidates = all_candidates[:12]

    return (
        all_candidates,
        contrast,
        best_mask
    )


# ============================================================
# TITLE
# ============================================================

st.title("🫁 Lung Candidate Detection")
st.caption(
    "AI-assisted CT image analysis prototype"
)

st.info(
    "Upload a CT image to identify and rank "
    "candidate regions for further review."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Analysis Pipeline")

    st.write(
        """
        **1. Image preprocessing**

        **2. Local contrast analysis**

        **3. Multi-threshold candidate detection**

        **4. Morphological cleaning**

        **5. Region filtering**

        **6. Feature extraction**

        **7. Random Forest scoring**

        **8. Candidate visualization**
        """
    )

    st.divider()

    st.caption(
        "Research/demo prototype only."
    )


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload CT Image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


if uploaded_file is not None:

    try:

        uploaded_image = Image.open(
            uploaded_file
        )

        image = normalize_image(
            uploaded_image
        )

    except Exception as e:

        st.error(
            "Unable to read the uploaded image."
        )

        st.code(str(e))

        st.stop()

    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    st.subheader("🖼️ CT Scan")

    st.image(
        image,
        caption="Uploaded CT Scan",
        use_container_width=True
    )


    # ========================================================
    # ANALYZE
    # ========================================================

    if st.button(
        "🔍 Analyze CT Scan",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing CT scan..."
        ):

            candidates, contrast, clean_mask = (
                detect_candidates(image)
            )


            # =================================================
            # FEATURE DATAFRAME
            # =================================================

            rows = []

            for i, candidate in enumerate(
                candidates
            ):

                row = {
                    "Region": f"Region {i + 1}",
                    "X": candidate["X"],
                    "Y": candidate["Y"],
                    "Width": candidate["Width"],
                    "Height": candidate["Height"],
                    "Area": candidate["Area"],
                    "Perimeter": candidate["Perimeter"],
                    "Aspect Ratio": candidate[
                        "Aspect Ratio"
                    ],
                    "Extent": candidate["Extent"],
                    "Mean Intensity": candidate[
                        "Mean Intensity"
                    ],
                    "Intensity Std": candidate[
                        "Intensity Std"
                    ],
                    "Circularity": candidate[
                        "Circularity"
                    ]
                }

                rows.append(row)


            features_df = pd.DataFrame(
                rows
            )


            # =================================================
            # RANDOM FOREST
            # =================================================

            feature_columns = [
                "Area",
                "Perimeter",
                "Aspect Ratio",
                "Extent",
                "Mean Intensity",
                "Intensity Std",
                "Circularity"
            ]


            if len(features_df) > 0:

                X_model = features_df[
                    feature_columns
                ]

                try:

                    predictions = model.predict(
                        X_model
                    )

                    probabilities = (
                        model.predict_proba(
                            X_model
                        )
                    )

                    if 1 in model.classes_:

                        positive_index = list(
                            model.classes_
                        ).index(1)

                        rf_probability = (
                            probabilities[
                                :,
                                positive_index
                            ]
                        )

                    else:

                        rf_probability = np.zeros(
                            len(features_df)
                        )

                except Exception:

                    predictions = np.zeros(
                        len(features_df),
                        dtype=int
                    )

                    rf_probability = np.zeros(
                        len(features_df)
                    )

            else:

                predictions = np.array([])
                rf_probability = np.array([])


            # =================================================
            # IMAGE-BASED PRIORITY SCORE
            # =================================================

            if len(features_df) > 0:

                score_df = features_df[
                    feature_columns
                ].copy()

                for column in [
                    "Area",
                    "Perimeter",
                    "Aspect Ratio",
                    "Extent",
                    "Mean Intensity",
                    "Intensity Std",
                    "Circularity"
                ]:

                    minimum = score_df[
                        column
                    ].min()

                    maximum = score_df[
                        column
                    ].max()

                    if maximum != minimum:

                        score_df[column] = (
                            score_df[column]
                            - minimum
                        ) / (
                            maximum
                            - minimum
                        )

                    else:

                        score_df[column] = 0.0


                # Compactness
                compactness = (
                    1
                    /
                    (
                        1
                        +
                        abs(
                            features_df[
                                "Aspect Ratio"
                            ]
                            - 1
                        )
                    )
                )


                # Prototype priority score
                candidate_score = (

                    0.30
                    * score_df[
                        "Intensity Std"
                    ]

                    +

                    0.25
                    * score_df[
                        "Extent"
                    ]

                    +

                    0.20
                    * score_df[
                        "Area"
                    ]

                    +

                    0.15
                    * score_df[
                        "Circularity"
                    ]

                    +

                    0.10
                    * compactness
                )


                features_df[
                    "Candidate Score"
                ] = candidate_score.values

            else:

                features_df[
                    "Candidate Score"
                ] = []


            # =================================================
            # ADD RF OUTPUT
            # =================================================

            features_df[
                "RF Prediction"
            ] = predictions

            features_df[
                "RF Probability"
            ] = rf_probability


            # =================================================
            # COMBINE PRIORITY
            #
            # We use both the prototype score and RF output.
            # This is still a research/demo ranking.
            # =================================================

            if len(features_df) > 0:

                features_df[
                    "Priority Score"
                ] = (

                    0.55
                    * features_df[
                        "Candidate Score"
                    ]

                    +

                    0.45
                    * features_df[
                        "RF Probability"
                    ]
                )


                features_df = (
                    features_df
                    .sort_values(
                        "Priority Score",
                        ascending=False
                    )
                    .reset_index(
                        drop=True
                    )
                )


            # =================================================
            # NUMBER OF HIGH PRIORITY CANDIDATES
            # =================================================

            if len(features_df) > 0:

                # Show top candidates rather than
                # relying entirely on the tiny RF.
                high_priority_count = min(
                    3,
                    len(features_df)
                )

            else:

                high_priority_count = 0


            # =================================================
            # SUCCESS
            # =================================================

            st.success(
                "Analysis completed successfully."
            )


            # =================================================
            # METRICS
            # =================================================

            st.subheader(
                "📊 Analysis Results"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Candidate Regions",
                    len(features_df)
                )

            with col2:

                st.metric(
                    "High-Priority Candidates",
                    high_priority_count
                )

            with col3:

                st.metric(
                    "Model",
                    "Random Forest"
                )


            # =================================================
            # VISUALIZATION
            # =================================================

            st.subheader(
                "🎯 Suspicious Candidate Regions"
            )

            result_image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )


            # Draw candidates
            for i, row in features_df.iterrows():

                x = int(row["X"])
                y = int(row["Y"])
                w = int(row["Width"])
                h = int(row["Height"])

                rank = i + 1

                priority = float(
                    row["Priority Score"]
                )

                # Top 3 = stronger visual emphasis
                if rank <= high_priority_count:

                    thickness = 3

                else:

                    thickness = 1


                # Draw rectangle
                cv2.rectangle(
                    result_image,

                    (x, y),

                    (
                        x + w,
                        y + h
                    ),

                    (255, 0, 0),

                    thickness
                )


                label = (
                    f"R{rank} "
                    f"{priority:.2f}"
                )


                cv2.putText(
                    result_image,

                    label,

                    (
                        x,
                        max(
                            y - 6,
                            15
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    (255, 0, 0),

                    1,

                    cv2.LINE_AA
                )


            st.image(
                result_image,
                caption=(
                    "Ranked candidate regions — "
                    "not confirmed lesions"
                ),
                use_container_width=True
            )


            # =================================================
            # TWO COLUMN PROCESSING VIEW
            # =================================================

            st.subheader(
                "🔬 Image Processing"
            )

            process_col1, process_col2 = (
                st.columns(2)
            )


            with process_col1:

                # Normalize contrast for display
                contrast_display = cv2.normalize(
                    contrast,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                ).astype(np.uint8)

                st.image(
                    contrast_display,
                    caption="Local Contrast Map",
                    use_container_width=True
                )


            with process_col2:

                if clean_mask is not None:

                    st.image(
                        clean_mask,
                        caption="Candidate Mask",
                        use_container_width=True
                    )

                else:

                    st.image(
                        np.zeros_like(image),
                        caption="Candidate Mask",
                        use_container_width=True
                    )


            # =================================================
            # RESULTS TABLE
            # =================================================

            st.subheader(
                "🔎 Ranked Candidate Regions"
            )


            display_df = features_df[
                [
                    "Region",
                    "Priority Score",
                    "Candidate Score",
                    "RF Probability",
                    "Area",
                    "Circularity"
                ]
            ].copy()


            display_df[
                "Priority Score"
            ] = display_df[
                "Priority Score"
            ].round(3)


            display_df[
                "Candidate Score"
            ] = display_df[
                "Candidate Score"
            ].round(3)


            display_df[
                "RF Probability"
            ] = display_df[
                "RF Probability"
            ].round(2)


            display_df[
                "Area"
            ] = display_df[
                "Area"
            ].round(1)


            display_df[
                "Circularity"
            ] = display_df[
                "Circularity"
            ].round(3)


            display_df["Priority"] = [
                (
                    "🔴 High"
                    if i < high_priority_count
                    else "🟡 Review"
                )
                for i in range(
                    len(display_df)
                )
            ]


            display_df = display_df[
                [
                    "Region",
                    "Priority",
                    "Priority Score",
                    "Candidate Score",
                    "RF Probability",
                    "Area",
                    "Circularity"
                ]
            ]


            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # FEATURE DETAILS
            # =================================================

            with st.expander(
                "📋 View Extracted Features"
            ):

                feature_display = features_df[
                    [
                        "Region",
                        "Area",
                        "Perimeter",
                        "Aspect Ratio",
                        "Extent",
                        "Mean Intensity",
                        "Intensity Std",
                        "Circularity"
                    ]
                ].copy()

                st.dataframe(
                    feature_display.round(3),
                    use_container_width=True,
                    hide_index=True
                )


            # =================================================
            # EXPLANATION
            # =================================================

            st.info(
                """
                **How the result is generated**

                The system first identifies image regions
                with local intensity variation. Candidate
                regions are filtered and their shape/intensity
                features are extracted. The trained Random
                Forest then provides a prototype probability,
                while the candidate score ranks regions for
                further review.

                A highlighted region is a **candidate region**,
                not a confirmed tumor or diagnosis.
                """
            )


            # =================================================
            # DISCLAIMER
            # =================================================

            st.warning(
                "⚠️ Research/demo prototype only. "
                "The current Random Forest was trained on a "
                "very small demonstration dataset and its "
                "scores are not clinical probabilities. "
                "This system must not be used to diagnose "
                "or rule out disease."
            )


else:

    # ========================================================
    # NO IMAGE
    # ========================================================

    st.info(
        "👆 Upload a CT image above to begin analysis."
    )

    st.markdown(
        """
        ### What this prototype does

        **CT Image**
        → Preprocessing
        → Local contrast
        → Candidate detection
        → Feature extraction
        → Random Forest
        → Candidate ranking
        → Visualization
        """
    )

    st.warning(
        "⚠️ Research/demo prototype — not a medical "
        "diagnostic tool."
    )