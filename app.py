import streamlit as st
import pandas as pd
import pickle
import ast

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Smart Shopping Recommendation System",
    page_icon="🛒",
    layout="wide"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------
st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

h1{
    color:#0d6efd;
    text-align:center;
}

.stButton > button{
    width:100%;
    background-color:black !important;
    color:white !important;
    border:none;
    border-radius:8px;
    height:3.2em;
    font-size:18px;
    font-weight:bold;
}

.stButton > button:hover{
    background-color:#0b5ed7 !important;
    color:white !important;
}

.product-card{
    padding:15px;
    border-radius:12px;
    border:1px solid #dddddd;
    background:white;
    margin-bottom:15px;
}

hr{
    margin-top:20px;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------
st.title("🛍 Smart Shopping Recommendation System")

st.write(
"""
Predict product ratings and get the best product recommendations
using **Random Forest Regression**.
"""
)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
@st.cache_resource
def load_model():

    model = pickle.load(
        open("random_forest_model.pkl", "rb")
    )

    category_encoder = pickle.load(
        open("category_encoder.pkl", "rb")
    )

    seller_encoder = pickle.load(
        open("seller_encoder.pkl", "rb")
    )

    return model, category_encoder, seller_encoder


model, category_encoder, seller_encoder = load_model()

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
@st.cache_data
def load_data():

    df = pd.read_csv("dataset/Combined_dataset.csv")

    remove_words = [
        "bra",
        "boxer",
        "brief","camisoles","dresses"
    ]

    df = df[
        ~df["category"]
        .astype(str)
        .str.lower()
        .str.contains("|".join(remove_words), na=False)
    ]

    return df.reset_index(drop=True)


products = load_data()

# ----------------------------------------------------
# DROPDOWNS
# ----------------------------------------------------
categories = sorted(
    products["category"]
    .dropna()
    .unique()
)

sellers = sorted(
    products["seller_name"]
    .dropna()
    .unique()
)

# ----------------------------------------------------
# USER INPUT
# ----------------------------------------------------
st.header("Enter Product Details")

left, right = st.columns(2)

with left:

    ratings_count = st.number_input(
        "Ratings Count",
        min_value=0,
        value=500
    )

    initial_price = st.number_input(
        "Initial Price (₹)",
        min_value=1.0,
        value=1000.0
    )

    discount = st.slider(
        "Discount (%)",
        0,
        90,
        20
    )

with right:

    category = st.selectbox(
        "Category",
        categories
    )

    seller = st.selectbox(
        "Seller Name",
        sellers
    )

st.markdown("---")

predict = st.button(
    "🔍 Predict Rating",
    type="primary",
    use_container_width=True
)

# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------
if predict:

    try:

        category_encoded = category_encoder.transform(
            [category]
        )[0]

        seller_encoded = seller_encoder.transform(
            [seller]
        )[0]

        input_df = pd.DataFrame({

            "ratings_count":[ratings_count],

            "initial_price":[initial_price],

            "discount":[discount],

            "category":[category_encoded],

            "seller_name":[seller_encoded]

        })

        prediction = model.predict(input_df)[0]

        st.markdown("---")

        st.header("⭐ Prediction Result")

        st.metric(
            "Predicted Rating",
            f"{prediction:.2f} ⭐"
        )

        if prediction >= 4.5:

            st.success("🌟 Highly Recommended Product")

        elif prediction >= 4:

            st.success("✅ Recommended Product")

        elif prediction >= 3:

            st.warning("👍 Average Product")

        else:

            st.error("❌ Not Recommended")

        # ----------------------------------------------------
        # RECOMMENDED PRODUCTS
        # ----------------------------------------------------

        st.markdown("---")

        st.header("🛍 Recommended Products")

        similar = products[
            products["category"] == category
        ].copy()

        similar = similar.sort_values(
            by="rating",
            ascending=False
        ).head(5)

        if similar.empty:

            st.warning("No products found.")

        else:

            for _, row in similar.iterrows():

                st.markdown("---")

                col1, col2 = st.columns([1,3])

                with col1:

                    image = row["images"]

                    try:

                        if pd.notna(image):

                            if (
                                isinstance(image, str)
                                and image.startswith("[")
                            ):

                                image = ast.literal_eval(image)[0]

                            st.image(
                                image,
                                width=180
                            )

                        else:

                            st.write("🖼 No Image")

                    except:

                        st.write("🖼 No Image")

                with col2:

                    st.subheader(row["title"])

                    st.write(
                        f"⭐ Rating : {row['rating']}"
                    )

                    st.write(
                        f"👥 Ratings Count : {row['ratings_count']}"
                    )

                    st.write(
                        f"💰 Price : ₹{row['initial_price']}"
                    )

                    st.write(
                        f"🏷 Discount : {row['discount']}%"
                    )

                    st.write(
                        f"🏪 Seller : {row['seller_name']}"
                    )

                    if row["rating"] >= 4:

                        st.success("✅ Recommended")

                    else:

                        st.warning("⚠ Average Product")
        # ----------------------------------------------------
        # CATEGORY STATISTICS
        # ----------------------------------------------------

        st.markdown("---")
        st.header("📊 Category Statistics")

        category_df = products[
            products["category"] == category
        ]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Products",
            len(category_df)
        )

        c2.metric(
            "Average Rating",
            round(category_df["rating"].mean(), 2)
        )

        c3.metric(
            "Average Price",
            f"₹{round(category_df['initial_price'].mean(), 2)}"
        )

        # ----------------------------------------------------
        # DOWNLOAD PREDICTION
        # ----------------------------------------------------

        st.markdown("---")

        result = pd.DataFrame({

            "Category":[category],
            "Seller":[seller],
            "Ratings Count":[ratings_count],
            "Initial Price":[initial_price],
            "Discount":[discount],
            "Predicted Rating":[round(prediction, 2)]

        })

        csv = result.to_csv(index=False)

        st.download_button(
            label="📥 Download Prediction",
            data=csv,
            file_name="prediction.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:

        st.error(f"Error: {e}")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div style="text-align:center; padding:15px;">

<h3>🛒 Smart Shopping Recommendation System</h3>

<p>
Developed using <b>Random Forest Regression</b> and <b>Streamlit</b>
</p>

</div>
""",
unsafe_allow_html=True
)                        
