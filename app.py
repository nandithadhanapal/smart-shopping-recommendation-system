import streamlit as st
import pandas as pd
import pickle


# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Smart Shopping Recommendation System",
    page_icon="🛒",
    layout="wide"
)


# ----------------------------
# Title
# ----------------------------

st.title("🛒 Smart Shopping Recommendation System")

st.write(
    "Find the best products within your budget using Machine Learning."
)



# ----------------------------
# Load Model
# ----------------------------

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



# ----------------------------
# Load Dataset
# ----------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "dataset/Combined_dataset.csv"
    )

    return df



df = load_data()
remove_words = ["bra", "boxer", "brief","camisoles","swim-tops","saree-blouse","thermal-bottoms","tights"]

df = df[
    ~df["category"].astype(str).str.lower().str.contains(
        "|".join(remove_words),
        na=False
    )
]



# ----------------------------
# Data Cleaning
# ----------------------------

df["discount"] = df["discount"].fillna(
    df["discount"].median()
)


df["seller_name"] = df["seller_name"].fillna(
    "Unknown"
)


df["ratings_count"] = df["ratings_count"].fillna(
    0
)



# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.header(
    "🛍️ Shopping Preferences"
)


category = st.sidebar.selectbox(
    "Select Category",
    sorted(df["category"].unique())
)


budget = st.sidebar.number_input(
    "Enter Budget ₹",
    min_value=100,
    value=5000,
    step=100
)


top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)



# ----------------------------
# Recommendation
# ----------------------------

if st.sidebar.button("Recommend Products"):


    # Filter category

    filtered = df[
        df["category"] == category
    ].copy()



    if len(filtered) == 0:

        st.error(
            "No products available in this category."
        )

        st.stop()



    # Budget filter

    budget_filtered = filtered[
        filtered["initial_price"] <= budget
    ].copy()



    if len(budget_filtered) >= top_n:

        filtered = budget_filtered


    else:

        st.warning(
            "Not enough products in this budget. Showing closest products."
        )

        filtered = filtered.sort_values(
            by="initial_price"
        )



    # Keep products

    filtered = filtered.head(50)



    # ----------------------------
    # Encoding
    # ----------------------------


    filtered["category"] = category_encoder.transform(
        filtered["category"]
    )



    known_sellers = set(
        seller_encoder.classes_
    )


    filtered["seller_name"] = filtered["seller_name"].apply(
        lambda x:
        x if x in known_sellers
        else seller_encoder.classes_[0]
    )



    filtered["seller_name"] = seller_encoder.transform(
        filtered["seller_name"]
    )



    # ----------------------------
    # Model Input
    # ----------------------------


    # ----------------------------
    # Model Input
    # ----------------------------

    X = filtered[
        [
            "ratings_count",
            "initial_price",
            "discount",
            "category",
            "seller_name"
        ]
    ]


    # ----------------------------
    # Random Forest Prediction
    # ----------------------------

    filtered["Predicted Rating"] = model.predict(X)


    st.subheader("🤖 Random Forest Prediction")

    best_product = filtered.iloc[0]

    st.write(
        "Product:",
        best_product["title"]
    )

    st.success(
        f"Predicted Rating: {best_product['Predicted Rating']:.2f} ⭐"
    )



    # ----------------------------
    # Ranking
    # ----------------------------


    result = filtered.sort_values(
        by=[
            "Predicted Rating",
            "discount",
            "initial_price"
        ],
        ascending=[
            False,
            False,
            True
        ]
    )


    result = result.head(top_n)



    # ----------------------------
    # Display
    # ----------------------------


    st.success(
        f"Top {len(result)} Recommended Products"
    )



    # ----------------------------
    # Display Results
    # ----------------------------

    for _, row in result.iterrows():

        st.markdown("---")

        # Product Image
        if pd.notna(row["images"]) and str(row["images"]).startswith("http"):
            st.image(
                row["images"],
                width=250
            )

        # Product Name
        st.subheader(
            "🛍️ " + str(row["title"])
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "⭐ Predicted Rating",
                f"{row['Predicted Rating']:.2f}"
            )

        with col2:
            st.metric(
                "💰 Price",
                f"₹{row['initial_price']:.2f}"
            )

        with col3:
            st.metric(
                "🏷️ Discount",
                f"{row['discount']}%"
            )

        st.write(
            "⭐ **Original Rating:**",
            row["rating"]
        )

        st.write(
            "👥 **Ratings Count:**",
            int(row["ratings_count"])
        )

        st.write(
            "🏪 **Seller:**",
            row["seller_name"]
        )

        st.write(
            "📂 **Category:**",
            category
        )


    st.write("⭐ **Original Rating:**", row["rating"])
    st.write("👥 **Ratings Count:**", int(row["ratings_count"]))
    st.write("🏪 **Seller:**", row["seller_name"])
    st.write("📂 **Category:**", category)

    # Description
    if pd.notna(row["product_description"]):
        st.write("📝 **Description:**")
        st.write(row["product_description"])

    # Delivery Options
    if "delivery_options" in row.index and pd.notna(row["delivery_options"]):
        st.write("🚚 **Delivery:**", row["delivery_options"])

    # Product Specifications
    if "product_specifications" in row.index and pd.notna(row["product_specifications"]):
        with st.expander("📋 Product Specifications"):
            st.write(row["product_specifications"])

    # Customer Reviews
    if "what_customers_said" in row.index and pd.notna(row["what_customers_said"]):
        with st.expander("💬 Customer Reviews"):
            st.write(row["what_customers_said"])



else:
    st.info(
        "Select category and budget, then click Recommend Products."
    )
 