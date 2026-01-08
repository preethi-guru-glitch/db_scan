import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

st.title("DBSCAN Clustering")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    if numeric_df.shape[1] < 2:
        st.error("Dataset must have at least 2 numeric columns for clustering.")
    else:
        st.subheader("Selected Features")
        st.write(numeric_df.columns.tolist())

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        # Sidebar parameters
        st.sidebar.header("DBSCAN Parameters")
        eps = st.sidebar.slider("EPS (Neighborhood radius)", 0.1, 5.0, 0.5)
        min_samples = st.sidebar.slider("Min Samples", 1, 20, 5)

        # DBSCAN model
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        df["Cluster"] = labels

        st.subheader("Clustered Data")
        st.dataframe(df.head())

        # Plot using first two numeric features
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            numeric_df.iloc[:, 0],
            numeric_df.iloc[:, 1],
            c=labels
        )
        ax.set_xlabel(numeric_df.columns[0])
        ax.set_ylabel(numeric_df.columns[1])
        st.pyplot(fig)

        # Cluster info
        st.subheader("Cluster Information")
        st.write("Unique cluster labels:", set(labels))
        st.write("Noise points count:", list(labels).count(-1))
