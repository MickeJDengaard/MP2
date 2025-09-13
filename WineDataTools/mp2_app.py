import streamlit as st
import pandas as pd
import plots
import stats
import data_io as io_utils
from sklearn.decomposition import PCA
import plotly.express as px

st.title("Wine Quality Data Exploration")

# ------------------------
# Upload Section | So the user can upload their own data
# ------------------------
st.sidebar.header("Upload Wine Data")
red_file = st.sidebar.file_uploader("Upload red wine Excel", type=["xlsx"])
white_file = st.sidebar.file_uploader("Upload white wine Excel", type=["xlsx"])

if red_file and white_file:
    # Læs data med data_io.py
    red_df = io_utils.read_wine_data(red_file, "red")
    white_df = io_utils.read_wine_data(white_file, "white")
    df = io_utils.combine_dataframes([red_df, white_df])
    df = io_utils.remove_duplicates(df)

    st.success("Data loaded successfully!!")

    # ------------------------
    # Data preview
    # ------------------------
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # ------------------------
    # Descriptive statistics
    # ------------------------
    st.subheader("Descriptive Statistics")
    summary_df = stats.describe_wine_data(df, verbose=False, round_digits=2)
    st.dataframe(summary_df)

    # ------------------------
    # Mean comparison
    # ------------------------
    st.subheader("Mean Comparison (White vs Red)")
    means, diffs = stats.mean_comparison(df)
    st.write("Means:")
    st.dataframe(means)
    st.write("Differences (White - Red):")
    st.dataframe(diffs)

    # ------------------------
    # Plots
    # ------------------------
    st.subheader("Visualizations")

    plot_type = st.selectbox("Select plot type:",
                             ["Histogram", "Grouped Histogram", "Boxplot", "Boxplot by type", "Scatter vs Quality",
                              "Correlation Heatmap"])

    if plot_type == "Histogram":
        plots.show_histograms(df, bins=15)
    elif plot_type == "Grouped Histogram":
        plots.show_grouped_histograms(df, bins=15)
    elif plot_type == "Boxplot":
        plots.show_boxplots(df)
    elif plot_type == "Boxplot by type":
        plots.boxplots_by_type(df)
    elif plot_type == "Scatter vs Quality":
        plots.scatter_vs_quality(df)
    elif plot_type == "Correlation Heatmap":
        plots.show_correlation_heatmap(df)

    # ------------------------
    # PCA 3D Visualization
    # ------------------------
    st.subheader("PCA 3D Visualization")
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) >= 3:
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(df[numeric_cols])
        pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2", "PC3"])
        pca_df["type"] = df["type"].values
        fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="type", opacity=0.7)
        st.plotly_chart(fig)
    else:
        st.info("Not enough numeric columns for 3D PCA plot.")

else:
    st.warning("Please upload both red and white wine Excel files.")
