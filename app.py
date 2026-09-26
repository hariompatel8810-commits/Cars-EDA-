import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cars EDA Report",
    page_icon=None,
    layout="wide"
)

# ---------------------------------------------------------
# Load and clean data
# Based on the work completed in the CARS EDA notebook
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Cars.csv")

    # Remove New_Price because the notebook found that more than
    # 85% of its values were null.
    if "New_Price" in df.columns:
        df.drop(columns=["New_Price"], inplace=True)

    # Feature engineering
    if "Mileage" in df.columns:
        df[["Mileage_value", "Mileage_unit"]] = (
            df["Mileage"].astype(str).str.split(" ", n=1, expand=True)
        )

    if "Engine" in df.columns:
        df[["Engine_value", "Engine_unit"]] = (
            df["Engine"].astype(str).str.split(" ", n=1, expand=True)
        )

    if "Power" in df.columns:
        df[["Power_value", "Power_unit"]] = (
            df["Power"].astype(str).str.split(" ", n=1, expand=True)
        )

    # Drop original columns after feature engineering
    drop_cols = [c for c in ["Mileage", "Engine", "Power"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Convert engineered columns to numeric
    for col in ["Mileage_value", "Engine_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Power_value" in df.columns:
        df["Power_value"] = pd.to_numeric(df["Power_value"], errors="coerce")

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Split Name into Company_name and Model_name
    if "Name" in df.columns:
        df[["Company_name", "Model_name"]] = (
            df["Name"].astype(str).str.rsplit(" ", n=1, expand=True)
        )
        df.drop(columns=["Name"], inplace=True)

    # Fill categorical missing values using mode
    numeric_cols = [
        c for c in [
            "Kilometers_Driven",
            "Mileage_value",
            "Engine_value",
            "Power_value"
        ]
        if c in df.columns
    ]

    filled_cols = [
        c for c in ["Company_name", "Model_name", "Fuel_Type", "Price"]
        if c in df.columns
    ]

    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols and c not in filled_cols
    ]

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value.iloc[0])

    # Fill numeric missing values using median
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Cars.csv was not found. Keep Cars.csv in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Unable to load the dataset: {e}")
    st.stop()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def bar_chart(series, title, x_label, y_label, rotation=0, top_n=None):
    data = series.value_counts()

    if top_n is not None:
        data = data.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data.index.astype(str), data.values)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=rotation)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def grouped_sum_chart(group_col, title):
    data = df.groupby(group_col)["Price"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data.index.astype(str), data.values)
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel("Total Price")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
st.sidebar.title("Cars EDA")
st.sidebar.markdown("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Introduction", "Report", "Conclusion"]
)

# ---------------------------------------------------------
# PAGE 1: INTRODUCTION
# ---------------------------------------------------------
if page == "Introduction":

    st.title("Cars Exploratory Data Analysis")
    st.subheader("Introduction")

    st.write(
        """
        This project presents an Exploratory Data Analysis (EDA) of a car
        dataset. The analysis is based on the work performed in the
        Jupyter notebook.

        The main purpose of the project is to understand the structure of
        the car dataset, clean the data, perform feature engineering, handle
        missing and duplicate values, and analyze car-related variables
        using visualizations.
        """
    )

    st.subheader("Objectives")

    objectives = [
        "Understand the structure and characteristics of the car dataset.",
        "Identify and handle missing values.",
        "Remove duplicate records.",
        "Perform feature engineering on Mileage, Engine, Power, and Name.",
        "Analyze categorical and numerical variables.",
        "Study relationships between price and different car attributes.",
        "Use visualizations to identify useful patterns in the dataset."
    ]

    for objective in objectives:
        st.write(f"- {objective}")

    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Rows after cleaning", f"{df.shape[0]:,}")

    with c2:
        st.metric("Columns after cleaning", f"{df.shape[1]:,}")

    with c3:
        st.metric("Duplicate rows", f"{df.duplicated().sum():,}")

    st.subheader("Main Data Preparation Steps")

    st.write(
        """
        1. Imported the car dataset.
        2. Removed the New_Price column because it contained more than
           85% missing values.
        3. Split Mileage, Engine, and Power into value and unit columns.
        4. Converted the value columns into numeric data types.
        5. Removed duplicate rows.
        6. Split the Name column into Company_name and Model_name.
        7. Filled categorical missing values using mode.
        8. Filled numerical missing values using median.
        9. Performed univariate, bivariate, and multivariate analysis.
        """
    )

# ---------------------------------------------------------
# PAGE 2: REPORT
# ---------------------------------------------------------
elif page == "Report":

    st.title("Cars EDA Report")

    # Dataset summary
    st.header("1. Dataset Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Rows", f"{df.shape[0]:,}")

    with c2:
        st.metric("Columns", f"{df.shape[1]:,}")

    with c3:
        st.metric("Missing values", f"{int(df.isnull().sum().sum()):,}")

    with c4:
        st.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": [df[col].nunique() for col in df.columns]
    })

    st.dataframe(info_df, use_container_width=True)

    # -----------------------------------------------------
    # Univariate Analysis
    # -----------------------------------------------------
    st.header("2. Univariate Analysis")

    st.subheader("Cars by Location")
    bar_chart(
        df["Location"],
        "Number of Cars by Location",
        "Location",
        "Number of Cars",
        rotation=75
    )

    st.subheader("Top 5 Locations")
    bar_chart(
        df["Location"],
        "Top 5 Locations with Highest Number of Cars",
        "Location",
        "Number of Cars",
        rotation=45,
        top_n=5
    )

    st.subheader("Cars by Year")
    bar_chart(
        df["Year"],
        "Number of Cars by Year",
        "Year",
        "Number of Cars",
        rotation=45
    )

    st.subheader("Cars by Fuel Type")
    bar_chart(
        df["Fuel_Type"],
        "Number of Cars by Fuel Type",
        "Fuel Type",
        "Number of Cars"
    )

    st.subheader("Cars by Owner Type")
    bar_chart(
        df["Owner_Type"],
        "Number of Cars by Owner Type",
        "Owner Type",
        "Number of Cars"
    )

    st.subheader("Cars by Transmission")
    bar_chart(
        df["Transmission"],
        "Number of Cars by Transmission",
        "Transmission",
        "Number of Cars"
    )

    st.subheader("Cars by Colour")
    bar_chart(
        df["Colour"],
        "Number of Cars by Colour",
        "Colour",
        "Number of Cars",
        rotation=45
    )

    st.subheader("Cars by Seats")
    bar_chart(
        df["Seats"],
        "Number of Cars by Seats",
        "Seats",
        "Number of Cars"
    )

    st.subheader("Cars by Number of Doors")
    bar_chart(
        df["No. of Doors"],
        "Number of Cars by Number of Doors",
        "Number of Doors",
        "Number of Cars"
    )

    st.subheader("Top 10 Companies")
    bar_chart(
        df["Company_name"],
        "Top 10 Companies by Number of Cars",
        "Company",
        "Number of Cars",
        rotation=60,
        top_n=10
    )

    st.subheader("Top 10 Models")
    bar_chart(
        df["Model_name"],
        "Top 10 Models by Number of Cars",
        "Model",
        "Number of Cars",
        rotation=60,
        top_n=10
    )

    # -----------------------------------------------------
    # Bivariate Analysis
    # -----------------------------------------------------
    st.header("3. Bivariate Analysis")

    st.subheader("Location-wise Total Price")
    grouped_sum_chart(
        "Location",
        "Location-wise Total Price"
    )

    st.subheader("Year-wise Total Price")
    grouped_sum_chart(
        "Year",
        "Year-wise Total Price"
    )

    st.subheader("Fuel Type-wise Total Price")
    grouped_sum_chart(
        "Fuel_Type",
        "Fuel Type-wise Total Price"
    )

    # -----------------------------------------------------
    # Multivariate Analysis
    # -----------------------------------------------------
    st.header("4. Multivariate Analysis")

    st.subheader("Fuel Type and Transmission-wise Average Price")

    fuel_transmission = (
        df.groupby(["Fuel_Type", "Transmission"])["Price"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    fuel_types = fuel_transmission["Fuel_Type"].unique()
    transmissions = fuel_transmission["Transmission"].unique()

    x = np.arange(len(fuel_types))
    width = 0.35

    for i, transmission in enumerate(transmissions):
        values = []
        for fuel in fuel_types:
            temp = fuel_transmission[
                (fuel_transmission["Fuel_Type"] == fuel) &
                (fuel_transmission["Transmission"] == transmission)
            ]

            values.append(
                temp["Price"].iloc[0] if not temp.empty else 0
            )

        offset = (i - (len(transmissions) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=str(transmission))

    ax.set_title("Fuel Type and Transmission-wise Average Price")
    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Average Price")
    ax.set_xticks(x)
    ax.set_xticklabels(fuel_types)
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Location and Fuel Type-wise Total Price")

    location_fuel = (
        df.groupby(["Location", "Fuel_Type"])["Price"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    locations = location_fuel["Location"].unique()
    fuels = location_fuel["Fuel_Type"].unique()

    x = np.arange(len(locations))
    width = 0.8 / max(len(fuels), 1)

    for i, fuel in enumerate(fuels):
        values = []

        for location in locations:
            temp = location_fuel[
                (location_fuel["Location"] == location) &
                (location_fuel["Fuel_Type"] == fuel)
            ]

            values.append(
                temp["Price"].iloc[0] if not temp.empty else 0
            )

        ax.bar(x + i * width, values, width, label=str(fuel))

    ax.set_title("Location and Fuel Type-wise Total Price")
    ax.set_xlabel("Location")
    ax.set_ylabel("Total Price")
    ax.set_xticks(x + width * (len(fuels) - 1) / 2)
    ax.set_xticklabels(locations, rotation=45)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Company and Transmission-wise Average Price")

    company_transmission = (
        df.groupby(["Company_name", "Transmission"])["Price"]
        .mean()
        .reset_index()
    )

    selected_companies = (
        company_transmission.groupby("Company_name")["Price"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .index
    )

    company_filtered = company_transmission[
        company_transmission["Company_name"].isin(selected_companies)
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    companies = list(selected_companies)
    transmissions = company_filtered["Transmission"].unique()

    x = np.arange(len(companies))
    width = 0.35

    for i, transmission in enumerate(transmissions):
        values = []

        for company in companies:
            temp = company_filtered[
                (company_filtered["Company_name"] == company) &
                (company_filtered["Transmission"] == transmission)
            ]

            values.append(
                temp["Price"].iloc[0] if not temp.empty else 0
            )

        offset = (i - (len(transmissions) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=str(transmission))

    ax.set_title("Company and Transmission-wise Average Price")
    ax.set_xlabel("Company")
    ax.set_ylabel("Average Price")
    ax.set_xticks(x)
    ax.set_xticklabels(companies, rotation=60)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------------------------------
    # Statistical Summary
    # -----------------------------------------------------
    st.header("5. Statistical Summary")

    st.subheader("Numerical Variables")
    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    st.subheader("Categorical Variables")
    st.dataframe(
        df.describe(include="object").T,
        use_container_width=True
    )

# ---------------------------------------------------------
# PAGE 3: CONCLUSION
# ---------------------------------------------------------
else:

    st.title("Conclusion")

    st.subheader("Conclusion of the Analysis")

    st.write(
        """
        The Cars EDA project demonstrates a complete data analysis workflow,
        starting from data ingestion and cleaning and continuing through
        feature engineering and visualization.

        The notebook identified a large amount of missing data in the
        New_Price column, so that column was removed. Mileage, Engine, and
        Power were transformed into separate value and unit columns, while
        the Name column was divided into Company_name and Model_name.

        Duplicate records were removed and missing categorical values were
        handled using mode. Numerical missing values were handled using
        median values.

        The analysis then examined the distribution of cars across
        locations, years, fuel types, owner types, transmissions, colours,
        seats, doors, companies, and models. Bivariate analysis examined
        total price across locations, years, and fuel types. Multivariate
        analysis examined price using combinations of fuel type,
        transmission, location, and company.

        Overall, the project provides a structured view of the car dataset
        and shows how exploratory data analysis can be used to understand
        patterns and relationships in automobile data.
        """
    )

    st.subheader("Key Analysis Areas")

    conclusion_points = [
        "Data cleaning and preprocessing were performed before analysis.",
        "Feature engineering created separate numerical and unit fields for vehicle specifications.",
        "Categorical and numerical missing values were handled using the methods used in the notebook.",
        "Duplicate records were removed.",
        "Location, year, fuel type, owner type, transmission, colour, seats, and doors were analyzed.",
        "Price was analyzed against location, year, fuel type, transmission, and company.",
        "Univariate, bivariate, and multivariate visualizations were used to understand the dataset."
    ]

    for point in conclusion_points:
        st.write(f"- {point}")

    st.subheader("Final Note")

    st.write(
        """
        This Streamlit application converts the analysis performed in the
        Jupyter notebook into an interactive three-page report that can be
        presented directly from a web browser.
        """
    )
