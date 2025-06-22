# FinalProjectAirBnb
# Airbnb Dynamic Pricing Recommendation Engine

This project presents a data analytics pipeline to recommend optimal pricing for Airbnb listings based on historical data. Using Python for data processing and machine learning, Excel for manual data cleaning, and Power BI for visualization, the project provides actionable insights for hosts and market analysts.

---

## Objective

To analyze Airbnb listing data and predict optimal prices by considering key factors such as location, seasonality, room type, and listing quality — and to visualize these insights through an interactive Power BI dashboard.

---

## Dataset

- **Source**: Cleaned manually using Excel (`listings_clean.csv`)
- **Key Fields**:
  - `room_type`, `neighbourhood_group`, `minimum_nights`, `availability_365`
  - `number_of_reviews`, `reviews_per_month`, `last_review`, `price`

---

## Workflow

### 1. **Data Cleaning**
- Performed manual cleaning in Excel: removed nulls, filtered errors
- Further cleaned in Python (e.g., handling missing values, converting date formats)

### 2. **Feature Engineering**
- Extracted review month from `last_review`
- One-hot encoded categorical columns (`room_type`, `neighbourhood_group`)

### 3. **Modeling (Python)**
- Split data into training and testing sets
- Imputed missing values
- Trained a `RandomForestRegressor` to predict listing price
- Flagged listings as:
  - **Underpriced** (Error > +50)
  - **Overpriced** (Error < –50)
  - **Fair** (otherwise)

### 4. **Output**
- Predictions and pricing flags saved to `airbnb_price_predictions.csv`
- Top pricing predictors visualized via bar charts and correlation heatmaps

### 5. **Visualization (Power BI)**
- Built an interactive dashboard (`ProjectFinal.pbix`) featuring:
  - Map of listings by neighbourhood
  - Scatter plot: Actual vs Predicted Price
  - Donut chart: Price share by neighbourhood group
  - Bar charts: Avg price by room type, total price by top hosts
  - KPI cards and slicers for filtering

---

## Key Insights

- **Entire homes** has the highest price listing of 2M and 8682 hosts.
- **Eixample** has the highest number of of hosts(1690) and has the highest total listing price of 1.17M (43.74%). It indicates a large number of active listings and/or higher pricing per night in this neighborhood.
- The host **Acomodis Apartments** has the highest total price listing of 177k.



---



