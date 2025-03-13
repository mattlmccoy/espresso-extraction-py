# espresso-extraction-py

## **Espresso Extraction Analysis**
### *A Metrology-Driven Approach to Perfecting Espresso Shots* ☕  

This repository contains Python scripts designed to **collect, analyze, and visualize espresso extraction data**. The project leverages metrology principles to study the effects of **temperature and grind size** on **extraction yield**, demonstrating how precise measurements impact coffee quality.  

---

## **📜 Features**
- **Data Collection:**  
  - Users input experimental parameters such as **temperature, pressure, grind size, extraction time, and Brix value**.
  - Automatically calculates **Total Dissolved Solids (TDS)** and **extraction yield**.
  - Saves data in structured CSV format under `extraction_data/`.

- **2D Data Visualization:**  
  - Plots **extraction yield vs. various parameters**.
  - Includes **uncertainty analysis** with **error bars**.
  - Linear regression applied to show data trends.

- **3D Data Visualization:**  
  - Plots **extraction yield vs. two selected parameters**.
  - Fits a **trend plane using least squares regression**.
  - Shows **real data points (red)** and **interpolated trend points (light blue)**.

---

## **📥 Installation**
### **Requirements**
Ensure you have **Python 3.x** installed and install the required dependencies using:

```bash
pip install numpy pandas matplotlib scipy
```

---

## **🚀 Usage**
Run the main script to start data collection or visualization:

```bash
python espresso-extraction.py
```

### **Menu Options**
1️⃣ **Collect Data** → Input experimental parameters and save data.  
2️⃣ **Visualize 2D Plot** → Select a variable and plot extraction yield.  
3️⃣ **Visualize 3D Plot** → Generate a 3D regression plane for deeper analysis.  

---

## **📊 Example Outputs**
### **2D Plot: Temperature vs. Extraction Yield**
- Shows a linear trend between extraction temperature and yield.
- **Error bars** represent measurement uncertainty.

### **3D Plot: Temperature & Grind Size vs. Extraction Yield**
- Fitted **least squares regression plane** highlights data trends.
- **Red points** = real data, **light blue points** = interpolated.

---

## **🧑‍🔬 Scientific Basis**
This project applies **metrology principles** to espresso brewing, emphasizing the **importance of precise measurement** in achieving a **consistent and high-quality shot**.  

It references scientific research such as:  
📖 Andueza et al. (2003) on grind size and extraction yield.  
📖 Klotz et al. (2020) on temperature's impact on taste.  
📖 UC Davis Coffee Center studies on consumer perception.  

---

## **📂 Repository Structure**
```
espresso-extraction-py/
│── sample-data/            # Stores collected data (CSV)
  │── figures/              # Saved plots
│── espresso_extraction.py  # Entry point for script
│── README.md               # This document
```

---

## **📌 Acknowledgments**
This project was developed as part of a **metrology education initiative**, illustrating how precision measurement impacts everyday experiences like making coffee. ☕🔬  
