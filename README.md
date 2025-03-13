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
  
![image](https://github.com/user-attachments/assets/2cbfe2de-1baf-441b-925f-e412e11c8c03)


### **3D Plot: Temperature & Grind Size vs. Extraction Yield**
- Fitted **least squares regression plane** highlights data trends.
- **Red points** = real data, **light blue points** = interpolated.

![extraction_3D_Temperature_90-95_Grind_200-600_20250312_233608](https://github.com/user-attachments/assets/f7f11192-1e68-4d54-9692-a46c6f8eeddb)


---

## **🧑‍🔬 Scientific Basis**
This project applies **metrology principles** to espresso brewing, emphasizing the **importance of precise measurement** in achieving a **consistent and high-quality shot**.  

It references scientific research such as:  
📖 Andueza et al. (2003) on grind size and extraction yield.  
📖 Klotz et al. (2020) on temperature's impact on taste.  
📖 UC Davis Coffee Center studies on consumer perception.  

**Reference List:**

[1] Várady, M., Tauchen, J., Klouček, P., & Popelka, P. (2022). Effects of Total Dissolved Solids, Extraction Yield, Grinding, and Method of Preparation on Antioxidant Activity in Fermented Specialty Coffee. Fermentation, 8(8), 375. https://doi.org/10.3390/fermentation8080375

[2] Andueza, S.F., Vila, M.A., Peña, M.P., & Cid, C. (2007). Influence of coffee/water ratio on the final quality of espresso coffee. Journal of the Science of Food and Agriculture, 87, 586-592.

[3] Batali, M. E., Cotter, A. R., Frost, S. C., Ristenpart, W. D., & Guinard, J.-X. (2021). Titratable Acidity, Perceived Sourness, and Liking of Acidity in Drip Brewed Coffee. ACS Food Science & Technology, 1(4), 559–569. https://doi.org/10.1021/acsfoodscitech.0c00078

[4] Klotz, J. A., Winkler, G., & Lachenmeier, D. W. (2020). Influence of the Brewing Temperature on the Taste of Espresso. Foods (Basel, Switzerland), 9(1), 36. https://doi.org/10.3390/foods9010036

[5] Andueza, S., De Peña, M. P., & Cid, C. (2003). Chemical and sensorial characteristics of espresso coffee as affected by grinding and torrefacto roast. Journal of Agricultural and Food Chemistry, 51(24), 7034–7039. https://doi.org/10.1021/jf034628f

[6] Liu, X., Tang, Y., & Wang, Y. (2022). Consumer Satisfaction for Starbucks. Proceedings of the 2022 7th International Conference on Social Sciences and Economic Development (ICSSED 2022), 1475–1482. https://doi.org/10.2991/aebmr.k.220405.247

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

## **🎥 Video**
A video was created to better explain and show the results from these studies and what can be done with the script.  
https://www.youtube.com/watch?v=FDcUICSV3XE

---

## **🔬 Equipment**

Cheap refractometer: https://a.co/d/f6HQd3G
--> Measures in Brix (sugar content). The script converts **Brix** to **TDS** and **Extraction Yield**.

---

## **📌 Acknowledgments**
This project was developed as part of a **metrology education initiative**, illustrating how precision measurement impacts everyday experiences like making coffee. ☕🔬  
