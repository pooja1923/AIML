import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. LOAD & PREPARE IMAGE
# -------------------------------
img = cv2.imread("bird1.png")   #chnage here

if img is None:
    print("Error: Could not find image. Check the filename!")
else:
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Flatten the channels
    R = img_rgb[:, :, 0].flatten()
    G = img_rgb[:, :, 1].flatten()
    B = img_rgb[:, :, 2].flatten()

    # Create DataFrame
    df = pd.DataFrame({'Red': R, 'Green': G, 'Blue': B})

    # -------------------------------
    # SHOW FIRST AND LAST ROWS
    # -------------------------------
    print("--- First 5 Rows of the Dataset ---")
    print(df.head())
    
    print("\n--- Last 5 Rows of the Dataset ---")
    print(df.tail())
    print("-" * 35)

    # -------------------------------
    # 2. COLOR HISTOGRAM
    # -------------------------------
    plt.figure(figsize=(10, 4))
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)

    plt.title("Color Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

    # -------------------------------
    # 3. SCATTER PLOT (DENSE OUTPUT)
    # -------------------------------
    plt.figure(figsize=(8, 6))

    # Using 25,000 samples to match the dense look in your image
    sample_size = 25000 
    
    # Adjusting alpha and size (s) to get the "cloud" blending effect
    plt.scatter(R[:sample_size], G[:sample_size], c='red', label='R vs G', alpha=0.2, s=8)
    plt.scatter(R[:sample_size], B[:sample_size], c='blue', label='R vs B', alpha=0.2, s=8)
    plt.scatter(G[:sample_size], B[:sample_size], c='green', label='G vs B', alpha=0.2, s=8)

    plt.xlabel("Color Intensity")
    plt.ylabel("Color Intensity")
    plt.title("Scatter Plot of RGB Channels")
    
    # Set limits to full range 0-255
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    plt.show()

    # -------------------------------
    # 4. ML MODEL
    # -------------------------------
    X = df[['Blue', 'Green']]
    y = df['Red']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Model Evaluation ---")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print("\nFirst 5 Predicted Red Values:")
    print(y_pred[:5])
