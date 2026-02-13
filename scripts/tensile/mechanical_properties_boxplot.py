import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- подготовка данных ---
def round_temp(t):
    if t < 30: return 22
    elif t < 45: return 40
    else: return 50

df = pd.read_csv("out/summary.csv")

# приведение имён из вашей таблицы
col_temp   = "Temp_C"
col_speed  = "TestSpeed_mm_min"
col_sigma  = "MaxStress_MPa_data"      # было MaxStressMPa_data — опечатка
col_Egpa   = "YoungsModulus_GPa"       # в исходной таблице есть такой столбец

# защита от разных регистров/пробелов
df = df.rename(columns={c: c.strip() for c in df.columns})

df["TempGroup"] = df[col_temp].apply(round_temp)
df["Speed_cat"] = df[col_speed].astype(str)  # для аккуратной легенды

# --- boxplot для прочности ---
plt.figure(figsize=(8,6))
sns.boxplot(x="TempGroup", y=col_sigma, hue="Speed_cat", data=df)
plt.xlabel("Temperature (°C)")
plt.ylabel("Max Stress (MPa)")
plt.title("Boxplot of Max Stress by Temperature Group and Speed")
plt.legend(title="Speed (mm/min)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("box_max_stress.png", dpi=300)


# --- boxplot для модуля Юнга ---
plt.figure(figsize=(8,6))
sns.boxplot(x="TempGroup", y=col_Egpa, hue="Speed_cat", data=df)
plt.xlabel("Temperature (°C)")
plt.ylabel("Young's Modulus (GPa)")
plt.title("Boxplot of Young's Modulus by Temperature Group and Speed")
plt.legend(title="Speed (mm/min)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("box_E_GPa.png", dpi=300)

