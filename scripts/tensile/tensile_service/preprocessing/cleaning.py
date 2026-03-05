import pandas as pd


def clean_tensile_data(df: pd.DataFrame, zero_shift: bool = True, break_threshold: float = 15.0):
    """
    break_threshold: процент от максимального напряжения (UTS).
    Например, при 15.0: если UTS = 100 MPa, программа обрежет всё, что идет после падения ниже 15 MPa.
    """
    df.columns = ['strain_pct', 'stress_mpa']

    if zero_shift:
        # Коррекция нулевой точки
        df['stress_mpa'] -= df['stress_mpa'].iloc[:10].mean()
        df['strain_pct'] -= df['strain_pct'].iloc[0]

    df = df[df['strain_pct'] >= 0].reset_index(drop=True)

    # --- УМНОЕ ОБРЕЗАНИЕ ПО ПОРОГУ ---
    if not df.empty:
        idx_uts = df['stress_mpa'].idxmax()
        max_stress = df['stress_mpa'].max()

        # Рассчитываем физический порог в MPa
        cutoff_value = max_stress * (break_threshold / 100.0)

        # Ищем данные после пика
        post_uts = df.iloc[idx_uts:]

        # Находим первую точку, которая упала ниже порога
        # Также проверяем на отрицательные значения (уход в минус)
        break_condition = (post_uts['stress_mpa'] <= cutoff_value) | (post_uts['stress_mpa'] < 0)

        if break_condition.any():
            idx_break = break_condition.idxmax()
            df = df.iloc[:idx_break].reset_index(drop=True)

    df['strain_abs'] = df['strain_pct'] / 100.0
    return df