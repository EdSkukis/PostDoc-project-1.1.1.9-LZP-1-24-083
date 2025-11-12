# TGA Friedman Isoconversional Analysis (Python)

## 1. Project Overview

This repository implements the **Friedman differential isoconversional method** for thermal analysis of solid-state reactions (e.g., polymer degradation, oxidation, pyrolysis) using thermogravimetric analysis (TGA) data recorded at different heating rates.

The method estimates the **apparent activation energy** \(E(\alpha)\) as a function of the **conversion degree** \( \alpha \), without assuming a specific kinetic model. It is based on:

$$
\frac{d\alpha}{dt} = A\,f(\alpha)\,\exp\!\left(-\frac{E(\alpha)}{R\,T}\right)
$$

Taking logarithms for multiple heating rates at the same \(\alpha\):

$$
\ln\!\left(\frac{d\alpha}{dt}\right)
   = \ln[A f(\alpha)] - \frac{E(\alpha)}{R\,T}
$$

Hence, for each conversion level \(\alpha\), the slope of
\(\ln(d\alpha/dt)\) vs \(1/T\) gives \(-E(\alpha)/R\).

---

## 2. Folder Structure

```
tga_friedman_project/
├── tga_friedman.py # Main analysis script
├── README.md # Documentation
├── data/
│ ├── template.csv # Input format example
│ ├── example_5Kmin.csv
│ ├── example_10Kmin.csv
│ └── example_20Kmin.csv
└── output/ # Results and plots are written here
```
## 6. Extending the Framework

The current architecture is modular. The main functions are summarized below:

| Function | Role |
|-----------|------|
| `read_run()` | Loads and preprocesses a single TGA dataset |
| `interp_at_alpha()` | Interpolates temperature and rate at each α-grid |
| `fit_friedman()` | Performs linear regression for Friedman analysis |

---

### Implementing Other Isoconversional Methods

#### (a) Ozawa–Flynn–Wall (OFW) Method

For each conversion level \( \alpha \):

$$
\log(\beta) = \text{const} - 0.4567\,\frac{E(\alpha)}{R\,T_{\alpha}}
$$

Perform a linear regression of \( \log \beta \) versus \( 1/T_{\alpha} \)  
for a set of heating rates \(\beta\).  
The slope gives the activation energy \(E(\alpha)\) through:

$$
E(\alpha) = -\,\frac{\text{slope} \times R}{0.4567}
$$

---

#### (b) Kissinger–Akahira–Sunose (KAS) Method

For each conversion level \( \alpha \):

$$
\ln\!\left(\frac{\beta}{T_{\alpha}^2}\right)
   = \text{const} - \frac{E(\alpha)}{R\,T_{\alpha}}
$$

Perform a regression of \( \ln(\beta/T_{\alpha}^2) \) versus \( 1/T_{\alpha} \).  
The slope directly yields \( -E(\alpha)/R \).

---

Both methods can be implemented by reusing the same interpolation step  
(`interp_at_alpha()`) and replacing only the regression expression.

To summarize:

| Method | Regression Variable | Equation Form | Slope Interpretation |
|:-------|:--------------------|:---------------|:---------------------|
| **Friedman** | \( \ln(d\alpha/dt) \) vs \( 1/T \) | \( \ln(d\alpha/dt) = \ln[A f(\alpha)] - E(\alpha)/(R T) \) | \( -E(\alpha)/R \) |
| **OFW** | \( \log(\beta) \) vs \( 1/T_{\alpha} \) | \( \log(\beta) = \text{const} - 0.4567 E(\alpha)/(R T_{\alpha}) \) | \( -0.4567\,E(\alpha)/R \) |
| **KAS** | \( \ln(\beta/T_{\alpha}^2) \) vs \( 1/T_{\alpha} \) | \( \ln(\beta/T_{\alpha}^2) = \text{const} - E(\alpha)/(R T_{\alpha}) \) | \( -E(\alpha)/R \) |

---

**Implementation Tip:**  
In `tga_friedman.py`, extend the function `fit_friedman()` or create  
`fit_ofw()` / `fit_kas()` that call the same interpolation logic but  
apply the alternative linear relationships above.
---
## 7. References

1. Friedman, H. L. Kinetics of thermal degradation of plastics from thermogravimetry. Application to polyethylene.
Journal of Polymer Science Part C: Polymer Symposia 6 (1964) 183–195.

2. Vyazovkin, S. A unified approach to kinetic analysis of nonisothermal data.
Thermochimica Acta 340–341 (1999) 53–68.

3. Vyazovkin, S., Burnham, A. K. et al. ICTAC Kinetics Committee Recommendations for performing kinetic computations on thermal analysis data.
Thermochimica Acta 520 (2011) 1–19.