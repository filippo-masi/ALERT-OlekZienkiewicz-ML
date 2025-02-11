### **Neural integration for constitutive equations – HANDS-ON for experimental data**  

This repository contains **Jupyter notebooks** and **code** developed for the **2025 ALERT Olek Zienkiewicz Doctoral School** on *Constitutive modelling of geomaterials* - Session: *Constitutive modelling meets Machine Learning*.

📌 **More details**: [ALERT OZ School 2025 Program](https://soilmodels.com/alert-oz-school-2025/)

---

<p align="center">
  <img src="./_images/front_page.png" alt="Neural Integration for Small Data" width="100%">
</p>

---

#### **📂 Hands-on Exercises**
Two exercises are considered, both uses experimental results of a sand specimen subjected to drained triaxial compression from [SoilModels](https://soilmodels.com/). The first exercise considers a filtered (smoothed) version of the original dataset, the second uses the original, raw dataset. The filtered data is used to enable a fast training which can be effectively used to vary the structure of the network (and its hyper-parameters).


#### **I. Learning constitutive equations from experiments – `[sand] training & validation`**
- *Material*: Sand with scattered gravel *(origin: Dobrany, Czech Republic)*
- *Tests*: Drained triaxial compression  
- *Dataset*: *Smoothed version* of the experimental data  

#### **II. Learning constitutive equations from experiments – `[sand-raw] training & validation`**
- *Material*: Sand with scattered gravel *(origin: Dobrany, Czech Republic)*
- *Tests*: Drained triaxial compression  
- *Dataset*: *Original experimental data*  

---

#### **📖 References**

If you find this repository helpful, please consider citing:

Masi, F., & Einav, I. (2023). [Neural integration for constitutive equations using small data](https://doi.org/10.1016/j.cma.2023.116698). *Computer Methods in Applied Mechanics and Engineering*, 420, 116698.

```bibtex
@article{masi2024neural,
  title={Neural integration for constitutive equations using small data},
  author={Masi, Filippo and Einav, Itai},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={420},
  pages={116698},
  year={2024},
  publisher={Elsevier}
}
```

```bibtext
@article{HandsOnNICE,
  title={NICE - Experiments Hands-on},
  author={Masi, Filippo and Louvard, Enzo},
  year={2025},
  url={https://github.com/filippo-masi/ALERT-OlekZienkiewicz-ML}
}
```
Authors: Filippo Masi, Enzo Louvard
