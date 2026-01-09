# Solar & Orbital Dynamics Simulation

![Language](https://img.shields.io/badge/Language-Python%203.x-blue)
![Libraries](https://img.shields.io/badge/Libraries-NumPy%20|%20Astropy%20|%20Plotly-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

##  Descripció i Objectiu
Aquest projecte simula i visualitza la dinàmica orbital de la Terra i la radiació solar incident en un panell situat a **Barcelona**.

El codi combina mètodes numèrics clàssics amb dades astrofísiques d'alta precisió per estudiar l'eficiència energètica fotovoltaica sota diferents condicions (incloent-hi ombres orogràfiques).

###  Informe Científic
Per veure la base teòrica detallada, el desenvolupament matemàtic i la discussió profunda dels resultats, consulteu el document adjunt:
 **[Llegir l'Informe Complet (PDF)](./Informe.pdf)**

---

##  Autors
Projecte desenvolupat per l'equip de Física:
* **Joel Sánchez**
* **Rubén Moreno**
* **Xavi Montero**
* **Arnau Rodríguez**

---

##  Fitxers de Sortida
En executar la simulació, es generen automàticament els següents resultats gràfics:

###  Dinàmica Orbital
| Fitxer | Descripció |
| :--- | :--- |
| `orbites_euler_vs_rk4.png` | Comparativa visual de les trajectòries (Euler vs RK4). |
| `energia_total.png` | Conservació de l'energia orbital en funció del mètode. |
| `desviacio_percentual.png` | Anàlisi de l'error numèric i estabilitat. |
| `errors.png` | Quantificació d'errors en el càlcul d'òrbites. |

###  Energia i Simulació Solar
| Fitxer | Descripció |
| :--- | :--- |
| `energia_simulacio.png` | Energia diària generada (Simulació pròpia). |
| `energia_analític.png` | Resultats utilitzant fórmules analítiques teòriques. |
| `energia__astropy_2024.png` | Simulació precisa amb dades d'**Astropy** (2024). |
| `energia_per_plaques.png` | Comparativa: 1 placa vs 2 plaques vs Optimització. |
| `energia_anual_en_funcio_inclinacio.png` | Estudi d'eficiència segons el *tilt* del panell. |

###  Efectes i Visualitzacions
| Fitxer | Descripció |
| :--- | :--- |
| `analema_solar_barcelona.png` | Projecció de l'analema vist des de Barcelona. |
| `energia_simulacio_serralada.png` | Impacte de l'orografia (serralada) en la captació. |
| `energia_simulacio_ombra.png` | Estudi de pèrdues per ombres matinals. |

> **Nota:** També es genera una visualització 3D interactiva de l'analema amb **Plotly**.

---

##  Requeriments i Instal·lació
El projecte requereix un entorn **Python 3**. Les dependències principals són per a càlcul vectorial, astrometria i gràfics.

```bash
pip install numpy matplotlib astropy plotly
