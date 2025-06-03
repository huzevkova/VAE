# Filtrácia medicínskeho datasteu a tréning VAE na generáciu nových snímkov
Projekt obsahuje metódy na filtráciu histopatologických snímok, možnosť automatizovanej optimalizácie hodnôt filtračných metód a tréning variačného autoenkóderu, aj s jednoduchým GUI. Bližšie informácie o fungovaní a použití sú v príručkách.

## Filtrácia podľa farby
Skript na analýzu jednotlivých farieb v snímke. S využitím metódy K-means zhlukovania sa v každej snímke určilo 7 farieb. Po dokončení zhlukovania vznikne reprodukcia pôvodného snímku, v ktorej
každý pixel patrí do jedného zo 7 nájdených zhlukov. Následne sa určia najbližšie farebné pomenovania daných zhlukov. Každá farba zaberá na snímku určité percento. Farby boli ďalej klasifikované ako odtiene bielej, ružovej, fialovej, alebo inej farby.

![color_cluster_comparison](https://github.com/user-attachments/assets/e754bc8b-a04b-476e-ae01-6278e1589fcc)
![color_distribution](https://github.com/user-attachments/assets/4427d5e3-40b6-469d-85ce-f65369b4e10d)

## Filtrácia podľa rozmazania
Skript na identifikáciu rozmazaných snímok. Snímky sú označované ako vhodné alebo nevhodné (rozmazané) na základe výpočtu rozptylu Laplaceovho operátora, ktorý je používaný na detekciu hrán. 

## Filtrácia podľa poču buniek
Skript na analýzu počtu potenciálnych buniek v snímkach. Využíva Sobel filtre, Gaussovu redukciu šumu, a na samotné identifikovanie buniek bola využitá binarizácia a adaptívne prahovanie (angl. adaptive thresholding).

![cell_script_example](https://github.com/user-attachments/assets/ae8911dd-47bf-46bf-bfe2-d42554cea24c)

## Variačný autoenkóder
Klasické komponenty enkóder a dekóder sa skladajú z CNN a ReLU vrstiev. Tradičná stratová fukncia (rekonštrukčná chyba a KL-divergencia) je rozšírená aj o chybu rozmazania. Prostredníctvom GUI možno meniť hyperparametre VAE - veľkosť obrázku, veľkosť latentného priestoru, veľkosť tréningovej dávky, hĺbka siete, počet epoch, koeficienty stratovej funkcie, rýchlosť učenia a počet kanálov. Modely sa priebežne ukladajú aj s hodnotami chýb.

Architektúra VAE s hĺbkou 4:
![VAE](https://github.com/user-attachments/assets/8b7de959-e9e1-4127-8d38-09a729179623)
