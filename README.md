
# 🎥 Multi-Camera Spatial Calibration & Stereo Tracking

Ce dépôt contient l'architecture logicielle et les scripts mathématiques développés dans le cadre d'un projet de recherche et développement (Système d'analyse comportementale). 

Il implémente un pipeline complet de vision par ordinateur orienté objet pour l'étalonnage spatial, l'application de la géométrie épipolaire et la triangulation 3D robuste à l'aide de marqueurs ChArUco, avec un système prédictif de gestion des occultations.

## ⚠️ Avertissement sur les Données (RGPD)
Pour des raisons de confidentialité et de respect de la vie privée, **aucune donnée visuelle (fichiers `.mp4`, images de la salle, visages ou données annotées) n'est incluse dans ce dépôt**. 
Les scripts sont fournis nus. Pour exécuter et tester ce pipeline, vous devez obligatoirement fournir vos propres flux vidéos et les placer dans les répertoires d'entrée appropriés.

## 🛠️ Prérequis et Installation

Assurez-vous d'avoir installé Python 3.x. Installez ensuite les dépendances mathématiques et de vision par ordinateur requises :

```bash
pip install -r requirements.txt
