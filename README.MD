## Introduction

Ce projet fournie un ensemble de méthodes permettant:
* De calculer différents type de features audio: 
    * Pause Time
    * Speak Time
    * Silence
    * Overlap
    * Turn duration
* De rechercher des backchannels
* De calculer la synchronie de groupe

## Données
Chaque partie à besoin de données, celles-ci peuvent être extraites grâce au fichier de configuration OpenSmile fourni avec le projet dans le dossier 'OpenSmileConfiguration'.
Pour convertir un fichier audio en donnée lisible par le programme il faut faire: 
```"path_to_smile_extract"/SMILExtract_Release.exe -C OpenSmileConfiguration/extract2.conf -I "path_to_audio" -O "path_to_out"```

Par la suite changer le fichier constante.py, pour ceci faites:
````python
session = <number_session>
if session == <number_session>:
    filenames = <liste de vos piste en csv>
    offset = <decalage de vos piste en secondes>
    max_len = <taille maximum de vos piste en seconde>
````

## Utilisation
Vous pouvez voir l'utilisation des différentes partie dans les main respectifs:
* backchannels: Pas au point actuellement
* calcule des features: 'features/main.py'
* calcule de la synchronie: 'synchronie/main_synchronie.py'



