# dnn-rotnet-sem-supervised-learning

L'apprentissage semi-supervisé est une technique d'apprentissage automatique qui combine à la fois l’utilisation de données étiquetées et non étiquetées. Il se situe ainsi entre l'apprentissage supervisé qui n'utilise que des données étiquetées et l'apprentissage non supervisé qui n'utilise que des données non étiquetées. Il est évident que l’objectif d’une telle approche est de permettre de former des RNP performant même quand les données étiquetées viennent à faire défaut, étant donné que l’approche n’a besoin que d’un nombre très limité d’étiquette pour fonctionner. Pour les taches de classification d’image, il existe deux approches pour faire de l'apprentissage semi-supervisé :
a. Faire appel à des algorithmes qui permettent d'apprendre simultanément à partir des données étiquetées et non étiquetées ;
b. Faire une approche en deux temps : apprendre à partir des données non étiquetées, par des techniques d’apprentissage auto-supervisé  , pour créer un premier RNP (*), et par la suite utiliser l'apprentissage par transfère  pour créer un deuxième RNP à partir du premier et ainsi résoudre la tâche de classification (**). 
(*) : cette première étape a pour but d’entrainer un RNP à connaitre les caractéristiques des images à classifier. Par caractéristique, il s’agit de la forme, de la disposition, de l’orientation, etc. des éléments dans l’image. Par exemple, pour pouvoir prédire qu’une image contient un visage de chat, il faudra d’abord être en mesure de reconnaitre les éléments constitutifs (forme des oreilles, nez, moustache etc.) qui constitut un visage de chat. Pour assurer cette première étape, de nombreuses approches ont été proposées, notamment la « Reconnaissance des transformations géométriques » : RotNet [2]. RotNet est une méthode auto-supervisée pour apprendre une caractéristique efficace des images en prédisant la rotation, discrétisée en 4 rotations de 0, 90, 180 et 270 degrés. 
 
Bien qu'il s'agisse d'une idée très simple, pour prédire les rotations, le modèle doit comprendre l'emplacement, les types et la pose des objets dans une image pour résoudre cette tâche et, à ce titre, les représentations apprises sont utiles pour les tâches en aval (c.-à-d. (**)).
(**) : cette deuxième étape a pour but de réaliser la tâche de classification en donnant une sémantique aux formes et aux caractéristiques apprises par le premier RNP dans le (*). Et grâce aux caractéristiques apprises dans le (*), l’apprentissage par transfert peut se réaliser de manière supervisée avec très peu d’étiquette.

Dans le contexte de ce travail, c’est la deuxième approche qui a été utilisée, couplé à l’utilisation de RotNet pour l’apprentissage/extraction des caractéristiques. 
