## CartPole - Policy Gradient

### Introduccion ###

Este es un proyecto basado en el entorn CartPole, uno de los juegos de ejemplo que estan disponibles en la libreria de Python GYM. Este proyecto tiene como objetivo principal explicar el funcionamiento de el metodo Policy Gradient en la resolucion de problemas relacionados con Aprendizaje Reforzado.

![5](https://user-images.githubusercontent.com/95035101/202281398-5cc3ad67-65c5-4ec1-bf6f-de54e8ae87c8.png)

### ¿Que es Policy Gradient? ###
Policy Gradient es un metodo derivado de los metodos Policy-Based que se encarga de elegir diferentes acciones disponibles en el entorno en un estado determinado. Cada accion tiene una cierta probabilidad de ser ejecutada y esta depende del estado actual y de un cierto nivel aleatoriedad ¡Exacto! Como podras haber intuido el algoritmo nos devuelve probabilidades para cada accion en un estado, entonces nosotros elegiremos una accion de manera aleatoria pero tomando en cuenta estas probabilidades, lo cual significaria que si en un entorno hipotetico tenemos dos acciones: La (A) y la (B) donde la A tiene un 90% de probabilidades de ser ejecutada y la (B) tiene 10%, eso no quiere decir que necesariamente va a ser ejecutada la (A), si no que tiene un 90% de posibilidades de ser la que se valla a ejecturar. Sin embargo, siempre existira un porcentaje de probabilidad, aunque minimo, de que se ejecute la acion (B).

![55](https://user-images.githubusercontent.com/95035101/202281437-99662b10-c5d6-485a-86d1-4aa8fa828fdb.png)

### Como ejecutar el codigo ###

Para lograr que el codigo ejecute y funcione correctamente, deben tener instalado en su computadora las siguientes librerias que se encuentran en el archivo "requirements.txt" y ademas tambien deben tener lo siguiente: os, numpy, gym, csv, pandas, matplotlib.pyplot y keras

