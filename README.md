## CartPole - Policy Gradient

### Introduccion ###

Este es un proyecto basado en el entorn CartPole, uno de los juegos de ejemplo que estan disponibles en la libreria de Python GYM. Este proyecto tiene como objetivo principal explicar el funcionamiento de el metodo Policy Gradient en la resolucion de problemas relacionados con Aprendizaje Reforzado.

![5](https://user-images.githubusercontent.com/95035101/202281398-5cc3ad67-65c5-4ec1-bf6f-de54e8ae87c8.png)

### ¿Que es Policy Gradient? ###
Policy Gradient es un metodo derivado de los metodos Policy-Based que se encarga de elegir diferentes acciones disponibles en el entorno en un estado determinado. Cada accion tiene una cierta probabilidad de ser ejecutada y esta depende del estado actual y de un cierto nivel aleatoriedad ¡Exacto! Como podras haber intuido el algoritmo nos devuelve probabilidades para cada accion en un estado, entonces nosotros elegiremos una accion de manera aleatoria pero tomando en cuenta estas probabilidades, lo cual significaria que si en un entorno hipotetico tenemos dos acciones: La (A) y la (B) donde la A tiene un 90% de probabilidades de ser ejecutada y la (B) tiene 10%, eso no quiere decir que necesariamente va a ser ejecutada la (A), si no que tiene un 90% de posibilidades de ser la que se valla a ejecturar. Sin embargo, siempre existira un porcentaje de probabilidad, aunque minimo, de que se ejecute la acion (B).

![55](https://user-images.githubusercontent.com/95035101/202281437-99662b10-c5d6-485a-86d1-4aa8fa828fdb.png)

### Definicion Matematica ###

#### Trayectoria ####
Lo primero que debemos definir es una trayectoria, solo una secuencia de estado-acción-recompensas (pero ignoramos la recompensa). Una trayectoria es un poco más flexible que un episodio porque no hay restricciones en su duración; puede corresponder a un episodio completo o solo a una parte de un episodio. Denotamos la longitud con una H mayúscula, donde H representa el Horizonte, y representamos una trayectoria con τ:  

![1_wxTnMR0P97Jvvuhb1PXAkQ](https://user-images.githubusercontent.com/95035101/202324606-bed180c5-4449-45a5-963e-c258f6f969b4.png)

El método REINFORCE se basa en trayectorias en lugar de episodios porque maximizar el rendimiento esperado sobre las trayectorias (en lugar de los episodios) permite que el método busque políticas óptimas para tareas episódicas y continuas.  

![1_YKP3TY95UHwxIlMLg_3IQQ](https://user-images.githubusercontent.com/95035101/202324878-abfb7dbb-125a-402c-a399-f055103ee64d.png)

Aunque para la gran mayoría de las tareas episódicas, donde solo se entrega una recompensa al final del episodio, solo tiene sentido usar el episodio completo como una trayectoria; de lo contrario, no tenemos suficiente información sobre recompensas para estimar significativamente el rendimiento esperado.  

#### Retorno de una trayectoria ####

Denotamos el retorno de una trayectoria τ con R(τ), y se calcula como la recompensa total de esa trayectoria τ:  

![1_TTrBrnE3Jrt6ShtPTZdX1Q](https://user-images.githubusercontent.com/95035101/202324858-4d567bf8-1127-4db9-818c-1af0720a0343.png)

El parámetro Gk se denomina rendimiento total, o rendimiento futuro, en el paso de tiempo k para la transición k  

![1_YKP3TY95UHwxIlMLg_3IQQ](https://user-images.githubusercontent.com/95035101/202325130-c4faeced-a169-4a1b-9eca-2c476350ff45.png)

Es el retorno que esperamos obtener desde el paso de tiempo k hasta el final de la trayectoria, y se puede aproximar sumando las recompensas de algún estado del episodio hasta el final del episodio usando gamma γ:  

#### Rendimiento esperado ####

![1_LMwxo_VGVIJSssUxt06kwA](https://user-images.githubusercontent.com/95035101/202325121-0484e865-6668-4d43-9aec-f711df106849.png)

Recuerde que el objetivo de este algoritmo es encontrar los pesos θ de la red neuronal que maximiza el rendimiento esperado que denotamos por U(θ) y se puede definir como:

Para ver cómo se corresponde con el rendimiento esperado, observe que hemos expresado el rendimiento R(τ) en función de la trayectoria τ. Luego, calculamos el promedio ponderado, donde los pesos están dados por P(τ;θ), la probabilidad de cada trayectoria posible, de todos los valores posibles que puede tomar el retorno R(τ). Tenga en cuenta que la probabilidad depende de los pesos θ en la red neuronal porque θ define la política utilizada para seleccionar las acciones en la trayectoria, que también juega un papel en la determinación de los estados que observa el agente.

#### Ascenso de gradiente ####  

Como ya indiqué, una forma de determinar el valor de θ que maximiza la función U(θ) es a través del ascenso de gradiente.  

Equivalente al algoritmo Hill-Climbing presentado en este Post, intuitivamente podemos visualizar que el ascenso de gradiente traza una estrategia para alcanzar el punto más alto de una colina, U(θ), simplemente dando pequeños pasos de forma iterativa en la dirección del gradiente:  

![1_kT_Ncn2NNkLCH9DKY2RTuA](https://user-images.githubusercontent.com/95035101/202325216-f46a02c5-7bff-4118-bce8-3b62327c8121.png)

Matemáticamente, nuestro paso de actualización para el ascenso de gradiente se puede expresar como:  

![1_Kx9muF8yH2IXahRaM2ahCg](https://user-images.githubusercontent.com/95035101/202325350-ee85e65f-7895-4155-9c21-a05791e7d791.png)

donde α es el tamaño de paso que generalmente se permite que disminuya con el tiempo (equivalente a la disminución de la tasa de aprendizaje en el aprendizaje profundo). Una vez que sabemos cómo calcular o estimar este gradiente, podemos aplicar repetidamente este paso de actualización, con la esperanza de que θ converja al valor que maximiza U(θ).  

### Como ejecutar el codigo ###

Para lograr que el codigo ejecute y funcione correctamente, deben tener previamente instalado en su computadora las siguientes librerias que se encuentran en el archivo "requirements.txt" y ademas tambien deben tener lo siguiente: os, numpy, gym, csv, pandas, matplotlib.pyplot y keras.

Una vez corroborado todo esto debes ir a una consola de comandos (En mi caso es cmd porque uso Windows pero puede ser cualquier otri sistema operativo).

![32](https://user-images.githubusercontent.com/95035101/202325630-f0622bc4-89b0-41c2-94d2-6e2e928108ab.png)

### Explicacion de las funciones ###

#### OurModel(input_dim, output_dim, lr) ####
Esta funcion se encarga de crear nuestro modelo de red neuronal, el cual entrenaremos para que tome las decisiones al momento de elegir una accion. Una detalle importante es que la capa final de esta red neuronal es de tipo Softmax, lo cual es importante porque nosotros necesimos que la salida de dicha red sea un arreglo con un conjunto de probabilidades.

#### ShowMetrics() ####
Esta funcion nos permite graficar tanto la perdida total (Loss) como la recompensa total (reward) por cada episodio.

#### __init__(env_name, render) ####
Esta funcion se va a encargar de definir las variables bases necesarias para que el agente pueda trabajar de manera correcta. 

#### get_action(state) ####
Esta funcion va a devolver una accion dado el estado en el que se encuentra el agente. Para hacerlo ejecuta la funcion 'predict' dentro de el modelo 'Actor' que se encuentra dentro del agente. El producto de esa prediccion es un arreglo con probabilidades que sera usado por la funcion np.random.choice el cual elegira una accion de manera aleatoria pero teniendo en cuenta las probabilidades del modelo.

#### remember(state, reward, action) ####
Esta funcion se encarga de guardar el estado, recompensa y accion en cada momento del episodio para ser usado posteriormente en el entrenamiento. En el caso de la accion se la guarda en formato "one_hot" que quiere decir que se la guarda en un arreglo donde el indice seleccionado es 1 y los demas valores son 0. La razon de esto es que posteriormente en la etapa de entrenamiento nosotros vamos a necesitar que las acciones tengan la misma dimensionalidad que el la salida de nuestra red neuronal, asi podemos compararlas en el entrenamiento.

#### forget() ####
Esta funcion se encarga de vaciar los arreglos que guardan los estados, recompensas y acciones para el proxima episodio. Esto se hace para que no se mesclen experiencias nuevas con viejas.

#### save() ####
Esta funcion se encarga de guardar los pesos del modelo para que no perdamos el progreso del entrenamiento en caso de que queramos detener el aprendizaje del agente.

#### load() ####
Esta funcion se encarga de cargar los pesos de nuestro modelo, en caso de que existan.

#### discounted_reward() ####
Esta es la piedra angular de el procedimiento de nuestro entrenamiento. El concepto de recompensa descontada se utiliza para darle a la recompensa de cada estado un valor de el siguiente modo:   

![ffff](https://user-images.githubusercontent.com/95035101/202328785-413b4da0-ba97-4323-9819-262d348aa70e.png)

Es decir que la recompensa de cada estado va a ser igual a la sumatoria de la recompensa actual mas las recompensas posteriores a esas multiplicadas por un factor gamma con un exponenete que es igual a t'-t-1. Esto probocara que a medida que se valla recorriendo el arreglo de recompensas, el valor va a ir disminuyendo y por lo tanto las primeras van a tener un valor mas alto y las ultimas van a tener el valor mas bajo. La logica esta en que las ultimas acciones generan recompensas mas bajas porque fueron malas y nos hicieron perder el episodio. Por lo tanto seran estas ultimas las que tendremos que cambiar.

![3](https://user-images.githubusercontent.com/95035101/202329584-7c3e601d-8941-4c13-9414-490c045e2f8e.png)

Paso 1: Definir las variables iniciales. Gamma es el factor de descuento que se usara para todas las recompensas. discounted_rewards es el arreglo que guardara todas las recompensas que han pasado por el algoritmo de descuento. reward_sum es el acumulativo de las recompensas futuras.



#### train() ####

#### run() ####


