#import "conf.typ": conf
#show: conf.with(
  title: ("Titulo de mi tfg", "Title of my thesis"),
  tutors: ("Javier Gonzalez", "José Raúl Ruiz Sarmiento"),
  dept: "Departamento de automática y control"
)

#include "resumen.typ"

= Tecnologías utilizadas
Apple DepthPro @Bochkovskii2024:arxiv

Apple HyperSim @roberts:2021

= Desarrollo
== Ensamblado del pipeline
#lorem(100)

== Entrenamiento
#lorem(100)
#lorem(200)

=== Datos
Entrenamiento con datos sinteticos @roberts:2021: Queremos que el modelo aprenda a modelar la incertidumbre de la estimación, necesitamos datos robustos, profundidad sintética -> perfecta

Aprendizaje semisupervisado: No hay una ground truth como tal


#lorem(200)


=== Preprocesado
#lorem(100)

=== Modelo
VAE convolucional

#lorem(200)

#lorem(200)

= Conclusiones y trabajo futuro
#lorem(200)
#lorem(200)

== Ideas
- idea: 2 frames, (D[t-1], Est[t-1]) y (D[t], Est[t]), calcular INC[t]