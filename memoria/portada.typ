#let portada(
  title: none,
  tutors: none,
  dept: none,
  doc
) = {
  let vspace = 1.5em
  page(fill: rgb("#183256"), margin: (x: 0.75in, y: 1in), header: [
      #place(
        horizon + start,
        dx: -0.4in,
        dy: 0.5in,
        image("images/MARCA UNIV. NEGATIVA VERTICAL.png", height: 1.54in)
      )
      #place(
        horizon + end,
        dx: 0.2in,
        dy: 0.5in,
        image("images/COLOR_NEGATIVO_INFORMATICA_recortado.png", height: 1.54in)
      ),
    ],
    text(size: 18pt, stretch: 50%, white)[
      #place(center + horizon, align(center)[
      Grado en Ingeniería Informática
      #v(vspace)

      #text(size: 22pt)[
        #title.at(0) \
        #title.at(1)
      ]
      #v(vspace)

      #text(gray, "Realizado por") \
      Rubén Jiménez Reina
      #v(vspace)

      #text(gray, "Tutorizado por") \
      #tutors.at(0) \
      #tutors.at(1)
      #v(vspace)

      #text(gray, "Departamento") \
      #dept \
      UNIVERSIDAD DE MÁLAGA
      #v(vspace)

      #text(gray, "Málaga, junio de 2025")
      ])
    ]
  )

  pagebreak()
  doc
}