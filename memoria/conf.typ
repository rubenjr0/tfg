#import "portada.typ": portada

#let conf(
  title: none,
  tutors: none,
  dept: none,
  doc
) = {
  set text(lang: "es", font: "Liberation Sans", kerning: true, hyphenate: false)
  set par(justify: true)
  set heading(numbering: "1.")

  show: portada.with(
    title: title,
    tutors: tutors,
    dept: dept
  )

  outline()

  pagebreak()
  counter(page).update(1)
  set page(numbering: "1")
  doc

  pagebreak()
  bibliography("bibliografia.bib", style: "apa", full: true)

  set heading(numbering: none)
  pagebreak()
  include "anexo.typ"
}