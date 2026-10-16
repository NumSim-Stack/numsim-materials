# Material reports

One folder per material: equations, implementation, and results generated from its `case.json`
(stress for prescribed strain paths, uniaxial stress with its Newton iterations, stress paths, tangent check, history).

| material | kind |
|---|---|
| [linear_elasticity](linear_elasticity/README.md) | small-strain elasticity |
| [j2_plasticity](j2_plasticity/README.md) | small-strain von Mises plasticity |

Regenerate the results:

```sh
cmake -B build -DNUMSIM_MATERIALS_BUILD_REPORTS=ON && cmake --build build --target material_report
python3 reports/make_report.py --tool build/reports/material_report docs/materials/<material>
```
